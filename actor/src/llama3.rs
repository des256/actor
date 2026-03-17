use {
    crate::*,
    half::f16,
    std::{
        ffi::c_void,
        sync::{Arc, mpsc as std_mpsc},
    },
    tokenizers::Tokenizer,
    tokio::sync::mpsc as tokio_mpsc,
};

const CHANNEL_CAPACITY: usize = 64;

const ENGINE_PATH: &str = "data/llama3/engine/model.engine";
const TOKENIZER_PATH: &str = "data/llama3/engine/tokenizer.json";

// Architecture constants (must match export.py)
const NUM_LAYERS: usize = 28;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const VOCAB_SIZE: usize = 128256;
const MAX_SEQ_LEN: usize = 2048;

// Token IDs
const BOS_TOKEN: i32 = 128000;
const EOS_TOKEN: i32 = 128001;
const EOT_TOKEN: i32 = 128009;

// KV cache buffer size per tensor (k or v) at max_seq_len
// Shape: [1, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM] f16
const KV_BYTES: usize = 1 * NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM * size_of::<f16>();

unsafe extern "C" {
    fn cudaStreamCreate(stream: *mut *mut c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
}

pub struct Input<T: Clone + Send + 'static> {
    pub payload: T,
    pub prompt: String,
    pub max_tokens: usize,
    pub stamp: u64,
}

pub enum Output<T: Clone + Send + 'static> {
    Token { payload: T, token: String, stamp: u64 },
    Eos { payload: T, stamp: u64 },
}

struct Direct {
    // CUDA stream
    stream: *mut c_void,

    // GPU buffers
    input_ids: tensorrt::Buffer,    // [1, seq_len]
    position_ids: tensorrt::Buffer, // [1, seq_len]
    logits: tensorrt::Buffer,       // [1, seq_len, VOCAB_SIZE]

    // KV cache double buffers (56 pairs = 112 total)
    kv_a: Vec<tensorrt::Buffer>, // past_key_values inputs (buffer A)
    kv_b: Vec<tensorrt::Buffer>, // past_key_values inputs (buffer B)

    // TRT context
    ctx: Arc<tensorrt::Context>,

    // Tokenizer
    tokenizer: Tokenizer,

    // Generation state
    past_len: usize,
    kv_idx: bool, // false → A is input, B is output
}

unsafe impl Send for Direct {}

impl Direct {
    fn new(trt: &Arc<tensorrt::Tensorrt>) -> Self {
        let engine = trt.load_engine(ENGINE_PATH);


        let ctx = engine.create_context();

        let mut stream: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { cudaStreamCreate(&mut stream) };
        assert!(rc == 0, "cudaStreamCreate failed: {rc}");

        let i64_size = size_of::<i64>();

        // Allocate I/O buffers
        let input_ids = tensorrt::Buffer::new(MAX_SEQ_LEN * i64_size);
        let position_ids = tensorrt::Buffer::new(MAX_SEQ_LEN * i64_size);
        let logits = tensorrt::Buffer::new(MAX_SEQ_LEN * VOCAB_SIZE * size_of::<f16>());

        // Allocate KV cache buffers (56 pairs)
        let mut kv_a = Vec::with_capacity(NUM_LAYERS * 2);
        let mut kv_b = Vec::with_capacity(NUM_LAYERS * 2);
        for _ in 0..(NUM_LAYERS * 2) {
            kv_a.push(tensorrt::Buffer::new(KV_BYTES));
            kv_b.push(tensorrt::Buffer::new(KV_BYTES));
        }

        let tokenizer = Tokenizer::from_file(TOKENIZER_PATH).unwrap_or_else(|e| panic!("failed to load tokenizer: {e}"));

        Self {
            stream,
            input_ids,
            position_ids,
            logits,
            kv_a,
            kv_b,
            ctx,
            tokenizer,
            past_len: 0,
            kv_idx: false,
        }
    }

    /// Run inference for a single step or prefill.
    fn infer(&mut self, input_token_ids: &[i64], seq_len: usize) {
        // Upload input_ids and position_ids
        self.input_ids.upload(input_token_ids);
        let positions: Vec<i64> = (self.past_len as i64..(self.past_len + seq_len) as i64).collect();
        self.position_ids.upload(&positions);

        // Set dynamic input shapes
        self.ctx.set_input_shape("input_ids", &[1, seq_len as i64]);
        self.ctx.set_input_shape("position_ids", &[1, seq_len as i64]);

        // Bind I/O tensors
        self.ctx.set_tensor_address("input_ids", self.input_ids.ptr);
        self.ctx.set_tensor_address("position_ids", self.position_ids.ptr);
        self.ctx.set_tensor_address("logits", self.logits.ptr);

        // Bind KV cache: past_key_values (input) and present (output)
        let (kv_in, kv_out) = if !self.kv_idx {
            (&self.kv_a, &self.kv_b)
        } else {
            (&self.kv_b, &self.kv_a)
        };

        for layer in 0..NUM_LAYERS {
            let k_idx = layer * 2;
            let v_idx = layer * 2 + 1;

            let past_k_name = format!("past_key_values.{}.key", layer);
            let past_v_name = format!("past_key_values.{}.value", layer);
            let present_k_name = format!("present.{}.key", layer);
            let present_v_name = format!("present.{}.value", layer);

            // Set KV cache shapes
            self.ctx
                .set_input_shape(&past_k_name, &[1, NUM_KV_HEADS as i64, self.past_len as i64, HEAD_DIM as i64]);
            self.ctx
                .set_input_shape(&past_v_name, &[1, NUM_KV_HEADS as i64, self.past_len as i64, HEAD_DIM as i64]);

            // Bind addresses
            self.ctx.set_tensor_address(&past_k_name, kv_in[k_idx].ptr);
            self.ctx.set_tensor_address(&past_v_name, kv_in[v_idx].ptr);
            self.ctx.set_tensor_address(&present_k_name, kv_out[k_idx].ptr);
            self.ctx.set_tensor_address(&present_v_name, kv_out[v_idx].ptr);
        }

        // Execute
        self.ctx.enqueue(self.stream);
        unsafe { cudaStreamSynchronize(self.stream) };

        // Update state
        self.past_len += seq_len;
        self.kv_idx = !self.kv_idx;
    }

    /// Generate tokens autoregressively. Returns generated token strings via callback.
    fn generate<F>(&mut self, prompt_tokens: &[i64], max_new_tokens: usize, mut on_token: F)
    where
        F: FnMut(String) -> bool, // Returns true to continue, false to stop
    {
        // Reset state
        self.past_len = 0;
        self.kv_idx = false;

        // Prefill phase: process prompt tokens
        if !prompt_tokens.is_empty() {
            self.infer(prompt_tokens, prompt_tokens.len());
        }

        // Decode phase: generate tokens one by one
        let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens);
        let mut prev_decoded_len = 0;

        for _ in 0..max_new_tokens {
            // Download logits for the last position
            // After prefill: logits shape is [1, prompt_len, vocab], read last position
            // After decode: logits shape is [1, 1, vocab], always read position 0
            let logits_offset = if generated_tokens.is_empty() {
                // First iteration after prefill: logits has shape [1, prompt_len, vocab]
                if prompt_tokens.is_empty() {
                    0 // Empty prompt edge case
                } else {
                    (prompt_tokens.len() - 1) * VOCAB_SIZE
                }
            } else {
                // Decode iterations: logits has shape [1, 1, vocab] from most recent infer()
                0
            };

            let mut last_logits_f16 = vec![f16::ZERO; VOCAB_SIZE];
            // Download just the last position's logits (f16)
            let download_offset = logits_offset * size_of::<f16>();
            unsafe {
                tensorrt::cudaMemcpy(
                    last_logits_f16.as_mut_ptr() as *mut c_void,
                    (self.logits.ptr as usize + download_offset) as *mut c_void,
                    VOCAB_SIZE * size_of::<f16>(),
                    2, // D2H
                );
            }

            // Convert f16 to f32 for argmax
            let last_logits: Vec<f32> = last_logits_f16.iter().map(|x| x.to_f32()).collect();

            // Greedy sampling: argmax
            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap();

            // Check for EOS/EOT
            if next_token == EOS_TOKEN as u32 || next_token == EOT_TOKEN as u32 {
                break;
            }

            // Add to generated sequence
            generated_tokens.push(next_token);

            // Decode entire sequence and send delta
            let full_decoded = self
                .tokenizer
                .decode(&generated_tokens, true)
                .unwrap_or_else(|e| panic!("decode failed for tokens {:?}: {}", generated_tokens, e));

            if full_decoded.len() > prev_decoded_len {
                let delta = &full_decoded[prev_decoded_len..];
                prev_decoded_len = full_decoded.len();

                if !on_token(delta.to_string()) {
                    break;
                }
            }

            // Run inference with next token
            let next_token_i64 = next_token as i64;
            self.infer(&[next_token_i64], 1);
        }
    }
}

pub struct Handle<T: Clone + Send + 'static> {
    tx: std_mpsc::Sender<Input<T>>,
}

pub struct Listener<T: Clone + Send + 'static> {
    rx: tokio_mpsc::Receiver<Output<T>>,
}

pub fn create<T: Clone + Send + 'static>(trt: &Arc<tensorrt::Tensorrt>, epoch: &Arc<Epoch>) -> (Handle<T>, Listener<T>) {
    let (input_tx, input_rx) = std_mpsc::channel::<Input<T>>();
    let (output_tx, output_rx) = tokio_mpsc::channel::<Output<T>>(CHANNEL_CAPACITY);

    std::thread::spawn({
        let trt = Arc::clone(trt);
        let _epoch = Arc::clone(epoch);
        move || {
            let mut direct = Direct::new(&trt);

            while let Ok(input) = input_rx.recv() {
                // Tokenize prompt
                let encoding = direct
                    .tokenizer
                    .encode(input.prompt.clone(), false)
                    .unwrap_or_else(|e| panic!("tokenization failed for prompt '{}': {}", input.prompt, e));

                let prompt_tokens: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

                let max_tokens = input.max_tokens.min(MAX_SEQ_LEN - prompt_tokens.len());
                if input.max_tokens > max_tokens {
                    eprintln!(
                        "Warning: max_tokens capped from {} to {} due to context limit (prompt {} tokens)",
                        input.max_tokens,
                        max_tokens,
                        prompt_tokens.len()
                    );
                }

                // Generate tokens
                direct.generate(&prompt_tokens, max_tokens, |token_str| {
                    output_tx
                        .blocking_send(Output::Token {
                            payload: input.payload.clone(),
                            token: token_str,
                            stamp: input.stamp,
                        })
                        .is_ok()
                });

                // Send EOS
                let _ = output_tx.blocking_send(Output::Eos {
                    payload: input.payload.clone(),
                    stamp: input.stamp,
                });
            }
        }
    });

    (Handle { tx: input_tx }, Listener { rx: output_rx })
}

impl<T: Clone + Send + 'static> Handle<T> {
    pub fn send(&self, input: Input<T>) {
        self.tx.send(input).unwrap();
    }
}

impl<T: Clone + Send + 'static> Listener<T> {
    pub async fn recv(&mut self) -> Output<T> {
        self.rx.recv().await.unwrap()
    }

    pub fn try_recv(&mut self) -> Option<Output<T>> {
        match self.rx.try_recv() {
            Ok(output) => Some(output),
            Err(tokio_mpsc::error::TryRecvError::Empty) => None,
            Err(tokio_mpsc::error::TryRecvError::Disconnected) => {
                panic!("Llama3: output channel disconnected")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_constants() {
        // Verify architecture constants match Llama 3.2 3B config
        assert_eq!(NUM_LAYERS, 28, "Llama 3.2 3B has 28 layers");
        assert_eq!(NUM_KV_HEADS, 8, "Llama 3.2 3B uses 8 KV heads (GQA)");
        assert_eq!(HEAD_DIM, 128, "Head dimension is 128");
        assert_eq!(VOCAB_SIZE, 128256, "Llama 3.2 vocab size");
        assert_eq!(MAX_SEQ_LEN, 2048, "Max sequence length set to 2048");
    }

    #[test]
    fn test_token_ids_in_range() {
        // Verify special token IDs are within vocab range
        assert!(BOS_TOKEN >= 0 && (BOS_TOKEN as usize) < VOCAB_SIZE);
        assert!(EOS_TOKEN >= 0 && (EOS_TOKEN as usize) < VOCAB_SIZE);
        assert!(EOT_TOKEN >= 0 && (EOT_TOKEN as usize) < VOCAB_SIZE);
    }

    #[test]
    fn test_kv_cache_buffer_size() {
        // Verify KV cache buffer calculation (f16 KV cache)
        let expected = NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM * size_of::<f16>();
        assert_eq!(KV_BYTES, expected);

        // Each layer has 2 KV tensors (key + value)
        let total_kv_buffers = NUM_LAYERS * 2;
        assert_eq!(total_kv_buffers, 56, "Should have 56 KV cache tensors");
    }

    #[test]
    fn test_kv_cache_indexing() {
        // Property test: verify KV cache indexing never goes out of bounds
        for layer in 0..NUM_LAYERS {
            let k_idx = layer * 2;
            let v_idx = layer * 2 + 1;

            assert!(k_idx < NUM_LAYERS * 2, "Key index in bounds for layer {}", layer);
            assert!(v_idx < NUM_LAYERS * 2, "Value index in bounds for layer {}", layer);
        }
    }

    #[test]
    fn test_channel_capacity() {
        // Verify channel capacity is reasonable
        assert!(CHANNEL_CAPACITY > 0);
        assert!(CHANNEL_CAPACITY <= 1024, "Channel capacity shouldn't be excessive");
    }
}
