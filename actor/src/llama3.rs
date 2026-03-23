use {
    crate::*,
    half::f16,
    rand::Rng,
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
const MAX_SEQ_LEN: usize = 512;

// Token IDs
const BOS_TOKEN: i32 = 128000;
const EOS_TOKEN: i32 = 128001;
const EOT_TOKEN: i32 = 128009;

// KV cache buffer size per tensor (k or v) at max_seq_len
// Shape: [1, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM] f16
const KV_BYTES: usize = 1 * NUM_KV_HEADS * MAX_SEQ_LEN * HEAD_DIM * size_of::<f16>();

pub struct Input<T: Clone + Send + 'static> {
    pub payload: T,        // pass-along payload
    pub prompt: String,    // the prompt to generate from
    pub max_tokens: usize, // maximum number of output tokens
    pub temperature: f32,  // sampling temperature (0.0 = greedy, >0 = stochastic)
    pub stamp: u64,        // epoch timestamp
}

pub enum Output<T: Clone + Send + 'static> {
    Token {
        payload: T,    // pass-along payload from input prompt
        token: String, // the generated token as string
        stamp: u64,    // epoch timestamp
    },
    Eos {
        payload: T, // pass-salong payload from input prompt
        stamp: u64, // epoch timestamp
    },
}

struct Direct {
    stream: *mut c_void,
    context: Arc<tensorrt::Context>,
    input_ids: tensorrt::Buffer,
    position_ids: tensorrt::Buffer,
    logits: tensorrt::Buffer,
    k_cache: [Vec<tensorrt::Buffer>; 2],
    v_cache: [Vec<tensorrt::Buffer>; 2],
    kv_cache_index: usize,
    tokenizer: Tokenizer,
    past_len: usize,
}

unsafe impl Send for Direct {}

impl Direct {
    fn new(trt: &Arc<tensorrt::Tensorrt>) -> Self {
        let engine = trt.load_engine(ENGINE_PATH);

        let context = engine.create_context();

        let mut stream: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { tensorrt::ffi::cudaStreamCreate(&mut stream) };
        assert!(rc == 0, "cudaStreamCreate failed: {rc}");

        let i64_size = size_of::<i64>();

        let input_ids = tensorrt::Buffer::new(MAX_SEQ_LEN * i64_size);
        let position_ids = tensorrt::Buffer::new(MAX_SEQ_LEN * i64_size);
        let logits = tensorrt::Buffer::new(MAX_SEQ_LEN * VOCAB_SIZE * size_of::<f16>());

        let mut k_cache = [Vec::with_capacity(NUM_LAYERS), Vec::with_capacity(NUM_LAYERS)];
        let mut v_cache = [Vec::with_capacity(NUM_LAYERS), Vec::with_capacity(NUM_LAYERS)];
        for _ in 0..(NUM_LAYERS) {
            k_cache[0].push(tensorrt::Buffer::new(KV_BYTES));
            v_cache[0].push(tensorrt::Buffer::new(KV_BYTES));
            k_cache[1].push(tensorrt::Buffer::new(KV_BYTES));
            v_cache[1].push(tensorrt::Buffer::new(KV_BYTES));
        }

        let tokenizer = Tokenizer::from_file(TOKENIZER_PATH).unwrap_or_else(|e| panic!("failed to load tokenizer: {e}"));

        Self {
            stream,
            context,
            input_ids,
            position_ids,
            logits,
            k_cache,
            v_cache,
            kv_cache_index: 0,
            tokenizer,
            past_len: 0,
        }
    }

    fn infer(&mut self, input_token_ids: &[i64], seq_len: usize) {
        self.input_ids.upload(input_token_ids);
        let positions: Vec<i64> = (self.past_len as i64..(self.past_len + seq_len) as i64).collect();
        self.position_ids.upload(&positions);
        self.context.set_input_shape("input_ids", &[1, seq_len as i64]);
        self.context.set_input_shape("position_ids", &[1, seq_len as i64]);
        self.context.set_tensor_address("input_ids", self.input_ids.ptr);
        self.context.set_tensor_address("position_ids", self.position_ids.ptr);
        self.context.set_tensor_address("logits", self.logits.ptr);

        for layer in 0..NUM_LAYERS {
            let past_k_name = format!("past_key_values.{}.key", layer);
            let past_v_name = format!("past_key_values.{}.value", layer);
            let present_k_name = format!("present.{}.key", layer);
            let present_v_name = format!("present.{}.value", layer);

            // Set KV cache shapes
            self.context
                .set_input_shape(&past_k_name, &[1, NUM_KV_HEADS as i64, self.past_len as i64, HEAD_DIM as i64]);
            self.context
                .set_input_shape(&past_v_name, &[1, NUM_KV_HEADS as i64, self.past_len as i64, HEAD_DIM as i64]);

            // Bind addresses
            self.context
                .set_tensor_address(&past_k_name, self.k_cache[1 - self.kv_cache_index][layer].ptr);
            self.context
                .set_tensor_address(&past_v_name, self.v_cache[1 - self.kv_cache_index][layer].ptr);
            self.context
                .set_tensor_address(&present_k_name, self.k_cache[self.kv_cache_index][layer].ptr);
            self.context
                .set_tensor_address(&present_v_name, self.v_cache[self.kv_cache_index][layer].ptr);
        }

        // Execute
        self.context.enqueue(self.stream);
        unsafe { tensorrt::ffi::cudaStreamSynchronize(self.stream) };

        // Update state
        self.past_len += seq_len;
        self.kv_cache_index = 1 - self.kv_cache_index;
    }

    /// Generate tokens autoregressively. Returns generated token strings via callback.
    fn generate<F>(&mut self, prompt_tokens: &[i64], max_new_tokens: usize, temperature: f32, mut on_token: F)
    where
        F: FnMut(String) -> bool, // Returns true to continue, false to stop
    {
        // Reset state
        self.past_len = 0;
        self.kv_cache_index = 0;

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
                tensorrt::ffi::cudaMemcpy(
                    last_logits_f16.as_mut_ptr() as *mut c_void,
                    (self.logits.ptr as usize + download_offset) as *mut c_void,
                    VOCAB_SIZE * size_of::<f16>(),
                    2, // D2H
                );
            }

            // Convert f16 to f32
            let last_logits: Vec<f32> = last_logits_f16.iter().map(|x| x.to_f32()).collect();

            let next_token = if temperature <= 0.0 {
                // Greedy: argmax
                last_logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx as u32)
                    .unwrap()
            } else {
                // Temperature sampling with softmax
                let max_logit = last_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut probs: Vec<f32> = last_logits.iter().map(|&l| ((l - max_logit) / temperature).exp()).collect();
                let sum: f32 = probs.iter().sum();
                for p in probs.iter_mut() {
                    *p /= sum;
                }
                let r: f32 = rand::thread_rng().r#gen();
                let mut cumulative = 0.0;
                let mut sampled = probs.len() as u32 - 1;
                for (i, &p) in probs.iter().enumerate() {
                    cumulative += p;
                    if r < cumulative {
                        sampled = i as u32;
                        break;
                    }
                }
                sampled
            };

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
                direct.generate(&prompt_tokens, max_tokens, input.temperature, |token_str| {
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

    pub async fn build_prompt(
        &self,
        identity: &str,
        personality: &str,
        tools: &str,
        facts: &str,
        history: &history::History,
    ) -> String {
        let (summary, history) = history.summarize(5).await;
        let mut short_history = String::new();
        for (role, message) in history.iter() {
            let role = match role {
                history::Role::Robot => "assistant",
                history::Role::User(_) => "user", // TODO: process ID
            };
            short_history.push_str(&format!(
                "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                role, message,
            ));
        }
        format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}\n{}\n{}\n{}\n{}<|eot_id|>{}<|start_header_id|>assistant<|end_header_id|>\n\n",
            identity, personality, tools, facts, summary, short_history,
        )
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
