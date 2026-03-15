use {
    crate::*,
    std::{
        ffi::c_void,
        ptr::null_mut,
        sync::{Arc, mpsc as std_mpsc},
    },
    tokenizers::Tokenizer,
    tokio::sync::mpsc as tokio_mpsc,
};

const MOONSHINE_ENCODER_PATH: &str = "data/moonshine/engine/encoder.engine";
const MOONSHINE_DECODER_PATH: &str = "data/moonshine/engine/decoder.engine";
const MOONSHINE_TOKENIZER_PATH: &str = "data/moonshine/source/tokenizer.json";

const SAMPLES_PER_FRAME: usize = 320;
const WINDOW_SIZE: usize = 100;
const WINDOW_SHIFT: usize = 20;
const AUDIO_SIZE: usize = SAMPLES_PER_FRAME * WINDOW_SIZE;
const AUDIO_SHIFT: usize = SAMPLES_PER_FRAME * WINDOW_SHIFT;
const VOCAB_SIZE: usize = 32768;
const BOS_ID: i64 = 1;
const EOS_ID: i64 = 2;
const MAX_TOKENS: usize = 64;
const REPETITION_PENALTY: f32 = 1.2;
const NO_REPEAT_NGRAM: usize = 3;
const DEPTH: usize = 14;
const NHEADS: usize = 10;
const HEAD_DIM: usize = 64;
const ENC_DIM: usize = 768;

const KV_MAX_BYTES: usize = DEPTH * NHEADS * MAX_TOKENS * HEAD_DIM * size_of::<f32>();

const CHANNEL_CAPACITY: usize = 64;

pub struct Input<T: Clone + Send + 'static> {
    pub payload: T,
    pub audio: Vec<i16>,
    pub flush: bool,
}

pub enum Output<T: Clone + Send + 'static> {
    Partial { payload: T, utterance: String },
    Final { payload: T, utterance: String },
}

fn find_longest_common_substring(s1: &str, s2: &str) -> Option<(usize, usize, usize)> {
    let (len1, len2) = (s1.len(), s2.len());
    let mut table = vec![vec![0usize; len2 + 1]; len1 + 1];
    let (mut max_len, mut end1, mut end2) = (0, 0, 0);
    let (b1, b2) = (s1.as_bytes(), s2.as_bytes());
    for i in 1..=len1 {
        for j in 1..=len2 {
            if b1[i - 1] == b2[j - 1] {
                table[i][j] = table[i - 1][j - 1] + 1;
                if table[i][j] > max_len {
                    max_len = table[i][j];
                    end1 = i - 1;
                    end2 = j - 1;
                }
            }
        }
    }
    if max_len == 0 { None } else { Some((max_len, end1, end2)) }
}

struct Direct {
    encoder_context: Arc<tensorrt::Context>,
    decoder_context: Arc<tensorrt::Context>,
    stream: *mut c_void,
    encoder_inputs: tensorrt::Buffer,
    encoder_attention_mask: tensorrt::Buffer,
    encoder_hidden: tensorrt::Buffer,
    encoder_output_mask: tensorrt::Buffer,
    decoder_input_ids: tensorrt::Buffer,
    decoder_encoder_mask: tensorrt::Buffer,
    decoder_logits: tensorrt::Buffer,
    cache_keys: [tensorrt::Buffer; 2],
    cache_values: [tensorrt::Buffer; 2],
    tokenizer: Tokenizer,
}

impl Direct {
    pub fn new(encoder_engine: &Arc<tensorrt::Engine>, decoder_engine: &Arc<tensorrt::Engine>) -> Self {
        let encoder_context = encoder_engine.create_context();
        let decoder_context = decoder_engine.create_context();
        let mut stream: *mut c_void = null_mut();
        let _ = unsafe { tensorrt::cudaStreamCreate(&mut stream) };
        let encoder_inputs = tensorrt::Buffer::new(AUDIO_SIZE * size_of::<f32>());
        let encoder_attention_mask = tensorrt::Buffer::new(AUDIO_SIZE * size_of::<i64>());
        let encoder_hidden = tensorrt::Buffer::new(WINDOW_SIZE * ENC_DIM * size_of::<f32>());
        let encoder_output_mask = tensorrt::Buffer::new(WINDOW_SIZE * size_of::<i64>());
        let decoder_input_ids = tensorrt::Buffer::new(size_of::<i64>());
        let decoder_encoder_mask = tensorrt::Buffer::new(WINDOW_SIZE * size_of::<i64>());
        let decoder_logits = tensorrt::Buffer::new(VOCAB_SIZE * size_of::<f32>());
        let cache_keys = [tensorrt::Buffer::new(KV_MAX_BYTES), tensorrt::Buffer::new(KV_MAX_BYTES)];
        let cache_values = [tensorrt::Buffer::new(KV_MAX_BYTES), tensorrt::Buffer::new(KV_MAX_BYTES)];
        let tokenizer = Tokenizer::from_file(MOONSHINE_TOKENIZER_PATH).unwrap();
        Self {
            encoder_context,
            decoder_context,
            stream,
            encoder_inputs,
            encoder_attention_mask,
            encoder_hidden,
            encoder_output_mask,
            decoder_input_ids,
            decoder_encoder_mask,
            decoder_logits,
            cache_keys,
            cache_values,
            tokenizer,
        }
    }

    pub fn transcribe(&mut self, audio: &[f32]) -> String {
        let attention_mask = vec![1i64; AUDIO_SIZE];
        self.encoder_inputs.upload(&audio);
        self.encoder_attention_mask.upload(&attention_mask);
        self.encoder_context.set_input_shape("input_values", &[1, AUDIO_SIZE as i64]);
        self.encoder_context
            .set_input_shape("attention_mask", &[1, AUDIO_SIZE as i64]);
        self.encoder_context
            .set_tensor_address("input_values", self.encoder_inputs.ptr);
        self.encoder_context
            .set_tensor_address("attention_mask", self.encoder_attention_mask.ptr);
        self.encoder_context
            .set_tensor_address("last_hidden_state", self.encoder_hidden.ptr);
        self.encoder_context
            .set_tensor_address("encoder_attention_mask", self.encoder_output_mask.ptr);
        self.encoder_context.enqueue(self.stream);
        unsafe { tensorrt::cudaStreamSynchronize(self.stream) };
        self.decoder_encoder_mask.upload(&vec![1i64; WINDOW_SIZE]);
        let mut tokens: Vec<i64> = Vec::new();
        let mut current_token = BOS_ID;
        let mut past_len: usize = 0;
        let mut kv_index: usize = 0;
        for _ in 0..MAX_TOKENS {
            self.decoder_input_ids.upload(&[current_token]);
            self.decoder_context.set_input_shape("input_ids", &[1, 1]);
            self.decoder_context
                .set_input_shape("encoder_hidden_states", &[1, WINDOW_SIZE as i64, ENC_DIM as i64]);
            self.decoder_context
                .set_input_shape("encoder_attention_mask", &[1, WINDOW_SIZE as i64]);
            self.decoder_context
                .set_input_shape("k_self", &[DEPTH as i64, 1, NHEADS as i64, past_len as i64, HEAD_DIM as i64]);
            self.decoder_context
                .set_input_shape("v_self", &[DEPTH as i64, 1, NHEADS as i64, past_len as i64, HEAD_DIM as i64]);
            self.decoder_context
                .set_tensor_address("input_ids", self.decoder_input_ids.ptr);
            self.decoder_context
                .set_tensor_address("encoder_hidden_states", self.encoder_hidden.ptr);
            self.decoder_context
                .set_tensor_address("encoder_attention_mask", self.encoder_output_mask.ptr);
            self.decoder_context
                .set_tensor_address("k_self", self.cache_keys[kv_index].ptr);
            self.decoder_context
                .set_tensor_address("v_self", self.cache_values[kv_index].ptr);
            self.decoder_context.set_tensor_address("logits", self.decoder_logits.ptr);
            self.decoder_context
                .set_tensor_address("out_k_self", self.cache_keys[1 - kv_index].ptr);
            self.decoder_context
                .set_tensor_address("out_v_self", self.cache_values[1 - kv_index].ptr);
            self.decoder_context.enqueue(self.stream);
            unsafe { tensorrt::cudaStreamSynchronize(self.stream) };
            let mut logits = vec![0f32; VOCAB_SIZE];
            self.decoder_logits.download(&mut logits);
            for &token in &tokens {
                let index = token as usize;
                if index < VOCAB_SIZE {
                    if logits[index] > 0.0 {
                        logits[index] /= REPETITION_PENALTY;
                    } else {
                        logits[index] *= REPETITION_PENALTY;
                    }
                }
            }
            if (NO_REPEAT_NGRAM >= 2) && (tokens.len() >= NO_REPEAT_NGRAM - 1) {
                let prefix = &tokens[tokens.len() - (NO_REPEAT_NGRAM - 1)..];
                for w in tokens.windows(NO_REPEAT_NGRAM) {
                    if w[..NO_REPEAT_NGRAM - 1] == *prefix {
                        let blocked = w[NO_REPEAT_NGRAM - 1] as usize;
                        if blocked < VOCAB_SIZE {
                            logits[blocked] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(index, _)| index as i64)
                .unwrap_or(0);
            if next_token == EOS_ID {
                break;
            }
            tokens.push(next_token);
            current_token = next_token;
            past_len += 1;
            kv_index = 1 - kv_index;
        }
        let tokens: Vec<u32> = tokens.iter().map(|&token| token as u32).collect();
        self.tokenizer.decode(&tokens, true).unwrap_or_default()
    }
}

pub struct Handle<T: Clone + Send + 'static> {
    tx: std_mpsc::Sender<Input<T>>,
}

pub struct Listener<T: Clone + Send + 'static> {
    rx: tokio_mpsc::Receiver<Output<T>>,
}

pub fn create<T: Clone + Send + 'static>(tensorrt: &Arc<tensorrt::Tensorrt>) -> (Handle<T>, Listener<T>) {
    let (input_tx, input_rx) = std_mpsc::channel::<Input<T>>();
    let (output_tx, output_rx) = tokio_mpsc::channel::<Output<T>>(CHANNEL_CAPACITY);

    let encoder_engine = tensorrt.load_engine(MOONSHINE_ENCODER_PATH);
    let decoder_engine = tensorrt.load_engine(MOONSHINE_DECODER_PATH);
    std::thread::spawn({
        move || {
            let mut direct = Direct::new(&encoder_engine, &decoder_engine);
            let mut samples = Vec::<f32>::new();
            let mut at_start = true;
            let mut accumulated_text = String::new();
            while let Ok(input) = input_rx.recv() {
                samples.extend(input.audio.iter().map(|&sample| sample as f32 / 32768.0));
                if at_start && samples.len() < AUDIO_SIZE {
                    let mut audio = samples.clone();
                    audio.resize(AUDIO_SIZE, 0.0);
                    let text = direct.transcribe(&audio);
                    if at_start {
                        accumulated_text = text.clone();
                    } else if let Some((_, end1, end2)) = find_longest_common_substring(&accumulated_text, &text) {
                        accumulated_text = format!("{}{}", &accumulated_text[..end1], &text[end2..]);
                    }
                    if let Err(error) = output_tx.blocking_send(Output::Partial {
                        payload: input.payload.clone(),
                        utterance: accumulated_text.clone(),
                    }) {
                        panic!("Moonshine: failed to send partial: {}", error);
                    }
                } else {
                    at_start = false;
                    while samples.len() >= AUDIO_SIZE {
                        let audio = &samples[..AUDIO_SIZE];
                        let text = direct.transcribe(audio);
                        samples.drain(..AUDIO_SHIFT);
                        if accumulated_text.is_empty() {
                            accumulated_text = text.clone();
                        } else if let Some((_, end1, end2)) = find_longest_common_substring(&accumulated_text, &text) {
                            accumulated_text = format!("{}{}", &accumulated_text[..end1], &text[end2..]);
                        }
                        if let Err(error) = output_tx.blocking_send(Output::Partial {
                            payload: input.payload.clone(),
                            utterance: accumulated_text.clone(),
                        }) {
                            panic!("Moonshine: failed to send partial: {}", error);
                        }
                    }
                    if input.flush && (samples.len() < AUDIO_SIZE) {
                        let mut audio = samples.clone();
                        audio.resize(AUDIO_SIZE, 0.0);
                        let text = direct.transcribe(&audio);
                        if accumulated_text.is_empty() {
                            accumulated_text = text.clone();
                        } else if let Some((_, end1, end2)) = find_longest_common_substring(&accumulated_text, &text) {
                            accumulated_text = format!("{}{}", &accumulated_text[..end1], &text[end2..]);
                        }
                        if let Err(error) = output_tx.blocking_send(Output::Partial {
                            payload: input.payload.clone(),
                            utterance: accumulated_text.clone(),
                        }) {
                            panic!("Moonshine: failed to send partial: {}", error);
                        }
                    }
                }
                if input.flush {
                    if let Err(error) = output_tx.blocking_send(Output::Final {
                        payload: input.payload.clone(),
                        utterance: accumulated_text.clone(),
                    }) {
                        panic!("Moonshine: failed to send final: {}", error);
                    }
                    at_start = true;
                    samples.clear();
                    accumulated_text.clear();
                }
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
                panic!("Moonshine: output channel disconnected")
            }
        }
    }
}
