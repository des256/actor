use crate::*;

const MOONSHINE_FRONTEND_PATH: &str = "data/moonshine/frontend.onnx";
const MOONSHINE_ENCODER_PATH: &str = "data/moonshine/encoder.onnx";
const MOONSHINE_DECODER_KV_PATH: &str = "data/moonshine/decoder_kv.onnx";
const MOONSHINE_CROSS_KV_PATH: &str = "data/moonshine/cross_kv.onnx";
const MOONSHINE_ADAPTER_PATH: &str = "data/moonshine/adapter.onnx";
const MOONSHINE_TOKENIZER_PATH: &str = "data/moonshine/source/tokenizer.bin";

const SAMPLES_PER_FRAME: usize = 320;
const FEATURE_DIM: usize = 768;
const DEPTH: usize = 14;
const NHEADS: usize = 10;
const HEAD_DIM: usize = 64;
const VOCAB_SIZE: usize = 32768;
const BOS_ID: i64 = 1;
const EOS_ID: i64 = 2;
const WINDOW_SIZE: usize = 100;
const WINDOW_SHIFT: usize = 20;
const MAX_TOKENS: usize = 64;
const CONV1_SIZE: usize = 768;
const CONV2_SIZE: usize = 1536;
const REPETITION_PENALTY: f32 = 1.2;
const NO_REPEAT_NGRAM: usize = 3;

pub struct Input<T: Clone + Send + 'static> {
    pub payload: T,
    pub audio: Vec<i16>,
    pub flush: bool,
}

pub enum Output<T: Clone + Send + 'static> {
    Partial { payload: T, utterance: String },
    Final { payload: T, utterance: String },
}

mod moonshine;
pub use moonshine::*;
