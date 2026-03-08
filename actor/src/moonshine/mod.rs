use crate::*;

const MOONSHINE_FRONTEND_PATH: &str = "data/moonshine/source/frontend.ort";
const MOONSHINE_ENCODER_PATH: &str = "data/moonshine/source/encoder.ort";
const MOONSHINE_DECODER_KV_PATH: &str = "data/moonshine/source/decoder_kv.ort";
const MOONSHINE_CROSS_KV_PATH: &str = "data/moonshine/source/cross_kv.ort";
const MOONSHINE_ADAPTER_PATH: &str = "data/moonshine/source/adapter.ort";
const MOONSHINE_TOKENIZER_PATH: &str = "data/moonshine/source/tokenizer.bin";

// Model dimensions (medium streaming model)
const ENCODER_DIM: usize = 768;
const DECODER_DIM: usize = 640;
const DEPTH: usize = 14;
const NHEADS: usize = 10;
const HEAD_DIM: usize = 64;
const VOCAB_SIZE: usize = 32768;
const BOS_ID: i64 = 1;
const EOS_ID: i64 = 2;
const TOTAL_LOOKAHEAD: usize = 16;
const D_MODEL_FRONTEND: usize = 768;
const C1: usize = 1536;
const MAX_SEQ_LEN: usize = 448;
const LEFT_CONTEXT_PER_LAYER: usize = 16;
const AUDIO_CHUNK_SIZE: usize = 1280;

pub enum Input<T: Clone + Send + 'static> {
    Initial { payload: T, audio: Vec<i16> },
    Continuing { payload: T, audio: Vec<i16>, flush: bool },
}

pub enum Output<T: Clone + Send + 'static> {
    Partial { payload: T, utterance: String },
    Final { payload: T, utterance: String },
}

mod moonshine;
pub use moonshine::*;

mod frontend;
use frontend::*;

mod encoder;
use encoder::*;

mod decoder;
use decoder::*;

mod cross;
use cross::*;

mod adapter;
use adapter::*;

mod tokenizer;
use tokenizer::*;
