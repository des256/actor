use {crate::*, std::sync::Arc};

const PARAKEET_ENCODER_PATH: &str = "data/asr/encoder.onnx";
const PARAKEET_DECODER_PATH: &str = "data/asr/decoder_joint.onnx";
const PARAKEET_TOKENIZER_PATH: &str = "data/asr/tokenizer.model";

const HANN_WINDOW_SIZE: usize = 400; // number of samples in Hann window
const FFT_SIZE: usize = 512; // number of bins in FFT
const SPECTRUM_SIZE: usize = FFT_SIZE / 2 + 1; // number of bins in spectrum
const MEL_SIZE: usize = 128; // number of bands in mel filterbank
const ENCODER_WINDOW_SIZE: usize = 121; // size of encoder window
const ENCODER_CHUNK_SHIFT: usize = 112; // number of frames to shift between chunks
const ENCODER_LAYERS: usize = 24; // number of layers in encoder
const ENCODER_OUTPUT_DIM: usize = 1024; // encoder output vector size
const ENCODER_CHANNEL_DIM: usize = 70; // channel-related context dimension
const ENCODER_TIME_DIM: usize = 8; // time-related context dimension
const BLANK_ID: i64 = 1024; // token ID for blank token
const VOCAB_SIZE: usize = 1025; // 1024 tokens + 1 blank
const DECODER_STATE_DIM: usize = 640; // decoder context dimension
const MAX_SYMBOLS_PER_STEP: usize = 16; // maximum number of tokens to decode per step

pub struct Input<T: Clone + Send + 'static> {
    pub payload: T,
    pub audio: Vec<i16>,
    pub flush: bool,
}

pub enum Output<T: Clone + Send + 'static> {
    Partial { payload: T, utterance: String },
    Final { payload: T, utterance: String },
}

mod asr;
pub use asr::*;

mod decoder;
use decoder::*;

mod encoder;
use encoder::*;

mod featureextractor;
use featureextractor::*;

mod tokenizer;
use tokenizer::*;
