use {crate::*, std::sync::Arc};

const PARAKEET_ENCODER_PATH: &str = "data/parakeet/onnx/encoder.onnx";
const PARAKEET_DECODER_PATH: &str = "data/parakeet/onnx/decoder_joint.onnx";
const PARAKEET_TOKENIZER_PATH: &str = "data/parakeet/onnx/tokenizer.model";

const HANN_WINDOW_SIZE: usize = 400; // number of samples in Hann window
const FFT_SIZE: usize = 512; // number of bins in FFT
const SPECTRUM_SIZE: usize = FFT_SIZE / 2 + 1; // number of bins in spectrum
const MEL_SIZE: usize = 128; // number of bands in mel filterbank

const WINDOW_SIZE: usize = 64000; // number of samples in encoder window
const WINDOW_SHIFT: usize = 1600; // number of samples to shift between windows

const ENCODER_OUTPUT_DIM: usize = 1024; // encoder output vector size

const BLANK_ID: i64 = 8192; // token ID for blank token (vocab_size index)
const VOCAB_SIZE: usize = 8193; // 8192 tokens + 1 blank
const NUM_DURATIONS: usize = 5; // TDT duration logit count
const TDT_DURATIONS: [usize; 5] = [0, 1, 2, 3, 4]; // TDT duration values
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

mod parakeet;
pub use parakeet::*;

mod decoder;
use decoder::*;

mod encoder;
use encoder::*;

mod featureextractor;
use featureextractor::*;

mod tokenizer;
use tokenizer::*;
