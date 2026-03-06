use crate::*;

const CONDITIONER_PATH: &str = "data/tts/text_conditioner.onnx";
const FLOW_MAIN_PATH: &str = "data/tts/flow_lm_main_int8.onnx";
const FLOW_STEP_PATH: &str = "data/tts/flow_lm_flow_int8.onnx";
const DECODER_PATH: &str = "data/tts/mimi_decoder_int8.onnx";
const TOKENIZER_PATH: &str = "data/tts/tokenizer.json";

const MAX_TOKENS: usize = 1000;
const LATENT_DIM: usize = 32;
const CONDITIONING_DIM: usize = 1024;
const DEFAULT_TEMPERATURE: f32 = 0.7;
const DEFAULT_LSD_STEPS: usize = 1;
const DEFAULT_EOS_THRESHOLD: f32 = -4.0;

pub struct Input<T: Clone + Send + 'static> {
    pub payload: T,
    pub sentence: String,
    pub stamp: u64,
}

pub struct Output<T: Clone + Send + 'static> {
    pub payload: T,
    pub audio: Vec<i16>,
    pub index: usize,
    pub last: bool,
    pub stamp: u64,
}

mod tts;
pub use tts::*;

mod tokenizer;
use tokenizer::*;

mod encoder;
use encoder::*;

mod decoder;
use decoder::*;
