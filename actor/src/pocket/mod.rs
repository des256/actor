use crate::*;

const BACKBONE_PATH: &str = "data/pocket/engine/backbone.engine";
const FLOW_NET_PATH: &str = "data/pocket/engine/flow_net.engine";
const DECODER_PATH: &str = "data/pocket/engine/mimi_decoder.engine";
const METADATA_PATH: &str = "data/pocket/engine/metadata.safetensors";
const TOKENIZER_PATH: &str = "data/pocket/engine/tokenizer.json";

const MAX_TOKENS: usize = 100;
const LATENT_DIM: usize = 32;
const CONDITIONING_DIM: usize = 1024;
const DEFAULT_TEMPERATURE: f32 = 0.7;
const DEFAULT_EOS_THRESHOLD: f32 = -4.0;
const FRAMES_AFTER_EOS: usize = 3;

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

mod pocket;
pub use pocket::*;

mod tokenizer;
use tokenizer::*;

mod trt;
use trt::*;
