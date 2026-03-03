use {
    crate::*,
    std::{
        fs,
        sync::{Arc, mpsc as std_mpsc},
    },
    tokio::sync::mpsc as tokio_mpsc,
};

const PARAKEET_ENCODER_PATH: &str = "data/asr/encoder.onnx";
const PARAKEET_DECODER_JOINT_PATH: &str = "data/asr/decoder_joint.onnx";
const PARAKEET_TOKENIZER_PATH: &str = "data/asr/tokenizer.model";
const MEL_FILTERBANK_PATH: &str = "data/asr/mel_filterbank.bin";
const HANN_WINDOW_PATH: &str = "data/asr/hann_window.bin";

const NUM_MEL_BINS: usize = 128; // 128 mel filterbank bins
const FFT_SIZE: usize = 512; // size of FFT
const SPECTRUM_BINS: usize = FFT_SIZE / 2 + 1; // number of bins in converted spectrum
const WINDOW_SIZE: usize = 400; // 25ms at 16kHz
const BLANK_ID: i64 = 1024; // token ID for blank token

const DECODER_STATE_DIM: usize = 640;
const NUM_LAYERS: usize = 24;
const ENCODER_DIM: usize = 1024;

const CACHE_CHANNEL_CONTEXT: usize = 70;
const CACHE_TIME_CONTEXT: usize = 8;

const HOP_SIZE: usize = 160; // 10ms at 16kHz
const PRE_EMPHASIS: f32 = 0.97;
const LOG_ZERO_GUARD: f32 = 5.960_464_5e-08;

const MAX_SYMBOLS_PER_STEP: usize = 16;

const VOCAB_SIZE: usize = 1025; // 1024 tokens + 1 blank

const CHANNEL_CAPACITY: usize = 64;

const DEFAULT_ENCODER_WINDOW_SIZE: usize = 121;
const DEFAULT_ENCODER_CHUNK_SHIFT: usize = 112;

pub enum AsrInput<T: Clone + Send + 'static> {
    Audio { payload: T, data: Vec<i16> },
    Flush { payload: T },
}

pub enum AsrOutput<T: Clone + Send + 'static> {
    Utterance { payload: T, utterance: String },
    Flush { payload: T },
}

pub struct AsrHandle<T: Clone + Send + 'static> {
    input_tx: std_mpsc::Sender<AsrInput<T>>,
}

pub struct AsrListener<T: Clone + Send + 'static> {
    output_rx: tokio_mpsc::Receiver<AsrOutput<T>>,
}

pub fn create_asr<T: Clone + Send + 'static>(
    onnx: &Arc<Onnx>,
    executor: &Executor,
) -> (AsrHandle<T>, AsrListener<T>) {
    let mut encoder = onnx.create_session(
        executor,
        &OptimizationLevel::EnableAll,
        4,
        PARAKEET_ENCODER_PATH,
    );
    let mut decoder_joint = onnx.create_session(
        executor,
        &OptimizationLevel::EnableAll,
        4,
        PARAKEET_DECODER_JOINT_PATH,
    );
    let mut metadata = encoder.metadata();
    let encoder_window_size: usize = metadata
        .get("window_size")
        .unwrap_or(&DEFAULT_ENCODER_WINDOW_SIZE);
    let encoder_chunk_shift: usize = metadata
        .get("chunk_shift")
        .unwrap_or(&DEFAULT_ENCODER_CHUNK_SHIFT);
    let data = fs::read(MEL_FILTERBANK_PATH).unwrap();
    let mel_filterbank: Vec<f32> = data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    if mel_filterbank.len() != NUM_MEL_BINS * SPECTRUM_BINS {
        panic!(
            "Mel filterbank size mismatch ({} != {})",
            mel_filterbank.len(),
            NUM_MEL_BINS * SPECTRUM_BINS
        );
    }
    let data = fs::read(HANN_WINDOW_PATH).unwrap();
    let hann_window: Vec<f32> = data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    if hann_window.len() != encoder_window_size {
        panic!(
            "Hann window size mismatch ({} != {})",
            hann_window.len(),
            encoder_window_size
        );
    }
    let tokenizer_tokens = load_tokens(&PARAKEET_TOKENIZER_PATH);
    let (input_tx, input_rx) = std_mpsc::channel::<AsrInput<T>>();
    let (output_tx, output_rx) = tokio_mpsc::channel::<AsrOutput<T>>(CHANNEL_CAPACITY);
    std::thread::spawn({});
    (AsrHandle { input_tx }, AsrListener { output_rx })
}

impl<T: Clone + Send + 'static> AsrHandle<T> {
    pub fn send(&self, input: AsrInput<T>) {
        self.input_tx.send(input).unwrap();
    }
}

impl<T: Clone + Send + 'static> AsrListener<T> {
    pub async fn recv(&mut self) -> AsrOutput<T> {
        self.output_rx.recv().await.unwrap()
    }

    pub fn try_recv(&mut self) -> Option<AsrOutput<T>> {
        match self.output_rx.try_recv() {
            Ok(output) => Some(output),
            Err(tokio_mpsc::error::TryRecvError::Empty) => None,
            Err(tokio_mpsc::error::TryRecvError::Disconnected) => {
                panic!("Asr: data channel disconnected")
            }
        }
    }
}
