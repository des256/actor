use {
    crate::*,
    std::{
        sync::{Arc, mpsc as std_mpsc},
        time::Instant,
    },
    tokenizers::Tokenizer,
    tokio::sync::mpsc as tokio_mpsc,
};

const CHANNEL_CAPACITY: usize = 64;

const ENGINE_PATH: &str = "data/llama3/engine";
const TOKENIZER_PATH: &str = "data/llama3/engine/tokenizer.json";

const BOS_TOKEN: i32 = 128000;
const EOS_TOKEN: i32 = 128001;
const EOT_TOKEN: i32 = 128009;
const MAX_NEW_TOKENS: i32 = 512;

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

pub struct Direct {
    executor: Arc<tensorrt::Executor>,
    tokenizer: Tokenizer,
}

impl Direct {
    pub fn new() -> Self {
        let tokenizer = Tokenizer::from_file(TOKENIZER_PATH).unwrap();
        let config = tensorrt::ExecutorConfig {
            model_type: tensorrt::ModelType::DecoderOnly,
            kv_cache_free_gpu_mem_fraction: 0.5,
            max_beam_width: 1,
        };
        let executor = Arc::new(tensorrt::Executor::new(ENGINE_PATH, &config));
        Self { executor, tokenizer }
    }
}

pub struct Handle<T: Clone + Send + 'static> {
    tx: std_mpsc::Sender<Input<T>>,
}

pub struct Listener<T: Clone + Send + 'static> {
    rx: tokio_mpsc::Receiver<Output<T>>,
}

pub fn create<T: Clone + Send + 'static>(epoch: &Arc<Epoch>) -> (Handle<T>, Listener<T>) {
    let (input_tx, input_rx) = std_mpsc::channel::<Input<T>>();
    let (output_tx, output_rx) = tokio_mpsc::channel::<Output<T>>(CHANNEL_CAPACITY);

    std::thread::spawn({
        let _epoch = Arc::clone(epoch);
        move || {
            let direct = Direct::new();
            while let Ok(input) = input_rx.recv() {
                let encoding: tokenizers::Encoding = match direct.tokenizer.encode(input.prompt, false) {
                    Ok(encoding) => encoding,
                    Err(error) => {
                        panic!("Llama3: failed to tokenize prompt: {}", error);
                    }
                };
                let mut input_tokens: Vec<i32> = Vec::with_capacity(encoding.get_ids().len() + 1);
                input_tokens.push(BOS_TOKEN);
                input_tokens.extend(encoding.get_ids().iter().map(|&token| token as i32));
                let sampling = tensorrt::SamplingParams::default();
                let request = direct.executor.enqueue(
                    &input_tokens,
                    MAX_NEW_TOKENS,
                    &sampling,
                    Some(EOT_TOKEN),
                    Some(EOS_TOKEN),
                    true,
                );
                loop {
                    let (tokens, is_final) = direct.executor.await_response(request as u64, 0);

                    if is_final || tokens.iter().any(|&t| t == EOS_TOKEN || t == EOT_TOKEN) {
                        break;
                    }

                    let token_ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
                    let token_str = direct.tokenizer.decode(&token_ids, true).unwrap_or_default();

                    if let Err(error) = output_tx.blocking_send(Output::Token {
                        payload: input.payload.clone(),
                        token: token_str,
                        stamp: input.stamp,
                    }) {
                        panic!("Llama3: failed to send token: {}", error);
                    }
                }
                if let Err(error) = output_tx.blocking_send(Output::Eos {
                    payload: input.payload.clone(),
                    stamp: input.stamp,
                }) {
                    panic!("Llama3: failed to send EOS: {}", error);
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
                panic!("Llama3: output channel disconnected")
            }
        }
    }
}
