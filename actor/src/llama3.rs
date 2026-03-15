use {
    crate::*,
    std::{
        sync::{Arc, mpsc as std_mpsc},
        time::Instant,
    },
    tokio::sync::mpsc as tokio_mpsc,
};

const CHANNEL_CAPACITY: usize = 64;

const ENGINE_PATH: &str = "data/llama3_3b/engine/model.engine";
const TOKENIZER_PATH: &str = "data/llama3_3b/engine/tokenizer.json";

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
    context: Arc<tensorrt::Context>,
}

impl Direct {
    pub fn new(model_engine: &Arc<tensorrt::Engine>) -> Self {
        let context = model_engine.create_context();

        Self { context }
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

    let model_engine = tensorrt.load_engine(ENGINE_PATH);
    std::thread::spawn({
        move || {
            let mut direct = Direct::new(&model_engine);
            while let Ok(input) = input_rx.recv() {
                let start = Instant::now();
                let encoding: tokenizers::Encoding = match tokenizer.encode(input.prompt, false) {
                    Ok(encoding) => encoding,
                    Err(error) => {
                        panic!("Llama3: failed to tokenize prompt: {}", error);
                    }
                };
                let mut input_tokens: Vec<i32> = Vec::with_capacity(encoding.get_ids().len() + 1);
                input_tokens.push(BOS_TOKEN);
                input_tokens.extend(encoding.get_ids().iter().map(|&token| token as i32));
                let sampling = SamplingParams::default();
                let request = executor.enqueue(&tokens, MAX_NEW_TOKENS, &sampling, Some(EOT_TOKEN), Some(EOS_TOKEN), true);
                let mut all_tokens: Vec<i32> = Vec::new();
                let mut ttft_ms = 0u64;
                loop {
                    let (tokens, is_final) = executor.await_response(request, 0);
                    if !tokens.is_empty() && ttft_ms == 0 {
                        ttft_ms = start.elapsed().as_millis() as u64;
                    }
                    // TODO: convert token to string
                    if let Err(error) = output_tx.blocking_send(Output::Token {
                        payload: input.payload.clone(),
                        token: "".to_string(),
                        stamp: input.stamp,
                    }) {
                        panic!("Llama3: failed to send token: {}", error);
                    }
                    // TODO: if EOS or EOT, exit loop
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
