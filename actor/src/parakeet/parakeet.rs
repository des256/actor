use {
    super::*,
    std::sync::{Arc, mpsc as std_mpsc},
    tokio::sync::mpsc as tokio_mpsc,
};

const CHANNEL_CAPACITY: usize = 64;

pub struct Handle<T: Clone + Send + 'static> {
    tx: std_mpsc::Sender<Input<T>>,
}

pub struct Listener<T: Clone + Send + 'static> {
    rx: tokio_mpsc::Receiver<Output<T>>,
}

pub fn create<T: Clone + Send + 'static>(onnx: &Arc<onnx::Onnx>, executor: onnx::Executor) -> (Handle<T>, Listener<T>) {
    // create channels
    let (input_tx, input_rx) = std_mpsc::channel::<parakeet::Input<T>>();
    let (output_tx, output_rx) = tokio_mpsc::channel::<parakeet::Output<T>>(CHANNEL_CAPACITY);

    // load models
    let mut feature_extractor = FeatureExtractor::new();
    let mut encoder = Encoder::new(&onnx, executor);
    let mut decoder = Decoder::new(&onnx, executor);
    let tokenizer = Tokenizer::new();

    // start audio pump
    std::thread::spawn({
        move || {
            let mut audio_buffer = Vec::new();
            let mut accumulator = String::new();

            // input pump
            while let Ok(input) = input_rx.recv() {
                // append audio to audio buffer
                audio_buffer.extend_from_slice(&input.audio);

                // on flush, make sure there is atleast one window available
                if input.flush && audio_buffer.len() < WINDOW_SIZE {
                    audio_buffer.resize(WINDOW_SIZE, 0);
                }

                // while a full window is available
                while audio_buffer.len() >= WINDOW_SIZE {
                    let window = audio_buffer[..WINDOW_SIZE].to_vec();
                    audio_buffer.drain(..WINDOW_SHIFT);

                    feature_extractor.reset();
                    decoder.reset();

                    let features = feature_extractor.extract_features(&window);
                    let encoder_frames = encoder.encode(&features);
                    let tokens = decoder.decode(&encoder_frames);

                    let text = tokenizer.tokenize(&tokens);
                    println!("{}", text);
                }

                if input.flush {
                    // NOTE: for now this will return an empty string
                    if let Err(error) = output_tx.blocking_send(parakeet::Output::<T>::Final {
                        payload: input.payload.clone(),
                        utterance: accumulator.clone(),
                    }) {
                        panic!("Asr: failed to send final output: {}", error);
                    }
                    audio_buffer.clear();
                    feature_extractor.reset();
                    encoder.reset();
                    decoder.reset();
                    accumulator.clear();
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
                panic!("Asr: output channel disconnected")
            }
        }
    }
}
