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
    let (input_tx, input_rx) = std_mpsc::channel::<asr::Input<T>>();
    let (output_tx, output_rx) = tokio_mpsc::channel::<asr::Output<T>>(CHANNEL_CAPACITY);

    // load models
    let mut feature_extractor = FeatureExtractor::new();
    let mut encoder = Encoder::new(&onnx, executor);
    let mut decoder = Decoder::new(&onnx, executor);
    let tokenizer = Tokenizer::new();

    // start audio pump
    std::thread::spawn({
        move || {
            let mut features = Vec::new();
            let mut frames = 0usize;
            let mut accumulator = String::new();

            // input pump
            while let Ok(input) = input_rx.recv() {
                // extract features and append them to `features`
                if !input.audio.is_empty() {
                    let new_features = feature_extractor.extract_features(&input.audio);
                    let new_frames = new_features.len() / MEL_SIZE;
                    if new_frames > 0 {
                        features.extend_from_slice(&new_features);
                        frames += new_frames;
                    }
                }

                // on flush, make sure there is at least one full window available
                if input.flush && (frames > 0) && (frames < ENCODER_WINDOW_SIZE) {
                    let pad_frames = ENCODER_WINDOW_SIZE - frames;
                    features.resize(features.len() + pad_frames * MEL_SIZE, 0.0);
                    frames = ENCODER_WINDOW_SIZE;
                }

                // process full windows
                while frames >= ENCODER_WINDOW_SIZE {
                    // transpose first window
                    let window = &features[..ENCODER_WINDOW_SIZE * MEL_SIZE];
                    let mut transposed_window = vec![0.0f32; ENCODER_WINDOW_SIZE * MEL_SIZE];
                    for frame in 0..ENCODER_WINDOW_SIZE {
                        for bin in 0..MEL_SIZE {
                            transposed_window[bin * ENCODER_WINDOW_SIZE + frame] = window[frame * MEL_SIZE + bin];
                        }
                    }

                    // shift `features` by `encoder.chunk_shift()` frames
                    let shift_frames = ENCODER_CHUNK_SHIFT.min(frames);
                    let shift_floats = shift_frames * MEL_SIZE;
                    features.drain(..shift_floats);
                    frames -= shift_frames;

                    // encode transposed window
                    let encoder_frames = encoder.encode_window(&transposed_window);

                    // greedy decode into tokens
                    let tokens = decoder.decode(&encoder_frames);

                    // extract text from tokens
                    let text = tokenizer.tokenize(&tokens);

                    // add to accumulator and send as partial output
                    if text.chars().any(|c| c.is_alphanumeric()) {
                        accumulator.push_str(&text);
                        if let Err(error) = output_tx.blocking_send(asr::Output::<T>::Partial {
                            payload: input.payload.clone(),
                            utterance: accumulator.clone(),
                        }) {
                            panic!("Asr: failed to send partial output: {}", error);
                        }
                    }
                }

                // when flushing, send final output and reset everything
                if input.flush {
                    if let Err(error) = output_tx.blocking_send(asr::Output::<T>::Final {
                        payload: input.payload.clone(),
                        utterance: accumulator.clone(),
                    }) {
                        panic!("Asr: failed to send final output: {}", error);
                    }
                    feature_extractor.reset();
                    features.clear();
                    frames = 0;
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
