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

pub fn create<T: Clone + Send + 'static>(
    onnx: &Arc<onnx::Onnx>,
    executor: onnx::Executor,
) -> (Handle<T>, Listener<T>) {
    let (input_tx, input_rx) = std_mpsc::channel::<Input<T>>();
    let (output_tx, output_rx) = tokio_mpsc::channel::<Output<T>>(CHANNEL_CAPACITY);

    let mut frontend = Frontend::new(&onnx, executor);
    let encoder = Encoder::new(&onnx, executor);
    let decoder = Decoder::new(&onnx, executor);
    let cross = Cross::new(&onnx, executor);
    let adapter = Adapter::new(&onnx, executor);
    let tokenizer = Tokenizer::new();

    std::thread::spawn({
        move || {
            let mut accumulated_features: Vec<f32> = Vec::new();
            let mut accumulated_feature_count: usize = 0;
            let mut encoder_frames_emitted: usize = 0;
            let mut adapter_pos_offset: i64 = 0;
            let mut memory: Vec<f32> = Vec::new();
            let mut memory_len: usize = 0;
            let mut last_text = String::new();
            let mut sample_buffer: Vec<f32> = Vec::new();

            while let Ok(input) = input_rx.recv() {
                let (payload, audio, flush) = match input {
                    Input::Initial { payload, audio } => {
                        frontend.reset();
                        accumulated_features.clear();
                        accumulated_feature_count = 0;
                        encoder_frames_emitted = 0;
                        adapter_pos_offset = 0;
                        memory.clear();
                        memory_len = 0;
                        last_text.clear();
                        sample_buffer.clear();
                        (payload, audio, false)
                    }
                    Input::Continuing { payload, audio, flush } => (payload, audio, flush),
                };

                // Convert and buffer audio samples
                if !audio.is_empty() {
                    sample_buffer.extend(audio.iter().map(|&s| s as f32 / 32768.0));
                }

                // Process complete AUDIO_CHUNK_SIZE chunks through frontend
                while sample_buffer.len() >= AUDIO_CHUNK_SIZE {
                    let chunk: Vec<f32> = sample_buffer.drain(..AUDIO_CHUNK_SIZE).collect();
                    let new_features = frontend.process(&chunk);
                    let new_count = new_features.len() / ENCODER_DIM;
                    if new_count > 0 {
                        accumulated_features.extend_from_slice(&new_features);
                        accumulated_feature_count += new_count;
                    }
                }

                // On flush, pad remaining samples to AUDIO_CHUNK_SIZE and process
                if flush && !sample_buffer.is_empty() {
                    sample_buffer.resize(AUDIO_CHUNK_SIZE, 0.0);
                    let new_features = frontend.process(&sample_buffer);
                    let new_count = new_features.len() / ENCODER_DIM;
                    if new_count > 0 {
                        accumulated_features.extend_from_slice(&new_features);
                        accumulated_feature_count += new_count;
                    }
                    sample_buffer.clear();
                }

                // Determine how many features are stable (not in lookahead)
                let stable_count = if flush {
                    accumulated_feature_count
                } else {
                    accumulated_feature_count.saturating_sub(TOTAL_LOOKAHEAD)
                };
                let new_frames = stable_count.saturating_sub(encoder_frames_emitted);

                if new_frames > 0 {
                    // Build encoder sliding window with left context
                    let left_context = LEFT_CONTEXT_PER_LAYER * DEPTH;
                    let window_start = encoder_frames_emitted.saturating_sub(left_context);
                    let window_size = accumulated_feature_count - window_start;

                    let window_data = &accumulated_features
                        [window_start * ENCODER_DIM..(window_start + window_size) * ENCODER_DIM];

                    // Run encoder on the full window
                    let encoded = encoder.encode(window_data, window_size);

                    // Extract only the newly stable frames from encoder output
                    let extract_start = (encoder_frames_emitted - window_start) * ENCODER_DIM;
                    let extract_end = extract_start + new_frames * ENCODER_DIM;
                    let new_encoded = &encoded[extract_start..extract_end];

                    // Project through adapter with positional offset
                    let new_memory = adapter.adapt(new_encoded, new_frames, adapter_pos_offset);

                    // Append to memory
                    memory.extend_from_slice(&new_memory);
                    memory_len += new_frames;
                    encoder_frames_emitted += new_frames;
                    adapter_pos_offset += new_frames as i64;

                    // Compute cross-attention KV from full memory
                    let (k_cross, v_cross) = cross.compute(&memory, memory_len);

                    // Full greedy decode from BOS
                    let duration_sec = memory_len as f32 * 0.020;
                    let max_tokens = ((duration_sec * 6.5).ceil() as usize).min(MAX_SEQ_LEN);
                    let tokens = decoder.decode(max_tokens, &k_cross, &v_cross);

                    last_text = tokenizer.decode(&tokens);

                    if !last_text.is_empty() {
                        if let Err(e) = output_tx.blocking_send(Output::Partial {
                            payload: payload.clone(),
                            utterance: last_text.clone(),
                        }) {
                            panic!("Moonshine: failed to send partial: {}", e);
                        }
                    }
                }

                if flush {
                    if let Err(e) = output_tx.blocking_send(Output::Final {
                        payload: payload.clone(),
                        utterance: last_text.clone(),
                    }) {
                        panic!("Moonshine: failed to send final: {}", e);
                    }
                    frontend.reset();
                    accumulated_features.clear();
                    accumulated_feature_count = 0;
                    encoder_frames_emitted = 0;
                    adapter_pos_offset = 0;
                    memory.clear();
                    memory_len = 0;
                    last_text.clear();
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
                panic!("Moonshine: output channel disconnected")
            }
        }
    }
}
