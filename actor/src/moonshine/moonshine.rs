use {
    super::*,
    std::sync::{Arc, mpsc as std_mpsc},
    tokio::sync::mpsc as tokio_mpsc,
};

const CHANNEL_CAPACITY: usize = 64;

fn transcribe(
    frontend_session: &onnx::Session,
    encoder_session: &onnx::Session,
    adapter_session: &onnx::Session,
    cross_session: &onnx::Session,
    decoder_session: &onnx::Session,
    tokenizer_vocab: &[Vec<u8>],
    window: &[f32],
) -> String {
    // frontend
    let audio_tensor = onnx::Value::from_slice(&frontend_session.onnx, &[1, window.len()], &window);
    let conv1_tensor = onnx::Value::zeros::<f32>(&frontend_session.onnx, &[1, CONV1_SIZE as i64, 4]);
    let conv2_tensor = onnx::Value::zeros::<f32>(&frontend_session.onnx, &[1, CONV2_SIZE as i64, 4]);
    let outputs = frontend_session.run(
        &[
            ("audio_chunk", &audio_tensor),
            ("conv1_buffer", &conv1_tensor),
            ("conv2_buffer", &conv2_tensor),
        ],
        &["features", "conv1_buffer_out", "conv2_buffer_out"],
    );
    let features = outputs[0].extract_as_f32();

    // encoder
    let features_tensor = onnx::Value::from_slice(&encoder_session.onnx, &[1, WINDOW_SIZE, FEATURE_DIM], &features);
    let outputs = encoder_session.run(&[("features", &features_tensor)], &["encoded"]);
    let encoded = outputs[0].extract_as_f32();

    // adapter
    let encoded_tensor = onnx::Value::from_slice(&adapter_session.onnx, &[1, WINDOW_SIZE, FEATURE_DIM], &encoded);
    let pos = onnx::Value::from_slice(&adapter_session.onnx, &[1], &[0i64]);
    let outputs = adapter_session.run(&[("encoded", &encoded_tensor), ("pos_offset", &pos)], &["memory"]);
    let memory = outputs[0].extract_as_f32();

    // cross-attention
    let memory_tensor = onnx::Value::from_slice(&cross_session.onnx, &[1, WINDOW_SIZE, NHEADS * HEAD_DIM], &memory);
    let mut outputs = cross_session.run(&[("memory", &memory_tensor)], &["k_cross", "v_cross"]);
    let v_cross = outputs.remove(1);
    let k_cross = outputs.remove(0);

    // decoder
    let tokens = decode(&decoder_session, MAX_TOKENS, &k_cross, &v_cross);

    // tokenizer
    let mut bytes = Vec::new();
    for &id in &tokens {
        let id = id as usize;
        // Skip special tokens: 0=<unk>, 1=<s> (BOS), 2=</s> (EOS)
        if id >= 3 && id < tokenizer_vocab.len() {
            bytes.extend_from_slice(&tokenizer_vocab[id]);
        }
    }
    String::from_utf8_lossy(&bytes)
        .into_owned()
        .replace("▁", " ")
        .trim()
        .to_string()
}

fn find_longest_common_substring(s1: &str, s2: &str) -> Option<(usize, usize, usize)> {
    let len1 = s1.len();
    let len2 = s2.len();
    let mut table = vec![vec![0; len2 + 1]; len1 + 1];
    let mut max_len = 0;
    let mut end_idx1 = 0;
    let mut end_idx2 = 0;
    let b1 = s1.as_bytes();
    let b2 = s2.as_bytes();
    for i in 1..=len1 {
        for j in 1..=len2 {
            if b1[i - 1] == b2[j - 1] {
                table[i][j] = table[i - 1][j - 1] + 1;
                if table[i][j] > max_len {
                    max_len = table[i][j];
                    end_idx1 = i - 1;
                    end_idx2 = j - 1;
                }
            }
        }
    }
    if max_len == 0 {
        None
    } else {
        Some((max_len, end_idx1, end_idx2))
    }
}

pub fn decode(session: &onnx::Session, max_tokens: usize, k_cross: &onnx::Value, v_cross: &onnx::Value) -> Vec<i64> {
    // Start with empty self-attention KV cache: [DEPTH, 1, NHEADS, 0, HEAD_DIM]
    let mut k_self = onnx::Value::zeros::<f32>(&session.onnx, &[DEPTH as i64, 1, NHEADS as i64, 0, HEAD_DIM as i64]);
    let mut v_self = onnx::Value::zeros::<f32>(&session.onnx, &[DEPTH as i64, 1, NHEADS as i64, 0, HEAD_DIM as i64]);

    let mut tokens = Vec::new();
    let mut current_token = BOS_ID;

    for _ in 0..max_tokens {
        let token_value = onnx::Value::from_slice(&session.onnx, &[1, 1], &[current_token]);

        let mut outputs = session.run(
            &[
                ("token", &token_value),
                ("k_self", &k_self),
                ("v_self", &v_self),
                ("out_k_cross", k_cross),
                ("out_v_cross", v_cross),
            ],
            &["logits", "out_k_self", "out_v_self"],
        );

        let mut logits = outputs[0].extract_as_f32();

        // Update self-attention KV cache
        k_self = outputs.remove(1);
        v_self = outputs.remove(1);

        // Repetition penalty: penalize previously generated tokens
        for &tok in &tokens {
            let idx = tok as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    logits[idx] /= REPETITION_PENALTY;
                } else {
                    logits[idx] *= REPETITION_PENALTY;
                }
            }
        }

        // No-repeat n-gram blocking: if generating token X would create an n-gram
        // that already appeared, set X's logit to -inf
        if NO_REPEAT_NGRAM >= 2 && tokens.len() >= NO_REPEAT_NGRAM - 1 {
            // The last (n-1) tokens form the current prefix
            let prefix = &tokens[tokens.len() - (NO_REPEAT_NGRAM - 1)..];
            // Scan history for matching prefixes and collect their continuations
            for window in tokens.windows(NO_REPEAT_NGRAM) {
                if window[..NO_REPEAT_NGRAM - 1] == *prefix {
                    let blocked = window[NO_REPEAT_NGRAM - 1] as usize;
                    if blocked < logits.len() {
                        logits[blocked] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        // Argmax over vocabulary
        let next_token = logits
            .iter()
            .take(VOCAB_SIZE)
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as i64)
            .unwrap_or(0);

        if next_token == EOS_ID {
            break;
        }

        tokens.push(next_token);
        current_token = next_token;
    }

    tokens
}

pub struct Handle<T: Clone + Send + 'static> {
    tx: std_mpsc::Sender<Input<T>>,
}

pub struct Listener<T: Clone + Send + 'static> {
    rx: tokio_mpsc::Receiver<Output<T>>,
}

pub fn create<T: Clone + Send + 'static>(onnx: &Arc<onnx::Onnx>, executor: onnx::Executor) -> (Handle<T>, Listener<T>) {
    let (input_tx, input_rx) = std_mpsc::channel::<Input<T>>();
    let (output_tx, output_rx) = tokio_mpsc::channel::<Output<T>>(CHANNEL_CAPACITY);

    let frontend_session = onnx.create_session(executor, onnx::OptimizationLevel::EnableAll, 4, MOONSHINE_FRONTEND_PATH);
    let encoder_session = onnx.create_session(executor, onnx::OptimizationLevel::EnableAll, 4, MOONSHINE_ENCODER_PATH);
    let adapter_session = onnx.create_session(executor, onnx::OptimizationLevel::EnableAll, 4, MOONSHINE_ADAPTER_PATH);
    let cross_session = onnx.create_session(executor, onnx::OptimizationLevel::EnableAll, 4, MOONSHINE_CROSS_KV_PATH);
    let decoder_session = onnx.create_session(executor, onnx::OptimizationLevel::EnableAll, 4, MOONSHINE_DECODER_KV_PATH);

    let mut tokenizer_vocab: Vec<Vec<u8>> = Vec::new();
    let data = std::fs::read(MOONSHINE_TOKENIZER_PATH).expect("Failed to read moonshine tokenizer");
    let mut pos = 0;
    while pos < data.len() {
        let len = data[pos] as usize;
        pos += 1;
        if pos + len > data.len() {
            break;
        }
        tokenizer_vocab.push(data[pos..pos + len].to_vec());
        pos += len;
    }

    std::thread::spawn({
        let onnx = Arc::clone(&onnx);
        move || {
            // initialize state
            let mut at_start = true;
            let mut samples = Vec::<f32>::new();
            let mut accumulated_text = String::new();

            while let Ok(input) = input_rx.recv() {
                // append audio to samples
                samples.extend(input.audio.iter().map(|&sample| sample as f32 / 32768.0));

                if at_start && samples.len() < SAMPLES_PER_FRAME * WINDOW_SIZE {
                    let mut window = samples.clone();
                    window.resize(SAMPLES_PER_FRAME * WINDOW_SIZE, 0.0);
                    let text = transcribe(
                        &frontend_session,
                        &encoder_session,
                        &adapter_session,
                        &cross_session,
                        &decoder_session,
                        &tokenizer_vocab,
                        &window,
                    );
                    if at_start {
                        accumulated_text = text.clone();
                    } else if let Some((_, end1, end2)) = find_longest_common_substring(&accumulated_text, &text) {
                        accumulated_text = format!("{}{}", &accumulated_text[..end1], &text[end2..]);
                    }
                    if let Err(error) = output_tx.blocking_send(Output::Partial {
                        payload: input.payload.clone(),
                        utterance: accumulated_text.clone(),
                    }) {
                        panic!("Moonshine: failed to send partial: {}", error);
                    }
                } else {
                    at_start = false;
                    while samples.len() >= SAMPLES_PER_FRAME * WINDOW_SIZE {
                        let window = Vec::<f32>::from(&samples[..SAMPLES_PER_FRAME * WINDOW_SIZE]);
                        samples.drain(..SAMPLES_PER_FRAME * WINDOW_SHIFT);
                        let text = transcribe(
                            &frontend_session,
                            &encoder_session,
                            &adapter_session,
                            &cross_session,
                            &decoder_session,
                            &tokenizer_vocab,
                            &window,
                        );
                        if accumulated_text.is_empty() {
                            accumulated_text = text.clone();
                        } else if let Some((_, end1, end2)) = find_longest_common_substring(&accumulated_text, &text) {
                            accumulated_text = format!("{}{}", &accumulated_text[..end1], &text[end2..]);
                        }
                        if let Err(error) = output_tx.blocking_send(Output::Partial {
                            payload: input.payload.clone(),
                            utterance: accumulated_text.clone(),
                        }) {
                            panic!("Moonshine: failed to send partial: {}", error);
                        }
                    }

                    if input.flush && (samples.len() < SAMPLES_PER_FRAME * WINDOW_SIZE) {
                        let mut window = samples.clone();
                        window.resize(SAMPLES_PER_FRAME * WINDOW_SIZE, 0.0);
                        let text = transcribe(
                            &frontend_session,
                            &encoder_session,
                            &adapter_session,
                            &cross_session,
                            &decoder_session,
                            &tokenizer_vocab,
                            &window,
                        );
                        if accumulated_text.is_empty() {
                            accumulated_text = text.clone();
                        } else if let Some((_, end1, end2)) = find_longest_common_substring(&accumulated_text, &text) {
                            accumulated_text = format!("{}{}", &accumulated_text[..end1], &text[end2..]);
                        }
                        if let Err(error) = output_tx.blocking_send(Output::Final {
                            payload: input.payload.clone(),
                            utterance: accumulated_text.clone(),
                        }) {
                            panic!("Moonshine: failed to send final: {}", error);
                        }
                    }
                }

                // finalize output if flush
                if input.flush {
                    if let Err(error) = output_tx.blocking_send(Output::Final {
                        payload: input.payload.clone(),
                        utterance: String::new(),
                    }) {
                        panic!("Moonshine: failed to send final: {}", error);
                    }
                    at_start = true;
                    samples.clear();
                    accumulated_text.clear();
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
