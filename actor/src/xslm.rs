use {
    super::*,
    std::sync::{Arc, mpsc as std_mpsc},
    tokenizers::Tokenizer,
    tokio::sync::mpsc as tokio_mpsc,
};

const MAX_TOKENS: usize = 512;

const SMOLLM2_360M_MODEL_PATH: &str = "data/xslm/smollm2_360m/model_q4f16.onnx";
const SMOLLM2_360M_TOKENIZER_PATH: &str = "data/xslm/smollm2_360m/tokenizer.json";
const SMOLLM2_360M_EOS_TOKENS: &[u32] = &[0, 2];
const SMOLLM2_360M_NUM_KV_HEADS: usize = 5;
const SMOLLM2_360M_HEAD_DIM: usize = 64;

const CHANNEL_CAPACITY: usize = 64;

#[derive(Clone, Copy)]
pub enum Model {
    Smollm2360m,
}

#[derive(Clone)]
pub struct Input<T: Clone + Send + 'static> {
    pub payload: T,
    pub prompt: String,
    pub stamp: u64,
}

pub enum Output<T: Clone + Send + 'static> {
    Token { payload: T, token: String, stamp: u64 },
    Eos { payload: T, stamp: u64 },
}

pub struct Handle<T: Clone + Send + 'static> {
    tx: std_mpsc::Sender<Input<T>>,
}

pub struct Listener<T: Clone + Send + 'static> {
    rx: tokio_mpsc::Receiver<Output<T>>,
}

pub fn create<T: Clone + Send + 'static>(
    onnx: &Arc<onnx::Onnx>,
    executor: onnx::Executor,
    model: Model,
    epoch: &Arc<Epoch>,
) -> (Handle<T>, Listener<T>) {
    // create channels
    let (input_tx, input_rx) = std_mpsc::channel::<Input<T>>();
    let (output_tx, output_rx) = tokio_mpsc::channel::<Output<T>>(CHANNEL_CAPACITY);

    // get model-specific parameters
    let (model_path, tokenizer_path, eos_tokens, num_kv_heads, head_dim) = match model {
        Model::Smollm2360m => (
            SMOLLM2_360M_MODEL_PATH,
            SMOLLM2_360M_TOKENIZER_PATH,
            SMOLLM2_360M_EOS_TOKENS,
            SMOLLM2_360M_NUM_KV_HEADS,
            SMOLLM2_360M_HEAD_DIM,
        ),
    };

    // load models
    let mut session = onnx.create_session(executor, onnx::OptimizationLevel::EnableAll, 4, model_path);
    let tokenizer = match Tokenizer::from_file(tokenizer_path) {
        Ok(tokenizer) => tokenizer,
        Err(error) => {
            panic!("Xslm: failed to load tokenizer: {}", error);
        }
    };
    let tokenizer = Arc::new(tokenizer);

    // start prompt pump
    std::thread::spawn({
        let epoch = Arc::clone(&epoch);
        move || {
            // extract input names
            let mut input_names = Vec::<String>::new();
            for i in 0..session.input_count() {
                input_names.push(session.input_name(i));
            }

            // extract output names
            let mut output_names = Vec::<String>::new();
            for i in 0..session.output_count() {
                output_names.push(session.output_name(i));
            }

            // count KV keys from input names
            let kv_key_count = input_names.iter().filter(|name| name.contains(".key")).count();
            let kv_value_count = input_names.iter().filter(|name| name.contains(".value")).count();
            if kv_key_count != kv_value_count {
                panic!("Xslm: KV key and value count mismatch");
            }
            let first_kv_idx = input_names
                .iter()
                .position(|name| name.contains(".key") || name.contains(".value"))
                .unwrap();
            let kv_dtype = session.input_element_type(first_kv_idx);

            // prompt pump
            while let Ok(input) = input_rx.recv() {
                // skip if stale
                if !epoch.is_current(input.stamp) {
                    continue;
                }

                // tokenize prompt
                let encoding = match tokenizer.encode(input.prompt.as_str(), false) {
                    Ok(encoding) => encoding,
                    Err(error) => {
                        panic!("Xslm: failed to tokenize prompt: {}", error);
                    }
                };
                let tokens = encoding.get_ids().to_vec();
                if tokens.is_empty() {
                    continue; // silently ignore empty prompts
                }

                // initial state
                let mut input_ids: Vec<i64> = tokens.iter().map(|&token| token as i64).collect();
                let mut attention_mask: Vec<i64> = vec![1; tokens.len()];
                let mut positions: Vec<i64> = (0..tokens.len() as i64).collect();

                // track generated tokens for incremental decoding
                let mut generated_tokens = Vec::<u32>::new();
                let mut prev_decoded_len = 0;

                // create KV tensor cache
                let mut kv_cache: Vec<onnx::Value> = Vec::with_capacity(kv_key_count * 2);
                for _ in 0..kv_key_count * 2 {
                    kv_cache.push(onnx::Value::empty_typed(
                        &session.onnx,
                        &[1, num_kv_heads, 0, head_dim],
                        kv_dtype,
                    ));
                }

                // generate tokens
                for _ in 0..MAX_TOKENS {
                    // build input tensors
                    let input_ids_tensor = onnx::Value::from_slice::<i64>(&session.onnx, &[1, input_ids.len()], &input_ids);
                    let attention_mask_tensor =
                        onnx::Value::from_slice::<i64>(&session.onnx, &[1, attention_mask.len()], &attention_mask);
                    let positions_tensor = onnx::Value::from_slice::<i64>(&session.onnx, &[1, positions.len()], &positions);
                    let mut inputs = Vec::with_capacity(3 + kv_cache.len());
                    for name in &input_names {
                        if name == "input_ids" {
                            inputs.push((name.as_str(), &input_ids_tensor));
                        } else if name == "attention_mask" {
                            inputs.push((name.as_str(), &attention_mask_tensor));
                        } else if name == "position_ids" {
                            inputs.push((name.as_str(), &positions_tensor));
                        } else {
                            let idx_str = name.strip_prefix("past_key_values.").unwrap();
                            let dot_pos = idx_str.find('.').unwrap();
                            let layer: usize = idx_str[..dot_pos].parse().unwrap();
                            let cache_idx = match &idx_str[dot_pos + 1..] {
                                "key" => layer * 2,
                                "value" => layer * 2 + 1,
                                _ => panic!("Xslm: invalid KV cache index: {}", idx_str),
                            };
                            if cache_idx < kv_cache.len() {
                                inputs.push((name.as_str(), &kv_cache[cache_idx]));
                            }
                        }
                    }
                    if inputs.len() != input_names.len() {
                        panic!("Xslm: missing input tensors");
                    }

                    // run model
                    let output_refs = output_names.iter().map(|name| name.as_str()).collect::<Vec<_>>();
                    let outputs = session.run(&inputs, &output_refs);

                    // exit if stale
                    if !epoch.is_current(input.stamp) {
                        break;
                    }

                    // extract token
                    let logits_idx = output_names.iter().position(|name| name == "logits").unwrap();
                    let logits_data = outputs[logits_idx].extract_as_f32();
                    let seq_len = input_ids.len();
                    if (seq_len == 0) || logits_data.is_empty() {
                        panic!("Xslm: invalid logits");
                    }
                    let vocab_size = logits_data.len() / seq_len;
                    let last_pos_offset = (seq_len - 1) * vocab_size;
                    if last_pos_offset + vocab_size > logits_data.len() {
                        panic!(
                            "Xslm: logits shape mismatch, expected {} but got {}",
                            last_pos_offset + vocab_size,
                            logits_data.len()
                        );
                    }
                    let last_pos_logits = &logits_data[last_pos_offset..last_pos_offset + vocab_size];
                    let next_token = match last_pos_logits
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(idx, _)| idx as i64)
                    {
                        Some(id) => id,
                        None => {
                            panic!("Xslm: no valid logits found");
                        }
                    };

                    // if EOS, send to output channel and exit loop
                    if eos_tokens.contains(&(next_token as u32)) {
                        output_tx
                            .blocking_send(Output::Eos {
                                payload: input.payload.clone(),
                                stamp: epoch.current(),
                            })
                            .unwrap();
                        break;
                    }

                    // add token to generated_tokens
                    generated_tokens.push(next_token as u32);

                    // decode entire sequence to output new part as token
                    let full_decoded = match tokenizer.decode(&generated_tokens, true) {
                        Ok(decoded) => decoded,
                        Err(error) => {
                            panic!("Xslm: failed to decode entire sequence: {}", error);
                        }
                    };
                    if full_decoded.len() > prev_decoded_len {
                        let delta = &full_decoded[prev_decoded_len..];
                        prev_decoded_len = full_decoded.len();
                        if let Err(error) = output_tx.blocking_send(Output::Token {
                            payload: input.payload.clone(),
                            token: delta.to_string(),
                            stamp: epoch.current(),
                        }) {
                            panic!("Xslm: failed to send token: {}", error);
                        }
                    }

                    // update KV cache
                    kv_cache = outputs
                        .into_iter()
                        .enumerate()
                        .filter(|(i, _)| *i != logits_idx)
                        .map(|(_, v)| v)
                        .collect();

                    // prepare for next iteration
                    input_ids = vec![next_token as i64];
                    attention_mask.push(1);
                    positions = vec![attention_mask.len() as i64 - 1];
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
                panic!("Xslm: output channel disconnected")
            }
        }
    }
}
