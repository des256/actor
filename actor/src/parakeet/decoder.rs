use super::*;

pub struct Decoder {
    session: onnx::Session,
    state1: onnx::Value,
    state2: onnx::Value,
    last_token: i64,
}

impl Decoder {
    pub fn new(onnx: &Arc<onnx::Onnx>, executor: onnx::Executor) -> Self {
        let session = onnx.create_session(executor, onnx::OptimizationLevel::EnableAll, 4, PARAKEET_DECODER_PATH);
        let state1 = onnx::Value::zeros::<f32>(&onnx, &[2, 1, DECODER_STATE_DIM as i64]);
        let state2 = onnx::Value::zeros::<f32>(&onnx, &[2, 1, DECODER_STATE_DIM as i64]);
        Self {
            session,
            state1,
            state2,
            last_token: BLANK_ID,
        }
    }

    pub fn decode(&mut self, encoder_frames: &[f32]) -> Vec<i64> {
        let mut tokens = Vec::new();
        let mut transposed_frame = vec![0.0f32; ENCODER_OUTPUT_DIM];

        // loop over every encoder frame
        let num_frames = encoder_frames.len() / ENCODER_OUTPUT_DIM;
        for i in 0..num_frames {
            // extract transposed frame
            for d in 0..ENCODER_OUTPUT_DIM {
                transposed_frame[d] = encoder_frames[d * num_frames + i];
            }

            // create `encoder_outputs` tensor from transposed frame
            let encoder_outputs = onnx::Value::from_slice(&self.session.onnx, &[1, ENCODER_OUTPUT_DIM, 1], &transposed_frame);

            // decode up to `MAX_SYMBOLS_PER_STEP` tokens
            for _ in 0..MAX_SYMBOLS_PER_STEP {
                // extract current decoder state so it can be restored at blank token
                let state1_data = self.state1.extract_tensor::<f32>().to_vec();
                let state2_data = self.state2.extract_tensor::<f32>().to_vec();

                // create `targets` and `target_length` tensors
                let targets = onnx::Value::from_slice(&self.session.onnx, &[1, 1], &[self.last_token as i32]);
                let target_length = onnx::Value::from_slice(&self.session.onnx, &[1], &[1i32]);

                // run decoder
                let mut outputs = self.session.run(
                    &[
                        ("encoder_outputs", &encoder_outputs),
                        ("targets", &targets),
                        ("target_length", &target_length),
                        ("input_states_1", &self.state1),
                        ("input_states_2", &self.state2),
                    ],
                    &["outputs", "prednet_lengths", "output_states_1", "output_states_2"],
                );

                // extract logits
                let logits = outputs[0].extract_tensor::<f32>().to_vec();

                // update decoder state
                self.state1 = outputs.remove(2);
                self.state2 = outputs.remove(2);

                // find most likely token
                let valid_range = &logits[..VOCAB_SIZE.min(logits.len())];
                let predicted_token = valid_range
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx as i64)
                    .unwrap_or(0);

                // if blank, restore state and exit loop
                if predicted_token == BLANK_ID {
                    self.state1 = onnx::Value::from_slice(&self.session.onnx, &[2, 1, DECODER_STATE_DIM], &state1_data);
                    self.state2 = onnx::Value::from_slice(&self.session.onnx, &[2, 1, DECODER_STATE_DIM], &state2_data);
                    break;
                }

                // otherwise add token
                tokens.push(predicted_token);
                self.last_token = predicted_token;
            }
        }
        tokens
    }

    pub fn reset(&mut self) {
        self.state1 = onnx::Value::zeros::<f32>(&self.session.onnx, &[2, 1, DECODER_STATE_DIM as i64]);
        self.state2 = onnx::Value::zeros::<f32>(&self.session.onnx, &[2, 1, DECODER_STATE_DIM as i64]);
        self.last_token = BLANK_ID;
    }
}
