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

    /// TDT label-looping decode.
    ///
    /// `encoder_frames` is row-major: encoder_frames[frame * ENCODER_OUTPUT_DIM + dim].
    /// Returns a vector of predicted token IDs.
    pub fn decode(&mut self, encoder_frames: &[f32]) -> Vec<i64> {
        let num_frames = encoder_frames.len() / ENCODER_OUTPUT_DIM;
        if num_frames == 0 {
            return Vec::new();
        }

        let mut tokens = Vec::new();
        let mut time_index: usize = 0;
        let mut symbols_added: usize = 0;

        while time_index < num_frames {
            // safety: force advance if too many symbols from same frame
            if symbols_added >= MAX_SYMBOLS_PER_STEP {
                time_index += 1;
                symbols_added = 0;
                continue;
            }

            // save decoder state for blank restoration
            let state1_data = self.state1.extract_tensor::<f32>().to_vec();
            let state2_data = self.state2.extract_tensor::<f32>().to_vec();

            // create encoder frame tensor [1, ENCODER_OUTPUT_DIM, 1]
            let frame_start = time_index * ENCODER_OUTPUT_DIM;
            let frame_data = &encoder_frames[frame_start..frame_start + ENCODER_OUTPUT_DIM];
            let encoder_outputs = onnx::Value::from_slice(
                &self.session.onnx,
                &[1, ENCODER_OUTPUT_DIM, 1],
                frame_data,
            );

            // create target tensors
            let targets = onnx::Value::from_slice(
                &self.session.onnx,
                &[1, 1],
                &[self.last_token as i32],
            );
            let target_length = onnx::Value::from_slice(
                &self.session.onnx,
                &[1],
                &[1i32],
            );

            // run decoder-joint
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

            // extract logits (8193 token logits + 5 duration logits = 8198)
            let logits = outputs[0].extract_as_f32();

            // update decoder state from model output
            self.state1 = outputs.remove(2);
            self.state2 = outputs.remove(2);

            // predict token: argmax over token logits [0..VOCAB_SIZE)
            let token_logits = &logits[..VOCAB_SIZE];
            let predicted_token = token_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as i64)
                .unwrap_or(0);

            // predict duration: argmax over duration logits [VOCAB_SIZE..VOCAB_SIZE+NUM_DURATIONS)
            let duration_logits = &logits[VOCAB_SIZE..VOCAB_SIZE + NUM_DURATIONS];
            let duration_index = duration_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            let mut duration = TDT_DURATIONS[duration_index];

            if predicted_token == BLANK_ID {
                // blank: restore decoder state, force duration >= 1
                self.state1 = onnx::Value::from_slice(
                    &self.session.onnx,
                    &[2, 1, DECODER_STATE_DIM],
                    &state1_data,
                );
                self.state2 = onnx::Value::from_slice(
                    &self.session.onnx,
                    &[2, 1, DECODER_STATE_DIM],
                    &state2_data,
                );
                if duration == 0 {
                    duration = 1;
                }
                time_index += duration;
                symbols_added = 0;
            } else {
                // non-blank: emit token, advance by duration
                tokens.push(predicted_token);
                self.last_token = predicted_token;
                time_index += duration;
                if duration > 0 {
                    symbols_added = 0;
                } else {
                    symbols_added += 1;
                }
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
