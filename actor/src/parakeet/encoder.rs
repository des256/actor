use {super::*, std::sync::Arc};

pub struct Encoder {
    session: onnx::Session,
    channel_tensor: onnx::Value,
    time_tensor: onnx::Value,
    channel_len_tensor: onnx::Value,
}

impl Encoder {
    pub fn new(onnx: &Arc<onnx::Onnx>, executor: onnx::Executor) -> Self {
        let session = onnx.create_session(executor, onnx::OptimizationLevel::EnableAll, 4, PARAKEET_ENCODER_PATH);
        Self {
            session,
            channel_tensor: onnx::Value::zeros::<f32>(
                &onnx,
                &[
                    1,
                    ENCODER_LAYERS as i64,
                    ENCODER_CHANNEL_DIM as i64,
                    ENCODER_OUTPUT_DIM as i64,
                ],
            ),
            time_tensor: onnx::Value::zeros::<f32>(
                &onnx,
                &[1, ENCODER_LAYERS as i64, ENCODER_OUTPUT_DIM as i64, ENCODER_TIME_DIM as i64],
            ),
            channel_len_tensor: onnx::Value::from_slice(&onnx, &[1], &[0i64]),
        }
    }

    // encode a single window
    pub fn encode_window(&mut self, window: &[f32]) -> Vec<f32> {
        // create `audio_signal` and `length` tensors
        let audio_signal = onnx::Value::from_slice(&self.session.onnx, &[1, MEL_SIZE, ENCODER_WINDOW_SIZE], &window);
        let length = onnx::Value::from_slice(&self.session.onnx, &[1], &[ENCODER_WINDOW_SIZE as i64]);

        // run encoder
        let mut outputs = self.session.run(
            &[
                ("audio_signal", &audio_signal),
                ("length", &length),
                ("cache_last_channel", &self.channel_tensor),
                ("cache_last_time", &self.time_tensor),
                ("cache_last_channel_len", &self.channel_len_tensor),
            ],
            &[
                "outputs",
                "encoded_lengths",
                "cache_last_channel_next",
                "cache_last_time_next",
                "cache_last_channel_next_len",
            ],
        );

        // extract encoder output
        let encoder_frames = outputs[0].extract_as_f32();

        // update encoder state
        self.channel_tensor = outputs.remove(2);
        self.time_tensor = outputs.remove(2);
        self.channel_len_tensor = outputs.remove(2);

        encoder_frames
    }

    pub fn reset(&mut self) {
        // reset encoder state
        self.channel_tensor = onnx::Value::zeros::<f32>(
            &self.session.onnx,
            &[
                1,
                ENCODER_LAYERS as i64,
                ENCODER_CHANNEL_DIM as i64,
                ENCODER_OUTPUT_DIM as i64,
            ],
        );
        self.time_tensor = onnx::Value::zeros::<f32>(
            &self.session.onnx,
            &[1, ENCODER_LAYERS as i64, ENCODER_OUTPUT_DIM as i64, ENCODER_TIME_DIM as i64],
        );
        self.channel_len_tensor = onnx::Value::from_slice(&self.session.onnx, &[1], &[0i64]);
    }
}
