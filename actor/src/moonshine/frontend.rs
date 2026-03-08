use {super::*, std::sync::Arc};

pub struct Frontend {
    session: onnx::Session,
    sample_buffer: onnx::Value,
    sample_len: onnx::Value,
    conv1_buffer: onnx::Value,
    conv2_buffer: onnx::Value,
    frame_count: onnx::Value,
}

impl Frontend {
    pub fn new(onnx: &Arc<onnx::Onnx>, executor: onnx::Executor) -> Self {
        let session = onnx.create_session(executor, onnx::OptimizationLevel::EnableAll, 4, MOONSHINE_FRONTEND_PATH);
        Self {
            sample_buffer: onnx::Value::zeros::<f32>(onnx, &[1, 79]),
            sample_len: onnx::Value::zeros::<i64>(onnx, &[1]),
            conv1_buffer: onnx::Value::zeros::<f32>(onnx, &[1, D_MODEL_FRONTEND as i64, 4]),
            conv2_buffer: onnx::Value::zeros::<f32>(onnx, &[1, C1 as i64, 4]),
            frame_count: onnx::Value::zeros::<i64>(onnx, &[1]),
            session,
        }
    }

    /// Process an audio chunk through the frontend, returning feature frames
    /// as a flat [N * ENCODER_DIM] f32 vec. Updates internal causal convolution state.
    pub fn process(&mut self, audio: &[f32]) -> Vec<f32> {
        let audio_value = onnx::Value::from_slice(&self.session.onnx, &[1, audio.len()], audio);
        let mut outputs = self.session.run(
            &[
                ("audio_chunk", &audio_value),
                ("sample_buffer", &self.sample_buffer),
                ("sample_len", &self.sample_len),
                ("conv1_buffer", &self.conv1_buffer),
                ("conv2_buffer", &self.conv2_buffer),
                ("frame_count", &self.frame_count),
            ],
            &[
                "features",
                "sample_buffer_out",
                "sample_len_out",
                "conv1_buffer_out",
                "conv2_buffer_out",
                "frame_count_out",
            ],
        );
        let features = outputs[0].extract_as_f32();
        let _ = outputs.remove(0);
        self.sample_buffer = outputs.remove(0);
        self.sample_len = outputs.remove(0);
        self.conv1_buffer = outputs.remove(0);
        self.conv2_buffer = outputs.remove(0);
        self.frame_count = outputs.remove(0);
        features
    }

    pub fn reset(&mut self) {
        self.sample_buffer = onnx::Value::zeros::<f32>(&self.session.onnx, &[1, 79]);
        self.sample_len = onnx::Value::zeros::<i64>(&self.session.onnx, &[1]);
        self.conv1_buffer = onnx::Value::zeros::<f32>(&self.session.onnx, &[1, D_MODEL_FRONTEND as i64, 4]);
        self.conv2_buffer = onnx::Value::zeros::<f32>(&self.session.onnx, &[1, C1 as i64, 4]);
        self.frame_count = onnx::Value::zeros::<i64>(&self.session.onnx, &[1]);
    }
}
