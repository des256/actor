use {super::*, std::sync::Arc};

pub struct Encoder {
    session: onnx::Session,
}

impl Encoder {
    pub fn new(onnx: &Arc<onnx::Onnx>, executor: onnx::Executor) -> Self {
        let session = onnx.create_session(executor, onnx::OptimizationLevel::EnableAll, 4, MOONSHINE_ENCODER_PATH);
        Self { session }
    }

    pub fn encode(&self, features: &[f32], window_size: usize) -> Vec<f32> {
        let input = onnx::Value::from_slice(&self.session.onnx, &[1, window_size, ENCODER_DIM], features);
        let outputs = self.session.run(&[("features", &input)], &["encoded"]);
        outputs[0].extract_as_f32()
    }
}
