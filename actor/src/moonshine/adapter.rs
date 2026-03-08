use {super::*, std::sync::Arc};

pub struct Adapter {
    session: onnx::Session,
}

impl Adapter {
    pub fn new(onnx: &Arc<onnx::Onnx>, executor: onnx::Executor) -> Self {
        let session = onnx.create_session(executor, onnx::OptimizationLevel::EnableAll, 4, MOONSHINE_ADAPTER_PATH);
        Self { session }
    }

    pub fn adapt(&self, encoded: &[f32], num_frames: usize, pos_offset: i64) -> Vec<f32> {
        let input = onnx::Value::from_slice(&self.session.onnx, &[1, num_frames, ENCODER_DIM], encoded);
        let pos = onnx::Value::from_slice(&self.session.onnx, &[1], &[pos_offset]);
        let outputs = self.session.run(
            &[("encoded", &input), ("pos_offset", &pos)],
            &["memory"],
        );
        outputs[0].extract_as_f32()
    }
}
