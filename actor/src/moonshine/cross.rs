use {super::*, std::sync::Arc};

pub struct Cross {
    session: onnx::Session,
}

impl Cross {
    pub fn new(onnx: &Arc<onnx::Onnx>, executor: onnx::Executor) -> Self {
        let session = onnx.create_session(executor, onnx::OptimizationLevel::EnableAll, 4, MOONSHINE_CROSS_KV_PATH);
        Self { session }
    }

    pub fn compute(&self, memory: &[f32], memory_len: usize) -> (onnx::Value, onnx::Value) {
        let input = onnx::Value::from_slice(&self.session.onnx, &[1, memory_len, DECODER_DIM], memory);
        let mut outputs = self.session.run(
            &[("memory", &input)],
            &["k_cross", "v_cross"],
        );
        let v_cross = outputs.remove(1);
        let k_cross = outputs.remove(0);
        (k_cross, v_cross)
    }
}
