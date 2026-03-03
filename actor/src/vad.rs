use {crate::*, std::sync::Arc};

const SILERO_VAD_MODEL_PATH: &str = "data/silero_vad.onnx";

const FRAME_SIZE: usize = 512;
const CONTEXT_SIZE: usize = 64;

pub struct Vad {
    session: Session,
    state_tensor: Value,
    sample_rate_tensor: Value,
    input_tensor: Value,
}

impl Vad {
    pub fn new(onnx: &Arc<Onnx>, executor: &Executor, sample_rate: usize) -> Self {
        let session = onnx.create_session(
            executor,
            &OptimizationLevel::EnableAll,
            4,
            SILERO_VAD_MODEL_PATH,
        );
        let state_tensor = Value::zeros::<f32>(&onnx, &[2, 1, 128]);
        let sample_rate_tensor = Value::from_slice::<i64>(&onnx, &[1], &[sample_rate as i64]);
        let input_tensor = Value::zeros::<f32>(&onnx, &[1, (CONTEXT_SIZE + FRAME_SIZE) as i64]);
        Self {
            session,
            state_tensor,
            sample_rate_tensor,
            input_tensor,
        }
    }

    pub fn analyze(&mut self, frame: &[i16]) -> f32 {
        let slice = self.input_tensor.as_slice_mut::<f32>();
        let mut context = [0f32; CONTEXT_SIZE];
        context.copy_from_slice(&slice[FRAME_SIZE..]);
        slice[..CONTEXT_SIZE].copy_from_slice(&context);
        for i in 0..FRAME_SIZE {
            slice[CONTEXT_SIZE + i] = frame[i] as f32 / 32768.0;
        }
        let inputs = [
            ("input", &self.input_tensor),
            ("state", &self.state_tensor),
            ("sr", &self.sample_rate_tensor),
        ];
        let mut outputs = self.session.run(&inputs, &["output", "stateN"]);
        let probability = outputs[0].extract_tensor::<f32>()[0];
        self.state_tensor = outputs.split_off(1).remove(0);
        probability
    }
}
