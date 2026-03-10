use {super::*, std::sync::Arc};

pub struct Encoder {
    session: onnx::Session,
}

impl Encoder {
    pub fn new(onnx: &Arc<onnx::Onnx>, executor: onnx::Executor) -> Self {
        let session = onnx.create_session(executor, onnx::OptimizationLevel::EnableAll, 4, PARAKEET_ENCODER_PATH);
        Self { session }
    }

    /// Encode mel features into encoder output frames.
    ///
    /// `features` is row-major: features[frame * MEL_SIZE + bin].
    /// Returns row-major encoder frames: result[frame * ENCODER_OUTPUT_DIM + dim].
    pub fn encode(&mut self, features: &[f32]) -> Vec<f32> {
        let num_frames = features.len() / MEL_SIZE;
        if num_frames == 0 {
            return Vec::new();
        }

        // transpose from row-major [frame, bin] to [1, bin, frame] for ONNX
        let mut transposed = vec![0.0f32; MEL_SIZE * num_frames];
        for frame in 0..num_frames {
            for bin in 0..MEL_SIZE {
                transposed[bin * num_frames + frame] = features[frame * MEL_SIZE + bin];
            }
        }

        // create ONNX tensors
        let audio_signal = onnx::Value::from_slice(
            &self.session.onnx,
            &[1, MEL_SIZE, num_frames],
            &transposed,
        );
        let length = onnx::Value::from_slice(
            &self.session.onnx,
            &[1],
            &[num_frames as i64],
        );

        // run encoder
        let outputs = self.session.run(
            &[("audio_signal", &audio_signal), ("length", &length)],
            &["outputs", "encoded_lengths"],
        );

        // extract output (dim-major: [1, ENCODER_OUTPUT_DIM, T_enc])
        let enc_output = outputs[0].extract_as_f32();
        let t_enc = enc_output.len() / ENCODER_OUTPUT_DIM;

        // transpose to row-major: result[frame * ENCODER_OUTPUT_DIM + dim]
        let mut result = vec![0.0f32; t_enc * ENCODER_OUTPUT_DIM];
        for frame in 0..t_enc {
            for dim in 0..ENCODER_OUTPUT_DIM {
                result[frame * ENCODER_OUTPUT_DIM + dim] = enc_output[dim * t_enc + frame];
            }
        }

        result
    }

    pub fn reset(&mut self) {}
}
