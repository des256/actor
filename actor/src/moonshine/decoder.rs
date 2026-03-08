use {super::*, std::sync::Arc};

pub struct Decoder {
    session: onnx::Session,
}

impl Decoder {
    pub fn new(onnx: &Arc<onnx::Onnx>, executor: onnx::Executor) -> Self {
        let session = onnx.create_session(executor, onnx::OptimizationLevel::EnableAll, 4, MOONSHINE_DECODER_KV_PATH);
        Self { session }
    }

    pub fn decode(&self, max_tokens: usize, k_cross: &onnx::Value, v_cross: &onnx::Value) -> Vec<i64> {
        // Start with empty self-attention KV cache: [DEPTH, 1, NHEADS, 0, HEAD_DIM]
        let mut k_self = onnx::Value::zeros::<f32>(
            &self.session.onnx,
            &[DEPTH as i64, 1, NHEADS as i64, 0, HEAD_DIM as i64],
        );
        let mut v_self = onnx::Value::zeros::<f32>(
            &self.session.onnx,
            &[DEPTH as i64, 1, NHEADS as i64, 0, HEAD_DIM as i64],
        );

        let mut tokens = Vec::new();
        let mut current_token = BOS_ID;

        for _ in 0..max_tokens {
            let token_value = onnx::Value::from_slice(&self.session.onnx, &[1, 1], &[current_token]);

            let mut outputs = self.session.run(
                &[
                    ("token", &token_value),
                    ("k_self", &k_self),
                    ("v_self", &v_self),
                    ("out_k_cross", k_cross),
                    ("out_v_cross", v_cross),
                ],
                &["logits", "out_k_self", "out_v_self"],
            );

            let logits = outputs[0].extract_as_f32();

            // Update self-attention KV cache
            k_self = outputs.remove(1);
            v_self = outputs.remove(1);

            // Argmax over vocabulary
            let next_token = logits
                .iter()
                .take(VOCAB_SIZE)
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as i64)
                .unwrap_or(0);

            if next_token == EOS_ID {
                break;
            }

            tokens.push(next_token);
            current_token = next_token;
        }

        tokens
    }
}
