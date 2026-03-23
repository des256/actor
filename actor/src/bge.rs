use {super::*, std::sync::Arc, tokenizers::Tokenizer};

const MODEL_PATH: &str = "data/bge/onnx/model.onnx";
const TOKENIZER_PATH: &str = "data/bge/onnx/tokenizer.json";
const EMBEDDING_SIZE: usize = 384;
const MAX_SEQ_LEN: usize = 128;

pub struct Embedding(Vec<f32>);

impl Embedding {
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    pub fn similarity(&self, other: &Embedding) -> f32 {
        self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum()
    }

    pub fn similarities(&self, others: &[Embedding]) -> Vec<f32> {
        others.iter().map(|other| self.similarity(other)).collect()
    }
}

pub struct Bge {
    session: onnx::Session,
    tokenizer: Tokenizer,
}

impl Bge {
    pub fn new(onnx_runtime: &Arc<onnx::Onnx>, executor: onnx::Executor) -> Result<Self, Box<dyn std::error::Error>> {
        let session = onnx_runtime.create_session(executor, onnx::OptimizationLevel::EnableAll, 4, MODEL_PATH);
        let tokenizer = Tokenizer::from_file(TOKENIZER_PATH)
            .map_err(|e| format!("Failed to load tokenizer from {}: {}", TOKENIZER_PATH, e))?;
        Ok(Self { session, tokenizer })
    }

    pub fn embed(&self, sentence: &str) -> Result<Embedding, Box<dyn std::error::Error>> {
        // Tokenize input
        let encoding = self
            .tokenizer
            .encode(sentence, true)
            .map_err(|e| format!("Failed to encode sentence: {}", e))?;
        let input_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();

        // Truncate to max sequence length if needed
        let seq_len = input_ids.len().min(MAX_SEQ_LEN);
        let input_ids_i64: Vec<i64> = input_ids[..seq_len].iter().map(|&id| id as i64).collect();
        let attention_mask_i64: Vec<i64> = attention_mask[..seq_len].iter().map(|&m| m as i64).collect();
        let token_type_ids_i64: Vec<i64> = vec![0i64; seq_len];

        // Create input tensors
        let input_ids_tensor = onnx::Value::from_slice::<i64>(&self.session.onnx, &[1, seq_len], &input_ids_i64);
        let attention_mask_tensor = onnx::Value::from_slice::<i64>(&self.session.onnx, &[1, seq_len], &attention_mask_i64);
        let token_type_ids_tensor = onnx::Value::from_slice::<i64>(&self.session.onnx, &[1, seq_len], &token_type_ids_i64);

        // Run inference
        // Note: ONNX model also exports 'tanh' output (pooler), but we don't use it.
        // We extract the CLS token from last_hidden_state manually and L2-normalize.
        let inputs = [
            ("input_ids", &input_ids_tensor),
            ("attention_mask", &attention_mask_tensor),
            ("token_type_ids", &token_type_ids_tensor),
        ];
        let outputs = self.session.run(&inputs, &["last_hidden_state"]);

        // Extract CLS token embedding (first token, index 0)
        let last_hidden_state = outputs[0].extract_tensor::<f32>();
        let mut cls_embedding = vec![0f32; EMBEDDING_SIZE];
        cls_embedding.copy_from_slice(&last_hidden_state[..EMBEDDING_SIZE]);

        // L2 normalize with epsilon guard
        let magnitude: f32 = cls_embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let epsilon = 1e-12;
        let norm_factor = (magnitude + epsilon).recip();
        for x in &mut cls_embedding {
            *x *= norm_factor;
        }

        Ok(Embedding(cls_embedding))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_as_slice() {
        let emb = Embedding(vec![1.0, 2.0, 3.0]);
        assert_eq!(emb.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_embedding_size() {
        assert_eq!(EMBEDDING_SIZE, 384);
    }

    #[test]
    fn test_max_seq_len() {
        assert_eq!(MAX_SEQ_LEN, 128);
    }
}
