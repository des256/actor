use super::*;

pub struct Tokenizer {
    vocab: Vec<Vec<u8>>,
}

impl Tokenizer {
    pub fn new() -> Self {
        let data = std::fs::read(MOONSHINE_TOKENIZER_PATH).expect("Failed to read moonshine tokenizer");
        let mut vocab = Vec::new();
        let mut pos = 0;
        while pos < data.len() {
            let len = data[pos] as usize;
            pos += 1;
            if pos + len > data.len() {
                break;
            }
            vocab.push(data[pos..pos + len].to_vec());
            pos += len;
        }
        Self { vocab }
    }

    pub fn decode(&self, tokens: &[i64]) -> String {
        let mut bytes = Vec::new();
        for &id in tokens {
            let id = id as usize;
            // Skip special tokens: 0=<unk>, 1=<s> (BOS), 2=</s> (EOS)
            if id >= 3 && id < self.vocab.len() {
                bytes.extend_from_slice(&self.vocab[id]);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
