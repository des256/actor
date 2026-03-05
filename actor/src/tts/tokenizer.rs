use super::*;

pub struct Tokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl Tokenizer {
    pub fn new() -> Self {
        let tokenizer = tokenizers::Tokenizer::from_file(TOKENIZER_PATH).unwrap();
        Self { tokenizer }
    }

    pub fn tokenize(&self, sentence: &str) -> (Vec<i64>, usize) {
        // prepare
        let mut prepared = sentence.trim().replace('\n', " ");
        let words = prepared.split_whitespace().count();
        let frames_after = if words <= 4 { 3 } else { 1 } + 2;
        if words < 5 {
            prepared = format!("        {}", prepared);
        }
        let prepared = format!("\u{2581}{}", prepared.replace(' ', "\u{2581}"));

        // tokenize
        let encoding = self.tokenizer.encode(prepared, false).unwrap();

        (encoding.get_ids().iter().map(|token| *token as i64).collect(), frames_after)
    }
}
