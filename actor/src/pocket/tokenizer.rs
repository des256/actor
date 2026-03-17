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
        let prepared = prepare_text(sentence);

        let encoding = self.tokenizer.encode(prepared, false).unwrap();

        (
            encoding.get_ids().iter().map(|token| *token as i64).collect(),
            FRAMES_AFTER_EOS,
        )
    }
}

/// Text normalization matching the model's expected input format.
fn prepare_text(text: &str) -> String {
    let mut t = text
        .trim()
        .replace('\n', " ")
        .replace('\r', " ")
        .replace("  ", " ");

    if t.is_empty() {
        return t;
    }

    // Uppercase first letter
    let mut chars = t.chars();
    if let Some(first) = chars.next() {
        t = first.to_uppercase().to_string() + chars.as_str();
    }

    // Ensure ends with punctuation
    if t.ends_with(|c: char| c.is_alphanumeric()) {
        t.push('.');
    }

    // Pad short texts
    if t.split_whitespace().count() < 5 {
        t = format!("        {}", t);
    }

    // Prepend sentencepiece marker and replace spaces
    format!("\u{2581}{}", t.replace(' ', "\u{2581}"))
}
