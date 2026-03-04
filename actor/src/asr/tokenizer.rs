use {super::*, std::path::Path};

fn load_tokens<P: AsRef<Path>>(path: P) -> Vec<String> {
    let path = path.as_ref();
    let data = std::fs::read(path).unwrap();
    parse_sentencepiece_model(&data)
}

fn parse_sentencepiece_model(data: &[u8]) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let (tag, wire_type, new_pos) = read_tag(data, pos);
        pos = new_pos;
        if tag == 1 && wire_type == 2 {
            let (sub_data, new_pos) = read_bytes(data, pos);
            pos = new_pos;
            if let Some(piece) = parse_sentencepiece(sub_data) {
                pieces.push(piece);
            }
        } else {
            pos = skip_field(data, pos, wire_type);
        }
    }
    if pieces.is_empty() {
        panic!("asr::Tokenizer: No pieces found in SentencePiece model");
    }
    pieces
}

fn parse_sentencepiece(data: &[u8]) -> Option<String> {
    let mut pos = 0;
    let mut piece = None;
    while pos < data.len() {
        let (tag, wire_type, new_pos) = read_tag(data, pos);
        pos = new_pos;
        if tag == 1 && wire_type == 2 {
            let (bytes, new_pos) = read_bytes(data, pos);
            pos = new_pos;
            piece = Some(String::from_utf8_lossy(bytes).into_owned());
        } else {
            pos = skip_field(data, pos, wire_type);
        }
    }
    piece
}

fn read_varint(data: &[u8], mut pos: usize) -> (u64, usize) {
    let mut result: u64 = 0;
    let mut shift = 0;
    loop {
        if pos >= data.len() {
            panic!("Tokenizer: Unexpected end of protobuf data");
        }
        let byte = data[pos];
        pos += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return (result, pos);
        }
        shift += 7;
        if shift >= 64 {
            panic!("Tokenizer: Varint too long");
        }
    }
}

fn read_tag(data: &[u8], pos: usize) -> (u32, u8, usize) {
    let (value, new_pos) = read_varint(data, pos);
    let tag = (value >> 3) as u32;
    let wire_type = (value & 0x07) as u8;
    (tag, wire_type, new_pos)
}

fn read_bytes<'a>(data: &'a [u8], pos: usize) -> (&'a [u8], usize) {
    let (len, pos) = read_varint(data, pos);
    let len = len as usize;
    let end = pos + len;
    if end > data.len() {
        panic!("Length-delimited field exceeds data");
    }
    (&data[pos..end], end)
}

fn skip_field(data: &[u8], pos: usize, wire_type: u8) -> usize {
    match wire_type {
        0 => {
            let (_, new_pos) = read_varint(data, pos);
            new_pos
        }
        1 => pos + 8,
        2 => {
            let (_, new_pos) = read_bytes(data, pos);
            new_pos
        }
        5 => pos + 4,
        _ => panic!("Unknown wire type: {}", wire_type),
    }
}

pub struct Tokenizer {
    mapping: Vec<String>,
}

impl Tokenizer {
    pub fn new() -> Self {
        let mapping = load_tokens(&PARAKEET_TOKENIZER_PATH);
        Self { mapping }
    }

    pub fn tokenize(&self, tokens: &[i64]) -> String {
        if !tokens.is_empty() {
            tokens
                .iter()
                .filter_map(|&token| {
                    if token as usize >= self.mapping.len() {
                        None
                    } else {
                        Some(self.mapping[token as usize].replace('▁', " "))
                    }
                })
                .collect::<String>()
        } else {
            String::new()
        }
    }
}
