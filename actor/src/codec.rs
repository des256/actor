pub trait Codec {
    fn encode(&self, target: &mut [u8]);
    fn decode(source: &[u8]) -> Self;
    fn size(&self) -> usize;
}

impl Codec for bool {
    fn encode(&self, target: &mut [u8]) {
        target[0] = if *self { 1 } else { 0 };
    }

    fn decode(source: &[u8]) -> Self {
        source[0] != 0
    }

    fn size(&self) -> usize {
        1
    }
}

macro_rules! impl_numeric {
    ($($ty:ty),*) => {
        $(
            impl Codec for $ty {
                fn encode(&self, target: &mut [u8]) {
                    target.copy_from_slice(&self.to_le_bytes());
                }

                fn decode(source: &[u8]) -> Self {
                    Self::from_le_bytes(source.try_into().unwrap())
                }

                fn size(&self) -> usize {
                    std::mem::size_of::<$ty>()
                }
            }
        )*
    }
}

impl_numeric!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64);

impl Codec for String {
    fn encode(&self, target: &mut [u8]) {
        let bytes = self.as_bytes();
        (bytes.len() as u32).encode(&mut target[0..4]);
        target[4..].copy_from_slice(bytes);
    }

    fn decode(source: &[u8]) -> Self {
        let len = u32::decode(&source[0..4]) as usize;
        Self::from_utf8(source[4..4 + len].to_vec()).unwrap()
    }

    fn size(&self) -> usize {
        4 + self.as_bytes().len()
    }
}

impl<T: Codec> Codec for Vec<T> {
    fn encode(&self, target: &mut [u8]) {
        (self.len() as u32).encode(&mut target[0..4]);
        let mut offset = 4;
        for item in self.iter() {
            let size = item.size();
            item.encode(&mut target[offset..offset + size]);
            offset += size;
        }
    }

    fn decode(source: &[u8]) -> Self {
        let len = u32::decode(&source[0..4]) as usize;
        let mut offset = 4;
        let mut result = Vec::with_capacity(len);
        for _ in 0..len {
            let item = T::decode(&source[offset..]);
            offset += item.size();
            result.push(item);
        }
        result
    }

    fn size(&self) -> usize {
        4 + self.iter().map(|item| item.size()).sum::<usize>()
    }
}
