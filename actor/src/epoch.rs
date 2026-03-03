use std::sync::atomic::{AtomicU64, Ordering};

pub struct Epoch(AtomicU64);

impl Epoch {
    pub fn new() -> Self {
        Epoch(AtomicU64::new(1))
    }

    pub fn current(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }

    pub fn advance(&self) -> u64 {
        self.0.fetch_add(1, Ordering::Relaxed)
    }

    pub fn is_current(&self, epoch: u64) -> bool {
        epoch == self.0.load(Ordering::Relaxed)
    }
}
