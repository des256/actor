use {
    libpulse_binding::{
        def::BufferAttr,
        sample::{Format, Spec},
        stream::Direction,
    },
    libpulse_simple_binding::Simple,
    tokio::sync::mpsc as tokio_mpsc,
};

const CHANNEL_CAPACITY: usize = 8;

pub struct AudioInListener {
    input_rx: tokio_mpsc::Receiver<Vec<i16>>,
}

pub fn create_audioin(
    sample_rate: usize,
    chunk_size: usize,
    device_name: Option<&str>,
    boost: usize,
) -> AudioInListener {
    let (input_tx, input_rx) = tokio_mpsc::channel::<Vec<i16>>(CHANNEL_CAPACITY);
    let listener = AudioInListener { input_rx };
    std::thread::spawn({
        let device_name = match device_name {
            Some(name) => Some(name.to_string()),
            None => None,
        };
        move || {
            let spec = Spec {
                format: Format::S16NE,
                channels: 1,
                rate: sample_rate as u32,
            };
            let bytes_per_chunk = chunk_size * 2;
            let mut buffer = vec![0u8; bytes_per_chunk];
            let buffer_attr = BufferAttr {
                maxlength: bytes_per_chunk as u32 * 16,
                tlength: u32::MAX,
                prebuf: u32::MAX,
                minreq: u32::MAX,
                fragsize: bytes_per_chunk as u32,
            };
            let pulse: Simple = match Simple::new(
                None,
                "actor-audioin",
                Direction::Record,
                device_name.as_deref(),
                "audio-capture",
                &spec,
                None,
                Some(&buffer_attr),
            ) {
                Ok(pulse) => pulse,
                Err(error) => panic!("AudioIn: failed to connect to PulseAudio: {}", error),
            };
            loop {
                match pulse.read(&mut buffer) {
                    Ok(()) => {
                        let mut sample = Vec::<i16>::with_capacity(buffer.len() / 2);
                        for bytes in buffer.as_slice().chunks(2) {
                            sample.push(i16::saturating_mul(
                                i16::from_le_bytes([bytes[0], bytes[1]]),
                                boost as i16,
                            ));
                        }
                        if let Err(error) = input_tx.blocking_send(sample) {
                            panic!("AudioIn: failed to send audio: {}", error);
                        }
                    }
                    Err(error) => {
                        panic!("AudioIn: PulseAudio read error: {}", error);
                    }
                }
            }
        }
    });
    listener
}

impl AudioInListener {
    pub async fn recv(&mut self) -> Vec<i16> {
        self.input_rx.recv().await.unwrap()
    }

    pub fn try_recv(&mut self) -> Option<Vec<i16>> {
        match self.input_rx.try_recv() {
            Ok(sample) => Some(sample),
            Err(tokio_mpsc::error::TryRecvError::Empty) => None,
            Err(tokio_mpsc::error::TryRecvError::Disconnected) => {
                panic!("AudioIn: data channel disconnected")
            }
        }
    }
}
