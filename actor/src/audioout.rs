use {
    crate::*,
    libpulse_binding::{
        sample::{Format, Spec},
        stream::Direction,
    },
    libpulse_simple_binding::Simple,
    std::sync::{Arc, mpsc as std_mpsc},
    tokio::sync::mpsc as tokio_mpsc,
};

const CHANNEL_CAPACITY: usize = 8;

pub struct Audio<T: Clone + Send + 'static> {
    pub payload: T,
    pub data: Vec<i16>,
    pub epoch: u64,
}

pub enum Status<T: Clone + Send + 'static> {
    Started(T),
    Finished { payload: T, index: usize },
    Canceled { payload: T, index: usize },
}

pub struct Handle<T: Clone + Send + 'static> {
    output_tx: std_mpsc::Sender<Audio<T>>,
}

pub struct Listener<T: Clone + Send + 'static> {
    status_rx: tokio_mpsc::Receiver<Status<T>>,
}

pub fn create<T: Clone + Send + 'static>(
    sample_rate: usize,
    chunk_size: usize,
    device_name: Option<&str>,
    epoch: &Arc<Epoch>,
) -> (Handle<T>, Listener<T>) {
    let (output_tx, output_rx) = std_mpsc::channel::<Audio<T>>();
    let (status_tx, status_rx) = tokio_mpsc::channel::<Status<T>>(CHANNEL_CAPACITY);
    let handle = Handle { output_tx };
    let listener = Listener { status_rx };
    std::thread::spawn({
        let device_name = match device_name {
            Some(name) => Some(name.to_string()),
            None => None,
        };
        let epoch = Arc::clone(&epoch);
        move || {
            let spec = Spec {
                format: Format::S16NE,
                channels: 1,
                rate: sample_rate as u32,
            };
            let pulse = match Simple::new(
                None,
                "actor-audioout",
                Direction::Playback,
                device_name.as_deref(),
                "audio-playback",
                &spec,
                None,
                None,
            ) {
                Ok(pulse) => pulse,
                Err(error) => panic!("AudioOut: failed to connect to PulseAudio: {}", error),
            };
            let mut current_chunk: Option<Audio<T>> = None;
            let mut current_index = 0usize;
            let mut buffer = vec![0i16; chunk_size];
            loop {
                if let Some(chunk) = &current_chunk {
                    if !epoch.is_current(chunk.epoch) {
                        let chunk = current_chunk.take().unwrap();
                        if let Err(error) = status_tx.blocking_send(Status::Canceled {
                            payload: chunk.payload,
                            index: current_index,
                        }) {
                            panic!("AudioOut: failed to send canceled status: {}", error);
                        }
                        current_index = 0;
                    }
                }
                let mut i = 0usize;
                while i < chunk_size {
                    if let Some(chunk) = &current_chunk {
                        let mut n = chunk.data.len() - current_index;
                        if n > chunk_size - i {
                            n = chunk_size - i;
                        }
                        buffer[i..i + n].copy_from_slice(&chunk.data[current_index..current_index + n]);
                        current_index += n;
                        i += n;
                        if current_index >= chunk.data.len() {
                            if let Err(error) = status_tx.blocking_send(Status::Finished {
                                payload: chunk.payload.clone(),
                                index: current_index,
                            }) {
                                panic!("AudioOut: failed to send finished status: {}", error);
                            }
                            current_chunk = None;
                        }
                    } else {
                        match output_rx.try_recv() {
                            Ok(chunk) => {
                                if !epoch.is_current(chunk.epoch) {
                                    continue;
                                }
                                if let Err(error) = status_tx.blocking_send(Status::Started(chunk.payload.clone())) {
                                    panic!("AudioOut: failed to send started status: {}", error);
                                }
                                current_chunk = Some(chunk);
                                current_index = 0;
                            }
                            Err(std_mpsc::TryRecvError::Empty) => {
                                if i < chunk_size {
                                    buffer[i..].fill(0);
                                    i = chunk_size;
                                }
                            }
                            Err(std_mpsc::TryRecvError::Disconnected) => {
                                panic!("AudioOut: data channel disconnected")
                            }
                        }
                    }
                }
                let slice = unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const u8, buffer.len() * 2) };
                if let Err(error) = pulse.write(slice) {
                    panic!("AudioOut: failed to write audio: {}", error);
                }
            }
        }
    });
    (handle, listener)
}

impl<T: Clone + Send + 'static> Handle<T> {
    pub fn send(&self, audio: Audio<T>) {
        if let Err(error) = self.output_tx.send(audio) {
            panic!("AudioOut: failed to send audio: {}", error);
        }
    }
}

impl<T: Clone + Send + 'static> Listener<T> {
    pub async fn recv(&mut self) -> Option<Status<T>> {
        self.status_rx.recv().await
    }

    pub fn try_recv(&mut self) -> Option<Status<T>> {
        match self.status_rx.try_recv() {
            Ok(status) => Some(status),
            Err(tokio_mpsc::error::TryRecvError::Empty) => None,
            Err(tokio_mpsc::error::TryRecvError::Disconnected) => {
                panic!("AudioOut: data channel disconnected")
            }
        }
    }
}
