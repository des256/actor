use {
    super::*,
    std::{
        fs::File,
        io::Read,
        path::Path,
        sync::{Arc, mpsc as std_mpsc},
    },
    tokio::sync::mpsc as tokio_mpsc,
};

const CHANNEL_CAPACITY: usize = 64;
const SAMPLES_PER_FRAME: usize = 1920;

pub struct Handle<T: Clone + Send + 'static> {
    tx: std_mpsc::Sender<Input<T>>,
}

pub struct Listener<T: Clone + Send + 'static> {
    rx: tokio_mpsc::Receiver<Output<T>>,
}

fn load_voice(voice_path: impl AsRef<Path>) -> Vec<f32> {
    let mut file = File::open(voice_path).unwrap();
    let mut buf4 = [0u8; 4];
    file.read_exact(&mut buf4).unwrap();
    let ndims = u32::from_le_bytes(buf4) as usize;
    let mut total_elements: usize = 1;
    for _ in 0..ndims {
        let mut buf8 = [0u8; 8];
        file.read_exact(&mut buf8).unwrap();
        let dim = u64::from_le_bytes(buf8) as usize;
        total_elements = total_elements.checked_mul(dim).unwrap();
    }
    let mut data = vec![0f32; total_elements];
    let slice = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, total_elements * 4) };
    file.read_exact(slice).unwrap();
    data
}

pub fn create<T: Clone + Send + 'static>(
    trt: &Arc<tensorrt::Tensorrt>,
    voice_path: impl AsRef<Path>,
    epoch: &Arc<Epoch>,
) -> (Handle<T>, Listener<T>) {
    let (input_tx, input_rx) = std_mpsc::channel::<pocket::Input<T>>();
    let (output_tx, output_rx) = tokio_mpsc::channel::<pocket::Output<T>>(CHANNEL_CAPACITY);

    let mut pocket = TrtPocket::new(&trt);
    let tokenizer = Tokenizer::new();
    let voice = load_voice(voice_path);

    pocket.init_voice(&voice);

    std::thread::spawn({
        let epoch = Arc::clone(&epoch);
        move || {
            while let Ok(input) = input_rx.recv() {
                if !epoch.is_current(input.stamp) {
                    continue;
                }

                // Reset to post-voice state
                pocket.reset(&voice);

                // Tokenize and condition
                let (tokens, eos_countdown_seed) = tokenizer.tokenize(&input.sentence);
                pocket.condition(&tokens);

                // Autoregressive loop — accumulate latents
                let mut eos_countdown: Option<usize> = None;
                for _ in 0..MAX_TOKENS {
                    let (_latent, is_eos) = pocket.step();

                    if !epoch.is_current(input.stamp) {
                        break;
                    }

                    if let Some(ref mut remaining) = eos_countdown {
                        if *remaining == 0 {
                            break;
                        }
                        *remaining -= 1;
                    } else if is_eos {
                        eos_countdown = Some(eos_countdown_seed);
                    }
                }

                // Skip decode if stale
                if !epoch.is_current(input.stamp) {
                    continue;
                }

                // Batch decode all latents to audio
                let audio = pocket.decode_audio();

                // Split audio into frame-sized chunks and send
                let chunks: Vec<&[i16]> = audio.chunks(SAMPLES_PER_FRAME).collect();
                let total_chunks = chunks.len();
                for (i, chunk) in chunks.into_iter().enumerate() {
                    if !epoch.is_current(input.stamp) {
                        break;
                    }
                    if let Err(error) = output_tx.blocking_send(Output {
                        payload: input.payload.clone(),
                        audio: chunk.to_vec(),
                        index: i,
                        last: i == total_chunks - 1,
                        stamp: input.stamp,
                    }) {
                        panic!("Tts: failed to send output: {}", error);
                    }
                }
            }
        }
    });

    (Handle { tx: input_tx }, Listener { rx: output_rx })
}

impl<T: Clone + Send + 'static> Handle<T> {
    pub fn send(&self, input: Input<T>) {
        self.tx.send(input).unwrap();
    }
}

impl<T: Clone + Send + 'static> Listener<T> {
    pub async fn recv(&mut self) -> Output<T> {
        self.rx.recv().await.unwrap()
    }

    pub fn try_recv(&mut self) -> Option<Output<T>> {
        match self.rx.try_recv() {
            Ok(output) => Some(output),
            Err(tokio_mpsc::error::TryRecvError::Empty) => None,
            Err(tokio_mpsc::error::TryRecvError::Disconnected) => {
                panic!("Tts: output channel disconnected")
            }
        }
    }
}
