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
    onnx: &Arc<onnx::Onnx>,
    executor: onnx::Executor,
    voice_path: impl AsRef<Path>,
    epoch: &Arc<Epoch>,
) -> (Handle<T>, Listener<T>) {
    // create channels
    let (input_tx, input_rx) = std_mpsc::channel::<tts::Input<T>>();
    let (output_tx, output_rx) = tokio_mpsc::channel::<tts::Output<T>>(CHANNEL_CAPACITY);

    // load models
    let mut encoder = Encoder::new(&onnx, executor);
    let mut decoder = Decoder::new(&onnx, executor);
    let tokenizer = Tokenizer::new();

    // condition voice
    encoder.init_voice(&load_voice(voice_path));

    // TODO: prepare re-usable tensors, or rather, do that in the constructors of each sub-module

    // start sentence pump
    std::thread::spawn({
        let epoch = Arc::clone(&epoch);
        move || {
            while let Ok(input) = input_rx.recv() {
                // skip if stale
                if !epoch.is_current(input.stamp) {
                    continue;
                }

                // initialize state
                encoder.reset();

                // tokenize sentence
                let (tokens, eos_countdown_seed) = tokenizer.tokenize(&input.sentence);

                // condition the sentence
                encoder.condition(&tokens);

                // token loop
                let mut eos_countdown: Option<usize> = None;
                let mut current_index = 0usize;
                for _ in 0..MAX_TOKENS {
                    // encoder step
                    let (latent, is_eos) = encoder.step();

                    // decode to audio
                    let audio = decoder.decode(&latent);

                    // exit if stale
                    if !epoch.is_current(input.stamp) {
                        break;
                    }

                    // send output
                    let is_last = if let Some(0) = eos_countdown { true } else { false };
                    if let Err(error) = output_tx.blocking_send(Output {
                        payload: input.payload.clone(),
                        audio,
                        index: current_index,
                        last: is_last,
                        stamp: input.stamp,
                    }) {
                        panic!("Tts: failed to send output: {}", error);
                    }

                    // next step
                    current_index += 1;
                    if let Some(ref mut remaining) = eos_countdown {
                        if *remaining == 0 {
                            break;
                        }
                        *remaining -= 1;
                    } else if is_eos {
                        eos_countdown = Some(eos_countdown_seed);
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
