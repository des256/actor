use {actor::*, std::collections::VecDeque};

const SAMPLE_RATE: usize = 16000;
const FRAME_SIZE: usize = 512;
const FRAMES_PER_CHUNK: usize = 8;
const VAD_LIMIT: f32 = 0.5;
const PREROLL_CHUNKS: usize = 1;
const COOLDOWN_FRAMES: usize = 10;

enum State {
    Idle,
    Speaking,
    Cooldown(usize),
}

#[tokio::main]
async fn main() {
    let mut audioin_listener = audioin::create(SAMPLE_RATE, FRAMES_PER_CHUNK * FRAME_SIZE, None, 3);
    let onnx = onnx::Onnx::new(17);
    let mut vad = vad::Vad::new(&onnx, onnx::Executor::Cpu, SAMPLE_RATE);
    let (asr_handle, mut asr_listener) = asr::create::<()>(&onnx, onnx::Executor::Cuda(0));

    // audioin pump
    tokio::spawn({
        async move {
            let mut state = State::Idle;
            let mut preroll: VecDeque<Vec<i16>> = VecDeque::with_capacity(PREROLL_CHUNKS + 1);
            loop {
                let audio = audioin_listener.recv().await;
                let mut speech_started = false;
                let mut speech_ended = false;
                for chunk in audio.chunks_exact(FRAME_SIZE) {
                    let probability = vad.analyze(chunk);
                    match &mut state {
                        State::Idle => {
                            if probability > VAD_LIMIT {
                                state = State::Speaking;
                                speech_started = true;
                            }
                        }
                        State::Speaking => {
                            if probability < VAD_LIMIT {
                                state = State::Cooldown(COOLDOWN_FRAMES);
                            }
                        }
                        State::Cooldown(remaining) => {
                            if probability > VAD_LIMIT {
                                state = State::Speaking;
                            } else {
                                *remaining -= 1;
                                if *remaining == 0 {
                                    speech_ended = true;
                                    state = State::Idle;
                                }
                            }
                        }
                    }
                }
                if speech_started {
                    for chunk in preroll.drain(..) {
                        asr_handle.send(asr::Input {
                            payload: (),
                            audio: chunk,
                            flush: false,
                        });
                    }
                }
                let mut in_speech = speech_ended;
                if let State::Speaking | State::Cooldown(_) = state {
                    in_speech = true;
                }
                if in_speech || speech_started {
                    asr_handle.send(asr::Input {
                        payload: (),
                        audio,
                        flush: speech_ended,
                    });
                } else {
                    if preroll.len() >= PREROLL_CHUNKS {
                        preroll.pop_front();
                    }
                    preroll.push_back(audio);
                }
            }
        }
    });

    // asr pump
    loop {
        match asr_listener.recv().await {
            asr::Output::Partial { payload: _, utterance } => {
                println!("Partial: {}", utterance);
            }
            asr::Output::Final { payload: _, utterance } => {
                println!("Final: {}", utterance);
            }
        }
    }
}
