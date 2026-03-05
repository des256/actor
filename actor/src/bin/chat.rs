use {
    actor::*,
    std::{
        collections::VecDeque,
        io::{Write, stdin, stdout},
        sync::Arc,
        time::{Duration, Instant},
    },
};

const VOICE_PATH: &str = "data/tts/voices/hannah.bin";
const ASR_SAMPLE_RATE: usize = 16000;
const VAD_FRAME_SIZE: usize = 512;
const VAD_FRAMES_PER_CHUNK: usize = 8;
const VAD_THRESHOLD: f32 = 0.5;
const VAD_PREROLL_CHUNKS: usize = 1;
const VAD_COOLDOWN_FRAMES: usize = 16;
const TTS_SAMPLE_RATE: usize = 24000;
const TTS_CHUNK_SIZE: usize = 2048;

enum VadState {
    Idle,
    Speaking,
    Cooldown(usize),
}

#[derive(Clone)]
struct AsrPayload {
    user_speech_end: Instant, // when user stopped speaking
}

#[derive(Clone)]
struct LlmPayload {
    user_speech_end: Instant, // when user stopped speaking
    user_sentence: String,    // the full utterance
    prompt_id: u64,           // assigned prompt ID
}

#[derive(Clone)]
struct TtsPayload {
    user_speech_end: Instant,  // when user stopped speaking
    user_sentence: String,     // the full utterance
    prompt_id: u64,            // assigned prompt ID
    response_id: u64,          // assigned response ID
    response_sentence: String, // the current response sentence
}

#[derive(Clone)]
struct AudioOutPayload {
    user_speech_end: Instant,  // when user stopped speaking
    user_sentence: String,     // the full utterance
    prompt_id: u64,            // assigned prompt ID
    response_id: u64,          // assigned response ID
    response_sentence: String, // the current response sentence
    last: bool,                // true if this is the last chunk of the response
}

fn ends_with_sentence_boundary(text: &str) -> bool {
    let trimmed = text.trim_end();
    matches!(trimmed.as_bytes().last(), Some(b'.' | b'!' | b'?' | b':' | b';'))
}

#[tokio::main]
async fn main() {
    // prepare pipeline components
    let mut epoch = Arc::new(Epoch::new());
    let onnx = onnx::Onnx::new(24);

    println!("initializing audio...");
    let mut audioin_listener = audioin::create(ASR_SAMPLE_RATE, VAD_FRAMES_PER_CHUNK * VAD_FRAME_SIZE, None, 3);
    let (audioout_handle, mut audioout_listener) =
        audioout::create::<AudioOutPayload>(TTS_SAMPLE_RATE, TTS_CHUNK_SIZE, None, &epoch);
    let audioout_handle = Arc::new(audioout_handle);

    print!("loading VAD...");
    stdout().flush().unwrap();
    let mut vad = vad::Vad::new(&onnx, onnx::Executor::Cpu, ASR_SAMPLE_RATE);
    println!(" done.");
    print!("loading ASR...");
    stdout().flush().unwrap();
    let (asr_handle, mut asr_listener) = asr::create::<AsrPayload>(&onnx, onnx::Executor::Cuda(0));
    let asr_handle = Arc::new(asr_handle);
    println!(" done.");
    print!("loading LLM...");
    stdout().flush().unwrap();
    let (llm_handle, mut llm_listener) =
        llm::create::<LlmPayload>(&onnx, onnx::Executor::Cuda(0), llm::Model::Llama33b, &epoch);
    let llm_handle = Arc::new(llm_handle);
    println!(" done.");
    print!("loading TTS...");
    stdout().flush().unwrap();
    let (tts_handle, mut tts_listener) = tts::create::<TtsPayload>(&onnx, onnx::Executor::Cpu, VOICE_PATH, &epoch);
    let tts_handle = Arc::new(tts_handle);
    println!(" done.");

    let history = Arc::new(history::History::new());

    // prepare for prompt engineering magic
    println!("everything loaded. select persona:");
    println!("1. grumpy");
    println!("2. wise");
    println!("3. arrogant");
    println!("4. happy");
    print!("> ");
    stdout().flush().unwrap();
    let mut input = String::new();
    stdin().read_line(&mut input).unwrap();
    let choice = input.trim().parse::<usize>().unwrap();
    let directive = "keep response short, do not use internal reasoning\npersona:\n- dramatic\n";
    let (identity, personality) = match choice {
        1 => ("you are very grumpy", format!("{}- sarcastic", directive)),
        2 => ("you are a wise sage", format!("{}- eloquent riddles\n- fuzzy", directive)),
        3 => ("you outshine the user", format!("{}- show off\n- not helpful", directive)),
        4 => ("you are happy", format!("{}- positive and upbeat", directive)),
        _ => panic!("invalid choice"),
    };
    let tools = "tools:\n- write [flash] to emphasize something important";
    let facts = "facts to use when needed:\n- today is march 5, it's sunny and warm outside";

    // AudioIn pump
    tokio::spawn({
        let asr_handle = Arc::clone(&asr_handle);
        async move {
            let mut state = VadState::Idle;
            let mut preroll: VecDeque<Vec<i16>> = VecDeque::with_capacity(VAD_PREROLL_CHUNKS + 1);
            loop {
                let audio = audioin_listener.recv().await;
                let mut speech_started = false;
                let mut speech_ended = false;
                let mut user_speech_end = Instant::now();
                for chunk in audio.chunks_exact(VAD_FRAME_SIZE) {
                    let probability = vad.analyze(chunk);
                    match &mut state {
                        VadState::Idle => {
                            if probability > VAD_THRESHOLD {
                                state = VadState::Speaking;
                                speech_started = true;
                            }
                        }
                        VadState::Speaking => {
                            if probability < VAD_THRESHOLD {
                                state = VadState::Cooldown(VAD_COOLDOWN_FRAMES);
                                user_speech_end = Instant::now();
                            }
                        }
                        VadState::Cooldown(remaining) => {
                            if probability > VAD_THRESHOLD {
                                state = VadState::Speaking;
                            } else {
                                *remaining -= 1;
                                if *remaining == 0 {
                                    speech_ended = true;
                                    state = VadState::Idle;
                                }
                            }
                        }
                    }
                }
                if speech_started {
                    for chunk in preroll.drain(..) {
                        asr_handle.send(asr::Input {
                            payload: AsrPayload { user_speech_end },
                            audio: chunk,
                            flush: false,
                        });
                    }
                }
                let mut in_speech = speech_ended;
                if let VadState::Speaking | VadState::Cooldown(_) = state {
                    in_speech = true;
                }
                if in_speech || speech_started {
                    asr_handle.send(asr::Input {
                        payload: AsrPayload { user_speech_end },
                        audio,
                        flush: speech_ended,
                    });
                } else {
                    if preroll.len() >= VAD_PREROLL_CHUNKS {
                        preroll.pop_front();
                    }
                    preroll.push_back(audio);
                }
            }
        }
    });

    // ASR pump
    tokio::spawn({
        let epoch = Arc::clone(&epoch);
        let llm_handle = Arc::clone(&llm_handle);
        let history = Arc::clone(&history);
        async move {
            let mut prompt_id = 0u64;
            loop {
                match asr_listener.recv().await {
                    asr::Output::Partial { payload: _, utterance } => {
                        println!("({}...)", utterance);
                    }
                    asr::Output::Final { payload, utterance } => {
                        println!("--> {}", utterance);
                        if !utterance.is_empty() {
                            history.add(history::Role::User(0), utterance.clone()).await;
                            let prompt =
                                prompt::build(llm::Model::Llama33b, identity, &personality, tools, facts, &history).await;
                            llm_handle.send(llm::Input {
                                payload: LlmPayload {
                                    user_speech_end: payload.user_speech_end,
                                    user_sentence: utterance,
                                    prompt_id,
                                },
                                prompt,
                                stamp: epoch.current(),
                            });
                            prompt_id += 1;
                        }
                    }
                }
            }
        }
    });

    // LLM pump
    tokio::spawn({
        let epoch = Arc::clone(&epoch);
        let tts_handle = Arc::clone(&tts_handle);
        async move {
            let mut current_prompt_id = u64::MAX;
            let mut current_response = String::new();
            let mut response_id = 0u64;
            loop {
                match llm_listener.recv().await {
                    llm::Output::Token { payload, token, stamp } => {
                        if stamp != epoch.current() {
                            current_response.clear();
                            continue;
                        }
                        current_response.push_str(&token);
                        if ends_with_sentence_boundary(&current_response) {
                            let trimmed = current_response.trim().to_string();
                            if !trimmed.is_empty() {
                                if payload.prompt_id != current_prompt_id {
                                    current_prompt_id = payload.prompt_id;
                                    response_id = 0;
                                }
                                let padded = format!("   {}     ", trimmed);
                                tts_handle.send(tts::Input {
                                    payload: TtsPayload {
                                        user_speech_end: payload.user_speech_end,
                                        user_sentence: payload.user_sentence,
                                        prompt_id: payload.prompt_id,
                                        response_id,
                                        response_sentence: trimmed,
                                    },
                                    sentence: padded,
                                    stamp: epoch.current(),
                                });
                                current_response.clear();
                                response_id += 1;
                            }
                        }
                    }
                    llm::Output::Eos { payload, stamp } => {
                        if stamp != epoch.current() {
                            current_response.clear();
                            continue;
                        }
                        let trimmed = current_response.trim().to_string();
                        if !trimmed.is_empty() {
                            if payload.prompt_id != current_prompt_id {
                                current_prompt_id = payload.prompt_id;
                                response_id = 0;
                            }
                            let padded = format!("   {}     ", trimmed);
                            tts_handle.send(tts::Input {
                                payload: TtsPayload {
                                    user_speech_end: payload.user_speech_end,
                                    user_sentence: payload.user_sentence,
                                    prompt_id: payload.prompt_id,
                                    response_id,
                                    response_sentence: trimmed,
                                },
                                sentence: padded,
                                stamp: epoch.current(),
                            });
                            current_response.clear();
                            response_id += 1;
                        }
                    }
                }
            }
        }
    });

    // TTS pump
    tokio::spawn({
        let epoch = Arc::clone(&epoch);
        let audioout_handle = Arc::clone(&audioout_handle);
        async move {
            loop {
                let output = tts_listener.recv().await;
                audioout_handle.send(audioout::Input {
                    payload: AudioOutPayload {
                        user_speech_end: output.payload.user_speech_end,
                        user_sentence: output.payload.user_sentence,
                        prompt_id: output.payload.prompt_id,
                        response_id: output.payload.response_id,
                        response_sentence: output.payload.response_sentence,
                        last: output.last,
                    },
                    data: output.audio,
                    stamp: epoch.current(),
                });
            }
        }
    });

    // AudioOut pump
    tokio::spawn({
        let history = Arc::clone(&history);
        async move {
            let mut current_prompt_id = u64::MAX;
            let mut current_response_id = u64::MAX;
            let mut current_response_sentence = String::new();
            let mut current_index = 0usize;
            let mut samples_per_char = 0.0f32;
            loop {
                match audioout_listener.recv().await {
                    audioout::Status::Started(payload) => {
                        if current_response_id != payload.response_id {
                            current_response_id = payload.response_id;
                            current_response_sentence = String::new();
                            current_index = 0;
                            if payload.prompt_id != current_prompt_id {
                                current_prompt_id = payload.prompt_id;
                                println!(
                                    "P{:03} thinking time: {}ms",
                                    payload.prompt_id,
                                    payload.user_speech_end.elapsed().as_millis()
                                );
                            }
                            println!("<-- {}", payload.response_sentence);
                        }
                    }
                    audioout::Status::Finished { payload, index } => {
                        current_index += index;
                        if payload.last {
                            history.add(history::Role::Robot, current_response_sentence.clone()).await;
                            if current_response_sentence.len() > 10 {
                                samples_per_char = current_index as f32 / current_response_sentence.len() as f32;
                            }
                        }
                    }
                    audioout::Status::Canceled { payload: _, index } => {
                        current_index += index;
                        let truncated: String = if samples_per_char > 0.0 {
                            let chars_played = (current_index as f32 / samples_per_char).round() as usize;
                            current_response_sentence.chars().take(chars_played).collect()
                        } else {
                            current_response_sentence.clone()
                        };
                        println!("<-- {}...", truncated);
                        history.add(history::Role::Robot, format!("{}...", truncated)).await;
                    }
                }
            }
        }
    });

    loop {
        println!("chat is running, press CTRL-C to exit...");
        std::thread::sleep(Duration::from_secs(10));
    }
}
