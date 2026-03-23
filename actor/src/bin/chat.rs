use {
    actor::*,
    std::{
        collections::VecDeque,
        io::{Write, stdin, stdout},
        sync::Arc,
        time::{Duration, Instant},
    },
};

const VOICE_PATH: &str = "data/pocket/voices/hannah.bin";
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
struct ChatPayload {
    user_speech_end: Instant, // when user stopped speaking
    user_sentence: String,    // the full utterance
    user_intent: String,      // the intent of the user's utterance
    prompt_id: u64,           // assigned prompt ID
}

#[derive(Clone)]
struct TtsPayload {
    user_speech_end: Instant,  // when user stopped speaking
    user_sentence: String,     // the full utterance
    user_intent: String,       // the intent of the user's utterance
    prompt_id: u64,            // assigned prompt ID
    response_id: u64,          // assigned response ID
    response_sentence: String, // the current response sentence
    response_nuance: String,   // the current response nuance
}

#[derive(Clone)]
struct AudioOutPayload {
    user_speech_end: Instant,  // when user stopped speaking
    _user_sentence: String,    // the full utterance
    user_intent: String,       // the intent of the user's utterance
    prompt_id: u64,            // assigned prompt ID
    response_id: u64,          // assigned response ID
    response_sentence: String, // the current response sentence
    response_nuance: String,   // the current response nuance
    last: bool,                // true if this is the last chunk of the response
}

fn ends_with_sentence_boundary(text: &str) -> bool {
    let trimmed = text.trim_end();
    matches!(trimmed.as_bytes().last(), Some(b'.' | b'!' | b'?' | b':' | b';'))
}

fn strip_markers(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '*' || c == '[' {
            let close = if c == '*' { '*' } else { ']' };
            let mut found_close = false;
            for inner in chars.by_ref() {
                if inner == close {
                    found_close = true;
                    break;
                }
            }
            if !found_close {
                result.push(c);
            }
        } else {
            result.push(c);
        }
    }
    result
}

const INTENTS: [&str; 3] = ["greeting", "statement", "question"];
const NUANCES: [&str; 9] = [
    "happy",
    "formal",
    "frustrated",
    "sad",
    "angry",
    "sarcastic",
    "excited",
    "surprised",
    "confused",
];

#[tokio::main]
async fn main() {
    // prepare pipeline components
    let epoch = Arc::new(Epoch::new());
    let onnx = onnx::Onnx::new(24);
    let tensorrt = tensorrt::Tensorrt::new();

    println!("initializing audio...");
    let mut audioin_listener = audioin::create(ASR_SAMPLE_RATE, VAD_FRAMES_PER_CHUNK * VAD_FRAME_SIZE, None, 3);
    let (audioout_handle, mut audioout_listener) =
        audioout::create::<AudioOutPayload>(TTS_SAMPLE_RATE, TTS_CHUNK_SIZE, None, &epoch);
    let audioout_handle = Arc::new(audioout_handle);

    print!("loading VAD (Silero)...");
    stdout().flush().unwrap();
    let mut vad = vad::Vad::new(&onnx, onnx::Executor::Cpu, ASR_SAMPLE_RATE);
    println!(" done.");
    print!("loading ASR (Moonshine STT)...");
    stdout().flush().unwrap();
    let (asr_handle, mut asr_listener) = moonshine::create::<AsrPayload>(&tensorrt);
    let asr_handle = Arc::new(asr_handle);
    println!(" done.");
    print!("loading LLM (Llama 3.2 Instruct)...");
    stdout().flush().unwrap();
    let (llm_handle, mut llm_listener) = llama3::create::<ChatPayload>(&tensorrt, &epoch);
    let llm_handle = Arc::new(llm_handle);
    println!(" done.");
    print!("loading TTS (Pocket TTS)...");
    stdout().flush().unwrap();
    let trt = tensorrt::Tensorrt::new();
    let (tts_handle, mut tts_listener) = pocket::create::<TtsPayload>(&trt, VOICE_PATH, &epoch);
    let tts_handle = Arc::new(tts_handle);
    println!(" done.");

    let bge = Arc::new(bge::Bge::new(&onnx, onnx::Executor::Cpu).unwrap());
    let intents: Vec<bge::Embedding> = INTENTS.iter().map(|intent| bge.embed(intent).unwrap()).collect();
    let nuances: Vec<bge::Embedding> = NUANCES.iter().map(|nuance| bge.embed(nuance).unwrap()).collect();

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
        let epoch = Arc::clone(&epoch);
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
                                epoch.advance(); // stop everything immediately
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
                        asr_handle.send(moonshine::Input {
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
                    asr_handle.send(moonshine::Input {
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
        let bge = Arc::clone(&bge);
        async move {
            let mut prompt_id = 0u64;
            loop {
                match asr_listener.recv().await {
                    moonshine::Output::Partial { payload: _, utterance } => {
                        println!("({}...)", utterance);
                    }
                    moonshine::Output::Final { payload, utterance } => {
                        if !utterance.is_empty() {
                            let intents = bge.embed(&utterance).unwrap().similarities(&intents);
                            let mut sorted_intents = intents.iter().zip(INTENTS).collect::<Vec<_>>();
                            sorted_intents.sort_by(|a, b| b.0.partial_cmp(a.0).unwrap_or(std::cmp::Ordering::Equal));
                            let intent = format!("{}, {}", sorted_intents[0].1, sorted_intents[1].1);
                            println!("--> {} ({})", utterance, intent);
                            history.add(history::Role::User(0), utterance.clone()).await;
                            let prompt = llm_handle.build_prompt(identity, &personality, tools, facts, &history).await;
                            llm_handle.send(llama3::Input {
                                payload: ChatPayload {
                                    user_speech_end: payload.user_speech_end,
                                    user_sentence: utterance.clone(),
                                    user_intent: intent,
                                    prompt_id,
                                },
                                prompt,
                                stamp: epoch.current(),
                                max_tokens: 50,
                                temperature: 0.0,
                            });
                            prompt_id += 1;
                        }
                    }
                }
            }
        }
    });

    // Chat pump
    tokio::spawn({
        let epoch = Arc::clone(&epoch);
        let tts_handle = Arc::clone(&tts_handle);
        async move {
            let mut current_prompt_id = u64::MAX;
            let mut current_response = String::new();
            let mut response_id = 0u64;
            loop {
                match llm_listener.recv().await {
                    llama3::Output::Token { payload, token, stamp } => {
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
                                let nuances = bge.embed(&trimmed).unwrap().similarities(&nuances); // potentially blocking?
                                let mut sorted_nuances = nuances.iter().zip(NUANCES).collect::<Vec<_>>();
                                sorted_nuances.sort_by(|a, b| b.0.partial_cmp(a.0).unwrap_or(std::cmp::Ordering::Equal));
                                let nuance = format!("{}, {}", sorted_nuances[0].1, sorted_nuances[1].1);
                                tts_handle.send(pocket::Input {
                                    payload: TtsPayload {
                                        user_speech_end: payload.user_speech_end,
                                        user_sentence: payload.user_sentence.clone(),
                                        user_intent: payload.user_intent.clone(),
                                        prompt_id: payload.prompt_id,
                                        response_id,
                                        response_sentence: trimmed.clone(),
                                        response_nuance: nuance,
                                    },
                                    sentence: strip_markers(&trimmed).to_string(),
                                    stamp: epoch.current(),
                                });
                                current_response.clear();
                                response_id += 1;
                            }
                        }
                    }
                    llama3::Output::Eos { payload, stamp } => {
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
                            let nuances = bge.embed(&trimmed).unwrap().similarities(&nuances); // potentially blocking?
                            let mut sorted_nuances = nuances.iter().zip(NUANCES).collect::<Vec<_>>();
                            sorted_nuances.sort_by(|a, b| b.0.partial_cmp(a.0).unwrap_or(std::cmp::Ordering::Equal));
                            let nuance = format!("{}, {}", sorted_nuances[0].1, sorted_nuances[1].1);
                            tts_handle.send(pocket::Input {
                                payload: TtsPayload {
                                    user_speech_end: payload.user_speech_end,
                                    user_sentence: payload.user_sentence.clone(),
                                    user_intent: payload.user_intent.clone(),
                                    prompt_id: payload.prompt_id,
                                    response_id,
                                    response_sentence: trimmed.clone(),
                                    response_nuance: nuance,
                                },
                                sentence: strip_markers(&trimmed).to_string(),
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
        let audioout_handle = Arc::clone(&audioout_handle);
        async move {
            loop {
                let output = tts_listener.recv().await;
                audioout_handle.send(audioout::Input {
                    payload: AudioOutPayload {
                        user_speech_end: output.payload.user_speech_end,
                        _user_sentence: output.payload.user_sentence,
                        user_intent: output.payload.user_intent,
                        prompt_id: output.payload.prompt_id,
                        response_id: output.payload.response_id,
                        response_sentence: output.payload.response_sentence,
                        response_nuance: output.payload.response_nuance,
                        last: output.last,
                    },
                    data: output.audio,
                    stamp: output.stamp,
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
                        if current_prompt_id != payload.prompt_id || current_response_id != payload.response_id {
                            if payload.prompt_id != current_prompt_id {
                                current_prompt_id = payload.prompt_id;
                                println!(
                                    "P{:03} thinking time: {}ms",
                                    payload.prompt_id,
                                    payload.user_speech_end.elapsed().as_millis()
                                );
                            }
                            current_response_id = payload.response_id;
                            current_response_sentence = payload.response_sentence.clone();
                            current_index = 0;
                            println!("<-- {} ({})", payload.response_sentence, payload.response_nuance);
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
                    audioout::Status::Canceled { payload, index } => {
                        current_index += index;
                        let truncated: String = if samples_per_char > 0.0 {
                            let chars_played = (current_index as f32 / samples_per_char).round() as usize;
                            current_response_sentence.chars().take(chars_played).collect()
                        } else {
                            current_response_sentence.clone()
                        };
                        println!("<-- {}... ({})", truncated, payload.response_nuance);
                        history.add(history::Role::Robot, format!("{}...", truncated)).await;
                    }
                }
            }
        }
    });

    // just wait forever
    loop {
        std::thread::sleep(Duration::from_secs(10));
    }
}
