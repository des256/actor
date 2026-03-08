use actor::*;
use std::time::Instant;

#[tokio::main]
async fn main() {
    // load wav file
    let mut reader = hound::WavReader::open("test.wav").expect("Failed to open test.wav");
    let spec = reader.spec();
    assert_eq!(
        spec.sample_rate, 16000,
        "Expected 16kHz sample rate, got {}",
        spec.sample_rate
    );
    assert_eq!(spec.channels, 1, "Expected mono audio, got {} channels", spec.channels);

    let samples: Vec<i16> = match spec.sample_format {
        hound::SampleFormat::Int => reader.samples::<i16>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| (s.unwrap() * 32767.0) as i16).collect(),
    };

    let duration_secs = samples.len() as f64 / spec.sample_rate as f64;
    println!("Loaded test.wav: {} samples ({:.2}s)", samples.len(), duration_secs);

    // init parakeet and moonshine
    let onnx = onnx::Onnx::new(17);
    println!("Loading models...");
    let (parakeet_handle, mut parakeet_listener) = parakeet::create::<()>(&onnx, onnx::Executor::Cuda(0));
    let (moonshine_handle, mut moonshine_listener) = moonshine::create::<()>(&onnx, onnx::Executor::Cpu);
    println!("Models loaded.");

    // send audio in 500ms chunks (8000 samples at 16kHz)
    let chunk_size = 8000;
    let chunks: Vec<Vec<i16>> = samples.chunks(chunk_size).map(|c| c.to_vec()).collect();
    let num_chunks = chunks.len();

    // test Moonshine
    let start = Instant::now();
    let mut first_result = None;
    tokio::spawn({
        let chunks = chunks.clone();
        async move {
            for (i, chunk) in chunks.iter().enumerate() {
                let is_last = i == num_chunks - 1;
                if i == 0 {
                    moonshine_handle.send(moonshine::Input::Initial {
                        payload: (),
                        audio: chunk.clone(),
                    });
                } else {
                    moonshine_handle.send(moonshine::Input::Continuing {
                        payload: (),
                        audio: chunk.clone(),
                        flush: is_last,
                    });
                }
            }
        }
    });
    loop {
        match moonshine_listener.recv().await {
            moonshine::Output::Partial { .. } => {
                if first_result.is_none() {
                    first_result = Some(start.elapsed());
                }
            }
            moonshine::Output::Final { payload: _, utterance } => {
                let elapsed = start.elapsed();
                if first_result.is_none() {
                    first_result = Some(elapsed);
                }
                println!("Moonshine: {}", utterance);
                println!("Time to first result: {:?}", first_result.unwrap());
                break;
            }
        }
    }

    // test Parakeet
    let start = Instant::now();
    let mut first_result = None;
    tokio::spawn({
        let chunks = chunks.clone();
        async move {
            for (i, chunk) in chunks.iter().enumerate() {
                let is_last = i == num_chunks - 1;
                parakeet_handle.send(parakeet::Input {
                    payload: (),
                    audio: chunk.clone(),
                    flush: is_last,
                });
            }
        }
    });
    loop {
        match parakeet_listener.recv().await {
            parakeet::Output::Partial { .. } => {
                if first_result.is_none() {
                    first_result = Some(start.elapsed());
                }
            }
            parakeet::Output::Final { payload: _, utterance } => {
                let elapsed = start.elapsed();
                if first_result.is_none() {
                    first_result = Some(elapsed);
                }
                println!("Parakeet: {}", utterance);
                println!("Time to first result: {:?}", first_result.unwrap());
                break;
            }
        }
    }
}
