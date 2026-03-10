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
    //let (parakeet_handle, mut parakeet_listener) = parakeet::create::<()>(&onnx, onnx::Executor::Cuda(0));
    let (moonshine_handle, mut moonshine_listener) = moonshine::create::<()>(&onnx, onnx::Executor::Cuda(0));
    println!("Models loaded.");

    // send audio in 50ms chunks (800 samples at 16kHz)
    let chunk_size = 800;
    let chunks: Vec<Vec<i16>> = samples.chunks(chunk_size).map(|c| c.to_vec()).collect();
    let num_chunks = chunks.len();

    // test Moonshine
    let start = Instant::now();
    let mut first_result = None;
    tokio::spawn({
        let chunks = chunks.clone();
        async move {
            for (i, chunk) in chunks.iter().enumerate() {
                moonshine_handle.send(moonshine::Input {
                    payload: (),
                    audio: chunk.clone(),
                    flush: i == num_chunks - 1,
                });
            }
        }
    });
    let mut partial_count = 0u32;
    loop {
        match moonshine_listener.recv().await {
            moonshine::Output::Partial { utterance, .. } => {
                partial_count += 1;
                let elapsed = start.elapsed();
                if first_result.is_none() {
                    first_result = Some(elapsed);
                }
                print!("{}\r", utterance);
            }
            moonshine::Output::Final { payload: _, utterance } => {
                let elapsed = start.elapsed();
                if first_result.is_none() {
                    first_result = Some(elapsed);
                }
                println!("{}", utterance);
                println!("total time: {:?}", elapsed);
                println!("latency: {:?}", first_result.unwrap());
                break;
            }
        }
    }

    /*
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
    let mut partial_count = 0u32;
    loop {
        match parakeet_listener.recv().await {
            parakeet::Output::Partial { utterance, .. } => {
                partial_count += 1;
                let elapsed = start.elapsed();
                if first_result.is_none() {
                    first_result = Some(elapsed);
                }
                println!("  Parakeet partial #{} @ {:?}: {}", partial_count, elapsed, utterance);
            }
            parakeet::Output::Final { payload: _, utterance } => {
                let elapsed = start.elapsed();
                if first_result.is_none() {
                    first_result = Some(elapsed);
                }
                println!("Parakeet final @ {:?}: {}", elapsed, utterance);
                println!("Time to first result: {:?}", first_result.unwrap());
                break;
            }
        }
    }
    */
}
