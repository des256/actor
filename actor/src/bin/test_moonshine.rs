use actor::*;
use std::time::Instant;

#[tokio::main]
async fn main() {
    // load wav file
    let mut reader = hound::WavReader::open("moonshine.wav").expect("Failed to open moonshine.wav");
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

    // init moonshine
    let tensorrt = tensorrt::Tensorrt::new();
    println!("Loading model...");
    let (moonshine_handle, mut moonshine_listener) = moonshine::create::<()>(&tensorrt);
    println!("Model loaded.");

    // split audio in 100ms chunks
    let chunk_size = 1600;
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
    loop {
        match moonshine_listener.recv().await {
            moonshine::Output::Partial { utterance, .. } => {
                let elapsed = start.elapsed();
                if first_result.is_none() {
                    first_result = Some(elapsed);
                }
                println!("partial: {}", utterance);
            }
            moonshine::Output::Final { payload: _, utterance } => {
                let elapsed = start.elapsed();
                if first_result.is_none() {
                    first_result = Some(elapsed);
                }
                println!("final: {}", utterance);
                println!("latency: {:?}", first_result.unwrap());
                // Use _exit to bypass C++ atexit handlers that crash during TensorRT cleanup
                unsafe { libc::_exit(0) };
            }
        }
    }
}
