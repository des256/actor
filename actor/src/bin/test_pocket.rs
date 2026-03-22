use actor::*;
use std::time::Instant;

#[tokio::main]
async fn main() {
    let sentence = "After days in the shade, sunlight now cuts through the rough surface, \
        sending shimmering rays dancing across the rocky bed of the river, and illuminating \
        the patches of bright-green algae that carpeted the rocks of deeper, slower pools.";

    // init pocket tts
    let tensorrt = tensorrt::Tensorrt::new();
    let epoch = std::sync::Arc::new(Epoch::new());
    println!("Loading model...");
    let (pocket_handle, mut pocket_listener) = pocket::create::<()>(&tensorrt, "data/pocket/voices/stephen.bin", &epoch);
    println!("Model loaded.");

    // send sentence
    let start = Instant::now();
    let stamp = epoch.current();
    pocket_handle.send(pocket::Input {
        payload: (),
        sentence: sentence.to_string(),
        stamp,
    });

    // collect audio chunks
    let mut all_samples: Vec<i16> = Vec::new();
    let mut first_chunk = None;
    loop {
        let output = pocket_listener.recv().await;
        let elapsed = start.elapsed();
        if first_chunk.is_none() {
            first_chunk = Some(elapsed);
        }
        println!(
            "chunk {}: {} samples{}",
            output.index,
            output.audio.len(),
            if output.last { " (last)" } else { "" },
        );
        all_samples.extend_from_slice(&output.audio);
        if output.last {
            break;
        }
    }

    let total_elapsed = start.elapsed();
    let duration_secs = all_samples.len() as f64 / 24000.0;
    println!(
        "Generated {} samples ({:.2}s audio) in {:?}",
        all_samples.len(),
        duration_secs,
        total_elapsed,
    );
    println!("First chunk latency: {:?}", first_chunk.unwrap());

    // save to wav
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create("pocket.wav", spec).expect("Failed to create pocket.wav");
    for &sample in &all_samples {
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
    println!("Saved pocket.wav");

    // bypass TensorRT cleanup crash
    unsafe { libc::_exit(0) };
}
