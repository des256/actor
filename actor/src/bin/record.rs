use {
    actor::*,
    hound::{SampleFormat, WavSpec, WavWriter},
};

const SAMPLE_RATE: usize = 24000;
const CHUNK_SIZE: usize = 4096;
const RECORDING_SECONDS: usize = 10;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        panic!("usage: record <wav_file>");
    }
    let path = &args[1];
    let mut audioin_listener = audioin::create(SAMPLE_RATE, CHUNK_SIZE, None, 3);
    let spec = WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE as u32,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec).unwrap();
    println!("recording {} seconds...", RECORDING_SECONDS);
    let mut total_samples = 0usize;
    while total_samples < SAMPLE_RATE * RECORDING_SECONDS {
        let audio = audioin_listener.recv().await;
        total_samples += audio.len();
        for sample in audio {
            writer.write_sample(sample).unwrap();
        }
    }
    println!("recording finished");
    writer.finalize().unwrap();
}
