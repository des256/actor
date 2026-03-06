use {
    actor::*,
    std::{
        io::{Write, stdout},
        path::PathBuf,
    },
};

const SAMPLE_RATE: u32 = 24000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_stdout_logger();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        println!("Usage: {} <voice.wav> <output.bin>", args[0]);
        println!();
        println!("Encodes a voice WAV file through the mimi encoder and saves the");
        println!("resulting latent tensor to a binary file.");
        println!();
        println!("Output format (little-endian):");
        println!("  4 bytes  - number of dimensions (u32)");
        println!("  N×8 bytes - each dimension (u64)");
        println!("  remainder - raw f32 data");
        std::process::exit(1);
    }
    let voice_path = &args[1];
    let output_path = &args[2];

    // Load mimi encoder model
    let encoder_path = PathBuf::from("data/tts/mimi_encoder.onnx");
    if !encoder_path.exists() {
        panic!("Missing: {}", encoder_path.display());
    }

    // Read WAV file
    print!("loading WAV: {}", voice_path);
    stdout().flush().unwrap();
    let mut reader = hound::WavReader::open(voice_path)?;
    let spec = reader.spec();

    if spec.channels != 1 {
        panic!("WAV must be mono, got {} channels", spec.channels);
    }
    if spec.sample_rate != SAMPLE_RATE {
        panic!("WAV must be {}Hz, got {}Hz", SAMPLE_RATE, spec.sample_rate);
    }
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|val| val as f32 / max_val))
                .collect::<Result<Vec<_>, _>>()?
        }
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
    };
    println!(
        "done: {} samples, {}Hz, {} bit",
        samples.len(),
        spec.sample_rate,
        spec.bits_per_sample,
    );

    // Initialize inference and create encoder session
    print!("loading mimi encoder...");
    stdout().flush().unwrap();
    let onnx = onnx::Onnx::new(24);
    let session = onnx.create_session(onnx::Executor::Cuda(0), onnx::OptimizationLevel::EnableAll, 4, encoder_path);
    println!(" done.");

    // Encode: input shape [1, 1, audio_len], output "latents"
    print!("encoding {} samples...", samples.len());
    stdout().flush().unwrap();
    let audio_tensor = onnx::Value::from_slice::<f32>(&onnx, &[1, 1, samples.len()], &samples);
    let outputs = session.run(&[("audio", &audio_tensor)], &["latents"]);
    let shape = outputs[0].tensor_shape();
    let latents = outputs[0].extract_tensor::<f32>();
    let shape_str: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    println!("encoded latents: shape [{}], {} floats", shape_str.join(", "), latents.len());

    // Write binary output: ndims (u32) + dims (u64 each) + raw f32 data
    let mut file = std::fs::File::create(output_path)?;
    let ndims = shape.len() as u32;
    file.write_all(&ndims.to_le_bytes())?;
    for &dim in &shape {
        file.write_all(&(dim as u64).to_le_bytes())?;
    }
    for &val in latents {
        file.write_all(&val.to_le_bytes())?;
    }

    let file_size = std::fs::metadata(output_path)?.len();
    println!("Wrote {} bytes to {}", file_size, output_path);

    Ok(())
}
