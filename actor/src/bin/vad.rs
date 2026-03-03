use actor::*;

const SAMPLE_RATE: usize = 16000;
const FRAME_SIZE: usize = 512;
const FRAMES_PER_CHUNK: usize = 8;

#[tokio::main]
async fn main() {
    let mut audioin_listener = create_audioin(SAMPLE_RATE, FRAMES_PER_CHUNK * FRAME_SIZE, None, 3);
    let onnx = Onnx::new(17);
    let mut vad = Vad::new(&onnx, &Executor::Cpu, SAMPLE_RATE);
    loop {
        let frame = audioin_listener.recv().await;
        let mut probability = 0.0f32;
        for i in 0..FRAMES_PER_CHUNK {
            probability += vad.analyze(&frame[i * FRAME_SIZE..(i + 1) * FRAME_SIZE]);
        }
        probability /= FRAMES_PER_CHUNK as f32;
        println!("{:.2}%", probability * 100.0);
    }
}
