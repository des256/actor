use {actor::*, std::sync::Arc};

const VOICE_PATH: &str = "data/tts/voices/hannah.bin";
const TTS_SAMPLE_RATE: usize = 24000;
const TTS_CHUNK_SIZE: usize = 512;

#[tokio::main]
async fn main() {
    let epoch = Arc::new(Epoch::new());
    let (audioout_handle, mut audioout_listener) = audioout::create(TTS_SAMPLE_RATE, TTS_CHUNK_SIZE, None, &epoch);
    let audioout_handle = Arc::new(audioout_handle);
    let onnx = onnx::Onnx::new(24);
    let (tts_handle, mut tts_listener) = tts::create::<()>(&onnx, onnx::Executor::Cpu, VOICE_PATH, &epoch);

    // spawn TTS pump
    tokio::spawn({
        let epoch = Arc::clone(&epoch);
        let audioout_handle = Arc::clone(&audioout_handle);
        async move {
            loop {
                let output = tts_listener.recv().await;
                println!("received audio {}", output.index);
                audioout_handle.send(audioout::Input {
                    payload: (),
                    data: output.audio,
                    stamp: epoch.current(),
                });
            }
        }
    });

    // spawn audioout status pump
    tokio::spawn({
        async move {
            loop {
                match audioout_listener.recv().await {
                    audioout::Status::Started(_) => {
                        println!("started");
                    }
                    audioout::Status::Finished { payload: _, index } => {
                        println!("finished {}", index);
                    }
                    audioout::Status::Canceled { payload: _, index } => {
                        println!("canceled {}", index);
                    }
                }
            }
        }
    });

    // send sentence to TTS
    println!("sending sentence...");
    tts_handle.send(tts::Input {
        payload: (),
        sentence: "The weather patterns in this region are dictated by the surrounding mountains. While the valleys remain dry, the peaks often collect moisture, creating a unique microclimate that shifts throughout the afternoon.".to_string(),
        stamp: epoch.current(),
    });

    // keep alive
    std::future::pending::<()>().await;
}
