use {actor::*, hound::WavReader, std::sync::Arc};

const CHUNK_SIZE: usize = 4096;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        panic!("usage: play <wav_file>");
    }
    let path = &args[1];
    let mut reader = WavReader::open(path).unwrap();
    let spec = reader.spec();
    if spec.channels != 1 {
        panic!("input file must be mono");
    }
    let epoch = Arc::new(Epoch::new());
    let chunk = AudioOutChunk {
        payload: (),
        data: reader
            .samples::<i16>()
            .collect::<Result<Vec<i16>, _>>()
            .unwrap(),
        epoch: epoch.current(),
    };
    let (audioout_handle, mut audioout_listener) =
        create_audioout(spec.sample_rate as usize, CHUNK_SIZE, None, &epoch);
    audioout_handle.send(chunk);
    while let Some(status) = audioout_listener.recv().await {
        match status {
            AudioOutStatus::Started(_) => {
                println!("started");
            }
            AudioOutStatus::Finished { payload: _, index } => {
                println!("finished: {}", index);
            }
            AudioOutStatus::Canceled { payload: _, index } => {
                println!("canceled: {}", index);
            }
        }
    }
}
