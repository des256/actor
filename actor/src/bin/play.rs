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
    let audio = audioout::Input {
        payload: (),
        data: reader.samples::<i16>().collect::<Result<Vec<i16>, _>>().unwrap(),
        stamp: epoch.current(),
    };
    let (audioout_handle, mut audioout_listener) = audioout::create::<()>(spec.sample_rate as usize, CHUNK_SIZE, None, &epoch);
    audioout_handle.send(audio);
    loop {
        match audioout_listener.recv().await {
            audioout::Status::Started(_) => {}
            audioout::Status::Finished { .. } => {}
            audioout::Status::Canceled { .. } => {}
        }
    }
}
