use {
    actor::*,
    std::{io, io::Write, sync::Arc},
};

#[tokio::main]
async fn main() {
    println!("(X)SLM nuance analyzer");
    let epoch = Arc::new(Epoch::new());
    let onnx = onnx::Onnx::new(24);
    let slm_core = Arc::new(slm::Core::new(&onnx, onnx::Executor::Cuda(0), slm::Model::Smollm3));
    let (slm_handle, mut slm_listener) = slm::create::<()>(&slm_core, &epoch);
    loop {
        print!("> ");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        epoch.advance();
        let prompt = prompt::build_slm_nuance(slm::Model::Smollm3, input.trim()).await;
        slm_handle.send(slm::Input {
            payload: (),
            prompt,
            stamp: epoch.current(),
            max_tokens: 10,
        });
        let mut response = String::new();
        loop {
            match slm_listener.recv().await {
                slm::Output::Token { token, stamp, .. } => {
                    if !epoch.is_current(stamp) {
                        continue;
                    }
                    if token.contains('}') {
                        break;
                    }
                    response.push_str(&token);
                }
                slm::Output::Eos { stamp, .. } => {
                    if !epoch.is_current(stamp) {
                        continue;
                    }
                    break;
                }
            }
        }
        let nuance = response.trim();
        println!("nuance: {}", nuance);
    }
}
