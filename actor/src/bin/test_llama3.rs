use {
    actor::*,
    std::{io, io::Write, sync::Arc, time::Instant},
};

#[tokio::main]
async fn main() {
    println!("loading model...");
    let epoch = Arc::new(Epoch::new());
    let (llm_handle, mut llm_listener) = llama3::create::<()>(&epoch);

    // warmup
    println!("warming up...");
    let prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
You're a useful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
Hi"
    .to_string();
    llm_handle.send(llama3::Input {
        payload: (),
        prompt,
        stamp: epoch.current(),
        max_tokens: 50,
    });
    while let llama3::Output::Token { .. } = llm_listener.recv().await {}

    // measure TTFT
    println!("testing...");
    let prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
You're a useful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
Hi, tell me something about cars<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        .to_string();
    let start = Instant::now();
    let mut ttft_ms: Option<u64> = None;
    llm_handle.send(llama3::Input {
        payload: (),
        prompt,
        stamp: epoch.current(),
        max_tokens: 50,
    });
    let mut response = String::new();
    loop {
        match llm_listener.recv().await {
            llama3::Output::Token { token, .. } => {
                if ttft_ms.is_none() {
                    ttft_ms = Some(start.elapsed().as_millis() as u64);
                }
                print!("{}", token);
                io::stdout().flush().unwrap();
                response.push_str(&token);
            }
            llama3::Output::Eos { .. } => {
                println!();
                break;
            }
        }
    }
    println!("TTFT: {}ms", ttft_ms.unwrap());
}
