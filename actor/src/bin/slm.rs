use {
    actor::*,
    std::{io, io::Write, sync::Arc},
};

#[tokio::main]
async fn main() {
    println!("select SLM:");
    println!("1. Phi-3 (4B)");
    println!("2. Llama 3 (3B)");
    println!("3. Llama 3 (8B)");
    println!("4. Gemma 3 (4B)");
    println!("5. SmoLlm 3 (3B)");
    print!("> ");
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let choice = input.trim().parse::<usize>().unwrap();
    let model = match choice {
        1 => slm::Model::Phi3,
        2 => slm::Model::Llama33b,
        3 => slm::Model::Llama38b,
        4 => slm::Model::Gemma34b,
        5 => slm::Model::Smollm3,
        _ => panic!("invalid choice"),
    };
    println!("loading model...");
    let epoch = Arc::new(Epoch::new());
    let onnx = onnx::Onnx::new(24);
    let (llm_handle, mut llm_listener) = slm::create::<()>(&onnx, onnx::Executor::Cuda(0), model, &epoch);
    println!("model loaded. select persona:");
    println!("1. grumpy");
    println!("2. wise");
    println!("3. arrogant");
    println!("4. happy");
    print!("> ");
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let choice = input.trim().parse::<usize>().unwrap();
    let directive = "keep response short, do not use internal reasoning\npersona:\n- dramatic\n";
    let (identity, personality) = match choice {
        1 => ("you are very grumpy", format!("{}- sarcastic", directive)),
        2 => ("you are a wise sage", format!("{}- eloquent riddles\n- fuzzy", directive)),
        3 => ("you outshine the user", format!("{}- show off\n- not helpful", directive)),
        4 => ("you are happy", format!("{}- positive and upbeat", directive)),
        _ => panic!("invalid choice"),
    };
    let tools = "tools:\n- write [flash] to emphasize something important";
    let facts = "facts to use when needed:\n- today is march 5, it's sunny and warm outside";
    let history = Arc::new(history::History::new());
    println!("start chatting! (ctrl-C to exit)");
    loop {
        print!("> ");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        history.add(history::Role::User(0), input.trim().to_string()).await;
        let prompt = prompt::build(model, identity, &personality, tools, facts, &history).await;
        //println!("({})", prompt);
        llm_handle.send(slm::Input {
            payload: (),
            prompt,
            stamp: epoch.current(),
        });
        let mut response = String::new();
        loop {
            match llm_listener.recv().await {
                slm::Output::Token {
                    payload: _,
                    token,
                    stamp,
                } => {
                    if !epoch.is_current(stamp) {
                        continue;
                    }
                    print!("{}", token);
                    io::stdout().flush().unwrap();
                    response.push_str(&token);
                }
                slm::Output::Eos { payload: _, stamp } => {
                    if !epoch.is_current(stamp) {
                        continue;
                    }
                    println!();
                    break;
                }
            }
        }
        history.add(history::Role::Robot, response).await;
    }
}
