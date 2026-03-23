use {
    actor::*,
    std::{io, io::Write, sync::Arc},
};

#[tokio::main]
async fn main() {
    println!("loading model...");
    let epoch = Arc::new(Epoch::new());
    let tensorrt = tensorrt::Tensorrt::new();
    let (llm_handle, mut llm_listener) = llama3::create::<()>(&tensorrt, &epoch);
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
        let prompt = llm_handle.build_prompt(identity, &personality, tools, facts, &history).await;
        llm_handle.send(llama3::Input {
            payload: (),
            prompt,
            stamp: epoch.current(),
            max_tokens: 200,
            temperature: 0.7,
        });
        let mut response = String::new();
        loop {
            match llm_listener.recv().await {
                llama3::Output::Token {
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
                llama3::Output::Eos { payload: _, stamp } => {
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
