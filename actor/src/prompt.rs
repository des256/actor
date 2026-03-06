use super::*;

pub async fn build_slm_main(
    model: slm::Model,
    identity: &str,
    personality: &str,
    tools: &str,
    facts: &str,
    history: &history::History,
) -> String {
    let (summary, history) = history.summarize(5).await;
    match model {
        slm::Model::Phi3 => {
            let mut short_history = String::new();
            for (role, message) in history.iter() {
                let role = match role {
                    history::Role::Robot => "assistant",
                    history::Role::User(_) => "user", // TODO: process ID
                };
                short_history.push_str(&format!(
                    "<|{}|>
{}<|end|>
",
                    role, message
                ));
            }
            format!(
                "<|system|>
{}
{}
{}
{}
{}<|end|>
{}<|assistant|>",
                identity, personality, tools, facts, summary, short_history,
            )
        }
        slm::Model::Llama33b | slm::Model::Llama38b => {
            let mut short_history = String::new();
            for (role, message) in history.iter() {
                let role = match role {
                    history::Role::Robot => "assistant",
                    history::Role::User(_) => "user", // TODO: process ID
                };
                short_history.push_str(&format!(
                    "<|start_header_id|>{}<|end_header_id|>

{}<|eot_id|>",
                    role, message,
                ));
            }
            format!(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{}
{}
{}
{}
{}<|eot_id|>{}<|start_header_id|>assistant<|end_header_id|>

",
                identity, personality, tools, facts, summary, short_history,
            )
        }
        slm::Model::Gemma34b => {
            let mut short_history = String::new();
            for (role, message) in history.iter() {
                let role = match role {
                    history::Role::Robot => "model",
                    history::Role::User(_) => "user", // TODO: process ID
                };
                short_history.push_str(&format!(
                    "<start_of_turn>{}
{}<end_of_turn>
",
                    role, message,
                ));
            }
            format!(
                "<start_of_turn>user
{}
{}
{}
{}
{}<end_of_turn>
{}
<start_of_turn>model
",
                identity, personality, tools, facts, summary, short_history,
            )
        }
        slm::Model::Smollm3 => {
            let mut short_history = String::new();
            for (role, message) in history.iter() {
                let role = match role {
                    history::Role::Robot => "assistant",
                    history::Role::User(_) => "user", // TODO: process ID
                };
                short_history.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, message,));
            }
            format!(
                "<|im_start|>system
{}
{}
{}
{}
{}<|im_end|>
{}<|im_start|>assistant
",
                identity, personality, tools, facts, summary, short_history,
            )
        }
    }
}

pub async fn build_xslm_main(
    model: xslm::Model,
    identity: &str,
    personality: &str,
    tools: &str,
    facts: &str,
    history: &history::History,
) -> String {
    let (summary, history) = history.summarize(5).await;
    match model {
        xslm::Model::Smollm2360m => {
            let mut short_history = String::new();
            for (role, message) in history.iter() {
                let role = match role {
                    history::Role::Robot => "assistant",
                    history::Role::User(_) => "user", // TODO: process ID
                };
                short_history.push_str(&format!(
                    "<|im_start|>{}
{}<|im_end|>
",
                    role, message,
                ));
            }
            format!(
                "<|im_start|>system
{}
{}
{}
{}
{}<|im_end|>
{}
<|im_start|>assistant\n",
                identity, personality, tools, facts, summary, short_history,
            )
        }
    }
}

const INTENT_SYSTEM: &str = "You are an expert at understanding what the user wants. Analyze the user input and return a JSON object with its intent.
Possible values: greeting, question, statement
Respond in JSON only. Do not use internal reasoning. respond immediately.
Examples:
- \"You are a jerk!\": {\"intent\": \"statement\"}
- \"How are you?\": {\"intent\": \"question\"}
- \"Would you like fries with that?\": {\"intent\": \"question\"}
- \"Good morning!\": {\"intent\": \"greeting\"}";

const INTENT_PREFIX: &str = "{\"intent\": \"";

pub async fn build_slm_intent(model: slm::Model, input: &str) -> String {
    match model {
        slm::Model::Phi3 => format!(
            "<|system|>\n{}<|end|>\n<|user|>\n{}<|end|>\n<|assistant|>\n{}",
            INTENT_SYSTEM, input, INTENT_PREFIX,
        ),
        slm::Model::Llama33b | slm::Model::Llama38b => format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{}",
            INTENT_SYSTEM, input, INTENT_PREFIX,
        ),
        slm::Model::Gemma34b => format!(
            "<start_of_turn>user\n{}\n{}<end_of_turn>\n<start_of_turn>model\n{}",
            INTENT_SYSTEM, input, INTENT_PREFIX,
        ),
        slm::Model::Smollm3 => format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}",
            INTENT_SYSTEM, input, INTENT_PREFIX,
        ),
    }
}

const NUANCE_SYSTEM: &str = "You are an expert at understanding emotional nuance. Analyze the user input and return a JSON object with its emotional nuance.
Possible values: sarcastic, ironic, joking, playful, serious, angry, frustrated, happy, excited, sad, depressed, anxious, scared, surprised, confused, bored, indifferent, annoyed, disappointed, embarrassed, guilty, ashamed, jealous, envious, greedy
Respond in JSON only. Do not use internal reasoning. respond immediately.
Examples:
- \"You are a jerk!\": {\"nuance\": \"aggressive\"}
- \"Wow, look at that airplane\": {\"nuance\": \"surprised\"}
- \"It's all going to shit\": {\"nuance\": \"depressed\"}";

const NUANCE_PREFIX: &str = "{\"nuance\": \"";

pub async fn build_slm_nuance(model: slm::Model, input: &str) -> String {
    match model {
        slm::Model::Phi3 => format!(
            "<|system|>\n{}<|end|>\n<|user|>\n{}<|end|>\n<|assistant|>\n{}",
            NUANCE_SYSTEM, input, NUANCE_PREFIX,
        ),
        slm::Model::Llama33b | slm::Model::Llama38b => format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{}",
            NUANCE_SYSTEM, input, NUANCE_PREFIX,
        ),
        slm::Model::Gemma34b => format!(
            "<start_of_turn>user\n{}\n{}<end_of_turn>\n<start_of_turn>model\n{}",
            NUANCE_SYSTEM, input, NUANCE_PREFIX,
        ),
        slm::Model::Smollm3 => format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}",
            NUANCE_SYSTEM, input, NUANCE_PREFIX,
        ),
    }
}
