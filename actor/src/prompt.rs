use super::*;

// 1. persona instructions
// 2. tool usage
// 3. basic facts
// 4. chat history
// 5. main setup with recap

pub fn build(
    model: llm::Model,
    identity: &str,
    personality: &str,
    tools: &str,
    facts: &str,
    history: &history::ChatHistory,
) -> String {
    let (summary, history) = history.summarize(5);
    match model {
        llm::Model::Phi3 => {
            let mut short_history = String::new();
            for (role, message) in history.iter() {
                let role = match role {
                    history::Role::Robot => "assistant",
                    history::Role::User(_) => "user", // TODO: process ID
                };
                short_history.push_str(&format!("<|{}|>\n{}<|end|>\n", role, message));
            }
            format!(
                "<|system|>\n{}\n{}\n{}\n{}\n{}<|end|>\n{}<|assistant|>",
                identity, personality, tools, facts, summary, short_history,
            )
        }
        llm::Model::Llama33b | llm::Model::Llama38b => {
            let mut short_history = String::new();
            for (role, message) in history.iter() {
                let role = match role {
                    history::Role::Robot => "assistant",
                    history::Role::User(_) => "user", // TODO: process ID
                };
                short_history.push_str(&format!(
                    "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                    role, message,
                ));
            }
            format!(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}\n{}\n{}\n{}\n{}<|eot_id|>{}<|start_header_id|>assistant<|end_header_id|>\n\n",
                identity, personality, tools, facts, summary, short_history,
            )
        }
        llm::Model::Gemma34b => {
            let mut short_history = String::new();
            for (role, message) in history.iter() {
                let role = match role {
                    history::Role::Robot => "model",
                    history::Role::User(_) => "user", // TODO: process ID
                };
                short_history.push_str(&format!("<start_of_turn>{}\n{}<end_of_turn>\n", role, message,));
            }
            format!(
                "<start_of_turn>user\n{}\n{}\n{}\n{}\n{}<end_of_turn>\n{}<start_of_turn>model\n",
                identity, personality, tools, facts, summary, short_history,
            )
        }
        llm::Model::Smollm3 => {
            let mut short_history = String::new();
            for (role, message) in history.iter() {
                let role = match role {
                    history::Role::Robot => "assistant",
                    history::Role::User(_) => "user", // TODO: process ID
                };
                short_history.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, message,));
            }
            format!(
                "<|im_start|>system\n{}\n{}\n{}\n{}\n{}<|im_end|>\n{}<|im_start|>assistant\n",
                identity, personality, tools, facts, summary, short_history,
            )
        }
    }
}
