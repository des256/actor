#[derive(Clone, Copy)]
pub enum Role {
    Robot,
    User(u64),
}

pub struct ChatHistory {
    messages: Vec<(Role, String)>,
}

impl ChatHistory {
    pub fn new() -> Self {
        Self { messages: Vec::new() }
    }

    pub fn summarize(&self, max_messages: usize) -> (String, Vec<(Role, String)>) {
        // take last max_messages messages
        let mut messages = Vec::<(Role, String)>::new();
        let max_messages = if max_messages > self.messages.len() {
            self.messages.len()
        } else {
            max_messages
        };
        for i in self.messages.len() - max_messages..self.messages.len() {
            messages.push(self.messages[i].clone());
        }

        // TODO: actually summarize the earlier messages
        let summary = String::new();

        (summary, messages)
    }

    pub fn add(&mut self, role: Role, message: String) {
        self.messages.push((role, message));
    }
}
