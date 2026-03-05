use tokio::sync::RwLock;

#[derive(Clone, Copy)]
pub enum Role {
    Robot,
    User(u64),
}

pub struct History {
    messages: RwLock<Vec<(Role, String)>>,
}

impl History {
    pub fn new() -> Self {
        Self {
            messages: RwLock::new(Vec::new()),
        }
    }

    pub async fn summarize(&self, max_messages: usize) -> (String, Vec<(Role, String)>) {
        // take last max_messages messages
        let read = self.messages.read().await;
        let max_messages = if max_messages > read.len() { read.len() } else { max_messages };
        let mut result = Vec::<(Role, String)>::new();
        for i in read.len() - max_messages..read.len() {
            result.push(read[i].clone());
        }

        // TODO: actually summarize the earlier messages
        let summary = String::new();

        (summary, result)
    }

    pub async fn add(&self, role: Role, message: String) {
        let mut write = self.messages.write().await;
        write.push((role, message));
    }
}
