use serde::{Serialize, Deserialize};

/// Enum for different LLM providers
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum LLM {
    OpenAI,
    Anthropic,
    Ollama,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AiConfig {
    pub llm: LLM,
    pub model: String,
    pub max_token: Option<u32>, // Optional, defaults to some number provided by LLM API
}

#[derive(Debug)]
pub struct AiPrompt {
    pub content: String,
    pub output: String,
}

pub struct Question {
    pub system_prompt: Option<String>,
    pub messages: Option<Vec<AiPrompt>>,
    pub user_input: String,
}

