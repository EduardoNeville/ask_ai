use serde::{Deserialize, Serialize};
use std::fmt;

/// Enum representing different Large Language Model (LLM) providers.
///
/// This enum is used to specify which LLM framework to use when interacting with AI models.
/// It supports three providers: OpenAI, Anthropic, and Ollama.
///
/// ### Example Usage:
///
/// ```rust,ignore
/// use ask_ai::config::Framework;
///
/// let framework = Framework::OpenAI; // Use OpenAI as the LLM provider
/// assert_eq!(framework.to_string(), "openai");
/// ```
#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum Framework {
    /// Represents the OpenAI framework (e.g., GPT models).
    OpenAI,
    /// Represents the Anthropic framework (e.g., Claude models).
    Anthropic,
    /// Represents the Ollama framework (e.g., locally hosted models).
    Ollama,
}

impl fmt::Display for Framework {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Framework::OpenAI => write!(f, "openai"),
            Framework::Anthropic => write!(f, "anthropic"),
            Framework::Ollama => write!(f, "ollama"),
        }
    }
}

/// Configuration for interacting with an AI model.
///
/// This struct defines the necessary configuration for querying an AI model, including the
/// framework provider, the specific model to use, and an optional maximum token limit for responses.
///
/// ### Example Usage:
///
/// ```rust,ignore
/// use ask_ai::config::{AiConfig, Framework};
///
/// let ai_config = AiConfig {
///     llm: Framework::OpenAI,           // Specify the framework provider
///     model: "gpt-4".to_string(),       // Specify the model to use
///     max_token: Some(1000),            // Optional: Limit the response to 1000 tokens
/// };
/// ```
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AiConfig {
    /// The LLM framework provider to use (e.g., OpenAI, Anthropic, Ollama).
    pub llm: Framework,
    /// The specific model to query (e.g., "gpt-4", "claude-2", "llama2").
    pub model: String,
    /// Optional maximum token limit for the AI's response. If `None`, the default limit
    /// provided by the LLM API will be used.
    pub max_token: Option<u32>,
}

/// Represents a single prompt and its corresponding AI response.
///
/// This struct is used to store a user's input (`content`) and the AI's output (`output`).
/// It is typically used in a conversation history to maintain context.
///
/// ### Example Usage:
///
/// ```rust,ignore
/// use ask_ai::config::AiPrompt;
///
/// let prompt = AiPrompt {
///     content: "What is Rust?".to_string(), // User's input
///     output: "Rust is a systems programming language...".to_string(), // AI's response
/// };
/// ```
#[derive(Debug, Clone)]
pub struct AiPrompt {
    /// The user's input or question.
    pub content: String,
    /// The AI's response to the input.
    pub output: String,
}

/// Represents a question or query to the AI, including optional context.
///
/// This struct is used to define a question or query to the AI, along with optional
/// system prompts and conversation history for context.
///
/// ### Example Usage:
///
/// ```rust,ignore
/// use ask_ai::config::{Question, AiPrompt};
///
/// let question = Question {
///     system_prompt: Some("You are a helpful assistant.".to_string()), // Optional system prompt
///     messages: Some(vec![
///         AiPrompt {
///             content: "What is Rust?".to_string(),
///             output: "Rust is a systems programming language...".to_string(),
///         },
///     ]), // Optional conversation history
///     new_prompt: "Tell me more about Rust.".to_string(), // New user prompt
/// };
/// ```
#[derive(Debug, Clone)]
pub struct Question {
    /// An optional system prompt to instruct the AI on how to behave.
    /// For example, "You are a helpful assistant."
    pub system_prompt: Option<String>,
    /// An optional list of prior messages in the conversation for context.
    /// Each message includes the user's input and the AI's response.
    pub messages: Option<Vec<AiPrompt>>,
    /// The new prompt or question from the user.
    pub new_prompt: String,
}
