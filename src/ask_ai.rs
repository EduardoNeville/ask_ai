use crate::config::{AiConfig, Framework, Question};
use crate::error::{AppError, Result};
use ollama_rs::{
    generation::chat::{request::ChatMessageRequest, ChatMessage, MessageRole},
    Ollama,
};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use serde_json::Value;
use std::env;

///### `get_openai_response`
///
///This is an internal function that interacts directly with OpenAI's API. It's called by `ask_question` when the configured `Framework` is `Framework::OpenAI`.
///
///---
///
///#### Signature:
///
///```rust,ignore
///async fn get_openai_response(question: Question, ai_config: &AiConfig) -> Result<String>
///```
///
///---
///
///#### Parameters:
///
///1. `ai_config`:
///   - Contains the user's configurations for the Framework provider, including the OpenAI model to query (`gpt-3.5-turbo`, `gpt-4`, etc.).
///
///2. `question`:
///   - A struct used to specify details about the prompt, context, and user question when querying OpenAI API.
///
///---
///
///#### Example Usage:
///
///This function is not meant to be directly used by end-users. Instead, it gets invoked through the `ask_question` function when the `llm` field of `AiConfig` is set to `Framework::OpenAI`.
async fn get_openai_response(question: Question, ai_config: &AiConfig) -> Result<String> {
    let api_key = env::var("OPENAI_API_KEY").map_err(|e| AppError::ApiError {
        model_name: ai_config.llm.to_string(),
        failure_str: format!("Missing or invalid OPENAI_API_KEY: {}", e),
    })?;

    // Messages array as before
    let mut messages = vec![];
    if let Some(sys_prompt) = &question.system_prompt {
        messages.push(serde_json::json!({
            "role": "system",
            "content": sys_prompt
        }));
    } else {
        messages.push(serde_json::json!({
            "role": "system",
            "content": ""
        }));
    }
    if let Some(prev_messages) = question.messages {
        for msg in prev_messages.iter() {
            if !msg.content.is_empty() {
                messages.push(serde_json::json!({
                    "role": "user",
                    "content": msg.content
                }));
            }
            if !msg.output.is_empty() {
                messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": msg.output
                }));
            }
        }
    }
    let usr_input = if question.new_prompt.is_empty() {
        ".".to_string()
    } else {
        question.new_prompt
    };
    messages.push(serde_json::json!({
        "role": "user",
        "content": usr_input
    }));

    let payload = serde_json::json!({
        "model": ai_config.model,
        "messages": messages
    });

    // Use env-var for endpoint (to allow httpmock substitution)
    let api_url = env::var("OPENAI_API_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1/chat/completions".to_string());

    let resp = reqwest::Client::new()
        .post(&api_url)
        .header(CONTENT_TYPE, "application/json")
        .header(AUTHORIZATION, format!("Bearer {}", api_key))
        .json(&payload)
        .send()
        .await
        .map_err(|e| AppError::ApiError {
            model_name: ai_config.llm.to_string(),
            failure_str: format!("Request error: {}", e),
        })?;

    if !resp.status().is_success() {
        let status = resp.status();
        let err_body = resp.text().await.unwrap_or_default();
        return Err(AppError::ApiError {
            model_name: ai_config.llm.to_string(),
            failure_str: format!("Status {}: {}", status, err_body),
        });
    }

    let response: Value = resp.json().await.map_err(|e| AppError::ModelError {
        model_name: ai_config.model.to_string(),
        failure_str: format!("Failed to parse JSON response: {}", e),
    })?;

    let answer = response["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| AppError::ModelError {
            model_name: ai_config.model.to_string(),
            failure_str: "Failed to extract content from OpenAI response".to_string(),
        })?
        .to_string();

    Ok(answer)
}

///### `get_anthropic_response`
///
///This internal function queries Anthropic's API to get a response from a Claude model (e.g., Claude-1, Claude-2). It is called by `ask_question` when the `Framework::Anthropic` is used in `AiConfig`.
///
///---
///
///#### Signature:
///
///```rust,ignore
///async fn get_anthropic_response(question: Question, ai_config: &AiConfig) -> Result<String>
///```
///
///---
///
///#### Parameters:
///
///1. `question`:
///   - A `Question` struct containing the system prompt, past messages, and user query.
///2. `ai_config`:
///   - A configuration object specifying the Anthropic Framework model and optional maximum token limit.
///
///---
///
///#### Example Usage:
///
///This function is also internal and should not be called directly. Use invocation through `ask_question`.
///
pub async fn get_anthropic_response(question: Question, ai_config: &AiConfig) -> Result<String> {
    let api_key = env::var("ANTHROPIC_API_KEY").map_err(|e| AppError::ApiError {
        model_name: ai_config.llm.to_string(),
        failure_str: format!("Missing or invalid ANTHROPIC_API_KEY: {}", e),
    })?;

    // Build messages array
    let mut messages = vec![];
    if let Some(prev_messages) = question.messages {
        for msg in prev_messages.iter() {
            if !msg.content.is_empty() {
                messages.push(serde_json::json!({
                    "role": "user",
                    "content": [{"type": "text", "text": msg.content}]
                }));
            }
            if !msg.output.is_empty() {
                messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": [{"type": "text", "text": msg.output}]
                }));
            }
        }
    }
    let usr_input = if question.new_prompt.is_empty() {
        ".".to_string()
    } else {
        question.new_prompt
    };
    messages.push(serde_json::json!({
        "role": "user",
        "content": [{"type": "text", "text": usr_input}]
    }));

    let system_prompt = question.system_prompt.unwrap_or_else(|| {String::from("")});
    let max_tokens = ai_config.max_token.unwrap_or(1024);

    let payload = serde_json::json!({
        "model": ai_config.model,
        "max_tokens": max_tokens,
        "messages": messages,
        "system": system_prompt
    });

    let api_url = env::var("ANTHROPIC_API_URL")
        .unwrap_or_else(|_| "https://api.anthropic.com/v1/messages".to_string());

    let resp = reqwest::Client::new()
        .post(&api_url)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header(CONTENT_TYPE, "application/json")
        .json(&payload)
        .send()
        .await
        .map_err(|e| AppError::ApiError {
            model_name: ai_config.llm.to_string(),
            failure_str: format!("Request error: {}", e),
        })?;

    if !resp.status().is_success() {
        let status = resp.status();
        let err_body = resp.text().await.unwrap_or_default();
        return Err(AppError::ApiError {
            model_name: ai_config.llm.to_string(),
            failure_str: format!("Status {}: {}", status, err_body),
        });
    }

    let response: Value = resp.json().await.map_err(|e| AppError::ModelError {
        model_name: ai_config.model.to_string(),
        failure_str: format!("Failed to parse JSON response: {}", e),
    })?;

    let answer = response["content"][0]["text"]
        .as_str()
        .ok_or_else(|| AppError::ModelError {
            model_name: ai_config.model.to_string(),
            failure_str: "Failed to extract content from Anthropic response".to_string(),
        })?
        .to_string();

    Ok(answer)
}

///### `get_ollama_response`
///
///An internal function that interacts with Ollama's API. Called when the Framework provider is `Framework::Ollama`.
///
///---
///
///#### Signature:
///
///```rust,ignore
///async fn get_ollama_response(question: Question, ai_config: &AiConfig) -> Result<String>
///```
///
///---
///
///#### Parameters:
///
///1. `question`:
///   - A `Question` struct with details of the system prompt, prior conversation, and user input.
///2. `ai_config`:
///   - Specifies the configured model and parameters for Ollama interaction.
///
///---
///
///#### Example Usage:
///
///This function is internal and used exclusively through `ask_question`.
async fn get_ollama_response(question: Question, ai_config: &AiConfig) -> Result<String> {
    let mut ollama = Ollama::default();

    // Creating the chain
    let mut msgs = vec![];

    if question.system_prompt.is_some() {
        msgs.push(ChatMessage {
            role: MessageRole::System,
            content: question.system_prompt.unwrap(),
            tool_calls: vec![],
            images: None,
        });
    } else {
        let default_sys_prompt = String::from("");
        msgs.push(ChatMessage {
            role: MessageRole::System,
            content: default_sys_prompt,
            tool_calls: vec![],
            images: None,
        });
    }

    if question.messages.is_some() {
        for msg in question.messages.unwrap().iter() {
            if !msg.content.is_empty() {
                msgs.push(ChatMessage {
                    role: MessageRole::User,
                    content: msg.content.to_owned(),
                    tool_calls: vec![],
                    images: None,
                });
            }

            if !msg.output.is_empty() {
                msgs.push(ChatMessage {
                    role: MessageRole::Assistant,
                    content: msg.output.to_owned(),
                    tool_calls: vec![],
                    images: None,
                });
            }
        }
    }

    if question.new_prompt.is_empty() {
        msgs.push(ChatMessage {
            role: MessageRole::User,
            content: String::from("."),
            tool_calls: vec![],
            images: None,
        });
    } else {
        msgs.push(ChatMessage {
            role: MessageRole::User,
            content: question.new_prompt.to_owned(),
            tool_calls: vec![],
            images: None,
        });
    }

    // Construct the chat completion request with the system and user messages
    let req = ChatMessageRequest::new(ai_config.model.to_owned(), msgs.to_owned());

    let result = ollama
        .send_chat_messages_with_history(&mut msgs, req)
        .await
        .map_err(|e| AppError::ModelError {
            model_name: ai_config.model.to_owned(),
            failure_str: e.to_string(),
        })?;

    let answer = result.message.content;

    Ok(answer)
}

pub async fn ask_question(ai_config: &AiConfig, question: Question) -> Result<String> {
    match ai_config.llm {
        Framework::OpenAI => get_openai_response(question, ai_config).await,
        Framework::Anthropic => get_anthropic_response(question, ai_config).await,
        Framework::Ollama => get_ollama_response(question, ai_config).await,
    }
}
