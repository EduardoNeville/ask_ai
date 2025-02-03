use ollama_rs::{
    generation::chat::{
        request::ChatMessageRequest,
        ChatMessage,
        MessageRole
    }, 
    Ollama
};

use openai_api_rs::v1::{
    api::OpenAIClient,
    chat_completion::{self, ChatCompletionMessage, ChatCompletionRequest},
};
use anthropic_rs::{
    client::Client,
    completion::message::{Content, ContentType, Message, MessageRequest, Role, System, SystemPrompt},
    config::Config,
    models::claude::ClaudeModel,
};

use std::{env, str::FromStr};
use anyhow::Result;

use crate::config::{AiConfig, LLM, Question};
use crate::error::AppError;

///### `get_ollama_response`
///
///An internal function that interacts with Ollama's API. Called when the LLM provider is `LLM::Ollama`.
///
///---
///
///#### Signature:
///
///```rust
///async fn get_ollama_response(question: Question, user_config: &AiConfig) -> Result<String>
///```
///
///---
///
///#### Parameters:
///
///1. `question`: 
///   - A `Question` struct with details of the system prompt, prior conversation, and user input.
///2. `user_config`: 
///   - Specifies the configured model and parameters for Ollama interaction.
///
///---
///
///#### Example Usage:
///
///This function is internal and used exclusively through `ask_question`.
async fn get_ollama_response(question: Question, user_config: &AiConfig) -> Result<String> {
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
        let default_sys_prompt = String::from("You are helpful assistant. Answer the question consicely.");
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

    if question.user_input.is_empty() {
        msgs.push(ChatMessage {
            role: MessageRole::User,
            content: String::from("."),
            tool_calls: vec![],
            images: None,
        });
    } else {
        msgs.push(ChatMessage {
            role: MessageRole::User,
            content: question.user_input.to_owned(),
            tool_calls: vec![],
            images: None,
        });
    }


    // Construct the chat completion request with the system and user messages
    let req = ChatMessageRequest::new(user_config.model.to_owned(), msgs.to_owned());

    let result = ollama
        .send_chat_messages_with_history(&mut msgs, req)
        .await
        .map_err(|e| {
            AppError::ModelError {
                model_name: user_config.model.to_owned(),
                failure_str: e.to_string(),
            }
        })?;

    let answer = result.message.content;

    Ok(answer)
}

///### `get_anthropic_response`
///
///This internal function queries Anthropic's API to get a response from a Claude model (e.g., Claude-1, Claude-2). It is called by `ask_question` when the `LLM::Anthropic` is used in `AiConfig`.
///
///---
///
///#### Signature:
///
///```rust
///async fn get_anthropic_response(question: Question, user_config: &AiConfig) -> Result<String>
///```
///
///---
///
///#### Parameters:
///
///1. `question`: 
///   - A `Question` struct containing the system prompt, past messages, and user query.
///2. `user_config`: 
///   - A configuration object specifying the Anthropic LLM model and optional maximum token limit.
///
///---
///
///#### Example Usage:
///
///This function is also internal and should not be called directly. Use invocation through `ask_question`.
///
async fn get_anthropic_response(question: Question, user_config: &AiConfig) -> Result<String> {
    let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|e| {
        AppError::ApiError { 
            model_name: user_config.llm,
            failure_str: e.to_string()
        }
    })?;

    let config = Config::new(api_key);
    let client = Client::new(config).unwrap();
    let claude_model = ClaudeModel::from_str(&user_config.model).map_err(|e| {
        AppError::ModelError {
            model_name: user_config.model.to_owned(),
            failure_str: e.to_string(),
        }
    })?;

    // System prompt
    let sys_prompt = if question.system_prompt.is_some() {
        System::Structured({
            SystemPrompt {
                text: question.system_prompt.unwrap(),
                content_type: ContentType::Text,
                cache_control: None,
            }
        })
    } else {
        System::Text(String::from("You are helpful assistant. Answer the question consicely."))
    };

    // Creating the chain
    let mut msgs = vec![];
    if question.messages.is_some() {
        for msg in question.messages.unwrap().iter() {
            if !msg.content.is_empty() {
                msgs.push(Message {
                    role: Role::User,
                    content: vec![Content {
                        content_type: ContentType::Text,
                        text: msg.content.to_owned(),
                    }],
                });
            }

            if !msg.output.is_empty() {
                msgs.push(Message {
                    role: Role::Assistant,
                    content: vec![Content {
                        content_type: ContentType::Text,
                        text: msg.output.to_owned(),
                    }],
                });
            }
        }
    }

    let usr_input = if question.user_input.is_empty() {
        String::from(".")
    } else {
        question.user_input.to_owned()
    };

    msgs.push(Message {
        role: Role::User,
        content: vec![Content {
            content_type: ContentType::Text,
            text: usr_input,
        }],
    });

    let max_token = if user_config.max_token.is_some() {
        user_config.max_token.unwrap()
    } else {
        1024_u32
    };

    // Message Request Building
    let message = MessageRequest {
        model: claude_model,
        max_tokens: max_token,
        messages: msgs,
        system: Some(sys_prompt),
        ..Default::default()
    };

    // Find result
    let result = client.create_message(message).await.map_err(|e| {
        AppError::ModelError {
            model_name: user_config.model.to_owned(),
            failure_str: e.to_string(),
        }
    })?;

    let answer = result.content[0].text.to_owned();

    Ok(answer)
}

///### `get_openai_response`
///
///This is an internal function that interacts directly with OpenAI's API. It's called by `ask_question` when the configured `LLM` is `LLM::OpenAI`.
///
///---
///
///#### Signature:
///
///```rust
///async fn get_openai_response(question: Question, user_config: &AiConfig) -> Result<String>
///```
///
///---
///
///#### Parameters:
///
///1. `user_config`: 
///   - Contains the user's configurations for the LLM provider, including the OpenAI model to query (`gpt-3.5-turbo`, `gpt-4`, etc.).
///
///2. `question`: 
///   - A struct used to specify details about the prompt, context, and user question when querying OpenAI API.
///
///---
///
///#### Example Usage:
///
///This function is not meant to be directly used by end-users. Instead, it gets invoked through the `ask_question` function when the `llm` field of `AiConfig` is set to `LLM::OpenAI`.
async fn get_openai_response(question: Question, user_config: &AiConfig) -> Result<String> {
    // Retrieve the OpenAI API key from the environment securely
    let api_key =
        env::var("OPENAI_API_KEY").map_err(|e| 
        AppError::ApiError {
            model_name: user_config.llm,
            failure_str: e.to_string(),
        }
    )?;

    // Initialize the OpenAI client with the API key
    let client = OpenAIClient::builder()
        .with_api_key(api_key)
        .build()
        .unwrap();

    let mut msgs = vec![];
    if question.system_prompt.is_some() {
        msgs.push(ChatCompletionMessage {
            role: chat_completion::MessageRole::system,
            content: chat_completion::Content::Text(question.system_prompt.unwrap().to_owned()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
    } else {
        let default_sys_prompt = String::from("You are helpful assistant. Answer the question consicely.");
        msgs.push(ChatCompletionMessage {
            role: chat_completion::MessageRole::system,
            content: chat_completion::Content::Text(default_sys_prompt),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }

    if question.messages.is_some() {
        for msg in question.messages.unwrap().iter() {
            if !msg.content.is_empty() {
                msgs.push(ChatCompletionMessage {
                    role: chat_completion::MessageRole::user,
                    content: chat_completion::Content::Text(msg.content.to_owned()),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                });
            }

            if !msg.output.is_empty() {
                msgs.push(ChatCompletionMessage {
                    role: chat_completion::MessageRole::assistant,
                    content: chat_completion::Content::Text(msg.output.to_owned()),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                });
            }

        }
    }

    let usr_input = if question.user_input.is_empty() {
        String::from(".")
    } else {
        question.user_input.to_owned()
    };

    msgs.push(ChatCompletionMessage {
        role: chat_completion::MessageRole::user,
        content: chat_completion::Content::Text(usr_input),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    });

    // Construct the chat completion request with the system and user messages
    let req = ChatCompletionRequest::new(user_config.model.to_owned(), msgs);

    let result = client.chat_completion(req).await.map_err(|e| {
        AppError::ModelError {
            model_name: user_config.model.to_owned(),
            failure_str: e.to_string(),
        }
    })?;
    let answer = result.choices[0].message.content.to_owned().unwrap();

    Ok(answer)
}



///### `ask_question`
///
///This function is the main entry point for querying one of the supported LLMs (OpenAI, Anthropic, or Ollama). It uses the provided configuration (`AiConfig`) and the inputs (`Question`) to produce a response from the AI model.
///
///---
///
///#### Signature: 
///
///```rust
///pub async fn ask_question(user_config: &AiConfig, question: Question) -> Result<String>
///```
///
///---
///
///#### Parameters:
///
///1. `user_config`: 
///   - A struct of type `AiConfig` that specifies which LLM provider, model, and optional maximum token limit to use. It wraps the properties of the model and provider required to query the LLM.
///
///2. `question`: 
///   - A struct of type `Question` struct containing:
///     - `system_prompt`: Optional prompt instructing the AI on how to behave or respond.
///     - `messages`: Optional list of prior chat messages (for maintaining conversation context).
///     - `user_input`: The user's current question or input.
///
///---
///
///#### Return:
///
///- **`Result<String>`**: Returns the model's response on success or an application-defined error (`AppError`) in case of failure.
pub async fn ask_question(user_config: &AiConfig, question: Question) -> Result<String> {
    match user_config.llm {
        LLM::OpenAI => {
            let ans = get_openai_response(question, user_config).await?;
            Ok(ans)
        }
        LLM::Anthropic => {
            let ans = get_anthropic_response(question, user_config).await?;
            Ok(ans)
        }
        LLM::Ollama => {
            let ans = get_ollama_response(question, user_config).await?;
            Ok(ans)
        }
    }
}
