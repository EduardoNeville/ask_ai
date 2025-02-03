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

/// This function is used to query using the openai API
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
