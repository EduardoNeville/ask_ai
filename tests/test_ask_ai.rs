use ask_ai::{
    config::{AiConfig, Framework, Question},
    ask_ai::ask_question,
    error::AppError,
};
use httpmock::prelude::*;
use serial_test::serial;
use std::env;

#[tokio::test]
#[serial]
async fn openai_reqwest_httpmock_success() {
    let server = MockServer::start();

    let mock = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions")
            .header("Authorization", "Bearer open_api_testkey")
            .header("Content-Type", "application/json")
            .body_contains("Say something, please.");
        then.status(200)
            .header("Content-Type", "application/json")
            .body(r#"{
                "choices": [
                    { "message": { "content": "Hello from OpenAI (mock)!" } }
                ]
            }"#);
    });

    env::set_var("OPENAI_API_KEY", "open_api_testkey");
    env::set_var("OPENAI_API_URL", &format!("{}/v1/chat/completions", server.base_url()));

    let ai_config = AiConfig {
        llm: Framework::OpenAI,
        model: "gpt-3.5-turbo".to_string(),
        max_token: Some(1000),
    };
    let question = Question {
        system_prompt: None,
        messages: None,
        new_prompt: "Say something, please.".to_string(),
    };

    let answer = ask_question(&ai_config, question).await.expect("Should succeed");
    mock.assert();
    assert_eq!(answer, "Hello from OpenAI (mock)!");

    env::remove_var("OPENAI_API_KEY");
    env::remove_var("OPENAI_API_URL");
}

#[tokio::test]
#[serial]
async fn anthropic_reqwest_httpmock_success() {
    let server = MockServer::start();

    let mock = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/messages")
            .header("x-api-key", "anthropic_testkey")
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .body_contains("Anthropic question!");
        then.status(200)
            .header("content-type", "application/json")
            .body(r#"{
                "content": [
                    { "text": "Answers from Claude (mock)!" }
                ]
            }"#);
    });

    env::set_var("ANTHROPIC_API_KEY", "anthropic_testkey");
    env::set_var("ANTHROPIC_API_URL", &format!("{}/v1/messages", server.base_url()));

    let ai_config = AiConfig {
        llm: Framework::Anthropic,
        model: "claude-2".to_string(),
        max_token: Some(80),
    };
    let question = Question {
        system_prompt: Some("You are friendly.".to_string()),
        messages: None,
        new_prompt: "Anthropic question!".to_string(),
    };

    let answer = ask_question(&ai_config, question).await.expect("Should succeed");
    mock.assert();
    assert_eq!(answer, "Answers from Claude (mock)!");

    env::remove_var("ANTHROPIC_API_KEY");
    env::remove_var("ANTHROPIC_API_URL");
}

#[tokio::test]
#[serial]
async fn openai_reqwest_httpmock_error() {
    let server = MockServer::start();

    let mock = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/chat/completions");
        then.status(401)
            .header("Content-Type", "application/json")
            .body(r#"{ "error": "unauthorized" }"#);
    });

    env::set_var("OPENAI_API_KEY", "bad_api_key");
    env::set_var("OPENAI_API_URL", &format!("{}/v1/chat/completions", server.base_url()));

    let ai_config = AiConfig {
        llm: Framework::OpenAI,
        model: "gpt-3.5-turbo".to_string(),
        max_token: Some(1000),
    };
    let question = Question {
        system_prompt: None,
        messages: None,
        new_prompt: "bad".to_string(),
    };

    match ask_question(&ai_config, question).await {
        Err(AppError::ApiError { model_name, failure_str }) => {
            assert_eq!(model_name, "openai");
            assert!(failure_str.contains("Status 401"));
        },
        other => panic!("Expected AppError::ApiError, got {:?}", other),
    };
    mock.assert();

    env::remove_var("OPENAI_API_KEY");
    env::remove_var("OPENAI_API_URL");
}

#[tokio::test]
#[serial]
async fn anthropic_reqwest_httpmock_error_model_parse() {
    let server = MockServer::start();

    let mock = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/messages");
        then.status(200)
            .header("content-type", "application/json")
            .body(r#"{
                "content": []
            }"#); // No text, causes ModelError in parsing
    });

    env::set_var("ANTHROPIC_API_KEY", "badkey");
    env::set_var("ANTHROPIC_API_URL", &format!("{}/v1/messages", server.base_url()));

    let ai_config = AiConfig {
        llm: Framework::Anthropic,
        model: "claude-2".to_string(),
        max_token: Some(80),
    };
    let question = Question {
        system_prompt: None,
        messages: None,
        new_prompt: "blah".to_string(),
    };

    match ask_question(&ai_config, question).await {
        Err(AppError::ModelError { model_name, failure_str }) => {
            assert_eq!(model_name, "claude-2");
            assert!(failure_str.contains("Failed to extract content"));
        },
        other => panic!("Expected AppError::ModelError, got {:?}", other),
    };
    mock.assert();

    env::remove_var("ANTHROPIC_API_KEY");
    env::remove_var("ANTHROPIC_API_URL");
}
