use ask_ai::{
    ask_ai::{ask_question, get_anthropic_response},
    config::{AiConfig, AiPrompt, Framework, Question},
    error::AppError,
};

use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
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
            .body(
                r#"{
                "choices": [
                    { "message": { "content": "Hello from OpenAI (mock)!" } }
                ]
            }"#,
            );
    });

    env::set_var("OPENAI_API_KEY", "open_api_testkey");
    env::set_var(
        "OPENAI_API_URL",
        &format!("{}/v1/chat/completions", server.base_url()),
    );

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

    let answer = ask_question(&ai_config, question)
        .await
        .expect("Should succeed");
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
            .body(
                r#"{
                "content": [
                    { "text": "Answers from Claude (mock)!" }
                ]
            }"#,
            );
    });

    env::set_var("ANTHROPIC_API_KEY", "anthropic_testkey");
    env::set_var(
        "ANTHROPIC_API_URL",
        &format!("{}/v1/messages", server.base_url()),
    );

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

    let answer = ask_question(&ai_config, question)
        .await
        .expect("Should succeed");
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
        when.method(POST).path("/v1/chat/completions");
        then.status(401)
            .header("Content-Type", "application/json")
            .body(r#"{ "error": "unauthorized" }"#);
    });

    env::set_var("OPENAI_API_KEY", "bad_api_key");
    env::set_var(
        "OPENAI_API_URL",
        &format!("{}/v1/chat/completions", server.base_url()),
    );

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
        Err(AppError::ApiError {
            model_name,
            failure_str,
        }) => {
            assert_eq!(model_name, "openai");
            assert!(failure_str.contains("Status 401"));
        }
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
        when.method(POST).path("/v1/messages");
        then.status(200)
            .header("content-type", "application/json")
            .body(
                r#"{
                "content": []
            }"#,
            ); // No text, causes ModelError in parsing
    });

    env::set_var("ANTHROPIC_API_KEY", "badkey");
    env::set_var(
        "ANTHROPIC_API_URL",
        &format!("{}/v1/messages", server.base_url()),
    );

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
        Err(AppError::ModelError {
            model_name,
            failure_str,
        }) => {
            assert_eq!(model_name, "claude-2");
            assert!(failure_str.contains("Failed to extract content"));
        }
        other => panic!("Expected AppError::ModelError, got {:?}", other),
    };
    mock.assert();

    env::remove_var("ANTHROPIC_API_KEY");
    env::remove_var("ANTHROPIC_API_URL");
}


#[tokio::test]
#[serial]
async fn anthropic_error_unsupported_model_httpmock() {
    use ask_ai::error::AppError;
    use ask_ai::config::{AiConfig, Framework, Question};

    let server = MockServer::start();
    let model_name = "claude-nonexistent-model";

    // The mock API responds with a model_not_supported error as Anthropic would
    let body = format!(
        r#"{{
            "type": "error",
            "error": {{
                "type": "model_not_supported",
                "message": "Model not supported: {}"
            }}
        }}"#,
        model_name
    );

    let mock = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/messages")
            .header("x-api-key", "fake_anthropic_key");
        then.status(400)
            .header("content-type", "application/json")
            .body(body.clone());
    });

    env::set_var("ANTHROPIC_API_KEY", "fake_anthropic_key");
    env::set_var(
        "ANTHROPIC_API_URL",
        &format!("{}/v1/messages", server.base_url()),
    );

    let ai_config = AiConfig {
        llm: Framework::Anthropic,
        model: model_name.to_string(),
        max_token: Some(256),
    };
    let question = Question {
        system_prompt: None,
        messages: None,
        new_prompt: "Will this fail?".to_string(),
    };

    let err = ask_question(&ai_config, question)
        .await
        .err()
        .expect("Should fail for unsupported model");

    match err {
        AppError::ApiError { model_name: n, failure_str } => {
            assert_eq!(n, "anthropic");
            assert!(failure_str.contains("model_not_supported") || failure_str.contains("Model not supported"),
                "failure_str: {}", failure_str);
        }
        other => panic!("Expected AppError::ApiError, got {:?}", other),
    }
    mock.assert();

    env::remove_var("ANTHROPIC_API_KEY");
    env::remove_var("ANTHROPIC_API_URL");
}

#[tokio::test]
#[serial]
async fn anthropic_real_api_key_unsupported_model() {
    use ask_ai::error::AppError;
    use ask_ai::config::{AiConfig, Framework, Question};

    // Run this test only if ANTHROPIC_API_KEY is set (to avoid spurious failure in CI)
    if env::var("ANTHROPIC_API_KEY").is_err() {
        eprintln!("Skipping test: ANTHROPIC_API_KEY not set");
        return;
    }

    let ai_config = AiConfig {
        llm: Framework::Anthropic,
        model: "claude-non-existing-xyz".to_string(),
        max_token: Some(128),
    };
    let question = Question {
        system_prompt: Some("You are a helpful assistant.".to_string()),
        messages: None,
        new_prompt: "Does this model exist?".to_string(),
    };

    let err = ask_question(&ai_config, question).await.err().expect("Should fail unsupported model");

    match err {
        AppError::ApiError { model_name: n, failure_str } => {
            assert_eq!(n, "anthropic");
            let failure_lc = failure_str.to_lowercase();
            assert!(
                failure_lc.contains("model not supported")
                || failure_lc.contains("unsupported model")
                || failure_lc.contains("not_found_error")
                || failure_lc.contains("404"),
                "failure_str: {}", failure_str
            );
        }
        other => panic!("Expected AppError::ApiError, got {:?}", other),
    }
}

#[tokio::test]
#[serial]
async fn replicate_get_anthropic_response_step_by_step() {
    use serde_json::json;
    use std::env;
    use ask_ai::config::{AiConfig, Framework, Question, AiPrompt};

    // ---- 1. Env var extraction ----
    let api_key = match env::var("ANTHROPIC_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            eprintln!("Skipping test: ANTHROPIC_API_KEY not set");
            return;
        }
    };

    let ai_config = AiConfig {
        llm: Framework::Anthropic,
        model: "claude-2".to_string(),
        max_token: Some(256),
    };

    // ---- 2. Build Question with some dummy conversation ----
    let question = Question {
        system_prompt: Some("You are a test system.".to_string()),
        messages: Some(vec![
            AiPrompt { content: "Hello, Claude!".into(), output: "Hi there, user!".into() },
            AiPrompt { content: "What's up?".into(), output: "Answer.".into() }
        ]),
        new_prompt: "Why is the sky blue?".into(),
    };

    // ---- 3. Build messages as in the function ----
    let mut messages = vec![];
    if let Some(prev_messages) = question.messages.clone() {
        for msg in prev_messages.iter() {
            if !msg.content.is_empty() {
                let user_msg = json!({
                    "role": "user",
                    "content": [ { "type": "text", "text": msg.content } ]
                });
                assert!(user_msg["content"][0]["text"].as_str().unwrap().len() > 0);
                messages.push(user_msg);
            }
            if !msg.output.is_empty() {
                let assistant_msg = json!({
                    "role": "assistant",
                    "content": [ { "type": "text", "text": msg.output } ]
                });
                assert!(assistant_msg["content"][0]["text"].as_str().unwrap().len() > 0);
                messages.push(assistant_msg);
            }
        }
    }
    // Add the new prompt
    let usr_input = if question.new_prompt.is_empty() {
        "This is the user input".to_string()
    } else {
        question.new_prompt.clone()
    };
    let last_msg = json!({
        "role": "user",
        "content": [ { "type": "text", "text": usr_input.clone() } ]
    });
    assert_eq!(last_msg["content"][0]["text"], usr_input.as_str());
    messages.push(last_msg);

    // At this point, messages should be (user, assistant, user, assistant, user(prompt))
    assert_eq!(messages.len(), 5);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[2]["role"], "user");
    assert_eq!(messages[3]["role"], "assistant");
    assert_eq!(messages[4]["role"], "user");

    // ---- 5. Compose system prompt and max_tokens ----
    let system_prompt = question.system_prompt.clone().unwrap_or("".to_string());
    assert!(system_prompt.contains("test system"));

    let max_tokens = ai_config.max_token.unwrap_or(1024);
    assert_eq!(max_tokens, 256);

    // ---- 6. Compose payload ----
    let payload = json!({
        "model": ai_config.model,
        "max_tokens": max_tokens,
        "messages": messages,
        "system": system_prompt
    });
    assert_eq!(payload["model"], "claude-2");
    assert_eq!(payload["max_tokens"], 256);
    assert_eq!(payload["messages"][0]["role"], "user");
    assert_eq!(payload["system"], "You are a test system.");

    let api_url = env::var("ANTHROPIC_API_URL")
        .unwrap_or_else(|_| "https://api.anthropic.com/v1/messages".to_string());

    let response = reqwest::Client::new()
        .post(&api_url)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&payload)
        .send()
        .await
        .map_err(|e| AppError::ApiError {
            model_name: ai_config.llm.to_string(),
            failure_str: format!("Request error: {}", e),
        });

    println!("Response: {:?}", response);

    env::remove_var("ANTHROPIC_API_KEY");
}

#[tokio::test]
#[serial]
async fn anthropic_response_real_api_key() {

    // Run this test only if ANTHROPIC_API_KEY is set (to avoid spurious failure in CI)
    if env::var("ANTHROPIC_API_KEY").is_err() {
        eprintln!("Skipping test: ANTHROPIC_API_KEY not set");
        return;
    }

    // ---- 1. Build configuration with real data with some dummy conversation ----
    let ai_config = AiConfig {
        llm: Framework::Anthropic,
        model: "claude-opus-4-20250514".to_string(),
        max_token: Some(80),
    };
    // ---- 2. Build Question with some dummy conversation ----
    let question = Question {
        system_prompt: Some("You are friendly.".to_string()),
        messages: None,
        new_prompt: "Anthropic question!".to_string(),
    };

    let err = get_anthropic_response(question, &ai_config).await.err().expect("Should fail unsupported model");

    match err {
        AppError::ApiError { model_name: n, failure_str } => {
            assert_eq!(n, "anthropic");
            let failure_lc = failure_str.to_lowercase();
            assert!(
                failure_lc.contains("model not supported")
                || failure_lc.contains("unsupported model")
                || failure_lc.contains("not_found_error")
                || failure_lc.contains("404"),
                "failure_str: {}", failure_str
            );
        }
        other => panic!("Expected AppError::ApiError, got {:?}", other),
    }

}
