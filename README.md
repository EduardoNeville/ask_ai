# AI Question-Answering Crate

This Rust crate provides a unified way to call different Large Language Model (Framework) providers, including OpenAI, Anthropic, and Ollama, enabling the user to ask questions and interact with these models seamlessly. The crate abstracts away the complexities of interacting with different Framework APIs and offers a unified interface to query these models.

---

## Table of Contents

1. **Features**
2. **Configuration**
3. **Usage**
   - Basic Example
   - Customizing System Prompts
   - Providing Chat History
4. **Environment Variables**
5. **Error Handling**
6. **Contributing**
7. **License**

---

## Features

- Support for multiple Framework providers: OpenAI, Anthropic, and Ollama.
- Unified interface to interact with different APIs.
- Ease of adding system-level prompts to guide responses.
- Support for maintaining chat history (multi-turn conversations).
- Error handling for API failures, model errors, and unexpected behavior.

---

## Configuration

Before you can use the crate, you need to configure it through the `AiConfig` structure. This configuration tells the system:
1. Which Framework provider to use (`Framework::OpenAI`, `Framework::Anthropic`, or `Framework::Ollama`).
2. The specific model you want to query, e.g., `"chatgpt-4o-latest"` for OpenAI or `"claude-2"` for Anthropic.
3. (Optional) Maximum tokens for the response output.

### Example `AiConfig`

```rust
use ask_ai::config::{AiConfig, Framework};

let ai_config = AiConfig {
    llm: Framework::OpenAI,           // Specify Framework provider
    model: "chatgpt-4o-latest".to_string(), // Specify model
    max_token: Some(1000),      // Optional: Limit max tokens in response
};
```

---

## Usage

### 1. Basic Example (Ask a Single Question)

You can ask a one-off question using the following example:

```rust
use ask_ai::{config::{AiConfig, Framework, Question}, model::ask_question};

#[tokio::main]
async fn main() {
    let ai_config = AiConfig {
        llm: Framework::OpenAI,
        model: "chatgpt-4o-latest".to_string(),
        max_token: Some(1000),
    };

    let question = Question {
        system_prompt: None,      // Optional system prompt
        messages: None,           // No previous history
        new_prompt: "What is Rust?".to_string(),
    };

    match ask_question(&ai_config, question).await {
        Ok(answer) => println!("Answer: {}", answer),
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

### 2. Customizing System Prompts

A system-level prompt modifies the assistant's behavior. For example, you might instruct the assistant to answer concisely or role-play as an expert.

```rust
let question = Question {
    system_prompt: Some("You are an expert Rust programmer. Answer concisely.".to_string()), // Custom prompt
    messages: None,
    new_prompt: "How do closures work in Rust?".to_string(),
};
```

### 3. Multi-Turn Conversation (With Chat History)

To maintain a conversation, you can include previous messages and their respective responses.

```rust
use ask_ai::config::{AiPrompt};

let previous_messages = vec![
    AiPrompt {
        content: "What is Rust?".to_string(),
        output: "Rust is a systems programming language focused on safety, speed, and concurrency.".to_string(),
    },
    AiPrompt {
        content: "Why is Rust popular?".to_string(),
        output: "Rust is popular because of features like memory safety, modern tooling, and high performance.".to_string(),
    },
];

let question = Question {
    system_prompt: None,
    messages: Some(previous_messages), // Include chat history
    new_prompt: "What are Rust's main drawbacks?".to_string(),
};
```

---

## Environment Variables

This crate requires API keys to interface with the Framework providers. Store these keys as environment variables to keep them secure. Below is a list of required variables:

| Provider   | Environment Variable      |
|------------|---------------------------|
| OpenAI     | `OPENAI_API_KEY`          |
| Anthropic  | `ANTHROPIC_API_KEY`       |
| Ollama     | No key required currently |

For security, avoid hardcoding API keys into your application code. Use a `.env` file or a secret storage mechanism.

---

## Error Handling

All interactions with Framework return `Result<String>`. Errors are encapsulated using the `AppError` enum, which defines three main error types:

1. **ModelError**: Occurs when querying a specific model fails.
2. **ApiError**: Indicates an issue with the API key or API call.
3. **UnexpectedError**: For any other unforeseen issues.

### Example: Handling Errors Gracefully

```rust
match ask_question(&ai_config, question).await {
    Ok(answer) => println!("Answer: {}", answer),
    Err(e) => match e {
        AppError::ModelError { model_name, failure_str } => {
            eprintln!("Model Error: {} - {}", model_name, failure_str);
        },
        AppError::ApiError { model_name, failure_str } => {
            eprintln!("API Error: {:?} - {}", model_name, failure_str);
        },
        AppError::UnexpectedError(msg) => {
            eprintln!("Unexpected Error: {}", msg);
        },
    },
}
```

## Contributing

Contributions, bug reports, and feature requests are welcome! Feel free to open an issue or submit a pull request in GitHub.

### How to Contribute:

1. Fork the repository.
2. Clone to your local system: `git clone <your-fork-url>`
3. Create a feature branch: `git checkout -b feature-name`
4. Push changes and open a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

