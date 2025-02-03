use std::error::Error;
use std::fmt;

use crate::config::LLM;

#[derive(Debug)]
pub enum AppError {
    ModelError {
        model_name: String,     // Name of the faulty model
        failure_str: String,    // Detailed explanation of the error
    },
    ApiError {
        model_name: LLM,        // LLM provider (OpenAI, Anthropic, Ollama)
        failure_str: String,    // Detailed explanation of the error
    },
    UnexpectedError(String),    // Catch-all for unexpected issues
}

/// Implement `std::fmt::Display` for `AppError`.
impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::UnexpectedError(msg) => write!(
                f,
                "Unexpected or unknown error: {}",
                msg
            ),
            AppError::ModelError {
                model_name,
                failure_str,
            } => write!(
                f,
                "Error requesting answer from {}. Error: {}",
                model_name, failure_str
            ),
            AppError::ApiError {
                model_name,
                failure_str,
            } => write!(
                f,
                "Error loading API {:?}. Error: {}",
                model_name, failure_str
            ),
        }
    }
}

/// Implement `std::error::Error` for `AppError`.
impl Error for AppError {}

/// Custom Result type that uses `AppError`.
pub type Result<T, E = AppError> = std::result::Result<T, E>;
