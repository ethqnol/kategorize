//! Error types for the k-modes crate

use thiserror::Error;

/// Result type alias for convenience
pub type Result<T> = std::result::Result<T, Error>;

/// Error types that can occur during clustering operations
#[derive(Error, Debug)]
pub enum Error {
    /// Invalid input parameters
    #[error("Invalid parameter: {message}")]
    InvalidParameter { 
        /// Error message
        message: String 
    },

    /// Empty or invalid data
    #[error("Invalid data: {message}")]
    InvalidData { 
        /// Error message
        message: String 
    },

    /// Convergence failure
    #[error("Convergence failure: {message}")]
    ConvergenceFailure { 
        /// Error message
        message: String 
    },

    /// Initialization failure
    #[error("Initialization failure: {message}")]
    InitializationFailure { 
        /// Error message
        message: String 
    },

    /// Mathematical computation error
    #[error("Computation error: {message}")]
    ComputationError { 
        /// Error message
        message: String 
    },
}

impl Error {
    /// Create a new InvalidParameter error
    pub fn invalid_parameter(message: impl Into<String>) -> Self {
        Self::InvalidParameter {
            message: message.into(),
        }
    }

    /// Create a new InvalidData error
    pub fn invalid_data(message: impl Into<String>) -> Self {
        Self::InvalidData {
            message: message.into(),
        }
    }

    /// Create a new ConvergenceFailure error
    pub fn convergence_failure(message: impl Into<String>) -> Self {
        Self::ConvergenceFailure {
            message: message.into(),
        }
    }

    /// Create a new InitializationFailure error
    pub fn initialization_failure(message: impl Into<String>) -> Self {
        Self::InitializationFailure {
            message: message.into(),
        }
    }

    /// Create a new ComputationError
    pub fn computation_error(message: impl Into<String>) -> Self {
        Self::ComputationError {
            message: message.into(),
        }
    }
}