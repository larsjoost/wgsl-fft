//! Error types for the wgsl-fft crate

use std::io;

use thiserror::Error;

/// Main error type for FFT operations
#[derive(Debug, Error)]
pub enum FftError {
    /// Input validation error (wrong size, empty, etc.)
    #[error("Input validation error: {0}")]
    ValidationError(String),

    /// WGPU-related error
    #[error("WGPU error: {0}")]
    WgpuError(String),

    /// I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] io::Error),

    /// No GPU device available
    #[error("No GPU device available")]
    NoDevice,

    /// Invalid FFT size
    #[error("Invalid FFT size: {0}")]
    InvalidSize(String),

    /// Compute shader error
    #[error("Compute shader error: {0}")]
    ShaderError(String),

    /// Batch processing error
    #[error("Batch processing error: {0}")]
    BatchError(String),

    /// GPU backend error (CUDA, HIP, ROCm)
    #[error("GPU backend error: {0}")]
    BackendError(String),

    /// Generic error message
    #[error("{0}")]
    Other(String),
}

/// Result type alias for FFT operations
pub type Result<T> = std::result::Result<T, FftError>;

/// Convert wgpu::Error to FftError
impl From<wgpu::Error> for FftError {
    fn from(e: wgpu::Error) -> Self {
        Self::WgpuError(e.to_string())
    }
}

/// Convert wgpu::PollError to FftError
impl From<wgpu::PollError> for FftError {
    fn from(e: wgpu::PollError) -> Self {
        Self::WgpuError(e.to_string())
    }
}

/// Convert wgpu::RequestAdapterError to FftError
impl From<wgpu::RequestAdapterError> for FftError {
    fn from(e: wgpu::RequestAdapterError) -> Self {
        Self::WgpuError(e.to_string())
    }
}

/// Convert wgpu::RequestDeviceError to FftError
impl From<wgpu::RequestDeviceError> for FftError {
    fn from(e: wgpu::RequestDeviceError) -> Self {
        Self::WgpuError(e.to_string())
    }
}
