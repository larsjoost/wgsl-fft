//! GPU-accelerated FFT using [wgpu](https://github.com/gfx-rs/wgpu) compute shaders.
//!
//! This crate provides fast Fourier transform (FFT) computation on GPU hardware
//! using WebGPU compute shaders via the [wgpu](https://github.com/gfx-rs/wgpu) library.
//!
//! ## Features
//!
//! - **GPU-accelerated**: Uses wgpu compute shaders for high-performance FFT computation.
//! - **Stockham Radix-4/2**: Fast algorithm for power-of-two FFT sizes.
//! - **Arbitrary sizes**: Bluestein's algorithm for non-power-of-two sizes (CPU fallback).
//! - **Batch processing**: Efficiently process multiple FFTs in a single call.
//! - **Fallback support**: Automatically falls back to CPU software rendering if no GPU is available.
//!
//! ## Quick Start
//!
//! ```no_run
//! use wgsl_fft::GpuFft;
//! use num_complex::Complex;
//!
//! let fft = GpuFft::new().expect("GPU or CPU fallback required");
//! let input = vec![vec![Complex::new(1.0, 0.0); 1024]];
//! let spectrum = fft.fft(&input).expect("FFT failed");
//! ```
//!
//! ## Module Structure
//!
//! - [`fft`] - Main FFT implementation with [`GpuFft`] and [`FftExecutor`] trait
//! - [`pipelines`] - Pre-compiled FFT pipelines for embedding in larger GPU pipelines
//! - [`shaders`] - WGSL compute shader source code
//! - [`benchmark`] - Benchmarking utilities
//! - [`error`] - Error types for FFT operations

mod error;
pub use error::{FftError, Result};

pub mod benchmark;
#[cfg(feature = "cuda")]
mod cufft_wrapper;
#[cfg(feature = "hipfft")]
pub mod hipfft_wrapper;
pub mod rivals;
#[cfg(feature = "rocm")]
mod rocfft_wrapper;
pub mod shaders;

mod fft;
mod pipelines;

pub use fft::{FftExecutor, FftUniforms, GpuFft, GpuFftTrait, SizeCache};
pub use pipelines::{FftDirection, FftPipelines};

#[cfg(feature = "cuda")]
pub use cufft_wrapper::CuFft;
#[cfg(feature = "hipfft")]
pub use hipfft_wrapper::HipFft;
#[cfg(feature = "rocm")]
pub use rocfft_wrapper::RocFft;
