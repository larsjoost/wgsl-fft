//! GPU-accelerated FFT using [wgpu](https://github.com/gfx-rs/wgpu) compute shaders.

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
