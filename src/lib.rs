//! GPU-accelerated FFT using [wgpu](https://github.com/gfx-rs/wgpu) compute shaders.
//!
//! Implements the **Stockham autosort** Radix-4/2 FFT — a two-buffer ping-pong formulation
//! where each stage reads from one buffer and writes to the other. This eliminates the separate
//! bit-reversal pass and removes all inter-stage memory hazards. The baseline dispatches
//! ⌊log₄N⌋ Radix-4 passes (plus one Radix-2 pass when log₂N is odd), halving the pass count
//! vs the old Radix-2 baseline for a significant throughput improvement.

// Module declarations
pub mod benchmark;
#[cfg(feature = "cuda")]
mod cufft_wrapper;
#[cfg(feature = "hipfft")]
pub mod hipfft_wrapper;
pub mod pipeline;
pub mod rivals;
#[cfg(feature = "rocm")]
mod rocfft_wrapper;
pub mod shaders;

// Internal modules
mod buffer;
mod fft;
mod pipelines;

// Re-exports for public API

// From fft.rs
pub use fft::{FftExecutor, FftUniforms, GpuFft, GpuFftTrait, SizeCache};

// From pipelines.rs
pub use pipelines::{FftDirection, FftPipelines};

// From buffer.rs
pub use buffer::{PingPongBuffers, PingPongState};

// From pipeline module
pub use pipeline::{
    ComputeStage, FftStage, MultiplyStage, NoiseStage, NormalizeStage, Pipeline,
    PipelineParameters, StageContext,
};

#[cfg(feature = "cuda")]
pub use cufft_wrapper::CuFft;
#[cfg(feature = "hipfft")]
pub use hipfft_wrapper::HipFft;
#[cfg(feature = "rocm")]
pub use rocfft_wrapper::RocFft;
