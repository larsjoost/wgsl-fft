//! # The FFT Rivalry Registry
//!
//! To add a new AI implementation to the rivalry:
//! 1. Create a new file in this directory (e.g., `my_ai_fft.rs`).
//! 2. Implement the `FftExecutor` trait.
//! 3. Register your rival in `examples/rivalry_leaderboard.rs`.

#[allow(clippy::all)]
pub mod baseline;
#[allow(clippy::all)]
pub mod claude;
#[allow(clippy::all)]
pub mod codex;
#[allow(clippy::all)]
pub mod devstral_2;
#[allow(clippy::all)]
pub mod gemini;
#[allow(clippy::all)]
pub mod mistral_vibe;
#[allow(clippy::all)]
pub mod radix4;
#[allow(clippy::all)]
pub mod radix4_proper;
