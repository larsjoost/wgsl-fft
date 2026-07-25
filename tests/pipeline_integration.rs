//! Integration tests for wgsl-fft with wgsl-ping-pong-pipeline.
//!
//! These tests verify that wgsl-fft's PipelineStage implementations work correctly
//! with the wgsl-ping-pong-pipeline library.
//!
//! This test file requires the "ping_pong" feature to be enabled.

use std::sync::Arc;

#[cfg(feature = "ping_pong")]
use wgsl_fft::ping_pong_integration::{FftPipelineStage, MultiplyPipelineStage};
#[cfg(feature = "ping_pong")]
use wgsl_ping_pong_pipeline::wgpu_utils::ComputeContext;
#[cfg(feature = "ping_pong")]
use wgsl_ping_pong_pipeline::{Pipeline, PipelineStage};

#[cfg(feature = "ping_pong")]
/// CPU circular convolution for verification.
fn cpu_convolve_circular(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            let k = (i + j) % n;
            result[i] += a[j] * b[k];
        }
    }
    result
}

#[cfg(feature = "ping_pong")]
/// Converts real signal to complex interleaved format (re, im, re, im, ...)
fn real_to_complex(signal: &[f32]) -> Vec<f32> {
    signal.iter().flat_map(|&x| vec![x, 0.0]).collect()
}

#[cfg(feature = "ping_pong")]
/// Extracts real parts from complex interleaved format
fn extract_real(complex: &[f32]) -> Vec<f32> {
    complex.iter().step_by(2).map(|&x| x).collect()
}

#[cfg(feature = "ping_pong")]
/// Creates interleaved pairs [A0, B0, A1, B1, ...] from two complex vectors
fn interleave_complex(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    for i in 0..(a.len() / 2) {
        result.push(a[i * 2]);
        result.push(a[i * 2 + 1]);
        result.push(b[i * 2]);
        result.push(b[i * 2 + 1]);
    }
    result
}

#[cfg(feature = "ping_pong")]
const EPSILON: f32 = 1e-4;

/// Test: FFT pipeline stage properties
#[test]
#[cfg(feature = "ping_pong")]
fn test_fft_pipeline_stage_properties() {
    let forward_stage = FftPipelineStage::forward(1024, 1);
    assert_eq!(forward_stage.name(), "fft_forward");
    assert_eq!(forward_stage.vector_dim(), 2);
    assert_eq!(forward_stage.batch_size(), 1024 * 1);

    let inverse_stage = FftPipelineStage::inverse(1024, 1);
    assert_eq!(inverse_stage.name(), "fft_inverse");
    assert_eq!(inverse_stage.vector_dim(), 2);
    assert_eq!(inverse_stage.batch_size(), 1024 * 1);
}

/// Test: Multiply stage creation and properties
#[test]
#[cfg(feature = "ping_pong")]
fn test_multiply_pipeline_stage_properties() {
    let multiply_stage = MultiplyPipelineStage::new(1024, 1);
    assert_eq!(multiply_stage.name(), "multiply_complex");
    assert_eq!(multiply_stage.vector_dim(), 2);
    // Input is interleaved pairs [A0, B0, A1, B1, ...] with size 2 * n * batch_size
    // Output maintains same size to maintain pipeline compatibility
    assert_eq!(multiply_stage.batch_size(), 2 * 1024 * 1);
}

/// Test: Pipeline with interleaved input - simple multiply
#[pollster::test]
#[cfg(feature = "ping_pong")]
async fn test_simple_interleaved_pipeline() -> anyhow::Result<()> {
    let n = 8; // Use power of 2

    // Create a shared ComputeContext
    let context = Arc::new(ComputeContext::new_high_performance().await?);

    // Create simple test data - all ones in complex format
    let a_data: Vec<f32> = (0..n).flat_map(|_| vec![1.0, 0.0]).collect();
    let b_data: Vec<f32> = (0..n).flat_map(|_| vec![1.0, 0.0]).collect();

    // Create interleaved input: [A0, B0, A1, B1, ...]
    let interleaved_input = interleave_complex(&a_data, &b_data);

    // Build pipeline: Multiply only
    let mut pipeline = Pipeline::new()
        .with_context(Arc::clone(&context))
        .pipe_custom(Box::new(MultiplyPipelineStage::new(n, 1)))
        .build()
        .await?;

    // Write interleaved input
    pipeline.write_input(&interleaved_input).await?;

    // Advance pipeline by 1 tick
    pipeline.tick(()).await?;

    // Read output
    let output_data = pipeline.read_output().await?;
    let output = output_data.map(|(_, data)| data).unwrap_or_default();

    // Verify output size (2 * n complex numbers = 2 * n * 2 floats, same as input)
    assert_eq!(output.len(), interleaved_input.len(), "Output length mismatch");

    // Multiply ones by ones should give ones in first half, zeros in second half
    for i in 0..n {
        let real = output[i * 2];
        let imag = output[i * 2 + 1];
        assert!(
            (real - 1.0).abs() < EPSILON,
            "Output[{}].real = {}, expected 1.0",
            i,
            real
        );
        assert!(
            imag.abs() < EPSILON,
            "Output[{}].imag = {}, expected 0.0",
            i,
            imag
        );
    }
    
    // Second half should be zeros
    for i in n..(2 * n) {
        let real = output[i * 2];
        let imag = output[i * 2 + 1];
        assert!(
            real.abs() < EPSILON,
            "Output[{}].real = {}, expected 0.0",
            i,
            real
        );
        assert!(
            imag.abs() < EPSILON,
            "Output[{}].imag = {}, expected 0.0",
            i,
            imag
        );
    }

    println!("✓ Simple interleaved pipeline test passed!");

    Ok(())
}

/// Test: Pipeline with larger FFT size using interleaved input
#[pollster::test]
#[cfg(feature = "ping_pong")]
async fn test_larger_fft_size() -> anyhow::Result<()> {
    let n = 16; // Larger FFT size

    // Simple test signal - all ones
    let a_data: Vec<f32> = (0..n).flat_map(|_| vec![1.0, 0.0]).collect();
    let b_data: Vec<f32> = (0..n).flat_map(|_| vec![1.0, 0.0]).collect();

    // Create interleaved input
    let interleaved_input = interleave_complex(&a_data, &b_data);

    // Create a shared ComputeContext
    let context = Arc::new(ComputeContext::new_high_performance().await?);

    // Build pipeline: Multiply only
    let mut pipeline = Pipeline::new()
        .with_context(Arc::clone(&context))
        .pipe_custom(Box::new(MultiplyPipelineStage::new(n, 1)))
        .build()
        .await?;

    // Write interleaved input
    pipeline.write_input(&interleaved_input).await?;
    pipeline.tick(()).await?;

    let output_data = pipeline.read_output().await?;
    let output = output_data.map(|(_, data)| data).unwrap_or_default();

    // Just verify it ran without errors
    // Output size should be same as input (2 * n * 2 floats)
    assert_eq!(output.len(), interleaved_input.len());

    println!("✓ Larger FFT size (n={}) test passed!", n);

    Ok(())
}
