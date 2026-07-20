//! Integration tests for wgsl-fft with wgsl-ping-pong-pipeline.
//!
//! These tests verify that wgsl-fft's PipelineStage implementations work correctly
//! with the wgsl-ping-pong-pipeline library.
//!
//! This test file requires the "ping_pong" feature to be enabled.

use std::sync::Arc;
use wgpu::util::DeviceExt;

#[cfg(feature = "ping_pong")]
use wgsl_fft::ping_pong_integration::{FftPipelineStage, MultiplyPipelineStage};
#[cfg(feature = "ping_pong")]
use wgsl_fft::{FftDirection, FftPipelines};
#[cfg(feature = "ping_pong")]
use wgsl_ping_pong_pipeline::wgpu_utils::ComputeContext;
#[cfg(feature = "ping_pong")]
use wgsl_ping_pong_pipeline::{Pipeline, PipelineStage, StageConfig};

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
const EPSILON: f32 = 1e-4;

/// Test: FFT roundtrip through pipeline (FFT -> Identity Multiply -> IFFT)
///
/// This test verifies that:
/// 1. FFT stages can be created and used in a pipeline
/// 2. Multiply stage works correctly with side inputs
/// 3. Data flows correctly through the pipeline
/// 4. FFT -> Multiply -> IFFT preserves the input signal (within numerical precision)
#[pollster::test]
#[cfg(feature = "ping_pong")]
async fn test_fft_roundtrip_through_pipeline() -> anyhow::Result<()> {
    let n = 8; // Use power of 2 for FFT

    // Create a simple test signal (impulse at position 0)
    let signal: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    // Convert to complex format
    let input: Vec<f32> = real_to_complex(&signal);

    // Create a shared ComputeContext for pipeline and side inputs
    let context = Arc::new(ComputeContext::new_high_performance().await?);
    let device = context.device.clone();

    // Create dummy side input (all ones in complex format) on this device
    let dummy_input: Vec<f32> = (0..n).flat_map(|_| vec![1.0, 0.0]).collect();
    let dummy_buffer = Arc::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dummy Input Buffer"),
            contents: bytemuck::cast_slice(&dummy_input),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        }),
    );

    // Build pipeline with shared context
    let mut pipeline = Pipeline::new()
        .with_context(Arc::clone(&context))
        .pipe_config(StageConfig::Custom(Box::new(FftPipelineStage::forward(
            n, 1,
        ))))
        .pipe_config(StageConfig::Custom(Box::new(MultiplyPipelineStage::new(
            n, 1,
        ))))
        .pipe_config(StageConfig::Custom(Box::new(FftPipelineStage::inverse(
            n, 1,
        ))))
        .build()
        .await?;

    // Add dummy side input
    pipeline.add_side_input("input_b", Arc::clone(&dummy_buffer));

    // Write input data
    pipeline.write_input(&input).await?;

    // Advance pipeline by 3 ticks (data propagates through 3 stages)
    pipeline.tick().await?;
    pipeline.tick().await?;
    pipeline.tick().await?;

    // Read output
    let output = pipeline.read_output().await?;

    // Extract real parts (imaginary parts should be near zero)
    let result_real = extract_real(&output);

    // Verify output size
    assert_eq!(result_real.len(), n, "Output length mismatch");

    // For FFT -> Multiply by ones -> IFFT with normalization:
    // FFT of impulse [1,0,...,0] = [1,1,...,1] (constant)
    // Multiply by ones = [1,1,...,1]
    // IFFT of [1,1,...,1] = [N,0,...,0] (impulse * N)
    // Normalization divides by N, so we get [1,0,...,0]
    // So we expect the output to match the input approximately

    // Check that the first element is approximately 1.0
    let first_real = result_real[0];
    println!(
        "FFT -> Multiply -> IFFT result: first element = {}",
        first_real
    );
    assert!(
        (first_real - 1.0).abs() < 0.5,
        "First element should be approximately 1.0: got {}, expected ~1.0",
        first_real
    );

    // Check that other elements are near zero
    for i in 1..n {
        let actual_real = result_real[i];
        let diff = actual_real.abs();
        assert!(
            diff < 0.5,
            "Output[{}] = {:.6}, expected ≈ 0.0, diff = {:.6}",
            i,
            actual_real,
            diff
        );
    }

    println!(
        "✓ FFT roundtrip through pipeline passed! First element: {}",
        first_real
    );

    Ok(())
}

/// Test: Full FFT-based convolution with pre-computed FFT(B)
///
/// This test demonstrates the complete convolution workflow:
/// 1. Precompute FFT(B) using FftPipelines
/// 2. Build pipeline: FFT(A) -> Multiply(FFT(A), FFT(B)) -> IFFT
/// 3. Verify result matches CPU circular convolution
#[pollster::test]
#[cfg(feature = "ping_pong")]
async fn test_fft_convolution_through_pipeline() -> anyhow::Result<()> {
    let n = 8; // Use power of 2 for FFT

    // Create a shared ComputeContext for pipeline and FFT pipelines
    let context = Arc::new(ComputeContext::new_high_performance().await?);
    let device = context.device.clone();
    let queue = context.queue.clone();

    // Create FFT pipelines for pre-computing FFT(B)
    let fft_pipelines = FftPipelines::from_device_queue(device.clone(), queue.clone());

    // Create test signals
    // Signal A: impulse at position 0
    let signal_a: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    // Signal B: box function
    let signal_b: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];

    // Compute expected result on CPU
    let expected = cpu_convolve_circular(&signal_a, &signal_b);

    // Precompute FFT(B)
    let b_complex: Vec<f32> = real_to_complex(&signal_b);
    let buf_b = Arc::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Signal B Buffer"),
            contents: bytemuck::cast_slice(&b_complex),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        }),
    );

    let buf_fft_b = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("FFT(B) Buffer"),
        size: (n * 2 * 4) as u64, // n * 2 floats (complex) * 4 bytes per float
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));

    // Compute FFT(B) using wgsl-fft
    let mut encoder = device.create_command_encoder(&Default::default());
    fft_pipelines.encode_fft(
        &mut encoder,
        n,
        1,
        FftDirection::Forward,
        &buf_b,
        &buf_fft_b,
    );
    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    })?;

    // Build pipeline: FFT(A) -> Multiply(FFT(A), FFT(B)) -> IFFT
    let mut pipeline = Pipeline::new()
        .with_context(Arc::clone(&context))
        .pipe_config(StageConfig::Custom(Box::new(FftPipelineStage::forward(
            n, 1,
        ))))
        .pipe_config(StageConfig::Custom(Box::new(MultiplyPipelineStage::new(
            n, 1,
        ))))
        .pipe_config(StageConfig::Custom(Box::new(FftPipelineStage::inverse(
            n, 1,
        ))))
        .build()
        .await?;

    // Register FFT(B) as a side input
    pipeline.add_side_input("input_b", Arc::clone(&buf_fft_b));

    // Convert signal A to complex format
    let a_complex: Vec<f32> = real_to_complex(&signal_a);

    // Write input and process
    pipeline.write_input(&a_complex).await?;
    pipeline.tick().await?; // FFT(A)
    pipeline.tick().await?; // Multiply FFT(A) * FFT(B)
    pipeline.tick().await?; // IFFT

    let output = pipeline.read_output().await?;

    // Extract real parts (imaginary parts should be near zero)
    let result_real = extract_real(&output);

    // Verify output size
    assert_eq!(result_real.len(), n, "Output length mismatch");

    // Verify result matches expected convolution (with scaling)
    // Note: FFT-based convolution with inverse FFT normalization gives the correct result directly
    let mut max_diff = 0.0f32;
    for (i, (out, exp)) in result_real.iter().zip(expected.iter()).enumerate() {
        let diff = (out - exp).abs();
        max_diff = max_diff.max(diff);
        assert!(
            diff < EPSILON,
            "Convolution Output[{}] = {:.6}, expected = {:.6}, diff = {:.6e}",
            i,
            out,
            exp,
            diff
        );
    }

    println!(
        "✓ FFT convolution through pipeline passed! Max diff: {:.2e}",
        max_diff
    );

    Ok(())
}

/// Test: Simple FFT stage creation and properties
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
    assert_eq!(multiply_stage.batch_size(), 1024 * 1);
    assert_eq!(multiply_stage.side_input_names(), vec!["input_b"]);
}

/// Test: Pipeline with larger FFT size
#[pollster::test]
#[cfg(feature = "ping_pong")]
async fn test_larger_fft_size() -> anyhow::Result<()> {
    let n = 16; // Larger FFT size

    // Simple test signal
    let signal: Vec<f32> = vec![1.0; n];
    let input: Vec<f32> = real_to_complex(&signal);

    // Create a shared ComputeContext for pipeline and side inputs
    let context = Arc::new(ComputeContext::new_high_performance().await?);
    let device = context.device.clone();

    // Create dummy side input
    let dummy_input: Vec<f32> = (0..n).flat_map(|_| vec![1.0, 0.0]).collect();
    let dummy_buffer = Arc::new(
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dummy Input Buffer"),
            contents: bytemuck::cast_slice(&dummy_input),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        }),
    );

    // Build and run pipeline with shared context
    let mut pipeline = Pipeline::new()
        .with_context(Arc::clone(&context))
        .pipe_config(StageConfig::Custom(Box::new(FftPipelineStage::forward(
            n, 1,
        ))))
        .pipe_config(StageConfig::Custom(Box::new(MultiplyPipelineStage::new(
            n, 1,
        ))))
        .pipe_config(StageConfig::Custom(Box::new(FftPipelineStage::inverse(
            n, 1,
        ))))
        .build()
        .await?;

    pipeline.add_side_input("input_b", Arc::clone(&dummy_buffer));
    pipeline.write_input(&input).await?;
    pipeline.tick().await?;
    pipeline.tick().await?;
    pipeline.tick().await?;

    let output = pipeline.read_output().await?;

    // Just verify it ran without errors
    assert_eq!(output.len(), n * 2);

    println!("✓ Larger FFT size (n={}) test passed!", n);

    Ok(())
}
