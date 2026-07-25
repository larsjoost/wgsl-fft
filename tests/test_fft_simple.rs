//! Simple test for FFT -> IFFT without multiply stage
//!
//! This test requires the "ping_pong" feature to be enabled.

#[cfg(feature = "ping_pong")]
use std::sync::Arc;
#[cfg(feature = "ping_pong")]
use wgsl_fft::ping_pong_integration::FftPipelineStage;
#[cfg(feature = "ping_pong")]
use wgsl_ping_pong_pipeline::wgpu_utils::ComputeContext;
#[cfg(feature = "ping_pong")]
use wgsl_ping_pong_pipeline::Pipeline;

#[cfg(feature = "ping_pong")]
fn real_to_complex(signal: &[f32]) -> Vec<f32> {
    signal.iter().flat_map(|&x| vec![x, 0.0]).collect()
}

#[cfg(feature = "ping_pong")]
fn extract_real(complex: &[f32]) -> Vec<f32> {
    complex.iter().step_by(2).map(|&x| x).collect()
}

#[pollster::test]
#[cfg(feature = "ping_pong")]
async fn test_fft_ifft_no_multiply() -> anyhow::Result<()> {
    let n = 8;
    let signal: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let input: Vec<f32> = real_to_complex(&signal);

    // Create a shared ComputeContext
    let context = Arc::new(ComputeContext::new_high_performance().await?);

    // Build pipeline: FFT -> IFFT (no multiply)
    let mut pipeline = Pipeline::new()
        .with_context(Arc::clone(&context))
        .pipe_custom(Box::new(FftPipelineStage::forward(n, 1)))
        .pipe_custom(Box::new(FftPipelineStage::inverse(n, 1)))
        .build()
        .await?;

    // Write input data
    pipeline.write_input(&input).await?;

    // Advance pipeline by 2 ticks
    pipeline.tick(()).await?;
    pipeline.tick(()).await?;

    // Read output
    let output_data = pipeline.read_output().await?;
    let output = output_data.map(|(_, data)| data).unwrap_or_default();

    // Extract real parts
    let result_real = extract_real(&output);

    println!("FFT -> IFFT (no multiply) result: {:?}", result_real);

    // Should be approximately [n,0,0,0,0,0,0,0] (no normalization applied)
    // FFT of impulse [1,0,...,0] = [1,1,...,1]
    // IFFT of [1,1,...,1] = [n,0,...,0]
    assert_eq!(result_real.len(), n);
    assert!(
        (result_real[0] - n as f32).abs() < 0.5,
        "Expected ~{}, got {}",
        n as f32,
        result_real[0]
    );
    for i in 1..n {
        assert!(
            result_real[i].abs() < 0.5,
            "Expected ~0.0, got {}",
            result_real[i]
        );
    }

    Ok(())
}
