//! Tests for arbitrary FFT sizes (not just powers of 2)
//!
//! These tests verify that Bluestein's algorithm correctly computes FFTs
//! for sizes that are not powers of two.

use num_complex::Complex;
use rustfft::FftPlanner;
use wgsl_fft::GpuFft;

const EPSILON: f32 = 1e-3;

/// Generate a test signal with multiple frequency components
fn make_test_signal(n: usize) -> Vec<Complex<f32>> {
    (0..n)
        .map(|i| {
            let t = i as f32 / n as f32;
            let signal = 0.7 * (2.0 * std::f32::consts::PI * 10.0 * t).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * 50.0 * t).sin();
            Complex {
                re: signal,
                im: 0.0,
            }
        })
        .collect()
}

fn test_fft_size(gpu: &GpuFft, n: usize) {
    println!("Testing FFT for size n={}", n);
    let input = make_test_signal(n);
    let gpu_out_batch = gpu.fft(&[input.clone()]).expect("GPU FFT failed");
    let gpu_out = &gpu_out_batch[0];
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let mut cpu_buf = input.clone();
    fft.process(&mut cpu_buf);
    assert_eq!(gpu_out.len(), n);
    let mut max_diff: f32 = 0.0;
    for (i, (g, c)) in gpu_out.iter().zip(cpu_buf.iter()).enumerate() {
        let diff = ((g.re - c.re).powi(2) + (g.im - c.im).powi(2)).sqrt();
        max_diff = max_diff.max(diff);
        assert!(
            diff < EPSILON,
            "element {i}: GPU={g:?}  CPU={c:?}  diff={diff:.2e}"
        );
    }
    println!("  FFT max element-wise error: {max_diff:.2e} - PASSED");
}

fn test_ifft_size(gpu: &GpuFft, n: usize) {
    println!("Testing IFFT for size n={}", n);
    let input = make_test_signal(n);
    let spectrum_batch = gpu.fft(&[input.clone()]).expect("FFT failed");
    let spectrum = &spectrum_batch[0];
    let reconstructed_batch = gpu.ifft(&[spectrum.to_vec()]).expect("IFFT failed");
    let reconstructed = &reconstructed_batch[0];
    assert_eq!(reconstructed.len(), n);
    let mut max_diff: f32 = 0.0;
    for (i, (orig, recon)) in input.iter().zip(reconstructed.iter()).enumerate() {
        let diff = ((orig.re - recon.re).powi(2) + (orig.im - recon.im).powi(2)).sqrt();
        max_diff = max_diff.max(diff);
        assert!(
            diff < EPSILON,
            "element {i}: original={orig:?} reconstructed={recon:?} diff={diff:.2e}"
        );
    }
    println!("  IFFT roundtrip max error: {max_diff:.2e} - PASSED");
}

#[test]
fn test_arbitrary_sizes_fft() {
    let gpu = GpuFft::new().expect("GPU required");
    let sizes = vec![
        3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 25, 30, 40, 50, 60, 100, 150, 200, 255,
        512,
    ];
    for &n in &sizes {
        test_fft_size(&gpu, n);
    }
}

#[test]
fn test_arbitrary_sizes_ifft() {
    let gpu = GpuFft::new().expect("GPU required");
    let sizes = vec![
        3, 5, 6, 7, 10, 11, 12, 15, 17, 20, 25, 30, 50, 100, 150, 200,
    ];
    for &n in &sizes {
        test_ifft_size(&gpu, n);
    }
}

#[test]
fn test_power_of_two_sizes_still_work() {
    let gpu = GpuFft::new().expect("GPU required");
    let sizes = vec![16, 32, 64, 128, 256, 512, 1024, 2048];
    for &n in &sizes {
        test_fft_size(&gpu, n);
        test_ifft_size(&gpu, n);
    }
}

#[test]
fn test_roundtrip_arbitrary_sizes() {
    let gpu = GpuFft::new().expect("GPU required");
    let sizes = vec![3, 5, 10, 15, 20, 50, 100, 150, 200, 255];
    for &n in &sizes {
        println!("Testing roundtrip for size n={}", n);
        let input = make_test_signal(n);
        let spectrum_batch = gpu.fft(&[input.clone()]).expect("FFT failed");
        let spectrum = &spectrum_batch[0];
        let reconstructed_batch = gpu.ifft(&[spectrum.to_vec()]).expect("IFFT failed");
        let reconstructed = &reconstructed_batch[0];
        assert_eq!(reconstructed.len(), n);
        let mut max_diff: f32 = 0.0;
        for (i, (orig, recon)) in input.iter().zip(reconstructed.iter()).enumerate() {
            let diff = ((orig.re - recon.re).powi(2) + (orig.im - recon.im).powi(2)).sqrt();
            max_diff = max_diff.max(diff);
            assert!(
                diff < EPSILON,
                "Roundtrip error too large at element {i}: diff={diff:.2e}"
            );
        }
        println!("  Roundtrip max error: {max_diff:.2e} - PASSED");
    }
}

#[test]
fn test_zero_size_rejected() {
    let gpu = GpuFft::new().expect("GPU required");
    let empty_input = vec![vec![]];
    let result = gpu.fft(&empty_input);
    assert!(result.is_err(), "Empty input should be rejected");
}

#[test]
fn test_batch_arbitrary_sizes() {
    let gpu = GpuFft::new().expect("GPU required");
    let n = 15;
    let batch_size = 4;
    let inputs: Vec<Vec<Complex<f32>>> = (0..batch_size).map(|_| make_test_signal(n)).collect();
    let results = gpu.fft(&inputs).expect("Batch FFT failed");
    assert_eq!(results.len(), batch_size);
    for result in &results {
        assert_eq!(result.len(), n);
    }
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    for (i, (input, result)) in inputs.iter().zip(results.iter()).enumerate() {
        let mut cpu_buf = input.clone();
        fft.process(&mut cpu_buf);
        let mut max_diff: f32 = 0.0;
        for (j, (g, c)) in result.iter().zip(cpu_buf.iter()).enumerate() {
            let diff = ((g.re - c.re).powi(2) + (g.im - c.im).powi(2)).sqrt();
            max_diff = max_diff.max(diff);
            assert!(diff < EPSILON, "Batch {i}, element {j}: diff={diff:.2e}");
        }
        println!("  Batch {i} max error: {max_diff:.2e}");
    }
}
