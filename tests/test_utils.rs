//! Test utilities and common test patterns for wgls-rs-fft
//!
//! This module provides reusable test utilities to reduce code duplication
//! and improve test maintainability following DRY and SOLID principles.

use num_complex::Complex;
use wgsl_fft::GpuFft;

/// Create a new GPU FFT instance for testing
///
/// # Panics
///
/// Panics if no GPU is available
pub fn create_test_fft() -> GpuFft {
    GpuFft::new().expect("GPU required for tests")
}

/// Generate a basic test signal with sine waves
///
/// Creates a signal with multiple frequency components for testing
pub fn generate_test_signal(n: usize) -> Vec<Complex<f32>> {
    (0..n)
        .map(|i| {
            let t = i as f32 / n as f32;
            // Multiple frequency components for better test coverage
            let signal = 0.7 * (2.0 * std::f32::consts::PI * 10.0 * t).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * 50.0 * t).sin();
            Complex {
                re: signal,
                im: 0.0,
            }
        })
        .collect()
}

/// Apply Hann window function to reduce spectral leakage
///
/// This is a common preprocessing step for FFT analysis
pub fn apply_hann_window(signal: &mut [Complex<f32>]) {
    let n = signal.len();
    for (i, sample) in signal.iter_mut().enumerate() {
        let window = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos());
        *sample = Complex {
            re: sample.re * window,
            im: sample.im * window,
        };
    }
}

/// Test FFT->IFFT roundtrip accuracy
///
/// Verifies that FFT followed by IFFT reconstructs the original signal
/// within acceptable numerical precision
pub fn test_roundtrip_accuracy(fft: &GpuFft, signal: &[Complex<f32>], epsilon: f32) {
    // FFT -> IFFT roundtrip
    let spectrum_batch = fft.fft(&[signal.to_vec()]).expect("FFT failed");
    let spectrum = &spectrum_batch[0];
    let reconstructed_batch = fft.ifft(&[spectrum.to_vec()]).expect("IFFT failed");
    let reconstructed = &reconstructed_batch[0];

    // Verify roundtrip accuracy
    assert_eq!(reconstructed.len(), signal.len());

    let mut max_diff: f32 = 0.0;
    for (orig, recon) in signal.iter().zip(reconstructed.iter()) {
        let diff = ((orig.re - recon.re).powi(2) + (orig.im - recon.im).powi(2)).sqrt();
        max_diff = max_diff.max(diff);
        assert!(
            diff < epsilon,
            "Roundtrip error too large: original={orig:?} reconstructed={recon:?} diff={diff:.2e}"
        );
    }
    println!("Roundtrip max error: {max_diff:.2e}");
}

/// Calculate maximum element-wise difference between two complex vectors
///
/// Useful for comparing FFT results
pub fn max_complex_diff(a: &[Complex<f32>], b: &[Complex<f32>]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");

    a.iter()
        .zip(b.iter())
        .map(|(c1, c2)| ((c1.re - c2.re).powi(2) + (c1.im - c2.im).powi(2)).sqrt())
        .fold(0.0, |max, val| max.max(val))
}

/// Calculate power spectrum from complex FFT output
///
/// Converts complex FFT results to power spectrum (magnitude squared)
pub fn calculate_power_spectrum(spectrum: &[Complex<f32>]) -> Vec<f32> {
    spectrum.iter().map(|c| c.re * c.re + c.im * c.im).collect()
}
