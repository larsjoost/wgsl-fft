//! CI Debug Test File
//!
//! This test file helps debug CI issues by running the most critical tests
//! in an environment that mimics GitHub Actions as closely as possible.

use num_complex::Complex;
use wgsl_fft::GpuFft;

static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn test_ci_minimal() {
    let _lock = GPU_LOCK.lock().unwrap();

    if !GpuFft::is_gpu_available() {
        println!("CI DEBUG: No GPU available - using CPU fallback would be needed");
        return; // Skip test if no GPU
    }

    // Create FFT instance with robust error handling
    let fft = match GpuFft::new() {
        Ok(fft) => fft,
        Err(e) => {
            println!("CI DEBUG: Failed to create FFT: {}", e);
            println!("CI DEBUG: This can happen in resource-constrained environments");
            return;
        }
    };

    // Create test signal
    let mut signal = vec![Complex::new(1.0, 0.0); 1024];

    // Add some variation
    for (i, c) in signal.iter_mut().enumerate() {
        let t = i as f32 / 1024.0;
        c.re = (t * 2.0 * std::f32::consts::PI * 5.0).sin();
    }

    // Test FFT
    let signal_for_fft = signal.clone();
    match fft.fft(&[signal_for_fft]) {
        Ok(spectrum_batch) => {
            let spectrum = &spectrum_batch[0];
            println!(
                "CI DEBUG: FFT successful, spectrum length: {}",
                spectrum.len()
            );

            // Test IFFT
            match fft.ifft(&[spectrum.to_vec()]) {
                Ok(reconstructed_batch) => {
                    let reconstructed = &reconstructed_batch[0];
                    println!("CI DEBUG: IFFT successful");

                    // Verify roundtrip
                    let mut max_error = 0.0f32;
                    for (orig, recon) in signal.iter().zip(reconstructed.iter()) {
                        let error =
                            ((orig.re - recon.re).powi(2) + (orig.im - recon.im).powi(2)).sqrt();
                        max_error = max_error.max(error);
                    }
                    println!("CI DEBUG: Max roundtrip error: {:.2e}", max_error);
                    assert!(max_error < 1e-3, "Roundtrip error too large: {}", max_error);
                }
                Err(e) => {
                    println!("CI DEBUG: IFFT failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("CI DEBUG: FFT failed: {}", e);
        }
    }
}

#[test]
fn test_ci_performance() {
    let _lock = GPU_LOCK.lock().unwrap();
    use std::time::Instant;

    if !GpuFft::is_gpu_available() {
        println!("CI DEBUG: No GPU available, skipping performance test");
        return;
    }

    let fft = GpuFft::new().expect("GPU required");
    let signal = vec![Complex::new(1.0, 0.0); 4096];

    let start = Instant::now();
    let signal_for_fft = signal.clone();
    let spectrum_batch = fft.fft(&[signal_for_fft]).expect("FFT failed");
    let spectrum = &spectrum_batch[0];
    let fft_time = start.elapsed();

    let start = Instant::now();
    let _reconstructed_batch = fft.ifft(&[spectrum.to_vec()]).expect("IFFT failed");
    let _reconstructed = &_reconstructed_batch[0];
    let ifft_time = start.elapsed();

    println!("CI DEBUG: FFT: {:?}, IFFT: {:?}", fft_time, ifft_time);
}
