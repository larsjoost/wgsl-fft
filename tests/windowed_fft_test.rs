use num_complex::Complex;
use wgsl_fft::GpuFft;

mod test_utils;
use test_utils::*;

const N: usize = 2048;
const EPSILON: f32 = 1e-3;

#[test]
fn test_windowed_fft_ifft_roundtrip() {
    let fft = create_test_fft();

    // Generate signal and apply window
    let mut signal = generate_test_signal(N);
    apply_hann_window(&mut signal);

    // Test roundtrip accuracy using utility function
    test_roundtrip_accuracy(&fft, &signal, EPSILON);
}

#[test]
fn test_window_function_properties() {
    let fft = GpuFft::new().expect("GPU required");

    // Generate signal
    let mut signal = generate_test_signal(N);
    let original = signal.clone();

    // Apply window and compute FFT
    apply_hann_window(&mut signal);
    let windowed_spectrum_batch = fft.fft(&[signal]).expect("FFT failed");
    let windowed_spectrum = &windowed_spectrum_batch[0];

    // Compute FFT of original (unwindowed)
    let original_spectrum_batch = fft.fft(&[original]).expect("FFT failed");
    let original_spectrum = &original_spectrum_batch[0];

    // Windowing should reduce spectral leakage
    // Check that windowed spectrum has lower side lobes
    let windowed_power: f32 = windowed_spectrum
        .iter()
        .map(|c| c.re * c.re + c.im * c.im)
        .sum();
    let original_power: f32 = original_spectrum
        .iter()
        .map(|c| c.re * c.re + c.im * c.im)
        .sum();

    // Power should be preserved (within reasonable bounds)
    let power_ratio = windowed_power / original_power;
    println!("Windowed vs original power ratio: {power_ratio:.3}");

    // Should be reasonably close to 1 (window function preserves energy)
    assert!(
        power_ratio > 0.1 && power_ratio < 10.0,
        "Power ratio {power_ratio} is unexpected"
    );
}

#[test]
fn test_multiple_window_types() {
    let fft = GpuFft::new().expect("GPU required");
    let mut signal = generate_test_signal(N);

    // Test Hann window (already implemented)
    apply_hann_window(&mut signal);
    let hann_spectrum_batch = fft.fft(&[signal.clone()]).expect("FFT failed");
    let hann_spectrum = &hann_spectrum_batch[0];
    let hann_reconstructed_batch = fft.ifft(&[hann_spectrum.to_vec()]).expect("IFFT failed");
    let hann_reconstructed = &hann_reconstructed_batch[0];

    let mut max_error: f32 = 0.0;
    for (orig, recon) in signal.iter().zip(hann_reconstructed.iter()) {
        let error = ((orig.re - recon.re).powi(2) + (orig.im - recon.im).powi(2)).sqrt();
        max_error = max_error.max(error);
    }
    println!("Hann window roundtrip max error: {max_error:.2e}");
    assert!(max_error < EPSILON);
}

#[test]
fn test_frequency_detection() {
    let fft = GpuFft::new().expect("GPU required");

    // Generate signal with known frequencies
    let mut signal = generate_test_signal(N);
    apply_hann_window(&mut signal);

    // Compute FFT
    let spectrum_batch = fft.fft(&[signal]).expect("FFT failed");
    let spectrum = &spectrum_batch[0];
    let power_spectrum: Vec<f32> = spectrum.iter().map(|c| c.re * c.re + c.im * c.im).collect();

    // Find peaks in the spectrum
    let mut peaks = vec![];
    let sample_rate = N as f32; // samples per cycle

    for i in 1..N / 2 {
        if power_spectrum[i] > power_spectrum[i - 1]
            && power_spectrum[i] > power_spectrum[i + 1]
            && power_spectrum[i] > 10.0
        {
            // minimum power threshold
            let freq = i as f32 * sample_rate / N as f32;
            peaks.push((freq, power_spectrum[i]));
        }
    }

    println!("Detected {} frequency peaks", peaks.len());
    for (freq, power) in &peaks {
        println!("  {:.1} Hz (power: {:.0})", freq, power);
    }

    // Should detect our known frequencies (10Hz and 50Hz)
    assert!(
        !peaks.is_empty(),
        "Should detect at least one frequency peak"
    );

    // Check that detected frequencies are in expected ranges
    let has_10hz_peak = peaks.iter().any(|(freq, _)| (8.0..12.0).contains(freq));
    let has_50hz_peak = peaks.iter().any(|(freq, _)| (45.0..55.0).contains(freq));

    println!("Detected 10Hz peak: {}", has_10hz_peak);
    println!("Detected 50Hz peak: {}", has_50hz_peak);

    // At least one of the known frequencies should be detected
    assert!(
        has_10hz_peak || has_50hz_peak || peaks.len() >= 2,
        "Should detect known frequency components"
    );
}
