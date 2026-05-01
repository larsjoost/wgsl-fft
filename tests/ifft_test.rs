use num_complex::Complex;
use rustfft::FftPlanner;
use wgsl_fft::GpuFft;

mod test_utils;
use test_utils::*;

const N: usize = 1024;
const EPSILON: f32 = 1e-3;

/// Generate a different test signal for IFFT tests
/// This creates a signal with different frequency characteristics
fn make_test_signal(n: usize) -> Vec<Complex<f32>> {
    (0..n)
        .map(|i| Complex {
            re: (i as f32 * 0.1).sin(),
            im: (i as f32 * 0.07).cos(),
        })
        .collect()
}

#[test]
fn test_ifft_roundtrip() {
    let fft = create_test_fft();
    let original = make_test_signal(N);

    // Test roundtrip accuracy using utility function
    test_roundtrip_accuracy(&fft, &original, EPSILON);
}

#[test]
fn test_ifft_matches_rustfft() {
    let fft = GpuFft::new().expect("GPU required");

    // Create a time-domain signal and compute its FFT first
    let time_domain = make_test_signal(N);
    let frequency_domain_batch = fft.fft(&[time_domain.clone()]).expect("FFT failed");
    let frequency_domain = frequency_domain_batch[0].clone();

    // Compute IFFT with our implementation
    let our_ifft_batch = fft.ifft(&[frequency_domain]).expect("IFFT failed");
    let our_ifft = &our_ifft_batch[0];

    // Compute forward FFT with rustfft for comparison (since our IFFT should reconstruct original)
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N);
    let mut rustfft_freq = time_domain.clone();
    fft.process(&mut rustfft_freq);

    let ifft = planner.plan_fft_inverse(N);
    let mut rustfft_time = rustfft_freq.clone();
    ifft.process(&mut rustfft_time);

    // Scale rustfft result by 1/N to match our unitary transform
    let scale = 1.0 / N as f32;
    for c in &mut rustfft_time {
        *c = Complex {
            re: c.re * scale,
            im: c.im * scale,
        };
    }

    // Compare our IFFT result with rustfft's roundtrip
    assert_eq!(our_ifft.len(), N);
    let mut max_diff: f32 = 0.0;
    for (_i, (ours, rustfft)) in our_ifft.iter().zip(rustfft_time.iter()).enumerate() {
        let diff = ((ours.re - rustfft.re).powi(2) + (ours.im - rustfft.im).powi(2)).sqrt();
        max_diff = max_diff.max(diff);
        assert!(
            diff < EPSILON,
            "ours={ours:?}  rustfft={rustfft:?}  diff={diff:.2e}"
        );
    }
    println!("IFFT vs rustfft roundtrip max element-wise error: {max_diff:.2e}");
}

#[test]
fn test_fft_ifft_properties() {
    let fft = GpuFft::new().expect("GPU required");
    let input = make_test_signal(N);

    // FFT then IFFT should return scaled original
    let spectrum_batch = fft.fft(&[input.clone()]).expect("FFT failed");
    let spectrum = &spectrum_batch[0];
    let reconstructed_batch = fft.ifft(&[spectrum.to_vec()]).expect("IFFT failed");
    let reconstructed = &reconstructed_batch[0];

    // Check that FFT(IFFT(x)) = x and IFFT(FFT(x)) = x (within numerical precision)
    let mut max_diff: f32 = 0.0;
    for (orig, recon) in input.iter().zip(reconstructed.iter()) {
        let diff = ((orig.re - recon.re).powi(2) + (orig.im - recon.im).powi(2)).sqrt();
        max_diff = max_diff.max(diff);
    }
    println!("FFT->IFFT roundtrip max error: {max_diff:.2e}");
    assert!(max_diff < EPSILON);

    // Test that IFFT(FFT(x)) = x
    let spectrum2_batch = fft.fft(&[reconstructed.to_vec()]).expect("FFT failed");
    let spectrum2 = &spectrum2_batch[0];
    let mut max_diff2: f32 = 0.0;
    for (s1, s2) in spectrum.iter().zip(spectrum2.iter()) {
        let diff = ((s1.re - s2.re).powi(2) + (s1.im - s2.im).powi(2)).sqrt();
        max_diff2 = max_diff2.max(diff);
    }
    println!("IFFT->FFT roundtrip max error: {max_diff2:.2e}");
    assert!(max_diff2 < EPSILON);
}
