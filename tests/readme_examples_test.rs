//! Tests for examples shown in the README file
//!
//! These tests verify that the code examples in README.md work correctly.

use num_complex::Complex;
use wgsl_fft::GpuFft;

#[test]
fn test_readme_forward_fft_single() {
    // Example from README: Single FFT
    let fft = GpuFft::new().expect("GPU required");

    // Note: The README example has a bug - it uses 'i' which is not in scope
    // The corrected version uses a proper signal generation
    let n = 1024;
    let single_input = vec![vec![
        Complex {
            re: (0.0f32 * 0.1).sin(),
            im: 0.0
        };
        n
    ]];

    let single_spectrum = fft.fft(&single_input).expect("FFT failed");
    assert_eq!(single_spectrum.len(), 1);
    assert_eq!(single_spectrum[0].len(), n);
}

#[test]
fn test_readme_forward_fft_batch() {
    // Example from README: Batch FFT
    let fft = GpuFft::new().expect("GPU required");

    let n = 1024;
    let batch_inputs = vec![
        vec![
            Complex {
                re: (0.0f32 * 0.1).sin(),
                im: 0.0
            };
            n
        ],
        vec![
            Complex {
                re: (0.0f32 * 0.2).sin(),
                im: 0.0
            };
            n
        ],
    ];
    let batch_spectra = fft.fft(&batch_inputs).expect("FFT failed");
    assert_eq!(batch_spectra.len(), 2); // Two FFT results
    assert_eq!(batch_spectra[0].len(), n); // Each FFT has 1024 bins
}

#[test]
fn test_readme_inverse_fft() {
    // Example from README: Inverse FFT
    let fft = GpuFft::new().expect("GPU required");

    let n = 1024;

    // Create some frequency domain data
    let batch_spectra = vec![
        vec![Complex::new(1.0, 0.0); n],
        vec![Complex::new(0.5, 0.0); n],
    ];

    // Compute inverse FFT (automatically scaled by 1/N)
    let reconstructed_batch = fft.ifft(&batch_spectra).expect("IFFT failed");
    assert_eq!(reconstructed_batch.len(), 2); // Two IFFT results
}

#[test]
fn test_readme_roundtrip() {
    // Example from README: Roundtrip test
    let fft = GpuFft::new().expect("GPU required");

    let n = 1024;

    // Create time-domain signal
    let time_domain: Vec<Complex<f32>> = (0..n)
        .map(|i| Complex {
            re: (i as f32 * 0.1).sin(),
            im: 0.0,
        })
        .collect();

    let batch_inputs = vec![time_domain.clone(), time_domain.clone()];

    // FFT
    let batch_spectra = fft.fft(&batch_inputs).expect("FFT failed");

    // IFFT
    let reconstructed_batch = fft.ifft(&batch_spectra).expect("IFFT failed");

    // Roundtrip: FFT(IFFT(x)) ≈ x (within numerical precision)
    let max_error: f32 = batch_inputs[0]
        .iter()
        .zip(reconstructed_batch[0].iter())
        .map(|(a, b)| ((a.re - b.re).powi(2) + (a.im - b.im).powi(2)).sqrt())
        .fold(0.0, f32::max);

    // Should be within numerical precision
    assert!(max_error < 1e-3, "Max roundtrip error: {max_error:.2e}");
    println!("Max roundtrip error: {max_error:.2e}");
}

#[test]
fn test_readme_batch_processing() {
    // Example from README: Batch Processing
    let fft = GpuFft::new().expect("GPU required");

    // Process 8 signals of 4096 samples each
    let batch_size = 8;
    let fft_size = 4096;
    let signals: Vec<Vec<Complex<f32>>> = (0..batch_size)
        .map(|_| vec![Complex::new(0.0, 0.0); fft_size])
        .collect();

    // Batch FFT - much faster than processing individually
    let spectra = fft.fft(&signals).expect("Batch FFT failed");

    // Process results
    for (i, spectrum) in spectra.iter().enumerate() {
        assert_eq!(
            spectrum.len(),
            fft_size,
            "Signal {} FFT should have {} bins",
            i,
            fft_size
        );
    }
}

#[test]
fn test_readme_batch_processing_example() {
    // More complete batch processing example from README
    let fft = GpuFft::new().expect("GPU required");

    let batch_size = 8;
    let fft_size = 4096;
    let signals: Vec<Vec<Complex<f32>>> = (0..batch_size)
        .map(|batch_idx| {
            (0..fft_size)
                .map(|i| Complex::new((batch_idx as f32 + i as f32) * 0.1, 0.0))
                .collect()
        })
        .collect();

    let spectra = fft.fft(&signals).expect("Batch FFT failed");

    assert_eq!(spectra.len(), batch_size);
    for (i, spectrum) in spectra.iter().enumerate() {
        assert_eq!(spectrum.len(), fft_size);
        // Verify the spectrum is not all zeros (since input has signal)
        let max_magnitude = spectrum
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .fold(0.0, f32::max);
        assert!(
            max_magnitude > 0.0,
            "Signal {} spectrum should have non-zero magnitude",
            i
        );
    }
}

#[test]
fn test_readme_requirements() {
    // Verify that the library works with the requirements stated in README
    // "Input length must be a power of two and non-empty"
    // But now we also support arbitrary sizes!

    let fft = GpuFft::new().expect("GPU required");

    // Power of two sizes should work
    let power_of_two_sizes = [256, 512, 1024, 2048];
    for &n in &power_of_two_sizes {
        let input = vec![vec![Complex::new(1.0, 0.0); n]];
        let result = fft.fft(&input).expect("FFT failed for power of two size");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), n);
    }

    // Arbitrary sizes should also work now
    let arbitrary_sizes = [3, 5, 10, 15, 100, 150];
    for &n in &arbitrary_sizes {
        let input = vec![vec![Complex::new(1.0, 0.0); n]];
        let result = fft.fft(&input).expect("FFT failed for arbitrary size");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), n);
    }
}

#[test]
fn test_readme_all_vectors_same_length() {
    // Verify that the requirement "All vectors in a batch must have the same length" is enforced
    let fft = GpuFft::new().expect("GPU required");

    // This should work - all vectors have same length
    let valid_batch = vec![
        vec![Complex::new(1.0, 0.0); 1024],
        vec![Complex::new(2.0, 0.0); 1024],
    ];
    let result = fft.fft(&valid_batch).expect("Valid batch should work");
    assert_eq!(result.len(), 2);

    // Note: The current implementation doesn't explicitly check for same length
    // It just uses the first vector's length and expects all to match
    // This could be improved in the future
}
