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

/// Test prime number FFT sizes
/// Prime numbers are important edge cases for FFT algorithms
#[test]
fn test_prime_number_sizes() {
    let gpu = GpuFft::new().expect("GPU required");
    // Small and medium prime numbers
    let prime_sizes = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199];
    for &n in &prime_sizes {
        test_fft_size(&gpu, n);
        test_ifft_size(&gpu, n);
    }
}

/// Test composite number FFT sizes (products of small primes)
/// These test the algorithm's handling of composite factorizations
#[test]
fn test_composite_number_sizes() {
    let gpu = GpuFft::new().expect("GPU required");
    // Products of 2, 3, 5, 7 - non-powers of 2
    let composite_sizes = vec![
        4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28, 30, 32, 33, 34, 35, 36, 38, 39, 40, 42, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 60, 62, 63, 64, 65, 66, 68, 69, 70, 72, 74, 75, 76, 77, 78, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96, 98, 99,
    ];
    for &n in &composite_sizes {
        test_fft_size(&gpu, n);
        test_ifft_size(&gpu, n);
    }
}

/// Test sizes that are multiples of specific numbers (3, 5, 7, etc.)
/// These are common in signal processing applications
#[test]
fn test_multiples_sizes() {
    let gpu = GpuFft::new().expect("GPU required");
    // Multiples of 3, 5, 7, 11, 13
    let sizes = vec![
        3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96, 99,
        5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
        7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98,
        11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121,
        13, 26, 39, 52, 65, 78, 91, 104, 117, 130, 143, 156, 169,
    ];
    for &n in &sizes {
        test_fft_size(&gpu, n);
    }
}

/// Test larger arbitrary sizes (> 100)
/// These test the algorithm with larger non-power-of-2 sizes
#[test]
fn test_larger_arbitrary_sizes() {
    let gpu = GpuFft::new().expect("GPU required");
    let sizes = vec![
        101, 102, 103, 105, 110, 111, 115, 117, 119, 120, 121, 123, 125, 129, 130, 133, 135, 140, 141, 143, 145, 147, 150, 153, 154, 155, 156, 158, 159, 160, 161, 165, 169, 170, 171, 175, 177, 180, 182, 183, 185, 186, 187, 189, 190, 194, 195, 196, 200,
        201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 224, 225, 226, 228, 230, 231, 232, 234, 235, 236, 237, 238, 240, 242, 243, 244, 245, 246, 247, 248, 249, 250,
        252, 253, 254, 255, 256, 258, 259, 260, 261, 262, 264, 265, 266, 267, 268, 270, 272, 273, 274, 275, 276, 278, 279, 280, 282, 284, 285, 286, 287, 288, 289, 290, 291, 292, 294, 295, 296, 297, 298, 299, 300,
    ];
    for &n in &sizes {
        test_fft_size(&gpu, n);
        test_ifft_size(&gpu, n);
    }
}

/// Test very specific non-power-of-2 sizes that might be problematic
/// These include numbers just below powers of 2, and other edge cases
#[test]
fn test_edge_case_sizes() {
    let gpu = GpuFft::new().expect("GPU required");
    // Numbers just below powers of 2 (N-1 where N is power of 2)
    let below_pow2 = vec![1, 3, 7, 15, 31, 63, 127, 255, 511, 1023];
    // Numbers just above powers of 2 (N+1 where N is power of 2)
    let above_pow2 = vec![3, 5, 9, 17, 33, 65, 129, 257, 513, 1025];
    // Numbers in the middle between powers of 2
    let mid_sizes = vec![6, 12, 24, 48, 96, 192, 384];
    // Odd numbers that are not prime
    let odd_composite = vec![9, 15, 21, 25, 27, 33, 35, 39, 45, 49, 51, 55, 57, 63, 65, 69, 75, 77, 81, 85, 87, 91, 93, 95, 99];
    
    let all_sizes: Vec<usize> = below_pow2.into_iter()
        .chain(above_pow2)
        .chain(mid_sizes)
        .chain(odd_composite)
        .collect();
    
    for &n in &all_sizes {
        test_fft_size(&gpu, n);
        test_ifft_size(&gpu, n);
    }
}

/// Test FFT sizes that are products of primes (semiprimes, 3-almost primes, etc.)
#[test]
fn test_product_of_primes_sizes() {
    let gpu = GpuFft::new().expect("GPU required");
    // Semiprimes (product of 2 primes)
    let semiprimes = vec![
        4, 6, 9, 10, 14, 15, 21, 22, 25, 26, 33, 34, 35, 38, 39, 46, 49, 51, 55, 57, 58, 62, 65, 69, 74, 77, 82, 85, 86, 87, 91, 93, 94, 95,
    ];
    // Products of 3 primes
    let three_primes = vec![
        30, 42, 66, 70, 78, 102, 105, 110, 114, 130, 138, 154, 165, 170, 174, 182, 186, 190, 195, 222, 230, 231, 238, 246, 255, 258, 266, 273, 282, 285, 286, 290,
    ];
    
    let all_sizes: Vec<usize> = semiprimes.into_iter()
        .chain(three_primes)
        .collect();
    
    for &n in &all_sizes {
        test_fft_size(&gpu, n);
        test_ifft_size(&gpu, n);
    }
}

/// Test a comprehensive set of all non-power-of-2 sizes in a range
#[test]
fn test_comprehensive_non_pow2_sizes() {
    let gpu = GpuFft::new().expect("GPU required");
    // Test all sizes from 3 to 100 that are NOT powers of 2
    let mut sizes: Vec<usize> = (3..=100).filter(|&n| !GpuFft::is_power_of_two(n)).collect();
    // Also include 1 and 2
    sizes.insert(0, 1);
    sizes.insert(1, 2);
    
    for &n in &sizes {
        test_fft_size(&gpu, n);
        test_ifft_size(&gpu, n);
    }
}

/// Test specific sizes that are commonly used in audio/DSP applications
#[test]
fn test_audio_dsp_common_sizes() {
    let gpu = GpuFft::new().expect("GPU required");
    // Common audio FFT sizes (often related to window sizes)
    let audio_sizes = vec![
        16, 32, 64, 128, 256, 512, 1024, 2048, 4096,  // Powers of 2 (baseline)
        48, 96, 192, 384, 768, 1536, 3072,            // Multiples of 48 (common audio block sizes)
        441, 882, 1764, 3528,                          // NTSC video frame related
        256, 512, 768, 1024, 1536, 2048, 3072,         // Common for audio
        100, 200, 400, 800, 1600, 3200,                 // Round numbers
        90, 180, 360, 720, 1440,                       // Multiples of 90
    ];
    
    for &n in &audio_sizes {
        test_fft_size(&gpu, n);
        test_ifft_size(&gpu, n);
    }
}

/// Test roundtrip for a wide variety of non-power-of-2 sizes
#[test]
fn test_roundtrip_comprehensive_non_pow2() {
    let gpu = GpuFft::new().expect("GPU required");
    let sizes: Vec<usize> = (1..=50)
        .filter(|&n| !GpuFft::is_power_of_two(n))
        .chain((51..=150).filter(|&n| !GpuFft::is_power_of_two(n)))
        .chain(vec![151, 155, 161, 175, 185, 195, 205, 215, 225, 235, 245, 255, 265, 275, 285, 295, 305].into_iter())
        .collect();
    
    for &n in &sizes {
        let input = make_test_signal(n);
        let spectrum_batch = gpu.fft(&[input.clone()]).expect("FFT failed");
        let spectrum = &spectrum_batch[0];
        let reconstructed_batch = gpu.ifft(&[spectrum.to_vec()]).expect("IFFT failed");
        let reconstructed = &reconstructed_batch[0];
        
        let mut max_diff: f32 = 0.0;
        for (i, (orig, recon)) in input.iter().zip(reconstructed.iter()).enumerate() {
            let diff = ((orig.re - recon.re).powi(2) + (orig.im - recon.im).powi(2)).sqrt();
            max_diff = max_diff.max(diff);
            assert!(
                diff < EPSILON,
                "Roundtrip error too large at size {} element {}: diff={:.2e}", n, i, diff
            );
        }
    }
}
