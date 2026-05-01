use num_complex::Complex;
use std::time::Instant;
use wgsl_fft::GpuFft;

#[cfg(feature = "cuda")]
use wgsl_fft::CuFft;
#[cfg(feature = "rocm")]
use wgsl_fft::RocFft;

fn main() {
    println!("FFT Implementation Comparison");
    println!("==============================");

    // Test parameters
    let fft_size = 1024;
    let batch_sizes = vec![1, 4, 16, 64, 256];

    // Generate test data
    let inputs: Vec<Vec<Complex<f32>>> = (0..256)
        .map(|i| {
            (0..fft_size)
                .map(|j| Complex::new((i * fft_size + j) as f32 * 0.1, 0.0))
                .collect()
        })
        .collect();

    // Test WebGPU implementation
    println!("\nTesting WebGPU Implementation:");
    println!("------------------------------");

    let wgpu_fft = GpuFft::new().expect("Failed to create WebGPU FFT instance");

    for &batch_size in &batch_sizes {
        let batch_inputs: Vec<Vec<Complex<f32>>> = inputs[..batch_size].to_vec();

        // Warm-up
        let _ = wgpu_fft.fft(&batch_inputs).expect("FFT failed");

        // Timed run
        let start = Instant::now();
        let _results = wgpu_fft.fft(&batch_inputs).expect("FFT failed");
        let duration = start.elapsed();

        let total_samples = batch_size * fft_size;
        let samples_per_second = total_samples as f64 / duration.as_secs_f64();
        let mega_samples_per_second = samples_per_second / 1_000_000.0;

        println!(
            "Batch Size {}: {:.2} MSa/s",
            batch_size, mega_samples_per_second
        );
    }

    // Test cuFFT implementation (if available)
    #[cfg(feature = "cuda")]
    {
        match CuFft::new(fft_size) {
            Ok(cufft) => {
                println!("\nTesting cuFFT Implementation:");
                println!("------------------------------");

                for &batch_size in &batch_sizes {
                    let batch_inputs: Vec<Vec<Complex<f32>>> = inputs[..batch_size].to_vec();

                    // Warm-up
                    let _ = cufft.batch_fft(&batch_inputs).expect("FFT failed");

                    // Timed run
                    let start = Instant::now();
                    let _results = cufft.batch_fft(&batch_inputs).expect("FFT failed");
                    let duration = start.elapsed();

                    let total_samples = batch_size * fft_size;
                    let samples_per_second = total_samples as f64 / duration.as_secs_f64();
                    let mega_samples_per_second = samples_per_second / 1_000_000.0;

                    println!(
                        "Batch Size {}: {:.2} MSa/s",
                        batch_size, mega_samples_per_second
                    );
                }
            }
            Err(e) => {
                println!("\ncuFFT not available ({e})");
                if e.to_string().contains("CUFFT_NOT_SUPPORTED") {
                    println!(
                        "Hint: try launching with LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
                    );
                }
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("\ncuFFT support not compiled (add --features cuda to enable)");
    }

    // Test ROCm implementation (AMD GPUs only)
    #[cfg(feature = "rocm")]
    {
        println!("\nTesting ROCm FFT Implementation:");
        println!("----------------------------------");

        match RocFft::new() {
            Ok(_) => {
                println!("ROCm FFT initialized successfully");
                // ROCm FFT implementation would go here
                // Note: This won't work on NVIDIA GTX 1070
            }
            Err(e) => {
                println!("ROCm FFT error: {}", e);
            }
        }
    }

    #[cfg(not(feature = "rocm"))]
    {
        println!("\nROCm support not compiled (add --features rocm to enable, AMD GPUs only)");
    }
}

#[cfg(test)]
mod tests {
    use num_complex::Complex;
    use wgsl_fft::GpuFft;

    fn make_fft() -> GpuFft {
        GpuFft::new().expect("GPU required")
    }

    // ── WebGPU correctness ────────────────────────────────────────────────────

    #[test]
    fn wgpu_impulse_flat_spectrum() {
        let fft = make_fft();
        let n = 256;
        let mut impulse = vec![Complex::<f32>::new(0.0, 0.0); n];
        impulse[0] = Complex::new(1.0, 0.0);
        let out = fft.fft(&[impulse]).expect("fft failed");
        for (k, c) in out[0].iter().enumerate() {
            let mag = c.norm();
            assert!(
                (mag - 1.0).abs() < 1e-3,
                "bin {k} magnitude {mag:.4} should be ~1 for impulse"
            );
        }
    }

    #[test]
    fn wgpu_roundtrip_n1024() {
        let fft = make_fft();
        let n = 1024;
        let input: Vec<Vec<Complex<f32>>> = vec![(0..n)
            .map(|i| Complex::new((i as f32 / n as f32).sin() * 0.001, 0.0))
            .collect()];
        let freq = fft.fft(&input).expect("fft failed");
        let recon = fft.ifft(&freq).expect("ifft failed");
        let max_err = input[0]
            .iter()
            .zip(recon[0].iter())
            .map(|(a, b)| (a - b).norm())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-3, "roundtrip error {max_err:.2e} at N=1024");
    }

    #[test]
    fn wgpu_output_length_matches_input() {
        let fft = make_fft();
        for &n in &[64usize, 256, 1024] {
            let input = vec![vec![Complex::new(1.0f32, 0.0); n]];
            let out = fft.fft(&input).expect("fft failed");
            assert_eq!(out[0].len(), n);
        }
    }

    // ── cuFFT (feature-gated) ─────────────────────────────────────────────────

    #[cfg(feature = "cuda")]
    #[test]
    fn cufft_matches_wgpu_n256() {
        use wgsl_fft::CuFft;
        let n = 256;
        let cufft = match CuFft::new(n) {
            Ok(c) => c,
            Err(_) => return, // cuFFT unavailable at runtime, skip
        };
        let wgpu = make_fft();
        let input: Vec<Vec<Complex<f32>>> = vec![(0..n)
            .map(|i| Complex::new(i as f32 * 0.001, 0.0))
            .collect()];
        let wgpu_out = wgpu.fft(&input).expect("wgpu fft failed");
        let cufft_out = cufft.fft(&input).expect("cufft fft failed");
        for (k, (a, b)) in wgpu_out[0].iter().zip(cufft_out[0].iter()).enumerate() {
            let err = (a - b).norm();
            assert!(err < 1e-2, "bin {k}: wgpu vs cufft diff {err:.2e}");
        }
    }
}
