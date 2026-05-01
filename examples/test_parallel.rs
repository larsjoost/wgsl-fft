use num_complex::Complex;
use std::time::Instant;
use wgsl_fft::GpuFft;

fn main() {
    let fft = GpuFft::new().expect("Failed to create FFT instance");

    println!("Testing Parallel Batch Processing");
    println!("==================================");

    // Test different batch sizes
    let fft_size = 1024;
    let batch_sizes = vec![1, 4, 16, 64, 256];

    for &batch_size in &batch_sizes {
        // Generate test data
        let inputs: Vec<Vec<Complex<f32>>> = (0..batch_size)
            .map(|i| {
                (0..fft_size)
                    .map(|j| Complex::new((i * fft_size + j) as f32 * 0.1, 0.0))
                    .collect()
            })
            .collect();

        // Warm-up
        let _ = fft.fft(&inputs).expect("FFT failed");

        // Timed run
        let start = Instant::now();
        let results = fft.fft(&inputs).expect("FFT failed");
        let duration = start.elapsed();

        let total_samples = batch_size * fft_size;
        let samples_per_second = total_samples as f64 / duration.as_secs_f64();
        let mega_samples_per_second = samples_per_second / 1_000_000.0;

        println!(
            "Batch Size {}: {:.2} MSa/s ({} FFTs, {} samples)",
            batch_size, mega_samples_per_second, batch_size, total_samples
        );

        // Verify results are correct (basic sanity check)
        for result in &results {
            assert_eq!(result.len(), fft_size);
        }
    }

    println!("\nParallel processing test completed successfully!");
}

#[cfg(test)]
mod tests {
    use num_complex::Complex;
    use wgsl_fft::GpuFft;

    fn impulse(n: usize) -> Vec<Complex<f32>> {
        let mut v = vec![Complex::new(0.0f32, 0.0); n];
        v[0] = Complex::new(1.0, 0.0);
        v
    }

    // ── Output shape ──────────────────────────────────────────────────────────

    #[test]
    fn output_length_matches_input() {
        let fft = GpuFft::new().expect("GPU required");
        for &n in &[64usize, 256, 1024] {
            let input = vec![vec![Complex::new(1.0f32, 0.0); n]];
            let out = fft.fft(&input).expect("fft failed");
            assert_eq!(out[0].len(), n, "output length should equal n={n}");
        }
    }

    #[test]
    fn batch_output_count_matches_input() {
        let fft = GpuFft::new().expect("GPU required");
        let n = 256;
        for &batch in &[1usize, 4, 16] {
            let input: Vec<Vec<Complex<f32>>> = (0..batch)
                .map(|_| vec![Complex::new(0.1f32, 0.0); n])
                .collect();
            let out = fft.fft(&input).expect("fft failed");
            assert_eq!(out.len(), batch, "output batch count should be {batch}");
        }
    }

    // ── Impulse → flat spectrum ───────────────────────────────────────────────

    #[test]
    fn impulse_spectrum_flat() {
        let fft = GpuFft::new().expect("GPU required");
        let n = 256;
        let out = fft.fft(&[impulse(n)]).expect("fft failed");
        for (k, c) in out[0].iter().enumerate() {
            let mag = c.norm();
            assert!(
                (mag - 1.0).abs() < 1e-3,
                "bin {k} magnitude {mag:.4} should be ~1 for impulse"
            );
        }
    }

    // ── Batch independence ────────────────────────────────────────────────────

    #[test]
    fn batch_results_independent() {
        // Each signal in a batch should produce the same result as running it alone.
        let fft = GpuFft::new().expect("GPU required");
        let n = 128;
        let signals: Vec<Vec<Complex<f32>>> = (0..4)
            .map(|i| {
                (0..n)
                    .map(|j| Complex::new((i * 10 + j) as f32 * 0.001, 0.0))
                    .collect()
            })
            .collect();

        let batch_out = fft.fft(&signals).expect("batch fft failed");
        for (i, sig) in signals.iter().enumerate() {
            let single_out = fft.fft(&[sig.clone()]).expect("single fft failed");
            for (k, (a, b)) in single_out[0].iter().zip(batch_out[i].iter()).enumerate() {
                let err = (a - b).norm();
                assert!(
                    err < 1e-4,
                    "signal {i} bin {k}: batch vs single diff {err:.2e}"
                );
            }
        }
    }
}
