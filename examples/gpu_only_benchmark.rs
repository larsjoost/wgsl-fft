use wgsl_fft::{
    benchmark::{benchmark_gpu_only, benchmark_rival, MAX_TOTAL_SAMPLES},
    GpuFft,
};

fn main() {
    println!("\n=== GPU-ONLY PERFORMANCE BENCHMARK ===");
    println!("This benchmark isolates GPU compute + DMA performance, excluding CPU overhead");

    let gpu_fft = GpuFft::new().expect("Failed to init WebGPU");
    let reference = GpuFft::new().expect("Failed to init WebGPU reference");

    let fft_sizes = [256, 1024, 16384, 65536, 1048576];

    for &n in &fft_sizes {
        let batch_size = choose_batch_size(n);
        println!("\n--- N = {} ---", n);

        // Full benchmark (includes CPU overhead)
        let full_result = benchmark_rival(&gpu_fft, &reference, n, batch_size);

        // GPU-only benchmark (isolated GPU + DMA)
        let gpu_only_result =
            benchmark_gpu_only(&gpu_fft, n, batch_size).expect("GPU-only benchmark failed");

        println!(
            "{:>40} | {:>8} | {:>14} | {:>10} | {:>14} | {:>10}",
            "Implementation", "Batch", "Full MS/s", "Full GFLOPS", "GPU MS/s", "GPU GFLOPS"
        );
        println!("{}", "-".repeat(110));

        println!(
            "{:>40} | {:>8} | {:>14.2} | {:>10.2} | {:>14.2} | {:>10.2}",
            "WebGPU (Full Pipeline)",
            batch_size,
            full_result.msamples_per_sec,
            full_result.gflops,
            "N/A",
            "N/A"
        );

        println!(
            "{:>40} | {:>8} | {:>14} | {:>10} | {:>14.2} | {:>10.2}",
            "WebGPU (GPU-only)",
            batch_size,
            "N/A",
            "N/A",
            gpu_only_result.gpu_msamples_per_sec,
            gpu_only_result.gpu_gflops
        );

        // Calculate overhead percentage
        let full_duration = 1.0 / full_result.msamples_per_sec * 1_000_000.0;
        let gpu_duration = gpu_only_result.gpu_duration_sec * 1_000_000.0; // convert to microseconds
        let overhead_pct = ((full_duration - gpu_duration) / full_duration) * 100.0;

        println!("\nOverhead analysis (N={}):", n);
        println!("  Full pipeline duration: {:.2} μs", full_duration);
        println!("  GPU-only duration: {:.2} μs", gpu_duration);
        println!("  CPU/DMA overhead: {:.2}%", overhead_pct);
    }
}

fn choose_batch_size(n: usize) -> usize {
    let preferred = if n <= 1024 {
        1024
    } else if n <= 16384 {
        256
    } else if n <= 65536 {
        64
    } else if n <= 262_144 {
        16
    } else {
        1
    };

    let mut batch = preferred;
    while n.saturating_mul(batch) > MAX_TOTAL_SAMPLES && batch > 1 {
        batch /= 2;
    }
    batch.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── choose_batch_size (pure) ──────────────────────────────────────────────

    #[test]
    fn batch_size_tiers() {
        assert_eq!(choose_batch_size(256), 1024);
        assert_eq!(choose_batch_size(1024), 1024);
        assert_eq!(choose_batch_size(16384), 256);
        assert_eq!(choose_batch_size(65536), 64);
        assert_eq!(choose_batch_size(1_048_576), 1);
    }

    #[test]
    fn batch_size_capped_by_max_total_samples() {
        for &n in &[256usize, 1024, 16384, 65536, 1_048_576] {
            let b = choose_batch_size(n);
            assert!(
                n * b <= MAX_TOTAL_SAMPLES,
                "n={n} * batch={b} exceeds MAX_TOTAL_SAMPLES"
            );
        }
    }

    // ── GPU benchmark smoke tests ─────────────────────────────────────────────

    #[test]
    fn gpu_only_returns_positive_throughput() {
        let gpu = GpuFft::new().expect("GPU required");
        let result = benchmark_gpu_only(&gpu, 256, 4).expect("benchmark failed");
        assert!(
            result.gpu_msamples_per_sec > 0.0,
            "throughput should be positive"
        );
        assert!(result.gpu_gflops > 0.0, "gflops should be positive");
        assert!(result.gpu_duration_sec > 0.0, "duration should be positive");
    }

    #[test]
    fn gpu_only_result_fields_consistent() {
        let gpu = GpuFft::new().expect("GPU required");
        let n = 256;
        let batch = 4;
        let result = benchmark_gpu_only(&gpu, n, batch).expect("benchmark failed");
        assert_eq!(result.n, n);
        assert_eq!(result.batch_size, batch);
        // gflops = 5 * N * batch * log2(N) / t / 1e9
        let expected_gflops =
            5.0 * (n * batch) as f64 * (n as f64).log2() / result.gpu_duration_sec / 1e9;
        let rel_diff = (result.gpu_gflops - expected_gflops).abs() / expected_gflops;
        assert!(
            rel_diff < 1e-6,
            "gflops formula mismatch: rel_diff={rel_diff:.2e}"
        );
    }
}
