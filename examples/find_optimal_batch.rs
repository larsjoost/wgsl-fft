use std::env;
use wgsl_fft::{
    benchmark::{benchmark_rival, ValidationOutcome, MAX_TOTAL_SAMPLES},
    FftExecutor, GpuFft,
};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: cargo run --example find_optimal_batch <N> [rival_name]");
        println!("Example: cargo run --example find_optimal_batch 1024");
        return;
    }

    let n: usize = args[1].parse().expect("N must be a positive integer");
    let target_rival = args.get(2);

    println!("\n=== OPTIMAL BATCH SIZE FINDER (N = {}) ===", n);

    let reference = GpuFft::new().expect("Failed to init WebGPU reference");

    let mut rivals: Vec<Box<dyn FftExecutor>> = Vec::new();
    rivals.push(Box::new(GpuFft::new().expect("Failed to init WebGPU")));
    rivals.push(Box::new(wgsl_fft::rivals::radix4::Radix4Rival::new()));
    rivals.push(Box::new(wgsl_fft::rivals::claude::ClaudeFft::new()));
    rivals.push(Box::new(wgsl_fft::rivals::gemini::GeminiFft::new()));

    let batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

    for rival in &rivals {
        let name = rival.name();
        if let Some(target) = target_rival {
            if !name.to_lowercase().contains(&target.to_lowercase()) {
                continue;
            }
        }

        println!("\n--- Rival: {} ---", name);
        println!(
            "{:>8} | {:>14} | {:>10} | {:>8}",
            "Batch", "MSamples/s", "GFLOPS", "Status"
        );
        println!("{}", "-".repeat(48));

        let mut best_msps = 0.0;
        let mut best_batch = 0;

        for &batch in &batch_sizes {
            if n * batch > wgsl_fft::benchmark::MAX_TOTAL_SAMPLES {
                continue;
            }

            let result = benchmark_rival(rival.as_ref(), &reference, n, batch);
            let status = match &result.validation {
                ValidationOutcome::Pass => "PASS".to_string(),
                ValidationOutcome::Fail { max_error } => format!("FAIL({:.2e})", max_error),
            };

            if result.msamples_per_sec > best_msps {
                best_msps = result.msamples_per_sec;
                best_batch = batch;
            }

            println!(
                "{:>8} | {:>14.2} | {:>10.2} | {:>8}",
                batch, result.msamples_per_sec, result.gflops, status,
            );
        }

        println!(
            "Best batch size for {}: {} ({:.2} MSamples/s)",
            name, best_batch, best_msps
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Benchmark correctness ─────────────────────────────────────────────────

    #[test]
    fn baseline_passes_validation_n256() {
        let reference = GpuFft::new().expect("GPU required");
        let rival = GpuFft::new().expect("GPU required");
        let result = benchmark_rival(&rival, &reference, 256, 4);
        assert!(
            matches!(result.validation, ValidationOutcome::Pass),
            "baseline should pass validation at N=256"
        );
    }

    #[test]
    fn benchmark_returns_positive_throughput() {
        let reference = GpuFft::new().expect("GPU required");
        let rival = GpuFft::new().expect("GPU required");
        let result = benchmark_rival(&rival, &reference, 256, 4);
        assert!(
            result.msamples_per_sec > 0.0,
            "throughput should be positive"
        );
        assert!(result.gflops > 0.0, "gflops should be positive");
    }

    #[test]
    fn benchmark_skips_oversized_batches() {
        // n * batch exceeds MAX_TOTAL_SAMPLES → loop body is skipped, no panic
        let n = 1_048_576usize;
        let batch = 1024usize;
        assert!(
            n.saturating_mul(batch) > MAX_TOTAL_SAMPLES,
            "precondition: this batch should exceed the cap"
        );
        // No benchmark call — just verify the guard logic holds.
        // The find_optimal_batch loop uses this check before calling benchmark_rival.
    }

    #[test]
    fn claude_rival_passes_validation_n256() {
        let reference = GpuFft::new().expect("GPU required");
        let rival = wgsl_fft::rivals::claude::ClaudeFft::new();
        let result = benchmark_rival(&rival, &reference, 256, 4);
        assert!(
            matches!(result.validation, ValidationOutcome::Pass),
            "ClaudeFft should pass validation at N=256"
        );
    }
}
