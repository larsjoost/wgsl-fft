use num_complex::Complex;
use std::time::{Duration, Instant};
use wgsl_fft::GpuFft;

fn make_input(n: usize) -> Vec<Complex<f32>> {
    (0..n)
        .map(|i| Complex {
            re: (i as f32 * 0.1).sin(),
            im: (i as f32 * 0.07).cos(),
        })
        .collect()
}

struct BenchResult {
    n: usize,
    iters: usize,
    ms_per_call: f64,
    msamples_per_s: f64,
    gflops: f64,
}

/// Measure GPU FFT throughput for a single size `n`.
///
/// Runs a short warmup, then measures for at least `min_duration` (and at
/// least `min_iters` iterations) to amortise per-submit overhead.
fn bench(fft: &GpuFft, n: usize, min_iters: usize, min_duration: Duration) -> BenchResult {
    let input = make_input(n);

    // Warmup: let the GPU driver / shader JIT settle
    for _ in 0..5 {
        fft.fft(&[input.clone()]).unwrap();
    }

    // Measurement loop
    let mut iters = 0usize;
    let start = Instant::now();
    loop {
        fft.fft(&[input.clone()]).unwrap();
        iters += 1;
        if iters >= min_iters && start.elapsed() >= min_duration {
            break;
        }
    }
    let elapsed_s = start.elapsed().as_secs_f64();

    // Standard FFT FLOP estimate: 5 N log2(N) real multiplies + additions
    let log2_n = (n as f64).log2();
    let flops_per_fft = 5.0 * n as f64 * log2_n;

    BenchResult {
        n,
        iters,
        ms_per_call: elapsed_s / iters as f64 * 1_000.0,
        msamples_per_s: (n * iters) as f64 / elapsed_s / 1e6,
        gflops: flops_per_fft * iters as f64 / elapsed_s / 1e9,
    }
}

/// Reports GPU FFT throughput across a range of sizes.
///
/// Run with:
/// ```
/// cargo test fft_throughput -- --nocapture
/// ```
///
/// Output columns:
/// - **N**          — FFT length (power of two)
/// - **iters**      — number of completed FFT calls during measurement
/// - **MSamples/s** — complex samples processed per second (millions)
/// - **GFLOPS**     — estimated floating-point throughput (5 N log₂N model)
/// - **ms/call**    — wall-clock time per single FFT call
#[test]
fn fft_throughput() {
    let fft = GpuFft::new().expect("GPU required");

    // (size, min_iters, min_duration_ms)
    // Larger sizes need fewer iterations to reach the time budget.
    let configs: &[(usize, usize, u64)] = &[
        (256, 50, 500),
        (1_024, 50, 500),
        (4_096, 20, 500),
        (16_384, 10, 500),
        (65_536, 10, 1_000),
        (262_144, 5, 1_000),
    ];

    println!();
    println!(
        "{:>8}  {:>8}  {:>12}  {:>10}  {:>10}",
        "N", "iters", "MSamples/s", "GFLOPS", "ms/call",
    );
    println!("{}", "-".repeat(58));

    for &(n, min_iters, min_ms) in configs {
        let r = bench(&fft, n, min_iters, Duration::from_millis(min_ms));
        println!(
            "{:>8}  {:>8}  {:>12.2}  {:>10.3}  {:>10.3}",
            r.n, r.iters, r.msamples_per_s, r.gflops, r.ms_per_call,
        );
    }
    println!();
}
