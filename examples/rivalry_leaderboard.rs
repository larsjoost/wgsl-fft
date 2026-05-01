use std::process::Command;
use wgsl_fft::{
    benchmark::{benchmark_gpu_pipeline, validate_rival, ValidationOutcome, MAX_TOTAL_SAMPLES},
    FftExecutor, GpuFft,
};

#[cfg(feature = "cuda")]
use wgsl_fft::CuFft;

#[cfg(feature = "hipfft")]
use wgsl_fft::HipFft;

/// Detect if NVIDIA GPU is available for cuFFT
#[cfg(feature = "cuda")]
fn probe_cufft_runtime() -> Result<(), String> {
    use std::panic;

    match panic::catch_unwind(|| CuFft::new(256)) {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => Err("panic during cuFFT initialization".to_string()),
    }
}

#[cfg(not(feature = "cuda"))]
fn probe_cufft_runtime() -> Result<(), String> {
    Err("binary was built without --features cuda".to_string())
}

fn is_host_nvidia_present() -> bool {
    if let Ok(output) = Command::new("nvidia-smi")
        .arg("--query-gpu=name")
        .arg("--format=csv,noheader")
        .output()
    {
        if output.status.success() {
            let name = String::from_utf8_lossy(&output.stdout);
            if !name.trim().is_empty() {
                return true;
            }
        }
    }

    if let Ok(output) = Command::new("lspci").output() {
        if output.status.success() {
            let listing = String::from_utf8_lossy(&output.stdout);
            return listing.to_lowercase().contains("nvidia");
        }
    }

    false
}

fn main() {
    println!("\n=== WGSL-FFT RIVALRY LEADERBOARD ===");
    println!("Measuring complete GPU pipeline performance (host-to-device + GPU compute + device-to-host)");

    // Distinguish host GPU detection from cuFFT runtime availability.
    let host_nvidia_present = is_host_nvidia_present();
    let cufft_probe = probe_cufft_runtime();
    let cufft_available = cufft_probe.is_ok();
    if cufft_available {
        println!("🟢 NVIDIA GPU detected - cuFFT benchmarks will be included");
    } else if host_nvidia_present {
        if cfg!(feature = "cuda") {
            println!("🟡 NVIDIA GPU detected, but cuFFT runtime init failed");
            if let Err(err) = &cufft_probe {
                println!("   cuFFT init error: {err}");
                if err.contains("CUFFT_NOT_SUPPORTED") {
                    println!("   Hint: try launching with LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH");
                }
            }
            println!("   cuFFT benchmarks will be skipped for this run");
        } else {
            println!("🟡 NVIDIA GPU detected, but this binary was built without --features cuda");
            println!("   cuFFT benchmarks will be skipped");
        }
    } else {
        println!("🔴 No NVIDIA GPU detected - cuFFT benchmarks will be skipped");
        println!("   To enable cuFFT: build with --features cuda and ensure NVIDIA drivers are installed");
    }

    // Baseline is the single source of truth for validation.
    let reference = GpuFft::new().expect("Failed to init WebGPU reference");

    let mut rivals: Vec<Box<dyn FftExecutor>> = Vec::new();
    rivals.push(Box::new(GpuFft::new().expect("Failed to init WebGPU")));
    rivals.push(Box::new(wgsl_fft::rivals::radix4::Radix4Rival::new()));
    rivals.push(Box::new(
        wgsl_fft::rivals::radix4_proper::Radix4ProperFft::new(),
    ));
    rivals.push(Box::new(wgsl_fft::rivals::claude::ClaudeFft::new()));
    rivals.push(Box::new(wgsl_fft::rivals::codex::CodexFft::new()));
    rivals.push(Box::new(wgsl_fft::rivals::devstral_2::Devstral2Fft::new()));
    rivals.push(Box::new(wgsl_fft::rivals::gemini::GeminiFft::new()));
    rivals.push(Box::new(
        wgsl_fft::rivals::mistral_vibe::MistralVibeFft::new(),
    ));

    #[cfg(feature = "cuda")]
    if let Ok(cufft) = CuFft::new(1024) {
        rivals.push(Box::new(cufft));
    }

    #[cfg(feature = "hipfft")]
    if let Ok(hipfft) = HipFft::new(1024) {
        rivals.push(Box::new(hipfft));
    }

    let fft_sizes = [256, 1024, 16384, 65536, 1048576];

    for &n in &fft_sizes {
        let batch_size = choose_batch_size(n);
        println!("\n--- N = {} ---", n);
        println!(
            "{:>32} | {:>8} | {:>14} | {:>10} | {:>8}",
            "Implementation", "Batch", "MSamples/s", "GFLOPS", "Status"
        );
        println!("{}", "-".repeat(84));

        for rival in &rivals {
            match benchmark_gpu_pipeline(rival.as_ref(), n, batch_size) {
                Ok(gpu_result) => {
                    let validation = validate_rival(rival.as_ref(), &reference, n, batch_size);
                    let status = match &validation {
                        ValidationOutcome::Pass => "PASS".to_string(),
                        ValidationOutcome::Fail { max_error } => {
                            format!("FAIL({:.2e})", max_error)
                        }
                    };
                    println!(
                        "{:>32} | {:>8} | {:>14.2} | {:>10.2} | {:>8}",
                        gpu_result.rival_name,
                        gpu_result.batch_size,
                        gpu_result.gpu_msamples_per_sec,
                        gpu_result.gpu_gflops,
                        status,
                    );
                }
                Err(e) => {
                    println!(
                        "{:>32} | {:>8} | {:>14} | {:>10} | {:>8}",
                        rival.name(),
                        batch_size,
                        "ERROR",
                        "-",
                        format!("{}", e),
                    );
                }
            }
        }
    }

    println!("\nAdd your implementation to src/rivals/ and register it above to compete.");
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
    use num_complex::Complex;

    // ── choose_batch_size (pure logic) ────────────────────────────────────────

    #[test]
    fn batch_size_small_n() {
        assert_eq!(choose_batch_size(256), 1024);
        assert_eq!(choose_batch_size(1024), 1024);
    }

    #[test]
    fn batch_size_medium_n() {
        assert_eq!(choose_batch_size(16384), 256);
        assert_eq!(choose_batch_size(65536), 64);
    }

    #[test]
    fn batch_size_large_n() {
        assert_eq!(choose_batch_size(262_144), 16);
        assert_eq!(choose_batch_size(1_048_576), 1);
    }

    #[test]
    fn batch_size_never_exceeds_cap() {
        for &n in &[256usize, 1024, 16384, 65536, 262_144, 1_048_576] {
            let b = choose_batch_size(n);
            assert!(
                n * b <= MAX_TOTAL_SAMPLES,
                "n={n} batch={b} exceeds MAX_TOTAL_SAMPLES"
            );
        }
    }

    #[test]
    fn batch_size_always_at_least_one() {
        for n in [1usize, 2, 256, 1024, 16384, 65536, 1_048_576] {
            assert!(choose_batch_size(n) >= 1, "n={n} gave batch_size 0");
        }
    }

    // ── GPU correctness ───────────────────────────────────────────────────────

    #[test]
    fn fft_roundtrip_n256() {
        let fft = GpuFft::new().expect("GPU required");
        let n = 256usize;
        let input: Vec<Vec<Complex<f32>>> = vec![(0..n)
            .map(|i| Complex::new((i as f32).sin() * 0.001, 0.0))
            .collect()];
        let spectrum = fft.fft(&input).expect("fft failed");
        let recon = fft.ifft(&spectrum).expect("ifft failed");
        for (a, b) in input[0].iter().zip(recon[0].iter()) {
            let err = (a - b).norm();
            assert!(err < 1e-3, "roundtrip error {err:.2e} at N=256");
        }
    }
}
