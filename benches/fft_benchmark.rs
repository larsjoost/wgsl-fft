use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_complex::Complex;
use wgsl_fft::GpuFft;

fn benchmark_fft(c: &mut Criterion) {
    let fft = GpuFft::new().expect("Failed to create FFT instance");

    // Test realistic FFT sizes for audio/signal processing
    let sizes = [4096, 8192, 16384, 32768, 65536];

    for &size in &sizes {
        // Create realistic test data (sine waves)
        let input: Vec<Complex<f32>> = (0..size)
            .map(|i| {
                let t = i as f32 / size as f32;
                Complex::new(
                    (t * 440.0 * 2.0 * std::f32::consts::PI).sin(),
                    (t * 880.0 * 2.0 * std::f32::consts::PI).sin() * 0.5,
                )
            })
            .collect();

        // Use larger sample size for more accurate measurements
        let mut group = c.benchmark_group(format!("FFT size {}", size));
        group.sample_size(20);
        group.warm_up_time(std::time::Duration::from_secs(1));
        group.measurement_time(std::time::Duration::from_secs(5));

        group.bench_function("Single FFT", |b| {
            b.iter(|| {
                let _result = fft.fft(black_box(&[input.clone()])).unwrap();
            })
        });

        // Benchmark batch processing (more realistic workload)
        group.bench_function("Batch of 10 FFTs", |b| {
            b.iter(|| {
                let batch = vec![input.clone(); 10];
                let _result = fft.fft(black_box(&batch)).unwrap();
            })
        });

        group.finish();
    }
}

fn benchmark_ifft(c: &mut Criterion) {
    let fft = GpuFft::new().expect("Failed to create FFT instance");

    // Test IFFT with realistic data
    let size = 16384;
    let input: Vec<Complex<f32>> = (0..size)
        .map(|i| {
            let t = i as f32 / size as f32;
            Complex::new(
                (t * 440.0 * 2.0 * std::f32::consts::PI).sin(),
                (t * 880.0 * 2.0 * std::f32::consts::PI).sin() * 0.5,
            )
        })
        .collect();

    // First compute FFT to get frequency domain data
    let spectrum_batch = fft.fft(&[input.clone()]).unwrap();
    let spectrum = &spectrum_batch[0];

    let mut group = c.benchmark_group("IFFT");
    group.sample_size(20);
    group.warm_up_time(std::time::Duration::from_secs(1));
    group.measurement_time(std::time::Duration::from_secs(5));

    group.bench_function("Single IFFT size 16384", |b| {
        b.iter(|| {
            let _result = fft.ifft(black_box(&[spectrum.clone()])).unwrap();
        })
    });

    group.bench_function("Batch of 5 IFFTs size 16384", |b| {
        b.iter(|| {
            let batch = vec![spectrum.clone(); 5];
            let _result = fft.ifft(black_box(&batch)).unwrap();
        })
    });

    group.finish();
}

fn benchmark_roundtrip(c: &mut Criterion) {
    let fft = GpuFft::new().expect("Failed to create FFT instance");

    // Test realistic audio processing workload
    let size = 8192;
    let input: Vec<Complex<f32>> = (0..size)
        .map(|i| {
            let t = i as f32 / size as f32;
            Complex::new(
                (t * 440.0 * 2.0 * std::f32::consts::PI).sin(),
                (t * 880.0 * 2.0 * std::f32::consts::PI).sin() * 0.5,
            )
        })
        .collect();

    let mut group = c.benchmark_group("Roundtrip");
    group.sample_size(20);
    group.warm_up_time(std::time::Duration::from_secs(1));
    group.measurement_time(std::time::Duration::from_secs(5));

    group.bench_function("Single FFT+IFFT size 8192", |b| {
        b.iter(|| {
            let spectrum_batch = fft.fft(black_box(&[input.clone()])).unwrap();
            let spectrum = &spectrum_batch[0];
            let _reconstructed = fft.ifft(black_box(&[spectrum.clone()])).unwrap();
        })
    });

    group.bench_function("Batch of 3 roundtrips size 8192", |b| {
        b.iter(|| {
            let batch = vec![input.clone(); 3];
            let spectrum_batch = fft.fft(black_box(&batch)).unwrap();
            let _reconstructed = fft.ifft(black_box(&spectrum_batch)).unwrap();
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_fft, benchmark_ifft, benchmark_roundtrip);

criterion_main!(benches);
