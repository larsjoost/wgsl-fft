use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_complex::Complex;
use wgsl_fft::GpuFft;

fn benchmark_batch_processing(c: &mut Criterion) {
    let fft = GpuFft::new().expect("Failed to create FFT instance");

    let fft_size = 1024;
    let batch_sizes = vec![1, 4, 16, 64, 256, 512];

    let mut group = c.benchmark_group("Batch FFT Processing");

    for batch_size in batch_sizes {
        // Generate test data
        let inputs: Vec<Vec<Complex<f32>>> = (0..batch_size)
            .map(|i| {
                (0..fft_size)
                    .map(|j| Complex::new((i * fft_size + j) as f32 * 0.1, 0.0))
                    .collect()
            })
            .collect();

        group.bench_function(format!("batch_size_{}", batch_size), |b| {
            b.iter(|| {
                let _results = fft.fft(black_box(&inputs)).expect("FFT failed");
            });
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_batch_processing);
criterion_main!(benches);
