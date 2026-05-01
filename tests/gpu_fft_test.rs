use num_complex::Complex;
use rustfft::FftPlanner;
use wgsl_fft::GpuFft;

const N: usize = 1024;
const EPSILON: f32 = 1e-3;

fn make_input(n: usize) -> Vec<Complex<f32>> {
    (0..n)
        .map(|i| Complex {
            re: (i as f32 * 0.1).sin(),
            im: (i as f32 * 0.07).cos(),
        })
        .collect()
}

#[test]
fn test_gpu_fft_matches_rustfft() {
    let input = make_input(N);

    let gpu = GpuFft::new().expect("GPU required");
    let input_for_fft = input.clone();
    let gpu_out_batch = gpu.fft(&[input_for_fft]).expect("GPU FFT failed");
    let gpu_out = &gpu_out_batch[0];

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N);
    let mut cpu_buf = input.clone();
    fft.process(&mut cpu_buf);

    assert_eq!(gpu_out.len(), N);
    let mut max_diff: f32 = 0.0;
    for (i, (g, c)) in gpu_out.iter().zip(cpu_buf.iter()).enumerate() {
        let diff = ((g.re - c.re).powi(2) + (g.im - c.im).powi(2)).sqrt();
        max_diff = max_diff.max(diff);
        assert!(
            diff < EPSILON,
            "element {i}: GPU={g:?}  CPU={c:?}  diff={diff:.2e}"
        );
    }
    println!("max element-wise error: {max_diff:.2e}");
}
