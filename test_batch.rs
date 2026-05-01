use wgsl_fft::GpuFft;
use num_complex::Complex;

fn main() {
    let fft = GpuFft::new().expect("GPU required");
    
    // Test single FFT (should still work)
    let single_input = vec![Complex::new(1.0, 0.0); 16];
    let single_result = fft.fft(&single_input).expect("Single FFT failed");
    println!("Single FFT result length: {}", single_result.len());
    
    // Test batch FFT
    let batch_inputs = vec![
        vec![Complex::new(1.0, 0.0); 16],
        vec![Complex::new(0.5, 0.0); 16],
        vec![Complex::new(0.25, 0.0); 16],
    ];
    
    let batch_results = fft.fft_batch(&batch_inputs).expect("Batch FFT failed");
    println!("Batch FFT results: {} vectors, each of length {}", 
             batch_results.len(), batch_results[0].len());
    
    // Test batch IFFT
    let batch_ifft_results = fft.ifft_batch(&batch_results).expect("Batch IFFT failed");
    println!("Batch IFFT results: {} vectors, each of length {}", 
             batch_ifft_results.len(), batch_ifft_results[0].len());
    
    // Verify roundtrip consistency for batch processing
    for (original, reconstructed) in batch_inputs.iter().zip(batch_ifft_results.iter()) {
        let max_error: f32 = original.iter().zip(reconstructed.iter())
            .map(|(orig, recon)| ((orig.re - recon.re).powi(2) + (orig.im - recon.im).powi(2)).sqrt())
            .fold(0.0, f32::max);
        println!("Max roundtrip error: {}", max_error);
        assert!(max_error < 1e-4, "Batch roundtrip error too large: {}", max_error);
    }
    
    println!("✅ All batch processing tests passed!");
}