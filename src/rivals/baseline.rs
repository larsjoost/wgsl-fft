use std::any::Any;

use crate::error::Result;
use crate::{FftExecutor, GpuFft};
use num_complex::Complex;

// Need wgpu for the GpuFftTrait implementation
use wgpu;

pub struct StockhamBaseline(pub GpuFft);

impl Default for StockhamBaseline {
    fn default() -> Self {
        Self::new()
    }
}

impl StockhamBaseline {
    pub fn new() -> Self {
        Self(GpuFft::new().unwrap())
    }
}

impl FftExecutor for StockhamBaseline {
    fn name(&self) -> &str {
        "Baseline (Stockham Radix-2)"
    }
    fn fft(&self, inputs: &[Vec<Complex<f32>>]) -> Result<Vec<Vec<Complex<f32>>>> {
        self.0.fft(inputs)
    }
    fn ifft(&self, inputs: &[Vec<Complex<f32>>]) -> Result<Vec<Vec<Complex<f32>>>> {
        self.0.ifft(inputs)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl crate::GpuFftTrait for StockhamBaseline {
    fn benchmark_gpu_only(
        &self,
        sc: &crate::SizeCache,
        batch_size: u32,
        n: usize,
        warmup_iters: usize,
        bench_iters: usize,
    ) -> Result<f64> {
        self.0
            .benchmark_gpu_only(sc, batch_size, n, warmup_iters, bench_iters)
    }

    fn get_or_build_size_cache(&self, n: usize, log_n: u32) -> crate::SizeCache {
        self.0.get_or_build_size_cache(n, log_n)
    }

    fn prepare_input_data(&self, input: &[Complex<f32>], inverse: bool) -> Vec<f32> {
        self.0.prepare_input_data(input, inverse)
    }

    fn queue(&self) -> &wgpu::Queue {
        self.0.queue()
    }
}
