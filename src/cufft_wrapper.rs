use std::any::Any;
use std::sync::Arc;

use cudarc::cufft::{sys, CudaFft, FftDirection};
use cudarc::driver::{CudaContext, CudaStream};
use num_complex::Complex;

use crate::error::{FftError, Result};
use crate::FftExecutor;

/// Wrapper for cuFFT (via cudarc) to compare performance with our WebGPU implementation.
pub struct CuFft {
    plan: CudaFft,
    stream: Arc<CudaStream>,
    fft_size: usize,
}

impl FftExecutor for CuFft {
    fn name(&self) -> &str {
        "cuFFT (NVIDIA Gold Standard)"
    }

    fn fft(&self, inputs: &[Vec<Complex<f32>>]) -> Result<Vec<Vec<Complex<f32>>>> {
        self.batch_fft(inputs)
    }

    fn ifft(&self, inputs: &[Vec<Complex<f32>>]) -> Result<Vec<Vec<Complex<f32>>>> {
        self.batch_ifft(inputs)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// Note: cuFFT uses a completely different architecture (CUDA vs wgpu)
// so it needs a specialized GPU-only benchmark implementation
// For now, cuFFT will use the full benchmarking path

impl CuFft {
    /// Create a new cuFFT instance.
    pub fn new(fft_size: usize) -> Result<Self> {
        if fft_size == 0 {
            return Err(FftError::InvalidSize("fft_size must be > 0".to_string()));
        }

        let ctx = CudaContext::new(0).map_err(|e| FftError::BackendError(e.to_string()))?;
        let stream = ctx.default_stream();
        let single_plan = CudaFft::plan_1d(
            fft_size as i32,
            sys::cufftType::CUFFT_C2C,
            1,
            stream.clone(),
        )
        .map_err(|e| FftError::BackendError(e.to_string()))?;

        Ok(Self {
            plan: single_plan,
            stream,
            fft_size,
        })
    }

    fn ensure_input_size(&self, input: &[Complex<f32>]) -> Result<()> {
        if input.len() != self.fft_size {
            return Err(FftError::ValidationError(format!(
                "input length {} does not match planned fft_size {}",
                input.len(),
                self.fft_size
            )));
        }
        Ok(())
    }

    fn validate_batch_inputs(inputs: &[Vec<Complex<f32>>]) -> Result<usize> {
        if inputs.is_empty() {
            return Ok(0);
        }

        let n = inputs[0].len();
        if n == 0 {
            return Err(FftError::ValidationError(
                "batch input vectors must be non-empty".to_string(),
            ));
        }

        for (idx, input) in inputs.iter().enumerate() {
            if input.len() != n {
                return Err(FftError::BatchError(format!(
                    "all batch inputs must have the same length; input[0]={n}, input[{idx}]={}",
                    input.len()
                )));
            }
        }

        Ok(n)
    }

    fn to_cuda_complex(values: &[Complex<f32>]) -> Vec<sys::float2> {
        values
            .iter()
            .map(|c| sys::float2 { x: c.re, y: c.im })
            .collect()
    }

    fn from_cuda_complex(values: &[sys::float2]) -> Vec<Complex<f32>> {
        values.iter().map(|c| Complex::new(c.x, c.y)).collect()
    }

    /// Convert from Complex<f32> vectors to flattened sys::float2 format
    pub fn to_flattened_input(inputs: &[Vec<Complex<f32>>]) -> Vec<sys::float2> {
        inputs
            .iter()
            .flat_map(|v| v.iter().map(|c| sys::float2 { x: c.re, y: c.im }))
            .collect()
    }

    /// Convert from flattened sys::float2 format to Complex<f32> vectors
    pub fn from_flattened_output(
        flat_output: Vec<sys::float2>,
        n: usize,
    ) -> Vec<Vec<Complex<f32>>> {
        flat_output
            .chunks(n)
            .map(|chunk| chunk.iter().map(|c| Complex::new(c.x, c.y)).collect())
            .collect()
    }

    fn scale_inverse(output: &mut [Complex<f32>], n: usize) {
        let scale = 1.0 / n as f32;
        for c in output {
            c.re *= scale;
            c.im *= scale;
        }
    }

    /// Perform forward FFT.
    pub fn fft(&self, input: &[Complex<f32>]) -> Result<Vec<Complex<f32>>> {
        self.ensure_input_size(input)?;

        let host_input = Self::to_cuda_complex(input);
        let mut d_input = self
            .stream
            .clone_htod(&host_input)
            .map_err(|e| FftError::BackendError(e.to_string()))?;
        let mut d_output = self
            .stream
            .alloc_zeros::<sys::float2>(self.fft_size)
            .map_err(|e| FftError::BackendError(e.to_string()))?;

        self.plan
            .exec_c2c(&mut d_input, &mut d_output, FftDirection::Forward)
            .map_err(|e| FftError::BackendError(e.to_string()))?;

        let host_output: Vec<sys::float2> = self
            .stream
            .clone_dtoh(&d_output)
            .map_err(|e| FftError::BackendError(e.to_string()))?;
        Ok(Self::from_cuda_complex(&host_output))
    }

    /// Perform inverse FFT.
    pub fn ifft(&self, input: &[Complex<f32>]) -> Result<Vec<Complex<f32>>> {
        self.ensure_input_size(input)?;

        let host_input = Self::to_cuda_complex(input);
        let mut d_input = self
            .stream
            .clone_htod(&host_input)
            .map_err(|e| FftError::BackendError(e.to_string()))?;
        let mut d_output = self
            .stream
            .alloc_zeros::<sys::float2>(self.fft_size)
            .map_err(|e| FftError::BackendError(e.to_string()))?;

        self.plan
            .exec_c2c(&mut d_input, &mut d_output, FftDirection::Inverse)
            .map_err(|e| FftError::BackendError(e.to_string()))?;

        let host_output: Vec<sys::float2> = self
            .stream
            .clone_dtoh(&d_output)
            .map_err(|e| FftError::BackendError(e.to_string()))?;
        let mut result = Self::from_cuda_complex(&host_output);
        Self::scale_inverse(&mut result, self.fft_size);
        Ok(result)
    }

    /// Perform batch FFT on pre-flattened input (optimized for benchmarking).
    /// Input: flattened vector of complex pairs [re0, im0, re1, im1, ...]
    /// Output: flattened vector of complex pairs
    pub fn batch_fft_flattened(
        &self,
        flat_input: &[sys::float2],
        n: usize,
        batch_size: usize,
    ) -> Result<Vec<sys::float2>> {
        if flat_input.is_empty() {
            return Ok(Vec::new());
        }

        let mut d_input = self
            .stream
            .clone_htod(flat_input)
            .map_err(|e| FftError::BackendError(e.to_string()))?;
        let mut d_output = self
            .stream
            .alloc_zeros::<sys::float2>(n * batch_size)
            .map_err(|e| FftError::BackendError(e.to_string()))?;

        self.plan
            .exec_c2c(&mut d_input, &mut d_output, FftDirection::Forward)
            .map_err(|e| FftError::BackendError(e.to_string()))?;

        let flat_output: Vec<sys::float2> = self
            .stream
            .clone_dtoh(&d_output)
            .map_err(|e| FftError::BackendError(e.to_string()))?;
        Ok(flat_output)
    }

    /// Perform batch FFT.
    pub fn batch_fft(&self, inputs: &[Vec<Complex<f32>>]) -> Result<Vec<Vec<Complex<f32>>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let n = Self::validate_batch_inputs(inputs)?;
        let batch_size = inputs.len();

        let flat_input: Vec<sys::float2> = inputs
            .iter()
            .flat_map(|v| v.iter().map(|c| sys::float2 { x: c.re, y: c.im }))
            .collect();

        let flat_output = self.batch_fft_flattened(&flat_input, n, batch_size)?;
        let result = flat_output.chunks(n).map(Self::from_cuda_complex).collect();

        Ok(result)
    }

    /// Perform batch IFFT on pre-flattened input (optimized for benchmarking).
    /// Input: flattened vector of complex pairs [re0, im0, re1, im1, ...]
    /// Output: flattened vector of complex pairs (scaled by 1/N)
    pub fn batch_ifft_flattened(
        &self,
        flat_input: &[sys::float2],
        n: usize,
        batch_size: usize,
    ) -> Result<Vec<sys::float2>> {
        if flat_input.is_empty() {
            return Ok(Vec::new());
        }

        let mut d_input = self
            .stream
            .clone_htod(flat_input)
            .map_err(|e| FftError::BackendError(e.to_string()))?;
        let mut d_output = self
            .stream
            .alloc_zeros::<sys::float2>(n * batch_size)
            .map_err(|e| FftError::BackendError(e.to_string()))?;

        self.plan
            .exec_c2c(&mut d_input, &mut d_output, FftDirection::Inverse)
            .map_err(|e| FftError::BackendError(e.to_string()))?;

        let mut flat_output: Vec<sys::float2> = self
            .stream
            .clone_dtoh(&d_output)
            .map_err(|e| FftError::BackendError(e.to_string()))?;

        // Apply scaling for inverse transform
        let scale = 1.0 / n as f32;
        for c in &mut flat_output {
            c.x *= scale;
            c.y *= scale;
        }

        Ok(flat_output)
    }

    fn batch_ifft(&self, inputs: &[Vec<Complex<f32>>]) -> Result<Vec<Vec<Complex<f32>>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let n = Self::validate_batch_inputs(inputs)?;
        let batch_size = inputs.len();

        let flat_input: Vec<sys::float2> = inputs
            .iter()
            .flat_map(|v| v.iter().map(|c| sys::float2 { x: c.re, y: c.im }))
            .collect();

        let flat_output = self.batch_ifft_flattened(&flat_input, n, batch_size)?;
        let mut result: Vec<Vec<Complex<f32>>> =
            flat_output.chunks(n).map(Self::from_cuda_complex).collect();

        Ok(result)
    }

    /// Check if CUDA is available.
    pub fn is_available() -> bool {
        Self::new(16).is_ok()
    }
}
