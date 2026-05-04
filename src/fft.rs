//! GPU-accelerated FFT implementation using wgpu compute shaders.
//!
//! Implements the **Stockham autosort** Radix-4/2 FFT — a two-buffer ping-pong formulation
//! where each stage reads from one buffer and writes to the other. This eliminates the separate
//! bit-reversal pass and removes all inter-stage memory hazards.
//!
//! Also implements **Bluestein's algorithm** for arbitrary FFT sizes (not just powers of 2).

use std::any::Any;
use std::cell::RefCell;
use std::num::NonZeroU64;

use num_complex::Complex;

use crate::error::{FftError, Result};
use crate::shaders;

/// Number of components in a complex number (real and imaginary)
const COMPLEX_COMPONENT_COUNT: usize = 2;

/// Byte size of an f32
const F32_BYTE_SIZE: usize = std::mem::size_of::<f32>();

/// Trait for FFT implementations that can be benchmarked.
pub trait FftExecutor {
    fn name(&self) -> &str;
    fn fft(&self, inputs: &[Vec<Complex<f32>>]) -> Result<Vec<Vec<Complex<f32>>>>;
    fn ifft(&self, inputs: &[Vec<Complex<f32>>]) -> Result<Vec<Vec<Complex<f32>>>>;

    /// Get a reference to the underlying type for downcasting.
    fn as_any(&self) -> &dyn Any;
}

/// Trait for GPU FFT implementations that support GPU-only benchmarking.
pub trait GpuFftTrait {
    /// Benchmark only the GPU compute pass and DMA operations (isolated from CPU overhead).
    /// Returns duration in seconds for the GPU operations only.
    fn benchmark_gpu_only(
        &self,
        sc: &SizeCache,
        batch_size: u32,
        n: usize,
        warmup_iters: usize,
        bench_iters: usize,
    ) -> Result<f64>;

    /// Get or build size-specific GPU resources.
    fn get_or_build_size_cache(&self, n: usize, log_n: u32) -> SizeCache;

    /// Prepare input data for GPU processing, applying conjugation for IFFT if needed.
    fn prepare_input_data(&self, input: &[Complex<f32>], inverse: bool) -> Vec<f32>;

    /// Get the queue for GPU operations.
    fn queue(&self) -> &wgpu::Queue;
}

/// Pre-allocated GPU resources for a specific FFT size.
#[derive(Clone, Debug)]
pub struct SizeCache {
    pub buf_a: wgpu::Buffer,
    pub buf_b: wgpu::Buffer,
    pub staging_buf: wgpu::Buffer,
    pub twiddle_buf: wgpu::Buffer,
    pub data_bytes: u64,
    /// R4 stages (R4 mode) or R2 stages (legacy with_shader mode).
    pub stage_bgs: Vec<wgpu::BindGroup>,
    /// Final R2 stage when log₂N is odd (R4 mode only).
    pub stage_bg_r2: Option<wgpu::BindGroup>,
    pub result_in_b: bool,
    /// Workgroup count for the main-stage dispatch (N/4 in R4 mode, N/2 in legacy mode).
    pub wg_n2: u32,
    /// Workgroup count for R4 dispatch (N/4). 0 in legacy mode.
    pub wg_r4: u32,
}

/// Uniforms passed to the compute shader (16-byte aligned).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FftUniforms {
    pub n: u32,
    pub stage: u32,
    pub log_n: u32,
    pub _pad: u32,
}

/// GPU-accelerated FFT engine backed by wgpu compute shaders.
///
/// Implements the Stockham autosort Radix-4 algorithm with an optional Radix-2
/// final stage for odd log₂N sizes. Use [`GpuFft::new`] for the default R4
/// pipeline or [`GpuFft::with_shader`] to supply a custom WGSL kernel.
///
/// For arbitrary FFT sizes (not powers of 2), Bluestein's algorithm is used automatically.
#[derive(Debug)]
pub struct GpuFft {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipeline: wgpu::ComputePipeline,
    /// Present only when created via `new()` (R4 mode). `None` in legacy `with_shader` mode.
    pub pipeline_r2: Option<wgpu::ComputePipeline>,
    pub cache: RefCell<std::collections::HashMap<usize, SizeCache>>,
}

impl FftExecutor for GpuFft {
    fn name(&self) -> &str {
        "Baseline (Stockham Radix-4/2)"
    }

    fn fft(&self, inputs: &[Vec<Complex<f32>>]) -> Result<Vec<Vec<Complex<f32>>>> {
        self.transform_batch_internal(inputs, false)
    }

    fn ifft(&self, inputs: &[Vec<Complex<f32>>]) -> Result<Vec<Vec<Complex<f32>>>> {
        self.transform_batch_internal(inputs, true)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl GpuFftTrait for GpuFft {
    fn benchmark_gpu_only(
        &self,
        sc: &SizeCache,
        batch_size: u32,
        n: usize,
        warmup_iters: usize,
        bench_iters: usize,
    ) -> Result<f64> {
        use std::time::Instant;

        // Warmup
        for _ in 0..warmup_iters {
            self.execute_compute_pass(sc, batch_size, n);
            self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })?;
        }

        // Benchmark
        let start = Instant::now();
        for _ in 0..bench_iters {
            self.execute_compute_pass(sc, batch_size, n);
        }

        self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })?;

        let duration = start.elapsed();
        Ok(duration.as_secs_f64() / bench_iters as f64)
    }

    fn get_or_build_size_cache(&self, n: usize, log_n: u32) -> SizeCache {
        self.get_or_build_size_cache(n, log_n)
    }

    fn prepare_input_data(&self, input: &[Complex<f32>], inverse: bool) -> Vec<f32> {
        self.prepare_input_data(input, inverse)
    }

    fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

impl GpuFft {
    /// Access the underlying wgpu device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Access the compiled compute pipeline.
    pub fn compute_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }

    /// Create a new [`GpuFft`] using the Radix-4/2 Stockham baseline.
    ///
    /// Dispatches ⌊log₄N⌋ Radix-4 passes (+ one Radix-2 pass when log₂N is odd),
    /// halving the pass count vs the old Radix-2 baseline.
    ///
    /// For arbitrary FFT sizes (not powers of 2), Bluestein's algorithm is used automatically.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use wgsl_fft::GpuFft;
    ///
    /// let fft = GpuFft::new().expect("GPU required");
    /// // Now use fft.fft() and fft.ifft()
    /// ```
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .or_else(|_| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: true,
            }))
        })?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            ..Default::default()
        }))?;
        Self::from_device_queue(device, queue)
    }

    /// Create a new [`GpuFft`] using the Radix-4/2 Stockham baseline with an existing device and queue.
    ///
    /// This constructor allows you to provide your own wgpu device and queue, which is useful
    /// when you want to share a single GPU context across multiple resources.
    ///
    /// # Arguments
    ///
    /// * `device` - A wgpu device to use for creating resources.
    /// * `queue` - A wgpu queue to use for submitting commands.
    pub fn from_device_queue(device: wgpu::Device, queue: wgpu::Queue) -> Result<Self> {
        let compile = |src: &str, label: &str| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{label}_pipeline")),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let pipeline = compile(shaders::R4_WGSL, "stockham_r4");
        let pipeline_r2 = Some(compile(shaders::R2_WGSL, "stockham_r2"));

        Ok(Self {
            device,
            queue,
            pipeline,
            pipeline_r2,
            cache: RefCell::new(std::collections::HashMap::new()),
        })
    }

    /// Create a new [`GpuFft`] with a custom WGSL shader.
    /// This allows AI rivals to swap kernels easily.
    pub fn with_shader(wgsl_source: String, label: &str) -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .or_else(|_| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: true,
            }))
        })?;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            ..Default::default()
        }))?;
        Self::with_shader_and_device(device, queue, wgsl_source, label)
    }

    /// Create a new [`GpuFft`] with a custom WGSL shader using an existing device and queue.
    ///
    /// This allows AI rivals to swap kernels easily while sharing a GPU context.
    ///
    /// # Arguments
    ///
    /// * `device` - A wgpu device to use for creating resources.
    /// * `queue` - A wgpu queue to use for submitting commands.
    /// * `wgsl_source` - The WGSL shader source code.
    /// * `label` - A label for the shader and pipeline.
    pub fn with_shader_and_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
        wgsl_source: String,
        label: &str,
    ) -> Result<Self> {
        let shader_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{}_pipeline", label)),
            layout: None,
            module: &shader_mod,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            pipeline_r2: None, // legacy single-pipeline mode
            cache: RefCell::new(std::collections::HashMap::new()),
        })
    }

    /// Check if a GPU is available without creating an instance.
    pub fn is_gpu_available() -> bool {
        let instance = wgpu::Instance::default();
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .is_ok()
    }

    /// Compute the forward FFT for a batch of input vectors.
    ///
    /// Processes multiple FFTs efficiently. For single vector processing,
    /// pass a vector containing one input vector.
    /// All input vectors must have the same length.
    ///
    /// For power-of-two sizes, uses the fast Stockham Radix-4/2 algorithm.
    /// For arbitrary sizes, uses Bluestein's algorithm.
    ///
    /// # Arguments
    ///
    /// * `inputs` - A vector of input vectors, each containing complex samples.
    ///
    /// # Returns
    ///
    /// A vector of FFT results, one for each input vector.
    ///
    /// # Panics
    ///
    /// Panics if any input vector is empty or has a different length than others.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU operation fails (buffer mapping, device lost, etc.).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use wgsl_fft::GpuFft;
    /// use num_complex::Complex;
    ///
    /// let fft = GpuFft::new().expect("GPU or CPU fallback required");
    ///
    /// // Single FFT (pass vector with one element)
    /// let single_input = vec![vec![Complex::new(1.0, 0.0); 1024]];
    /// let single_spectrum = fft.fft(&single_input).expect("FFT failed");
    ///
    /// // Batch FFT
    /// let batch_inputs = vec![
    ///     vec![Complex::new(1.0, 0.0); 1024],
    ///     vec![Complex::new(0.5, 0.0); 1024],
    /// ];
    /// let batch_spectra = fft.fft(&batch_inputs).expect("Batch FFT failed");
    ///
    /// // Arbitrary size FFT (not power of two)
    /// let arbitrary_input = vec![vec![Complex::new(1.0, 0.0); 150]];
    /// let arbitrary_spectrum = fft.fft(&arbitrary_input).expect("Arbitrary size FFT failed");
    /// ```
    pub fn fft(&self, inputs: &[Vec<Complex<f32>>]) -> Result<Vec<Vec<Complex<f32>>>> {
        self.transform_batch_internal(inputs, false)
    }

    /// Compute the inverse FFT for a batch of input vectors.
    ///
    /// Processes multiple IFFTs efficiently. For single vector processing,
    /// pass a vector containing one input vector.
    /// All input vectors must have the same length.
    /// The output is automatically scaled by `1/N` to maintain the unitary transform property.
    ///
    /// For power-of-two sizes, uses the fast Stockham Radix-4/2 algorithm.
    /// For arbitrary sizes, uses Bluestein's algorithm.
    ///
    /// # Arguments
    ///
    /// * `inputs` - A vector of input vectors, each containing complex samples.
    ///
    /// # Returns
    ///
    /// A vector of IFFT results, one for each input vector.
    ///
    /// # Panics
    ///
    /// Panics if any input vector is empty or has a different length than others.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU operation fails (buffer mapping, device lost, etc.).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use wgsl_fft::GpuFft;
    /// use num_complex::Complex;
    ///
    /// let fft = GpuFft::new().expect("GPU or CPU fallback required");
    ///
    /// // Single IFFT (pass vector with one element)
    /// let single_spectrum = vec![vec![Complex::new(1.0, 0.0); 1024]];
    /// let single_reconstructed = fft.ifft(&single_spectrum).expect("IFFT failed");
    ///
    /// // Batch IFFT
    /// let batch_spectra = vec![
    ///     vec![Complex::new(1.0, 0.0); 1024],
    ///     vec![Complex::new(0.5, 0.0); 1024],
    /// ];
    /// let batch_reconstructed = fft.ifft(&batch_spectra).expect("Batch IFFT failed");
    ///
    /// // Arbitrary size IFFT (not power of two)
    /// let arbitrary_spectrum = vec![vec![Complex::new(1.0, 0.0); 150]];
    /// let arbitrary_reconstructed = fft.ifft(&arbitrary_spectrum).expect("Arbitrary size IFFT failed");
    /// ```
    pub fn ifft(&self, inputs: &[Vec<Complex<f32>>]) -> Result<Vec<Vec<Complex<f32>>>> {
        self.transform_batch_internal(inputs, true)
    }

    /// Validate that the input size is non-zero.
    /// Arbitrary sizes are now supported via Bluestein's algorithm.
    pub fn validate_input_size(&self, n: usize) -> Result<()> {
        if n == 0 {
            return Err(FftError::ValidationError(
                "Transform length must be non-zero".to_string(),
            ));
        }
        Ok(())
    }

    /// Check if a size is a power of two.
    pub fn is_power_of_two(n: usize) -> bool {
        n > 0 && (n & (n - 1)) == 0
    }

    /// Internal batch transform implementation that handles both FFT and IFFT for multiple inputs.
    ///
    /// When `inverse` is true, computes IFFT (with conjugation and 1/N scaling).
    /// When `inverse` is false, computes standard FFT.
    ///
    /// For power-of-two sizes, uses the Stockham Radix-4/2 algorithm.
    /// For arbitrary sizes, uses Bluestein's algorithm.
    pub fn transform_batch_internal(
        &self,
        inputs: &[Vec<Complex<f32>>],
        inverse: bool,
    ) -> Result<Vec<Vec<Complex<f32>>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        self.validate_batch_inputs(inputs)?;

        let n = inputs[0].len();
        let batch_size = inputs.len() as u32;

        if Self::is_power_of_two(n) {
            return self.transform_power_of_two(inputs, inverse, n, batch_size);
        }

        self.transform_batch_bluestein(inputs, inverse)
    }

    /// Validate that all inputs in a batch have the same size.
    fn validate_batch_inputs(&self, inputs: &[Vec<Complex<f32>>]) -> Result<()> {
        let n = inputs[0].len();

        for input in inputs {
            if input.len() != n {
                return Err(FftError::BatchError(
                    "All input vectors in a batch must have the same length".to_string(),
                ));
            }
            self.validate_input_size(input.len())?;
        }

        Ok(())
    }

    /// Transform batch for power-of-two sizes using Stockham Radix-4/2.
    fn transform_power_of_two(
        &self,
        inputs: &[Vec<Complex<f32>>],
        inverse: bool,
        n: usize,
        batch_size: u32,
    ) -> Result<Vec<Vec<Complex<f32>>>> {
        let log_n = n.trailing_zeros();
        let sc = self.get_or_build_size_cache(n, log_n);

        let all_raw_data = self.prepare_batch_input_data(inputs, inverse);

        self.upload_batch_data(&sc, &all_raw_data);
        self.execute_compute_pass(&sc, batch_size, n);

        let mut output = self.readback_results(&sc, batch_size, n)?;

        if inverse {
            self.apply_inverse_postprocessing(&mut output, n);
        }

        Ok(self.split_results(output, n))
    }

    /// Prepare input data for all inputs in a batch.
    fn prepare_batch_input_data(
        &self,
        inputs: &[Vec<Complex<f32>>],
        inverse: bool,
    ) -> Vec<f32> {
        let batch_size = inputs.len();
        let n = inputs[0].len();

        let mut all_raw_data = Vec::with_capacity(n * COMPLEX_COMPONENT_COUNT * batch_size);

        for input in inputs {
            let raw = self.prepare_input_data(input, inverse);
            all_raw_data.extend_from_slice(&raw);
        }

        all_raw_data
    }

    /// Upload batch data to GPU buffer.
    fn upload_batch_data(&self, sc: &SizeCache, data: &[f32]) {
        self.queue.write_buffer(&sc.buf_a, 0, bytemuck::cast_slice(data));
    }

    /// Apply inverse transform postprocessing to all chunks.
    fn apply_inverse_postprocessing(&self, output: &mut [Complex<f32>], n: usize) {
        for chunk in output.chunks_mut(n) {
            self.apply_inverse_transform_postprocessing(chunk, n);
        }
    }

    /// Split output into individual results.
    fn split_results(&self, output: Vec<Complex<f32>>, n: usize) -> Vec<Vec<Complex<f32>>> {
        output.chunks(n).map(|chunk| chunk.to_vec()).collect()
    }

    /// Get the result buffer based on whether result is in buffer B.
    fn get_result_buffer<'a>(&self, sc: &'a SizeCache) -> &'a wgpu::Buffer {
        if sc.result_in_b {
            return &sc.buf_b;
        }
        &sc.buf_a
    }

    /// Calculate number of R4 stages.
    fn calculate_num_r4_stages(&self, is_r4_mode: bool, log_n: u32) -> usize {
        if is_r4_mode {
            return (log_n / 2) as usize;
        }
        0
    }

    /// Calculate total number of stages.
    fn calculate_total_stages(
        &self,
        is_r4_mode: bool,
        num_r4: usize,
        has_r2: bool,
        log_n: u32,
    ) -> usize {
        if is_r4_mode {
            return num_r4 + has_r2 as usize;
        }
        log_n as usize
    }

    /// Calculate twiddle table count.
    fn calculate_twiddle_count(&self, is_r4_mode: bool, n: usize) -> usize {
        if is_r4_mode {
            return n;
        }
        n / 2
    }

    /// Transform using Bluestein's algorithm for arbitrary FFT sizes.
    ///
    /// For non-power-of-two sizes, we use a CPU-based implementation via rustfft.
    /// This provides correctness for arbitrary sizes while maintaining GPU acceleration
    /// for power-of-two sizes.
    ///
    /// Note: In the current implementation, arbitrary sizes are computed on the CPU
    /// using rustfft as a fallback. Future optimizations could implement a GPU-accelerated
    /// Bluestein's algorithm for full GPU support of arbitrary sizes.
    ///
    /// The implementation matches the GPU FFT convention:
    /// - Forward FFT: X[k] = Σ_n x[n] * exp(-2πi * k * n / N)
    /// - Inverse FFT: x[n] = (1/N) * Σ_k X[k] * exp(2πi * k * n / N)
    fn transform_batch_bluestein(
        &self,
        inputs: &[Vec<Complex<f32>>],
        inverse: bool,
    ) -> Result<Vec<Vec<Complex<f32>>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let n = inputs[0].len();
        let batch_size = inputs.len();

        let mut planner = rustfft::FftPlanner::<f32>::new();
        let mut results = Vec::with_capacity(batch_size);

        for input in inputs {
            results.push(self.process_bluestein_single(&mut planner, input, n, inverse));
        }

        Ok(results)
    }

    /// Process a single input using Bluestein's algorithm.
    fn process_bluestein_single(
        &self,
        planner: &mut rustfft::FftPlanner<f32>,
        input: &[Complex<f32>],
        n: usize,
        inverse: bool,
    ) -> Vec<Complex<f32>> {
        if inverse {
            return self.process_bluestein_inverse(planner, input, n);
        }

        let mut result = input.to_vec();
        let fft = planner.plan_fft_forward(n);
        fft.process(&mut result);
        result
    }

    /// Process inverse FFT using Bluestein's algorithm with scaling.
    fn process_bluestein_inverse(
        &self,
        planner: &mut rustfft::FftPlanner<f32>,
        input: &[Complex<f32>],
        n: usize,
    ) -> Vec<Complex<f32>> {
        let mut result = input.to_vec();

        let ifft = planner.plan_fft_inverse(n);
        ifft.process(&mut result);

        self.apply_bluestein_scaling(&mut result, n);
        result
    }

    /// Apply 1/N scaling to match GPU implementation.
    fn apply_bluestein_scaling(&self, result: &mut [Complex<f32>], n: usize) {
        let scale = 1.0 / n as f32;

        for x in result {
            *x = Complex::new(x.re * scale, x.im * scale);
        }
    }

    /// Prepare input data for GPU processing, applying conjugation for IFFT if needed.
    pub fn prepare_input_data(&self, input: &[Complex<f32>], inverse: bool) -> Vec<f32> {
        if inverse {
            return input.iter().flat_map(|c| [c.re, -c.im]).collect();
        }
        input.iter().flat_map(|c| [c.re, c.im]).collect()
    }

    /// Execute the compute shader pass.
    pub fn execute_compute_pass(&self, sc: &SizeCache, batch_size: u32, n: usize) {
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FFT Pass"),
        });

        self.run_compute_pass(&mut enc, sc, batch_size);

        let result_buf = self.get_result_buffer(sc);
        let single_fft_bytes = (n * COMPLEX_COMPONENT_COUNT * F32_BYTE_SIZE) as u64;

        enc.copy_buffer_to_buffer(result_buf, 0, &sc.staging_buf, 0, single_fft_bytes * batch_size as u64);

        self.queue.submit(std::iter::once(enc.finish()));
    }

    /// Run compute pass on encoder.
    fn run_compute_pass(&self, enc: &mut wgpu::CommandEncoder, sc: &SizeCache, batch_size: u32) {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FFT Compute"),
            timestamp_writes: None,
        });

        if sc.wg_r4 > 0 {
            self.dispatch_r4_mode_pass(&mut pass, sc, batch_size);
            return;
        }

        self.dispatch_legacy_mode_pass(&mut pass, sc, batch_size);
    }

    /// Dispatch R4 mode compute pass.
    fn dispatch_r4_mode_pass(&self, pass: &mut wgpu::ComputePass, sc: &SizeCache, batch_size: u32) {
        pass.set_pipeline(&self.pipeline);

        for bg in &sc.stage_bgs {
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(sc.wg_r4, batch_size, 1);
        }

        if let Some(r2_bg) = &sc.stage_bg_r2 {
            self.dispatch_r2_stage_pass(pass, r2_bg, sc, batch_size);
        }
    }

    /// Dispatch R2 stage pass.
    fn dispatch_r2_stage_pass(
        &self,
        pass: &mut wgpu::ComputePass,
        r2_bg: &wgpu::BindGroup,
        sc: &SizeCache,
        batch_size: u32,
    ) {
        pass.set_pipeline(self.pipeline_r2.as_ref().unwrap());
        pass.set_bind_group(0, r2_bg, &[]);
        pass.dispatch_workgroups(sc.wg_n2, batch_size, 1);
    }

    /// Dispatch legacy mode compute pass.
    fn dispatch_legacy_mode_pass(&self, pass: &mut wgpu::ComputePass, sc: &SizeCache, batch_size: u32) {
        pass.set_pipeline(&self.pipeline);

        for bg in &sc.stage_bgs {
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(sc.wg_n2, batch_size, 1);
        }
    }

    /// Read back results from GPU and convert to complex numbers.
    pub fn readback_results(
        &self,
        sc: &SizeCache,
        batch_size: u32,
        n: usize,
    ) -> Result<Vec<Complex<f32>>> {
        // Readback
        let single_fft_bytes = (n * COMPLEX_COMPONENT_COUNT * F32_BYTE_SIZE) as u64;
        let total_bytes = single_fft_bytes * batch_size as u64;
        let slice = sc.staging_buf.slice(0..total_bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })?;

        let mapped = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&mapped);
        let output: Vec<Complex<f32>> = floats
            .chunks_exact(2)
            .map(|p| Complex { re: p[0], im: p[1] })
            .collect();

        drop(mapped);
        sc.staging_buf.unmap();

        Ok(output)
    }

    /// Apply postprocessing for inverse transform (conjugation and 1/N scaling).
    pub fn apply_inverse_transform_postprocessing(&self, output: &mut [Complex<f32>], n: usize) {
        let scale = 1.0 / n as f32;
        for c in output {
            *c = Complex {
                re: c.re * scale,
                im: -c.im * scale,
            };
        }
    }

    /// Get or build size-specific GPU resources.
    pub fn get_or_build_size_cache(&self, n: usize, log_n: u32) -> SizeCache {
        let mut cache = self.cache.borrow_mut();
        if let Some(sc) = cache.get(&n) {
            return sc.clone();
        }

        let sc = self.build_size_cache(n, log_n);
        cache.insert(n, sc.clone());
        sc
    }

    /// Build GPU buffers and bind groups for a specific FFT size.
    pub fn build_size_cache(&self, n: usize, log_n: u32) -> SizeCache {
        let is_r4_mode = self.pipeline_r2.is_some();

        let num_r4 = self.calculate_num_r4_stages(is_r4_mode, log_n);
        let has_r2 = is_r4_mode && log_n % 2 == 1;
        let total_stages = self.calculate_total_stages(is_r4_mode, num_r4, has_r2, log_n);

        let single_fft_bytes = n as u64 * 2 * std::mem::size_of::<f32>() as u64;
        // Cap at 1024 to avoid excessive pre-allocation; hardware limits are often much larger.
        let max_batch_size = (self.device.limits().max_storage_buffer_binding_size
            / single_fft_bytes)
            .min(1024) as u32;
        let data_bytes = single_fft_bytes * max_batch_size as u64;

        let make_buf = |label| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: data_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let buf_a = make_buf("fft_buf_a");
        let buf_b = make_buf("fft_buf_b");
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft_staging"),
            size: data_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Twiddle table: N entries for R4 mode (max accessed index = 3N/2−5 < 2N),
        // N/2 entries for legacy R2 mode (max accessed index = N−2 < N).
        let twiddle_count = self.calculate_twiddle_count(is_r4_mode, n);
        let twiddles: Vec<f32> = (0..twiddle_count)
            .flat_map(|j| {
                let angle = -std::f32::consts::TAU * j as f32 / n as f32;
                [angle.cos(), angle.sin()]
            })
            .collect();
        let twiddle_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft_twiddles"),
            size: (twiddles.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&twiddle_buf, 0, bytemuck::cast_slice(&twiddles));

        let alignment = self.device.limits().min_uniform_buffer_offset_alignment as u64;
        let entry_bytes = std::mem::size_of::<FftUniforms>() as u64;
        let stride = entry_bytes.div_ceil(alignment) * alignment;

        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft_uniforms"),
            size: stride * total_stages.max(1) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_size = NonZeroU64::new(entry_bytes);
        let layout_r4 = self.pipeline.get_bind_group_layout(0);
        let layout_r2_opt = self
            .pipeline_r2
            .as_ref()
            .map(|p| p.get_bind_group_layout(0));

        let make_bg_with_layout = |layout: &wgpu::BindGroupLayout,
                                   src: &wgpu::Buffer,
                                   dst: &wgpu::Buffer,
                                   uniform_offset: u64| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &uniform_buf,
                            offset: uniform_offset,
                            size: uniform_size,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: dst.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: twiddle_buf.as_entire_binding(),
                    },
                ],
            })
        };

        let make_bg = |src: &wgpu::Buffer, dst: &wgpu::Buffer, uniform_offset: u64| {
            make_bg_with_layout(&layout_r4, src, dst, uniform_offset)
        };

        if is_r4_mode {
            // R4 mode: ⌊log₄N⌋ Radix-4 stages + optional Radix-2
            for s in 0..num_r4 {
                let p = 1u32 << (s as u32 * 2);
                self.queue.write_buffer(
                    &uniform_buf,
                    stride * s as u64,
                    bytemuck::bytes_of(&FftUniforms {
                        n: n as u32,
                        stage: p,
                        log_n,
                        _pad: 0,
                    }),
                );
            }
            if has_r2 {
                let p = 1u32 << (num_r4 as u32 * 2);
                self.queue.write_buffer(
                    &uniform_buf,
                    stride * num_r4 as u64,
                    bytemuck::bytes_of(&FftUniforms {
                        n: n as u32,
                        stage: p,
                        log_n,
                        _pad: 0,
                    }),
                );
            }

            let stage_bgs: Vec<wgpu::BindGroup> = (0..num_r4)
                .map(|s| {
                    let (src, dst) = if s % 2 == 0 {
                        (&buf_a, &buf_b)
                    } else {
                        (&buf_b, &buf_a)
                    };
                    make_bg(src, dst, stride * s as u64)
                })
                .collect();

            let stage_bg_r2 = if has_r2 {
                let (src, dst) = if num_r4 % 2 == 0 {
                    (&buf_a, &buf_b)
                } else {
                    (&buf_b, &buf_a)
                };
                let layout_r2 = layout_r2_opt.as_ref().unwrap();
                Some(make_bg_with_layout(
                    layout_r2,
                    src,
                    dst,
                    stride * num_r4 as u64,
                ))
            } else {
                None
            };

            SizeCache {
                buf_a,
                buf_b,
                staging_buf,
                twiddle_buf,
                data_bytes,
                stage_bgs,
                stage_bg_r2,
                result_in_b: total_stages % 2 == 1,
                wg_n2: (n as u32 / 2).div_ceil(256),
                wg_r4: (n as u32 / 4).div_ceil(256),
            }
        } else {
            // Legacy mode (with_shader): log₂N Radix-2 stages, stage-index uniforms
            for stage in 0..log_n {
                self.queue.write_buffer(
                    &uniform_buf,
                    stride * stage as u64,
                    bytemuck::bytes_of(&FftUniforms {
                        n: n as u32,
                        stage,
                        log_n,
                        _pad: 0,
                    }),
                );
            }

            let stage_bgs = (0..log_n as usize)
                .map(|s| {
                    let (src, dst) = if s % 2 == 0 {
                        (&buf_a, &buf_b)
                    } else {
                        (&buf_b, &buf_a)
                    };
                    make_bg(src, dst, stride * s as u64)
                })
                .collect();

            SizeCache {
                buf_a,
                buf_b,
                staging_buf,
                twiddle_buf,
                data_bytes,
                stage_bgs,
                stage_bg_r2: None,
                result_in_b: log_n % 2 == 1,
                wg_n2: (n as u32 / 2).div_ceil(256),
                wg_r4: 0,
            }
        }
    }
}

impl Default for GpuFft {
    fn default() -> Self {
        Self::new().expect("No GPU available for default GpuFft instance")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_prepare_input_data_fft() {
        let fft = GpuFft::new().expect("Failed to create FFT instance");
        let input = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let result = fft.prepare_input_data(&input, false);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_prepare_input_data_ifft() {
        let fft = GpuFft::new().expect("Failed to create FFT instance");
        let input = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let result = fft.prepare_input_data(&input, true);
        assert_eq!(result, vec![1.0, -2.0, 3.0, -4.0]);
    }

    #[test]
    fn test_apply_inverse_transform_postprocessing() {
        let fft = GpuFft::new().expect("Failed to create FFT instance");
        let mut output = vec![Complex::new(2.0, 4.0), Complex::new(6.0, 8.0)];
        fft.apply_inverse_transform_postprocessing(&mut output, 2);
        assert_eq!(output[0].re, 1.0);
        assert_eq!(output[0].im, -2.0);
        assert_eq!(output[1].re, 3.0);
        assert_eq!(output[1].im, -4.0);
    }
}
