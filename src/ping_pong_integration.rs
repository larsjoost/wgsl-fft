//! Integration with wgsl-ping-pong-pipeline
//!
//! This module provides PipelineStage implementations for wgsl-fft that can be used
//! as custom stages in the wgsl-ping-pong-pipeline.
//!
//! The main types provided are:
//! - [`FftPipelineStage`] - A PipelineStage that performs FFT or IFFT operations
//! - [`MultiplyPipelineStage`] - A PipelineStage that performs element-wise complex multiplication
//!
//! # Example
//!
//! ```ignore
//! use wgsl_ping_pong_pipeline::{Pipeline, StageConfig};
//! use wgsl_fft::ping_pong_integration::{FftPipelineStage, MultiplyPipelineStage};
//!
//! let pipeline = Pipeline::new()
//!     .pipe_config(StageConfig::Custom(
//!         Box::new(FftPipelineStage::forward(1024, 1))
//!     ))
//!     .pipe_config(StageConfig::Custom(
//!         Box::new(MultiplyPipelineStage::new(1024, 1))
//!     ))
//!     .pipe_config(StageConfig::Custom(
//!         Box::new(FftPipelineStage::inverse(1024, 1))
//!     ))
//!     .build()
//!     .await?;
//! ```

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use wgpu::CommandEncoder;

use wgsl_ping_pong_pipeline::pipeline::pipeline_stage::PipelineStage;
use wgsl_ping_pong_pipeline::wgpu_utils::ComputeContext;

use anyhow::{Result, anyhow};

use crate::{FftDirection, FftPipelines};

/// A PipelineStage that performs FFT or IFFT operations using wgsl-fft's FftPipelines.
///
/// This stage wraps the `FftPipelines::encode_fft()` method and handles the GPU resource
/// initialization lazily during the `initialize()` call.
///
/// # Usage
///
/// ```ignore
/// // Forward FFT stage
/// let forward_stage = FftPipelineStage::forward(1024, 1);
///
/// // Inverse FFT stage
/// let inverse_stage = FftPipelineStage::inverse(1024, 1);
/// ```
pub struct FftPipelineStage {
    n: usize,
    batch_size: u32,
    direction: FftDirection,
    /// Lazily initialized FftPipelines (created during initialize())
    fft_pipelines: Option<FftPipelines>,
}

impl FftPipelineStage {
    /// Creates a new forward FFT stage.
    ///
    /// # Arguments
    /// * `n` - FFT size (number of complex elements, must be power of 2)
    /// * `batch_size` - Number of FFTs to process in parallel
    pub fn forward(n: usize, batch_size: u32) -> Self {
        Self {
            n,
            batch_size,
            direction: FftDirection::Forward,
            fft_pipelines: None,
        }
    }

    /// Creates a new inverse FFT stage.
    ///
    /// # Arguments
    /// * `n` - FFT size (number of complex elements, must be power of 2)
    /// * `batch_size` - Number of FFTs to process in parallel
    pub fn inverse(n: usize, batch_size: u32) -> Self {
        Self {
            n,
            batch_size,
            direction: FftDirection::Inverse,
            fft_pipelines: None,
        }
    }
}

impl Debug for FftPipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FftPipelineStage")
            .field("n", &self.n)
            .field("batch_size", &self.batch_size)
            .field("direction", &self.direction)
            .field("fft_pipelines_initialized", &self.fft_pipelines.is_some())
            .finish()
    }
}

impl PipelineStage for FftPipelineStage {
    fn name(&self) -> &str {
        match self.direction {
            FftDirection::Forward => "fft_forward",
            FftDirection::Inverse => "fft_inverse",
        }
    }

    fn vector_dim(&self) -> usize {
        2 // Complex numbers as vec2<f32>
    }

    fn batch_size(&self) -> usize {
        self.n * self.batch_size as usize
    }

    fn encode(
        &self,
        encoder: &mut CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        _side_inputs: &HashMap<String, Arc<wgpu::Buffer>>,
    ) -> Result<()> {
        let fft_pipelines = self
            .fft_pipelines
            .as_ref()
            .expect("FftPipelineStage must be initialized before encode()");

        // Calculate batch_size from output buffer size and our internal n
        // This allows us to handle resized buffers without rebuilding the stage
        let element_size = 2 * std::mem::size_of::<f32>() as u64;
        let total_elements = (output.size() / element_size) as usize;
        let batch_size = (total_elements / self.n) as u32;
        
        // Perform FFT/IFFT
        // Note: encode_fft handles the direction but doesn't normalize
        // Normalization will be applied separately by NormalizePipelineStage
        fft_pipelines.encode_fft(
            encoder,
            self.n,
            batch_size,
            self.direction,
            input,
            output,
        );

        Ok(())
    }

    fn initialize(&mut self, context: &ComputeContext) -> Result<()> {
        let device = context.device.clone();
        let queue = context.queue.clone();
        self.fft_pipelines = Some(FftPipelines::from_device_queue(device, queue));
        Ok(())
    }

    fn requires_initialization(&self) -> bool {
        true
    }
    
    fn supports_dynamic_resizing(&self) -> bool {
        true
    }
    
    fn resize(&mut self, new_batch_size: usize, new_vector_dim: usize) -> Result<()> {
        // For FftPipelineStage, we expect vector_dim to always be 2 (complex numbers)
        if new_vector_dim != 2 {
            anyhow::bail!("FftPipelineStage requires vector_dim of 2 for complex numbers");
        }
        
        // For now, we'll treat new_batch_size as the total size (n * batch_size)
        // and keep our internal n the same, just update batch_size
        // This is a simplified approach - in a full implementation, we'd need to handle
        // both n and batch_size changes more carefully
        let total_elements = new_batch_size;
        if total_elements % self.n != 0 {
            anyhow::bail!("New batch size {} is not a multiple of FFT size {}", total_elements, self.n);
        }
        self.batch_size = (total_elements / self.n) as u32;
        
        Ok(())
    }
    
    fn update_n(&mut self, new_n: usize) -> Result<()> {
        self.n = new_n;
        Ok(())
    }
}

/// A PipelineStage that performs element-wise complex multiplication.
///
/// This stage multiplies two complex buffers element-wise: `output[i] = input_a[i] * input_b[i]`
/// where `input_a` is the pipeline's input buffer and `input_b` is provided as a side input
/// buffer named "input_b".
///
/// # Bind Group Layout
///
/// This stage uses a custom bind group layout with 3 bindings:
/// - Binding 0: input_a (storage, read-only)
/// - Binding 1: input_b (storage, read-only) - side input
/// - Binding 2: output (storage, read-write)
///
/// # Usage
///
/// ```ignore
/// use std::sync::Arc;
/// use wgsl_ping_pong_pipeline::{Pipeline, StageConfig};
/// use wgsl_fft::ping_pong_integration::MultiplyPipelineStage;
///
/// let multiply_stage = MultiplyPipelineStage::new(1024, 1);
/// let pipeline = Pipeline::new()
///     .pipe_config(StageConfig::Custom(Box::new(multiply_stage)))
///     .build()
///     .await?;
///
/// // Add side input buffer
/// let input_b_buffer: Arc<wgpu::Buffer> = ...;
/// pipeline.add_side_input("input_b", Arc::clone(&input_b_buffer));
/// ```
pub struct MultiplyPipelineStage {
    n: usize,
    batch_size: u32,
    /// The compiled compute pipeline for complex multiplication
    compute_pipeline: Option<wgpu::ComputePipeline>,
    /// The bind group layout
    bind_group_layout: Option<wgpu::BindGroupLayout>,
    /// The device for creating bind groups
    device: Option<wgpu::Device>,
}

impl MultiplyPipelineStage {
    /// Creates a new complex multiplication stage.
    ///
    /// # Arguments
    /// * `n` - Number of complex elements
    /// * `batch_size` - Number of batches
    pub fn new(n: usize, batch_size: u32) -> Self {
        Self {
            n,
            batch_size,
            compute_pipeline: None,
            bind_group_layout: None,
            device: None,
        }
    }
}

impl Debug for MultiplyPipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiplyPipelineStage")
            .field("n", &self.n)
            .field("batch_size", &self.batch_size)
            .field("compute_pipeline_initialized", &self.compute_pipeline.is_some())
            .field("device_initialized", &self.device.is_some())
            .finish()
    }
}

impl PipelineStage for MultiplyPipelineStage {
    fn name(&self) -> &str {
        "multiply_complex"
    }

    fn vector_dim(&self) -> usize {
        2 // Complex numbers as vec2<f32>
    }

    fn batch_size(&self) -> usize {
        self.n * self.batch_size as usize
    }

    fn side_input_names(&self) -> Vec<&str> {
        vec!["input_b"]
    }

    fn encode(
        &self,
        encoder: &mut CommandEncoder,
        input_a: &wgpu::Buffer,
        output: &wgpu::Buffer,
        side_inputs: &HashMap<String, Arc<wgpu::Buffer>>,
    ) -> Result<()> {
        let pipeline = self
            .compute_pipeline
            .as_ref()
            .ok_or_else(|| anyhow!("MultiplyPipelineStage must be initialized before encode()"))?;
        let bgl = self
            .bind_group_layout
            .as_ref()
            .ok_or_else(|| anyhow!("MultiplyPipelineStage bind group layout must be initialized before encode()"))?;

        let device = self
            .device
            .as_ref()
            .ok_or_else(|| anyhow!("MultiplyPipelineStage device must be initialized before encode()"))?;

        // Get the side input buffer
        let input_b = side_inputs
            .get("input_b")
            .ok_or_else(|| anyhow!("MultiplyPipelineStage requires side input 'input_b'"))?;

        // Create bind group with all three buffers
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MultiplyPipelineStage Bind Group"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MultiplyPipelineStage Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        // Dispatch based on actual output buffer size to handle resized buffers
        let element_size = 2 * std::mem::size_of::<f32>() as u64;
        let total_elements = (output.size() / element_size) as u32;
        let workgroup_size = 256u32;
        let dispatch_count = (total_elements + workgroup_size - 1) / workgroup_size;
        pass.dispatch_workgroups(dispatch_count, 1, 1);

        Ok(())
    }

    fn initialize(&mut self, context: &ComputeContext) -> Result<()> {
        let device = &context.device;
        
        // Create bind group layout with 3 bindings
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("MultiplyPipelineStage Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create the compute pipeline with the complex multiplication shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Complex Multiply Shader"),
            source: wgpu::ShaderSource::Wgsl(COMPLEX_MULTIPLY_WGSL.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("MultiplyPipelineStage Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MultiplyPipelineStage Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        self.compute_pipeline = Some(compute_pipeline);
        self.bind_group_layout = Some(bgl);
        self.device = Some(device.clone());

        Ok(())
    }

    fn requires_initialization(&self) -> bool {
        true
    }
    
    fn supports_dynamic_resizing(&self) -> bool {
        true
    }
    
    fn resize(&mut self, new_batch_size: usize, new_vector_dim: usize) -> Result<()> {
        // For MultiplyPipelineStage, we expect vector_dim to always be 2 (complex numbers)
        if new_vector_dim != 2 {
            anyhow::bail!("MultiplyPipelineStage requires vector_dim of 2 for complex numbers");
        }
        
        // Update our cached values
        let total_elements = new_batch_size;
        if total_elements % self.n != 0 {
            anyhow::bail!("New batch size {} is not a multiple of element count {}", total_elements, self.n);
        }
        self.batch_size = (total_elements / self.n) as u32;
        
        Ok(())
    }
    
    fn update_n(&mut self, new_n: usize) -> Result<()> {
        self.n = new_n;
        Ok(())
    }
}

/// WGSL shader for complex multiplication: output[i] = input_a[i] * input_b[i]
///
/// Each complex number is stored as vec2<f32> where:
/// - .x = real part
/// - .y = imaginary part
///
/// Complex multiplication formula:
/// (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
const COMPLEX_MULTIPLY_WGSL: &str = r#"
@group(0) @binding(0)
var<storage, read> input_a: array<vec2<f32>>;

@group(0) @binding(1)
var<storage, read> input_b: array<vec2<f32>>;

@group(0) @binding(2)
var<storage, read_write> output: array<vec2<f32>>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&output)) { return; }
    output[idx] = cmul(input_a[idx], input_b[idx]);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_stage_creation() {
        let stage = FftPipelineStage::forward(1024, 1);
        assert_eq!(stage.name(), "fft_forward");
        assert_eq!(stage.vector_dim(), 2);
        assert_eq!(stage.batch_size(), 1024 * 1);
    }

    #[test]
    fn test_ifft_stage_creation() {
        let stage = FftPipelineStage::inverse(1024, 1);
        assert_eq!(stage.name(), "fft_inverse");
        assert_eq!(stage.vector_dim(), 2);
        assert_eq!(stage.batch_size(), 1024 * 1);
    }

    #[test]
    fn test_multiply_stage_creation() {
        let stage = MultiplyPipelineStage::new(1024, 1);
        assert_eq!(stage.name(), "multiply_complex");
        assert_eq!(stage.vector_dim(), 2);
        assert_eq!(stage.batch_size(), 1024 * 1);
        assert_eq!(stage.side_input_names(), vec!["input_b"]);
    }
}
