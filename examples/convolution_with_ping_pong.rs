//! Example demonstrating FFT-based convolution using wgsl-fft with wgsl-ping-pong-pipeline.
//!
//! This example shows how to integrate wgsl-fft's FFT functionality with the generic
//! ping-pong pipeline infrastructure to create a convolution pipeline.
//!
//! Convolution formula: conv(A, B) = IFFT(FFT(A) * FFT(B))
//!
//! Pipeline structure:
//!   Input A -> [FFT] -> [Multiply (with B_fft as side input)] -> [IFFT] -> [Normalize] -> Output
//!
//! Note: B must be pre-processed with FFT separately and registered as a side input.

use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use wgsl_fft::{FftDirection, FftPipelines};
use wgsl_ping_pong_pipeline::pipeline::pipeline_stage::PipelineStage;
use wgsl_ping_pong_pipeline::pipeline::Pipeline;
use wgsl_ping_pong_pipeline::wgpu_utils::ComputeContext;

/// A pipeline stage that performs FFT or IFFT using wgsl-fft's FftPipelines.
///
/// This implements the PipelineStage trait from wgsl-ping-pong-pipeline,
/// allowing FFT operations to be integrated into any pipeline.
struct FftPipelineStage {
    name: String,
    n: usize,
    batch_size: u32,
    direction: FftDirection,
    fft_pipelines: Arc<FftPipelines>,
}

impl FftPipelineStage {
    /// Creates a new FFT pipeline stage.
    ///
    /// # Arguments
    /// * `name` - Name for this stage (for debugging)
    /// * `n` - FFT size (number of complex elements, must be power of 2)
    /// * `batch_size` - Number of FFTs to process in parallel
    /// * `direction` - FFT direction (Forward or Inverse)
    /// * `fft_pipelines` - Shared FftPipelines instance
    pub fn new(
        name: impl Into<String>,
        n: usize,
        batch_size: u32,
        direction: FftDirection,
        fft_pipelines: Arc<FftPipelines>,
    ) -> Self {
        Self {
            name: name.into(),
            n,
            batch_size,
            direction,
            fft_pipelines,
        }
    }
}

impl std::fmt::Debug for FftPipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FftPipelineStage")
            .field("name", &self.name)
            .field("n", &self.n)
            .field("batch_size", &self.batch_size)
            .field("direction", &self.direction)
            .field("fft_pipelines", &"Arc<FftPipelines>")
            .finish()
    }
}

impl PipelineStage for FftPipelineStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn vector_dim(&self) -> usize {
        // FFT operates on complex numbers (vec2<f32>)
        2
    }

    fn batch_size(&self) -> usize {
        self.n * self.batch_size as usize
    }

    fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        _side_inputs: &HashMap<String, Arc<wgpu::Buffer>>,
    ) -> Result<()> {
        self.fft_pipelines.encode_fft(
            encoder,
            self.n,
            self.batch_size,
            self.direction,
            input,
            output,
        );
        Ok(())
    }

    fn initialize(&mut self, _context: &ComputeContext) -> Result<()> {
        // Already initialized with FftPipelines
        Ok(())
    }

    fn requires_initialization(&self) -> bool {
        false
    }
}

/// A pipeline stage that performs element-wise complex multiplication.
///
/// For convolution: multiplies FFT(A) by FFT(B)
/// Each element is a vec2<f32> representing a complex number (re, im)
struct MultiplyPipelineStage {
    name: String,
    n: usize,
    batch_size: u32,
    side_input_name: String,
    /// Cached resources
    device: Option<wgpu::Device>,
    compute_pipeline: Option<wgpu::ComputePipeline>,
    bind_group_layout: Option<wgpu::BindGroupLayout>,
}

impl MultiplyPipelineStage {
    /// Creates a new multiply stage with a side input.
    pub fn with_side_input(
        name: impl Into<String>,
        n: usize,
        batch_size: u32,
        side_input_name: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            n,
            batch_size,
            side_input_name: side_input_name.into(),
            device: None,
            compute_pipeline: None,
            bind_group_layout: None,
        }
    }
}

impl std::fmt::Debug for MultiplyPipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiplyPipelineStage")
            .field("name", &self.name)
            .field("n", &self.n)
            .field("batch_size", &self.batch_size)
            .field("side_input_name", &self.side_input_name)
            .field("device", &self.device.as_ref().map_or("None", |_| "Some"))
            .field(
                "compute_pipeline",
                &self.compute_pipeline.as_ref().map_or("None", |_| "Some"),
            )
            .field(
                "bind_group_layout",
                &self.bind_group_layout.as_ref().map_or("None", |_| "Some"),
            )
            .finish()
    }
}

impl PipelineStage for MultiplyPipelineStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn vector_dim(&self) -> usize {
        2 // Complex numbers (vec2<f32>)
    }

    fn batch_size(&self) -> usize {
        self.n * self.batch_size as usize
    }

    fn side_input_names(&self) -> Vec<&str> {
        vec![self.side_input_name.as_str()]
    }

    fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input_a: &wgpu::Buffer,
        output: &wgpu::Buffer,
        side_inputs: &HashMap<String, Arc<wgpu::Buffer>>,
    ) -> Result<()> {
        let device = self
            .device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("MultiplyPipelineStage not initialized"))?;
        let pipeline = self
            .compute_pipeline
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("MultiplyPipelineStage compute pipeline not created"))?;
        let bgl = self.bind_group_layout.as_ref().ok_or_else(|| {
            anyhow::anyhow!("MultiplyPipelineStage bind group layout not created")
        })?;

        // Get side input buffer
        let input_b = side_inputs
            .get(&self.side_input_name)
            .ok_or_else(|| anyhow::anyhow!("Side input '{}' not found", self.side_input_name))?;

        // Create bind group with all 3 buffers
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} Bind Group", self.name)),
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

        // Dispatch compute pass
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("{} Pass", self.name)),
            timestamp_writes: None,
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        let workgroup_size = 256u32;
        let total_elements = self.batch_size() as u32;
        let dispatch_count = (total_elements + workgroup_size - 1) / workgroup_size;
        pass.dispatch_workgroups(dispatch_count, 1, 1);

        Ok(())
    }

    fn initialize(&mut self, context: &ComputeContext) -> Result<()> {
        self.device = Some(context.device.clone());

        // Create bind group layout for 3 bindings
        self.bind_group_layout = Some(context.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("{} Bind Group Layout", self.name)),
                entries: &[
                    // Binding 0: input_a
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
                    // Binding 1: input_b (side input)
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
                    // Binding 2: output
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
            },
        ));

        // Create compute pipeline
        let shader_source = r#"
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

        let shader = context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{} Shader", self.name)),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(&format!("{} Pipeline Layout", self.name)),
                    bind_group_layouts: &[Some(self.bind_group_layout.as_ref().unwrap())],
                    immediate_size: 0,
                });

        self.compute_pipeline = Some(context.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{} Pipeline", self.name)),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));

        Ok(())
    }

    fn requires_initialization(&self) -> bool {
        self.compute_pipeline.is_none()
    }
}

/// A pipeline stage that performs normalization (divide by N).
///
/// Required after IFFT to scale the result correctly.
struct NormalizePipelineStage {
    name: String,
    n: usize,
    batch_size: u32,
    fft_pipelines: Arc<FftPipelines>,
}

impl NormalizePipelineStage {
    pub fn new(
        name: impl Into<String>,
        n: usize,
        batch_size: u32,
        fft_pipelines: Arc<FftPipelines>,
    ) -> Self {
        Self {
            name: name.into(),
            n,
            batch_size,
            fft_pipelines,
        }
    }
}

impl std::fmt::Debug for NormalizePipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NormalizePipelineStage")
            .field("name", &self.name)
            .field("n", &self.n)
            .field("batch_size", &self.batch_size)
            .field("fft_pipelines", &"Arc<FftPipelines>")
            .finish()
    }
}

impl PipelineStage for NormalizePipelineStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn vector_dim(&self) -> usize {
        2 // Complex numbers
    }

    fn batch_size(&self) -> usize {
        self.n * self.batch_size as usize
    }

    fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        _side_inputs: &HashMap<String, Arc<wgpu::Buffer>>,
    ) -> Result<()> {
        self.fft_pipelines
            .encode_normalize(encoder, self.n, self.batch_size, output);

        // Note: encode_normalize does in-place normalization on the output buffer
        // But for pipeline integration, we need to copy from input to output first
        // Actually, looking at the implementation, encode_normalize takes the buffer to normalize
        // So we need to copy input to output first, then normalize
        encoder.copy_buffer_to_buffer(input, 0, output, 0, output.size());
        self.fft_pipelines
            .encode_normalize(encoder, self.n, self.batch_size, output);

        Ok(())
    }

    fn initialize(&mut self, _context: &ComputeContext) -> Result<()> {
        Ok(())
    }

    fn requires_initialization(&self) -> bool {
        false
    }
}

/// Helper function to compute FFT of a signal using wgsl-fft.
///
/// Returns a buffer containing the FFT result.
async fn compute_fft(
    fft_pipelines: &Arc<FftPipelines>,
    input_data: &[f32],
    n: usize,
    batch_size: u32,
) -> Result<wgpu::Buffer> {
    let device = fft_pipelines.device();
    let queue = fft_pipelines.queue();

    // Create input and output buffers
    let buffer_size = (n * 2 * batch_size as usize * std::mem::size_of::<f32>()) as u64;

    let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fft_input"),
        size: buffer_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fft_output"),
        size: buffer_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Write input data
    queue.write_buffer(&input_buffer, 0, bytemuck::cast_slice(input_data));

    // Create encoder and encode FFT
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("fft_encoder"),
    });

    fft_pipelines.encode_fft(
        &mut encoder,
        n,
        batch_size,
        FftDirection::Forward,
        &input_buffer,
        &output_buffer,
    );

    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    })?;

    Ok(output_buffer)
}

/// Generates a simple test signal (impulse at the beginning).
fn generate_impulse_signal(n: usize) -> Vec<f32> {
    let mut signal = vec![0.0; n * 2]; // n complex numbers = 2n f32 values
    if n > 0 {
        signal[0] = 1.0; // Real part of first element = 1.0
        signal[1] = 0.0; // Imag part of first element = 0.0
    }
    signal
}

/// Generates a simple test signal (impulse at position n/4).
fn generate_delayed_impulse_signal(n: usize) -> Vec<f32> {
    let mut signal = vec![0.0; n * 2];
    let delay = n / 4;
    if delay < n {
        signal[delay * 2] = 1.0; // Real part
        signal[delay * 2 + 1] = 0.0; // Imag part
    }
    signal
}

fn main() -> Result<()> {
    pollster::block_on(async {
        println!("=== FFT-based Convolution with wgsl-ping-pong-pipeline ===\n");

        // Create FFT pipelines (shared across all FFT stages)
        println!("Creating FftPipelines...");
        let fft_pipelines = Arc::new(FftPipelines::new()?);
        println!("✓ FftPipelines created");

        // FFT parameters
        let n = 1024; // Must be power of 2 for wgsl-fft
        let batch_size = 1u32;

        println!("\nParameters:");
        println!("  FFT size (n): {}", n);
        println!("  Batch size: {}", batch_size);

        // Generate test signals
        println!("\nGenerating test signals...");
        let a_data = generate_impulse_signal(n);
        let b_data = generate_delayed_impulse_signal(n);
        println!("✓ Test signals generated");

        // Pre-compute FFT(B) - this will be used as a side input
        println!("\nPre-computing FFT(B)...");
        let b_fft_buffer = compute_fft(&fft_pipelines, &b_data, n, batch_size).await?;
        println!("✓ FFT(B) computed");

        // Build the convolution pipeline: FFT(A) -> Multiply -> IFFT -> Normalize
        println!("\nBuilding convolution pipeline...");
        let mut pipeline: Pipeline<u64> = Pipeline::new()
            // Stage 0: FFT of input A
            .pipe_custom(Box::new(FftPipelineStage::new(
                "fft_a",
                n,
                batch_size,
                FftDirection::Forward,
                Arc::clone(&fft_pipelines),
            )))
            // Stage 1: Multiply FFT(A) by FFT(B) (B_fft is side input)
            .pipe_custom(Box::new(MultiplyPipelineStage::with_side_input(
                "multiply", n, batch_size, "b_fft",
            )))
            // Stage 2: Inverse FFT
            .pipe_custom(Box::new(FftPipelineStage::new(
                "ifft",
                n,
                batch_size,
                FftDirection::Inverse,
                Arc::clone(&fft_pipelines),
            )))
            // Stage 3: Normalize (divide by N)
            .pipe_custom(Box::new(NormalizePipelineStage::new(
                "normalize",
                n,
                batch_size,
                Arc::clone(&fft_pipelines),
            )))
            .build()
            .await?;

        println!("✓ Pipeline built with {} stages", pipeline.num_stages());

        // Register B_fft as a side input for the multiply stage
        println!("\nRegistering FFT(B) as side input...");
        pipeline.add_side_input("b_fft", Arc::new(b_fft_buffer));
        println!("✓ Side input registered");

        // Write input A to the pipeline
        println!("\nWriting input A to pipeline...");
        pipeline.write_input(&a_data).await?;
        println!("✓ Input A written");

        // Tick the pipeline - data needs to propagate through all 4 stages
        println!("\nTicking pipeline...");
        for i in 0..pipeline.num_stages() {
            pipeline.tick(0u64).await?;
            println!("  Tick {}: Data propagated to stage {}", i + 1, i + 1);
        }
        println!("✓ Pipeline complete");

        // Read the output
        println!("\nReading output...");
        let result = pipeline.read_output().await?;
        let (tag, data) = result.expect("Expected output to be ready");
        println!("✓ Output read (tag: {:?}, {} elements)", tag, data.len());

        // Print first few values
        println!("\nFirst 10 output values:");
        for i in 0..std::cmp::min(10, data.len()) {
            print!("  [{}] = {:.6}, ", i, data[i]);
        }
        println!("...");

        println!("\n=== Convolution Example Complete ===");
        println!("This demonstrates FFT-based convolution using:");
        println!("  - wgsl-fft for FFT/IFFT operations");
        println!("  - wgsl-ping-pong-pipeline for pipeline infrastructure");
        println!("\nThe pipeline structure was:");
        println!("  Input A -> FFT -> Multiply (with FFT(B)) -> IFFT -> Normalize -> Output");

        Ok(())
    })
}
