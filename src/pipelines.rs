//! Pre-compiled Cooley-Tukey Radix-2 FFT pipelines for embedding in a larger GPU pipeline.
//!
//! `FftPipelines` owns its wgpu device and queue. Use [`FftPipelines::device`] and
//! [`FftPipelines::queue`] to share them with the rest of your pipeline so all GPU
//! resources live on a single device.

use std::cell::RefCell;

use wgpu::util::DeviceExt;
use wgpu::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    BufferBindingType, ComputePipeline, Device, ShaderStages,
};

use crate::buffer::{PingPongBuffers, PingPongState};
use crate::shaders;

/// Direction for FFT operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftDirection {
    /// Forward FFT
    Forward = 0,
    /// Inverse FFT
    Inverse = 1,
}

// Private helper functions

fn fft_storage_entry(binding: u32, read_only: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn fft_uniform_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn fft_make_pipeline(
    device: &Device,
    label: &str,
    bgl: &BindGroupLayout,
    src: &str,
) -> ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&format!("{label}_shader")),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label}_layout")),
        bind_group_layouts: &[Some(bgl)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Pre-compiled Cooley-Tukey Radix-2 FFT pipelines for embedding in a larger GPU pipeline.
///
/// `FftPipelines` owns its wgpu device and queue. Use [`FftPipelines::device`] and
/// [`FftPipelines::queue`] to share them with the rest of your pipeline so all GPU
/// resources live on a single device.
///
/// # Example
///
/// ```no_run
/// use wgsl_fft::{FftPipelines, FftDirection};
///
/// let fft = FftPipelines::new().expect("GPU required");
/// let device = fft.device();
/// let queue  = fft.queue();
///
/// let n: usize = 1024;
/// // allocate input_buf, scratch0, scratch1 as STORAGE buffers of size n * 8 bytes
/// # let input_buf: wgpu::Buffer = unimplemented!();
/// # let output_buf: wgpu::Buffer = unimplemented!();
/// let mut encoder = device.create_command_encoder(&Default::default());
/// fft.encode_fft(&mut encoder, n, FftDirection::Forward, &input_buf, &output_buf);
/// fft.encode_normalize(&mut encoder, n, &output_buf);
/// queue.submit(std::iter::once(encoder.finish()));
/// ```
pub struct FftPipelines {
    device: Device,
    /// The wgpu queue for submitting command encoders.
    pub queue: wgpu::Queue,
    pipeline_butterfly: ComputePipeline,
    pipeline_bit_reverse: ComputePipeline,
    pipeline_normalize: ComputePipeline,
    bgl: BindGroupLayout,
    bgl_norm: BindGroupLayout,
    scratch: RefCell<std::collections::HashMap<usize, wgpu::Buffer>>,
}

impl FftPipelines {
    /// Initialize GPU and compile all three FFT compute pipelines.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
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
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                ..Default::default()
            }))?;
        Ok(Self::from_device_queue(device, queue))
    }

    /// Build pipelines from an existing device and queue.
    ///
    /// Use this when you already have a `wgpu::Device` (e.g. from a window surface)
    /// and want to avoid creating a second GPU context.
    pub fn from_device_queue(device: Device, queue: wgpu::Queue) -> Self {
        let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("fft_pipelines_bgl"),
            entries: &[
                fft_storage_entry(0, true),
                fft_storage_entry(1, false),
                fft_uniform_entry(2),
            ],
        });
        let bgl_norm = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("fft_pipelines_norm_bgl"),
            entries: &[fft_storage_entry(0, false), fft_uniform_entry(1)],
        });
        let pipeline_butterfly = fft_make_pipeline(
            &device,
            "fft_butterfly",
            &bgl,
            shaders::COOLEY_TUKEY_R2_WGSL,
        );
        let pipeline_bit_reverse =
            fft_make_pipeline(&device, "fft_bit_reverse", &bgl, shaders::BIT_REVERSAL_WGSL);
        let pipeline_normalize = fft_make_pipeline(
            &device,
            "fft_normalize",
            &bgl_norm,
            shaders::NORMALIZE_VEC2_WGSL,
        );
        Self {
            device,
            queue,
            pipeline_butterfly,
            pipeline_bit_reverse,
            pipeline_normalize,
            bgl,
            bgl_norm,
            scratch: RefCell::new(std::collections::HashMap::new()),
        }
    }

    /// The wgpu device that owns all GPU resources in this instance.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The wgpu queue for submitting command encoders.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Encode one FFT or IFFT into `encoder`. The result is written to `output_buf`.
    ///
    /// An internal scratch buffer (allocated lazily per `n`) handles the ping-pong.
    /// Compute passes within a command buffer are sequential, so one scratch buffer
    /// per size is safe even when multiple FFTs of the same `n` are encoded back-to-back.
    pub fn encode_fft(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        n: usize,
        direction: FftDirection,
        input_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
    ) {
        let log2_n = n.trailing_zeros();
        let dir = direction as u32;
        let byte_size = (n * 8) as u64; // n * sizeof(vec2<f32>)

        // Ensure a scratch buffer exists for this n
        {
            let mut map = self.scratch.borrow_mut();
            map.entry(n).or_insert_with(|| {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("fft_scratch"),
                    size: byte_size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            });
        }
        let scratch_map = self.scratch.borrow();
        let scratch_buf = scratch_map.get(&n).unwrap();

        // Assign the two ping-pong slots so that after log2_n butterfly passes
        // the result naturally lands in output_buf without any extra copy.
        // Bit-reversal writes to bufs[0]; each butterfly pass shifts current by 1.
        // After log2_n passes, result is in bufs[log2_n % 2], so we set that to output_buf.
        let (buf0, buf1): (&wgpu::Buffer, &wgpu::Buffer) = if log2_n % 2 == 0 {
            (output_buf, scratch_buf)
        } else {
            (scratch_buf, output_buf)
        };

        let br_params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bit_rev_params"),
                contents: bytemuck::cast_slice(&[n as u32, log2_n, 0u32, 0u32]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        {
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fft_bit_rev_bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf0.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: br_params.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bit_reversal_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_bit_reverse);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(256), 1, 1);
        }

        let bufs = [buf0, buf1];
        for stage in 0..log2_n {
            let src = bufs[stage as usize % 2];
            let dst = bufs[(stage as usize + 1) % 2];
            let fft_params = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("fft_stage{stage}_params")),
                    contents: bytemuck::cast_slice(&[n as u32, stage, dir, 0u32]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            {
                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("fft_butterfly_bg_stage{stage}")),
                    layout: &self.bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: src.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: dst.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: fft_params.as_entire_binding(),
                        },
                    ],
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("fft_butterfly_stage{stage}")),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_butterfly);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(((n / 2) as u32).div_ceil(256), 1, 1);
            }
        }
        // result is now in bufs[log2_n % 2] = output_buf
    }

    /// Encode an in-place divide-by-N pass on `buf` (IFFT normalization).
    pub fn encode_normalize(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        n: usize,
        buf: &wgpu::Buffer,
    ) {
        let params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("normalize_params"),
                contents: bytemuck::cast_slice(&[n as u32, 0u32, 0u32, 0u32]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        {
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("normalize_bg"),
                layout: &self.bgl_norm,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("normalize_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_normalize);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(256), 1, 1);
        }
    }

    // New simplified API for ping-pong buffer integration

    /// Create a ping-pong buffer pair for use with the ping-pong API.
    /// Each buffer has the specified size in bytes.
    pub fn create_pingpong(&self, size: u64, label: &str) -> PingPongBuffers {
        PingPongBuffers::new(&self.device, size, label)
    }

    /// Encode an FFT using a ping-pong buffer pair with explicit state control.
    ///
    /// This method uses the provided ping-pong buffers for all intermediate storage.
    /// The FFT result is written to the write buffer determined by `state`.
    ///
    /// For multi-stage pipelines: after calling this, toggle the state and pass it
    /// to the next stage, which will read from the buffer this stage wrote to.
    ///
    /// # Example
    /// ```ignore
    /// use wgsl_fft::{FftPipelines, FftDirection, PingPongState, PingPongBuffers};
    ///
    /// let fft = FftPipelines::new()?;
    /// let mut state = PingPongState::Read0Write1;
    /// let pp = fft.create_pingpong(size, "my_pipeline");
    /// let input_buf = /* ... */;
    ///
    /// let mut encoder = fft.device().create_command_encoder(&Default::default());
    ///
    /// // Stage 1: FFT forward - reads from A (via input copy), writes to B
    /// fft.encode_fft_with_pingpong(&mut encoder, n, FftDirection::Forward, state, &pp, input_buf);
    /// state.toggle(); // Now: Read1Write0
    ///
    /// // Stage 2: Multiply - reads from B, writes to A
    /// // (would use similar API for multiply stage)
    /// state.toggle(); // Now: Read0Write1
    ///
    /// // Stage 3: IFFT - reads from A, writes to B
    /// fft.encode_fft_with_pingpong(&mut encoder, n, FftDirection::Inverse, state, &pp, input_buf);
    /// ```
    pub fn encode_fft_with_pingpong(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        n: usize,
        direction: FftDirection,
        state: PingPongState,
        ping_pong: &PingPongBuffers,
        input_buf: &wgpu::Buffer,
    ) {
        let (_, write_buf) = ping_pong.get(state);
        self.encode_fft(encoder, n, direction, input_buf, write_buf);
    }

    /// Encode normalize on the write buffer of a ping-pong pair.
    /// Call this after IFFT, passing the same state that was used for the IFFT.
    pub fn encode_normalize_with_pingpong(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        n: usize,
        state: PingPongState,
        ping_pong: &PingPongBuffers,
    ) {
        let (_, write_buf) = ping_pong.get(state);
        self.encode_normalize(encoder, n, write_buf);
    }
}
