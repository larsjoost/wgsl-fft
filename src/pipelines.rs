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

use crate::error::Result;
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

// Pre-baked resources for a single encode_fft call (specific n, direction, input/output buffers).
// bind_groups[0] = bit-reversal pass, bind_groups[1..] = butterfly stages.
// params holds the uniform buffers alive (wgpu BindGroups ref-count their resources,
// but keeping them here makes the ownership explicit).
struct FftCallCache {
    #[allow(dead_code)]
    params: Vec<wgpu::Buffer>,
    bind_groups: Vec<wgpu::BindGroup>,
}

struct FftNormCache {
    #[allow(dead_code)]
    params: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
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
/// // allocate input_buf and output_buf as STORAGE buffers of size n * 8 bytes (scratch is managed internally)
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
    // Keyed by (n, direction_as_u32, input_ptr, output_ptr)
    call_cache: RefCell<std::collections::HashMap<(usize, u32, usize, usize), FftCallCache>>,
    // Keyed by (n, buf_ptr)
    norm_cache: RefCell<std::collections::HashMap<(usize, usize), FftNormCache>>,
}

impl FftPipelines {
    /// Get buffer pair based on log2_n parity.
    fn get_buffer_pair_for_mode<'a>(
        log2_n: u32,
        output_buf: &'a wgpu::Buffer,
        scratch_buf: &'a wgpu::Buffer,
    ) -> (&'a wgpu::Buffer, &'a wgpu::Buffer) {
        if log2_n % 2 == 0 {
            return (output_buf, scratch_buf);
        }
        (scratch_buf, output_buf)
    }

    /// Initialize GPU and compile all three FFT compute pipelines.
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
            call_cache: RefCell::new(std::collections::HashMap::new()),
            norm_cache: RefCell::new(std::collections::HashMap::new()),
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
    /// All bind groups and uniform buffers are cached after the first call for a given
    /// (n, direction, input_buf, output_buf) combination — subsequent calls encode with
    /// zero allocations.
    pub fn encode_fft(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        n: usize,
        direction: FftDirection,
        input_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
    ) {
        let log2_n = n.trailing_zeros();

        // Ensure scratch buffer exists for this n
        {
            let byte_size = (n * 8) as u64;
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

        let key = (
            n,
            direction as u32,
            input_buf as *const _ as usize,
            output_buf as *const _ as usize,
        );

        // Build and cache on first call for this key
        {
            let scratch_guard = self.scratch.borrow();
            let scratch_buf = scratch_guard.get(&n).unwrap();
            let mut cache = self.call_cache.borrow_mut();
            if !cache.contains_key(&key) {
                let entry = Self::build_fft_cache(
                    &self.device, &self.bgl, n, direction, input_buf, output_buf, scratch_buf,
                );
                cache.insert(key, entry);
            }
        }

        // Encode using cached bind groups — zero allocations
        let cache_guard = self.call_cache.borrow();
        let cached = cache_guard.get(&key).unwrap();

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bit_reversal_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_bit_reverse);
            pass.set_bind_group(0, &cached.bind_groups[0], &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(256), 1, 1);
        }

        for stage in 0..log2_n as usize {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fft_butterfly_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_butterfly);
            pass.set_bind_group(0, &cached.bind_groups[1 + stage], &[]);
            pass.dispatch_workgroups(((n / 2) as u32).div_ceil(256), 1, 1);
        }
    }

    fn build_fft_cache(
        device: &Device,
        bgl: &BindGroupLayout,
        n: usize,
        direction: FftDirection,
        input_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        scratch_buf: &wgpu::Buffer,
    ) -> FftCallCache {
        let log2_n = n.trailing_zeros();
        let dir = direction as u32;

        let (buf0, buf1) = Self::get_buffer_pair_for_mode(log2_n, output_buf, scratch_buf);

        let mut params = Vec::with_capacity(1 + log2_n as usize);
        let mut bind_groups = Vec::with_capacity(1 + log2_n as usize);

        // Bit-reversal pass
        let br_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bit_rev_params"),
            contents: bytemuck::cast_slice(&[n as u32, log2_n, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let br_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fft_bit_rev_bg"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: br_params.as_entire_binding() },
            ],
        });
        params.push(br_params);
        bind_groups.push(br_bg);

        // Butterfly passes
        let bufs = [buf0, buf1];
        for stage in 0..log2_n {
            let src = bufs[stage as usize % 2];
            let dst = bufs[(stage as usize + 1) % 2];
            let stage_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("fft_stage{stage}_params")),
                contents: bytemuck::cast_slice(&[n as u32, stage, dir, 0u32]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let stage_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("fft_butterfly_bg_stage{stage}")),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: stage_params.as_entire_binding() },
                ],
            });
            params.push(stage_params);
            bind_groups.push(stage_bg);
        }

        FftCallCache { params, bind_groups }
    }

    /// Encode an in-place divide-by-N pass on `buf` (IFFT normalization).
    ///
    /// Bind group and uniform buffer are cached after the first call for a given (n, buf).
    pub fn encode_normalize(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        n: usize,
        buf: &wgpu::Buffer,
    ) {
        let key = (n, buf as *const _ as usize);
        {
            let mut cache = self.norm_cache.borrow_mut();
            if !cache.contains_key(&key) {
                let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("normalize_params"),
                    contents: bytemuck::cast_slice(&[n as u32, 0u32, 0u32, 0u32]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("normalize_bg"),
                    layout: &self.bgl_norm,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: params.as_entire_binding() },
                    ],
                });
                cache.insert(key, FftNormCache { params, bind_group: bg });
            }
        }
        let cache_guard = self.norm_cache.borrow();
        let cached = cache_guard.get(&key).unwrap();
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("normalize_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_normalize);
            pass.set_bind_group(0, &cached.bind_group, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(256), 1, 1);
        }
    }
}
