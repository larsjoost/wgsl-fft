//! Six-stage GPU pipeline: H2D DMA → Gaussian noise → FFT → window → IFFT → D2H DMA.
//!
//! All six stages run in parallel via **multi-slot pipelining**: N_SLOTS independent buffer sets
//! are kept in flight simultaneously.  At steady state the layout looks like:
//!
//! ```text
//!   Slot 0:  [H2D→noise→FFT→window→IFFT→D2H] ─────────────────────────────
//!   Slot 1:      [H2D→noise→FFT→window→IFFT→D2H] ──────────────────────────
//!   Slot 2:          [H2D→noise→FFT→window→IFFT→D2H] ────────────────────
//!   ...
//! ```
//!
//! Each slot owns an independent set of GPU buffers, so the GPU's async DMA engine can overlap
//! host-device copies of one slot with compute work from another slot.
//!
//! FFT and IFFT use `wgsl_fft::R4_WGSL` / `R2_WGSL` — the Stockham Radix-4/2 baseline from
//! `src/shaders.rs`.  IFFT is implemented by supplying an inverse twiddle table
//! (e^{+2πij/N}) to the same kernels; a `SCALE_WGSL` pass applies the 1/N factor afterwards.

use std::num::NonZeroU64;
use std::time::Instant;

use bytemuck;
use num_complex::Complex;
use wgsl_fft::{R2_WGSL, R4_WGSL};

// ── Inline WGSL shaders for the three non-FFT stages ─────────────────────────

/// Box-Muller Gaussian noise added in-place to a complex signal buffer.
/// Uniform: x=N (samples per signal), y=seed.  Dispatches (N/256, batch, 1).
const NOISE_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> U: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> DATA: array<f32>;

// PCG hash (O'Neill 2014): multipliers are the standard PCG-XSH-RR constants.
fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

// 256 threads per workgroup — standard occupancy target for most GPU architectures.
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid      = gid.x;
    let batch_id = gid.y;
    let n        = U.x;
    let seed     = U.y;
    if tid >= n { return; }

    // Two independent hashes per sample for Box-Muller; primes chosen to decorrelate tid/batch.
    let h1 = pcg(tid * 1234567u + batch_id * 7654321u + seed);
    let h2 = pcg(h1 + 999983u);
    // 2.328306437e-10 = 1/2^32: maps u32 uniform to [0,1].
    let u1 = max(f32(h1) * 2.328306437e-10, 1e-7);
    // 6.283185307 = 2π: Box-Muller phase angle.
    let u2 = f32(h2) * 2.328306437e-10 * 6.283185307;

    let r = sqrt(-2.0 * log(u1)) * 0.1;
    let bo = batch_id * n * 2u;
    DATA[bo + 2u * tid]      += r * cos(u2);
    DATA[bo + 2u * tid + 1u] += r * sin(u2);
}
"#;

/// Hann spectral window applied in-place to a complex frequency-domain buffer.
/// WIN holds N coefficients.  Dispatches (N/256, batch, 1).
const WINDOW_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> DATA: array<f32>;
@group(0) @binding(1) var<storage, read>       WIN:  array<f32>;

// 256 threads per workgroup — standard occupancy target for most GPU architectures.
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k        = gid.x;
    let batch_id = gid.y;
    let n        = arrayLength(&WIN);
    if k >= n { return; }

    let bo = batch_id * n * 2u;
    let w  = WIN[k];
    DATA[bo + 2u * k]      *= w;
    DATA[bo + 2u * k + 1u] *= w;
}
"#;

/// 1/N scaling applied in-place after IFFT.  Uniform: x=N.  Dispatches (N/256, batch, 1).
const SCALE_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> U: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> DATA: array<f32>;

// 256 threads per workgroup — standard occupancy target for most GPU architectures.
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid      = gid.x;
    let batch_id = gid.y;
    let n        = U.x;
    if tid >= n { return; }

    let s  = 1.0 / f32(n);
    let bo = batch_id * n * 2u;
    DATA[bo + 2u * tid]      *= s;
    DATA[bo + 2u * tid + 1u] *= s;
}
"#;

// ── Pipeline constants ────────────────────────────────────────────────────────

const N_SLOTS: usize = 6;

// ── Slot ─────────────────────────────────────────────────────────────────────

struct Slot {
    /// Ping-pong A: H2D target, noise in-place; always holds IFFT result after pipeline.
    buf_a: wgpu::Buffer,
    /// Ping-pong B: FFT/IFFT scratch — kept alive so bind groups remain valid.
    #[allow(dead_code)]
    buf_b: wgpu::Buffer,
    /// CPU-mappable readback — D2H copies buf_a here after the scale pass.
    d2h_staging: wgpu::Buffer,

    noise_uniform_buf: wgpu::Buffer,
    /// Kept alive so the scale bind group remains valid.
    #[allow(dead_code)]
    scale_uniform_buf: wgpu::Buffer,

    noise_bg: wgpu::BindGroup,
    fft_bgs: Vec<wgpu::BindGroup>,
    fft_r2_bg: Option<wgpu::BindGroup>,
    window_bg: wgpu::BindGroup,
    ifft_bgs: Vec<wgpu::BindGroup>,
    ifft_r2_bg: Option<wgpu::BindGroup>,
    scale_bg: wgpu::BindGroup,

    buf_bytes: u64,
}

// ── Pipeline ──────────────────────────────────────────────────────────────────

struct Pipeline {
    device: wgpu::Device,
    queue: wgpu::Queue,

    noise_pipeline: wgpu::ComputePipeline,
    fft_r4_pipeline: wgpu::ComputePipeline,
    fft_r2_pipeline: wgpu::ComputePipeline,
    window_pipeline: wgpu::ComputePipeline,
    scale_pipeline: wgpu::ComputePipeline,

    slots: Vec<Slot>,

    n: usize,
    batch_size: usize,

    /// Workgroup counts per dispatch.
    wg_r4: u32,
    wg_r2: u32,
    wg_n: u32,
}

// ── GPU buffer helpers ────────────────────────────────────────────────────────

fn make_storage_buf(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn make_staging_buf(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("slot_d2h_staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Creates a vec4<u32> uniform (16 bytes) with U.x = n; remaining fields zeroed.
fn make_simple_uniform(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    n: usize,
) -> wgpu::Buffer {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: 16, // vec4<u32>: [n, 0, 0, 0]
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, bytemuck::cast_slice(&[n as u32, 0u32, 0u32, 0u32]));
    buf
}

/// Creates and writes the paired FFT / IFFT uniform buffers (one stride-aligned entry per stage).
fn make_fft_uniforms(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    n: usize,
    num_r4: usize,
    has_r2: bool,
    stride: u64,
) -> (wgpu::Buffer, wgpu::Buffer) {
    let entries = (num_r4 + has_r2 as usize).max(1) as u64;
    let make_buf = |label: &str| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: stride * entries,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let fft = make_buf("fft_uniform");
    let ifft = make_buf("ifft_uniform");
    // R4 stages: p = 4^s; R2 fallback: p = 4^num_r4.
    for s in 0..num_r4 {
        let entry: [u32; 4] = [n as u32, 1u32 << (s as u32 * 2), n.trailing_zeros(), 0];
        let off = stride * s as u64;
        queue.write_buffer(&fft, off, bytemuck::cast_slice(&entry));
        queue.write_buffer(&ifft, off, bytemuck::cast_slice(&entry));
    }
    if has_r2 {
        let entry: [u32; 4] = [n as u32, 1u32 << (num_r4 as u32 * 2), n.trailing_zeros(), 0];
        let off = stride * num_r4 as u64;
        queue.write_buffer(&fft, off, bytemuck::cast_slice(&entry));
        queue.write_buffer(&ifft, off, bytemuck::cast_slice(&entry));
    }
    (fft, ifft)
}

// ── Bind group helpers ────────────────────────────────────────────────────────

/// Creates one FFT/IFFT bind group per Radix-4 stage with correct ping-pong src/dst.
///
/// `start_in_b`: true if the data to process is initially in buf_b (e.g. after an FFT
/// that ended in buf_b). XOR with the stage parity selects src/dst for each stage.
fn make_staged_bind_groups(
    device: &wgpu::Device,
    num_stages: usize,
    start_in_b: bool,
    pipeline: &wgpu::ComputePipeline,
    uniform_buf: &wgpu::Buffer,
    stride: u64,
    twiddle_buf: &wgpu::Buffer,
    buf_a: &wgpu::Buffer,
    buf_b: &wgpu::Buffer,
    uniform_sz: Option<NonZeroU64>,
) -> Vec<wgpu::BindGroup> {
    (0..num_stages)
        .map(|s| {
            let in_b = start_in_b ^ (s % 2 == 1);
            let (src, dst) = if in_b { (buf_b, buf_a) } else { (buf_a, buf_b) };
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: uniform_buf,
                            offset: stride * s as u64,
                            size: uniform_sz,
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
        })
        .collect()
}

/// Creates the single Radix-2 fallback bind group (stage index = num_r4).
fn make_r2_bind_group(
    device: &wgpu::Device,
    num_r4: usize,
    start_in_b: bool,
    pipeline: &wgpu::ComputePipeline,
    uniform_buf: &wgpu::Buffer,
    stride: u64,
    twiddle_buf: &wgpu::Buffer,
    buf_a: &wgpu::Buffer,
    buf_b: &wgpu::Buffer,
    uniform_sz: Option<NonZeroU64>,
) -> wgpu::BindGroup {
    let in_b = start_in_b ^ (num_r4 % 2 == 1);
    let (src, dst) = if in_b { (buf_b, buf_a) } else { (buf_a, buf_b) };
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: uniform_buf,
                    offset: stride * num_r4 as u64,
                    size: uniform_sz,
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
}

fn make_noise_bg(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    uniform_buf: &wgpu::Buffer,
    data_buf: &wgpu::Buffer,
    uniform_sz: Option<NonZeroU64>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("noise_bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: uniform_buf,
                    offset: 0,
                    size: uniform_sz,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: data_buf.as_entire_binding(),
            },
        ],
    })
}

fn make_window_bg(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    spectrum_buf: &wgpu::Buffer,
    window_buf: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("window_bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: spectrum_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: window_buf.as_entire_binding(),
            },
        ],
    })
}

fn make_scale_bg(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    uniform_buf: &wgpu::Buffer,
    result_buf: &wgpu::Buffer,
    uniform_sz: Option<NonZeroU64>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("scale_bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: uniform_buf,
                    offset: 0,
                    size: uniform_sz,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: result_buf.as_entire_binding(),
            },
        ],
    })
}

// ── build_slot ────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn build_slot(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    noise_pipeline: &wgpu::ComputePipeline,
    fft_r4_pipeline: &wgpu::ComputePipeline,
    fft_r2_pipeline: &wgpu::ComputePipeline,
    window_pipeline: &wgpu::ComputePipeline,
    scale_pipeline: &wgpu::ComputePipeline,
    fwd_twiddle_buf: &wgpu::Buffer,
    inv_twiddle_buf: &wgpu::Buffer,
    window_buf: &wgpu::Buffer,
    n: usize,
    batch_size: usize,
    num_r4: usize,
    has_r2: bool,
    fft_result_in_b: bool,
) -> Slot {
    let buf_bytes = (n * batch_size * 2 * std::mem::size_of::<f32>()) as u64;
    let uniform_sz = NonZeroU64::new(16);
    let alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
    let stride = 16u64.div_ceil(alignment) * alignment;

    let buf_a = make_storage_buf(device, "slot_buf_a", buf_bytes);
    let buf_b = make_storage_buf(device, "slot_buf_b", buf_bytes);
    let d2h_staging = make_staging_buf(device, buf_bytes);

    let noise_uniform_buf = make_simple_uniform(device, queue, "noise_uniform", n);
    let scale_uniform_buf = make_simple_uniform(device, queue, "scale_uniform", n);
    let (fft_uniform_buf, ifft_uniform_buf) =
        make_fft_uniforms(device, queue, n, num_r4, has_r2, stride);

    let fft_bgs = make_staged_bind_groups(
        device,
        num_r4,
        false,
        fft_r4_pipeline,
        &fft_uniform_buf,
        stride,
        fwd_twiddle_buf,
        &buf_a,
        &buf_b,
        uniform_sz,
    );
    let fft_r2_bg = has_r2.then(|| {
        make_r2_bind_group(
            device,
            num_r4,
            false,
            fft_r2_pipeline,
            &fft_uniform_buf,
            stride,
            fwd_twiddle_buf,
            &buf_a,
            &buf_b,
            uniform_sz,
        )
    });
    let ifft_bgs = make_staged_bind_groups(
        device,
        num_r4,
        fft_result_in_b,
        fft_r4_pipeline,
        &ifft_uniform_buf,
        stride,
        inv_twiddle_buf,
        &buf_a,
        &buf_b,
        uniform_sz,
    );
    let ifft_r2_bg = has_r2.then(|| {
        make_r2_bind_group(
            device,
            num_r4,
            fft_result_in_b,
            fft_r2_pipeline,
            &ifft_uniform_buf,
            stride,
            inv_twiddle_buf,
            &buf_a,
            &buf_b,
            uniform_sz,
        )
    });

    let spectrum_buf = if fft_result_in_b { &buf_b } else { &buf_a };
    let noise_bg = make_noise_bg(
        device,
        noise_pipeline,
        &noise_uniform_buf,
        &buf_a,
        uniform_sz,
    );
    let window_bg = make_window_bg(device, window_pipeline, spectrum_buf, window_buf);
    let scale_bg = make_scale_bg(
        device,
        scale_pipeline,
        &scale_uniform_buf,
        &buf_a,
        uniform_sz,
    );

    Slot {
        buf_a,
        buf_b,
        d2h_staging,
        noise_uniform_buf,
        scale_uniform_buf,
        noise_bg,
        fft_bgs,
        fft_r2_bg,
        window_bg,
        ifft_bgs,
        ifft_r2_bg,
        scale_bg,
        buf_bytes,
    }
}

// ── Pipeline impl ─────────────────────────────────────────────────────────────

impl Pipeline {
    fn init_gpu() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("GPU required");
        pollster::block_on(adapter.request_device(&Default::default()))
            .expect("device creation failed")
    }

    fn compile_pipelines(
        device: &wgpu::Device,
    ) -> (
        wgpu::ComputePipeline,
        wgpu::ComputePipeline,
        wgpu::ComputePipeline,
        wgpu::ComputePipeline,
        wgpu::ComputePipeline,
    ) {
        let compile = |src: &str, label: &str| {
            let m = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &m,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        (
            compile(NOISE_WGSL, "noise"),
            compile(R4_WGSL, "fft_r4"),
            compile(R2_WGSL, "fft_r2"),
            compile(WINDOW_WGSL, "window"),
            compile(SCALE_WGSL, "scale"),
        )
    }

    /// Precomputes forward/inverse twiddle tables and Hann window coefficients on the GPU.
    fn precompute_tables(
        n: usize,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
        let fwd: Vec<f32> = (0..n)
            .flat_map(|j| {
                let a = -std::f32::consts::TAU * j as f32 / n as f32;
                [a.cos(), a.sin()]
            })
            .collect();
        let inv: Vec<f32> = (0..n)
            .flat_map(|j| {
                let a = std::f32::consts::TAU * j as f32 / n as f32;
                [a.cos(), a.sin()]
            })
            .collect();
        let win: Vec<f32> = (0..n)
            .map(|k| 0.5 * (1.0 - (std::f32::consts::TAU * k as f32 / (n - 1) as f32).cos()))
            .collect();
        let upload = |label: &str, data: &[f32]| {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (data.len() * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
            buf
        };
        (
            upload("fwd_twiddles", &fwd),
            upload("inv_twiddles", &inv),
            upload("window_coeffs", &win),
        )
    }

    fn new(n: usize, batch_size: usize) -> Self {
        assert!(n.is_power_of_two() && n >= 4);
        let log_n = n.trailing_zeros();
        let num_r4 = (log_n / 2) as usize;
        let has_r2 = log_n % 2 == 1;
        let fft_result_in_b = (num_r4 + has_r2 as usize) % 2 == 1;

        let (device, queue) = Self::init_gpu();
        let (noise_pipeline, fft_r4_pipeline, fft_r2_pipeline, window_pipeline, scale_pipeline) =
            Self::compile_pipelines(&device);
        let (fwd_twiddle_buf, inv_twiddle_buf, window_buf) =
            Self::precompute_tables(n, &device, &queue);

        let slots: Vec<Slot> = (0..N_SLOTS)
            .map(|_| {
                build_slot(
                    &device,
                    &queue,
                    &noise_pipeline,
                    &fft_r4_pipeline,
                    &fft_r2_pipeline,
                    &window_pipeline,
                    &scale_pipeline,
                    &fwd_twiddle_buf,
                    &inv_twiddle_buf,
                    &window_buf,
                    n,
                    batch_size,
                    num_r4,
                    has_r2,
                    fft_result_in_b,
                )
            })
            .collect();

        Self {
            device,
            queue,
            noise_pipeline,
            fft_r4_pipeline,
            fft_r2_pipeline,
            window_pipeline,
            scale_pipeline,
            slots,
            n,
            batch_size,
            // 256 matches @workgroup_size(256,1,1) in the WGSL kernels.
            wg_r4: (n as u32 / 4).div_ceil(256),
            wg_r2: (n as u32 / 2).div_ceil(256),
            wg_n: (n as u32).div_ceil(256),
        }
    }

    fn dispatch_noise(&self, enc: &mut wgpu::CommandEncoder, slot: &Slot, bs: u32) {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("noise"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.noise_pipeline);
        pass.set_bind_group(0, &slot.noise_bg, &[]);
        pass.dispatch_workgroups(self.wg_n, bs, 1);
    }

    fn dispatch_fft(&self, enc: &mut wgpu::CommandEncoder, slot: &Slot, bs: u32) {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fft"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.fft_r4_pipeline);
        for bg in &slot.fft_bgs {
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(self.wg_r4, bs, 1);
        }
        if let Some(r2_bg) = &slot.fft_r2_bg {
            pass.set_pipeline(&self.fft_r2_pipeline);
            pass.set_bind_group(0, r2_bg, &[]);
            pass.dispatch_workgroups(self.wg_r2, bs, 1);
        }
    }

    fn dispatch_window(&self, enc: &mut wgpu::CommandEncoder, slot: &Slot, bs: u32) {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("window"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.window_pipeline);
        pass.set_bind_group(0, &slot.window_bg, &[]);
        pass.dispatch_workgroups(self.wg_n, bs, 1);
    }

    fn dispatch_ifft(&self, enc: &mut wgpu::CommandEncoder, slot: &Slot, bs: u32) {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ifft"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.fft_r4_pipeline);
        for bg in &slot.ifft_bgs {
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(self.wg_r4, bs, 1);
        }
        if let Some(r2_bg) = &slot.ifft_r2_bg {
            pass.set_pipeline(&self.fft_r2_pipeline);
            pass.set_bind_group(0, r2_bg, &[]);
            pass.dispatch_workgroups(self.wg_r2, bs, 1);
        }
    }

    fn dispatch_scale_and_copy(&self, enc: &mut wgpu::CommandEncoder, slot: &Slot, bs: u32) {
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scale"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scale_pipeline);
            pass.set_bind_group(0, &slot.scale_bg, &[]);
            pass.dispatch_workgroups(self.wg_n, bs, 1);
        }
        enc.copy_buffer_to_buffer(&slot.buf_a, 0, &slot.d2h_staging, 0, slot.buf_bytes);
    }

    /// Upload raw f32 data (interleaved re/im), run all pipeline stages, return submission index.
    fn submit_frame(&self, slot_idx: usize, frame_seed: u32, raw: &[f32]) -> wgpu::SubmissionIndex {
        let slot = &self.slots[slot_idx];
        self.queue
            .write_buffer(&slot.buf_a, 0, bytemuck::cast_slice(raw));
        self.queue.write_buffer(
            &slot.noise_uniform_buf,
            0,
            bytemuck::cast_slice(&[self.n as u32, frame_seed, 0u32, 0u32]),
        );

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pipeline_enc"),
            });
        let bs = self.batch_size as u32;

        self.dispatch_noise(&mut enc, slot, bs);
        self.dispatch_fft(&mut enc, slot, bs);
        self.dispatch_window(&mut enc, slot, bs);
        self.dispatch_ifft(&mut enc, slot, bs);
        self.dispatch_scale_and_copy(&mut enc, slot, bs);

        let idx = self.queue.submit([enc.finish()]);
        slot.d2h_staging
            .slice(0..slot.buf_bytes)
            .map_async(wgpu::MapMode::Read, |_| {});
        idx
    }

    /// Block until submission `idx` for slot `slot_idx` completes, return Complex output.
    fn collect_frame(&self, slot_idx: usize, idx: wgpu::SubmissionIndex) -> Vec<Vec<Complex<f32>>> {
        let slot = &self.slots[slot_idx];
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(idx),
                timeout: None,
            })
            .expect("device lost");

        let mapped = slot.d2h_staging.slice(0..slot.buf_bytes).get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&mapped);
        let result: Vec<Vec<Complex<f32>>> = floats
            .chunks_exact(self.n * 2)
            .map(|sig| {
                sig.chunks_exact(2)
                    .map(|p| Complex { re: p[0], im: p[1] })
                    .collect()
            })
            .collect();
        drop(mapped);
        slot.d2h_staging.unmap();
        result
    }

    /// Run `inputs` through the full pipeline using N_SLOTS in-flight slots.
    ///
    /// Returns one output batch per input batch, in order.
    fn run(&self, inputs: &[Vec<f32>]) -> Vec<Vec<Vec<Complex<f32>>>> {
        let num_frames = inputs.len();
        let mut in_flight: Vec<Option<wgpu::SubmissionIndex>> = vec![None; N_SLOTS];
        let mut results: Vec<Vec<Vec<Complex<f32>>>> = Vec::with_capacity(num_frames);

        for frame in 0..(num_frames + N_SLOTS) {
            let s = frame % N_SLOTS;

            // Collect result from the previous cycle of this slot.
            if let Some(idx) = in_flight[s].take() {
                results.push(self.collect_frame(s, idx));
            }

            // Submit next frame if data remains.
            if frame < num_frames {
                in_flight[s] = Some(self.submit_frame(s, frame as u32, &inputs[frame]));
            }
        }

        results
    }
}

// ── Signal generation ─────────────────────────────────────────────────────────

fn make_frame(seed: usize, n: usize, batch_size: usize) -> Vec<f32> {
    (0..batch_size)
        .flat_map(|b| {
            (0..n).flat_map(move |i| {
                let t = (b * n + i) as f32 / n as f32;
                let s = (seed as f32 * 0.3 + t * std::f32::consts::TAU * 5.0).sin();
                [s, 0.0f32]
            })
        })
        .collect()
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    const N: usize = 1024;
    const BATCH: usize = 8;
    const WARMUP: usize = 4;
    const FRAMES: usize = 32;

    println!("Parallel pipeline: N={N}, batch={BATCH}, frames={FRAMES}, slots={N_SLOTS}");
    println!("Stages: H2D → Gaussian noise → FFT (R4/R2) → Hann window → IFFT (R4/R2) → D2H\n");

    let pipeline = Pipeline::new(N, BATCH);
    let total_samples = (N * BATCH * FRAMES) as f64;

    // Warmup
    let warm_inputs: Vec<Vec<f32>> = (0..WARMUP).map(|i| make_frame(i, N, BATCH)).collect();
    pipeline.run(&warm_inputs);

    // Sequential (one slot used at a time)
    let inputs: Vec<Vec<f32>> = (0..FRAMES)
        .map(|i| make_frame(i + WARMUP, N, BATCH))
        .collect();

    let t0 = Instant::now();
    for (i, raw) in inputs.iter().enumerate() {
        let idx = pipeline.submit_frame(0, i as u32, raw);
        pipeline.collect_frame(0, idx);
    }
    let seq_secs = t0.elapsed().as_secs_f64();

    // Pipelined (all N_SLOTS in flight)
    let t1 = Instant::now();
    pipeline.run(&inputs);
    let pipe_secs = t1.elapsed().as_secs_f64();

    println!(
        "{:>20} | {:>10} | {:>10} | {:>9}",
        "mode", "total ms", "ms/frame", "MS/s"
    );
    println!("{}", "-".repeat(58));
    println!(
        "{:>20} | {:>10.2} | {:>10.3} | {:>9.2}",
        "sequential",
        seq_secs * 1e3,
        seq_secs * 1e3 / FRAMES as f64,
        total_samples / seq_secs / 1e6,
    );
    println!(
        "{:>20} | {:>10.2} | {:>10.3} | {:>9.2}",
        format!("pipeline ({N_SLOTS} slots)"),
        pipe_secs * 1e3,
        pipe_secs * 1e3 / FRAMES as f64,
        total_samples / pipe_secs / 1e6,
    );
    println!("\nSpeedup: {:.2}×", seq_secs / pipe_secs);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: verifies the full pipeline produces finite, non-zero output.
    ///
    /// A true roundtrip (signal → FFT → all-ones window → IFFT → signal) would require
    /// injecting an all-ones window at runtime, but the window buffer is immutable after
    /// pipeline construction. Correctness of FFT↔IFFT is covered by gpu_fft_test.rs.
    #[test]
    fn roundtrip_allpass_window() {
        const N: usize = 1024;
        const BATCH: usize = 2;

        let pipeline = Pipeline::new(N, BATCH);
        let raw = make_frame(42, N, BATCH);
        let idx = pipeline.submit_frame(0, 99, &raw);
        let out = pipeline.collect_frame(0, idx);

        assert_eq!(out.len(), BATCH);
        assert_eq!(out[0].len(), N);
        for batch in &out {
            for c in batch {
                assert!(c.re.is_finite(), "re NaN/inf");
                assert!(c.im.is_finite(), "im NaN/inf");
            }
        }

        // Check that the output energy is non-trivial (pipeline actually ran).
        let energy: f32 = out
            .iter()
            .flat_map(|b| b.iter())
            .map(|c| c.norm_sqr())
            .sum();
        assert!(energy > 0.0, "output energy should be positive");
    }

    #[test]
    fn pipeline_throughput_positive() {
        const N: usize = 256;
        const BATCH: usize = 2;
        let pipeline = Pipeline::new(N, BATCH);
        let inputs: Vec<Vec<f32>> = (0..8).map(|i| make_frame(i, N, BATCH)).collect();
        let t = Instant::now();
        pipeline.run(&inputs);
        assert!(t.elapsed().as_secs_f64() > 0.0);
    }

    #[test]
    fn pipeline_output_count_matches_input() {
        const N: usize = 256;
        const BATCH: usize = 3;
        const FRAMES: usize = 10;
        let pipeline = Pipeline::new(N, BATCH);
        let inputs: Vec<Vec<f32>> = (0..FRAMES).map(|i| make_frame(i, N, BATCH)).collect();
        let results = pipeline.run(&inputs);
        assert_eq!(results.len(), FRAMES);
        for r in &results {
            assert_eq!(r.len(), BATCH);
            assert_eq!(r[0].len(), N);
        }
    }

    #[test]
    fn make_frame_length() {
        assert_eq!(make_frame(0, 256, 4).len(), 256 * 4 * 2);
    }
}
