use std::any::Any;
use std::cell::RefCell;
use std::num::NonZeroU64;

use num_complex::Complex;

use crate::error::Result;
use crate::FftExecutor;

/// Stockham Radix-8 DIT
const GEMINI_R8_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> U: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> SRC: array<f32>;
@group(0) @binding(2) var<storage, read_write> DST: array<f32>;
@group(0) @binding(3) var<storage, read> TWIDDLE: array<f32>;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid      = gid.x;
    let batch_id = gid.y;
    let n        = U.x;
    let eighth_n  = n >> 3u;
    if tid >= eighth_n { return; }

    let p = U.y;
    let eight_p = p << 3u;
    let k = tid % p;
    let j = tid / p;

    let bo = batch_id * n * 2u;

    // Source indices (Stockham DIT read)
    let i0 = j*p + k;
    let i1 = i0 + eighth_n;      let i2 = i0 + 2u*eighth_n;  let i3 = i0 + 3u*eighth_n;
    let i4 = i0 + 4u*eighth_n;   let i5 = i0 + 5u*eighth_n;  let i6 = i0 + 6u*eighth_n;  let i7 = i0 + 7u*eighth_n;

    var x: array<vec2<f32>, 8>;
    x[0] = vec2<f32>(SRC[bo + 2u*i0], SRC[bo + 2u*i0 + 1u]);
    x[1] = vec2<f32>(SRC[bo + 2u*i1], SRC[bo + 2u*i1 + 1u]);
    x[2] = vec2<f32>(SRC[bo + 2u*i2], SRC[bo + 2u*i2 + 1u]);
    x[3] = vec2<f32>(SRC[bo + 2u*i3], SRC[bo + 2u*i3 + 1u]);
    x[4] = vec2<f32>(SRC[bo + 2u*i4], SRC[bo + 2u*i4 + 1u]);
    x[5] = vec2<f32>(SRC[bo + 2u*i5], SRC[bo + 2u*i5 + 1u]);
    x[6] = vec2<f32>(SRC[bo + 2u*i6], SRC[bo + 2u*i6 + 1u]);
    x[7] = vec2<f32>(SRC[bo + 2u*i7], SRC[bo + 2u*i7 + 1u]);

    // Apply twiddles
    let stride = eighth_n / p;
    let tw = k * stride;
    for (var m = 1u; m < 8u; m++) {
        let wr = TWIDDLE[2u * m * tw];
        let wi = TWIDDLE[2u * m * tw + 1u];
        x[m] = complex_mul(vec2<f32>(wr, wi), x[m]);
    }

    // 8-point DFT
    let s04 = x[0] + x[4]; let d04 = x[0] - x[4];
    let s26 = x[2] + x[6]; let d26 = x[2] - x[6];
    let e0 = s04 + s26; let e2 = s04 - s26;
    let e1 = vec2<f32>(d04.x + d26.y, d04.y - d26.x);
    let e3 = vec2<f32>(d04.x - d26.y, d04.y + d26.x);

    let s15 = x[1] + x[5]; let d15 = x[1] - x[5];
    let s37 = x[3] + x[7]; let d37 = x[3] - x[7];
    let o0 = s15 + s37; let o2 = s15 - s37;
    let o1 = vec2<f32>(d15.x + d37.y, d15.y - d37.x);
    let o3 = vec2<f32>(d15.x - d37.y, d15.y + d37.x);

    let s8 = 0.70710678118654752;
    let w1o1 = vec2<f32>((o1.x + o1.y) * s8, (o1.y - o1.x) * s8);
    let w2o2 = vec2<f32>(o2.y, -o2.x);
    let w3o3 = vec2<f32>((-o3.x + o3.y) * s8, -(o3.x + o3.y) * s8);

    var y: array<vec2<f32>, 8>;
    y[0] = e0 + o0;   y[4] = e0 - o0;
    y[1] = e1 + w1o1; y[5] = e1 - w1o1;
    y[2] = e2 + w2o2; y[6] = e2 - w2o2;
    y[3] = e3 + w3o3; y[7] = e3 - w3o3;

    let d_base = bo + 2u*(j * eight_p + k);
    for (var m = 0u; m < 8u; m++) {
        DST[d_base + 2u*m*p] = y[m].x;
        DST[d_base + 2u*m*p + 1u] = y[m].y;
    }
}
"#;

const GEMINI_R4_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> U: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> SRC: array<f32>;
@group(0) @binding(2) var<storage, read_write> DST: array<f32>;
@group(0) @binding(3) var<storage, read> TWIDDLE: array<f32>;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let batch_id = gid.y;
    let n = U.x;
    let quarter_n = n >> 2u;
    if (tid >= quarter_n) { return; }

    let p = U.y;
    let four_p = p << 2u;
    let k = tid % p;
    let j = tid / p;

    let bo = batch_id * n * 2u;
    let i0 = j * p + k;
    let i1 = i0 + quarter_n;
    let i2 = i1 + quarter_n;
    let i3 = i2 + quarter_n;

    var x: array<vec2<f32>, 4>;
    x[0] = vec2<f32>(SRC[bo + 2u*i0], SRC[bo + 2u*i0 + 1u]);
    x[1] = vec2<f32>(SRC[bo + 2u*i1], SRC[bo + 2u*i1 + 1u]);
    x[2] = vec2<f32>(SRC[bo + 2u*i2], SRC[bo + 2u*i2 + 1u]);
    x[3] = vec2<f32>(SRC[bo + 2u*i3], SRC[bo + 2u*i3 + 1u]);

    let stride = quarter_n / p;
    let tw = k * stride;
    x[1] = complex_mul(vec2<f32>(TWIDDLE[2u*tw], TWIDDLE[2u*tw+1u]), x[1]);
    x[2] = complex_mul(vec2<f32>(TWIDDLE[4u*tw], TWIDDLE[4u*tw+1u]), x[2]);
    x[3] = complex_mul(vec2<f32>(TWIDDLE[6u*tw], TWIDDLE[6u*tw+1u]), x[3]);

    let s02 = x[0] + x[2]; let d02 = x[0] - x[2];
    let s13 = x[1] + x[3]; let d13 = x[1] - x[3];

    var y: array<vec2<f32>, 4>;
    y[0] = s02 + s13;
    y[1] = vec2<f32>(d02.x + d13.y, d02.y - d13.x);
    y[2] = s02 - s13;
    y[3] = vec2<f32>(d02.x - d13.y, d02.y + d13.x);

    let d_base = bo + 2u*(j * four_p + k);
    for (var m = 0u; m < 4u; m++) {
        DST[d_base + 2u*m*p] = y[m].x;
        DST[d_base + 2u*m*p + 1u] = y[m].y;
    }
}
"#;

const GEMINI_R2_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> U: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> SRC: array<f32>;
@group(0) @binding(2) var<storage, read_write> DST: array<f32>;
@group(0) @binding(3) var<storage, read> TWIDDLE: array<f32>;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let batch_id = gid.y;
    let n = U.x;
    let half_n = n >> 1u;
    if (tid >= half_n) { return; }

    let p = U.y;
    let two_p = p + p;
    let k = tid % p;
    let j = tid / p;

    let bo = batch_id * n * 2u;
    let i1 = j * p + k;
    let i2 = i1 + half_n;

    let x1 = vec2<f32>(SRC[bo + 2u*i1], SRC[bo + 2u*i1 + 1u]);
    let x2 = vec2<f32>(SRC[bo + 2u*i2], SRC[bo + 2u*i2 + 1u]);

    let tw = k * (half_n / p);
    let t = complex_mul(vec2<f32>(TWIDDLE[2u*tw], TWIDDLE[2u*tw+1u]), x2);

    let d_base = bo + 2u*(j * two_p + k);
    DST[d_base] = x1.x + t.x;
    DST[d_base + 1u] = x1.y + t.y;
    DST[d_base + 2u*p] = x1.x - t.x;
    DST[d_base + 2u*p + 1u] = x1.y - t.y;
}
"#;

const GEMINI_LOCAL_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> U: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> SRC: array<f32>;
@group(0) @binding(2) var<storage, read_write> DST: array<f32>;
@group(0) @binding(3) var<storage, read> TWIDDLE: array<f32>;

var<workgroup> lds_a: array<vec2<f32>, 1024>;
var<workgroup> lds_b: array<vec2<f32>, 1024>;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let n = U.x;
    let log_n = U.z;
    let batch_id = wid.x;
    let tid = lid.x;
    let bo = batch_id * n * 2u;
    
    for (var i = tid; i < n; i += 256u) {
        lds_a[i] = vec2<f32>(SRC[bo + i * 2u], SRC[bo + i * 2u + 1u]);
    }
    workgroupBarrier();

    var ping_pong = true;
    let num_r4 = log_n / 2u;
    for (var s = 0u; s < num_r4; s++) {
        let p = 1u << (s * 2u);
        let quarter_n = n >> 2u;
        for (var i = tid; i < quarter_n; i += 256u) {
            let k = i % p;
            let j = i / p;
            let i0 = j * p + k;
            let i1 = i0 + quarter_n;
            let i2 = i1 + quarter_n;
            let i3 = i2 + quarter_n;

            var x: array<vec2<f32>, 4>;
            if (ping_pong) { x[0] = lds_a[i0]; x[1] = lds_a[i1]; x[2] = lds_a[i2]; x[3] = lds_a[i3]; } 
            else { x[0] = lds_b[i0]; x[1] = lds_b[i1]; x[2] = lds_b[i2]; x[3] = lds_b[i3]; }

            let stride = quarter_n / p;
            let tw = k * stride;
            x[1] = complex_mul(vec2<f32>(TWIDDLE[2u*tw], TWIDDLE[2u*tw+1u]), x[1]);
            x[2] = complex_mul(vec2<f32>(TWIDDLE[4u*tw], TWIDDLE[4u*tw+1u]), x[2]);
            x[3] = complex_mul(vec2<f32>(TWIDDLE[6u*tw], TWIDDLE[6u*tw+1u]), x[3]);

            let s02 = x[0] + x[2]; let d02 = x[0] - x[2];
            let s13 = x[1] + x[3]; let d13 = x[1] - x[3];

            let o0 = j * (p * 4u) + k;
            if (ping_pong) {
                lds_b[o0] = s02 + s13; lds_b[o0+p] = vec2<f32>(d02.x + d13.y, d02.y - d13.x);
                lds_b[o0+2u*p] = s02 - s13; lds_b[o0+3u*p] = vec2<f32>(d02.x - d13.y, d02.y + d13.x);
            } else {
                lds_a[o0] = s02 + s13; lds_a[o0+p] = vec2<f32>(d02.x + d13.y, d02.y - d13.x);
                lds_a[o0+2u*p] = s02 - s13; lds_a[o0+3u*p] = vec2<f32>(d02.x - d13.y, d02.y + d13.x);
            }
        }
        ping_pong = !ping_pong;
        workgroupBarrier();
    }

    if (log_n % 2u == 1u) {
        let half_n = n >> 1u;
        let p = 1u << (log_n - 1u);
        for (var i = tid; i < half_n; i += 256u) {
            let k = i % p; let j = i / p;
            let i1 = j * p + k; let i2 = i1 + half_n;
            var x1: vec2<f32>; var x2: vec2<f32>;
            if (ping_pong) { x1 = lds_a[i1]; x2 = lds_a[i2]; } else { x1 = lds_b[i1]; x2 = lds_b[i2]; }
            let t = complex_mul(vec2<f32>(TWIDDLE[k * 2u], TWIDDLE[k * 2u + 1u]), x2);
            let o1 = j * (p * 2u) + k;
            if (ping_pong) { lds_b[o1] = x1 + t; lds_b[o1+p] = x1 - t; } else { lds_a[o1] = x1 + t; lds_a[o1+p] = x1 - t; }
        }
        ping_pong = !ping_pong;
        workgroupBarrier();
    }

    for (var i = tid; i < n; i += 256u) {
        var val: vec2<f32>;
        if (ping_pong) { val = lds_a[i]; } else { val = lds_b[i]; }
        DST[bo + i * 2u] = val.x;
        DST[bo + i * 2u + 1u] = val.y;
    }
}
"#;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GeminiUniforms {
    n: u32,
    p: u32,
    log_n: u32,
    _pad: u32,
}

#[derive(Clone)]
struct GeminiCache {
    buf_a: wgpu::Buffer,
    buf_b: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    #[allow(dead_code)]
    twiddle_buf: wgpu::Buffer,
    stage_bgs: Vec<(wgpu::ComputePipeline, wgpu::BindGroup, u32)>,
    local_bg: Option<wgpu::BindGroup>,
    result_in_b: bool,
    is_local: bool,
}

pub struct GeminiFft {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline_r8: wgpu::ComputePipeline,
    pipeline_r4: wgpu::ComputePipeline,
    pipeline_r2: wgpu::ComputePipeline,
    pipeline_local: wgpu::ComputePipeline,
    cache: RefCell<std::collections::HashMap<usize, GeminiCache>>,
}

impl GeminiFft {
    pub fn new() -> Self {
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
        })
        .expect("no wgpu adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::empty(),
            ..Default::default()
        }))
        .expect("no wgpu device");
        let compile = |device: &wgpu::Device, src: &str, label: &str| {
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
        let p_r8 = compile(&device, GEMINI_R8_WGSL, "gemini_r8");
        let p_r4 = compile(&device, GEMINI_R4_WGSL, "gemini_r4");
        let p_r2 = compile(&device, GEMINI_R2_WGSL, "gemini_r2");
        let p_local = compile(&device, GEMINI_LOCAL_WGSL, "gemini_local");
        Self {
            device,
            queue,
            pipeline_r8: p_r8,
            pipeline_r4: p_r4,
            pipeline_r2: p_r2,
            pipeline_local: p_local,
            cache: RefCell::new(std::collections::HashMap::new()),
        }
    }

    fn build_cache(&self, n: usize, log_n: u32) -> GeminiCache {
        let is_local = n <= 1024;
        let num_r8 = (log_n / 3) as usize;
        let rem = log_n % 3;
        let has_r4 = rem == 2;
        let has_r2 = rem == 1;
        let total_stages = if is_local {
            1
        } else {
            num_r8 + has_r4 as usize + has_r2 as usize
        };
        let single_bytes = (n * 2 * std::mem::size_of::<f32>()) as u64;
        let max_batch =
            (self.device.limits().max_storage_buffer_binding_size as u64 / single_bytes).min(1024);
        let data_bytes = single_bytes * max_batch;
        let make_buf = |label, usage| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: data_bytes,
                usage,
                mapped_at_creation: false,
            })
        };
        let buf_a = make_buf(
            "gemini_buf_a",
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let buf_b = make_buf(
            "gemini_buf_b",
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let staging_buf = make_buf(
            "gemini_staging",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let twiddles: Vec<f32> = (0..n)
            .flat_map(|j| {
                let angle = -std::f32::consts::TAU * j as f32 / n as f32;
                [angle.cos(), angle.sin()]
            })
            .collect();
        let twiddle_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gemini_twiddles"),
            size: (twiddles.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&twiddle_buf, 0, bytemuck::cast_slice(&twiddles));
        let alignment = self.device.limits().min_uniform_buffer_offset_alignment as u64;
        let stride = (std::mem::size_of::<GeminiUniforms>() as u64).div_ceil(alignment) * alignment;
        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gemini_uniforms"),
            size: stride * total_stages as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let make_bg = |device: &wgpu::Device,
                       pipeline: &wgpu::ComputePipeline,
                       uniform_buf: &wgpu::Buffer,
                       twiddle_buf: &wgpu::Buffer,
                       src: &wgpu::Buffer,
                       dst: &wgpu::Buffer,
                       offset: u64| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: uniform_buf,
                            offset,
                            size: NonZeroU64::new(std::mem::size_of::<GeminiUniforms>() as u64),
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
        if is_local {
            self.queue.write_buffer(
                &uniform_buf,
                0,
                bytemuck::bytes_of(&GeminiUniforms {
                    n: n as u32,
                    p: 0,
                    log_n,
                    _pad: 0,
                }),
            );
            let local_bg = make_bg(
                &self.device,
                &self.pipeline_local,
                &uniform_buf,
                &twiddle_buf,
                &buf_a,
                &buf_b,
                0,
            );
            return GeminiCache {
                buf_a,
                buf_b,
                staging_buf,
                twiddle_buf,
                stage_bgs: vec![],
                local_bg: Some(local_bg),
                result_in_b: true,
                is_local: true,
            };
        }
        let mut stage_bgs = Vec::new();
        let mut slot = 0;
        for s in 0..num_r8 {
            let p = 1u32 << (s * 3);
            self.queue.write_buffer(
                &uniform_buf,
                stride * slot as u64,
                bytemuck::bytes_of(&GeminiUniforms {
                    n: n as u32,
                    p,
                    log_n,
                    _pad: 0,
                }),
            );
            let (src, dst) = if slot % 2 == 0 {
                (&buf_a, &buf_b)
            } else {
                (&buf_b, &buf_a)
            };
            stage_bgs.push((
                self.pipeline_r8.clone(),
                make_bg(
                    &self.device,
                    &self.pipeline_r8,
                    &uniform_buf,
                    &twiddle_buf,
                    src,
                    dst,
                    stride * slot as u64,
                ),
                (n as u32 / 8).div_ceil(256),
            ));
            slot += 1;
        }
        if has_r4 {
            let p = 1u32 << (num_r8 * 3);
            self.queue.write_buffer(
                &uniform_buf,
                stride * slot as u64,
                bytemuck::bytes_of(&GeminiUniforms {
                    n: n as u32,
                    p,
                    log_n,
                    _pad: 0,
                }),
            );
            let (src, dst) = if slot % 2 == 0 {
                (&buf_a, &buf_b)
            } else {
                (&buf_b, &buf_a)
            };
            stage_bgs.push((
                self.pipeline_r4.clone(),
                make_bg(
                    &self.device,
                    &self.pipeline_r4,
                    &uniform_buf,
                    &twiddle_buf,
                    src,
                    dst,
                    stride * slot as u64,
                ),
                (n as u32 / 4).div_ceil(256),
            ));
            slot += 1;
        }
        if has_r2 {
            let p = 1u32 << (num_r8 * 3 + if has_r4 { 2 } else { 0 });
            self.queue.write_buffer(
                &uniform_buf,
                stride * slot as u64,
                bytemuck::bytes_of(&GeminiUniforms {
                    n: n as u32,
                    p,
                    log_n,
                    _pad: 0,
                }),
            );
            let (src, dst) = if slot % 2 == 0 {
                (&buf_a, &buf_b)
            } else {
                (&buf_b, &buf_a)
            };
            stage_bgs.push((
                self.pipeline_r2.clone(),
                make_bg(
                    &self.device,
                    &self.pipeline_r2,
                    &uniform_buf,
                    &twiddle_buf,
                    src,
                    dst,
                    stride * slot as u64,
                ),
                (n as u32 / 2).div_ceil(256),
            ));
            slot += 1;
        }
        GeminiCache {
            buf_a,
            buf_b,
            staging_buf,
            twiddle_buf,
            stage_bgs,
            local_bg: None,
            result_in_b: slot % 2 == 1,
            is_local: false,
        }
    }

    fn get_or_build_cache(&self, n: usize) -> GeminiCache {
        let mut map = self.cache.borrow_mut();
        if let Some(c) = map.get(&n) {
            return c.clone();
        }
        let c = self.build_cache(n, n.trailing_zeros());
        map.insert(n, c.clone());
        c
    }

    fn transform_batch_internal(
        &self,
        inputs: &[Vec<Complex<f32>>],
        inverse: bool,
    ) -> Result<Vec<Vec<Complex<f32>>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let n = inputs[0].len();
        let batch_size = inputs.len() as u32;
        let cache = self.get_or_build_cache(n);
        let mut raw = Vec::with_capacity(n * 2 * inputs.len());
        for input in inputs {
            if inverse {
                raw.extend(input.iter().flat_map(|c| [c.re, -c.im]));
            } else {
                raw.extend(input.iter().flat_map(|c| [c.re, c.im]));
            }
        }
        self.queue
            .write_buffer(&cache.buf_a, 0, bytemuck::cast_slice(&raw));
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gemini_fft"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gemini_fft_compute"),
                timestamp_writes: None,
            });
            if cache.is_local {
                pass.set_pipeline(&self.pipeline_local);
                pass.set_bind_group(0, cache.local_bg.as_ref().unwrap(), &[]);
                pass.dispatch_workgroups(batch_size, 1, 1);
            } else {
                for (pipeline, bg, wg_count) in &cache.stage_bgs {
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(*wg_count, batch_size, 1);
                }
            }
        }
        let result_buf = if cache.result_in_b {
            &cache.buf_b
        } else {
            &cache.buf_a
        };
        let out_bytes = (n * 2 * std::mem::size_of::<f32>()) as u64 * batch_size as u64;
        enc.copy_buffer_to_buffer(result_buf, 0, &cache.staging_buf, 0, out_bytes);
        self.queue.submit(std::iter::once(enc.finish()));
        let slice = cache.staging_buf.slice(0..out_bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })?;
        let mapped = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&mapped);
        let mut output: Vec<Complex<f32>> = floats
            .chunks_exact(2)
            .map(|p| Complex { re: p[0], im: p[1] })
            .collect();
        drop(mapped);
        cache.staging_buf.unmap();
        if inverse {
            let scale = 1.0 / n as f32;
            for c in &mut output {
                *c = Complex {
                    re: c.re * scale,
                    im: -c.im * scale,
                };
            }
        }
        Ok(output.chunks(n).map(|ch| ch.to_vec()).collect())
    }
}

impl FftExecutor for GeminiFft {
    fn name(&self) -> &str {
        "Gemini (Mixed-Radix Stockham)"
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
