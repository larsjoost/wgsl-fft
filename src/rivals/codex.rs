use std::any::Any;
use std::cell::RefCell;
use std::num::NonZeroU64;

use num_complex::Complex;

use crate::error::Result;
use crate::FftExecutor;

// ── WGSL: Single-dispatch local-memory radix-4 FFT for N = 256 ───────────────
//
// One workgroup handles one full 256-point FFT using workgroup memory only.
// This removes all intermediate global ping-pong traffic and collapses the
// transform to a single dispatch for the smallest leaderboard size.
const CODEX_LOCAL_256_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> U: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> SRC: array<f32>;
@group(0) @binding(2) var<storage, read_write> DST: array<f32>;
@group(0) @binding(3) var<storage, read> TWIDDLE: array<f32>;

var<workgroup> BUF0: array<vec2<f32>, 256>;
var<workgroup> BUF1: array<vec2<f32>, 256>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn radix4(x0: vec2<f32>, x1: vec2<f32>, x2: vec2<f32>, x3: vec2<f32>) -> array<vec2<f32>, 4> {
    let y0 = x0 + x1 + x2 + x3;
    let y1 = vec2<f32>(x0.x + x1.y - x2.x - x3.y, x0.y - x1.x - x2.y + x3.x);
    let y2 = x0 - x1 + x2 - x3;
    let y3 = vec2<f32>(x0.x - x1.y - x2.x + x3.y, x0.y + x1.x - x2.y - x3.x);
    return array<vec2<f32>, 4>(y0, y1, y2, y3);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = lid.x;
    let batch_id = gid.y;
    let n = U.x;
    if n != 256u {
        return;
    }
    let batch_offset = batch_id * 512u;

    let g0 = batch_offset + 2u * tid;
    let g1 = g0 + 128u;
    let g2 = g0 + 256u;
    let g3 = g0 + 384u;

    BUF0[tid] = vec2<f32>(SRC[g0], SRC[g0 + 1u]);
    BUF0[tid + 64u] = vec2<f32>(SRC[g1], SRC[g1 + 1u]);
    BUF0[tid + 128u] = vec2<f32>(SRC[g2], SRC[g2 + 1u]);
    BUF0[tid + 192u] = vec2<f32>(SRC[g3], SRC[g3 + 1u]);
    workgroupBarrier();

    {
        let out = radix4(BUF0[tid], BUF0[tid + 64u], BUF0[tid + 128u], BUF0[tid + 192u]);
        let base = tid << 2u;
        BUF1[base] = out[0];
        BUF1[base + 1u] = out[1];
        BUF1[base + 2u] = out[2];
        BUF1[base + 3u] = out[3];
    }
    workgroupBarrier();

    {
        let k = tid & 3u;
        let tw1 = 16u * k;
        let tw2 = tw1 << 1u;
        let tw3 = tw1 * 3u;
        let out = radix4(
            BUF1[tid],
            cmul(BUF1[tid + 64u], vec2<f32>(TWIDDLE[2u * tw1], TWIDDLE[2u * tw1 + 1u])),
            cmul(BUF1[tid + 128u], vec2<f32>(TWIDDLE[2u * tw2], TWIDDLE[2u * tw2 + 1u])),
            cmul(BUF1[tid + 192u], vec2<f32>(TWIDDLE[2u * tw3], TWIDDLE[2u * tw3 + 1u])),
        );
        let j = tid >> 2u;
        let base = (j << 4u) + k;
        BUF0[base] = out[0];
        BUF0[base + 4u] = out[1];
        BUF0[base + 8u] = out[2];
        BUF0[base + 12u] = out[3];
    }
    workgroupBarrier();

    {
        let k = tid & 15u;
        let tw1 = 4u * k;
        let tw2 = tw1 << 1u;
        let tw3 = tw1 * 3u;
        let out = radix4(
            BUF0[tid],
            cmul(BUF0[tid + 64u], vec2<f32>(TWIDDLE[2u * tw1], TWIDDLE[2u * tw1 + 1u])),
            cmul(BUF0[tid + 128u], vec2<f32>(TWIDDLE[2u * tw2], TWIDDLE[2u * tw2 + 1u])),
            cmul(BUF0[tid + 192u], vec2<f32>(TWIDDLE[2u * tw3], TWIDDLE[2u * tw3 + 1u])),
        );
        let j = tid >> 4u;
        let base = (j << 6u) + k;
        BUF1[base] = out[0];
        BUF1[base + 16u] = out[1];
        BUF1[base + 32u] = out[2];
        BUF1[base + 48u] = out[3];
    }
    workgroupBarrier();

    {
        let out = radix4(
            BUF1[tid],
            cmul(BUF1[tid + 64u], vec2<f32>(TWIDDLE[2u * tid], TWIDDLE[2u * tid + 1u])),
            cmul(BUF1[tid + 128u], vec2<f32>(TWIDDLE[4u * tid], TWIDDLE[4u * tid + 1u])),
            cmul(BUF1[tid + 192u], vec2<f32>(TWIDDLE[6u * tid], TWIDDLE[6u * tid + 1u])),
        );
        BUF0[tid] = out[0];
        BUF0[tid + 64u] = out[1];
        BUF0[tid + 128u] = out[2];
        BUF0[tid + 192u] = out[3];
    }
    workgroupBarrier();

    let y0 = BUF0[tid];
    let y1 = BUF0[tid + 64u];
    let y2 = BUF0[tid + 128u];
    let y3 = BUF0[tid + 192u];

    DST[g0] = y0.x;
    DST[g0 + 1u] = y0.y;
    DST[g1] = y1.x;
    DST[g1 + 1u] = y1.y;
    DST[g2] = y2.x;
    DST[g2 + 1u] = y2.y;
    DST[g3] = y3.x;
    DST[g3 + 1u] = y3.y;
}
"#;

// ── WGSL: Stockham Radix-8 DIT ───────────────────────────────────────────────
//
// Uniform U: .x = N, .y = p, .z = twiddle stride
const CODEX_R8_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> U: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> SRC: array<f32>;
@group(0) @binding(2) var<storage, read_write> DST: array<f32>;
@group(0) @binding(3) var<storage, read> TWIDDLE: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let batch_id = gid.y;
    let n = U.x;
    let eighth_n = n >> 3u;
    if tid >= eighth_n { return; }

    let p = U.y;
    let tw_stride = U.z;
    let eight_p = p << 3u;

    let k = tid % p;
    let j = tid / p;
    let bo = batch_id * n * 2u;

    let i0 = j * p + k;
    let i1 = i0 + eighth_n;      let i2 = i0 + 2u * eighth_n;  let i3 = i0 + 3u * eighth_n;
    let i4 = i0 + 4u * eighth_n; let i5 = i0 + 5u * eighth_n;  let i6 = i0 + 6u * eighth_n;  let i7 = i0 + 7u * eighth_n;

    let s0 = bo + 2u * i0; let s1 = bo + 2u * i1; let s2 = bo + 2u * i2; let s3 = bo + 2u * i3;
    let s4 = bo + 2u * i4; let s5 = bo + 2u * i5; let s6 = bo + 2u * i6; let s7 = bo + 2u * i7;

    let x0r = SRC[s0]; let x0i = SRC[s0 + 1u];
    let x1r = SRC[s1]; let x1i = SRC[s1 + 1u];
    let x2r = SRC[s2]; let x2i = SRC[s2 + 1u];
    let x3r = SRC[s3]; let x3i = SRC[s3 + 1u];
    let x4r = SRC[s4]; let x4i = SRC[s4 + 1u];
    let x5r = SRC[s5]; let x5i = SRC[s5 + 1u];
    let x6r = SRC[s6]; let x6i = SRC[s6 + 1u];
    let x7r = SRC[s7]; let x7i = SRC[s7 + 1u];

    let tw1 = k * tw_stride; let tw2 = 2u * tw1; let tw3 = 3u * tw1;
    let tw4 = 4u * tw1; let tw5 = 5u * tw1; let tw6 = 6u * tw1; let tw7 = 7u * tw1;

    let b0r = x0r; let b0i = x0i;

    let wr1 = TWIDDLE[2u * tw1]; let wi1 = TWIDDLE[2u * tw1 + 1u];
    let b1r = wr1 * x1r - wi1 * x1i; let b1i = wr1 * x1i + wi1 * x1r;
    let wr2 = TWIDDLE[2u * tw2]; let wi2 = TWIDDLE[2u * tw2 + 1u];
    let b2r = wr2 * x2r - wi2 * x2i; let b2i = wr2 * x2i + wi2 * x2r;
    let wr3 = TWIDDLE[2u * tw3]; let wi3 = TWIDDLE[2u * tw3 + 1u];
    let b3r = wr3 * x3r - wi3 * x3i; let b3i = wr3 * x3i + wi3 * x3r;
    let wr4 = TWIDDLE[2u * tw4]; let wi4 = TWIDDLE[2u * tw4 + 1u];
    let b4r = wr4 * x4r - wi4 * x4i; let b4i = wr4 * x4i + wi4 * x4r;
    let wr5 = TWIDDLE[2u * tw5]; let wi5 = TWIDDLE[2u * tw5 + 1u];
    let b5r = wr5 * x5r - wi5 * x5i; let b5i = wr5 * x5i + wi5 * x5r;
    let wr6 = TWIDDLE[2u * tw6]; let wi6 = TWIDDLE[2u * tw6 + 1u];
    let b6r = wr6 * x6r - wi6 * x6i; let b6i = wr6 * x6i + wi6 * x6r;
    let wr7 = TWIDDLE[2u * tw7]; let wi7 = TWIDDLE[2u * tw7 + 1u];
    let b7r = wr7 * x7r - wi7 * x7i; let b7i = wr7 * x7i + wi7 * x7r;

    let s04r = b0r + b4r; let s04i = b0i + b4i;
    let d04r = b0r - b4r; let d04i = b0i - b4i;
    let s26r = b2r + b6r; let s26i = b2i + b6i;
    let d26r = b2r - b6r; let d26i = b2i - b6i;

    let e0r = s04r + s26r; let e0i = s04i + s26i;
    let e2r = s04r - s26r; let e2i = s04i - s26i;
    let e1r = d04r + d26i; let e1i = d04i - d26r;
    let e3r = d04r - d26i; let e3i = d04i + d26r;

    let s15r = b1r + b5r; let s15i = b1i + b5i;
    let d15r = b1r - b5r; let d15i = b1i - b5i;
    let s37r = b3r + b7r; let s37i = b3i + b7i;
    let d37r = b3r - b7r; let d37i = b3i - b7i;

    let o0r = s15r + s37r; let o0i = s15i + s37i;
    let o2r = s15r - s37r; let o2i = s15i - s37i;
    let o1r = d15r + d37i; let o1i = d15i - d37r;
    let o3r = d15r - d37i; let o3i = d15i + d37r;

    let s = 0.70710678118654752;
    let w1o1r = (o1r + o1i) * s; let w1o1i = (o1i - o1r) * s;
    let w2o2r = o2i; let w2o2i = -o2r;
    let w3o3r = (-o3r + o3i) * s; let w3o3i = -(o3r + o3i) * s;

    let y0r = e0r + o0r; let y0i = e0i + o0i;
    let y1r = e1r + w1o1r; let y1i = e1i + w1o1i;
    let y2r = e2r + w2o2r; let y2i = e2i + w2o2i;
    let y3r = e3r + w3o3r; let y3i = e3i + w3o3i;
    let y4r = e0r - o0r; let y4i = e0i - o0i;
    let y5r = e1r - w1o1r; let y5i = e1i - w1o1i;
    let y6r = e2r - w2o2r; let y6i = e2i - w2o2i;
    let y7r = e3r - w3o3r; let y7i = e3i - w3o3i;

    let d0 = bo + 2u * (j * eight_p + k);
    let d1 = d0 + 2u * p; let d2 = d0 + 4u * p; let d3 = d0 + 6u * p;
    let d4 = d0 + 8u * p; let d5 = d0 + 10u * p; let d6 = d0 + 12u * p; let d7 = d0 + 14u * p;

    DST[d0] = y0r; DST[d0 + 1u] = y0i;
    DST[d1] = y1r; DST[d1 + 1u] = y1i;
    DST[d2] = y2r; DST[d2 + 1u] = y2i;
    DST[d3] = y3r; DST[d3 + 1u] = y3i;
    DST[d4] = y4r; DST[d4 + 1u] = y4i;
    DST[d5] = y5r; DST[d5 + 1u] = y5i;
    DST[d6] = y6r; DST[d6 + 1u] = y6i;
    DST[d7] = y7r; DST[d7 + 1u] = y7i;
}
"#;

// ── WGSL: Stockham Radix-4 DIT ───────────────────────────────────────────────
//
// Uniform U: .x = N, .y = p, .z = twiddle stride
//
// Source indices (natural-order Stockham read):
//   i0 = j*p+k,  i1 = i0+N/4,  i2 = i0+N/2,  i3 = i0+3N/4
//
// Twiddle indices into the N-entry table (max tw3 < 3N/4 < N — always in bounds):
//   tw1 = k*tw_stride,  tw2 = 2*tw1,  tw3 = 3*tw1
//
// Correct radix-4 DIT butterfly (b = W_tw1·x1, c = W_tw2·x2, d = W_tw3·x3):
//   y0 = x0 + b + c + d
//   y1 = x0 - i·b - c + i·d  ->  re: x0r+bi-cr-di,  im: x0i-br-ci+dr
//   y2 = x0 - b + c - d
//   y3 = x0 + i·b - c - i·d  ->  re: x0r-bi-cr+di,  im: x0i+br-ci-dr
//
// Output indices (Stockham autosort):
//   o0 = j*4p+k,  o1 = o0+p,  o2 = o0+2p,  o3 = o0+3p
const CODEX_R4_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> U: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> SRC: array<f32>;
@group(0) @binding(2) var<storage, read_write> DST: array<f32>;
@group(0) @binding(3) var<storage, read> TWIDDLE: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let batch_id = gid.y;
    let n = U.x;
    let quarter_n = n >> 2u;
    if tid >= quarter_n {
        return;
    }

    let p = U.y;
    let tw_stride = U.z;
    let four_p = p << 2u;

    let k = tid % p;
    let j = tid / p;

    let batch_offset = batch_id * n * 2u;

    let i0 = j * p + k;
    let i1 = i0 + quarter_n;
    let i2 = i0 + quarter_n + quarter_n;
    let i3 = i2 + quarter_n;

    let s0 = batch_offset + 2u * i0;
    let s1 = batch_offset + 2u * i1;
    let s2 = batch_offset + 2u * i2;
    let s3 = batch_offset + 2u * i3;

    let x0r = SRC[s0];
    let x0i = SRC[s0 + 1u];
    let x1r = SRC[s1];
    let x1i = SRC[s1 + 1u];
    let x2r = SRC[s2];
    let x2i = SRC[s2 + 1u];
    let x3r = SRC[s3];
    let x3i = SRC[s3 + 1u];

    let tw1 = k * tw_stride;
    let tw2 = tw1 * 2u;
    let tw3 = tw1 * 3u;

    let wr1 = TWIDDLE[2u * tw1];
    let wi1 = TWIDDLE[2u * tw1 + 1u];
    let wr2 = TWIDDLE[2u * tw2];
    let wi2 = TWIDDLE[2u * tw2 + 1u];
    let wr3 = TWIDDLE[2u * tw3];
    let wi3 = TWIDDLE[2u * tw3 + 1u];

    let br = wr1 * x1r - wi1 * x1i;
    let bi = wr1 * x1i + wi1 * x1r;
    let cr = wr2 * x2r - wi2 * x2i;
    let ci = wr2 * x2i + wi2 * x2r;
    let dr = wr3 * x3r - wi3 * x3i;
    let di = wr3 * x3i + wi3 * x3r;

    let o0 = j * four_p + k;
    let o1 = o0 + p;
    let o2 = o0 + p + p;
    let o3 = o2 + p;

    let d0 = batch_offset + 2u * o0;
    let d1 = batch_offset + 2u * o1;
    let d2 = batch_offset + 2u * o2;
    let d3 = batch_offset + 2u * o3;

    DST[d0] = x0r + br + cr + dr;
    DST[d0 + 1u] = x0i + bi + ci + di;
    DST[d1] = x0r + bi - cr - di;
    DST[d1 + 1u] = x0i - br - ci + dr;
    DST[d2] = x0r - br + cr - dr;
    DST[d2 + 1u] = x0i - bi + ci - di;
    DST[d3] = x0r - bi - cr + di;
    DST[d3 + 1u] = x0i + br - ci - dr;
}
"#;

// ── WGSL: Stockham Radix-2 (finalisation for odd log2N) ─────────────────────
//
// Identical to the baseline stockham kernel, but `p` and `tw_stride` are
// passed directly in the uniform instead of being reconstructed from a stage id.
const CODEX_R2_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> U: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> SRC: array<f32>;
@group(0) @binding(2) var<storage, read_write> DST: array<f32>;
@group(0) @binding(3) var<storage, read> TWIDDLE: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let batch_id = gid.y;
    let n = U.x;
    let half_n = n >> 1u;
    if tid >= half_n {
        return;
    }

    let p = U.y;
    let tw_stride = U.z;
    let two_p = p + p;

    let k = tid % p;
    let j = tid / p;

    let batch_offset = batch_id * n * 2u;

    let i1 = j * p + k;
    let i2 = i1 + half_n;

    let src1 = batch_offset + 2u * i1;
    let src2 = batch_offset + 2u * i2;

    let re1 = SRC[src1];
    let im1 = SRC[src1 + 1u];
    let re2 = SRC[src2];
    let im2 = SRC[src2 + 1u];

    let twiddle_idx = k * tw_stride;
    let wr = TWIDDLE[2u * twiddle_idx];
    let wi = TWIDDLE[2u * twiddle_idx + 1u];

    let tr = wr * re2 - wi * im2;
    let ti = wr * im2 + wi * re2;

    let out1 = j * two_p + k;
    let out2 = out1 + p;

    let dst1 = batch_offset + 2u * out1;
    let dst2 = batch_offset + 2u * out2;

    DST[dst1] = re1 + tr;
    DST[dst1 + 1u] = im1 + ti;
    DST[dst2] = re1 - tr;
    DST[dst2 + 1u] = im1 - ti;
}
"#;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    n: u32,
    p: u32,
    tw_stride: u32,
    _pad: u32,
}

#[derive(Clone)]
struct CodexCache {
    buf_a: wgpu::Buffer,
    buf_b: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    #[allow(dead_code)]
    twiddle_buf: wgpu::Buffer,
    local_256_bg: Option<wgpu::BindGroup>,
    stage_bgs_r8: Vec<wgpu::BindGroup>,
    stage_bg_r4: Option<wgpu::BindGroup>,
    stage_bg_r2: Option<wgpu::BindGroup>,
    wg_n8: u32,
    wg_n4: u32,
    wg_n2: u32,
    result_in_b: bool,
}

pub struct CodexFft {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline_local_256: wgpu::ComputePipeline,
    pipeline_r8: wgpu::ComputePipeline,
    pipeline_r4: wgpu::ComputePipeline,
    pipeline_r2: wgpu::ComputePipeline,
    cache: RefCell<std::collections::HashMap<usize, CodexCache>>,
}

impl CodexFft {
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
            ..Default::default()
        }))
        .expect("no wgpu device");

        let compile = |src: String, label: &str| {
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

        let pipeline_local_256 = compile(CODEX_LOCAL_256_WGSL.to_string(), "codex_local_256");
        let pipeline_r8 = compile(CODEX_R8_WGSL.to_string(), "codex_r8");
        let pipeline_r4 = compile(CODEX_R4_WGSL.to_string(), "codex_r4");
        let pipeline_r2 = compile(CODEX_R2_WGSL.to_string(), "codex_r2");

        Self {
            device,
            queue,
            pipeline_local_256,
            pipeline_r8,
            pipeline_r4,
            pipeline_r2,
            cache: RefCell::new(std::collections::HashMap::new()),
        }
    }

    fn build_cache(&self, n: usize, log_n: u32) -> CodexCache {
        let num_r8 = (log_n / 3) as usize;
        let rem = log_n % 3;
        let has_r4 = rem == 2;
        let has_r2 = rem == 1;
        let total_stages = num_r8 + has_r4 as usize + has_r2 as usize;

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
            "codex_buf_a",
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let buf_b = make_buf(
            "codex_buf_b",
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let staging_buf = make_buf(
            "codex_staging",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        // N-entry twiddle table: e^{-2πij/N} for j = 0..N.
        // Max radix-4 twiddle index ≈ 3N/4 < N, so this table is always sufficient.
        let twiddles: Vec<f32> = (0..n)
            .flat_map(|j| {
                let angle = -std::f32::consts::TAU * j as f32 / n as f32;
                [angle.cos(), angle.sin()]
            })
            .collect();
        let twiddle_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("codex_twiddles"),
            size: (twiddles.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&twiddle_buf, 0, bytemuck::cast_slice(&twiddles));

        let alignment = self.device.limits().min_uniform_buffer_offset_alignment as u64;
        let entry_bytes = std::mem::size_of::<Uniforms>() as u64;
        let stride = entry_bytes.div_ceil(alignment) * alignment;

        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("codex_uniforms"),
            size: stride * total_stages as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut slot = 0usize;
        for s in 0..num_r8 {
            let p = 1u32 << (3 * s as u32);
            self.queue.write_buffer(
                &uniform_buf,
                stride * slot as u64,
                bytemuck::bytes_of(&Uniforms {
                    n: n as u32,
                    p,
                    tw_stride: (n as u32 / 8) / p,
                    _pad: 0,
                }),
            );
            slot += 1;
        }
        if has_r4 {
            let p = 1u32 << (3 * num_r8 as u32);
            self.queue.write_buffer(
                &uniform_buf,
                stride * slot as u64,
                bytemuck::bytes_of(&Uniforms {
                    n: n as u32,
                    p,
                    tw_stride: (n as u32 / 4) / p,
                    _pad: 0,
                }),
            );
            slot += 1;
        }
        if has_r2 {
            let p = 1u32 << (3 * num_r8 as u32 + if has_r4 { 2 } else { 0 });
            self.queue.write_buffer(
                &uniform_buf,
                stride * slot as u64,
                bytemuck::bytes_of(&Uniforms {
                    n: n as u32,
                    p,
                    tw_stride: (n as u32 / 2) / p,
                    _pad: 0,
                }),
            );
        }

        let uniform_size = NonZeroU64::new(entry_bytes);
        let make_bg = |pipeline: &wgpu::ComputePipeline,
                       src: &wgpu::Buffer,
                       dst: &wgpu::Buffer,
                       offset: u64| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &uniform_buf,
                            offset,
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

        let stage_bgs_r8: Vec<wgpu::BindGroup> = (0..num_r8)
            .map(|s| {
                let (src, dst) = if s % 2 == 0 {
                    (&buf_a, &buf_b)
                } else {
                    (&buf_b, &buf_a)
                };
                make_bg(&self.pipeline_r8, src, dst, stride * s as u64)
            })
            .collect();

        let mut cur_slot = num_r8;

        let stage_bg_r4 = if has_r4 {
            let (src, dst) = if cur_slot % 2 == 0 {
                (&buf_a, &buf_b)
            } else {
                (&buf_b, &buf_a)
            };
            let bg = make_bg(&self.pipeline_r4, src, dst, stride * cur_slot as u64);
            cur_slot += 1;
            Some(bg)
        } else {
            None
        };

        let local_256_bg = if n == 256 {
            Some(make_bg(&self.pipeline_local_256, &buf_a, &buf_b, 0))
        } else {
            None
        };

        let stage_bg_r2 = if has_r2 {
            let (src, dst) = if cur_slot % 2 == 0 {
                (&buf_a, &buf_b)
            } else {
                (&buf_b, &buf_a)
            };
            Some(make_bg(
                &self.pipeline_r2,
                src,
                dst,
                stride * cur_slot as u64,
            ))
        } else {
            None
        };

        CodexCache {
            buf_a,
            buf_b,
            staging_buf,
            twiddle_buf,
            local_256_bg,
            stage_bgs_r8,
            stage_bg_r4,
            stage_bg_r2,
            wg_n8: (n as u32 / 8).div_ceil(256),
            wg_n4: (n as u32 / 4).div_ceil(256),
            wg_n2: (n as u32 / 2).div_ceil(256),
            result_in_b: if n == 256 {
                true
            } else {
                total_stages % 2 == 1
            },
        }
    }

    fn get_or_build_cache(&self, n: usize, log_n: u32) -> CodexCache {
        let mut map = self.cache.borrow_mut();
        if let Some(c) = map.get(&n) {
            return c.clone();
        }
        let c = self.build_cache(n, log_n);
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
        assert!(
            n.is_power_of_two() && n > 0,
            "FFT size must be a non-zero power of two"
        );
        let log_n = n.trailing_zeros();
        let batch_size = inputs.len() as u32;

        let cache = self.get_or_build_cache(n, log_n);

        let mut raw = vec![0.0f32; n * 2 * inputs.len()];
        for (batch_idx, input) in inputs.iter().enumerate() {
            assert_eq!(input.len(), n, "all inputs must have the same length");
            let base = batch_idx * n * 2;
            for (i, c) in input.iter().enumerate() {
                let p = base + i * 2;
                raw[p] = c.re;
                raw[p + 1] = if inverse { -c.im } else { c.im };
            }
        }
        self.queue
            .write_buffer(&cache.buf_a, 0, bytemuck::cast_slice(&raw));

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("codex_fft"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("codex_fft_compute"),
                timestamp_writes: None,
            });
            if let Some(local_256_bg) = &cache.local_256_bg {
                pass.set_pipeline(&self.pipeline_local_256);
                pass.set_bind_group(0, local_256_bg, &[]);
                pass.dispatch_workgroups(1, batch_size, 1);
            } else {
                for bg in &cache.stage_bgs_r8 {
                    pass.set_pipeline(&self.pipeline_r8);
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(cache.wg_n8, batch_size, 1);
                }
                if let Some(r4_bg) = &cache.stage_bg_r4 {
                    pass.set_pipeline(&self.pipeline_r4);
                    pass.set_bind_group(0, r4_bg, &[]);
                    pass.dispatch_workgroups(cache.wg_n4, batch_size, 1);
                }
                if let Some(r2_bg) = &cache.stage_bg_r2 {
                    pass.set_pipeline(&self.pipeline_r2);
                    pass.set_bind_group(0, r2_bg, &[]);
                    pass.dispatch_workgroups(cache.wg_n2, batch_size, 1);
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
        let mut output = vec![Complex::new(0.0f32, 0.0f32); n * batch_size as usize];
        for (i, c) in output.iter_mut().enumerate() {
            let j = i * 2;
            c.re = floats[j];
            c.im = floats[j + 1];
        }
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

impl FftExecutor for CodexFft {
    fn name(&self) -> &str {
        "Codex (Stockham Radix-8/4/2 Mixed, HW-Preferred)"
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
