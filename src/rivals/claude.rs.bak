use std::any::Any;
use std::cell::RefCell;
use std::num::NonZeroU64;

use num_complex::Complex;
use wgsl_rs::wgsl;

use crate::FftExecutor;

// ── WGSL: Stockham Radix-8 DIT ───────────────────────────────────────────────
const CLAUDE_R8_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> U: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> SRC: array<f32>;
@group(0) @binding(2) var<storage, read_write> DST: array<f32>;
@group(0) @binding(3) var<storage, read> TWIDDLE: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid      = gid.x;
    let batch_id = gid.y;
    let n        = U.x;
    let e        = n >> 3u;
    if tid >= e { return; }

    let stage  = U.y;
    let p      = 1u << (stage * 3u);
    let eight_p = p << 3u;

    let k = tid % p;
    let j = tid / p;

    let bo = batch_id * n * 2u;

    let i0 = j*p + k;
    let i1 = i0 + e;      let i2 = i0 + 2u*e;  let i3 = i0 + 3u*e;
    let i4 = i0 + 4u*e;   let i5 = i0 + 5u*e;  let i6 = i0 + 6u*e;  let i7 = i0 + 7u*e;

    let s0 = bo + 2u*i0; let s1 = bo + 2u*i1; let s2 = bo + 2u*i2; let s3 = bo + 2u*i3;
    let s4 = bo + 2u*i4; let s5 = bo + 2u*i5; let s6 = bo + 2u*i6; let s7 = bo + 2u*i7;

    let x0r = SRC[s0]; let x0i = SRC[s0+1u];
    let x1r = SRC[s1]; let x1i = SRC[s1+1u];
    let x2r = SRC[s2]; let x2i = SRC[s2+1u];
    let x3r = SRC[s3]; let x3i = SRC[s3+1u];
    let x4r = SRC[s4]; let x4i = SRC[s4+1u];
    let x5r = SRC[s5]; let x5i = SRC[s5+1u];
    let x6r = SRC[s6]; let x6i = SRC[s6+1u];
    let x7r = SRC[s7]; let x7i = SRC[s7+1u];

    let stride = e >> (stage * 3u);
    let tw1 = k*stride; let tw2 = 2u*tw1; let tw3 = 3u*tw1;
    let tw4 = 4u*tw1;   let tw5 = 5u*tw1; let tw6 = 6u*tw1; let tw7 = 7u*tw1;

    let b0r = x0r; let b0i = x0i;

    let wr1 = TWIDDLE[2u*tw1]; let wi1 = TWIDDLE[2u*tw1+1u];
    let b1r = wr1*x1r - wi1*x1i; let b1i = wr1*x1i + wi1*x1r;

    let wr2 = TWIDDLE[2u*tw2]; let wi2 = TWIDDLE[2u*tw2+1u];
    let b2r = wr2*x2r - wi2*x2i; let b2i = wr2*x2i + wi2*x2r;

    let wr3 = TWIDDLE[2u*tw3]; let wi3 = TWIDDLE[2u*tw3+1u];
    let b3r = wr3*x3r - wi3*x3i; let b3i = wr3*x3i + wi3*x3r;

    let wr4 = TWIDDLE[2u*tw4]; let wi4 = TWIDDLE[2u*tw4+1u];
    let b4r = wr4*x4r - wi4*x4i; let b4i = wr4*x4i + wi4*x4r;

    let wr5 = TWIDDLE[2u*tw5]; let wi5 = TWIDDLE[2u*tw5+1u];
    let b5r = wr5*x5r - wi5*x5i; let b5i = wr5*x5i + wi5*x5r;

    let wr6 = TWIDDLE[2u*tw6]; let wi6 = TWIDDLE[2u*tw6+1u];
    let b6r = wr6*x6r - wi6*x6i; let b6i = wr6*x6i + wi6*x6r;

    let wr7 = TWIDDLE[2u*tw7]; let wi7 = TWIDDLE[2u*tw7+1u];
    let b7r = wr7*x7r - wi7*x7i; let b7i = wr7*x7i + wi7*x7r;

    let s04r = b0r+b4r; let s04i = b0i+b4i;
    let d04r = b0r-b4r; let d04i = b0i-b4i;
    let s26r = b2r+b6r; let s26i = b2i+b6i;
    let d26r = b2r-b6r; let d26i = b2i-b6i;

    let e0r = s04r+s26r; let e0i = s04i+s26i;
    let e2r = s04r-s26r; let e2i = s04i-s26i;
    let e1r = d04r+d26i; let e1i = d04i-d26r;
    let e3r = d04r-d26i; let e3i = d04i+d26r;

    let s15r = b1r+b5r; let s15i = b1i+b5i;
    let d15r = b1r-b5r; let d15i = b1i-b5i;
    let s37r = b3r+b7r; let s37i = b3i+b7i;
    let d37r = b3r-b7r; let d37i = b3i-b7i;

    let o0r = s15r+s37r; let o0i = s15i+s37i;
    let o2r = s15r-s37r; let o2i = s15i-s37i;
    let o1r = d15r+d37i; let o1i = d15i-d37r;
    let o3r = d15r-d37i; let o3i = d15i+d37r;

    let s = 0.70710678118654752;

    let w1o1r = (o1r+o1i)*s; let w1o1i = (o1i-o1r)*s;
    let w2o2r = o2i;  let w2o2i = -o2r;
    let w3o3r = (-o3r+o3i)*s; let w3o3i = -(o3r+o3i)*s;

    let y0r = e0r+o0r;    let y0i = e0i+o0i;
    let y1r = e1r+w1o1r;  let y1i = e1i+w1o1i;
    let y2r = e2r+w2o2r;  let y2i = e2i+w2o2i;
    let y3r = e3r+w3o3r;  let y3i = e3i+w3o3i;
    let y4r = e0r-o0r;    let y4i = e0i-o0i;
    let y5r = e1r-w1o1r;  let y5i = e1i-w1o1i;
    let y6r = e2r-w2o2r;  let y6i = e2i-w2o2i;
    let y7r = e3r-w3o3r;  let y7i = e3i-w3o3i;

    let d0 = bo + 2u*(j*eight_p + k);
    let d1 = d0 + 2u*p; let d2 = d0 + 4u*p; let d3 = d0 + 6u*p;
    let d4 = d0 + 8u*p; let d5 = d0 + 10u*p; let d6 = d0 + 12u*p; let d7 = d0 + 14u*p;

    DST[d0]=y0r; DST[d0+1u]=y0i;
    DST[d1]=y1r; DST[d1+1u]=y1i;
    DST[d2]=y2r; DST[d2+1u]=y2i;
    DST[d3]=y3r; DST[d3+1u]=y3i;
    DST[d4]=y4r; DST[d4+1u]=y4i;
    DST[d5]=y5r; DST[d5+1u]=y5i;
    DST[d6]=y6r; DST[d6+1u]=y6i;
    DST[d7]=y7r; DST[d7+1u]=y7i;
}
"#;

// ── WGSL: Stockham Radix-4 DIT ───────────────────────────────────────────────
#[wgsl]
pub mod claude_r4_kernel {
    use wgsl_rs::std::*;

    uniform!(group(0), binding(0), U: Vec4u);
    storage!(group(0), binding(1), read_write, SRC: RuntimeArray<f32>);
    storage!(group(0), binding(2), read_write, DST: RuntimeArray<f32>);
    storage!(group(0), binding(3), TWIDDLE: RuntimeArray<f32>);

    #[compute]
    #[workgroup_size(256, 1, 1)]
    pub fn main(#[builtin(global_invocation_id)] gid: Vec3u) {
        let tid = gid.x;
        let batch_id = gid.y;
        let n = get!(U).x;
        let quarter_n = n >> 2u32;
        if tid >= quarter_n {
            return;
        }

        let stage = get!(U).y;
        let p = 1u32 << (stage + stage);
        let four_p = p << 2u32;

        let k = tid % p;
        let j = tid / p;

        let batch_offset = batch_id * n * 2u32;

        let i0 = j * p + k;
        let i1 = i0 + quarter_n;
        let i2 = i0 + quarter_n + quarter_n;
        let i3 = i2 + quarter_n;

        let s0 = batch_offset + 2u32 * i0;
        let s1 = batch_offset + 2u32 * i1;
        let s2 = batch_offset + 2u32 * i2;
        let s3 = batch_offset + 2u32 * i3;

        let x0r = get!(SRC)[s0];
        let x0i = get!(SRC)[s0 + 1u32];
        let x1r = get!(SRC)[s1];
        let x1i = get!(SRC)[s1 + 1u32];
        let x2r = get!(SRC)[s2];
        let x2i = get!(SRC)[s2 + 1u32];
        let x3r = get!(SRC)[s3];
        let x3i = get!(SRC)[s3 + 1u32];

        let stride = quarter_n >> (stage + stage);
        let tw1 = k * stride;
        let tw2 = tw1 * 2u32;
        let tw3 = tw1 * 3u32;

        let wr1 = get!(TWIDDLE)[2u32 * tw1];
        let wi1 = get!(TWIDDLE)[2u32 * tw1 + 1u32];
        let wr2 = get!(TWIDDLE)[2u32 * tw2];
        let wi2 = get!(TWIDDLE)[2u32 * tw2 + 1u32];
        let wr3 = get!(TWIDDLE)[2u32 * tw3];
        let wi3 = get!(TWIDDLE)[2u32 * tw3 + 1u32];

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

        let d0 = batch_offset + 2u32 * o0;
        let d1 = batch_offset + 2u32 * o1;
        let d2 = batch_offset + 2u32 * o2;
        let d3 = batch_offset + 2u32 * o3;

        get_mut!(DST)[d0] = x0r + br + cr + dr;
        get_mut!(DST)[d0 + 1u32] = x0i + bi + ci + di;
        get_mut!(DST)[d1] = x0r + bi - cr - di;
        get_mut!(DST)[d1 + 1u32] = x0i - br - ci + dr;
        get_mut!(DST)[d2] = x0r - br + cr - dr;
        get_mut!(DST)[d2 + 1u32] = x0i - bi + ci - di;
        get_mut!(DST)[d3] = x0r - bi - cr + di;
        get_mut!(DST)[d3 + 1u32] = x0i + br - ci - dr;
    }
}

// ── WGSL: Stockham Radix-2 ───────────────────────────────────────────────────
#[wgsl]
pub mod claude_r2_kernel {
    use wgsl_rs::std::*;

    uniform!(group(0), binding(0), U: Vec4u);
    storage!(group(0), binding(1), read_write, SRC: RuntimeArray<f32>);
    storage!(group(0), binding(2), read_write, DST: RuntimeArray<f32>);
    storage!(group(0), binding(3), TWIDDLE: RuntimeArray<f32>);

    #[compute]
    #[workgroup_size(256, 1, 1)]
    pub fn main(#[builtin(global_invocation_id)] gid: Vec3u) {
        let tid = gid.x;
        let batch_id = gid.y;
        let n = get!(U).x;
        let half_n = n >> 1u32;
        if tid >= half_n {
            return;
        }

        let stage = get!(U).y;
        let p = 1u32 << stage;
        let two_p = p + p;

        let k = tid % p;
        let j = tid / p;

        let batch_offset = batch_id * n * 2u32;

        let i1 = j * p + k;
        let i2 = i1 + half_n;

        let src1 = batch_offset + 2u32 * i1;
        let src2 = batch_offset + 2u32 * i2;

        let re1 = get!(SRC)[src1];
        let im1 = get!(SRC)[src1 + 1u32];
        let re2 = get!(SRC)[src2];
        let im2 = get!(SRC)[src2 + 1u32];

        let twiddle_idx = k * (half_n >> stage);
        let wr = get!(TWIDDLE)[2u32 * twiddle_idx];
        let wi = get!(TWIDDLE)[2u32 * twiddle_idx + 1u32];

        let tr = wr * re2 - wi * im2;
        let ti = wr * im2 + wi * re2;

        let out1 = j * two_p + k;
        let out2 = out1 + p;

        let dst1 = batch_offset + 2u32 * out1;
        let dst2 = batch_offset + 2u32 * out2;

        get_mut!(DST)[dst1] = re1 + tr;
        get_mut!(DST)[dst1 + 1u32] = im1 + ti;
        get_mut!(DST)[dst2] = re1 - tr;
        get_mut!(DST)[dst2 + 1u32] = im1 - ti;
    }
}

// ── WGSL: Workgroup-local Stockham R8/4/2 (N ≤ 1024) ────────────────────────
//
// Single dispatch per batch: each workgroup processes one signal entirely in
// shared memory, eliminating all global-memory round-trips between stages.
// Uses R8 stages (log_n/3) then an optional R4 (rem==2) or R2 (rem==1) stage.
// Fewer barriers vs Gemini's local R4: N=256 → 3 vs 4, N=1024 → 4 vs 5.
const CLAUDE_LOCAL_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> U: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> SRC: array<f32>;
@group(0) @binding(2) var<storage, read_write> DST: array<f32>;
@group(0) @binding(3) var<storage, read> TWIDDLE: array<f32>;

var<workgroup> lds_a: array<vec2<f32>, 1024>;
var<workgroup> lds_b: array<vec2<f32>, 1024>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let n     = U.x;
    let log_n = U.z;
    let batch = wid.x;
    let tid   = lid.x;
    let bo    = batch * n * 2u;

    // Load signal from global memory into lds_a
    for (var i = tid; i < n; i += 256u) {
        lds_a[i] = vec2<f32>(SRC[bo + i*2u], SRC[bo + i*2u + 1u]);
    }
    workgroupBarrier();

    let num_r8 = log_n / 3u;
    let rem    = log_n % 3u;
    var pp = true;   // true = read lds_a, write lds_b

    // Radix-8 stages in shared memory
    for (var s = 0u; s < num_r8; s++) {
        let p        = 1u << (s * 3u);
        let eight_p  = p << 3u;
        let eighth_n = n >> 3u;
        let stride   = eighth_n / p;

        for (var i = tid; i < eighth_n; i += 256u) {
            let k    = i % p;
            let j    = i / p;
            let base = j*p + k;

            var x: array<vec2<f32>, 8>;
            if (pp) {
                x[0]=lds_a[base];              x[1]=lds_a[base+  eighth_n];
                x[2]=lds_a[base+2u*eighth_n];  x[3]=lds_a[base+3u*eighth_n];
                x[4]=lds_a[base+4u*eighth_n];  x[5]=lds_a[base+5u*eighth_n];
                x[6]=lds_a[base+6u*eighth_n];  x[7]=lds_a[base+7u*eighth_n];
            } else {
                x[0]=lds_b[base];              x[1]=lds_b[base+  eighth_n];
                x[2]=lds_b[base+2u*eighth_n];  x[3]=lds_b[base+3u*eighth_n];
                x[4]=lds_b[base+4u*eighth_n];  x[5]=lds_b[base+5u*eighth_n];
                x[6]=lds_b[base+6u*eighth_n];  x[7]=lds_b[base+7u*eighth_n];
            }

            // External Stockham twiddles: W_N^{m*k*stride}
            let tw = k * stride;
            for (var m = 1u; m < 8u; m++) {
                let ti = 2u * m * tw;
                x[m] = cmul(vec2<f32>(TWIDDLE[ti], TWIDDLE[ti+1u]), x[m]);
            }

            // 8-point DFT
            let s04 = x[0]+x[4]; let d04 = x[0]-x[4];
            let s26 = x[2]+x[6]; let d26 = x[2]-x[6];
            let e0  = s04+s26;
            let e2  = s04-s26;
            let e1  = vec2<f32>(d04.x+d26.y, d04.y-d26.x);
            let e3  = vec2<f32>(d04.x-d26.y, d04.y+d26.x);

            let s15 = x[1]+x[5]; let d15 = x[1]-x[5];
            let s37 = x[3]+x[7]; let d37 = x[3]-x[7];
            let o0  = s15+s37;
            let o2  = s15-s37;
            let o1  = vec2<f32>(d15.x+d37.y, d15.y-d37.x);
            let o3  = vec2<f32>(d15.x-d37.y, d15.y+d37.x);

            let sq   = 0.70710678118654752;
            let w1o1 = vec2<f32>((o1.x+o1.y)*sq, (o1.y-o1.x)*sq);
            let w2o2 = vec2<f32>(o2.y, -o2.x);
            let w3o3 = vec2<f32>((-o3.x+o3.y)*sq, -(o3.x+o3.y)*sq);

            var y: array<vec2<f32>, 8>;
            y[0]=e0+o0;   y[4]=e0-o0;
            y[1]=e1+w1o1; y[5]=e1-w1o1;
            y[2]=e2+w2o2; y[6]=e2-w2o2;
            y[3]=e3+w3o3; y[7]=e3-w3o3;

            let db = j*eight_p + k;
            if (pp) {
                for (var m = 0u; m < 8u; m++) { lds_b[db + m*p] = y[m]; }
            } else {
                for (var m = 0u; m < 8u; m++) { lds_a[db + m*p] = y[m]; }
            }
        }
        pp = !pp;
        workgroupBarrier();
    }

    // Radix-4 remainder stage (rem == 2)
    if (rem == 2u) {
        let p         = 1u << (num_r8 * 3u);
        let four_p    = p << 2u;
        let quarter_n = n >> 2u;
        let stride    = quarter_n / p;

        for (var i = tid; i < quarter_n; i += 256u) {
            let k    = i % p;
            let j    = i / p;
            let base = j*p + k;

            var x: array<vec2<f32>, 4>;
            if (pp) {
                x[0]=lds_a[base]; x[1]=lds_a[base+quarter_n];
                x[2]=lds_a[base+2u*quarter_n]; x[3]=lds_a[base+3u*quarter_n];
            } else {
                x[0]=lds_b[base]; x[1]=lds_b[base+quarter_n];
                x[2]=lds_b[base+2u*quarter_n]; x[3]=lds_b[base+3u*quarter_n];
            }

            let tw = k * stride;
            x[1] = cmul(vec2<f32>(TWIDDLE[2u*tw],   TWIDDLE[2u*tw+1u]), x[1]);
            x[2] = cmul(vec2<f32>(TWIDDLE[4u*tw],   TWIDDLE[4u*tw+1u]), x[2]);
            x[3] = cmul(vec2<f32>(TWIDDLE[6u*tw],   TWIDDLE[6u*tw+1u]), x[3]);

            let s02 = x[0]+x[2]; let d02 = x[0]-x[2];
            let s13 = x[1]+x[3]; let d13 = x[1]-x[3];

            let db = j*four_p + k;
            if (pp) {
                lds_b[db]      = s02+s13;
                lds_b[db+p]    = vec2<f32>(d02.x+d13.y, d02.y-d13.x);
                lds_b[db+2u*p] = s02-s13;
                lds_b[db+3u*p] = vec2<f32>(d02.x-d13.y, d02.y+d13.x);
            } else {
                lds_a[db]      = s02+s13;
                lds_a[db+p]    = vec2<f32>(d02.x+d13.y, d02.y-d13.x);
                lds_a[db+2u*p] = s02-s13;
                lds_a[db+3u*p] = vec2<f32>(d02.x-d13.y, d02.y+d13.x);
            }
        }
        pp = !pp;
        workgroupBarrier();
    }

    // Radix-2 remainder stage (rem == 1)
    if (rem == 1u) {
        let p      = 1u << (num_r8 * 3u);
        let two_p  = p + p;
        let half_n = n >> 1u;
        let stride = half_n / p;

        for (var i = tid; i < half_n; i += 256u) {
            let k  = i % p;
            let j  = i / p;
            let i1 = j*p + k;
            let i2 = i1 + half_n;

            var x1: vec2<f32>;
            var x2: vec2<f32>;
            if (pp) { x1=lds_a[i1]; x2=lds_a[i2]; } else { x1=lds_b[i1]; x2=lds_b[i2]; }

            let tw = k * stride;
            let t  = cmul(vec2<f32>(TWIDDLE[2u*tw], TWIDDLE[2u*tw+1u]), x2);
            let db = j*two_p + k;

            if (pp) { lds_b[db]=x1+t; lds_b[db+p]=x1-t; }
            else    { lds_a[db]=x1+t; lds_a[db+p]=x1-t; }
        }
        pp = !pp;
        workgroupBarrier();
    }

    // Write result back to global memory (DST = buf_b)
    for (var i = tid; i < n; i += 256u) {
        var val: vec2<f32>;
        if (pp) { val = lds_a[i]; } else { val = lds_b[i]; }
        DST[bo + i*2u]    = val.x;
        DST[bo + i*2u+1u] = val.y;
    }
}
"#;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    n: u32,
    stage: u32,
    log_n: u32,
    _pad: u32,
}

#[derive(Clone)]
struct ClaudeCache {
    buf_a: wgpu::Buffer,
    buf_b: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    #[allow(dead_code)]
    twiddle_buf: wgpu::Buffer,
    // Global ping-pong path (N > 1024)
    stage_bgs_r8: Vec<wgpu::BindGroup>,
    stage_bg_r4: Option<wgpu::BindGroup>,
    stage_bg_r2: Option<wgpu::BindGroup>,
    wg_n8: u32,
    wg_n4: u32,
    wg_n2: u32,
    result_in_b: bool,
    // Workgroup-local path (N ≤ 1024)
    is_local: bool,
    local_bg: Option<wgpu::BindGroup>,
}

pub struct ClaudeFft {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline_r8: wgpu::ComputePipeline,
    pipeline_r4: wgpu::ComputePipeline,
    pipeline_r2: wgpu::ComputePipeline,
    pipeline_local: wgpu::ComputePipeline,
    cache: RefCell<std::collections::HashMap<usize, ClaudeCache>>,
}

impl ClaudeFft {
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

        let compile_wgsl = |src: &str, label: &str| {
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

        let pipeline_r8 = compile_wgsl(CLAUDE_R8_WGSL, "claude_r8");
        let pipeline_r4 = compile_wgsl(
            &claude_r4_kernel::WGSL_MODULE.wgsl_source().join("\n"),
            "claude_r4",
        );
        let pipeline_r2 = compile_wgsl(
            &claude_r2_kernel::WGSL_MODULE.wgsl_source().join("\n"),
            "claude_r2",
        );
        let pipeline_local = compile_wgsl(CLAUDE_LOCAL_WGSL, "claude_local");

        Self {
            device,
            queue,
            pipeline_r8,
            pipeline_r4,
            pipeline_r2,
            pipeline_local,
            cache: RefCell::new(std::collections::HashMap::new()),
        }
    }

    fn build_cache(&self, n: usize, log_n: u32) -> ClaudeCache {
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
            "claude_buf_a",
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let buf_b = make_buf(
            "claude_buf_b",
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let staging_buf = make_buf(
            "claude_staging",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        let twiddles: Vec<f32> = (0..n)
            .flat_map(|j| {
                let angle = -std::f32::consts::TAU * j as f32 / n as f32;
                [angle.cos(), angle.sin()]
            })
            .collect();
        let twiddle_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("claude_twiddles"),
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
            label: Some("claude_uniforms"),
            size: stride * total_stages as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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

        // ── Local path (N ≤ 1024) ────────────────────────────────────────────
        if is_local {
            self.queue.write_buffer(
                &uniform_buf,
                0,
                bytemuck::bytes_of(&Uniforms {
                    n: n as u32,
                    stage: 0,
                    log_n,
                    _pad: 0,
                }),
            );
            let local_bg = make_bg(&self.pipeline_local, &buf_a, &buf_b, 0);
            return ClaudeCache {
                buf_a,
                buf_b,
                staging_buf,
                twiddle_buf,
                stage_bgs_r8: vec![],
                stage_bg_r4: None,
                stage_bg_r2: None,
                wg_n8: 0,
                wg_n4: 0,
                wg_n2: 0,
                result_in_b: true,
                is_local: true,
                local_bg: Some(local_bg),
            };
        }

        // ── Global ping-pong path (N > 1024) ─────────────────────────────────
        for s in 0..num_r8 {
            self.queue.write_buffer(
                &uniform_buf,
                stride * s as u64,
                bytemuck::bytes_of(&Uniforms {
                    n: n as u32,
                    stage: s as u32,
                    log_n,
                    _pad: 0,
                }),
            );
        }
        let mut slot = num_r8;
        if has_r4 {
            let r4_stage = (3 * num_r8 as u32) / 2;
            self.queue.write_buffer(
                &uniform_buf,
                stride * slot as u64,
                bytemuck::bytes_of(&Uniforms {
                    n: n as u32,
                    stage: r4_stage,
                    log_n,
                    _pad: 0,
                }),
            );
            slot += 1;
        }
        if has_r2 {
            let r2_stage = 3 * num_r8 as u32;
            self.queue.write_buffer(
                &uniform_buf,
                stride * slot as u64,
                bytemuck::bytes_of(&Uniforms {
                    n: n as u32,
                    stage: r2_stage,
                    log_n,
                    _pad: 0,
                }),
            );
        }

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

        ClaudeCache {
            buf_a,
            buf_b,
            staging_buf,
            twiddle_buf,
            stage_bgs_r8,
            stage_bg_r4,
            stage_bg_r2,
            wg_n8: (n as u32 / 8).div_ceil(256),
            wg_n4: (n as u32 / 4).div_ceil(256),
            wg_n2: (n as u32 / 2).div_ceil(256),
            result_in_b: total_stages % 2 == 1,
            is_local: false,
            local_bg: None,
        }
    }

    fn get_or_build_cache(&self, n: usize, log_n: u32) -> ClaudeCache {
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
    ) -> Result<Vec<Vec<Complex<f32>>>, Box<dyn std::error::Error>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let n = inputs[0].len();
        assert!(n.is_power_of_two() && n > 0);
        let log_n = n.trailing_zeros();
        let batch_size = inputs.len() as u32;

        let cache = self.get_or_build_cache(n, log_n);

        let mut raw: Vec<f32> = Vec::with_capacity(n * 2 * inputs.len());
        for input in inputs {
            assert_eq!(input.len(), n);
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
                label: Some("claude_fft"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("claude_fft_compute"),
                timestamp_writes: None,
            });

            if cache.is_local {
                // Single dispatch: entire FFT in shared memory, one workgroup per signal
                pass.set_pipeline(&self.pipeline_local);
                pass.set_bind_group(0, cache.local_bg.as_ref().unwrap(), &[]);
                pass.dispatch_workgroups(batch_size, 1, 1);
            } else {
                // Multi-pass global ping-pong
                for bg in &cache.stage_bgs_r8 {
                    pass.set_pipeline(&self.pipeline_r8);
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(cache.wg_n8, batch_size, 1);
                }
                if let Some(bg) = &cache.stage_bg_r4 {
                    pass.set_pipeline(&self.pipeline_r4);
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(cache.wg_n4, batch_size, 1);
                }
                if let Some(bg) = &cache.stage_bg_r2 {
                    pass.set_pipeline(&self.pipeline_r2);
                    pass.set_bind_group(0, bg, &[]);
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

impl FftExecutor for ClaudeFft {
    fn name(&self) -> &str {
        "Claude (R8/4/2 + Local N≤1024)"
    }

    fn fft(
        &self,
        inputs: &[Vec<Complex<f32>>],
    ) -> Result<Vec<Vec<Complex<f32>>>, Box<dyn std::error::Error>> {
        self.transform_batch_internal(inputs, false)
    }

    fn ifft(
        &self,
        inputs: &[Vec<Complex<f32>>],
    ) -> Result<Vec<Vec<Complex<f32>>>, Box<dyn std::error::Error>> {
        self.transform_batch_internal(inputs, true)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
