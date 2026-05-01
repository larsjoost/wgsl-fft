use std::any::Any;
use std::cell::RefCell;
use std::num::NonZeroU64;

use num_complex::Complex;

use crate::FftExecutor;

// ── WGSL: Stockham Radix-8 DIT with Subgroup Shuffles ──────────────────────
//
// Subgroup-optimized version that uses subgroupShuffle for intra-wave communication
// Falls back to shared memory when subgroups are not available
const MISTRAL_R8_SUBGROUP_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> U: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> SRC: array<f32>;
@group(0) @binding(2) var<storage, read_write> DST: array<f32>;
@group(0) @binding(3) var<storage, read> TWIDDLE: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid      = gid.x;
    let batch_id = gid.y;
    let n        = U.x;
    let eighth_n  = n >> 3u;
    if tid >= eighth_n { return; }

    let stage  = U.y;
    let p      = 1u << (stage * 3u);
    let eight_p = p << 3u;

    let k = tid % p;
    let j = tid / p;
    let bo = batch_id * n * 2u;

    let i0 = j*p + k;
    let i1 = i0 + eighth_n;      let i2 = i0 + 2u*eighth_n;  let i3 = i0 + 3u*eighth_n;
    let i4 = i0 + 4u*eighth_n;   let i5 = i0 + 5u*eighth_n;  let i6 = i0 + 6u*eighth_n;  let i7 = i0 + 7u*eighth_n;

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

    // External twiddles from pre-computed table
    let stride = eighth_n >> (stage * 3u);
    let tw1 = k*stride; let tw2 = 2u*tw1; let tw3 = 3u*tw1;
    let tw4 = 4u*tw1;   let tw5 = 5u*tw1; let tw6 = 6u*tw1; let tw7 = 7u*tw1;

    // b0 = x0 (W^0 = 1), bm = W_N^twm * xm
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

    // 4-point DFT of even group [b0, b2, b4, b6]
    let s04r = b0r+b4r; let s04i = b0i+b4i;
    let d04r = b0r-b4r; let d04i = b0i-b4i;
    let s26r = b2r+b6r; let s26i = b2i+b6i;
    let d26r = b2r-b6r; let d26i = b2i-b6i;

    // E[0]=s04+s26, E[2]=s04-s26, E[1]=d04+(-i)*d26, E[3]=d04+i*d26
    let e0r = s04r+s26r; let e0i = s04i+s26i;
    let e2r = s04r-s26r; let e2i = s04i-s26i;
    let e1r = d04r+d26i; let e1i = d04i-d26r;
    let e3r = d04r-d26i; let e3i = d04i+d26r;

    // 4-point DFT of odd group [b1, b3, b5, b7]
    let s15r = b1r+b5r; let s15i = b1i+b5i;
    let d15r = b1r-b5r; let d15i = b1i-b5i;
    let s37r = b3r+b7r; let s37i = b3i+b7i;
    let d37r = b3r-b7r; let d37i = b3i-b7i;

    let o0r = s15r+s37r; let o0i = s15i+s37i;
    let o2r = s15r-s37r; let o2i = s15i-s37i;
    let o1r = d15r+d37i; let o1i = d15i-d37r;
    let o3r = d15r-d37i; let o3i = d15i+d37r;

    // Combine with internal W_8^k constants (1/sqrt(2) = 0.70710678...)
    let s = 0.70710678118654752;

    // W_8^1 * o1: re=(o1r+o1i)*s, im=(o1i-o1r)*s
    let w1o1r = (o1r+o1i)*s; let w1o1i = (o1i-o1r)*s;
    // W_8^2 * o2 = -i*o2: (o2i, -o2r)
    let w2o2r = o2i;  let w2o2i = -o2r;
    // W_8^3 * o3 = -(1+i)/sqrt(2) * o3: re=(-o3r+o3i)*s, im=-(o3r+o3i)*s
    let w3o3r = (-o3r+o3i)*s; let w3o3i = -(o3r+o3i)*s;

    // Y[k] = E[k] + W_8^k * O[k],  Y[k+4] = E[k] - W_8^k * O[k]
    let y0r = e0r+o0r;    let y0i = e0i+o0i;
    let y1r = e1r+w1o1r;  let y1i = e1i+w1o1i;
    let y2r = e2r+w2o2r;  let y2i = e2i+w2o2i;
    let y3r = e3r+w3o3r;  let y3i = e3i+w3o3i;
    let y4r = e0r-o0r;    let y4i = e0i-o0i;
    let y5r = e1r-w1o1r;  let y5i = e1i-w1o1i;
    let y6r = e2r-w2o2r;  let y6i = e2i-w2o2i;
    let y7r = e3r-w3o3r;  let y7i = e3i-w3o3i;

    // Stockham output: o_m = j*eight_p + k + m*p
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

// ── WGSL: Stockham Radix-8 DIT ───────────────────────────────────────────────
//
// Uniform U: .x = N, .y = radix-8 stage index s (0 .. log_8N-1)
//
// For stage s:
//   p        = 8^s  = 1u << (s*3)
//   eighth_n  = N/8
//   stride   = eighth_n >> (s*3)   = eighth_n / p
//
// Source indices (natural-order Stockham read):
//   i0 = j*p+k,  i1 = i0+N/8,  i2 = i0+N/4,  i3 = i0+3N/8
//   i4 = i0+N/2,  i5 = i0+5N/8,  i6 = i0+3N/4,  i7 = i0+7N/8
//
// Twiddle indices into the N-entry table:
//   tw1 = k*stride,  tw2 = 2*tw1,  tw3 = 3*tw1,  tw4 = 4*tw1
//   tw5 = 5*tw1,  tw6 = 6*tw1,  tw7 = 7*tw1
//
// Output indices (Stockham autosort):
//   o0 = j*8p+k,  o1 = o0+p,  o2 = o0+2p,  o3 = o0+3p
//   o4 = o0+4p,  o5 = o0+5p,  o6 = o0+6p,  o7 = o0+7p
const MISTRAL_R8_WGSL: &str = r#"
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
    if tid >= eighth_n {
        return;
    }

    let stage = U.y;
    let p = 1u << (stage + stage + stage);
    let eight_p = p << 3u;

    let k = tid % p;
    let j = tid / p;

    let batch_offset = batch_id * n * 2u;

    let i0 = j * p + k;
    let i1 = i0 + eighth_n;
    let i2 = i0 + eighth_n + eighth_n;
    let i3 = i2 + eighth_n;
    let i4 = i0 + eighth_n + eighth_n + eighth_n + eighth_n;
    let i5 = i4 + eighth_n;
    let i6 = i4 + eighth_n + eighth_n;
    let i7 = i6 + eighth_n;

    let s0 = batch_offset + 2u * i0;
    let s1 = batch_offset + 2u * i1;
    let s2 = batch_offset + 2u * i2;
    let s3 = batch_offset + 2u * i3;
    let s4 = batch_offset + 2u * i4;
    let s5 = batch_offset + 2u * i5;
    let s6 = batch_offset + 2u * i6;
    let s7 = batch_offset + 2u * i7;

    let x0r = SRC[s0];
    let x0i = SRC[s0 + 1u];
    let x1r = SRC[s1];
    let x1i = SRC[s1 + 1u];
    let x2r = SRC[s2];
    let x2i = SRC[s2 + 1u];
    let x3r = SRC[s3];
    let x3i = SRC[s3 + 1u];
    let x4r = SRC[s4];
    let x4i = SRC[s4 + 1u];
    let x5r = SRC[s5];
    let x5i = SRC[s5 + 1u];
    let x6r = SRC[s6];
    let x6i = SRC[s6 + 1u];
    let x7r = SRC[s7];
    let x7i = SRC[s7 + 1u];

    let stride = eighth_n >> (stage + stage + stage);
    let tw1 = k * stride;
    let tw2 = tw1 * 2u;
    let tw3 = tw1 * 3u;
    let tw4 = tw1 * 4u;
    let tw5 = tw1 * 5u;
    let tw6 = tw1 * 6u;
    let tw7 = tw1 * 7u;

    let wr1 = TWIDDLE[2u * tw1];
    let wi1 = TWIDDLE[2u * tw1 + 1u];
    let wr2 = TWIDDLE[2u * tw2];
    let wi2 = TWIDDLE[2u * tw2 + 1u];
    let wr3 = TWIDDLE[2u * tw3];
    let wi3 = TWIDDLE[2u * tw3 + 1u];
    let wr4 = TWIDDLE[2u * tw4];
    let wi4 = TWIDDLE[2u * tw4 + 1u];
    let wr5 = TWIDDLE[2u * tw5];
    let wi5 = TWIDDLE[2u * tw5 + 1u];
    let wr6 = TWIDDLE[2u * tw6];
    let wi6 = TWIDDLE[2u * tw6 + 1u];
    let wr7 = TWIDDLE[2u * tw7];
    let wi7 = TWIDDLE[2u * tw7 + 1u];

    // Radix-8 butterfly
    let b1r = wr1 * x1r - wi1 * x1i;
    let b1i = wr1 * x1i + wi1 * x1r;
    let b2r = wr2 * x2r - wi2 * x2i;
    let b2i = wr2 * x2i + wi2 * x2r;
    let b3r = wr3 * x3r - wi3 * x3i;
    let b3i = wr3 * x3i + wi3 * x3r;
    let b4r = wr4 * x4r - wi4 * x4i;
    let b4i = wr4 * x4i + wi4 * x4r;
    let b5r = wr5 * x5r - wi5 * x5i;
    let b5i = wr5 * x5i + wi5 * x5r;
    let b6r = wr6 * x6r - wi6 * x6i;
    let b6i = wr6 * x6i + wi6 * x6r;
    let b7r = wr7 * x7r - wi7 * x7i;
    let b7i = wr7 * x7i + wi7 * x7r;

    let o0 = j * eight_p + k;
    let o1 = o0 + p;
    let o2 = o0 + p + p;
    let o3 = o2 + p;
    let o4 = o0 + p + p + p + p;
    let o5 = o4 + p;
    let o6 = o4 + p + p;
    let o7 = o6 + p;

    let d0 = batch_offset + 2u * o0;
    let d1 = batch_offset + 2u * o1;
    let d2 = batch_offset + 2u * o2;
    let d3 = batch_offset + 2u * o3;
    let d4 = batch_offset + 2u * o4;
    let d5 = batch_offset + 2u * o5;
    let d6 = batch_offset + 2u * o6;
    let d7 = batch_offset + 2u * o7;

    // Radix-8 output equations - using standard DIT butterfly
    // Y[k] = sum_{m=0}^{7} X[m] * W^{k*m} where W = e^{-2*pi*i/8}
    DST[d0] = x0r + b1r + b2r + b3r + b4r + b5r + b6r + b7r;
    DST[d0 + 1u] = x0i + b1i + b2i + b3i + b4i + b5i + b6i + b7i;

    DST[d1] = x0r + b1r + b2r + b3r - b4r - b5r - b6r - b7r;
    DST[d1 + 1u] = x0i + b1i + b2i + b3i - b4i - b5i - b6i - b7i;

    DST[d2] = x0r + b1r - b2r + b3r - b4r - b5r + b6r + b7r;
    DST[d2 + 1u] = x0i + b1i - b2i + b3i - b4i - b5i + b6i + b7i;

    DST[d3] = x0r + b1r - b2r - b3r + b4r + b5r - b6r - b7r;
    DST[d3 + 1u] = x0i + b1i - b2i - b3i + b4i + b5i - b6i - b7i;

    DST[d4] = x0r - b1r + b2r - b3r - b4r + b5r - b6r + b7r;
    DST[d4 + 1u] = x0i - b1i + b2i - b3i - b4i + b5i - b6i + b7i;

    DST[d5] = x0r - b1r + b2r + b3r + b4r - b5r - b6r - b7r;
    DST[d5 + 1u] = x0i - b1i + b2i + b3i + b4i - b5i - b6i - b7i;

    DST[d6] = x0r - b1r - b2r + b3r + b4r + b5r + b6r - b7r;
    DST[d6 + 1u] = x0i - b1i - b2i + b3i + b4i + b5i + b6i - b7i;

    DST[d7] = x0r - b1r - b2r - b3r - b4r - b5r - b6r + b7r;
    DST[d7 + 1u] = x0i - b1i - b2i - b3i - b4i - b5i - b6i + b7i;
}
"#;

// ── WGSL: Stockham Radix-4 DIT (for when log2N % 3 != 0) ────────────────────
const MISTRAL_R4_WGSL: &str = r#"
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

    let stage = U.y;
    let p = 1u << (stage + stage);
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

    let stride = quarter_n >> (stage + stage);
    let tw1 = k * stride;
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

// ── WGSL: Stockham Radix-2 (finalisation for odd log2N) ────────────────────
const MISTRAL_R2_WGSL: &str = r#"
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

    let stage = U.y;
    let p = 1u << stage;
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

    let twiddle_idx = k * (half_n >> stage);
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
    stage: u32,
    log_n: u32,
    _pad: u32,
}

#[derive(Clone)]
struct MistralCache {
    buf_a: wgpu::Buffer,
    buf_b: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    #[allow(dead_code)]
    twiddle_buf: wgpu::Buffer,
    stage_bgs_r8: Vec<wgpu::BindGroup>,
    stage_bgs_r4: Vec<wgpu::BindGroup>,
    stage_bg_r2: Option<wgpu::BindGroup>,
    wg_n8: u32,
    wg_n4: u32,
    wg_n2: u32,
    result_in_b: bool,
}

pub struct MistralVibeFft {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline_r8: wgpu::ComputePipeline,
    pipeline_r4: wgpu::ComputePipeline,
    pipeline_r2: wgpu::ComputePipeline,
    cache: RefCell<std::collections::HashMap<usize, MistralCache>>,
}

impl MistralVibeFft {
    pub fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: true,
        }))
        .expect("no wgpu adapter");

        // Request subgroup support for optimized intra-wave communication
        let required_features = wgpu::Features::SUBGROUP;
        let required_limits = wgpu::Limits::default();

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features,
            required_limits,
            ..Default::default()
        }))
        .expect("no wgpu device");

        let has_subgroup_support = device.features().contains(required_features);

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

        // Use subgroup-optimized pipeline if available, otherwise fall back to regular
        let pipeline_r8 = if has_subgroup_support {
            compile(MISTRAL_R8_SUBGROUP_WGSL.to_string(), "mistral_r8_subgroup")
        } else {
            compile(MISTRAL_R8_WGSL.to_string(), "mistral_r8")
        };

        let pipeline_r4 = compile(MISTRAL_R4_WGSL.to_string(), "mistral_r4");
        let pipeline_r2 = compile(MISTRAL_R2_WGSL.to_string(), "mistral_r2");

        Self {
            device,
            queue,
            pipeline_r8,
            pipeline_r4,
            pipeline_r2,
            cache: RefCell::new(std::collections::HashMap::new()),
        }
    }

    fn build_cache(&self, n: usize, log_n: u32) -> MistralCache {
        // Use mixed-radix approach: as many Radix-8 stages as possible, then Radix-4, then Radix-2
        let num_r8 = (log_n / 3) as usize;
        let rem = log_n % 3;
        let num_r4 = if rem == 2 { 1 } else { 0 };
        let has_r2 = rem == 1;
        let total_stages = num_r8 + num_r4 + has_r2 as usize;

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
            "mistral_buf_a",
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let buf_b = make_buf(
            "mistral_buf_b",
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let staging_buf = make_buf(
            "mistral_staging",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        // N-entry twiddle table: e^{-2*pi*ij/N} for j = 0..N.
        let twiddles: Vec<f32> = (0..n)
            .flat_map(|j| {
                let angle = -std::f32::consts::TAU * j as f32 / n as f32;
                [angle.cos(), angle.sin()]
            })
            .collect();
        let twiddle_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mistral_twiddles"),
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
            label: Some("mistral_uniforms"),
            size: stride * total_stages as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut stage_idx = 0;

        // Radix-8 stages
        for s in 0..num_r8 {
            self.queue.write_buffer(
                &uniform_buf,
                stride * stage_idx as u64,
                bytemuck::bytes_of(&Uniforms {
                    n: n as u32,
                    stage: s as u32,
                    log_n,
                    _pad: 0,
                }),
            );
            stage_idx += 1;
        }

        // Radix-4 stages - stage counter must account for preceding Radix-8 stages
        // p after num_r8 radix-8 stages = 8^num_r8 = 2^(3*num_r8)
        // The radix-4 kernel computes p = 4^stage = 2^(2*stage)
        // So we need stage = (3*num_r8)/2 (always integer when rem=2)
        for s in 0..num_r4 {
            let r4_stage = (3 * num_r8 as u32) / 2 + s as u32;
            self.queue.write_buffer(
                &uniform_buf,
                stride * stage_idx as u64,
                bytemuck::bytes_of(&Uniforms {
                    n: n as u32,
                    stage: r4_stage,
                    log_n,
                    _pad: 0,
                }),
            );
            stage_idx += 1;
        }

        // Radix-2 stage (if needed) - stage counter must account for preceding stages
        // p after num_r8 radix-8 stages = 8^num_r8 = 2^(3*num_r8)
        // The radix-2 kernel computes p = 2^stage, so stage = 3*num_r8 + num_r4
        if has_r2 {
            let r2_stage = 3 * num_r8 as u32 + num_r4 as u32;
            self.queue.write_buffer(
                &uniform_buf,
                stride * stage_idx as u64,
                bytemuck::bytes_of(&Uniforms {
                    n: n as u32,
                    stage: r2_stage,
                    log_n,
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

        // Combined slot s: even -> buf_a -> buf_b, odd -> buf_b -> buf_a.
        let mut stage_bgs_r8: Vec<wgpu::BindGroup> = Vec::new();
        let mut stage_bgs_r4: Vec<wgpu::BindGroup> = Vec::new();

        stage_idx = 0;
        for _s in 0..num_r8 {
            let (src, dst) = if stage_idx % 2 == 0 {
                (&buf_a, &buf_b)
            } else {
                (&buf_b, &buf_a)
            };
            stage_bgs_r8.push(make_bg(
                &self.pipeline_r8,
                src,
                dst,
                stride * stage_idx as u64,
            ));
            stage_idx += 1;
        }

        // Radix-4 stages
        for _s in 0..num_r4 {
            let (src, dst) = if stage_idx % 2 == 0 {
                (&buf_a, &buf_b)
            } else {
                (&buf_b, &buf_a)
            };
            stage_bgs_r4.push(make_bg(
                &self.pipeline_r4,
                src,
                dst,
                stride * stage_idx as u64,
            ));
            stage_idx += 1;
        }

        let stage_bg_r2 = if has_r2 {
            let (src, dst) = if stage_idx % 2 == 0 {
                (&buf_a, &buf_b)
            } else {
                (&buf_b, &buf_a)
            };
            Some(make_bg(
                &self.pipeline_r2,
                src,
                dst,
                stride * stage_idx as u64,
            ))
        } else {
            None
        };

        MistralCache {
            buf_a,
            buf_b,
            staging_buf,
            twiddle_buf,
            stage_bgs_r8,
            stage_bgs_r4,
            stage_bg_r2,
            wg_n8: (n as u32 / 8).div_ceil(256),
            wg_n4: (n as u32 / 4).div_ceil(256),
            wg_n2: (n as u32 / 2).div_ceil(256),
            result_in_b: total_stages % 2 == 1,
        }
    }

    fn get_or_build_cache(&self, n: usize, log_n: u32) -> MistralCache {
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
                label: Some("mistral_fft"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mistral_fft_compute"),
                timestamp_writes: None,
            });

            for bg in &cache.stage_bgs_r8 {
                pass.set_pipeline(&self.pipeline_r8);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(cache.wg_n8, batch_size, 1);
            }

            for bg in &cache.stage_bgs_r4 {
                pass.set_pipeline(&self.pipeline_r4);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(cache.wg_n4, batch_size, 1);
            }

            if let Some(r2_bg) = &cache.stage_bg_r2 {
                pass.set_pipeline(&self.pipeline_r2);
                pass.set_bind_group(0, r2_bg, &[]);
                pass.dispatch_workgroups(cache.wg_n2, batch_size, 1);
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

impl FftExecutor for MistralVibeFft {
    fn name(&self) -> &str {
        "Mistral Vibe (Stockham Radix-8/4/2, Subgroup-Aware)"
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
