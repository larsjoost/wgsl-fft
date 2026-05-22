/// Stockham Radix-4 DIT kernel (derived from Gemini rivalry winner).
///
/// Uniform U: .x = N (transform length), .y = p (stride = 4^stage_index).
/// SRC / DST:  interleaved complex pairs [re₀, im₀, re₁, im₁, …].
/// TWIDDLE:    N complex pairs e^{-2πij/N} for j = 0..N.
///
/// Dispatched log₄N times; each dispatch processes N/4 butterfly groups.
pub const R4_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> U: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> SRC: array<f32>;
@group(0) @binding(2) var<storage, read_write> DST: array<f32>;
@group(0) @binding(3) var<storage, read> TWIDDLE: array<f32>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid       = gid.x;
    let batch_id  = gid.y;
    let n         = U.x;
    let p         = U.y;
    let quarter_n = n >> 2u;
    if tid >= quarter_n { return; }

    let four_p = p << 2u;
    let k      = tid % p;
    let j      = tid / p;
    let bo     = batch_id * n * 2u;

    let i0 = j*p + k;
    let i1 = i0 + quarter_n;
    let i2 = i1 + quarter_n;
    let i3 = i2 + quarter_n;

    var x: array<vec2<f32>, 4>;
    x[0] = vec2<f32>(SRC[bo + 2u*i0], SRC[bo + 2u*i0+1u]);
    x[1] = vec2<f32>(SRC[bo + 2u*i1], SRC[bo + 2u*i1+1u]);
    x[2] = vec2<f32>(SRC[bo + 2u*i2], SRC[bo + 2u*i2+1u]);
    x[3] = vec2<f32>(SRC[bo + 2u*i3], SRC[bo + 2u*i3+1u]);

    let stride = quarter_n / p;
    let tw     = k * stride;
    let tw_im_sign = 1.0;
    x[1] = cmul(vec2<f32>(TWIDDLE[2u*tw],   tw_im_sign * TWIDDLE[2u*tw+1u]),   x[1]);
    x[2] = cmul(vec2<f32>(TWIDDLE[4u*tw],   tw_im_sign * TWIDDLE[4u*tw+1u]),   x[2]);
    x[3] = cmul(vec2<f32>(TWIDDLE[6u*tw],   tw_im_sign * TWIDDLE[6u*tw+1u]),   x[3]);

    let s02 = x[0] + x[2]; let d02 = x[0] - x[2];
    let s13 = x[1] + x[3]; let d13 = x[1] - x[3];

    let y0 = s02 + s13;
    let y1 = vec2<f32>(d02.x + d13.y, d02.y - d13.x);
    let y2 = s02 - s13;
    let y3 = vec2<f32>(d02.x - d13.y, d02.y + d13.x);

    let d_base = bo + 2u*(j*four_p + k);
    DST[d_base]          = y0.x; DST[d_base+1u]          = y0.y;
    DST[d_base + 2u*p]   = y1.x; DST[d_base + 2u*p+1u]   = y1.y;
    DST[d_base + 4u*p]   = y2.x; DST[d_base + 4u*p+1u]   = y2.y;
    DST[d_base + 6u*p]   = y3.x; DST[d_base + 6u*p+1u]   = y3.y;
}
"#;

/// Stockham Radix-2 DIT kernel — fallback for the final stage when log₂N is odd.
///
/// Uniform U: .x = N, .y = p (stride = 2^(num_r4_stages * 2)).
/// TWIDDLE:   N complex pairs e^{-2πij/N} for j = 0..N.
pub const R2_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> U: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> SRC: array<f32>;
@group(0) @binding(2) var<storage, read_write> DST: array<f32>;
@group(0) @binding(3) var<storage, read> TWIDDLE: array<f32>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid      = gid.x;
    let batch_id = gid.y;
    let n        = U.x;
    let p        = U.y;
    let half_n   = n >> 1u;
    if tid >= half_n { return; }

    let two_p = p + p;
    let k     = tid % p;
    let j     = tid / p;
    let bo    = batch_id * n * 2u;

    let i1 = j*p + k;
    let i2 = i1 + half_n;

    let x1 = vec2<f32>(SRC[bo + 2u*i1], SRC[bo + 2u*i1+1u]);
    let x2 = vec2<f32>(SRC[bo + 2u*i2], SRC[bo + 2u*i2+1u]);

    let tw = k * (half_n / p);
    let tw_im_sign = 1.0;
    let t  = cmul(vec2<f32>(TWIDDLE[2u*tw], tw_im_sign * TWIDDLE[2u*tw+1u]), x2);

    let d_base = bo + 2u*(j*two_p + k);
    DST[d_base]          = x1.x + t.x; DST[d_base+1u]          = x1.y + t.y;
    DST[d_base + 2u*p]   = x1.x - t.x; DST[d_base + 2u*p+1u]   = x1.y - t.y;
}
"#;

/// Cooley-Tukey radix-2 DIT FFT shader using vec2<f32> complex pairs.
/// Computes twiddle factors on the fly (no TWIDDLE buffer needed).
/// This matches the pipeline's existing FFT shader format.
///
/// Uniform params: .x = N, .y = stage, .z = direction (0=FFT, 1=IFFT)
/// @binding(0): input_data (storage, read) - array<vec2<f32>>
/// @binding(1): output_data (storage, read_write) - array<vec2<f32>>
pub const COOLEY_TUKEY_R2_WGSL: &str = r#"
struct FftParams {
    n:         u32,
    stage:     u32,
    direction: u32,
    _pad:      u32,
};

@group(0) @binding(0) var<storage, read>       input_data:  array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output_data: array<vec2<f32>>;
@group(0) @binding(2) var<uniform>             params:      FftParams;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x,
    );
}

fn twiddle(k: u32, span: u32, direction: u32) -> vec2<f32> {
    let pi2: f32 = 6.283185307179586;
    let sign = select(-1.0, 1.0, direction == 1u);
    let angle = sign * pi2 * f32(k) / f32(span);
    return vec2<f32>(cos(angle), sin(angle));
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let batch_id = gid.y;
    if i >= params.n / 2u { return; }

    let bo = batch_id * params.n;
    let span: u32 = 1u << (params.stage + 1u);
    let half: u32 = span >> 1u;
    let group: u32 = i / half;
    let k: u32     = i % half;
    let even: u32  = bo + group * span + k;
    let odd: u32   = even + half;

    let u = input_data[even];
    let v = input_data[odd];
    let w  = twiddle(k, span, params.direction);
    let wv = cmul(w, v);
    output_data[even] = u + wv;
    output_data[odd]  = u - wv;
}
"#;

/// Normalize shader for vec2<f32> format
/// Divides each element by N
/// @binding(0): data (storage, read_write) - array<vec2<f32>>
/// @binding(1): params (uniform) - vec4<u32> where .x = N
pub const NORMALIZE_VEC2_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: vec4<u32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let batch_id = gid.y;
    let n = params.x;
    if i >= n { return; }
    
    let idx = batch_id * n + i;
    let scale = 1.0 / f32(n);
    data[idx] = vec2<f32>(data[idx].x * scale, data[idx].y * scale);
}
"#;

/// Cooley-Tukey radix-2 bit-reversal permutation (vec2<f32> format).
/// Bindings: 0 = src (read), 1 = dst (read_write), 2 = BitRevParams uniform { n, log2_n, _pad0, _pad1 }
pub const BIT_REVERSAL_WGSL: &str = r#"
struct BitRevParams {
    n: u32,
    log2_n: u32,
    _pad0: u32,
    _pad1: u32,
};
@group(0) @binding(0) var<storage, read>       src:    array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> dst:    array<vec2<f32>>;
@group(0) @binding(2) var<uniform>             params: BitRevParams;
fn bit_reverse(x: u32, bits: u32) -> u32 {
    var r: u32 = 0u;
    var v: u32 = x;
    for (var i: u32 = 0u; i < bits; i++) {
        r = (r << 1u) | (v & 1u);
        v >>= 1u;
    }
    return r;
}
@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let batch_id = gid.y;
    if i >= params.n { return; }
    let j = bit_reverse(i, params.log2_n);
    
    let bo = batch_id * params.n;
    dst[bo + i] = src[bo + j];
}
"#;

/// Bluestein's algorithm shader for arbitrary-size FFT.
///
/// Bluestein's algorithm converts an N-point FFT into a 2N-point convolution:
/// X[k] = exp(-πi*k²/N) * Σ_m x[m]*exp(-πi*(k-m)²/N) * exp(πi*m²/N)
///
/// This shader multiplies the input by the chirp sequence: a[m] = x[m] * exp(πi*m²/N)
///
/// Bindings:
/// @group(0) @binding(0): params - uniform with N, padded_N, _, _
/// @group(0) @binding(1): input - storage read - interleaved complex pairs
/// @group(0) @binding(2): output - storage write - interleaved complex pairs
/// @group(0) @binding(3): chirp - storage read - precomputed chirp values (exp(πi*m²/N))
pub const BLUESTEIN_CHIRP_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> params: vec4<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> chirp: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let batch_id = gid.y;
    let n = params.x;
    let padded_n = params.y;
    
    if i >= n { return; }
    
    let bo = batch_id * padded_n * 2u;
    let idx = i * 2u;
    
    // Load input complex value
    let x_re = input[bo + idx];
    let x_im = input[bo + idx + 1u];
    
    // Load chirp complex value (exp(πi*i²/N))
    let c_re = chirp[idx];
    let c_im = chirp[idx + 1u];
    
    // Multiply: output = x * chirp
    // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    let out_re = x_re * c_re - x_im * c_im;
    let out_im = x_re * c_im + x_im * c_re;
    
    output[bo + idx] = out_re;
    output[bo + idx + 1u] = out_im;
}
"#;

/// Bluestein's algorithm shader for multiplying by inverse chirp and extracting result.
///
/// After FFT of the chirp-multiplied input, we need to:
/// 1. Multiply by the inverse chirp: exp(-πi*k²/N)
/// 2. Extract the first N points from the padded result
/// 3. Apply scaling (1/N for IFFT)
///
/// Bindings:
/// @group(0) @binding(0): params - uniform with N, padded_N, _, _
/// @group(0) @binding(1): input - storage read - interleaved complex pairs (FFT result)
/// @group(0) @binding(2): output - storage write - interleaved complex pairs
/// @group(0) @binding(3): inv_chirp - storage read - precomputed inverse chirp values (with scaling)
pub const BLUESTEIN_INV_CHIRP_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> params: vec4<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> inv_chirp: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let batch_id = gid.y;
    let n = params.x;
    let padded_n = params.y;
    
    if i >= n { return; }
    
    let bo_in = batch_id * padded_n * 2u;
    let bo_out = batch_id * n * 2u;
    let idx = i * 2u;
    
    // Load FFT result complex value
    let x_re = input[bo_in + idx];
    let x_im = input[bo_in + idx + 1u];
    
    // Load inverse chirp complex value (exp(-πi*i²/N) * scale)
    let c_re = inv_chirp[idx];
    let c_im = inv_chirp[idx + 1u];
    
    // Multiply: output = x * inv_chirp (scaling already included in inv_chirp)
    let out_re = x_re * c_re - x_im * c_im;
    let out_im = x_re * c_im + x_im * c_re;
    
    output[bo_out + idx] = out_re;
    output[bo_out + idx + 1u] = out_im;
}
"#;

/// Zero-pad input from N to padded_N (for Bluestein's algorithm).
///
/// Bindings:
/// @group(0) @binding(0): params - uniform with N, padded_N, _, _
/// @group(0) @binding(1): input - storage read - interleaved complex pairs
/// @group(0) @binding(2): output - storage write - interleaved complex pairs (zero-padded)
pub const BLUESTEIN_ZERO_PAD_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> params: vec4<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let batch_id = gid.y;
    let n = params.x;
    let padded_n = params.y;
    
    let bo_in = batch_id * padded_n * 2u;
    let bo_out = batch_id * padded_n * 2u;
    
    if i >= padded_n * 2u { return; }
    
    let idx = i;
    
    // Copy from input if within original size, otherwise write zero
    if idx < n * 2u {
        output[bo_out + idx] = input[bo_in + idx];
    } else {
        output[bo_out + idx] = 0.0;
    }
}
"#;
