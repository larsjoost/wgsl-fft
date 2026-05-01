use std::any::Any;

use crate::{FftExecutor, GpuFft};
use num_complex::Complex;

const RADIX4_WGSL: &str = r#"
@group(0) @binding(0)
var<uniform> U: vec4<u32>;
@group(0) @binding(1)
var<storage, read_write> SRC: array<f32>;
@group(0) @binding(2)
var<storage, read_write> DST: array<f32>;
@group(0) @binding(3)
var<storage, read> TWIDDLE: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let batch_id = gid.y;
    let n = U.x;
    let stage = U.y;
    let batch_offset = batch_id * n * 2u;

    // For Radix-4, we only process even stages (0, 2, 4, ...)
    // Odd stages (1, 3, 5, ...) are identity operations (copy SRC to DST)
    if stage % 2u != 0u {
        let idx1 = tid;
        let idx2 = tid + (n >> 1u);

        if idx1 < n {
            let s1 = batch_offset + 2u * idx1;
            DST[s1] = SRC[s1];
            DST[s1 + 1u] = SRC[s1 + 1u];
        }
        if idx2 < n {
            let s2 = batch_offset + 2u * idx2;
            DST[s2] = SRC[s2];
            DST[s2 + 1u] = SRC[s2 + 1u];
        }
        return;
    }

    let quarter_n = n >> 2u;
    if tid >= quarter_n {
        return;
    }

    let r4_stage = stage / 2u;
    let p = 1u << (r4_stage + r4_stage);
    let four_p = p << 2u;

    let k = tid % p;
    let j = tid / p;

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

    let stride = quarter_n >> (r4_stage + r4_stage);
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

pub struct Radix4Rival(pub GpuFft);

impl Radix4Rival {
    pub fn new() -> Self {
        Self(GpuFft::with_shader(RADIX4_WGSL.to_string(), "radix4_rival").unwrap())
    }
}

impl FftExecutor for Radix4Rival {
    fn name(&self) -> &str {
        "Radix-4 Rival"
    }

    fn fft(
        &self,
        inputs: &[Vec<Complex<f32>>],
    ) -> Result<Vec<Vec<Complex<f32>>>, Box<dyn std::error::Error>> {
        self.0.fft(inputs)
    }

    fn ifft(
        &self,
        inputs: &[Vec<Complex<f32>>],
    ) -> Result<Vec<Vec<Complex<f32>>>, Box<dyn std::error::Error>> {
        self.0.ifft(inputs)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl crate::GpuFftTrait for Radix4Rival {
    fn benchmark_gpu_only(
        &self,
        sc: &crate::SizeCache,
        batch_size: u32,
        n: usize,
        warmup_iters: usize,
        bench_iters: usize,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        self.0
            .benchmark_gpu_only(sc, batch_size, n, warmup_iters, bench_iters)
    }

    fn get_or_build_size_cache(&self, n: usize, log_n: u32) -> crate::SizeCache {
        self.0.get_or_build_size_cache(n, log_n)
    }

    fn prepare_input_data(&self, input: &[Complex<f32>], inverse: bool) -> Vec<f32> {
        self.0.prepare_input_data(input, inverse)
    }

    fn queue(&self) -> &wgpu::Queue {
        self.0.queue()
    }
}
