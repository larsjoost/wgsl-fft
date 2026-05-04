use crate::error::{FftError, Result};
use crate::FftExecutor;
use num_complex::Complex;
use thiserror::Error;

// ── hipFFT C API (FFI) ─────────────────────────────────────────────────────
// Compatible with both AMD ROCm and NVIDIA HIP-CUDA backends.
// Link against -lhipfft and -lamdhip64 (AMD) or -lhipfft-cuda -lcuda (NVIDIA).

#[allow(non_camel_case_types)]
type hipfftHandle = i32;

#[allow(non_camel_case_types)]
#[repr(C)]
struct hipfftComplex {
    x: f32,
    y: f32,
}

#[allow(non_camel_case_types)]
type hipStream_t = *mut std::ffi::c_void;

const HIPFFT_C2C: i32 = 0x29;
const HIPFFT_FORWARD: i32 = -1;
const HIPFFT_BACKWARD: i32 = 1;
const HIP_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const HIP_MEMCPY_DEVICE_TO_HOST: i32 = 2;

#[link(name = "hipfft")]
extern "C" {
    fn hipfftCreate(plan: *mut hipfftHandle) -> i32;
    fn hipfftDestroy(plan: hipfftHandle) -> i32;
    fn hipfftPlan1d(plan: *mut hipfftHandle, nx: i32, type_: i32, batch: i32) -> i32;
    fn hipfftExecC2C(
        plan: hipfftHandle,
        idata: *mut hipfftComplex,
        odata: *mut hipfftComplex,
        direction: i32,
    ) -> i32;
    fn hipfftSetStream(plan: hipfftHandle, stream: hipStream_t) -> i32;
}

#[link(name = "amdhip64")]
extern "C" {
    fn hipMalloc(ptr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    fn hipFree(ptr: *mut std::ffi::c_void) -> i32;
    fn hipMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: i32,
    ) -> i32;
    fn hipDeviceSynchronize() -> i32;
}

// ── Error type ───────────────────────────────────────────────────────────────

/// Error type for hipFFT operations
#[derive(Debug, Error)]
#[error("hipFFT error code {0}")]
struct HipFftError(i32);

fn check(code: i32) -> std::result::Result<(), HipFftError> {
    if code == 0 {
        Ok(())
    } else {
        Err(HipFftError(code))
    }
}

impl From<HipFftError> for FftError {
    fn from(e: HipFftError) -> Self {
        FftError::BackendError(format!("hipFFT error: {}", e.0))
    }
}

// ── HipFft struct ────────────────────────────────────────────────────────────

pub struct HipFft {
    fft_size: usize,
}

impl HipFft {
    pub fn new(fft_size: usize) -> Result<Self> {
        // /dev/kfd is the AMD Kernel Fusion Driver — only present when an AMD GPU
        // with ROCm is active. Without it, calling hipfftCreate segfaults immediately.
        if !std::path::Path::new("/dev/kfd").exists() {
            return Err(FftError::BackendError(
                "No AMD GPU detected (/dev/kfd absent); hipFFT requires ROCm".to_string(),
            ));
        }

        // Probe by creating and immediately destroying a tiny plan.
        let mut probe: hipfftHandle = 0;
        // SAFETY: /dev/kfd presence verified above; hipFFT C API requires raw pointers with no safe wrapper.
        unsafe {
            check(hipfftCreate(&mut probe)).map_err(FftError::from)?;
            check(hipfftPlan1d(&mut probe, 1, HIPFFT_C2C, 1)).map_err(FftError::from)?;
            hipfftDestroy(probe);
        }
        Ok(Self { fft_size })
    }

    fn exec_batch(&self, inputs: &[Vec<Complex<f32>>], direction: i32) -> Result<Vec<Vec<Complex<f32>>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let n = inputs[0].len();
        let batch = inputs.len();

        // Flatten to interleaved f32 pairs.
        let mut host_in: Vec<hipfftComplex> = inputs
            .iter()
            .flat_map(|v| v.iter().map(|c| hipfftComplex { x: c.re, y: c.im }))
            .collect();

        let byte_count = host_in.len() * std::mem::size_of::<hipfftComplex>();

        // SAFETY: hipFFT C API requires raw pointers; host_in is valid for byte_count bytes and
        // outlives the GPU copy. All allocations are freed before returning.
        unsafe {
            let mut d_in: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_out: *mut std::ffi::c_void = std::ptr::null_mut();
            check(hipMalloc(&mut d_in, byte_count)).map_err(FftError::from)?;
            check(hipMalloc(&mut d_out, byte_count)).map_err(FftError::from)?;

            check(hipMemcpy(
                d_in,
                host_in.as_ptr() as *const std::ffi::c_void,
                byte_count,
                HIP_MEMCPY_HOST_TO_DEVICE,
            ))
            .map_err(FftError::from)?;

            let mut plan: hipfftHandle = 0;
            check(hipfftCreate(&mut plan)).map_err(FftError::from)?;
            check(hipfftPlan1d(&mut plan, n as i32, HIPFFT_C2C, batch as i32)).map_err(FftError::from)?;

            check(hipfftExecC2C(
                plan,
                d_in as *mut hipfftComplex,
                d_out as *mut hipfftComplex,
                direction,
            ))
            .map_err(FftError::from)?;

            check(hipDeviceSynchronize()).map_err(FftError::from)?;

            check(hipMemcpy(
                host_in.as_mut_ptr() as *mut std::ffi::c_void,
                d_out,
                byte_count,
                HIP_MEMCPY_DEVICE_TO_HOST,
            ))
            .map_err(FftError::from)?;

            hipfftDestroy(plan);
            hipFree(d_in);
            hipFree(d_out);
        }

        let scale = if direction == HIPFFT_BACKWARD {
            1.0 / n as f32
        } else {
            1.0
        };

        let output: Vec<Vec<Complex<f32>>> = host_in
            .chunks(n)
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|c| Complex::new(c.x * scale, c.y * scale))
                    .collect()
            })
            .collect();

        Ok(output)
    }
}

impl FftExecutor for HipFft {
    fn name(&self) -> &str {
        "hipFFT (ROCm/HIP reference)"
    }

    fn fft(&self, inputs: &[Vec<Complex<f32>>]) -> Result<Vec<Vec<Complex<f32>>>> {
        self.exec_batch(inputs, HIPFFT_FORWARD)
    }

    fn ifft(&self, inputs: &[Vec<Complex<f32>>]) -> Result<Vec<Vec<Complex<f32>>>> {
        self.exec_batch(inputs, HIPFFT_BACKWARD)
    }
}
