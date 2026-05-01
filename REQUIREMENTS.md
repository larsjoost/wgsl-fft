# Requirements: 1D Batched WebGPU FFT (High-Performance)

## 0. Execution Mode and Benchmark Validity (Mandatory)

* **Prefer real GPU adapters:** Request adapter with `force_fallback_adapter: false` first. If unavailable, then fall back to `true`.
* **Report adapter info:** Print backend + adapter name in benchmark output so software fallback runs are obvious.
* **Release-only performance runs:** All leaderboard/performance numbers must be collected with `--release`.
* **Separate timing from readback:** Timed benchmark path must not include host readback (`map_async`, staging-buffer map, CPU polling for result extraction). Readback is only for correctness validation.

## 1. Batch-Aware Memory Architecture

* **Contiguous Layout:** Input must be a single `wgpu::Buffer` containing B signals of length N.
* **Interleaved Complex Data:** Use `vec2<f32>` (Real, Imaginary) so the memory controller pulls both components in a single 64-bit fetch.
* **Stride Management (future):** Supporting configurable strides and distances between signals would allow sub-sections of larger datasets to be processed without copying. This is not yet implemented and is not a requirement for rivals.

## 2. Kernel Dispatch Strategies

The planner must choose a dispatch strategy based on the relationship between FFT size (N) and workgroup size (W).

### Strategy A: Small FFTs (One or More per Workgroup)
* **Occupancy Focus:** If N is small (e.g., 64, 128, 256), a single workgroup should process **multiple** FFTs from the batch.
* **Shared Memory Tiling:** Use `var<workgroup>` memory as a staging area where multiple signals are loaded, processed, and written back in bulk.
* **Single-dispatch local FFT for small N:** For `N <= 256`, perform the full FFT in workgroup memory in one dispatch (no global ping-pong passes).
* **Why:** Prevents under-utilization where a workgroup has 256 threads but only 64 are doing work.

### Strategy B: Medium FFTs (One per Workgroup)
* **Shared Memory Resident:** For sizes where N fits in workgroup memory (typically up to 4096), the entire 1D FFT is completed in one pass.
* **Synchronization:** Use `workgroupBarrier()` between butterfly stages.

### Strategy C: Large FFTs (Multi-Pass Batched)
* **Multi-Pass via Separate Dispatches:** WebGPU has no intra-dispatch global barrier. When N exceeds shared memory, each pass is a separate `dispatch_workgroups` call submitted to the queue. Every signal in the batch goes through Pass 1, then Pass 2, etc.
* **Stockham Algorithm:** Use the Stockham formulation to keep the batch coalesced across passes — no bit-reversal step required.

## 3. High-Performance WGSL Kernels

* **Mixed-Radix (4 or 8):** Use Radix-4 or Radix-8 butterflies to reduce pass count from log₂N to log₄N or log₈N, and to reduce register pressure. Any Radix size maybe be used.
* **WGSL source format:** Write WGSL as raw `const &str` strings passed to `wgpu::ShaderSource::Wgsl`. Raw strings are used for all kernels in `src/shaders.rs`.
* **Subgroup Shuffle Optimization (where available):** `subgroupShuffle` allows threads to exchange data within a wave (32–64 threads) without touching shared memory, which is significantly faster. However, this requires `wgpu::Features::SUBGROUP` to be explicitly requested at device creation and is not available on all backends (notably some Metal and DX12 configurations). Guard its use behind a runtime feature check and fall back to `workgroupBarrier`-based exchange.
* **Twiddle Factor Pre-computation:** Twiddle factors are identical for every signal in the batch. Store them in a read-only storage buffer to leverage the GPU's constant cache. The baseline already implements this.

## 4. Rust Orchestrator (wgpu)

* **Pipeline Overridable Constants:** WebGPU `override` declarations let you specialize scalar constants at pipeline creation time without a full shader recompilation. This is useful for fixed parameters like workgroup size, but cannot change loop trip counts or array dimensions in a way the driver can unroll.
* **Command Encoding:** Use `dispatch_workgroups(groups_x, batch_size, 1)` where `groups_x` covers the butterflies-per-stage at 256 threads per group. For a Radix-R kernel, each stage computes N/R butterflies: `groups_x = (n / R).div_ceil(256)`. The formula `(n/2).div_ceil(256)` is Radix-2 specific. The Y dimension carries the batch index into the kernel.

## 5. Performance Monitoring

* **Timestamp Queries (optional):** Use `wgpu::QuerySet` with `wgpu::Features::TIMESTAMP_QUERY` (must be explicitly requested) to measure GPU kernel time. The leaderboard benchmark uses **wall-clock time** (`std::time::Instant`) which includes dispatch + synchronization overhead — GPU-only timestamps give more precise numbers but are not required.
* **Timing scope:** If using timestamp queries, report GPU-only time (dispatch execution) separately from host orchestration and readback.
* **Throughput Calculation:** Report performance as **GFLOPS** using the standard FFT operation count: `5 · B · N · log₂N / time`.
* **Alignment:** Use `bytemuck` (already a project dependency) to cast buffer data. Uniform buffer offsets must respect `device.limits().min_uniform_buffer_offset_alignment` — the baseline already handles this.

## 6. Validation and Benchmark API Shape

* **Two-path execution model:** Implement:
  1. **Benchmark path:** upload + dispatch + timestamp resolve, no host decode of FFT outputs.
  2. **Validation path:** run FFT once, read back results, compare to baseline within tolerance.
* **Correctness is unchanged:** Validation still compares rival output vs baseline output sample-by-sample under the existing tolerance.

## Summary

> Implement a **1D Batched FFT** in WebGPU/Rust. Focus on a **Stockham-based Mixed-Radix** approach with an explicit small-FFT local-memory path. Benchmark on real GPU adapters when available, in **release mode**, and measure **GPU-only execution time** (separate from readback). Use subgroup shuffles where available, with workgroup fallbacks. Data is interleaved `vec2<f32>`. The goal is maximum VRAM throughput and occupancy.
