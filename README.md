# wgsl-fft

GPU-accelerated FFT in Rust using [wgpu](https://github.com/gfx-rs/wgpu) compute shaders.

Implements the FFT using a two-buffer ping-pong formulation where each stage reads from one buffer and writes to the other. This eliminates the separate bit-reversal pass and removes all inter-stage memory hazards, allowing the entire transform to run in a single GPU compute pass with one `queue.submit()` call.

**GPU-accelerated**: This library uses wgpu compute shaders for GPU acceleration. If no GPU is available, wgpu's fallback adapter provides CPU-based software rendering.

**Supports both power-of-two and arbitrary FFT sizes**: Power-of-two sizes use fast Stockham Radix-4/2 GPU acceleration, while arbitrary sizes use Bluestein's algorithm. The implementation is fully GPU-accelerated for all sizes and optimized for batch processing.

**Multi-backend**: Supports Vulkan, Metal, DX12, and WebGPU via wgpu. Optional CUDA support via `cuda` feature, ROCm support via `rocm` feature, and HIP FFT via `hipfft` feature.

The WGSL compute kernels are embedded as raw WGSL strings in [`src/shaders.rs`](src/shaders.rs).

## Table of Contents

- [Features](#features)
- [Usage](#usage)
- [Examples](#examples)
- [Requirements](#requirements)
- [Algorithms](#algorithms)
- [Module Structure](#module-structure)
- [Simple Pipeline Interface](#simple-pipeline-interface)
- [Performance Characteristics](#performance-characteristics)
- [Numerical Accuracy](#numerical-accuracy)
- [Shader Development](#shader-development)
- [Performance Optimization](#performance-optimization)
- [License](#license)

## Features

- **GPU-accelerated**: Uses wgpu compute shaders for high-performance FFT computation
- **Stockham Radix-4/2**: Fast algorithm for power-of-two FFT sizes in a single compute pass
- **Arbitrary sizes**: Bluestein's algorithm for non-power-of-two sizes (fully GPU-accelerated)
- **Batch processing**: Efficiently process multiple FFTs in a single call (up to 7.5x speedup)
- **Multi-backend**: Vulkan, Metal, DX12, WebGPU via wgpu
- **CUDA support**: Optional `cuda` feature for NVIDIA GPUs via CuFFT
- **ROCm support**: Optional `rocm` feature for AMD GPUs
- **HIP FFT**: Optional `hipfft` feature for AMD HIP
- **Fallback support**: Automatically falls back to CPU software rendering if no GPU is available

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
wgsl-fft = "0.4.3"
num-complex = "0.4"
```

For CUDA support:
```toml
wgsl-fft = { version = "0.4.3", features = ["cuda"] }
```

For ROCm support:
```toml
wgsl-fft = { version = "0.4.3", features = ["rocm"] }
```

### Forward FFT (Batch Processing)

```rust
use wgsl_fft::GpuFft;
use num_complex::Complex;

// Create FFT instance - returns Result since GPU might not be available
let fft = GpuFft::new()?;

// Single FFT (pass vector with one element)
let single_input = vec![vec![Complex { re: (i as f32 * 0.1).sin(), im: 0.0 }; 1024]];
let single_spectrum = fft.fft(&single_input)?;
assert_eq!(single_spectrum.len(), 1);
assert_eq!(single_spectrum[0].len(), 1024);

// Batch FFT (process multiple signals efficiently)
let batch_inputs = vec![
    vec![Complex { re: (i as f32 * 0.1).sin(), im: 0.0 }; 1024],
    vec![Complex { re: (i as f32 * 0.2).sin(), im: 0.0 }; 1024],
];
let batch_spectra = fft.fft(&batch_inputs)?;
assert_eq!(batch_spectra.len(), 2); // Two FFT results
assert_eq!(batch_spectra[0].len(), 1024); // Each FFT has 1024 bins
```

### Inverse FFT (Batch Processing)

```rust
// Compute inverse FFT (automatically scaled by 1/N)
let reconstructed_batch = fft.ifft(&batch_spectra)?;
assert_eq!(reconstructed_batch.len(), 2); // Two IFFT results

// Roundtrip: FFT(IFFT(x)) ≈ x (within numerical precision)
let roundtrip_error = batch_inputs[0].iter().zip(reconstructed_batch[0].iter())
    .map(|(a, b)| ((a.re - b.re).powi(2) + (a.im - b.im).powi(2)).sqrt())
    .fold(0.0, f32::max);
println!("Max roundtrip error: {roundtrip_error:.2e}");
```

### Arbitrary (Non-Power-of-Two) Sizes

```rust
// Non-power-of-two FFT (e.g., 150 samples)
let arbitrary_input = vec![vec![Complex::new(1.0, 0.0); 150]];
let arbitrary_spectrum = fft.fft(&arbitrary_input)?;
assert_eq!(arbitrary_spectrum[0].len(), 150);

// Non-power-of-two IFFT
let arbitrary_reconstructed = fft.ifft(&arbitrary_spectrum)?;
assert_eq!(arbitrary_reconstructed[0].len(), 150);
```

### Running Benchmarks

```sh
# Single FFT throughput test
cargo test fft_throughput --release -- --nocapture

# Criterion benchmarks
cargo bench

# Accuracy test vs rustfft
cargo test test_gpu_fft_matches_rustfft --release
```

## Examples

The repository includes several examples demonstrating various features:

| Example | Description |
|---------|-------------|
| `test_batch_processing` | Demonstrates batch FFT processing |
| `find_optimal_batch` | Finds optimal batch size for your GPU |
| `gpu_only_benchmark` | Benchmarks GPU-only performance |
| `compare_cufft` | Compares wgsl-fft with NVIDIA CuFFT |
| `windowed_fft` | Shows windowed FFT for signal processing |
| `parallel_pipeline` | Parallel FFT pipeline example |
| `rivalry_leaderboard` | Compares against multiple FFT implementations |
| `test_parallel` | Tests parallel FFT execution |
| `double_buffer_fft` | Demonstrates double-buffering technique |

## Requirements

- **Rust**: 1.73+ (as specified in Cargo.toml)
- **GPU**: wgpu-compatible (Vulkan 1.3+, Metal, DX12, or WebGPU)
- **OS**: Linux, Windows, or macOS

### Linux Dependencies

For Vulkan backend on Linux, install:
```sh
sudo apt-get update && sudo apt-get install -y \
    pkg-config \
    libx11-dev \
    libasound2-dev \
    libudev-dev \
    libwayland-dev \
    libxkbcommon-dev
```

## Algorithms

### Power-of-Two: Stockham Auto-sort Radix-4/2
- **Single compute pass**: All log₂(N) butterfly stages execute in one shader dispatch
- **Ping-pong buffers**: Even stages read from buffer A and write to buffer B; odd stages read from B and write to A
- **Natural order output**: No separate bit-reversal pass needed due to autosort property
- **Memory efficient**: No inter-stage synchronization required since consecutive stages access different buffers
- **Complexity**: O(N log N)

### Arbitrary Sizes: Bluestein's Algorithm
- Automatically used for non-power-of-two input lengths
- Converts any size $N$ FFT into a power-of-two convolution of size $M \ge 2N-1$
- Uses GPU-accelerated power-of-2 FFTs internally for the expensive convolution computation
- **High precision**: Uses `f64` for all precomputations (twiddle factors and chirps) to maintain accuracy
- **Cached**: Precomputed chirp FFTs are cached to eliminate redundant work for repeated sizes
- Seamlessly integrated — the same `fft()` and `ifft()` methods work for all sizes

## Module Structure

The crate is organized into the following modules:

| Module | Purpose |
|--------|---------|
| `fft` | Main FFT implementation (`GpuFft`, `FftExecutor`, `SizeCache`) |
| `pipelines` | Pre-compiled FFT pipelines (`FftPipelines`, `FftDirection`) for embedding in larger GPU pipelines |
| `pipeline_builder` | Simple pipeline interface (`PipelineBuilder`, `ComputeStage`, `PingPongState`, etc.) for declarative pipeline construction |
| `shaders` | WGSL compute shader source code |
| `benchmark` | Benchmarking utilities and throughput measurement |
| `rivals` | Alternative FFT implementations for comparison |
| `error` | Error types for FFT operations |

All public types are re-exported from the crate root, so you can use `wgsl_fft::GpuFft` directly.

## Simple Pipeline Interface

For building streaming GPU pipelines with automatic buffer management, wgsl-fft provides a simple, declarative interface based on the bucket brigade pattern. This interface is designed to be used both by external consumers (like gpu-pipeline) and internally within wgsl-fft itself.

### Key Concepts

#### Bucket Brigade Pattern

The bucket brigade pattern uses a global `PingPongState` to control buffer read/write assignments across all stages:
- **Tick N**: All stages read from buffer index 0 and write to buffer index 1
- **Tick N+1**: All stages read from buffer index 1 and write to buffer index 0

This creates automatic data flow between stages without manual synchronization.

#### Core Components

| Component | Purpose |
|-----------|---------|
| `PingPongState` | Controls global read/write buffer indexing |
| `PingPongBuffers` | A pair of buffers for ping-pong operations |
| `ComputeStage` | Trait for pipeline stages that encode GPU operations |
| `FftStage` | Built-in FFT stage (forward or inverse) |
| `NormalizeStage` | Built-in normalization stage for IFFT scaling |
| `PipelineBuilder` | Declarative builder for creating streaming GPU pipelines |
| `Pipeline` | Executes stages in sequence with automatic buffer management |

### Basic Usage

```rust
use wgsl_fft::pipeline_builder::{PipelineBuilder, FftDirection, PingPongState};
use wgsl_fft::FftPipelines;

// Create FftPipelines for GPU operations
let fft_pipelines = FftPipelines::new().expect("GPU required");

// Build a simple FFT->Normalize->IFFT pipeline
let pipeline = PipelineBuilder::new(
    fft_pipelines.device().clone(),
    fft_pipelines.queue().clone()
)
.fft(FftDirection::Forward)      // Stage 1: Forward FFT
.normalize()                     // Stage 2: Normalize (scale by 1/N)
.fft(FftDirection::Inverse)     // Stage 3: Inverse FFT
.build();

// Execute the pipeline
let mut state = PingPongState::default();
let n = 1024;
let batch_size = 1;

// Execute one tick - all stages process data simultaneously
let command_buffer = pipeline.tick(&mut state, n, batch_size);

// Submit to GPU
pipeline.queue().submit(std::iter::once(command_buffer));
```

### Managing Ping-Pong State

```rust
use wgsl_fft::pipeline_builder::PingPongState;

let mut state = PingPongState::default();

// Initial state: read from buffer 0, write to buffer 1
assert_eq!(state.read_index(), 0);
assert_eq!(state.write_index(), 1);

// Toggle state for next tick
state.toggle();

// After toggle: read from buffer 1, write to buffer 0
assert_eq!(state.read_index(), 1);
assert_eq!(state.write_index(), 0);
```

### Creating Custom Stages

You can implement the `ComputeStage` trait to create custom processing stages:

```rust
use wgsl_fft::pipeline_builder::{ComputeStage, PipelineBuilder};
use wgpu::{CommandEncoder, Buffer};

struct MyCustomStage {
    // Your stage-specific parameters
    scale_factor: f32,
}

impl ComputeStage for MyCustomStage {
    fn name(&self) -> &str {
        "my_custom_stage"
    }

    fn encode(
        &self,
        encoder: &mut CommandEncoder,
        input: &Buffer,
        output: &Buffer,
        n: usize,
        batch_size: u32,
    ) {
        // Encode your custom compute operations here
        // Use encoder to dispatch compute shaders, etc.
    }
}

// Use your custom stage in a pipeline
let pipeline = PipelineBuilder::new(device, queue)
    .fft(FftDirection::Forward)
    .add_stage(Box::new(MyCustomStage { scale_factor: 2.0 }))
    .build();
```

### Parameter Passing

The pipeline interface supports type-safe parameter passing for stage configuration:

```rust
use wgsl_fft::pipeline_builder::PipelineParameters;

let mut params = PipelineParameters::new();
params.insert(1024usize);      // FFT size
params.insert(1u32);          // Batch size
params.insert(0.5f32);        // Some scale factor

// Stages can retrieve their specific parameters
if let Some(n) = params.get::<usize>() {
    println!("FFT size: {}", n);
}
```

### Using Ping-Pong Buffers

```rust
use wgsl_fft::pipeline_builder::{PingPongBuffers, PingPongState};

// Create a pair of buffers for ping-pong operations
let buffers = PingPongBuffers::new(&device, 1024 * 8, "fft_data");

// Get read/write buffers based on current state
let (read_buf, write_buf) = buffers.get(PingPongState::Read0Write1);

// After state toggle, the read/write buffers will be swapped
let mut state = PingPongState::Read0Write1;
state.toggle();
let (read_buf, write_buf) = buffers.get(state);
// Now read_buf is buffers[1] and write_buf is buffers[0]
```

### Complete Example: FFT Pipeline

Here's a complete example showing how to build and run a streaming FFT pipeline:

```rust
use wgsl_fft::pipeline_builder::{PipelineBuilder, FftDirection, PingPongState, PingPongBuffers};
use wgsl_fft::FftPipelines;

// Setup
let fft_pipelines = FftPipelines::new().expect("GPU required");
let device = fft_pipelines.device();
let queue = fft_pipelines.queue();

// Create input/output buffers
let input_data = PingPongBuffers::new(device, 1024 * 8, "input");
let output_data = PingPongBuffers::new(device, 1024 * 8, "output");

// Build pipeline
let pipeline = PipelineBuilder::new(device.clone(), queue.clone())
    .fft(FftDirection::Forward)
    .normalize()
    .build();

// Execute multiple ticks
let mut state = PingPongState::default();
let n = 1024;
let batch_size = 1;

for _ in 0..10 {
    // Get current read/write buffers
    let (input_read, input_write) = input_data.get(state);
    let (output_read, output_write) = output_data.get(state);
    
    // Execute pipeline tick
    let cmd_buf = pipeline.tick(&mut state, n, batch_size);
    
    // Submit to GPU
    queue.submit(std::iter::once(cmd_buf));
    
    // State automatically toggled for next iteration
}
```

This simple interface hides the complexity of GPU pipeline management while providing full flexibility for custom stages and configurations.

## Integration with wgsl-ping-pong-pipeline

wgsl-fft provides `PipelineStage` implementations that can be used as custom stages in the [wgsl-ping-pong-pipeline](https://github.com/larsjoost/wgsl-ping-pong-pipeline) library. This allows you to use wgsl-fft's FFT operations as part of a larger ping-pong pipeline without the pipeline needing to know anything about wgsl-fft.

### Feature Flag

The integration module is gated behind the `ping_pong` feature flag. To enable it:

```toml
[dependencies]
wgsl-fft = { version = "0.4.4", features = ["ping_pong"] }
```

### Available Stages

The `ping_pong_integration` module provides:

| Stage | Purpose | Side Inputs |
|-------|---------|-------------|
| `FftPipelineStage::forward(n, batch_size)` | Forward FFT | None |
| `FftPipelineStage::inverse(n, batch_size)` | Inverse FFT (with normalization) | None |
| `MultiplyPipelineStage::new(n, batch_size)` | Complex multiplication | `"input_b"` |

### Example: FFT-Based Convolution

```rust
use std::sync::Arc;
use wgsl_ping_pong_pipeline::{Pipeline, StageConfig};
use wgsl_fft::ping_pong_integration::{FftPipelineStage, MultiplyPipelineStage};

// Create pipeline: FFT -> Multiply (with pre-computed FFT(B)) -> IFFT
let mut pipeline = Pipeline::new()
    .pipe_config(StageConfig::Custom(
        Box::new(FftPipelineStage::forward(1024, 1))
    ))
    .pipe_config(StageConfig::Custom(
        Box::new(MultiplyPipelineStage::new(1024, 1))
    ))
    .pipe_config(StageConfig::Custom(
        Box::new(FftPipelineStage::inverse(1024, 1))
    ))
    .build()
    .await?;

// Add pre-computed FFT(B) as a side input
let fft_b_buffer: Arc<wgpu::Buffer> = /* ... */;
pipeline.add_side_input("input_b", Arc::clone(&fft_b_buffer));

// Write input signal A
let input_a: Vec<f32> = /* complex data as interleaved re/im */;
pipeline.write_input(&input_a).await?;

// Propagate through pipeline
pipeline.tick().await?;  // FFT(A)
pipeline.tick().await?;  // FFT(A) * FFT(B)
pipeline.tick().await?;  // IFFT(FFT(A) * FFT(B)) = convolution result

// Read output
let result = pipeline.read_output().await?;
```

This integration demonstrates the clean separation of concerns: wgsl-ping-pong-pipeline remains completely general, while wgsl-fft provides the domain-specific FFT implementations.

## Performance Characteristics

Performance measurements are from **NVIDIA GTX 1070** with Vulkan backend, release build.

### Single FFT Performance (Power-of-Two Sizes)

| Size (N)  | Throughput      | GFLOPS  | Latency      |
|------------|-----------------|---------|--------------|
| 256        | 2.7 MSamples/s  | 0.11    | 0.095 ms     |
| 1,024      | 9.6 MSamples/s  | 0.48    | 0.107 ms     |
| 4,096      | 32.0 MSamples/s | 1.92    | 0.128 ms     |
| 16,384     | 86.2 MSamples/s | 6.04    | 0.190 ms     |
| 65,536     | 145 MSamples/s  | 11.60   | 0.452 ms     |
| 262,144    | 160 MSamples/s  | 14.45   | 1.632 ms     |

### Power-of-Two vs Non-Power-of-Two Throughput Comparison

Non-power-of-two sizes use Bluestein's algorithm with GPU-accelerated power-of-2 FFTs internally.

| Size (N)  | Type            | Throughput      | GFLOPS  | Overhead |
|------------|-----------------|-----------------|---------|----------|
| 4,096      | Power-of-2      | 32.0 MSamples/s | 1.92    | baseline |
| 4,095      | Non-POT (Bluestein) | 22.1 MSamples/s | 1.33    | ~31% slower |
| 4,097      | Non-POT (Bluestein) | 21.8 MSamples/s | 1.31    | ~32% slower |
| 8,192      | Power-of-2      | 71.5 MSamples/s | 5.06    | baseline |
| 8,191      | Non-POT (Bluestein) | 48.3 MSamples/s | 3.42    | ~32% slower |
| 16,384     | Power-of-2      | 86.2 MSamples/s | 6.04    | baseline |
| 15,000     | Non-POT (Bluestein) | 56.1 MSamples/s | 3.85    | ~35% slower |

> **Note**: Bluestein's algorithm for non-power-of-two sizes achieves **~65-70%** of the throughput of native power-of-two sizes. The overhead comes from the additional convolution steps (3 FFTs: forward chirp, multiply, inverse chirp) required by Bluestein's method.

### Batch Processing Performance

Throughput **increases with batch size** due to reduced per-call overhead:

| Batch Size | FFT Size | Total Throughput | Speedup vs Single |
|------------|----------|------------------|------------------|
| 1          | 4,096    | 32.0 MSamples/s  | 1.0x (baseline)  |
| 4          | 4,096    | 98.5 MSamples/s  | 3.1x             |
| 8          | 4,096    | 162 MSamples/s   | 5.1x             |
| 16         | 4,096    | 205 MSamples/s   | 6.4x             |
| 32         | 4,096    | 228 MSamples/s   | 7.1x             |
| 64         | 4,096    | 240 MSamples/s   | 7.5x             |

**Why throughput increases with batch size:**
- **Reduced kernel launch overhead**: A single `queue.submit()` processes all FFTs in the batch
- **Better GPU utilization**: Keeps compute units busy with continuous work instead of idle between submissions
- **Memory locality**: Sequential processing of similar-sized FFTs improves cache efficiency
- **Amortized fixed costs**: Setup/teardown costs are spread across more FFTs

> **Optimal batch sizes**: For FFT sizes \>=4096, batch sizes of **8-64** provide the best throughput. Beyond ~64, diminishing returns set in as GPU memory bandwidth becomes the bottleneck.

## Numerical Accuracy

- **Precision**: All computations use single-precision (`f32`) floating point
- **Bluestein precision**: Uses `f64` for precomputations (twiddle factors, chirps) to maintain accuracy
- **Max error**: < **1e-3** element-wise L₂ error vs. `rustfft` for N=1024
- **Roundtrip accuracy**: FFT(IFFT(x)) ≈ x within numerical precision
- **Batch consistency**: Batch processing maintains identical numerical accuracy to single-vector processing

## Shader Development

The canonical shader source is in [`src/shaders.rs`](src/shaders.rs) as raw WGSL strings:

| Shader | Purpose |
|--------|---------|
| `R4_WGSL` | Stockham Radix-4 kernel (main FFT algorithm for even stages) |
| `R2_WGSL` | Stockham Radix-2 kernel (for odd stages when N is not divisible by 4) |
| `BLUESTEIN_CHIRP_WGSL` | Bluestein chirp multiplication kernel |
| `BLUESTEIN_INV_CHIRP_WGSL` | Bluestein inverse chirp multiplication kernel |
| `BLUESTEIN_ZERO_PAD_WGSL` | Zero-padding kernel for Bluestein's algorithm |

The Stockham autosort kernel implements the entire FFT in a single compute shader that processes all log₂(N) stages sequentially.

To test shader changes:
```sh
cargo test --release
```

## Performance Optimization

The implementation includes several optimizations:

- **Power-of-2**: Single compute pass (all log₂(N) stages in one shader dispatch)
- **Arbitrary sizes**: Multiple optimized passes using Bluestein's convolution
- **Ping-pong buffers**: Eliminates memory hazards between stages
- **Optimized synchronization**: Reduced CPU-GPU synchronization overhead
- **Efficient memory access**: Coalesced memory reads/writes for better GPU utilization
- **Automatic fallback**: Uses wgpu's software fallback adapter when no GPU is available

**For best performance:**
- Use `--release` builds (debug builds are significantly slower)
- Target larger FFT sizes (\>=4096) where GPU advantages are most pronounced
- Use batch processing for multiple transforms (8-64 batch size optimal)
- Ensure your system has a modern wgpu-compatible GPU

## License

MIT License - Copyright (c) 2024 Lars Joost Jensen

See [LICENSE](LICENSE) for details.
