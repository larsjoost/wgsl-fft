# NVIDIA Docker Scripts for FFT Rivalry Leaderboard

This directory contains scripts to run the FFT rivalry leaderboard in NVIDIA Docker containers with CUDA support.

## Prerequisites

To use these scripts, you need:

1. **Docker** installed on your system
2. **NVIDIA Container Toolkit** installed
3. **NVIDIA GPU** with proper drivers
4. **Linux system** (these scripts are designed for Linux)

## Installation

### 1. Install Docker

```bash
# For Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER
```

### 2. Install NVIDIA Container Toolkit

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 3. Verify Installation

```bash
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

This should show your NVIDIA GPU information.

## Available Scripts

### 1. `build_and_run_docker.sh` (Recommended)

**Main script** that uses a dedicated Dockerfile for better maintainability.

**Usage:**
```bash
./scripts/build_and_run_docker.sh
```

**Features:**
- Uses dedicated `scripts/Dockerfile.cuda` for Docker configuration
- Builds a custom Docker image with all dependencies
- Caches the built image for faster subsequent runs
- Automatically detects NVIDIA GPU availability
- Clean separation of Docker configuration from script logic

### 2. `run_leaderboard_nvidia_simple.sh`

**Simpler script** that uses the official NVIDIA CUDA image and installs dependencies on-the-fly.

**Usage:**
```bash
./scripts/run_leaderboard_nvidia_simple.sh
```

**Features:**
- Uses official NVIDIA CUDA Docker image
- Installs Rust and dependencies when container starts
- No custom image build required
- Good for quick testing

### 3. `scripts/Dockerfile.cuda`

**Dedicated Dockerfile** for FFT rivalry leaderboard with CUDA support.

**Usage:**
```bash
# Build manually
docker build -t fft-rivalry-leaderboard -f scripts/Dockerfile.cuda .

# Run manually
docker run --gpus all --rm -it fft-rivalry-leaderboard
```

**Features:**
- Clean, maintainable Docker configuration
- Proper CUDA environment setup
- Rust installation and project build
- Can be used independently or with scripts

## Expected Output

When running successfully, you should see:

```
🟢 NVIDIA GPU detected - cuFFT benchmarks will be included
=== WGSL-FFT RIVALRY LEADERBOARD ===

--- N = 256 ---
                  Implementation |    Batch |     MSamples/s |     GFLOPS |   Status
------------------------------------------------------------------------------------
     Baseline (Stockham Radix-2) |     1024 |         134.23 |       5.37 |     PASS
                   Radix-4 Rival |     1024 |          55.05 |       2.20 | FAIL(1.36e-1)
Radix-4 Proper (Mixed Radix-4/2) |     1024 |         152.03 |       6.08 |     PASS
Claude (Stockham Radix-8/4/2 Mixed) |     1024 |         149.54 |       5.98 |     PASS
Codex (Stockham Radix-8/4/2 Mixed, HW-Preferred) |     1024 |         175.99 |       7.04 |     PASS
   Gemini (Mixed-Radix Stockham) |     1024 |         151.13 |       6.05 |     PASS
cuFFT (NVIDIA Gold Standard) |     1024 |        250.45 |      10.02 |     PASS
```

## Troubleshooting

### "NVIDIA Container Toolkit may not be properly installed"

This means either:
1. NVIDIA Container Toolkit is not installed
2. Docker is not configured to use NVIDIA GPUs
3. No NVIDIA GPU is available on your system

**Solution:** Install NVIDIA Container Toolkit as shown above and restart Docker.

### "Failed to build with CUDA features"

This can happen if:
1. CUDA toolkit is not properly installed in the container
2. There are version mismatches between CUDA components

**Solution:** Try the simple script first, or check CUDA version compatibility.

### "Permission denied" errors

Make sure the scripts are executable:
```bash
chmod +x scripts/run_leaderboard_nvidia*.sh
```

And that your user is in the docker group:
```bash
sudo usermod -aG docker $USER
newgrp docker  # or log out and back in
```

## Dockerfile Details

The `scripts/Dockerfile.cuda` provides a clean, maintainable Docker configuration:

```dockerfile
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Key features:
- Ubuntu 22.04 base with CUDA 11.8
- Rust installation via rustup
- Proper CUDA environment variables
- Project build with --features cuda
- Automatic verification of CUDA installation
```

**Customization:** You can modify the Dockerfile to:
- Change CUDA version (e.g., `nvidia/cuda:12.0-devel-ubuntu22.04`)
- Add additional system dependencies
- Modify Rust installation options
- Change build flags or environment variables

## Performance Notes

- The first run will be slower as it downloads Docker images and installs dependencies
- Subsequent runs will be much faster due to Docker caching
- GPU-accelerated cuFFT should show significantly higher performance than WebGPU implementations
- Release mode (`--release`) is used for accurate performance measurements

## Alternative: Local CUDA Build

If you have CUDA installed locally, you can run without Docker:

```bash
# Install CUDA toolkit locally first
cargo build --release --features cuda
cargo run --example rivalry_leaderboard --release --features cuda
```

This requires:
- NVIDIA CUDA Toolkit 11.8 or compatible version
- Proper environment variables (CUDA_HOME, LD_LIBRARY_PATH, etc.)
- NVIDIA drivers installed on host system

## License

These scripts are provided under the MIT License, same as the main project.