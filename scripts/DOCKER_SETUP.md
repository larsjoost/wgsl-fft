# Docker Setup for FFT Rivalry Leaderboard

This document provides a quick reference for the Docker-based setup.

## Quick Start

```bash
# Make scripts executable
chmod +x scripts/build_and_run_docker.sh

# Build and run with Docker
./scripts/build_and_run_docker.sh
```

## Files Structure

```
scripts/
├── Dockerfile.cuda          # Main Docker configuration
├── build_and_run_docker.sh  # Primary script (recommended)
├── run_leaderboard_nvidia_simple.sh  # Alternative simple script
├── README_NVIDIA_DOCKER.md  # Comprehensive documentation
└── DOCKER_SETUP.md          # This file
```

## Dockerfile.cuda

The dedicated Dockerfile provides:

- **Base Image**: `nvidia/cuda:11.8.0-devel-ubuntu22.04`
- **System Dependencies**: build tools, git, cmake, etc.
- **Rust Installation**: via rustup
- **CUDA Environment**: Proper PATH and LD_LIBRARY_PATH setup
- **Project Build**: `cargo build --release --features cuda`

## Build Options

### Option 1: Using the Script (Recommended)

```bash
./scripts/build_and_run_docker.sh
```

### Option 2: Manual Build

```bash
# Build the image
docker build -t fft-rivalry-leaderboard -f scripts/Dockerfile.cuda .

# Run with GPU
docker run --gpus all --rm -it fft-rivalry-leaderboard

# Run without GPU
docker run --rm -it fft-rivalry-leaderboard
```

### Option 3: Simple Script (No Custom Image)

```bash
./scripts/run_leaderboard_nvidia_simple.sh
```

## Docker Commands Cheat Sheet

```bash
# List images
docker images

# Remove image (free up ~5GB)
docker rmi fft-rivalry-leaderboard

# List running containers
docker ps

# Stop all containers
docker stop $(docker ps -aq)

# Remove all containers
docker rm $(docker ps -aq)

# Clean up unused images and containers
docker system prune
```

## GPU Detection

The scripts automatically detect NVIDIA GPU availability:

- **With GPU**: `--gpus all` flag is used
- **Without GPU**: Runs in CPU-only mode
- **Detection**: Uses `nvidia-smi` via NVIDIA Container Toolkit

## Requirements Checklist

- [ ] Docker installed
- [ ] NVIDIA Container Toolkit installed (for GPU support)
- [ ] NVIDIA drivers installed (for GPU support)
- [ ] Scripts are executable (`chmod +x scripts/*.sh`)
- [ ] Dockerfile.cuda exists in scripts/ directory

## Troubleshooting Tips

1. **Docker build fails**: Check internet connection and Docker daemon
2. **CUDA features not available**: Verify NVIDIA Container Toolkit installation
3. **Permission denied**: Ensure scripts are executable and user is in docker group
4. **Out of disk space**: Run `docker system prune` to clean up

## Performance Tips

- First build downloads ~5GB Docker image (takes time)
- Subsequent builds use cached image (much faster)
- Built image can be reused for multiple runs
- Use `--gpus all` only when NVIDIA GPU is available

## Customization

To modify the Docker setup:

1. **Edit Dockerfile.cuda**: Change base image, dependencies, or build flags
2. **Rebuild image**: `docker build -t fft-rivalry-leaderboard -f scripts/Dockerfile.cuda .`
3. **Run with new image**: `docker run --gpus all --rm -it fft-rivalry-leaderboard`

Common customizations:
- Change CUDA version (e.g., `nvidia/cuda:12.0-devel-ubuntu22.04`)
- Add more system dependencies in the Dockerfile
- Modify Rust installation options
- Change cargo build flags

## Expected Output

```
🟢 NVIDIA GPU detected - GPU acceleration will be enabled
🐳 Building Docker image with CUDA support...
✅ Docker image built successfully
🚀 Running FFT rivalry leaderboard...
=== WGSL-FFT RIVALRY LEADERBOARD ===
🟢 NVIDIA GPU detected - cuFFT benchmarks will be included
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

## Cleanup

After testing, you can clean up Docker resources:

```bash
# Remove the built image
docker rmi fft-rivalry-leaderboard

# Clean up all unused Docker resources
docker system prune -a
```

This will free up approximately 5GB of disk space.
