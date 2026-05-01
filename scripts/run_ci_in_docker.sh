#!/bin/bash

# Script to run CI tests in Docker container
# This ensures consistent environment between local and GitHub CI

set -e  # Exit immediately if any command fails

docker build -t wgsl-fft-ci -f Dockerfile.ci .

docker run --rm \
    -v $(pwd):/workspace \
    -w /workspace \
    -e RUST_BACKTRACE=1 \
    -e CARGO_TERM_COLOR=always \
    wgsl-fft-ci \
    ./scripts/ci_test.sh