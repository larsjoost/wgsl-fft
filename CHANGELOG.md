# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added `rust-version = "1.70"` to Cargo.toml
- Added `exclude` directive in Cargo.toml for examples and benches
- Added crate-level documentation to lib.rs
- Added `Default` implementations for `StockhamBaseline` and `ClaudeFft`
- Added `#[derive(Debug)]` to `GpuFft`

### Changed
- Updated README.md to clarify that arbitrary FFT sizes are supported (not just power-of-two)
- Removed `#[allow(dead_code)]` attributes from `SizeCache` fields in fft.rs

### Fixed
- Fixed clippy warnings in benchmark.rs (unnecessary casts, map_or simplification)
- Fixed clippy warnings in rivals/baseline.rs and rivals/claude.rs (missing Default impls)

## [0.3.1] - 2025-01-01

### Added
- Support for all FFT sizes (not just powers of 2) via Bluestein's algorithm
- Batch processing for multiple FFTs

## [0.3.0] - 2025-01-01

Initial release with Stockham Radix-4/2 implementation.
