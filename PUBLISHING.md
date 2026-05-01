# Publishing Status and Guidelines

## Current Publishing Status

✅ **This crate can now be published to crates.io** - the `wgsl-rs` dependency has been removed.

## Previous Block

The project previously depended on a specific git commit of [`wgsl-rs`](https://github.com/schell/wgsl-rs) that used a different API than any published version:

```toml
# Old dependency (removed)
wgsl-rs = { git = "https://github.com/schell/wgsl-rs", rev = "07175c49dfc231f309c5fc6a4b86750d00dfd7cc" }
```

## Solution Implemented

All WGSL shader code has been converted from `wgsl-rs` procedural macros to raw WGSL string constants embedded directly in the Rust source files. This eliminates the external dependency entirely.

## Publishing to crates.io

To publish:

```bash
# Verify everything works
cargo test --all-targets
cargo package

# Publish
cargo publish
```

## Package Configuration

- **Package name on crates.io:** `wgsl-fft`
- **Lib name (for Rust imports):** `wgsl_fft`
- **Version:** 0.1.0
- **Dependencies:** wgpu, bytemuck, pollster, num-complex (no procedural macros)
