# CI/CD Setup Guide

This document explains how to set up and use the CI/CD pipeline for the wgsl-fft library.

## 📁 Files Created

### `.github/workflows/ci.yml`

A comprehensive GitHub Actions workflow that runs on every push and pull request to the `main` branch.

## 🔧 CI Pipeline Jobs

### 1. **Test Job**
- **Purpose**: Run all tests and build the project
- **Steps**:
  - Checkout code
  - Setup Rust toolchain
  - Install system dependencies (for wgpu)
  - Build the project
  - Run all tests
  - Check formatting
  - Run clippy

### 2. **Lint Job**
- **Purpose**: Code quality checks
- **Steps**:
  - Checkout code
  - Setup Rust toolchain with clippy and rustfmt
  - Check code formatting
  - Run clippy lints

### 3. **Doc Test Job**
- **Purpose**: Verify documentation examples
- **Steps**:
  - Checkout code
  - Setup Rust toolchain
  - Run documentation tests

### 4. **Benchmark Job**
- **Purpose**: Performance benchmarking
- **Steps**:
  - Checkout code
  - Setup Rust toolchain
  - Install system dependencies
  - Run benchmark tests

## 🚀 How to Use

### Local Testing

To test the CI setup locally before pushing:

```bash
# Test the complete workflow locally using act (GitHub Actions runner)
act -j test

# Or test specific jobs
act -j test
act -j lint
act -j doc-test
```

### GitHub Actions

The workflow will automatically run on:
- Every push to `main` branch
- Every pull request targeting `main` branch

### Manual Trigger

To manually trigger the workflow:

```bash
# From GitHub UI:
1. Go to "Actions" tab
2. Select "CI" workflow
3. Click "Run workflow" dropdown
4. Select branch and run

# Or via GitHub CLI:
gh workflow run ci.yml
```

## 📊 CI Badges

Add these badges to your README.md:

```markdown
[![CI Status](https://github.com/your-username/wgsl-fft/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/wgsl-fft/actions/workflows/ci.yml)
[![Lint Status](https://github.com/your-username/wgsl-fft/actions/workflows/ci.yml/badge.svg?branch=main&event=push)](https://github.com/your-username/wgsl-fft/actions/workflows/ci.yml)
```

## 🔧 Customization

### Adding More Tests

To add additional test jobs, add them to the `jobs` section in `.github/workflows/ci.yml`:

```yaml
  your_test_job:
    name: Your Test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - name: Run your custom tests
        run: cargo test --your-flags
```

### Environment Variables

Add environment variables in the `env` section:

```yaml
env:
  CARGO_TERM_COLOR: always
  YOUR_VAR: value
```

### Matrix Testing

To test across multiple Rust versions:

```yaml
strategy:
  matrix:
    rust:
      - stable
      - beta
      - nightly
    os:
      - ubuntu-latest
      - macos-latest
      - windows-latest
```

## 📖 CI Best Practices

### 1. Fast Feedback
- Keep CI runs under 5 minutes
- Parallelize jobs when possible
- Use caching (Swatinem/rust-cache)

### 2. Comprehensive Testing
- Test on multiple platforms
- Include linting and formatting
- Test documentation examples
- Add benchmark tests

### 3. Security
- Use official GitHub Actions
- Pin action versions
- Limit secrets exposure

### 4. Maintainability
- Document CI setup
- Keep workflows simple
- Add comments for complex steps

## 🎯 CI/CD Features Implemented

✅ **Automatic testing** on push/PR
✅ **Code quality checks** (clippy, rustfmt)
✅ **Documentation testing** (doc tests)
✅ **Performance benchmarking**
✅ **Dependency caching** for fast builds
✅ **Multi-platform support** (Ubuntu)
✅ **GPU dependency installation**
✅ **Parallel job execution**

## 🔗 Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Rust Cache Action](https://github.com/Swatinem/rust-cache)
- [Rust Toolchain Action](https://github.com/dtolnay/rust-toolchain)
- [wgpu Setup Guide](https://wgpu.rs/)

---

**CI/CD Setup Complete!** 🎉

The workflow is ready to use and will automatically test your code on every push and pull request.
