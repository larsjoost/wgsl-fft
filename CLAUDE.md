# CLAUDE.md — wgsl-fft Coding Standards

## Error Handling
- Prefer `Result<T, E>` over `assert!` / `panic!` for validation at API boundaries
- Use descriptive error types rather than `Box<dyn Error>` in public APIs
- No `unwrap()` in library code; only in tests and examples

## Safety
- Avoid `unsafe` blocks; only use them when there is no safe alternative (e.g., FFI calls to C GPU libraries)
- Every `unsafe` block must have a `// SAFETY:` comment explaining why it is sound

## Comments
- Explain WHY, not WHAT; never restate what the code says
- No TODO without a tracking comment explaining the known limitation
- No debug artifacts: `println!`, `dbg!`, `eprintln!` must not be committed

## Documentation
- All public items (`pub fn`, `pub struct`, `pub trait`) must have a doc comment
- WGSL workgroup sizes must be documented with hardware rationale

## Testing
- New GPU kernels must have a correctness test comparing against RustFFT with tolerance ≤ 1e-3
- Tests must not use `unwrap()` silently — prefer `expect("reason")` for clarity

## Performance
- Cache GPU resources (buffers, bind groups) per FFT size; no per-call allocations
- Magic numeric constants must have a comment explaining their origin

## Style
- snake_case for all identifiers
- No near-duplicate code blocks — extract shared logic into a function
- No unnecessary nesting; flatten with early returns and guard clauses
- Small functions. Functions size should optimally be 5 lines of code