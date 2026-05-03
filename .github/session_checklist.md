# End-of-Session Checklist

A checklist to verify at the end of each coding session.

## Code Quality
- [ ] `cargo build` - compiles without errors
- [ ] `cargo build --release` - release build compiles
- [ ] `cargo test --all` - all tests pass (including `cargo test --release`)
- [ ] `cargo fmt` - code formatting is correct
- [ ] `cargo clippy -- -D warnings` - no linting warnings (consider as errors)
- [ ] No unused imports or variables
- [ ] No compiler warnings

## Testing
- [ ] All new tests pass
- [ ] All existing tests still pass (no regressions)
- [ ] CI test suite passes (`scripts/ci_test.sh`)
- [ ] Edge cases tested:
  - [ ] Empty inputs
  - [ ] Single element inputs
  - [ ] Boundary values
  - [ ] Large inputs (if applicable)
- [ ] Integration tests pass
- [ ] Examples in documentation work

## Documentation
- [ ] Code comments are up to date
- [ ] Doc comments compile without errors (`cargo test --doc`)
- [ ] README.md is updated (if applicable)
- [ ] README examples work (if modified)
- [ ] CHANGELOG updated (if applicable)
- [ ] Commit messages are descriptive

## Performance
- [ ] No performance regressions (if applicable)
- [ ] Leaderboard/benchmarks still work (if applicable)
- [ ] Benchmark results are reasonable

## Cleanup
- [ ] No debug print statements left in code
- [ ] No `println!`, `eprintln!`, or `dbg!` in production code
- [ ] No temporary files or test artifacts in git
- [ ] All TODOs/FIXMEs addressed or documented as issues
- [ ] No hardcoded paths or credentials
- [ ] git status is clean (or changes are intentional)
- [ ] Staged changes are ready for commit

## Final Verification
- [ ] Run full test suite one final time: `cargo test --all`
- [ ] Verify all tests pass
- [ ] Check that CI would pass (if possible): `scripts/ci_test.sh`
