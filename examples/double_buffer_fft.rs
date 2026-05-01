//! Double-buffered FFT pipeline using [`wgsl_fft::GpuFft`].
//!
//! Demonstrates how to pipeline GPU work across consecutive batches with a
//! single compute pipeline and two independent buffer sets (slots).
//!
//! Sequential (slot 0 only):
//!
//!   CPU:  [upload 0          ][proc 0  ][upload 1          ][proc 1  ] ...
//!   DMA:  [H→D 0  ]           [D→H 0  ][H→D 1  ]           [D→H 1  ] ...
//!   GPU:            [fft 0   ]                    [fft 1   ]          ...
//!
//! Double-buffered (slot 0 / slot 1 alternating):
//!
//!   CPU:  [upload 0][upload 1][proc 0   ][upload 2][proc 1   ][upload 3] ...
//!   DMA:  [H→D 0  ][H→D 1  ]  [D→H 0  ][H→D 2  ] [D→H 1  ]           ...
//!   GPU:            [fft 0   ][fft 1   ]            [fft 2   ]          ...
//!          ─── GPU and DMA engines overlap between consecutive batches ───
//!
//! Each slot owns a complete, independent set of GPU buffers:
//!   buf_a + buf_b — FFT ping-pong storage (Stockham autosort)
//!   staging       — CPU-mappable readback buffer
//!   stage_bgs     — one bind group per FFT stage, bound to this slot's buffers
//!
//! `submit()` uploads a batch, encodes all FFT stages, copies the result to the
//! staging buffer, and returns a `wgpu::SubmissionIndex`.  `collect()` blocks
//! only on that specific submission via `poll(Wait { submission_index: … })`,
//! letting the other slot's work continue on the GPU uninterrupted.

use std::num::NonZeroU64;
use std::time::Instant;

use bytemuck;
use num_complex::Complex;
use wgsl_fft::GpuFft;

// ── Buffer slot ───────────────────────────────────────────────────────────────

/// One complete set of GPU buffers for a single FFT batch.
///
/// The pipeline owns two of these and alternates between them each submission.
struct Slot {
    /// Ping-pong input/source buffer (also used as the final output when
    /// `result_in_b == false`).
    buf_a: wgpu::Buffer,
    /// Ping-pong destination buffer (final output when `result_in_b == true`).
    buf_b: wgpu::Buffer,
    /// CPU-mappable readback buffer — result is copied here after the last stage.
    staging: wgpu::Buffer,
    /// One bind group per FFT stage: even stages read A/write B, odd read B/write A.
    stage_bgs: Vec<wgpu::BindGroup>,
    /// True when the last FFT stage wrote to buf_b (log_n is odd).
    result_in_b: bool,
    /// Byte size of one full batch: `n * batch_size * 2 * sizeof(f32)`.
    buf_bytes: u64,
}

fn build_slot(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    n: usize,
    batch_size: usize,
) -> Slot {
    let buf_bytes = (n * batch_size * 2 * std::mem::size_of::<f32>()) as u64;
    let log_n = n.trailing_zeros();

    let make_storage = |label: &str| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: buf_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };

    let buf_a = make_storage("buf_a");
    let buf_b = make_storage("buf_b");

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: buf_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Baseline pipeline is now Radix-4: ⌊log₄N⌋ stages, U.y = p (= 4^s).
    let num_r4 = (log_n / 2) as usize;
    let total_stages = num_r4; // even log_n only; odd sizes would need +1 R2 stage

    // Uniform buffer: one stride-aligned entry per R4 stage — {n, p, log_n, _pad}
    let alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
    let stride = 16u64.div_ceil(alignment) * alignment; // sizeof([u32;4]) = 16
    let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("uniforms"),
        size: stride * total_stages.max(1) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    for s in 0..num_r4 {
        let p = 1u32 << (s as u32 * 2); // p = 4^s
        let entry: [u32; 4] = [n as u32, p, log_n, 0];
        queue.write_buffer(
            &uniform_buf,
            stride * s as u64,
            bytemuck::cast_slice(&entry),
        );
    }

    // Twiddle factors: N complex pairs e^{-2πij/N} (R4 accesses up to index 3N/2).
    let twiddles: Vec<f32> = (0..n)
        .flat_map(|j| {
            let a = -std::f32::consts::TAU * j as f32 / n as f32;
            [a.cos(), a.sin()]
        })
        .collect();
    let twiddle_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("twiddles"),
        size: (twiddles.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&twiddle_buf, 0, bytemuck::cast_slice(&twiddles));

    // Bind groups: R4 stage s reads from A when s is even, from B when s is odd.
    let layout = pipeline.get_bind_group_layout(0);
    let uniform_sz = NonZeroU64::new(16);
    let make_bg = |src: &wgpu::Buffer, dst: &wgpu::Buffer, slot: usize| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform_buf,
                        offset: stride * slot as u64,
                        size: uniform_sz,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dst.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: twiddle_buf.as_entire_binding(),
                },
            ],
        })
    };

    let stage_bgs = (0..num_r4)
        .map(|s| {
            let (src, dst) = if s % 2 == 0 {
                (&buf_a, &buf_b)
            } else {
                (&buf_b, &buf_a)
            };
            make_bg(src, dst, s)
        })
        .collect();

    Slot {
        buf_a,
        buf_b,
        staging,
        stage_bgs,
        result_in_b: total_stages % 2 == 1,
        buf_bytes,
    }
}

// ── Pipeline ──────────────────────────────────────────────────────────────────

struct DoubleBufferFft {
    fft: GpuFft,
    slots: [Slot; 2],
    n: usize,
    batch_size: usize,
    /// Workgroups dispatched per R4 stage: ⌈N/4 / 256⌉
    wg_r4: u32,
}

impl DoubleBufferFft {
    fn new(n: usize, batch_size: usize) -> Self {
        assert!(n.is_power_of_two() && n > 1);
        let fft = GpuFft::new().expect("GPU required");
        let slots = std::array::from_fn(|_| {
            build_slot(
                fft.device(),
                &fft.queue,
                fft.compute_pipeline(),
                n,
                batch_size,
            )
        });
        let wg_r4 = (n as u32 / 4).div_ceil(256);
        Self {
            fft,
            slots,
            n,
            batch_size,
            wg_r4,
        }
    }

    /// Upload `inputs` to slot `s`, dispatch all FFT stages, copy the result to
    /// the staging buffer, and schedule a deferred `map_async`.
    ///
    /// Returns a `SubmissionIndex` — pass it to [`collect`] to block only on
    /// this batch while the other slot continues on the GPU uninterrupted.
    fn submit(&self, s: usize, inputs: &[Vec<Complex<f32>>]) -> wgpu::SubmissionIndex {
        assert_eq!(inputs.len(), self.batch_size);
        let slot = &self.slots[s];

        // ① Host → device: upload interleaved re/im floats into buf_a.
        let raw: Vec<f32> = inputs
            .iter()
            .flat_map(|v| v.iter())
            .flat_map(|c| [c.re, c.im])
            .collect();
        self.fft
            .queue
            .write_buffer(&slot.buf_a, 0, bytemuck::cast_slice(&raw));

        // ② GPU compute: run all Stockham stages using this slot's bind groups.
        //    Stage s reads from A if s is even, from B if s is odd — the bind
        //    groups already encode the correct src/dst buffer assignment.
        let mut enc = self
            .fft
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(self.fft.compute_pipeline());
            for bg in &slot.stage_bgs {
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(self.wg_r4, self.batch_size as u32, 1);
            }
        }

        // ③ Device → host: copy the final result buffer into the staging buffer.
        let result = if slot.result_in_b {
            &slot.buf_b
        } else {
            &slot.buf_a
        };
        enc.copy_buffer_to_buffer(result, 0, &slot.staging, 0, slot.buf_bytes);

        let idx = self.fft.queue.submit([enc.finish()]);

        // Queue the CPU mapping now, deferred until the GPU copy in ③ finishes.
        // `poll(Wait { Some(idx) })` in collect() processes this callback.
        slot.staging
            .slice(0..slot.buf_bytes)
            .map_async(wgpu::MapMode::Read, |_| {});

        idx
    }

    /// Block until submission `idx` (slot `s`) completes, then read and return
    /// the FFT output as `batch_size` vectors of length `n`.
    ///
    /// Only blocks on *this* submission — any other slot's in-flight work
    /// continues running on the GPU uninterrupted.
    fn collect(&self, s: usize, idx: wgpu::SubmissionIndex) -> Vec<Vec<Complex<f32>>> {
        let slot = &self.slots[s];

        self.fft
            .device()
            .poll(wgpu::PollType::Wait {
                submission_index: Some(idx),
                timeout: None,
            })
            .expect("device lost");

        let mapped = slot.staging.slice(0..slot.buf_bytes).get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&mapped);
        let result = floats
            .chunks_exact(2)
            .map(|c| Complex { re: c[0], im: c[1] })
            .collect::<Vec<_>>()
            .chunks(self.n)
            .map(<[_]>::to_vec)
            .collect();
        drop(mapped);
        slot.staging.unmap();

        result
    }
}

// ── Benchmark helpers ─────────────────────────────────────────────────────────

fn make_batch(seed: usize, n: usize, batch_size: usize) -> Vec<Vec<Complex<f32>>> {
    (0..batch_size)
        .map(|b| {
            (0..n)
                .map(|i| {
                    let t = (b * n + i) as f32 / n as f32;
                    Complex::new(
                        (seed as f32 * 0.1 + t).sin() * 0.001,
                        (seed as f32 * 0.1 + t * std::f32::consts::TAU).cos() * 0.001,
                    )
                })
                .collect()
        })
        .collect()
}

#[inline(never)]
fn consume(output: &[Vec<Complex<f32>>]) -> f32 {
    output
        .iter()
        .flat_map(|v| v.iter())
        .map(|c| c.norm_sqr())
        .sum()
}

// ── Sequential run ────────────────────────────────────────────────────────────

fn run_sequential(p: &DoubleBufferFft, batches: &[Vec<Vec<Complex<f32>>>]) -> (f64, f32) {
    let mut sink = 0.0f32;
    let start = Instant::now();
    for batch in batches {
        let idx = p.submit(0, batch);
        sink += consume(&p.collect(0, idx));
    }
    (start.elapsed().as_secs_f64(), sink)
}

// ── Double-buffered run ───────────────────────────────────────────────────────

fn run_double_buffered(p: &DoubleBufferFft, batches: &[Vec<Vec<Complex<f32>>>]) -> (f64, f32) {
    assert!(batches.len() >= 2, "need at least 2 batches to pipeline");
    let n = batches.len();
    let mut sink = 0.0f32;

    // Prime: submit batches 0 and 1 to separate slots so both are in flight.
    let idx0 = p.submit(0, &batches[0]);
    let idx1 = p.submit(1, &batches[1]);
    let mut pending: [Option<wgpu::SubmissionIndex>; 2] = [Some(idx0), Some(idx1)];

    let start = Instant::now();
    let mut next_submit = 2;
    let mut next_collect = 0;

    while next_collect < n {
        let slot = next_collect % 2;

        // Collect the oldest result for this slot.
        // While we wait here, the *other* slot's submission is still in flight —
        // the GPU continues computing and DMAs overlap with this readback.
        sink += consume(&p.collect(slot, pending[slot].take().unwrap()));

        // Re-use the now-free slot for the next batch.
        if next_submit < n {
            pending[slot] = Some(p.submit(slot, &batches[next_submit]));
            next_submit += 1;
        }

        next_collect += 1;
    }

    (start.elapsed().as_secs_f64(), sink)
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let n = 1024;
    let batch_size = 64;
    let num_batches = 32;
    let warmup = 4;

    let pipeline = DoubleBufferFft::new(n, batch_size);
    println!("N={n}, signals_per_batch={batch_size}, num_batches={num_batches}");

    let all_batches: Vec<Vec<Vec<Complex<f32>>>> = (0..num_batches + warmup)
        .map(|i| make_batch(i, n, batch_size))
        .collect();
    let (warm_batches, timed_batches) = all_batches.split_at(warmup);

    println!("\nWarming up ({warmup} batches)...");
    run_double_buffered(&pipeline, warm_batches);

    let total_samples = (n * batch_size * num_batches) as f64;

    let (seq_secs, seq_sink) = run_sequential(&pipeline, timed_batches);
    let (db_secs, db_sink) = run_double_buffered(&pipeline, timed_batches);

    println!(
        "\n{:>20} | {:>12} | {:>12} | {:>8}",
        "mode", "total (ms)", "ms/batch", "MS/s"
    );
    println!("{}", "-".repeat(62));
    println!(
        "{:>20} | {:>12.2} | {:>12.3} | {:>8.2}",
        "sequential",
        seq_secs * 1000.0,
        seq_secs * 1000.0 / num_batches as f64,
        total_samples / seq_secs / 1_000_000.0,
    );
    println!(
        "{:>20} | {:>12.2} | {:>12.3} | {:>8.2}",
        "double-buffered",
        db_secs * 1000.0,
        db_secs * 1000.0 / num_batches as f64,
        total_samples / db_secs / 1_000_000.0,
    );
    println!("\nSpeedup: {:.2}×", seq_secs / db_secs);

    let diff = (seq_sink - db_sink).abs() / seq_sink.abs().max(1e-9);
    println!("Output checksum diff: {diff:.2e}  (should be ~0)");
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── make_batch (pure) ─────────────────────────────────────────────────────

    #[test]
    fn make_batch_length() {
        let result = make_batch(0, 256, 4);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].len(), 256);
    }

    #[test]
    fn make_batch_finite_values() {
        for sig in make_batch(42, 64, 2) {
            for v in sig {
                assert!(v.re.is_finite() && v.im.is_finite());
            }
        }
    }

    #[test]
    fn make_batch_different_seeds_differ() {
        let a = make_batch(0, 64, 1);
        let b = make_batch(1, 64, 1);
        assert!(a[0]
            .iter()
            .zip(b[0].iter())
            .any(|(x, y)| (x - y).norm() > 1e-9));
    }

    // ── consume (pure) ────────────────────────────────────────────────────────

    #[test]
    fn consume_zero_input() {
        let zeros = vec![vec![Complex::<f32>::new(0.0, 0.0); 64]];
        assert_eq!(consume(&zeros), 0.0);
    }

    #[test]
    fn consume_non_negative() {
        let input = vec![(0..64)
            .map(|i| Complex::new(i as f32, -(i as f32)))
            .collect::<Vec<_>>()];
        assert!(consume(&input) > 0.0);
    }

    // ── GPU ───────────────────────────────────────────────────────────────────

    #[test]
    fn double_buffer_matches_sequential() {
        let n = 256;
        let batch_size = 4;
        let p = DoubleBufferFft::new(n, batch_size);
        let batches: Vec<_> = (0..6).map(|i| make_batch(i, n, batch_size)).collect();
        let (_, seq_sink) = run_sequential(&p, &batches);
        let (_, db_sink) = run_double_buffered(&p, &batches);
        let rel_diff = (seq_sink - db_sink).abs() / seq_sink.abs().max(1e-9);
        assert!(
            rel_diff < 1e-5,
            "checksums differ: seq={seq_sink:.6} db={db_sink:.6} rel={rel_diff:.2e}"
        );
    }

    #[test]
    fn double_buffer_throughput_positive() {
        let p = DoubleBufferFft::new(256, 2);
        let batches: Vec<_> = (0..4).map(|i| make_batch(i, 256, 2)).collect();
        let (secs, _) = run_double_buffered(&p, &batches);
        assert!(secs > 0.0);
    }
}
