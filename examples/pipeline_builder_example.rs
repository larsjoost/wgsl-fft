//! Example demonstrating the new Streaming Pipeline API.
//!
//! This example shows how to build a multi-stage FFT pipeline using the
//! synchronous "Bucket Brigade" streaming pattern.
//!
//! Run with: cargo run --example pipeline_builder_example

use wgsl_fft::{
    ComputeStage, FftPipelines, FftStage, NormalizeStage, Pipeline as StreamingPipeline,
    PipelineParameters,
};

fn main() {
    println!("Streaming Pipeline Example");
    println!("==========================");
    println!();
    println!("This example demonstrates how to build a multi-stage GPU pipeline");
    println!("using the synchronous bucket-brigade pattern.");
    println!();

    // 1. Initialize GPU
    let fft = FftPipelines::new().expect("GPU required");
    let device = fft.device().clone();
    let queue = fft.queue().clone();

    println!("Initialized GPU: {}", "");

    // 2. Define stages
    // A simple Forward FFT -> Inverse FFT pipeline
    let stages: Vec<Box<dyn ComputeStage>> = vec![
        Box::new(FftStage::forward()),
        Box::new(FftStage::inverse()),
        Box::new(NormalizeStage),
    ];

    // 3. Build the pipeline
    // We process a batch of 1 vector, size 1024
    let n = 1024;
    let buffer_size = (n * 8) as u64; // complex floats
    let mut pipeline = StreamingPipeline::new(
        device,
        queue,
        fft,
        stages,
        1, // batch size
        buffer_size,
    );

    println!("Built pipeline with {} stages", 3);

    // 4. Run one tick
    println!("Ticking pipeline...");
    pipeline.tick(PipelineParameters::new());

    println!();
    println!("Example complete!");
    println!();
    println!("The streaming pipeline provides:");
    println!("  - High-throughput 'Bucket Brigade' architecture");
    println!("  - Simultaneous execution of all stages");
    println!("  - Automatic buffer flow between stages via Global Toggle");
    println!("  - Built-in stages: FftStage, MultiplyStage, NormalizeStage, NoiseStage");
    println!();
}
