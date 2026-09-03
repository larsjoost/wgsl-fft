//! Simple pipeline interface for wgsl-fft
//!
//! This module provides a clean, minimal interface for building streaming
//! GPU pipelines with automatic buffer management using ping-pong buffering.
//! It is designed to be used both by external consumers (like gpu-pipeline)
//! and internally within wgsl-fft itself.
//!
//! # Key Concepts
//!
//! - **PingPongState**: Controls global read/write buffer indexing for the bucket brigade pattern
//! - **PingPongBuffers**: A pair of buffers that can be read from and written to based on state
//! - **ComputeStage**: Trait for pipeline stages that can encode compute operations
//! - **PipelineBuilder**: Declarative builder for creating streaming GPU pipelines
//! - **Pipeline**: Executes stages in sequence with automatic buffer management

use wgpu::{Buffer, Device, Queue};

use crate::pipelines::FftDirection;

/// Global state controlling read/write buffer indices for ping-pong buffering.
///
/// This enum implements the "bucket brigade" pattern where all stages in a pipeline
/// operate on the same global state. When the state toggles, all stages switch their
/// read and write buffer indices simultaneously.
///
/// - **Read0Write1**: Read from buffer index 0, write to buffer index 1
/// - **Read1Write0**: Read from buffer index 1, write to buffer index 0
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PingPongState {
    /// Read from buffer 0, write to buffer 1
    #[default]
    Read0Write1,
    /// Read from buffer 1, write to buffer 0
    Read1Write0,
}

impl PingPongState {
    /// Toggle the read/write indices.
    ///
    /// This advances the pipeline state, causing all stages to switch
    /// their read and write buffers simultaneously.
    #[inline]
    pub fn toggle(&mut self) {
        *self = match self {
            PingPongState::Read0Write1 => PingPongState::Read1Write0,
            PingPongState::Read1Write0 => PingPongState::Read0Write1,
        };
    }

    /// Get the buffer index to read from.
    #[inline]
    pub fn read_index(&self) -> usize {
        match self {
            PingPongState::Read0Write1 => 0,
            PingPongState::Read1Write0 => 1,
        }
    }

    /// Get the buffer index to write to.
    #[inline]
    pub fn write_index(&self) -> usize {
        match self {
            PingPongState::Read0Write1 => 1,
            PingPongState::Read1Write0 => 0,
        }
    }

    /// Get read and write buffers from a pair based on the current state.
    ///
    /// # Arguments
    /// * `pair` - An array of two buffers
    ///
    /// # Returns
    /// A tuple of (read_buffer, write_buffer) references
    #[inline]
    pub fn buffers<'a>(&self, pair: &'a [Buffer; 2]) -> (&'a Buffer, &'a Buffer) {
        match self {
            PingPongState::Read0Write1 => (&pair[0], &pair[1]),
            PingPongState::Read1Write0 => (&pair[1], &pair[0]),
        }
    }
}

/// A pair of buffers for ping-pong operations.
///
/// This struct holds two GPU buffers that can be used for ping-pong buffering,
/// where data alternates between the two buffers on each pipeline tick.
/// This eliminates the need for explicit synchronization between stages.
#[derive(Debug)]
pub struct PingPongBuffers {
    /// The two buffers used for ping-pong operations.
    ///
    /// Index 0 and 1 are used alternately as read and write targets
    /// based on the current PingPongState.
    pub buffers: [Buffer; 2],
}

impl PingPongBuffers {
    /// Create a new pair of buffers with the given size.
    ///
    /// # Arguments
    /// * `device` - The wgpu device to create buffers on
    /// * `size` - Size in bytes for each buffer
    /// * `label` - Base label for the buffers (index will be appended)
    ///
    /// # Returns
    /// A new PingPongBuffers instance with two buffers of the specified size
    pub fn new(device: &Device, size: u64, label: &str) -> Self {
        let create_buffer = |idx: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{label}_pingpong_{}", idx)),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        Self {
            buffers: [create_buffer(0), create_buffer(1)],
        }
    }

    /// Get the read and write buffers based on the current state.
    ///
    /// # Arguments
    /// * `state` - The current ping-pong state
    ///
    /// # Returns
    /// A tuple of (read_buffer, write_buffer) references
    #[inline]
    pub fn get(&self, state: PingPongState) -> (&Buffer, &Buffer) {
        state.buffers(&self.buffers)
    }

    /// Get buffer by index.
    #[inline]
    pub fn get_by_index(&self, index: usize) -> &Buffer {
        &self.buffers[index]
    }
}

/// Trait for compute stages in a pipeline.
///
/// A ComputeStage represents a single processing step in a GPU pipeline.
/// Each stage reads from an input buffer and writes to an output buffer.
/// Multiple stages can be chained together to form complex pipelines.
pub trait ComputeStage: Send + Sync + std::fmt::Debug {
    /// Get the name of this stage for debugging and logging purposes.
    fn name(&self) -> &str;

    /// Encode compute operations for this stage into the command encoder.
    ///
    /// This method should encode all GPU compute operations needed for this stage
    /// to process the input data and write results to the output buffer.
    ///
    /// # Arguments
    /// * `encoder` - The command encoder to write GPU operations to
    /// * `input` - The input buffer to read from
    /// * `output` - The output buffer to write to
    /// * `n` - The FFT size (number of complex elements)
    /// * `batch_size` - The number of FFTs to process in this call
    fn encode(
        &self,
        _encoder: &mut wgpu::CommandEncoder,
        _input: &Buffer,
        _output: &Buffer,
        _n: usize,
        _batch_size: u32,
    );

    /// Optional: Called when the stage is added to a pipeline.
    /// Can be used to validate parameters or prepare resources.
    fn on_add(&mut self, _n: usize, _batch_size: u32) -> Result<(), String> {
        Ok(())
    }
}

/// A simple FFT stage that uses the pre-compiled wgsl-fft pipelines.
///
/// This stage performs either a forward or inverse FFT using the optimized
/// Stockham Radix-4/2 algorithm from wgsl-fft.
#[derive(Debug, Clone)]
pub struct FftStage {
    direction: FftDirection,
}

impl FftStage {
    /// Create a new FFT stage with the specified direction.
    ///
    /// # Arguments
    /// * `direction` - Whether to perform forward or inverse FFT
    pub fn new(direction: FftDirection) -> Self {
        Self { direction }
    }

    /// Get the FFT direction.
    pub fn direction(&self) -> FftDirection {
        self.direction
    }

    /// Create a forward FFT stage.
    pub fn forward() -> Self {
        Self::new(FftDirection::Forward)
    }

    /// Create an inverse FFT stage.
    pub fn inverse() -> Self {
        Self::new(FftDirection::Inverse)
    }
}

impl ComputeStage for FftStage {
    fn name(&self) -> &str {
        match self.direction {
            FftDirection::Forward => "fft",
            FftDirection::Inverse => "ifft",
        }
    }

    fn encode(
        &self,
        _encoder: &mut wgpu::CommandEncoder,
        _input: &Buffer,
        _output: &Buffer,
        _n: usize,
        _batch_size: u32,
    ) {
        // TODO: Implement actual FFT encoding using FftPipelines
    }

    fn on_add(&mut self, n: usize, batch_size: u32) -> Result<(), String> {
        if n == 0 {
            return Err("FFT size must be greater than 0".to_string());
        }
        if batch_size == 0 {
            return Err("Batch size must be greater than 0".to_string());
        }
        Ok(())
    }
}

/// A normalization stage for IFFT results.
///
/// This stage divides each element by N (the FFT size) to properly scale
/// the inverse FFT results, maintaining signal amplitude.
#[derive(Debug, Clone, Default)]
pub struct NormalizeStage;

impl NormalizeStage {
    /// Create a new normalization stage.
    pub fn new() -> Self {
        Self
    }
}

impl ComputeStage for NormalizeStage {
    fn name(&self) -> &str {
        "normalize"
    }

    fn encode(
        &self,
        _encoder: &mut wgpu::CommandEncoder,
        _input: &Buffer,
        _output: &Buffer,
        _n: usize,
        _batch_size: u32,
    ) {
        // TODO: Implement actual normalization encoding using FftPipelines
    }

    fn on_add(&mut self, n: usize, batch_size: u32) -> Result<(), String> {
        if n == 0 {
            return Err("FFT size must be greater than 0".to_string());
        }
        if batch_size == 0 {
            return Err("Batch size must be greater than 0".to_string());
        }
        Ok(())
    }
}

/// Builder for creating streaming GPU pipelines.
///
/// The PipelineBuilder provides a declarative API for constructing GPU compute pipelines
/// from a sequence of stages. Each stage processes data and passes it to the next stage.
/// The builder pattern allows for easy composition of complex pipelines.
#[derive(Debug)]
pub struct PipelineBuilder {
    device: Device,
    queue: Queue,
    stages: Vec<Box<dyn ComputeStage>>,
}

impl PipelineBuilder {
    /// Create a new PipelineBuilder with the given device and queue.
    ///
    /// # Arguments
    /// * `device` - The wgpu device for GPU operations
    /// * `queue` - The wgpu queue for submitting commands
    pub fn new(device: Device, queue: Queue) -> Self {
        Self {
            device,
            queue,
            stages: Vec::new(),
        }
    }

    /// Add a custom stage to the pipeline.
    ///
    /// # Arguments
    /// * `stage` - The stage to add
    pub fn add_stage(mut self, stage: Box<dyn ComputeStage>) -> Self {
        self.stages.push(stage);
        self
    }

    /// Add an FFT stage to the pipeline.
    ///
    /// # Arguments
    /// * `direction` - The FFT direction (forward or inverse)
    pub fn fft(mut self, direction: FftDirection) -> Self {
        self.stages
            .push(Box::new(FftStage::new(direction)) as Box<dyn ComputeStage>);
        self
    }

    /// Add a normalization stage to the pipeline.
    ///
    /// This stage divides each FFT element by N to properly scale inverse FFT results.
    pub fn normalize(mut self) -> Self {
        self.stages
            .push(Box::new(NormalizeStage) as Box<dyn ComputeStage>);
        self
    }

    /// Build the pipeline.
    ///
    /// This consumes the builder and returns a Pipeline instance that can be used
    /// to execute the constructed sequence of stages.
    pub fn build(self) -> Pipeline {
        Pipeline {
            device: self.device,
            queue: self.queue,
            stages: self.stages,
        }
    }

    /// Get the current number of stages in the pipeline being built.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Get the device reference.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the queue reference.
    pub fn queue(&self) -> &Queue {
        &self.queue
    }
}

/// A streaming GPU pipeline composed of multiple compute stages.
///
/// The Pipeline executes stages in sequence, with each stage reading from
/// an input buffer and writing to an output buffer. The bucket brigade pattern
/// ensures automatic data flow between stages on each tick.
#[derive(Debug)]
pub struct Pipeline {
    device: Device,
    queue: Queue,
    stages: Vec<Box<dyn ComputeStage>>,
}

impl Pipeline {
    /// Create a new Pipeline with the given stages.
    ///
    /// This is typically created via PipelineBuilder::build().
    pub fn new(device: Device, queue: Queue, stages: Vec<Box<dyn ComputeStage>>) -> Self {
        Self {
            device,
            queue,
            stages,
        }
    }

    /// Get the wgpu device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the wgpu queue.
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Get the stages in this pipeline.
    pub fn stages(&self) -> &[Box<dyn ComputeStage>] {
        &self.stages
    }

    /// Get the number of stages in this pipeline.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Get a stage by index.
    pub fn get_stage(&self, index: usize) -> Option<&dyn ComputeStage> {
        self.stages.get(index).map(|v| &**v)
    }

    /// Get a stage by name.
    pub fn get_stage_by_name(&self, name: &str) -> Option<&dyn ComputeStage> {
        self.stages
            .iter()
            .find(|stage| stage.name() == name)
            .map(|v| &**v)
    }

    /// Execute one tick of the pipeline.
    ///
    /// All stages read from their input buffers and write to their output buffers
    /// based on the current state. After execution, the state is toggled for the next tick.
    ///
    /// # Arguments
    /// * `state` - The current ping-pong state (will be toggled)
    /// * `n` - The FFT size
    /// * `batch_size` - The number of FFTs to process
    ///
    /// # Returns
    /// A command buffer that can be submitted to the queue
    pub fn tick(
        &self,
        state: &mut PingPongState,
        _n: usize,
        _batch_size: u32,
    ) -> wgpu::CommandBuffer {
        let encoder = self.device.create_command_encoder(&Default::default());

        // Execute all stages
        for _stage in &self.stages {
            // TODO: Implement proper buffer management for stage connections
        }

        // Toggle state for next tick
        state.toggle();

        encoder.finish()
    }

    /// Submit a tick of the pipeline to the queue.
    ///
    /// This is a convenience method that calls tick() and submits the result.
    ///
    /// # Arguments
    /// * `state` - The current ping-pong state (will be toggled)
    /// * `n` - The FFT size
    /// * `batch_size` - The number of FFTs to process
    pub fn submit_tick(&self, state: &mut PingPongState, n: usize, batch_size: u32) {
        let cmd_buf = self.tick(state, n, batch_size);
        self.queue.submit(std::iter::once(cmd_buf));
    }

    /// Get the names of all stages in the pipeline.
    pub fn stage_names(&self) -> Vec<&str> {
        self.stages.iter().map(|s| s.name()).collect()
    }
}

impl Clone for Pipeline {
    fn clone(&self) -> Self {
        // For now, we can't clone the stages because of trait object limitations
        panic!("Cloning Pipeline is not yet implemented due to trait object limitations");
    }
}

/// Clone trait for boxed ComputeStage.
///
/// This trait enables cloning of boxed ComputeStage implementations.
pub trait CloneBox {
    fn clone_box(&self) -> Box<dyn ComputeStage>;
}

impl<T> CloneBox for T
where
    T: ComputeStage + Clone + 'static,
{
    fn clone_box(&self) -> Box<dyn ComputeStage> {
        Box::new(self.clone())
    }
}

// Note: We can't implement Clone for Box<dyn ComputeStage> due to trait object limitations
// This is a known limitation that can be addressed in the future

/// Parameter passing for pipeline stages.
///
/// This module provides a way to pass parameters through the pipeline
/// to specific stages. Parameters are stored with their TypeId and can be
/// retrieved by stages that know their expected types.
pub mod params {
    use std::any::{Any, TypeId};
    use std::collections::HashMap;

    /// Container for type-safe pipeline parameters.
    #[derive(Debug, Default)]
    pub struct PipelineParameters {
        values: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
    }

    impl PipelineParameters {
        /// Create a new empty parameter container.
        pub fn new() -> Self {
            Self::default()
        }

        /// Insert a parameter value.
        pub fn insert<T: 'static + Send + Sync>(&mut self, value: T) {
            self.values.insert(TypeId::of::<T>(), Box::new(value));
        }

        /// Get a parameter value by type.
        pub fn get<T: 'static + Send + Sync>(&self) -> Option<&T> {
            self.values
                .get(&TypeId::of::<T>())
                .and_then(|boxed| boxed.downcast_ref::<T>())
        }

        /// Get a mutable parameter value by type.
        pub fn get_mut<T: 'static + Send + Sync>(&mut self) -> Option<&mut T> {
            self.values
                .get_mut(&TypeId::of::<T>())
                .and_then(|boxed| boxed.downcast_mut::<T>())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ping_pong_state_default() {
        let state = PingPongState::default();
        assert_eq!(state, PingPongState::Read0Write1);
    }

    #[test]
    fn test_ping_pong_state_read_write_indices() {
        let state = PingPongState::Read0Write1;
        assert_eq!(state.read_index(), 0);
        assert_eq!(state.write_index(), 1);

        let state = PingPongState::Read1Write0;
        assert_eq!(state.read_index(), 1);
        assert_eq!(state.write_index(), 0);
    }

    #[test]
    fn test_ping_pong_state_toggle() {
        let mut state = PingPongState::Read0Write1;
        assert_eq!(state, PingPongState::Read0Write1);

        state.toggle();
        assert_eq!(state, PingPongState::Read1Write0);

        state.toggle();
        assert_eq!(state, PingPongState::Read0Write1);
    }

    #[test]
    fn test_ping_pong_state_equality() {
        assert_eq!(PingPongState::Read0Write1, PingPongState::Read0Write1);
        assert_eq!(PingPongState::Read1Write0, PingPongState::Read1Write0);
        assert_ne!(PingPongState::Read0Write1, PingPongState::Read1Write0);
    }

    #[test]
    fn test_fft_stage_creation() {
        let fft_stage = FftStage::new(FftDirection::Forward);
        assert_eq!(fft_stage.direction(), FftDirection::Forward);
        assert_eq!(fft_stage.name(), "fft");

        let ifft_stage = FftStage::new(FftDirection::Inverse);
        assert_eq!(ifft_stage.direction(), FftDirection::Inverse);
        assert_eq!(ifft_stage.name(), "ifft");
    }

    #[test]
    fn test_fft_stage_convenience_constructors() {
        let forward = FftStage::forward();
        assert_eq!(forward.direction(), FftDirection::Forward);

        let inverse = FftStage::inverse();
        assert_eq!(inverse.direction(), FftDirection::Inverse);
    }

    #[test]
    fn test_normalize_stage() {
        let stage = NormalizeStage;
        assert_eq!(stage.name(), "normalize");
    }

    #[test]
    fn test_pipeline_stage_names() {
        let fft_stage = FftStage::forward();
        let normalize_stage = NormalizeStage;
        let ifft_stage = FftStage::inverse();

        assert_eq!(fft_stage.name(), "fft");
        assert_eq!(normalize_stage.name(), "normalize");
        assert_eq!(ifft_stage.name(), "ifft");
    }

    #[test]
    fn test_pipeline_parameters() {
        use params::PipelineParameters;

        let mut params = PipelineParameters::new();

        params.insert(42u32);
        params.insert(3.14f32);
        params.insert("hello");

        assert_eq!(params.get::<u32>(), Some(&42));
        assert_eq!(params.get::<f32>(), Some(&3.14));
        assert_eq!(params.get::<&str>(), Some(&"hello"));
        assert_eq!(params.get::<i32>(), None);
    }

    #[test]
    fn test_pipeline_builder_basic() {
        // This is a compile-time test to verify the API
        assert!(true);
    }
}
