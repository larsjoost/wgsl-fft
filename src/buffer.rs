//! Ping-pong buffer infrastructure for GPU compute pipelines.

use wgpu::Device;

/// Ping-pong state for double-buffering.
/// A single external controller holds this state and passes it to all stages.
/// After each stage completes, the controller toggles the state so the next stage
/// reads from what the previous stage wrote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PingPongState {
    /// Read from buffer A (index 0), write to buffer B (index 1)
    Read0Write1,
    /// Read from buffer B (index 1), write to buffer A (index 0)
    Read1Write0,
}

impl PingPongState {
    /// Toggle the ping-pong state for the next stage.
    #[inline]
    pub fn toggle(&mut self) {
        *self = match self {
            Self::Read0Write1 => Self::Read1Write0,
            Self::Read1Write0 => Self::Read0Write1,
        };
    }

    /// Get the read buffer index (0 or 1).
    #[inline]
    pub fn read_index(&self) -> usize {
        match self {
            Self::Read0Write1 => 0,
            Self::Read1Write0 => 1,
        }
    }

    /// Get the write buffer index (0 or 1).
    #[inline]
    pub fn write_index(&self) -> usize {
        match self {
            Self::Read0Write1 => 1,
            Self::Read1Write0 => 0,
        }
    }

    /// Given a pair of buffers, return (read_buf, write_buf) references.
    #[inline]
    pub fn buffers<'a>(&self, pair: &'a [wgpu::Buffer; 2]) -> (&'a wgpu::Buffer, &'a wgpu::Buffer) {
        match self {
            Self::Read0Write1 => (&pair[0], &pair[1]),
            Self::Read1Write0 => (&pair[1], &pair[0]),
        }
    }
}

impl Default for PingPongState {
    fn default() -> Self {
        Self::Read0Write1
    }
}

/// A pair of buffers for ping-pong rendering.
/// All stages share the same buffer pair and use PingPongState to coordinate access.
#[derive(Debug)]
pub struct PingPongBuffers {
    /// The two buffers in the ping-pong pair.
    pub buffers: [wgpu::Buffer; 2],
}

impl PingPongBuffers {
    /// Create a new ping-pong buffer pair for the given size.
    pub fn new(device: &Device, size: u64, label: &str) -> Self {
        let make_buf = |idx: usize| {
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
            buffers: [make_buf(0), make_buf(1)],
        }
    }

    /// Get the buffer references for the current state.
    pub fn get(&self, state: PingPongState) -> (&wgpu::Buffer, &wgpu::Buffer) {
        state.buffers(&self.buffers)
    }
}
