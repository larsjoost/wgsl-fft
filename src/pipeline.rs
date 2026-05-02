//! Streaming pipeline builder and compute stage infrastructure.
//!
//! This module implements a synchronous "Bucket Brigade" GPU pipeline.
//! Every stage reads from a 'Read' buffer and writes to a 'Write' buffer.
//! All stages toggle their Read/Write sides simultaneously at the end of a 'tick'.

use std::any::{Any, TypeId};
use std::collections::HashMap;

use crate::{FftPipelines, PingPongBuffers, PingPongState};
use wgpu::{Buffer, CommandEncoder, Device, Queue};

/// A collection of typed parameters for pipeline stages.
#[derive(Default)]
pub struct PipelineParameters {
    params: HashMap<TypeId, Box<dyn Any>>,
}

impl PipelineParameters {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn set<T: Send + Sync + 'static>(&mut self, value: T) {
        self.params.insert(TypeId::of::<T>(), Box::new(value));
    }
    pub fn get<T: 'static>(&self) -> Option<&T> {
        self.params
            .get(&TypeId::of::<T>())
            .and_then(|b| b.downcast_ref::<T>())
    }
    pub fn get_f32(&self) -> Option<f32> {
        self.get::<f32>().copied()
    }
    pub fn get_u32(&self) -> Option<u32> {
        self.get::<u32>().copied()
    }
}

/// Context passed to each stage during encoding.
pub struct StageContext<'a> {
    pub encoder: &'a mut CommandEncoder,
    pub device: &'a Device,
    pub queue: &'a Queue,
    pub fft: &'a FftPipelines,
    pub state: PingPongState,
}

/// A compute stage that processes a batch of buffers.
pub trait ComputeStage: Send + Sync {
    fn name(&self) -> &str;
    fn encode(
        &self,
        ctx: &mut StageContext,
        inputs: &[&Buffer],
        outputs: &[&Buffer],
        params: &PipelineParameters,
    );
}

/// A streaming pipeline that advances all stages synchronously.
pub struct Pipeline {
    device: Device,
    queue: Queue,
    fft: FftPipelines,
    stages: Vec<Box<dyn ComputeStage>>,
    /// Each element is a batch of PingPongBuffers for one "register" between stages.
    registers: Vec<Vec<PingPongBuffers>>,
    state: PingPongState,
    /// Parameters flowing through the pipeline alongside the data.
    stage_params: Vec<PipelineParameters>,
}

impl Pipeline {
    pub fn new(
        device: Device,
        queue: Queue,
        fft: FftPipelines,
        stages: Vec<Box<dyn ComputeStage>>,
        batch_size: usize,
        buffer_size: u64,
    ) -> Self {
        let mut registers = Vec::with_capacity(stages.len() + 1);
        for i in 0..=stages.len() {
            let mut batch = Vec::with_capacity(batch_size);
            for j in 0..batch_size {
                batch.push(PingPongBuffers::new(
                    &device,
                    buffer_size,
                    &format!("reg{i}_b{j}"),
                ));
            }
            registers.push(batch);
        }

        let mut stage_params = Vec::with_capacity(stages.len());
        for _ in 0..stages.len() {
            stage_params.push(PipelineParameters::new());
        }

        Self {
            device,
            queue,
            fft,
            stages,
            registers,
            state: PingPongState::Read0Write1,
            stage_params,
        }
    }

    pub fn tick(&mut self, next_params: PipelineParameters) {
        let mut encoder = self.device.create_command_encoder(&Default::default());

        for i in (1..self.stages.len()).rev() {
            self.stage_params[i] =
                std::mem::replace(&mut self.stage_params[i - 1], PipelineParameters::new());
        }
        self.stage_params[0] = next_params;

        for (i, stage) in self.stages.iter().enumerate() {
            let inputs: Vec<&Buffer> = self.registers[i]
                .iter()
                .map(|pp| pp.get(self.state).0)
                .collect();
            let outputs: Vec<&Buffer> = self.registers[i + 1]
                .iter()
                .map(|pp| pp.get(self.state).1)
                .collect();

            let mut ctx = StageContext {
                encoder: &mut encoder,
                device: &self.device,
                queue: &self.queue,
                fft: &self.fft,
                state: self.state,
            };
            stage.encode(&mut ctx, &inputs, &outputs, &self.stage_params[i]);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.state.toggle();
    }

    pub fn get_input_buffer(&self, batch_idx: usize) -> &Buffer {
        self.registers[0][batch_idx].get(self.state).0
    }

    pub fn get_output_buffer(&self, batch_idx: usize) -> &Buffer {
        self.registers[self.stages.len()][batch_idx]
            .get(self.state)
            .0
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn queue(&self) -> &Queue {
        &self.queue
    }
}

pub struct FftStage {
    direction: crate::FftDirection,
}

impl FftStage {
    pub fn forward() -> Self {
        Self {
            direction: crate::FftDirection::Forward,
        }
    }
    pub fn inverse() -> Self {
        Self {
            direction: crate::FftDirection::Inverse,
        }
    }
}

impl ComputeStage for FftStage {
    fn name(&self) -> &str {
        "fft"
    }
    fn encode(
        &self,
        ctx: &mut StageContext,
        inputs: &[&Buffer],
        outputs: &[&Buffer],
        _params: &PipelineParameters,
    ) {
        for (input, output) in inputs.iter().zip(outputs.iter()) {
            let n = (input.size() / 8) as usize;
            ctx.fft
                .encode_fft(ctx.encoder, n, self.direction, input, output);
        }
    }
}

pub struct NormalizeStage;

impl ComputeStage for NormalizeStage {
    fn name(&self) -> &str {
        "normalize"
    }
    fn encode(
        &self,
        ctx: &mut StageContext,
        inputs: &[&Buffer],
        outputs: &[&Buffer],
        _params: &PipelineParameters,
    ) {
        for (input, output) in inputs.iter().zip(outputs.iter()) {
            let n = (input.size() / 8) as usize;
            ctx.encoder
                .copy_buffer_to_buffer(input, 0, output, 0, input.size());
            ctx.fft.encode_normalize(ctx.encoder, n, output);
        }
    }
}

pub struct MultiplyStage {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl MultiplyStage {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("multiply_shader"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
                @group(0) @binding(0) var<storage, read> a: array<vec2<f32>>;
                @group(0) @binding(1) var<storage, read> b: array<vec2<f32>>;
                @group(0) @binding(2) var<storage, read_write> out: array<vec2<f32>>;
                @compute @workgroup_size(256)
                fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                    let i = id.x;
                    if i >= arrayLength(&a) { return; }
                    let ca = a[i]; let cb = b[i];
                    out[i] = vec2<f32>(ca.x * cb.x - ca.y * cb.y, ca.x * cb.y + ca.y * cb.x);
                }
            "#
                .into(),
            ),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("multiply_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline, bgl }
    }
}

impl ComputeStage for MultiplyStage {
    fn name(&self) -> &str {
        "multiply"
    }
    fn encode(
        &self,
        ctx: &mut StageContext,
        inputs: &[&Buffer],
        outputs: &[&Buffer],
        _params: &PipelineParameters,
    ) {
        if inputs.len() < 2 || outputs.is_empty() {
            return;
        }
        let a = inputs[0];
        let b = inputs[1];
        let out = outputs[0];
        let n = (a.size() / 8) as u32;
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
            ],
        });
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
        drop(pass);
        if inputs.len() >= 2 && outputs.len() >= 2 {
            ctx.encoder
                .copy_buffer_to_buffer(inputs[1], 0, outputs[1], 0, inputs[1].size());
        }
    }
}

pub struct NoiseStage {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl NoiseStage {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("noise_shader"),
            source: wgpu::ShaderSource::Wgsl(r#"
struct NoiseParams { sigma: f32, seed: u32, n: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: NoiseParams;
fn xorshift32(state: u32) -> u32 { var s = state; s ^= s << 13u; s ^= s >> 17u; s ^= s << 5u; return s; }
fn u32_to_f32_01(x: u32) -> f32 { return f32(x >> 9u) * (1.0 / 8388608.0); }
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; if i >= params.n { return; }
    let base = params.seed ^ (i * 2654435761u);
    let s1 = xorshift32(base); let s2 = xorshift32(s1);
    let u1 = max(u32_to_f32_01(s1), 1e-7); let u2 = u32_to_f32_01(s2);
    let pi2: f32 = 6.283185307179586;
    let r = sqrt(-2.0 * log(u1)); let theta = pi2 * u2;
    let z0 = r * cos(theta); let z1 = r * sin(theta);
    data[i] = data[i] + vec2<f32>(z0 * params.sigma, z1 * params.sigma);
}
"#.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("noise_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline, bgl }
    }
}

impl ComputeStage for NoiseStage {
    fn name(&self) -> &str {
        "noise"
    }
    fn encode(
        &self,
        ctx: &mut StageContext,
        inputs: &[&Buffer],
        outputs: &[&Buffer],
        params: &PipelineParameters,
    ) {
        let sigma = params.get_f32().unwrap_or(0.0);
        let base_seed = params.get_u32().unwrap_or(0);
        for (idx, (input, output)) in inputs.iter().zip(outputs.iter()).enumerate() {
            let n = (input.size() / 8) as u32;
            ctx.encoder
                .copy_buffer_to_buffer(input, 0, output, 0, input.size());
            if sigma == 0.0 {
                continue;
            }
            #[repr(C)]
            #[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
            struct NoiseShaderParams {
                sigma: f32,
                seed: u32,
                n: u32,
                _pad: u32,
            }
            let seed = base_seed.wrapping_add(idx as u32);
            let p_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            ctx.queue.write_buffer(
                &p_buf,
                0,
                bytemuck::bytes_of(&NoiseShaderParams {
                    sigma,
                    seed,
                    n,
                    _pad: 0,
                }),
            );
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: p_buf.as_entire_binding(),
                    },
                ],
            });
            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
        }
    }
}
