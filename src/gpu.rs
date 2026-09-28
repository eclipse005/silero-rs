//! wgpu GPU 路径（Vulkan）：一帧 = 8 个 dispatch（3 个 pass）+ 1 次 submit + 4 字节回读。
//! context 留在 host（与 host 路径共用 x640 组装语义），LSTM h/c 留在设备。
//! 后端强制 Vulkan（本机 DX12 hang，见 D:\wgpu 手册 §10）。
//!
//! 资源结构（受 max_storage_buffers_per_shader_stage=8 与用途域规则约束）：
//! - 每个逻辑张量一个物理 buffer；15 个权重张量合成一个 W buffer（永远只读）
//! - 消费者 kernel 对输入做"垫片写"保持 rw 用途（见 gpu.wgsl 头注），避免 ro/rw 冲突
//! - pass A: stft+conv1-4（X/W ro；MAG/E1/E2/E3/E4 rw）—— 每 buffer 用途一致
//!   pass B: gates+point（E4/H/C/W ro；GATES/HN/CN rw）
//!   [copy hn→h, cn→c]
//!   pass C: final（W/H ro；PROB rw）

use crate::Weights;
use std::sync::mpsc;

const WGSL: &str = include_str!("gpu.wgsl");

// kind 索引（与 buffers 向量顺序一致）
// 0=x 1=W 2=mag 3=e1 4=e2 5=e3 6=e4 7=gates 8=h 9=c 10=prob
// W buffer 内部区域：(kind1 的条目还带 W 区域序号)
const W_REGION_ORDER: [(&str, u64); 15] = [
    ("basis", 258 * 256),
    ("c1w", 128 * 129 * 3),
    ("c1b", 128),
    ("c2w", 64 * 128 * 3),
    ("c2b", 64),
    ("c3w", 64 * 64 * 3),
    ("c3b", 64),
    ("c4w", 128 * 64 * 3),
    ("c4b", 128),
    ("wih", 512 * 128),
    ("whh", 512 * 128),
    ("bih", 512),
    ("bhh", 512),
    ("fw", 128),
    ("fb", 1),
];

// (binding, W区域序号) —— kind=1 的条目用
const PIPELINES: [(&str, &[(u32, u8, bool, u8)]); 8] = [
    // (entry_point, [(binding, kind, read_only, w_region)])
    // h/c 一律 rw：gates 读 h（垫片写保持 STORE）、point 原地写 h/c、final 读 h（垫片写）
    ("main_stft", &[(0, 0, true, 0), (1, 1, true, 0), (2, 2, false, 0)]),
    ("main_conv1", &[(3, 2, false, 0), (4, 1, true, 1), (5, 1, true, 2), (6, 3, false, 0)]),
    ("main_conv2", &[(7, 3, false, 0), (8, 1, true, 3), (9, 1, true, 4), (10, 4, false, 0)]),
    ("main_conv3", &[(11, 4, false, 0), (12, 1, true, 5), (13, 1, true, 6), (14, 5, false, 0)]),
    ("main_conv4", &[(15, 5, false, 0), (16, 1, true, 7), (17, 1, true, 8), (18, 6, false, 0)]),
    (
        "main_gates",
        &[
            (19, 6, true, 0),
            (20, 8, false, 0),
            (21, 1, true, 9),
            (22, 1, true, 10),
            (23, 1, true, 11),
            (24, 1, true, 12),
            (25, 7, false, 0),
        ],
    ),
    ("main_point", &[(26, 7, false, 0), (27, 9, false, 0), (28, 8, false, 0)]),
    ("main_final", &[(30, 8, false, 0), (31, 1, true, 13), (32, 1, true, 14), (33, 10, false, 0)]),
];

pub struct GpuVad {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Vec<wgpu::ComputePipeline>,
    bind_groups: Vec<wgpu::BindGroup>,
    x: wgpu::Buffer,
    h: wgpu::Buffer,
    c: wgpu::Buffer,
    prob: wgpu::Buffer,
    /// 双 staging ping-pong：一层深流水线用（手册 §3）
    stagings: [wgpu::Buffer; 2],
    pending: Option<(usize, mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>)>,
    parity: usize,
}

/// 跨平台后端选择：默认 PRIMARY（Vulkan/Metal/DX12 自动），可用环境变量
/// SILERO_BACKEND=vulkan|dx12|metal|gl|primary 覆盖（驱动问题时的逃生门）。
pub fn backend_from_env() -> wgpu::Backends {
    match std::env::var("SILERO_BACKEND").as_deref() {
        Ok("vulkan") => wgpu::Backends::VULKAN,
        Ok("dx12") => wgpu::Backends::DX12,
        Ok("metal") => wgpu::Backends::METAL,
        Ok("gl") => wgpu::Backends::GL,
        Ok("primary") | _ => wgpu::Backends::PRIMARY,
    }
}

impl GpuVad {
    pub fn new(w: &Weights) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: backend_from_env(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .expect("no vulkan adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("silero-vad"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                ..Default::default()
            }))
            .expect("request_device failed");

        use wgpu::util::DeviceExt;
        let w_all: Vec<f32> = [
            &w.basis, &w.c1w, &w.c1b, &w.c2w, &w.c2b, &w.c3w, &w.c3b, &w.c4w, &w.c4b, &w.wih,
            &w.whh, &w.bih, &w.bhh, &w.fw, &w.fb,
        ]
        .into_iter()
        .flatten()
        .copied()
        .collect();

        let st = wgpu::BufferUsages::STORAGE;
        let buffers: Vec<wgpu::Buffer> = vec![
            device.create_buffer(&wgpu::BufferDescriptor { label: Some("x"), size: 640 * 4, usage: st | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }),          // 0
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("W"), contents: bytemuck_f32(&w_all), usage: st }),                                      // 1
            device.create_buffer(&wgpu::BufferDescriptor { label: Some("mag"), size: 517 * 4, usage: st, mapped_at_creation: false }),                                         // 2
            device.create_buffer(&wgpu::BufferDescriptor { label: Some("e1"), size: 513 * 4, usage: st, mapped_at_creation: false }),                                          // 3
            device.create_buffer(&wgpu::BufferDescriptor { label: Some("e2"), size: 129 * 4, usage: st, mapped_at_creation: false }),                                          // 4
            device.create_buffer(&wgpu::BufferDescriptor { label: Some("e3"), size: 65 * 4, usage: st, mapped_at_creation: false }),                                           // 5
            device.create_buffer(&wgpu::BufferDescriptor { label: Some("e4"), size: 129 * 4, usage: st, mapped_at_creation: false }),                                          // 6
            device.create_buffer(&wgpu::BufferDescriptor { label: Some("gates"), size: 513 * 4, usage: st, mapped_at_creation: false }),                                       // 7
            device.create_buffer(&wgpu::BufferDescriptor { label: Some("h"), size: 129 * 4, usage: st | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }),            // 8（末位垫片）
            device.create_buffer(&wgpu::BufferDescriptor { label: Some("c"), size: 128 * 4, usage: st | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }),            // 9
            device.create_buffer(&wgpu::BufferDescriptor { label: Some("prob"), size: 4, usage: st | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false }),               // 10
        ];

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("silero.wgsl"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });

        let mut pipelines = Vec::new();
        let mut bind_groups = Vec::new();
        for (ep, binds) in &PIPELINES {
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(ep),
                entries: &binds
                    .iter()
                    .map(|(b, _, ro, _)| wgpu::BindGroupLayoutEntry {
                        binding: *b,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: *ro },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    })
                    .collect::<Vec<_>>(),
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(ep),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
            let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(ep),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(ep),
                compilation_options: Default::default(),
                cache: None,
            });
            let entries: Vec<wgpu::BindGroupEntry> = binds
                .iter()
                .map(|(b, kind, _, w_region)| {
                    let buf = &buffers[*kind as usize];
                    let (off, size) = if *kind == 1 {
                        // W buffer：取该张量的子范围
                        let mut off_elems = 0u64;
                        for (_, n) in W_REGION_ORDER.iter().take(*w_region as usize) {
                            off_elems += n;
                        }
                        let n_elems = W_REGION_ORDER[*w_region as usize].1;
                        (off_elems * 4, n_elems * 4)
                    } else {
                        (0, buf.size())
                    };
                    wgpu::BindGroupEntry {
                        binding: *b,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: buf,
                            offset: off,
                            size: Some(std::num::NonZeroU64::new(size).unwrap()),
                        }),
                    }
                })
                .collect();
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(ep),
                layout: &bgl,
                entries: &entries,
            });
            pipelines.push(pipe);
            bind_groups.push(bg);
        }

        let staging0 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging0"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging1 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging1"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            x: buffers[0].clone(),
            h: buffers[8].clone(),
            c: buffers[9].clone(),
            prob: buffers[10].clone(),
            device,
            queue,
            pipelines,
            bind_groups,
            stagings: [staging0, staging1],
            pending: None,
            parity: 0,
        }
    }

    pub fn reset(&mut self) {
        self.queue.write_buffer(&self.h, 0, &[0u8; 512]);
        self.queue.write_buffer(&self.c, 0, &[0u8; 512]);
    }

    /// 编码并提交一帧（单 pass：8 个 dispatch 顺序执行，dispatch 间有隐式屏障），
    /// 把 prob 拷到 staging[slot]，武装该 staging 的 map。
    fn submit_frame(&mut self, x640: &[f32], slot: usize) -> mpsc::Receiver<Result<(), wgpu::BufferAsyncError>> {
        assert_eq!(x640.len(), 640);
        let mut bytes = Vec::with_capacity(640 * 4);
        for v in x640 {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.queue.write_buffer(&self.x, 0, &bytes);

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });

        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("silero"),
                timestamp_writes: None,
            });
            for (p, n) in [
                (0usize, 129 * 4u32), // stft
                (1, 128 * 4),         // conv1
                (2, 64 * 2),          // conv2
                (3, 64),              // conv3
                (4, 128),             // conv4
                (5, 512),             // gates
                (6, 128),             // point（workgroup 128）
                (7, 1),               // final（workgroup 1）
            ] {
                pass.set_pipeline(&self.pipelines[p]);
                pass.set_bind_group(0, &self.bind_groups[p], &[]);
                pass.dispatch_workgroups(n.div_ceil(64).max(1), 1, 1);
            }
        }
        enc.copy_buffer_to_buffer(&self.prob, 0, &self.stagings[slot], 0, 4);
        self.queue.submit(Some(enc.finish()));

        // 回读 4 字节（map 在提交之后武装，手册 §9）
        let (tx, rx) = mpsc::channel();
        self.stagings[slot].slice(..4).map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        rx
    }

    fn read_staging(
        &mut self,
        slot: usize,
        rx: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    ) -> f32 {
        // 非阻塞泵送：只等本 slot 的 map 回调（其提交早已完成），
        // 不等待最新提交 —— 这是流水线能重叠的关键（poll(Wait) 会串行化一切）。
        loop {
            match rx.try_recv() {
                Ok(r) => {
                    r.expect("map failed");
                    break;
                }
                Err(_) => {
                    let _ = self.device.poll(wgpu::PollType::Poll);
                    std::hint::spin_loop();
                }
            }
        }
        let view = self.stagings[slot].slice(..4).get_mapped_range().expect("map view");
        let p = f32::from_le_bytes([view[0], view[1], view[2], view[3]]);
        drop(view);
        self.stagings[slot].unmap();
        p
    }

    /// 同步一帧（交互语义：返回本帧概率）。
    pub fn frame(&mut self, x640: &[f32]) -> f32 {
        let rx = self.submit_frame(x640, 0);
        self.read_staging(0, rx)
    }

    /// 流水线一帧（吞吐语义）：返回**上一帧**的概率，首帧返回 None；
    /// 流结束时用 flush 取最后一帧。概率流与同步路径完全一致。
    pub fn frame_pipelined(&mut self, x640: &[f32]) -> Option<f32> {
        let slot = self.parity % 2;
        let rx = self.submit_frame(x640, slot);
        self.parity += 1;
        let out = self
            .pending
            .take()
            .map(|(prev_slot, prev_rx)| self.read_staging(prev_slot, prev_rx));
        self.pending = Some((slot, rx));
        out
    }

    /// 流水线收尾：返回最后一帧的概率（若在途）。
    pub fn flush(&mut self) -> Option<f32> {
        self.pending
            .take()
            .map(|(slot, rx)| self.read_staging(slot, rx))
    }
}

fn bytemuck_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
