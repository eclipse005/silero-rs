//! 批量 GPU 路径（Vulkan）：BATCH 帧一次提交 —— 本项目 wgpu 的主战场。
//!
//! 预算依据（见项目记忆 + 2026-09 实测剖析）：
//! - conv 部分逐帧独立 → 5 个批量 dispatch；gates_in 批量 GEMV 1 个。
//!   **Intel 集显上 conv 是隐藏的大头**（优化前每帧 ~37µs 且行间几乎不并行）：
//!   conv 权重 host 侧重排 [co][ic][k]→[co][k][ic]（线程沿 ic 连续读）+ shader 内
//!   4 深软件流水（先取 ic+4..ic+7 的权重/激活再做本拍 FMA）。均为纯重排 → 逐位一致。
//! - LSTM 递归步 T/CHUNK 个顺序单 workgroup dispatch（gin/whh·h 融合 + pointwise
//!   原地 h/c，帧号经 entry point 烘进 shader）；每批 1 次 submit、1 次 map(T*256B)。
//! - prob 归约拆出为批量 kernel（每帧 1 个 workgroup(128)，线程 0 顺序累加 ——
//!   与旧 in-loop 归约逐位一致）。
//! 每帧 ≈ 6 + T/CHUNK + 1 个 dispatch，主机成本摊薄 ~T 倍。

use crate::Weights;
use std::collections::HashMap;
use std::sync::mpsc;

const WGSL_TEMPLATE: &str = include_str!("gpu_batch.wgsl");
/// 厂商无关超越函数层，见 `wgsl_math.wgsl` 头注。
const WGSL_MATH: &str = include_str!("wgsl_math.wgsl");
pub const MAX_T: usize = 512;
/// LSTM 递归每 dispatch 串行处理的帧数：发射次数 512 → MAX_T/LSTM_CHUNK。
/// 每帧计算量仅 ~65K MAC，发射开销远大于计算；kernel 内 for 循环递归。
const LSTM_CHUNK: usize = 32;

// W 区域（f32 元素偏移/长度），与 gpu_batch.wgsl 的 OFF_* 注入一致
const W_SIZES: [u64; 15] = [
    258 * 256,
    128 * 129 * 3,
    128,
    64 * 128 * 3,
    64,
    64 * 64 * 3,
    64,
    128 * 64 * 3,
    128,
    512 * 128,
    512 * 128,
    512,
    512,
    128,
    1,
];

fn w_offsets() -> Vec<u64> {
    let mut offs = Vec::with_capacity(W_SIZES.len());
    let mut acc = 0u64;
    for &n in &W_SIZES {
        offs.push(acc);
        acc += n;
    }
    offs
}

const OFF_NAMES: [&str; 14] = [
    "OFF_C1W", "OFF_C1B", "OFF_C2W", "OFF_C2B", "OFF_C3W", "OFF_C3B", "OFF_C4W", "OFF_C4B",
    "OFF_WIH", "OFF_WHH", "OFF_BIH", "OFF_BHH", "OFF_FW", "OFF_FB",
];

// kind: 0=x 1=W 2=mag 3=e1 4=e2 5=e3 6=e4 7=gin 8=h 9=c 10=prob 11=h_all
// 批量 kernel 的 (binding, kind, w_region)；lstm chunk 单独建 bind group
const BATCH_PIPELINES: [(&str, &[(u32, u8, bool, u8)]); 7] = [
    // 批量 kernel 一律经 b_w（binding 1，整个权重 concat）+ 注入偏移读权重；
    // binding 编号与 gpu_batch.wgsl 的全局声明一致
    ("main_stft", &[(0, 0, true, 0), (1, 1, true, 0), (2, 2, false, 0)]),
    ("main_conv1", &[(1, 1, true, 0), (2, 2, false, 0), (3, 3, false, 0)]),
    ("main_conv2", &[(1, 1, true, 0), (3, 3, false, 0), (4, 4, false, 0)]),
    ("main_conv3", &[(1, 1, true, 0), (4, 4, false, 0), (5, 5, false, 0)]),
    ("main_conv4", &[(1, 1, true, 0), (5, 5, false, 0), (6, 6, false, 0)]),
    ("main_gates_in", &[(1, 1, true, 0), (6, 6, false, 0), (7, 7, false, 0)]),
    // prob：每帧 1 个 workgroup(128)，读 h_all 快照 + 权重，写 prob。
    // h_all 在 shader 模块级声明为 read_write（lstm 要写），故此处也绑 rw ——
    // wgpu 仍会对 lstm(写)→prob(读) 的 storage 冒险插入 dispatch 间屏障
    ("main_prob", &[(1, 1, true, 0), (12, 11, false, 0), (11, 10, false, 0)]),
];

// lstm chunk 的 (binding, kind)：1=W整块（含转置 whh，绝对偏移索引） 7=gin整块
// 8=h 9=c 11=h_all整块
const LSTM_BINDS: [(u32, u8); 5] = [(8, 7), (1, 1), (9, 8), (10, 9), (12, 11)];

pub struct GpuBatch {
    device: wgpu::Device,
    queue: wgpu::Queue,
    buffers: Vec<wgpu::Buffer>,
    // x 双缓冲：第 N+1 批的 x 上传与第 N 批的 GPU 执行并行
    x_alt: wgpu::Buffer,
    staging: [wgpu::Buffer; 2],
    // 每个用过的 T 一套 pipeline（shader 含 BATCH 常量）
    pipelines: HashMap<usize, Vec<wgpu::ComputePipeline>>,
    // 批量 kernel 的 bind group（T 无关）：[slot][pipeline]，slot=0 用 buffers[0]，slot=1 用 x_alt
    batch_bgs: Vec<Vec<wgpu::BindGroup>>,
    // lstm chunk 的 bind group（gin/h_all/W 全整块，1 个）
    lstm_bg: wgpu::BindGroup,
    step_bgl: wgpu::BindGroupLayout,
    pl_layout: wgpu::PipelineLayout,
    // 一层深流水线：已提交未读回的批
    inflight: Option<Inflight>,
    flip: usize,
    // 调试计时（TIMESTAMP_QUERY）
    ts_query: wgpu::QuerySet,
    ts_resolve: wgpu::Buffer,
    ts_map: wgpu::Buffer,
}

struct Inflight {
    slot: usize,
    bytes: u64,
}

impl GpuBatch {
    pub fn new(w: &Weights) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: crate::gpu::backend_from_env(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = crate::gpu::pick_adapter(&instance);
        crate::gpu::maybe_log_adapter(&adapter);
        // 本项目两张实测适配器（Intel iGPU / GTX 1070）均支持 1024（HANDOFF §4）；
        // conv/lstm 的 workgroup 实际只用 64/256，这里仅留余量
        let mut limits = wgpu::Limits::default();
        limits.max_compute_invocations_per_workgroup = 1024;
        limits.max_compute_workgroup_size_x = 1024;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("silero-batch"),
                required_features: wgpu::Features::TIMESTAMP_QUERY,
                required_limits: limits,
                memory_hints: Default::default(),
                ..Default::default()
            }))
            .expect("request_device failed");

        use wgpu::util::DeviceExt;
        // conv 权重重排 [co][ic][k] → [co][k][ic]：conv 线程沿 ic 连续读权重。
        // 旧布局线程读步长 3（12B 里用 4B），缓存行利用率 33%，实测 conv 是集显上
        // 最大的瓶颈（每帧 ~37µs 且行间几乎不并行，L2 带宽浪费 2/3）。
        // 纯缓冲区重排，(co,ic,k) 取的值不变、累加链序不变 → 数值逐位一致。
        // （与下方 whh 预转置同理。）
        fn repack_conv_w(w: &[f32], n_co: usize, n_ic: usize) -> Vec<f32> {
            let mut out = vec![0.0f32; w.len()];
            for co in 0..n_co {
                for ic in 0..n_ic {
                    for k in 0..3 {
                        out[co * n_ic * 3 + k * n_ic + ic] = w[co * n_ic * 3 + ic * 3 + k];
                    }
                }
            }
            out
        }
        let c1w = repack_conv_w(&w.c1w, 128, 129);
        let c2w = repack_conv_w(&w.c2w, 64, 128);
        let c3w = repack_conv_w(&w.c3w, 64, 64);
        let c4w = repack_conv_w(&w.c4w, 128, 64);
        // whh 预转置 (r,k)→(k,r)：lstm chunk 是单 workgroup（低占用、延迟暴露），
        // 线程 r 读 whh 行 r 时 warp 内 32 线程命中 32 条不同缓存行（扇区效率 12.5%），
        // 转置后 warp 读连续地址（100%）；每行 k 顺序累加不变 → 数值逐位一致
        let whh_t: Vec<f32> = (0..w.whh.len())
            .map(|i| {
                let k = i / 512;
                let r = i % 512;
                w.whh[r * 128 + k]
            })
            .collect();
        let w_all: Vec<f32> = [
            &w.basis, &c1w, &w.c1b, &c2w, &w.c2b, &c3w, &w.c3b, &c4w, &w.c4b, &w.wih,
            &whh_t, &w.bih, &w.bhh, &w.fw, &w.fb,
        ]
        .into_iter()
        .flatten()
        .copied()
        .collect();

        let t = MAX_T as u64;
        let st = wgpu::BufferUsages::STORAGE;
        let st_dbg = st | wgpu::BufferUsages::COPY_SRC; // 中间量可读回（调试用）
        let buffers: Vec<wgpu::Buffer> = vec![
            device.create_buffer(&wgpu::BufferDescriptor { label: Some("x"), size: t * 640 * 4, usage: st | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false }),
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("W"), contents: bytemuck_f32(&w_all), usage: st }),
            device.create_buffer(&wgpu::BufferDescriptor { label: Some("mag"), size: t * 517 * 4, usage: st_dbg, mapped_at_creation: false }),
            device.create_buffer(&wgpu::BufferDescriptor { label: Some("e1"), size: t * 513 * 4, usage: st_dbg, mapped_at_creation: false }),
            device.create_buffer(&wgpu::BufferDescriptor { label: Some("e2"), size: t * 129 * 4, usage: st_dbg, mapped_at_creation: false }),
            device.create_buffer(&wgpu::BufferDescriptor { label: Some("e3"), size: t * 65 * 4, usage: st_dbg, mapped_at_creation: false }),
            device.create_buffer(&wgpu::BufferDescriptor { label: Some("e4"), size: t * 129 * 4, usage: st_dbg, mapped_at_creation: false }),
            device.create_buffer(&wgpu::BufferDescriptor { label: Some("gin"), size: t * 576 * 4, usage: st_dbg, mapped_at_creation: false }),
            device.create_buffer(&wgpu::BufferDescriptor { label: Some("h"), size: 128 * 4, usage: st | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }),
            device.create_buffer(&wgpu::BufferDescriptor { label: Some("c"), size: 128 * 4, usage: st | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }),
            device.create_buffer(&wgpu::BufferDescriptor { label: Some("prob"), size: t * 64 * 4, usage: st | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false }),
            device.create_buffer(&wgpu::BufferDescriptor { label: Some("h_all"), size: t * 128 * 4, usage: st, mapped_at_creation: false }),
        ];
        let staging: [wgpu::Buffer; 2] = [0, 1].map(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging"),
                size: t * 256,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let x_alt = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("x_alt"),
            size: t * 640 * 4,
            usage: st | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 批量 kernel bind groups（wgpu 的 BindGroup 内部持有 BGL 引用，drop 安全）
        // 两套：slot 0 绑 buffers[0]，slot 1 绑 x_alt
        let mut batch_bgs: Vec<Vec<wgpu::BindGroup>> = Vec::new();
        for x_buf in [&buffers[0], &x_alt] {
            let mut per_slot = Vec::new();
            for (_, binds) in &BATCH_PIPELINES {
                let bgl = Self::make_bgl(&device, binds);
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &bgl,
                    entries: &Self::make_entries(binds, &buffers, x_buf),
                });
                per_slot.push(bg);
            }
            batch_bgs.push(per_slot);
        }

        // lstm BGL + 单个 bind group（gin/h_all/W 全整块绑定，chunk kernel 按帧号 d 索引）
        let step_binds_ro = LSTM_BINDS
            .iter()
            .map(|(b, kind)| {
                let ro = matches!(kind, 1);
                wgpu::BindGroupLayoutEntry {
                    binding: *b,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: ro },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            })
            .collect::<Vec<_>>();
        let step_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lstm_chunk"),
            entries: &step_binds_ro,
        });
        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lstm_chunk_pl"),
            bind_group_layouts: &[Some(&step_bgl)],
            immediate_size: 0,
        });
        let lstm_bg = {
            let entries: Vec<wgpu::BindGroupEntry> = LSTM_BINDS
                .iter()
                .map(|(b, kind)| {
                    // 全整块绑定（W 绝对偏移 / gin、h_all 按帧号 d 索引）；
                    // 子范围绑定会越界读 0（症状：prob 恒 0.5）
                    let full_w: u64 = W_SIZES.iter().sum();
                    let (buf, off, size) = match kind {
                        7 => (&buffers[7], 0, 576 * MAX_T as u64 * 4),
                        1 => (&buffers[1], 0, full_w * 4),
                        8 => (&buffers[8], 0, 128 * 4),
                        9 => (&buffers[9], 0, 128 * 4),
                        11 => (&buffers[11], 0, 128 * MAX_T as u64 * 4),
                        _ => unreachable!(),
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
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lstm_bg"),
                layout: &step_bgl,
                entries: &entries,
            })
        };

        let ts_query = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("ts"),
            ty: wgpu::QueryType::Timestamp,
            count: 16,
        });
        let ts_resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ts_resolve"),
            size: 16 * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let ts_map = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ts_map"),
            size: 16 * 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            buffers,
            x_alt,
            staging,
            pipelines: HashMap::new(),
            batch_bgs,
            lstm_bg,
            step_bgl,
            pl_layout,
            inflight: None,
            flip: 0,
            ts_query,
            ts_resolve,
            ts_map,
        }
    }

    fn make_bgl(device: &wgpu::Device, binds: &[(u32, u8, bool, u8)]) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
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
        })
    }

    fn make_entries<'a>(
        binds: &[(u32, u8, bool, u8)],
        buffers: &'a [wgpu::Buffer],
        x_buf: &'a wgpu::Buffer,
    ) -> Vec<wgpu::BindGroupEntry<'a>> {
        binds
            .iter()
            .map(|(b, kind, _, _)| {
                let buf = if *kind == 0 { x_buf } else { &buffers[*kind as usize] };
                // 批量 kernel（含 lstm/prob）都用绝对 OFF_* 偏移索引权重，
                // W 必须绑整块 concat——子范围绑定会越界读 0（症状：prob 恒 0.5）
                let size = if *kind == 1 {
                    W_SIZES.iter().sum::<u64>() * 4
                } else {
                    buf.size()
                };
                wgpu::BindGroupEntry {
                    binding: *b,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: buf,
                        offset: 0,
                        size: Some(std::num::NonZeroU64::new(size).unwrap()),
                    }),
                }
            })
            .collect()
    }

    /// 处理一批 t 帧（x_all: t*640 f32）。
    /// 一层深流水线：**先**注册上一批 staging 的 map（fence 在本批提交之前，否则
    /// wgpu 的 map 会等到本批完成 = 假流水线），**再**提交本批，然后收割上一批——
    /// 收割期间本批在 GPU 上执行。首批返回 None；末批由 [`GpuBatch::flush`] 收尾。
    /// h/c 状态常驻 GPU，批间无需主机同步。
    pub fn frame_batch(&mut self, x_all: &[f32], t: usize) -> Option<Vec<f32>> {
        assert!(t >= 1 && t <= MAX_T);
        assert_eq!(x_all.len(), t * 640);
        if !self.pipelines.contains_key(&t) {
            self.build_pipelines(t);
        }
        // 1) 注册上一批的读回（map 的 fence 截止到此刻已提交的工作）
        let pending = self.inflight.take().map(|p| {
            let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let flag = done.clone();
            self.staging[p.slot]
                .slice(..p.bytes)
                .map_async(wgpu::MapMode::Read, move |r| {
                    r.expect("map failed");
                    flag.store(true, std::sync::atomic::Ordering::SeqCst);
                });
            (p, done)
        });
        let pipes = &self.pipelines[&t];
        let slot = self.flip;

        // 2) 上传 + 编码 + 提交本批
        let mut bytes = Vec::with_capacity(t * 640 * 4);
        for v in x_all {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        if slot == 0 {
            self.queue.write_buffer(&self.buffers[0], 0, &bytes);
        } else {
            self.queue.write_buffer(&self.x_alt, 0, &bytes);
        }

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("batch") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("batch"),
                timestamp_writes: None,
            });
            let tt = t as u32;
            let disps: [(usize, u32); 6] = [
                (0, tt * 516),
                (1, tt * 512),
                (2, tt * 128),
                (3, tt * 64),
                (4, tt * 128),
                (5, tt * 512),
            ];
            for (p, n) in disps {
                pass.set_pipeline(&pipes[p]);
                pass.set_bind_group(0, &self.batch_bgs[slot][p], &[]);
                // 2D dispatch：x 撞 65535 上限时折到 y 维（shader 用 nw.x 折算线性 idx）
                let nx = n.div_ceil(64).min(65535);
                let ny = n.div_ceil(64).div_ceil(nx);
                pass.dispatch_workgroups(nx, ny, 1);
            }
            // LSTM 递归：t/CHUNK 个顺序 dispatch，每个 chunk kernel 内串行 LSTM_CHUNK 帧
            // （隐式屏障保证 chunk 间串行；帧界由 shader 内 BATCH 守卫）
            let chunks = t.div_ceil(LSTM_CHUNK);
            for c in 0..chunks {
                pass.set_pipeline(&pipes[7 + c]);
                pass.set_bind_group(0, &self.lstm_bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            // prob：每帧 1 个 workgroup(128)，批量归约 + sigmoid（BATCH_PIPELINES[6]）
            pass.set_pipeline(&pipes[6]);
            pass.set_bind_group(0, &self.batch_bgs[slot][6], &[]);
            pass.dispatch_workgroups(tt, 1, 1);
        }
        // prob 行距 64 f32 = 256B/帧：整块拷回 staging（t*256B），host 每行取首 f32
        let copy_bytes = (t as u64) * 256;
        enc.copy_buffer_to_buffer(&self.buffers[10], 0, &self.staging[slot], 0, copy_bytes);
        self.queue.submit(Some(enc.finish()));
        self.inflight = Some(Inflight { slot, bytes: copy_bytes });
        self.flip ^= 1;

        // 3) 收割上一批（本批已在 GPU 上执行）
        pending.map(|(p, done)| self.finish_read(p.slot, p.bytes, done))
    }

    /// 读回最后一批的概率（流水线收尾）。
    pub fn flush(&mut self) -> Option<Vec<f32>> {
        let p = self.inflight.take()?;
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = done.clone();
        self.staging[p.slot]
            .slice(..p.bytes)
            .map_async(wgpu::MapMode::Read, move |r| {
                r.expect("map failed");
                flag.store(true, std::sync::atomic::Ordering::SeqCst);
            });
        Some(self.finish_read(p.slot, p.bytes, done))
    }

    /// 一步式：整段 16k 音频 → 逐帧概率。内部管理 context / reflect pad / 分批 /
    /// 流水线收尾（等价 host 路径的 [`crate::SileroVad::frame`] 循环）。
    pub fn stream(&mut self, audio: &[f32]) -> Vec<f32> {
        self.reset();
        const FRAME: usize = 512;
        const CTX: usize = 64;
        const PAD: usize = 64;
        let row = CTX + FRAME + PAD;
        let n_frames = audio.len().div_ceil(FRAME);
        let mut ctx = vec![0.0f32; CTX];
        let mut x_all = vec![0.0f32; MAX_T * row];
        let mut frame_buf = vec![0.0f32; FRAME];
        let mut probs = Vec::with_capacity(n_frames);
        for batch_start in (0..n_frames).step_by(MAX_T) {
            let t = MAX_T.min(n_frames - batch_start);
            for f in 0..t {
                let fi = batch_start + f;
                let s = fi * FRAME;
                let e = (s + FRAME).min(audio.len());
                frame_buf[..e - s].copy_from_slice(&audio[s..e]);
                frame_buf[e - s..].fill(0.0);
                let base = f * row;
                x_all[base..base + CTX].copy_from_slice(&ctx);
                x_all[base + CTX..base + CTX + FRAME].copy_from_slice(&frame_buf);
                for i in 0..PAD {
                    x_all[base + CTX + FRAME + i] = x_all[base + CTX + FRAME - 2 - i];
                }
                ctx.copy_from_slice(&frame_buf[FRAME - CTX..]);
            }
            probs.extend(self.frame_batch(&x_all[..t * row], t).unwrap_or_default());
        }
        if let Some(p) = self.flush() {
            probs.extend(p);
        }
        probs
    }

    fn finish_read(
        &self,
        slot: usize,
        copy_bytes: u64,
        done: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> Vec<f32> {
        while !done.load(std::sync::atomic::Ordering::SeqCst) {
            let _ = self.device.poll(wgpu::PollType::Poll);
            std::hint::spin_loop();
        }
        let view = self
            .staging[slot]
            .slice(..copy_bytes)
            .get_mapped_range()
            .expect("map view");
        let probs: Vec<f32> = view
            .chunks_exact(256)
            .map(|r| {
                let c = &r[..4];
                f32::from_le_bytes([c[0], c[1], c[2], c[3]])
            })
            .collect();
        drop(view);
        self.staging[slot].unmap();
        probs
    }

    /// 调试：GPU 时间戳逐 kernel 计时（跑一个满 BATCH=512 批，x 全填同一行）。
    /// 返回 (阶段名, 毫秒)，时间戳周期按 ns 近似（相对占比有效）。
    pub fn debug_timing(&mut self, x_row: &[f32]) -> Vec<(String, f64)> {
        let t = MAX_T;
        if !self.pipelines.contains_key(&t) {
            self.build_pipelines(t);
        }
        let pipes = &self.pipelines[&t];
        let mut bytes = vec![0u8; t * 640 * 4];
        for r in 0..t {
            for (i, v) in x_row.iter().enumerate() {
                let o = r * 640 * 4 + i * 4;
                bytes[o..o + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        self.queue.write_buffer(&self.buffers[0], 0, &bytes);

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ts") });
        let mut ts_i = 0u32;
        let mut marks: Vec<(&str, u32, u32)> = Vec::new();
        let defs = [
            ("stft", 0usize, 516u32),
            ("conv1", 1, 512),
            ("conv2", 2, 128),
            ("conv3", 3, 64),
            ("conv4", 4, 128),
            ("gates_in", 5, 512),
        ];
        for (name, pi, n) in defs {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(name),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &self.ts_query,
                    beginning_of_pass_write_index: Some(ts_i),
                    end_of_pass_write_index: Some(ts_i + 1),
                }),
            });
            pass.set_pipeline(&pipes[pi]);
            pass.set_bind_group(0, &self.batch_bgs[0][pi], &[]);
            let nx = n.div_ceil(64).min(65535);
            let ny = n.div_ceil(64).div_ceil(nx);
            pass.dispatch_workgroups(nx, ny, 1);
            drop(pass);
            marks.push((name, ts_i, ts_i + 1));
            ts_i += 2;
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lstm"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &self.ts_query,
                    beginning_of_pass_write_index: Some(ts_i),
                    end_of_pass_write_index: Some(ts_i + 1),
                }),
            });
            for c in 0..MAX_T / LSTM_CHUNK {
                pass.set_pipeline(&pipes[7 + c]);
                pass.set_bind_group(0, &self.lstm_bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            drop(pass);
            marks.push(("lstm", ts_i, ts_i + 1));
            ts_i += 2;
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prob"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &self.ts_query,
                    beginning_of_pass_write_index: Some(ts_i),
                    end_of_pass_write_index: Some(ts_i + 1),
                }),
            });
            pass.set_pipeline(&pipes[6]);
            pass.set_bind_group(0, &self.batch_bgs[0][6], &[]);
            pass.dispatch_workgroups(t as u32, 1, 1);
            drop(pass);
            marks.push(("prob", ts_i, ts_i + 1));
            ts_i += 2;
        }
        enc.resolve_query_set(&self.ts_query, 0..ts_i, &self.ts_resolve, 0);
        enc.copy_buffer_to_buffer(&self.ts_resolve, 0, &self.ts_map, 0, ts_i as u64 * 8);
        self.queue.submit(Some(enc.finish()));

        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = done.clone();
        self.ts_map
            .slice(..ts_i as u64 * 8)
            .map_async(wgpu::MapMode::Read, move |r| {
                r.expect("map failed");
                flag.store(true, std::sync::atomic::Ordering::SeqCst);
            });
        while !done.load(std::sync::atomic::Ordering::SeqCst) {
            let _ = self.device.poll(wgpu::PollType::Poll);
            std::hint::spin_loop();
        }
        let view = self
            .ts_map
            .slice(..ts_i as u64 * 8)
            .get_mapped_range()
            .expect("map view");
        let raw: Vec<u64> = view
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        drop(view);
        self.ts_map.unmap();
        marks.iter()
            .map(|(name, b, e)| ((*name).to_string(), (raw[*e as usize] - raw[*b as usize]) as f64 / 1e6))
            .collect()
    }

    /// 调试：单帧（BATCH=1）逐 kernel dispatch + 立即读回，定位不写入的 kernel。
    pub fn debug_intermediates(&mut self, x_row: &[f32]) -> Vec<(String, Vec<f32>)> {
        if !self.pipelines.contains_key(&1) {
            self.build_pipelines(1);
        }
        let pipes = &self.pipelines[&1];
        let bytes: Vec<u8> = x_row.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.queue.write_buffer(&self.buffers[0], 0, &bytes);

        // (kernel 名, pipeline idx, 线程数, 输出 buffer idx, 读回元素数)
        let stages = [
            ("stft", 0usize, 516u32, 2usize, 8usize),
            ("conv1", 1, 512, 3, 8),
            ("conv2", 2, 128, 4, 8),
            ("conv3", 3, 64, 5, 8),
            ("conv4", 4, 128, 6, 8),
            ("gates_in", 5, 512, 7, 8),
        ];
        let mut out = Vec::new();
        for (name, pi, n, bi, rn) in stages {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("dbg") });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("dbg"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipes[pi]);
                pass.set_bind_group(0, &self.batch_bgs[0][pi], &[]);
                let nx = n.div_ceil(64).min(65535);
                let ny = n.div_ceil(64).div_ceil(nx);
                pass.dispatch_workgroups(nx, ny, 1);
            }
            let dbg = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dbg"),
                size: (rn * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            enc.copy_buffer_to_buffer(&self.buffers[bi], 0, &dbg, 0, (rn * 4) as u64);
            self.queue.submit(Some(enc.finish()));
            let (tx, rx) = mpsc::channel();
            dbg.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).unwrap();
            });
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
            let view = dbg.slice(..).get_mapped_range().expect("map view");
            let vals: Vec<f32> = view
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            drop(view);
            dbg.unmap();
            dbg.destroy();
            out.push((name.to_string(), vals));
        }
        out
    }

    fn build_pipelines(&mut self, t: usize) {
        let offs = w_offsets();
        let mut src = format!("{}\n{}", WGSL_MATH, WGSL_TEMPLATE);
        src = src.replace("{BATCH}", &t.to_string());
        src = src.replace("{LSTM_CHUNK}", &LSTM_CHUNK.to_string());
        // 为每个 chunk 生成入口点：lstm_chunk_i 处理帧 [i*CHUNK, i*CHUNK+CHUNK)
        let chunks_src: String = (0..MAX_T.div_ceil(LSTM_CHUNK))
            .map(|i| {
                format!(
                    "@compute @workgroup_size(256)\nfn lstm_chunk_{i}(@builtin(local_invocation_id) lid: vec3<u32>) {{\n    lstm_chunk_impl({i}u * {LSTM_CHUNK}u, lid.x);\n}}\n"
                )
            })
            .collect();
        src = src.replace("{LSTM_CHUNKS}", &chunks_src);
        for (name, &off) in OFF_NAMES.iter().zip(offs[1..15].iter()) {
            src = src.replace(&format!("{{{}}}", name), &(off as u32).to_string());
        }
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_batch"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let mut pipes = Vec::new();
        for (ep, binds) in BATCH_PIPELINES.iter() {
            let bgl = Self::make_bgl(&self.device, binds);
            let pl = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(ep),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
            let pipe = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(ep),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(ep),
                compilation_options: Default::default(),
                cache: None,
            });
            pipes.push(pipe);
        }
        // lstm chunk pipelines（entry point lstm_chunk_i，D0 已烘进 shader）
        for i in 0..MAX_T.div_ceil(LSTM_CHUNK) {
            let ep = format!("lstm_chunk_{i}");
            let pipe = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("lstm_chunk"),
                layout: Some(&self.pl_layout),
                module: &shader,
                entry_point: Some(&ep),
                compilation_options: Default::default(),
                cache: None,
            });
            pipes.push(pipe);
        }
        self.pipelines.insert(t, pipes);
    }

    pub fn reset(&mut self) {
        self.inflight = None; // 约定 reset 前已 flush（gate 的 run 每流先 flush 再 reset）
        self.queue.write_buffer(&self.buffers[8], 0, &[0u8; 512]);
        self.queue.write_buffer(&self.buffers[9], 0, &[0u8; 512]);
    }
}

fn bytemuck_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
