//! 长音频端到端回归测试（GPU 后端）。
//!
//! **为什么需要它**：`src/bin/gate.rs` 的 10 变体扫描用的是 ≤60s 片段，而 silero 的
//! LSTM 是指数不稳定递归 —— 实测在 1525s 音频上，Intel 用硬件 `exp` 时从 frame 2784
//! 开始偏离、到 frame 12789 偏差已达 0.75，时间戳从 567 段掉到 555 段。
//! 也就是说**任何只覆盖短片段的 gate 都在结构上看不见这类缺陷**，它必须用长音频。
//!
//! 选 `v02`（25842 帧 ≈ 16.2 分钟）而不是更短的 fixture：发散在 ~frame 13000 才
//! 放大到能翻转时间戳的程度，`v05`/`v07`（174s/344s）不足以暴露，`v01`（97MB）又太重。
//!
//! 覆盖三件事：
//!   1. host CPU 与 GPU 都不得出现 |p - golden| >= 1e-3 的帧
//!   2. 两条路径的语音时间戳都必须与 golden 逐段精确相等
//!   3. GPU 侧不存在非有限值
//!
//! 更完整的跨厂商四路比对仍以 `cargo run --release --features gpu --bin align` 为准；
//! 本测试是不依赖人工执行的兜底。

#![cfg(feature = "gpu")]

use silero_vad_wgpu::gpu_batch::GpuBatch;
use silero_vad_wgpu::timestamps::{speech_timestamps_from_probs, TsParams};
use silero_vad_wgpu::{SileroVad, Weights, CFG_16K};

const CASE: &str = "v02";
/// 时间戳边界翻转的偏差量级是 O(0.1~1)；正常路径在 1e-5 量级。
const MAX_FRAME_DIFF: f32 = 1e-3;

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// 时间戳 golden 是扁平 i64 流：每段 `(start, end)` 共 2 个 i64（16 字节）。
fn read_pairs(path: &std::path::Path) -> Vec<(i64, i64)> {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    raw.chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect::<Vec<i64>>()
        .chunks_exact(2)
        .map(|c| (c[0], c[1]))
        .collect()
}

#[test]
fn long_audio_keeps_timestamps_exact() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = root.join("ref").join("video");
    let wav = read_f32(&dir.join(format!("{CASE}.f32")));
    let golden = read_f32(&dir.join(format!("{CASE}.probs.f32")));
    let ts_golden = read_pairs(&dir.join(format!("{CASE}.ts.i64")));
    let wpath = root.join("ref").join("silero_vad_16k_jit.safetensors");
    let wstr = wpath.to_str().expect("weights path");
    let params = TsParams::default();

    assert_eq!(wav.len().div_ceil(512), golden.len(), "fixture 帧数不一致");

    // host CPU
    let mut host = SileroVad::new(Weights::load(wstr).expect("weights"), CFG_16K).expect("cfg");
    host.reset();
    let mut pc = Vec::with_capacity(golden.len());
    let mut buf = vec![0.0f32; 512];
    for f in 0..golden.len() {
        let s = f * 512;
        let e = (s + 512).min(wav.len());
        buf[..e - s].copy_from_slice(&wav[s..e]);
        buf[e - s..].fill(0.0);
        pc.push(host.frame(&buf));
    }

    // GPU
    let mut gb = GpuBatch::new(&Weights::load(wstr).expect("weights"));
    let pd = gb.stream(&wav);
    assert_eq!(pd.len(), golden.len(), "GPU 帧数不一致");

    let worst = |p: &[f32]| -> (f32, usize) {
        let mut w = (0.0f32, 0usize);
        for (i, v) in p.iter().enumerate() {
            let d = (v - golden[i]).abs();
            if d > w.0 {
                w = (d, i);
            }
        }
        w
    };
    for (name, p) in [("host", &pc), ("gpu", &pd)] {
        assert!(
            p.iter().all(|v| v.is_finite()),
            "{name}: 概率流出现非有限值"
        );
        let (d, i) = worst(p);
        assert!(
            d < MAX_FRAME_DIFF,
            "{name}: frame {i} 偏差 {d:e} >= {MAX_FRAME_DIFF:e} —— \
             长音频上出现递归发散（检查 wgsl_math.wgsl 是否被改动）"
        );
    }

    let to_ts = |p: &[f32]| {
        speech_timestamps_from_probs(p, 16000, 512, Some(p.len() as i64 * 512), &params)
    };
    assert_eq!(
        to_ts(&pc),
        ts_golden,
        "host 时间戳与 golden 不一致（{} 段 vs {} 段）",
        to_ts(&pc).len(),
        ts_golden.len()
    );
    assert_eq!(
        to_ts(&pd),
        ts_golden,
        "GPU 时间戳与 golden 不一致（{} 段 vs {} 段）",
        to_ts(&pd).len(),
        ts_golden.len()
    );
}
