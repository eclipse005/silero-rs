//! 长音频 GPU 发散定位：流式跑 ref 数据，报告第一处显著偏离 golden 的帧。
//!
//! 用法：
//!   cargo run --release --features gpu --bin trace -- video v01
//!   cargo run --release --features gpu --bin trace -- gate test60s
//!
//! 目标是区分两种失效模式：
//!   A) 从第 0 帧起就稳定偏离 → 单帧计算（kernel 数值语义）问题
//!   B) 前 N 帧正常、某帧突然跳变 → 批边界 / 流水线 / 状态搬运问题

use silero_vad_wgpu::gpu_batch::GpuBatch;
use silero_vad_wgpu::timestamps::{speech_timestamps_from_probs, TsParams};
use silero_vad_wgpu::Weights;

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_i64(path: &std::path::Path) -> Vec<i64> {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    raw.chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}

/// 逐帧 GpuVad 路径（gpu.wgsl，8 dispatch/帧，独立实现）跑同一段音频。
/// 与 GpuBatch 对照可判别：两条路径都发散 = 驱动/精度层面；只有 batch 发散 = 批量 kernel。
fn stream_framewise(wav: &[f32], weights: &Weights) -> Vec<f32> {
    use silero_vad_wgpu::gpu::GpuVad;
    let mut gv = GpuVad::new(weights);
    let n = wav.len().div_ceil(512);
    let mut out = Vec::with_capacity(n);
    let mut buf = vec![0.0f32; 512];
    for f in 0..n {
        let s = f * 512;
        let e = (s + 512).min(wav.len());
        buf[..e - s].copy_from_slice(&wav[s..e]);
        buf[e - s..].fill(0.0);
        out.push(gv.frame(&buf));
    }
    out
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let framewise = a.iter().any(|x| x == "--framewise");
    let dump = a
        .iter()
        .position(|x| x == "--dump")
        .and_then(|i| a.get(i + 1))
        .cloned();
    let a: Vec<&String> = a
        .iter()
        .filter(|x| *x != "--framewise" && *x != "--dump")
        .collect();
    let (kind, name) = if a.len() >= 2 {
        (a[0].clone(), a[1].clone())
    } else {
        ("gate".into(), "test60s".into())
    };
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let (wav_path, golden_path, ts_path) = match kind.as_str() {
        "video" => {
            let d = root.join("ref").join("video");
            (
                d.join(format!("{name}.f32")),
                d.join(format!("{name}.probs.f32")),
                d.join(format!("{name}.ts.i64")),
            )
        }
        _ => {
            let d = root.join("ref").join("gate");
            (
                d.join(format!("{name}.wav.f32")),
                d.join(format!("{name}.probs.f32")),
                d.join(format!("{name}.ts.i64")),
            )
        }
    };
    let wstr = root.join("ref").join("silero_vad_16k_jit.safetensors");
    let wstr = wstr.to_str().unwrap();
    let wav = read_f32(&wav_path);
    let pg = read_f32(&golden_path);
    let tgold: Vec<(i64, i64)> = read_i64(&ts_path)
        .chunks_exact(2)
        .map(|c| (c[0], c[1]))
        .collect();
    println!(
        "{kind}/{name}: samples={} frames={} dur={:.1}s",
        wav.len(),
        wav.len().div_ceil(512),
        wav.len() as f64 / 16000.0
    );
    silero_vad_wgpu::gpu::maybe_dump_adapters();

    let mut gb = GpuBatch::new(&Weights::load(wstr).expect("weights"));
    let pd = if framewise {
        println!("mode: 逐帧 GpuVad (gpu.wgsl)");
        stream_framewise(&wav, &Weights::load(wstr).expect("weights"))
    } else {
        println!("mode: 批量 GpuBatch (gpu_batch.wgsl)");
        gb.stream(&wav)
    };
    assert_eq!(pd.len(), pg.len(), "frame count mismatch");

    if let Some(path) = dump.as_deref() {
        let mut b = Vec::with_capacity(pd.len() * 4);
        for v in &pd {
            b.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(path, &b).expect("dump");
        println!("dumped {} probs -> {path}", pd.len());
    }

    // 逐帧扫描：找出第一处超过阈值的位置，以及整体分布
    const TOL: f32 = 1e-3;
    let mut first_bad: Option<usize> = None;
    let mut first_mid: Option<usize> = None;
    let mut hist = [0usize; 5]; // <1e-5, <1e-4, <1e-3, <1e-1, >=1e-1
    for i in 0..pd.len() {
        let d = (pd[i] - pg[i]).abs();
        hist[if d < 1e-5 {
            0
        } else if d < 1e-4 {
            1
        } else if d < TOL {
            2
        } else if d < 1e-1 {
            3
        } else {
            4
        }] += 1;
        if first_mid.is_none() && d >= 1e-4 {
            first_mid = Some(i);
        }
        if first_bad.is_none() && d >= TOL {
            first_bad = Some(i);
        }
    }
    println!(
        "diff histogram: <1e-5:{}  <1e-4:{}  <1e-3:{}  <1e-1:{}  >=1e-1:{}",
        hist[0], hist[1], hist[2], hist[3], hist[4]
    );
    println!("first frame with |d|>=1e-4 : {first_mid:?}");
    println!("first frame with |d|>=1e-3 : {first_bad:?}");

    if let Some(i) = first_bad {
        let lo = i.saturating_sub(4);
        let hi = (i + 5).min(pd.len());
        println!("--- around first bad frame {i} (batch {}) ---", i / 512);
        for j in lo..hi {
            let mark = if j == i { "  <== BAD" } else { "" };
            println!(
                "  f{j:>6}: gpu={:.6} golden={:.6} cpu-ref |d|={:.3e}{mark}",
                pd[j],
                pg[j],
                (pd[j] - pg[j]).abs()
            );
        }
    }

    let params = TsParams::default();
    let td = speech_timestamps_from_probs(&pd, 16000, 512, Some(pd.len() as i64 * 512), &params);
    println!("ts golden={} gpu={} equal={}", tgold.len(), td.len(), tgold == td);

    // 定位第一处时间戳差异
    if tgold != td {
        for (i, (g, d)) in tgold.iter().zip(td.iter()).enumerate() {
            if g != d {
                println!("  first ts diff at seg {i}: golden={g:?} gpu={d:?}");
                break;
            }
        }
    }
}
