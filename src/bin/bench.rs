//! GPU 逐 kernel 耗时 + 吞吐基准，用于 wgpu RTFx 优化定位。
//!
//! 用法：
//!   SILERO_BACKEND=vulkan SILERO_ADAPTER=Intel  cargo run --release --features gpu --bin bench
//!   SILERO_BACKEND=vulkan SILERO_ADAPTER=NVIDIA cargo run --release --features gpu --bin bench
//!
//! 输出两部分：
//!   1) `debug_timing`：一个满 BATCH=512 批里每个 kernel 的 GPU 时间（相对占比有效）
//!   2) 端到端 RTFx：host 侧整段音频的墙钟吞吐，与 CPU 路径对照
//!
//! 注意：TIMESTAMP_QUERY 的时间戳周期按 1ns 近似，绝对值不可信，**看占比**。

use silero_vad_wgpu::gpu_batch::GpuBatch;
use silero_vad_wgpu::{SileroVad, Weights, CFG_16K};
use std::time::Instant;

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// 构造首帧的 x 行（ctx 64 + frame 512 + reflect pad 64）。
fn first_x_row(wav: &[f32]) -> Vec<f32> {
    let mut xr = vec![0.0f32; 640];
    let n = wav.len().min(512);
    xr[64..64 + n].copy_from_slice(&wav[..n]);
    for i in 0..64 {
        xr[576 + i] = xr[574 - i];
    }
    xr
}

fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let wstr = root.join("ref").join("silero_vad_16k_jit.safetensors");
    let wstr = wstr.to_str().unwrap();
    let wav = read_f32(&root.join("ref").join("video").join("v01.f32"));
    let dur = wav.len() as f64 / 16000.0;

    let info = silero_vad_wgpu::gpu::dump_adapters_env();
    println!("adapter: {info}");
    println!("audio: {} samples = {dur:.1}s (v01)\n", wav.len());

    let mut gb = GpuBatch::new(&Weights::load(wstr).expect("weights"));

    // --- 1) 逐 kernel 时间 ---
    let xr = first_x_row(&wav);
    let timings = gb.debug_timing(&xr);
    let total: f64 = timings.iter().map(|(_, ms)| *ms).sum();
    println!("per-kernel GPU time in one full BATCH=512 batch (share of total):");
    println!("  {:<10} {:>12} {:>8}", "kernel", "ms", "share");
    for (name, ms) in &timings {
        println!(
            "  {name:<10} {ms:>12.4} {:>7.1}%",
            if total > 0.0 { ms / total * 100.0 } else { 0.0 }
        );
    }
    println!("  {:<10} {total:>12.4}\n", "TOTAL");

    // --- 2) 单批墙钟（host 侧，不依赖时间戳周期，可与上面的 kernel 占比对照）---
    // 用真实音频拼一个满批 x(t*640)，反复提交，观察每批的 host 墙钟。
    const FRAME: usize = 512;
    const CTX: usize = 64;
    const PAD: usize = 64;
    const MAX_T: usize = 512;
    let mut ctx = vec![0.0f32; CTX];
    let mut x_batch = vec![0.0f32; MAX_T * (CTX + FRAME + PAD)];
    let mut fbuf = vec![0.0f32; FRAME];
    for f in 0..MAX_T {
        let s = f * FRAME;
        let e = (s + FRAME).min(wav.len());
        fbuf[..e - s].copy_from_slice(&wav[s..e]);
        fbuf[e - s..].fill(0.0);
        let b = f * (CTX + FRAME + PAD);
        x_batch[b..b + CTX].copy_from_slice(&ctx);
        x_batch[b + CTX..b + CTX + FRAME].copy_from_slice(&fbuf);
        for i in 0..PAD {
            x_batch[b + CTX + FRAME + i] = x_batch[b + CTX + FRAME - 2 - i];
        }
        ctx.copy_from_slice(&fbuf[FRAME - CTX..]);
    }

    gb.reset();
    let _ = gb.frame_batch(&x_batch, MAX_T); // warmup
    let mut bw = Vec::new();
    for _ in 0..20 {
        gb.reset();
        let t0 = Instant::now();
        let _ = gb.frame_batch(&x_batch, MAX_T);
        let _ = gb.flush();
        bw.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    bw.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let batch_ms = bw[bw.len() / 2];
    println!(
        "single full batch (512 frames) host wall: {batch_ms:.3} ms  \
         => {:.1} us/frame",
        batch_ms * 1e3 / MAX_T as f64
    );

    // 变批大小：若墙钟随工作量线性变化 → GPU 受限；若是平的小常数 → 开销/延迟受限。
    println!("\nbatch-size scaling (t 帧/批, 固定总帧数 = 4096):");
    println!("  {:>6} {:>10} {:>12} {:>14}", "t", "batches", "ms/batch", "us/frame");
    const TOTAL: usize = 4096;
    for t in [16usize, 64, 128, 256, 512] {
        gb.reset();
        let sub = &x_batch[..t * (CTX + FRAME + PAD)];
        let _ = gb.frame_batch(sub, t);
        let _ = gb.flush();
        let mut v = Vec::new();
        let reps = 8;
        for _ in 0..reps {
            gb.reset();
            let t0 = Instant::now();
            for _ in 0..(TOTAL / t) {
                let _ = gb.frame_batch(sub, t);
            }
            let _ = gb.flush();
            v.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ms = v[v.len() / 2];
        println!(
            "  {t:>6} {:>10} {ms:>12.4} {:>14.2}",
            ms / (TOTAL / t) as f64,
            ms * 1e3 / TOTAL as f64
        );
    }
    println!();

    // --- 3) 端到端 RTFx ---
    // warmup
    let _ = gb.stream(&wav[..16000 * 30]);
    let mut runs = Vec::new();
    for _ in 0..3 {
        let t0 = Instant::now();
        let probs = gb.stream(&wav);
        runs.push(dur / t0.elapsed().as_secs_f64());
        std::hint::black_box(probs);
    }
    runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let gpu = runs[runs.len() / 2];

    let mut host = SileroVad::new(Weights::load(wstr).expect("weights"), CFG_16K).expect("cfg");
    let mut buf = vec![0.0f32; 512];
    let mut run_cpu = |h: &mut SileroVad| -> Vec<f32> {
        h.reset();
        let n = wav.len().div_ceil(512);
        let mut out = Vec::with_capacity(n);
        for f in 0..n {
            let s = f * 512;
            let e = (s + 512).min(wav.len());
            buf[..e - s].copy_from_slice(&wav[s..e]);
            buf[e - s..].fill(0.0);
            out.push(h.frame(&buf));
        }
        out
    };
    let _ = run_cpu(&mut host);
    let mut cruns = Vec::new();
    for _ in 0..3 {
        let t0 = Instant::now();
        std::hint::black_box(run_cpu(&mut host));
        cruns.push(dur / t0.elapsed().as_secs_f64());
    }
    cruns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cpu = cruns[cruns.len() / 2];

    println!("end-to-end RTFx (median of 3, v01 = {dur:.0}s):");
    println!("  CPU (1 thread) : {cpu:>8.0}");
    println!("  GPU (wgpu)     : {gpu:>8.0}   ({:.2}x CPU)", gpu / cpu);
    println!(
        "  per-frame: cpu {:.1}us  gpu {:.1}us",
        1e6 / cpu / 31.25,
        1e6 / gpu / 31.25
    );
}
