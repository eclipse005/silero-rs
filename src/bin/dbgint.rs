//! 调试：帧 0 前馈中间量读回（与 numpy 参考比对），定位设备相关数值差异。
//! 用法：cargo run --release --features gpu --bin dbgint -- <fixture.f32> [frame_idx]

use silero_vad_wgpu::gpu_batch::GpuBatch;
use silero_vad_wgpu::Weights;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("fixture path");
    let frame_idx: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(0);

    let weights =
        Weights::load("ref/silero_vad_16k_jit.safetensors").expect("weights");
    let mut gb = GpuBatch::new(&weights);

    // 构造 x 行：ctx(64, 首帧为 0) + frame(512) + reflect pad(64)
    let wav_raw = std::fs::read(path).unwrap();
    let wav: Vec<f32> = wav_raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let s = frame_idx * 512;
    let e = (s + 512).min(wav.len());
    let mut xr = vec![0.0f32; 640];
    xr[64..64 + (e - s)].copy_from_slice(&wav[s..e]);
    for i in 0..64 {
        xr[576 + i] = xr[574 - i];
    }

    let stages = gb.debug_intermediates(&xr);
    for (name, vals) in &stages {
        println!("{name}: {vals:?}");
    }
}
