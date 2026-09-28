//! 视频库音频四方对齐比对：
//!   A = 原版 CPU（golden，video_golden.py）
//!   B = 原版 CUDA（golden）
//!   C = 移植版 Rust CPU（host）
//!   D = 移植版 Rust WGPU（GpuBatch，512 帧批量）
//! 比对：逐帧概率 max-abs-diff + 时间戳逐对相等（A 为参照）。
//!
//! 用法：cargo run --release --features gpu --bin align

use silero_vad_wgpu::gpu_batch::{GpuBatch, MAX_T};
use silero_vad_wgpu::timestamps::{speech_timestamps_from_probs, TsParams};
use silero_vad_wgpu::vad_iterator::{VadEvent, VadIterator};
use silero_vad_wgpu::{SileroVad, Weights, CFG_16K};

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

fn pairs(raw: Vec<i64>) -> Vec<(i64, i64)> {
    raw.chunks_exact(2).map(|c| (c[0], c[1])).collect()
}

fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn main() {
    let root = env!("CARGO_MANIFEST_DIR");
    let vdir = std::path::Path::new(root).join("ref").join("video");
    let params = TsParams::default();
    let weights = Weights::load(
        std::path::Path::new(root)
            .join("ref")
            .join("silero_vad_16k_jit.safetensors")
            .to_str()
            .unwrap(),
    )
    .expect("weights");

    let mut names: Vec<String> = std::fs::read_dir(&vdir)
        .expect("vdir")
        .filter_map(|e| {
            let p = e.ok()?.path();
            let n = p.file_name()?.to_str()?.to_string();
            // 只取 vNN.f32（跳过 vNN.probs*.f32 等）
            if n.ends_with(".f32")
                && !n.contains(".probs")
                && n.starts_with('v')
                && n[..n.len() - 4].chars().skip(1).all(|c| c.is_ascii_digit())
            {
                Some(n[..n.len() - 4].to_string())
            } else {
                None
            }
        })
        .collect();
    names.sort();

    let mut host = SileroVad::new(
        Weights::load(
            std::path::Path::new(root)
                .join("ref")
                .join("silero_vad_16k_jit.safetensors")
                .to_str()
                .unwrap(),
        )
        .unwrap(),
        CFG_16K,
    )
    .expect("cfg");
    // E：Rust VadIterator（增量式，API 对齐验证）
    let mut vit = VadIterator::new(
        Weights::load(
            std::path::Path::new(root)
                .join("ref")
                .join("silero_vad_16k_jit.safetensors")
                .to_str()
                .unwrap(),
        )
        .unwrap(),
        CFG_16K,
    )
    .expect("vadit");
    let mut gb = GpuBatch::new(&weights);

    println!(
        "{:<4} {:>8} {:>10} {:>10} {:>10} {:>10}  {:>4} {:>4} {:>4} {:>4} {:>4}  rtfx_cpu/rtfx_gpu",
        "name",
        "frames",
        "B-A(dif)",
        "C-A(dif)",
        "D-A(dif)",
        "D-C(dif)",
        "B=A",
        "C=A",
        "D=A",
        "D=C",
        "E=P"
    );
    let mut all_ok = true;
    for name in &names {
        let wav = read_f32(&vdir.join(format!("{name}.f32")));
        let pa = read_f32(&vdir.join(format!("{name}.probs.f32")));
        let pb = read_f32(&vdir.join(format!("{name}.probs_cuda.f32")));
        let ta = pairs(read_i64(&vdir.join(format!("{name}.ts.i64"))));
        let tb = pairs(read_i64(&vdir.join(format!("{name}.ts_cuda.i64"))));
        // 原版 VADIterator 增量事件 golden：kind(0=start/1=end)+sample 交错
        let ev_py: Vec<VadEvent> = read_i64(&vdir.join(format!("{name}.vadit.i64")))
            .chunks_exact(2)
            .map(|c| if c[0] == 0 { VadEvent::Start(c[1]) } else { VadEvent::End(c[1]) })
            .collect();
        let dur = wav.len() as f64 / 16000.0;

        // C：Rust host CPU
        let stream_host = |vad: &mut SileroVad| -> Vec<f32> {
            vad.reset();
            let frame = vad.cfg().frame as usize;
            let n_frames = (wav.len() + frame - 1) / frame;
            let mut probs = Vec::with_capacity(n_frames);
            let mut buf = vec![0.0f32; frame];
            for f in 0..n_frames {
                let s = f * frame;
                let e = (s + frame).min(wav.len());
                buf[..e - s].copy_from_slice(&wav[s..e]);
                buf[e - s..].fill(0.0);
                probs.push(vad.frame(&buf));
            }
            probs
        };
        stream_host(&mut host); // warmup
        let t0 = std::time::Instant::now();
        let pc = stream_host(&mut host);
        let rtfx_cpu = dur / t0.elapsed().as_secs_f64();
        let tc = speech_timestamps_from_probs(
            &pc,
            16000,
            512,
            Some(pc.len() as i64 * 512),
            &params,
        );

        // D：Rust WGPU 批量
        let frame = 512usize;
        let n_frames = (wav.len() + frame - 1) / frame;
        let run_batch = |gb: &mut GpuBatch| -> Vec<f32> {
            gb.reset();
            let mut ctx = vec![0.0f32; 64];
            let mut x_all = vec![0.0f32; MAX_T * 640];
            let mut frame_buf = vec![0.0f32; frame];
            let mut probs = Vec::with_capacity(n_frames);
            for batch_start in (0..n_frames).step_by(MAX_T) {
                let t = MAX_T.min(n_frames - batch_start);
                for f in 0..t {
                    let fi = batch_start + f;
                    let s = fi * frame;
                    let e = (s + frame).min(wav.len());
                    frame_buf[..e - s].copy_from_slice(&wav[s..e]);
                    frame_buf[e - s..].fill(0.0);
                    let base = f * 640;
                    x_all[base..base + 64].copy_from_slice(&ctx);
                    x_all[base + 64..base + 64 + frame].copy_from_slice(&frame_buf);
                    for i in 0..64 {
                        x_all[base + 64 + frame + i] = x_all[base + 64 + frame - 2 - i];
                    }
                    ctx.copy_from_slice(&frame_buf[448..]);
                }
                probs.extend(gb.frame_batch(&x_all[..t * 640], t).unwrap_or_default());
            }
            if let Some(p) = gb.flush() {
                probs.extend(p);
            }
            probs
        };
        let _ = run_batch(&mut gb); // warmup
        let t0 = std::time::Instant::now();
        let pd = run_batch(&mut gb);
        let rtfx_gpu = dur / t0.elapsed().as_secs_f64();
        let td = speech_timestamps_from_probs(
            &pd,
            16000,
            512,
            Some(pd.len() as i64 * 512),
            &params,
        );

        assert_eq!(pc.len(), pa.len(), "{name} host frame count");
        assert_eq!(pd.len(), pa.len(), "{name} batch frame count");

        // E：Rust VadIterator 增量事件
        vit.reset();
        let mut ev_rs: Vec<VadEvent> = Vec::new();
        for f in 0..n_frames {
            let s = f * 512;
            let e = (s + 512).min(wav.len());
            let mut chunk = vec![0.0f32; 512];
            chunk[..e - s].copy_from_slice(&wav[s..e]);
            if let Some(ev) = vit.push(&chunk) {
                ev_rs.push(ev);
            }
        }
        let te = ev_rs == ev_py;
        all_ok &= te;

        let eq = |x: &Vec<(i64, i64)>, y: &Vec<(i64, i64)>| if x == y { "OK" } else { "FAIL" };
        let db_a = maxdiff(&pb, &pa);
        let dc_a = maxdiff(&pc, &pa);
        let dd_a = maxdiff(&pd, &pa);
        let dd_c = maxdiff(&pd, &pc);
        let ok_ts = ta == tb && ta == tc && ta == td;
        all_ok &= ok_ts;
        println!(
            "{:<4} {:>8} {:>10.2e} {:>10.2e} {:>10.2e} {:>10.2e}  {:>4} {:>4} {:>4} {:>4} {:>4}  {rtfx_cpu:.0}/{rtfx_gpu:.0}",
            name,
            pa.len(),
            db_a,
            dc_a,
            dd_a,
            dd_c,
            eq(&tb, &ta),
            eq(&tc, &ta),
            eq(&td, &ta),
            eq(&td, &tc),
            if te { "OK" } else { "FAIL" },
        );
        if !ok_ts {
            println!("    ts mismatch: A={} B={} C={} D={}", ta.len(), tb.len(), tc.len(), td.len());
        }
        if !te {
            println!(
                "    vadit mismatch: py={} rust={}",
                ev_py.len(),
                ev_rs.len()
            );
            for (i, (a, b)) in ev_py.iter().zip(ev_rs.iter()).enumerate() {
                if a != b {
                    println!("    first diff at event {i}: py={a:?} rust={b:?}");
                    break;
                }
            }
        }
    }
    println!(
        "{}",
        if all_ok {
            "ALIGN: PASS (ts 4-way exact + vadit events exact)"
        } else {
            "ALIGN: FAIL"
        }
    );
}
