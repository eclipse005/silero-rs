//! silero-vad 命令行工具（跨平台）：
//!   silero-vad [options] <audio.wav>
//!
//! 后端：cpu（默认，SIMD 自动检测）/ gpu（wgpu，需 --features gpu 编译）。
//! 输出：JSON 语音段 [{"start":..,"end":..}]（样本或秒）。
//! 音频：16k/8k mono WAV（其他采样率先重采样，等价原版 read_audio 约束）。
//! 模型：默认内嵌官方 16k 权重；8k 用 --model 指定权重文件。

use silero_vad_wgpu::timestamps::TsParams;

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(2);
}

struct Args {
    input: String,
    backend: String,
    model: Option<String>,
    seconds: bool,
    time_resolution: u32,
    threshold: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        input: String::new(),
        backend: "cpu".into(),
        model: None,
        seconds: false,
        time_resolution: 1,
        threshold: 0.5,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--backend" => a.backend = it.next().unwrap_or_else(|| die("--backend needs a value")),
            "--model" => a.model = Some(it.next().unwrap_or_else(|| die("--model needs a path"))),
            "--seconds" => a.seconds = true,
            "--threshold" => {
                a.threshold = it
                    .next()
                    .unwrap_or_else(|| die("--threshold needs a value"))
                    .parse()
                    .unwrap_or_else(|_| die("--threshold must be f64"))
            }
            "--time-resolution" => {
                a.time_resolution = it
                    .next()
                    .unwrap_or_else(|| die("--time-resolution needs a value"))
                    .parse()
                    .unwrap_or_else(|_| die("--time-resolution must be u32"))
            }
            _ if arg.starts_with('-') => die(&format!("unknown option {arg}")),
            _ => a.input = arg,
        }
    }
    if a.input.is_empty() {
        eprintln!(
            "usage: silero-vad [--backend cpu|gpu] [--seconds] [--threshold 0.5] \
             [--time-resolution 1] [--model weights.safetensors] <audio.wav>"
        );
        std::process::exit(2);
    }
    a
}

/// wav → mono f32（f32 原样 / int 线性归一），多声道按块平均；返回 (samples, sr)。
fn read_wav_mono(path: &str) -> (Vec<f32>, u32) {
    let mut r = hound::WavReader::open(path).unwrap_or_else(|e| die(&format!("open wav: {e}")));
    let spec = r.spec();
    let nch = spec.channels as usize;
    let avg = |c: &[f32]| c.iter().sum::<f32>() / nch as f32;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            let all: Vec<f32> = r.samples::<f32>().map(|s| s.unwrap()).collect();
            if nch == 1 {
                all
            } else {
                all.chunks(nch).map(avg).collect()
            }
        }
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            let all: Vec<f32> = r
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / max)
                .collect();
            if nch == 1 {
                all
            } else {
                all.chunks(nch).map(avg).collect()
            }
        }
    };
    (samples, spec.sample_rate)
}

fn main() {
    let args = parse_args();
    let (wav, sr) = read_wav_mono(&args.input);
    if sr != 16000 && sr != 8000 {
        die(&format!(
            "sample rate {sr} not supported (8000/16000 only) — resample first"
        ));
    }
    let params = TsParams {
        threshold: args.threshold,
        ..TsParams::default()
    };
    let to_out = |x: i64| -> OutVal {
        if args.seconds {
            OutVal::Sec(silero_vad_wgpu::to_seconds(x, sr as i64, args.time_resolution))
        } else {
            OutVal::Sample(x)
        }
    };

    // 加载权重：16k 用内嵌，8k/自定义用 --model
    let load = |want_sr: u32| -> silero_vad_wgpu::Weights {
        match (&args.model, want_sr) {
            (Some(p), _) => silero_vad_wgpu::Weights::load(p).unwrap_or_else(|e| die(&format!("weights: {e}"))),
            (None, 16000) => {
                silero_vad_wgpu::Weights::embedded_16k().unwrap_or_else(|e| die(&format!("embedded weights: {e}")))
            }
            (None, _) => die("8k audio needs --model (embedded weights are 16k only)"),
        }
    };

    let ts: Vec<(i64, i64)> = match args.backend.as_str() {
        "cpu" => {
            let mut vad = silero_vad_wgpu::SileroVad::new(load(sr), cfg_for(sr)).unwrap_or_else(|e| die(&e));
            silero_vad_wgpu::get_speech_timestamps(&mut vad, &wav, &params)
        }
        #[cfg(feature = "gpu")]
        "gpu" => {
            use silero_vad_wgpu::timestamps::speech_timestamps_from_probs;
            if sr != 16000 {
                die("gpu backend is 16k-only");
            }
            let mut gb = silero_vad_wgpu::gpu_batch::GpuBatch::new(&load(16000));
            let probs = gb.stream(&wav);
            speech_timestamps_from_probs(&probs, 16000, 512, Some(wav.len() as i64), &params)
        }
        #[cfg(not(feature = "gpu"))]
        "gpu" => die("built without gpu support — rebuild with --features gpu"),
        other => die(&format!("unknown backend {other} (cpu|gpu)")),
    };

    let items: Vec<String> = ts
        .iter()
        .map(|(s, e)| match (to_out(*s), to_out(*e)) {
            (OutVal::Sample(a), OutVal::Sample(b)) => {
                format!("{{\"start\": {a}, \"end\": {b}}}")
            }
            (OutVal::Sec(a), OutVal::Sec(b)) => {
                format!("{{\"start\": {a}, \"end\": {b}}}")
            }
            _ => unreachable!(),
        })
        .collect();
    println!("[{}]", items.join(", "));
}

fn cfg_for(sr: u32) -> silero_vad_wgpu::ModelCfg {
    if sr == 8000 {
        silero_vad_wgpu::CFG_8K
    } else {
        silero_vad_wgpu::CFG_16K
    }
}

/// 输出值的双态（样本 / 秒）
enum OutVal {
    Sample(i64),
    Sec(f64),
}
