//! API smoke test：内嵌权重 + 合成音频，验证公开 API 行为（CI 无数据依赖可跑）。
//!
//! 合成信号：语音段（440Hz 正弦 + 谐波）与静音交替。VAD 对正弦的响应与真实语音
//! 不同（可能不触发），因此只断言结构性质：概率数量、时间戳单调、事件配对、
//! collect_chunks 长度合法、GPU stream 与 CPU 帧数一致（gpu feature 下）。

use silero_vad_wgpu::timestamps::TsParams;
use silero_vad_wgpu::vad_iterator::{VadEvent, VadIterator};
use silero_vad_wgpu::{collect_chunks, get_speech_timestamps, SileroVad, CFG_16K};

const SR: usize = 16000;
const FRAME: usize = 512;

fn synth() -> Vec<f32> {
    // 10 段 1s "语音"（有能量）+ 10 段 1s 静音交替，共 20s
    let mut v = Vec::with_capacity(20 * SR);
    for seg in 0..20 {
        let speech = seg % 2 == 0;
        for i in 0..SR {
            if speech {
                let t = i as f32 / SR as f32;
                let a = 0.4 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 880.0 * t).sin();
                v.push(a);
            } else {
                v.push(0.0);
            }
        }
    }
    v
}

fn frames(audio: &[f32]) -> Vec<Vec<f32>> {
    let n = audio.len().div_ceil(FRAME);
    (0..n)
        .map(|f| {
            let s = f * FRAME;
            let e = (s + FRAME).min(audio.len());
            let mut c = vec![0.0f32; FRAME];
            c[..e - s].copy_from_slice(&audio[s..e]);
            c
        })
        .collect()
}

#[test]
fn embedded_weights_load() {
    let w = silero_vad_wgpu::Weights::embedded_16k().expect("embedded weights");
    assert_eq!(w.basis.len(), 258 * 256);
    assert_eq!(w.wih.len(), 512 * 128);
    assert!(w.fb.len() == 1);
}

#[test]
fn one_shot_timestamps_are_monotonic() {
    let mut vad = SileroVad::new(silero_vad_wgpu::Weights::embedded_16k().unwrap(), CFG_16K).unwrap();
    let audio = synth();
    let ts = get_speech_timestamps(&mut vad, &audio, &TsParams::default());
    for w in ts.windows(2) {
        assert!(w[0].1 <= w[1].0, "segments must not overlap: {ts:?}");
    }
    for (s, e) in &ts {
        assert!(e > s);
        assert!(*e <= audio.len() as i64);
    }
    let speech = collect_chunks(&ts, &audio);
    assert!(speech.len() <= audio.len());
    // 正弦段能量高，VAD 应该至少触发一部分段（宽松断言，避免调参耦合）
    if !ts.is_empty() {
        assert!(speech.len() > 0);
    }
}

#[test]
fn vad_iterator_events_are_paired() {
    let it_weights = silero_vad_wgpu::Weights::embedded_16k().unwrap();
    let mut it = VadIterator::new(it_weights, CFG_16K).unwrap();
    let audio = synth();
    let mut depth = 0i32;
    let mut n_events = 0usize;
    for chunk in frames(&audio) {
        match it.push(&chunk) {
            Some(VadEvent::Start(_)) => {
                assert_eq!(depth, 0, "Start while already in speech");
                depth += 1;
                n_events += 1;
            }
            Some(VadEvent::End(_)) => {
                assert_eq!(depth, 1, "End without Start");
                depth -= 1;
                n_events += 1;
            }
            None => {}
        }
    }
    // 事件配对：流结束时不能悬空
    assert_eq!(depth, 0, "unterminated speech segment");
    assert!(n_events % 2 == 0, "events must pair up: {n_events}");
}

#[test]
fn probs_shape_matches_frames() {
    let mut vad = SileroVad::new(silero_vad_wgpu::Weights::embedded_16k().unwrap(), CFG_16K).unwrap();
    let audio = synth();
    let n_frames = audio.len().div_ceil(FRAME);
    let mut probs = Vec::with_capacity(n_frames);
    vad.reset();
    for chunk in frames(&audio) {
        probs.push(vad.frame(&chunk));
    }
    assert_eq!(probs.len(), n_frames);
    assert!(probs.iter().all(|p| (0.0..=1.0).contains(p)));
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_stream_matches_frame_count() {
    use silero_vad_wgpu::gpu_batch::GpuBatch;
    let audio = synth();
    let mut gb = GpuBatch::new(&silero_vad_wgpu::Weights::embedded_16k().unwrap());
    let probs = gb.stream(&audio);
    assert_eq!(probs.len(), audio.len().div_ceil(FRAME));
    assert!(probs.iter().all(|p| (0.0..=1.0).contains(p)));
}
