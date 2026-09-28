//! `get_speech_timestamps_from_probs` 的逐行移植（utils_vad.py L421-588）。
//! 目标是与 Python 版在相同概率序列下产出**完全一致**的时间戳（G2 门槛）。
//! 默认参数与 Python 一致；返回 (start, end) 样本坐标对（return_seconds=False）。

#[derive(Debug, Clone)]
pub struct TsParams {
    pub threshold: f64,
    pub min_speech_duration_ms: f64,
    pub max_speech_duration_s: f64,
    pub min_silence_duration_ms: f64,
    pub speech_pad_ms: f64,
    pub neg_threshold: Option<f64>,
    pub min_silence_at_max_speech: f64,
    pub use_max_poss_sil_at_max_speech: bool,
}

impl Default for TsParams {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_speech_duration_ms: 250.0,
            max_speech_duration_s: f64::INFINITY,
            min_silence_duration_ms: 100.0,
            speech_pad_ms: 30.0,
            neg_threshold: None,
            min_silence_at_max_speech: 98.0,
            use_max_poss_sil_at_max_speech: true,
        }
    }
}

/// audio_length_samples=None → probs.len() * window（与 Python 默认一致）。
pub fn speech_timestamps_from_probs(
    probs: &[f32],
    sampling_rate: i64,
    window_size_samples: i64,
    audio_length_samples: Option<i64>,
    p: &TsParams,
) -> Vec<(i64, i64)> {
    let audio_length_samples =
        audio_length_samples.unwrap_or(probs.len() as i64 * window_size_samples);

    let min_speech_samples = sampling_rate as f64 * p.min_speech_duration_ms / 1000.0;
    let speech_pad_samples = sampling_rate as f64 * p.speech_pad_ms / 1000.0;
    let max_speech_samples =
        sampling_rate as f64 * p.max_speech_duration_s - window_size_samples as f64
            - 2.0 * speech_pad_samples;
    let min_silence_samples = sampling_rate as f64 * p.min_silence_duration_ms / 1000.0;
    let min_silence_samples_at_max_speech =
        sampling_rate as f64 * p.min_silence_at_max_speech / 1000.0;

    let neg_threshold = p.neg_threshold.unwrap_or((p.threshold - 0.15).max(0.01));

    let mut triggered = false;
    let mut speeches: Vec<(i64, i64)> = Vec::new();
    let mut cur_start: Option<i64> = None; // current_speech dict 只有 start/end 两个键

    let mut temp_end: i64 = 0;
    let mut prev_end: i64 = 0;
    let mut next_start: i64 = 0;
    let mut possible_ends: Vec<(i64, i64)> = Vec::new();

    for (i, &prob) in probs.iter().enumerate() {
        let cur_sample = window_size_samples * i as i64;
        let above = prob as f64 >= p.threshold;

        // 语音在 temp_end 之后回归：记录候选静音（足够长才记），清 temp_end
        if above && temp_end != 0 {
            let sil_dur = cur_sample - temp_end;
            if sil_dur as f64 > min_silence_samples_at_max_speech {
                possible_ends.push((temp_end, sil_dur));
            }
            temp_end = 0;
            if next_start < prev_end {
                next_start = cur_sample;
            }
        }

        // 语音开始
        if above && !triggered {
            triggered = true;
            cur_start = Some(cur_sample);
            continue;
        }

        // 达到最大语音长度：决定切割点
        if triggered
            && cur_start.is_some()
            && (cur_sample - cur_start.unwrap()) as f64 > max_speech_samples
        {
            if p.use_max_poss_sil_at_max_speech && !possible_ends.is_empty() {
                // Python max(key=x[1]) 返回首个最大值；Rust max_by_key 返回最后一个，
                // 手写 fold 保持"取首个最大"语义
                let best = possible_ends.iter().fold(possible_ends[0], |acc, &x| {
                    if x.1 > acc.1 {
                        x
                    } else {
                        acc
                    }
                });
                prev_end = best.0;
                let dur = best.1;
                if let Some(s) = cur_start.take() {
                    speeches.push((s, prev_end));
                }
                next_start = prev_end + dur;

                if next_start < prev_end + cur_sample {
                    cur_start = Some(next_start);
                } else {
                    triggered = false;
                }
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                possible_ends.clear();
            } else if prev_end != 0 {
                let s = cur_start.take().unwrap();
                speeches.push((s, prev_end));
                if next_start < prev_end {
                    triggered = false;
                } else {
                    cur_start = Some(next_start);
                }
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                possible_ends.clear();
            } else {
                let s = cur_start.take().unwrap();
                speeches.push((s, cur_sample));
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                triggered = false;
                possible_ends.clear();
                continue;
            }
        }

        // 语音中检测静音
        if ((prob as f64) < neg_threshold) && triggered {
            if temp_end == 0 {
                temp_end = cur_sample;
            }
            let sil_dur_now = cur_sample - temp_end;

            if !p.use_max_poss_sil_at_max_speech
                && sil_dur_now as f64 > min_silence_samples_at_max_speech
            {
                prev_end = temp_end;
            }

            if (sil_dur_now as f64) < min_silence_samples {
                continue;
            } else {
                let end = temp_end;
                if let Some(s) = cur_start {
                    if (end - s) as f64 > min_speech_samples {
                        speeches.push((s, end));
                    }
                }
                cur_start = None;
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                triggered = false;
                possible_ends.clear();
                continue;
            }
        }
    }

    if let Some(s) = cur_start {
        if (audio_length_samples - s) as f64 > min_speech_samples {
            speeches.push((s, audio_length_samples));
        }
    }

    // padding（与 Python 的循环逐行对应）
    let n = speeches.len();
    for i in 0..n {
        if i == 0 {
            let (s, e) = speeches[i];
            speeches[i].0 = (s as f64 - speech_pad_samples).max(0.0) as i64;
            let _ = e;
        }
        if i != n - 1 {
            let silence_duration = speeches[i + 1].0 - speeches[i].1;
            if (silence_duration as f64) < 2.0 * speech_pad_samples {
                speeches[i].1 += silence_duration / 2;
                speeches[i + 1].0 =
                    (speeches[i + 1].0 as f64 - (silence_duration / 2) as f64).max(0.0)
                        as i64;
            } else {
                speeches[i].1 = (speeches[i].1 as f64 + speech_pad_samples)
                    .min(audio_length_samples as f64) as i64;
                speeches[i + 1].0 = (speeches[i + 1].0 as f64 - speech_pad_samples)
                    .max(0.0) as i64;
            }
        } else {
            speeches[i].1 = (speeches[i].1 as f64 + speech_pad_samples)
                .min(audio_length_samples as f64) as i64;
        }
    }

    speeches
}
