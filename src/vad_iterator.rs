//! 原版 `VADIterator`（v6.2.3 的 utils_vad.py）的逐行移植 —— 增量式流状态机。
//!
//! 与离线版 [`crate::timestamps`] 的区别：VADIterator 每喂一个窗口就增量输出
//! Start/End 事件（语音起点 = 当前样本 - speech_pad - 窗长，终点判定用
//! threshold - 0.15 的滞回 + min_silence），语义与原版逐位一致。
//!
//! 注意：事件幅度含 speech_pad_ms 的"超前"（原版如此），流结束时若仍处于
//! 语音段，原版不会补发 End 事件，本实现保持一致。

use crate::{ModelCfg, SileroVad, Weights};

/// 增量事件：语音段起点 / 终点（样本坐标，含 pad 语义，与原版一致）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadEvent {
    Start(i64),
    End(i64),
}

/// 默认参数（与原版 VADIterator 相同）：threshold=0.5, min_silence=100ms, pad=30ms。
pub struct VadIterator {
    vad: SileroVad,
    threshold: f64,
    min_silence_samples: f64,
    speech_pad_samples: f64,
    triggered: bool,
    temp_end: i64,
    current_sample: i64,
}

impl VadIterator {
    /// 默认参数构造（threshold=0.5, min_silence_duration_ms=100, speech_pad_ms=30）。
    pub fn new(w: Weights, cfg: ModelCfg) -> Result<Self, String> {
        Self::new_with(w, cfg, 0.5, 100.0, 30.0)
    }

    /// 显式参数构造（同原版 VADIterator 构造参数）。
    pub fn new_with(
        w: Weights,
        cfg: ModelCfg,
        threshold: f64,
        min_silence_duration_ms: f64,
        speech_pad_ms: f64,
    ) -> Result<Self, String> {
        if cfg.sr != 8000 && cfg.sr != 16000 {
            return Err("VADIterator does not support sampling rates other than [8000, 16000]".into());
        }
        Ok(Self {
            vad: SileroVad::new(w, cfg)?,
            threshold,
            min_silence_samples: cfg.sr as f64 * min_silence_duration_ms / 1000.0,
            speech_pad_samples: cfg.sr as f64 * speech_pad_ms / 1000.0,
            triggered: false,
            temp_end: 0,
            current_sample: 0,
        })
    }

    /// 等效原版 reset_states()。
    pub fn reset(&mut self) {
        self.vad.reset();
        self.triggered = false;
        self.temp_end = 0;
        self.current_sample = 0;
    }

    pub fn cfg(&self) -> ModelCfg {
        self.vad.cfg()
    }

    /// 已消费的样本数（原版 current_sample）。
    pub fn current_sample(&self) -> i64 {
        self.current_sample
    }

    /// 喂一个窗口（长度须等于 cfg.frame：16k=512 / 8k=256），返回增量事件。
    /// 语义逐行对应 utils_vad.py（v6.2.3）的 VADIterator.__call__。
    pub fn push(&mut self, chunk: &[f32]) -> Option<VadEvent> {
        let prob = self.vad.frame(chunk) as f64;
        self.step(prob, chunk.len() as i64)
    }

    /// 状态机步进：`prob` 为当前窗口语音概率（模型已求出），`window` 为窗口样本数。
    /// 与模型调用解耦——差分模糊测试可直接注入概率序列（见
    /// `state_machine_fuzz_gen.py` 与 `tests/data/state_machine_fuzz.json`）。
    /// 语义逐行对应原版 `__call__` 的状态机部分（current_sample 先自增，再进状态机）。
    fn step(&mut self, prob: f64, window: i64) -> Option<VadEvent> {
        self.current_sample += window;

        // 滞回：语音恢复时清掉临时终点
        if prob >= self.threshold && self.temp_end != 0 {
            self.temp_end = 0;
        }
        if prob >= self.threshold && !self.triggered {
            self.triggered = true;
            let speech_start = (self.current_sample as f64
                - self.speech_pad_samples
                - window as f64)
                .max(0.0);
            return Some(VadEvent::Start(speech_start as i64));
        }
        if prob < self.threshold - 0.15 && self.triggered {
            if self.temp_end == 0 {
                self.temp_end = self.current_sample;
            }
            if ((self.current_sample - self.temp_end) as f64) < self.min_silence_samples {
                return None;
            }
            let speech_end =
                self.temp_end as f64 + self.speech_pad_samples - window as f64;
            self.temp_end = 0;
            self.triggered = false;
            return Some(VadEvent::End(speech_end as i64));
        }
        None
    }
}

#[cfg(test)]
mod state_machine_fuzz {
    //! 与 Python 原版的差分模糊测试：语料 `tests/data/state_machine_fuzz.json`
    //! 由 `state_machine_fuzz_gen.py` 生成（期望输出即原版 VADIterator + FakeModel
    //! 注入概率的结果），回放要求逐位一致。覆盖 threshold/neg_threshold 的 f32
    //! 邻域、min_silence 边界、pad 0/30/77ms、窗口 512/256/100、末窗非整等分支。

    use super::{VadEvent, VadIterator};
    use crate::{CFG_16K, Weights};

    // 语料不入库（.gitignore: tests/data/），本地由 state_machine_fuzz_gen.py 生成；
    // 缺失时跳过而非编译失败——语料属于本地测试数据，不是构建依赖。
    fn corpus() -> Option<String> {
        std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/state_machine_fuzz.json"
        ))
        .ok()
    }

    #[test]
    fn iterator_replay_matches_python_bit_exact() {
        let Some(corpus) = corpus() else {
            eprintln!("skip: 语料缺失——用 state_machine_fuzz_gen.py 重新生成后可回放");
            return;
        };
        let corpus: serde_json::Value = serde_json::from_str(&corpus).unwrap();
        let cases = corpus["iterator"].as_array().unwrap();
        assert!(cases.len() >= 200, "corpus too small: {}", cases.len());
        let mut events = 0usize;
        for case in cases {
            let probs: Vec<f32> = case["probs_bits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|b| f32::from_bits(b.as_u64().unwrap() as u32))
                .collect();
            let windows: Vec<i64> = case["windows"]
                .as_array()
                .unwrap()
                .iter()
                .map(|w| w.as_i64().unwrap())
                .collect();
            assert_eq!(probs.len(), windows.len(), "corpus case inconsistent");
            let mut it = VadIterator::new_with(
                Weights::embedded_16k().unwrap(),
                CFG_16K,
                case["threshold"].as_f64().unwrap(),
                case["min_silence_duration_ms"].as_f64().unwrap(),
                case["speech_pad_ms"].as_f64().unwrap(),
            )
            .unwrap();
            let mut got: Vec<(i64, i64)> = Vec::new();
            for (i, w) in windows.iter().enumerate() {
                if let Some(ev) = it.step(probs[i] as f64, *w) {
                    match ev {
                        VadEvent::Start(s) => got.push((0, s)),
                        VadEvent::End(e) => got.push((1, e)),
                    }
                }
            }
            let want: Vec<(i64, i64)> = case["expected"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| {
                    let pair = d.as_array().unwrap();
                    (pair[0].as_i64().unwrap(), pair[1].as_i64().unwrap())
                })
                .collect();
            events += want.len();
            assert_eq!(got, want, "case {}", case["name"].as_str().unwrap());
        }
        assert!(events >= 2000, "coverage too thin: {events} events");
    }
}
