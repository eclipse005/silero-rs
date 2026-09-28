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
        let window = chunk.len() as i64;
        self.current_sample += window;

        let speech_prob = self.vad.frame(chunk) as f64;

        // 滞回：语音恢复时清掉临时终点
        if speech_prob >= self.threshold && self.temp_end != 0 {
            self.temp_end = 0;
        }
        if speech_prob >= self.threshold && !self.triggered {
            self.triggered = true;
            let speech_start = (self.current_sample as f64
                - self.speech_pad_samples
                - window as f64)
                .max(0.0);
            return Some(VadEvent::Start(speech_start as i64));
        }
        if speech_prob < self.threshold - 0.15 && self.triggered {
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
