//! Browser bindings for the Silero VAD port.
//!
//! The 16 kHz weights are compiled in (1.2 MB), so the page downloads nothing:
//!
//! ```js
//! import init, { SileroVad } from "./silero_vad_wasm.js";
//! await init();
//! const vad = SileroVad.create();
//!
//! // 16 kHz mono f32 samples in, cut times out:
//! const cuts = JSON.parse(vad.cuts(samples, 60));   // [] when nothing to cut
//! ```
//!
//! `cuts()` is the silence-midpoint planner from the parent crate: between two
//! speech runs it cuts in the middle of the silence, and a run longer than
//! `maxSeconds` is split at the last pause inside the window (or hard-cut when the
//! speech has no pause at all).

use silero_vad_wgpu::chunk;
use silero_vad_wgpu::timestamps::TsParams;
use silero_vad_wgpu::{SileroVad as Vad, Weights, CFG_16K};
use wasm_bindgen::prelude::*;

/// A VAD instance. Cheap to create; keep one per page.
#[wasm_bindgen]
pub struct SileroVad {
    vad: Vad,
    params: TsParams,
}

#[wasm_bindgen]
impl SileroVad {
    /// Builds the detector with the embedded 16 kHz weights.
    #[wasm_bindgen(js_name = create)]
    pub fn create() -> Result<SileroVad, JsError> {
        console_error_panic_hook::set_once();
        let weights =
            Weights::embedded_16k().map_err(|e| JsError::new(&format!("weights: {e}")))?;
        let vad = Vad::new(weights, CFG_16K).map_err(|e| JsError::new(&e))?;
        Ok(SileroVad {
            vad,
            params: TsParams::default(),
        })
    }

    /// Speech runs as JSON `[[start, end], ...]` in seconds.
    pub fn speech(&mut self, samples: &[f32]) -> Result<String, JsError> {
        let segments = self.segments(samples, f64::INFINITY);
        let items: Vec<String> = segments
            .iter()
            .map(|(s, e)| format!("[{s:.3},{e:.3}]"))
            .collect();
        Ok(format!("[{}]", items.join(",")))
    }

    /// Cut times in seconds — the silence midpoints a chunker should split on.
    ///
    /// `max_seconds` (default 60) caps a chunk; a speech run that does not fit is
    /// cut at the last pause before the limit, as described in the module docs.
    pub fn cuts(
        &mut self,
        samples: &[f32],
        max_seconds: Option<f64>,
    ) -> Result<String, JsError> {
        let max = max_seconds.unwrap_or(60.0);
        let segments = self.segments(samples, max.max(0.1));
        let cuts = chunk::chunk_cuts(&segments, max as f32);
        let items: Vec<String> = cuts.iter().map(|c| format!("{c:.3}")).collect();
        Ok(format!("[{}]", items.join(",")))
    }

    /// Everything the page needs in one call: both lists plus the speech duration.
    pub fn analyse(&mut self, samples: &[f32], max_seconds: Option<f64>) -> String {
        let max = max_seconds.unwrap_or(60.0);
        let segments = self.segments(samples, max.max(0.1));
        let cuts = chunk::chunk_cuts(&segments, max as f32);
        let segs: Vec<String> = segments
            .iter()
            .map(|(s, e)| format!("[{s:.3},{e:.3}]"))
            .collect();
        let cs: Vec<String> = cuts.iter().map(|c| format!("{c:.3}")).collect();
        format!(
            "{{\"duration\":{:.3},\"speech\":[{}],\"cuts\":[{}]}}",
            samples.len() as f64 / CFG_16K.sr as f64,
            segs.join(","),
            cs.join(",")
        )
    }
}

impl SileroVad {
    /// Runs the frame state machine and converts the result to seconds.
    fn segments(&mut self, samples: &[f32], max_speech: f64) -> Vec<(f32, f32)> {
        let mut params = self.params.clone();
        params.max_speech_duration_s = max_speech;
        let sr = CFG_16K.sr as f64;
        silero_vad_wgpu::get_speech_timestamps(&mut self.vad, samples, &params)
            .into_iter()
            .map(|(s, e)| ((s as f64 / sr) as f32, (e as f64 / sr) as f32))
            .collect()
    }
}
