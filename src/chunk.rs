//! Chunk planning for transcription: speech runs in, silence midpoints out.
//!
//! A recogniser works best on a few seconds to a minute of audio, but a recorder
//! hands it one long file. The natural cut is the **middle of a silence**: both
//! neighbouring speech runs keep their onsets and offsets, and the cut lands
//! where there is nothing to clip.
//!
//! [`chunk_cuts`] therefore turns VAD segments into cut times:
//!
//! * between two speech runs, cut at the midpoint of the silence;
//! * a run longer than `max_seconds` is split at the last silence midpoint that
//!   still fits inside the window (looking *back* from the limit), so a chunk
//!   never ends mid-word if there was any pause to use;
//! * a run with no usable pause inside the window is cut hard at the limit — the
//!   recogniser will see a clipped word there, which is the honest fallback
//!   rather than an unbounded chunk.

/// A speech run in seconds, as the VAD reports it.
pub type Segment = (f32, f32);

/// Plans the cut times (absolute seconds) for a whole recording.
///
/// `segments` must be sorted and non-overlapping; the returned times are strictly
/// increasing and lie inside `0..end`, so the caller can always add the file
/// boundaries itself.
pub fn chunk_cuts(segments: &[Segment], max_seconds: f32) -> Vec<f32> {
    if !max_seconds.is_finite() || max_seconds <= 0.0 {
        return Vec::new();
    }
    let max = max_seconds;
    let mut cuts = Vec::new();
    let mut start = 0.0f32;
    let mut i = 0usize;
    while i < segments.len() {
        // Everything that fits in this window stays: walk forward.
        while i < segments.len() && segments[i].1 <= start + max {
            i += 1;
        }
        if i >= segments.len() {
            break;
        }
        // `segments[i]` spills past the limit: cut at the last silence midpoint
        // that still fits, looking back from the limit.
        let limit = start + max;
        let mut best = None;
        for j in (1..=i).rev() {
            let (prev_end, next_start) = (segments[j - 1].1, segments[j].0);
            let mid = 0.5 * (prev_end + next_start);
            if next_start > prev_end && mid > start && mid <= limit {
                best = Some(mid);
                break;
            }
        }
        let cut = best.unwrap_or(limit);
        cuts.push(cut);
        start = cut;
        if cut >= segments[i].1 {
            i += 1;
        }
    }
    cuts.dedup_by(|a, b| (*a - *b).abs() < 1e-3);
    cuts
}

/// Midpoint of every silence between two runs (no length limit applied).
pub fn silence_midpoints(segments: &[Segment]) -> Vec<f32> {
    segments
        .windows(2)
        .filter(|w| w[1].0 > w[0].1)
        .map(|w| 0.5 * (w[0].1 + w[1].0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuts_at_silence_midpoints_without_a_limit() {
        let segs = [(0.0, 2.0), (4.0, 6.0), (9.0, 10.0)];
        assert_eq!(silence_midpoints(&segs), vec![3.0, 7.5]);
        assert!(chunk_cuts(&segs, f32::INFINITY).is_empty());
    }

    #[test]
    fn caps_a_long_recording_at_the_last_pause() {
        // Runs every 2 s with 1 s silences: the limit lands at 10 s, and the last
        // midpoint that fits is 9.5.
        let segs = [
            (0.0, 2.0),
            (3.0, 5.0),
            (6.0, 8.0),
            (9.0, 11.0),
            (12.0, 14.0),
        ];
        let cuts = chunk_cuts(&segs, 10.0);
        assert_eq!(cuts, vec![8.5]);
    }

    #[test]
    fn falls_back_to_a_hard_cut_when_there_is_no_pause() {
        let segs = [(0.0, 45.0)];
        assert_eq!(chunk_cuts(&segs, 20.0), vec![20.0, 40.0]);
    }

    #[test]
    fn keeps_splitting_a_run_that_spans_several_windows() {
        let segs = [(0.0, 70.0)];
        assert_eq!(chunk_cuts(&segs, 30.0), vec![30.0, 60.0]);
    }

    #[test]
    fn short_recordings_need_no_cuts() {
        assert!(chunk_cuts(&[(0.0, 5.0), (6.0, 9.0)], 60.0).is_empty());
    }
}
