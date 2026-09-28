# silero-rs

[Silero VAD](https://github.com/snakers4/silero-vad) (v6.2.3 model) ported to Rust — dependency-free hand-written kernels, host CPU (SIMD) and GPU (wgpu: Vulkan/Metal/DX12) backends, **speech-timestamp identical to the Python original**.

Use it as a library (official 16k weights embedded) or as a cross-platform CLI.

## Highlights

- **Aligned with the original** — timestamps are bit-exact vs the Python implementation (gate-verified across 10 model variants and real-media four-way checks: original CPU/CUDA vs Rust CPU/wgpu); per-frame probabilities stay within the f32 feedback-noise class.
- **No ML runtime** — no libtorch, no ONNX runtime, no Python. STFT-as-conv encoder → conv stack → LSTM → head, all hand-written.
- **Fast** — RTFx (× realtime, 16k, single thread): CPU ≈ 500–900 (AVX2+FMA), GPU batched ≈ 800–900, vs ≈ 90–120 for the original Python.
- **Cross-platform** — x86-64 (runtime SIMD detection, scalar fallback) and GPU via wgpu; backend selectable with `SILERO_BACKEND`.

## Library

```rust
use silero_vad_wgpu::{get_speech_timestamps, collect_chunks, SileroVad, CFG_16K};
use silero_vad_wgpu::timestamps::TsParams;

let mut vad = SileroVad::new(silero_vad_wgpu::Weights::embedded_16k()?, CFG_16K)?;
let ts = get_speech_timestamps(&mut vad, &wav_16k_mono, &TsParams::default()); // [(start, end)] samples
let speech = collect_chunks(&ts, &wav_16k_mono);

// streaming (≈ Python VADIterator)
use silero_vad_wgpu::vad_iterator::VadIterator;
let mut it = VadIterator::new(silero_vad_wgpu::Weights::embedded_16k()?, CFG_16K)?;
for chunk in windows_of_512(&stream) {
    if let Some(ev) = it.push(&chunk) { println!("{ev:?}"); } // Start(i64) / End(i64)
}
```

## CLI

```bash
cargo run --release --bin silero-vad -- --seconds audio.wav          # CPU
cargo run --release --features gpu --bin silero-vad -- --backend gpu --seconds audio.wav
```

```json
[{"start": 8.6, "end": 9.5}, {"start": 9.7, "end": 11.7}, ...]
```

Options: `--threshold 0.5`, `--model weights.safetensors` (8k/custom), `--seconds`, `--time-resolution 1`.
Input: 8k/16k mono WAV (GPU backend is 16k-only). Resample other rates first.

## Correctness

`cargo test` runs an API smoke test on the embedded weights. The full gate compares frozen golden
outputs against 10 model variants (G1: probabilities within the documented f32 noise bound,
G2: timestamps exactly equal) and a real-media four-way alignment — see `src/bin/gate.rs`,
`src/bin/align.rs` and the Python tooling (`export_for_rust.py`, `variant_baselines.py`,
`video_golden.py`). Regenerating goldens requires the upstream Python repo
(`SILERO_PY_REPO`, default `D:\silero-vad`).

## License

MIT (same as upstream Silero VAD).
