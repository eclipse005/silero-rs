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

`cargo test` runs an API smoke test on the embedded weights plus `tests/math_parity.rs`,
which exercises the transcendental layer on a **real GPU**. The full gate compares frozen
golden outputs against 10 model variants (G1: probabilities within the documented f32 noise bound,
G2: timestamps exactly equal) and a real-media four-way alignment — see `src/bin/gate.rs`,
`src/bin/align.rs` and the Python tooling (`export_for_rust.py`, `variant_baselines.py`,
`video_golden.py`). Regenerating goldens requires the upstream Python repo
(`SILERO_PY_REPO`, default `D:\silero-vad`).

### Numerics: why there is a hand-written transcendental layer

Silero's LSTM is an **exponentially unstable recurrence**. A 1 ULP per-step difference is
amplified to O(1) probability error over long audio, which flips speech-timestamp
boundaries. Measured on the 1525 s `v01` fixture, using the GPU's own `exp`/`tanh`:

| step | result |
|---|---|
| frame 0 | 1 ULP (NVIDIA vs Intel) |
| frame 314 | 1e-6 |
| frame 12789 | 0.75 |
| timestamps | 567 → 555 segments (vs golden) |

The cause is that **hardware transcendentals are not portable**: on the same shader Intel
and NVIDIA disagree by 1–2 ULP on `exp` (94/275 sampled inputs), `tanh` (110/275) and
`sigmoid`. A probe also confirmed denormal flushing is *not* the cause (zero denormal
outputs) and that `+ - * /` are bit-identical across vendors with no FMA contraction.

So `src/wgsl_math.wgsl` (GPU) and `src/transcendental.rs` (host) implement `exp`, `expm1`,
`sigmoid` and `tanh` from `+ - * /` only — Cody-Waite range reduction plus
`exp(r) = 1 + r + r²·Q(r)`. `tanh` uses `-expm1(-2x)/(expm1(-2x)+2)` rather than
`2·sigmoid(2x)-1`, which loses ~64 ULP at `x ≈ 0` — precisely where LSTM gates sit.

What this buys, on the 8-fixture four-way align:

| | frames > 1e-5 | timestamps |
|---|---|---|
| NVIDIA, hardware `exp` | 15 | 567 = golden |
| Intel, hardware `exp` | 22695 | 555 ≠ golden |
| Intel, software layer | 7 | 567 = golden |
| NVIDIA, software layer | 14 | 567 = golden |

Bit-exactness *across vendors* is not achievable and is not claimed: WGSL has no `precise`
qualifier and naga emits no SPIR-V `NoContraction`, so drivers contract `a*b+c` into FMA
while Rust/LLVM does not. `tests/math_parity.rs` pins host-vs-GPU to ≤ 4 ULP (measured) and
guards against *algorithm* drift; the real guarantee is the end-to-end align on long audio.

### Selecting a GPU

`SILERO_ADAPTER=<substring>` picks an adapter by name, driver or device type — the only
reliable way to target an iGPU on a dual-GPU machine, since `power_preference` only ever
selects "high performance". `SILERO_ADAPTER_INFO=1` prints the adapter inventory with the
limits and features that matter.

```bash
SILERO_BACKEND=vulkan SILERO_ADAPTER=Intel  SILERO_ADAPTER_INFO=1 cargo run --release --features gpu --bin align
SILERO_BACKEND=vulkan SILERO_ADAPTER=NVIDIA cargo run --release --features gpu --bin align
```

**Cross-vendor changes must be validated on every adapter, and on long audio**: the
10-variant gate only uses clips up to 60 s, which is not long enough to expose recurrence
instability. `src/bin/trace.rs` localises divergence for a given fixture
(`trace video v01`), and `--dump` exports the probability stream for cross-adapter diffing.

### Debugging

```bash
cargo run --release --features gpu --bin trace -- gate test60s
cargo run --release --features gpu --bin trace -- video v01 --framewise   # per-frame GpuVad path
```

## License

MIT (same as upstream Silero VAD).
