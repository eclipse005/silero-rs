"""Phase 0: 原版基线 + 金标准 dump（供 Rust/wgpu 移植对照）。

三件事：
1. 跑通原版 TorchScript JIT 模型（默认路径，非 ONNX）
2. 基线 RTFx：CPU 1线程（官方姿势）/ CPU 4线程 / CUDA 逐帧同步 / CUDA 吞吐上限
3. 金标准 dump：CPU 单线程逐帧概率 + 时间戳 → wgpu/ref/*.npz（Rust 门禁用）

测法（移植手册 §3）：先整段 warmup，再跑 5 次取全部数字；
RTFx = 音频时长 / 墙钟。CUDA 数值不作为金标准（与 CPU 不逐位一致）。

用法：
    python wgpu/phase0_baseline.py            # 基准 + dump 全跑
    python wgpu/phase0_baseline.py --bench    # 只跑基准
    python wgpu/phase0_baseline.py --dump     # 只 dump 金标准
"""

import argparse
import json
import os
import sys
import time
import wave

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch  # noqa: E402
from silero_vad.utils_vad import (  # noqa: E402
    get_speech_timestamps_from_probs,
    init_jit_model,
)

JIT_PATH = os.path.join(ROOT, "src", "silero_vad", "data", "silero_vad.jit")
REF_DIR = os.path.join(ROOT, "wgpu", "ref")

# (name, path, sr) —— 8k 分支不在移植范围内，仅 16k
# datasets/vad_gate/ 来自 D:\mini_asr_data（FLEURS/MLS test 切分，16k mono float32 wav）
# 命名规则：<语种>_<short|long>（每组最短+最长），mls 取中位附近一条
_GATE_DIR = os.path.join(ROOT, "datasets", "vad_gate")
FIXTURES = [
    ("test60s", os.path.join(ROOT, "tests", "data", "test.wav"), 16000),
    ("aepyx169s", os.path.join(ROOT, "examples", "c++", "aepyx.wav"), 16000),
    ("zh_short", os.path.join(_GATE_DIR, "fleurs_cmn_hans_cn_17381936933530140952.wav"), 16000),
    ("zh_long", os.path.join(_GATE_DIR, "fleurs_cmn_hans_cn_1626393343718951419.wav"), 16000),
    ("en_short", os.path.join(_GATE_DIR, "fleurs_en_us_2254021119996292300.wav"), 16000),
    ("en_long", os.path.join(_GATE_DIR, "fleurs_en_us_17498257810809617374.wav"), 16000),
    ("yue_short", os.path.join(_GATE_DIR, "fleurs_yue_hant_hk_15798427843044240865.wav"), 16000),
    ("yue_long", os.path.join(_GATE_DIR, "fleurs_yue_hant_hk_1120333469782447474.wav"), 16000),
    ("ja_short", os.path.join(_GATE_DIR, "fleurs_ja_jp_3678765754439539021.wav"), 16000),
    ("ja_long", os.path.join(_GATE_DIR, "fleurs_ja_jp_3010933638415181533.wav"), 16000),
    ("mls_es", os.path.join(_GATE_DIR, "mls_spanish_3503_2162_000014.wav"), 16000),
    ("mls_fr", os.path.join(_GATE_DIR, "mls_french_12977_10625_000017.wav"), 16000),
]

N_RUNS = 5
WARMUP_RUNS = 1  # CUDA 首跑含 init，必须 warmup（移植手册 §3.1）


def read_wav(path):
    """单声道 wav → f32 [-1, 1]。

    PCM16 走 stdlib wave（与既有 golden 的量化路径逐位一致）；float32 WAV
    （mini_asr_data 的 FLEURS/MLS）stdlib 读不了，走 soundfile。
    """
    import soundfile as sf

    try:
        with wave.open(path, "rb") as w:
            width, sr = w.getsampwidth(), w.getframerate()
            nch = w.getnchannels()
    except wave.Error:
        # float32 WAV（format 3）stdlib 不识别，直接走 soundfile
        data, sr = sf.read(path, dtype="float32")
        assert data.ndim == 1, f"expect mono, got shape {data.shape}"
        return data, sr
    assert nch == 1, f"expect mono, got {nch} ch"
    if width == 2:
        with wave.open(path, "rb") as w:
            raw = w.readframes(w.getnframes())
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr
    data, sr = sf.read(path, dtype="float32")
    assert data.ndim == 1, f"expect mono, got shape {data.shape}"
    return data, sr


def stream_probs(model, wav, sr, device, sync_each_frame):
    """按 get_speech_timestamps 的语义逐帧流式。

    device=cpu: chunk 在主机上（官方姿势）。
    device=cuda + sync_each_frame: 每帧 .item() 强制同步 —— 真实流式语义。
    device=cuda + not sync_each_frame: 整段 wav 一次性上卡，帧内切片，
        概率留在 GPU 最后一次回读 —— 纯吞吐上限（天花板参考臂）。
    """
    window = 512 if sr == 16000 else 256
    wav_dev = wav.to(device)
    if not sync_each_frame and device != "cpu":
        full = wav_dev  # 已整段上卡
    model.reset_states()
    probs = []
    n = wav_dev.numel()
    for start in range(0, n, window):
        if not sync_each_frame and device != "cpu":
            chunk = full[start : start + window]
            if chunk.numel() < window:
                chunk = torch.nn.functional.pad(chunk, (0, window - chunk.numel()))
        else:
            chunk_np = wav[start : start + window]
            if chunk_np.numel() < window:
                chunk_np = torch.nn.functional.pad(
                    chunk_np, (0, window - chunk_np.numel())
                )
            chunk = chunk_np.to(device)
        p = model(chunk, sr)
        if sync_each_frame:
            probs.append(p.item())
        else:
            probs.append(p)
    if device != "cpu" and not sync_each_frame:
        probs = torch.stack(probs).cpu().tolist()
    else:
        probs = [float(p) for p in probs]
    return probs


def bench_arm(model, wav_np, sr, device, sync_each_frame, n_threads=None):
    if n_threads is not None:
        torch.set_num_threads(n_threads)
    wav_t = torch.from_numpy(wav_np)
    dur = wav_np.size / sr

    for _ in range(WARMUP_RUNS):
        stream_probs(model, wav_t, sr, device, sync_each_frame)
    walls = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        stream_probs(model, wav_t, sr, device, sync_each_frame)
        walls.append(time.perf_counter() - t0)

    return {
        "walls_s": [round(w, 4) for w in walls],
        "median_s": round(sorted(walls)[len(walls) // 2], 4),
        "min_s": round(min(walls), 4),
        "rtfx_median": round(dur / sorted(walls)[len(walls) // 2], 2),
        "rtfx_best": round(dur / min(walls), 2),
        "frames": (wav_np.size + 511) // 512,
    }


def dump_golden(model, wav_np, sr, name):
    """CPU 单线程金标准：wav f32 + 逐帧概率 + 默认参数时间戳。"""
    torch.set_num_threads(1)
    wav_t = torch.from_numpy(wav_np)
    model.reset_states()
    probs = stream_probs(model, wav_t, sr, "cpu", sync_each_frame=True)
    ts = get_speech_timestamps_from_probs(
        probs, sampling_rate=sr, return_seconds=False
    )
    # return_seconds=False 返回 [{'start': s, 'end': e}, ...]
    ts_pairs = [(int(d["start"]), int(d["end"])) for d in ts]
    out = os.path.join(REF_DIR, f"golden_{name}.npz")
    np.savez_compressed(
        out,
        wav=wav_np.astype(np.float32),
        probs=np.asarray(probs, dtype=np.float32),
        sr=np.int64(sr),
        ts=np.asarray(ts_pairs, dtype=np.int64).reshape(-1, 2),
    )
    speech_total = sum(e - s for s, e in ts_pairs)
    print(
        f"  golden_{name}.npz: {len(probs)} frames, "
        f"speech {speech_total / sr:.1f}s / {wav_np.size / sr:.1f}s, "
        f"probs[min={min(probs):.4f} max={max(probs):.4f}]"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--dump", action="store_true")
    args = ap.parse_args()
    do_bench = args.bench or not (args.bench or args.dump)
    do_dump = args.dump or not (args.bench or args.dump)

    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    wavs = {}
    for name, path, sr in FIXTURES:
        wav_np, sr_actual = read_wav(path)
        assert sr_actual == sr, f"{name}: expect {sr}, got {sr_actual}"
        wavs[name] = (wav_np, sr)
        print(f"fixture {name}: {path}  {wav_np.size/sr:.2f}s @ {sr}Hz")

    os.makedirs(REF_DIR, exist_ok=True)
    results = {"torch": torch.__version__, "gpu": None, "fixtures": {}}
    if torch.cuda.is_available():
        results["gpu"] = torch.cuda.get_device_name(0)

    for name, (wav_np, sr) in wavs.items():
        print(f"\n=== {name} ===")
        results["fixtures"][name] = {}

        if do_bench:
            arms = []

            m_cpu = init_jit_model(JIT_PATH, device="cpu")
            arms.append(("cpu_1thread", bench_arm(m_cpu, wav_np, sr, "cpu", True, 1)))
            arms.append(("cpu_4thread", bench_arm(m_cpu, wav_np, sr, "cpu", True, 4)))

            if torch.cuda.is_available():
                m_gpu = init_jit_model(JIT_PATH, device="cuda")
                arms.append(
                    ("cuda_sync_per_frame", bench_arm(m_gpu, wav_np, sr, "cuda", True))
                )
                arms.append(
                    (
                        "cuda_throughput_ceiling",
                        bench_arm(m_gpu, wav_np, sr, "cuda", False),
                    )
                )

            print(f"  {'arm':<24}{'median':>10}{'best':>10}{'RTFx(med)':>12}{'RTFx(best)':>12}")
            for arm_name, r in arms:
                results["fixtures"][name][arm_name] = r
                print(
                    f"  {arm_name:<24}{r['median_s']:>9.4f}s{r['min_s']:>9.4f}s"
                    f"{r['rtfx_median']:>12.2f}{r['rtfx_best']:>12.2f}"
                )

        if do_dump:
            m_cpu = init_jit_model(JIT_PATH, device="cpu")
            dump_golden(m_cpu, wav_np, sr, name)

    if do_bench:
        out_json = os.path.join(REF_DIR, "baseline_results.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nresults -> {out_json}")


if __name__ == "__main__":
    main()
