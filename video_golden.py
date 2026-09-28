"""原版（torch jit）CPU + CUDA 逐帧概率 + 时间戳 golden（视频库音频对齐测试）。

对 wgpu/ref/video/vNN.f32（16k mono f32）：
  vNN.probs.f32       原版 CPU（1 线程官方姿势）逐帧概率
  vNN.probs_cuda.f32  原版 CUDA（P104，逐帧同步）逐帧概率
  vNN.ts.i64          CPU 概率 → 时间戳（get_speech_timestamps_from_probs，样本对）
  vNN.ts_cuda.i64     CUDA 概率 → 时间戳
  video_golden.json   RTFx（warmup 1 + 2 计时）
"""

import glob
import json
import os
import sys
import time

import numpy as np
import torch

# 上游 Python 仓库（含 src/silero_vad 与 models/），可用环境变量覆盖
ROOT = os.environ.get("SILERO_PY_REPO", r"D:\silero-vad")
sys.path.insert(0, os.path.join(ROOT, "src"))
VDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref", "video")
SR = 16000
WINDOW = 512


def stream_jit(sess, wav, device):
    t = torch.from_numpy(wav).to(device)
    sess.reset_states()
    probs = []
    n = t.numel()
    for s in range(0, n, WINDOW):
        chunk = t[s : s + WINDOW]
        if chunk.numel() < WINDOW:
            chunk = torch.nn.functional.pad(chunk, (0, WINDOW - chunk.numel()))
        probs.append(float(sess(chunk, SR).item()))
    return probs


def ts_from_probs(probs):
    from silero_vad.utils_vad import get_speech_timestamps_from_probs

    ts = get_speech_timestamps_from_probs(probs, sampling_rate=SR, return_seconds=False)
    return [(int(d["start"]), int(d["end"])) for d in ts]


def vadit_events(model, wav, device):
    """原版 VADIterator（增量状态机）逐窗口事件：0=start, 1=end（样本坐标）。"""
    from silero_vad.utils_vad import VADIterator

    it = VADIterator(model, threshold=0.5, sampling_rate=SR,
                     min_silence_duration_ms=100, speech_pad_ms=30)
    events = []
    for s in range(0, wav.size, WINDOW):
        chunk = wav[s : s + WINDOW]
        if chunk.size < WINDOW:
            chunk = np.pad(chunk, (0, WINDOW - chunk.size))
        r = it(torch.from_numpy(chunk).to(device))
        if r is not None:
            kind = 0 if "start" in r else 1
            events.append((kind, int(r["start"] if kind == 0 else r["end"])))
    return events


def main():
    torch.set_num_threads(1)
    jit_cpu = torch.jit.load(os.path.join(ROOT, "models", "silero_vad.jit"), map_location="cpu")
    jit_cuda = torch.jit.load(os.path.join(ROOT, "models", "silero_vad.jit"), map_location="cuda")

    out_json = {}
    for p in sorted(glob.glob(os.path.join(VDIR, "v*.f32"))):
        base = p[: -len(".f32")]
        if base.endswith(".probs") or base.endswith(".probs_cuda"):
            continue
        name = os.path.basename(base)
        wav = np.fromfile(p, dtype=np.float32)
        dur = wav.size / SR
        print(f"=== {name} ({dur:.0f}s, {wav.size} samples) ===")

        # CPU
        t0 = time.perf_counter()
        probs_cpu = stream_jit(jit_cpu, wav, "cpu")
        cpu_first = time.perf_counter() - t0
        walls = []
        for _ in range(2):
            t0 = time.perf_counter()
            stream_jit(jit_cpu, wav, "cpu")
            walls.append(time.perf_counter() - t0)
        cpu_rtfx = dur / sorted(walls)[0]
        ts_cpu = ts_from_probs(probs_cpu)

        # CUDA
        t0 = time.perf_counter()
        probs_cuda = stream_jit(jit_cuda, wav, "cuda")
        torch.cuda.synchronize()
        cuda_first = time.perf_counter() - t0
        walls = []
        for _ in range(2):
            t0 = time.perf_counter()
            stream_jit(jit_cuda, wav, "cuda")
            torch.cuda.synchronize()
            walls.append(time.perf_counter() - t0)
        cuda_rtfx = dur / sorted(walls)[0]
        ts_cuda = ts_from_probs(probs_cuda)

        probs_a = np.asarray(probs_cpu, dtype=np.float32)
        probs_b = np.asarray(probs_cuda, dtype=np.float32)
        max_diff = float(np.abs(probs_a - probs_b).max())
        np.asarray(probs_cpu, dtype=np.float32).tofile(base + ".probs.f32")
        np.asarray(probs_cuda, dtype=np.float32).tofile(base + ".probs_cuda.f32")
        np.asarray(ts_cpu, dtype=np.int64).tofile(base + ".ts.i64")
        np.asarray(ts_cuda, dtype=np.int64).tofile(base + ".ts_cuda.i64")

        # VADIterator 增量事件（CPU 权威；kind/sample 交错 i64）
        ev_cpu = vadit_events(jit_cpu, wav, "cpu")
        np.asarray([v for e in ev_cpu for v in e], dtype=np.int64).tofile(base + ".vadit.i64")

        ts_eq = ts_cpu == ts_cuda
        out_json[name] = {
            "dur_s": round(dur, 2),
            "frames": len(probs_cpu),
            "cpu_rtfx": round(cpu_rtfx, 1),
            "cuda_rtfx": round(cuda_rtfx, 1),
            "cpu_vs_cuda_maxdiff": max_diff,
            "cpu_vs_cuda_ts_equal": ts_eq,
            "cpu_ts_segments": len(ts_cpu),
            "cpu_vadit_events": len(ev_cpu),
        }
        print(
            f"  frames={len(probs_cpu)} cpu_rtfx={cpu_rtfx:.1f} cuda_rtfx={cuda_rtfx:.1f} "
            f"cpu-vs-cuda maxdiff={max_diff:.2e} ts_equal={ts_eq} ({len(ts_cpu)} segments, "
            f"{len(ev_cpu)} vadit events)"
        )

    with open(os.path.join(VDIR, "video_golden.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)
    print("-> video_golden.json")


if __name__ == "__main__":
    main()
