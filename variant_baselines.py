"""多变体金标准 + Python 原版基线（对齐目标与 RTFx 对照）。

每个变体用"原版自己的驱动方式"跑 12 个 16k 夹具（8k 变体用 aepyx_8k.wav）：
  jit16k / jit8k        torch jit（内部 context 管理）
  op16_16k / op16_8k    OnnxWrapper 语义（外部 context + state 每帧进出 + sr）
  op15 / op18 / half / openvino   同 op16 语义（half/openvino 无 sr 输入）
  sequence              SileroVADSequence 语义（≤512 帧/次调用，h/c 状态）
  st16k                 tinygrad 架构（torch 重建）+ models/silero_vad_16k.safetensors

输出：
  wgpu/ref/gate/<variant>/<fixture>.probs.f32 / .ts.i64（Rust 门禁输入）
  wgpu/ref/baseline_results.json 增补 per-variant RTFx
"""

import glob
import json
import os
import sys
import time
import wave

import numpy as np

# 上游 Python 仓库（含 src/silero_vad、models/、examples/），可用环境变量覆盖
ROOT = os.environ.get("SILERO_PY_REPO", r"D:\silero-vad")
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch  # noqa: E402
from silero_vad.utils_vad import get_speech_timestamps_from_probs  # noqa: E402

M = os.path.join(ROOT, "models")
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref")
N_RUNS, WARMUP = 5, 1


def read_wav(path):
    import soundfile as sf

    with wave.open(path, "rb") as w:
        width, sr, nch = w.getsampwidth(), w.getframerate(), w.getnchannels()
    assert nch == 1
    if width == 2:
        with wave.open(path, "rb") as w:
            raw = w.readframes(w.getnframes())
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr
    data, sr = sf.read(path, dtype="float32")
    assert data.ndim == 1
    return data, sr


# ---------- 驱动 ----------

def stream_jit(sess, wav, sr):
    """torch jit：内部 context/state 管理。sess = jit module。"""
    window = 512 if sr == 16000 else 256
    t = torch.from_numpy(wav)
    sess.reset_states()
    probs = []
    n = t.numel()
    for s in range(0, n, window):
        chunk = t[s : s + window]
        if chunk.numel() < window:
            chunk = torch.nn.functional.pad(chunk, (0, window - chunk.numel()))
        probs.append(float(sess(chunk, sr).item()))
    return probs


class OrtStream:
    """OnnxWrapper 语义：外部 context + state 每帧进出（sr 输入按需）。"""

    def __init__(self, path, sr):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        self.names = {i.name for i in self.sess.get_inputs()}
        self.sr = sr
        self.window = 512 if sr == 16000 else 256
        self.ctx_n = 64 if sr == 16000 else 32
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.ctx = np.zeros(self.ctx_n, dtype=np.float32)

    def frame(self, chunk):
        x = np.concatenate([self.ctx, chunk])[None].astype(np.float32)
        feeds = {"input": x, "state": self.state}
        if "sr" in self.names:
            feeds["sr"] = np.array(self.sr, dtype=np.int64)
        out, self.state = self.sess.run(None, feeds)
        self.ctx = x[0, -self.ctx_n :].copy()
        return float(out[0, 0])


def stream_ort(os_, wav):
    probs = []
    n = wav.size
    os_.reset()
    for s in range(0, n, os_.window):
        chunk = wav[s : s + os_.window]
        if chunk.size < os_.window:
            chunk = np.pad(chunk, (0, os_.window - chunk.size))
        probs.append(os_.frame(chunk))
    return probs


def stream_sequence(sess, wav, sr=16000, max_frames=512):
    """SileroVADSequence 语义（sequence_vad.py frame_blocks）。每次流式前重置 h/c。"""
    frame, ctx_n = 512, 64
    sess.h = np.zeros((1, 1, 128), dtype=np.float32)
    sess.c = np.zeros((1, 1, 128), dtype=np.float32)
    count = (wav.size + frame - 1) // frame
    prev_ctx = np.zeros(ctx_n, dtype=np.float32)
    probs = []
    for first in range(0, count, max_frames):
        bf = min(max_frames, count - first)
        samples = wav[first * frame : min(wav.size, (first + bf) * frame)]
        frames = np.zeros((bf, frame), dtype=np.float32)
        frames.reshape(-1)[: samples.size] = samples
        contexts = np.empty((bf, ctx_n), dtype=np.float32)
        contexts[0] = prev_ctx
        if bf > 1:
            contexts[1:] = frames[:-1, -ctx_n:]
        prev_ctx = frames[-1, -ctx_n:].copy()
        block = np.concatenate((contexts, frames), axis=1)
        p, hn, cn = sess.run(None, {"input": block, "h": sess.h, "c": sess.c})
        sess.h, sess.c = hn, cn
        probs.extend(float(v) for v in p)
    return probs


def make_sequence_sess():
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 1
    s = ort.InferenceSession(
        os.path.join(M, "silero_vad_16k_sequence.onnx"),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    s.h = np.zeros((1, 1, 128), dtype=np.float32)
    s.c = np.zeros((1, 1, 128), dtype=np.float32)
    return s


def make_st16k():
    """torch 重建 tinygrad 架构 + W_C 权重（外部 context/hc 状态）。"""
    from safetensors.torch import load_file

    st = load_file(os.path.join(M, "silero_vad_16k.safetensors"))
    st = {k: v.contiguous() for k, v in st.items()}

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stft_conv = torch.nn.Conv1d(1, 258, 256, stride=128, bias=False)
            self.conv1 = torch.nn.Conv1d(129, 128, 3, padding=1)
            self.conv2 = torch.nn.Conv1d(128, 64, 3, stride=2, padding=1)
            self.conv3 = torch.nn.Conv1d(64, 64, 3, stride=2, padding=1)
            self.conv4 = torch.nn.Conv1d(64, 128, 3, padding=1)
            self.final = torch.nn.Conv1d(128, 1, 1)
            with torch.no_grad():
                self.stft_conv.weight.copy_(st["stft_conv.weight"])
                self.conv1.weight.copy_(st["conv1.weight"]); self.conv1.bias.copy_(st["conv1.bias"])
                self.conv2.weight.copy_(st["conv2.weight"]); self.conv2.bias.copy_(st["conv2.bias"])
                self.conv3.weight.copy_(st["conv3.weight"]); self.conv3.bias.copy_(st["conv3.bias"])
                self.conv4.weight.copy_(st["conv4.weight"]); self.conv4.bias.copy_(st["conv4.bias"])
                self.final.weight.copy_(st["final_conv.weight"]); self.final.bias.copy_(st["final_conv.bias"])

        def forward(self, x576, h, c):
            # torch reflect pad 不支持 1D：先升 (1,576) 再 pad 最后一维
            x = torch.nn.functional.pad(x576[None], (0, 64), "reflect").unsqueeze(1)
            x = self.stft_conv(x)
            x = (x[:, :129] ** 2 + x[:, 129:] ** 2).sqrt()
            x = self.conv1(x).relu()
            x = self.conv2(x).relu()
            x = self.conv3(x).relu()
            x = self.conv4(x).relu().squeeze(-1)
            h2, c2 = torch.lstm_cell(
                x, (h, c), st["lstm_cell.weight_ih"], st["lstm_cell.weight_hh"],
                st["lstm_cell.bias_ih"], st["lstm_cell.bias_hh"],
            )
            p = self.final(h2.relu().unsqueeze(-1)).sigmoid()
            return p.squeeze(), h2, c2

    m = Tiny().eval()
    torch.set_num_threads(1)
    return m


def stream_st16k(m, wav):
    probs = []
    h = torch.zeros(1, 128)
    c = torch.zeros(1, 128)
    ctx = np.zeros(64, dtype=np.float32)
    n = wav.size
    for s in range(0, n, 512):
        chunk = wav[s : s + 512]
        if chunk.size < 512:
            chunk = np.pad(chunk, (0, 512 - chunk.size))
        x = np.concatenate([ctx, chunk])
        with torch.no_grad():
            p, h, c = m.forward(torch.from_numpy(x), h, c)
        ctx = x[-64:].copy()
        probs.append(float(p.item()))
    return probs


# ---------- 主流程 ----------

VARIANTS_16K = ["jit16k", "op16_16k", "op15", "op18", "sequence", "half", "openvino", "st16k"]
VARIANTS_8K = ["jit8k", "op16_8k"]


def build_sessions():
    s = {}
    torch.set_num_threads(1)
    s["jit16k"] = torch.jit.load(os.path.join(M, "silero_vad.jit"), map_location="cpu")
    s["jit8k"] = s["jit16k"]
    s["op16_16k"] = OrtStream(os.path.join(M, "silero_vad.onnx"), 16000)
    s["op16_8k"] = OrtStream(os.path.join(M, "silero_vad.onnx"), 8000)
    s["op15"] = OrtStream(os.path.join(M, "silero_vad_16k_op15.onnx"), 16000)
    s["op18"] = OrtStream(os.path.join(M, "silero_vad_op18_ifless.onnx"), 16000)
    s["half"] = OrtStream(os.path.join(M, "silero_vad_half.onnx"), 16000)
    s["openvino"] = OrtStream(os.path.join(M, "silero_vad_openvino_16k.onnx"), 16000)
    s["sequence"] = make_sequence_sess()
    s["st16k"] = make_st16k()
    return s


def stream(v, sess, wav, sr):
    if v == "jit16k":
        return stream_jit(sess, wav, 16000)
    if v == "jit8k":
        return stream_jit(sess, wav, 8000)
    if v == "sequence":
        return stream_sequence(sess, wav)
    if v == "st16k":
        return stream_st16k(sess, wav)
    return stream_ort(sess, wav)


def bench(v, sess, wav, sr):
    dur = wav.size / sr
    for _ in range(WARMUP):
        stream(v, sess, wav, sr)
    walls = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        stream(v, sess, wav, sr)
        walls.append(time.perf_counter() - t0)
    walls.sort()
    return {"median_s": round(walls[N_RUNS // 2], 4), "best_s": round(walls[0], 4),
            "rtfx_median": round(dur / walls[N_RUNS // 2], 2), "rtfx_best": round(dur / walls[0], 2)}


def main():
    fixtures16 = []
    for p in sorted(glob.glob(os.path.join(REF, "gate", "jit16k", "*.wav.f32"))):
        name = os.path.basename(p)[: -len(".wav.f32")]
        fixtures16.append((name, np.fromfile(p, dtype=np.float32)))
    wav8, sr8 = read_wav(os.path.join(ROOT, "examples", "c++", "aepyx_8k.wav"))
    fixtures8 = [("aepyx8k", wav8)]

    sess = build_sessions()
    results = {}
    if os.path.exists(os.path.join(REF, "baseline_results.json")):
        with open(os.path.join(REF, "baseline_results.json"), encoding="utf-8") as f:
            results = json.load(f)
    results.setdefault("variants", {})

    plan = [(v, fixtures16, 16000) for v in VARIANTS_16K] + [(v, fixtures8, 8000) for v in VARIANTS_8K]

    for v, fixtures, sr in plan:
        vdir = os.path.join(REF, "gate", v)
        os.makedirs(vdir, exist_ok=True)
        s = sess[v]
        total_dur = sum(w.size / sr for _, w in fixtures)
        print(f"=== {v} ({sr}Hz, {len(fixtures)} fixtures, {total_dur:.0f}s) ===")
        rtfx = []
        for name, wav in fixtures:
            np.asarray(wav, dtype=np.float32).tofile(os.path.join(vdir, f"{name}.wav.f32"))
            t0 = time.perf_counter()
            probs = stream(v, s, wav, sr)
            first_pass = time.perf_counter() - t0
            ts = get_speech_timestamps_from_probs(probs, sampling_rate=sr, return_seconds=False)
            ts_pairs = [(int(d["start"]), int(d["end"])) for d in ts]
            np.asarray(probs, dtype=np.float32).tofile(os.path.join(vdir, f"{name}.probs.f32"))
            np.asarray(ts_pairs, dtype=np.int64).tofile(os.path.join(vdir, f"{name}.ts.i64"))
            b = bench(v, s, wav, sr)
            rtfx.append(b["rtfx_median"])
            print(
                f"  {name:<12} {len(probs):>5} frames  first_pass={first_pass:.2f}s  "
                f"RTFx med={b['rtfx_median']:.1f} best={b['rtfx_best']:.1f}  "
                f"probs[min={min(probs):.3f} max={max(probs):.3f}]"
            )
        results["variants"][v] = {
            "rtfx_median_all": rtfx,
            "rtfx_median_of_medians": sorted(rtfx)[len(rtfx) // 2],
        }
        print(f"  -> median-of-medians RTFx: {sorted(rtfx)[len(rtfx)//2]:.1f}")

    with open(os.path.join(REF, "baseline_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("results -> baseline_results.json (variants section)")


if __name__ == "__main__":
    main()
