"""状态机差分模糊测试语料生成器。

用 Python 原版（utils_vad.get_speech_timestamps_from_probs / VADIterator）对
"构造性边界 + 随机"概率序列求期望输出，写入 tests/data/state_machine_fuzz.json；
Rust 侧单元测试（timestamps.rs / vad_iterator.rs）回放比对，要求**逐位一致**。

边界设计（状态机每个分支的两侧都要命中）：
- prob 恰好 == threshold / == f32 邻域（neg_threshold 是 f64，f32 无法精确命中，
  覆盖其上下两个 f32 邻点）
- min_silence_samples / min_speech_samples 恰好等于整数个窗口（== 走边界分支）
  与不等的两种参数
- max_speech 切割：possible_ends 空/非空 × use_max_poss_sil true/false
- padding：silence_duration 奇偶 × pad 大小（0 / 30ms / 77ms）
- 末段：以语音结束 / 以静音结束 / 空序列 / 单帧 / audio_length_samples 截断

用法（需含 torch 的环境）：
  python state_machine_fuzz_gen.py
输出：tests/data/state_machine_fuzz.json（提交入库，Rust 测试不再依赖 Python）
"""

import json
import os
import sys

import numpy as np
import torch

ROOT = os.environ.get("SILERO_PY_REPO", r"D:\silero-vad")
sys.path.insert(0, os.path.join(ROOT, "src"))
from silero_vad.utils_vad import get_speech_timestamps_from_probs, VADIterator  # noqa: E402

rng = np.random.default_rng(20260929)


def bits(a) -> list:
    """f32 数组 → u32 位型（JSON 存位型，Rust f32::from_bits 还原，无精度损失）。"""
    return np.asarray(a, dtype=np.float32).view(np.uint32).astype(np.int64).tolist()


def f32(x) -> float:
    return float(np.float32(x))


# f32 域内的临界值：threshold / neg_threshold 上下邻点、0、1
def boundary_values(threshold: float, neg_threshold: float) -> list:
    t = np.float32(threshold)
    g = np.float32(neg_threshold)
    vals = [
        np.float32(0.0), np.float32(1.0),
        t, np.nextafter(t, np.float32(0), dtype=np.float32),
        np.nextafter(t, np.float32(1), dtype=np.float32),
        g, np.nextafter(g, np.float32(0), dtype=np.float32),
        np.nextafter(g, np.float32(1), dtype=np.float32),
    ]
    return [float(v) for v in vals]


def seq_blocks(n, boundary, hi=(0.6, 1.0), lo=(0.0, 0.25), mid=(0.3, 0.5)):
    """语音/静音交替块，块长在 1..n/3 随机；偶尔插入滞回带（mid）与边界值。"""
    out = []
    hi_state = bool(rng.integers(0, 2))
    while len(out) < n:
        run = int(rng.integers(1, max(2, n // 3)))
        for _ in range(run):
            if hi_state:
                v = rng.uniform(*hi)
            else:
                v = rng.uniform(*lo)
            if rng.random() < 0.06:
                v = boundary[int(rng.integers(0, len(boundary)))]
            elif rng.random() < 0.04:
                v = rng.uniform(*mid)
            out.append(np.float32(v))
            if len(out) >= n:
                break
        hi_state = not hi_state
    return np.asarray(out[:n], dtype=np.float32)


def build_offline_cases():
    """离线状态机（get_speech_timestamps_from_probs）用例。"""
    variants = [
        dict(tag="default"),
        dict(tag="max3s", max_speech_duration_s=3.0),
        dict(tag="max3s_nouse", max_speech_duration_s=3.0, use_max_poss_sil_at_max_speech=False),
        dict(tag="max2s_nouse_neg", max_speech_duration_s=2.0,
             use_max_poss_sil_at_max_speech=False, neg_threshold=0.2),
        dict(tag="sil64", min_silence_duration_ms=64),          # 1024 样本 = 恰好 2 窗
        dict(tag="speech32", min_speech_duration_ms=32),        # 512 样本 = 恰好 1 窗
        dict(tag="pad0", speech_pad_ms=0),
        dict(tag="pad77", speech_pad_ms=77),
        dict(tag="th03", threshold=0.3),
        dict(tag="th08neg", threshold=0.8, neg_threshold=0.65),
        dict(tag="silmax512", min_silence_at_max_speech=512, max_speech_duration_s=3.0),
        dict(tag="sr8k", sampling_rate=8000),
    ]
    cases = []
    for v in variants:
        sr = v.get("sampling_rate", 16000)
        # 原版 from_probs 硬编码 window（16k→512 / 8k→256），不接受该参数
        window = 512 if sr == 16000 else 256
        threshold = v.get("threshold", 0.5)
        neg = v.get("neg_threshold", None)
        neg_eff = neg if neg is not None else max(threshold - 0.15, 0.01)
        boundary = boundary_values(threshold, neg_eff)
        params = dict(
            sampling_rate=sr, threshold=threshold,
            min_speech_duration_ms=v.get("min_speech_duration_ms", 250),
            max_speech_duration_s=v.get("max_speech_duration_s", float("inf")),
            min_silence_duration_ms=v.get("min_silence_duration_ms", 100),
            speech_pad_ms=v.get("speech_pad_ms", 30),
            neg_threshold=neg,
            min_silence_at_max_speech=v.get("min_silence_at_max_speech", 98),
            use_max_poss_sil_at_max_speech=v.get("use_max_poss_sil_at_max_speech", True),
        )

        seqs = []
        # 随机 + 块状，多长度
        for n in [1, 2, 37, 130, 333, 600]:
            seqs.append(rng.random(n))
            seqs.append(seq_blocks(n, boundary))
        # 临界注入：随机位置替换为边界值
        base = seq_blocks(400, boundary)
        inj = base.copy()
        pos = rng.integers(0, 400, size=40)
        for i, p in enumerate(pos):
            inj[p] = np.float32(boundary[i % len(boundary)])
        seqs.append(inj)
        # max_speech 触发：长 above 段 + 稀疏 dip（构造 possible_ends）/ 无 dip
        ms = v.get("max_speech_duration_s", float("inf"))
        if ms != float("inf"):
            need = int(sr * ms / window) + 5
            dip = np.full(need + 10, np.float32(0.9))
            dip[::max(2, need // 4)] = np.float32(0.1)  # 稀疏 dip → possible_ends
            seqs.append(dip)
            nodip = np.full(need + 10, np.float32(0.9))
            seqs.append(nodip)
        # 手工临界：silence / speech 恰好整数窗（对齐 min_silence / min_speech 参数）
        for k in [1, 2, 3, 4]:
            s = np.full(k * 2 + 3, np.float32(0.0))
            s[k + 1:k + 1 + k] = np.float32(0.9)  # 恰好 k 窗语音 + k 窗静音交替
            seqs.append(s)
        # 末端形态：全 above（末段以语音结束）/ 全 below / 单帧
        seqs.append(np.ones(137, dtype=np.float32))
        seqs.append(np.zeros(137, dtype=np.float32))

        for si, probs in enumerate(seqs):
            audio_len = None
            if si % 7 == 3:  # 末窗非整：audio_length_samples 截断
                audio_len = (len(probs) - 1) * window + 137
            exp = get_speech_timestamps_from_probs(
                [float(x) for x in probs], audio_length_samples=audio_len, **params)
            # JSON 无法表示 inf（serde_json 拒收）：存 null，Rust 侧还原
            params_json = {**params, "max_speech_duration_s": (
                None if params["max_speech_duration_s"] == float("inf")
                else params["max_speech_duration_s"])}
            cases.append({
                "name": f"{v['tag']}_{si}",
                "params": params_json,
                "audio_length_samples": audio_len,
                "probs_bits": bits(probs),
                "expected": [[int(d["start"]), int(d["end"])] for d in exp],
            })
    return cases


class FakeModel:
    """把概率序列注入 VADIterator（模型调用只消耗序列，不碰音频内容）。"""

    def __init__(self, probs):
        self.p = list(probs)
        self.i = 0

    def reset_states(self):
        pass  # VADIterator.__init__/reset_states 会调用；纯注入无状态可重置

    def __call__(self, x, sr):
        v = self.p[self.i]
        self.i += 1
        return torch.tensor(v, dtype=torch.float32)


def build_iterator_cases():
    """流式状态机（VADIterator）用例：FakeModel 注入概率，任意窗口长度。"""
    variants = [
        dict(tag="default"),
        dict(tag="th08", threshold=0.8),
        dict(tag="sil64", min_silence_duration_ms=64),
        dict(tag="sil0", min_silence_duration_ms=0),
        dict(tag="pad0", speech_pad_ms=0),
        dict(tag="pad77", speech_pad_ms=77),
        dict(tag="w256", window_size_samples=256),
        dict(tag="w100", window_size_samples=100),
    ]
    cases = []
    for v in variants:
        window = v.get("window_size_samples", 512)
        threshold = v.get("threshold", 0.5)
        boundary = boundary_values(threshold, max(threshold - 0.15, 0.01))
        for si in range(30):
            n = int(rng.integers(2, 260))
            if si % 3 == 0:
                probs = seq_blocks(n, boundary)
            elif si % 3 == 1:
                probs = rng.random(n)
            else:
                probs = np.asarray([boundary[int(rng.integers(0, len(boundary)))]
                                    for _ in range(n)], dtype=np.float32)
            windows = [window] * n
            if si % 5 == 2 and n > 3:  # 末窗非整
                windows[-1] = max(1, window // 2)
            it = VADIterator(
                FakeModel([float(x) for x in probs]), threshold=threshold,
                sampling_rate=16000, min_silence_duration_ms=v.get("min_silence_duration_ms", 100),
                speech_pad_ms=v.get("speech_pad_ms", 30))
            events = []
            for w in windows:
                r = it(torch.zeros(w))
                if r is not None:
                    kind = 0 if "start" in r else 1
                    events.append([kind, int(r["start"] if kind == 0 else r["end"])])
            cases.append({
                "name": f"{v['tag']}_{si}",
                "threshold": threshold,
                "min_silence_duration_ms": v.get("min_silence_duration_ms", 100),
                "speech_pad_ms": v.get("speech_pad_ms", 30),
                "sampling_rate": 16000,
                "probs_bits": bits(probs),
                "windows": windows,
                "expected": events,
            })
    return cases


def main():
    offline = build_offline_cases()
    it = build_iterator_cases()
    out = {
        "seed": 20260929,
        "generator": "state_machine_fuzz_gen.py（期望输出来自 Python 原版，逐位对齐契约）",
        "offline": offline,
        "iterator": it,
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "tests", "data", "state_machine_fuzz.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    size = os.path.getsize(path) / 1e6
    print(f"offline cases={len(offline)}  iterator cases={len(it)}  -> {path} ({size:.2f} MB)")
    segs = sum(len(c["expected"]) for c in offline)
    evs = sum(len(c["expected"]) for c in it)
    print(f"non-trivial coverage: offline segments={segs}, iterator events={evs}")


if __name__ == "__main__":
    main()
