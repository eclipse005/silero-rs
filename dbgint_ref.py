"""帧 N 前馈参考（numpy，f32）：与 GpuBatch::debug_intermediates 读回比对。

用法：python dbgint_ref.py <fixture.f32> [frame_idx]
输出：stft/conv1..4/gates_in 各级前 8 个值 + e4/gin 全量摘要。
"""

import sys
import numpy as np
from safetensors.torch import load_file

W = load_file(r"D:\silero-rs\ref\silero_vad_16k_jit.safetensors")
path = sys.argv[1]
frame_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
wav = np.fromfile(path, dtype="<f4")

# x 行：ctx 64（frame_idx>0 时取上一帧尾部 64 样本）+ frame 512 + reflect 64
s = frame_idx * 512
e = min(s + 512, wav.size)
chunk = np.zeros(512, np.float32)
chunk[: e - s] = wav[s:e]
ctx = np.zeros(64, np.float32) if frame_idx == 0 else wav[s - 64 : s].copy()
x = np.concatenate([ctx, chunk])
x = np.concatenate([x, np.array([x[576 - 2 - i] for i in range(64)], np.float32)]).astype(np.float32)

basis = W["stft_conv.weight"].numpy().reshape(258, 256).astype(np.float32)
mag = np.zeros((4, 129), np.float32)
for p in range(4):
    win = x[p * 128 : p * 128 + 256]
    re = basis[:129] @ win
    im = basis[129:] @ win
    mag[p] = np.sqrt(re * re + im * im).astype(np.float32)

def conv(inp, w, b, positions):
    oc, ic, K = w.shape
    out = np.zeros((len(positions), oc), np.float32)
    for pi, valid in enumerate(positions):
        acc = b.copy()
        for k, ip in valid:
            acc += w[:, :, k] @ inp[ip]
        out[pi] = np.maximum(acc, 0).astype(np.float32)
    return out

e1 = conv(mag, W["conv1.weight"].numpy(), W["conv1.bias"].numpy(),
          [[(k, p + k - 1) for k in range(3) if 1 <= p + k <= 4] for p in range(4)])
e2 = conv(e1, W["conv2.weight"].numpy(), W["conv2.bias"].numpy(),
          [[(k, p * 2 + k - 1) for k in range(3) if 1 <= p * 2 + k <= 4] for p in range(2)])
e3 = conv(e2, W["conv3.weight"].numpy(), W["conv3.bias"].numpy(),
          [[(k, k - 1) for k in range(3) if 1 <= k <= 2]])
e4 = conv(e3, W["conv4.weight"].numpy(), W["conv4.bias"].numpy(),
          [[(k, k - 1) for k in range(3) if 1 <= k <= 1]])

gin = (W["lstm_cell.weight_ih"].numpy() @ e4[0] + W["lstm_cell.bias_ih"].numpy()).astype(np.float32)

np.set_printoptions(precision=7, linewidth=200)
print("ref mag[:8]  =", mag.reshape(-1)[:8])
print("ref e1[:8]   =", e1.reshape(-1)[:8])
print("ref e2[:8]   =", e2.reshape(-1)[:8])
print("ref e3[:8]   =", e3.reshape(-1)[:8])
print("ref e4[:8]   =", e4.reshape(-1)[:8])
print("ref gin[:8]  =", gin[:8])
print("ref gin[256:264] =", gin[256:264])
