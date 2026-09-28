"""从 JIT 模型导出 Rust 移植所需输入（权重 + 门禁数据）。

1. 权重：直接取 silero_vad.jit 的 16k 分支参数（bit-exact 权威源），
   写 wgpu/ref/silero_vad_16k_jit.safetensors。
   注意：仓库自带的 silero_vad_16k.safetensors 与 JIT **不是同一份权重**
   （conv/lstm 的 max_abs_diff ~1.0），不能作为移植权重源。
2. 门禁数据：把 wgpu/ref/golden_*.npz 展开成 Rust 直读的裸文件：
   gate/<name>.wav.f32   输入音频 f32 LE
   gate/<name>.probs.f32 期望逐帧概率 f32 LE
   gate/<name>.ts.i64    期望时间戳 (start,end) 对 i64 LE
"""

import glob
import os
import sys

import numpy as np
import torch

# 上游 Python 仓库（含 models/ 与 src/silero_vad），可用环境变量覆盖
PY_REPO = os.environ.get("SILERO_PY_REPO", r"D:\silero-vad")
ROOT = PY_REPO
REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref")
GATE = os.path.join(REF, "gate")


def dump_weights():
    m = torch.jit.load(
        os.path.join(ROOT, "models", "silero_vad.jit"),
        map_location="cpu",
    )
    enc = list(m._model.encoder.children())
    tensors = {
        "stft_conv.weight": m._model.stft.forward_basis_buffer,
        "conv1.weight": enc[0].reparam_conv.weight,
        "conv1.bias": enc[0].reparam_conv.bias,
        "conv2.weight": enc[1].reparam_conv.weight,
        "conv2.bias": enc[1].reparam_conv.bias,
        "conv3.weight": enc[2].reparam_conv.weight,
        "conv3.bias": enc[2].reparam_conv.bias,
        "conv4.weight": enc[3].reparam_conv.weight,
        "conv4.bias": enc[3].reparam_conv.bias,
        "lstm_cell.weight_ih": m._model.decoder.rnn.weight_ih,
        "lstm_cell.weight_hh": m._model.decoder.rnn.weight_hh,
        "lstm_cell.bias_ih": m._model.decoder.rnn.bias_ih,
        "lstm_cell.bias_hh": m._model.decoder.rnn.bias_hh,
        "final_conv.weight": getattr(m._model.decoder.decoder, "2").weight,
        "final_conv.bias": getattr(m._model.decoder.decoder, "2").bias,
    }
    tensors = {k: v.contiguous().clone() for k, v in tensors.items()}
    out = os.path.join(REF, "silero_vad_16k_jit.safetensors")
    from safetensors.torch import save_file

    save_file(tensors, out)
    # crate 内嵌权重（Weights::embedded_16k 用）——随源码入库
    assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    os.makedirs(assets, exist_ok=True)
    save_file(tensors, os.path.join(assets, "silero_vad_16k_jit.safetensors"))
    print(f"weights -> {out} + assets/ ({os.path.getsize(out)} bytes, {len(tensors)} tensors)")


def dump_variant_weights():
    """W_B: half.onnx（独立权重集）；W_8k: jit 的 8k 分支。

    W_A = silero_vad_16k_jit.safetensors（op15/op16/op18/openvino/sequence 共享）。
    W_C = models/silero_vad_16k.safetensors（键名已是规范格式，Rust 直接读）。
    """
    import onnx
    from onnx import numpy_helper
    from safetensors.torch import save_file

    # W_B: half.onnx
    og = onnx.load(os.path.join(ROOT, "models", "silero_vad_half.onnx"))
    inits = {i.name: numpy_helper.to_array(i) for i in og.graph.initializer}
    keymap = {
        "stft_conv.weight": "stft.forward_basis_buffer",
        "conv1.weight": "encoder.0.reparam_conv.weight",
        "conv1.bias": "encoder.0.reparam_conv.bias",
        "conv2.weight": "encoder.1.reparam_conv.weight",
        "conv2.bias": "encoder.1.reparam_conv.bias",
        "conv3.weight": "encoder.2.reparam_conv.weight",
        "conv3.bias": "encoder.2.reparam_conv.bias",
        "conv4.weight": "encoder.3.reparam_conv.weight",
        "conv4.bias": "encoder.3.reparam_conv.bias",
        "lstm_cell.weight_ih": "decoder.rnn.weight_ih",
        "lstm_cell.weight_hh": "decoder.rnn.weight_hh",
        "lstm_cell.bias_ih": "decoder.rnn.bias_ih",
        "lstm_cell.bias_hh": "decoder.rnn.bias_hh",
        "final_conv.weight": "decoder.decoder.2.weight",
        "final_conv.bias": "decoder.decoder.2.bias",
    }
    tensors = {}
    for ck, ok in keymap.items():
        v = inits[ok]
        tensors[ck] = torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
    out = os.path.join(REF, "weights_half16k.safetensors")
    save_file(tensors, out)
    print(f"weights -> {out} ({os.path.getsize(out)} bytes)")

    # W_8k: jit 8k 分支
    m8 = torch.jit.load(os.path.join(ROOT, "models", "silero_vad.jit"), map_location="cpu")._model_8k
    enc8 = list(m8.encoder.children())
    t8 = {
        "stft_conv.weight": m8.stft.forward_basis_buffer,
        "conv1.weight": enc8[0].reparam_conv.weight,
        "conv1.bias": enc8[0].reparam_conv.bias,
        "conv2.weight": enc8[1].reparam_conv.weight,
        "conv2.bias": enc8[1].reparam_conv.bias,
        "conv3.weight": enc8[2].reparam_conv.weight,
        "conv3.bias": enc8[2].reparam_conv.bias,
        "conv4.weight": enc8[3].reparam_conv.weight,
        "conv4.bias": enc8[3].reparam_conv.bias,
        "lstm_cell.weight_ih": m8.decoder.rnn.weight_ih,
        "lstm_cell.weight_hh": m8.decoder.rnn.weight_hh,
        "lstm_cell.bias_ih": m8.decoder.rnn.bias_ih,
        "lstm_cell.bias_hh": m8.decoder.rnn.bias_hh,
        "final_conv.weight": getattr(m8.decoder.decoder, "2").weight,
        "final_conv.bias": getattr(m8.decoder.decoder, "2").bias,
    }
    t8 = {k: v.contiguous().clone() for k, v in t8.items()}
    out = os.path.join(REF, "weights_jit8k.safetensors")
    save_file(t8, out)
    print(
        f"weights -> {out} ({os.path.getsize(out)} bytes; "
        f"basis {tuple(t8['stft_conv.weight'].shape)})"
    )


def export_gate():
    os.makedirs(GATE, exist_ok=True)
    n = 0
    for npz_path in sorted(glob.glob(os.path.join(REF, "golden_*.npz"))):
        name = os.path.basename(npz_path)[len("golden_"):-len(".npz")]
        vdir = os.path.join(GATE, "jit16k")
        os.makedirs(vdir, exist_ok=True)
        z = np.load(npz_path)
        np.concatenate([z["wav"].astype(np.float32)]).tofile(
            os.path.join(vdir, f"{name}.wav.f32")
        )
        z["probs"].astype(np.float32).tofile(os.path.join(vdir, f"{name}.probs.f32"))
        z["ts"].astype(np.int64).tofile(os.path.join(vdir, f"{name}.ts.i64"))
        n += 1
        print(
            f"gate/jit16k/{name}: {z['wav'].size} samples, {z['probs'].size} probs, "
            f"{z['ts'].shape[0]} ts pairs"
        )
    print(f"exported {n} fixtures -> {GATE}/jit16k")


if __name__ == "__main__":
    dump_weights()
    dump_variant_weights()
    export_gate()
