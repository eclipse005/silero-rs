# HANDOFF — wgpu RTFx 优化交接

> 面向接手优化工作的 AI/工程师。本文档只提供**事实与环境**，不含优化方案。
> 目标：把 `SILERO_BACKEND=vulkan` 的 GPU 路径 RTFx 提上去（当前集显低于单线程 CPU）。
>
> **硬约束：优化不得破坏数值对齐**。见 §6，破坏对齐的"提速"一律不算成果。

---

## 1. 任务与现状一句话

把 `D:\silero-rs`（Silero VAD 的 Rust 移植，CPU SIMD + wgpu GPU 双后端）的 **GPU 路径 RTFx** 提上去。
2026-09 优化后：Intel 集显单流 RTFx ≈ 930–1000（原 512–590），已高于同机单线程 CPU（485–827）；
GTX 1070 单流 ≈ 1460（原 1104–1144）。公开 API 未变（`stream`/`frame_batch`，与原版对齐），
全部提速来自 kernel 内部的逐位等价重排。

---

## 2. 仓库状态

| 项 | 值 |
|---|---|
| 路径 | `D:\silero-rs`（git repo，分支 `master`） |
| 远端 | `origin https://github.com/eclipse005/silero-rs.git` |
| HEAD | `b3cf7c3 Add bench: per-kernel GPU timing and RTFx baseline on both adapters` |
| 与远端 | **ahead 3，未 push**（`e846ab1` 数值修复、`e79bc4c` dbgint、`b3cf7c3` bench） |
| 工作区 | 干净 |
| 构建 | `cargo build --release --features gpu` |
| 依赖 | `wgpu 30.0.1`、`pollster 0.4`、`safetensors 0.4`、`hound 3.5`；edition 2021；`lto=true, codegen-units=1` |

---

## 3. 原版参照环境

| 项 | 值 |
|---|---|
| Python 原版仓库 | `D:\silero-vad`（Silero VAD v6.2.3 上游） |
| golden 基线工具 | `SILERO_PY_REPO` 环境变量，默认 `D:\silero-vad` |
| 生成 golden 的脚本 | `export_for_rust.py`、`variant_baselines.py`、`video_golden.py`、`phase0_baseline.py` |
| 原版运行环境 | PyTorch `2.8.0+cu128`，硬件加速 `NVIDIA P104-100` |
| 原版 CPU RTFx（16k，单线程） | **≈ 117**（`ref/baseline_results.json`，test60s / aepyx169s） |
| 原版 4 线程 CPU | ≈ 65 |
| 原版 CUDA 逐帧同步 | ≈ 66 |
| 原版 CUDA 吞吐上限 | ≈ 74 |

> ⚠️ 原版的 CUDA 数字是**逐帧同步**语义，不是本项目的批量语义，不可直接比较。

**golden 数据位置**（全部已入库，直接读，不要重新生成）：

- `ref/gate/<case>.{wav.f32, probs.f32, ts.i64}` — 12 个模型变体 × 短片段（**≤60s**，见 §6 警告）
- `ref/gate/{half,jit16k,jit8k,op15,op16_16k,op16_8k,op18,openvino,sequence,st16k}/` — 各变体的冻结基线
- `ref/video/v01..v08.{f32, probs.f32, probs_cuda.f32, ts.i64, ts_cuda.i64, vadit.i64}` — 真实媒体，8 段，最长 1525s
- `ref/video/manifest.json`、`ref/video/video_golden.json`
- `ref/*.safetensors` — 权重（`silero_vad_16k_jit.safetensors` 是 jit16k 及全部 ONNX 变体共享的 W_A）

---

## 4. 本机硬件与驱动

```
OS        Windows 11 (win32 10.0.26200 x64), PowerShell 7
CPU       单线程路径只用 1 核（SIMD: AVX2+FMA，运行时检测）
GPU 1     Intel(R) Graphics           — IntegratedGpu
GPU 2     NVIDIA GeForce GTX 1070     — DiscreteGpu (报告名 NVIDIA P104-100)
显示适配器 GameViewer Virtual Display Adapter（无 GPU 能力）
Vulkan    Loader 1.4.313.0 (C:\Windows\System32\vulkan-1.dll)
```

**多卡选择方式（重要）**：`power_preference` 只会选独显，**没有别的办法选中集显**。
必须用 `SILERO_ADAPTER=<名字子串>`（大小写不敏感，匹配 adapter name / driver / device_type）。

### 已实测的适配器 limits / features（Vulkan）

| limit | Intel iGPU | GTX 1070 |
|---|---|---|
| `max_storage_buffers_per_shader_stage` | 3355442 | 524288 |
| `max_compute_invocations_per_workgroup` | 1024 | 1536 |
| `max_compute_workgroup_size_x` | 1024 | 1536 |
| `max_compute_workgroup_storage_size` | 49152 B | 49152 B |
| `max_storage_buffer_binding_size` | **1073741820 B (~1 GiB)** | 2147483644 B |
| `max_buffer_size` | 4294901760 B | 4292870144 B |
| `TIMESTAMP_QUERY` | true | true |
| `SHADER_F16` | true | false |

> 注意集显的 `max_storage_buffer_binding_size` 只有 ~1 GiB（独显 2 GiB）。本项目最大 buffer 约 1.3 MiB，
> 目前不构成约束，但若将来加大 batch 需要重新评估。

`diag` 已删除；用 `SILERO_ADAPTER_INFO=1` 跑任一 bin 即可打印适配器清单与 limits。

---

## 5. 代码结构（GPU 路径）

### 文件

| 文件 | 作用 |
|---|---|
| `src/lib.rs` | host CPU 路径（SIMD）、`SileroVad`、`Weights`、`CFG_16K/8K` |
| `src/gpu.rs` | GPU 公共设施（后端/适配器选择、诊断打印），供 `gpu_batch` 与各 bin 使用。逐帧 `GpuVad` + `gpu.wgsl` 已删除（Intel 发散诊断使命完成，git 历史保留） |
| `src/gpu_batch.rs` | **批量** GPU 路径 `GpuBatch`（`gpu_batch.wgsl`）。CLI / align / bench 实际走的路径 |
| `src/gpu_batch.wgsl` | 批量 kernels（模板，`{BATCH}`/`{LSTM_CHUNKS}` 等占位由 Rust 注入） |
| `src/wgsl_math.wgsl` | 厂商无关超越函数层，注入到上面两个 shader 前面 |
| `src/transcendental.rs` | 同上算法的 host 镜像 |
| `src/bin/bench.rs` | **基线工具**：逐 kernel 占比 / 单批墙钟 / 批大小扫描 / 端到端 RTFx |
| `src/bin/align.rs` | 四路对齐校验（golden CPU / golden CUDA / Rust CPU / Rust GPU） |
| `src/bin/gate.rs` | 10 模型变体 gate + RTFx 基准 |
| `src/bin/trace.rs` | 长音频发散定位，`--dump` 导出概率流 |
| `src/bin/dbgint.rs` | 单帧中间量 dump |

### 批量路径的执行结构（`GpuBatch::frame_batch`）

常量：`MAX_T = 512`（每批帧数）、`LSTM_CHUNK = 32`（每个 dispatch 串行处理的帧数）。

一个满批（T=512）共 **23 个 dispatch（6 conv + 16 lstm chunk + 1 prob）+ 1 次 submit +
1 次 map_async**：

| # | kernel | 线程数 (T=512) | workgroup 数 | 说明 |
|---|---|---|---|---|
| 0 | `main_stft` | 264192 | 4128 | x640 → mag，workgroup_size 64，2D dispatch（防 65535 上限） |
| 1 | `main_conv1` | 262144 | 4096 | 128 输出通道 |
| 2 | `main_conv2` | 65536 | 1024 | |
| 3 | `main_conv3` | 32768 | 512 | |
| 4 | `main_conv4` | 65536 | 1024 | |
| 5 | `main_gates_in` | 262144 | 4096 | Wih@e4 |
| 6..21 | `lstm_chunk_0..15` | 256 | **1（每个）** | **单 workgroup，串行 32 帧** |
| 22 | `main_prob` | 128×T | T | fw·relu(h) 顺序归约 + sigmoid（每帧 1 个 workgroup(128)） |

- 前 5 个 conv + stft + gates_in：逐帧独立。**显式软件流水**（先取 ic+4..ic+7 的权重/激活，
  再做本拍 FMA）+ **conv 权重 host 侧重排** `[co][ic][k]→[co][k][ic]`（线程沿 ic 连续读）。
  Intel 集显上 conv 曾是隐藏的最大瓶颈（见 §9）。
- **LSTM 递归不可并行**，每 32 帧打包进 1 个 dispatch，单 workgroup(256) 内 for 循环串行；
  每帧 2 次 `workgroupBarrier()`；跨帧 h/c 在 workgroup 内存，chunk 首尾与 `b_h`/`b_c` 搬运一次。
- prob 从递归循环拆出为批量 kernel：与旧 in-loop 归约**逐位一致**（乘积先存 workgroup
  内存、线程 0 顺序累加 128 项 + sigmoid）。
- 流水深度为 **1**：`frame_batch` 先给上一批注册 map，再提交本批，然后收割上一批。

### 缓冲区（T=512 时）

| buffer | 字节 | 用途 |
|---|---|---|
| `x` / `x_alt` | 1310720 各一 | 输入（ctx64 + frame512 + pad64 = 640/帧），双缓冲交替 |
| `W` | 1238532 | 15 个权重张量 concat（conv 已重排、whh 已转置），只读 |
| `mag` | 1058816 | stft 输出 |
| `e1` | 1050624 | |
| `e2` / `e4` | 264192 各一 | |
| `e3` | 133120 | |
| `gin` | 1179648 | Wih@e4 + bih |
| `h` / `c` | 512 各一 | LSTM 递归状态，常驻 GPU |
| `prob` | 131072 | 行距 64 f32 = 256 B/帧（首 f32 有效） |
| `h_all` | 262144 | 每帧 h 快照（lstm 写，main_prob 读） |
| `staging[2]` | 131072 各一 | 读回缓冲，双缓冲 |

合计约 9.5 MiB。

### 计算量（16k，每帧单块 T=1）

约 **60 万 MAC/帧**（≈1.2 MFLOP）：stft 264,192 + conv1 148,608 + conv2 40,960 + conv3 8,192 +
conv4 8,192 + gates_in 65,536 + LSTM(whh·h + final) 65,664。
16k 每帧 32 ms，故每秒音频约 37.5 MFLOP。

---

## 6. 数值契约（**硬约束**）

Silero 的 LSTM 是**指数不稳定递归**：每步 1 ULP 差异会被放大。实测在 1525s 音频上，
1 ULP → frame 314 时 1e-6 → frame 12789 时 0.75 → 时间戳从 567 段变 555 段。

因此：**任何改变浮点运算顺序 / 累加顺序 / 激活函数实现的改动，都有真实的概率风险，
而短片段测试看不见它。**

### 为什么有 `wgsl_math.wgsl`（不要删、不要绕过）

GPU 的**硬件**超越函数不可复现：同一份 WGSL 在 Intel 与 NVIDIA 上，
`exp` 有 94/275、`tanh` 有 110/275 个采样输入位模式不同（差 1–2 ULP）。
次正规刷零已被证伪（次正规输出为 0）。

所以 `exp` / `expm1` / `sigmoid` / `tanh` 被替换为**只用 `+ - * /`** 的手写实现
（Cody-Waite 区间归约 + `exp(r) = 1 + r + r²·Q(r)`）。
`+ - * /` 已实测在这两家上逐位一致。

`tanh` 用 `-expm1(-2x)/(expm1(-2x)+2)`，**不要**改回 `2·sigmoid(2x)-1`：后者在 x≈0 处病态
（实测 64 ULP），而 LSTM 的 gate 常落在 0 附近。

**跨厂商逐位一致做不到，也不声称**：WGSL 无 `precise` 限定符，naga 不发 SPIR-V `NoContraction`，
驱动会把 `a*b+c` 收缩成 FMA，Rust/LLVM 不会 —— 两端最大差 4 ULP。

### 验收标准

| 项 | 要求 | 现状 |
|---|---|---|
| 10 变体 gate | `ts=OK diff=OK`，prob diff ≤ ~1e-4 | PASS（最大 1.67e-5） |
| 四路 align（8 段视频） | 时间戳 4-way 逐段精确相等 | **Intel PASS / NVIDIA PASS** |
| `cargo test` | 全绿 | PASS |
| 逐位回归 | 优化后 GPU 概率流与旧实现 **byte-exact**（v01 47665 帧 + v02 25842 帧，`trace --dump` 比对） | PASS |
| 修复前后对照 | Intel：22695 帧 >1e-5 → **7 帧**；时间戳 555 → **567 = golden** | — |

### ⚠️ 60s 用例抓不到这类缺陷

`ref/gate/` 的 12 个变体 fixture **最长只有 60s**。发散要到 ~frame 13000 才放大到能翻转时间戳，
所以**短用例全绿 ≠ 没有发散**。跨厂商改动必须跑长音频。

---

## 7. 基线数据（本机实测，`bench` / `align`）

RTFx = 音频时长 / 墙钟时间，>1 表示快于实时。测量对象：`ref/video/v01.f32`，1525.2s。

### 端到端（2026-09 优化后）

| 路径 | RTFx | 每帧 |
|---|---|---|
| Rust CPU 单线程（AVX2+FMA） | 485 – 827（**波动大，见 §9**） | 38.7 – 66.0 µs |
| **Rust GPU / Intel iGPU / Vulkan** | **930 – 1000** | **31.9 – 33 µs** |
| Rust GPU / GTX 1070 / Vulkan | ≈ 1460 | 21.9 µs |
| Python 原版 CPU 单线程 | ≈ 117 | — |

优化前基线（供对照）：Intel 单流 512–590（54.6–58.9 µs/帧），1070 单流 1104–1144。

`align` 逐段 RTFx（rtfx_cpu/rtfx_gpu，2026-09 优化后）：

| fixture | Intel 集显 | GTX 1070 |
|---|---|---|
| v01 (47665 帧) | 823 / 929 | 803 / 1462 |
| v02 | 813 / 933 | 824 / 1456 |
| v03 | 808 / 830 | 818 / 1444 |
| v04 | 814 / 914 | 814 / 1461 |
| v05 | 825 / 929 | 822 / 1400 |
| v06 | 829 / 927 | 785 / 1396 |
| v07 | 831 / 932 | 827 / 1434 |
| v08 | 814 / 931 | 829 / 1438 |

### 逐 kernel 时间占比（⚠️ 不可信，勿再用它定位）

优化前的占比表声称 lstm 97.6%、conv 合计 2.4% —— **这是错的**（timestamp 周期按 1ns 折算
导致的假象，Intel 上占比也不可靠）。真实占比（host 墙钟逐 kernel 实测，优化前满批 512 帧）：
conv 合计 ≈ 18.7ms（conv1 8.4 / stft 3.9 / gates_in 3.6 / conv2 1.8 / conv3+4 各 0.5）、
LSTM+prob ≈ 10.7ms。**定位瓶颈只能用 host 墙钟 + 逐 kernel 单独 dispatch（SILERO_ONLY 式
探针）或逐个跳过 dispatch，不要信 timestamp。**

### 单批墙钟（host 侧，不依赖时间戳）

Intel：满批 512 帧 = **28.4 ms**（55.4 µs/帧）。

### 批大小扫描（Intel，固定总帧数 4096）

| t（帧/批） | 批数 | ms/批 | µs/帧 |
|---|---|---|---|
| 16 | 256 | 0.98 | 61.4 |
| 64 | 64 | 3.42 | 53.4 |
| 128 | 32 | 7.90 | 61.8 |
| 256 | 16 | 14.49 | 56.6 |
| 512 | 8 | 29.14 | 56.9 |

**µs/帧 在 t=16..512 的 32 倍范围内基本恒定** → 每批的 submit/readback 固定开销不是瓶颈，
成本是真正与帧数成正比的 GPU 工作。

---

## 8. 常用命令

```powershell
cd D:\silero-rs

# —— 基线 / 性能 ——
$env:SILERO_BACKEND="vulkan"; $env:SILERO_ADAPTER="Intel"
cargo run --release --features gpu --bin bench
$env:SILERO_ADAPTER="NVIDIA"; cargo run --release --features gpu --bin bench

# —— 正确性：四路对齐（必须两张卡都跑）——
$env:SILERO_ADAPTER="Intel";  cargo run --release --features gpu --bin align
$env:SILERO_ADAPTER="NVIDIA"; cargo run --release --features gpu --bin align

# —— 正确性：10 变体 gate ——
cargo run --release --bin gate -- --no-bench

# —— 测试（含真实 GPU 的数值一致性 + 16 分钟长音频回归）——
$env:SILERO_BACKEND="vulkan"; $env:SILERO_ADAPTER="Intel"
cargo test --release --features gpu

# —— 适配器清单 ——
$env:SILERO_ADAPTER_INFO="1"; cargo run --release --features gpu --bin bench

# —— 发散定位 / 跨卡概率流 diff ——
cargo run --release --features gpu --bin trace -- video v01
cargo run --release --features gpu --bin trace -- video v01 --dump v01_intel.f32

# —— 改 wgsl 后（include_str! 依赖，务必确认真的重编了）——
cargo build --release --features gpu   # 输出里应出现 "Compiling silero-vad-wgpu"
```

环境变量一览：`SILERO_BACKEND=vulkan|dx12|metal|gl|primary`、`SILERO_ADAPTER=<子串>`、
`SILERO_POWER=low`、`SILERO_ADAPTER_INFO=1`、`SILERO_SIMD=0`、`SILERO_MATH_DUMP=<path>`、`SILERO_PY_REPO`。

---

## 9. 已知坑（会浪费你时间的地方）

1. **`debug_timing` 的绝对毫秒不可信，Intel 上连占比都不可信**。它按「1 tick = 1 ns」折算，
   但各适配器 `timestampPeriod` 不同（实测 Intel 与 NVIDIA 差 ~150×）。优化前它声称
   lstm 占 97.6%、conv 占 2.4%，而 host 墙钟实测 conv 才是大头 —— 这误导了两轮优化。
   **定位瓶颈：host 墙钟 + 单独 dispatch/跳过 dispatch 的探针。**
2. **本机 Intel 集显的奇特执行模型**（Xe iGPU，驱动 101.8860，实测规律）：
   - 单 workgroup 的串行帧循环每帧有 ~38µs 的固定下限（掏空循环体也一样）——负载太低，
     频率/调度不上去了；这正是当初集显跑不过 CPU 的根因。
   - conv 的行间并行度几乎为 0（每行 ~21-37µs，不随行数变快）；LSTM 若开多个 workgroup
     则可完美并行（实测 k≤16 墙钟与 k=1 持平）——当前单流用不到，留作将来参考。
   - **workgroup_size(512) 会严重劣化**（LSTM 与 conv 都试过，256/64 反而快），
     最终用 64（conv）/256（lstm）。
   - conv 全部走 2D dispatch（shader 用 `num_workgroups` 折算线性 idx，防 1D dispatch
     的 65535 workgroup 上限）。
3. **`Copy-Item` 会保留源文件 mtime**。用备份覆盖被改文件后 cargo 可能认为没变化而不重编，
   导致测的还是旧代码。用 `(Get-Item x).LastWriteTime = Get-Date` 强制触发。
4. **CPU RTFx 噪声大**（485–827）。本机有其它负载时波动明显。基准取 3 次中位数，
   且不要在跑其它重任务时测。
5. **PowerShell 不支持 heredoc**（`<<EOF`），写多行文件用 here-string `@' ... '@`。
6. `src/gpu.rs` 的逐帧 `GpuVad` + `gpu.wgsl` **已删除**（含 gate 的流式 GPU 臂、trace 的
   `--framewise`）。历史上 HANDOFF 写过"无人调用"，实际 gate/trace 诊断臂在用——该说法
   曾经过时；诊断使命（Intel 发散定位）完成后连同调用点一起删除，git 历史保留。
   现 `src/gpu.rs` 只剩后端/适配器公共设施；改它仍不会影响任何测量结果。
7. 长音频 fixture 很大（`v01.f32` = 97 MB）。`cargo test` 用的 `tests/long_audio.rs` 选的是
   `v02`（25842 帧 / 52 MB），因为发散要 ~frame 13000 才暴露时间戳翻转，`v05`/`v07` 太短抓不到。
8. **数值逐位等价的改法清单**（本轮验证过的安全手法，改之前先对照）：
   线程↔行的映射重排（每行算式不变）、纯 host 侧缓冲区重排（如 conv 权重 [co][ic][k]→[co][k][ic]、
   whh 预转置）、load 时机重排（软件流水/预取，值与界内性不变）、prob 拆成独立批量 kernel
   （乘积先存 workgroup、线程 0 顺序累加的结构照搬）。每改一次，用
   `trace -- video v01 --dump` 与旧实现字节比对 + 双卡 align 验证。

---

## 10. 建议的验证顺序（任何提速改动后）

```powershell
# 1. 单元测试（快，~10s）
cargo test --release --features gpu --lib

# 2. 真实 GPU 数值一致性 + 16 分钟长音频回归（~10s）
$env:SILERO_BACKEND="vulkan"; $env:SILERO_ADAPTER="Intel"
cargo test --release --features gpu

# 3. 长音频四路对齐 —— 集显
$env:SILERO_ADAPTER="Intel"; cargo run --release --features gpu --bin align
# 4. 长音频四路对齐 —— 独显（跨厂商改动必须两张卡都过）
$env:SILERO_ADAPTER="NVIDIA"; cargo run --release --features gpu --bin align

# 5. 10 变体 gate
cargo run --release --bin gate -- --no-bench

# 6. 性能对比
$env:SILERO_ADAPTER="Intel";  cargo run --release --features gpu --bin bench
$env:SILERO_ADAPTER="NVIDIA"; cargo run --release --features gpu --bin bench
```

判据：`align` 输出末尾必须是 `ALIGN: PASS (ts 4-way exact + vadit events exact)`，
`gate` 必须是 `GATE: PASS`。任一失败即数值被破坏，**提速作废**。

---

## 11. 尚未提交 / 可选后续

- `master` 领先 `origin/master` 4 个提交，**未 push**。是否推送由维护者决定。
- `e79bc4c` 引入的 `dbgint.rs` / `dbgint_ref.py` 是上上轮遗留的调试工具，单独成 commit 便于区分。
- 2026-09 优化（conv 重排/软件流水 + prob 批量 kernel）为纯内部改动，公开 API 未变。
  后续可选：conv 的 8 深软件流水（本机收益递减）；公开 API 若要保持与原版对齐，
  **不要**在此之上加新接口（教训见 git log）。
- §9.1 的教训值得记住：**Intel 上 GPU timestamp 的占比会撒谎**，已有后来者想"优化 LSTM"
  的话，先让他看 §9。
