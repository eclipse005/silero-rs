//! Silero VAD（v6.2.3 包内置 jit 模型）纯 Rust 移植 —— host CPU 路径，16k/8k 双配置。
//!
//! 语义来源：`silero_vad.jit` 的 TorchScript 反编译（phase0 阶段逐模块 dump 确认）。
//! ONNX 变体（op15/op16/op18/openvino/sequence）与 jit 逐位同权重（已验证），
//! 仅 context/state 的管理位置不同（外部 vs 内部），逐帧数学一致。
//!
//! 权重来源（export_for_rust.py / models/）：
//! - W_A `ref/silero_vad_16k_jit.safetensors`：jit16k + 全部 ONNX 变体共享
//! - W_B `ref/weights_half16k.safetensors`：half.onnx（独立权重集）
//! - W_C `models/silero_vad_16k.safetensors`：tinygrad 权重（键名即规范格式）
//! - W_8k `ref/weights_jit8k.safetensors`：jit 8k 分支

pub mod timestamps;
pub mod vad_iterator;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "gpu")]
pub mod gpu_batch;

/// 全部 f32 权重（行主序，与 safetensors 存储一致）。
pub struct Weights {
    pub basis: Vec<f32>, // (2*cutoff, n_fft)
    pub c1w: Vec<f32>,   // (128, cutoff*3)
    pub c1b: Vec<f32>,   // (128,)
    pub c2w: Vec<f32>,   // (64, 128*3)
    pub c2b: Vec<f32>,   // (64,)
    pub c3w: Vec<f32>,   // (64, 64*3)
    pub c3b: Vec<f32>,   // (64,)
    pub c4w: Vec<f32>,   // (128, 64*3)
    pub c4b: Vec<f32>,   // (128,)
    pub wih: Vec<f32>,   // (512, 128)
    pub whh: Vec<f32>,   // (512, 128)
    pub bih: Vec<f32>,   // (512,)
    pub bhh: Vec<f32>,   // (512,)
    pub fw: Vec<f32>,    // (128,)
    pub fb: Vec<f32>,    // (1,)
}

/// 采样率配置（jit 的 _model / _model_8k 两分支）。
#[derive(Clone, Copy, Debug)]
pub struct ModelCfg {
    pub sr: u32,
    pub frame: usize,  // 512@16k / 256@8k
    pub ctx: usize,    // 64@16k / 32@8k
    pub pad: usize,    // reflect pad（= n_fft/4）
    pub n_fft: usize,  // 256@16k / 128@8k
    pub hop: usize,    // 128@16k / 64@8k
    pub cutoff: usize, // 129@16k / 65@8k
}

pub const CFG_16K: ModelCfg = ModelCfg {
    sr: 16000,
    frame: 512,
    ctx: 64,
    pad: 64,
    n_fft: 256,
    hop: 128,
    cutoff: 129,
};

pub const CFG_8K: ModelCfg = ModelCfg {
    sr: 8000,
    frame: 256,
    ctx: 32,
    pad: 32,
    n_fft: 128,
    hop: 64,
    cutoff: 65,
};

impl ModelCfg {
    pub fn x_len(&self) -> usize {
        self.ctx + self.frame + self.pad // 640 / 320
    }
    pub fn frames_t(&self) -> usize {
        (self.ctx + self.frame + self.pad - self.n_fft) / self.hop + 1 // 4
    }
}

impl Weights {
    /// 从 safetensors 文件加载（数据 f32 LE）。
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let raw = std::fs::read(path)?;
        Self::from_bytes(&raw)
    }

    /// 从内存字节加载（crate 内嵌权重用）。
    pub fn from_bytes(raw: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let st = safetensors::SafeTensors::deserialize(raw)?;

        fn take(
            st: &safetensors::SafeTensors,
            name: &str,
            expect: Option<usize>,
        ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            let t = st.tensor(name)?;
            if t.dtype() != safetensors::Dtype::F32 {
                return Err(format!("{name}: dtype != F32").into());
            }
            let data = t.data();
            if let Some(expect) = expect {
                if data.len() != expect * 4 {
                    return Err(format!(
                        "{name}: {} bytes, expect {expect} f32",
                        data.len()
                    )
                    .into());
                }
            }
            Ok(data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect())
        }

        Ok(Self {
            basis: take(&st, "stft_conv.weight", None)?, // 长度由 cfg 校验（16k/8k 不同）
            c1w: take(&st, "conv1.weight", None)?,
            c1b: take(&st, "conv1.bias", Some(128))?,
            c2w: take(&st, "conv2.weight", Some(64 * 128 * 3))?,
            c2b: take(&st, "conv2.bias", Some(64))?,
            c3w: take(&st, "conv3.weight", Some(64 * 64 * 3))?,
            c3b: take(&st, "conv3.bias", Some(64))?,
            c4w: take(&st, "conv4.weight", Some(128 * 64 * 3))?,
            c4b: take(&st, "conv4.bias", Some(128))?,
            wih: take(&st, "lstm_cell.weight_ih", Some(512 * 128))?,
            whh: take(&st, "lstm_cell.weight_hh", Some(512 * 128))?,
            bih: take(&st, "lstm_cell.bias_ih", Some(512))?,
            bhh: take(&st, "lstm_cell.bias_hh", Some(512))?,
            fw: take(&st, "final_conv.weight", Some(128))?,
            fb: take(&st, "final_conv.bias", Some(1))?,
        })
    }

    /// 内嵌官方 16k 权重（crate 开箱即用；由 wgpu/export_for_rust.py 从
    /// models/silero_vad.jit 导出，与 Python 原版逐位同权重）。
    pub fn embedded_16k() -> Result<Self, Box<dyn std::error::Error>> {
        Self::from_bytes(include_bytes!("../assets/silero_vad_16k_jit.safetensors"))
    }
}

/// 定点 dot：AVX2+FMA（8 lane 累加，固定水平归约顺序），SILERO_SIMD=0 强制标量。
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if simd_enabled() {
            // SAFETY: is_x86_feature_detected 已确认 avx2+fma
            return unsafe { dot_avx2(a, b) };
        }
    }
    dot_scalar(a, b)
}

fn simd_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SILERO_SIMD").map(|v| v != "0").unwrap_or(true)
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
    })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps};
    let n = a.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        acc = _mm256_fmadd_ps(
            _mm256_loadu_ps(a.as_ptr().add(i)),
            _mm256_loadu_ps(b.as_ptr().add(i)),
            acc,
        );
        i += 8;
    }
    // 固定顺序水平归约：((l0+l1)+(l2+l3)) + ((l4+l5)+(l6+l7))
    let arr: [f32; 8] = std::mem::transmute(acc);
    let mut s = (arr[0] + arr[1]) + (arr[2] + arr[3]);
    s += (arr[4] + arr[5]) + (arr[6] + arr[7]);
    let mut tail = 0.0f32;
    while i < n {
        tail += a[i] * b[i];
        i += 1;
    }
    s + tail
}

fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// 单流 VAD：wrapper（context/state）+ 模型前向，16k/8k 通用。
/// conv 权重在构造时预转置为 (oc, 3, ic)——推理热路径零 gather（手册 §5.2/§5.4）。
pub struct SileroVad {
    w: Weights,
    cfg: ModelCfg,
    wt1: Vec<f32>, // (128, 3, cutoff)
    wt2: Vec<f32>, // (64, 3, 128)
    wt3: Vec<f32>, // (64, 3, 64)
    wt4: Vec<f32>, // (128, 3, 64)
    ctx: Vec<f32>,
    h: Vec<f32>,
    c: Vec<f32>,
    x: Vec<f32>,
    mag_t: Vec<f32>, // (t, cutoff)
    e1_t: Vec<f32>,  // (t, 128)
    e2_t: Vec<f32>,  // (2, 64)
    e3_t: Vec<f32>,  // (1, 64)
    e4_t: Vec<f32>,  // (1, 128)
    gates: Vec<f32>,
    hn: Vec<f32>,
    cn: Vec<f32>,
}

impl SileroVad {
    pub fn new(w: Weights, cfg: ModelCfg) -> Result<Self, String> {
        let t = cfg.frames_t();
        let basis_len = 2 * cfg.cutoff * cfg.n_fft;
        if w.basis.len() != basis_len {
            return Err(format!(
                "basis len {} != 2*cutoff*n_fft = {}",
                w.basis.len(),
                basis_len
            ));
        }
        let c1_len = 128 * cfg.cutoff * 3;
        if w.c1w.len() != c1_len {
            return Err(format!("conv1.weight len {} != {c1_len}", w.c1w.len()));
        }
        // 预转置 conv 权重 (oc, ic, 3) → (oc, 3, ic)：内层变 ic 连续
        let transpose_w = |w: &[f32], oc: usize, ic: usize| -> Vec<f32> {
            let mut out = vec![0.0; oc * 3 * ic];
            for co in 0..oc {
                for k in 0..3 {
                    for i in 0..ic {
                        out[co * 3 * ic + k * ic + i] = w[co * ic * 3 + i * 3 + k];
                    }
                }
            }
            out
        };
        Ok(Self {
            wt1: transpose_w(&w.c1w, 128, cfg.cutoff),
            wt2: transpose_w(&w.c2w, 64, 128),
            wt3: transpose_w(&w.c3w, 64, 64),
            wt4: transpose_w(&w.c4w, 128, 64),
            w,
            cfg,
            ctx: vec![0.0; cfg.ctx],
            h: vec![0.0; 128],
            c: vec![0.0; 128],
            x: vec![0.0; cfg.x_len()],
            mag_t: vec![0.0; t * cfg.cutoff],
            e1_t: vec![0.0; t * 128],
            e2_t: vec![0.0; 2 * 64],
            e3_t: vec![0.0; 64],
            e4_t: vec![0.0; 128],
            gates: vec![0.0; 512],
            hn: vec![0.0; 128],
            cn: vec![0.0; 128],
        })
    }

    /// 等效 wrapper.reset_states()。
    pub fn reset(&mut self) {
        self.ctx.iter_mut().for_each(|v| *v = 0.0);
        self.h.iter_mut().for_each(|v| *v = 0.0);
        self.c.iter_mut().for_each(|v| *v = 0.0);
    }

    pub fn cfg(&self) -> ModelCfg {
        self.cfg
    }

    /// 一帧（cfg.frame 样本），返回语音概率。
    pub fn frame(&mut self, chunk: &[f32]) -> f32 {
        let cfg = self.cfg;
        assert_eq!(chunk.len(), cfg.frame, "wrong frame size for cfg");
        let t = cfg.frames_t();
        let head = cfg.ctx + cfg.frame; // 576 / 288
        let x = &mut self.x;
        x[..cfg.ctx].copy_from_slice(&self.ctx);
        x[cfg.ctx..head].copy_from_slice(chunk);
        for i in 0..cfg.pad {
            x[head + i] = x[head - 2 - i];
        }

        // STFT → magnitude (t, cutoff) 行主序（t 主序直接喂 conv）
        let basis = &self.w.basis;
        for p in 0..t {
            let w0 = p * cfg.hop;
            let win = &x[w0..w0 + cfg.n_fft];
            for f in 0..cfg.cutoff {
                let re = dot(win, &basis[f * cfg.n_fft..(f + 1) * cfg.n_fft]);
                let im = dot(
                    win,
                    &basis[(cfg.cutoff + f) * cfg.n_fft..(cfg.cutoff + f + 1) * cfg.n_fft],
                );
                self.mag_t[p * cfg.cutoff + f] = (re * re + im * im).sqrt();
            }
        }

        // encoder：每块 conv → ReLU（窗口点积，零 gather）
        conv3_t(&self.mag_t, t, cfg.cutoff, &self.wt1, &self.w.c1b, 1, true, &mut self.e1_t);
        conv3_t(&self.e1_t, t, 128, &self.wt2, &self.w.c2b, 2, true, &mut self.e2_t);
        conv3_t(&self.e2_t, 2, 64, &self.wt3, &self.w.c3b, 2, true, &mut self.e3_t);
        conv3_t(&self.e3_t, 1, 64, &self.wt4, &self.w.c4b, 1, true, &mut self.e4_t);

        // lstm_cell（块序 i,f,g,o）
        let e4 = &self.e4_t;
        for r in 0..512 {
            let gi = dot(&self.w.wih[r * 128..(r + 1) * 128], e4) + self.w.bih[r];
            let gh = dot(&self.w.whh[r * 128..(r + 1) * 128], &self.h) + self.w.bhh[r];
            self.gates[r] = gi + gh;
        }
        for j in 0..128 {
            let i = sigmoid(self.gates[j]);
            let f = sigmoid(self.gates[128 + j]);
            let g = self.gates[256 + j].tanh();
            let o = sigmoid(self.gates[384 + j]);
            self.cn[j] = f * self.c[j] + i * g;
            self.hn[j] = o * self.cn[j].tanh();
        }
        std::mem::swap(&mut self.h, &mut self.hn);
        std::mem::swap(&mut self.c, &mut self.cn);

        // ReLU → conv1x1 → sigmoid → mean(T=1)
        let mut z = self.w.fb[0];
        for j in 0..128 {
            let a = if self.h[j] > 0.0 { self.h[j] } else { 0.0 };
            z += self.w.fw[j] * a;
        }
        let p = sigmoid(z);

        // context = (ctx+chunk)[−ctx:] = chunk 尾部
        let tail = cfg.frame - cfg.ctx;
        self.ctx.copy_from_slice(&chunk[tail..]);
        p
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// conv1d(k=3, pad=1)：输入 (T_in, IC) 行主序，权重预转置 (OC, 3, IC)，
/// 输出 (T_out, OC)。每输出 = 3 次连续窗口 dot，零 gather、零分支写。
fn conv3_t(
    input_t: &[f32],
    t_in: usize,
    ic: usize,
    wt: &[f32],
    b: &[f32],
    stride: usize,
    relu: bool,
    out_t: &mut Vec<f32>,
) {
    let t_out = (t_in - 1) / stride + 1;
    let oc = wt.len() / (3 * ic);
    debug_assert_eq!(wt.len(), oc * 3 * ic);
    out_t.clear();
    out_t.resize(t_out * oc, 0.0);
    for t in 0..t_out {
        for co in 0..oc {
            let wb = &wt[co * 3 * ic..(co + 1) * 3 * ic];
            let mut acc = b[co];
            for k in 0..3 {
                let row = t * stride + k; // 实际输入行 = row - 1
                if row >= 1 && row <= t_in {
                    let r = (row - 1) * ic;
                    acc += dot(&wb[k * ic..(k + 1) * ic], &input_t[r..r + ic]);
                }
            }
            let v = acc;
            out_t[t * oc + co] = if relu && v < 0.0 { 0.0 } else { v };
        }
    }
}

// ---------- API 对齐层（对应原版 silero_vad.utils_vad 的顶层函数） ----------

/// 一步式：整段音频 → 语音段时间戳（样本对），等价原版
/// `get_speech_timestamps(audio, model, ...)`（默认参数组合见 [`TsParams`]）。
/// 概率路径与门禁一致：内部 reset + 逐帧（末帧补零）+ 离线状态机。
pub fn get_speech_timestamps(
    vad: &mut SileroVad,
    audio: &[f32],
    params: &timestamps::TsParams,
) -> Vec<(i64, i64)> {
    let cfg = vad.cfg();
    let frame = cfg.frame as usize;
    let n_frames = audio.len().div_ceil(frame);
    let mut buf = vec![0.0f32; frame];
    let mut probs = Vec::with_capacity(n_frames);
    vad.reset();
    for f in 0..n_frames {
        let s = f * frame;
        let e = (s + frame).min(audio.len());
        buf[..e - s].copy_from_slice(&audio[s..e]);
        buf[e - s..].fill(0.0);
        probs.push(vad.frame(&buf));
    }
    timestamps::speech_timestamps_from_probs(
        &probs,
        cfg.sr as i64,
        cfg.frame as i64,
        Some(audio.len() as i64),
        params,
    )
}

/// 样本 → 秒（等价原版 `return_seconds=True` 的 `round(x / sr, time_resolution)`；
/// Python round 是 ties-even，与 f64::round（half-away）不同，用 round_ties_even 对齐）。
pub fn to_seconds(x: i64, sr: i64, time_resolution: u32) -> f64 {
    let p = 10f64.powi(time_resolution as i32);
    let v = x as f64 / sr as f64 * p;
    v.round_ties_even() / p
}

/// 等价原版 `collect_chunks(tss, wav)`：按时间戳（样本对）拼接语音段。
pub fn collect_chunks(tss: &[(i64, i64)], audio: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    for (s, e) in tss {
        let s = (*s as usize).min(audio.len());
        let e = (*e as usize).min(audio.len());
        if e > s {
            out.extend_from_slice(&audio[s..e]);
        }
    }
    out
}
