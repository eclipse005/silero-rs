//! 厂商无关的超越函数层 —— [`crate::wgsl_math.wgsl`] 的 host 侧镜像实现。
//!
//! 两份实现必须是**同一算法的两份镜像**：host 与 GPU 各自偏离参考的差异同样会被
//! silero 不稳定的 LSTM 递归放大（实测 1525s 音频上 1 ULP → 0.75 概率差，时间戳
//! 567 → 555 段）。`tests/math_parity.rs` 在真实 GPU 上校验两端一致（容差 <= 4 ULP，
//! 差额来自 GPU 侧的 FMA 收缩，见 wgsl_math.wgsl 头注），本模块的单元测试则对拍
//! libm 要求 <= 1 ULP。
//!
//! 注意 `round_ties_even`：WGSL 的 `round()` 是**就近偶数**，Rust 的 `f32::round()`
//! 是**远离零**，必须用 `round_ties_even()` 才与 WGSL 对齐。

pub const TF_LN2_HI: f32 = 6.931_457_5e-1;
pub const TF_LN2_LO: f32 = 1.428_606_8e-6;
pub const TF_INV_LN2: f32 = 1.442_695;

// exp(r) = 1 + r + r²·Q(r)，Q = Σ_{k=0..6} r^k / (k+2)!
const TF_Q0: f32 = 0.5;                            // 1/2!
const TF_Q1: f32 = 1.666_666_6e-1;                 // 1/3!
const TF_Q2: f32 = 4.166_666_5e-2;                 // 1/4!
const TF_Q3: f32 = 8.333_333e-3;                   // 1/5!
const TF_Q4: f32 = 1.388_888_9e-3;                 // 1/6!
const TF_Q5: f32 = 1.984_127e-4;                   // 1/7!
const TF_Q6: f32 = 2.480_158_7e-5;                 // 1/8!

/// 只用 `+ - * /` 的 `exp`（跨厂商逐位一致，见 `wgsl_math.wgsl` 头注）。
pub fn exp(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    // 溢出饱和到 FLT_MAX；下溢门取结果低于最小次正规数之处。
    // （silero 的 sigmoid/tanh 只会传入非正参数，溢出分支实际不可达，
    //   这里仍保持函数自身正确，且两侧 gate 逐位对称。）
    if x >= 88.722_84 {
        return f32::MAX;
    }
    if x <= -103.972_09 {
        return 0.0;
    }
    // 与 WGSL `round()` 同语义：就近偶数
    let nf = (x * TF_INV_LN2).round_ties_even();
    // Cody-Waite 双项拆分：n*ln2_hi 精确（ln2_hi 仅 13 位有效位，|n|<=128 时乘积
    // 不丢位），残差再做一次减法得到 |r| <= ln2/2。
    let r = (x - nf * TF_LN2_HI) - nf * TF_LN2_LO;

    // exp(r) = 1 + r + r²·Q(r)：把一阶项单独提出，Horner 只作用在 Q 上，
    // 最后 1 + (r + r²Q) 只舍入一次 —— 这是 f32 下达到 ~1 ULP 的关键。
    // Q 取 exp 的泰勒余项 1/2! + r/3! + ... + r^7/8!（|r|<=ln2/2 时截断误差
    // r^9/8! ≈ 1.8e-9，相对误差远小于 1 ULP）。
    let r2 = r * r;
    let q = TF_Q0 + r * (TF_Q1 + r * (TF_Q2 + r * (TF_Q3 + r * (TF_Q4 + r * (TF_Q5 + r * TF_Q6)))));
    let y = 1.0 + (r + r2 * q);

    // 2^n 缩放：按位构造指数域，避免 powf（powf 同样不可跨厂商复现）。
    // n=127 合法（y∈[1,2) → 结果 < 2^128，指数域 254，仍可表示）；
    // n=128 才会让指数域溢出到 255。
    let ni = nf as i32;
    if ni >= 128 {
        return f32::MAX;
    }
    if ni >= -126 {
        return y * f32::from_bits(((ni + 127) as u32) << 23);
    }
    if ni >= -149 {
        // 次正规区：2^n = 2^(n+64)·2^-64，两步缩放且中间值仍在正规范围
        let s1 = f32::from_bits(((ni + 191) as u32) << 23);
        return (y * s1) * f32::from_bits(63u32 << 23); // 2^-64
    }
    0.0
}

/// `exp(x) - 1`，与 [`exp`] 共用区间归约，但保留 `exp(r)-1` 的低幅部分，
/// 使 `x → 0` 时不发生相消（`n == 0` 时直接返回 `r + r²Q`，精度是满的）。
pub fn expm1(x: f32) -> f32 {
    if x >= 88.722_84 {
        return f32::MAX;
    }
    if x <= -103.972_09 {
        return -1.0;
    }
    let nf = (x * TF_INV_LN2).round_ties_even();
    let r = (x - nf * TF_LN2_HI) - nf * TF_LN2_LO;
    let r2 = r * r;
    let q = TF_Q0 + r * (TF_Q1 + r * (TF_Q2 + r * (TF_Q3 + r * (TF_Q4 + r * (TF_Q5 + r * TF_Q6)))));
    let m1 = r + r2 * q; // == exp(r) - 1，无相消

    let ni = nf as i32;
    if ni == 0 {
        return m1;
    }
    // expm1 = 2^n·(1+m1) - 1 = 2^n·m1 + (2^n - 1)。
    // n<0 时两项同为负、n>0 时两项同为正，均不发生相消。
    if ni >= -126 {
        let s = f32::from_bits(((ni + 127) as u32) << 23);
        return s * m1 + (s - 1.0);
    }
    if ni >= -149 {
        let s1 = f32::from_bits(((ni + 191) as u32) << 23);
        let s = s1 * f32::from_bits(63u32 << 23);
        return s * m1 + (s - 1.0);
    }
    -1.0
}

/// 稳定形式 sigmoid（与 `wgsl_math.wgsl` 的 `tf_sigmoid` 对应）。
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        return 1.0 / (1.0 + exp(-x));
    }
    let e = exp(x);
    e / (1.0 + e)
}

/// `tanh(x) = -expm1(-2x) / (expm1(-2x) + 2)`（与 `wgsl_math.wgsl` 的 `tf_tanh` 对应）。
///
/// **不能**写成 `2*sigmoid(2x) - 1`：sigmoid 在 0.5 附近的绝对舍入误差（~2^-25）
/// 会被减法放大成相对误差，实测 x≈0.01 处可达 64 ULP。LSTM 的 gate 与 cell 值经常
/// 落在 0 附近，那种病态会直接喂进递归。
pub fn tanh(x: f32) -> f32 {
    if x >= 20.0 {
        return 1.0;
    }
    if x <= -20.0 {
        return -1.0;
    }
    let u = expm1(-2.0 * x);
    -u / (u + 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 与 libm 对拍：正规结果区要求 <= 1 ULP；次正规区 ULP 间距本身是 2^-149，
    /// 改用绝对容差（几个最小次正规数），否则整数差会被放大到无意义的量级。
    #[test]
    fn exp_matches_reference() {
        let subnormal_tol = 4.0 * f32::from_bits(1);
        let mut worst = 0i64;
        let mut worst_x = 0.0f32;
        let mut x = -88.0f32;
        while x <= 88.0 {
            let got = exp(x);
            let want = x.exp();
            assert!(got.is_finite(), "exp({x}) not finite");
            if want.abs() >= f32::MIN_POSITIVE {
                let ulp = (got.to_bits() as i64 - want.to_bits() as i64).abs();
                if ulp > worst {
                    worst = ulp;
                    worst_x = x;
                }
            } else {
                assert!(
                    (got - want).abs() <= subnormal_tol,
                    "exp({x}) 次正规区误差过大: {got} vs {want}"
                );
            }
            x += 0.00037;
        }
        assert!(
            worst <= 1,
            "exp 正规区偏离 libm 最多 {worst} ULP（x={worst_x}），需 <= 1 ULP"
        );
    }

    /// sigmoid / tanh 相对 libm 对拍。
    #[test]
    fn activations_match_reference() {
        let mut x = -30.0f32;
        while x <= 30.0 {
            let s = sigmoid(x);
            let s_ref = 1.0 / (1.0 + (-x).exp());
            assert!(
                (s - s_ref).abs() <= 2e-7,
                "sigmoid({x}) = {s} vs {s_ref}"
            );
            let t = tanh(x);
            let t_ref = x.tanh();
            assert!((t - t_ref).abs() <= 2e-7, "tanh({x}) = {t} vs {t_ref}");
            x += 0.00037;
        }
    }

    #[test]
    fn sigmoid_edges() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert!(sigmoid(100.0) == 1.0);
        // exp(-100) = 3.7e-44 是次正规数，保留而非冲零（跨厂商一致优先）
        assert!(sigmoid(-100.0) > 0.0 && sigmoid(-100.0) < 1e-38);
        let s = sigmoid(-1.0);
        assert!((s - 0.268_941_4).abs() < 1e-6, "got {s}");
    }

    #[test]
    fn tanh_edges() {
        assert_eq!(tanh(0.0), 0.0);
        assert!(tanh(50.0) == 1.0);
        assert!(tanh(-50.0) == -1.0);
        let t = tanh(1.0);
        assert!((t - 0.761_594_2).abs() < 1e-6, "got {t}");
    }

    #[test]
    fn exp_never_nonfinite() {
        for k in -120..=120 {
            let x = k as f32;
            assert!(exp(x).is_finite(), "exp({x}) not finite");
        }
    }
}
