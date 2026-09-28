// 厂商无关的超越函数层（exp / expm1 / sigmoid / tanh），host 与 GPU 共用同一份定义
// （`src/transcendental.rs` 是同一算法的 Rust 镜像）。
//
// 为什么需要它：silero 的 LSTM 是**指数不稳定**的递归 —— 实测同一条 1525s 音频上，
// Intel 与 NVIDIA 跑同一份 WGSL，概率从第 0 帧就差 1 ULP，到 frame 314 放大到 1e-6、
// frame 12789 放大到 0.75，时间戳因此从 567 段变成 555 段。根因是两家的硬件
// `exp`/`tanh` 实现差 1–2 ULP（实测 275 个输入里 exp 有 94 个、tanh 有 110 个不一致；
// 同一探针也确认**不是**次正规刷零 —— 次正规输出为 0）。
//
// 本文件因此用**只含 + - * /** 的形式重新实现这些函数。已实测该基础算术在
// Intel/NVIDIA 上逐位一致，所以结果不再随 GPU 厂商漂移。
//
// **但不要指望跨厂商逐位相同**：WGSL 没有 `precise` 限定符，naga 也不发射 SPIR-V 的
// `NoContraction`，于是驱动会把 Horner 链上的 `a*b+c` 收缩成 FMA（单次舍入），而
// Rust/LLVM 默认不收缩（两次舍入）。实测两端最大差 4 ULP —— 这条差异无法从 shader
// 源码侧消除，因此 `tests/math_parity.rs` 把契约定在 <= 4 ULP 并守住算法不漂移，
// 真正的正确性保证是长音频端到端 align（`tests/long_audio.rs`）。
//
// 算法：Cody-Waite 双项区间归约 + exp(r) = 1 + r + r²·Q(r)（Q 为泰勒余项
// 1/2! + r/3! + … + r^7/8!）。把一阶项单独提出后，最后一次加法只舍入一次，
// 在 f32 下可稳定达到 ~1 ULP（`transcendental::tests` 对拍 libm 实测 <= 1 ULP）。
//
// 边界区行为（有意为之，且不参与 silero 计算）：
//   - x >= 88.72284   → 返回 FLT_MAX（真值已溢出；silero 的 sigmoid 只关心
//                      「饱和到 1」，FLT_MAX 足够且避免 inf 传播）
//   - x <= -103.97209 → 返回 0（结果已低于 f32 最小次正规数）
//   - n < -126        → 次正规结果，用 2^(n+64)·2^-64 两步缩放正确生成，而非直接冲零

const TF_LN2_HI: f32 = 6.93145751953125e-01;      // 0x3f317200，尾数仅 12 bit → 乘 n（|n|<=128）精确
const TF_LN2_LO: f32 = 1.42860682030941723212e-06;
const TF_INV_LN2: f32 = 1.44269504088896338700e+00;

// exp(r) = 1 + r + r²·Q(r)，Q = Σ_{k=0..6} r^k / (k+2)!
const TF_Q0: f32 = 0.5;                    // 1/2!
const TF_Q1: f32 = 1.6666666e-1;           // 1/3!
const TF_Q2: f32 = 4.1666665e-2;           // 1/4!
const TF_Q3: f32 = 8.333333e-3;            // 1/5!
const TF_Q4: f32 = 1.3888889e-3;           // 1/6!
const TF_Q5: f32 = 1.984127e-4;            // 1/7!
const TF_Q6: f32 = 2.4801587e-5;           // 1/8!

fn tf_exp(x: f32) -> f32 {
    if (x >= 88.72284) { return 3.4028234663852886e+38; }
    if (x <= -103.97209) { return 0.0; }

    // 区间归约：r = x - n*ln2hi - n*ln2lo，|r| <= ln2/2
    let nf = round(x * TF_INV_LN2);   // WGSL round() = 就近偶数，与 Rust round_ties_even 对齐
    let r = (x - nf * TF_LN2_HI) - nf * TF_LN2_LO;

    // exp(r) = 1 + r + r²·Q(r)：一阶项单独提出，Horner 只作用在 Q 上，
    // 最后 1 + (r + r²Q) 只舍入一次 —— f32 下达到 ~1 ULP 的关键。
    let r2 = r * r;
    let q = TF_Q0 + r * (TF_Q1 + r * (TF_Q2 + r * (TF_Q3 + r * (TF_Q4 + r * (TF_Q5 + r * TF_Q6)))));
    let y = 1.0 + (r + r2 * q);

    // 2^n 缩放（按位构造，避免 powf —— powf 同样是不可复现的超越函数）
    // n=127 合法（y∈[1,2) → 结果 < 2^128，指数域 254 仍可表示）；n=128 才溢出。
    let ni = i32(nf);
    if (ni >= 128) { return 3.4028234663852886e+38; }
    if (ni >= -126) {
        return y * bitcast<f32>(u32(ni + 127) << 23u);
    }
    if (ni >= -149) {
        // 次正规区：2^n = 2^(n+64)·2^-64，两步缩放且中间值仍在正规范围
        let s1 = bitcast<f32>(u32(ni + 191) << 23u);
        return (y * s1) * bitcast<f32>(u32(63) << 23u); // 2^-64
    }
    return 0.0;
}

/// exp(x) - 1。与 `tf_exp` 共用区间归约，但**保留** exp(r)-1 的低幅部分，
/// 使 x→0 时不发生相消：n==0 时 expm1(x) 直接就是 `r + r²Q`，精度是满的。
fn tf_expm1(x: f32) -> f32 {
    if (x >= 88.72284) { return 3.4028234663852886e+38; }
    if (x <= -103.97209) { return -1.0; }

    let nf = round(x * TF_INV_LN2);
    let r = (x - nf * TF_LN2_HI) - nf * TF_LN2_LO;
    let r2 = r * r;
    let q = TF_Q0 + r * (TF_Q1 + r * (TF_Q2 + r * (TF_Q3 + r * (TF_Q4 + r * (TF_Q5 + r * TF_Q6)))));
    let m1 = r + r2 * q;              // == exp(r) - 1，无相消

    let ni = i32(nf);
    if (ni == 0) { return m1; }       // x 已在主区间内

    // expm1 = 2^n·(1+m1) - 1 = 2^n·m1 + (2^n - 1)
    // n<0 时两项同号（均为负），n>0 时两项同号（均为正），均不发生相消。
    if (ni >= -126) {
        let s = bitcast<f32>(u32(ni + 127) << 23u);
        return s * m1 + (s - 1.0);
    }
    if (ni >= -149) {
        let s1 = bitcast<f32>(u32(ni + 191) << 23u);
        let s = s1 * bitcast<f32>(u32(63) << 23u);
        return s * m1 + (s - 1.0);
    }
    return -1.0;
}

/// 数值稳定形式的 sigmoid：x >= 0 走 1/(1+e^-x)，x < 0 走 e^x/(1+e^x)，
/// 避免 exp(-x) 在 x 很负时溢出成 inf。
fn tf_sigmoid(x: f32) -> f32 {
    if (x >= 0.0) {
        return 1.0 / (1.0 + tf_exp(-x));
    }
    let e = tf_exp(x);
    return e / (1.0 + e);
}

/// tanh(x) = -expm1(-2x) / (expm1(-2x) + 2)。
///
/// **不能**用 `2·sigmoid(2x) - 1`：sigmoid 在 0.5 附近的绝对舍入误差（~2^-25）
/// 会被减法放大成相对误差，实测 x≈0.01 处可达 64 ULP。LSTM 的 gate 与 cell 值
/// 经常落在 0 附近，那种病态会直接喂进递归。
fn tf_tanh(x: f32) -> f32 {
    if (x >= 20.0) { return 1.0; }    // tanh(20) 在 f32 下已是精确的 1.0
    if (x <= -20.0) { return -1.0; }
    let u = tf_expm1(-2.0 * x);
    return -u / (u + 2.0);
}
