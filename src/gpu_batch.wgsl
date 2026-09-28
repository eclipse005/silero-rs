// Silero VAD 16k 批量 GPU kernels —— {BATCH} 帧一次提交（模板注入 BATCH 常量）。
//
// 设计（因材施教预算分析）：
// - conv/stft 逐帧独立：T 帧仍是 5 个 dispatch，线程数 ×T
// - gates_in（Wih@e4 全帧）1 个批量 dispatch；whh·h 留给递归步
// - LSTM 递归步：单 workgroup(512) fused kernel —— 全组算 whh·h 进 workgroup
//   storage → barrier → 128 线程 pointwise 原地更新 h/c → 归约出 prob[t]
// - 每 BATCH 帧：6 + BATCH 个 dispatch、1 次 submit、1 次 map(T*4B)
//
// 布局：所有批量张量按 (帧 t, 帧内偏移) 行主序，行尾 1 个垫片槽
// （读者 kernel 写垫片保持 rw 用途，wgpu 用途域规则）。

const BATCH: u32 = {BATCH}u;

@group(0) @binding(0) var<storage, read> b_x: array<f32>;       // BATCH*640
@group(0) @binding(1) var<storage, read> b_w: array<f32>;       // 权重 concat（区域见 rust 侧）
@group(0) @binding(2) var<storage, read_write> b_mag: array<f32>; // BATCH*517

// W 区域偏移（f32）：basis=0, c1w=66048, c1b=115584, c2w=115712, c2b=140288,
// c2b 后：c3w=140352, c3b=152704, c3b 尾：c4w=152768, c4b=177280, c4b 尾：
// wih=177408, whh=242944, bih=308480, bhh=308992, bhh 尾：fw=309504, fb=309632
// 由 rust 侧传入实际偏移 —— 这里用 const 注入更稳：
// （占位，rust 侧 format 时替换 {OFF_*}）
const OFF_C1W: u32 = {OFF_C1W}u;
const OFF_C1B: u32 = {OFF_C1B}u;
const OFF_C2W: u32 = {OFF_C2W}u;
const OFF_C2B: u32 = {OFF_C2B}u;
const OFF_C3W: u32 = {OFF_C3W}u;
const OFF_C3B: u32 = {OFF_C3B}u;
const OFF_C4W: u32 = {OFF_C4W}u;
const OFF_C4B: u32 = {OFF_C4B}u;
const OFF_WIH: u32 = {OFF_WIH}u;
const OFF_WHH: u32 = {OFF_WHH}u;
const OFF_BIH: u32 = {OFF_BIH}u;
const OFF_BHH: u32 = {OFF_BHH}u;
const OFF_FW: u32 = {OFF_FW}u;
const OFF_FB: u32 = {OFF_FB}u;

// ---------- 批量 STFT：x(BATCH*640) → mag(BATCH*517，行内 (p,f)) ----------
@compute @workgroup_size(64)
fn main_stft(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= BATCH * 516u) { return; }
    let t = idx / 516u;
    let r = idx % 516u;
    let f = r / 4u;
    let p = r % 4u;
    let w0 = t * 640u + p * 128u;
    var re = 0.0;
    var im = 0.0;
    let base_r = f * 256u;
    let base_i = (129u + f) * 256u;
    for (var k = 0u; k < 256u; k = k + 1u) {
        let v = b_x[w0 + k];
        re = re + b_w[base_r + k] * v;
        im = im + b_w[base_i + k] * v;
    }
    b_mag[t * 517u + p * 129u + f] = sqrt(re * re + im * im);
}

// ---------- 批量 conv1：mag(4,129) → e1(4,128) ----------
@group(0) @binding(3) var<storage, read_write> b_e1: array<f32>; // BATCH*513

@compute @workgroup_size(64)
fn main_conv1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let t = idx / 512u;
    let r = idx % 512u;
    b_mag[t * 517u + 516u] = 0.0; // 垫片
    if (idx >= BATCH * 512u) { return; }
    let p4 = r / 128u;
    let co = r % 128u;
    var acc = b_w[OFF_C1B + co];
    for (var k = 0u; k < 3u; k = k + 1u) {
        let p = p4 + k;
        if (p >= 1u && p <= 4u) {
            let row = (p - 1u) * 129u;
            for (var ic = 0u; ic < 129u; ic = ic + 1u) {
                acc = acc + b_w[OFF_C1W + co * 387u + ic * 3u + k] * b_mag[t * 517u + row + ic];
            }
        }
    }
    b_e1[t * 513u + p4 * 128u + co] = max(acc, 0.0);
}

// ---------- 批量 conv2：e1(4,128) → e2(2,64)，stride 2 ----------
@group(0) @binding(4) var<storage, read_write> b_e2: array<f32>; // BATCH*129

@compute @workgroup_size(64)
fn main_conv2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let t = idx / 128u;
    let r = idx % 128u;
    b_e1[t * 513u + 512u] = 0.0;
    if (idx >= BATCH * 128u) { return; }
    let p2 = r / 64u;
    let co = r % 64u;
    var acc = b_w[OFF_C2B + co];
    for (var k = 0u; k < 3u; k = k + 1u) {
        let p = p2 * 2u + k;
        if (p >= 1u && p <= 4u) {
            let row = (p - 1u) * 128u;
            for (var ic = 0u; ic < 128u; ic = ic + 1u) {
                acc = acc + b_w[OFF_C2W + co * 384u + ic * 3u + k] * b_e1[t * 513u + row + ic];
            }
        }
    }
    b_e2[t * 129u + p2 * 64u + co] = max(acc, 0.0);
}

// ---------- 批量 conv3：e2(2,64) → e3(1,64)，stride 2 ----------
@group(0) @binding(5) var<storage, read_write> b_e3: array<f32>; // BATCH*65

@compute @workgroup_size(64)
fn main_conv3(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let t = idx / 64u;
    let co = idx % 64u;
    b_e2[t * 129u + 128u] = 0.0;
    if (idx >= BATCH * 64u) { return; }
    var acc = b_w[OFF_C3B + co];
    for (var k = 0u; k < 3u; k = k + 1u) {
        let p = k;
        if (p >= 1u && p <= 2u) {
            let row = (p - 1u) * 64u;
            for (var ic = 0u; ic < 64u; ic = ic + 1u) {
                acc = acc + b_w[OFF_C3W + co * 192u + ic * 3u + k] * b_e2[t * 129u + row + ic];
            }
        }
    }
    b_e3[t * 65u + co] = max(acc, 0.0);
}

// ---------- 批量 conv4：e3(1,64) → e4(1,128) ----------
@group(0) @binding(6) var<storage, read_write> b_e4: array<f32>; // BATCH*129

@compute @workgroup_size(64)
fn main_conv4(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let t = idx / 128u;
    let co = idx % 128u;
    b_e3[t * 65u + 64u] = 0.0;
    if (idx >= BATCH * 128u) { return; }
    var acc = b_w[OFF_C4B + co];
    for (var k = 0u; k < 3u; k = k + 1u) {
        let p = k;
        if (p >= 1u && p <= 1u) {
            let row = (p - 1u) * 64u;
            for (var ic = 0u; ic < 64u; ic = ic + 1u) {
                acc = acc + b_w[OFF_C4W + co * 192u + ic * 3u + k] * b_e3[t * 65u + row + ic];
            }
        }
    }
    b_e4[t * 129u + co] = max(acc, 0.0);
}

// ---------- 批量 gates_in：Wih@e4 + bih（全帧一次） ----------
@group(0) @binding(7) var<storage, read_write> b_gin: array<f32>; // BATCH*576（行 256B 对齐）

@compute @workgroup_size(64)
fn main_gates_in(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let t = idx / 512u;
    let r = idx % 512u;
    b_e4[t * 129u + 128u] = 0.0;
    if (idx >= BATCH * 512u) { return; }
    var acc = b_w[OFF_BIH + r];
    for (var k = 0u; k < 128u; k = k + 1u) {
        acc = acc + b_w[OFF_WIH + r * 128u + k] * b_e4[t * 129u + k];
    }
    b_gin[t * 576u + r] = acc;
}

// ---------- LSTM 递归（CHUNK 帧打包进 1 个 dispatch）+ final 融合 ----------
// 递归不可并行，但发射开销 ~50µs/次：把 CHUNK 帧串进 kernel 内 for 循环，
// 每帧仍是 全组算 whh·h → barrier → 128 线程 pointwise 原地 h/c → barrier → 归约 prob[d]。
// 512 个 dispatch → MAX_T/CHUNK 个。帧界由 BATCH（rust 侧注入 t）守卫。
// gin/prob 绑整块 buffer（帧号 d 直接索引）；整块 rw 绑定 + 垫片写保持用途域一致。
@group(0) @binding(8) var<storage, read_write> b_gin_all: array<f32>;  // MAX_T*576
@group(0) @binding(9) var<storage, read_write> b_h: array<f32>;  // 128
@group(0) @binding(10) var<storage, read_write> b_c: array<f32>; // 128
@group(0) @binding(11) var<storage, read_write> b_prob_all: array<f32>; // MAX_T*64

var<workgroup> wg_g: array<f32, 512>;
var<workgroup> wg_red: array<f32, 128>;

const LSTM_CHUNK: u32 = {LSTM_CHUNK}u;

// 处理 d0 起的 LSTM_CHUNK 帧（2 barrier/帧已足够：
// barrier2 后无人再读 wg_g；wg_red 的跨帧读写被 r0 程序序 + 下帧 barrier1 隔开）
fn chunk_impl(d0: u32, r: u32) {
    b_gin_all[d0 * 576u + 512u] = 0.0; // 垫片：保持 STORE 语义
    for (var fi = 0u; fi < LSTM_CHUNK; fi = fi + 1u) {
        let d = d0 + fi;
        if (d >= BATCH) { return; }
        // gates：whh 预转置 (k,r) + 8 路展开 —— 单 workgroup 延迟暴露，
        // 8 个独立累加器让 8 个 L2 加载在飞，消去逐迭代串行等待
        for (var half = 0u; half < 2u; half = half + 1u) {
            let row = r + half * 256u;
            var gh = b_w[OFF_BHH + row];
            var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
            var a4 = 0.0; var a5 = 0.0; var a6 = 0.0; var a7 = 0.0;
            var k = 0u;
            loop {
                if (k >= 128u) { break; }
                a0 = a0 + b_w[OFF_WHH + (k) * 512u + row] * b_h[k];
                a1 = a1 + b_w[OFF_WHH + (k + 1u) * 512u + row] * b_h[k + 1u];
                a2 = a2 + b_w[OFF_WHH + (k + 2u) * 512u + row] * b_h[k + 2u];
                a3 = a3 + b_w[OFF_WHH + (k + 3u) * 512u + row] * b_h[k + 3u];
                a4 = a4 + b_w[OFF_WHH + (k + 4u) * 512u + row] * b_h[k + 4u];
                a5 = a5 + b_w[OFF_WHH + (k + 5u) * 512u + row] * b_h[k + 5u];
                a6 = a6 + b_w[OFF_WHH + (k + 6u) * 512u + row] * b_h[k + 6u];
                a7 = a7 + b_w[OFF_WHH + (k + 7u) * 512u + row] * b_h[k + 7u];
                k = k + 8u;
            }
            gh = gh + a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
            wg_g[row] = b_gin_all[d * 576u + row] + gh;
        }
        workgroupBarrier();
        if (r < 128u) {
            let j = r;
            let i = 1.0 / (1.0 + exp(-wg_g[j]));
            let f = 1.0 / (1.0 + exp(-wg_g[128u + j]));
            let g = tanh(wg_g[256u + j]);
            let o = 1.0 / (1.0 + exp(-wg_g[384u + j]));
            let cn = f * b_c[j] + i * g;
            b_c[j] = cn;
            let hn = o * tanh(cn);
            b_h[j] = hn;
            wg_red[j] = b_w[OFF_FW + j] * max(hn, 0.0);
        }
        workgroupBarrier();
        if (r == 0u) {
            var z = b_w[OFF_FB];
            for (var j = 0u; j < 128u; j = j + 1u) {
                z = z + wg_red[j];
            }
            b_prob_all[d * 64u] = 1.0 / (1.0 + exp(-z));
        }
    }
}

{LSTM_CHUNKS}


