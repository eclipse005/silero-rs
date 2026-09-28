// Silero VAD 16k 批量 GPU kernels —— {BATCH} 帧一次提交（模板注入 BATCH 常量）。
//
// 设计：
// - conv/stft 逐帧独立：T 帧仍是 5 个 dispatch，线程数 ×T。**显式软件流水**：每个
//   卷积线程先取 ic+4..ic+7 的权重/激活，再做本拍 FMA —— Intel 集显实测卷积是每线程
//   ~100 周期的取数延迟链、占用率不足以掩盖（优化前每帧 ~37µs），预取后取数与 FMA
//   重叠。累加链顺序不变 → 数值逐位一致。配套 host 侧 conv 权重重排
//   [co][ic][k]→[co][k][ic]（线程沿 ic 连续读，缓存行利用率 33%→100%）。
// - gates_in（Wih@e4 全帧）1 个批量 dispatch；whh·h 留给递归步。
// - LSTM 递归步：单 workgroup(256) fused kernel —— 全组算 whh·h 进 workgroup
//   storage → barrier → 128 线程 pointwise 原地更新 h/c；每 CHUNK 帧打包进 1 个
//   dispatch（发射开销远大于计算），帧号经 entry point 烘进 shader。
// - prob（fw·relu(h) 顺序归约 + sigmoid）拆出递归循环：每帧 1 个 workgroup(128)
//   批量并行；与旧 in-loop 归约**逐位一致**（乘积先存 workgroup 内存、线程 0 顺序累加）。
// - 每 T 帧：8 + MAX_T/LSTM_CHUNK 个 dispatch、1 次 submit、1 次 map(T*256B)。
//
// 布局：所有批量张量按 (帧 t, 帧内偏移) 行主序，行尾 1 个垫片槽
// （读者 kernel 写垫片保持 rw 用途，wgpu 用途域规则）。
// ⚠️ 本机 Intel 集显：workgroup_size(512) 会显著劣化（实测），conv 用 64 / lstm 用 256。

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
fn main_stft(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let idx = gid.y * nw.x * 64u + gid.x;
    if (idx >= BATCH * 516u) { return; }
    let t = idx / 516u;
    let r = idx % 516u;
    let f = r / 4u;
    let p = r % 4u;
    let w0 = t * 640u + p * 128u;
    // 软件流水（4 深）：每拍预取 k+4..k+7 的 (x, wr, wi) 在飞，再做本拍 4 组 FMA；
    // re/im 两条累加链各自保持 k 升序 → 数值逐位一致
    var xn0 = b_x[w0]; var wrn0 = b_w[f * 256u]; var win0 = b_w[(129u + f) * 256u];
    var xn1 = b_x[w0 + 1u]; var wrn1 = b_w[f * 256u + 1u]; var win1 = b_w[(129u + f) * 256u + 1u];
    var xn2 = b_x[w0 + 2u]; var wrn2 = b_w[f * 256u + 2u]; var win2 = b_w[(129u + f) * 256u + 2u];
    var xn3 = b_x[w0 + 3u]; var wrn3 = b_w[f * 256u + 3u]; var win3 = b_w[(129u + f) * 256u + 3u];
    var re = 0.0;
    var im = 0.0;
    for (var k = 0u; k < 256u; k = k + 4u) {
        let g0 = min(k + 4u, 255u);
        let g1 = min(k + 5u, 255u);
        let g2 = min(k + 6u, 255u);
        let x4 = b_x[w0 + g0]; let w4 = b_w[f * 256u + g0]; let v4 = b_w[(129u + f) * 256u + g0];
        let x5 = b_x[w0 + g1]; let w5 = b_w[f * 256u + g1]; let v5 = b_w[(129u + f) * 256u + g1];
        let x6 = b_x[w0 + g2]; let w6 = b_w[f * 256u + g2]; let v6 = b_w[(129u + f) * 256u + g2];
        re = re + wrn0 * xn0; im = im + win0 * xn0;
        re = re + wrn1 * xn1; im = im + win1 * xn1;
        re = re + wrn2 * xn2; im = im + win2 * xn2;
        re = re + wrn3 * xn3; im = im + win3 * xn3;
        xn0 = x4; wrn0 = w4; win0 = v4;
        xn1 = x5; wrn1 = w5; win1 = v5;
        xn2 = x6; wrn2 = w6; win2 = v6;
        xn3 = b_x[w0 + min(k + 7u, 255u)];
        wrn3 = b_w[f * 256u + min(k + 7u, 255u)];
        win3 = b_w[(129u + f) * 256u + min(k + 7u, 255u)];
    }
    b_mag[t * 517u + p * 129u + f] = sqrt(re * re + im * im);
}

// ---------- 批量 conv1：mag(4,129) → e1(4,128) ----------
@group(0) @binding(3) var<storage, read_write> b_e1: array<f32>; // BATCH*513

@compute @workgroup_size(64)
fn main_conv1(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let idx = gid.y * nw.x * 64u + gid.x;
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
            // 软件流水（4 深）：预取 ic+4..ic+7 的 (w, m) 在飞，再做本拍 4 次 FMA；
            // 权重已重排为 [co][k][ic]，沿 ic 连续读。129 = 32×4 + 1：主循环 32 拍，
            // 尾拍 ic=128 在循环后补一次 FMA
            let wbase = OFF_C1W + co * 387u + k * 129u;
            var wn0 = b_w[wbase]; var mn0 = b_mag[t * 517u + row];
            var wn1 = b_w[wbase + 1u]; var mn1 = b_mag[t * 517u + row + 1u];
            var wn2 = b_w[wbase + 2u]; var mn2 = b_mag[t * 517u + row + 2u];
            var wn3 = b_w[wbase + 3u]; var mn3 = b_mag[t * 517u + row + 3u];
            for (var ic = 0u; ic < 128u; ic = ic + 4u) {
                let g0 = min(ic + 4u, 128u); let g1 = min(ic + 5u, 128u);
                let g2 = min(ic + 6u, 128u); let g3 = min(ic + 7u, 128u);
                let w4 = b_w[wbase + g0]; let m4 = b_mag[t * 517u + row + g0];
                let w5 = b_w[wbase + g1]; let m5 = b_mag[t * 517u + row + g1];
                let w6 = b_w[wbase + g2]; let m6 = b_mag[t * 517u + row + g2];
                let w7 = b_w[wbase + g3]; let m7 = b_mag[t * 517u + row + g3];
                acc = acc + wn0 * mn0;
                acc = acc + wn1 * mn1;
                acc = acc + wn2 * mn2;
                acc = acc + wn3 * mn3;
                wn0 = w4; mn0 = m4;
                wn1 = w5; mn1 = m5;
                wn2 = w6; mn2 = m6;
                wn3 = w7; mn3 = m7;
            }
            acc = acc + wn0 * mn0; // 尾拍：ic=128
        }
    }
    b_e1[t * 513u + p4 * 128u + co] = max(acc, 0.0);
}

// ---------- 批量 conv2：e1(4,128) → e2(2,64)，stride 2 ----------
@group(0) @binding(4) var<storage, read_write> b_e2: array<f32>; // BATCH*129

@compute @workgroup_size(64)
fn main_conv2(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let idx = gid.y * nw.x * 64u + gid.x;
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
            let wbase = OFF_C2W + co * 384u + k * 128u;
            var wn0 = b_w[wbase]; var mn0 = b_e1[t * 513u + row];
            var wn1 = b_w[wbase + 1u]; var mn1 = b_e1[t * 513u + row + 1u];
            var wn2 = b_w[wbase + 2u]; var mn2 = b_e1[t * 513u + row + 2u];
            var wn3 = b_w[wbase + 3u]; var mn3 = b_e1[t * 513u + row + 3u];
            for (var ic = 0u; ic < 128u; ic = ic + 4u) {
                let g0 = min(ic + 4u, 127u); let g1 = min(ic + 5u, 127u);
                let g2 = min(ic + 6u, 127u); let g3 = min(ic + 7u, 127u);
                let w4 = b_w[wbase + g0]; let m4 = b_e1[t * 513u + row + g0];
                let w5 = b_w[wbase + g1]; let m5 = b_e1[t * 513u + row + g1];
                let w6 = b_w[wbase + g2]; let m6 = b_e1[t * 513u + row + g2];
                let w7 = b_w[wbase + g3]; let m7 = b_e1[t * 513u + row + g3];
                acc = acc + wn0 * mn0;
                acc = acc + wn1 * mn1;
                acc = acc + wn2 * mn2;
                acc = acc + wn3 * mn3;
                wn0 = w4; mn0 = m4;
                wn1 = w5; mn1 = m5;
                wn2 = w6; mn2 = m6;
                wn3 = w7; mn3 = m7;
            }
        }
    }
    b_e2[t * 129u + p2 * 64u + co] = max(acc, 0.0);
}

// ---------- 批量 conv3：e2(2,64) → e3(1,64)，stride 2 ----------
@group(0) @binding(5) var<storage, read_write> b_e3: array<f32>; // BATCH*65

@compute @workgroup_size(64)
fn main_conv3(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let idx = gid.y * nw.x * 64u + gid.x;
    let t = idx / 64u;
    let co = idx % 64u;
    b_e2[t * 129u + 128u] = 0.0;
    if (idx >= BATCH * 64u) { return; }
    var acc = b_w[OFF_C3B + co];
    for (var k = 0u; k < 3u; k = k + 1u) {
        let p = k;
        if (p >= 1u && p <= 2u) {
            let row = (p - 1u) * 64u;
            let wbase = OFF_C3W + co * 192u + k * 64u;
            var wn0 = b_w[wbase]; var mn0 = b_e2[t * 129u + row];
            var wn1 = b_w[wbase + 1u]; var mn1 = b_e2[t * 129u + row + 1u];
            var wn2 = b_w[wbase + 2u]; var mn2 = b_e2[t * 129u + row + 2u];
            var wn3 = b_w[wbase + 3u]; var mn3 = b_e2[t * 129u + row + 3u];
            for (var ic = 0u; ic < 64u; ic = ic + 4u) {
                let g0 = min(ic + 4u, 63u); let g1 = min(ic + 5u, 63u);
                let g2 = min(ic + 6u, 63u); let g3 = min(ic + 7u, 63u);
                let w4 = b_w[wbase + g0]; let m4 = b_e2[t * 129u + row + g0];
                let w5 = b_w[wbase + g1]; let m5 = b_e2[t * 129u + row + g1];
                let w6 = b_w[wbase + g2]; let m6 = b_e2[t * 129u + row + g2];
                let w7 = b_w[wbase + g3]; let m7 = b_e2[t * 129u + row + g3];
                acc = acc + wn0 * mn0;
                acc = acc + wn1 * mn1;
                acc = acc + wn2 * mn2;
                acc = acc + wn3 * mn3;
                wn0 = w4; mn0 = m4;
                wn1 = w5; mn1 = m5;
                wn2 = w6; mn2 = m6;
                wn3 = w7; mn3 = m7;
            }
        }
    }
    b_e3[t * 65u + co] = max(acc, 0.0);
}

// ---------- 批量 conv4：e3(1,64) → e4(1,128) ----------
@group(0) @binding(6) var<storage, read_write> b_e4: array<f32>; // BATCH*129

@compute @workgroup_size(64)
fn main_conv4(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let idx = gid.y * nw.x * 64u + gid.x;
    let t = idx / 128u;
    let co = idx % 128u;
    b_e3[t * 65u + 64u] = 0.0;
    if (idx >= BATCH * 128u) { return; }
    var acc = b_w[OFF_C4B + co];
    for (var k = 0u; k < 3u; k = k + 1u) {
        let p = k;
        if (p >= 1u && p <= 1u) {
            let row = (p - 1u) * 64u;
            let wbase = OFF_C4W + co * 192u + k * 64u;
            var wn0 = b_w[wbase]; var mn0 = b_e3[t * 65u + row];
            var wn1 = b_w[wbase + 1u]; var mn1 = b_e3[t * 65u + row + 1u];
            var wn2 = b_w[wbase + 2u]; var mn2 = b_e3[t * 65u + row + 2u];
            var wn3 = b_w[wbase + 3u]; var mn3 = b_e3[t * 65u + row + 3u];
            for (var ic = 0u; ic < 64u; ic = ic + 4u) {
                let g0 = min(ic + 4u, 63u); let g1 = min(ic + 5u, 63u);
                let g2 = min(ic + 6u, 63u); let g3 = min(ic + 7u, 63u);
                let w4 = b_w[wbase + g0]; let m4 = b_e3[t * 65u + row + g0];
                let w5 = b_w[wbase + g1]; let m5 = b_e3[t * 65u + row + g1];
                let w6 = b_w[wbase + g2]; let m6 = b_e3[t * 65u + row + g2];
                let w7 = b_w[wbase + g3]; let m7 = b_e3[t * 65u + row + g3];
                acc = acc + wn0 * mn0;
                acc = acc + wn1 * mn1;
                acc = acc + wn2 * mn2;
                acc = acc + wn3 * mn3;
                wn0 = w4; mn0 = m4;
                wn1 = w5; mn1 = m5;
                wn2 = w6; mn2 = m6;
                wn3 = w7; mn3 = m7;
            }
        }
    }
    b_e4[t * 129u + co] = max(acc, 0.0);
}

// ---------- 批量 gates_in：Wih@e4 + bih（全帧一次） ----------
@group(0) @binding(7) var<storage, read_write> b_gin: array<f32>; // BATCH*576（行 256B 对齐）

@compute @workgroup_size(64)
fn main_gates_in(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let idx = gid.y * nw.x * 64u + gid.x;
    let t = idx / 512u;
    let r = idx % 512u;
    b_e4[t * 129u + 128u] = 0.0;
    if (idx >= BATCH * 512u) { return; }
    // 软件流水（4 深）：预取 k+4..k+7 的 (wih, e4) 在飞，再做本拍 4 次 FMA
    let wbase = OFF_WIH + r * 128u;
    var wn0 = b_w[wbase]; var en0 = b_e4[t * 129u];
    var wn1 = b_w[wbase + 1u]; var en1 = b_e4[t * 129u + 1u];
    var wn2 = b_w[wbase + 2u]; var en2 = b_e4[t * 129u + 2u];
    var wn3 = b_w[wbase + 3u]; var en3 = b_e4[t * 129u + 3u];
    var acc = b_w[OFF_BIH + r];
    for (var k = 0u; k < 128u; k = k + 4u) {
        let g0 = min(k + 4u, 127u); let g1 = min(k + 5u, 127u);
        let g2 = min(k + 6u, 127u); let g3 = min(k + 7u, 127u);
        let w4 = b_w[wbase + g0]; let e4v = b_e4[t * 129u + g0];
        let w5 = b_w[wbase + g1]; let e5 = b_e4[t * 129u + g1];
        let w6 = b_w[wbase + g2]; let e6 = b_e4[t * 129u + g2];
        let w7 = b_w[wbase + g3]; let e7 = b_e4[t * 129u + g3];
        acc = acc + wn0 * en0;
        acc = acc + wn1 * en1;
        acc = acc + wn2 * en2;
        acc = acc + wn3 * en3;
        wn0 = w4; en0 = e4v;
        wn1 = w5; en1 = e5;
        wn2 = w6; en2 = e6;
        wn3 = w7; en3 = e7;
    }
    b_gin[t * 576u + r] = acc;
}

// ---------- LSTM 递归（CHUNK 帧打包进 1 个 dispatch）+ final 拆分 ----------
// 递归不可并行，但发射开销远大于计算：把 CHUNK 帧串进 kernel 内 for 循环。
// 每帧 2 barrier：全组算 whh·h 进 wg_g → barrier → 128 线程 pointwise 原地 h/c +
// h 快照 → barrier。帧循环体的浮点运算/累加序与旧版逐位一致（8 路展开链 + 固定合并序）。
@group(0) @binding(8) var<storage, read_write> b_gin_all: array<f32>;  // BATCH*576
@group(0) @binding(9) var<storage, read_write> b_h: array<f32>;  // 128
@group(0) @binding(10) var<storage, read_write> b_c: array<f32>; // 128
@group(0) @binding(11) var<storage, read_write> b_prob_all: array<f32>; // BATCH*64
@group(0) @binding(12) var<storage, read_write> b_h_all: array<f32>; // BATCH*128 每帧 h 快照（供 main_prob）

var<workgroup> wg_g: array<f32, 512>;
// 跨帧递归状态放 workgroup 内存：规范只保证 workgroupBarrier 同步 **workgroup
// 内存**的可见性，不保证同一 dispatch 内跨帧读写 storage buffer 的可见性
//（那是 dispatch 之间的隐式屏障负责的事）。h/c 常驻 workgroup 内存，chunk 首尾
// 与 b_h/b_c 搬运一次。（历史注：这**不是**「Intel 发散」的根因 —— 真正根因是硬件
// exp/tanh 跨厂商差 1–2 ULP，见 wgsl_math.wgsl 头注。）
var<workgroup> wg_h: array<f32, 128>;
var<workgroup> wg_c: array<f32, 128>;

const LSTM_CHUNK: u32 = {LSTM_CHUNK}u;

// 处理 d0 起的 min(LSTM_CHUNK, BATCH-d0) 帧。r ∈ [0,256)：每线程 2 个 gate 行
// （r 与 r+256）。chunk 间由 dispatch 隐式屏障串行。
fn lstm_chunk_impl(d0: u32, r: u32) {
    b_gin_all[d0 * 576u + 512u] = 0.0; // 垫片：保持 STORE 语义
    if (r < 128u) {
        wg_h[r] = b_h[r];
        wg_c[r] = b_c[r];
    }
    workgroupBarrier();
    for (var fi = 0u; fi < LSTM_CHUNK; fi = fi + 1u) {
        let d = d0 + fi;
        if (d >= BATCH) { break; } // uniform：全员跳出并执行 chunk 尾的 h/c 写回
        // gates：whh 预转置 (k,r) + 8 路展开 —— 8 个独立累加器让 8 个 L2 加载在飞，
        // 消去逐迭代串行等待；链归属（k ≡ m mod 8）与合并序与旧版完全一致
        for (var half = 0u; half < 2u; half = half + 1u) {
            let row = r + half * 256u;
            var gh = b_w[OFF_BHH + row];
            var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
            var a4 = 0.0; var a5 = 0.0; var a6 = 0.0; var a7 = 0.0;
            var k = 0u;
            loop {
                if (k >= 128u) { break; }
                a0 = a0 + b_w[OFF_WHH + (k) * 512u + row] * wg_h[k];
                a1 = a1 + b_w[OFF_WHH + (k + 1u) * 512u + row] * wg_h[k + 1u];
                a2 = a2 + b_w[OFF_WHH + (k + 2u) * 512u + row] * wg_h[k + 2u];
                a3 = a3 + b_w[OFF_WHH + (k + 3u) * 512u + row] * wg_h[k + 3u];
                a4 = a4 + b_w[OFF_WHH + (k + 4u) * 512u + row] * wg_h[k + 4u];
                a5 = a5 + b_w[OFF_WHH + (k + 5u) * 512u + row] * wg_h[k + 5u];
                a6 = a6 + b_w[OFF_WHH + (k + 6u) * 512u + row] * wg_h[k + 6u];
                a7 = a7 + b_w[OFF_WHH + (k + 7u) * 512u + row] * wg_h[k + 7u];
                k = k + 8u;
            }
            gh = gh + a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
            wg_g[row] = b_gin_all[d * 576u + row] + gh;
        }
        workgroupBarrier();
        if (r < 128u) {
            let j = r;
            let i = tf_sigmoid(wg_g[j]);
            let f = tf_sigmoid(wg_g[128u + j]);
            let g = tf_tanh(wg_g[256u + j]);
            let o = tf_sigmoid(wg_g[384u + j]);
            let cn = f * wg_c[j] + i * g;
            wg_c[j] = cn;
            let hn = o * tf_tanh(cn);
            wg_h[j] = hn;
            b_h_all[d * 128u + j] = hn; // 快照：main_prob 的归约输入（f32 存取精确）
        }
        workgroupBarrier();
    }
    // chunk 结束：workgroup → storage（chunk 间 dispatch 隐式屏障保证可见）
    if (r < 128u) {
        b_h[r] = wg_h[r];
        b_c[r] = wg_c[r];
    }
}

{LSTM_CHUNKS}

// ---------- prob：fw·relu(h) 顺序归约 + sigmoid（每帧 1 个 workgroup(128)） ----------
// 与旧 in-loop 归约逐位一致：乘积由线程 j 写 workgroup、线程 0 顺序累加 128 项。
var<workgroup> wg_pr: array<f32, 128>;

@compute @workgroup_size(128)
fn main_prob(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x / 128u;
    let j = gid.x % 128u;
    if (d >= BATCH) { return; } // uniform（整 workgroup 同帧）
    wg_pr[j] = b_w[OFF_FW + j] * max(b_h_all[d * 128u + j], 0.0);
    workgroupBarrier();
    if (j == 0u) {
        var z = b_w[OFF_FB];
        for (var k = 0u; k < 128u; k = k + 1u) {
            z = z + wg_pr[k];
        }
        b_prob_all[d * 64u] = tf_sigmoid(z);
    }
}
