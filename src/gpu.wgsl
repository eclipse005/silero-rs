// Silero VAD 16k 的 WGSL kernels —— 每个 entry point 对应一帧里的一个 dispatch。
// 语义与 lib.rs host 路径逐行对应；累加顺序为顺序归约（确定性）。
// 全文件 binding 唯一编号，每个 pipeline 用自己的 bind group。
//
// 注意：消费者 kernel（conv1-4 读前一级输出、point 读 gates）对输入声明为
// read_write 并向"垫片槽"（数组末尾多出的一个元素，永不参与计算）写 0 ——
// 这保持 STORE 访问语义，使同一物理 buffer 在整个 pass 内保持 rw 用途，
// 避免 ro/rw usage 冲突（wgpu 用途域规则）。

// ---------- STFT + magnitude: x640 → mag (129*4) ----------
@group(0) @binding(0) var<storage, read> stft_x: array<f32>;       // 640
@group(0) @binding(1) var<storage, read> stft_basis: array<f32>;   // 258*256
@group(0) @binding(2) var<storage, read_write> stft_mag: array<f32>; // 517（末位垫片）

@compute @workgroup_size(64)
fn main_stft(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= 129u * 4u) { return; }
    let f = idx / 4u;
    let p = idx % 4u;
    let w0 = p * 128u;
    var re = 0.0;
    var im = 0.0;
    let base_r = f * 256u;
    let base_i = (129u + f) * 256u;
    for (var k = 0u; k < 256u; k = k + 1u) {
        let v = stft_x[w0 + k];
        re = re + stft_basis[base_r + k] * v;
        im = im + stft_basis[base_i + k] * v;
    }
    stft_mag[idx] = sqrt(re * re + im * im);
}

// ---------- conv1d(k=3, pad=1) + relu ----------
// conv1: mag(129,4) → e1(128,4)，stride 1；垫片槽 516
@group(0) @binding(3) var<storage, read_write> conv1_in: array<f32>;
@group(0) @binding(4) var<storage, read> conv1_w: array<f32>;
@group(0) @binding(5) var<storage, read> conv1_b: array<f32>;
@group(0) @binding(6) var<storage, read_write> conv1_out: array<f32>; // 513（末位垫片）

@compute @workgroup_size(64)
fn main_conv1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    conv1_in[516] = 0.0; // 垫片：保持 STORE 语义
    if (idx >= 128u * 4u) { return; }
    let co = idx / 4u;
    let t = idx % 4u;
    let ic = 129u;
    let t_in = 4u;
    var acc = conv1_b[co];
    let wbase = co * ic * 3u;
    for (var i = 0u; i < ic; i = i + 1u) {
        for (var k = 0u; k < 3u; k = k + 1u) {
            let tt = t + k;
            var v = 0.0;
            if (tt != 0u && tt != t_in + 1u) {
                v = conv1_in[i * t_in + (tt - 1u)];
            }
            acc = acc + conv1_w[wbase + i * 3u + k] * v;
        }
    }
    conv1_out[idx] = max(acc, 0.0);
}

// conv2: e1(128,4) → e2(64,2)，stride 2；垫片槽 512
@group(0) @binding(7) var<storage, read_write> conv2_in: array<f32>;
@group(0) @binding(8) var<storage, read> conv2_w: array<f32>;
@group(0) @binding(9) var<storage, read> conv2_b: array<f32>;
@group(0) @binding(10) var<storage, read_write> conv2_out: array<f32>; // 129

@compute @workgroup_size(64)
fn main_conv2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    conv2_in[512] = 0.0;
    if (idx >= 64u * 2u) { return; }
    let co = idx / 2u;
    let t = idx % 2u;
    let ic = 128u;
    let t_in = 4u;
    var acc = conv2_b[co];
    let wbase = co * ic * 3u;
    for (var i = 0u; i < ic; i = i + 1u) {
        for (var k = 0u; k < 3u; k = k + 1u) {
            let tt = t * 2u + k;
            var v = 0.0;
            if (tt != 0u && tt != t_in + 1u) {
                v = conv2_in[i * t_in + (tt - 1u)];
            }
            acc = acc + conv2_w[wbase + i * 3u + k] * v;
        }
    }
    conv2_out[idx] = max(acc, 0.0);
}

// conv3: e2(64,2) → e3(64,1)，stride 2；垫片槽 128
@group(0) @binding(11) var<storage, read_write> conv3_in: array<f32>;
@group(0) @binding(12) var<storage, read> conv3_w: array<f32>;
@group(0) @binding(13) var<storage, read> conv3_b: array<f32>;
@group(0) @binding(14) var<storage, read_write> conv3_out: array<f32>; // 65

@compute @workgroup_size(64)
fn main_conv3(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    conv3_in[128] = 0.0;
    if (idx >= 64u) { return; }
    let co = idx;
    let ic = 64u;
    let t_in = 2u;
    var acc = conv3_b[co];
    let wbase = co * ic * 3u;
    for (var i = 0u; i < ic; i = i + 1u) {
        for (var k = 0u; k < 3u; k = k + 1u) {
            let tt = k;
            var v = 0.0;
            if (tt != 0u && tt != t_in + 1u) {
                v = conv3_in[i * t_in + (tt - 1u)];
            }
            acc = acc + conv3_w[wbase + i * 3u + k] * v;
        }
    }
    conv3_out[idx] = max(acc, 0.0);
}

// conv4: e3(64,1) → e4(128,1)，stride 1；垫片槽 64
@group(0) @binding(15) var<storage, read_write> conv4_in: array<f32>;
@group(0) @binding(16) var<storage, read> conv4_w: array<f32>;
@group(0) @binding(17) var<storage, read> conv4_b: array<f32>;
@group(0) @binding(18) var<storage, read_write> conv4_out: array<f32>; // 129

@compute @workgroup_size(64)
fn main_conv4(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    conv4_in[64] = 0.0;
    if (idx >= 128u) { return; }
    let co = idx;
    let ic = 64u;
    let t_in = 1u;
    var acc = conv4_b[co];
    let wbase = co * ic * 3u;
    for (var i = 0u; i < ic; i = i + 1u) {
        for (var k = 0u; k < 3u; k = k + 1u) {
            let tt = k;
            var v = 0.0;
            if (tt != 0u && tt != t_in + 1u) {
                v = conv4_in[i * t_in + (tt - 1u)];
            }
            acc = acc + conv4_w[wbase + i * 3u + k] * v;
        }
    }
    conv4_out[idx] = max(acc, 0.0);
}

// ---------- LSTM gates: gates = (Wih@e4 + bih) + (Whh@h + bhh) ----------
@group(0) @binding(19) var<storage, read> gates_e4: array<f32>;   // 129（含垫片，读前 128）
@group(0) @binding(20) var<storage, read_write> gates_h: array<f32>;  // 129（含垫片）
@group(0) @binding(21) var<storage, read> gates_wih: array<f32>;  // 512*128
@group(0) @binding(22) var<storage, read> gates_whh: array<f32>;  // 512*128
@group(0) @binding(23) var<storage, read> gates_bih: array<f32>;  // 512
@group(0) @binding(24) var<storage, read> gates_bhh: array<f32>;  // 512
@group(0) @binding(25) var<storage, read_write> gates_out: array<f32>; // 513

@compute @workgroup_size(64)
fn main_gates(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    gates_h[128] = 0.0; // 垫片：保持 STORE 语义（h 在本 pass 内被 point 原地写）
    if (r >= 512u) { return; }
    var gi = gates_bih[r];
    var gh = gates_bhh[r];
    let base = r * 128u;
    for (var k = 0u; k < 128u; k = k + 1u) {
        gi = gi + gates_wih[base + k] * gates_e4[k];
        gh = gh + gates_whh[base + k] * gates_h[k];
    }
    gates_out[r] = gi + gh;
}

// ---------- LSTM pointwise: (gates, c, h) 原地更新，块序 i,f,g,o；垫片槽 512 ----------
// LSTMCell 对 h/c 按索引逐元素更新：线程 j 只碰自己的 j —— 原地无冒险。
// dispatch 间隐式屏障保证 gates 先读旧 h、point 后写，无需 hn/cn 缓冲与拷贝。
@group(0) @binding(26) var<storage, read_write> pt_gates: array<f32>; // 513
@group(0) @binding(27) var<storage, read_write> pt_c: array<f32>;     // 128（原地更新）
@group(0) @binding(28) var<storage, read_write> pt_h: array<f32>;     // 129（原地更新 h）

@compute @workgroup_size(128)
fn main_point(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    pt_gates[512] = 0.0;
    if (j >= 128u) { return; }
    let i = tf_sigmoid(pt_gates[j]);
    let f = tf_sigmoid(pt_gates[128u + j]);
    let g = tf_tanh(pt_gates[256u + j]);
    let o = tf_sigmoid(pt_gates[384u + j]);
    let cn = f * pt_c[j] + i * g;
    pt_c[j] = cn;
    pt_h[j] = o * tf_tanh(cn);
}

// ---------- final: relu(h) → conv1x1 → sigmoid ----------
@group(0) @binding(30) var<storage, read_write> fin_h: array<f32>;  // 129（含垫片）
@group(0) @binding(31) var<storage, read> fin_w: array<f32>;   // 128
@group(0) @binding(32) var<storage, read> fin_b: array<f32>;   // 1
@group(0) @binding(33) var<storage, read_write> fin_out: array<f32>; // 1

@compute @workgroup_size(1)
fn main_final() {
    fin_h[128] = 0.0; // 垫片：保持 STORE 语义
    var z = fin_b[0];
    for (var j = 0u; j < 128u; j = j + 1u) {
        z = z + fin_w[j] * max(fin_h[j], 0.0);
    }
    fin_out[0] = tf_sigmoid(z);
}

