//! GPU 公共设施：后端/适配器选择、适配器枚举与诊断打印。
//! 供 `gpu_batch`（批量路径）与各 bin 使用；环境变量见各函数文档。
//! （逐帧 GpuVad 路径已删除——Intel 发散诊断使命完成，git 历史保留。）

/// 跨平台后端选择：默认 PRIMARY（Vulkan/Metal/DX12 自动），可用环境变量
/// SILERO_BACKEND=vulkan|dx12|metal|gl|primary 覆盖（驱动问题时的逃生门）。
pub fn backend_from_env() -> wgpu::Backends {
    match std::env::var("SILERO_BACKEND").as_deref() {
        Ok("vulkan") => wgpu::Backends::VULKAN,
        Ok("dx12") => wgpu::Backends::DX12,
        Ok("metal") => wgpu::Backends::METAL,
        Ok("gl") => wgpu::Backends::GL,
        Ok("primary") | _ => wgpu::Backends::PRIMARY,
    }
}

/// 适配器偏好：SILERO_POWER=low 选核显/低功耗设备（LowPower），默认 HighPerformance（独显）。
pub fn power_preference_from_env() -> wgpu::PowerPreference {
    match std::env::var("SILERO_POWER").as_deref() {
        Ok("low") => wgpu::PowerPreference::LowPower,
        _ => wgpu::PowerPreference::HighPerformance,
    }
}

/// SILERO_ADAPTER_INFO=1 时向 stderr 打印适配器信息（诊断选中的设备）。
pub fn maybe_log_adapter(adapter: &wgpu::Adapter) {
    if std::env::var("SILERO_ADAPTER_INFO").as_deref() == Ok("1") {
        let info = adapter.get_info();
        let l = adapter.limits();
        eprintln!(
            "adapter: {} | {:?} | {:?} | driver={} {}",
            info.name, info.backend, info.device_type, info.driver, info.driver_info
        );
        eprintln!(
            "  limits: stor_buf/stage={} wg_invoc={} wg_x={} wg_mem={}B stor_bind={}B buf={}B",
            l.max_storage_buffers_per_shader_stage,
            l.max_compute_invocations_per_workgroup,
            l.max_compute_workgroup_size_x,
            l.max_compute_workgroup_storage_size,
            l.max_storage_buffer_binding_size,
            l.max_buffer_size,
        );
        eprintln!(
            "  features: TIMESTAMP_QUERY={} SHADER_F16={}",
            adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY),
            adapter.features().contains(wgpu::Features::SHADER_F16),
        );
    }
}

/// SILERO_ADAPTER=<子串> 时按名字/类型/驱动名挑选适配器（大小写不敏感）。
/// 多显卡机器上 `power_preference` 只会挑"高性能"，无法指定核显 —— 这是唯一可靠入口。
pub fn adapter_filter_from_env() -> Option<String> {
    std::env::var("SILERO_ADAPTER").ok().filter(|s| !s.is_empty())
}

/// 统一的适配器获取：先按 SILERO_ADAPTER 子串过滤枚举结果，过滤不到再回落到
/// wgpu 默认选择（并提示）。
pub fn pick_adapter(instance: &wgpu::Instance) -> wgpu::Adapter {
    let power = power_preference_from_env();
    let backends = backend_from_env();
    let filter = adapter_filter_from_env();

    if let Some(pat) = filter.as_deref() {
        let pat_lc = pat.to_ascii_lowercase();
        let all = pollster::block_on(instance.enumerate_adapters(backends));
        let hit = all.iter().find(|a| {
            let i = a.get_info();
            i.name.to_ascii_lowercase().contains(&pat_lc)
                || i.driver.to_ascii_lowercase().contains(&pat_lc)
                || i.driver_info.to_ascii_lowercase().contains(&pat_lc)
                || format!("{:?}", i.device_type).to_ascii_lowercase().contains(&pat_lc)
        });
        if let Some(a) = hit {
            return a.clone();
        }
        eprintln!(
            "warn: SILERO_ADAPTER={pat} 未匹配到适配器，回落到默认选择；可用适配器："
        );
        for a in &all {
            let i = a.get_info();
            eprintln!("  - {} | {:?} | {:?}", i.name, i.backend, i.device_type);
        }
    }

    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: power,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .expect("no suitable GPU adapter")
}

/// 枚举并打印全部适配器 + 关键 limits/features（SILERO_ADAPTER_INFO=1 或 --adapters 时用）。
pub fn dump_adapters(instance: &wgpu::Instance, backends: wgpu::Backends) {
    let all = pollster::block_on(instance.enumerate_adapters(backends));
    eprintln!("adapters({backends:?}) = {}", all.len());
    for a in &all {
        let i = a.get_info();
        let l = a.limits();
        eprintln!(
            "  - {} | {:?} | {:?} | vendor=0x{:04x} | driver={} {}",
            i.name, i.backend, i.device_type, i.vendor, i.driver, i.driver_info
        );
        eprintln!(
            "      stor_buf/stage={} wg_invoc={} wg_x={} wg_mem={}B stor_bind={}B buf={}B ts_query={}",
            l.max_storage_buffers_per_shader_stage,
            l.max_compute_workgroup_size_x,
            l.max_compute_workgroup_size_x,
            l.max_compute_workgroup_storage_size,
            l.max_storage_buffer_binding_size,
            l.max_buffer_size,
            a.features().contains(wgpu::Features::TIMESTAMP_QUERY),
        );
    }
}

/// 诊断入口：按环境变量决定是否打印适配器清单。
pub fn maybe_dump_adapters() {
    if std::env::var("SILERO_ADAPTER_INFO").as_deref() == Ok("1") {
        let inst = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: backend_from_env(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        dump_adapters(&inst, backend_from_env());
    }
}

/// 选定适配器的「名称 | 后端 | 类型」单行摘要（供 bench 等工具打印）。
pub fn dump_adapters_env() -> String {
    let inst = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: backend_from_env(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let a = pick_adapter(&inst);
    let i = a.get_info();
    format!("{} | {:?} | {:?}", i.name, i.backend, i.device_type)
}
