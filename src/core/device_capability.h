#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::core {

// Compute capability (major*10+minor) of the CUDA device bound to the calling thread, cached on
// first use. Ops dispatchers use this to select the launch path whose kernel SASS the current
// device can actually run: a universal binary bakes one host-side dispatcher per op, so
// per-architecture launch policy is resolved here instead of at compile time. Must be called
// after a device has been set (the engine binds a device before any op runs).
[[nodiscard]] inline std::int32_t current_device_compute_capability() {
    static const std::int32_t cached = [] {
        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceProp props{};
        cudaGetDeviceProperties(&props, device);
        return props.major * 10 + props.minor;
    }();
    return cached;
}

// sm_89 (Ada) and later: plain FP8/FP4 tensor-core encodings.
[[nodiscard]] inline bool device_supports_sm89_or_newer() {
    return current_device_compute_capability() >= 89;
}

// Blackwell (sm_100) and later: per-block static shared memory above the pre-Blackwell 48KB
// (up to ~99KB). sm_86 and sm_89 both cap static shared at 48KB, so any kernel that bakes more
// than that is Blackwell-only: its pre-Blackwell SASS is a trap stub and the host dispatcher
// must fall back to a smaller-footprint path on every pre-Blackwell device.
[[nodiscard]] inline bool device_supports_sm100_or_newer() {
    return current_device_compute_capability() >= 100;
}

// Pre-Blackwell devices (sm_86/sm_89) cap static shared memory at 49152 bytes per block;
// Blackwell lifts the limit. Kernels that bake more than that launch their storage as dynamic
// shared memory on pre-Blackwell devices (with the MaxDynamicSharedMemorySize carveout) and
// keep the static layout on Blackwell. Host launchers and the kernel-side storage switch must
// use the same 49152-byte threshold to stay consistent with the per-architecture SASS.
inline constexpr std::size_t kPreBlackwellStaticSharedLimit = 49152;

[[nodiscard]] inline std::size_t dynamic_shared_carveout(std::size_t storage_bytes) {
    return storage_bytes > kPreBlackwellStaticSharedLimit &&
               current_device_compute_capability() < 100
        ? storage_bytes
        : 0;
}

} // namespace ninfer::core
