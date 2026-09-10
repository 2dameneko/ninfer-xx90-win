#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

namespace ninfer::pdl {

struct LaunchConfig {
    dim3 grid;
    dim3 block;
    std::size_t dynamic_smem_bytes = 0;
    cudaStream_t stream            = nullptr;
};

// Programmatic stream serialization is a sm_90+ capability. A universal binary carries one host
// launch policy for every baked architecture, so the attribute path is gated by a process-wide
// flag that DeviceContext sets from the attached device's resolved features. Until a device is
// attached (or on pre-Hopper hardware) consumers launch normally, which is always correct.
inline bool& enabled_flag() {
    static bool enabled = false;
    return enabled;
}

inline void set_programmatic_launch_enabled(bool enabled) { enabled_flag() = enabled; }

[[nodiscard]] inline bool programmatic_launch_enabled() { return enabled_flag(); }

// Launches a consumer kernel as a programmatic dependent of the immediately preceding producer
// kernel in the same stream. Every consumer control path that reads producer output must first call
// wait_for_dependencies().
template <class... KernelArgs, class... CallArgs>
[[nodiscard]] inline cudaError_t
launch_dependent(const LaunchConfig& launch, void (*kernel)(KernelArgs...), CallArgs&&... args) {
    if (!programmatic_launch_enabled()) {
        kernel<<<launch.grid, launch.block, launch.dynamic_smem_bytes, launch.stream>>>(
            std::forward<CallArgs>(args)...);
        return cudaGetLastError();
    }

    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config{};
    config.gridDim          = launch.grid;
    config.blockDim         = launch.block;
    config.dynamicSmemBytes = launch.dynamic_smem_bytes;
    config.stream           = launch.stream;
    config.attrs            = &attribute;
    config.numAttrs         = 1;

    return cudaLaunchKernelEx(&config, kernel, std::forward<CallArgs>(args)...);
}

// Every producer CTA must call this at least once or exit. This enables dependent scheduling but
// does not make producer writes visible to the consumer. The intrinsics exist on sm_90+ only; on
// older baked architectures consumers launched without the PDL attribute need no-ops.
__device__ __forceinline__ void trigger_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// Call on every consumer control path before its first access to producer-dependent data.
__device__ __forceinline__ void wait_for_dependencies() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
#endif
}

} // namespace ninfer::pdl
