#pragma once

#include <cstdint>

namespace ninfer::arch {

// Compute capability encoded as major * 10 + minor; matches
// DeviceContext::compute_capability().
enum : std::int32_t {
    kCapAmpere86     = 86,
    kCapAda89        = 89,
    kCapBlackwell120 = 120,
};

// Compile-time bitmask of the architectures whose native SASS this binary was built with. The
// build system defines NINFER_BAKED_CAPABILITIES from CMAKE_CUDA_ARCHITECTURES so every target,
// including tests and apps, sees the same answer. A new Nvidia architecture is added by
// introducing one capability constant, one bit, one row in features_for() and (when needed)
// device-code guards; no source file outside this table needs to know the card model.
enum : std::uint32_t {
    kBit86  = 1u << 0,
    kBit89  = 1u << 1,
    kBit120 = 1u << 2,
};

#if !defined(NINFER_BAKED_CAPABILITIES)
#error NINFER_BAKED_CAPABILITIES must be defined by the build system (see CMakeLists.txt)
#endif

inline constexpr std::uint32_t kBakedCapabilities =
    static_cast<std::uint32_t>(NINFER_BAKED_CAPABILITIES);

// Capability-resolved kernel feature set. Device code keeps using __CUDA_ARCH__ directly; this
// table is the single source of truth for host-side launch policy, route selection, KV storage
// validation, and graph/workspace planning.
struct Features {
    std::int32_t capability = 0;
    bool fp8_a8   = false;  // FP8 E4M3 activation x weight tensor-core GEMMs (sm_89 plain MMA;
                            // sm_120 kind::f8f6f4).
    bool nvfp4_a4 = false;  // NVFP4 block-scale MMA (Blackwell sm_120a only).
    bool fp8_kv   = false;  // FP8 E4M3 KV-cache attention.
    bool nvfp4_kv = false;  // NVFP4 KV-cache attention.
    bool kv_k8v4  = false;  // FP8-key / NVFP4-value KV-cache attention.
    bool pdl      = false;  // Programmatic dependent launch (sm_90+).
    bool tma      = false;  // TMA bulk async copy (sm_90+).
};

[[nodiscard]] constexpr Features features_for(std::int32_t capability) {
    Features f;
    f.capability = capability;
    const bool ada_or_newer = capability >= kCapAda89;
    const bool blackwell    = capability >= kCapBlackwell120;
    f.fp8_a8   = ada_or_newer;
    f.nvfp4_a4 = blackwell;
    f.fp8_kv   = ada_or_newer;
    f.nvfp4_kv = blackwell;
    f.kv_k8v4  = blackwell;
    f.pdl      = capability >= 90;
    f.tma      = capability >= 90;
    return f;
}

[[nodiscard]] constexpr bool capability_baked(std::int32_t capability) {
    switch (capability) {
    case kCapAmpere86:
        return (kBakedCapabilities & kBit86) != 0;
    case kCapAda89:
        return (kBakedCapabilities & kBit89) != 0;
    case kCapBlackwell120:
        return (kBakedCapabilities & kBit120) != 0;
    default:
        return false;
    }
}

} // namespace ninfer::arch