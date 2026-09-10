#include "ops/attn_input_proj/w8/w8_attn_input_kernels.h"

#include "core/device.h"
#include "core/device_capability.h"
#include "ops/linear/w8/w8_small_t_mma.cuh"
#include "ops/linear/w8/w8_rowsplit_gemm_medium_t_splitk.cuh"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kTargetRows             = 9216;
constexpr int kCompanionRows          = 6144;
constexpr int kHidden                 = 2048;
constexpr int kRowsPerCta             = 16;
constexpr int kFirstExactCols         = 2;
constexpr int kLastTargetExactCols    = 48;
constexpr int kLastCompanionExactCols = 32;
using TargetOutput                    = W8SplitOutput4<4096, 512, 4096, 512>;
using CompanionOutput                 = W8SplitOutput3<4096, 1024, 1024>;
using TargetLauncher    = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                                cudaStream_t);
using CompanionLauncher = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&,
                                   cudaStream_t);

template <int ActiveCols, int Rows, class Output>
void launch_output(const Tensor& x, const Weight& weight, Output output, cudaStream_t stream) {
    constexpr int TileCols = ActiveCols <= 8    ? 8
                             : ActiveCols <= 16 ? 16
                             : ActiveCols <= 24 ? 24
                             : ActiveCols <= 32 ? 32
                             : ActiveCols <= 40 ? 40
                                                : 48;
    using Geometry         = W8LinearGeometry<Rows, kHidden>;
    using Schedule         = W8SmallTMmaDefaultSchedule<TileCols, ActiveCols>;
    const std::size_t dynamic_shared = core::dynamic_shared_carveout(
        sizeof(W8SmallTMmaSharedStorage<Schedule>));
    if (dynamic_shared != 0) {
        static const bool carveout_ok = [] {
            return cudaFuncSetAttribute(
                       w8_small_t_mma_kernel<Geometry, ActiveCols, Schedule, Output>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(sizeof(W8SmallTMmaSharedStorage<Schedule>))) == cudaSuccess;
        }();
        if (!carveout_ok) {
            throw std::runtime_error("W8 small-T shared-memory carveout is unavailable");
        }
    }
    w8_small_t_mma_kernel<Geometry, ActiveCols, Schedule>
        <<<Rows / kRowsPerCta, Schedule::kThreads, dynamic_shared, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output);
}

template <int ActiveCols>
void launch_target_active_cols(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                               Tensor& k, Tensor& v, cudaStream_t stream) {
    static_assert((4096 % kRowsPerCta) == 0 && (512 % kRowsPerCta) == 0);
    const TargetOutput output{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
    launch_output<ActiveCols, kTargetRows>(x, weight, output, stream);
}

template <int ActiveCols>
void launch_companion_active_cols(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                  Tensor& v, cudaStream_t stream) {
    static_assert((4096 % kRowsPerCta) == 0 && (1024 % kRowsPerCta) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    launch_output<ActiveCols, kCompanionRows>(x, weight, output, stream);
}

template <std::size_t... Offsets>
constexpr auto make_target_launchers(std::index_sequence<Offsets...>) {
    return std::array<TargetLauncher, sizeof...(Offsets)>{
        &launch_target_active_cols<kFirstExactCols + static_cast<int>(Offsets)>...};
}

template <std::size_t... Offsets>
constexpr auto make_companion_launchers(std::index_sequence<Offsets...>) {
    return std::array<CompanionLauncher, sizeof...(Offsets)>{
        &launch_companion_active_cols<kFirstExactCols + static_cast<int>(Offsets)>...};
}

constexpr auto kTargetLaunchers =
    make_target_launchers(std::make_index_sequence<kLastTargetExactCols - kFirstExactCols + 1>{});
constexpr auto kCompanionLaunchers = make_companion_launchers(
    std::make_index_sequence<kLastCompanionExactCols - kFirstExactCols + 1>{});

template <int TileCols, int KSplits, int NGroups, int MinBlocks>
void launch_target_medium_cols(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                               Tensor& k, Tensor& v, cudaStream_t stream) {
    static_assert((4096 % kRowsPerCta) == 0 && (512 % kRowsPerCta) == 0);
    const TargetOutput output{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
    w8_rowsplit_medium_t_splitk_kernel<kHidden, TileCols, KSplits, NGroups, MinBlocks>
        <<<kTargetRows / kRowsPerCta, KSplits * NGroups * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, x.ne[1]);
}

template <int TileCols, int KSplits, int NGroups, int MinBlocks>
void launch_companion_medium_cols(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                  Tensor& v, cudaStream_t stream) {
    static_assert((4096 % kRowsPerCta) == 0 && (1024 % kRowsPerCta) == 0);
    const CompanionOutput output{static_cast<__nv_bfloat16*>(q.data),
                                 static_cast<__nv_bfloat16*>(k.data),
                                 static_cast<__nv_bfloat16*>(v.data)};
    w8_rowsplit_medium_t_splitk_kernel<kHidden, TileCols, KSplits, NGroups, MinBlocks>
        <<<kCompanionRows / kRowsPerCta, KSplits * NGroups * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, x.ne[1]);
}

} // namespace

void w8_attn_input_splitk_mma_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream) {
    if (x.ne[1] < kFirstExactCols || x.ne[1] > 64) {
        throw std::invalid_argument("W8 attention input split-K MMA requires T=2..64");
    }
    if (x.ne[1] <= kLastTargetExactCols) {
        kTargetLaunchers[x.ne[1] - kFirstExactCols](x, weight, q, gate, k, v, stream);
    } else if (!core::device_supports_sm100_or_newer()) {
        // Pre-Blackwell SASS has no medium split-K kernels (48KB per-block shared-memory limit);
        // 32-column exact-T chunking covers T=49..64 without a one-column tail.
        std::int32_t offset = 0;
        while (x.ne[1] - offset >= 2) {
            std::int32_t count = x.ne[1] - offset < 32 ? x.ne[1] - offset : 32;
            if (x.ne[1] - offset - count == 1) {
                --count;
            }
            const Tensor x_slice = x.slice(1, offset, count);
            Tensor q_slice        = q.slice(1, offset, count);
            Tensor gate_slice     = gate.slice(1, offset, count);
            Tensor k_slice        = k.slice(1, offset, count);
            Tensor v_slice        = v.slice(1, offset, count);
            kTargetLaunchers[count - kFirstExactCols](x_slice, weight, q_slice, gate_slice, k_slice,
                                                      v_slice, stream);
            offset += count;
        }
    } else {
        launch_target_medium_cols<64, 4, 2, 2>(x, weight, q, gate, k, v, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

void w8_attn_input_splitk_mma_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& k,
                                     Tensor& v, cudaStream_t stream) {
    if (x.ne[1] < kFirstExactCols || x.ne[1] > 96) {
        throw std::invalid_argument("W8 companion attention input split-K MMA requires T=2..96");
    }
    if (x.ne[1] <= kLastCompanionExactCols) {
        kCompanionLaunchers[x.ne[1] - kFirstExactCols](x, weight, q, k, v, stream);
    } else if (!core::device_supports_sm100_or_newer()) {
        // Pre-Blackwell fallback: 32-column exact-T chunking covers T=33..96.
        std::int32_t offset = 0;
        while (x.ne[1] - offset >= 2) {
            std::int32_t count = x.ne[1] - offset < 32 ? x.ne[1] - offset : 32;
            if (x.ne[1] - offset - count == 1) {
                --count;
            }
            const Tensor x_slice = x.slice(1, offset, count);
            Tensor q_slice       = q.slice(1, offset, count);
            Tensor k_slice       = k.slice(1, offset, count);
            Tensor v_slice       = v.slice(1, offset, count);
            kCompanionLaunchers[count - kFirstExactCols](x_slice, weight, q_slice, k_slice, v_slice,
                                                         stream);
            offset += count;
        }
    } else if (x.ne[1] <= 48) {
        launch_companion_medium_cols<48, 4, 2, 3>(x, weight, q, k, v, stream);
    } else if (x.ne[1] <= 64) {
        launch_companion_medium_cols<64, 4, 2, 2>(x, weight, q, k, v, stream);
    } else {
        launch_companion_medium_cols<96, 2, 4, 3>(x, weight, q, k, v, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
