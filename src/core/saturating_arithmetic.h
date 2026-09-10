#pragma once

#include <cstdint>
#include <limits>

namespace ninfer::core {

inline constexpr std::uint64_t kUint64Maximum = std::numeric_limits<std::uint64_t>::max();

// Saturating 64-bit unsigned addition.
constexpr std::uint64_t saturating_add(std::uint64_t left, std::uint64_t right) noexcept {
    return right > kUint64Maximum - left ? kUint64Maximum : left + right;
}

// Saturating 64-bit unsigned multiplication.
constexpr std::uint64_t saturating_multiply(std::uint64_t left, std::uint64_t right) noexcept {
    if (left == 0 || right == 0) { return 0; }
    return left > kUint64Maximum / right ? kUint64Maximum : left * right;
}

// High 64 bits of the exact 128-bit product. The runtime cost arithmetic needs wide products, and
// MSVC has no native 128-bit integer type, so the product is assembled from 32-bit limbs:
// left*right = p00 + 2^32 * (p01 + p10) + 2^64 * p11. Every partial sum below is bounded by the
// exact high half, which is itself below 2^64.
constexpr std::uint64_t multiply_high(std::uint64_t left, std::uint64_t right) noexcept {
    constexpr std::uint64_t kMask32 = 0xffff'ffffULL;
    const std::uint64_t left_low   = left & kMask32;
    const std::uint64_t left_high  = left >> 32U;
    const std::uint64_t right_low  = right & kMask32;
    const std::uint64_t right_high = right >> 32U;

    const std::uint64_t low_product  = left_low * right_low;
    const std::uint64_t cross_left   = left_low * right_high;
    const std::uint64_t cross_right  = left_high * right_low;
    const std::uint64_t high_product = left_high * right_high;

    const std::uint64_t carry =
        ((low_product >> 32U) + (cross_left & kMask32) + (cross_right & kMask32)) >> 32U;
    return high_product + (cross_left >> 32U) + (cross_right >> 32U) + carry;
}

} // namespace ninfer::core
