#ifndef BVH_MORTON_HPP
#define BVH_MORTON_HPP

#include <cstddef>
#include <climits>

#include "bvh/utilities.hpp"

namespace bvh {

/// Split an unsigned integer such that its bits are spaced by 2 zeros.
/// For instance, morton_split(0b00110010) = 0b000000001001000000001000.
template <typename T>
T morton_split(T x) {
    constexpr size_t log_bits = RoundUpLog2<sizeof(T) * CHAR_BIT>::value;
    auto mask = T(-1);
    for (size_t i = log_bits, n = 1 << log_bits; i > 0; --i, n >>= 1) {
        mask = (mask | (mask << n)) & ~(mask << (n / 2));
        x = (x | (x << n)) & mask;
    }
    return x;
}

/// Morton-encode three unsigned integers into one integer.
template <typename T>
T morton_encode(T x, T y, T z) {
    return morton_split(x)       |
          (morton_split(y) << 1) |
          (morton_split(z) << 2);
}

} // namespace bvh

#endif
