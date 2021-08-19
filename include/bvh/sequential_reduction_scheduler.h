#ifndef BVH_SEQUENTIAL_REDUCTION_SCHEDULER_H
#define BVH_SEQUENTIAL_REDUCTION_SCHEDULER_H

#include <type_traits>
#include <numeric>
#include <ranges>

namespace bvh {

/// Helper class that performs a reduction using C++'s standard library.
class SequentialReductionScheduler {
public:
    template <
        typename T,
        typename Index,
        typename BinaryOp,
        typename UnaryOp,
        std::enable_if_t<std::is_integral_v<Index>, int> = 0>
    static T run(
        Index begin, Index end, T init,
        const BinaryOp& bin_op,
        const UnaryOp& un_op)
    {
        auto view = std::views::iota(begin, end);
        return std::transform_reduce(view.begin(), view.end(), init, bin_op, un_op);
    }
};

} // namespace bvh

#endif
