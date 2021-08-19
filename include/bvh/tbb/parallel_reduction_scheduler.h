#ifndef BVH_TBB_PARALLEL_REDUCTION_SCHEDULER_H
#define BVH_TBB_PARALLEL_REDUCTION_SCHEDULER_H

#include <type_traits>
#include <numeric>
#include <ranges>

#define TBB_SUPPRESS_DEPRECATED_MESSAGES 1
#include <tbb/tbb.h>

namespace bvh::tbb {

/// Helper class that performs a reduction in parallel using TBB.
class ParallelReductionScheduler {
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
        return ::tbb::parallel_reduce(
            ::tbb::blocked_range<Index>(begin, end), init,
            [&] (auto& range, auto result) {
                auto view = std::views::iota(range.begin(), range.end());
                return std::transform_reduce(view.begin(), view.end(), result, bin_op, un_op);
            }, bin_op);
    }
};

} // namespace bvh::tbb

#endif
