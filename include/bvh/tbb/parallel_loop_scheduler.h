#ifndef BVH_TBB_PARALLEL_LOOP_SCHEDULER_H
#define BVH_TBB_PARALLEL_LOOP_SCHEDULER_H

#include <type_traits>
#include <numeric>
#include <ranges>

#define TBB_SUPPRESS_DEPRECATED_MESSAGES 1
#include <tbb/tbb.h>

namespace bvh::tbb {

/// Helper class that runs a loop in parallel using TBB.
class ParallelLoopScheduler {
public:
    template <typename Index, typename UnaryOp, std::enable_if_t<std::is_integral_v<Index>, int> = 0>
    static void run(Index begin, Index end, const UnaryOp& un_op) {
        ::tbb::parallel_for(::tbb::blocked_range<Index>(begin, end),
            [&] (auto& range) {
                auto view = std::views::iota(range.begin(), range.end());
                std::for_each(view.begin(), view.end(), un_op);
            });
    }
};

} // namespace bvh::tbb

#endif
