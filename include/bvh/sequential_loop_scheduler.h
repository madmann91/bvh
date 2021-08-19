#ifndef BVH_SEQUENTIAL_LOOP_SCHEDULER_H
#define BVH_SEQUENTIAL_LOOP_SCHEDULER_H

#include <type_traits>
#include <ranges>

namespace bvh {

/// Helper class that runs a loop using C++'s standard library.
class SequentialLoopScheduler {
public:
    template <typename Index, typename UnaryOp, std::enable_if_t<std::is_integral_v<Index>, int> = 0>
    static void run(Index begin, Index end, const UnaryOp& un_op) {
        auto view = std::views::iota(begin, end);
        std::for_each(view.begin(), view.end(), un_op);
    }
};

} // namespace bvh

#endif
