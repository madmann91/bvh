#ifndef BVH_OMP_PARALLEL_LOOP_SCHEDULER_H
#define BVH_OMP_PARALLEL_LOOP_SCHEDULER_H

#include <type_traits>

namespace bvh::omp {

/// Helper class that runs a loop in parallel using OpenMP.
class ParallelLoopScheduler {
public:
    template <typename Index, typename UnaryOp, std::enable_if_t<std::is_integral_v<Index>, int> = 0>
    static void run(Index begin, Index end, const UnaryOp& un_op) {
        #pragma omp parallel for
        for (auto i = begin; i < end; ++i)
            un_op(i);
    }
};

} // namespace bvh::omp

#endif
