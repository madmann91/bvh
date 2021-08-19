#ifndef BVH_TBB_PARALLEL_SORT_ALGORITHM_H
#define BVH_TBB_PARALLEL_SORT_ALGORITHM_H

#define TBB_SUPPRESS_DEPRECATED_MESSAGES 1
#include <tbb/tbb.h>

namespace bvh::tbb {

/// Helper class that sorts an array in parallel using TBB.
class ParallelSortAlgorithm {
public:
    template <typename Iterator, typename CmpOp>
    static void run(Iterator begin, Iterator end, const CmpOp& cmp_op) {
        ::tbb::parallel_sort(begin, end, cmp_op);
    }
};

} // namespace bvh::tbb

#endif
