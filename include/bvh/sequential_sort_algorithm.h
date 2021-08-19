#ifndef BVH_SEQUENTIAL_SORT_ALGORITHM_H
#define BVH_SEQUENTIAL_SORT_ALGORITHM_H

#include <algorithm>

namespace bvh {

/// Helper class that sorts an array using C++'s standard library.
class SequentialSortAlgorithm {
public:
    template <typename Iterator, typename CmpOp>
    static void run(Iterator begin, Iterator end, const CmpOp& cmp_op) {
        std::sort(begin, end, cmp_op);
    }
};

} // namespace bvh

#endif
