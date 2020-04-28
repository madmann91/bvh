#ifndef BVH_BOTTOM_UP_ALGORITHM_HPP
#define BVH_BOTTOM_UP_ALGORITHM_HPP

#include <memory>
#include <cstddef>

#include "bvh/bvh.hpp"
#include "bvh/platform.hpp"

namespace bvh {

/// Base class for bottom-up BVH traversal algorithms. The implementation is inspired
/// from T. Karras' bottom-up refitting algorithm, explained in the article
/// "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees".
template <typename Bvh, bool MaintainChildIndices = false>
class BottomUpAlgorithm {
protected:
    std::unique_ptr<size_t[]> parents, children;
    std::unique_ptr<int[]> flags;

    Bvh& bvh;

    BottomUpAlgorithm(Bvh& bvh)
        : bvh(bvh)
    {
        bvh__assert_not_in_parallel();
        parents = std::make_unique<size_t[]>(bvh.node_count);
        flags   = std::make_unique<int[]>(bvh.node_count);

        if (MaintainChildIndices)
            children = std::make_unique<size_t[]>(bvh.node_count);

        parents[0] = 0;

        // Compute parent/children indices
        #pragma omp parallel for
        for (size_t i = 0; i < bvh.node_count; i++) {
            auto& node = bvh.nodes[i];
            if (node.is_leaf)
                continue;
            auto first_child = node.first_child_or_primitive;
            parents[first_child + 0] = i;
            parents[first_child + 1] = i;
            if (MaintainChildIndices)
                children[i] = first_child;
        }
    }

    ~BottomUpAlgorithm() {}

    template <typename ProcessLeaf, typename ProcessInnerNode>
    void traverse_in_parallel(
        const ProcessLeaf& process_leaf,
        const ProcessInnerNode& process_inner_node)
    {
        bvh__assert_in_parallel();

        #pragma omp for
        for (size_t i = 1; i < bvh.node_count; ++i) {
            // Only process leaves
            if (MaintainChildIndices ? children[i] != 0 : bvh.nodes[i].is_leaf)
                continue;

            process_leaf(i);

            // Process inner nodes on the path from that leaf up to the root
            size_t j = i;
            do {
                j = parents[j];

                // Make sure that the children of this inner node have been processed
                int previous_flag;
                #pragma omp atomic capture
                { previous_flag = flags[j]; flags[j]++; }
                if (previous_flag != 1)
                    break;
                flags[j] = 0;

                process_inner_node(j);
            } while (j != 0);
        }
    }
};

} // namespace bvh

#endif
