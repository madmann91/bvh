#ifndef BVH_PARALLEL_HIERARCHY_REFITTER_H
#define BVH_PARALLEL_HIERARCHY_REFITTER_H

#include <cstddef>

#include "bvh/bvh.h"
#include "bvh/parallel_bottom_up_traverser.h"

namespace bvh {

template <typename Bvh>
class ParallelHierarchyRefitter {
    using Node = typename Bvh::Node;

    ParallelBottomUpTraverser<Bvh> traverser_;

public:
    /// Refits every node of the BVH in parallel, using the given function to
    /// update the contents of a leaf.
    template <typename UpdateLeaf>
    void refit(Bvh& bvh, const std::vector<size_t>& parents, UpdateLeaf&& update_leaf) {
        traverser_.traverse(
            bvh, parents,
            [&] (size_t i) { update_leaf(bvh.nodes[i]); },
            [&] (size_t i) {
                auto& node  = bvh.nodes[i];
                auto& left  = bvh.nodes[node.first_index + 0];
                auto& right = bvh.nodes[node.first_index + 1];
                node.bbox_proxy() = left.bbox().extend(right.bbox());
            });
    }

    void refit(Bvh& bvh, const std::vector<size_t>& parents) {
        refit(bvh, parents, [] (const Node&) {});
    }

    void refit(Bvh& bvh) {
        refit(bvh, bvh.parents(std::execution::par_unseq));
    }
};

} // namespace bvh

#endif
