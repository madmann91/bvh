#ifndef BVH_PARALLEL_HIERARCHY_REFITTER_H
#define BVH_PARALLEL_HIERARCHY_REFITTER_H

#include <cstddef>

#include "bvh/bvh.h"
#include "bvh/parallel_bottom_up_traverser.h"
#include "bvh/sequential_loop_scheduler.h"

namespace bvh {

template <typename Bvh, typename LoopScheduler>
class ParallelHierarchyRefitter {
    using Node = typename Bvh::Node;

    ParallelBottomUpTraverser<Bvh, LoopScheduler> traverser_;

public:
    ParallelHierarchyRefitter(LoopScheduler& loop_scheduler)
        : traverser_(loop_scheduler)
    {}

    LoopScheduler& loop_scheduler() const {
        return traverser_.loop_scheduler;
    }

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
        refit(bvh, bvh.parents(loop_scheduler()));
    }
};

} // namespace bvh

#endif
