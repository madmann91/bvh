#ifndef BVH_HIERARCHY_REFITTER_HPP
#define BVH_HIERARCHY_REFITTER_HPP

#include "bvh/bvh.hpp"
#include "bvh/bottom_up_algorithm.hpp"

namespace bvh {

template <typename Bvh>
class HierarchyRefitter : public BottomUpAlgorithm<Bvh> {
    using BottomUpAlgorithm<Bvh>::bvh;
    using BottomUpAlgorithm<Bvh>::traverse;

public:
    HierarchyRefitter(Bvh& bvh)
        : BottomUpAlgorithm<Bvh>(bvh)
    {}

    template <typename UpdateLeaf>
    void refit(const UpdateLeaf& update_leaf) {
        #pragma omp parallel
        {
            // Refit every node of the tree in parallel
            traverse(
                [&] (size_t i) { update_leaf(bvh.nodes[i]); },
                [&] (size_t i) {
                    auto& node = bvh.nodes[i];
                    auto first_child = node.first_child_or_primitive;
                    node.bounding_box_proxy() = bvh.nodes[first_child + 0]
                        .bounding_box_proxy()
                        .to_bounding_box()
                        .extend(bvh.nodes[first_child + 1].bounding_box_proxy());
                });
        }
    }
};

} // namespace bvh

#endif
