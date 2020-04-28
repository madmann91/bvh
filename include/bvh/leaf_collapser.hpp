#ifndef BVH_LEAF_COLLAPSER_HPP
#define BVH_LEAF_COLLAPSER_HPP

#include <memory>

#include "bvh/bvh.hpp"
#include "bvh/sah_based_algorithm.hpp"
#include "bvh/bottom_up_algorithm.hpp"
#include "bvh/prefix_sum.hpp"
#include "bvh/platform.hpp"

namespace bvh {

/// Collapses leaves of the BVH according to the SAH. This optimization
/// is only helpful for bottom-up builders, as top-down builders already
/// have a termination criterion that prevents leaf creation when the SAH
/// cost does not improve.
template <typename Bvh>
class LeafCollapser : public SahBasedAlgorithm<Bvh>, public BottomUpAlgorithm<Bvh, true> {
    using Scalar = typename Bvh::ScalarType;

    PrefixSum<size_t> prefix_sum;

    using BottomUpAlgorithm<Bvh, true>::traverse_in_parallel;
    using BottomUpAlgorithm<Bvh, true>::children;
    using BottomUpAlgorithm<Bvh, true>::parents;
    using BottomUpAlgorithm<Bvh, true>::bvh;

public:
    using SahBasedAlgorithm<Bvh>::traversal_cost;

    LeafCollapser(Bvh& bvh)
        : BottomUpAlgorithm<Bvh, true>(bvh)
    {}

    void collapse() {
        if (bvh__unlikely(bvh.nodes[0].is_leaf))
            return;

        std::unique_ptr<size_t[]> primitive_indices_copy;
        std::unique_ptr<typename Bvh::Node[]> nodes_copy;

        auto node_index       = std::make_unique<size_t[]>(bvh.node_count / 2 + 1);
        auto primitive_counts = std::make_unique<size_t[]>(bvh.node_count);

        node_index[0] = 1;

        #pragma omp parallel
        {
            // Bottom-up traversal to collapse leaves
            traverse_in_parallel(
                [&] (size_t i) { primitive_counts[i] = bvh.nodes[i].primitive_count; },
                [&] (size_t i) {
                    auto& node = bvh.nodes[i];
                    assert(!node.is_leaf);
                    auto first_child  = node.first_child_or_primitive;
                    auto& left_child  = bvh.nodes[first_child + 0];
                    auto& right_child = bvh.nodes[first_child + 1];

                    auto left_primitive_count  = primitive_counts[first_child + 0];
                    auto right_primitive_count = primitive_counts[first_child + 1];
                    auto total_primitive_count = left_primitive_count + right_primitive_count;
                    primitive_counts[i] = total_primitive_count;

                    // Compute the cost of collapsing this node when both children are leaves
                    if (left_child.is_leaf && right_child.is_leaf) {
                        auto collapse_cost =
                            node.bounding_box_proxy().to_bounding_box().half_area() * (Scalar(total_primitive_count) - traversal_cost);
                        auto base_cost =
                            left_child .bounding_box_proxy().to_bounding_box().half_area() * left_primitive_count +
                            right_child.bounding_box_proxy().to_bounding_box().half_area() * right_primitive_count;
                        if (collapse_cost <= base_cost) {
                            node_index[(first_child + 1) / 2] = 0;
                            node.is_leaf = true;
                            return;
                        }
                    }

                    node_index[(first_child + 1) / 2] = 2;
                });

            prefix_sum.sum_in_parallel(node_index.get(), node_index.get(), bvh.node_count / 2 + 1);

            #pragma omp single
            {
                nodes_copy = std::make_unique<typename Bvh::Node[]>(node_index[bvh.node_count / 2]);
                primitive_indices_copy = std::make_unique<size_t[]>(primitive_counts[0]);
                nodes_copy[0] = bvh.nodes[0];
                nodes_copy[0].first_child_or_primitive =
                    node_index[(bvh.nodes[0].first_child_or_primitive - 1) / 2];
            }

            #pragma omp for
            for (size_t i = 1; i < bvh.node_count; i++) {
                if (!bvh.nodes[i].is_leaf || node_index[(i - 1) / 2] == node_index[(i + 1) / 2])
                    continue;

                // Find the index of the first primitive in this subtree
                size_t first_primitive = 0;
                size_t j = i;
                while (j != 0) {
                    if (!bvh.is_left_sibling(j))
                        first_primitive += primitive_counts[bvh.sibling(j)];
                    j = parents[j];
                }

                // Top-down traversal to store the primitives contained in this subtree.
                j = i;
                while (true) {
                    if (children[j] == 0) {
                        auto& node = bvh.nodes[j];
                        std::copy(
                            bvh.primitive_indices.get() + node.first_child_or_primitive,
                            bvh.primitive_indices.get() + node.first_child_or_primitive + node.primitive_count,
                            primitive_indices_copy.get() + first_primitive);
                        first_primitive += node.primitive_count;
                        while (!bvh.is_left_sibling(j) && j != i)
                            j = parents[j];
                        if (j == i)
                            break;
                        j = bvh.sibling(j);
                    } else
                        j = children[j];
                }

                bvh.nodes[i].first_child_or_primitive = first_primitive - primitive_counts[i];
                bvh.nodes[i].primitive_count = primitive_counts[i];
            }

            // Create the new nodes
            #pragma omp for
            for (size_t i = 1; i < bvh.node_count; i += 2) {
                auto j = node_index[(i - 1) / 2];
                if (j == node_index[(i + 1) / 2])
                    continue;
                nodes_copy[j + 0] = bvh.nodes[i + 0];
                nodes_copy[j + 1] = bvh.nodes[i + 1];
                if (!bvh.nodes[i + 0].is_leaf)
                    nodes_copy[j + 0].first_child_or_primitive =
                        node_index[(bvh.nodes[i + 0].first_child_or_primitive - 1) / 2];
                if (!bvh.nodes[i + 1].is_leaf)
                    nodes_copy[j + 1].first_child_or_primitive =
                        node_index[(bvh.nodes[i + 1].first_child_or_primitive - 1) / 2];
            }
        }

        std::swap(bvh.nodes, nodes_copy);
        std::swap(bvh.primitive_indices, primitive_indices_copy);
        bvh.node_count = node_index[bvh.node_count / 2];
    }
};

} // namespace bvh

#endif
