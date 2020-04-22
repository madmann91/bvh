#ifndef BVH_LEAF_COLLAPSER_HPP
#define BVH_LEAF_COLLAPSER_HPP

#include <memory>
#include <stack>

#include "bvh/bvh.hpp"
#include "bvh/sah_based_algorithm.hpp"
#include "bvh/prefix_sum.hpp"

namespace bvh {

/// Collapses leaves of the BVH according to the SAH. This optimization
/// is only helpful for bottom-up builders, as top-down builders already
/// have a termination criterion that prevents leaf creation when the SAH
/// cost does not improve.
template <typename Bvh>
class LeafCollapser : public SahBasedAlgorithm<Bvh> {
    using Scalar = typename Bvh::ScalarType;

    PrefixSum<size_t> prefix_sum;

    Bvh& bvh;

    void compute_first_primitives(
        size_t node_index,
        const size_t* bvh__restrict__ children,
        const size_t* bvh__restrict__ primitive_counts,
        size_t* bvh__restrict__ first_primitives)
    {
        static constexpr size_t task_spawn_threshold = 256;
        std::stack<size_t> stack;
        stack.push(node_index); 
        while (!stack.empty()) {
            auto i = stack.top();
            stack.pop();
            auto first_child = children[i];
            if (first_child != 0) {
                auto first_primitive = first_primitives[i];
                auto primitive_count = primitive_counts[i];
                first_primitives[first_child + 0] = first_primitive;
                first_primitives[first_child + 1] = first_primitive + primitive_counts[first_child + 0];
                if (primitive_count > task_spawn_threshold) {
                    #pragma omp task
                    { compute_first_primitives(first_child + 0, children, primitive_counts, first_primitives); }
                    #pragma omp task
                    { compute_first_primitives(first_child + 1, children, primitive_counts, first_primitives); }
                } else {
                    stack.push(first_child + 0);
                    stack.push(first_child + 1);
                }
            }
        }
    }

public:
    using SahBasedAlgorithm<Bvh>::traversal_cost;

    LeafCollapser(Bvh& bvh)
        : bvh(bvh)
    {}

    void collapse(size_t) {
        auto parents  = std::make_unique<size_t[]>(bvh.node_count);
        auto children = std::make_unique<size_t[]>(bvh.node_count);
        auto flags    = std::make_unique<int[]>(bvh.node_count);

        auto node_index       = std::make_unique<size_t[]>(bvh.node_count / 2 + 1);
        auto primitive_counts = std::make_unique<size_t[]>(bvh.node_count);
        auto first_primitives = std::make_unique<size_t[]>(bvh.node_count);

        std::unique_ptr<size_t[]> primitive_indices_copy;
        std::unique_ptr<typename Bvh::Node[]> nodes_copy;

        parents[0] = 0;
        node_index[0] = 1;

        #pragma omp parallel
        {
            // Compute parent indices
            #pragma omp for
            for (size_t i = 0; i < bvh.node_count; i++) {
                auto& node = bvh.nodes[i];
                if (node.is_leaf) {
                    primitive_counts[i] = node.primitive_count;
                    children[i] = 0;
                    continue;
                }
                auto first_child = node.first_child_or_primitive;
                parents[first_child + 0] = i;
                parents[first_child + 1] = i;
                primitive_counts[i] = 0;
                children[i] = first_child;
            }

            // Bottom-up traversal that collapses leaves according to the SAH
            #pragma omp for
            for (size_t i = 1; i < bvh.node_count; ++i) {
                // Only process leaves
                if (children[i] != 0)
                    continue;
                // Merge up to the root
                size_t j = i;
                do {
                    j = parents[j];

                    // Make sure that the children of this node have been processed
                    int previous_flag;
                    #pragma omp atomic capture
                    { previous_flag = flags[j]; flags[j]++; }
                    if (previous_flag != 1)
                        break;

                    auto& node = bvh.nodes[j];
                    assert(!node.is_leaf);
                    auto first_child  = node.first_child_or_primitive;
                    auto& left_child  = bvh.nodes[first_child + 0];
                    auto& right_child = bvh.nodes[first_child + 1];

                    auto left_primitive_count  = primitive_counts[first_child + 0];
                    auto right_primitive_count = primitive_counts[first_child + 1];
                    auto total_primitive_count = left_primitive_count + right_primitive_count;
                    primitive_counts[j] = total_primitive_count;

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
                            continue;
                        }
                    }

                    node_index[(first_child + 1) / 2] = 2;
                } while (j != 0);
            }

            prefix_sum.sum(node_index.get(), node_index.get(), bvh.node_count / 2 + 1);

            #pragma omp single
            {
                nodes_copy = std::make_unique<typename Bvh::Node[]>(node_index[bvh.node_count / 2]);
                primitive_indices_copy = std::make_unique<size_t[]>(primitive_counts[0]);
                nodes_copy[0] = bvh.nodes[0];

                compute_first_primitives(0, children.get(), primitive_counts.get(), first_primitives.get());
            }

            #pragma omp for
            for (size_t i = 1; i < bvh.node_count; i++) {
                if (!bvh.nodes[i].is_leaf || node_index[(i - 1) / 2] == node_index[(i + 1) / 2])
                    continue;

                // Find the index of the first primitive in this subtree
                size_t first_primitive = first_primitives[i];//0;
                size_t j = i;
                /*while (j != 0) {
                    if (!bvh.is_left_sibling(j))
                        first_primitive += primitive_counts[bvh.sibling(j)];
                    j = parents[j];
                }*/

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

            #pragma omp for
            for (size_t i = 1; i < bvh.node_count; i += 2) {
                auto j = node_index[(i - 1) / 2];
                if (j == node_index[(i + 1) / 2])
                    continue;
                nodes_copy[j + 0] = bvh.nodes[i + 0];
                nodes_copy[j + 1] = bvh.nodes[i + 1];
                if (!bvh.nodes[i + 0].is_leaf)
                    nodes_copy[j + 0].first_child_or_primitive = node_index[(bvh.nodes[i + 0].first_child_or_primitive - 1) / 2];
                if (!bvh.nodes[i + 1].is_leaf)
                    nodes_copy[j + 1].first_child_or_primitive = node_index[(bvh.nodes[i + 1].first_child_or_primitive - 1) / 2];
            }
        }

        std::swap(bvh.nodes, nodes_copy);
        std::swap(bvh.primitive_indices, primitive_indices_copy);
        bvh.node_count = node_index[bvh.node_count / 2];
    }
};

} // namespace bvh

#endif
