#ifndef BVH_LOCALLY_ORDERED_CLUSTERING_BUILDER_HPP
#define BVH_LOCALLY_ORDERED_CLUSTERING_BUILDER_HPP

#include <numeric>

#include "bvh/morton_code_based_builder.hpp"
#include "bvh/prefix_sum.hpp"

namespace bvh {

/// Bottom-up BVH builder based on agglomerative clustering. The algorithm starts
/// by sorting primitives by their Morton code, and then clusters them iteratively
/// to form the BVH nodes. Clusters are built starting from each primitive, by
/// agglomerating nearby clusters that minimize a distance metric. The distance
/// metric is in this case the area of the union of the bounding boxes of the two
/// clusters of interest.
/// See "Parallel Locally-Ordered Clustering for Bounding Volume Hierarchy Construction",
/// by D. Meister and J. Bittner.
template <typename Bvh, typename Morton>
class LocallyOrderedClusteringBuilder : public MortonCodeBasedBuilder<Bvh, Morton> {
    using Scalar = typename Bvh::ScalarType;
    using Node   = typename Bvh::Node;

    using ParentBuilder = MortonCodeBasedBuilder<Bvh, Morton>;
    using ParentBuilder::sort_primitives_by_morton_code;

    Bvh& bvh;

    PrefixSum<size_t> prefix_sum;

    std::pair<size_t, size_t> cluster(
        const Node* bvh__restrict__ input,
        Node* bvh__restrict__ output,
        size_t* bvh__restrict__ neighbors,
        size_t* bvh__restrict__ merged_index,
        size_t begin, size_t end,
        size_t previous_end)
    {
        size_t next_begin = 0;
        size_t next_end   = 0;

        #pragma omp parallel if (end - begin > loop_parallel_threshold)
        {
            // Nearest neighbor search
            #pragma omp for
            for (size_t i = begin; i < end; ++i) {
                size_t search_begin = i > begin + search_radius   ? i - search_radius     : begin;
                size_t search_end   = i + search_radius + 1 < end ? i + search_radius + 1 : end;
                Scalar best_distance = std::numeric_limits<Scalar>::max();
                size_t best_neighbor = -1;
                for (size_t j = search_begin; j < search_end; ++j) {
                    if (j == i)
                        continue;
                    auto distance = input[i]
                        .bounding_box_proxy()
                        .to_bounding_box()
                        .extend(input[j].bounding_box_proxy())
                        .half_area();
                    if (distance < best_distance) {
                        best_distance = distance;
                        best_neighbor = j;
                    }
                }
                neighbors[i] = best_neighbor;
            }

            // Mark nodes that are the closest as merged, but keep
            // the one with lowest index to act as the parent
            #pragma omp for
            for (size_t i = begin; i < end; ++i) {
                auto j = neighbors[i];
                bool is_mergeable = neighbors[j] == i;
                merged_index[i] = i < j && is_mergeable ? 1 : 0;
            }

            // Perform a prefix sum to compute the insertion indices
            prefix_sum.sum(merged_index + begin, merged_index + begin, end - begin);
            size_t merged_count   = merged_index[end - 1];
            size_t unmerged_count = end - begin - merged_count;
            size_t children_count = merged_count * 2;
            size_t children_begin = end - children_count;
            size_t unmerged_begin = end - (children_count + unmerged_count);

            #pragma omp single nowait
            {
                next_begin = unmerged_begin;
                next_end   = children_begin;
            }

            // Finally, merge nodes that are marked for merging and create
            // their parents using the indices computed previously.
            #pragma omp for nowait
            for (size_t i = begin; i < end; ++i) {
                auto j = neighbors[i];
                if (neighbors[j] == i) {
                    if (i < j) {
                        auto& unmerged_node = output[unmerged_begin + j - begin - merged_index[j]];
                        auto first_child = children_begin + (merged_index[i] - 1) * 2;
                        unmerged_node.bounding_box_proxy() = input[j]
                            .bounding_box_proxy()
                            .to_bounding_box()
                            .extend(input[i].bounding_box_proxy());
                        unmerged_node.is_leaf = false;
                        unmerged_node.first_child_or_primitive = first_child;
                        output[first_child + 0] = input[i];
                        output[first_child + 1] = input[j];
                    }
                } else {
                    output[unmerged_begin + i - begin - merged_index[i]] = input[i];
                }
            }

            // Copy the nodes of the previous level into the current array of nodes.
            #pragma omp for nowait
            for (size_t i = end; i < previous_end; ++i)
                output[i] = input[i];
        }

        return std::make_pair(next_begin, next_end);
    }

public:
    using ParentBuilder::loop_parallel_threshold;

    /// Parameter of the algorithm. The larger the search radius,
    /// the longer the search for neighboring nodes lasts.
    size_t search_radius = 14;

    LocallyOrderedClusteringBuilder(Bvh& bvh)
        : bvh(bvh)
    {}

    void build(
        const BoundingBox<Scalar>& global_bbox,
        const BoundingBox<Scalar>* bboxes,
        const Vector3<Scalar>* centers,
        size_t primitive_count)
    {
        auto primitive_indices =
            sort_primitives_by_morton_code(global_bbox, centers, primitive_count).first;

        auto node_count     = 2 * primitive_count - 1;
        auto nodes          = std::make_unique<Node[]>(node_count);
        auto nodes_copy     = std::make_unique<Node[]>(node_count);
        auto auxiliary_data = std::make_unique<size_t[]>(node_count * 3);

        size_t begin        = node_count - primitive_count;
        size_t end          = node_count;
        size_t previous_end = end;

        // Create the leaves
        #pragma omp parallel for
        for (size_t i = 0; i < primitive_count; ++i) {
            auto& node = nodes[begin + i];
            node.bounding_box_proxy()     = bboxes[primitive_indices[i]];
            node.is_leaf                  = true;
            node.primitive_count          = 1;
            node.first_child_or_primitive = i;
        }

        while (end - begin > 1) {
            auto [next_begin, next_end] = cluster(
                nodes.get(),
                nodes_copy.get(),
                auxiliary_data.get(),
                auxiliary_data.get() + node_count,
                begin, end,
                previous_end);

            std::swap(nodes_copy, nodes);

            previous_end = end;
            begin        = next_begin;
            end          = next_end;
        }

        std::swap(bvh.nodes, nodes);
        std::swap(bvh.primitive_indices, primitive_indices);
        bvh.node_count = node_count;
    }
};

} // namespace bvh

#endif
