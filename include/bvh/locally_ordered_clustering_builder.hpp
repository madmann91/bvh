#ifndef BVH_LOCALLY_ORDERED_CLUSTERING_BUILDER_HPP
#define BVH_LOCALLY_ORDERED_CLUSTERING_BUILDER_HPP

#include "bvh/morton_code_based_builder.hpp"

namespace bvh {

template <typename Bvh, typename Morton>
class LocallyOrderedClusteringBuilder : MortonCodeBasedBuilder<Bvh, Morton> {
public:
    using Scalar = typename Bvh::ScalarType;
    using Node   = typename Bvh::Node;

    using ParentBuilder = MortonCodeBasedBuilder<Bvh, Morton>;
    using ParentBuilder::sort_primitives_by_morton_code;

    Bvh& bvh;

    /// Parameter of the algorithm. The larger the search radius,
    /// the longer the search for neighboring nodes lasts.
    size_t search_radius = 14;

    /// Threshold (number of nodes) under which the algorithm
    /// executes loops serially.
    size_t parallel_threshold = 1024;

    LocallyOrderedClusteringBuilder(Bvh& bvh)
        : bvh(bvh)
    {}

    std::pair<size_t, size_t> cluster(
        const Node* __restrict input,
        Node* __restrict output,
        size_t* auxiliary_data,
        size_t data_size,
        size_t begin, size_t end,
        size_t previous_end)
    {
        size_t* __restrict neighbors   = auxiliary_data;
        size_t* __restrict next_index  = auxiliary_data + data_size;
        size_t* __restrict child_index = auxiliary_data + data_size * 2;

        size_t next_begin;
        size_t next_end;

        size_t unmerged_count = 0;
        size_t children_count = 0;

        #pragma omp parallel if (end - begin > parallel_threshold)
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
                next_index[i]  = i > j && is_mergeable ? 0 : 1;
                child_index[i] = i < j && is_mergeable ? 2 : 0;
            }

            // Perform a prefix sum to compute the insertion indices
            #pragma omp sections
            {
                #pragma omp section
                {
                    unmerged_count = 0;
                    for (size_t i = begin; i < end; ++i) {
                        size_t count = unmerged_count;
                        unmerged_count += next_index[i];
                        next_index[i] = count;
                    }
                }
                #pragma omp section
                {
                    children_count = 0;
                    for (size_t i = begin; i < end; ++i) {
                        size_t count = children_count;
                        children_count += child_index[i];
                        child_index[i] = count;
                    }
                }
            }

            size_t children_begin = end - children_count;
            size_t unmerged_begin = children_begin - unmerged_count;

            // Finally, merge nodes that are marked for merging and create
            // their parents using the indices computed previously.
            #pragma omp for nowait
            for (size_t i = begin; i < end; ++i) {
                auto j = neighbors[i];
                if (neighbors[j] == i) {
                    if (i < j) {
                        auto& merged_node = output[unmerged_begin + next_index[i]];
                        auto first_child = children_begin + child_index[i];
                        merged_node.bounding_box_proxy() = input[j]
                            .bounding_box_proxy()
                            .to_bounding_box()
                            .extend(input[i].bounding_box_proxy());
                        merged_node.is_leaf = false;
                        merged_node.first_child_or_primitive = first_child;
                        output[first_child + 0] = input[i];
                        output[first_child + 1] = input[j];
                    }
                } else {
                    output[unmerged_begin + next_index[i]] = input[i];
                }
            }

            // Copy the nodes of the previous level into the current array of nodes.
            #pragma omp for nowait
            for (size_t i = end; i < previous_end; ++i)
                output[i] = input[i];

            #pragma omp single nowait
            {
                next_begin = unmerged_begin;
                next_end   = children_begin;
            }
        }

        return std::make_pair(next_begin, next_end);
    }

    void build(
        const BoundingBox<Scalar>* bboxes,
        const Vector3<Scalar>* centers,
        size_t primitive_count)
    {
        auto primitive_indices = sort_primitives_by_morton_code(bboxes, centers, primitive_count);

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
                node_count,
                begin, end,
                previous_end);

            std::swap(nodes_copy, nodes);

            begin        = next_begin;
            end          = next_end;
            previous_end = end;
        }

        std::swap(bvh.nodes, nodes);
        std::swap(bvh.primitive_indices, primitive_indices);
        bvh.node_count = node_count;
    }
};

} // namespace bvh

#endif
