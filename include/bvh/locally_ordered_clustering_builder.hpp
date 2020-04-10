#ifndef BVH_LOCALLY_ORDERED_CLUSTERING_BUILDER_HPP
#define BVH_LOCALLY_ORDERED_CLUSTERING_BUILDER_HPP

#include "bvh/morton_code_based_builder.hpp"

namespace bvh {

template <typename Bvh, typename Morton>
class LocallyOrderedClusteringBuilder : MortonCodeBasedBuilder<Bvh, Morton> {
public:
    using Scalar = typename Bvh::ScalarType;

    using ParentBuilder = MortonCodeBasedBuilder<Bvh, Morton>;
    using ParentBuilder::sort_primitives_by_morton_code;

    Bvh& bvh;

    /// Parameter of the algorithm. The larger the search radius,
    /// the longer the search for neighboring nodes lasts.
    size_t search_radius = 14;

    LocallyOrderedClusteringBuilder(Bvh& bvh)
        : bvh(bvh)
    {}

    void build(
        const BoundingBox<Scalar>* bboxes,
        const Vector3<Scalar>* centers,
        size_t primitive_count)
    {
        auto primitive_indices = sort_primitives_by_morton_code(bboxes, centers, primitive_count);

        auto node_count = 2 * primitive_count - 1;
        auto nodes      = std::make_unique<typename Bvh::Node[]>(node_count);
        auto nodes_copy = std::make_unique<typename Bvh::Node[]>(node_count);

        auto neighbors   = std::make_unique<size_t[]>(node_count);
        auto next_index  = std::make_unique<size_t[]>(node_count);
        auto child_index = std::make_unique<size_t[]>(node_count);

        size_t begin = node_count - primitive_count, end = node_count;
        size_t unmerged_count = 0;
        size_t children_count = 0;
        size_t previous_end = end;

        #pragma omp parallel
        {
            // Create the leaves
            #pragma omp for
            for (size_t i = 0; i < primitive_count; ++i) {
                auto& node = nodes[begin + i];
                node.bounding_box_proxy()     = bboxes[primitive_indices[i]];
                node.is_leaf                  = true;
                node.primitive_count          = 1;
                node.first_child_or_primitive = i;
            }

            while (end - begin > 1) {
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
                        auto distance = nodes[i].bounding_box_proxy().to_bounding_box().extend(nodes[j].bounding_box_proxy()).half_area();
                        if (distance < best_distance) {
                            best_distance = distance;
                            best_neighbor = j;
                        }
                    }
                    neighbors[i] = best_neighbor;
                }

                // Mark nodes that are the closest as merged, but keep the one with lowest index act as the parent
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
                size_t next_begin = children_begin - unmerged_count;

                #pragma omp for nowait
                for (size_t i = begin; i < end; ++i) {
                    auto j = neighbors[i];
                    if (neighbors[j] == i) {
                        if (i < j) {
                            auto& merged_node = nodes_copy[next_begin + next_index[i]];
                            auto first_child = children_begin + child_index[i];
                            merged_node.bounding_box_proxy() = nodes[j].bounding_box_proxy().to_bounding_box().extend(nodes[i].bounding_box_proxy());
                            merged_node.is_leaf = false;
                            merged_node.first_child_or_primitive = first_child;
                            nodes_copy[first_child + 0] = nodes[i];
                            nodes_copy[first_child + 1] = nodes[j];
                        }
                    } else {
                        nodes_copy[next_begin + next_index[i]] = nodes[i];
                    }
                }

                #pragma omp for
                for (size_t i = end; i < previous_end; ++i)
                    nodes_copy[i] = nodes[i];

                #pragma omp single
                {
                    std::swap(nodes_copy, nodes);
                    previous_end = end;
                    begin        = next_begin;
                    end          = children_begin;
                }
            }
        }

        std::swap(bvh.nodes, nodes);
        std::swap(bvh.primitive_indices, primitive_indices);
        bvh.node_count = node_count;
    }
};

} // namespace bvh

#endif
