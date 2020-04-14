#ifndef BVH_EDGE_VOLUME_HEURISTIC_HPP
#define BVH_EDGE_VOLUME_HEURISTIC_HPP

#include <algorithm>
#include <optional>
#include <stack>

#include "bvh/bvh.hpp"
#include "bvh/bounding_box.hpp"

namespace bvh {

/// Edge-volume heuristic triangle splitting. This pre-pass splits very large and
/// thin triangles that are difficult to handle with BVHs.
/// See "The Edge Volume Heuristic - Robust Triangle Subdivision for Improved BVH Performance",
/// by H. Dammertz and A. Keller.
template <typename Triangle>
class EdgeVolumeHeuristic {
    using Scalar = typename Triangle::ScalarType;

    static std::pair<Scalar, size_t> find_largest_edge(const Triangle& triangle) {
        Scalar max_volume = 0;
        size_t edge_index = 0;
        for (size_t i = 0; i < 3; ++i) {
            auto [p0, p1] = triangle.edge(i);
            auto volume = BoundingBox<Scalar>(p0).extend(p1).volume();
            if (volume >= max_volume) {
                max_volume = volume;
                edge_index = i;
            }
        }
        return std::make_pair(max_volume, edge_index);
    }

    struct Split {
        std::pair<Triangle, size_t> left, right;
    };

    static std::optional<Split> try_split(
        const Triangle& triangle,
        size_t index,
        size_t& reference_count,
        size_t max_reference_count,
        Scalar threshold)
    {
        auto [volume, edge_index] = find_largest_edge(triangle);
        if (volume > threshold) {
            size_t other_index;
            #pragma omp atomic capture
            { other_index = reference_count; reference_count++; }

            if (other_index < max_reference_count) {
                auto [p0, p1] = triangle.edge(edge_index);
                auto p2 = triangle.edge((edge_index + 1) % 3).second;
                auto q = (p0 + p1) * Scalar(0.5);
                return std::make_optional(Split {
                    std::make_pair(Triangle(p0, q, p2), index),
                    std::make_pair(Triangle(p2, q, p1), other_index)
                });
            }
        }
        return std::nullopt;
    }

public:
    /// Derives a splitting threshold based on the scene bounding box.
    static Scalar threshold(const BoundingBox<Scalar>& global_bbox, size_t exponent = 14) {
        return global_bbox.volume() / Scalar(size_t(1) << exponent);
    }

    /// Performs triangle splitting on the given array of triangles.
    /// This function takes an array of bounding box and centers, and an
    /// array of triangle indices. It fills the triangle indices with the
    /// original triangle indices before splitting, and returns the number
    /// of triangles after splitting.
    static size_t pre_split(
        const Triangle* triangles,
        BoundingBox<Scalar>* bounding_boxes,
        Vector3<Scalar>* centers,
        size_t* triangle_indices,
        size_t triangle_count,
        size_t max_reference_count,
        Scalar threshold)
    {
        std::stack<std::pair<Triangle, size_t>> stack;
        size_t reference_count = triangle_count;

        #pragma omp parallel for private(stack)
        for (size_t i = 0; i < triangle_count; ++i) {
            // Avoid recomputing the bounding boxes/centers of every triangle if possible.
            auto split = try_split(triangles[i], i, reference_count, max_reference_count, threshold);
            if (split) {
                stack.push(split->left);
                stack.push(split->right);
                while (!stack.empty()) {
                    auto [triangle, j] = stack.top();
                    stack.pop();

                    auto split = try_split(triangle, j, reference_count, max_reference_count, threshold);
                    if (split) {
                        stack.push(split->left);
                        stack.push(split->right);
                    } else {
                        triangle_indices[j] = i;
                        bounding_boxes[j]   = triangle.bounding_box();
                        centers[j]          = triangle.center();
                    }
                }
            } else
                triangle_indices[i] = i;
        }

        return std::min(max_reference_count, reference_count);
    }

    /// Remaps BVH primitive indices and removes duplicate triangle references in the BVH leaves.
    static void repair_bvh_leaves(Bvh<Scalar>& bvh, const size_t* triangle_indices) {
        #pragma omp parallel for
        for (size_t i = 0; i < bvh.node_count; ++i) {
            auto& node = bvh.nodes[i];
            if (node.is_leaf) {
                auto begin = bvh.primitive_indices.get() + node.first_child_or_primitive;
                auto end   = begin + node.primitive_count;
                std::transform(begin, end, begin, [&] (size_t i) { return triangle_indices[i]; });
                std::sort(begin, end);
                node.primitive_count = std::unique(begin, end) - begin;
            }
        }
    }
};

} // namespace bvh

#endif
