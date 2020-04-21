#ifndef BVH_SINGLE_RAY_TRAVERSAL_HPP
#define BVH_SINGLE_RAY_TRAVERSAL_HPP

#include <cassert>

#include "bvh/bvh.hpp"
#include "bvh/ray.hpp"
#include "bvh/utilities.hpp"

namespace bvh {

template <typename Bvh, size_t StackSize = 64, bool Robust = false>
class SingleRayTraversal {
public:
    static constexpr size_t stack_size = StackSize;

private:
    using Scalar = typename Bvh::ScalarType;

    struct Stack {
        using Element = const typename Bvh::Node*;

        Element elements[stack_size];
        size_t size = 0;

        void push(const Element& t) {
            assert(size < stack_size);
            elements[size++] = t;
        }

        Element pop() {
            assert(!empty());
            return elements[--size];
        }

        bool empty() const { return size == 0; }
    };

    struct Octant {
        Octant(int x, int y, int z)
            : x(x), y(y), z(z)
        {}

        Octant(const Ray<Scalar>& ray)
            : x(ray.direction[0] < Scalar(0))
            , y(ray.direction[1] < Scalar(0))
            , z(ray.direction[2] < Scalar(0))
        {}

        int x, y, z;
    };

    std::pair<Scalar, Scalar> intersect_node(
        const typename Bvh::Node& node,
        const Ray<Scalar>& ray,
        const Vector3<Scalar>& inverse_origin,
        const Vector3<Scalar>& inverse_direction,
        const Vector3<Scalar>& padded_inverse_direction,
        const Octant& octant) const
    {
        Vector3<Scalar> entry, exit;
        if (Robust) {
            entry[0] = (node.bounds[0 +     octant.x] - ray.origin[0]) * inverse_direction[0];
            entry[1] = (node.bounds[2 +     octant.y] - ray.origin[1]) * inverse_direction[1];
            entry[2] = (node.bounds[4 +     octant.z] - ray.origin[2]) * inverse_direction[2];
            exit[0]  = (node.bounds[0 + 1 - octant.x] - ray.origin[0]) * padded_inverse_direction[0];
            exit[1]  = (node.bounds[2 + 1 - octant.y] - ray.origin[1]) * padded_inverse_direction[1];
            exit[2]  = (node.bounds[4 + 1 - octant.z] - ray.origin[2]) * padded_inverse_direction[2];
        } else {
            entry[0] = multiply_add(node.bounds[0 +     octant.x], inverse_direction[0], inverse_origin[0]);
            entry[1] = multiply_add(node.bounds[2 +     octant.y], inverse_direction[1], inverse_origin[1]);
            entry[2] = multiply_add(node.bounds[4 +     octant.z], inverse_direction[2], inverse_origin[2]);
            exit[0]  = multiply_add(node.bounds[0 + 1 - octant.x], inverse_direction[0], inverse_origin[0]);
            exit[1]  = multiply_add(node.bounds[2 + 1 - octant.y], inverse_direction[1], inverse_origin[1]);
            exit[2]  = multiply_add(node.bounds[4 + 1 - octant.z], inverse_direction[2], inverse_origin[2]);
        }
        // Note: This order for the min/max operations is guaranteed not to produce NaNs
        return std::make_pair(
            robust_max(entry[0], robust_max(entry[1], robust_max(entry[2], ray.tmin))),
            robust_min(exit [0], robust_min(exit [1], robust_min(exit [2], ray.tmax)))
        );
    }

    template <typename Intersector, typename Statistics>
    std::optional<typename Intersector::Result>& intersect_leaf(
        const typename Bvh::Node& node,
        Ray<Scalar>& ray,
        std::optional<typename Intersector::Result>& best_hit,
        Intersector& intersector,
        Statistics& statistics) const
    {
        assert(node.is_leaf);
        size_t begin = node.first_child_or_primitive;
        size_t end   = begin + node.primitive_count;
        statistics.intersections += end - begin;
        for (size_t i = begin; i < end; ++i) {
            if (auto hit = intersector(i, ray)) {
                best_hit = hit;
                if (intersector.any_hit)
                    return best_hit;
                ray.tmax = hit->distance();
            }
        }
        return best_hit;
    }

    template <typename Intersector, typename Statistics>
    std::optional<typename Intersector::Result> traverse(Ray<Scalar> ray, Intersector& intersector, Statistics& statistics) const {
        auto best_hit = std::optional<typename Intersector::Result>(std::nullopt);

        // If the root is a leaf, intersect it and return
        if (bvh__unlikely(bvh.nodes[0].is_leaf))
            return intersect_leaf(bvh.nodes[0], ray, best_hit, intersector, statistics);

        // Precompute the inverse direction to avoid divisions and refactor
        // the computation to allow the use of FMA instructions (when available).
        auto inverse_direction = ray.direction.inverse();
        auto inverse_origin    = -ray.origin * inverse_direction;

        // Padded inverse direction to avoid false-negatives in the ray-node test.
        // Only used when the robust ray-node intersection code is enabled.
        auto padded_inverse_direction = Vector3<Scalar>(
            add_ulp_magnitude(inverse_direction[0], 2),
            add_ulp_magnitude(inverse_direction[1], 2),
            add_ulp_magnitude(inverse_direction[2], 2));

        // Precompute the octant of the ray to speed up the ray-node test
        Octant octant(ray);

        // This traversal loop is eager, because it immediately processes leaves instead of pushing them on the stack.
        // This is generally beneficial for performance because intersections will likely be found which will
        // allow to cull more subtrees with the ray-box test of the traversal loop.
        Stack stack;
        const auto* node = bvh.nodes.get();
        while (true) {
            statistics.traversal_steps++;

            auto first_child = node->first_child_or_primitive;
            const auto* left_child  = &bvh.nodes[first_child + 0];
            const auto* right_child = &bvh.nodes[first_child + 1];
            auto distance_left  = intersect_node(*left_child,  ray, inverse_origin, inverse_direction, padded_inverse_direction, octant);
            auto distance_right = intersect_node(*right_child, ray, inverse_origin, inverse_direction, padded_inverse_direction, octant);

            if (bvh__unlikely(distance_left.first <= distance_left.second)) {
                if (left_child->is_leaf) {
                    if (intersect_leaf(*left_child, ray, best_hit, intersector, statistics) && intersector.any_hit)
                        break;
                    left_child = nullptr;
                }
            } else
                left_child = nullptr;

            if (bvh__unlikely(distance_right.first <= distance_right.second)) {
                if (right_child->is_leaf) {
                    if (intersect_leaf(*right_child, ray, best_hit, intersector, statistics) && intersector.any_hit)
                        break;
                    right_child = nullptr;
                }
            } else
                right_child = nullptr;

            if (bvh__likely((left_child != NULL) ^ (right_child != NULL))) {
                node = left_child != NULL ? left_child : right_child;
            } else if (bvh__unlikely((left_child != NULL) & (right_child != NULL))) {
                if (distance_left.first > distance_right.first)
                    std::swap(left_child, right_child);
                stack.push(right_child);
                node = left_child;
            } else {
                if (stack.empty())
                    break;
                node = stack.pop();
            }
        }

        return best_hit;
    }

    const Bvh& bvh;

public:
    /// Statistics collected during traversal.
    struct Statistics {
        size_t traversal_steps = 0;
        size_t intersections   = 0;
    };

    SingleRayTraversal(const Bvh& bvh)
        : bvh(bvh)
    {}

    /// Intersects the BVH with the given ray and intersector.
    template <typename Intersector>
    bvh__always_inline__
    std::optional<typename Intersector::Result> intersect(const Ray<Scalar>& ray, Intersector& intersector) const {
        struct {
            struct Empty {
                Empty& operator ++ (int)    { return *this; }
                Empty& operator ++ ()       { return *this; }
                Empty& operator += (size_t) { return *this; }
            } traversal_steps, intersections;
        } statistics;
        return traverse(ray, intersector, statistics);
    }

    /// Intersects the BVH with the given ray and intersector.
    /// Record statistics on the number of traversal and intersection steps.
    template <typename Intersector>
    bvh__always_inline__
    std::optional<typename Intersector::Result> intersect(const Ray<Scalar>& ray, Intersector& intersector, Statistics& statistics) const {
        return traverse(ray, intersector, statistics);
    }
};

} // namespace bvh

#endif
