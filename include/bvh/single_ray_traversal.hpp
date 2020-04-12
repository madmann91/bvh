#ifndef BVH_SINGLE_RAY_TRAVERSAL_HPP
#define BVH_SINGLE_RAY_TRAVERSAL_HPP

#include <cassert>

#include "bvh.hpp"
#include "ray.hpp"

namespace bvh {

template <typename Bvh, size_t StackSize = 64>
class SingleRayTraversal {
public:
    static constexpr size_t stack_size = StackSize;

private:
    using Scalar = typename Bvh::ScalarType;

    struct Stack {
        using Element = typename Bvh::IndexType;

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
            : x(ray.direction[0] > Scalar(0) ? 0 : 3)
            , y(ray.direction[1] > Scalar(0) ? 1 : 4)
            , z(ray.direction[2] > Scalar(0) ? 2 : 5)
        {}

        Octant inverse() const { return Octant(3 - x, 5 - y, 7 - z); }

        int x, y, z;
    };

    std::pair<Scalar, Scalar> intersect_node(
        const typename Bvh::Node& node,
        const Vector3<Scalar>& inverse_origin,
        const Vector3<Scalar>& inverse_direction,
        Scalar tmin, Scalar tmax,
        const Octant& octant) const
    {
        auto inverse_octant = octant.inverse();
        Scalar entry_x = multiply_add(node.bounds[        octant.x], inverse_direction[0], inverse_origin[0]);
        Scalar entry_y = multiply_add(node.bounds[        octant.y], inverse_direction[1], inverse_origin[1]);
        Scalar entry_z = multiply_add(node.bounds[        octant.z], inverse_direction[2], inverse_origin[2]);
        Scalar exit_x  = multiply_add(node.bounds[inverse_octant.x], inverse_direction[0], inverse_origin[0]);
        Scalar exit_y  = multiply_add(node.bounds[inverse_octant.y], inverse_direction[1], inverse_origin[1]);
        Scalar exit_z  = multiply_add(node.bounds[inverse_octant.z], inverse_direction[2], inverse_origin[2]);
        return std::make_pair(
            std::max(std::max(entry_x, entry_y), std::max(entry_z, tmin)),
            std::min(std::min(exit_x,  exit_y),  std::min(exit_z,  tmax))
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
    std::optional<typename Intersector::Result> intersect_bvh(Ray<Scalar> ray, Intersector& intersector, Statistics& statistics) const {
        auto best_hit = std::optional<typename Intersector::Result>(std::nullopt);

        // If the root is a leaf, intersect it and return
        if (bvh.nodes[0].is_leaf)
            return intersect_leaf(bvh.nodes[0], ray, best_hit, intersector, statistics);

        // Precompute the inverse direction to avoid divisions and refactor
        // the computation to allow the use of FMA instructions (when available).
        auto inverse_direction = ray.direction.inverse();
        auto inverse_origin    = -ray.origin * inverse_direction;

        // Precompute the octant of the ray to speed up the ray-node test
        Octant octant(ray);

        // This traversal loop is eager, because it immediately processes leaves instead of pushing them on the stack.
        // This is generally beneficial for performance because intersections will likely be found which will
        // allow to cull more subtrees with the ray-box test of the traversal loop.
        Stack stack;
        auto node = bvh.nodes.get();
        while (true) {
            statistics.traversal_steps++;

            auto first_child = node->first_child_or_primitive;

            auto& left  = bvh.nodes[first_child + 0];
            auto& right = bvh.nodes[first_child + 1];
            auto distance_left  = intersect_node(left,  inverse_origin, inverse_direction, ray.tmin, ray.tmax, octant);
            auto distance_right = intersect_node(right, inverse_origin, inverse_direction, ray.tmin, ray.tmax, octant);
            bool hit_left  = distance_left.first  <= distance_left.second;
            bool hit_right = distance_right.first <= distance_right.second;

            if (hit_left && left.is_leaf) {
                if (intersect_leaf(left, ray, best_hit, intersector, statistics) && intersector.any_hit)
                    break;
                hit_left = false;
            }

            if (hit_right && right.is_leaf) {
                if (intersect_leaf(right, ray, best_hit, intersector, statistics) && intersector.any_hit)
                    break;
                hit_right = false;
            }

            if (hit_left && hit_right) {
                int order = distance_left.first < distance_right.first ? 0 : 1;
                stack.push(first_child + (1 - order));
                node = &bvh.nodes[first_child + order];
            } else if (hit_left ^ hit_right) {
                node = &bvh.nodes[first_child + (hit_left ? 0 : 1)];
            } else {
                if (stack.empty())
                    break;
                node = &bvh.nodes[stack.pop()];
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
    std::optional<typename Intersector::Result> intersect(const Ray<Scalar>& ray, Intersector& intersector) const {
        struct {
            struct Empty {
                Empty& operator ++ (int)    { return *this; }
                Empty& operator ++ ()       { return *this; }
                Empty& operator += (size_t) { return *this; }
            } traversal_steps, intersections;
        } statistics;
        return intersect_bvh(ray, intersector, statistics);
    }

    /// Intersects the BVH with the given ray and intersector.
    /// Record statistics on the number of traversal and intersection steps.
    template <typename Intersector>
    std::optional<typename Intersector::Result> intersect(const Ray<Scalar>& ray, Intersector& intersector, Statistics& statistics) const {
        return intersect_bvh(ray, intersector, statistics);
    }
};

} // namespace bvh

#endif
