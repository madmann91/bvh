#ifndef BVH_NODE_INTERSECTORS_HPP
#define BVH_NODE_INTERSECTORS_HPP

#include "bvh/vector.hpp"
#include "bvh/ray.hpp"
#include "bvh/platform.hpp"
#include "bvh/utilities.hpp"

namespace bvh {

/// Base class for ray-node intersection algorithms. Does ray octant classification.
template <typename Bvh, typename Derived>
struct NodeIntersector {
    using Scalar = typename Bvh::ScalarType;

    std::array<int, 3> octant;

    NodeIntersector(const Ray<Scalar>& ray)
        : octant {
            ray.direction[0] < Scalar(0),
            ray.direction[1] < Scalar(0),
            ray.direction[2] < Scalar(0)
        }
    {}

    bvh__always_inline__
    Scalar intersect_axis(int axis, const Vector3<Scalar>& p, const Ray<Scalar>& ray) const {
        return static_cast<const Derived*>(this)->intersect_axis(axis, p, ray);
    }

    bvh__always_inline__
    std::pair<Scalar, Scalar> intersect(const typename Bvh::Node& node, const Ray<Scalar>& ray) const {
        Vector3<Scalar> entry, exit;
        entry[0] = intersect_axis(0, node.bounds[0 * 2 +     octant[0]], ray);
        entry[1] = intersect_axis(1, node.bounds[1 * 2 +     octant[1]], ray);
        entry[2] = intersect_axis(2, node.bounds[2 * 2 +     octant[2]], ray);
        exit [0] = intersect_axis(0, node.bounds[0 * 2 + 1 - octant[0]], ray);
        exit [1] = intersect_axis(1, node.bounds[1 * 2 + 1 - octant[1]], ray);
        exit [2] = intersect_axis(2, node.bounds[2 * 2 + 1 - octant[2]], ray);
        // Note: This order for the min/max operations is guaranteed not to produce NaNs
        return std::make_pair(
            robust_max(entry[0], robust_max(entry[1], robust_max(entry[2], ray.tmin))),
            robust_min(exit [0], robust_min(exit [1], robust_min(exit [2], ray.tmax))));
    }

protected:
    ~NodeIntersector() {}
};

/// Fully robust ray-node intersection algorithm (see "Robust BVH Ray Traversal", by T. Ize).
template <typename Bvh>
struct RobustNodeIntersector : public NodeIntersector<Bvh, RobustNodeIntersector<Bvh>> {
    using Scalar = typename Bvh::ScalarType;

    // Padded inverse direction to avoid false-negatives in the ray-node test.
    Vector3<Scalar> padded_inverse_direction;

    RobustNodeIntersector(const Ray<Scalar>& ray)
        : NodeIntersector<Bvh, RobustNodeIntersector<Bvh>>(ray)
    {
        auto inverse_direction = ray.direction.inverse();

        padded_inverse_direction = Vector3<Scalar>(
            add_ulp_magnitude(inverse_direction[0], 2),
            add_ulp_magnitude(inverse_direction[1], 2),
            add_ulp_magnitude(inverse_direction[2], 2));
    }

    bvh__always_inline__
    Scalar intersect_axis(int axis, const Vector3<Scalar>& p, const Ray<Scalar>& ray) const {
        return (p[axis] - ray.origin[axis]) * padded_inverse_direction[axis];
    }

    using NodeIntersector<Bvh, RobustNodeIntersector<Bvh>>::intersect;
};

/// Semi-robust, fast ray-node intersection algorithm.
template <typename Bvh>
struct FastNodeIntersector : public NodeIntersector<Bvh, FastNodeIntersector<Bvh>> {
    using Scalar = typename Bvh::ScalarType;

    Vector3<Scalar> scaled_origin;
    Vector3<Scalar> inverse_direction;

    FastNodeIntersector(const Ray<Scalar>& ray)
        : NodeIntersector<Bvh, FastNodeIntersector<Bvh>>(ray) 
    {
        inverse_direction = ray.direction.inverse();
        scaled_origin     = -ray.origin * inverse_direction;
    }

    bvh__always_inline__
    Scalar intersect_axis(int axis, const Vector3<Scalar>& p, const Ray<Scalar>&) const {
        return fast_multiply_add(p[axis], inverse_direction[axis], scaled_origin[axis]);
    }

    using NodeIntersector<Bvh, FastNodeIntersector<Bvh>>::intersect;
};

} // namespace bvh

#endif
