#ifndef BVH_PRIMITIVES_INTERSECTORS_HPP
#define BVH_PRIMITIVES_INTERSECTORS_HPP

#include <optional>

#include "bvh/ray.hpp"

namespace bvh {

/// An intersector that looks for the closest intersection.
template <bool PreShuffled, typename Bvh, typename Primitive>
struct ClosestIntersector {
    using Scalar       = typename Primitive::ScalarType;
    using Intersection = typename Primitive::IntersectionType;

    struct Result {
        size_t       primitive_index;
        Intersection intersection;

        Scalar distance() const { return intersection.distance(); }
    };

    ClosestIntersector(const Bvh& bvh, const Primitive* primitives)
        : bvh(bvh), primitives(primitives)
    {}

    std::optional<Result> operator () (size_t index, const Ray<Scalar>& ray) const {
        index = PreShuffled ? index : bvh.primitive_indices[index];
        if (auto hit = primitives[index].intersect(ray))
            return std::make_optional(Result { index, *hit });
        return std::nullopt;
    }

    static constexpr bool any_hit = false;

    const Bvh& bvh;
    const Primitive* primitives = nullptr;
};

/// An intersector that only stores the distance to the primitive.
template <bool PreShuffled, typename Bvh, typename Primitive>
struct AnyIntersector {
    using Scalar       = typename Primitive::ScalarType;

    struct Result {
        float t;
        Scalar distance() const { return t; }
    };

    AnyIntersector(const Bvh& bvh, const Primitive* primitives)
        : bvh(bvh), primitives(primitives)
    {}

    std::optional<Result> operator () (size_t index, const Ray<Scalar>& ray) const {
        index = PreShuffled ? index : bvh.primitive_indices[index];
        if (auto hit = primitives[index].intersect(ray))
            return std::make_optional(Result { hit->t });
        return std::nullopt;
    }

    static constexpr bool any_hit = true;

    const Bvh& bvh;
    const Primitive* primitives = nullptr;
};

} // namespace bvh

#endif
