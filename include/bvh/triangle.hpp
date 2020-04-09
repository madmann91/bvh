#ifndef BVH_TRIANGLE_HPP
#define BVH_TRIANGLE_HPP

#include "bvh/vector.hpp"
#include "bvh/bounding_box.hpp"
#include "bvh/ray.hpp"

namespace bvh {

/// Triangle primitive, defined by three points, and using the Moeller-Trumbore test.
template <typename Scalar>
struct Triangle {
    struct Intersection {
        Scalar t, u, v;
        Scalar distance() const { return t; }
    };

    using ScalarType       = Scalar;
    using IntersectionType = Intersection;

    Vector3<Scalar> p0, e1, e2, n;

    Triangle() = default;
    Triangle(const Vector3<Scalar>& p0, const Vector3<Scalar>& p1, const Vector3<Scalar>& p2)
        : p0(p0), e1(p0 - p1), e2(p2 - p0)
    {
        n = cross(e1, e2);
    }

    Vector3<Scalar> p1() const { return p0 - e1; }
    Vector3<Scalar> p2() const { return p0 + e2; }

    BoundingBox<Scalar> bounding_box() const {
        BoundingBox<Scalar> bbox(p0);
        bbox.extend(p1());
        bbox.extend(p2());
        return bbox;
    }

    Vector3<Scalar> center() const {
        return (p0 + p1() + p2()) * (Scalar(1.0) / Scalar(3.0));
    }

    std::optional<Intersection> intersect(const Ray<Scalar>& ray) const {
        static constexpr Scalar tolerance = Scalar(1e-9);

        auto c = p0 - ray.origin;
        auto r = cross(ray.direction, c);
        auto det = dot(n, ray.direction);
        auto abs_det = std::fabs(det);

        auto u = product_sign(dot(r, e2), det);
        auto v = product_sign(dot(r, e1), det);
        auto w = abs_det - u - v;

        if (u >= -tolerance && v >= -tolerance && w >= -tolerance) {
            auto t = product_sign(dot(n, c), det);
            if (t >= abs_det * ray.tmin && abs_det * ray.tmax > t) {
                auto inv_det = Scalar(1.0) / abs_det;
                return std::make_optional(Intersection{ t * inv_det, u * inv_det, v * inv_det });
            }
        }

        return std::nullopt;
    }
};

} // namespace bvh

#endif
