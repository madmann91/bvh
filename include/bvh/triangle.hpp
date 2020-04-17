#ifndef BVH_TRIANGLE_HPP
#define BVH_TRIANGLE_HPP

#include <optional>
#include <cassert>

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

    std::pair<Vector3<Scalar>, Vector3<Scalar>> edge(size_t i) const {
        assert(i < 3);
        Vector3<Scalar> p[] = { p0, p1(), p2() };
        return std::make_pair(p[i], p[(i + 1) % 3]);
    }

    Scalar area() const {
        return length(n) * Scalar(0.5);
    }

    std::pair<BoundingBox<Scalar>, BoundingBox<Scalar>> split(size_t axis, Scalar position) const {
        Vector3<Scalar> p[] = { p0, p1(), p2() };
        auto left  = BoundingBox<Scalar>::empty();
        auto right = BoundingBox<Scalar>::empty();
        auto split_edge = [=] (const Vector3<Scalar>& a, const Vector3<Scalar>& b) {
            auto t = (position - a[axis]) / (b[axis] - a[axis]);
            return a + t * (b - a);
        };
        auto q0 = p[0][axis] <= position;
        auto q1 = p[1][axis] <= position;
        auto q2 = p[2][axis] <= position;
        if (q0) left.extend(p[0]);
        else    right.extend(p[0]);
        if (q1) left.extend(p[1]);
        else    right.extend(p[1]);
        if (q2) left.extend(p[2]);
        else    right.extend(p[2]);
        if (q0 ^ q1) {
            auto m = split_edge(p[0], p[1]);
            left.extend(m);
            right.extend(m);
        }
        if (q1 ^ q2) {
            auto m = split_edge(p[1], p[2]);
            left.extend(m);
            right.extend(m);
        }
        if (q2 ^ q0) {
            auto m = split_edge(p[2], p[0]);
            left.extend(m);
            right.extend(m);
        }
        return std::make_pair(left, right);
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
