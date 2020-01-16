#include <vector>
#include <iostream>
#include "bvh.hpp"

struct Sphere {
    bvh::Vec3 origin;
    bvh::Scalar radius;

    Sphere() = default;
    Sphere(const bvh::Vec3& origin, bvh::Scalar radius)
        : origin(origin), radius(radius)
    {}

    // Required member: returns the center of the primitive
    bvh::Vec3 center() const {
        return origin;
    }

    // Required member: returns a bounding box for the primitive (tighter is better)
    bvh::BBox bounding_box() const {
        return bvh::BBox(origin - bvh::Vec3(radius), origin + bvh::Vec3(radius));
    }

    // Required type: contains the result of intersection with a ray
    struct Intersection {
        // Required member: contains the distance along the ray
        bvh::Scalar distance;

        Intersection() = default;
        Intersection(bvh::Scalar distance)
            : distance(distance)
        {}
    };

    std::optional<Intersection> intersect(const bvh::Ray& ray) const {
        bvh::Vec3 oc = ray.origin - origin;
        auto a = dot(ray.direction, ray.direction);
        auto b = 2 * dot(ray.direction, oc);
        auto c = dot(oc, oc) - radius * radius;

        auto delta = b * b - 4 * a * c;
        if (delta >= 0) {
            auto inv = bvh::Scalar(0.5) / a;
            auto t0 = -(b + std::sqrt(delta)) * inv;
            auto t1 = -(b - std::sqrt(delta)) * inv;
            auto t = std::fmin(t0 > ray.tmin ? t0 : t1, t1 > ray.tmin ? t1 : t0);
            if (t > ray.tmin && t < ray.tmax) {
                return std::make_optional(Intersection(t));
            }
        }

        return std::nullopt;
    }
};

int main() {
    // Create an array of spheres 
    std::vector<Sphere> spheres;
    spheres.emplace_back(bvh::Vec3(0.0f, 0.0f, 0.0f), 1.0f);
    spheres.emplace_back(bvh::Vec3(0.0f, 0.0f, 1.0f), 1.0f);

    // Create an acceleration data structure on those triangles
    bvh::Accel<Sphere> accel(spheres.data(), spheres.size());
    accel.build();

    // Intersect a ray with the data structure
    bvh::Ray ray(
        bvh::Vec3(0.0f, 0.0f, 0.0f), // origin
        bvh::Vec3(0.0f, 0.0f, 1.0f), // direction
        0.0f,                        // minimum distance
        100.0f                       // maximum distance
    );
    auto hit = accel.intersect_closest(ray);
    if (hit) {
        auto sphere_index = hit->first;
        auto intersection = hit->second;
        std::cout << "Hit sphere " << sphere_index          << "\n"
                  << "distance: "  << intersection.distance << std::endl;
        return 0;
    }
    return 1;
}
