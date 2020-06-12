#include <vector>
#include <iostream>

#include <bvh/bvh.hpp>
#include <bvh/vector.hpp>
#include <bvh/ray.hpp>
#include <bvh/sweep_sah_builder.hpp>
#include <bvh/single_ray_traverser.hpp>
#include <bvh/primitive_intersectors.hpp>

using Scalar      = float;
using Vector3     = bvh::Vector3<Scalar>;
using BoundingBox = bvh::BoundingBox<Scalar>;
using Ray         = bvh::Ray<Scalar>;
using Bvh         = bvh::Bvh<Scalar>;

struct CustomPrimitive  {
    struct Intersection {
        Scalar t;

        // Required member: returns the distance along the ray
        Scalar distance() const { return t; }
    };

    // Required type: the floating-point type used
    using ScalarType = Scalar;
    // Required type: the intersection data type returned by the intersect() method
    using IntersectionType = Intersection;

    CustomPrimitive() = default;

    // Required member: returns the center of the primitive
    Vector3 center() const {
        return Vector3(0, 0, 0);
    }

    // Required member: returns a bounding box for the primitive (tighter is better)
    BoundingBox bounding_box() const {
        return BoundingBox(Vector3(-1, -1, -1), Vector3(1, 1, 1));
    }

    // Required member: computes the intersection between a ray and the primitive
    std::optional<Intersection> intersect(const Ray& ray) const {
        return std::make_optional<Intersection>(Intersection { (ray.tmin + ray.tmax) * Scalar(0.5) });
    }
};

int main() {
    // Create an array of spheres 
    std::vector<CustomPrimitive> primitives;
    primitives.emplace_back();
    primitives.emplace_back();

    Bvh bvh;

    // Create an acceleration data structure on those triangles
    bvh::SweepSahBuilder<Bvh> builder(bvh);
    auto [bboxes, centers] = bvh::compute_bounding_boxes_and_centers(primitives.data(), primitives.size());
    auto global_bbox = bvh::compute_bounding_boxes_union(bboxes.get(), primitives.size());
    builder.build(global_bbox, bboxes.get(), centers.get(), primitives.size());

    // Intersect a ray with the data structure
    Ray ray(
        Vector3(0.0, 0.0, 0.0), // origin
        Vector3(0.0, 0.0, 1.0), // direction
        0.0,                    // minimum distance
        100.0                   // maximum distance
    );
    bvh::ClosestPrimitiveIntersector<Bvh, CustomPrimitive> intersector(bvh, primitives.data());
    bvh::SingleRayTraverser<Bvh> traverser(bvh);

    auto hit = traverser.traverse(ray, intersector);
    if (hit) {
        auto primitive_index = hit->primitive_index;
        auto intersection = hit->intersection;
        std::cout << "Hit primitive " << primitive_index         << "\n"
                  << "distance: "     << intersection.distance() << std::endl;
        return 0;
    }
    return 1;
}
