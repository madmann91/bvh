#include <vector>
#include <iostream>
#include <type_traits>

#include <bvh/bvh.hpp>
#include <bvh/vector.hpp>
#include <bvh/triangle.hpp>
#include <bvh/sphere.hpp>
#include <bvh/ray.hpp>
#include <bvh/sweep_sah_builder.hpp>
#include <bvh/single_ray_traverser.hpp>
#include <bvh/primitive_intersectors.hpp>

using Scalar   = float;
using Vector3  = bvh::Vector3<Scalar>;
using Triangle = bvh::Triangle<Scalar>;
using Sphere   = bvh::Sphere<Scalar>;
using Ray      = bvh::Ray<Scalar>;
using Bvh      = bvh::Bvh<Scalar>;

// This function builds a BVH from a vector of primitives, and intersects it with a ray.
template <typename Primitive>
static bool build_and_intersect_bvh(const std::vector<Primitive>& primitives) {
    Bvh bvh;

    // Compute the global bounding box and the centers of the primitives.
    // This is the input of the BVH construction algorithm.
    // Note: Using the bounding box centers instead of the primitive centers is possible,
    // but usually leads to lower-quality BVHs.
    auto [bboxes, centers] = bvh::compute_bounding_boxes_and_centers(primitives.data(), primitives.size());
    auto global_bbox = bvh::compute_bounding_boxes_union(bboxes.get(), primitives.size());

    // Create an acceleration data structure on the primitives
    bvh::SweepSahBuilder<Bvh> builder(bvh);
    builder.build(global_bbox, bboxes.get(), centers.get(), primitives.size());

    // Intersect a ray with the data structure
    Ray ray(
        Vector3(0.0, 0.0, 0.0), // origin
        Vector3(0.0, 0.0, 1.0), // direction
        0.0,                    // minimum distance
        100.0                   // maximum distance
    );
    bvh::ClosestPrimitiveIntersector<Bvh, Primitive> primitive_intersector(bvh, primitives.data());
    bvh::SingleRayTraverser<Bvh> traverser(bvh);

    if (auto hit = traverser.traverse(ray, primitive_intersector)) {
        auto triangle_index = hit->primitive_index;
        auto intersection = hit->intersection;
        std::cout
            << "Hit primitive " << triangle_index << "\n"
            << "distance: "     << intersection.t << "\n";
        if constexpr (std::is_same_v<Primitive, Triangle>) {
            std::cout
                << "u: " << intersection.u << "\n"
                << "v: " << intersection.v << "\n";
        }
        return true;
    }
    return false;
}

int main() {
    std::vector<Triangle> triangles;
    triangles.emplace_back(
        Vector3( 1.0, -1.0, 1.0),
        Vector3( 1.0,  1.0, 1.0),
        Vector3(-1.0,  1.0, 1.0)
    );
    triangles.emplace_back(
        Vector3( 1.0, -1.0, 1.0),
        Vector3(-1.0, -1.0, 1.0),
        Vector3(-1.0,  1.0, 1.0)
    );
    if (!build_and_intersect_bvh(triangles))
        return 1;

    std::vector<Sphere> spheres;
    spheres.emplace_back(Vector3( 1.0, -1.0, 1.0), Scalar(1.0));
    spheres.emplace_back(Vector3( 0.0,  0.0, 2.0), Scalar(0.5));
    if (!build_and_intersect_bvh(spheres))
        return 1;
    return 0;
}
