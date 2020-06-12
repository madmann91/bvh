#include <vector>
#include <iostream>

#include <bvh/bvh.hpp>
#include <bvh/vector.hpp>
#include <bvh/triangle.hpp>
#include <bvh/ray.hpp>
#include <bvh/sweep_sah_builder.hpp>
#include <bvh/single_ray_traverser.hpp>
#include <bvh/primitive_intersectors.hpp>

using Scalar   = float;
using Vector3  = bvh::Vector3<Scalar>;
using Triangle = bvh::Triangle<Scalar>;
using Ray      = bvh::Ray<Scalar>;
using Bvh      = bvh::Bvh<Scalar>;

int main() {
    // Create an array of triangles
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

    Bvh bvh;

    // Create an acceleration data structure on those triangles
    bvh::SweepSahBuilder<Bvh> builder(bvh);
    auto [bboxes, centers] = bvh::compute_bounding_boxes_and_centers(triangles.data(), triangles.size());
    auto global_bbox = bvh::compute_bounding_boxes_union(bboxes.get(), triangles.size());
    builder.build(global_bbox, bboxes.get(), centers.get(), triangles.size());

    // Intersect a ray with the data structure
    Ray ray(
        Vector3(0.0, 0.0, 0.0), // origin
        Vector3(0.0, 0.0, 1.0), // direction
        0.0,                    // minimum distance
        100.0                   // maximum distance
    );
    bvh::ClosestPrimitiveIntersector<Bvh, Triangle> primitive_intersector(bvh, triangles.data());
    bvh::SingleRayTraverser<Bvh> traverser(bvh);

    auto hit = traverser.traverse(ray, primitive_intersector);
    if (hit) {
        auto triangle_index = hit->primitive_index;
        auto intersection = hit->intersection;
        std::cout << "Hit triangle " << triangle_index << "\n"
                  << "distance: "    << intersection.t << "\n"
                  << "u: "           << intersection.u << "\n"
                  << "v: "           << intersection.v << std::endl;
        return 0;
    }
    return 1;
}
