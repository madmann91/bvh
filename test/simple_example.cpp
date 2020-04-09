#include <vector>
#include <iostream>

#include <bvh/bvh.hpp>
#include <bvh/vector.hpp>
#include <bvh/triangle.hpp>
#include <bvh/ray.hpp>
#include <bvh/binned_sah_builder.hpp>
#include <bvh/single_ray_traversal.hpp>
#include <bvh/intersectors.hpp>

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

    static constexpr bool pre_shuffle = true;
    static constexpr size_t bin_count = 32;

    // Create an acceleration data structure on those triangles
    Bvh bvh;
    bvh::BinnedSahBuilder<Bvh, bin_count> builder(&bvh);
    auto [bboxes, centers] = bvh::compute_bounding_boxes_and_centers(triangles.data(), triangles.size());
    builder.build(bboxes.get(), centers.get(), triangles.size());
    if (pre_shuffle)
        bvh::shuffle_primitives(triangles.data(), bvh.primitive_indices.get(), triangles.size());

    // Intersect a ray with the data structure
    Ray ray(
        Vector3(0.0, 0.0, 0.0), // origin
        Vector3(0.0, 0.0, 1.0), // direction
        0.0,                    // minimum distance
        100.0                   // maximum distance
    );
    bvh::ClosestIntersector<pre_shuffle, Bvh, Triangle> intersector(&bvh, triangles.data());
    bvh::SingleRayTraversal<Bvh> traversal(&bvh);

    auto hit = traversal.intersect(ray, intersector);
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
