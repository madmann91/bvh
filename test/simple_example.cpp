#include <vector>
#include <iostream>

#include <bvh/bvh.h>
#include <bvh/proto_imports.h>
#include <bvh/builders/binned_sah_builder.h>
#include <bvh/builders/seq_top_down_scheduler.h>
#include <bvh/traversers/single_ray_traverser.h>

using Scalar   = float;
using Triangle = bvh::Triangle<Scalar>;
using Ray      = bvh::Ray<Scalar>;
using Vec3     = bvh::Vec3<Scalar>;
using BBox     = bvh::BBox<Scalar>;
using Bvh      = bvh::Bvh<Scalar>;

int main() {
    // Create an array of triangles
    std::vector<Triangle> triangles;
    triangles.emplace_back(
        Vec3( 1.0, -1.0, 1.0),
        Vec3( 1.0,  1.0, 1.0),
        Vec3(-1.0,  1.0, 1.0)
    );
    triangles.emplace_back(
        Vec3( 1.0, -1.0, 1.0),
        Vec3(-1.0, -1.0, 1.0),
        Vec3(-1.0,  1.0, 1.0)
    );

    std::vector<BBox> bboxes;
    std::vector<Vec3> centers;
    auto global_bbox = BBox::empty();
    for (auto& triangle : triangles) {
        bboxes.push_back(triangle.bbox());
        centers.push_back(triangle.center());
        global_bbox.extend(bboxes.back());
    }

    using Builder = bvh::BinnedSahBuilder<Bvh>;
    bvh::SeqTopDownScheduler<Builder> scheduler;
    auto bvh = Builder::build(scheduler, global_bbox, bboxes.data(), centers.data(), triangles.size());

    // Intersect a ray with the data structure
    Ray ray(
        Vec3(0.0, 0.0, 0.0), // origin
        Vec3(0.0, 0.0, 1.0), // direction
        0.0,                 // minimum distance
        100.0                // maximum distance
    );

    struct Hit {
        std::pair<Scalar, Scalar> uv;
        size_t prim_index;
    };

    auto hit = bvh::SingleRayTraverser<Bvh>::traverse<false>(ray, bvh, [&] (Ray& ray, const Bvh::Node& leaf) {
        std::optional<Hit> hit;
        for (size_t i = 0; i < leaf.prim_count; ++i) {
            size_t prim_index = bvh.prim_indices[leaf.first_index + i];
            if (auto uv = triangles[prim_index].intersect(ray))
                hit = Hit { *uv, prim_index }; 
        }
        return hit;
    });

    if (hit) {
        std::cout
            << "Hit triangle " << hit->prim_index << "\n"
            << "distance: "    << ray.tmax        << "\n"
            << "u: "           << hit->uv.first   << "\n"
            << "v: "           << hit->uv.second  << std::endl;
        return 0;
    }
    return 1;
}
