#include <vector>
#include <iostream>
#include <ranges>
#include <numeric>

#include <proto/vec.h>
#include <proto/bbox.h>
#include <proto/ray.h>
#include <proto/triangle.h>
#include <par/sequential_executor.h>

#include <bvh/bvh.h>
#include <bvh/sweep_sah_builder.h>
#include <bvh/single_ray_traverser.h>
#include <bvh/sequential_top_down_scheduler.h>

using Scalar   = float;
using Triangle = proto::Triangle<Scalar>;
using Ray      = proto::Ray<Scalar>;
using Vec3     = proto::Vec3<Scalar>;
using BBox     = proto::BBox<Scalar>;
using Bvh      = bvh::Bvh<Scalar>;

int main() {
    par::SequentialExecutor executor;

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

    // Compute bounding boxes and centers for every triangle
    auto bboxes  = std::make_unique<BBox[]>(triangles.size());
    auto centers = std::make_unique<Vec3[]>(triangles.size());
    auto global_bbox = par::transform_reduce(
        executor, par::range_1d(size_t{0}, triangles.size()), BBox::empty(),
        [] (BBox left, const BBox& right) { return left.extend(right); },
        [&] (size_t i) -> BBox {
            auto bbox  = triangles[i].bbox();
            centers[i] = triangles[i].center();
            return bboxes[i] = bbox;
        });

    using Builder = bvh::SweepSahBuilder<Bvh>;
    bvh::SequentialTopDownScheduler<Builder> scheduler;
    auto bvh = Builder::build(scheduler, executor, global_bbox, bboxes.get(), centers.get(), triangles.size());

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
