#include <vector>
#include <ranges>
#include <numeric>
#include <iostream>

#include <proto/vec.h>
#include <proto/bbox.h>
#include <proto/ray.h>
#include <proto/triangle.h>
#include <par/sequential_executor.h>

#include <bvh/bvh.h>
#include <bvh/single_ray_traverser.h>
#include <bvh/binned_sah_builder.h>
#include <bvh/sequential_top_down_scheduler.h>

using Scalar   = float;
using Triangle = proto::Triangle<Scalar>;
using Ray      = proto::Ray<Scalar>;
using Vec3     = proto::Vec3<Scalar>;
using BBox     = proto::BBox<Scalar>;
using Bvh      = bvh::Bvh<Scalar>;

/// This is an example of custom primitive.
/// Since the new version of the library is more explicit,
/// the methods and types of this primitive need not have a specific name.
struct CustomPrimitive  {
    CustomPrimitive() = default;

    struct Intersection {
        float some_value;
    };

    Vec3 center() const { return Vec3(0, 0, 0); }
    BBox bbox() const { return BBox(Vec3(-1, -1, -1), Vec3(1, 1, 1)); }

    bool intersect(Ray& ray) const {
        // When an intersection is found, we update the ray's `tmax` member to reflect
        // the distance at which the intersection has been found.
        // In turn, this informs the traversal to ignore intersections that appear further
        // than this distance.
        ray.tmax = (ray.tmin + ray.tmax) * Scalar(0.5);
        return true;
    }
};

int main() {
    par::SequentialExecutor executor;

    // Create an array of primitives
    std::vector<CustomPrimitive> primitives;
    primitives.emplace_back();
    primitives.emplace_back();

    // Compute bounding boxes and centers for every primitive
    auto bboxes  = std::make_unique<BBox[]>(primitives.size());
    auto centers = std::make_unique<Vec3[]>(primitives.size());
    auto global_bbox = par::transform_reduce(
        executor, par::range_1d(size_t{0}, primitives.size()), BBox::empty(),
        [] (BBox left, const BBox& right) { return left.extend(right); },
        [&] (size_t i) -> BBox {
            auto bbox  = primitives[i].bbox();
            centers[i] = primitives[i].center();
            return bboxes[i] = bbox;
        });

    // Build a BVH on those primitives
    using Builder = bvh::BinnedSahBuilder<Bvh>;
    bvh::SequentialTopDownScheduler<Builder> scheduler;
    auto bvh = Builder::build(scheduler, global_bbox, bboxes.get(), centers.get(), primitives.size());

    // Intersect a ray with the data structure
    Ray ray(
        Vec3(0.0, 0.0, 0.0), // origin
        Vec3(0.0, 0.0, 1.0), // direction
        0.0,                 // minimum distance
        100.0                // maximum distance
    );

    struct Hit {
        // The user is free to add whatever information is required in this structure.
        // For now, we only really need to know which primitive was intersected.
        size_t prim_index;
    };

    auto hit = bvh::SingleRayTraverser<Bvh>::traverse<false>(ray, bvh, [&] (Ray& ray, const Bvh::Node& leaf) {
        std::optional<Hit> hit;
        for (size_t i = 0; i < leaf.prim_count; ++i) {
            size_t prim_index = bvh.prim_indices[leaf.first_index + i];
            if (primitives[prim_index].intersect(ray))
                hit = Hit { prim_index };
        }
        return hit;
    });

    if (hit) {
        std::cout
            << "Hit primitive: " << hit->prim_index << "\n"
            << "distance: "      << ray.tmax        << std::endl;
        return 0;
    }
    return 1;
}
