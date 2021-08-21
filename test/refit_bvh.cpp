#include <vector>
#include <iostream>
#include <random>
#include <ranges>
#include <numeric>
#include <array>

#include <proto/vec.h>
#include <proto/ray.h>
#include <proto/bbox.h>
#include <proto/triangle.h>
#include <par/sequential_executor.h>

#include <bvh/bvh.h>
#include <bvh/sweep_sah_builder.h>
#include <bvh/sequential_top_down_scheduler.h>
#include <bvh/single_ray_traverser.h>
#include <bvh/parallel_hierarchy_refitter.h>

using Scalar   = float;
using Vec3     = proto::Vec3<Scalar>;
using BBox     = proto::BBox<Scalar>;
using Triangle = proto::Triangle<Scalar>;
using Ray      = proto::Ray<Scalar>;
using Bvh      = bvh::Bvh<Scalar>;

static Vec3 random_vector() {
    std::default_random_engine gen;
    std::uniform_real_distribution<Scalar> uniform(0, 1);
    return Vec3(uniform(gen), uniform(gen), uniform(gen));
}

static std::vector<Triangle> random_triangles(size_t triangle_count) {
    std::vector<Triangle> triangles(triangle_count);
    for (size_t i = 0; i < triangle_count; ++i)
        triangles[i] = Triangle(random_vector(), random_vector(), random_vector());
    return triangles;
}

template <typename Prim>
bool check_bvh(const Bvh& bvh, const std::vector<Prim>& prims) {
    return std::all_of(
        bvh.nodes.begin(), bvh.nodes.end(),
        [&] (const Bvh::Node& node) {
            if (node.is_leaf()) {
                return std::all_of(
                    bvh.prim_indices.begin() + node.first_index,
                    bvh.prim_indices.begin() + node.first_index + node.prim_count,
                    [&] (size_t index) {
                        return prims[index].bbox().is_contained_in(node.bbox());
                    });
            } else {
                auto left_bbox  = bvh.nodes[node.first_index + 0].bbox();
                auto right_bbox = bvh.nodes[node.first_index + 1].bbox();
                return
                    left_bbox.is_contained_in(node.bbox()) &&
                    right_bbox.is_contained_in(node.bbox());
            }
        });
}

static bool create_and_refit_bvh(size_t primitive_count) {
    par::SequentialExecutor executor;

    auto triangles = random_triangles(primitive_count);

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

    Bvh bvh;

    // Create an acceleration data structure on those triangles
    using Builder = bvh::SweepSahBuilder<Bvh>;
    bvh::SequentialTopDownScheduler<Builder> scheduler;
    Builder::build(scheduler, executor, global_bbox, bboxes.get(), centers.get(), triangles.size());

    std::cout << "Created BVH with " << bvh.nodes.size() << " nodes" << std::endl;

    // Randomly modify triangles
    for (auto& triangle : triangles) {
        triangle.v0 += random_vector();
        triangle.v1 += random_vector();
        triangle.v2 += random_vector();
    }

    // Refit the BVH
    bvh::ParallelHierarchyRefitter<Bvh> refitter;
    refitter.refit(executor, bvh, bvh.parents(executor), [&] (Bvh::Node& leaf) {
        assert(leaf.is_leaf());
        auto bbox = BBox::empty();
        for (size_t i = 0; i < leaf.prim_count; ++i) {
            auto& triangle = triangles[bvh.prim_indices[leaf.first_index + i]];
            bbox.extend(triangle.bbox());
        }
        leaf.bbox_proxy() = bbox;
    });

    return check_bvh(bvh, triangles);
}

int main() {
    std::array<size_t, 5> sizes { 1, 32, 64, 256, 1024 };
    for (auto size : sizes) {
        if (!create_and_refit_bvh(size)) {
            std::cerr << "Failed to refit BVH" << std::endl;
            return 1;
        }
    }
    return 0;
}
