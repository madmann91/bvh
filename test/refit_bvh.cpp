#include <vector>
#include <iostream>
#include <random>
#include <array>

#include <bvh/bvh.hpp>
#include <bvh/vector.hpp>
#include <bvh/triangle.hpp>
#include <bvh/ray.hpp>
#include <bvh/sweep_sah_builder.hpp>
#include <bvh/single_ray_traverser.hpp>
#include <bvh/primitive_intersectors.hpp>
#include <bvh/bounding_box.hpp>
#include <bvh/hierarchy_refitter.hpp>

using Scalar      = float;
using Vector3     = bvh::Vector3<Scalar>;
using BoundingBox = bvh::BoundingBox<Scalar>;
using Triangle    = bvh::Triangle<Scalar>;
using Ray         = bvh::Ray<Scalar>;
using Bvh         = bvh::Bvh<Scalar>;

static Vector3 random_vector() {
    std::default_random_engine gen;
    std::uniform_real_distribution<Scalar> uniform(0, 1);
    return Vector3(uniform(gen), uniform(gen), uniform(gen));
}

static std::vector<Triangle> random_triangles(size_t triangle_count) {
    std::vector<Triangle> triangles(triangle_count);
    for (size_t i = 0; i < triangle_count; ++i)
        triangles[i] = Triangle(random_vector(), random_vector(), random_vector());
    return triangles;
}

template <typename Primitive>
bool check_bvh(const Bvh& bvh, const std::vector<Primitive>& primitives) {
    return std::all_of(
        bvh.nodes.get(), bvh.nodes.get() + bvh.node_count,
        [&] (const Bvh::Node& node) {
            if (node.is_leaf()) {
                return std::all_of(
                    bvh.primitive_indices.get() + node.first_child_or_primitive,
                    bvh.primitive_indices.get() + node.first_child_or_primitive + node.primitive_count,
                    [&] (size_t index) {
                        return primitives[index].bounding_box().is_contained_in(node.bounding_box_proxy());
                    });
            } else {
                auto left_bbox  = bvh.nodes[node.first_child_or_primitive + 0].bounding_box_proxy().to_bounding_box();
                auto right_bbox = bvh.nodes[node.first_child_or_primitive + 1].bounding_box_proxy().to_bounding_box();
                return
                    left_bbox.is_contained_in(node.bounding_box_proxy()) &&
                    right_bbox.is_contained_in(node.bounding_box_proxy());
            }
        });
}

static bool create_and_refit_bvh(size_t primitive_count) {
    auto triangles = random_triangles(primitive_count);

    Bvh bvh;

    // Create an acceleration data structure on those triangles
    bvh::SweepSahBuilder<Bvh> builder(bvh);
    auto [bboxes, centers] = bvh::compute_bounding_boxes_and_centers(triangles.data(), triangles.size());
    auto global_bbox = bvh::compute_bounding_boxes_union(bboxes.get(), triangles.size());
    builder.build(global_bbox, bboxes.get(), centers.get(), triangles.size());

    std::cout << "Created BVH with " << bvh.node_count << " nodes" << std::endl;

    // Randomly modify triangles
    for (auto& triangle : triangles) {
        triangle.p0 += random_vector();
        triangle.e1 += random_vector();
        triangle.e2 += random_vector();
    }

    // Refit the BVH
    bvh::HierarchyRefitter<Bvh> refitter(bvh);
    refitter.refit([&] (Bvh::Node& leaf) {
        assert(leaf.is_leaf());
        auto bbox = BoundingBox::empty();
        for (size_t i = 0; i < leaf.primitive_count; ++i) {
            auto& triangle = triangles[bvh.primitive_indices[leaf.first_child_or_primitive + i]];
            bbox.extend(triangle.bounding_box());
        }
        leaf.bounding_box_proxy() = bbox;
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
