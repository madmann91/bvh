#include <cassert>
#include <optional>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cstring>
#include <vector>

static constexpr size_t stack_size = 64;
static constexpr bool   use_robust_traversal = false;

#include <bvh/v2/bvh.h>
#include <bvh/v2/vec.h>
#include <bvh/v2/ray.h>
#include <bvh/v2/node.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/stack.h>
#include <bvh/v2/tri.h>
#include <bvh/v2/sphere.h>

#include "load_obj.h"

using Scalar  = float;
using Vec3    = bvh::v2::Vec<Scalar, 3>;
using BBox    = bvh::v2::BBox<Scalar, 3>;
using Tri     = bvh::v2::Tri<Scalar, 3>;
using Node    = bvh::v2::Node<Scalar, 3>;
using Bvh     = bvh::v2::Bvh<Node>;
using Ray     = bvh::v2::Ray<Scalar, 3>;

template struct bvh::v2::Sphere<Scalar, 3>;

using PrecomputedTri = bvh::v2::PrecomputedTri<Scalar>;

#if USE_BVHLIB
#include <bvh/bvh.hpp>
#include <bvh/triangle.hpp>
#include <bvh/vector.hpp>
#include <bvh/bounding_box.hpp>
#include <bvh/binned_sah_builder.hpp>
#include <bvh/sweep_sah_builder.hpp>
#include <bvh/locally_ordered_clustering_builder.hpp>
#include <bvh/parallel_reinsertion_optimizer.hpp>
#include <bvh/leaf_collapser.hpp>
#include <bvh/single_ray_traverser.hpp>
#include <bvh/primitive_intersectors.hpp>
#endif

static const size_t width = 4096;
static const size_t height = 4096;
static const auto output_file = "out.ppm";

template <typename Clock = std::chrono::high_resolution_clock, typename F>
auto profile(F&& f) -> typename Clock::duration {
    auto start = Clock::now();
    f();
    auto end = Clock::now();
    return end - start;
}

template <typename Duration>
static size_t to_ms(const Duration& duration) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

#if USE_BVHLIB
namespace bvhlib {

using Vector3      = bvh::Vector3<Scalar>;
using BoundingBox  = bvh::BoundingBox<Scalar>;
using Triangle     = bvh::Triangle<Scalar>;
using Ray          = bvh::Ray<Scalar>;
using Bvh          = bvh::Bvh<Scalar>;

template <bool>
struct Traverser {};

template <>
struct Traverser<true> {
    using Type = bvh::SingleRayTraverser<Bvh, stack_size, bvh::RobustNodeIntersector<Bvh>>;
};

template <>
struct Traverser<false> {
    using Type = bvh::SingleRayTraverser<Bvh, stack_size, bvh::FastNodeIntersector<Bvh>>;
};

Vector3 convert(const Vec3& vec) {
    return Vector3(vec[0], vec[1], vec[2]);
}

BoundingBox convert(const BBox& bbox) {
    return BoundingBox(convert(bbox.min), convert(bbox.max));
}

Triangle convert(const Tri& triangle) {
    return Triangle(convert(triangle.p0), convert(triangle.p1), convert(triangle.p2));
}

}
#endif

int main(int argc, char** argv) {
#if USE_BVHLIB
    std::cout << "Using BVH library" << std::endl;
#else
    std::cout << "Using mini BVH library" << std::endl;
#endif
    Vec3 eye(0, 2, 0);
    Vec3 dir(0.1, -0.5, -1.0);
    Vec3 up(0, 1, 0);

    if (argc < 2) {
        std::cerr << "Missing input file" << std::endl;
        return 1;
    }
    auto tris = load_obj<Scalar>(argv[1]);
    if (tris.empty()) {
        std::cerr << "No triangle was found in input OBJ file" << std::endl;
        return 1;
    }
    std::cout << "Loaded file with " << tris.size() << " triangle(s)" << std::endl;

    size_t node_count = 0;
#if USE_BVHLIB
    auto global_bbox = bvhlib::BoundingBox::empty();;
    std::vector<bvhlib::BoundingBox> bboxes(tris.size());
    std::vector<bvhlib::Vector3>     centers(tris.size());
    for (size_t i = 0; i < tris.size(); ++i) {
        bboxes[i] = bvhlib::BoundingBox(bvhlib::convert(tris[i].p0))
            .extend(bvhlib::convert(tris[i].p1))
            .extend(bvhlib::convert(tris[i].p2));
        centers[i] = bvhlib::convert((tris[i].p0 + tris[i].p1 + tris[i].p2) * (1.0f / 3.0f));
        global_bbox.extend(bboxes[i]);
    }

    bvhlib::Bvh bvh;
    auto build_time = profile<std::chrono::system_clock>([&] {
        //bvh::SweepSahBuilder<bvhlib::Bvh> builder(bvh);
        bvh::LocallyOrderedClusteringBuilder<bvhlib::Bvh, uint32_t> builder(bvh);
        builder.build(global_bbox, bboxes.data(), centers.data(), tris.size());
        bvh::LeafCollapser<bvhlib::Bvh> collapser(bvh);
        collapser.collapse();
    });
    node_count = bvh.node_count;
    //bvh::ParallelReinsertionOptimizer<bvhlib::Bvh> optimizer(bvh);
    //optimizer.optimize();

    std::vector<bvhlib::Triangle> permuted_tris(tris.size());
    for (size_t i = 0; i < tris.size(); ++i)
        permuted_tris[i] = bvhlib::convert(tris[bvh.primitive_indices[i]]);

    bvh::ClosestPrimitiveIntersector<bvhlib::Bvh, bvhlib::Triangle, true> primitive_intersector(bvh, permuted_tris.data());
    bvhlib::Traverser<use_robust_traversal>::Type traverser(bvh);
#else
    bvh::v2::ThreadPool thread_pool;

    std::vector<BBox> bboxes(tris.size());
    std::vector<Vec3> centers(tris.size());
    thread_pool.parallel_for(0, tris.size(), [&] (size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            bboxes[i]  = tris[i].get_bbox();
            centers[i] = tris[i].get_center();
        }
    });

    Bvh bvh;
    auto build_time = profile<std::chrono::system_clock>([&] {
        typename bvh::v2::DefaultBuilder<Node>::Config config;
        config.quality = bvh::v2::DefaultBuilder<Node>::Quality::High;
        bvh = bvh::v2::DefaultBuilder<Node>::build(thread_pool, bboxes, centers);
    });

    std::vector<PrecomputedTri> permuted_tris(tris.size());
    thread_pool.parallel_for(0, tris.size(), [&] (size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i)
            permuted_tris[i] = tris[bvh.prim_ids[i]];
    });

    node_count = bvh.nodes.size();
#endif

    std::cout << "Built BVH with " << node_count << " node(s) built in " << to_ms(build_time) << "ms" << std::endl;

    dir = normalize(dir);
    auto right = normalize(cross(dir, up));
    up = cross(right, dir);

    std::vector<uint8_t> image(width * height * 3);
    size_t intersections = 0;
    size_t visited_leaves = 0;
    size_t visited_nodes = 0;
    auto intersection_time = profile([&] {
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                auto u = 2.0f * static_cast<float>(x)/static_cast<float>(width) - 1.0f;
                auto v = 2.0f * static_cast<float>(y)/static_cast<float>(height) - 1.0f;
                static constexpr uint32_t invalid_id = -1;
                uint32_t prim_id = invalid_id;
#if USE_BVHLIB
                bvhlib::Ray ray(bvhlib::convert(eye), bvhlib::convert(dir + u * right + v * up));
                auto hit = traverser.traverse(ray, primitive_intersector);
                if (hit) {
                    prim_id = hit->primitive_index;
                    intersections++;
                }
#else
                Ray ray(eye, dir + u * right + v * up);
                bvh::v2::SmallStack<Bvh::Index, stack_size> stack;
                bvh.intersect<false, use_robust_traversal>(ray, bvh.get_root().index, stack,
                    [&] (size_t begin, size_t end) {
                        visited_leaves++;
                        for (size_t i = begin; i < end; ++i) {
                            if (permuted_tris[i].intersect(ray))
                                prim_id = i;
                        }
                        return prim_id != invalid_id;
                    },
                    [&] (auto&&, auto&&) {
                        visited_nodes++;
                    });
                if (prim_id != invalid_id)
                    intersections++;
#endif
                auto pixel = 3 * (y * width + x);
                image[pixel + 0] = prim_id * 37;
                image[pixel + 1] = prim_id * 91;
                image[pixel + 2] = prim_id * 51;
            }
        }
    });
    std::cout
        << intersections << " intersection(s) found in " << to_ms(intersection_time)  << "ms\n"
        << "Traversal visited " << visited_nodes << " nodes and " << visited_leaves << " leaves" << std::endl;

    std::ofstream out(output_file, std::ofstream::binary);
    out << "P6 " << width << " " << height << " " << 255 << "\n";
    for(size_t j = height; j > 0; --j)
        out.write(reinterpret_cast<char*>(image.data() + (j - 1) * 3 * width), sizeof(uint8_t) * 3 * width);
    std::cout << "Image saved as " << output_file << std::endl;
    return 0;
}
