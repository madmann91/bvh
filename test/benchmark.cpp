#include <cstdio>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <functional>

#include <bvh/bvh.hpp>
#include <bvh/binned_sah_builder.hpp>
#include <bvh/sweep_sah_builder.hpp>
#include <bvh/spatial_split_bvh_builder.hpp>
#include <bvh/locally_ordered_clustering_builder.hpp>
#include <bvh/linear_bvh_builder.hpp>
#include <bvh/parallel_reinsertion_optimizer.hpp>
#include <bvh/node_layout_optimizer.hpp>
#include <bvh/leaf_collapser.hpp>
#include <bvh/heuristic_primitive_splitter.hpp>
#include <bvh/hierarchy_refitter.hpp>
#include <bvh/single_ray_traverser.hpp>
#include <bvh/primitive_intersectors.hpp>
#include <bvh/triangle.hpp>

using Scalar      = float;
using Vector3     = bvh::Vector3<Scalar>;
using Triangle    = bvh::Triangle<Scalar>;
using BoundingBox = bvh::BoundingBox<Scalar>;
using Ray         = bvh::Ray<Scalar>;
using Bvh         = bvh::Bvh<Scalar>;

#include "obj.hpp"

template <typename F>
void profile(const char* task, F f, size_t runs = 1) {
    using namespace std::chrono;
    using Timing = std::chrono::milliseconds::rep;

    std::vector<Timing> timings;

    for (size_t i = 0; i < runs; ++i) {
        auto start_tick = high_resolution_clock::now();
        f();
        auto end_tick = high_resolution_clock::now();
        timings.push_back(duration_cast<milliseconds>(end_tick - start_tick).count());
    }

    std::sort(timings.begin(), timings.end());
    if (timings.size() == 1)
        std::cout << task << " took " << timings.front() << "ms" << std::endl;
    else {
        std::cout
            << task << " took "
            << timings.front() << "/"
            << timings[timings.size() / 2] << "/"
            << timings.back() << "ms (min/med/max of " << runs << " runs)" << std::endl;
    }
}

static size_t compute_bvh_depth(const Bvh& bvh, size_t node_index = 0) {
    auto& node = bvh.nodes[node_index];
    if (node.primitive_count == 0) {
        return 1 + std::max(
            compute_bvh_depth(bvh, node.first_child_or_primitive + 0),
            compute_bvh_depth(bvh, node.first_child_or_primitive + 1));
    } else
        return 0;
}

static int not_enough_arguments(const char* option) {
    std::cerr << "Not enough arguments for '" << option << "'" << std::endl;
    return 1;
}

static void usage() {
    std::cout <<
        "Usage: benchmark [options] file.obj\n"
        "\nOptions:\n"
        "  --help                  Shows this message.\n"
        "  --builder <name>        Sets the BVH builder to use (defaults to 'binned_sah').\n"
        "  --permute               Activates the primitive permutation optimization (disabled by default).\n"
        "  --optimize-layout       Activates the node layout optimization (disabled by default).\n"
        "  --collapse-leaves       Activates the leaf collapse optimization (disabled by default).\n"
        "  --parallel-reinsertion  Activates the parallel reinsertion optimization (disabled by default).\n"
        "  --pre-split <percent>   Activates pre-splitting and sets the percentage of references (disabled by default).\n"
        "  --build-iterations <n>  Sets the number of construction iterations (equal to 1 by default).\n"
        "  --eye <x> <y> <z>       Sets the position of the camera.\n"
        "  --dir <x> <y> <z>       Sets the direction of the camera.\n"
        "  --up  <x> <y> <z>       Sets the up vector of the camera.\n"
        "  --fov <degrees>         Sets the field of view.\n"
        "  --width <pixels>        Sets the image width.\n"
        "  --height <pixels>       Sets the image height.\n"
        "  -o <file.ppm>           Sets the output file name (defaults to 'render.ppm').\n\n"
        "  --rotate <axis> <degrees>\n\n"
        "    Rotates the scene by the given amount of degrees on the\n"
        "    given axis (valid axes are 'x', 'y', or 'z'). This is mainly\n"
        "    intended to test the impact of pre-splitting.\n\n"
        "  --collect-statistics <t> <i> <c>\n\n"
        "    Collects traversal statistics per pixel.\n"
        "    The arguments represent the weight of traversal steps (t),\n"
        "    primitive intersections (i), and the sum of the two (s).\n"
        "    These statistics are then converted to bytes and stored in\n"
        "    the red, green, and blue channels of the image, respectively.\n\n"
        "Builders:\n"
        "  binned_sah,\n"
        "  sweep_sah,\n"
        "  spatial_split,\n"
        "  locally_ordered_clustering,\n"
        "  linear\n"
        << std::endl;
}

struct Camera {
    Vector3 eye;
    Vector3 dir;
    Vector3 up;
    Scalar  fov;
};

template <bool Permute, bool CollectStatistics>
void render(
    const Camera& camera,
    const Bvh& bvh,
    const Triangle* triangles,
    Scalar* pixels,
    size_t width, size_t height,
    const Scalar* statistics_weights = NULL)
{
    auto dir = bvh::normalize(camera.dir);
    auto image_u = bvh::normalize(bvh::cross(dir, camera.up));
    auto image_v = bvh::normalize(bvh::cross(image_u, dir));
    auto image_w = std::tan(camera.fov * Scalar(3.14159265 * (1.0 / 180.0) * 0.5));
    auto ratio = Scalar(height) / Scalar(width);
    image_u = image_u * image_w;
    image_v = image_v * image_w * ratio;

    bvh::ClosestPrimitiveIntersector<Bvh, Triangle, Permute> intersector(bvh, triangles);
    bvh::SingleRayTraverser<Bvh> traverser(bvh);

    size_t traversal_steps = 0, intersections = 0;

    #pragma omp parallel for collapse(2) reduction(+: traversal_steps, intersections)
    for(size_t i = 0; i < width; ++i) {
        for(size_t j = 0; j < height; ++j) {
            size_t index = 3 * (width * j + i);

            auto u = 2 * (static_cast<Scalar>(i) + Scalar(0.5)) / static_cast<Scalar>(width)  - Scalar(1);
            auto v = 2 * (static_cast<Scalar>(j) + Scalar(0.5)) / static_cast<Scalar>(height) - Scalar(1);

            Ray ray(camera.eye, bvh::normalize(image_u * u + image_v * v + dir));

            bvh::SingleRayTraverser<Bvh>::Statistics statistics;
            auto hit = CollectStatistics
                ? traverser.traverse(ray, intersector, statistics)
                : traverser.traverse(ray, intersector);
            if (CollectStatistics) {
                traversal_steps += statistics.traversal_steps;
                intersections   += statistics.intersections;
            }
            if (!hit) {
                pixels[index] = pixels[index + 1] = pixels[index + 2] = 0;
            } else if (CollectStatistics) {
                auto combined = statistics.traversal_steps + statistics.intersections;
                pixels[index    ] = std::min(static_cast<Scalar>(statistics.traversal_steps) * statistics_weights[0], Scalar(1));
                pixels[index + 1] = std::min(static_cast<Scalar>(statistics.intersections)   * statistics_weights[1], Scalar(1));
                pixels[index + 2] = std::min(static_cast<Scalar>(combined)                   * statistics_weights[2], Scalar(1));
            } else {
                auto normal = bvh::normalize(triangles[hit->primitive_index].n);
                pixels[index    ] = std::fabs(normal[0]);
                pixels[index + 1] = std::fabs(normal[1]);
                pixels[index + 2] = std::fabs(normal[2]);
            }
        }
    }

    if (CollectStatistics) {
        std::cout
            << intersections << " total primitive intersection(s)\n"
            << traversal_steps << " total traversal step(s)" << std::endl;
    }
}

template <size_t Axis>
static void rotate_triangles(Scalar degrees, Triangle* triangles, size_t triangle_count) {
    static constexpr Scalar pi = Scalar(3.14159265359);
    auto cos = std::cos(degrees * pi / Scalar(180));
    auto sin = std::sin(degrees * pi / Scalar(180));
    auto rotate = [&] (const Vector3& p) {
        if (Axis == 0)
            return Vector3(p[0], p[1] * cos - p[2] * sin, p[1] * sin + p[2] * cos);
        else if (Axis == 1)
            return Vector3(p[0] * cos + p[2] * sin, p[1], -p[0] * sin + p[2] * cos);
        else
            return Vector3(p[0] * cos - p[1] * sin, p[0] * sin + p[1] * cos, p[2]);
    };
    #pragma omp parallel for
    for (size_t i = 0; i < triangle_count; ++i) {
        auto p0 = rotate(triangles[i].p0);
        auto p1 = rotate(triangles[i].p1());
        auto p2 = rotate(triangles[i].p2());
        triangles[i] = Triangle(p0, p1, p2);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }

    const char* output_file  = "render.ppm";
    const char* input_file   = NULL;
    const char* builder_name = "binned_sah";
    Camera camera = {
        Vector3(0, 0, -10),
        Vector3(0, 0, 1),
        Vector3(0, 1, 0),
        60
    };
    bool permute = false;
    bool optimize_layout = false;
    bool parallel_reinsertion = false;
    bool collapse_leaves = false;
    size_t build_iterations = 1;
    Scalar pre_split_factor = 0;
    bool collect_statistics = false;
    size_t rotation_axis = 3;
    Scalar rotation_degrees = 0;
    Scalar statistics_weights[3];
    size_t width  = 1080;
    size_t height = 720;
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') {
            if (!strcmp(argv[i], "--help")) {
                usage();
                return 1;
            } else if (!strcmp(argv[i], "--eye") ||
                !strcmp(argv[i], "--dir") ||
                !strcmp(argv[i], "--up")) {
                if (i + 3 >= argc)
                    return not_enough_arguments(argv[i]);
                Vector3* destination;
                switch (argv[i][2]) {
                    case 'd': destination = &camera.dir; break;
                    case 'u': destination = &camera.up;  break;
                    default:  destination = &camera.eye; break;
                }
                (*destination)[0] = strtof(argv[++i], NULL);
                (*destination)[1] = strtof(argv[++i], NULL);
                (*destination)[2] = strtof(argv[++i], NULL);
            } else if (!strcmp(argv[i], "--fov")) {
                if (i + 1 >= argc)
                    return not_enough_arguments(argv[i]);
                camera.fov = strtof(argv[++i], NULL);
            } else if (!strcmp(argv[i], "--width") ||
                       !strcmp(argv[i], "--height")) {
                if (i + 1 >= argc)
                    return not_enough_arguments(argv[i]);
                size_t* destination = argv[i][2] == 'w' ? &width : &height;
                *destination = strtoull(argv[++i], NULL, 10);
            } else if (!strcmp(argv[i], "--builder")) {
                if (i + 1 >= argc)
                    return not_enough_arguments(argv[i]);
                builder_name = argv[++i];
            } else if (!strcmp(argv[i], "--permute")) {
                permute = true;
            } else if (!strcmp(argv[i], "--optimize-layout")) {
                optimize_layout = true;
            } else if (!strcmp(argv[i], "--parallel-reinsertion")) {
                parallel_reinsertion = true;
            } else if (!strcmp(argv[i], "--collapse-leaves")) {
                collapse_leaves = true;
            } else if (!strcmp(argv[i], "--pre-split")) {
                if (i + 1 >= argc)
                    return not_enough_arguments(argv[i]);
                pre_split_factor = strtof(argv[++i], NULL) / Scalar(100.0);
                if (pre_split_factor < 0) {
                    std::cerr << "Invalid pre-split factor." << std::endl;
                    return 1;
                }
            } else if (!strcmp(argv[i], "--build-iterations")) {
                if (i + 1 >= argc)
                    return not_enough_arguments(argv[i]);
                build_iterations = strtoull(argv[++i], NULL, 10);
                if (build_iterations == 0) {
                    std::cerr << "Invalid number of construction iterations." << std::endl;
                    return 1;
                }
            } else if (!strcmp(argv[i], "--rotate")) {
                if (i + 2 >= argc)
                    return not_enough_arguments(argv[i]);
                rotation_axis = argv[++i][0] - 'x';
                rotation_degrees = strtof(argv[++i], NULL);
                if (rotation_axis > 2) {
                    std::cerr << "Invalid rotation axis" << std::endl;
                    return 1;
                }
            } else if (!strcmp(argv[i], "--collect-statistics")) {
                if (i + 2 >= argc)
                    return not_enough_arguments(argv[i]);
                collect_statistics = true;
                statistics_weights[0] = strtof(argv[++i], NULL);
                statistics_weights[1] = strtof(argv[++i], NULL);
                statistics_weights[2] = strtof(argv[++i], NULL);
            } else if (!strcmp(argv[i], "-o")) {
                if (i + 1 >= argc)
                    return not_enough_arguments(argv[i]);
                output_file = argv[++i];
            } else {
                std::cerr << "Unknown option: '" << argv[i] << "'" << std::endl;
                return 1;
            }
        } else {
            if (input_file) {
                std::cerr << "Scene file specified twice" << std::endl;
                return 1;
            }
            input_file = argv[i];
        }
    }

    if (!input_file) {
        std::cerr << "Missing a command line argument for the scene file" << std::endl;
        return 1;
    }

    std::function<size_t(Bvh&, const Triangle*, const BoundingBox&, const BoundingBox*, const Vector3*, size_t)> builder;
    if (!strcmp(builder_name, "binned_sah")) {
        builder = [] (Bvh& bvh, const Triangle*, const BoundingBox& global_bbox, const BoundingBox* bboxes, const Vector3* centers, size_t primitive_count) {
            static constexpr size_t bin_count = 16;
            bvh::BinnedSahBuilder<Bvh, bin_count> builder(bvh);
            builder.build(global_bbox, bboxes, centers, primitive_count);
            return primitive_count;
        };
    } else if (!strcmp(builder_name, "sweep_sah")) {
        builder = [] (Bvh& bvh, const Triangle*, const BoundingBox& global_bbox, const BoundingBox* bboxes, const Vector3* centers, size_t primitive_count) {
            bvh::SweepSahBuilder<Bvh> builder(bvh);
            builder.build(global_bbox, bboxes, centers, primitive_count);
            return primitive_count;
        };
    } else if (!strcmp(builder_name, "spatial_split")) {
        builder = [] (Bvh& bvh, const Triangle* triangles, const BoundingBox& global_bbox, const BoundingBox* bboxes, const Vector3* centers, size_t primitive_count) {
            static constexpr size_t bin_count = 64;
            bvh::SpatialSplitBvhBuilder<Bvh, Triangle, bin_count> builder(bvh);
            return builder.build(global_bbox, triangles, bboxes, centers, primitive_count);
        };
    } else if (!strcmp(builder_name, "locally_ordered_clustering")) {
        builder = [] (Bvh& bvh, const Triangle*, const BoundingBox& global_bbox, const BoundingBox* bboxes, const Vector3* centers, size_t primitive_count) {
            using Morton = uint32_t;
            bvh::LocallyOrderedClusteringBuilder<Bvh, Morton> builder(bvh);
            builder.build(global_bbox, bboxes, centers, primitive_count);
            return primitive_count;
        };
    } else if (!strcmp(builder_name, "linear")) {
        builder = [] (Bvh& bvh, const Triangle*, const BoundingBox& global_bbox, const BoundingBox* bboxes, const Vector3* centers, size_t primitive_count) {
            using Morton = uint32_t;
            bvh::LinearBvhBuilder<Bvh, Morton> builder(bvh);
            builder.build(global_bbox, bboxes, centers, primitive_count);
            return primitive_count;
        };
    } else {
        std::cerr << "Unknown BVH builder name" << std::endl;
        return 1;
    }

    // Load mesh from file
    auto triangles = obj::load_from_file(input_file);
    if (triangles.size() == 0) {
        std::cerr << "The given scene is empty or cannot be loaded" << std::endl;
        return 1;
    }

    // Rotate triangles if requested
    if (rotation_axis == 0)
        rotate_triangles<0>(rotation_degrees, triangles.data(), triangles.size());
    else if (rotation_axis == 1)
        rotate_triangles<1>(rotation_degrees, triangles.data(), triangles.size());
    else if (rotation_axis == 2)
        rotate_triangles<2>(rotation_degrees, triangles.data(), triangles.size());

    Bvh bvh;

    size_t reference_count = triangles.size();
    std::unique_ptr<Triangle[]> shuffled_triangles;

    // Build an acceleration data structure for this object set
    std::cout << "Building BVH (" << builder_name;
    if (pre_split_factor)
        std::cout << " + pre-split";
    if (parallel_reinsertion)
        std::cout << " + parallel-reinsertion";
    if (optimize_layout)
        std::cout << " + optimize-layout";
    if (collapse_leaves)
        std::cout << " + collapse-leaves";
    if (permute)
        std::cout << " + permute";
    std::cout << ")..." << std::endl;
    profile("BVH construction", [&] {
        auto [bboxes, centers] =
            bvh::compute_bounding_boxes_and_centers(triangles.data(), triangles.size());
        auto global_bbox = bvh::compute_bounding_boxes_union(bboxes.get(), triangles.size());
        bvh::HeuristicPrimitiveSplitter<Triangle> splitter;
        if (pre_split_factor > 0)
            std::tie(reference_count, bboxes, centers) = splitter.split(global_bbox, triangles.data(), triangles.size(), pre_split_factor);
        reference_count = builder(bvh, triangles.data(), global_bbox, bboxes.get(), centers.get(), reference_count);
        if (pre_split_factor > 0)
            splitter.repair_bvh_leaves(bvh);
        if (parallel_reinsertion) {
            bvh::ParallelReinsertionOptimizer<Bvh> reinsertion_optimizer(bvh);
            reinsertion_optimizer.optimize();
        }
        if (optimize_layout) {
            bvh::NodeLayoutOptimizer layout_optimizer(bvh);
            layout_optimizer.optimize();
        }
        if (collapse_leaves) {
            bvh::LeafCollapser leaf_collapser(bvh);
            leaf_collapser.collapse();
        }
        if (permute)
            shuffled_triangles = bvh::permute_primitives(triangles.data(), bvh.primitive_indices.get(), reference_count);
    }, build_iterations);

    // This is just to make sure that refitting works
    bvh::HierarchyRefitter refitter(bvh);
    refitter.refit([] (Bvh::Node&) {});

    std::cout
        << "BVH depth of " << compute_bvh_depth(bvh) << ", "
        << bvh.node_count << " node(s), "
        << reference_count << " reference(s)" << std::endl;

    auto pixels = std::make_unique<Scalar[]>(3 * width * height);

    std::cout << "Rendering image (" << width << "x" << height << ")..." << std::endl;
    profile("Rendering", [&] {
        if (permute) {
            if (collect_statistics)
                render<true, true>(camera, bvh, shuffled_triangles.get(), pixels.get(), width, height, statistics_weights);
            else
                render<true, false>(camera, bvh, shuffled_triangles.get(), pixels.get(), width, height);
        } else {
            if (collect_statistics)
                render<false, true>(camera, bvh, triangles.data(), pixels.get(), width, height, statistics_weights);
            else
                render<false, false>(camera, bvh, triangles.data(), pixels.get(), width, height);
        }
    });

    std::ofstream out(output_file, std::ofstream::binary);
    out << "P6 " << width << " " << height << " " << 255 << "\n";
    for(size_t j = height; j > 0; --j) {
        for(size_t i = 0; i < width; ++i) {
            size_t index = 3* (width * (j - 1) + i);
            uint8_t pixel[3] = {
                static_cast<uint8_t>(std::max(std::min(pixels[index    ] * 255, Scalar(255)), Scalar(0))),
                static_cast<uint8_t>(std::max(std::min(pixels[index + 1] * 255, Scalar(255)), Scalar(0))),
                static_cast<uint8_t>(std::max(std::min(pixels[index + 2] * 255, Scalar(255)), Scalar(0)))
            };
            out.write(reinterpret_cast<char*>(pixel), sizeof(uint8_t) * 3);
        }
    }
}
