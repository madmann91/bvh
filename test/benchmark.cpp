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
#include <bvh/locally_ordered_clustering_builder.hpp>
#include <bvh/linear_bvh_builder.hpp>
#include <bvh/parallel_reinsertion_optimization.hpp>
#include <bvh/single_ray_traversal.hpp>
#include <bvh/intersectors.hpp>
#include <bvh/triangle.hpp>

using Scalar      = float;
using Vector3     = bvh::Vector3<Scalar>;
using Triangle    = bvh::Triangle<Scalar>;
using BoundingBox = bvh::BoundingBox<Scalar>;
using Ray         = bvh::Ray<Scalar>;
using Bvh         = bvh::Bvh<Scalar>;

#include "obj.hpp"

template <typename F>
void profile(const char* task, F f) {
    auto start_tick = std::chrono::high_resolution_clock::now();
    f();
    auto end_tick = std::chrono::high_resolution_clock::now();
    size_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_tick - start_tick).count();
    std::cout << task << " took " << ms << "ms" << std::endl;
}

static int not_enough_arguments(const char* option) {
    std::cerr << "Not enough arguments for '" << option << "'" << std::endl;
    return 1;
}

static void usage() {
    std::cout <<
        "Usage: benchmark [options] file.obj\n"
        "\nOptions:\n"
        "  --help              Shows this message.\n"
        "  --builder <name>    Sets the BVH builder to use (defaults to 'binned_sah').\n"
        "  --optimizer <name>  Sets the BVH optimizer to use (none by default).\n"
        "  --pre-shuffle       Activates the pre-shuffling optimization.\n"
        "  --eye <x> <y> <z>   Sets the position of the camera.\n"
        "  --dir <x> <y> <z>   Sets the direction of the camera.\n"
        "  --up  <x> <y> <z>   Sets the up vector of the camera.\n"
        "  --fov <degrees>     Sets the field of view.\n"
        "  --width <pixels>    Sets the image width.\n"
        "  --height <pixels>   Sets the image height.\n"
        "  -o <file.ppm>       Sets the output file name (defaults to 'render.ppm').\n\n"
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
        "\nBuilders:\n"
        "  binned_sah,\n"
        "  sweep_sah,\n"
        "  locally_ordered_clustering,\n"
        "  linear\n"
        "\nOptimizers:\n"
        "  parallel_reinsertion\n"
        << std::endl;
}

struct Camera {
    Vector3 eye;
    Vector3 dir;
    Vector3 up;
    Scalar  fov;
};

template <bool PreShuffle, bool CollectStatistics>
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

    bvh::ClosestIntersector<PreShuffle, Bvh, Triangle> intersector(bvh, triangles);
    bvh::SingleRayTraversal<Bvh> traversal(bvh);

    size_t traversal_steps = 0, intersections = 0;

    #pragma omp parallel for collapse(2) reduction(+: traversal_steps, intersections)
    for(size_t i = 0; i < width; ++i) {
        for(size_t j = 0; j < height; ++j) {
            size_t index = 3 * (width * j + i);

            auto u = 2 * (i + Scalar(0.5)) / Scalar(width)  - Scalar(1);
            auto v = 2 * (j + Scalar(0.5)) / Scalar(height) - Scalar(1);

            Ray ray(camera.eye, bvh::normalize(image_u * u + image_v * v + dir));

            bvh::SingleRayTraversal<Bvh>::Statistics statistics;
            auto hit = CollectStatistics
                ? traversal.intersect(ray, intersector, statistics)
                : traversal.intersect(ray, intersector);
            if (CollectStatistics) {
                traversal_steps += statistics.traversal_steps;
                intersections   += statistics.intersections;
            }
            if(!hit) {
                pixels[index] = pixels[index + 1] = pixels[index + 2] = 0;
            } else {
                if (CollectStatistics) {
                    auto combined = statistics.traversal_steps + statistics.intersections; 
                    pixels[index    ] = std::min(statistics.traversal_steps * statistics_weights[0], 1.0f);
                    pixels[index + 1] = std::min(statistics.intersections   * statistics_weights[1], 1.0f);
                    pixels[index + 2] = std::min(combined                   * statistics_weights[2], 1.0f);
                } else {
                    auto normal = bvh::normalize(triangles[hit->primitive_index].n);
                    pixels[index    ] = std::fabs(normal[0]);
                    pixels[index + 1] = std::fabs(normal[1]);
                    pixels[index + 2] = std::fabs(normal[2]);
                }
            }
        }
    }

    if (CollectStatistics) {
        std::cout << intersections << " total primitive intersection(s)" << std::endl;
        std::cout << traversal_steps << " total traversal step(s)" << std::endl;
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

    const char* output_file    = "render.ppm";
    const char* input_file     = NULL;
    const char* builder_name   = "binned_sah";
    const char* optimizer_name = NULL;
    Camera camera = {
        Vector3(0, 0, -10),
        Vector3(0, 0, 1),
        Vector3(0, 1, 0),
        60
    };
    bool pre_shuffle = false;
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
            } else if (!strcmp(argv[i], "--builder") ||
                       !strcmp(argv[i], "--optimizer")) {
                if (i + 1 >= argc)
                    return not_enough_arguments(argv[i]);
                const char** name = argv[i][2] == 'b' ? &builder_name : &optimizer_name;
                *name = argv[++i];
            } else if (!strcmp(argv[i], "--pre-shuffle")) {
                pre_shuffle = true;
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

    std::function<void(Bvh&, const BoundingBox&, const BoundingBox*, const Vector3*, size_t)> builder;
    if (!strcmp(builder_name, "binned_sah")) {
        builder = [] (Bvh& bvh, const BoundingBox& global_bbox, const BoundingBox* bboxes, const Vector3* centers, size_t primitive_count) {
            static constexpr size_t bin_count = 16;
            bvh::BinnedSahBuilder<Bvh, bin_count> builder(bvh);
            builder.build(global_bbox, bboxes, centers, primitive_count);
        };
    } else if (!strcmp(builder_name, "sweep_sah")) {
        builder = [] (Bvh& bvh, const BoundingBox& global_bbox, const BoundingBox* bboxes, const Vector3* centers, size_t primitive_count) {
            bvh::SweepSahBuilder<Bvh> builder(bvh);
            builder.build(global_bbox, bboxes, centers, primitive_count);
        };
    } else if (!strcmp(builder_name, "locally_ordered_clustering")) {
        builder = [] (Bvh& bvh, const BoundingBox& global_bbox, const BoundingBox* bboxes, const Vector3* centers, size_t primitive_count) {
            using Morton = uint32_t;
            bvh::LocallyOrderedClusteringBuilder<Bvh, Morton> builder(bvh);
            builder.build(global_bbox, bboxes, centers, primitive_count);
        };
    } else if (!strcmp(builder_name, "linear")) {
        builder = [] (Bvh& bvh, const BoundingBox& global_bbox, const BoundingBox* bboxes, const Vector3* centers, size_t primitive_count) {
            using Morton = uint32_t;
            bvh::LinearBvhBuilder<Bvh, Morton> builder(bvh);
            builder.build(global_bbox, bboxes, centers, primitive_count);
        };
    } else {
        std::cerr << "Unknown BVH builder name" << std::endl;
        return 1;
    }

    std::function<void(Bvh&)> optimizer;
    if (!optimizer_name) {
        optimizer = [] (Bvh&) {};
    } else if (!strcmp(optimizer_name, "parallel_reinsertion")) {
        optimizer = [] (Bvh& bvh) {
            bvh::ParallelReinsertionOptimization<Bvh> optimization(bvh);
            optimization.optimize();
        };
    } else {
        std::cerr << "Unknown BVH optimizer name" << std::endl;
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
    if (optimizer_name)
        std::cout << " + " << optimizer_name;
    if (pre_shuffle)
        std::cout << " + pre-shuffle";
    std::cout << ")..." << std::endl;
    profile("BVH construction", [&] {
        auto [bboxes, centers] =
            bvh::compute_bounding_boxes_and_centers(triangles.data(), triangles.size());
        auto global_bbox = bvh::compute_bounding_boxes_union(bboxes.get(), triangles.size());
        builder(bvh, global_bbox, bboxes.get(), centers.get(), triangles.size());
        optimizer(bvh);
        if (pre_shuffle)
            shuffled_triangles = bvh::shuffle_primitives(triangles.data(), bvh.primitive_indices.get(), triangles.size());
    });

    std::cout << bvh.node_count << " node(s), " << reference_count << " reference(s)" << std::endl;

    auto pixels = std::make_unique<Scalar[]>(3 * width * height);

    std::cout << "Rendering image (" << width << "x" << height << ")..." << std::endl;
    profile("Rendering", [&] {
        if (pre_shuffle) {
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
