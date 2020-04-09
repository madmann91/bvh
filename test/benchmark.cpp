#include <cstdio>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cstdint>

#include <bvh/bvh.hpp>
#include <bvh/binned_sah_builder.hpp>
#include <bvh/sweep_sah_builder.hpp>
#include <bvh/parallel_reinsertion_optimization.hpp>
#include <bvh/single_ray_traversal.hpp>
#include <bvh/intersectors.hpp>
#include <bvh/triangle.hpp>

using Scalar   = float;
using Vector3  = bvh::Vector3<Scalar>;
using Triangle = bvh::Triangle<Scalar>;
using Ray      = bvh::Ray<Scalar>;
using Bvh      = bvh::Bvh<Scalar>;

#include "obj.hpp"

template <typename F>
void profile(const char* task, F f) {
    auto start_tick = std::chrono::high_resolution_clock::now();
    f();
    auto end_tick = std::chrono::high_resolution_clock::now();
    size_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_tick - start_tick).count();
    std::cout << task << " took " << ms << "ms" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout <<
            "Usage: benchmark"
            "    [--fast-bvh-build]"
            "    [--eye x y z]"
            "    [--dir x y z]"
            "    [--up x y z]"
            "    [--fov d]"
            "    [--width pixels]"
            "    [--height pixels]"
            "    file.obj"
            << std::endl;
        return 1;
    }

    bool fast_build = false;
    const char* input_file = NULL;
    Scalar fov = 60;
    Vector3 eye(0, 0, -10);
    Vector3 dir(0, 0, 1);
    Vector3 up (0, 1, 0);
    size_t width  = 1080;
    size_t height = 720;
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') {
            if (!strcmp(argv[i], "--fast-bvh-build"))
                fast_build = true;
            else if (!strcmp(argv[i], "--eye") ||
                     !strcmp(argv[i], "--dir") ||
                     !strcmp(argv[i], "--up")) {
                if (i + 3 >= argc) {
                    std::cerr << "Not enough arguments for '" << argv[i] << "'" << std::endl;
                    return 1;
                }
                Vector3* destination;
                switch (argv[i][2]) {
                    case 'd': destination = &dir; break;
                    case 'u': destination = &up;  break;
                    default:  destination = &eye; break;
                }
                (*destination)[0] = strtof(argv[++i], NULL);
                (*destination)[1] = strtof(argv[++i], NULL);
                (*destination)[2] = strtof(argv[++i], NULL);
            } else if (!strcmp(argv[i], "--fov")) {
                if (i + 1 >= argc) {
                    std::cerr << "Not enough arguments for '" << argv[i] << "'" << std::endl;
                    return 1;
                }
                fov = strtof(argv[++i], NULL);
            } else if (!strcmp(argv[i], "--width") ||
                       !strcmp(argv[i], "--height")) {
                if (i + 1 >= argc) {
                    std::cerr << "Not enough arguments for '" << argv[i] << "'" << std::endl;
                    return 1;
                }
                size_t* destination = argv[i][2] == 'w' ? &width : &height;
                *destination = strtoull(argv[++i], NULL, 10);
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

    // Load mesh from file
    auto objects = obj::load_from_file(input_file);
    if (objects.size() == 0) {
        std::cerr << "The given scene is empty or cannot be loaded" << std::endl;
        return 1;
    }

    // Build an acceleration data structure for this object set
    static constexpr bool pre_shuffle = true;
    static constexpr size_t bin_count = 32;
    Bvh bvh;

    profile("BVH construction", [&] {
        auto [bboxes, centers] = bvh::compute_bounding_boxes_and_centers(objects.data(), objects.size());
        if (fast_build) {
            bvh::BinnedSahBuilder<Bvh, bin_count> builder(bvh);
            builder.build(bboxes.get(), centers.get(), objects.size());
        } else {
            bvh::SweepSahBuilder<Bvh> builder(bvh);
            builder.build(bboxes.get(), centers.get(), objects.size());
            bvh::ParallelReinsertionOptimization<Bvh> optimization(bvh);
            optimization.optimize();
        }
        if (pre_shuffle)
            bvh::shuffle_primitives(objects.data(), bvh.primitive_indices.get(), objects.size());
    });

    std::cout << bvh.node_count << " node(s)" << std::endl;

    auto pixels = std::make_unique<Scalar[]>(3 * width * height);

    bvh::ClosestIntersector<pre_shuffle, Bvh, Triangle> intersector(&bvh, objects.data());
    bvh::SingleRayTraversal<Bvh> traversal(bvh);

    // Camera tangent space
    dir = bvh::normalize(dir);
    auto image_u = bvh::normalize(bvh::cross(dir, up));
    auto image_v = bvh::normalize(bvh::cross(image_u, dir));
    auto image_w = std::tan(fov * Scalar(3.14159265 * (1.0 / 180.0) * 0.5));
    auto ratio = Scalar(height) / Scalar(width);
    image_u = image_u * image_w;
    image_v = image_v * image_w * ratio;
    std::cout << "Rendering image (" << width << "x" << height << ")..." << std::endl;

    profile("Rendering", [&] {
        #pragma omp parallel for collapse(2)
        for(size_t i = 0; i < width; ++i) {
            for(size_t j = 0; j < height; ++j) {
                size_t index = 3 * (width * j + i);

                auto u = 2 * (i + Scalar(0.5)) / Scalar(width)  - Scalar(1);
                auto v = 2 * (j + Scalar(0.5)) / Scalar(height) - Scalar(1);

                Ray ray(eye, bvh::normalize(image_u * u + image_v * v + dir));

                auto hit = traversal.intersect(ray, intersector);
                if(!hit) {
                    pixels[index] = pixels[index + 1] = pixels[index + 2] = 0;
                } else {
                    auto normal = bvh::normalize(objects[hit->primitive_index].n);
                    pixels[index    ] = std::fabs(normal[0]);
                    pixels[index + 1] = std::fabs(normal[1]);
                    pixels[index + 2] = std::fabs(normal[2]);
                }
            }
        }
    });

    std::ofstream out("render.ppm", std::ofstream::binary);
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
