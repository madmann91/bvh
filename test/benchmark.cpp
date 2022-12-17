#include <bvh/v2/bvh.h>
#include <bvh/v2/vec.h>
#include <bvh/v2/ray.h>
#include <bvh/v2/node.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/thread_pool.h>
#include <bvh/v2/stack.h>
#include <bvh/v2/tri.h>
#include <bvh/v2/sphere.h>

#include "load_obj.h"

#include <cassert>
#include <optional>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cstring>
#include <vector>
#include <string_view>

using Scalar  = float;
using Vec3    = bvh::v2::Vec<Scalar, 3>;
using BBox    = bvh::v2::BBox<Scalar, 3>;
using Tri     = bvh::v2::Tri<Scalar, 3>;
using Node    = bvh::v2::Node<Scalar, 3>;
using Bvh     = bvh::v2::Bvh<Node>;
using Ray     = bvh::v2::Ray<Scalar, 3>;

template struct bvh::v2::Sphere<Scalar, 3>;

using PrecomputedTri = bvh::v2::PrecomputedTri<Scalar>;

static constexpr size_t stack_size = 64;

static void usage() {
    std::cout <<
        "Usage: benchmark [options] file.obj\n"
        "\nOptions:\n"
        "      --help                  Shows this message.\n"
        "  -q  --quality <quality>     Sets the desired BVH quality (low, med, or high, defaults to high).\n"
        "  -p  --permute-primitives    Activates the primitive permutation optimization (disabled by default).\n"
        "  -i  --build-iterations <n>  Sets the number of construction iterations (1 by default).\n"
        "      --robust-traversal      Use the robust traversal algorithm (disabled by default).\n"
        "  -e  --eye <x> <y> <z>       Sets the position of the camera.\n"
        "  -d  --dir <x> <y> <z>       Sets the direction of the camera.\n"
        "  -u  --up  <x> <y> <z>       Sets the up vector of the camera.\n"
        "      --fov <degrees>         Sets the field of view.\n"
        "  -w  --width <pixels>        Sets the image width.\n"
        "  -h  --height <pixels>       Sets the image height.\n"
        "  -m  --render-mode <mode>    Sets the render mode (eyelight or debug, defaults to eyelight).\n"
        "  -o <file.ppm>               Sets the output file name (defaults to 'render.ppm').\n"
        "\nRender-mode specific options:\n"
        "  --debug-threshold <integer>\n\n"
        "    Sets the maximum number of traversal and intersection steps used by\n"
        "    the debug visualization (by default auto-detected).\n"
        << std::endl;
}

template <typename Clock = std::chrono::high_resolution_clock, typename F>
auto profile(F&& f, size_t iter_count = 1) -> typename Clock::duration {
    std::vector<typename Clock::duration> durations;
    for (size_t i = 0; i < iter_count; ++i) {
        auto start = Clock::now();
        f();
        auto end = Clock::now();
        durations.push_back(end - start);
    }
    std::sort(durations.begin(), durations.end());
    return durations[durations.size() / 2];
}

template <typename Duration>
static size_t to_ms(const Duration& duration) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

enum class RenderMode {
    EyeLight, Debug
};

struct Options {
    using Quality = bvh::v2::DefaultBuilder<Node>::Quality;

    bool show_usage = false;
    Vec3 eye = Vec3(0);
    Vec3 dir = Vec3(0, 0, 1);
    Vec3 up = Vec3(0, 1, 0);
    size_t width = 1024;
    size_t height = 1024;
    size_t build_iters = 1;
    bool permute_prims = false;
    bool robust_traversal = false;
    size_t debug_threshold = 0;
    RenderMode render_mode = RenderMode::EyeLight;
    Quality quality = Quality::High;
    std::string input_model;
    std::string output_image = "render.ppm";

    bool parse(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            if (argv[i][0] != '-') {
                if (!input_model.empty()) {
                    std::cerr << "Input model specified twice" << std::endl;
                    return false;
                }
                input_model = argv[i];
                continue;
            }

            using namespace std::literals;
            if (argv[i] == "--help"sv) {
                show_usage = true;
                return true;
            } else if (argv[i] == "--quality"sv || argv[i] == "-q"sv) {
                if (!check_arg(argc, argv, i))
                    return false;
                ++i;
                if (argv[i] == "low"sv)       quality = Quality::Low;
                else if (argv[i] == "med"sv)  quality = Quality::Medium;
                else if (argv[i] == "high"sv) quality = Quality::High;
                else {
                    std::cerr << "Invalid BVH quality '" << argv[i] << "'" << std::endl;
                    return false;
                }
            } else if (argv[i] == "--permute_prims"sv || argv[i] == "-p"sv) {
                permute_prims = true;
            } else if (argv[i] == "--build-iterations"sv || argv[i] == "-i"sv) {
                if (!check_arg(argc, argv, i))
                    return false;
                build_iters = std::strtoul(argv[++i], nullptr, 10);
            } else if (argv[i] == "--robust-traversal"sv) {
                robust_traversal = true;
            } else if (argv[i] == "--eye"sv || argv[i] == "-e"sv) {
                if (!check_arg(argc, argv, i, 3))
                    return false;
                eye[0] = static_cast<Scalar>(std::strtod(argv[++i], nullptr));
                eye[1] = static_cast<Scalar>(std::strtod(argv[++i], nullptr));
                eye[2] = static_cast<Scalar>(std::strtod(argv[++i], nullptr));
            } else if (argv[i] == "--dir"sv || argv[i] == "-d"sv) {
                if (!check_arg(argc, argv, i, 3))
                    return false;
                dir[0] = static_cast<Scalar>(std::strtod(argv[++i], nullptr));
                dir[1] = static_cast<Scalar>(std::strtod(argv[++i], nullptr));
                dir[2] = static_cast<Scalar>(std::strtod(argv[++i], nullptr));
            } else if (argv[i] == "--up"sv || argv[i] == "-u"sv) {
                if (!check_arg(argc, argv, i, 3))
                    return false;
                up[0] = static_cast<Scalar>(std::strtod(argv[++i], nullptr));
                up[1] = static_cast<Scalar>(std::strtod(argv[++i], nullptr));
                up[2] = static_cast<Scalar>(std::strtod(argv[++i], nullptr));
            } else if (argv[i] == "--width"sv || argv[i] == "-w"sv) {
                if (!check_arg(argc, argv, i))
                    return false;
                width = std::strtoul(argv[++i], nullptr, 10);
            } else if (argv[i] == "--height"sv || argv[i] == "-h"sv) {
                if (!check_arg(argc, argv, i))
                    return false;
                height = std::strtoul(argv[++i], nullptr, 10);
            } else if (argv[i] == "--render-mode"sv || argv[i] == "-m"sv) {
                if (!check_arg(argc, argv, i))
                    return false;
                ++i;
                if (argv[i] == "eyelight"sv)    render_mode = RenderMode::EyeLight;
                else if (argv[i] == "debug"sv)  render_mode = RenderMode::Debug;
                else {
                    std::cerr << "Invalid render mode '" << argv[i] << "'" << std::endl;
                    return false;
                }
            } else if (argv[i] == "-o"sv) {
                if (!check_arg(argc, argv, i))
                    return false;
                output_image = argv[++i];
            } else if (argv[i] == "--debug-threshold"sv) {
                if (!check_arg(argc, argv, i))
                    return false;
                debug_threshold = std::strtoull(argv[++i], nullptr, 10);
            }
        }
        if (input_model.empty()) {
            std::cerr << "Missing input model file name" << std::endl;
            return false;
        }
        return true;
    }

private:
    static bool check_arg(int argc, char** argv, int i, int j = 1) {
        if (i + j >= argc) {
            std::cerr << "Missing argument for option '" << argv[i] << "'" << std::endl;
            return false;
        }
        return true;
    }
};

struct Accel {
    Bvh bvh;
    std::vector<PrecomputedTri> tris;
};

static Accel build_accel(bvh::v2::ThreadPool& thread_pool, const std::vector<Tri>& tris, const Options& options) {
    bvh::v2::ParallelExecutor executor(thread_pool);

    std::vector<BBox> bboxes(tris.size());
    std::vector<Vec3> centers(tris.size());
    executor.for_each(0, tris.size(), [&] (size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            bboxes[i]  = tris[i].get_bbox();
            centers[i] = tris[i].get_center();
        }
    });

    Accel accel;

    typename bvh::v2::DefaultBuilder<Node>::Config config;
    config.quality = options.quality;
    accel.bvh = bvh::v2::DefaultBuilder<Node>::build(thread_pool, bboxes, centers, config);

    accel.tris.resize(tris.size());
    if (options.permute_prims) {
        executor.for_each(0, tris.size(), [&] (size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i)
                accel.tris[i] = tris[accel.bvh.prim_ids[i]];
        });
    } else {
        executor.for_each(0, tris.size(), [&] (size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i)
                accel.tris[i] = tris[i];
        });
    }

    return accel;
}

struct Image {
    size_t width = 0;
    size_t height = 0;
    std::unique_ptr<uint8_t[]> data;

    Image() = default;
    Image(Image&&) = default;
    Image(const Image&) = delete;
    Image& operator = (Image&&) = default;

    Image(size_t width, size_t height)
        : width(width), height(height), data(std::make_unique<uint8_t[]>(width * height * 3))
    {}

    void save(const std::string& file_name) {
        std::ofstream out(file_name, std::ofstream::binary);
        out << "P6 " << width << " " << height << " " << 255 << "\n";
        for (size_t j = height; j > 0; --j)
            out.write(reinterpret_cast<char*>(data.get() + (j - 1) * 3 * width), sizeof(uint8_t) * 3 * width);
    }
};

struct TraversalStats {
    size_t visited_nodes = 0;
    size_t visited_leaves = 0;

    TraversalStats& operator += (const TraversalStats& other) {
        visited_nodes += other.visited_nodes;
        visited_leaves += other.visited_leaves;
        return *this;
    }
};

struct RenderStats {
    TraversalStats traversal_stats;
    size_t intersections = 0;
};

static constexpr size_t invalid_id = std::numeric_limits<size_t>::max();

template <bool UseRobustTraversal, bool PermutePrims, bool CaptureStats>
static size_t intersect_accel(Ray& ray, const Accel& accel, TraversalStats& stats) {
    size_t prim_id = invalid_id;
    bvh::v2::SmallStack<Bvh::Index, stack_size> stack;
    accel.bvh.intersect<false, UseRobustTraversal>(ray, accel.bvh.get_root().index, stack,
        [&] (size_t begin, size_t end) {
            if constexpr (CaptureStats)
                stats.visited_leaves++;
            for (size_t i = begin; i < end; ++i) {
                size_t j = PermutePrims ? i : accel.bvh.prim_ids[i];
                if (accel.tris[j].intersect(ray))
                    prim_id = j;
            }
            return prim_id != invalid_id;
        },
        [&] (auto&&, auto&&) {
            if constexpr (CaptureStats)
                stats.visited_nodes++;
        });
    return prim_id;
}

using IntersectAccelFn = size_t (*)(Ray&, const Accel&, TraversalStats&);
static IntersectAccelFn get_intersect_accel_fn(const Options& options) {
    static const IntersectAccelFn intersect_accel_fns[] = {
        intersect_accel<false, false, false>,
        intersect_accel<false, false, true>,
        intersect_accel<false, true, false>,
        intersect_accel<false, true, true>,
        intersect_accel<true, false, false>,
        intersect_accel<true, false, true>,
        intersect_accel<true, true, false>,
        intersect_accel<true, true, true>
    };
    return intersect_accel_fns[
        (options.robust_traversal ? 4 : 0) |
        (options.permute_prims ? 2 : 0) |
        (options.render_mode == RenderMode::Debug ? 1 : 0)];
}

std::tuple<uint8_t, uint8_t, uint8_t> intensity_to_color(Scalar k) {
    static const Vec3 g[] = {
        Vec3(0, 0, 255),
        Vec3(0, 255, 255),
        Vec3(0, 128, 0),
        Vec3(255, 255, 0),
        Vec3(255, 0, 0)
    };
    static constexpr size_t n = sizeof(g) / sizeof(g[0]);
    static constexpr auto s = Scalar{1} / static_cast<Scalar>(n);

    size_t i = std::min(n - 1, static_cast<size_t>(k * n));
    size_t j = std::min(n - 1, i + 1);

    auto t = (k - static_cast<Scalar>(i) * s) / s;
    auto c = (1.0f - t) * g[i] + t * g[j];
    return std::make_tuple(
        static_cast<uint8_t>(c[0]),
        static_cast<uint8_t>(c[1]),
        static_cast<uint8_t>(c[2]));
}

static Image render(const Accel& accel, RenderStats& render_stats, const Options& options) {
    auto intersect_accel = get_intersect_accel_fn(options);

    auto dir = normalize(options.dir);
    auto right = normalize(cross(dir, options.up));
    auto up = cross(right, dir);

    auto debug_data = options.render_mode == RenderMode::Debug
        ? std::make_unique<size_t[]>(options.width * options.height) : nullptr;

    Image image(options.width, options.height);
    for (size_t y = 0; y < options.height; ++y) {
        for (size_t x = 0; x < options.width; ++x) {
            auto u = Scalar{2} * static_cast<Scalar>(x) / static_cast<Scalar>(options.width) - Scalar{1};
            auto v = Scalar{2} * static_cast<Scalar>(y) / static_cast<Scalar>(options.height) - Scalar{1};
            auto pixel_id = y * options.width + x;

            TraversalStats traversal_stats;
            Ray ray(options.eye, dir + u * right + v * up);
            auto prim_id = intersect_accel(ray, accel, traversal_stats);
            if (prim_id != invalid_id)
                render_stats.intersections++;
            Scalar intensity = 0;
            if (options.render_mode == RenderMode::EyeLight) {
                if (prim_id != invalid_id)
                    intensity = std::abs(dot(normalize(accel.tris[prim_id].n), ray.dir));
                uint8_t pixel = static_cast<uint8_t>(
                    std::min(std::max(0, static_cast<int>(intensity * Scalar{256})), 255));
                image.data[3 * pixel_id + 0] = pixel;
                image.data[3 * pixel_id + 1] = pixel;
                image.data[3 * pixel_id + 2] = pixel;
            } else {
                debug_data[pixel_id] =
                    traversal_stats.visited_nodes + traversal_stats.visited_leaves;
                render_stats.traversal_stats += traversal_stats;
            }
        }
    }

    if (options.render_mode == RenderMode::Debug) {
        Scalar debug_threshold = static_cast<Scalar>(options.debug_threshold);
        if (debug_threshold == static_cast<Scalar>(0))
            debug_threshold = static_cast<Scalar>(
                *std::max_element(debug_data.get(), debug_data.get() + options.width * options.height));
        for (size_t i = 0; i < options.width * options.height; ++i) {
            std::tie(
                image.data[3 * i + 0],
                image.data[3 * i + 1],
                image.data[3 * i + 2]) = intensity_to_color(static_cast<Scalar>(debug_data[i]) / debug_threshold);
        }
    }

    return image;
}

int main(int argc, char** argv) {
    Options options;
    if (!options.parse(argc, argv))
        return 1;

    if (options.show_usage) {
        usage();
        return 0;
    }

    auto tris = load_obj<Scalar>(options.input_model);
    if (tris.empty()) {
        std::cerr << "No triangle was found in input OBJ file" << std::endl;
        return 1;
    }
    std::cout << "Loaded file with " << tris.size() << " triangle(s)" << std::endl;

    Accel accel;
    bvh::v2::ThreadPool thread_pool;
    auto build_time = profile([&] { accel = build_accel(thread_pool, tris, options); }, options.build_iters);
    std::cout
        << "Built BVH with " << accel.bvh.nodes.size() << " node(s) in "
        << to_ms(build_time) << "ms" << std::endl;

    RenderStats stats;
    Image image;
    auto intersection_time = profile([&] { image = render(accel, stats, options); });

    std::cout << stats.intersections << " intersection(s) found in " << to_ms(intersection_time)  << "ms\n";
    if (options.render_mode == RenderMode::Debug) {
        std::cout
            << "Traversal visited " << stats.traversal_stats.visited_nodes
            << " nodes and " << stats.traversal_stats.visited_leaves << " leaves" << std::endl;
    }

    image.save(options.output_image);
    std::cout << "Image saved as " << options.output_image << std::endl;
    return 0;
}
