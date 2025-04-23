#include <bvh/v2/c_api/bvh.h>

#include "load_obj.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
#include <float.h>
#include <time.h>
#include <stdbool.h>
#include <assert.h>
#include <tgmath.h>

#define INVALID_TRI_ID UINT32_MAX

struct scene {
    struct bvh3f* bvh;
    struct tri* tris;
    size_t tri_count;
};

struct options {
    const char* input_scene;
    const char* output_image;
    uint32_t width, height;
    struct bvh_vec3f eye, dir, up;
};

struct image {
    uint8_t* pixels;
    uint32_t width, height;
};

struct hit {
    float t, u, v;
    uint32_t tri_id;
};

struct intersect_user_data {
    const struct bvh_ray3f* ray;
    const struct scene* scene;
    struct hit* hit;
};

static inline float dot(struct bvh_vec3f v, struct bvh_vec3f w) {
    return v.x * w.x + v.y * w.y + v.z * w.z;
}

static inline float length(struct bvh_vec3f v) {
    return dot(v, v);
}

static inline struct bvh_vec3f sub(struct bvh_vec3f v, struct bvh_vec3f w) {
    return (struct bvh_vec3f) {
        .x = v.x - w.x,
        .y = v.y - w.y,
        .z = v.z - w.z,
    };
}

static inline struct bvh_vec3f cross(struct bvh_vec3f v, struct bvh_vec3f w) {
    return (struct bvh_vec3f) {
        .x = v.y * w.z - v.z * w.y,
        .y = v.z * w.x - v.x * w.z,
        .z = v.x * w.y - v.y * w.x,
    };
}

static inline struct bvh_vec3f normalize(struct bvh_vec3f v) {
    const float inv_len = 1.f / length(v);
    return (struct bvh_vec3f) {
        .x = v.x * inv_len,
        .y = v.y * inv_len,
        .z = v.z * inv_len,
    };
}

static inline struct bvh_vec3f tri_center(const struct tri* tri) {
    return (struct bvh_vec3f) {
        .x = (tri->v[0].x + tri->v[1].x + tri->v[2].x) * (1.f / 3.f),
        .y = (tri->v[0].y + tri->v[1].y + tri->v[2].y) * (1.f / 3.f),
        .z = (tri->v[0].z + tri->v[1].z + tri->v[2].z) * (1.f / 3.f)
    };
}

static inline struct bvh_bbox3f tri_bbox(const struct tri* tri) {
    struct bvh_vec3f min = tri->v[0], max = min;
    for (int i = 1; i < 3; ++i) {
        min.x = min.x < tri->v[i].x ? min.x : tri->v[i].x;
        min.y = min.y < tri->v[i].y ? min.y : tri->v[i].y;
        min.z = min.z < tri->v[i].z ? min.z : tri->v[i].z;
        max.x = max.x > tri->v[i].x ? max.x : tri->v[i].x;
        max.y = max.y > tri->v[i].y ? max.y : tri->v[i].y;
        max.z = max.z > tri->v[i].z ? max.z : tri->v[i].z;
    }
    return (struct bvh_bbox3f) { .min = min, .max = max };
}

static inline void build_bvh(struct scene* scene) {
    assert(!scene->bvh);
    struct bvh_bbox3f* bboxes = malloc(sizeof(struct bvh_bbox3f) * scene->tri_count);
    struct bvh_vec3f* centers = malloc(sizeof(struct bvh_vec3f) * scene->tri_count);
    for (size_t i = 0; i < scene->tri_count; ++i) {
        bboxes[i]  = tri_bbox(&scene->tris[i]);
        centers[i] = tri_center(&scene->tris[i]);
    }

    struct bvh_thread_pool* thread_pool = bvh_thread_pool_create(0);
    struct timespec ts0, ts1;
    timespec_get(&ts0, TIME_UTC);
    scene->bvh = bvh3f_build(thread_pool, bboxes, centers, scene->tri_count, NULL);
    timespec_get(&ts1, TIME_UTC);
    bvh_thread_pool_destroy(thread_pool);
    free(bboxes);
    free(centers);
    const uint64_t build_time = (ts1.tv_sec - ts0.tv_sec) * 1000 + (ts1.tv_nsec - ts0.tv_nsec) / 1000000;
    printf("Built BVH with %zu node(s) in %"PRIu64"\n", bvh3f_get_node_count(scene->bvh), build_time);
}

static inline void destroy_scene(struct scene* scene) {
    bvh3f_destroy(scene->bvh);
    free(scene->tris);
    memset(scene, 0, sizeof(struct scene));
}

struct image alloc_image(size_t width, size_t height) {
    return (struct image) {
        .pixels = malloc(sizeof(uint8_t) * width * height * 3),
        .width = width,
        .height = height
    };
}

static void free_image(struct image* image) {
    free(image->pixels);
    memset(image, 0, sizeof(struct image));
}

static bool save_image(const struct image* image, const char* file_name) {
    FILE* file = fopen(file_name, "wb");
    if (!file)
        return false;
    fprintf(file, "P6 %"PRIu32" %"PRIu32" 255\n", image->width, image->height);
    for (uint32_t y = image->height; y-- > 0;)
        fwrite(image->pixels + y * 3 * image->width, sizeof(uint8_t), 3 * image->width, file);
    fclose(file);
    return true;
}

static void set_pixel(struct image* image, uint32_t x, uint32_t y, uint8_t r, uint8_t g, uint8_t b) {
    uint8_t* p = &image->pixels[(y * image->width + x) * 3];
    p[0] = r;
    p[1] = g;
    p[2] = b;
}

static void usage() {
    printf(
        "Usage: c_api_example [options] file.obj\n"
        "\nOptions:\n"
        "  -h  --help                  Shows this message.\n"
        "  -e  --eye <x> <y> <z>       Sets the position of the camera.\n"
        "  -d  --dir <x> <y> <z>       Sets the direction of the camera.\n"
        "  -u  --up  <x> <y> <z>       Sets the up vector of the camera.\n"
        "  -w  --width <pixels>        Sets the image width.\n"
        "  -h  --height <pixels>       Sets the image height.\n"
        "  -o <file.ppm>               Sets the output file name (defaults to 'render.ppm').\n");
}

static inline bool check_arg(int i, int n, int argc, char** argv) {
    if (i + n > argc) {
        fprintf(stderr, "Missing argument(s) for '%s'\n", argv[i]);
        return false;
    }
    return true;
}

static inline bool parse_options(int argc, char** argv, struct options* options) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') {
            if (!strcmp(argv[i], "-e") || !strcmp(argv[i], "--eye")) {
                if (!check_arg(i, 3, argc, argv))
                    return false;
                options->eye.x = strtof(argv[++i], NULL);
                options->eye.y = strtof(argv[++i], NULL);
                options->eye.z = strtof(argv[++i], NULL);
            } else if (!strcmp(argv[i], "-d") || !strcmp(argv[i], "--dir")) {
                if (!check_arg(i, 3, argc, argv))
                    return false;
                options->dir.x = strtof(argv[++i], NULL);
                options->dir.y = strtof(argv[++i], NULL);
                options->dir.z = strtof(argv[++i], NULL);
            } else if (!strcmp(argv[i], "-u") || !strcmp(argv[i], "--up")) {
                if (!check_arg(i, 3, argc, argv))
                    return false;
                options->up.x = strtof(argv[++i], NULL);
                options->up.y = strtof(argv[++i], NULL);
                options->up.z = strtof(argv[++i], NULL);
            } else if (!strcmp(argv[i], "-w") || !strcmp(argv[i], "--width")) {
                if (!check_arg(i, 1, argc, argv))
                    return false;
                options->width = strtoul(argv[++i], NULL, 10);
            } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--height")) {
                if (!check_arg(i, 1, argc, argv))
                    return false;
                options->height = strtoul(argv[++i], NULL, 10);
            } else if (!strcmp(argv[i], "-o")) {
                if (!check_arg(i, 1, argc, argv))
                    return false;
                options->output_image = argv[++i];
            } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
                usage();
                return false;
            }
        } else {
            if (options->input_scene) {
                fprintf(stderr, "Input scene file specified twice\n");
                return false;
            }
            options->input_scene = argv[i];
        }
    }
    if (!options->input_scene) {
        fprintf(stderr, "Missing input scene file\n");
        return false;
    }
    return true;
}

static inline bool intersect_ray_tri(const struct bvh_ray3f* ray, const struct tri* tri, struct hit* hit) {
    const struct bvh_vec3f e1 = sub(tri->v[0], tri->v[1]);
    const struct bvh_vec3f e2 = sub(tri->v[2], tri->v[0]);
    const struct bvh_vec3f n = cross(e1, e2);
    const struct bvh_vec3f c = sub(tri->v[0], ray->org);
    const struct bvh_vec3f r = cross(ray->dir, c);

    const float inv_det = 1.f / dot(n, ray->dir);

    const float u = dot(r, e2) * inv_det;
    const float v = dot(r, e1) * inv_det;
    const float w = 1.f - u - v;

    static const float tolerance = -FLT_EPSILON;
    if (u >= tolerance && v >= tolerance && w >= tolerance) {
        const float t = dot(n, c) * inv_det;
        if (t >= ray->tmin && t <= hit->t) {
            hit->t = t;
            hit->u = u;
            hit->v = v;
            return true;
        }
    }
    return false;
}

static bool intersect_bvh_leaf(void* user_data, float* t, size_t begin, size_t end) {
    const struct bvh_ray3f* ray = ((struct intersect_user_data*)user_data)->ray;
    const struct scene* scene = ((struct intersect_user_data*)user_data)->scene;
    struct hit* hit = ((struct intersect_user_data*)user_data)->hit;
    bool was_hit = false;
    for (size_t i = begin; i < end; ++i) {
        const uint32_t tri_id = bvh3f_get_prim_id(scene->bvh, i);
        if (intersect_ray_tri(ray, &scene->tris[tri_id], hit)) {
            *t = hit->t;
            hit->tri_id = tri_id;
            was_hit = true;
        }
    }
    return was_hit;
}

static inline void render_image(
    const struct scene* scene,
    const struct options* options,
    struct image* image)
{
    const struct bvh_vec3f dir   = normalize(options->dir);
    const struct bvh_vec3f right = normalize(cross(dir, options->up));
    const struct bvh_vec3f up    = cross(right, dir);

    struct intersect_user_data user_data = { .scene = scene };
    const struct bvh_intersect_callbackf callback = {
        .user_data = &user_data,
        .user_fn = intersect_bvh_leaf 
    };

    size_t intr_count = 0;
    struct timespec ts0, ts1;
    timespec_get(&ts0, TIME_UTC);
    for (uint32_t y = 0; y < options->height; ++y) {
        for (uint32_t x = 0; x < options->width; ++x) {
            const float u = 2.f * ((float)x) / ((float)options->width)  - 1.f;
            const float v = 2.f * ((float)y) / ((float)options->height) - 1.f;

            const struct bvh_ray3f ray = {
                .org = options->eye,
                .dir = {
                    .x = options->dir.x + u * right.x + v * up.x,
                    .y = options->dir.y + u * right.y + v * up.y,
                    .z = options->dir.z + u * right.z + v * up.z
                },
                .tmin = 0.f,
                .tmax = FLT_MAX
            };

            struct hit hit = { .tri_id = INVALID_TRI_ID, .t = FLT_MAX };

            user_data.ray = &ray;
            user_data.hit = &hit;
            bvh3f_intersect_ray(scene->bvh, &ray, &callback);

            uint8_t r = 0;
            uint8_t g = 0;
            uint8_t b = 0;
            if (hit.tri_id != INVALID_TRI_ID) {
                r = (hit.tri_id * 79) % 255 + 1;
                g = (hit.tri_id * 43) % 255 + 1;
                b = (hit.tri_id * 57) % 255 + 1;
                intr_count++;
            }
            set_pixel(image, x, y, r, g, b);
        }
    }
    timespec_get(&ts1, TIME_UTC);
    const uint64_t render_time = (ts1.tv_sec - ts0.tv_sec) * 1000 + (ts1.tv_nsec - ts0.tv_nsec) / 1000000;
    printf("%zu intersection(s) found in %"PRIu64"\n", intr_count, render_time);
}

int main(int argc, char** argv) {
    struct options options = {
        .eye = (struct bvh_vec3f) { 0, 0, 0 },
        .dir = (struct bvh_vec3f) { 0, 0, 1 },
        .up  = (struct bvh_vec3f) { 0, 1, 0 },
        .output_image = "render.ppm",
        .width  = 1024,
        .height = 1024
    };
    if (!parse_options(argc, argv, &options))
        return 1;

    struct scene scene = { 0 };
    scene.tris = load_obj(argv[1], &scene.tri_count);
    if (!scene.tris) {
        fprintf(stderr, "Invalid or empty OBJ file\n");
        return 1;
    }
    printf("Loaded file with %zu triangle(s)\n", scene.tri_count);

    build_bvh(&scene);

    struct image image = alloc_image(options.width, options.height);
    render_image(&scene, &options, &image);
    if (!save_image(&image, options.output_image)) {
        fprintf(stderr, "Could not save rendered image to '%s'\n", options.output_image);
        return 1;
    } else {
        printf("Image saved as '%s'\n", options.output_image);
    }

    free_image(&image);
    destroy_scene(&scene);
    return 0;
}
