#ifndef LOAD_OBJ_H
#define LOAD_OBJ_H

#include <bvh/v2/c_api/bvh.h>

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct tri { struct bvh_vec3f v[3]; };
struct tri* load_obj(const char*, size_t*);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <bvh/v2/tri.h>

#include <vector>
#include <string>

template <typename T>
std::vector<bvh::v2::Tri<T, 3>> load_obj(const std::string& file);
#endif

#endif
