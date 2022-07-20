#ifndef LOAD_OBJ_H
#define LOAD_OBJ_H

#include <bvh/v2/tri.h>

#include <vector>
#include <string>

template <typename T>
std::vector<bvh::v2::Tri<T, 3>> load_obj(const std::string& file);

#endif
