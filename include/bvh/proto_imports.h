#ifndef BVH_PROTO_IMPORTS_H
#define BVH_PROTO_IMPORTS_H

#include <proto/vec.h>
#include <proto/ray.h>
#include <proto/triangle.h>
#include <proto/sphere.h>

namespace bvh {

template <typename T> using Ray      = proto::Ray<T>;
template <typename T> using Vec3     = proto::Vec3<T>;
template <typename T> using BBox     = proto::BBox<T>;
template <typename T> using Triangle = proto::Triangle<T>;
template <typename T> using Sphere   = proto::Sphere<T>;

using proto::normalize;
using proto::cross;
using proto::dot;
using proto::length;

} // namespace bvh

#endif
