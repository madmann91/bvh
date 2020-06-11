#ifndef BVH_FWD_HPP
#define BVH_FWD_HPP

#include <cstddef>

namespace bvh {

template <typename Scalar> struct Bvh;
template <typename Scalar> struct Ray;
template <typename Scalar> struct BoundingBox;

template <typename Scalar, size_t> struct Vector;

template <typename Scalar>
using Vector3 = Vector<Scalar, 3>;

} // namespace bvh

#endif
