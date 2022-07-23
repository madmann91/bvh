#ifndef BVH_V2_NODE_H
#define BVH_V2_NODE_H

#include "bvh/v2/utils.h"
#include "bvh/v2/vec.h"
#include "bvh/v2/bbox.h"
#include "bvh/v2/ray.h"

#include <cassert>
#include <limits>

namespace bvh::v2 {

template <
    typename T,
    size_t Dim,
    size_t IndexBits = sizeof(T) * CHAR_BIT,
    size_t PrimCountBits = 4>
struct Node {
    using Scalar = T;
    static constexpr size_t dimension = Dim;
    static constexpr size_t prim_count_bits = PrimCountBits;
    static constexpr size_t index_bits = IndexBits;
    static constexpr size_t max_prim_count = make_bitmask<size_t>(prim_count_bits);

    T bounds[Dim * 2];
    struct Index {
        using Type = UnsignedIntType<IndexBits>;
        Type first_id   : std::numeric_limits<Type>::digits - prim_count_bits;
        Type prim_count : prim_count_bits;
    } index;

    static_assert(sizeof(Index) == sizeof(typename Index::Type));

    Node() = default;

    BVH_ALWAYS_INLINE bool is_leaf() const { return index.prim_count != 0; }

    BVH_ALWAYS_INLINE void make_leaf(size_t first_prim, size_t prim_count) {
        assert(prim_count != 0);
        assert(prim_count <= max_prim_count);
        index.prim_count = prim_count;
        index.first_id = first_prim;
    }

    BVH_ALWAYS_INLINE void make_inner(size_t first_child) {
        index.prim_count = 0;
        index.first_id = first_child;
    }

    BVH_ALWAYS_INLINE BBox<T, Dim> get_bbox() const {
        return BBox<T, Dim>(
            Vec<T, Dim>::generate([&] (size_t i) { return bounds[i * 2]; }),
            Vec<T, Dim>::generate([&] (size_t i) { return bounds[i * 2 + 1]; }));
    }

    BVH_ALWAYS_INLINE void set_bbox(const BBox<T, Dim>& bbox) {
        static_for<0, Dim>([&] (size_t i) {
            bounds[i * 2 + 0] = bbox.min[i];
            bounds[i * 2 + 1] = bbox.max[i];
        });
    }

    BVH_ALWAYS_INLINE Vec<T, Dim> get_min_bounds(const Octant& octant) const {
        return Vec<T, Dim>::generate([&] (size_t i) { return bounds[2 * i + octant[i]]; });
    }

    BVH_ALWAYS_INLINE Vec<T, Dim> get_max_bounds(const Octant& octant) const {
        return Vec<T, Dim>::generate([&] (size_t i) { return bounds[2 * i + 1 - octant[i]]; });
    }

    /// Robust ray-node intersection routine. See "Robust BVH Ray Traversal", by T. Ize.
    BVH_ALWAYS_INLINE std::pair<T, T> intersect_robust(
        const Ray<T, Dim>& ray,
        const Vec<T, Dim>& inv_dir,
        const Vec<T, Dim>& inv_dir_pad,
        const Octant& octant) const
    {
        auto tmin = (get_min_bounds(octant) - ray.org) * inv_dir;
        auto tmax = (get_max_bounds(octant) - ray.org) * inv_dir_pad;
        return make_intersection_result(ray, tmin, tmax);
    }

    BVH_ALWAYS_INLINE std::pair<T, T> intersect_fast(
        const Ray<T, Dim>& ray,
        const Vec<T, Dim>& inv_dir,
        const Vec<T, Dim>& inv_org,
        const Octant& octant) const
    {
        auto tmin = fast_mul_add(get_min_bounds(octant), inv_dir, inv_org);
        auto tmax = fast_mul_add(get_max_bounds(octant), inv_dir, inv_org);
        return make_intersection_result(ray, tmin, tmax);
    }

private:
    BVH_ALWAYS_INLINE static std::pair<T, T> make_intersection_result(
        const Ray<T, Dim>& ray,
        const Vec<T, Dim>& tmin,
        const Vec<T, Dim>& tmax)
    {
        auto t0 = ray.tmin;
        auto t1 = ray.tmax;
        static_for<0, Dim>([&] (size_t i) {
            t0 = robust_max(tmin[i], t0);
            t1 = robust_min(tmax[i], t1);
        });
        return std::pair<T, T> { t0, t1 };
    }
};

} // namespace bvh::v2

#endif
