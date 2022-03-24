#ifndef BVH_BVH_H
#define BVH_BVH_H

#include <climits>
#include <cassert>
#include <algorithm>
#include <type_traits>
#include <vector>
#include <numeric>

#include <proto/bbox.h>
#include <proto/utils.h>
#include <par/for_each.h>
#include <par/transform_reduce.h>

namespace bvh {

/// This structure represents a BVH with a list of nodes and primitives indices.
/// The root of the BVH is located at index 0, and the children of a node are packed
/// together in the array of nodes.
///
/// - For inner nodes, the first child of an inner node is located at `first_index`,
///   and the second one is located at `first_index + 1`.
/// - For leaf nodes, `first_index` is the index of the first primitive of the leaf.
///
template <typename T>
struct Bvh {
    using Scalar = T;
    using Index  = std::make_unsigned_t<proto::SizedIntegerType<sizeof(T) * CHAR_BIT>>;

    /// Default traversal cost used when none is supplied.
    /// This corresponds to a BVH holding primitives that have a ray intersection routine that is
    /// as expensive as a ray-box intersection routine.
    static constexpr auto default_traversal_cost = Scalar(1);

    struct Node {
        T bounds[6];
        Index prim_count;
        Index first_index;

        bool is_leaf() const { return prim_count != 0; }

        Scalar sah_cost(Scalar traversal_cost = default_traversal_cost) const {
            return bbox().half_area() * (is_leaf() ? static_cast<Scalar>(prim_count) : traversal_cost);
        }

        /// Accessor to simplify the manipulation of the bounding box of a node.
        /// This type is convertible to a `BBox`.
        struct BBoxProxy {
            Node& node;

            BBoxProxy(Node& node)
                : node(node)
            {}

            BBoxProxy& operator = (const proto::BBox<T>& bbox) {
                node.bounds[0] = bbox.min[0];
                node.bounds[1] = bbox.max[0];
                node.bounds[2] = bbox.min[1];
                node.bounds[3] = bbox.max[1];
                node.bounds[4] = bbox.min[2];
                node.bounds[5] = bbox.max[2];
                return *this;
            }

            operator proto::BBox<T> () const {
                return proto::BBox<T>(
                    proto::Vec3<T>(node.bounds[0], node.bounds[2], node.bounds[4]),
                    proto::Vec3<T>(node.bounds[1], node.bounds[3], node.bounds[5]));
            }

            proto::BBox<T> to_bbox() const {
                return static_cast<proto::BBox<T>>(*this);
            }

            BBoxProxy& extend(const proto::BBox<T>& bbox)   { return *this = to_bbox().extend(bbox); }
            BBoxProxy& extend(const proto::Vec3<T>& vector) { return *this = to_bbox().extend(vector); }
        };

        BBoxProxy bbox_proxy() { return BBoxProxy(*this); }
        const BBoxProxy bbox_proxy() const { return BBoxProxy(*const_cast<Node*>(this)); }

        proto::BBox<T> bbox() const { return bbox_proxy().to_bbox(); }
    };

    /// Nodes of the BVH. The root is located at index 0.
    std::vector<Node> nodes;

    /// Indices of the primitives contained in the leaves of the BVH.
    /// Each leaf covers a range of indices in that array, equal to:
    /// `[first_index, first_index + prim_count]`.
    std::vector<size_t> prim_indices;

    static bool is_left_child(size_t i)  { return i % 2 == 1; }
    static bool is_right_child(size_t i) { return !is_left_child(i); }
    static size_t left_child(size_t i)  { return is_left_child(i)  ? i : i - 1; }
    static size_t right_child(size_t i) { return is_right_child(i) ? i : i + 1; }
    static size_t sibling(size_t i) { return is_left_child(i) ? i + 1 : i - 1; }

    /// Evaluates the SAH cost of this BVH.
    /// The `traversal_cost` parameter represents the ratio of the cost
    /// of traversing a node over the cost of intersecting a primitive.
    template <typename Executor>
    Scalar sah_cost(Executor& executor, Scalar traversal_cost = default_traversal_cost) const {
        return par::transform_reduce(
            executor, par::range_1d(size_t{0}, nodes.size()), Scalar(0),
            std::plus<Scalar> {},
            [&] (size_t i) { return nodes[i].sah_cost(traversal_cost); }) / nodes[0].bbox().half_area();
    }

    /// Computes the parent-child indices.
    /// The parent of the root node (at index 0) is the root node itself, by convention.
    template <typename Executor>
    std::vector<size_t> parents(Executor& executor) const {
        std::vector<size_t> parents(nodes.size(), 0);
        par::for_each(executor, par::range_1d(size_t{0}, nodes.size()), [&] (size_t i) {
            if (!nodes[i].is_leaf()) {
                parents[nodes[i].first_index + 0] = i;
                parents[nodes[i].first_index + 1] = i;
            }
        });
        return parents;
    }
};

} // namespace bvh

#endif
