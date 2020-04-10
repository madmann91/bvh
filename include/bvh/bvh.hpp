#ifndef BVH_BVH_HPP
#define BVH_BVH_HPP

#include <climits>
#include <memory>

#include "bvh/bounding_box.hpp"
#include "bvh/utilities.hpp"

namespace bvh {

/// This structure represents a BVH with a list of nodes and primitives indices.
template <typename Scalar, size_t MaxDepth = 64> 
struct Bvh {
    using IndexType  = typename SizedIntegerType<sizeof(Scalar) * CHAR_BIT>::Unsigned;
    using ScalarType = Scalar;

    static constexpr size_t max_depth = MaxDepth;

    // The size of this structure should be 32 bytes in
    // single precision and 64 bytes in double precision.
    struct Node {
        Scalar bounds[6];
        bool is_leaf : 1;
        IndexType primitive_count : sizeof(IndexType) * CHAR_BIT - 1;
        IndexType first_child_or_primitive;

        /// Accessor to simplify the manipulation of the bounding box of a node.
        /// This type is convertible to a `BoundingBox`.
        struct BoundingBoxProxy {
            Node& node;

            BoundingBoxProxy(Node& node)
                : node(node)
            {}

            BoundingBoxProxy& operator = (const BoundingBox<Scalar>& bbox) {
                node.bounds[0] = bbox.min[0];
                node.bounds[1] = bbox.min[1];
                node.bounds[2] = bbox.min[2];
                node.bounds[3] = bbox.max[0];
                node.bounds[4] = bbox.max[1];
                node.bounds[5] = bbox.max[2];
                return *this;
            }

            operator BoundingBox<Scalar> () const {
                return BoundingBox<Scalar>(
                    Vector3<Scalar>(node.bounds[0], node.bounds[1], node.bounds[2]),
                    Vector3<Scalar>(node.bounds[3], node.bounds[4], node.bounds[5]));
            }

            BoundingBox<Scalar> to_bounding_box() const {
                return static_cast<BoundingBox<Scalar>>(*this);
            }

            Scalar half_area() const { return to_bounding_box().half_area(); }

            BoundingBoxProxy& extend(const BoundingBox<Scalar>& bbox) {
                return *this = to_bounding_box().extend(bbox);
            }

            BoundingBoxProxy& extend(const Vector3<Scalar>& vector) {
                return *this = to_bounding_box().extend(vector);
            }
        };

        BoundingBoxProxy bounding_box_proxy() {
            return BoundingBoxProxy(*this);
        }

        const BoundingBoxProxy bounding_box_proxy() const {
            return BoundingBoxProxy(const_cast<Node>(*this));
        }
    };

    std::unique_ptr<Node[]>   nodes;
    std::unique_ptr<size_t[]> primitive_indices;

    size_t node_count = 0;

    /// Cost of intersecting a ray with a node of the data structure.
    /// This cost is relative to the cost of intersecting a primitive,
    /// which is assumed to be equal to 1.
    Scalar traversal_cost = 1;
};

} // namespace bvh

#endif
