#ifndef BVH_TRAVERSERS_SINGLE_RAY_TRAVERSER_H
#define BVH_TRAVERSERS_SINGLE_RAY_TRAVERSER_H

#include <optional>
#include <cassert>
#include <type_traits>
#include <climits>

#include <proto/ray.h>
#include <proto/vec.h>
#include <proto/utils.h>

namespace bvh {

/// Single ray traversal algorithm. Can be configured to be fully robust, or
/// only partially robust. The fully robust variant is slightly slower, but is
/// guaranteed not to produce false misses.
template <typename Bvh>
class SingleRayTraverser {
    using Scalar = typename Bvh::Scalar;
    using Node   = typename Bvh::Node;
    using Index  = typename Bvh::Index;
    using Vec3   = proto::Vec3<Scalar>;
    using Ray    = proto::Ray<Scalar>;
    using Octant = typename Ray::Octant;

    template <typename LeafIntersector>
    using Hit = std::invoke_result_t<LeafIntersector, Ray&, const Node&>;

    template <size_t Capacity>
    struct Stack {
        Index elems[Capacity];
        size_t size = 0;

        bool is_empty() const { return size == 0; }
        bool is_full() const { return size == Capacity; }
        void push(Index i) {
            assert(!is_full());
            elems[size++] = i;
        }
        Index pop() {
            assert(!is_empty());
            return elems[--size];
        }
    };

public:
    /// Default stack size used in `traverse()`.
    /// Should be enough for most uses.
    static constexpr size_t default_stack_size = 64;

    /// CRTP-based mixin to create node intersectors.
    template <typename Derived>
    struct NodeIntersector {
        Octant octant;

        NodeIntersector(const Ray& ray)
            : octant(ray.octant())
        {}

        proto_always_inline Scalar intersect_axis_min(int axis, Scalar p, const Ray& ray) const {
            return static_cast<const Derived*>(this)->intersect_axis_min(axis, p, ray);
        }

        proto_always_inline Scalar intersect_axis_max(int axis, Scalar p, const Ray& ray) const {
            return static_cast<const Derived*>(this)->intersect_axis_max(axis, p, ray);
        }

        proto_always_inline
        std::pair<Scalar, Scalar> intersect_node(const Ray& ray, const Node& node) const {
            Vec3 entry, exit;
            entry[0] = intersect_axis_min(0, node.bounds[0 * 2 +     octant[0]], ray);
            entry[1] = intersect_axis_min(1, node.bounds[1 * 2 +     octant[1]], ray);
            entry[2] = intersect_axis_min(2, node.bounds[2 * 2 +     octant[2]], ray);
            exit [0] = intersect_axis_max(0, node.bounds[0 * 2 + 1 - octant[0]], ray);
            exit [1] = intersect_axis_max(1, node.bounds[1 * 2 + 1 - octant[1]], ray);
            exit [2] = intersect_axis_max(2, node.bounds[2 * 2 + 1 - octant[2]], ray);
            // Note: This order for the min/max operations is guaranteed not to produce NaNs
            return std::make_pair(
                proto::robust_max(entry[0], proto::robust_max(entry[1], proto::robust_max(entry[2], ray.tmin))),
                proto::robust_min(exit [0], proto::robust_min(exit [1], proto::robust_min(exit [2], ray.tmax))));
        }

    protected:
        ~NodeIntersector() {}
    };

    /// Fully robust ray-node intersection algorithm (see "Robust BVH Ray Traversal", by T. Ize).
    struct RobustNodeIntersector : NodeIntersector<RobustNodeIntersector> {
        // Padded inverse direction to avoid false-negatives in the ray-node test.
        Vec3 pad_inv_dir;
        Vec3 inv_dir;

        RobustNodeIntersector(const Ray& ray)
            : NodeIntersector<RobustNodeIntersector>(ray)
        {
            inv_dir = Scalar(1) / ray.dir;
            pad_inv_dir = Vec3(
                proto::add_ulp_magnitude(inv_dir[0], 2),
                proto::add_ulp_magnitude(inv_dir[1], 2),
                proto::add_ulp_magnitude(inv_dir[2], 2));
        }

        proto_always_inline Scalar intersect_axis_min(int axis, Scalar p, const Ray& ray) const {
            return (p - ray.org[axis]) * inv_dir[axis];
        }

        proto_always_inline Scalar intersect_axis_max(int axis, Scalar p, const Ray& ray) const {
            return (p - ray.org[axis]) * pad_inv_dir[axis];
        }

        using NodeIntersector<RobustNodeIntersector>::intersect_node;
    };

    /// Semi-robust, fast ray-node intersection algorithm.
    struct FastNodeIntersector : public NodeIntersector<FastNodeIntersector> {
        Vec3 scaled_org;
        Vec3 inv_dir;

        FastNodeIntersector(const Ray& ray)
            : NodeIntersector<FastNodeIntersector>(ray)
        {
            inv_dir    = safe_inverse(ray.dir);
            scaled_org = -ray.org * inv_dir;
        }

        proto_always_inline Scalar intersect_axis_min(int axis, Scalar p, const Ray&) const {
            return proto::fast_mul_add(p, inv_dir[axis], scaled_org[axis]);
        }

        proto_always_inline Scalar intersect_axis_max(int axis, Scalar p, const Ray& ray) const {
            return intersect_axis_min(axis, p, ray);
        }

        using NodeIntersector<FastNodeIntersector>::intersect_node;
    };

    /// Traverses the BVH with a given ray.
    template <
        bool AnyHit,
        typename LeafIntersector,
        typename NodeVisitor,
        typename NodeIntersector = FastNodeIntersector,
        size_t StackSize = default_stack_size>
    static proto_always_inline Hit<LeafIntersector> traverse(
        Ray& ray,
        const Bvh& bvh,
        LeafIntersector&& leaf_intersector,
        NodeVisitor&& node_visitor)
    {
        auto leaf_mask = Index(1) << (sizeof(Index) * CHAR_BIT - 1);
        auto stack_value_from_node = [&] (Index index) {
            return bvh.nodes[index].is_leaf() ? ~index : bvh.nodes[index].first_index;
        };

        NodeIntersector node_intersector(ray);
        Stack<StackSize> stack;
        Hit<LeafIntersector> hit;
        Index top = stack_value_from_node(0);

        node_visitor(bvh.nodes[0]);
        while (true) {
            if (proto_unlikely(top & leaf_mask)) {
                auto& leaf = bvh.nodes[~top];
                node_visitor(leaf);
                if (auto leaf_hit = leaf_intersector(ray, leaf)) {
                    hit = leaf_hit;
                    if constexpr (AnyHit)
                        break;
                }
            } else {
                node_visitor(bvh.nodes[top + 0]);
                node_visitor(bvh.nodes[top + 1]);
                auto t_left  = node_intersector.intersect_node(ray, bvh.nodes[top + 0]);
                auto t_right = node_intersector.intersect_node(ray, bvh.nodes[top + 1]);
                bool hit_left  = t_left .first <= t_left .second;
                bool hit_right = t_right.first <= t_right.second;
                if (hit_left) {
                    auto first = stack_value_from_node(top + 0);
                    if (hit_right) {
                        auto second = stack_value_from_node(top + 1);
                        if (t_left.first > t_right.first)
                            std::swap(first, second);
                        stack.push(second);
                    }
                    top = first;
                    continue;
                } else if (hit_right) {
                    top = stack_value_from_node(top + 1);
                    continue;
                }
            }
            if (proto_unlikely(stack.is_empty()))
                break;
            top = stack.pop();
        }
        return hit;
    }

    template <
        bool AnyHit,
        typename LeafIntersector,
        typename NodeIntersector = FastNodeIntersector,
        size_t StackSize = default_stack_size>
    static proto_always_inline Hit<LeafIntersector> traverse(Ray& ray, const Bvh& bvh, LeafIntersector&& leaf_intersector) {
        return traverse<AnyHit>(ray, bvh, leaf_intersector, [] (const Node&) {});
    }
};

} // namespace bvh

#endif
