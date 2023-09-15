#ifndef BVH_V2_BVH_H
#define BVH_V2_BVH_H

#include "bvh/v2/node.h"

#include <cstddef>
#include <iterator>
#include <vector>
#include <stack>
#include <utility>

namespace bvh::v2 {

template <typename Node>
struct Bvh {
    using Index = typename Node::Index;
    using Scalar = typename Node::Scalar;

    std::vector<Node> nodes;
    std::vector<size_t> prim_ids;

    Bvh() = default;
    Bvh(Bvh&&) = default;

    Bvh& operator = (Bvh&&) = default;

    bool operator == (const Bvh& other) const = default;
    bool operator != (const Bvh& other) const = default;

    /// Returns the root node of this BVH.
    const Node& get_root() const { return nodes[0]; }

    /// Extracts the BVH rooted at the given node index.
    inline Bvh extract_bvh(size_t root_id) const;

    /// Intersects the BVH with a single ray, using the given function to intersect the contents
    /// of a leaf. The algorithm starts at the node index `top` and uses the given stack object.
    /// When `IsAnyHit` is true, the function stops at the first intersection (useful for shadow
    /// rays), otherwise it finds the closest intersection. When `IsRobust` is true, a slower but
    /// numerically robust ray-box test is used, otherwise a fast, but less precise test is used.
    template <bool IsAnyHit, bool IsRobust, typename Stack, typename LeafFn, typename InnerFn = IgnoreArgs>
    inline void intersect(Ray<Scalar, Node::dimension>& ray, Index top, Stack&, LeafFn&&, InnerFn&& = {}) const;

    inline void serialize(OutputStream&) const;
    static inline Bvh deserialize(InputStream&);
};

template <typename Node>
auto Bvh<Node>::extract_bvh(size_t root_id) const -> Bvh {
    assert(root_id != 0);

    Bvh bvh;
    bvh.nodes.emplace_back();

    std::stack<std::pair<size_t, size_t>> stack;
    stack.emplace(root_id, 0);
    while (!stack.empty()) {
        auto [src_id, dst_id] = stack.top();
        stack.pop();
        auto& src_node = nodes[src_id];
        auto& dst_node = bvh.nodes[dst_id];
        dst_node = src_node;
        if (src_node.is_leaf()) {
            dst_node.index.first_id = static_cast<typename Index::Type>(bvh.prim_ids.size());
            std::copy_n(
                prim_ids.begin() + src_node.index.first_id,
                src_node.index.prim_count,
                std::back_inserter(bvh.prim_ids));
        } else {
            size_t first_id = bvh.nodes.size();
            dst_node.index.first_id = static_cast<typename Index::Type>(first_id);
            bvh.nodes.emplace_back();
            bvh.nodes.emplace_back();
            stack.emplace(src_node.index.first_id + 0, first_id + 0);
            stack.emplace(src_node.index.first_id + 1, first_id + 1);
        }
    }
    return bvh;
}

template <typename Node>
template <bool IsAnyHit, bool IsRobust, typename Stack, typename LeafFn, typename InnerFn>
void Bvh<Node>::intersect(Ray<Scalar, Node::dimension>& ray, Index start, Stack& stack, LeafFn&& leaf_fn, InnerFn&& inner_fn) const {
    auto inv_dir = ray.template get_inv_dir<!IsRobust>();
    auto inv_org = -inv_dir * ray.org;
    auto inv_dir_pad = Ray<Scalar, Node::dimension>::pad_inv_dir(inv_dir);
    auto octant = ray.get_octant();

    auto intersect_node = [&] (const Node& node) {
        return IsRobust
            ? node.intersect_robust(ray, inv_dir, inv_dir_pad, octant)
            : node.intersect_fast(ray, inv_dir, inv_org, octant);
    };

    stack.push(start);
    while (!stack.is_empty()) {
        auto top = stack.pop();
        while (top.prim_count == 0) {
            auto& left  = nodes[top.first_id];
            auto& right = nodes[top.first_id + 1];

            inner_fn(left, right);

            auto intr_left  = intersect_node(left);
            auto intr_right = intersect_node(right);

            bool hit_left  = intr_left.first <= intr_left.second;
            bool hit_right = intr_right.first <= intr_right.second;

            if (hit_left) {
                auto near = left.index;
                if (hit_right) {
                    auto far = right.index;
                    if (!IsAnyHit && intr_left.first > intr_right.first)
                        std::swap(near, far);
                    stack.push(far);
                }
                top = near;
            } else if (hit_right)
                top = right.index;
            else [[unlikely]]
                break;
        }

        [[maybe_unused]] auto was_hit = leaf_fn(top.first_id, top.first_id + top.prim_count);
        if constexpr (IsAnyHit) {
            if (was_hit) return;
        }
    }
}

template <typename Node>
void Bvh<Node>::serialize(OutputStream& stream) const {
    stream.write(nodes.size());
    stream.write(prim_ids.size());
    for (auto&& node : nodes)
        node.serialize(stream);
    for (auto&& prim_id : prim_ids)
        stream.write(prim_id);
}

template <typename Node>
Bvh<Node> Bvh<Node>::deserialize(InputStream& stream) {
    Bvh bvh;
    bvh.nodes.resize(stream.read<size_t>());
    bvh.prim_ids.resize(stream.read<size_t>());
    for (auto& node : bvh.nodes)
        node = Node::deserialize(stream);
    for (auto& prim_id : bvh.prim_ids)
        prim_id = stream.read<size_t>();
    return bvh;
}

} // namespace bvh::v2

#endif
