#ifndef BVH_TOPOLOGY_MODIFIER_H
#define BVH_TOPOLOGY_MODIFIER_H

#include <limits>
#include <vector>
#include <cassert>
#include <algorithm>

#include "bvh/bvh.h"

namespace bvh {

/// Helper class that can be used to implement various algorithms that modify the
/// topology of a BVH. It automatically takes care of handling parent-child indices,
/// and updates them when removing or inserting nodes.
/// It also keeps around a free list of nodes, which is reused when inserting
/// new ones.
/// Note that the methods of this class cannot be called in parallel.
template <typename Bvh>
class TopologyModifier {
    using Index = typename Bvh::Index;
    using Node  = typename Bvh::Node;

public:
    Bvh& bvh;

    /// Array containing the index of the parent node of every node in the BVH.
    std::vector<size_t> parents;

    /// Nodes that are free to be used during insertion are placed in this list.
    /// Each index in this list corresponds to two nodes, starting at this index.
    std::vector<size_t> free_list;

    TopologyModifier(Bvh& bvh, std::vector<size_t>&& parents)
        : bvh(bvh), parents(std::move(parents))
    {}

    /// Removes the node at the given index, and places the two children in the free list.
    void remove_node(size_t index) {
        assert(index != 0);
        auto sibling = Bvh::sibling(index);
        auto parent  = parents[index];
        bvh.nodes[parent] = bvh.nodes[sibling];
        parents[index]   = 0;
        parents[sibling] = 0;
        if (!bvh.nodes[sibling].is_leaf()) {
            parents[bvh.nodes[sibling].first_index + 0] = parent;
            parents[bvh.nodes[sibling].first_index + 1] = parent;
        }
        free_list.push_back(Bvh::left_child(index));
        refit_from(parents[parent]);
    }

    /// Inserts the given node in the BVH, by making it the child of the node located at `target`.
    /// This requires to have at least two nodes in the free list (this can be done with
    /// a call to `remove_node()`).
    void insert_node(const Node& node, size_t target) {
        assert(!free_list.empty());
        auto old_node = bvh.nodes[target];
        auto first_child = free_list.back();
        free_list.pop_back();
        bvh.nodes[target].first_index = static_cast<Index>(first_child);
        bvh.nodes[target].prim_count  = 0;
        bvh.nodes[first_child + 0] = node;
        bvh.nodes[first_child + 1] = old_node;
        if (!node.is_leaf()) {
            parents[node.first_index + 0] = first_child + 0;
            parents[node.first_index + 1] = first_child + 0;
        }
        if (!old_node.is_leaf()) {
            parents[old_node.first_index + 0] = first_child + 1;
            parents[old_node.first_index + 1] = first_child + 1;
        }
        parents[first_child + 0] = target;
        parents[first_child + 1] = target;
        refit_from(target);
    }

private:
    void refit_from(size_t index) {
        do {
            auto& node = bvh.nodes[index];
            assert(!node.is_leaf());
            node.bbox_proxy() =
                bvh.nodes[node.first_index + 0].bbox().extend(
                bvh.nodes[node.first_index + 1].bbox());
            index = parents[index];
        } while (index != 0);
    }
};

} // namespace bvh

#endif
