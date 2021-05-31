#ifndef BVH_TOPOLOGY_MODIFIER_H
#define BVH_TOPOLOGY_MODIFIER_H

#include <limits>
#include <vector>
#include <cassert>

namespace bvh {

/// Helper class that can be used to implement various algorithms that modify the
/// topology of a BVH. It automatically takes care of handling parent-child indices,
/// and updates them when removing or inserting nodes.
/// It also keeps around a free list of nodes, which is reused when inserting
/// new ones.
/// Note that the methods of this class cannot be called in parallel.
template <typename Bvh>
class TopologyModifier {
    using Node = typename Bvh::Node;

public:
    TopologyModifier(Bvh& bvh)
        : bvh_(bvh), parents_(bvh.nodes.size())
    {
        recompute_parents();
    }

    /// Recomputes the parent-child indices, if the topology of the BVH has been
    /// changed through some other means that this object.
    void recompute_parents() {
        std::fill(parents_.begin(), parents_.end(), 0);
        for (size_t i = 0; i < bvh_.nodes.size(); ++i) {
            if (!bvh_.nodes[i].is_leaf()) {
                parents_[bvh_.nodes[i].first_index + 0] = i;
                parents_[bvh_.nodes[i].first_index + 1] = i;
            }
        }
    }

    /// Removes the node at the given index, and places the two children in the free list.
    void remove_node(size_t index) {
        assert(index != 0);
        auto sibling = Bvh::sibling(index);
        auto parent  = parents_[index];
        bvh_.nodes[parent] = bvh_.nodes[sibling];
        parents_[index]   = 0;
        parents_[sibling] = 0;
        if (!bvh_.nodes[sibling].is_leaf()) {
            parents_[bvh_.nodes[sibling].first_index + 0] = parent;
            parents_[bvh_.nodes[sibling].first_index + 1] = parent;
        }
        mark_as_free(Bvh::left_child(index));
        refit_from(parents_[parent]);
    }

    /// Adds the two nodes at `index` and `index + 1` to the free list, so that they can
    /// be used when inserting other nodes.
    void mark_as_free(size_t index) {
        free_list_.push_back(index);
    }

    /// Inserts the given node in the BVH, by making it the child of the node located at `target`.
    /// This requires to have at least two nodes in the free list (this can be done either
    /// by a call to `remove_node()` or `mark_as_free()`).
    void insert_node(const Node& node, size_t target) {
        assert(!free_list_.empty());
        auto old_node = bvh_.nodes[target];
        auto first_child = free_list_.back();
        free_list_.pop_back();
        bvh_.nodes[target].first_index = first_child;
        bvh_.nodes[target].prim_count  = 0;
        bvh_.nodes[first_child + 0] = node;
        bvh_.nodes[first_child + 1] = old_node;
        if (!node.is_leaf()) {
            parents_[node.first_index + 0] = first_child + 0;
            parents_[node.first_index + 1] = first_child + 0;
        }
        if (!old_node.is_leaf()) {
            parents_[old_node.first_index + 0] = first_child + 1;
            parents_[old_node.first_index + 1] = first_child + 1;
        }
        parents_[first_child + 0] = target;
        parents_[first_child + 1] = target;
        refit_from(target);
    }

private:
    void refit_from(size_t index) {
        do {
            auto& node = bvh_.nodes[index];
            assert(!node.is_leaf());
            node.bbox_proxy() =
                bvh_.nodes[node.first_index + 0].bbox().extend(
                bvh_.nodes[node.first_index + 1].bbox());
            index = parents_[index];
        } while (index != 0);
    }

    Bvh& bvh_;
    std::vector<size_t> free_list_;
    std::vector<size_t> parents_;
};

} // namespace bvh

#endif
