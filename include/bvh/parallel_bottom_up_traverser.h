#ifndef BVH_PARALLEL_BOTTOM_UP_TRAVERSER_H
#define BVH_PARALLEL_BOTTOM_UP_TRAVERSER_H

#include <cstddef>
#include <ranges>
#include <atomic>
#include <algorithm>
#include <ranges>
#include <vector>
#include <cassert>

#include "bvh/bvh.h"

namespace bvh {

/// Base class for bottom-up BVH traversal algorithms. The implementation is inspired
/// from T. Karras' bottom-up refitting algorithm, explained in the article
/// "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees".
template <typename Bvh>
class ParallelBottomUpTraverser {
    struct Flag {
        std::atomic_int value;
        Flag() = default;
        Flag(int init) : value(init) {}
        Flag(const Flag& other) : value(other.value.load()) {}
        Flag& operator = (const Flag& other) {
            value.store(other.value.load());
            return *this;
        }
        int mark() { return value.fetch_add(1); }
        void reset() { value.store(0); }
    };

    std::vector<Flag> flags_;

public:
    /// Traverses the BVH with an array of parent-child indices, and two functions that
    /// control how to process leaves and inner nodes.
    template <typename ProcessLeaf, typename ProcessInnerNode>
    void traverse(
        const Bvh& bvh,
        const std::vector<size_t>& parents,
        ProcessLeaf&& process_leaf,
        ProcessInnerNode&& process_inner_node)
    {
        assert(bvh.nodes.size() == parents.size());
        // Create a set of flags for this BVH (resize() cannot be used here
        if (flags_.size() != bvh.nodes.size())
            flags_.resize(bvh.nodes.size(), 0);

        // Special case if the BVH is just a leaf
        if (bvh.nodes.size() == 1)
            process_leaf(0);

        auto range = std::views::iota(size_t{1}, bvh.nodes.size());
        std::for_each(
            std::execution::par_unseq,
            range.begin(), range.end(),
            [&] (size_t i) {
                // Only process leaves
                if (!bvh.nodes[i].is_leaf())
                    return;

                process_leaf(i);

                // Process inner nodes on the path from that leaf up to the root
                size_t j = i;
                do {
                    j = parents[j];

                    // Make sure that the children of this inner node have been processed
                    if (int previous_flag = flags_[j].mark(); previous_flag != 1)
                        break;
                    flags_[j].reset();

                    process_inner_node(j);
                } while (j != 0);
            });
    }

    /// Same as the other version of `traverse()` but recomputes the parent indices from the BVH.
    template <typename ProcessLeaf, typename ProcessInnerNode>
    void traverse(
        const Bvh& bvh,
        ProcessLeaf&& process_leaf,
        ProcessInnerNode&& process_inner_node)
    {
        return traverse(
            bvh, bvh.parents(std::execution::par_unseq),
            std::move(process_leaf),
            std::move(process_inner_node));
    }
};

} // namespace bvh

#endif
