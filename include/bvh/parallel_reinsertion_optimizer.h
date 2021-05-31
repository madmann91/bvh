#ifndef BVH_PARALLEL_REINSERTION_OPTIMIZER_H
#define BVH_PARALLEL_REINSERTION_OPTIMIZER_H

#include <cassert>
#include <cstddef>
#include <array>
#include <utility>
#include <execution>
#include <ranges>
#include <vector>

#include <proto/utils.h>
#include <proto/bbox.h>

#include "bvh/bvh.h"
#include "bvh/parallel_hierarchy_refitter.h"

namespace bvh {

/// Optimization that tries to re-insert BVH nodes in such a way that the
/// SAH cost of the tree decreases after the re-insertion. Inspired from the
/// article "Parallel Reinsertion for Bounding Volume Hierarchy Optimization",
/// by D. Meister and J. Bittner.
template <typename Bvh>
class ParallelReinsertionOptimizer {
    using Scalar = typename Bvh::Scalar;
    using Node   = typename Bvh::Node;
    using BBox   = proto::BBox<Scalar>;

    using Insertion    = std::pair<size_t, Scalar>;
    using ConflictList = std::array<size_t, 6>;

    Bvh& bvh_;
    std::vector<size_t> parents_;
    ParallelHierarchyRefitter<Bvh> refitter_;

public:
    ParallelReinsertionOptimizer(Bvh& bvh)
        : bvh_(bvh), parents_(bvh.parents(std::execution::par_unseq))
    {}

private:
    ConflictList conflicts(size_t in, size_t out) const {
        // Return an array of re-insertion conflicts for the given nodes
        auto parent_in = parents_[in];
        return ConflictList {
            in,
            Bvh::sibling(in),
            parent_in,
            parent_in == 0 ? in : parents_[parent_in],
            out,
            out == 0 ? out : parents_[out],
        };
    }

    void reinsert(size_t in, size_t out) {
        auto sibling_in   = Bvh::sibling(in);
        auto parent_in    = parents_[in];
        auto sibling_node = bvh_.nodes[sibling_in];
        auto out_node     = bvh_.nodes[out];

        // Re-insert it into the destination
        bvh_.nodes[out].bbox_proxy().extend(bvh_.nodes[in].bbox());
        bvh_.nodes[out].first_index = std::min(in, sibling_in);
        bvh_.nodes[out].prim_count = 0;
        bvh_.nodes[sibling_in] = out_node;
        bvh_.nodes[parent_in] = sibling_node;

        // Update parent-child indices
        if (!out_node.is_leaf()) {
            parents_[out_node.first_index + 0] = sibling_in;
            parents_[out_node.first_index + 1] = sibling_in;
        }
        if (!sibling_node.is_leaf()) {
            parents_[sibling_node.first_index + 0] = parent_in;
            parents_[sibling_node.first_index + 1] = parent_in;
        }
        parents_[sibling_in] = out;
        parents_[in] = out;
    }

    Insertion search(size_t in) const {
        bool   down  = true;
        size_t pivot = parents_[in];
        size_t out   = Bvh::sibling(in);
        size_t out_best = out;

        auto bbox_in = bvh_.nodes[in].bbox();
        auto bbox_parent = bvh_.nodes[pivot].bbox();
        auto bbox_pivot = BBox::empty();

        Scalar d = 0;
        Scalar d_best = 0;
        const Scalar d_bound = bbox_parent.half_area() - bbox_in.half_area();

        // Perform a search to find a re-insertion position for the given node
        while (true) {
            auto bbox_out = bvh_.nodes[out].bbox();
            auto bbox_merged = BBox(bbox_in).extend(bbox_out);
            if (down) {
                auto d_direct = bbox_parent.half_area() - bbox_merged.half_area();
                if (d_best < d_direct + d) {
                    d_best = d_direct + d;
                    out_best = out;
                }
                d = d + bbox_out.half_area() - bbox_merged.half_area();
                if (bvh_.nodes[out].is_leaf() || d_bound + d <= d_best)
                    down = false;
                else
                    out = bvh_.nodes[out].first_index;
            } else {
                d = d - bbox_out.half_area() + bbox_merged.half_area();
                if (pivot == parents_[out]) {
                    bbox_pivot.extend(bbox_out);
                    out = pivot;
                    bbox_out = bvh_.nodes[out].bbox();
                    if (out != parents_[in]) {
                        bbox_merged = BBox(bbox_in).extend(bbox_pivot);
                        auto d_direct = bbox_parent.half_area() - bbox_merged.half_area();
                        if (d_best < d_direct + d) {
                            d_best = d_direct + d;
                            out_best = out;
                        }
                        d = d + bbox_out.half_area() - bbox_pivot.half_area();
                    }
                    if (out == 0)
                        break;
                    out = Bvh::sibling(pivot);
                    pivot = parents_[out];
                    down = true;
                } else {
                    if (Bvh::is_left_child(out)) {
                        down = true;
                        out = Bvh::sibling(out);
                    } else {
                        out = parents_[out];
                    }
                }
            }
        }

        if (in == out_best || Bvh::sibling(in) == out_best || parents_[in] == out_best)
            return Insertion { 0, 0 };
        return Insertion { out_best, d_best };
    }

    template <typename F>
    proto_always_inline void forall_nodes_in_iter(size_t first_node, size_t u, F&& f) {
        auto range = std::views::iota(size_t{0}, (bvh_.nodes.size() - first_node) / u);
        std::for_each(std::execution::par_unseq, range.begin(), range.end(), [&] (size_t i) {
            f(first_node + i * u);
        });
    }

public:
    void optimize(size_t u = 9, Scalar threshold = 0.1, Scalar traversal_cost = 1) {
        auto locks = std::make_unique<std::atomic<uint64_t>[]>(bvh_.nodes.size());
        auto outs  = std::make_unique<Insertion[]>(bvh_.nodes.size());

        auto old_cost = bvh_.sah_cost(std::execution::par_unseq, traversal_cost);
        for (size_t iter = 0; ; ++iter) {
            size_t first_node = iter % u + 1;

            // Clear the locks
            std::for_each(
                std::execution::par_unseq,
                locks.get(), locks.get() + bvh_.nodes.size(),
                [&] (std::atomic<uint64_t>& u) { u.store(0); });

            // Search for insertion candidates
            forall_nodes_in_iter(first_node, u, [&] (size_t i) { outs[i] = search(i); });

            // Resolve topological conflicts with locking
            forall_nodes_in_iter(first_node, u, [&] (size_t i) {
                if (outs[i].second <= 0)
                    return;
                // Encode locks into 64 bits using the highest 32 bits for the cost and the
                // lowest 32 bits for the index of the node requesting the re-insertion.
                // This takes advantage of the fact that IEEE-754 floats can be compared
                // with regular integer comparisons.
                auto lock = (uint64_t(proto::as<uint32_t>(float(outs[i].second))) << 32) | (uint64_t(i) & UINT64_C(0xFFFFFFFF));
                for (auto c : conflicts(i, outs[i].first))
                    proto::atomic_max(locks[c], lock);
            });

            // Check the locks to disable conflicting re-insertions
            forall_nodes_in_iter(first_node, u, [&] (size_t i) {
                if (outs[i].second <= 0)
                    return;
                auto c = conflicts(i, outs[i].first);
                // Make sure that this node owns all the locks for each and every conflicting node
                bool is_conflict_free = std::all_of(c.begin(), c.end(), [&] (size_t j) {
                    return (locks[j] & UINT64_C(0xFFFFFFFF)) == i;
                });
                if (!is_conflict_free)
                    outs[i] = Insertion { 0, 0 };
            });

            // Perform the reinsertions
            forall_nodes_in_iter(first_node, u, [&] (size_t i) {
                if (outs[i].second > 0)
                    reinsert(i, outs[i].first);
            });

            // Update the bounding boxes of each node in the tree
            refitter_.refit(bvh_, parents_);

            // Compare the old SAH cost to the new one and decrease the number
            // of nodes that are ignored during the optimization if the change
            // in cost is below the threshold.
            auto new_cost = bvh_.sah_cost(std::execution::par_unseq, traversal_cost);
            if (std::abs(new_cost - old_cost) <= threshold || iter >= u) {
                if (u <= 1)
                    break;
                u = u - 1;
                iter = 0;
            }
            old_cost = new_cost;
        }
    }
};

} // namespace bvh

#endif
