#ifndef BVH_V2_SWEEP_SAH_BUILDER_H
#define BVH_V2_SWEEP_SAH_BUILDER_H

#include "bvh/v2/bvh.h"
#include "bvh/v2/vec.h"
#include "bvh/v2/bbox.h"

#include <stack>
#include <tuple>
#include <algorithm>
#include <optional>
#include <numeric>
#include <cassert>

namespace bvh::v2 {

/// Single-threaded top-down builder that partitions primitives based on the Surface
/// Area Heuristic (SAH). Primitives are only sorted once along each axis.
template <typename Node>
class SweepSahBuilder {
    using Scalar = typename Node::Scalar;
    using Vec  = bvh::v2::Vec<Scalar, Node::dimension>;
    using BBox = bvh::v2::BBox<Scalar, Node::dimension>;

public:
    struct Config {
        /// Nodes containing less than this amount of primitives will not be split.
        /// This is mostly to speed up BVH construction, and should not be increased beyond reason.
        size_t min_leaf_size = 1;

        /// Nodes that cannot be split based on the SAH and have a number of primitives larger than
        /// this will be split using a fallback strategy. This should not happen often, but may
        /// happen in worst-case scenarios or poorly designed scenes.
        size_t max_leaf_size = 16;

        /// Log of the size of primitive clusters in base 2. It may sometimes be useful to intersect
        /// N primitives at a time, in which case intersecting 1, 2, ... N - 1 primitives has the
        /// same cost as intersecting N primitives. This parameter controls the SAH splitting
        /// criterion to reflect this, and should be set to log2(N).
        size_t log_cluster_size = 0;

        /// Ratio representing the performance cost of intersecting a node (usually a ray-box
        /// intersection), over the performance cost of intersecting a primitive. A ratio of 1 means
        /// that a ray-box intersection is as expensive as a ray-primitive intersection.
        Scalar cost_ratio = static_cast<Scalar>(1.);
    };

    /// Starts building a BVH with the given primitive data. The build algorithm is single-threaded.
    /// Auxiliary buffers are kept from one build to the other, allowing efficient memory re-use
    /// within the same BVH builder.
    Bvh<Node> build(
        const BBox* bboxes,
        const Vec* centers,
        size_t prim_count,
        const Config& config = {})
    {
        config_ = &config;
        marks_.resize(prim_count);
        accum_.resize(prim_count);
        bboxes_ = bboxes;
        prim_offset_ = make_bitmask<size_t>(config.log_cluster_size);

        for (unsigned axis = 0; axis < Node::dimension; ++axis) {
            prim_ids_[axis].resize(prim_count);
            std::iota(prim_ids_[axis].begin(), prim_ids_[axis].end(), 0);
            std::sort(prim_ids_[axis].begin(), prim_ids_[axis].end(), [&] (size_t i, size_t j) {
                return centers[i][axis] < centers[j][axis];
            });
        }

        return build();
    }

protected:
    struct WorkItem {
        size_t node_id;
        size_t begin;
        size_t end;

        BVH_ALWAYS_INLINE size_t size() const { return end - begin; }
    };

    struct Split {
        size_t pos;
        Scalar cost;
        unsigned axis;
    };

    std::vector<bool> marks_;
    std::vector<Scalar> accum_;
    std::vector<size_t> prim_ids_[Node::dimension];

    const BBox* bboxes_;
    size_t prim_offset_;
    const Config* config_;

    Bvh<Node> build() {
        // Create a BVH with a default-initialized root node
        Bvh<Node> bvh;
        bvh.nodes.reserve((2 * marks_.size()) / config_->min_leaf_size);
        bvh.nodes.emplace_back();
        bvh.nodes.back().set_bbox(compute_bbox(0, 0, marks_.size()));

        std::stack<WorkItem> stack;
        stack.push(WorkItem { 0, 0, marks_.size() });
        while (!stack.empty()) {
            auto item = stack.top();
            stack.pop();

            auto& node = bvh.nodes[item.node_id];
            if (item.size() > config_->min_leaf_size) {
                if (auto split = try_split(node.get_bbox(), item.begin, item.end)) {
                    auto first_child = bvh.nodes.size();
                    node.make_inner(first_child);

                    bvh.nodes.resize(first_child + 2);
                    bvh.nodes[first_child + 0].set_bbox(compute_bbox(0, item.begin, split->pos));
                    bvh.nodes[first_child + 1].set_bbox(compute_bbox(0, split->pos, item.end));
                    auto first_item  = WorkItem { first_child + 0, item.begin, split->pos };
                    auto second_item = WorkItem { first_child + 1, split->pos, item.end };
                    if (first_item.size() < second_item.size())
                        std::swap(first_item, second_item);
                    stack.push(first_item);
                    stack.push(second_item);
                    continue;
                }
            }

            node.make_leaf(item.begin, item.size());
        }
        bvh.prim_ids = std::move(prim_ids_[0]);
        bvh.nodes.shrink_to_fit();

        return bvh;
    }

    void find_best_split(unsigned axis, size_t begin, size_t end, Split& best_split) {
        size_t first_right = begin;

        // Sweep from the right to the left, computing the partial SAH cost
        auto right_bbox = BBox::make_empty();
        for (size_t i = end - 1; i > begin;) {
            static constexpr size_t chunk_size = 32;
            size_t next = i - std::min(i - begin, chunk_size);
            auto right_cost = static_cast<Scalar>(0.);
            for (; i > next; --i) {
                right_bbox.extend(bboxes_[prim_ids_[axis][i]]);
                accum_[i] = right_cost = compute_leaf_cost(i, end, right_bbox);
            }
            // Every `chunk_size` elements, check that we are not above the maximum cost
            if (right_cost > best_split.cost) {
                first_right = i;
                break;
            }
        }

        // Sweep from the left to the right, computing the full cost
        auto left_bbox = BBox::make_empty();
        for (size_t i = begin; i < first_right; ++i)
            left_bbox.extend(bboxes_[prim_ids_[axis][i]]);
        for (size_t i = first_right; i < end - 1; ++i) {
            left_bbox.extend(bboxes_[prim_ids_[axis][i]]);
            auto left_cost = compute_leaf_cost(begin, i + 1, left_bbox);
            auto cost = left_cost + accum_[i + 1];
            if (cost < best_split.cost)
                best_split = Split { i + 1, cost, axis };
            else if (left_cost > best_split.cost)
                break;
        }
    }

    std::optional<Split> try_split(const BBox& bbox, size_t begin, size_t end) {
        // Find the best split over all axes
        auto leaf_cost = compute_leaf_cost(begin, end, bbox, config_->cost_ratio);
        auto best_split = Split { begin + 1, leaf_cost, 0 };
        for (unsigned axis = 0; axis < Node::dimension; ++axis)
            find_best_split(axis, begin, end, best_split);

        // Make sure that the split is good before proceeding with it
        if (best_split.cost >= leaf_cost) {
            if (end - begin <= config_->max_leaf_size)
                return std::nullopt;

            // If the number of primitives is too high, fallback on a split at the
            // median on the largest axis.
            best_split.pos = (begin + end) / 2;
            best_split.axis = bbox.get_diagonal().get_largest_axis();
        }

        // Partition primitives (keeping the order intact so that the next recursive calls do not
        // need to sort primitives again).
        mark_primitives(best_split.axis, begin, best_split.pos, end);
        for (unsigned axis = 0; axis < Node::dimension; ++axis) {
            if (axis == best_split.axis)
                continue;
            std::stable_partition(
                prim_ids_[axis].begin() + begin,
                prim_ids_[axis].begin() + end,
                [&] (size_t i) { return marks_[i]; });
        }

        return std::make_optional(best_split);
    }

    BVH_ALWAYS_INLINE void mark_primitives(size_t axis, size_t begin, size_t split_pos, size_t end) {
        for (size_t i = begin; i < split_pos; ++i) marks_[prim_ids_[axis][i]] = true;
        for (size_t i = split_pos; i < end; ++i)   marks_[prim_ids_[axis][i]] = false;
    }

    BVH_ALWAYS_INLINE BBox compute_bbox(size_t axis, size_t begin, size_t end) const {
        auto bbox = BBox::make_empty();
        for (size_t i = begin; i < end; ++i)
            bbox.extend(bboxes_[prim_ids_[axis][i]]);
        return bbox;
    }

    BVH_ALWAYS_INLINE size_t compute_prim_count(size_t size) const {
        return (size + prim_offset_) >> config_->log_cluster_size;
    }

    BVH_ALWAYS_INLINE Scalar compute_leaf_cost(
        size_t begin,
        size_t end,
        const BBox& bbox,
        Scalar offset = static_cast<Scalar>(0.)) const
    {
        return bbox.get_half_area() * (static_cast<Scalar>(compute_prim_count(end - begin)) - offset);
    }
};

} // namespace bvh::v2

#endif
