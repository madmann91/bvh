#ifndef BVH_BUILDERS_SWEEP_SAH_BUILDER_H
#define BVH_BUILDERS_SWEEP_SAH_BUILDER_H

#include <cstddef>
#include <numeric>
#include <memory>
#include <atomic>
#include <array>
#include <type_traits>
#include <cassert>

#include <proto/bbox.h>
#include <proto/vec.h>
#include <proto/utils.h>

#include "bvh/bvh.h"
#include "bvh/builders/top_down_scheduler.h"
#include "bvh/builders/top_down_builder_config.h"

namespace bvh {

/// Top-down, full-sweep SAH-based BVH builder. Primitives are only
/// sorted once, and a stable partitioning algorithm is used when splitting,
/// so as to keep the relative order of primitives within each partition intact.
template <typename Bvh>
class SweepSahBuilder {
    using Scalar = typename Bvh::Scalar;
    using Node   = typename Bvh::Node;
    using BBox   = proto::BBox<Scalar>;
    using Vec3   = proto::Vec3<Scalar>;

    using Mark = uint_fast8_t;

public:
    using Config = TopDownBuilderConfig<Scalar>;

private:
    struct WorkItem {
        size_t node_index;
        size_t depth;
        size_t begin, end;

        size_t size() const { return end - begin; }
    };

    struct Split {
        int axis = -1;
        Scalar cost = std::numeric_limits<Scalar>::max();
        size_t prim_index = 0;

        operator bool () const { return axis >= 0; }
    };

    class Task {
    public:
        Task(
            Bvh& bvh,
            const Config& config,
            const BBox* bboxes,
            Mark* marks,
            Scalar* costs,
            const std::array<size_t*, 3>& sorted_indices,
            std::atomic<size_t>& node_count)
            : bvh_(bvh)
            , config_(config)
            , bboxes_(bboxes)
            , marks_(marks)
            , costs_(costs)
            , sorted_indices_(sorted_indices)
            , node_count_(node_count)
        {}

        std::optional<std::pair<WorkItem, WorkItem>> run(WorkItem&& item) const {
            // Make the current node a leaf
            auto& node = bvh_.nodes[item.node_index];
            node.prim_count  = item.size();
            node.first_index = item.begin;

            if (item.depth >= config_.max_depth ||
                item.size() <= config_.min_prims_per_leaf)
                return std::nullopt;

            Split split;
            for (int axis = 0; axis < 3; ++axis) {
                auto indices = sorted_indices_[axis];

                // Right sweep to compute partial costs
                auto right_bbox = BBox::empty();
                size_t begin = item.begin;
                for (size_t i = item.end - 1; i > item.begin; --i) {
                    right_bbox.extend(bboxes_[indices[i]]);
                    auto right_cost = right_bbox.half_area() * (item.end - i);
                    if (proto_unlikely(right_cost >= split.cost)) {
                        begin = i;
                        break;
                    }
                    costs_[i] = right_cost;
                }

                // Left sweep to compute full cost
                auto left_bbox = BBox::empty();
                for (size_t i = item.begin; i < begin; ++i)
                    left_bbox.extend(bboxes_[indices[i]]);
                for (size_t i = begin; i < item.end; ++i) {
                    left_bbox.extend(bboxes_[indices[i]]);
                    auto left_cost = left_bbox.half_area() * (i - item.begin + 1);
                    if (proto_unlikely(left_cost >= split.cost))
                        break;
                    auto cost = left_cost + costs_[i + 1];
                    if (cost < split.cost)
                        split = Split { axis, cost, i + 1 };
                }
            }

            // See the discussion on the termination criterion in `binned_bvh_builder.h`
            auto leaf_cost = node.bbox().half_area() * (item.size() - config_.traversal_cost);
            if (!split || split.cost >= leaf_cost) {
                if (item.size() > config_.max_prims_per_leaf) {
                    // This effectively creates a median split along the largest axis
                    split.axis = node.bbox().largest_axis();
                    split.prim_index = (item.begin + item.end + 1) / 2;
                } else
                    return std::nullopt;
            }

            // Mark the primitives that are on each side of the split,
            // and compute the bounding boxes of the children.
            auto mark_primitives = [&] (size_t begin, size_t end, Mark mark) {
                auto bbox = BBox::empty();
                for (size_t i = begin; i < end; ++i) {
                    auto prim_index = sorted_indices_[split.axis][i];
                    bbox.extend(bboxes_[prim_index]);
                    marks_[prim_index] = mark;
                }
                return bbox;
            };
            auto left_bbox  = mark_primitives(item.begin, split.prim_index, 1);
            auto right_bbox = mark_primitives(split.prim_index, item.end,   0);

            // We now partition the set of objects based on whether they appear
            // on the left or the right side of the split.
            // It is important to use a stable partitioning algorithm here, so as
            // to keep the order of primitive indices intact.
            auto is_marked = [&] (size_t i) { return marks_[i] != 0; };
            auto other_axes = std::array { (split.axis + 1) % 3, (split.axis + 2) % 3 };
            for (auto axis : other_axes)
                std::stable_partition(sorted_indices_[axis] + item.begin, sorted_indices_[axis] + item.end, is_marked);

            // Create an inner node
            assert(split.prim_index > item.begin && split.prim_index < item.end);
            auto left_index = node_count_.fetch_add(2);
            bvh_.nodes[left_index + 0].bbox_proxy() = left_bbox;
            bvh_.nodes[left_index + 1].bbox_proxy() = right_bbox;
            node.first_index = left_index;
            node.prim_count = 0;
            return std::make_optional(std::pair {
                WorkItem { left_index,     item.depth + 1, item.begin,       split.prim_index },
                WorkItem { left_index + 1, item.depth + 1, split.prim_index, item.end         } });
        }

    private:
        Bvh& bvh_;
        const Config& config_;
        const BBox* bboxes_;
        Mark* marks_;
        Scalar* costs_;
        std::array<size_t*, 3> sorted_indices_;
        std::atomic<size_t>& node_count_;
    };

    friend class TopDownScheduler<SweepSahBuilder>;

public:
    /// Builds the BVH from primitive bounding boxes and centers provided as two pointers.
    static Bvh build(
        TopDownScheduler<SweepSahBuilder>& scheduler,
        const BBox& global_bbox,
        const BBox* bboxes,
        const Vec3* centers,
        size_t prim_count,
        const Config& config = {})
    {
        Bvh bvh;

        // Initialize primitive indices and allocate nodes
        bvh.nodes = std::make_unique<Node[]>(2 * prim_count - 1);
        bvh.prim_indices   = std::make_unique<size_t[]>(prim_count);
        auto other_indices = std::make_unique<size_t[]>(prim_count * 2);

        auto marks = std::make_unique<Mark[]>(prim_count);
        auto costs = std::make_unique<Scalar[]>(prim_count);

        std::array<size_t*, 3> sorted_indices {
            bvh.prim_indices.get(),
            other_indices.get(),
            other_indices.get() + prim_count
        };

        for (int axis = 0; axis < 3; ++axis) {
            auto indices = sorted_indices[axis];
            std::iota(indices, indices + prim_count, 0);
            std::sort(indices, indices + prim_count,
                [&] (size_t i, size_t j) { return centers[i][axis] < centers[j][axis]; });
        }
        bvh.nodes[0].bbox_proxy() = global_bbox;

        // Start the root task and wait for the result
        std::atomic<size_t> node_count(1);
        scheduler.run(
            Task(bvh, config, bboxes, marks.get(), costs.get(), sorted_indices, node_count),
            WorkItem { 0, 0, 0, prim_count });

        bvh.node_count = node_count;
        bvh.nodes = proto::copy(bvh.nodes, bvh.node_count);
        return bvh;
    }
};

} // namespace bvh

#endif
