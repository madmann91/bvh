#ifndef BVH_BUILDERS_BINNED_SAH_BUILDER_H
#define BVH_BUILDERS_BINNED_SAH_BUILDER_H

#include <cstddef>
#include <numeric>
#include <memory>
#include <atomic>
#include <array>
#include <type_traits>
#include <cassert>

#include <proto/bbox.h>
#include <proto/vec.h>

#include "bvh/bvh.h"
#include "bvh/builders/top_down_scheduler.h"
#include "bvh/builders/top_down_builder_config.h"

namespace bvh {

/// BVH builder using the method described in
/// "On Fast Construction of SAH-based Bounding Volume Hierarchies", by I. Wald.
/// The number of bins can be configured as a template parameter.
/// The default number of 16 is usually enough for most applications.
template <typename Bvh, size_t BinCount = 16>
class BinnedSahBuilder {
    using Scalar = typename Bvh::Scalar;
    using Node   = typename Bvh::Node;
    using BBox   = proto::BBox<Scalar>;
    using Vec3   = proto::Vec3<Scalar>;

public:
    using Config = TopDownBuilderConfig<Scalar>;
    static constexpr size_t bin_count = BinCount;

private:
    struct WorkItem {
        size_t node_index;
        size_t begin, end;

        size_t size() const { return end - begin; }
    };

    struct Bin {
        BBox bbox = BBox::empty();
        size_t prim_count = 0;

        Bin() = default;

        Bin& merge(const Bin& other) {
            bbox.extend(other.bbox);
            prim_count += other.prim_count;
            return *this;
        }

        Scalar cost() const {
            return prim_count * bbox.half_area();
        }
    };

    struct Split {
        int axis = -1;
        Scalar cost = std::numeric_limits<Scalar>::max();
        size_t bin_index = 0;

        operator bool () const { return axis >= 0; }
    };

    static size_t bin_index(Scalar pos, Scalar min, Scalar max) {
        auto scale = bin_count / (max - min);
        return std::min(bin_count - 1, size_t(std::max(ptrdiff_t{0}, ptrdiff_t((pos - min) * scale))));
    }

    static size_t bin_index(int axis, const Vec3& pos, const BBox& bbox) {
        return bin_index(pos[axis], bbox.min[axis], bbox.max[axis]);
    }

    class Task {
    public:
        Task(
            Bvh& bvh,
            const Config& config,
            const BBox* bboxes,
            const Vec3* centers)
            : bvh_(bvh)
            , config_(config)
            , bboxes_(bboxes)
            , centers_(centers)
        {}

        std::optional<std::pair<WorkItem, WorkItem>> run(WorkItem&& item) const {
            // Make the current node a leaf
            auto& node = bvh_.nodes[item.node_index];
            node.prim_count  = item.size();
            node.first_index = item.begin;

            if (item.size() <= config_.min_prims_per_leaf)
                return std::nullopt;

            std::array<Bin, bin_count> bins_per_axis[3];

            for (size_t i = item.begin; i < item.end; ++i) {
                for (int axis = 0; axis < 3; ++axis) {
                    auto prim_index = bvh_.prim_indices[i];
                    auto& bin = bins_per_axis[axis][bin_index(axis, centers_[prim_index], node.bbox())];
                    bin.bbox.extend(bboxes_[prim_index]);
                    bin.prim_count++;
                }
            }

            Split split;
            std::array<Scalar, bin_count> right_cost;
            for (int axis = 0; axis < 3; ++axis) {
                auto& bins = bins_per_axis[axis];

                Bin left_accum, right_accum;
                for (size_t i = bin_count - 1; i > 0; --i) {
                    auto& bin = bins[i];
                    right_accum.merge(bin);
                    right_cost[i] = right_accum.cost();
                }

                for (size_t i = 0; i < bin_count - 1; ++i) {
                    auto& bin = bins[i];
                    left_accum.merge(bin);
                    auto cost = left_accum.cost() + right_cost[i + 1];
                    if (cost < split.cost)
                        split = Split { axis, cost, i + 1 };
                }
            }

            size_t right_begin = 0;
            auto left_bbox  = BBox::empty();
            auto right_bbox = BBox::empty();

            // Test the validity of the split using the SAH stopping criterion:
            // If the SAH cost of the split is higher than the SAH cost of the current node as a leaf,
            // then the split is not useful (in the sense of the SAH).
            auto leaf_cost = node.bbox().half_area() * (item.size() - config_.traversal_cost);
            if (!split || split.cost >= leaf_cost) {
                // If the split is not useful, then we can create a leaf.
                // However, if there are too many primitives to create a leaf,
                // we apply the fallback strategy: A median split.
                if (item.size() > config_.max_prims_per_leaf) {
                    auto axis = node.bbox().largest_axis();
                    right_begin = (item.begin + item.end + 1) / 2;
                    std::partial_sort(
                        bvh_.prim_indices.get() + item.begin,
                        bvh_.prim_indices.get() + right_begin,
                        bvh_.prim_indices.get() + item.end,
                        [&] (size_t i, size_t j) { return centers_[i][axis] < centers_[j][axis]; });

                    // Compute left and right bounding boxes
                    for (size_t i = item.begin; i < right_begin; ++i)
                        left_bbox.extend(bboxes_[bvh_.prim_indices[i]]);
                    for (size_t i = right_begin; i < item.end; ++i)
                        right_bbox.extend(bboxes_[bvh_.prim_indices[i]]);
                } else
                    return std::nullopt;
            } else {
                // If the split is useful, we partition the set of objects based on the bins they fall in.
                right_begin = std::partition(
                    bvh_.prim_indices.get() + item.begin,
                    bvh_.prim_indices.get() + item.end,
                    [&] (size_t i) {
                        return bin_index(split.axis, centers_[i], node.bbox()) < split.bin_index;
                    }) - bvh_.prim_indices.get();

                // Compute the left and right bounding boxes
                auto& bins = bins_per_axis[split.axis];
                for (size_t i = 0; i < split.bin_index; ++i)
                    left_bbox.extend(bins[i].bbox);
                for (size_t i = bin_count; i > split.bin_index; --i)
                    right_bbox.extend(bins[i - 1].bbox);
            }

            // Create an inner node
            assert(right_begin > item.begin && right_begin < item.end);
            auto left_index = std::atomic_ref(bvh_.node_count).fetch_add(2);
            bvh_.nodes[left_index + 0].bbox_proxy() = left_bbox;
            bvh_.nodes[left_index + 1].bbox_proxy() = right_bbox;
            node.first_index = left_index;
            node.prim_count = 0;
            return std::make_optional(std::pair {
                WorkItem { left_index,     item.begin,  right_begin },
                WorkItem { left_index + 1, right_begin, item.end    } });
        }

    private:
        Bvh& bvh_;
        const Config& config_;
        const BBox* bboxes_;
        const Vec3* centers_;
    };

    friend class TopDownScheduler<BinnedSahBuilder>;

public:
    /// Builds the BVH from primitive bounding boxes and centers provided as two pointers.
    static Bvh build(
        TopDownScheduler<BinnedSahBuilder>& scheduler,
        const BBox& global_bbox,
        const BBox* bboxes,
        const Vec3* centers,
        size_t prim_count,
        const Config& config = {})
    {
        Bvh bvh;

        // Initialize primitive indices and allocate nodes
        bvh.prim_indices = std::make_unique<size_t[]>(prim_count);
        bvh.nodes = std::make_unique<Node[]>(2 * prim_count - 1);
        std::iota(bvh.prim_indices.get(), bvh.prim_indices.get() + prim_count, 0);

        // Compute a global bounding box around the root
        bvh.nodes[0].bbox_proxy() = global_bbox;
        bvh.node_count = 1;

        // Start the root task and wait for the result
        scheduler.run(Task(bvh, config, bboxes, centers), WorkItem { 0, 0, prim_count });

        bvh.nodes = proto::copy(bvh.nodes, bvh.node_count);
        return bvh;
    }
};

} // namespace bvh

#endif
