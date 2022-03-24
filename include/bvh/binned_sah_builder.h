#ifndef BVH_BINNED_SAH_BUILDER_H
#define BVH_BINNED_SAH_BUILDER_H

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
#include "bvh/top_down_builder_common.h"

namespace bvh {

/// BVH builder using the method described in
/// "On Fast Construction of SAH-based Bounding Volume Hierarchies", by I. Wald.
/// The number of bins can be configured as a template parameter.
/// The default number of 16 is usually enough for most applications.
template <typename Bvh, size_t BinCount = 16>
class BinnedSahBuilder {
    using Scalar = typename Bvh::Scalar;
    using Index  = typename Bvh::Index;
    using Node   = typename Bvh::Node;
    using BBox   = proto::BBox<Scalar>;
    using Vec3   = proto::Vec3<Scalar>;

public:
    using Config = TopDownBuilderConfig<Scalar>;
    static constexpr size_t bin_count = BinCount;

private:
    struct WorkItem {
        size_t node_index;
        size_t depth;
        size_t begin, end;

        size_t size() const { return end - begin; }
    };

    struct Bin {
        BBox bbox = BBox::empty();
        size_t prim_count = 0;

        Bin() = default;

        Bin& proto_always_inline merge(const Bin& other) {
            bbox.extend(other.bbox);
            prim_count += other.prim_count;
            return *this;
        }

        Scalar proto_always_inline cost() const {
            return static_cast<Scalar>(prim_count) * bbox.half_area();
        }

        static proto_always_inline std::pair<Scalar, Scalar> offset_and_scale(Scalar min, Scalar max) {
            auto scale = bin_count / (max - min);
            auto offset = -min * scale;
            return std::pair { offset, scale };
        }

        static proto_always_inline size_t index(Scalar pos, Scalar offset, Scalar scale) {
            return std::min(bin_count - 1, size_t(std::max(ptrdiff_t{0}, ptrdiff_t(proto::fast_mul_add(pos, scale, offset)))));
        }

        static proto_always_inline size_t index(Scalar pos, std::pair<Scalar, Scalar> offset_and_scale) {
            return index(pos, offset_and_scale.first, offset_and_scale.second);
        }
    };

    struct Split {
        int axis = -1;
        Scalar cost = std::numeric_limits<Scalar>::max();
        size_t bin_index = 0;

        operator bool () const { return axis >= 0; }
    };

    class Task {
    public:
        Task(
            Bvh& bvh,
            const Config& config,
            const BBox* bboxes,
            const Vec3* centers,
            std::atomic<size_t>& node_count)
            : bvh_(bvh)
            , config_(config)
            , bboxes_(bboxes)
            , centers_(centers)
            , node_count_(node_count)
        {}

        std::optional<std::pair<WorkItem, WorkItem>> run(WorkItem&& item) const {
            // Make the current node a leaf
            auto& node = bvh_.nodes[item.node_index];
            node.prim_count  = static_cast<Index>(item.size());
            node.first_index = static_cast<Index>(item.begin);

            if (item.depth >= config_.max_depth ||
                item.size() <= config_.min_prims_per_leaf)
                return std::nullopt;

            std::array<Bin, bin_count> bins_per_axis[3];
            auto offsets_and_scales_per_axis = std::array {
                Bin::offset_and_scale(node.bbox().min[0], node.bbox().max[0]),
                Bin::offset_and_scale(node.bbox().min[1], node.bbox().max[1]),
                Bin::offset_and_scale(node.bbox().min[2], node.bbox().max[2]) };

            for (size_t i = item.begin; i < item.end; ++i) {
                proto::static_for<0, 3>([&] (int axis) {
                    auto prim_index = bvh_.prim_indices[i];
                    auto bin_index = Bin::index(centers_[prim_index][axis], offsets_and_scales_per_axis[axis]);
                    auto& bin = bins_per_axis[axis][bin_index];
                    bin.bbox.extend(bboxes_[prim_index]);
                    bin.prim_count++;
                });
            }

            Split split;
            Bin left_accums_per_axis[3], right_accums_per_axis[3];
            std::array<Scalar, bin_count> right_costs_per_axis[3];
            for (size_t i = bin_count - 1; i > 0; --i) {
                proto::static_for<0, 3>([&] (int axis) {
                    auto& bin = bins_per_axis[axis][i];
                    right_accums_per_axis[axis].merge(bin);
                    right_costs_per_axis[axis][i] = right_accums_per_axis[axis].cost();
                });
            }

            for (size_t i = 0; i < bin_count - 1; ++i) {
                proto::static_for<0, 3>([&] (int axis) {
                    auto& bin = bins_per_axis[axis][i];
                    left_accums_per_axis[axis].merge(bin);
                    auto cost = left_accums_per_axis[axis].cost() + right_costs_per_axis[axis][i + 1];
                    if (cost < split.cost)
                        split = Split { axis, cost, i + 1 };
                });
            }

            size_t right_begin = 0;
            auto left_bbox  = BBox::empty();
            auto right_bbox = BBox::empty();

            if (!split || !config_.is_good_split(split.cost, node.bbox().half_area(), item.size())) {
                // If the split is not useful, then we can create a leaf.
                // However, if there are too many primitives to create a leaf,
                // we apply the fallback strategy: A median split.
                if (item.size() > config_.max_prims_per_leaf) {
                    auto axis = node.bbox().largest_axis();
                    right_begin = (item.begin + item.end + 1) / 2;
                    std::partial_sort(
                        bvh_.prim_indices.data() + item.begin,
                        bvh_.prim_indices.data() + right_begin,
                        bvh_.prim_indices.data() + item.end,
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
                auto offset_and_scale = offsets_and_scales_per_axis[split.axis];
                right_begin = std::partition(
                    bvh_.prim_indices.data() + item.begin,
                    bvh_.prim_indices.data() + item.end,
                    [&] (size_t i) {
                        return Bin::index(centers_[i][split.axis], offset_and_scale) < split.bin_index;
                    }) - bvh_.prim_indices.data();

                // Compute the left and right bounding boxes
                auto& bins = bins_per_axis[split.axis];
                for (size_t i = 0; i < split.bin_index; ++i)
                    left_bbox.extend(bins[i].bbox);
                for (size_t i = bin_count; i > split.bin_index; --i)
                    right_bbox.extend(bins[i - 1].bbox);
            }

            // Create an inner node
            assert(right_begin > item.begin && right_begin < item.end);
            auto left_index = node_count_.fetch_add(2);
            bvh_.nodes[left_index + 0].bbox_proxy() = left_bbox;
            bvh_.nodes[left_index + 1].bbox_proxy() = right_bbox;
            node.first_index = static_cast<Index>(left_index);
            node.prim_count = 0;
            return std::make_optional(std::pair {
                WorkItem { left_index,     item.depth + 1, item.begin,  right_begin },
                WorkItem { left_index + 1, item.depth + 1, right_begin, item.end    } });
        }

    private:
        Bvh& bvh_;
        const Config& config_;
        const BBox* bboxes_;
        const Vec3* centers_;
        std::atomic<size_t>& node_count_;
    };

    friend TopDownScheduler<BinnedSahBuilder>;

public:
    /// Builds the BVH from primitive bounding boxes and centers provided as two pointers.
    template <template <typename> typename TopDownScheduler>
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
        bvh.prim_indices.resize(prim_count);
        bvh.nodes.resize(2 * prim_count - 1);
        std::iota(bvh.prim_indices.begin(), bvh.prim_indices.end(), 0);
        bvh.nodes[0].bbox_proxy() = global_bbox;

        // Start the root task and wait for the result
        std::atomic<size_t> node_count(1);
        scheduler.run(Task(bvh, config, bboxes, centers, node_count), WorkItem { 0, 0, 0, prim_count });
        bvh.nodes.resize(node_count);
        return bvh;
    }
};

} // namespace bvh

#endif
