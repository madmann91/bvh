#ifndef BVH_SPATIAL_SPLIT_BVH_BUILDER_HPP
#define BVH_SPATIAL_SPLIT_BVH_BUILDER_HPP

#include <numeric>
#include <optional>

#include "bvh/bvh.hpp"
#include "bvh/bounding_box.hpp"
#include "bvh/top_down_builder.hpp"
#include "bvh/sah_based_algorithm.hpp"

namespace bvh {

template <typename, typename, size_t> class SpatialSplitBvhBuildTask;

/// This is a top-down, spatial split BVH builder based on:
/// "Spatial Splits in Bounding Volume Hierarchies", by M. Stich et al.
/// Even though the object splitting strategy is a full-sweep SAH evaluation,
/// this builder is not as efficient as bvh::SweepSahBuilder when spatial splits
/// are disabled, because it needs to sort primitive references at every step.
template <typename Bvh, typename Primitive, size_t BinCount>
class SpatialSplitBvhBuilder :
    public TopDownBuilder<Bvh, SpatialSplitBvhBuildTask<Bvh, Primitive, BinCount>>,
    public SahBasedAlgorithm<Bvh>
{
    using Scalar    = typename Bvh::ScalarType;
    using BuildTask = SpatialSplitBvhBuildTask<Bvh, Primitive, BinCount>;
    using Reference = typename BuildTask::ReferenceType;

    using ParentBuilder = TopDownBuilder<Bvh, BuildTask>;
    using ParentBuilder::bvh;
    using ParentBuilder::run_task;

public:
    using ParentBuilder::max_depth;
    using ParentBuilder::max_leaf_size;
    using SahBasedAlgorithm<Bvh>::traversal_cost;

    /// Number of spatial binning passes that are run in order to
    /// find a spatial split. This brings additional accuracy without
    /// increasing the number of bins.
    size_t binning_pass_count = 2;

    SpatialSplitBvhBuilder(Bvh& bvh)
        : ParentBuilder(bvh)
    {}

    size_t build(
        const BoundingBox<Scalar>& global_bbox,
        const Primitive* primitives,
        const BoundingBox<Scalar>* bboxes,
        const Vector3<Scalar>* centers,
        size_t primitive_count,
        Scalar alpha = Scalar(1e-5),
        Scalar split_factor = Scalar(0.5))
    {
        size_t max_reference_count = primitive_count + primitive_count * split_factor;
        size_t reference_count = 0;

        std::unique_ptr<size_t[]> primitive_indices_copy;
        std::unique_ptr<typename Bvh::Node[]> nodes_copy;

        bvh.nodes = std::make_unique<typename Bvh::Node[]>(2 * max_reference_count + 1);
        bvh.primitive_indices = std::make_unique<size_t[]>(max_reference_count); 

        auto accumulated_bboxes = std::make_unique<BoundingBox<Scalar>[]>(max_reference_count);
        auto references         = std::make_unique<Reference[]>(max_reference_count);

        // Compute the spatial split threshold, as specified in the original publication
        auto spatial_threshold = alpha * Scalar(2) * global_bbox.half_area();

        bvh.node_count = 1;
        bvh.nodes[0].bounding_box_proxy() = global_bbox;

        #pragma omp parallel
        {
            #pragma omp for
            for (size_t i = 0; i < primitive_count; ++i) {
                references[i].bbox   = bboxes[i];
                references[i].center = centers[i];
                references[i].primitive_index = i;
            }

            #pragma omp single
            {
                BuildTask first_task(
                    *this,
                    primitives,
                    accumulated_bboxes.get(),
                    references.get(),
                    reference_count,
                    spatial_threshold);
                run_task(first_task, 0, 0, primitive_count, max_reference_count, 0);
            }

            #pragma omp single
            {
                primitive_indices_copy = std::make_unique<size_t[]>(reference_count);
                nodes_copy = std::make_unique<typename Bvh::Node[]>(bvh.node_count);
            }

            #pragma omp for nowait
            for (size_t i = 0; i < reference_count; ++i)
                primitive_indices_copy[i] = bvh.primitive_indices[i];

            #pragma omp for nowait
            for (size_t i = 0; i < bvh.node_count; ++i)
                nodes_copy[i] = bvh.nodes[i];
        }

        std::swap(bvh.nodes, nodes_copy);
        std::swap(bvh.primitive_indices, primitive_indices_copy);
        return reference_count;
    }
};

template <typename Bvh, typename Primitive, size_t BinCount>
class SpatialSplitBvhBuildTask : public TopDownBuildTask {
    using Scalar  = typename Bvh::ScalarType;
    using Builder = TopDownBuilder<Bvh, SpatialSplitBvhBuildTask>;

    struct WorkItem : public TopDownBuildTask::WorkItem {
        size_t split_end;

        WorkItem() = default;
        WorkItem(size_t node_index, size_t begin, size_t end, size_t split_end, size_t depth)
            : TopDownBuildTask::WorkItem(node_index, begin, end, depth), split_end(split_end)
        {}
    };

    struct Reference {
        BoundingBox<Scalar> bbox;
        Vector3<Scalar> center;
        size_t primitive_index;
    };

    struct Bin {
        BoundingBox<Scalar> bbox;
        BoundingBox<Scalar> accumulated_bbox;
        size_t entry;
        size_t exit;
    };

    struct ObjectSplit {
        Scalar cost;
        size_t index;
        int    axis;

        BoundingBox<Scalar> left_bbox;
        BoundingBox<Scalar> right_bbox;

        ObjectSplit(
            Scalar cost = std::numeric_limits<Scalar>::max(),
            size_t index = 1,
            int axis = 0,
            const BoundingBox<Scalar>& left_bbox = BoundingBox<Scalar>::empty(),
            const BoundingBox<Scalar>& right_bbox = BoundingBox<Scalar>::empty())
            : cost(cost), index(index), axis(axis), left_bbox(left_bbox), right_bbox(right_bbox)
        {}
    };

    struct SpatialSplit {
        Scalar cost;
        Scalar position;
        int    axis;

        SpatialSplit(
            Scalar cost = std::numeric_limits<Scalar>::max(),
            Scalar position = 0,
            int axis = 0)
            : cost(cost), position(position), axis(axis)
        {}
    };

    Builder& builder;

    const Primitive*     primitives;
    BoundingBox<Scalar>* accumulated_bboxes;
    Reference*           references;

    size_t& reference_count;
    Scalar  spatial_threshold;

    static constexpr size_t bin_count = BinCount;
    std::array<Bin, bin_count> bins;

    void sort_references_by_center(int axis, size_t begin, size_t end) const {
        // Sort references by the projection of their centers on this axis
        std::sort(references + begin, references + end, [&] (const Reference& a, const Reference& b) {
            return a.center[axis] < b.center[axis];
        });
    }

    ObjectSplit find_object_split(size_t begin, size_t end) const {
        ObjectSplit best_split;
        for (int axis = 0; axis < 3; ++axis) {
            sort_references_by_center(axis, begin, end);

            // Sweep from the right to the left to accumulate bounding boxes
            auto bbox = BoundingBox<Scalar>::empty();
            for (size_t i = end - 1; i > begin; --i) {
                bbox.extend(references[i].bbox);
                accumulated_bboxes[i] = bbox;
            }

            // Sweep from the left to the right to compute the SAH cost
            bbox = BoundingBox<Scalar>::empty();
            for (size_t i = begin; i < end - 1; ++i) {
                bbox.extend(references[i].bbox);
                auto cost = bbox.half_area() * (i + 1 - begin) + accumulated_bboxes[i + 1].half_area() * (end - (i + 1));
                if (cost < best_split.cost)
                    best_split = ObjectSplit(cost, i + 1, axis, bbox, accumulated_bboxes[i + 1]);
            }
        }
        return best_split;
    }

    std::pair<WorkItem, WorkItem> allocate_children(
        Bvh& bvh,
        const WorkItem& item,
        typename Bvh::Node& parent,
        size_t right_begin, size_t right_end,
        const BoundingBox<Scalar>& left_bbox,
        const BoundingBox<Scalar>& right_bbox)
    {
        // Allocate two nodes for the children
        size_t first_child;
        #pragma omp atomic capture
        { first_child = bvh.node_count; bvh.node_count += 2; }

        auto& left  = bvh.nodes[first_child + 0];
        auto& right = bvh.nodes[first_child + 1];
        parent.first_child_or_primitive = first_child;
        parent.primitive_count          = 0;
        parent.is_leaf                  = false;
                
        left.bounding_box_proxy()  = left_bbox;
        right.bounding_box_proxy() = right_bbox;

        // Allocate split space for the two children based on their SAH cost.
        // This assumes that reference ranges look like this:
        // - [item.begin...right_begin[ is the range of references on the left,
        // - [right_begin...right_end[ is the range of references on the right,
        // - [right_end...item.split_end[ is the free split space
        size_t remaining_split_count = item.split_end - right_end;
        auto left_cost  = left_bbox.half_area() * (right_begin - item.begin);
        auto right_cost = right_bbox.half_area() * (right_end - right_begin);
        size_t left_split_count = remaining_split_count * Scalar(left_cost / (left_cost + right_cost));

        // Move references of the right child to leave some split space for the left one 
        if (left_split_count > 0)
            std::move_backward(references + right_begin, references + right_end, references + right_end + left_split_count);

        size_t left_end = right_begin;
        right_begin += left_split_count;
        right_end   += left_split_count;
        return std::make_pair(
            WorkItem(first_child + 0, item.begin,  left_end,  right_begin,    item.depth + 1),
            WorkItem(first_child + 1, right_begin, right_end, item.split_end, item.depth + 1));
    }

    std::pair<WorkItem, WorkItem> apply_object_split(
        Bvh& bvh,
        typename Bvh::Node& parent,
        const ObjectSplit& split,
        const WorkItem& item)
    {
        // Sort references again if the axis is not the last sorted axis
        auto left_bbox  = split.left_bbox; 
        auto right_bbox = split.right_bbox; 
        if (split.axis != 2) {
            sort_references_by_center(split.axis, item.begin, item.end);

            // Because of the fact that we are sorting again, it is possible
            // that we do not get exactly the same bounding boxes as when
            // the split was first found. To make sure we have the right
            // bounds, we need to recompute them here.
            left_bbox = BoundingBox<Scalar>::empty(); 
            for (size_t i = item.begin; i < split.index; ++i)
                left_bbox.extend(references[i].bbox);
            right_bbox = BoundingBox<Scalar>::empty(); 
            for (size_t i = split.index; i < item.end; ++i)
                right_bbox.extend(references[i].bbox);
        }

        return allocate_children(bvh, item, parent, split.index, item.end, left_bbox, right_bbox);
    }

    std::optional<std::pair<Scalar, Scalar>>
    run_binning_pass(SpatialSplit& split, int axis, size_t begin, size_t end, Scalar min, Scalar max) {
        for (size_t i = 0; i < bin_count; ++i) {
            bins[i].bbox = BoundingBox<Scalar>::empty();
            bins[i].entry = 0;
            bins[i].exit  = 0;
        }

        // Split primitives and add the bounding box of the fragments to the bins
        auto bin_size = (max - min) / bin_count;
        auto inv_size = Scalar(1) / bin_size;
        for (size_t i = begin; i < end; ++i) {
            auto& reference = references[i];
            auto first_bin = std::min(bin_count - 1, size_t(std::max(Scalar(0), inv_size * (reference.bbox.min[axis] - min))));
            auto last_bin  = std::min(bin_count - 1, size_t(std::max(Scalar(0), inv_size * (reference.bbox.max[axis] - min))));
            auto current_bbox = reference.bbox;
            for (size_t j = first_bin; j < last_bin; ++j) {
                auto [left_bbox, right_bbox] = primitives[references[i].primitive_index].split(axis, min + (j + 1) * bin_size);
                bins[j].bbox.extend(left_bbox.shrink(current_bbox));
                current_bbox.shrink(right_bbox);
            }
            bins[last_bin].bbox.extend(current_bbox);
            bins[first_bin].entry++;
            bins[last_bin].exit++;
        }

        // Accumulate bounding boxes
        auto current_bbox = BoundingBox<Scalar>::empty();
        for (size_t i = bin_count; i > 0; --i)
            bins[i - 1].accumulated_bbox = current_bbox.extend(bins[i - 1].bbox);

        // Sweep and compute SAH cost
        size_t left_count = 0, right_count = end - begin;
        current_bbox = BoundingBox<Scalar>::empty();
        bool found = false;
        for (size_t i = 0; i < bin_count - 1; ++i) {
            left_count  += bins[i].entry;
            right_count -= bins[i].exit;
            current_bbox.extend(bins[i].bbox);

            auto cost = left_count * current_bbox.half_area() + right_count * bins[i + 1].accumulated_bbox.half_area();
            if (cost < split.cost) {
                split.cost = cost;
                split.axis = axis;
                split.position = min + (i + 1) * bin_size;
                found = true;
            }
        }

        return found ? std::make_optional(std::make_pair(split.position - bin_size, split.position + bin_size)) : std::nullopt;
    }

    SpatialSplit find_spatial_split(const BoundingBox<Scalar>& node_bbox, size_t begin, size_t end, size_t binning_pass_count) {
        SpatialSplit split;
        for (int axis = 0; axis < 3; ++axis) {
            auto min = node_bbox.min[axis];
            auto max = node_bbox.max[axis];
            // Run several binning passes to get the best possible split
            for (size_t pass = 0; pass < binning_pass_count; ++pass) {
                auto next_bounds = run_binning_pass(split, axis, begin, end, min, max);
                if (next_bounds)
                    std::tie(min, max) = *next_bounds;
                else
                    break;
            }
        } 
        return split;
    }

    std::pair<WorkItem, WorkItem> apply_spatial_split(
        Bvh& bvh,
        typename Bvh::Node& parent,
        const SpatialSplit& split,
        const WorkItem& item)
    {
        size_t left_end    = item.begin;
        size_t right_begin = item.end;
        size_t right_end   = item.end;

        auto left_bbox  = BoundingBox<Scalar>::empty();
        auto right_bbox = BoundingBox<Scalar>::empty();

        // Partition references such that:
        // - [item.begin...left_end[ is on the left,
        // - [left_end...right_begin[ is in between,
        // - [right_begin...item.end[ is on the right
        for (size_t i = item.begin; i < right_begin;) {
            auto& bbox = references[i].bbox;
            if (bbox.max[split.axis] <= split.position) {
                left_bbox.extend(bbox);
                std::swap(references[i++], references[left_end++]);
            } else if (bbox.min[split.axis] >= split.position) {
                right_bbox.extend(bbox);
                std::swap(references[i], references[--right_begin]);
            } else {
                i++;
            }
        }

        // Handle straddling references
        while (left_end < right_begin) {
            auto reference = references[left_end];
            auto [left_primitive_bbox, right_primitive_bbox] =
                primitives[reference.primitive_index].split(split.axis, split.position);
            left_primitive_bbox .shrink(reference.bbox);
            right_primitive_bbox.shrink(reference.bbox);

            size_t left_count  = left_end  - item.begin;
            size_t right_count = right_end - right_begin;

            // Make sure there is enough space to split that reference
            if (item.split_end - right_end > 0) {
                left_bbox .extend(left_primitive_bbox);
                right_bbox.extend(right_primitive_bbox);
                references[right_end++] = Reference {
                    right_primitive_bbox,
                    right_primitive_bbox.center(),
                    reference.primitive_index
                };
                references[left_end++] = Reference {
                    left_primitive_bbox,
                    left_primitive_bbox.center(),
                    reference.primitive_index
                };
            } else if (left_count < right_count) {
                left_bbox.extend(reference.bbox);
                left_end++;
            } else {
                right_bbox.extend(reference.bbox);
                std::swap(references[--right_begin], references[left_end]);
            }
        } 

        assert(left_end == right_begin);
        assert(right_end < item.split_end);
        return allocate_children(bvh, item, parent, right_begin, right_end, left_bbox, right_bbox);
    }

public:
    using ReferenceType = Reference;
    using WorkItemType  = WorkItem;

    SpatialSplitBvhBuildTask(
        Builder& builder,
        const Primitive* primitives,
        BoundingBox<Scalar>* accumulated_bboxes,
        Reference* references,
        size_t& reference_count,
        Scalar spatial_threshold)
        : builder(builder)
        , primitives(primitives)
        , accumulated_bboxes(accumulated_bboxes)
        , references(references) 
        , reference_count(reference_count)
        , spatial_threshold(spatial_threshold)
    {}

    std::optional<std::pair<WorkItem, WorkItem>> build(const WorkItem& item) {
        auto& bvh  = builder.bvh;
        auto& node = bvh.nodes[item.node_index];

        auto make_leaf = [&] (typename Bvh::Node& node, size_t begin, size_t end) {
            size_t primitive_count = end - begin;

            // Reserve space for the primitives
            size_t first_primitive;
            #pragma omp atomic capture
            { first_primitive = reference_count; reference_count += primitive_count; }

            // Copy the primitives indices from the references to the BVH
            for (size_t i = 0; i < primitive_count; ++i)
                bvh.primitive_indices[first_primitive + i] = references[begin + i].primitive_index;
            node.first_child_or_primitive = first_primitive;
            node.primitive_count          = primitive_count;
            node.is_leaf                  = true;
        };

        if (item.work_size() <= 1 || item.depth >= builder.max_depth) {
            make_leaf(node, item.begin, item.end);
            return std::nullopt;
        }

        ObjectSplit best_object_split = find_object_split(item.begin, item.end);

        // Find a spatial split when the size
        SpatialSplit best_spatial_split;
        auto overlap = BoundingBox<Scalar>(best_object_split.left_bbox).shrink(best_object_split.right_bbox).half_area();
        if (overlap > spatial_threshold && item.split_end - item.end > 0) {
            auto binning_pass_count = static_cast<SpatialSplitBvhBuilder<Bvh, Primitive, BinCount>&>(builder).binning_pass_count;
            best_spatial_split = find_spatial_split(node.bounding_box_proxy(), item.begin, item.end, binning_pass_count);
        }

        auto best_cost = std::min(best_spatial_split.cost, best_object_split.cost);
        bool use_spatial_split = best_cost < best_object_split.cost;

        auto traversal_cost = static_cast<SpatialSplitBvhBuilder<Bvh, Primitive, BinCount>&>(builder).traversal_cost;
        auto max_leaf_size  = static_cast<SpatialSplitBvhBuilder<Bvh, Primitive, BinCount>&>(builder).max_leaf_size;

        // Make sure the cost of splitting does not exceed the cost of not splitting
        if (best_cost >= node.bounding_box_proxy().half_area() * (item.work_size() - traversal_cost)) {
            if (item.work_size() > max_leaf_size) {
                // Fallback strategy: median split on the largest axis
                use_spatial_split = false;
                best_object_split.index = (item.begin + item.end) / 2;
                best_object_split.axis  = node.bounding_box_proxy().to_bounding_box().largest_axis();
            } else {
                make_leaf(node, item.begin, item.end);
                return std::nullopt;
            }
        }

        // Apply the (object/spatial) split
        return use_spatial_split
            ? std::make_optional(apply_spatial_split(bvh, node, best_spatial_split, item))
            : std::make_optional(apply_object_split(bvh, node, best_object_split, item));
    }
};

} // namespace bvh

#endif
