#ifndef BVH_SWEEP_SAH_BUILDER_HPP
#define BVH_SWEEP_SAH_BUILDER_HPP

#include <numeric>
#include <optional>

#include "bvh/bvh.hpp"
#include "bvh/bounding_box.hpp"
#include "bvh/top_down_builder.hpp"
#include "bvh/sah_based_algorithm.hpp"

namespace bvh {

template <typename> class SweepSahBuildTask;

template <typename Bvh>
class SweepSahBuilder :
    public TopDownBuilder<Bvh, SweepSahBuildTask<Bvh>>,
    public SahBasedAlgorithm<Bvh>
{
    using Scalar = typename Bvh::ScalarType;

    using ParentBuilder = TopDownBuilder<Bvh, SweepSahBuildTask<Bvh>>;
    using ParentBuilder::bvh;
    using ParentBuilder::run_task;

    using SahBasedAlgorithm<Bvh>::cost;

public:
    using ParentBuilder::max_depth;
    using SahBasedAlgorithm<Bvh>::traversal_cost;

    SweepSahBuilder(Bvh& bvh)
        : ParentBuilder(bvh)
    {}

    void build(const BoundingBox<Scalar>* bboxes, const Vector3<Scalar>* centers, size_t primitive_count) {
        // Allocate buffers
        bvh.nodes.reset(new typename Bvh::Node[2 * primitive_count + 1]);
        bvh.primitive_indices.reset(new size_t[primitive_count]);
        auto reference_data = std::make_unique<size_t[]>(primitive_count * 2);
        auto cost_data      = std::make_unique<Scalar[]>(primitive_count * 3);

        std::array<Scalar*, 3> costs = {
            cost_data.get(),
            cost_data.get() + primitive_count,
            cost_data.get() + 2 * primitive_count
        };
        std::array<size_t*, 3> references = {
            reference_data.get(),
            reference_data.get() + primitive_count,
            bvh.primitive_indices.get()
        };

        // Initialize root node
        auto root_bbox = BoundingBox<Scalar>::empty();
        bvh.node_count = 1;

        #pragma omp parallel
        {
            #pragma omp for nowait
            for (int axis = 0; axis < 3; ++axis) {
                std::iota(references[axis], references[axis] + primitive_count, 0);
                std::sort(references[axis], references[axis] + primitive_count, [&] (size_t reference1, size_t reference2) {
                    auto center1 = centers[reference1][axis];
                    auto center2 = centers[reference2][axis];
                    return center1 < center2 || (center1 == center2 && reference1 < reference2);
                });
            }

            #pragma omp declare reduction \
                (bbox_extend:BoundingBox<Scalar>:omp_out.extend(omp_in)) \
                initializer(omp_priv = BoundingBox<Scalar>::empty())

            #pragma omp for reduction(bbox_extend: root_bbox)
            for (size_t i = 0; i < primitive_count; ++i)
                root_bbox.extend(bboxes[i]);

            #pragma omp single
            {
                bvh.nodes[0].bounding_box_proxy() = root_bbox;
                SweepSahBuildTask<Bvh> first_task(*this, bboxes, centers, references, costs);
                run_task(first_task, 0, 0, primitive_count, 0);
            }
        }
    }
};

template <typename Bvh>
struct SweepSahBuildTask {
    using Scalar   = typename Bvh::ScalarType;
    using Builder  = TopDownBuilder<Bvh, SweepSahBuildTask>;
    using WorkItem = typename Builder::WorkItem;

    Builder& builder;
    const BoundingBox<Scalar>* bboxes;
    const Vector3<Scalar>* centers;

    std::array<size_t*, 3> references;
    std::array<Scalar*, 3> costs;

    std::pair<Scalar, size_t> find_split(int axis, size_t begin, size_t end) {
        auto bbox = BoundingBox<Scalar>::empty();
        for (size_t i = end - 1; i > begin; --i) {
            bbox.extend(bboxes[references[axis][i]]);
            costs[axis][i] = bbox.half_area() * (end - i);
        }
        bbox = BoundingBox<Scalar>::empty();
        auto best_split = std::pair<Scalar, size_t>(std::numeric_limits<Scalar>::max(), end);
        for (size_t i = begin; i < end - 1; ++i) {
            bbox.extend(bboxes[references[axis][i]]);
            auto cost = bbox.half_area() * (i + 1 - begin) + costs[axis][i + 1];
            if (cost < best_split.first)
                best_split = std::make_pair(cost, i + 1);
        }
        return best_split;
    }

public:
    SweepSahBuildTask(
        Builder& builder,
        const BoundingBox<Scalar>* bboxes,
        const Vector3<Scalar>* centers,
        const std::array<size_t*, 3>& references,
        const std::array<Scalar*, 3>& costs)
        : builder(builder), bboxes(bboxes), centers(centers), references(references), costs(costs)
    {}

    std::optional<std::pair<WorkItem, WorkItem>> build(const WorkItem& item) {
        auto& bvh  = builder.bvh;
        auto& node = bvh.nodes[item.node_index];

        auto make_leaf = [] (typename Bvh::Node& node, size_t begin, size_t end) {
            node.first_child_or_primitive = begin;
            node.primitive_count          = end - begin;
            node.is_leaf                  = true;
        };

        if (item.work_size() <= 1 || item.depth >= builder.max_depth) {
            make_leaf(node, item.begin, item.end);
            return std::nullopt;
        }

        std::pair<Scalar, size_t> best_splits[3];
        bool should_spawn_tasks = item.work_size() > builder.task_spawn_threshold;

        // Sweep primitives to find the best cost
        #pragma omp taskloop if (should_spawn_tasks) grainsize(1) default(shared)
        for (int axis = 0; axis < 3; ++axis)
            best_splits[axis] = find_split(axis, item.begin, item.end);

        int best_axis = 0;
        if (best_splits[0].first > best_splits[1].first)
            best_axis = 1;
        if (best_splits[best_axis].first > best_splits[2].first)
            best_axis = 2;

        auto traversal_cost = static_cast<SweepSahBuilder<Bvh>&>(builder).traversal_cost;

        // Make sure the cost of splitting does not exceed the cost of not splitting
        if (best_splits[best_axis].first >= node.bounding_box_proxy().half_area() * (item.work_size() - traversal_cost)) {
            make_leaf(node, item.begin, item.end);
            return std::nullopt;
        }

        int other_axis[2] = { (best_axis + 1) % 3, (best_axis + 2) % 3 };
        auto best_split = best_splits[best_axis];
        auto best_reference = references[best_axis][best_split.second];
        auto split_position = centers[best_reference][best_axis];
        auto partition_predicate = [&] (size_t reference) {
            auto position = centers[reference][best_axis];
            return position < split_position || (position == split_position && reference < best_reference);
        };

        auto left_bbox  = BoundingBox<Scalar>::empty();
        auto right_bbox = BoundingBox<Scalar>::empty();

        // Partition reference arrays and compute bounding boxes
        #pragma omp taskgroup
        {
            #pragma omp task if (should_spawn_tasks) default(shared)
            { std::stable_partition(references[other_axis[0]] + item.begin, references[other_axis[0]] + item.end, partition_predicate); }
            #pragma omp task if (should_spawn_tasks) default(shared)
            { std::stable_partition(references[other_axis[1]] + item.begin, references[other_axis[1]] + item.end, partition_predicate); }
            #pragma omp task if (should_spawn_tasks) default(shared)
            {
                for (size_t i = item.begin; i < best_split.second; ++i)
                    left_bbox.extend(bboxes[references[best_axis][i]]);
            }
            #pragma omp task if (should_spawn_tasks) default(shared)
            {
                for (size_t i = item.end - 1; i >= best_split.second; --i)
                    right_bbox.extend(bboxes[references[best_axis][i]]);
            }
        }

        // Allocate space for children
        size_t left_index;
        #pragma omp atomic capture
        { left_index = bvh.node_count; bvh.node_count += 2; }

        auto& left  = bvh.nodes[left_index + 0];
        auto& right = bvh.nodes[left_index + 1];
        node.first_child_or_primitive = left_index;
        node.primitive_count          = 0;
        node.is_leaf                  = false;
                
        left.bounding_box_proxy()  = left_bbox;
        right.bounding_box_proxy() = right_bbox;
        WorkItem first_item (left_index + 0, item.begin, best_split.second, item.depth + 1);
        WorkItem second_item(left_index + 1, best_split.second, item.end,   item.depth + 1);
        return std::make_optional(std::make_pair(first_item, second_item));
    }
};

} // namespace bvh

#endif
