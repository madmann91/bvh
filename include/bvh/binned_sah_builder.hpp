#ifndef BVH_BINNED_SAH_BUILDER_HPP
#define BVH_BINNED_SAH_BUILDER_HPP

#include <stack>

#include "bvh/bvh.hpp"
#include "bvh/bounding_box.hpp"

namespace bvh {

template <typename Bvh, size_t BinCount>
class BinnedSahBuilder {
public:
    BinnedSahBuilder(Bvh* bvh)
        : bvh(bvh)
    {}

private:
    using Scalar = typename Bvh::ScalarType;

    struct Bin {
        BoundingBox<Scalar> bbox;
        size_t primitive_count;
        Scalar right_cost;
    };

    static constexpr size_t bin_count = BinCount;
    size_t parallel_threshold = 1024;
    Bvh* bvh = nullptr;

    struct BuildTask {
        struct WorkItem {
            size_t node_index;
            size_t begin;
            size_t end;
            size_t depth;

            WorkItem() = default;
            WorkItem(size_t node_index, size_t begin, size_t end, size_t depth)
                : node_index(node_index), begin(begin), end(end), depth(depth)
            {}

            size_t work_size() const { return end - begin; }
        };

        BinnedSahBuilder* builder;

        const BoundingBox<Scalar>* bboxes;
        const Vector3<Scalar>* centers;

        BuildTask(BinnedSahBuilder* builder, const BoundingBox<Scalar>* bboxes, const Vector3<Scalar>* centers)
            : builder(builder), bboxes(bboxes), centers(centers)
        {}

        std::optional<std::pair<WorkItem, WorkItem>> build(const WorkItem& item) {
            auto  bvh  = builder->bvh;
            auto& node = bvh->nodes[item.node_index];

            auto make_leaf = [] (typename Bvh::Node& node, size_t begin, size_t end) {
                node.first_child_or_primitive = begin;
                node.primitive_count          = end - begin;
                node.is_leaf                  = true;
            };

            if (item.work_size() <= 1 || item.depth >= Bvh::max_depth()) {
                make_leaf(node, item.begin, item.end);
                return std::nullopt;
            }

            auto primitive_indices = bvh->primitive_indices.get();

            // Compute the bounding box of the centers of the primitives in this node
            auto center_bbox = BoundingBox<Scalar>::empty();
            for (size_t i = item.begin; i < item.end; ++i)
                center_bbox.extend(centers[primitive_indices[i]]);

            size_t best_split[3] = { 0, 0, 0 };
            Scalar best_cost[3] = {
                std::numeric_limits<Scalar>::max(),
                std::numeric_limits<Scalar>::max(),
                std::numeric_limits<Scalar>::max()
            };
            std::array<Bin, bin_count> bins_per_axis[3];

            auto inverse = center_bbox.diagonal().inverse() * Scalar(bin_count);
            auto base    = -center_bbox.min * inverse;
            auto bin_index = [=] (const Vector3<Scalar>& center, int axis) {
                return std::min(size_t(center[axis] * inverse[axis] + base[axis]), size_t(bin_count - 1));
            };

            #pragma omp taskloop if (item.work_size() > builder->parallel_threshold) grainsize(1) default(shared)
            for (int axis = 0; axis < 3; ++axis) {
                auto& bins = bins_per_axis[axis];

                // Setup bins
                for (auto& bin : bins) {
                    bin.bbox = BoundingBox<Scalar>::empty();
                    bin.primitive_count = 0;
                }

                // Fill bins
                for (size_t i = item.begin; i < item.end; ++i) {
                    auto primitive_index = primitive_indices[i];
                    Bin& bin = bins[bin_index(centers[primitive_index], axis)];
                    bin.primitive_count++;
                    bin.bbox.extend(bboxes[primitive_index]);
                }

                // Right sweep to compute partial SAH
                auto   current_bbox  = BoundingBox<Scalar>::empty();
                size_t current_count = 0;
                for (size_t i = bin_count - 1; i > 0; --i) {
                    current_bbox.extend(bins[i].bbox);
                    current_count += bins[i].primitive_count;
                    bins[i].right_cost = current_bbox.half_area() * current_count;
                }

                // Left sweep to compute full cost and find minimum
                current_bbox  = BoundingBox<Scalar>::empty();
                current_count = 0;
                for (size_t i = 0; i < bin_count - 1; ++i) {
                    current_bbox.extend(bins[i].bbox);
                    current_count += bins[i].primitive_count;
                    auto cost = current_bbox.half_area() * current_count + bins[i + 1].right_cost;
                    if (cost < best_cost[axis]) {
                        best_split[axis] = i + 1;
                        best_cost[axis]  = cost;
                    }
                }
            }

            int best_axis = 0;
            if (best_cost[0] > best_cost[1])
                best_axis = 1;
            if (best_cost[best_axis] > best_cost[2])
                best_axis = 2;

            size_t total_primitives = item.end - item.begin;
            Scalar half_total_area  = node.bounding_box_proxy().half_area();

            // Check that the split is useful
            if (best_split[best_axis] != 0 && best_cost[best_axis] + bvh->traversal_cost * half_total_area < total_primitives * half_total_area) {
                // Split primitives according to split position
                size_t begin_right = std::partition(primitive_indices + item.begin, primitive_indices + item.end, [&] (size_t i) {
                    return bin_index(centers[i], best_axis) < best_split[best_axis];
                }) - primitive_indices;

                // Check that the split does not leave one side empty
                if (begin_right > item.begin && begin_right < item.end) {
                    // Allocate two nodes
                    size_t left_index;
                    #pragma omp atomic capture
                    { left_index = bvh->node_count; bvh->node_count += 2; }
                    auto& left  = bvh->nodes[left_index + 0];
                    auto& right = bvh->nodes[left_index + 1];
                    node.first_child_or_primitive = left_index;
                    node.primitive_count          = 0;
                    node.is_leaf                  = false;
                    
                    // Compute the bounding boxes of each node
                    auto& bins = bins_per_axis[best_axis];
                    auto left_bbox  = BoundingBox<Scalar>::empty();
                    auto right_bbox = BoundingBox<Scalar>::empty();
                    for (size_t i = 0; i < best_split[best_axis]; ++i)
                        left_bbox.extend(bins[i].bbox);
                    for (size_t i = best_split[best_axis]; i < bin_count; ++i)
                        right_bbox.extend(bins[i].bbox);
                    left.bounding_box_proxy()  = left_bbox;
                    right.bounding_box_proxy() = right_bbox;

                    // Return new work items
                    WorkItem first_item (left_index + 0, item.begin, begin_right, item.depth + 1);
                    WorkItem second_item(left_index + 1, begin_right, item.end,   item.depth + 1);
                    return std::make_optional(std::make_pair(first_item, second_item));
                }
            }

            make_leaf(node, item.begin, item.end);
            return std::nullopt;
        }

        template <typename... Args>
        void run(Args&&... args) {
            std::stack<WorkItem> stack;
            stack.emplace(std::forward<Args&&>(args)...);
            while (!stack.empty()) {
                auto work_item = stack.top();
                stack.pop();

                auto more_work = build(work_item);
                if (more_work) {
                    auto [first_item, second_item] = *more_work;
                    if (first_item.work_size() > second_item.work_size())
                        std::swap(first_item, second_item);

                    stack.push(second_item);
                    if (first_item.work_size() > builder->parallel_threshold) {
                        BuildTask task(*this);
                        #pragma omp task firstprivate(task)
                        { task.run(first_item); }
                    } else {
                        stack.push(first_item);
                    }
                }
            }
        }
    };

public:
    void build(const BoundingBox<Scalar>* bboxes, const Vector3<Scalar>* centers, size_t primitive_count) {
        // Allocate buffers
        bvh->nodes.reset(new typename Bvh::Node[2 * primitive_count + 1]);
        bvh->primitive_indices.reset(new size_t[primitive_count]);

        // Initialize root node
        auto root_bbox = BoundingBox<Scalar>::empty();
        bvh->node_count = 1;

        #pragma omp parallel
        {
            #pragma omp declare reduction \
                (bbox_extend:BoundingBox<Scalar>:omp_out.extend(omp_in)) \
                initializer(omp_priv = BoundingBox<Scalar>::empty())

            #pragma omp for reduction(bbox_extend: root_bbox)
            for (size_t i = 0; i < primitive_count; ++i) {
                root_bbox.extend(bboxes[i]);
                bvh->primitive_indices[i] = i;
            }

            #pragma omp single
            {
                bvh->nodes[0].bounding_box_proxy() = root_bbox;
                BuildTask(this, bboxes, centers).run(0, 0, primitive_count, 0);
            }
        }
    }
};

} // namespace bvh

#endif
