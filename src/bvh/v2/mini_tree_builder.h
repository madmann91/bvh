#ifndef BVH_V2_MINI_TREE_BUILDER_H
#define BVH_V2_MINI_TREE_BUILDER_H

#include "bvh/v2/sweep_sah_builder.h"

#include <stack>
#include <tuple>
#include <algorithm>
#include <optional>
#include <numeric>
#include <cassert>

namespace bvh::v2 {

/// Multi-threaded top-down builder that partitions primitives using a grid. Multiple instances
/// of a single-threaded builder are run in parallel on that partition, generating many small
/// trees. Finally, a top-level tree is built on these smaller trees to form the final BVH.
/// This builder is inspired by
/// "Rapid Bounding Volume Hierarchy Generation using Mini Trees", by P. Ganestam et al.
template <typename Node, typename MortonCode = uint32_t>
class MiniTreeBuilder {
    using Scalar = typename Node::Scalar;
    using Vec  = bvh::v2::Vec<Scalar, Node::dimension>;
    using BBox = bvh::v2::BBox<Scalar, Node::dimension>;

public:
    struct Config : SweepSahBuilder<Node>::Config {
        /// Threshold on the area of a mini-tree node above which it is pruned, expressed in
        /// fraction of the area of bounding box around the entire set of primitives.
        Scalar pruning_area_ratio = static_cast<Scalar>(0.01);

        /// Minimum number of primitives per parallel task.
        size_t parallel_threshold = 1024;

        /// Number of bins used to split the workload horizontally.
        size_t bin_count = 4096;
    };

    Bvh<Node> build(
        ThreadPool& thread_pool,
        const BBox* bboxes,
        const Vec* centers,
        size_t prim_count,
        const Config& config = {})
    {
        if (prim_count <= config.parallel_threshold)
            return SweepSahBuilder<Node>().build(bboxes, centers, prim_count, config);

        config_ = &config;
        thread_pool_ = &thread_pool;
        bboxes_ = bboxes;
        centers_ = centers;

        log2_bin_count_ = round_up_log2(config.bin_count);
        bin_count_ = size_t{1} << log2_bin_count_;

        auto mini_trees = build_mini_trees(prim_count);
        auto pruned_trees = prune_mini_trees(std::move(mini_trees));
        return build_top_bvh(pruned_trees);
    }

private:
    friend struct BuildTask;

    struct Bin {
        std::vector<size_t> ids;

        BVH_ALWAYS_INLINE size_t size() const { return ids.size(); }
        BVH_ALWAYS_INLINE void add(size_t id) { ids.push_back(id); }

        BVH_ALWAYS_INLINE void merge(Bin&& other) {
            if (ids.empty())
                ids = std::move(other.ids);
            else
                ids.insert(ids.end(), other.ids.begin(), other.ids.end());
        }
    };

    struct LocalBins {
        std::vector<Bin> bins;

        LocalBins() = default;
        BVH_ALWAYS_INLINE LocalBins(size_t bin_count)
            : bins(bin_count)
        {}

        BVH_ALWAYS_INLINE Bin& operator [] (size_t i) { return bins[i]; }
        BVH_ALWAYS_INLINE const Bin& operator [] (size_t i) const { return bins[i]; }

        BVH_ALWAYS_INLINE void merge(LocalBins&& other) {
            if (bins.empty())
                bins = std::move(other.bins);
            else if (!other.bins.empty()) {
                bins.resize(std::max(bins.size(), other.bins.size()));
                for (size_t i = 0; i < bins.size(); ++i)
                    bins[i].merge(std::move(other[i]));
            }
        }
    };

    struct BuildTask {
        MiniTreeBuilder* builder;
        std::vector<Bvh<Node>>& mini_trees;
        std::vector<size_t> prim_ids;

        std::vector<BBox> bboxes;
        std::vector<Vec> centers;

        BuildTask(
            MiniTreeBuilder* builder,
            std::vector<Bvh<Node>>& mini_trees,
            std::vector<size_t>&& prim_ids)
            : builder(builder)
            , mini_trees(mini_trees)
            , prim_ids(std::move(prim_ids))
        {}

        BVH_ALWAYS_INLINE void run() {
            // Make sure that rebuilds produce the same BVH
            std::sort(prim_ids.begin(), prim_ids.end());

            // Extract bounding boxes and centers for this set of primitives
            bboxes.resize(prim_ids.size());
            centers.resize(prim_ids.size());
            for (size_t i = 0; i < prim_ids.size(); ++i) {
                bboxes[i] = builder->bboxes_[prim_ids[i]];
                centers[i] = builder->centers_[prim_ids[i]];
            }

            auto bvh = SweepSahBuilder<Node>().build(
                bboxes.data(),
                centers.data(),
                prim_ids.size(),
                *builder->config_);

            // Permute primitive indices so that they index the proper set of primitives
            for (size_t i = 0; i < bvh.prim_ids.size(); ++i)
                bvh.prim_ids[i] = prim_ids[bvh.prim_ids[i]];

            std::unique_lock<std::mutex> lock(builder->thread_pool_->get_mutex());
            mini_trees.emplace_back(std::move(bvh));
        }
    };

    const Vec* centers_;
    const BBox* bboxes_;

    size_t log2_bin_count_;
    size_t bin_count_;

    ThreadPool* thread_pool_;
    const Config* config_;

    std::vector<Bvh<Node>> build_mini_trees(size_t prim_count) {
        static constexpr size_t max_grid_size =
            size_t{1} << (std::numeric_limits<MortonCode>::digits / Node::dimension);

        // Compute the bounding box of all centers
        auto center_bbox = thread_pool_->parallel_reduce(0, prim_count, BBox::make_empty(),
            [this] (size_t begin, size_t end) {
                auto bbox = BBox::make_empty();
                for (size_t i = begin; i < end; ++i)
                    bbox.extend(centers_[i]);
                return bbox;
            },
            [] (BBox& bbox, const BBox& other) { bbox.extend(other); });

        auto grid_size = std::min(max_grid_size, size_t{1} << (log2_bin_count_ / Node::dimension));
        auto max_dim = static_cast<MortonCode>(grid_size - 1);
        auto grid_scale = Vec(static_cast<Scalar>(grid_size)) * safe_inverse(center_bbox.get_diagonal());
        auto grid_offset = -center_bbox.min * grid_scale;

        // Place primitives in bins
        auto bins = thread_pool_->parallel_reduce(0, prim_count, LocalBins{}, 
            [&] (size_t begin, size_t end) {
                LocalBins local_bins(bin_count_);
                auto bin_mask = make_bitmask<size_t>(log2_bin_count_);
                for (size_t i = begin; i < end; ++i) {
                    auto p = fast_mul_add(centers_[i], grid_scale, grid_offset);
                    auto x = std::min(max_dim, static_cast<MortonCode>(std::max(p[0], static_cast<Scalar>(0.))));
                    auto y = std::min(max_dim, static_cast<MortonCode>(std::max(p[1], static_cast<Scalar>(0.))));
                    auto z = std::min(max_dim, static_cast<MortonCode>(std::max(p[2], static_cast<Scalar>(0.))));
                    local_bins[morton_encode(x, y, z) & bin_mask].add(i);
                }
                return local_bins;
            },
            [&] (LocalBins& result, LocalBins&& other) { result.merge(std::move(other)); });

        // Iterate over bins to collect groups of primitives and build BVHs over them in parallel
        std::vector<Bvh<Node>> mini_trees;
        for (size_t i = 0, j = 0; i < bin_count_; ++i) {
            if (i != j)
                bins[j].merge(std::move(bins[i]));
            if (bins[j].size() >= config_->parallel_threshold || i == bin_count_ - 1) {
                auto task = new BuildTask(this, mini_trees, std::move(bins[j].ids));
                thread_pool_->push([task] { task->run(); delete task; });
                j = i + 1;
            }
        }
        thread_pool_->wait();

        return mini_trees;
    }

    std::vector<Bvh<Node>> prune_mini_trees(std::vector<Bvh<Node>>&& mini_trees) {
        // Compute the area threshold based on the area of the entire set of primitives
        auto avg_area = static_cast<Scalar>(0.);
        for (auto& mini_tree : mini_trees)
            avg_area += mini_tree.get_root().get_bbox().get_half_area();
        auto threshold = (avg_area * config_->pruning_area_ratio) / static_cast<Scalar>(mini_trees.size());

        // Cull nodes whose area is above the threshold
        std::stack<size_t> stack;
        std::vector<std::pair<size_t, size_t>> pruned_roots;
        for (size_t i = 0; i < mini_trees.size(); ++i) {
            stack.push(0);
            auto& mini_tree = mini_trees[i];
            while (!stack.empty()) {
                auto node_id = stack.top();
                auto& node = mini_tree.nodes[node_id];
                stack.pop();
                if (node.get_bbox().get_half_area() < threshold || node.is_leaf()) {
                    pruned_roots.emplace_back(i, node_id);
                } else {
                    stack.push(node.index.first_id);
                    stack.push(node.index.first_id + 1);
                }
            }
        }

        // Extract the BVHs rooted at the previously computed indices
        std::vector<Bvh<Node>> pruned_trees(pruned_roots.size());
        thread_pool_->parallel_for(0, pruned_roots.size(),
            [&] (size_t begin, size_t end) {
                for (size_t i = begin; i < end; ++i) {
                    if (pruned_roots[i].second == 0)
                        pruned_trees[i] = std::move(mini_trees[pruned_roots[i].first]);
                    else
                        pruned_trees[i] = mini_trees[pruned_roots[i].first]
                            .extract_bvh(pruned_roots[i].second);
                }
            });
        return pruned_trees;
    }

    Bvh<Node> build_top_bvh(std::vector<Bvh<Node>>& mini_trees) {
        // Build a BVH using the mini trees as leaves
        std::vector<Vec> centers(mini_trees.size());
        std::vector<BBox> bboxes(mini_trees.size());
        for (size_t i = 0; i < mini_trees.size(); ++i) {
            bboxes[i] = mini_trees[i].get_root().get_bbox();
            centers[i] = bboxes[i].get_center();
        }
        typename SweepSahBuilder<Node>::Config config;
        config.max_leaf_size = 1;
        auto bvh = SweepSahBuilder<Node>().build(bboxes.data(), centers.data(), mini_trees.size(), config);

        // Compute the offsets to apply to primitive and node indices
        std::vector<size_t> node_offsets(mini_trees.size());
        std::vector<size_t> prim_offsets(mini_trees.size());
        size_t node_count = bvh.nodes.size();
        size_t prim_count = 0;
        for (size_t i = 0; i < mini_trees.size(); ++i) {
            node_offsets[i] = node_count - 1; // Skip root node
            prim_offsets[i] = prim_count;
            node_count += mini_trees[i].nodes.size() - 1; // idem
            prim_count += mini_trees[i].prim_ids.size();
        }

        // Helper function to copy and fix the child/primitive index of a node
        auto copy_node = [&] (size_t i, Node& dst_node, const Node& src_node) {
            dst_node = src_node;
            dst_node.index.first_id += src_node.is_leaf() ? prim_offsets[i] : node_offsets[i];
        };

        // Make the leaves of the top BVH point to the right internal nodes
        for (auto& node : bvh.nodes) {
            if (!node.is_leaf())
                continue;
            assert(node.index.prim_count == 1);
            size_t tree_id = bvh.prim_ids[node.index.first_id];
            copy_node(tree_id, node, mini_trees[tree_id].get_root());
        }

        bvh.nodes.resize(node_count);
        bvh.prim_ids.resize(prim_count);
        thread_pool_->parallel_for(0, mini_trees.size(), 
            [&] (size_t begin, size_t end) {
                for (size_t i = begin; i < end; ++i) {
                    auto& mini_tree = mini_trees[i];

                    // Copy the nodes of the mini tree with the offsets applied, without copying
                    // the root node (since it is already copied to the top-level part of the BVH).
                    for (size_t j = 1; j < mini_tree.nodes.size(); ++j)
                        copy_node(i, bvh.nodes[node_offsets[i] + j], mini_tree.nodes[j]);

                    std::copy(
                        mini_tree.prim_ids.begin(),
                        mini_tree.prim_ids.end(),
                        bvh.prim_ids.begin() + prim_offsets[i]);
                }
            });

        return bvh;
    }
};

} // namespace bvh::v2

#endif
