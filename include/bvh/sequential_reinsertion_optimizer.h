#ifndef BVH_OPTIMIZERS_SEQUENTIAL_REINSERTION_OPTIMIZER_H
#define BVH_OPTIMIZERS_SEQUENTIAL_REINSERTION_OPTIMIZER_H

#include <cstddef>
#include <vector>
#include <algorithm>
#include <cassert>
#include <functional>
#include <queue>

#include "bvh/topology_modifier.h"

namespace bvh {

/// Sequential re-insertion optimizer, based on the article:
/// "Fast, Insertion-based Optimization of Bounding Volume Hierarchies", by J. Bittner et al.
/// Can be used over `bvh::ParallelReinsertionOptimizer` when parallelism is not required.
template <typename Bvh>
class SequentialReinsertionOptimizer {
    using Scalar = typename Bvh::Scalar;
    using Node   = typename Bvh::Node;

    struct Candidate {
        size_t index;
        Scalar cost;

        bool operator > (const Candidate& other) const { return cost > other.cost; }
    };

    static std::vector<Candidate> find_candidates(const Bvh& bvh, size_t count) {
        std::vector<Candidate> candidates;
        for (size_t i = 1; i < bvh.nodes.size(); ++i) {
            auto& node = bvh.nodes[i];
            if (node.is_leaf())
                continue;
            auto cost = node.bbox().half_area();
            if (candidates.size() < count) {
                candidates.push_back(Candidate { i, cost });
                std::push_heap(candidates.begin(), candidates.end(), std::greater<>{});
            } else if (cost > candidates.front().cost) {
                // Removes the element with smallest cost and replace it with this candidate
                std::pop_heap(candidates.begin(), candidates.end(), std::greater<>{});
                candidates.back() = Candidate { i, cost };
                std::push_heap(candidates.begin(), candidates.end(), std::greater<>{});
            }
        }
        std::sort_heap(candidates.begin(), candidates.end(), std::greater<>{});
        return candidates;
    }

    static size_t find_reinsertion(const Bvh& bvh, const Node& node) {
        auto best = Candidate { 0, std::numeric_limits<Scalar>::max() };
        auto node_area = node.bbox().half_area();
        std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> queue;
        queue.emplace(Candidate { 0, 0 });
        while (!queue.empty()) {
            auto candidate = queue.top();
            queue.pop();
            if (candidate.cost + node_area >= best.cost)
                break;
            auto& candidate_node = bvh.nodes[candidate.index];
            auto direct_cost = node.bbox().extend(candidate_node.bbox()).half_area();
            auto total_cost  = candidate.cost + direct_cost;
            if (total_cost < best.cost) {
                best.index = candidate.index;
                best.cost = total_cost;
            }
            auto child_cost = total_cost - candidate_node.bbox().half_area();
            if (child_cost + node_area < best.cost) {
                if (!candidate_node.is_leaf()) {
                    queue.emplace(Candidate { candidate_node.first_index + 0, child_cost });
                    queue.emplace(Candidate { candidate_node.first_index + 1, child_cost });
                }
            }
        }
        return best.index;
    }

public:
    static void optimize(TopologyModifier<Bvh>& topo_modifier, size_t iter_count = 3, double batch_ratio = 0.01) {
        size_t batch_size = std::max<size_t>(topo_modifier.bvh.nodes.size() * batch_ratio, 1);
        for (size_t j = 0; j < iter_count; ++j) {
            auto candidates = find_candidates(topo_modifier.bvh, batch_size);
            for (size_t i = candidates.size(); i > 0; --i) {
                auto& candidate = candidates[i - 1];
                auto node = topo_modifier.bvh.nodes[candidate.index];
                topo_modifier.remove_node(candidate.index);
                topo_modifier.insert_node(node, find_reinsertion(topo_modifier.bvh, node));
            }
        }
    }
};

} // namespace bvh

#endif
