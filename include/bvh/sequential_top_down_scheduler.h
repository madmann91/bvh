#ifndef BVH_SEQUENTIAL_TOP_DOWN_SCHEDULER_H
#define BVH_SEQUENTIAL_TOP_DOWN_SCHEDULER_H

#include <stack>
#include <execution>

#include "bvh/top_down_builder_common.h"

namespace bvh {
    
/// Top-down scheduler that makes top-down algorithms run serially.
template <typename Builder>
class SequentialTopDownScheduler final : public TopDownScheduler<Builder> {
    using InnerTask = typename TopDownScheduler<Builder>::InnerTask;
    using WorkItem  = typename TopDownScheduler<Builder>::WorkItem;

public:
    static constexpr auto&& execution_policy() { return std::execution::unseq; }

    void run(InnerTask&& root, WorkItem&& work_item) {
        std::stack<WorkItem> stack;
        stack.emplace(std::move(work_item));
        while (!stack.empty()) {
            auto top = std::move(stack.top());
            stack.pop();
            if (auto children = root.run(std::move(top))) {
                if (children->first.size() < children->second.size())
                    std::swap(children->first, children->second);
                stack.emplace(std::move(children->second));
                stack.emplace(std::move(children->first));
            }
        }
    }
};

} // namespace bvh

#endif
