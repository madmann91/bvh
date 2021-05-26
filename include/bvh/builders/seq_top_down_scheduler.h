#ifndef BVH_BUILDERS_SEQ_TOP_DOWN_SCHEDULER_H
#define BVH_BUILDERS_SEQ_TOP_DOWN_SCHEDULER_H

#include <stack>

#include "bvh/builders/top_down_scheduler.h"

namespace bvh {
    
template <typename Builder>
class SeqTopDownScheduler : public TopDownScheduler<Builder> {
    using InnerTask = typename TopDownScheduler<Builder>::InnerTask;
    using WorkItem  = typename TopDownScheduler<Builder>::WorkItem;

public:
    void run(InnerTask&& root, WorkItem&& work_item) override {
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
