#ifndef BVH_TOP_DOWN_SCHEDULER_H
#define BVH_TOP_DOWN_SCHEDULER_H

#include <stack>

namespace bvh {
    
/// Object that controls how a top-down construction algorithm is executed.
/// Custom top-down schedulers can be implemented to support various
/// parallel libraries or frameworks.
template <typename Builder>
class TopDownScheduler {
protected:
    using InnerTask = typename Builder::Task;
    using WorkItem  = typename Builder::WorkItem;

public:
    virtual void run(InnerTask&& root, WorkItem&& work_item) {
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
