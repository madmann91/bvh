#ifndef BVH_TBB_TOP_DOWN_SCHEDULER_H
#define BVH_TBB_TOP_DOWN_SCHEDULER_H

#include <stack>

#define TBB_SUPPRESS_DEPRECATED_MESSAGES 1
#include <tbb/tbb.h>

#include "bvh/top_down_builder_common.h"

namespace bvh::tbb {

/// TBB-based top-down construction algorithm scheduler.
template <typename Builder>
class ParallelTopDownScheduler final : public TopDownScheduler<Builder> {
    using InnerTask = typename TopDownScheduler<Builder>::InnerTask;
    using WorkItem  = typename TopDownScheduler<Builder>::WorkItem;

    class Task {
    public:
        Task(
            ParallelTopDownScheduler& scheduler,
            const InnerTask& inner_task,
            WorkItem&& first_item)
            : scheduler_(scheduler), inner_task_(inner_task), first_item_(std::move(first_item))
        {}

        void operator () () const {
            std::stack<WorkItem> stack;
            stack.emplace(std::move(first_item_));
            while (!stack.empty()) {
                auto top = std::move(stack.top());
                stack.pop();

                if (auto children = inner_task_.run(std::move(top))) {
                    if (children->first.size() < children->second.size())
                        std::swap(children->first, children->second);
                    if (children->second.size() > scheduler_.parallel_threshold)
                        scheduler_.task_group_.run(Task(scheduler_, inner_task_, std::move(children->second)));
                    else
                        stack.emplace(std::move(children->second));
                    stack.emplace(std::move(children->first));
                }
            }
        }

    private:
        ParallelTopDownScheduler& scheduler_;
        InnerTask inner_task_;
        WorkItem first_item_;
    };

public:
    /// Work items whose size is below this threshold execute serially.
    size_t parallel_threshold = 1024;

    void run(InnerTask&& root, WorkItem&& work_item) {
        task_group_.run_and_wait(Task(*this, std::move(root), std::move(work_item)));
    }

private:
    friend class Task;
    ::tbb::task_group task_group_;
};

} // namespace bvh::tbb

#endif
