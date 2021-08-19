#ifndef BVH_OMP_PARALLEL_TOP_DOWN_SCHEDULER_H
#define BVH_OMP_PARALLEL_TOP_DOWN_SCHEDULER_H

#include <stack>

#include "bvh/top_down_builder_common.h"

namespace bvh::omp {

/// OpenMP-based top-down construction algorithm scheduler.
template <typename Builder>
class ParallelTopDownScheduler final : public TopDownScheduler<Builder> {
    using InnerTask = typename TopDownScheduler<Builder>::InnerTask;
    using WorkItem  = typename TopDownScheduler<Builder>::WorkItem;

    void run_task(const InnerTask& inner_task, WorkItem&& work_item) const {
        std::stack<WorkItem> stack;
        stack.emplace(std::move(work_item));
        while (!stack.empty()) {
            auto top = std::move(stack.top());
            stack.pop();

            if (auto children = inner_task.run(std::move(top))) {
                if (children->first.size() < children->second.size())
                    std::swap(children->first, children->second);
                if (children->second.size() > parallel_threshold) {
                    WorkItem second_child(std::move(children->second));
                    #pragma omp task firstprivate(inner_task, second_child)
                    {
                        run_task(inner_task, std::move(second_child));
                    }
                } else
                    stack.emplace(std::move(children->second));
                stack.emplace(std::move(children->first));
            }
        }
    }

public:
    /// Work items whose size is below this threshold execute serially.
    size_t parallel_threshold = 1024;

    void run(InnerTask&& root, WorkItem&& work_item) {
        #pragma omp parallel
        {
            #pragma omp single
            {
                run_task(root, std::move(work_item));
            }
        }
    }
};

} // namespace bvh::omp

#endif
