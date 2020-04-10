#ifndef BVH_TOP_DOWN_BUILDER_HPP
#define BVH_TOP_DOWN_BUILDER_HPP

#include <stack>

namespace bvh {

template <typename Bvh, typename BuildTask>
class TopDownBuilder {
public:
    using WorkItem = typename BuildTask::WorkItem;

    size_t parallel_threshold = 1024;
    Bvh& bvh;

protected:
    TopDownBuilder(Bvh& bvh)
        : bvh(bvh)
    {}

    template <typename... Args>
    void run(BuildTask& task, Args&&... args) {
        std::stack<WorkItem> stack;
        stack.emplace(std::forward<Args&&>(args)...);
        while (!stack.empty()) {
            auto work_item = stack.top();
            stack.pop();

            auto more_work = task.build(work_item);
            if (more_work) {
                auto [first_item, second_item] = *more_work;
                if (first_item.work_size() > second_item.work_size())
                    std::swap(first_item, second_item);

                stack.push(second_item);
                if (first_item.work_size() > parallel_threshold) {
                    BuildTask new_task(task);
                    #pragma omp task firstprivate(new_task, first_item)
                    { run(new_task, first_item); }
                } else {
                    stack.push(first_item);
                }
            }
        }
    }
};

} // namespace bvh

#endif
