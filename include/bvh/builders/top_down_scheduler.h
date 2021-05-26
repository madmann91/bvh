#ifndef BVH_BUILDERS_TOP_DOWN_SCHEDULER_H
#define BVH_BUILDERS_TOP_DOWN_SCHEDULER_H

namespace bvh {
    
template <typename Builder>
class TopDownScheduler {
public:
    using InnerTask = typename Builder::Task;
    using WorkItem  = typename Builder::WorkItem;

    virtual void run(InnerTask&& task, WorkItem&& first) = 0;
};

} // namespace bvh

#endif
