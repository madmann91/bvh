#ifndef BVH_V2_THREAD_POOL_H
#define BVH_V2_THREAD_POOL_H

#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <queue>
#include <functional>

namespace bvh::v2 {

class ThreadPool {
public:
    using Task = std::function<void(size_t)>;

    ThreadPool(size_t thread_count = 0) { start(thread_count); }
    ~ThreadPool() {
        wait();
        stop();
        join();
    }

    void push(Task&& fun);
    void wait();

    size_t get_thread_count() const { return threads_.size(); }
    std::mutex& get_mutex() { return mutex_; }

    template <typename Loop>
    inline void parallel_for(size_t begin, size_t end, const Loop&);

    template <typename T, typename Reduce, typename Join>
    inline T parallel_reduce(size_t begin, size_t end, const T& init, const Reduce&, const Join&);

private:
    static void worker(ThreadPool*, size_t);

    void start(size_t);
    void stop();
    void join();

    int busy_count_ = 0;
    bool should_stop_ = false;
    std::mutex mutex_;
    std::vector<std::thread> threads_;
    std::condition_variable avail_;
    std::condition_variable done_;
    std::queue<Task> tasks_;
};

template <typename Loop>
void ThreadPool::parallel_for(size_t begin, size_t end, const Loop& loop) {
    auto chunk_size = std::max(size_t{1}, (end - begin) / get_thread_count());
    for (size_t i = begin; i < end; i += chunk_size) {
        size_t next = std::min(end, i + chunk_size);
        push([=] (size_t) { loop(i, next); });
    }
    wait();
}

template <typename T, typename Reduce, typename Join>
T ThreadPool::parallel_reduce(
    size_t begin,
    size_t end,
    const T& init,
    const Reduce& reduce,
    const Join& join)
{
    auto chunk_size = std::max(size_t{1}, (end - begin) / get_thread_count());
    std::vector<T> per_thread_result(get_thread_count(), init);
    for (size_t i = begin; i < end; i += chunk_size) {
        size_t next = std::min(end, i + chunk_size);
        push([&, i, next] (size_t thread_id) {
            auto& result = per_thread_result[thread_id];
            reduce(result, i, next);
        });
    }
    wait();
    for (size_t i = 1; i < get_thread_count(); ++i)
        join(per_thread_result[0], std::move(per_thread_result[i]));
    return per_thread_result[0];
}

} // namespace bvh::v2

#endif
