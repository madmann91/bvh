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

    /// Creates a thread pool with the given number of threads (a value of 0 tries to autodetect
    /// the number of threads and uses that as a thread count).
    ThreadPool(size_t thread_count = 0) { start(thread_count); }

    ~ThreadPool() {
        wait();
        stop();
        join();
    }

    void push(Task&& fun);
    void wait();

    size_t get_thread_count() const { return threads_.size(); }

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

} // namespace bvh::v2

#endif
