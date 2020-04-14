#ifndef BVH_RADIX_SORT_HPP
#define BVH_RADIX_SORT_HPP

#include <memory>
#include <algorithm>
#include <cstddef>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_num_threads() 1
#define omp_get_thread_num() 0
#endif

namespace bvh {

template <size_t BitsPerIteration = 10, typename Key, typename Value>
void radix_sort(
    std::unique_ptr<Key[]>& keys,      std::unique_ptr<Value[]>& values,
    std::unique_ptr<Key[]>& keys_copy, std::unique_ptr<Value[]>& values_copy,
    size_t count, size_t bit_count,
    size_t parallel_threshold)
{
    static constexpr size_t bucket_count = 1 << BitsPerIteration;
    static constexpr Key mask = (Key(1) << BitsPerIteration) - 1;

    // Allocate temporary storage
    std::unique_ptr<size_t[]> per_thread_buckets;

    #pragma omp parallel if (count > parallel_threshold)
    {
        size_t thread_count = omp_get_num_threads();
        size_t thread_id = omp_get_thread_num();

        #pragma omp single
        { per_thread_buckets = std::make_unique<size_t[]>((thread_count + 1) * bucket_count); }

        for (size_t bit = 0; bit < bit_count; bit += BitsPerIteration) {
            auto buckets = &per_thread_buckets[thread_id * bucket_count];
            std::fill(buckets, buckets + bucket_count, 0);

            #pragma omp for
            for (size_t i = 0; i < count; ++i)
                buckets[(keys[i] >> bit) & mask]++;

            #pragma omp for
            for (size_t i = 0; i < bucket_count; i++) {
                // Do a prefix sum of the elements in one bucket over all threads
                size_t count = 0;
                for (size_t j = 0; j < thread_count; ++j) {
                    size_t old_count = count;
                    count += per_thread_buckets[j * bucket_count + i];
                    per_thread_buckets[j * bucket_count + i] = old_count;
                }
                per_thread_buckets[thread_count * bucket_count + i] = count;
            }

            for (size_t i = 0, count = 0; i < bucket_count; ++i) {
                size_t old_count = count;
                count += per_thread_buckets[thread_count * bucket_count + i];
                buckets[i] += old_count;
            }

            #pragma omp for
            for (size_t i = 0; i < count; ++i) {
                size_t j = buckets[(keys[i] >> bit) & mask]++;
                keys_copy[j]   = keys[i];
                values_copy[j] = values[i];
            }

            #pragma omp single
            {
                std::swap(keys_copy, keys);
                std::swap(values_copy, values);
            }
        }
    }
}

} // namespace bvh

#endif
