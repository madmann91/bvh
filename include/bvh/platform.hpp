#ifndef BVH_PLATFORM_HPP
#define BVH_PLATFORM_HPP

#ifdef _OPENMP
#include <omp.h>
#include <cassert>
#define bvh__get_num_threads()        omp_get_num_threads()
#define bvh__get_thread_num()         omp_get_thread_num()
#define bvh__assert_not_in_parallel() assert(omp_get_level() == 0);
#define bvh__assert_in_parallel()     assert(omp_get_level() > 0);
#else
#define bvh__get_num_threads()        1
#define bvh__get_thread_num()         0
#define bvh__assert_not_in_parallel() (void)0
#define bvh__assert_in_parallel()     (void)0
#endif

#if defined(__GNUC__) || defined(__clang__)
#define bvh__restrict__      __restrict
#define bvh__always_inline__ __attribute__((always_inline))
#define bvh__likely(x)       __builtin_expect(x, true)
#define bvh__unlikely(x)     __builtin_expect(x, false)
#elif defined(_MSC_VER)
#define bvh__restrict__      __restrict
#define bvh__always_inline__ __forceinline
#define bvh__likely(x)       x
#define bvh__unlikely(x)     x
#else
#define bvh__restrict__
#define bvh__always_inline__
#define bvh__likely(x)       x
#define bvh__unlikely(x)     x
#endif

#endif
