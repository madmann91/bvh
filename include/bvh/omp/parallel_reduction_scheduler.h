#ifndef BVH_OMP_PARALLEL_REDUCTION_SCHEDULER_H
#define BVH_OMP_PARALLEL_REDUCTION_SCHEDULER_H

#include <type_traits>

namespace bvh::omp {

/// Helper class that performs a reduction in parallel using OpenMP.
class ParallelReductionScheduler {
public:
    template <
        typename T,
        typename Index,
        typename BinaryOp,
        typename UnaryOp,
        std::enable_if_t<std::is_integral_v<Index>, int> = 0>
    static T run(
        Index begin, Index end, T init,
        const BinaryOp& bin_op,
        const UnaryOp& un_op)
    {
        struct Custom {
            BinaryOp bin_op;
            T val;

            Custom(T val, const BinaryOp& bin_op) : val(val), bin_op(bin_op) {}
            Custom(const Custom&) = default;
            void combine(const Custom& other) { val = bin_op(val, other.val); }
        };

        Custom res(init, bin_op);
        #pragma omp declare reduction(ReduceOp:Custom:omp_out.combine(omp_in)) initializer(omp_priv=omp_orig)
        #pragma omp parallel for reduction(ReduceOp: res)
        for (auto i = begin; i < end; ++i)
            res.val = res.bin_op(res.val, un_op(i));
        return res.val;
    }
};

} // namespace bvh::omp

#endif
