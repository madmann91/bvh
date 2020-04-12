#ifndef BVH_MORTON_CODE_BASED_BUILDER_HPP
#define BVH_MORTON_CODE_BASED_BUILDER_HPP

#include <algorithm>
#include <memory>
#include <cassert>

#include "bvh/bounding_box.hpp"
#include "bvh/vector.hpp"
#include "bvh/morton.hpp"
#include "bvh/radix_sort.hpp"

namespace bvh {

template <typename Bvh, typename Morton>
class MortonCodeBasedBuilder {
    using Scalar = typename Bvh::ScalarType;

public:
    using MortonType = Morton;

    /// Maximum number of bits available per dimension.
    static constexpr size_t max_bit_count = (sizeof(Morton) * CHAR_BIT) / 3;

    /// Number of bits to use per dimension.
    size_t bit_count = max_bit_count;

    /// Threshold (number of nodes) under which the loops execute serially.
    size_t parallel_threshold = 256;

    /// Threshold (number of primitives) under which the radix sort executes serially.
    size_t radix_sort_parallel_threshold = 1024;

protected:
    using SortedPairs = std::pair<std::unique_ptr<size_t[]>, std::unique_ptr<Morton[]>>;

    SortedPairs sort_primitives_by_morton_code(
        const BoundingBox<Scalar>* bboxes,
        const Vector3<Scalar>* centers,
        size_t primitive_count) const
    {
        assert(bit_count <= max_bit_count);
        auto morton_codes      = std::make_unique<Morton[]>(primitive_count);
        auto primitive_indices = std::make_unique<size_t[]>(primitive_count);

        auto dim = Morton(1) << bit_count;
        auto global_bbox = BoundingBox<Scalar>::empty();

        #pragma omp parallel if (primitive_count > parallel_threshold)
        {
            #pragma omp declare reduction \
                (bbox_extend:BoundingBox<Scalar>:omp_out.extend(omp_in)) \
                initializer(omp_priv = BoundingBox<Scalar>::empty())

            #pragma omp for reduction(bbox_extend: global_bbox)
            for (size_t i = 0; i < primitive_count; ++i)
                global_bbox.extend(bboxes[i]);

            auto world_to_grid = Scalar(dim) * global_bbox.diagonal().inverse();
            auto grid_offset = -global_bbox.min * world_to_grid;

            #pragma omp for
            for (size_t i = 0; i < primitive_count; ++i) {
                auto grid_position = centers[i] * world_to_grid + grid_offset;
                Morton x = std::min(dim - 1, Morton(std::max(grid_position[0], Scalar(0))));
                Morton y = std::min(dim - 1, Morton(std::max(grid_position[1], Scalar(0))));
                Morton z = std::min(dim - 1, Morton(std::max(grid_position[2], Scalar(0))));
                morton_codes[i] = morton_encode(x, y, z);
                primitive_indices[i] = i;
            }
        }

        // Sort primitives by morton code
        {
            auto morton_codes_copy      = std::make_unique<uint32_t[]>(primitive_count);
            auto primitive_indices_copy = std::make_unique<size_t[]>(primitive_count);
            radix_sort(
                morton_codes, primitive_indices,
                morton_codes_copy, primitive_indices_copy,
                primitive_count, bit_count * 3,
                radix_sort_parallel_threshold);
            assert(std::is_sorted(morton_codes.get(), morton_codes.get() + primitive_count));
        }

        return std::make_pair(std::move(primitive_indices), std::move(morton_codes));
    }
};

} // namespace bvh

#endif
