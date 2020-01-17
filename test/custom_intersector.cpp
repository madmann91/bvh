#include <vector>
#include <iostream>
#include "bvh.hpp"

int main() {
    // The input of the BVH construction algorithm is just bounding boxes and centers
    std::vector<bvh::BBox> bboxes;
    std::vector<bvh::Vec3> centers;

    // Fill the bboxes and centers with the proper values taken from the geometry of interest
    // ...

    assert(bboxes.size() == centers.size());

    // Number of bins to evaluate the SAH.
    // Higher values means a more precise estimate of the SAH,
    // at a higher memory and computation cost. Values of 32 or
    // 64 are good enough for most scenes.
    static constexpr size_t bin_count = 64;

    // Maximum depth of the tree.
    // Increasing this value increases the traversal stack size.
    static constexpr size_t max_depth = 64;

    // The minimum number of primitives necessary to start a new builder task.
    // This number should be high enough to avoid creating too many tasks, but
    // low enough to ensure that there is enough parallelism. A value of 1024
    // is good enough for most primitive types.
    static constexpr size_t parallel_threshold = 1024;

    bvh::BVH<bin_count, max_depth, parallel_threshold> bvh;
    bvh.build(bboxes.data(), centers.data(), bboxes.size());

    // Boolean controlling whether the intersection routine exits
    // immediately after an intersection is found.
    static constexpr bool any_hit = false;

    // Intersectors are used by the traversal algorithm to intersect the primitives
    // the BVH. Since the BVH itself has no knowledge of the primitives, this structure
    // does the role of a proxy between the traversal algorithm and the primitive data.
    // It can also be used to cache precomputed intersection data, so as to speed up
    // primitive intersection.
    struct Intersector {
        // Required type: result of the intersection function
        struct Result {
            // Required member: distance along the ray
            bvh::Scalar distance;
        };

        // Required member: intersect the primitive at index `bvh.primitive_indices[index]`
        std::optional<Result> operator () (size_t index, const bvh::Ray& ray) const {
            // Note: a common optimization is to reorder the primitives such that
            // there is no need for an indirection through `bvh.primitive_indices`.
            return std::nullopt;
        }
    };

    Intersector intersector;

    // Setup the ray (see above for an example)
    bvh::Ray ray(bvh::Vec3(0), bvh::Vec3(1), 0, 1);
    auto hit = bvh.intersect<any_hit>(ray, intersector);
    if (hit) {
        auto primitive_index = hit->first;
        auto intersection = hit->second;

        // Do something with the intersection here
        // ...
    }
    return 0;
}
