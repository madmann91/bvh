#include <vector>
#include <iostream>

#include <bvh/bvh.hpp>
#include <bvh/vector.hpp>
#include <bvh/ray.hpp>
#include <bvh/sweep_sah_builder.hpp>
#include <bvh/single_ray_traverser.hpp>

using Scalar      = float;
using Vector3     = bvh::Vector3<Scalar>;
using BoundingBox = bvh::BoundingBox<Scalar>;
using Ray         = bvh::Ray<Scalar>;
using Bvh         = bvh::Bvh<Scalar>;

// Intersectors are used by the traversal algorithm to intersect the primitives
// the BVH. Since the BVH itself has no knowledge of the primitives, this structure
// does the role of a proxy between the traversal algorithm and the primitive data.
// It can also be used to cache precomputed intersection data, so as to speed up
// primitive intersection.
struct Intersector {
    // Required type: result of the intersection function
    struct Result {
        // More members can be added here
        int dummy = 42;

        // Required member: distance along the ray
        Scalar distance() {
            return std::numeric_limits<Scalar>::max();
        }
    };

    // Required member: intersect the primitive at index `bvh.primitive_indices[index]`
    std::optional<Result> intersect(size_t /*index*/, const Ray& /*ray*/) const {
        // Note: a common optimization is to reorder the primitives such that
        // there is no need for an indirection through `bvh.primitive_indices`.
        return std::nullopt;
    }

    // Required member: flag to indicate whether this intersector should stop at the first intersection
    static constexpr bool any_hit = false;
};

int main() {
    // The input of the BVH construction algorithm is just bounding boxes and centers
    std::vector<BoundingBox> bboxes;
    std::vector<Vector3> centers;

    // Fill the bboxes and centers with the proper values taken from the geometry of interest
    bboxes.emplace_back(Vector3(-1, -1, -1), Vector3(1, 1, 1));
    centers.emplace_back(Scalar(0), Scalar(0), Scalar(0));

    assert(bboxes.size() == centers.size());

    // Compute the union of all the bounding boxes
    auto global_bbox = bvh::compute_bounding_boxes_union(bboxes.data(), bboxes.size());

    Bvh bvh;

    bvh::SweepSahBuilder<Bvh> builder(bvh);
    builder.build(global_bbox, bboxes.data(), centers.data(), bboxes.size());

    Intersector intersector;
    bvh::SingleRayTraverser<Bvh> traverser(bvh);

    // Setup the ray (see above for an example)
    Ray ray(Vector3(0.0), Vector3(1.0), 0, 1);
    auto hit = traverser.traverse(ray, intersector);
    if (hit) {
        auto dummy_value = hit->dummy;

        // Do something with the intersection here
        // ...

        std::cout << "Dummy value: " << dummy_value << std::endl;
    }
    return 0;
}
