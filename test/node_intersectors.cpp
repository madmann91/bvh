#include <proto/vec.h>
#include <proto/ray.h>
#include <proto/bbox.h>

#include <bvh/bvh.h>
#include <bvh/single_ray_traverser.h>

using Scalar   = float;
using Ray      = proto::Ray<Scalar>;
using Vec3     = proto::Vec3<Scalar>;
using BBox     = proto::BBox<Scalar>;
using Bvh      = bvh::Bvh<Scalar>;

template <typename NodeIntersector>
bool intersect_bvh_node(const Bvh::Node& node, const Ray& ray) {
    NodeIntersector node_intersector(ray);
    auto [tentry, texit] = node_intersector.intersect_node(ray, node);
    return tentry <= texit;
}

int main() {
    Bvh::Node node;
    node.bounds[0] =  -1;
    node.bounds[1] =   1;
    node.bounds[2] =  -1;
    node.bounds[3] =   1;
    node.bounds[4] = 2.1;
    node.bounds[5] = 2.1;

    Ray ray(
        Vec3(0.25, 0.25, 0.0), // origin
        Vec3(0.0, -0.0, 1.0),  // direction
        0.0,                   // minimum distance
        100.0                  // maximum distance
    );

    return
        intersect_bvh_node<bvh::SingleRayTraverser<Bvh>::FastNodeIntersector>(node, ray) &&
        intersect_bvh_node<bvh::SingleRayTraverser<Bvh>::RobustNodeIntersector>(node, ray) ? 0 : 1;
}
