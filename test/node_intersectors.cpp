#include <bvh/bvh.hpp>
#include <bvh/vector.hpp>
#include <bvh/ray.hpp>
#include <bvh/node_intersectors.hpp>

using Scalar   = float;
using Vector3  = bvh::Vector3<Scalar>;
using Ray      = bvh::Ray<Scalar>;
using Bvh      = bvh::Bvh<Scalar>;

template <typename NodeIntersector>
bool intersect_bvh_node(const Bvh::Node& node, const Ray& ray) {
    NodeIntersector node_intersector(ray);
    auto [tentry, texit] = node_intersector.intersect(node, ray);
    return tentry <= texit;
}

int main() {
    Bvh::Node node;
    node.bounds[0] = Scalar( -1);
    node.bounds[1] = Scalar(  1);
    node.bounds[2] = Scalar( -1);
    node.bounds[3] = Scalar(  1);
    node.bounds[4] = Scalar(2.1);
    node.bounds[5] = Scalar(2.1);

    Ray ray(
        Vector3(0.25, 0.25, 0.0), // origin
        Vector3(0.0, -0.0, 1.0),  // direction
        0.0,                      // minimum distance
        100.0                     // maximum distance
    );

    return
        intersect_bvh_node<bvh::FastNodeIntersector<Bvh>>(node, ray) &&
        intersect_bvh_node<bvh::RobustNodeIntersector<Bvh>>(node, ray) ? 0 : 1;
}
