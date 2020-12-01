#include <bvh/bvh.hpp>
#include <bvh/vector.hpp>
#include <bvh/ray.hpp>
#include <bvh/node_intersectors.hpp>

using Scalar   = float;
using Vector3  = bvh::Vector3<Scalar>;
using Ray      = bvh::Ray<Scalar>;
using Bvh      = bvh::Bvh<Scalar>;

template <typename NodeIntersector>
bool test() {
    Ray ray(
        Vector3(0.25, 0.25, 0.0), // origin
        Vector3(0.0, -0.0, 1.0),  // direction
        0.0,                      // minimum distance
        100.0                     // maximum distance
    );
    NodeIntersector node_intersector(ray);
    Bvh::Node node;
    node.bounds[0] =  -1;
    node.bounds[1] =   1;
    node.bounds[2] =  -1;
    node.bounds[3] =   1;
    node.bounds[4] = 2.1;
    node.bounds[5] = 2.1;
    auto [tentry, texit] = node_intersector.intersect(node, ray);
    return tentry <= texit;
}

int main() {
    return
        test<bvh::FastNodeIntersector<Bvh>>() &&
        test<bvh::RobustNodeIntersector<Bvh>>() ? 0 : 1;
}
