#include <vector>
#include <iostream>
#include "bvh.hpp"

int main() {
    // Create an array of triangles
    std::vector<bvh::Triangle> triangles;
    triangles.emplace_back(
        bvh::Vec3( 1.0f, -1.0f, 1.0f),
        bvh::Vec3( 1.0f,  1.0f, 1.0f),
        bvh::Vec3(-1.0f,  1.0f, 1.0f)
    );
    triangles.emplace_back(
        bvh::Vec3( 1.0f, -1.0f, 1.0f),
        bvh::Vec3(-1.0f, -1.0f, 1.0f),
        bvh::Vec3(-1.0f,  1.0f, 1.0f)
    );

    // Create an acceleration data structure on those triangles
    bvh::Accel<bvh::Triangle> accel(triangles.data(), triangles.size());
    accel.build();

    // Intersect a ray with the data structure
    bvh::Ray ray(
        bvh::Vec3(0.0f, 0.0f, 0.0f), // origin
        bvh::Vec3(0.0f, 0.0f, 1.0f), // direction
        0.0f,                        // minimum distance
        100.0f                       // maximum distance
    );
    auto hit = accel.intersect_closest(ray);
    if (hit) {
        auto triangle_index = hit->first;
        auto intersection = hit->second;
        std::cout << "Hit triangle " << triangle_index        << "\n"
                  << "distance: "    << intersection.distance << "\n"
                  << "u: "           << intersection.u        << "\n"
                  << "v: "           << intersection.v        << std::endl;
        return 0;
    }
    return 1;
}
