#include <bvh/v2/bvh.h>
#include <bvh/v2/vec.h>
#include <bvh/v2/ray.h>
#include <bvh/v2/tri.h>
#include <bvh/v2/node.h>
#include <bvh/v2/stream.h>
#include <bvh/v2/default_builder.h>

#include <fstream>
#include <iostream>

using Scalar  = float;
using Vec3    = bvh::v2::Vec<Scalar, 3>;
using BBox    = bvh::v2::BBox<Scalar, 3>;
using Tri     = bvh::v2::Tri<Scalar, 3>;
using Node    = bvh::v2::Node<Scalar, 3>;
using Bvh     = bvh::v2::Bvh<Node>;
using Ray     = bvh::v2::Ray<Scalar, 3>;

using PrecomputedTri = bvh::v2::PrecomputedTri<Scalar>;
using StdOutputStream = bvh::v2::StdOutputStream;
using StdInputStream = bvh::v2::StdInputStream;

static bool save_bvh(const Bvh& bvh, const std::string& file_name) {
    std::ofstream out(file_name, std::ofstream::binary);
    if (!out)
        return false;
    StdOutputStream stream(out);
    bvh.serialize(stream);
    return true;
}

static std::optional<Bvh> load_bvh(const std::string& file_name) {
    std::ifstream in(file_name, std::ofstream::binary);
    if (!in)
        return std::nullopt;
    StdInputStream stream(in);
    return std::make_optional(Bvh::deserialize(stream));
}

int main() {
    std::vector<Tri> tris;
    tris.emplace_back(
        Vec3( 1.0, -1.0, 1.0),
        Vec3( 1.0,  1.0, 1.0),
        Vec3(-1.0,  1.0, 1.0)
    );
    tris.emplace_back(
        Vec3( 1.0, -1.0, 1.0),
        Vec3(-1.0, -1.0, 1.0),
        Vec3(-1.0,  1.0, 1.0)
    );

    std::vector<BBox> bboxes(tris.size());
    std::vector<Vec3> centers(tris.size());
    for (size_t i = 0; i < tris.size(); ++i) {
        bboxes[i]  = tris[i].get_bbox();
        centers[i] = tris[i].get_center();
    }

    auto bvh = bvh::v2::DefaultBuilder<Node>::build(bboxes, centers);

    save_bvh(bvh, "bvh.bin");
    auto other_bvh = load_bvh("bvh.bin");
    if (!other_bvh) {
        std::cerr << "Cannot load bvh file" << std::endl;
        return 1;
    }

    if (bvh == other_bvh) {
        std::cout << "The deserialized BVH is the same as the original one" << std::endl;
        return 0;
    } else {
        std::cerr << "The deserialized BVH does not match the original one" << std::endl;
        return 1;
    }
}
