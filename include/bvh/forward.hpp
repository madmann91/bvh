#ifndef BVH_FORWARD_DECL_HPP
#define BVH_FORWARD_DECL_HPP

// Note: This is an automatically generated header.
//       Changes to this file may be lost.

namespace bvh {

template <typename Bvh, size_t BinCount> class BinnedSahBuilder;
template <typename Bvh, size_t BinCount> class BinnedSahBuildTask;
template <typename Bvh, bool MaintainChildIndices> class BottomUpAlgorithm;
template <typename Scalar> struct BoundingBox;
template <typename Scalar> struct Bvh;
template <typename Primitive> class HeuristicPrimitiveSplitter;
template <typename Bvh> class HierarchyRefitter;
template <bool PreShuffled, typename Bvh, typename Primitive> struct ClosestIntersector;
template <bool PreShuffled, typename Bvh, typename Primitive> struct AnyIntersector;
template <typename Bvh> class LeafCollapser;
template <typename Bvh, typename Morton> class LinearBvhBuilder;
template <typename Bvh, typename Morton> class LocallyOrderedClusteringBuilder;
template <typename Bvh, typename Morton> class MortonCodeBasedBuilder;
template <typename Morton, typename Scalar> class MortonEncoder;
template <typename Bvh> class NodeLayoutOptimizer;
template <typename Bvh> class ParallelReinsertionOptimizer;
template <typename T> class PrefixSum;
template <size_t BitsPerIteration> class RadixSort;
template <typename Scalar> struct Ray;
template <typename Bvh> class SahBasedAlgorithm;
template <typename Bvh, size_t StackSize, bool Robust> class SingleRayTraverser;
template <typename Bvh, typename Primitive, size_t BinCount> class SpatialSplitBvhBuilder;
template <typename Bvh, typename Primitive, size_t BinCount> class SpatialSplitBvhBuildTask;
template <typename Scalar> struct Sphere;
template <typename Bvh> class SweepSahBuilder;
template <typename Bvh> class SweepSahBuildTask;
template <typename Scalar> struct Triangle;
template <size_t Bits> struct SizedIntegerType;
template <size_t I, size_t N> struct VectorSetter;
template <typename Scalar, size_t N> struct Vector;

} // namespace bvh

#endif
