# BVH Construction and Traversal Library

This library is a small, standalone library for BVH construction and traversal. It is licensed
under the MIT license.

## Features

- A high-quality, single-threaded sweeping SAH builder,
- A fast, medium-quality, single-threaded binned SAH builder inspired by
  "On Fast Construction of SAH-based Bounding Volume Hierarchies", by I. Wald,
- A fast, high-quality, multithreaded mini-tree BVH builder inspired by
  "Rapid Bounding Volume Hierarchy Generation using Mini Trees", by P. Ganestam et al.,
- A reinsertion optimizer based on "Parallel Reinsertion for Bounding Volume Hierarchy
  Optimization", by D. Meister and J. Bittner,
- A fast and robust traversal algorithm using "Robust BVH Ray Traversal", by T. Ize.
- A fast ray-triangle intersection algorithm based on
  "Fast, Minimum Storage Ray/Triangle Intersection", by T. Möller and B. Trumbore,
- A serialization/deserialization interface,
- A variable amount of dimensions (e.g. 2D, 3D, 4D BVHs are supported) and different scalar types
  (e.g. `float` or `double`),
- Few dependencies: Only depends on the standard library (parallelization uses a custom thread pool
  based on `std::thread`).

## Usage

![simple example](https://github.com/madmann91/bvh/blob/v2/test/main.cpp)
