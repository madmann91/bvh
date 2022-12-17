# BVH Construction and Traversal Library

This library is a small, standalone library for BVH construction and traversal. It is licensed
under the MIT license.

## Features

- High-quality, single-threaded sweeping SAH builder,
- Fast, medium-quality, single-threaded binned SAH builder inspired by
  "On Fast Construction of SAH-based Bounding Volume Hierarchies", by I. Wald,
- Fast, high-quality, multithreaded mini-tree BVH builder inspired by
  "Rapid Bounding Volume Hierarchy Generation using Mini Trees", by P. Ganestam et al.,
- Reinsertion optimizer based on "Parallel Reinsertion for Bounding Volume Hierarchy
  Optimization", by D. Meister and J. Bittner,
- Fast and robust traversal algorithm using "Robust BVH Ray Traversal", by T. Ize.
- Fast ray-triangle intersection algorithm based on
  "Fast, Minimum Storage Ray/Triangle Intersection", by T. Möller and B. Trumbore,
- Surface area traversal order heuristic for shadow rays based on "SATO: Surface Area Traversal
  Order for Shadow Ray Tracing", by J. Nah and D. Manocha,
- Fast ray-sphere intersection routine,
- Serialization/deserialization interface,
- Variable amount of dimensions (e.g. 2D, 3D, 4D BVHs are supported) and different scalar types
  (e.g. `float` or `double`),
- Only depends on the standard library (parallelization uses a custom thread pool based on
  `std::thread`).

## Usage

The library contains two examples that are kept up-to-date with the API:

- A [basic example](test/simple_example.cpp) that traces one ray on a scene made of a couple of triangles,
- A [benchmarking utility](test/benchmark.cpp) that showcases what the library can do.
