# bvh

This is a modern C++17 header-only BVH library optimized for ray-tracing. Traversal and
construction routines support different primitive types. The design is such that the
BVH holds no data and only holds nodes. There is no hardware- or platform-specific
intrinsic used. Parallelization is done using OpenMP tasks.

The construction algorithm is very fast and produces high-quality, SAH-optimized trees.
(see _On fast Construction of SAH-based Bounding Volume Hierarchies_, by Ingo Wald)
It only requires a bounding box and center for each primitive.

The traversal algorithm is optimized in two ways:

  - Rays are classified by octant, to make the ray-box test more efficient,
  - The order of traversal is such that the closest node is chosen first,
  - The ray-box test does not use divisions, and uses FMA instructions
    when possible,
  - The primitive data of the client is permuted in such a way that no
    indirection is done when looking up primitives.

The traversal algorithm can work in two modes: closest intersection,
or any intersection (for shadow rays, usually around 20% faster).
It only requires an intersection routine for the primitives.

# Building

There is no need to build anything, since this library is header-only.
To build the tests, type:

    mkdir build
    cd build
    cmake ..
    cmake --build .

# Usage

The API comes in two flavours: A low level version where the build algorithm
can be fine-tuned, and a high-level version.

## High-level API

The high-level API hides the details of the construction or traversal algorithms.
Default values are used for the builder and indirections are avoided in intersectors
by shuffling primitive data. For an example of how to use the high-level API,
see [this simple_example](test/simple_example.cpp).

## Low-level API

The low-level API allows direct access to the building algorithm and allows to create custom intersectors.
See [this example](test/custom_intersector.cpp).

## Custom primitive types

Custom primitives types can be implemented, but some members are required to maintain compatibility with the high-level API.
See [this file](test/custom_primitive.cpp) for an example.

## Double precision

Double precision can be enabled by defining `BVH_DOUBLE` before including `bvh.hpp`:

```cpp
// Uncomment to enable double precision
//#define BVH_DOUBLE
#include "bvh.hpp"
```
