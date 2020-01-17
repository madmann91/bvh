# bvh

This is a modern C++17 header-only BVH library optimized for ray-tracing. Traversal and
construction routines support different primitive types. The design is such that the
BVH holds no data and only holds nodes. There is no hardware- or platform-specific
intrinsic used. Parallelization is done using OpenMP tasks. There is no dependency
except the C++ standard library.

The construction algorithm is very fast and produces high-quality, SAH-optimized trees
(see _On fast Construction of SAH-based Bounding Volume Hierarchies_, by Ingo Wald).
It only requires a bounding box and center for each primitive.

The traversal algorithm is optimized in several ways:

  - Rays are classified by octant, to make the ray-box test more efficient,
  - The order of traversal is such that the closest node is chosen first,
  - The ray-box test does not use divisions, and uses FMA instructions
    when possible,
  - The primitive data of the client is permuted in such a way that no
    indirection is done when looking up primitives.

The traversal algorithm can work in two modes: closest intersection,
or any intersection (for shadow rays, usually around 20% faster).
It only requires an intersection routine for the primitives.

# Performance

This library is carefully crafted to ensure good single-ray performance, while still
being simple, portable and high-level. These are performance results versus another
simple library available on GitHub, [brandonpelfrey/Fast-BVH](https://github.com/brandonpelfrey/Fast-BVH)
(top of the list for the search "bvh"):

|                   | brandonpelfrey/Fast-BVH |    This library    |
|-------------------|-------------------------|--------------------|
| BVH construction  |           25ms          |          20ms      |
| Rendering         |         3540ms          |        1254ms      |

These numbers where generated by taking the benchmark code from brandonpelfrey/Fast-BVH
and [porting it to this library](test/benchmark_vs_fastbvh.cpp). Instead of using random
spheres, the algorithm has been modified to load an OBJ file and render triangles colored
by their normal. The scene used for this table is Sponza. The machine used is an AMD
Threadripper 2950X with 16 physical cores and 32 threads. Both libraries use OpenMP and run
on multiple cores (except that the construction algorithm in Fast-BVH is not multithreaded,
unlike the one in this library). The rendering resolution is 8Kx8K.

The results are not surprising as the BVH used in Fast-BVH is using the middle split technique
which is known to be seriously bad for anything but small examples or uniformly distributed
primitives. The SSE3 routines of Fast-BVH are also not better than the code generated by
the compiler (gcc 8.3.1 in this example).

Compared to [Embree](https://github.com/embree/embree), the ray-tracing kernels from Intel, this
library only supports single-ray tracing and binary BVHs. As a result, it cannot be as fast as the
stream or packet traversal algorithms of Embree. Compared to the single-ray kernels of Embree running on the
high-quality BVHs also built by Embree, this library is around 50% slower, but the actual number can vary
depending on the type of ray (e.g. coherent/incoherent) and the scene. To match that level of performance,
this library would have to implement:

  - Spatial splits (see _Spatial Splits in Bounding Volume Hierarchies_, by Stich et al.),
  - Higher-arity BVHs (BVH4 or BVH8, with more than just two children per node),
  - Vectorization of the traversal and intersection routines.

While spatial splits may be added in the future, wider BVHs and vectorization go against the principles of
simplicity and portability followed by this library and will most likely never be implemented. Instead, algorithmic
improvements such as post-build optimizations will be investigated, as long as they offer a good compromise
between implementation complexity and performance improvements.

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
see [this simple example](test/simple_example.cpp).

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

# License

This library is distributed under the MIT license. See [bvh.hpp](bvh.hpp) for details.
