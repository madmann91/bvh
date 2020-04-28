#ifndef BVH_VECTOR_HPP
#define BVH_VECTOR_HPP

#include <cstddef>
#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>

#include "bvh/platform.hpp"

namespace bvh {

/// Helper class to set the elements of a vector.
template <size_t I, size_t N>
struct VectorSetter {
    template <typename Vector, typename Scalar, typename... Args>
    bvh__always_inline__ static void set(Vector& v, Scalar s, Args... args) {
        v[I] = s;
        VectorSetter<I + 1, N>::set(v, args...);
    }
};

template <size_t N>
struct VectorSetter<N, N> {
    template <typename Vector>
    bvh__always_inline__ static void set(Vector&) {}
};

/// An N-dimensional vector class.
template <typename Scalar, size_t N>
struct Vector {
    Scalar values[N];

    Vector() = default;
    bvh__always_inline__ Vector(Scalar s) { std::fill(values, values + N, s); }

    template <size_t M, std::enable_if_t<(M > N), int> = 0>
    bvh__always_inline__ explicit Vector(const Vector<Scalar, M>& other) {
        std::copy(other.values, other.values + N, values);
    }

    template <typename... Args>
    bvh__always_inline__ Vector(Scalar first, Scalar second, Args... args) {
        set(first, second, args...);
    }

    template <typename F, std::enable_if_t<std::is_invocable<F, size_t>::value, int> = 0>
    bvh__always_inline__ Vector(F f) {
        for (size_t i = 0; i < N; ++i)
            values[i] = f(i);
    }

    template <typename... Args>
    bvh__always_inline__ void set(Args... args) {
        VectorSetter<0, N>::set(*this, args...);
    }

    bvh__always_inline__ Vector operator - () const {
        return Vector([this] (size_t i) { return -values[i]; });
    }

    bvh__always_inline__ Vector inverse() const {
        return Vector([this] (size_t i) { return Scalar(1) / values[i]; });
    }

    bvh__always_inline__ Scalar& operator [] (size_t i) { return values[i]; }
    bvh__always_inline__ Scalar  operator [] (size_t i) const { return values[i]; }
};

template <typename Scalar, size_t N>
bvh__always_inline__
inline Vector<Scalar, N> operator + (const Vector<Scalar, N>& a, const Vector<Scalar, N>& b) {
    return Vector<Scalar, N>([=] (size_t i) { return a[i] + b[i]; });
}

template <typename Scalar, size_t N>
bvh__always_inline__
inline Vector<Scalar, N> operator - (const Vector<Scalar, N>& a, const Vector<Scalar, N>& b) {
    return Vector<Scalar, N>([=] (size_t i) { return a[i] - b[i]; });
}

template <typename Scalar, size_t N>
bvh__always_inline__
inline Vector<Scalar, N> operator * (const Vector<Scalar, N>& a, const Vector<Scalar, N>& b) {
    return Vector<Scalar, N>([=] (size_t i) { return a[i] * b[i]; });
}

template <typename Scalar, size_t N>
bvh__always_inline__
inline Vector<Scalar, N> min(const Vector<Scalar, N>& a, const Vector<Scalar, N>& b) {
    return Vector<Scalar, N>([=] (size_t i) { return std::min(a[i], b[i]); });
}

template <typename Scalar, size_t N>
bvh__always_inline__
inline Vector<Scalar, N> max(const Vector<Scalar, N>& a, const Vector<Scalar, N>& b) {
    return Vector<Scalar, N>([=] (size_t i) { return std::max(a[i], b[i]); });
}

template <typename Scalar, size_t N>
bvh__always_inline__
inline Vector<Scalar, N> operator * (const Vector<Scalar, N>& a, Scalar s) {
    return Vector<Scalar, N>([=] (size_t i) { return a[i] * s; });
}

template <typename Scalar, size_t N>
bvh__always_inline__
inline Vector<Scalar, N> operator * (Scalar s, const Vector<Scalar, N>& b) {
    return b * s;
}

template <typename Scalar, size_t N>
bvh__always_inline__
inline Scalar dot(const Vector<Scalar, N>& a, const Vector<Scalar, N>& b) {
    Scalar sum = a[0] * b[0];
    for (size_t i = 1; i < N; ++i)
        sum += a[i] * b[i];
    return sum;
}

template <typename Scalar, size_t N>
bvh__always_inline__
inline Scalar length(const Vector<Scalar, N>& v) {
    return std::sqrt(dot(v, v));
}

template <typename Scalar, size_t N>
bvh__always_inline__
inline Vector<Scalar, N> normalize(const Vector<Scalar, N>& v) {
    auto inv = Scalar(1) / length(v);
    return v * inv;
}

template <typename Scalar>
using Vector3 = Vector<Scalar, 3>;

template <typename Scalar>
bvh__always_inline__
inline Vector3<Scalar> cross(const Vector3<Scalar>& a, const Vector3<Scalar>& b) {
    return Vector3<Scalar>([=] (size_t i) {
        size_t j = (i + 1) % 3;
        size_t k = (i + 2) % 3;
        return a[j] * b[k] - a[k] * b[j];
    });
}

} // namespace bvh

#endif
