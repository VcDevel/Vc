/*  This file is part of the Vc library. {{{
Copyright © 2014 Matthias Kretz <kretz@kde.org>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the names of contributing organizations nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#ifndef VC_COMMON_SIMD_CAST_H
#define VC_COMMON_SIMD_CAST_H

#include <type_traits>
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
template <typename To, typename From>
Vc_INTRINSIC Vc_CONST To simd_cast(From x, enable_if<std::is_same<To, From>::value> = nullarg)
{
    return x;
}

/*
 * I don't want to have the following visible in overload resolution:
template <typename To, typename From>
inline To simd_cast(From x0,
                    From x1,
                    enable_if<!(sizeof(To) == 16 && sizeof(From) == 16)> = nullarg)
{
    static_assert(std::is_same<From, void>::value,
                  "simd_cast for the given type combination is not implemented.");
    return To();
}
*/

#define Vc_1_SIMDARRAY_TO_1__(simdarray_type__, trait__)                                 \
    template <typename Return, typename T, std::size_t N, typename V>                    \
    Vc_INTRINSIC Vc_CONST Return simd_cast(const simdarray_type__<T, N, V, N> &k,        \
                                           enable_if<trait__<Return>::value> = nullarg)  \
    {                                                                                    \
        return simd_cast<Return>(internal_data(k));                                      \
    }                                                                                    \
    template <typename Return, typename T, std::size_t N, typename V, std::size_t M>     \
    Vc_INTRINSIC Vc_CONST Return simd_cast(                                              \
        const simdarray_type__<T, N, V, M> &k,                                           \
        enable_if<(N <= Return::Size && ((N - 1) & N) == 0 && trait__<Return>::value)> = \
            nullarg)                                                                     \
    {                                                                                    \
        return simd_cast<Return>(internal_data0(k), internal_data1(k));                  \
    }                                                                                    \
    template <typename Return, typename T, std::size_t N, typename V, std::size_t M>     \
    Vc_INTRINSIC Vc_CONST Return                                                         \
        simd_cast(const simdarray_type__<T, N, V, M> &k,                                 \
                  enable_if<(N > Return::Size && trait__<Return>::value)> = nullarg)     \
    {                                                                                    \
        /* This relies on the assumption that internal_data0(k).size() >= Return::Size   \
         It holds because Return::Size is a power of two && N > Return::Size &&          \
         internal_data0(k).size() is the largest power of two < N */                     \
        return simd_cast<Return>(internal_data0(k));                                     \
    }                                                                                    \
    template <typename Return, typename T, std::size_t N, typename V, std::size_t M>     \
    Vc_INTRINSIC Vc_CONST Return simd_cast(                                              \
        const simdarray_type__<T, N, V, M> &k,                                           \
        enable_if<(N <= Return::Size && ((N - 1) & N) != 0 && trait__<Return>::value)> = \
            nullarg)                                                                     \
    {                                                                                    \
        const auto &lo = internal_data0(k);                                              \
        const auto &hi = internal_data1(k);                                              \
        Return r = simd_cast<Return>(lo);                                                \
        for (size_t i = 0; i < hi.size(); ++i) {                                         \
            r[i + lo.size()] = static_cast<typename Return::EntryType>(hi[i]);           \
        }                                                                                \
        return r;                                                                        \
    }

#define Vc_2_SIMDARRAY_TO_1__(simdarray_type__, trait__)                                 \
    /* indivisible simdarray_type__ */                                                   \
    template <typename Return, typename T, std::size_t N, typename V>                    \
    Vc_INTRINSIC Vc_CONST Return simd_cast(                                              \
        const simdarray_type__<T, N, V, N> &k0,                                          \
        const simdarray_type__<T, N, V, N> &k1,                                          \
        enable_if<(N * 2 <= Return::Size && trait__<Return>::value)> = nullarg)          \
    {                                                                                    \
        return simd_cast<Return>(internal_data(k0), internal_data(k1));                  \
    }                                                                                    \
    /* bisectable simdarray_type__ (N = 2^n) */                                          \
    template <typename Return, typename T, std::size_t N, typename V, std::size_t M>     \
    Vc_INTRINSIC Vc_CONST Return                                                         \
        simd_cast(const simdarray_type__<T, N, V, M> &k0,                                \
                  const simdarray_type__<T, N, V, M> &k1,                                \
                  enable_if<(N * 2 <= Return::Size && ((N - 1) & N) == 0 &&              \
                             trait__<Return>::value)> = nullarg)                         \
    {                                                                                    \
        return simd_cast<Return>(internal_data0(k0),                                     \
                                 internal_data1(k0),                                     \
                                 internal_data0(k1),                                     \
                                 internal_data1(k1));                                    \
    }                                                                                    \
    /* remaining simdarray_type__ input never larger (N != 2^n) */                       \
    template <typename Return, typename T, std::size_t N, typename V, std::size_t M>     \
    Vc_INTRINSIC Vc_CONST Return                                                         \
        simd_cast(const simdarray_type__<T, N, V, M> &k0,                                \
                  const simdarray_type__<T, N, V, M> &k1,                                \
                  enable_if<(N * 2 <= Return::Size && ((N - 1) & N) != 0 &&              \
                             trait__<Return>::value)> = nullarg)                         \
    {                                                                                    \
        /* FIXME: this needs optimized implementation (unless compilers are smart        \
         * enough) */                                                                    \
        Return r = simd_cast<Return>(k0);                                                \
        for (size_t i = 0; i < N; ++i) {                                                 \
            r[i + N] = static_cast<typename Return::EntryType>(k1[i]);                   \
        }                                                                                \
        return r;                                                                        \
    }                                                                                    \
    /* remaining simdarray_type__ input larger (N != 2^n) */                             \
    template <typename Return, typename T, std::size_t N, typename V, std::size_t M>     \
    Vc_INTRINSIC Vc_CONST Return                                                         \
        simd_cast(const simdarray_type__<T, N, V, M> &k0,                                \
                  const simdarray_type__<T, N, V, M> &k1,                                \
                  enable_if<(N * 2 > Return::Size && N < Return::Size &&                 \
                             ((N - 1) & N) != 0 && trait__<Return>::value)> = nullarg)   \
    {                                                                                    \
        /* FIXME: this needs optimized implementation (unless compilers are smart        \
         * enough) */                                                                    \
        Return r = simd_cast<Return>(k0);                                                \
        for (size_t i = N; i < Return::Size; ++i) {                                      \
            r[i] = static_cast<typename Return::EntryType>(k1[i - N]);                   \
        }                                                                                \
        return r;                                                                        \
    }

#define Vc_3_SIMDARRAY_TO_1__(simdarray_type__, trait__)                                 \
    /* indivisible simdarray_type__ */                                                   \
    template <typename Return, typename T, std::size_t N, typename V>                    \
    Vc_INTRINSIC Vc_CONST Return simd_cast(                                              \
        const simdarray_type__<T, N, V, N> &k0,                                          \
        const simdarray_type__<T, N, V, N> &k1,                                          \
        const simdarray_type__<T, N, V, N> &k2,                                          \
        enable_if<(N * 3 <= Return::Size && trait__<Return>::value)> = nullarg)          \
    {                                                                                    \
        return simd_cast<Return>(                                                        \
            internal_data(k0), internal_data(k1), internal_data(k2));                    \
    }                                                                                    \
    /* remaining simdarray_type__ with input never larger */                             \
    template <typename Return, typename T, std::size_t N, typename V, std::size_t M>     \
    Vc_INTRINSIC Vc_CONST Return simd_cast(                                              \
        const simdarray_type__<T, N, V, M> &k0,                                          \
        const simdarray_type__<T, N, V, M> &k1,                                          \
        const simdarray_type__<T, N, V, M> &k2,                                          \
        enable_if<(N * 3 <= Return::Size && trait__<Return>::value)> = nullarg)          \
    {                                                                                    \
        using R = typename Return::EntryType;                                            \
        /* FIXME: this needs optimized implementation (unless compilers are smart        \
         * enough) */                                                                    \
        Return r = simd_cast<Return>(k0, k1);                                            \
        for (size_t i = 0; i < N; ++i) {                                                 \
            r[i + 2 * N] = static_cast<R>(k2[i]);                                        \
        }                                                                                \
        return r;                                                                        \
    }                                                                                    \
    /* remaining simdarray_type__ with input larger */                                   \
    template <typename Return, typename T, std::size_t N, typename V, std::size_t M>     \
    Vc_INTRINSIC Vc_CONST Return                                                         \
        simd_cast(const simdarray_type__<T, N, V, M> &k0,                                \
                  const simdarray_type__<T, N, V, M> &k1,                                \
                  const simdarray_type__<T, N, V, M> &k2,                                \
                  enable_if<(N * 3 > Return::Size && trait__<Return>::value)> = nullarg) \
    {                                                                                    \
        using R = typename Return::EntryType;                                            \
        /* FIXME: this needs optimized implementation (unless compilers are smart        \
         * enough) */                                                                    \
        Return r = simd_cast<Return>(k0, k1);                                            \
        for (size_t i = 2 * N; i < Return::Size; ++i) {                                  \
            r[i] = static_cast<R>(k2[i - 2 * N]);                                        \
        }                                                                                \
        return r;                                                                        \
    }

#define Vc_4_SIMDARRAY_TO_1__(simdarray_type__, trait__)                                 \
    /* indivisible simdarray_type__ */                                                   \
    template <typename Return, typename T, std::size_t N, typename V>                    \
    Vc_INTRINSIC Vc_CONST Return simd_cast(                                              \
        const simdarray_type__<T, N, V, N> &k0,                                          \
        const simdarray_type__<T, N, V, N> &k1,                                          \
        const simdarray_type__<T, N, V, N> &k2,                                          \
        const simdarray_type__<T, N, V, N> &k3,                                          \
        enable_if<(N * 4 <= Return::Size && trait__<Return>::value)> = nullarg)          \
    {                                                                                    \
        return simd_cast<Return>(                                                        \
            internal_data(k0), internal_data(k1), internal_data(k2), internal_data(k3)); \
    }                                                                                    \
    /* remaining simdarray_type__ with input never larger */                             \
    template <typename Return, typename T, std::size_t N, typename V, std::size_t M>     \
    Vc_INTRINSIC Vc_CONST Return simd_cast(                                              \
        const simdarray_type__<T, N, V, M> &k0,                                          \
        const simdarray_type__<T, N, V, M> &k1,                                          \
        const simdarray_type__<T, N, V, M> &k2,                                          \
        const simdarray_type__<T, N, V, M> &k3,                                          \
        enable_if<(N * 4 <= Return::Size && trait__<Return>::value)> = nullarg)          \
    {                                                                                    \
        using R = typename Return::EntryType;                                            \
        /* FIXME: this needs optimized implementation (unless compilers are smart        \
         * enough) */                                                                    \
        Return r = simd_cast<Return>(k0, k1, k2);                                        \
        for (size_t i = 0; i < N; ++i) {                                                 \
            r[i + 3 * N] = static_cast<R>(k3[i]);                                        \
        }                                                                                \
        return r;                                                                        \
    }                                                                                    \
    /* remaining simdarray_type__ with input larger */                                   \
    template <typename Return, typename T, std::size_t N, typename V, std::size_t M>     \
    Vc_INTRINSIC Vc_CONST Return                                                         \
        simd_cast(const simdarray_type__<T, N, V, M> &k0,                                \
                  const simdarray_type__<T, N, V, M> &k1,                                \
                  const simdarray_type__<T, N, V, M> &k2,                                \
                  const simdarray_type__<T, N, V, M> &k3,                                \
                  enable_if<(N * 4 > Return::Size && trait__<Return>::value)> = nullarg) \
    {                                                                                    \
        using R = typename Return::EntryType;                                            \
        /* FIXME: this needs optimized implementation (unless compilers are smart        \
         * enough) */                                                                    \
        Return r = simd_cast<Return>(k0, k1, k2);                                        \
        for (size_t i = 3 * N; i < Return::Size; ++i) {                                  \
            r[i] = static_cast<R>(k3[i - 3 * N]);                                        \
        }                                                                                \
        return r;                                                                        \
    }
}

#include "undomacros.h"

#endif // VC_COMMON_SIMD_CAST_H
