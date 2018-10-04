/*  This file is part of the Vc library. {{{
Copyright © 2016-2017 Matthias Kretz <kretz@kde.org>

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
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#ifndef VC_DETAIL_SCALAR_H_
#define VC_DETAIL_SCALAR_H_

#include "simd.h"
#include "detail.h"
#include <bitset>
#include <cmath>
#include <cstdlib>
#include "simd_tuple.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail {
template <class T> using scalar_mask = simd_mask<T, simd_abi::scalar>;
template <class T> using scalar_simd = simd<T, simd_abi::scalar>;
// simd impl {{{1
struct scalar_simd_impl {
    // member types {{{2
    using abi = Vc::simd_abi::scalar;
    using mask_member_type = bool;
    template <class T> using simd_member_type = T;
    template <class T> using simd = Vc::simd<T, abi>;
    template <class T> using simd_mask = Vc::simd_mask<T, abi>;
    template <class T> using type_tag = T *;

    // broadcast {{{2
    template <class T> static constexpr Vc_INTRINSIC T broadcast(T x) noexcept
    {
        return x;
    }

    // generator {{{2
    template <class F, class T>
    static Vc_INTRINSIC T generator(F &&gen, type_tag<T>)
    {
        return gen(size_constant<0>());
    }

    // load {{{2
    template <class T, class U, class F>
    static inline T load(const U *mem, F, type_tag<T>) noexcept
    {
        return static_cast<T>(mem[0]);
    }

    // masked load {{{2
    template <class T, class U, class F>
    static inline T masked_load(T merge, bool k, const U *mem, F) noexcept
    {
        if (k) {
            merge = static_cast<T>(mem[0]);
        }
        return merge;
    }

    // store {{{2
    template <class T, class U, class F>
    static inline void store(T v, U *mem, F, type_tag<T>) noexcept
    {
        mem[0] = static_cast<T>(v);
    }

    // masked store {{{2
    template <class T, class U, class F>
    static inline void masked_store(const T v, U *mem, F, const bool k) noexcept
    {
        if (k) {
            mem[0] = v;
        }
    }

    // negation {{{2
    template <class T> static inline bool negate(T x) noexcept { return !x; }

    // reductions {{{2
    template <class T, class BinaryOperation>
    static inline T reduce(const simd<T> &x, BinaryOperation &)
    {
        return x.d;
    }

    // min, max, clamp {{{2
    template <class T> static inline T min(const T a, const T b)
    {
        return std::min(a, b);
    }

    template <class T> static inline T max(const T a, const T b)
    {
        return std::max(a, b);
    }

    // complement {{{2
    template <class T> static inline T complement(T x) noexcept
    {
        return static_cast<T>(~x);
    }

    // unary minus {{{2
    template <class T> static inline T unary_minus(T x) noexcept
    {
        return static_cast<T>(-x);
    }

    // arithmetic operators {{{2
    template <class T> static inline T plus(T x, T y)
    {
        return static_cast<T>(detail::promote_preserving_unsigned(x) +
                              detail::promote_preserving_unsigned(y));
    }

    template <class T> static inline T minus(T x, T y)
    {
        return static_cast<T>(detail::promote_preserving_unsigned(x) -
                              detail::promote_preserving_unsigned(y));
    }

    template <class T> static inline T multiplies(T x, T y)
    {
        return static_cast<T>(detail::promote_preserving_unsigned(x) *
                              detail::promote_preserving_unsigned(y));
    }

    template <class T> static inline T divides(T x, T y)
    {
        return static_cast<T>(detail::promote_preserving_unsigned(x) /
                              detail::promote_preserving_unsigned(y));
    }

    template <class T> static inline T modulus(T x, T y)
    {
        return static_cast<T>(detail::promote_preserving_unsigned(x) %
                              detail::promote_preserving_unsigned(y));
    }

    template <class T> static inline T bit_and(T x, T y)
    {
        return static_cast<T>(detail::promote_preserving_unsigned(x) &
                              detail::promote_preserving_unsigned(y));
    }
    static inline float bit_and(float x, float y)
    {
        static_assert(sizeof(float) == sizeof(uint), "");
        const uint r = reinterpret_cast<const may_alias<uint> &>(x) &
                       reinterpret_cast<const may_alias<uint> &>(y);
        return reinterpret_cast<const may_alias<float> &>(r);
    }
    static inline double bit_and(double x, double y)
    {
        static_assert(sizeof(double) == sizeof(ullong), "");
        const ullong r = reinterpret_cast<const may_alias<ullong> &>(x) &
                         reinterpret_cast<const may_alias<ullong> &>(y);
        return reinterpret_cast<const may_alias<double> &>(r);
    }

    template <class T> static inline T bit_or(T x, T y)
    {
        return static_cast<T>(detail::promote_preserving_unsigned(x) |
                              detail::promote_preserving_unsigned(y));
    }
    static inline float bit_or(float x, float y)
    {
        static_assert(sizeof(float) == sizeof(uint), "");
        const uint r = reinterpret_cast<const may_alias<uint> &>(x) |
                       reinterpret_cast<const may_alias<uint> &>(y);
        return reinterpret_cast<const may_alias<float> &>(r);
    }
    static inline double bit_or(double x, double y)
    {
        static_assert(sizeof(double) == sizeof(ullong), "");
        const ullong r = reinterpret_cast<const may_alias<ullong> &>(x) |
                         reinterpret_cast<const may_alias<ullong> &>(y);
        return reinterpret_cast<const may_alias<double> &>(r);
    }


    template <class T> static inline T bit_xor(T x, T y)
    {
        return static_cast<T>(detail::promote_preserving_unsigned(x) ^
                              detail::promote_preserving_unsigned(y));
    }

    template <class T> static inline T bit_shift_left(T x, int y)
    {
        return static_cast<T>(detail::promote_preserving_unsigned(x) << y);
    }

    template <class T> static inline T bit_shift_right(T x, int y)
    {
        return static_cast<T>(detail::promote_preserving_unsigned(x) >> y);
    }

    // math {{{2
    template <class T> static Vc_INTRINSIC T abs(T x) { return T(std::abs(x)); }
    template <class T> static Vc_INTRINSIC T sqrt(T x) { return std::sqrt(x); }
    template <class T> static Vc_INTRINSIC T trunc(T x) { return std::trunc(x); }
    template <class T> static Vc_INTRINSIC T floor(T x) { return std::floor(x); }
    template <class T> static Vc_INTRINSIC T ceil(T x) { return std::ceil(x); }

    template <class T> static Vc_INTRINSIC simd_tuple<int, abi> fpclassify(T x)
    {
        return {std::fpclassify(x)};
    }
    template <class T> static Vc_INTRINSIC bool isfinite(T x) { return std::isfinite(x); }
    template <class T> static Vc_INTRINSIC bool isinf(T x) { return std::isinf(x); }
    template <class T> static Vc_INTRINSIC bool isnan(T x) { return std::isnan(x); }
    template <class T> static Vc_INTRINSIC bool isnormal(T x) { return std::isnormal(x); }
    template <class T> static Vc_INTRINSIC bool signbit(T x) { return std::signbit(x); }
    template <class T> static Vc_INTRINSIC bool isunordered(T x, T y) { return std::isunordered(x, y); }

    // increment & decrement{{{2
    template <class T> static inline void increment(T &x) { ++x; }
    template <class T> static inline void decrement(T &x) { --x; }

    // compares {{{2
    template <class T> static bool equal_to(T x, T y) { return x == y; }
    template <class T> static bool not_equal_to(T x, T y) { return x != y; }
    template <class T> static bool less(T x, T y) { return x < y; }
    template <class T> static bool greater(T x, T y) { return x > y; }
    template <class T> static bool less_equal(T x, T y) { return x <= y; }
    template <class T> static bool greater_equal(T x, T y) { return x >= y; }

    // smart_reference access {{{2
    template <class T, class U> static void set(T &v, int i, U &&x) noexcept
    {
        Vc_ASSERT(i == 0);
        unused(i);
        v = std::forward<U>(x);
    }

    // masked_assign {{{2
    template <typename T> static Vc_INTRINSIC void masked_assign(bool k, T &lhs, T rhs)
    {
        if (k) {
            lhs = rhs;
        }
    }

    // masked_cassign {{{2
    template <template <typename> class Op, typename T>
    static Vc_INTRINSIC void masked_cassign(const bool k, T &lhs, const T rhs)
    {
        if (k) {
            lhs = Op<T>{}(lhs, rhs);
        }
    }

    // masked_unary {{{2
    template <template <typename> class Op, typename T>
    static Vc_INTRINSIC T masked_unary(const bool k, const T v)
    {
        return static_cast<T>(k ? Op<T>{}(v) : v);
    }

    // }}}2
};

// simd_mask impl {{{1
struct scalar_mask_impl {
    // member types {{{2
    template <class T> using simd_mask = Vc::simd_mask<T, simd_abi::scalar>;
    template <class T> using type_tag = T *;

    // from_bitset {{{2
    template <class T>
    static Vc_INTRINSIC bool from_bitset(std::bitset<1> bs, type_tag<T>) noexcept
    {
        return bs[0];
    }

    // masked load {{{2
    template <class F>
    static Vc_INTRINSIC bool masked_load(bool merge, bool mask, const bool *mem,
                                         F) noexcept
    {
        if (mask) {
            merge = mem[0];
        }
        return merge;
    }

    // store {{{2
    template <class F> static Vc_INTRINSIC void store(bool v, bool *mem, F) noexcept
    {
        mem[0] = v;
    }

    // masked store {{{2
    template <class F>
    static Vc_INTRINSIC void masked_store(const bool v, bool *mem, F,
                                          const bool k) noexcept
    {
        if (k) {
            mem[0] = v;
        }
    }

    // logical and bitwise operators {{{2
    static constexpr bool logical_and(bool x, bool y) { return x && y; }
    static constexpr bool logical_or(bool x, bool y) { return x || y; }
    static constexpr bool bit_and(bool x, bool y) { return x && y; }
    static constexpr bool bit_or(bool x, bool y) { return x || y; }
    static constexpr bool bit_xor(bool x, bool y) { return x != y; }

    // smart_reference access {{{2
    static void set(bool &k, int i, bool x) noexcept
    {
        Vc_ASSERT(i == 0);
        detail::unused(i);
        k = x;
    }

    // masked_assign {{{2
    static Vc_INTRINSIC void masked_assign(bool k, bool &lhs, bool rhs)
    {
        if (k) {
            lhs = rhs;
        }
    }

    // }}}2
};

// simd_converter scalar -> scalar {{{1
template <class T> struct simd_converter<T, simd_abi::scalar, T, simd_abi::scalar> {
    Vc_INTRINSIC T operator()(T a) { return a; }
};
template <class From, class To>
struct simd_converter<From, simd_abi::scalar, To, simd_abi::scalar> {
    Vc_INTRINSIC To operator()(From a)
    {
        return static_cast<To>(a);
    }
};

// }}}1
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DETAIL_SCALAR_H_
// vim: foldmethod=marker
