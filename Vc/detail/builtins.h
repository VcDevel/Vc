/*  This file is part of the Vc library. {{{
Copyright © 2018 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_DETAIL_BUILTINS_H_
#define VC_DETAIL_BUILTINS_H_

#include <cstring>
#include "detail.h"

#if defined Vc_HAVE_SSE || defined Vc_HAVE_MMX
#include <x86intrin.h>
#endif  // Vc_HAVE_SSE

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
// intrinsic_type {{{
template <class T, size_t Bytes, class = std::void_t<>> struct intrinsic_type;
template <class T, size_t Size>
using intrinsic_type_t = typename intrinsic_type<T, Size * sizeof(T)>::type;
template <class T> using intrinsic_type2_t   = typename intrinsic_type<T, 2>::type;
template <class T> using intrinsic_type4_t   = typename intrinsic_type<T, 4>::type;
template <class T> using intrinsic_type8_t   = typename intrinsic_type<T, 8>::type;
template <class T> using intrinsic_type16_t  = typename intrinsic_type<T, 16>::type;
template <class T> using intrinsic_type32_t  = typename intrinsic_type<T, 32>::type;
template <class T> using intrinsic_type64_t  = typename intrinsic_type<T, 64>::type;
template <class T> using intrinsic_type128_t = typename intrinsic_type<T, 128>::type;

// }}}
// is_intrinsic{{{1
template <class T> struct is_intrinsic : public std::false_type {};
template <class T> inline constexpr bool is_intrinsic_v = is_intrinsic<T>::value;

// }}}
// builtin_type {{{1
template <class T, size_t N, class = void> struct builtin_type_n {};

// special case 1-element to be T itself
template <class T>
struct builtin_type_n<T, 1, std::enable_if_t<detail::is_vectorizable_v<T>>> {
    using type = T;
};

// else, use GNU-style builtin vector types
template <class T, size_t N>
struct builtin_type_n<T, N, std::enable_if_t<detail::is_vectorizable_v<T>>> {
    static constexpr size_t Bytes = N * sizeof(T);
    using type [[gnu::vector_size(Bytes)]] = T;
};

template <class T, size_t Bytes>
struct builtin_type : builtin_type_n<T, Bytes / sizeof(T)> {
    static_assert(Bytes % sizeof(T) == 0);
};

template <class T, size_t Size>
using builtin_type_t = typename builtin_type_n<T, Size>::type;
template <class T> using builtin_type2_t  = typename builtin_type<T, 2>::type;
template <class T> using builtin_type4_t  = typename builtin_type<T, 4>::type;
template <class T> using builtin_type8_t  = typename builtin_type<T, 8>::type;
template <class T> using builtin_type16_t = typename builtin_type<T, 16>::type;
template <class T> using builtin_type32_t = typename builtin_type<T, 32>::type;
template <class T> using builtin_type64_t = typename builtin_type<T, 64>::type;
template <class T> using builtin_type128_t = typename builtin_type<T, 128>::type;

// is_builtin_vector {{{1
template <class T, class = std::void_t<>> struct is_builtin_vector : std::false_type {};
template <class T>
struct is_builtin_vector<
    T,
    std::void_t<typename builtin_type<decltype(std::declval<T>()[0]), sizeof(T)>::type>>
    : std::is_same<
          T, typename builtin_type<decltype(std::declval<T>()[0]), sizeof(T)>::type> {
};

template <class T> inline constexpr bool is_builtin_vector_v = is_builtin_vector<T>::value;

// builtin_traits{{{1
template <class T, class = std::void_t<>> struct builtin_traits;
template <class T>
struct builtin_traits<T, std::void_t<std::enable_if_t<is_builtin_vector_v<T>>>> {
    using type = T;
    using value_type = decltype(std::declval<T>()[0]);
    static constexpr int width = sizeof(T) / sizeof(value_type);
};
template <class T, size_t N>
struct builtin_traits<Storage<T, N>, std::void_t<builtin_type_t<T, N>>> {
    using type = builtin_type_t<T, N>;
    using value_type = T;
    static constexpr int width = N;
};

// }}}
// builtin_cast{{{1
template <class To, class From, class Traits = builtin_traits<From>>
constexpr Vc_INTRINSIC typename builtin_type<To, sizeof(From)>::type builtin_cast(From x)
{
    return reinterpret_cast<typename builtin_type<To, sizeof(From)>::type>(x);
}
template <class To, class T, size_t N>
constexpr Vc_INTRINSIC builtin_type_t<To, N> builtin_cast(const Storage<T, N> &x)
{
    return reinterpret_cast<builtin_type_t<To, N>>(x.d);
}

// }}}
// to_intrin {{{
template <class T, class Traits = builtin_traits<T>,
          class R = intrinsic_type_t<typename Traits::value_type, Traits::width>>
constexpr Vc_INTRINSIC R to_intrin(T x)
{
    return reinterpret_cast<R>(x);
}
template <class T, size_t N, class R = intrinsic_type_t<T, N>>
constexpr Vc_INTRINSIC R to_intrin(Storage<T, N> x)
{
    return reinterpret_cast<R>(x.d);
}

// }}}
// make_builtin{{{
template <class T, class... Args>
constexpr Vc_INTRINSIC builtin_type_t<T, sizeof...(Args)> make_builtin(Args &&... args)
{
    return builtin_type_t<T, sizeof...(Args)>{static_cast<T>(args)...};
}

// }}}
// builtin_broadcast{{{
template <size_t N, class T>
constexpr Vc_INTRINSIC builtin_type_t<T, N> builtin_broadcast(T x)
{
    if constexpr (N == 2) {
        return builtin_type_t<T, 2>{x, x};
    } else if constexpr (N == 4) {
        return builtin_type_t<T, 4>{x, x, x, x};
    } else if constexpr (N == 8) {
        return builtin_type_t<T, 8>{x, x, x, x, x, x, x, x};
    } else if constexpr (N == 16) {
        return builtin_type_t<T, 16>{x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x};
    } else if constexpr (N == 32) {
        return builtin_type_t<T, 32>{x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
                                     x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x};
    } else if constexpr (N == 64) {
        return builtin_type_t<T, 64>{x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
                                     x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
                                     x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
                                     x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x};
    } else if constexpr (N == 128) {
        return builtin_type_t<T, 128>{
            x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
            x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
            x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
            x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
            x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x};
    }
}

// }}}
// auto_broadcast{{{
template <class T> struct auto_broadcast {
    const T x;
    constexpr Vc_INTRINSIC auto_broadcast(T xx) : x(xx) {}
    template <class V> constexpr Vc_INTRINSIC operator V() const
    {
        static_assert(is_builtin_vector_v<V>);
        return reinterpret_cast<V>(builtin_broadcast<sizeof(V) / sizeof(T)>(x));
    }
};

// }}}
// generate_builtin{{{
template <class T, size_t N, class G, size_t... I>
constexpr Vc_INTRINSIC builtin_type_t<T, N> generate_builtin_impl(
    G &&gen, std::index_sequence<I...>)
{
    return builtin_type_t<T, N>{static_cast<T>(gen(size_constant<I>()))...};
}

template <class V, class Traits = builtin_traits<V>, class G>
constexpr Vc_INTRINSIC V generate_builtin(G &&gen)
{
    return generate_builtin_impl<typename Traits::value_type, Traits::width>(
        std::forward<G>(gen), std::make_index_sequence<Traits::width>());
}

template <class T, size_t N, class G>
constexpr Vc_INTRINSIC builtin_type_t<T, N> generate_builtin(G &&gen)
{
    return generate_builtin_impl<T, N>(std::forward<G>(gen),
                                       std::make_index_sequence<N>());
}

// }}}
// builtin_load{{{
template <class T, size_t N, size_t M = N * sizeof(T), class F>
builtin_type_t<T, N> builtin_load(const void *p, F)
{
#ifdef Vc_WORKAROUND_XXX_2
    using U = std::conditional_t<
        (std::is_integral_v<T> || M < 4), long long,
        std::conditional_t<(std::is_same_v<T, double> || M < 8), float, T>>;
    using V = builtin_type_t<U, N * sizeof(T) / sizeof(U)>;
#else   // Vc_WORKAROUND_XXX_2
    using V = builtin_type_t<T, N>;
#endif  // Vc_WORKAROUND_XXX_2
    V r;
    static_assert(M <= sizeof(V));
    if constexpr(std::is_same_v<F, element_aligned_tag>) {
    } else if constexpr(std::is_same_v<F, vector_aligned_tag>) {
        p = __builtin_assume_aligned(p, alignof(builtin_type_t<T, N>));
    } else {
        p = __builtin_assume_aligned(p, F::alignment);
    }
    std::memcpy(&r, p, M);
    return reinterpret_cast<builtin_type_t<T, N>>(r);
}

// }}}
// builtin_load16 {{{
template <class T, size_t M = 16, class F>
builtin_type16_t<T> builtin_load16(const void *p, F f)
{
    return builtin_load<T, 16 / sizeof(T), M>(p, f);
}

// }}}
// builtin_store{{{
template <size_t M = 0, class B, class Traits = builtin_traits<B>, class F>
void builtin_store(const B v, void *p, F)
{
    using T = typename Traits::value_type;
    constexpr size_t N = Traits::width;
    constexpr size_t Bytes = M == 0 ? N * sizeof(T) : M;
    static_assert(Bytes <= sizeof(v));
#ifdef Vc_WORKAROUND_XXX_2
    using U = std::conditional_t<
        (std::is_integral_v<T> || Bytes < 4), long long,
        std::conditional_t<(std::is_same_v<T, double> || Bytes < 8), float, T>>;
    const auto vv = builtin_cast<U>(v);
#else   // Vc_WORKAROUND_XXX_2
    const builtin_type_t<T, N> vv = v;
#endif  // Vc_WORKAROUND_XXX_2
    if constexpr(std::is_same_v<F, vector_aligned_tag>) {
        p = __builtin_assume_aligned(p, alignof(builtin_type_t<T, N>));
    } else if constexpr(!std::is_same_v<F, element_aligned_tag>) {
        p = __builtin_assume_aligned(p, F::alignment);
    }
    if constexpr ((Bytes & (Bytes - 1)) != 0) {
        constexpr size_t MoreBytes = next_power_of_2(Bytes);
        alignas(MoreBytes) char tmp[MoreBytes];
        std::memcpy(tmp, &vv, MoreBytes);
        std::memcpy(p, tmp, Bytes);
    } else {
        std::memcpy(p, &vv, Bytes);
    }
}

// }}}
// xor_{{{
template <class T, class Traits = builtin_traits<T>>
constexpr Vc_INTRINSIC T xor_(T a, typename Traits::type b) noexcept
{
    return reinterpret_cast<T>(builtin_cast<unsigned>(a) ^ builtin_cast<unsigned>(b));
}

// }}}
// or_{{{
template <class T, class Traits = builtin_traits<T>>
constexpr Vc_INTRINSIC T or_(T a, typename Traits::type b) noexcept
{
    return reinterpret_cast<T>(builtin_cast<unsigned>(a) | builtin_cast<unsigned>(b));
}

// }}}
// and_{{{
template <class T, class Traits = builtin_traits<T>>
constexpr Vc_INTRINSIC T and_(T a, typename Traits::type b) noexcept
{
    return reinterpret_cast<T>(builtin_cast<unsigned>(a) & builtin_cast<unsigned>(b));
}

// }}}
// andnot_{{{
template <class T, class Traits = builtin_traits<T>>
constexpr Vc_INTRINSIC T andnot_(T a, typename Traits::type b) noexcept
{
    return reinterpret_cast<T>(~builtin_cast<unsigned>(a) & builtin_cast<unsigned>(b));
}

// }}}
// not_{{{
template <class T, class Traits = builtin_traits<T>>
constexpr Vc_INTRINSIC T not_(T a) noexcept
{
    return reinterpret_cast<T>(~builtin_cast<unsigned>(a));
}

// }}}
// concat{{{
template <class T, class Trait = builtin_traits<T>,
          class R = builtin_type_t<typename Trait::value_type, Trait::width * 2>>
constexpr R concat(T a_, T b_) {
#ifdef Vc_WORKAROUND_XXX_1
    using W = std::conditional_t<std::is_floating_point_v<typename Trait::value_type>,
                                 double, long long>;
    constexpr int input_width = sizeof(T) / sizeof(W);
    using TT = builtin_type_t<W, input_width>;
    const TT a = reinterpret_cast<TT>(a_);
    const TT b = reinterpret_cast<TT>(b_);
    using U = builtin_type_t<W, sizeof(R) / sizeof(W)>;
#else
    constexpr int input_width = Trait::width;
    const T &a = a_;
    const T &b = b_;
    using U = R;
#endif
    if constexpr(input_width == 2) {
        return reinterpret_cast<R>(U{a[0], a[1], b[0], b[1]});
    } else if constexpr (input_width == 4) {
        return reinterpret_cast<R>(U{a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]});
    } else if constexpr (input_width == 8) {
        return reinterpret_cast<R>(U{a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], b[0],
                                     b[1], b[2], b[3], b[4], b[5], b[6], b[7]});
    } else if constexpr (input_width == 16) {
        return reinterpret_cast<R>(
            U{a[0],  a[1],  a[2],  a[3],  a[4],  a[5],  a[6],  a[7],  a[8],  a[9], a[10],
              a[11], a[12], a[13], a[14], a[15], b[0],  b[1],  b[2],  b[3],  b[4], b[5],
              b[6],  b[7],  b[8],  b[9],  b[10], b[11], b[12], b[13], b[14], b[15]});
    } else if constexpr (input_width == 32) {
        return reinterpret_cast<R>(
            U{a[0],  a[1],  a[2],  a[3],  a[4],  a[5],  a[6],  a[7],  a[8],  a[9],  a[10],
              a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21],
              a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], b[0],
              b[1],  b[2],  b[3],  b[4],  b[5],  b[6],  b[7],  b[8],  b[9],  b[10], b[11],
              b[12], b[13], b[14], b[15], b[16], b[17], b[18], b[19], b[20], b[21], b[22],
              b[23], b[24], b[25], b[26], b[27], b[28], b[29], b[30], b[31]});
    }
}

// }}}
// extract<N, By>{{{
template <int Offset, int SplitBy, class T, class Trait = builtin_traits<T>,
          class R = builtin_type_t<typename Trait::value_type, Trait::width / SplitBy>>
constexpr Vc_INTRINSIC R extract(T x_)
{
#ifdef Vc_WORKAROUND_XXX_1
    using W = std::conditional_t<std::is_floating_point_v<typename Trait::value_type>,
                                 double, long long>;
    constexpr int return_width = sizeof(R) / sizeof(W);
    using U = builtin_type_t<W, return_width>;
    const auto x = reinterpret_cast<builtin_type_t<W, sizeof(T) / sizeof(W)>>(x_);
#else
    constexpr int return_width = Trait::width / SplitBy;
    using U = R;
    const builtin_type_t<typename Traits::value_type, Trait::width> &x = x_;
#endif
    constexpr int O = Offset * return_width;
    if constexpr (return_width == 2) {
        return reinterpret_cast<R>(U{x[O + 0], x[O + 1]});
    } else if constexpr (return_width == 4) {
        return reinterpret_cast<R>(U{x[O + 0], x[O + 1], x[O + 2], x[O + 3]});
    } else if constexpr (return_width == 8) {
        return reinterpret_cast<R>(U{x[O + 0], x[O + 1], x[O + 2], x[O + 3], x[O + 4],
                                     x[O + 5], x[O + 6], x[O + 7]});
    } else if constexpr (return_width == 16) {
        return reinterpret_cast<R>(U{x[O + 0], x[O + 1], x[O + 2], x[O + 3], x[O + 4],
                                     x[O + 5], x[O + 6], x[O + 7], x[O + 8], x[O + 9],
                                     x[O + 10], x[O + 11], x[O + 12], x[O + 13],
                                     x[O + 14], x[O + 15]});
    } else if constexpr (return_width == 32) {
        return reinterpret_cast<R>(
            U{x[O + 0],  x[O + 1],  x[O + 2],  x[O + 3],  x[O + 4],  x[O + 5],  x[O + 6],
              x[O + 7],  x[O + 8],  x[O + 9],  x[O + 10], x[O + 11], x[O + 12], x[O + 13],
              x[O + 14], x[O + 15], x[O + 16], x[O + 17], x[O + 18], x[O + 19], x[O + 20],
              x[O + 21], x[O + 22], x[O + 23], x[O + 24], x[O + 25], x[O + 26], x[O + 27],
              x[O + 28], x[O + 29], x[O + 30], x[O + 31]});
    }
}

// }}}
// intrin_cast{{{
template <class To, class From> constexpr Vc_INTRINSIC To intrin_cast(From v)
{
    static_assert(is_builtin_vector_v<From> && is_builtin_vector_v<To>);
    if constexpr (sizeof(To) == sizeof(From)) {
        return reinterpret_cast<To>(v);
    } else if constexpr (sizeof(From) > sizeof(To)) {
        return reinterpret_cast<const To &>(v);
    } else if constexpr (have_avx && sizeof(From) == 16 && sizeof(To) == 32) {
        return reinterpret_cast<To>(_mm256_castps128_ps256(
            reinterpret_cast<intrinsic_type_t<float, sizeof(From) / sizeof(float)>>(v)));
    } else if constexpr (have_avx512f && sizeof(From) == 16 && sizeof(To) == 64) {
        return reinterpret_cast<To>(_mm512_castps128_ps512(
            reinterpret_cast<intrinsic_type_t<float, sizeof(From) / sizeof(float)>>(v)));
    } else if constexpr (have_avx512f && sizeof(From) == 32 && sizeof(To) == 64) {
        return reinterpret_cast<To>(_mm512_castps256_ps512(
            reinterpret_cast<intrinsic_type_t<float, sizeof(From) / sizeof(float)>>(v)));
    } else {
        assert_unreachable<To>();
    }
}

// }}}
// auto_cast{{{
template <class T> struct auto_cast_t {
    static_assert(is_builtin_vector_v<T>);
    const T x;
    template <class U> constexpr Vc_INTRINSIC operator U() const
    {
        return intrin_cast<U>(x);
    }
};
template <class T> constexpr Vc_INTRINSIC auto_cast_t<T> auto_cast(const T &x)
{
    return {x};
}
template <class T, size_t N>
constexpr Vc_INTRINSIC auto_cast_t<typename Storage<T, N>::register_type> auto_cast(
    const Storage<T, N> &x)
{
    return {x.d};
}

// }}}
// to_bitset{{{
constexpr Vc_INTRINSIC std::bitset<1> to_bitset(bool x) { return unsigned(x); }

template <class T, class Trait = builtin_traits<T>>
Vc_INTRINSIC std::bitset<Trait::width> to_bitset(T x)
{
    constexpr bool is_sse = have_sse && sizeof(T) == 16;
    constexpr bool is_avx = have_avx && sizeof(T) == 32;
    constexpr bool is_neon128 = have_neon && sizeof(T) == 16;
    constexpr int w = sizeof(typename Trait::value_type);
    const auto intrin = detail::to_intrin(x);
    constexpr auto zero = decltype(intrin)();
    detail::unused(zero);

    if constexpr (is_neon128 && w == 1) {
        x &= T{0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80,
               0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80};
        return builtin_cast<ushort>(
            vpaddq_s8(vpaddq_s8(vpaddq_s8(x, zero), zero), zero))[0];
    } else if constexpr (is_neon128 && w == 2) {
        x &= T{0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80};
        return vpaddq_s16(vpaddq_s16(vpaddq_s16(x, zero), zero), zero)[0];
    } else if constexpr (is_neon128 && w == 4) {
        x &= T{0x1, 0x2, 0x4, 0x8};
        return vpaddq_s32(vpaddq_s32(x, zero), zero)[0];
    } else if constexpr (is_neon128 && w == 8) {
        x &= T{0x1, 0x2};
        return x[0] | x[1];
    } else if constexpr (is_sse && w == 1) {
        return _mm_movemask_epi8(intrin);
    } else if constexpr (is_sse && w == 2) {
        if constexpr (detail::have_avx512bw_vl) {
            return _mm_cmplt_epi16_mask(intrin, zero);
        } else {
            return _mm_movemask_epi8(_mm_packs_epi16(intrin, zero));
        }
    } else if constexpr (is_sse && w == 4) {
        if constexpr (detail::have_avx512vl && std::is_integral_v<T>) {
            return _mm_cmplt_epi32_mask(intrin, zero);
        } else {
            return _mm_movemask_ps(builtin_cast<float>(x));
        }
    } else if constexpr (is_sse && w == 8) {
        if constexpr (detail::have_avx512vl && std::is_integral_v<T>) {
            return _mm_cmplt_epi64_mask(intrin, zero);
        } else {
            return _mm_movemask_pd(builtin_cast<double>(x));
        }
    } else if constexpr (is_avx && w == 1) {
        return _mm256_movemask_epi8(intrin);
    } else if constexpr (is_avx && w == 2) {
        if constexpr (detail::have_avx512bw_vl) {
            return _mm256_cmplt_epi16_mask(intrin, zero);
        } else {
            return _mm_movemask_epi8(_mm_packs_epi16(detail::extract<0, 2>(intrin),
                                                     detail::extract<1, 2>(intrin)));
        }
    } else if constexpr (is_avx && w == 4) {
        if constexpr (detail::have_avx512vl && std::is_integral_v<T>) {
            return _mm256_cmplt_epi32_mask(intrin, zero);
        } else {
            return _mm256_movemask_ps(builtin_cast<float>(x));
        }
    } else if constexpr (is_avx && w == 8) {
        if constexpr (detail::have_avx512vl && std::is_integral_v<T>) {
            return _mm256_cmplt_epi64_mask(intrin, zero);
        } else {
            return _mm256_movemask_pd(builtin_cast<double>(x));
        }
    } else {
        assert_unreachable<T>();
    }
    }

// }}}
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DETAIL_BUILTINS_H_

// vim: foldmethod=marker
