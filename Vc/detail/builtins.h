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

#include "detail.h"

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
template <class T, size_t Bytes, class = std::void_t<>> struct builtin_type {};
template <class T, size_t Bytes>
struct builtin_type<
    T, Bytes,
    std::void_t<std::enable_if_t<std::conjunction_v<
        detail::is_equal_to<Bytes % sizeof(T), 0>, detail::is_vectorizable<T>>>>> {
    using type [[gnu::vector_size(Bytes)]] = T;
};

template <class T, size_t Size>
using builtin_type_t = typename builtin_type<T, Size * sizeof(T)>::type;
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

// }}}
// builtin_cast{{{1
template <class To, class From, class Traits = builtin_traits<From>>
constexpr Vc_INTRINSIC typename builtin_type<To, sizeof(From)>::type builtin_cast(From x)
{
    return reinterpret_cast<typename builtin_type<To, sizeof(From)>::type>(x);
}

// }}}
// to_intrin {{{
template <class T, class Traits = builtin_traits<T>,
          class R = intrinsic_type_t<typename Traits::value_type, Traits::width>>
constexpr Vc_INTRINSIC R to_intrin(T x)
{
    return reinterpret_cast<R>(x);
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
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DETAIL_BUILTINS_H_

// vim: foldmethod=marker
