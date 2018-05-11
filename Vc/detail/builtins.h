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
template <class T, size_t Bytes, class = detail::void_t<>> struct intrinsic_type;
template <class T, size_t Size>
using intrinsic_type_t = typename intrinsic_type<T, Size * sizeof(T)>::type;

// }}}
// is_intrinsic{{{1
template <class T> struct is_intrinsic : public std::false_type {};
template <class T> inline constexpr bool is_intrinsic_v = is_intrinsic<T>::value;

// }}}
// builtin_type {{{1
template <class T, size_t Bytes, class = detail::void_t<>> struct builtin_type {};
template <class T, size_t Bytes>
struct builtin_type<
    T, Bytes,
    detail::void_t<std::enable_if_t<detail::conjunction_v<
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
template <class T, class = void_t<>> struct is_builtin_vector : std::false_type {};
template <class T>
struct is_builtin_vector<
    T, void_t<typename builtin_type<decltype(std::declval<T>()[0]), sizeof(T)>::type>>
    : std::is_same<
          T, typename builtin_type<decltype(std::declval<T>()[0]), sizeof(T)>::type> {
};

template <class T> inline constexpr bool is_builtin_vector_v = is_builtin_vector<T>::value;

// builtin_traits{{{1
template <class T, class = void_t<>> struct builtin_traits;
template <class T>
struct builtin_traits<T, void_t<std::enable_if_t<is_builtin_vector_v<T>>>> {
    using type = T;
    using value_type = decltype(std::declval<T>()[0]);
    static constexpr int width = sizeof(T) / sizeof(value_type);

    /*
    using intrin_type = intrinsic_type_t<value_type, width>;
    static constexpr bool is_epi = std::is_integral_v<value_type>;
    static constexpr bool is_ps = std::is_same_v<value_type, float>;
    static constexpr bool is_pd = std::is_same_v<value_type, double>;
    */
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
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DETAIL_BUILTINS_H_

// vim: foldmethod=marker
