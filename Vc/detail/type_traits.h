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

#ifndef VC_SIMD_TYPE_TRAITS_H_
#define VC_SIMD_TYPE_TRAITS_H_

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail {
// integer type aliases{{{
using uchar = unsigned char;
using schar = signed char;
using ushort = unsigned short;
using uint = unsigned int;
using ulong = unsigned long;
using llong = long long;
using ullong = unsigned long long;
using wchar = wchar_t;
using char16 = char16_t;
using char32 = char32_t;

//}}}
// void_t{{{
template <class... Ts> using void_t = void;
//}}}

template <class... Ts> using all = std::conjunction<Ts...>;
template <class... Ts> using any = std::disjunction<Ts...>;
using std::negation;

// imports
using std::is_arithmetic;
using std::is_convertible;
using std::is_same;
using std::is_signed;
using std::is_unsigned;
using std::enable_if_t;

// is_equal
template <class T, T a, T b> struct is_equal : public std::false_type {
};
template <class T, T a> struct is_equal<T, a, a> : public std::true_type {
};
template <class T, T a, T b> inline constexpr bool is_equal_v = is_equal<T, a, b>::value;

// none
template <class... Ts> struct none : public negation<std::disjunction<Ts...>> {};

// sizeof
template <class T, std::size_t Expected>
struct has_expected_sizeof : public std::integral_constant<bool, sizeof(T) == Expected> {
};

// value aliases
template <class... Ts>
inline constexpr bool conjunction_v = all<Ts...>::value;
template <class... Ts> inline constexpr bool disjunction_v = std::disjunction<Ts...>::value;
template <class T> inline constexpr bool negation_v = negation<T>::value;
template <class... Ts> inline constexpr bool none_v = none<Ts...>::value;
template <class T, std::size_t Expected>
inline constexpr bool has_expected_sizeof_v = has_expected_sizeof<T, Expected>::value;

// value_type_or_identity
template <class T> typename T::value_type value_type_or_identity_impl(int);
template <class T> T value_type_or_identity_impl(float);
template <class T>
using value_type_or_identity = decltype(value_type_or_identity_impl<T>(int()));

// is_vectorizable {{{
template <class T> struct is_vectorizable : public std::is_arithmetic<T> {};
template <> struct is_vectorizable<bool> : public std::false_type {};
template <class T> inline constexpr bool is_vectorizable_v = is_vectorizable<T>::value;

// }}}
// is_possible_loadstore_conversion {{{
template <class Ptr, class ValueType>
struct is_possible_loadstore_conversion
    : all<is_vectorizable<Ptr>, is_vectorizable<ValueType>> {
};
template <> struct is_possible_loadstore_conversion<bool, bool> : std::true_type {
};

// }}}
// is_less_than {{{
template <int A, int B> struct is_less_than : public std::integral_constant<bool, (A < B)> {
};
// }}}
// is_equal_to {{{
template <int A, int B>
struct is_equal_to : public std::integral_constant<bool, (A == B)> {
};
// }}}
template <size_t X> using size_constant = std::integral_constant<size_t, X>;

// is_bitmask{{{
template <class T, class = std::void_t<>> struct is_bitmask : std::false_type {
    constexpr is_bitmask(const T &) noexcept {}
};
template <class T> inline constexpr bool is_bitmask_v = is_bitmask<T>::value;

/* the Storage<bool, N> case:
template <class T>
struct is_bitmask<T, std::enable_if_t<(sizeof(T) < T::width)>> : std::true_type {
    constexpr is_bitmask(const T &) noexcept {}
};*/

// the __mmaskXX case:
template <class T>
struct is_bitmask<
    T, std::void_t<decltype(std::declval<unsigned &>() = std::declval<T>() & 1u)>>
    : std::true_type {
    constexpr is_bitmask(const T &) noexcept {}
};

// }}}
// int_for_sizeof{{{
template <class T, size_t = sizeof(T)> struct int_for_sizeof;
template <class T> struct int_for_sizeof<T, 1> { using type = signed char; };
template <class T> struct int_for_sizeof<T, 2> { using type = signed short; };
template <class T> struct int_for_sizeof<T, 4> { using type = signed int; };
template <class T> struct int_for_sizeof<T, 8> { using type = signed long long; };
template <class T> using int_for_sizeof_t = typename int_for_sizeof<T>::type;

// }}}
// equal_int_type{{{
/**
 * \internal
 * TODO: rename to same_value_representation
 * Type trait to find the equivalent integer type given a(n) (un)signed long type.
 */
template <class T, size_t = sizeof(T)> struct equal_int_type;
template <> struct equal_int_type< long, sizeof(int)> { using type =    int; };
template <> struct equal_int_type< long, sizeof(llong)> { using type =  llong; };
template <> struct equal_int_type<ulong, sizeof(uint)> { using type =   uint; };
template <> struct equal_int_type<ulong, sizeof(ullong)> { using type = ullong; };
template <> struct equal_int_type< char, 1> { using type = std::conditional_t<std::is_signed_v<char>, schar, uchar>; };
template <size_t N> struct equal_int_type<char16_t, N> { using type = std::uint_least16_t; };
template <size_t N> struct equal_int_type<char32_t, N> { using type = std::uint_least32_t; };
template <> struct equal_int_type<wchar_t, 1> { using type = std::conditional_t<std::is_signed_v<wchar_t>, schar, uchar>; };
template <> struct equal_int_type<wchar_t, sizeof(short)> { using type = std::conditional_t<std::is_signed_v<wchar_t>, short, ushort>; };
template <> struct equal_int_type<wchar_t, sizeof(int)> { using type = std::conditional_t<std::is_signed_v<wchar_t>, int, uint>; };

template <class T> using equal_int_type_t = typename equal_int_type<T>::type;

// }}}
// has_same_value_representation{{{
template <class T, class = void_t<>>
struct has_same_value_representation : std::false_type {
};

template <class T>
struct has_same_value_representation<T, void_t<typename equal_int_type<T>::type>>
    : std::true_type {
};

template <class T>
inline constexpr bool has_same_value_representation_v =
    has_same_value_representation<T>::value;

// }}}
// is_fixed_size_abi{{{
template <class T> struct is_fixed_size_abi : std::false_type {
};

template <int N> struct is_fixed_size_abi<simd_abi::fixed_size<N>> : std::true_type {
};

template <class T> inline constexpr bool is_fixed_size_abi_v = is_fixed_size_abi<T>::value;

// }}}
// is_callable{{{
/*
template <class F, class... Args>
auto is_callable_dispatch(int, F &&fun, Args &&... args)
    -> std::conditional_t<true, std::true_type,
                          decltype(fun(std::forward<Args>(args)...))>;
template <class F, class... Args>
std::false_type is_callable_dispatch(float, F &&, Args &&...);

template <class... Args, class F> constexpr bool is_callable(F &&)
{
    return decltype(is_callable_dispatch(int(), std::declval<F>(), std::declval<Args>()...))::value;
}
*/

template <class F, class = std::void_t<>, class... Args>
struct is_callable : std::false_type {
};

template <class F, class... Args>
struct is_callable<F, std::void_t<decltype(std::declval<F>()(std::declval<Args>()...))>,
                   Args...> : std::true_type {
};

template <class F, class... Args>
inline constexpr bool is_callable_v = is_callable<F, void, Args...>::value;

#define Vc_TEST_LAMBDA(...)                                                              \
    decltype(                                                                            \
        [](auto &&... arg_s_) -> decltype(fun_(std::declval<decltype(arg_s_)>()...)) * { \
            return nullptr;                                                              \
        })

#define Vc_IS_CALLABLE(fun_, ...)                                                        \
    decltype(Vc::detail::is_callable_dispatch(int(), Vc_TEST_LAMBDA(fun_),               \
                                              __VA_ARGS__))::value

// }}}
// constexpr feature detection{{{
constexpr inline bool have_mmx = 0
#ifdef Vc_HAVE_MMX
                                 + 1
#endif
    ;
constexpr inline bool have_sse = 0
#ifdef Vc_HAVE_SSE
                                 + 1
#endif
    ;
constexpr inline bool have_sse2 = 0
#ifdef Vc_HAVE_SSE2
                                  + 1
#endif
    ;
constexpr inline bool have_sse3 = 0
#ifdef Vc_HAVE_SSE3
                                  + 1
#endif
    ;
constexpr inline bool have_ssse3 = 0
#ifdef Vc_HAVE_SSSE3
                                   + 1
#endif
    ;
constexpr inline bool have_sse4_1 = 0
#ifdef Vc_HAVE_SSE4_1
                                    + 1
#endif
    ;
constexpr inline bool have_sse4_2 = 0
#ifdef Vc_HAVE_SSE4_2
                                    + 1
#endif
    ;
constexpr inline bool have_xop = 0
#ifdef Vc_HAVE_XOP
                                 + 1
#endif
    ;
constexpr inline bool have_avx = 0
#ifdef Vc_HAVE_AVX
                                 + 1
#endif
    ;
constexpr inline bool have_avx2 = 0
#ifdef Vc_HAVE_AVX2
                                  + 1
#endif
    ;
constexpr inline bool have_bmi = 0
#ifdef Vc_HAVE_BMI
                                 + 1
#endif
    ;
constexpr inline bool have_bmi2 = 0
#ifdef Vc_HAVE_BMI2
                                  + 1
#endif
    ;
constexpr inline bool have_lzcnt = 0
#ifdef Vc_HAVE_LZCNT
                                   + 1
#endif
    ;
constexpr inline bool have_sse4a = 0
#ifdef Vc_HAVE_SSE4A
                                   + 1
#endif
    ;
constexpr inline bool have_fma = 0
#ifdef Vc_HAVE_FMA
                                 + 1
#endif
    ;
constexpr inline bool have_fma4 = 0
#ifdef Vc_HAVE_FMA4
                                  + 1
#endif
    ;
constexpr inline bool have_f16c = 0
#ifdef Vc_HAVE_F16C
                                  + 1
#endif
    ;
constexpr inline bool have_popcnt = 0
#ifdef Vc_HAVE_POPCNT
                                    + 1
#endif
    ;
constexpr inline bool have_avx512f = 0
#ifdef Vc_HAVE_AVX512F
                                     + 1
#endif
    ;
constexpr inline bool have_avx512dq = 0
#ifdef Vc_HAVE_AVX512DQ
                                      + 1
#endif
    ;
constexpr inline bool have_avx512vl = 0
#ifdef Vc_HAVE_AVX512VL
                                      + 1
#endif
    ;
constexpr inline bool have_avx512bw = 0
#ifdef Vc_HAVE_AVX512BW
                                      + 1
#endif
    ;
constexpr inline bool have_avx512dq_vl = have_avx512dq && have_avx512vl;
constexpr inline bool have_avx512bw_vl = have_avx512bw && have_avx512vl;

constexpr inline bool have_neon = 0
#ifdef Vc_HAVE_NEON
                                  + 1
#endif
    ;
// }}}

}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_SIMD_TYPE_TRAITS_H_
