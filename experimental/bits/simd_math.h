#ifndef _GLIBCXX_EXPERIMENTAL_SIMD_MATH_H_
#define _GLIBCXX_EXPERIMENTAL_SIMD_MATH_H_

//#pragma GCC system_header

#if __cplusplus >= 201703L

#include "simd_abis.h"
#include <utility>
#include <iomanip>

static_assert(std::is_same_v<bool, decltype(std::isnan(double()))>);
static_assert(std::is_same_v<bool, decltype(std::isinf(double()))>);

_GLIBCXX_SIMD_BEGIN_NAMESPACE
template <class _T, class _V> using __samesize = fixed_size_simd<_T, _V::size()>;
// __math_return_type {{{
template <class DoubleR, class _T, class _Abi> struct __math_return_type;
template <class DoubleR, class _T, class _Abi>
using __math_return_type_t = typename __math_return_type<DoubleR, _T, _Abi>::type;

template <class _T, class _Abi> struct __math_return_type<double, _T, _Abi> {
    using type = std::experimental::simd<_T, _Abi>;
};
template <class _T, class _Abi> struct __math_return_type<bool, _T, _Abi> {
    using type = std::experimental::simd_mask<_T, _Abi>;
};
template <class DoubleR, class _T, class _Abi> struct __math_return_type {
    using type = std::experimental::fixed_size_simd<DoubleR, simd_size_v<_T, _Abi>>;
};
//}}}
// _GLIBCXX_SIMD_MATH_CALL_ {{{
#define _GLIBCXX_SIMD_MATH_CALL_(__name)                                                 \
    template <class _T, class _Abi, class...,                                            \
              class _R = std::experimental::__math_return_type_t<                        \
                  decltype(std::__name(std::declval<double>())), _T, _Abi>>              \
    enable_if_t<std::is_floating_point_v<_T>, _R> __name(                                \
        std::experimental::simd<_T, _Abi> __x)                                           \
    {                                                                                    \
        using _V = std::experimental::simd<_T, _Abi>;                                    \
        return std::experimental::__impl_or_fallback(                                    \
            [](const auto& __xx)                                                         \
                -> decltype(                                                             \
                    _R(std::experimental::__private_init,                                \
                       std::experimental::__get_impl_t<decltype(__xx)>::__##__name(      \
                           std::experimental::__data(__xx)))) {                          \
                return {std::experimental::__private_init,                               \
                        std::experimental::__get_impl_t<decltype(__xx)>::__##__name(     \
                            std::experimental::__data(__xx))};                           \
            },                                                                           \
            [](const _V& __xx) {                                                         \
                if constexpr (std::experimental::is_simd_mask_v<_R>) {                   \
                    return _R(std::experimental::__private_init,                         \
                              [&](auto __i) { return std::__name(__xx[__i]); });         \
                } else {                                                                 \
                    return _R([&](auto __i) { return std::__name(__xx[__i]); });         \
                }                                                                        \
            },                                                                           \
            __x);                                                                        \
    }

// }}}
//__extra_argument_type{{{
template <class _U, class _T, class _Abi> struct __extra_argument_type;

template <class _T, class _Abi> struct __extra_argument_type<_T *, _T, _Abi> {
    using type = std::experimental::simd<_T, _Abi> *;
    static constexpr double *declval();
    _GLIBCXX_SIMD_INTRINSIC static constexpr auto __data(type __x) { return &std::experimental::__data(*__x); }
    static constexpr bool __needs_temporary_scalar = true;
};
template <class _U, class _T, class _Abi> struct __extra_argument_type<_U *, _T, _Abi> {
    static_assert(std::is_integral_v<_U>);
    using type = std::experimental::fixed_size_simd<_U, std::experimental::simd_size_v<_T, _Abi>> *;
    static constexpr _U *declval();
    _GLIBCXX_SIMD_INTRINSIC static constexpr auto __data(type __x) { return &std::experimental::__data(*__x); }
    static constexpr bool __needs_temporary_scalar = true;
};
template <class _T, class _Abi> struct __extra_argument_type<_T, _T, _Abi> {
    using type = std::experimental::simd<_T, _Abi>;
    static constexpr double declval();
    _GLIBCXX_SIMD_INTRINSIC static constexpr decltype(auto) __data(const type &__x)
    {
        return std::experimental::__data(__x);
    }
    static constexpr bool __needs_temporary_scalar = false;
};
template <class _U, class _T, class _Abi> struct __extra_argument_type {
    static_assert(std::is_integral_v<_U>);
    using type = std::experimental::fixed_size_simd<_U, std::experimental::simd_size_v<_T, _Abi>>;
    static constexpr _U declval();
    _GLIBCXX_SIMD_INTRINSIC static constexpr decltype(auto) __data(const type &__x)
    {
        return std::experimental::__data(__x);
    }
    static constexpr bool __needs_temporary_scalar = false;
};
//}}}
// _GLIBCXX_SIMD_MATH_CALL2_ {{{
#define _GLIBCXX_SIMD_MATH_CALL2_(__name, arg2_)                                           \
    template <                                                                             \
        class _T, class _Abi, class...,                                                    \
        class _Arg2 = std::experimental::__extra_argument_type<arg2_, _T, _Abi>,           \
        class _R    = std::experimental::__math_return_type_t<                             \
            decltype(std::__name(std::declval<double>(), _Arg2::declval())), _T, _Abi>> \
    enable_if_t<std::is_floating_point_v<_T>, _R> __name(                                  \
        const std::experimental::simd<_T, _Abi>& __xx, const typename _Arg2::type& __yy)   \
    {                                                                                      \
        using _V = std::experimental::simd<_T, _Abi>;                                      \
        return std::experimental::__impl_or_fallback(                                      \
            [](const auto& __x, const auto& __y)                                           \
                -> decltype(                                                               \
                    _R(std::experimental::__private_init,                                  \
                       std::experimental::__get_impl_t<decltype(__x)>::__##__name(         \
                           std::experimental::__data(__x), _Arg2::__data(__y)))) {         \
                return {std::experimental::__private_init,                                 \
                        std::experimental::__get_impl_t<decltype(__x)>::__##__name(        \
                            std::experimental::__data(__x), _Arg2::__data(__y))};          \
            },                                                                             \
            [](const _V& __x, const auto& __y) {                                           \
                auto&& gen = [&](auto __i) {                                               \
                    if constexpr (_Arg2::__needs_temporary_scalar) {                       \
                        const auto& yy = *__y;                                             \
                        auto __tmp     = yy[__i];                                          \
                        auto __ret     = std::__name(__x[__i], &__tmp);                    \
                        (*__y)[__i]    = __tmp;                                            \
                        return __ret;                                                      \
                    } else {                                                               \
                        return std::__name(__x[__i], __y[__i]);                            \
                    }                                                                      \
                };                                                                         \
                if constexpr (std::experimental::is_simd_mask_v<_R>) {                     \
                    return _R(std::experimental::__private_init, gen);                     \
                } else {                                                                   \
                    return _R(gen);                                                        \
                }                                                                          \
            },                                                                             \
            __xx, __yy);                                                                   \
    }                                                                                      \
    template <class _U, class _T, class _Abi>                                              \
    _GLIBCXX_SIMD_INTRINSIC std::experimental::__math_return_type_t<                       \
        decltype(std::__name(                                                              \
            std::declval<double>(),                                                        \
            std::declval<enable_if_t<                                                      \
                std::conjunction_v<                                                        \
                    std::is_same<arg2_, _T>,                                               \
                    std::negation<std::is_same<std::decay_t<_U>,                           \
                                               std::experimental::simd<_T, _Abi>>>,        \
                    std::is_convertible<_U, std::experimental::simd<_T, _Abi>>,            \
                    std::is_floating_point<_T>>,                                           \
                double>>())),                                                              \
        _T, _Abi>                                                                          \
    __name(_U&& __xx, const std::experimental::simd<_T, _Abi>& __yy)                       \
    {                                                                                      \
        return std::experimental::__name(                                                  \
            std::experimental::simd<_T, _Abi>(std::forward<_U>(__xx)), __yy);              \
    }

// }}}
// _GLIBCXX_SIMD_MATH_CALL3_ {{{
#define _GLIBCXX_SIMD_MATH_CALL3_(__name, arg2_, arg3_)                                  \
    template <class _T, class _Abi, class...,                                            \
              class _Arg2 = std::experimental::__extra_argument_type<arg2_, _T, _Abi>,   \
              class _Arg3 = std::experimental::__extra_argument_type<arg3_, _T, _Abi>,   \
              class _R    = std::experimental::__math_return_type_t<                     \
                  decltype(std::__name(std::declval<double>(), _Arg2::declval(),      \
                                       _Arg3::declval())),                            \
                  _T, _Abi>>                                                          \
    enable_if_t<std::is_floating_point_v<_T>, _R> __name(                                \
        std::experimental::simd<_T, _Abi> __xx, typename _Arg2::type __yy,               \
        typename _Arg3::type __zz)                                                       \
    {                                                                                    \
        using _V = std::experimental::simd<_T, _Abi>;                                    \
        return std::experimental::__impl_or_fallback(                                    \
            [](const auto& __x, const auto& __y, const auto& __z)                        \
                -> decltype(                                                             \
                    _R(std::experimental::__private_init,                                \
                       std::experimental::__get_impl_t<decltype(__x)>::__##__name(       \
                           std::experimental::__data(__x), _Arg2::__data(__y),           \
                           _Arg3::__data(__z)))) {                                       \
                return {std::experimental::__private_init,                               \
                        std::experimental::__get_impl_t<decltype(__x)>::__##__name(      \
                            std::experimental::__data(__x), _Arg2::__data(__y),          \
                            _Arg3::__data(__z))};                                        \
            },                                                                           \
            [](const _V& __x, const auto& __y, const auto& __z) {                        \
                return _R([&](auto __i) {                                                \
                    if constexpr (_Arg3::__needs_temporary_scalar) {                     \
                        const auto& __zz = *__z;                                         \
                        auto __tmp       = __zz[__i];                                    \
                        auto __ret       = std::__name(__x[__i], __y[__i], &__tmp);      \
                        (*__z)[__i]      = __tmp;                                        \
                        return __ret;                                                    \
                    } else {                                                             \
                        return std::__name(__x[__i], __y[__i], __z[__i]);                \
                    }                                                                    \
                });                                                                      \
            },                                                                           \
            __xx, __yy, __zz);                                                           \
    }                                                                                    \
    template <class _T, class _U, class _V, class..., class _TT = std::decay_t<_T>,      \
              class _UU = std::decay_t<_U>, class _VV = std::decay_t<_V>,                \
              class _Simd =                                                              \
                  std::conditional_t<std::experimental::is_simd_v<_UU>, _UU, _VV>>       \
    _GLIBCXX_SIMD_INTRINSIC decltype(                                                    \
        std::experimental::__name(_Simd(std::declval<_T>()), _Simd(std::declval<_U>()),  \
                                  _Simd(std::declval<_V>())))                            \
    __name(_T&& __xx, _U&& __yy, _V&& __zz)                                              \
    {                                                                                    \
        return std::experimental::__name(_Simd(std::forward<_T>(__xx)),                  \
                                         _Simd(std::forward<_U>(__yy)),                  \
                                         _Simd(std::forward<_V>(__zz)));                 \
    }

// }}}
// __cosSeries {{{
template < typename _Abi>
_GLIBCXX_SIMD_ALWAYS_INLINE static simd<float, _Abi> __cosSeries(const simd<float, _Abi> &__x)
{
    using _C = __trig<float>;
    const simd<float, _Abi> __x2 = __x * __x;
    return ((_C::cos_c2  * __x2 +
             _C::cos_c1) * __x2 +
             _C::cos_c0) * (__x2 * __x2)
        - .5f * __x2 + 1.f;
}
template <typename _Abi>
_GLIBCXX_SIMD_ALWAYS_INLINE static simd<double, _Abi> __cosSeries(const simd<double, _Abi> &__x)
{
    using _C = __trig<double>;
    const simd<double, _Abi> __x2 = __x * __x;
    return (((((_C::cos_c5  * __x2 +
                _C::cos_c4) * __x2 +
                _C::cos_c3) * __x2 +
                _C::cos_c2) * __x2 +
                _C::cos_c1) * __x2 +
                _C::cos_c0) * (__x2 * __x2)
        - .5 * __x2 + 1.;
}

// }}}
// __sinSeries {{{
template <typename _Abi>
_GLIBCXX_SIMD_ALWAYS_INLINE static simd<float, _Abi> __sinSeries(const simd<float, _Abi>& __x)
{
    using _C = __trig<float>;
    const simd<float, _Abi> __x2 = __x * __x;
    return ((_C::sin_c2  * __x2 +
             _C::sin_c1) * __x2 +
             _C::sin_c0) * (__x2 * __x)
        + __x;
}

template <typename _Abi>
_GLIBCXX_SIMD_ALWAYS_INLINE static simd<double, _Abi> __sinSeries(const simd<double, _Abi> &__x)
{
    using _C = __trig<double>;
    const simd<double, _Abi> __x2 = __x * __x;
    return (((((_C::sin_c5  * __x2 +
                _C::sin_c4) * __x2 +
                _C::sin_c3) * __x2 +
                _C::sin_c2) * __x2 +
                _C::sin_c1) * __x2 +
                _C::sin_c0) * (__x2 * __x)
        + __x;
}

// }}}
// __foldInput {{{
template <class _Abi>
_GLIBCXX_SIMD_ALWAYS_INLINE std::pair<simd<float, _Abi>, rebind_simd_t<int, simd<float, _Abi>>> __foldInput(
    simd<float, _Abi> __x)
{
    using _V = simd<float, _Abi>;
    using _C = __trig<float>;
    using _IV = rebind_simd_t<int, _V>;

    __x = abs(__x);
#if _GLIBCXX_SIMD_HAVE_FMA4 || _GLIBCXX_SIMD_HAVE_FMA
    rebind_simd_t<int, _V> __quadrant =
        static_simd_cast<_IV>(__x * _C::_4_pi + 1.f);  // prefer the fma here
    __quadrant &= ~1;
#else
    rebind_simd_t<int, _V> __quadrant = static_simd_cast<_IV>(__x * _C::_4_pi);
    __quadrant += __quadrant & 1;
#endif
    const _V __y = static_simd_cast<_V>(__quadrant);
    __quadrant &= 7;

    return {((__x - __y * _C::pi_4_hi) - __y * _C::pi_4_rem1) - __y * _C::pi_4_rem2, __quadrant};
}

template <typename _Abi>
_GLIBCXX_SIMD_ALWAYS_INLINE static std::pair<simd<double, _Abi>, rebind_simd_t<int, simd<double, _Abi>>>
__foldInput(simd<double, _Abi> __x)
{
    using _V = simd<double, _Abi>;
    using _C = __trig<double>;
    using _IV = rebind_simd_t<int, _V>;

    __x = abs(__x);
    _V __y = trunc(__x / _C::pi_4);  // * _C::4_pi would work, but is >twice as imprecise
    _V __z = __y - trunc(__y * _C::_1_16) * _C::_16;  // __y modulo 16
    _IV __quadrant = static_simd_cast<_IV>(__z);
    const auto mask = (__quadrant & 1) != 0;
    ++where(mask, __quadrant);
    where(static_simd_cast<typename _V::mask_type>(mask), __y) += _V(1);
    __quadrant &= 7;

    // since __y is an integer we don't need to split __y into low and high parts until the
    // integer
    // requires more bits than there are zero bits at the end of _pi_4_hi (30 bits -> 1e9)
    return {((__x - __y * _C::pi_4_hi) - __y * _C::pi_4_rem1) - __y * _C::pi_4_rem2, __quadrant};
}

// }}}
// __extract_exponent_bits {{{
template <class _Abi>
rebind_simd_t<int, simd<float, _Abi>> __extract_exponent_bits(const simd<float, _Abi> &__v)
{
    using namespace std::experimental::__proposed;
    using namespace std::experimental::__proposed::float_bitwise_operators;
    constexpr simd<float, _Abi> __exponent_mask =
        std::numeric_limits<float>::infinity();  // 0x7f800000
    return simd_reinterpret_cast<rebind_simd_t<int, simd<float, _Abi>>>(__v & __exponent_mask);
}

template <class _Abi>
rebind_simd_t<int, simd<double, _Abi>> __extract_exponent_bits(const simd<double, _Abi> &__v)
{
    using namespace std::experimental::__proposed;
    using namespace std::experimental::__proposed::float_bitwise_operators;
    const simd<double, _Abi> __exponent_mask =
        std::numeric_limits<double>::infinity();  // 0x7ff0000000000000
    constexpr auto _N = simd_size_v<double, _Abi> * 2;
    constexpr auto _Max = simd_abi::max_fixed_size<int>;
    if constexpr (_N > _Max) {
        const auto tup = split<_Max / 2, (_N - _Max) / 2>(__v & __exponent_mask);
        return concat(
            shuffle<strided<2, 1>>(
                simd_reinterpret_cast<simd<int, simd_abi::deduce_t<int, _Max>>>(
                    std::get<0>(tup))),
            shuffle<strided<2, 1>>(
                simd_reinterpret_cast<simd<int, simd_abi::deduce_t<int, _N - _Max>>>(
                    std::get<1>(tup))));
    } else {
        return shuffle<strided<2, 1>>(
            simd_reinterpret_cast<simd<int, simd_abi::deduce_t<int, _N>>>(__v &
                                                                         __exponent_mask));
    }
}

// }}}
// __impl_or_fallback {{{
template <class ImplFun, class FallbackFun, class... Args>
_GLIBCXX_SIMD_INTRINSIC auto __impl_or_fallback_dispatch(int, ImplFun&& __impl_fun,
                                                         FallbackFun&&, Args&&... __args)
    -> decltype(__impl_fun(std::forward<Args>(__args)...))
{
    return __impl_fun(std::forward<Args>(__args)...);
}

template <class ImplFun, class FallbackFun, class... Args>
inline auto __impl_or_fallback_dispatch(float, ImplFun&&, FallbackFun&& __fallback_fun,
                                        Args&&... __args)
    -> decltype(__fallback_fun(std::forward<Args>(__args)...))
{
    return __fallback_fun(std::forward<Args>(__args)...);
}

template <class... Args> _GLIBCXX_SIMD_INTRINSIC auto __impl_or_fallback(Args&&... __args)
{
    return __impl_or_fallback_dispatch(int(), std::forward<Args>(__args)...);
}  //}}}

// trigonometric functions {{{
_GLIBCXX_SIMD_MATH_CALL_(acos)
_GLIBCXX_SIMD_MATH_CALL_(asin)
_GLIBCXX_SIMD_MATH_CALL_(atan)
_GLIBCXX_SIMD_MATH_CALL2_(atan2, _T)

/*
 * algorithm for sine and cosine:
 *
 * The result can be calculated with sine or cosine depending on the π/4 section the input
 * is in.
 * sine   ≈ __x + __x³
 * cosine ≈ 1 - __x²
 *
 * sine:
 * Map -__x to __x and invert the output
 * Extend precision of __x - n * π/4 by calculating
 * ((__x - n * p1) - n * p2) - n * p3 (p1 + p2 + p3 = π/4)
 *
 * Calculate Taylor series with tuned coefficients.
 * Fix sign.
 */
//cos{{{
template <class _T, class _Abi>
enable_if_t<std::is_floating_point<_T>::value, simd<_T, _Abi>> cos(simd<_T, _Abi> __x)
{
    using _V = simd<_T, _Abi>;
    using _M = typename _V::mask_type;

    auto __folded = __foldInput(__x);
    const _V &__z = __folded.first;
    auto &__quadrant = __folded.second;
    _M __sign = static_simd_cast<_M>(__quadrant > 3);
    where(__quadrant > 3, __quadrant) -= 4;
    __sign ^= static_simd_cast<_M>(__quadrant > 1);

    _V __y = __cosSeries(__z);
    where(static_simd_cast<_M>(__quadrant == 1 || __quadrant == 2), __y) = __sinSeries(__z);
    where(__sign, __y) = -__y;
    _GLIBCXX_SIMD_DEBUG(_Cosine)
        (_GLIBCXX_SIMD_PRETTY_PRINT(__x))
        (_GLIBCXX_SIMD_PRETTY_PRINT(__sign))
        (_GLIBCXX_SIMD_PRETTY_PRINT(__z))
        (_GLIBCXX_SIMD_PRETTY_PRINT(__folded.second))
        (_GLIBCXX_SIMD_PRETTY_PRINT(__quadrant))
        (_GLIBCXX_SIMD_PRETTY_PRINT(__y));
    return __y;
}

template <class _T>
_GLIBCXX_SIMD_ALWAYS_INLINE
    enable_if_t<std::is_floating_point<_T>::value, simd<_T, simd_abi::scalar>>
    cos(simd<_T, simd_abi::scalar> __x)
{
    return std::cos(__data(__x));
}
//}}}
//sin{{{
template <class _T, class _Abi>
enable_if_t<std::is_floating_point<_T>::value, simd<_T, _Abi>> sin(simd<_T, _Abi> __x)
{
    using _V = simd<_T, _Abi>;
    using _M = typename _V::mask_type;

    auto __folded = __foldInput(__x);
    const _V &__z = __folded.first;
    auto &__quadrant = __folded.second;
    const _M __sign = (__x < 0) ^ static_simd_cast<_M>(__quadrant > 3);
    where(__quadrant > 3, __quadrant) -= 4;

    _V __y = __sinSeries(__z);
    where(static_simd_cast<_M>(__quadrant == 1 || __quadrant == 2), __y) = __cosSeries(__z);
    where(__sign, __y) = -__y;
    _GLIBCXX_SIMD_DEBUG(_Sine)
        (_GLIBCXX_SIMD_PRETTY_PRINT(__x))
        (_GLIBCXX_SIMD_PRETTY_PRINT(__sign))
        (_GLIBCXX_SIMD_PRETTY_PRINT(__z))
        (_GLIBCXX_SIMD_PRETTY_PRINT(__folded.second))
        (_GLIBCXX_SIMD_PRETTY_PRINT(__quadrant))
        (_GLIBCXX_SIMD_PRETTY_PRINT(__y));
    return __y;
}

template <class _T>
_GLIBCXX_SIMD_ALWAYS_INLINE
    enable_if_t<std::is_floating_point<_T>::value, simd<_T, simd_abi::scalar>>
    sin(simd<_T, simd_abi::scalar> __x)
{
    return std::sin(__data(__x));
}
//}}}

_GLIBCXX_SIMD_MATH_CALL_(tan)
_GLIBCXX_SIMD_MATH_CALL_(acosh)
_GLIBCXX_SIMD_MATH_CALL_(asinh)
_GLIBCXX_SIMD_MATH_CALL_(atanh)
_GLIBCXX_SIMD_MATH_CALL_(cosh)
_GLIBCXX_SIMD_MATH_CALL_(sinh)
_GLIBCXX_SIMD_MATH_CALL_(tanh)
// }}}
// exponential functions {{{
_GLIBCXX_SIMD_MATH_CALL_(exp)
_GLIBCXX_SIMD_MATH_CALL_(exp2)
_GLIBCXX_SIMD_MATH_CALL_(expm1)
// }}}
// frexp {{{
template <class _T, size_t _N> __storage<_T, _N> __getexp(__storage<_T, _N> __x)
{
    if constexpr (__have_avx512vl && __is_sse_ps<_T, _N>()) {
        return _mm_getexp_ps(__x);
    } else if constexpr (__have_avx512f && __is_sse_ps<_T, _N>()) {
        return __lo128(_mm512_getexp_ps(__auto_bitcast(__x)));
    } else if constexpr (__have_avx512vl && __is_sse_pd<_T, _N>()) {
        return _mm_getexp_pd(__x);
    } else if constexpr (__have_avx512f && __is_sse_pd<_T, _N>()) {
        return __lo128(_mm512_getexp_pd(__auto_bitcast(__x)));
    } else if constexpr (__have_avx512vl && __is_avx_ps<_T, _N>()) {
        return _mm256_getexp_ps(__x);
    } else if constexpr (__have_avx512f && __is_avx_ps<_T, _N>()) {
        return __lo256(_mm512_getexp_ps(__auto_bitcast(__x)));
    } else if constexpr (__have_avx512vl && __is_avx_pd<_T, _N>()) {
        return _mm256_getexp_pd(__x);
    } else if constexpr (__have_avx512f && __is_avx_pd<_T, _N>()) {
        return __lo256(_mm512_getexp_pd(__auto_bitcast(__x)));
    } else if constexpr (__is_avx512_ps<_T, _N>()) {
        return _mm512_getexp_ps(__x);
    } else if constexpr (__is_avx512_pd<_T, _N>()) {
        return _mm512_getexp_pd(__x);
    } else {
        __assert_unreachable<_T>();
    }
}

template <class _T, size_t _N> __storage<_T, _N> __getmant(__storage<_T, _N> __x)
{
    if constexpr (__have_avx512vl && __is_sse_ps<_T, _N>()) {
        return _mm_getmant_ps(__x, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src);
    } else if constexpr (__have_avx512f && __is_sse_ps<_T, _N>()) {
        return __lo128(
            _mm512_getmant_ps(__auto_bitcast(__x), _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src));
    } else if constexpr (__have_avx512vl && __is_sse_pd<_T, _N>()) {
        return _mm_getmant_pd(__x, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src);
    } else if constexpr (__have_avx512f && __is_sse_pd<_T, _N>()) {
        return __lo128(
            _mm512_getmant_pd(__auto_bitcast(__x), _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src));
    } else if constexpr (__have_avx512vl && __is_avx_ps<_T, _N>()) {
        return _mm256_getmant_ps(__x, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src);
    } else if constexpr (__have_avx512f && __is_avx_ps<_T, _N>()) {
        return __lo256(
            _mm512_getmant_ps(__auto_bitcast(__x), _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src));
    } else if constexpr (__have_avx512vl && __is_avx_pd<_T, _N>()) {
        return _mm256_getmant_pd(__x, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src);
    } else if constexpr (__have_avx512f && __is_avx_pd<_T, _N>()) {
        return __lo256(
            _mm512_getmant_pd(__auto_bitcast(__x), _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src));
    } else if constexpr (__is_avx512_ps<_T, _N>()) {
        return _mm512_getmant_ps(__x, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src);
    } else if constexpr (__is_avx512_pd<_T, _N>()) {
        return _mm512_getmant_pd(__x, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src);
    } else {
        __assert_unreachable<_T>();
    }
}

/**
 * splits \p __v into exponent and mantissa, the sign is kept with the mantissa
 *
 * The return value will be in the range [0.5, 1.0[
 * The \p __e value will be an integer defining the power-of-two exponent
 */
template <class _T, class _Abi>
enable_if_t<std::is_floating_point_v<_T>, simd<_T, _Abi>> frexp(
    const simd<_T, _Abi> &__x, __samesize<int, simd<_T, _Abi>> *__exp)
{
    if constexpr (simd_size_v<_T, _Abi> == 1) {
        int __tmp;
        const auto __r = std::frexp(__x[0], &__tmp);
        (*__exp)[0] = __tmp;
        return __r;
    } else if constexpr (__is_fixed_size_abi_v<_Abi>) {
        return {__private_init, __get_impl_t<simd<_T, _Abi>>::__frexp(__data(__x), __data(*__exp))};
    } else if constexpr (__have_avx512f) {
        using _IV = __samesize<int, simd<_T, _Abi>>;
        constexpr size_t _N = simd_size_v<_T, _Abi>;
        constexpr size_t NI = _N < 4 ? 4 : _N;
        const auto __v = __data(__x);
        const auto isnonzero = __get_impl_t<simd<_T, _Abi>>::isnonzerovalue_mask(__v._M_data);
        const auto __e =
            __to_intrin(__blend(isnonzero, __vector_type_t<int, NI>(),
                                1 + __convert<__storage<int, NI>>(__getexp(__v))._M_data));
        _GLIBCXX_SIMD_DEBUG(_Frexp)
        (std::hex, _GLIBCXX_SIMD_PRETTY_PRINT(int(isnonzero)), std::dec,
         _GLIBCXX_SIMD_PRETTY_PRINT(__e), _GLIBCXX_SIMD_PRETTY_PRINT(__getexp(__v)),
         _GLIBCXX_SIMD_PRETTY_PRINT(
             __to_intrin(1 + __convert<__storage<int, NI>>(__getexp(__v))._M_data)));
        __vector_store<_N * sizeof(int)>(__e, __exp, overaligned<alignof(_IV)>);
        return {__private_init, __blend(isnonzero, __v, __getmant(__v))};
    } else {
        // fallback implementation
        static_assert(sizeof(_T) == 4 || sizeof(_T) == 8);
        using _V = simd<_T, _Abi>;
        using _IV = rebind_simd_t<int, _V>;
        using _IM = typename _IV::mask_type;
        using _Limits = std::numeric_limits<_T>;
        using namespace std::experimental::__proposed;
        using namespace std::experimental::__proposed::float_bitwise_operators;

        constexpr int __exp_shift = sizeof(_T) == 4 ? 23 : 20;
        constexpr int __exp_adjust = sizeof(_T) == 4 ? 0x7e : 0x3fe;
        constexpr int __exp_offset = sizeof(_T) == 4 ? 0x70 : 0x200;
        constexpr _T __subnorm_scale =
            __double_const<1, 0, __exp_offset>;  // double->float converts as intended
        constexpr _V __exponent_mask =
            _Limits::infinity();  // 0x7f800000 or 0x7ff0000000000000
        constexpr _V __p5_1_exponent =
            _T(sizeof(_T) == 4 ? __float_const<-1, 0x007fffffu, -1>
                             : __double_const<-1, 0x000fffffffffffffull, -1>);

        _V __mant = __p5_1_exponent & (__exponent_mask | __x);
        const _IV __exponent_bits = __extract_exponent_bits(__x);
        if (_GLIBCXX_SIMD_IS_LIKELY(all_of(isnormal(__x)))) {
            *__exp = simd_cast<__samesize<int, _V>>((__exponent_bits >> __exp_shift) -
                                                       __exp_adjust);
            return __mant;
        }
        const auto __iszero_inf_nan = isunordered(__x * _Limits::infinity(), __x * _V());
        const _V __scaled_subnormal = __x * __subnorm_scale;
        const _V __mant_subnormal = __p5_1_exponent & (__exponent_mask | __scaled_subnormal);
        where(!isnormal(__x), __mant) = __mant_subnormal;
        where(__iszero_inf_nan, __mant) = __x;
        _IV __e = __extract_exponent_bits(__scaled_subnormal);
        const _IM __value_isnormal = static_simd_cast<_IM>(isnormal(__x));
        where(__value_isnormal, __e) = __exponent_bits;
        const _IV __offset = (simd_reinterpret_cast<_IV>(__value_isnormal) & _IV(__exp_adjust)) |
                          (simd_reinterpret_cast<_IV>((__exponent_bits == 0) &
                                                     (static_simd_cast<_IM>(__x != 0))) &
                           _IV(__exp_adjust + __exp_offset));
        *__exp = simd_cast<__samesize<int, _V>>((__e >> __exp_shift) - __offset);
        return __mant;
    }
}
// }}}
_GLIBCXX_SIMD_MATH_CALL2_(ldexp, int)
_GLIBCXX_SIMD_MATH_CALL_(ilogb)

// logarithms {{{
_GLIBCXX_SIMD_MATH_CALL_(log)
_GLIBCXX_SIMD_MATH_CALL_(log10)
_GLIBCXX_SIMD_MATH_CALL_(log1p)
_GLIBCXX_SIMD_MATH_CALL_(log2)
//}}}
//logb{{{
template <class _T, class _Abi>
enable_if_t<std::is_floating_point<_T>::value, simd<_T, _Abi>> logb(
    const simd<_T, _Abi> &__x)
{
    constexpr size_t _N = simd_size_v<_T, _Abi>;
    if constexpr (_N == 1) {
        return std::logb(__x[0]);
    } else if constexpr (__is_fixed_size_abi_v<_Abi>) {
        return {__private_init,
                __simd_tuple_apply(
                    [](auto __impl, auto __xx) {
                        using _V = typename decltype(__impl)::simd_type;
                        return __data(std::experimental::logb(_V(__private_init, __xx)));
                    },
                    __data(__x))};
    } else if constexpr (__have_avx512vl && __is_sse_ps<_T, _N>()) {
        return {__private_init, _mm_fixupimm_ps(_mm_getexp_ps(__abs(__data(__x))),
                                              __data(__x), __auto_broadcast(0x00550433), 0x00)};
    } else if constexpr (__have_avx512vl && __is_sse_pd<_T, _N>()) {
        return {__private_init, _mm_fixupimm_pd(_mm_getexp_pd(__abs(__data(__x))),
                                              __data(__x), __auto_broadcast(0x00550433), 0x00)};
    } else if constexpr (__have_avx512vl && __is_avx_ps<_T, _N>()) {
        return {__private_init,
                _mm256_fixupimm_ps(_mm256_getexp_ps(__abs(__data(__x))), __data(__x),
                                   __auto_broadcast(0x00550433), 0x00)};
    } else if constexpr (__have_avx512vl && __is_avx_pd<_T, _N>()) {
        return {__private_init,
                _mm256_fixupimm_pd(_mm256_getexp_pd(__abs(__data(__x))), __data(__x),
                                   __auto_broadcast(0x00550433), 0x00)};
    } else if constexpr (__have_avx512f && __is_avx_ps<_T, _N>()) {
        const __m512 __v = __auto_bitcast(__data(__x));
        return {__private_init,
                __lo256(_mm512_fixupimm_ps(_mm512_getexp_ps(_mm512_abs_ps(__v)), __v,
                                         __auto_broadcast(0x00550433), 0x00))};
    } else if constexpr (__have_avx512f && __is_avx_pd<_T, _N>()) {
        return {__private_init, __lo256(_mm512_fixupimm_pd(
                                  _mm512_getexp_pd(_mm512_abs_pd(__auto_bitcast(__data(__x)))),
                                  __auto_bitcast(__data(__x)), __auto_broadcast(0x00550433), 0x00))};
    } else if constexpr (__is_avx512_ps<_T, _N>()) {
        return {__private_init, _mm512_fixupimm_ps(_mm512_getexp_ps(__abs(__data(__x))), __data(__x),
                                                 __auto_broadcast(0x00550433), 0x00)};
    } else if constexpr (__is_avx512_pd<_T, _N>()) {
        return {__private_init, _mm512_fixupimm_pd(_mm512_getexp_pd(__abs(__data(__x))), __data(__x),
                                                 __auto_broadcast(0x00550433), 0x00)};
    } else {
        using _V = simd<_T, _Abi>;
        using namespace std::experimental::__proposed;
        auto __is_normal = isnormal(__x);

        // work on __abs(__x) to reflect the return value on Linux for negative inputs
        // (domain-error => implementation-defined value is returned)
        const _V abs_x = abs(__x);

        // __exponent(__x) returns the exponent value (bias removed) as simd<_U> with
        // integral _U
        auto &&__exponent =
            [](const _V &__v) {
                using namespace std::experimental::__proposed;
                using _IV = rebind_simd_t<
                    std::conditional_t<sizeof(_T) == sizeof(__llong), __llong, int>, _V>;
                return (simd_reinterpret_cast<_IV>(__v) >>
                        (std::numeric_limits<_T>::digits - 1)) -
                       (std::numeric_limits<_T>::max_exponent - 1);
            };
        _V __r = static_simd_cast<_V>(__exponent(abs_x));
        if (_GLIBCXX_SIMD_IS_LIKELY(all_of(__is_normal))) {
            // without corner cases (nan, inf, subnormal, zero) we have our answer:
            return __r;
        }
        const auto __is_zero = __x == 0;
        const auto __is_nan = isnan(__x);
        const auto __is_inf = isinf(__x);
        where(__is_zero, __r) = -std::numeric_limits<_T>::infinity();
        where(__is_nan, __r) = __x;
        where(__is_inf, __r) = std::numeric_limits<_T>::infinity();
        __is_normal |= __is_zero || __is_nan || __is_inf;
        if (all_of(__is_normal)) {
            // at this point everything but subnormals is handled
            return __r;
        }
        // subnormals repeat the exponent extraction after multiplication of the input
        // with __a floating point value that has 0x70 in its exponent (not too big for
        // sp and large enough for dp)
        const _V __scaled =
            abs_x * _T(std::is_same<_T, float>::value ? __float_const<1, 0, 0x70>
                                                    : __double_const<1, 0, 0x70>);
        _V __scaled_exp = static_simd_cast<_V>(__exponent(__scaled) - 0x70);
        _GLIBCXX_SIMD_DEBUG(_Logarithm)(__x, __scaled)(__is_normal)(__r, __scaled_exp);
        where(__is_normal, __scaled_exp) = __r;
        return __scaled_exp;
    }
}
//}}}
_GLIBCXX_SIMD_MATH_CALL2_(modf, _T *)
_GLIBCXX_SIMD_MATH_CALL2_(scalbn, int)
_GLIBCXX_SIMD_MATH_CALL2_(scalbln, long)

_GLIBCXX_SIMD_MATH_CALL_(cbrt)

_GLIBCXX_SIMD_MATH_CALL_(abs)
_GLIBCXX_SIMD_MATH_CALL_(fabs)

// [parallel.simd.math] only asks for is_floating_point_v<_T> and forgot to allow
// signed integral _T
template <class _T, class _Abi>
enable_if_t<!std::is_floating_point_v<_T> && std::is_signed_v<_T>, simd<_T, _Abi>> abs(
    const simd<_T, _Abi> &__x)
{
    return {__private_init, _Abi::_Simd_impl_type::__abs(__data(__x))};
}
template <class _T, class _Abi>
enable_if_t<!std::is_floating_point_v<_T> && std::is_signed_v<_T>, simd<_T, _Abi>> fabs(
    const simd<_T, _Abi> &__x)
{
    return {__private_init, _Abi::_Simd_impl_type::__abs(__data(__x))};
}

// the following are overloads for functions in <cstdlib> and not covered by
// [parallel.simd.math]. I don't see much value in making them work, though
/*
template <class _Abi> simd<long, _Abi> labs(const simd<long, _Abi> &__x)
{
    return {__private_init, _Abi::_Simd_impl_type::abs(__data(__x))};
}
template <class _Abi> simd<long long, _Abi> llabs(const simd<long long, _Abi> &__x)
{
    return {__private_init, _Abi::_Simd_impl_type::abs(__data(__x))};
}
*/

template <class _T, class _Abi>
simd<_T, _Abi> hypot(const simd<_T, _Abi> &__x, const simd<_T, _Abi> &__y)
{
    using namespace __proposed::float_bitwise_operators;
    auto __hi = max(abs(__x), abs(__y));                                    // no error
    auto __lo = min(abs(__y), abs(__x));                                    // no error
    auto he = __hi & simd<_T, _Abi>(std::numeric_limits<_T>::infinity());  // no error
    where(he == 0, he) = std::numeric_limits<_T>::min();
    auto h1 = __hi / he;                                                     // no error
    auto l1 = __lo / he;                                                     // no error
    auto __r = he * sqrt(h1 * h1 + l1 * l1);
    where(l1 == 0, __r) = __hi;
    where(isinf(__x) || isinf(__y), __r) = std::numeric_limits<_T>::infinity();
    return __r;
}

template <class _T, class _Abi>
simd<_T, _Abi> hypot(simd<_T, _Abi> __x, simd<_T, _Abi> __y, simd<_T, _Abi> __z)
{
    using namespace __proposed::float_bitwise_operators;
    __x = abs(__x);                                                       // no error
    __y = abs(__y);                                                       // no error
    __z = abs(__z);                                                       // no error
    auto __hi = max(max(__x, __y), __z);                                      // no error
    auto l0 = min(__z, max(__x, __y));                                      // no error
    auto l1 = min(__y, __x);                                              // no error
    auto he = __hi & simd<_T, _Abi>(std::numeric_limits<_T>::infinity());  // no error
    where(he == 0, he) = std::numeric_limits<_T>::min();
    auto h1 = __hi / he;                                                // no error
    l0 *= 1 / he;                                                     // no error
    l1 *= 1 / he;                                                     // no error
    auto __lo = l0 * l0 + l1 * l1;  // add the two smaller values first
    auto __r = he * sqrt(__lo + h1 * h1);
    where(l0 + l1 == 0, __r) = __hi;
    where(isinf(__x) || isinf(__y) || isinf(__z), __r) = std::numeric_limits<_T>::infinity();
    return __r;
}

_GLIBCXX_SIMD_MATH_CALL2_(pow, _T)

_GLIBCXX_SIMD_MATH_CALL_(sqrt)
_GLIBCXX_SIMD_MATH_CALL_(erf)
_GLIBCXX_SIMD_MATH_CALL_(erfc)
_GLIBCXX_SIMD_MATH_CALL_(lgamma)
_GLIBCXX_SIMD_MATH_CALL_(tgamma)
_GLIBCXX_SIMD_MATH_CALL_(ceil)
_GLIBCXX_SIMD_MATH_CALL_(floor)
_GLIBCXX_SIMD_MATH_CALL_(nearbyint)
_GLIBCXX_SIMD_MATH_CALL_(rint)
_GLIBCXX_SIMD_MATH_CALL_(lrint)
_GLIBCXX_SIMD_MATH_CALL_(llrint)

_GLIBCXX_SIMD_MATH_CALL_(round)
_GLIBCXX_SIMD_MATH_CALL_(lround)
_GLIBCXX_SIMD_MATH_CALL_(llround)

_GLIBCXX_SIMD_MATH_CALL_(trunc)

_GLIBCXX_SIMD_MATH_CALL2_(fmod, _T)
_GLIBCXX_SIMD_MATH_CALL2_(remainder, _T)
_GLIBCXX_SIMD_MATH_CALL3_(remquo, _T, int *)
_GLIBCXX_SIMD_MATH_CALL2_(copysign, _T)

_GLIBCXX_SIMD_MATH_CALL2_(nextafter, _T)
// not covered in [parallel.simd.math]:
// _GLIBCXX_SIMD_MATH_CALL2_(nexttoward, long double)
_GLIBCXX_SIMD_MATH_CALL2_(fdim, _T)
_GLIBCXX_SIMD_MATH_CALL2_(fmax, _T)
_GLIBCXX_SIMD_MATH_CALL2_(fmin, _T)

_GLIBCXX_SIMD_MATH_CALL3_(fma, _T, _T)
_GLIBCXX_SIMD_MATH_CALL_(fpclassify)
_GLIBCXX_SIMD_MATH_CALL_(isfinite)
_GLIBCXX_SIMD_MATH_CALL_(isinf)
_GLIBCXX_SIMD_MATH_CALL_(isnan)
_GLIBCXX_SIMD_MATH_CALL_(isnormal)
_GLIBCXX_SIMD_MATH_CALL_(signbit)

_GLIBCXX_SIMD_MATH_CALL2_(isgreater, _T)
_GLIBCXX_SIMD_MATH_CALL2_(isgreaterequal, _T)
_GLIBCXX_SIMD_MATH_CALL2_(isless, _T)
_GLIBCXX_SIMD_MATH_CALL2_(islessequal, _T)
_GLIBCXX_SIMD_MATH_CALL2_(islessgreater, _T)
_GLIBCXX_SIMD_MATH_CALL2_(isunordered, _T)

/* not covered in [parallel.simd.math]
template <class _Abi> __doublev<_Abi> nan(const char* tagp);
template <class _Abi> __floatv<_Abi> nanf(const char* tagp);
template <class _Abi> __ldoublev<_Abi> nanl(const char* tagp);

template <class _V> struct simd_div_t {
    _V quot, rem;
};
template <class _Abi>
simd_div_t<__scharv<_Abi>> div(__scharv<_Abi> numer,
                                         __scharv<_Abi> denom);
template <class _Abi>
simd_div_t<__shortv<_Abi>> div(__shortv<_Abi> numer,
                                         __shortv<_Abi> denom);
template <class _Abi>
simd_div_t<__intv<_Abi>> div(__intv<_Abi> numer, __intv<_Abi> denom);
template <class _Abi>
simd_div_t<__longv<_Abi>> div(__longv<_Abi> numer,
                                        __longv<_Abi> denom);
template <class _Abi>
simd_div_t<__llongv<_Abi>> div(__llongv<_Abi> numer,
                                         __llongv<_Abi> denom);
*/

// special math {{{
template <class _T, class _Abi>
enable_if_t<std::is_floating_point_v<_T>, simd<_T, _Abi>> assoc_laguerre(
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<_T, _Abi>> &__n,
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<_T, _Abi>> &__m,
    const std::experimental::simd<_T, _Abi> &__x)
{
    return std::experimental::simd<_T, _Abi>([&](auto __i) { return std::assoc_laguerre(__n[__i], __m[__i], __x[__i]); });
}

template <class _T, class _Abi>
enable_if_t<std::is_floating_point_v<_T>, simd<_T, _Abi>> assoc_legendre(
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<_T, _Abi>> &__n,
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<_T, _Abi>> &__m,
    const std::experimental::simd<_T, _Abi> &__x)
{
    return std::experimental::simd<_T, _Abi>([&](auto __i) { return std::assoc_legendre(__n[__i], __m[__i], __x[__i]); });
}

_GLIBCXX_SIMD_MATH_CALL2_(beta, _T)
_GLIBCXX_SIMD_MATH_CALL_(comp_ellint_1)
_GLIBCXX_SIMD_MATH_CALL_(comp_ellint_2)
_GLIBCXX_SIMD_MATH_CALL2_(comp_ellint_3, _T)
_GLIBCXX_SIMD_MATH_CALL2_(cyl_bessel_i, _T)
_GLIBCXX_SIMD_MATH_CALL2_(cyl_bessel_j, _T)
_GLIBCXX_SIMD_MATH_CALL2_(cyl_bessel_k, _T)
_GLIBCXX_SIMD_MATH_CALL2_(cyl_neumann, _T)
_GLIBCXX_SIMD_MATH_CALL2_(ellint_1, _T)
_GLIBCXX_SIMD_MATH_CALL2_(ellint_2, _T)
_GLIBCXX_SIMD_MATH_CALL3_(ellint_3, _T, _T)
_GLIBCXX_SIMD_MATH_CALL_(expint)

template <class _T, class _Abi>
enable_if_t<std::is_floating_point_v<_T>, simd<_T, _Abi>> hermite(
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<_T, _Abi>> &__n,
    const std::experimental::simd<_T, _Abi> &__x)
{
    return std::experimental::simd<_T, _Abi>([&](auto __i) { return std::hermite(__n[__i], __x[__i]); });
}

template <class _T, class _Abi>
enable_if_t<std::is_floating_point_v<_T>, simd<_T, _Abi>> laguerre(
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<_T, _Abi>> &__n,
    const std::experimental::simd<_T, _Abi> &__x)
{
    return std::experimental::simd<_T, _Abi>([&](auto __i) { return std::laguerre(__n[__i], __x[__i]); });
}

template <class _T, class _Abi>
enable_if_t<std::is_floating_point_v<_T>, simd<_T, _Abi>> legendre(
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<_T, _Abi>> &__n,
    const std::experimental::simd<_T, _Abi> &__x)
{
    return std::experimental::simd<_T, _Abi>([&](auto __i) { return std::legendre(__n[__i], __x[__i]); });
}

_GLIBCXX_SIMD_MATH_CALL_(riemann_zeta)

template <class _T, class _Abi>
enable_if_t<std::is_floating_point_v<_T>, simd<_T, _Abi>> sph_bessel(
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<_T, _Abi>> &__n,
    const std::experimental::simd<_T, _Abi> &__x)
{
    return std::experimental::simd<_T, _Abi>([&](auto __i) { return std::sph_bessel(__n[__i], __x[__i]); });
}

template <class _T, class _Abi>
enable_if_t<std::is_floating_point_v<_T>, simd<_T, _Abi>> sph_legendre(
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<_T, _Abi>> &__l,
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<_T, _Abi>> &__m,
    const std::experimental::simd<_T, _Abi> &theta)
{
    return std::experimental::simd<_T, _Abi>([&](auto __i) { return std::assoc_legendre(__l[__i], __m[__i], theta[__i]); });
}

template <class _T, class _Abi>
enable_if_t<std::is_floating_point_v<_T>, simd<_T, _Abi>> sph_neumann(
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<_T, _Abi>> &__n,
    const std::experimental::simd<_T, _Abi> &__x)
{
    return std::experimental::simd<_T, _Abi>([&](auto __i) { return std::sph_neumann(__n[__i], __x[__i]); });
}
// }}}

#undef _GLIBCXX_SIMD_MATH_CALL_
#undef _GLIBCXX_SIMD_MATH_CALL2_
#undef _GLIBCXX_SIMD_MATH_CALL3_

_GLIBCXX_SIMD_END_NAMESPACE

#endif  // __cplusplus >= 201703L
#endif  // _GLIBCXX_EXPERIMENTAL_SIMD_MATH_H_
// vim: foldmethod=marker
