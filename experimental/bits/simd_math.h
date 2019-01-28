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
template <class _Tp, class _V> using __samesize = fixed_size_simd<_Tp, _V::size()>;
// __math_return_type {{{
template <class _DoubleR, class _Tp, class _Abi> struct __math_return_type;
template <class _DoubleR, class _Tp, class _Abi>
using __math_return_type_t = typename __math_return_type<_DoubleR, _Tp, _Abi>::type;

template <class _Tp, class _Abi> struct __math_return_type<double, _Tp, _Abi> {
    using type = std::experimental::simd<_Tp, _Abi>;
};
template <class _Tp, class _Abi> struct __math_return_type<bool, _Tp, _Abi> {
    using type = std::experimental::simd_mask<_Tp, _Abi>;
};
template <class _DoubleR, class _Tp, class _Abi> struct __math_return_type {
    using type = std::experimental::fixed_size_simd<_DoubleR, simd_size_v<_Tp, _Abi>>;
};
//}}}
// TODO: rely on __simd_math_fallback to get rid of the SFINAE magic here:
// _GLIBCXX_SIMD_MATH_CALL_ {{{
#define _GLIBCXX_SIMD_MATH_CALL_(__name)                                                 \
    template <class _Tp, class _Abi, class...,                                            \
              class _R = std::experimental::__math_return_type_t<                        \
                  decltype(std::__name(std::declval<double>())), _Tp, _Abi>>              \
    enable_if_t<std::is_floating_point_v<_Tp>, _R> __name(                                \
        std::experimental::simd<_Tp, _Abi> __x)                                           \
    {                                                                                    \
        using _V = std::experimental::simd<_Tp, _Abi>;                                    \
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
template <class _U, class _Tp, class _Abi> struct __extra_argument_type;

template <class _Tp, class _Abi> struct __extra_argument_type<_Tp *, _Tp, _Abi> {
    using type = std::experimental::simd<_Tp, _Abi> *;
    static constexpr double *declval();
    _GLIBCXX_SIMD_INTRINSIC static constexpr auto __data(type __x) { return &std::experimental::__data(*__x); }
    static constexpr bool __needs_temporary_scalar = true;
};
template <class _U, class _Tp, class _Abi> struct __extra_argument_type<_U *, _Tp, _Abi> {
    static_assert(std::is_integral_v<_U>);
    using type = std::experimental::fixed_size_simd<_U, std::experimental::simd_size_v<_Tp, _Abi>> *;
    static constexpr _U *declval();
    _GLIBCXX_SIMD_INTRINSIC static constexpr auto __data(type __x) { return &std::experimental::__data(*__x); }
    static constexpr bool __needs_temporary_scalar = true;
};
template <class _Tp, class _Abi> struct __extra_argument_type<_Tp, _Tp, _Abi> {
    using type = std::experimental::simd<_Tp, _Abi>;
    static constexpr double declval();
    _GLIBCXX_SIMD_INTRINSIC static constexpr decltype(auto) __data(const type &__x)
    {
        return std::experimental::__data(__x);
    }
    static constexpr bool __needs_temporary_scalar = false;
};
template <class _U, class _Tp, class _Abi> struct __extra_argument_type {
    static_assert(std::is_integral_v<_U>);
    using type = std::experimental::fixed_size_simd<_U, std::experimental::simd_size_v<_Tp, _Abi>>;
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
        class _Tp, class _Abi, class...,                                                    \
        class _Arg2 = std::experimental::__extra_argument_type<arg2_, _Tp, _Abi>,           \
        class _R    = std::experimental::__math_return_type_t<                             \
            decltype(std::__name(std::declval<double>(), _Arg2::declval())), _Tp, _Abi>> \
    enable_if_t<std::is_floating_point_v<_Tp>, _R> __name(                                  \
        const std::experimental::simd<_Tp, _Abi>& __xx, const typename _Arg2::type& __yy)   \
    {                                                                                      \
        using _V = std::experimental::simd<_Tp, _Abi>;                                      \
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
    template <class _U, class _Tp, class _Abi>                                              \
    _GLIBCXX_SIMD_INTRINSIC std::experimental::__math_return_type_t<                       \
        decltype(std::__name(                                                              \
            std::declval<double>(),                                                        \
            std::declval<enable_if_t<                                                      \
                std::conjunction_v<                                                        \
                    std::is_same<arg2_, _Tp>,                                               \
                    std::negation<std::is_same<std::decay_t<_U>,                           \
                                               std::experimental::simd<_Tp, _Abi>>>,        \
                    std::is_convertible<_U, std::experimental::simd<_Tp, _Abi>>,            \
                    std::is_floating_point<_Tp>>,                                           \
                double>>())),                                                              \
        _Tp, _Abi>                                                                          \
    __name(_U&& __xx, const std::experimental::simd<_Tp, _Abi>& __yy)                       \
    {                                                                                      \
        return std::experimental::__name(                                                  \
            std::experimental::simd<_Tp, _Abi>(std::forward<_U>(__xx)), __yy);              \
    }

// }}}
// _GLIBCXX_SIMD_MATH_CALL3_ {{{
#define _GLIBCXX_SIMD_MATH_CALL3_(__name, arg2_, arg3_)                                  \
    template <class _Tp, class _Abi, class...,                                            \
              class _Arg2 = std::experimental::__extra_argument_type<arg2_, _Tp, _Abi>,   \
              class _Arg3 = std::experimental::__extra_argument_type<arg3_, _Tp, _Abi>,   \
              class _R    = std::experimental::__math_return_type_t<                     \
                  decltype(std::__name(std::declval<double>(), _Arg2::declval(),      \
                                       _Arg3::declval())),                            \
                  _Tp, _Abi>>                                                          \
    enable_if_t<std::is_floating_point_v<_Tp>, _R> __name(                                \
        std::experimental::simd<_Tp, _Abi> __xx, typename _Arg2::type __yy,               \
        typename _Arg3::type __zz)                                                       \
    {                                                                                    \
        using _V = std::experimental::simd<_Tp, _Abi>;                                    \
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
    template <class _Tp, class _U, class _V, class..., class _TT = std::decay_t<_Tp>,      \
              class _UU = std::decay_t<_U>, class _VV = std::decay_t<_V>,                \
              class _Simd =                                                              \
                  std::conditional_t<std::experimental::is_simd_v<_UU>, _UU, _VV>>       \
    _GLIBCXX_SIMD_INTRINSIC decltype(                                                    \
        std::experimental::__name(_Simd(std::declval<_Tp>()), _Simd(std::declval<_U>()),  \
                                  _Simd(std::declval<_V>())))                            \
    __name(_Tp&& __xx, _U&& __yy, _V&& __zz)                                              \
    {                                                                                    \
        return std::experimental::__name(_Simd(std::forward<_Tp>(__xx)),                  \
                                         _Simd(std::forward<_U>(__yy)),                  \
                                         _Simd(std::forward<_V>(__zz)));                 \
    }

// }}}
// __cosSeries {{{
template <typename _Abi>
_GLIBCXX_SIMD_ALWAYS_INLINE static simd<float, _Abi>
  __cosSeries(const simd<float, _Abi>& __x)
{
  const simd<float, _Abi> __x2 = __x * __x;
  simd<float, _Abi>       __y;
  __y = 0x1.ap-16f;                  //  1/8!
  __y = __y * __x2 - 0x1.6c1p-10f;   // -1/6!
  __y = __y * __x2 + 0x1.555556p-5f; //  1/4!
  return __y * (__x2 * __x2) - .5f * __x2 + 1.f;
}
template <typename _Abi>
_GLIBCXX_SIMD_ALWAYS_INLINE static simd<double, _Abi>
  __cosSeries(const simd<double, _Abi>& __x)
{
  const simd<double, _Abi> __x2 = __x * __x;
  simd<double, _Abi>       __y;
  __y = 0x1.AC00000000000p-45;               //  1/16!
  __y = __y * __x2 - 0x1.9394000000000p-37;  // -1/14!
  __y = __y * __x2 + 0x1.1EED8C0000000p-29;  //  1/12!
  __y = __y * __x2 - 0x1.27E4FB7400000p-22;  // -1/10!
  __y = __y * __x2 + 0x1.A01A01A018000p-16;  //  1/8!
  __y = __y * __x2 - 0x1.6C16C16C16C00p-10;  // -1/6!
  __y = __y * __x2 + 0x1.5555555555554p-5;   //  1/4!
  return (__y * __x2 - .5f) * __x2 + 1.f;
}

// }}}
// __sinSeries {{{
template <typename _Abi>
_GLIBCXX_SIMD_ALWAYS_INLINE static simd<float, _Abi>
  __sinSeries(const simd<float, _Abi>& __x)
{
  const simd<float, _Abi> __x2 = __x * __x;
  simd<float, _Abi>       __y;
  __y = -0x1.9CC000p-13f;             // -1/7!
  __y = __y * __x2 + 0x1.111100p-7f;  //  1/5!
  __y = __y * __x2 - 0x1.555556p-3f;  // -1/3!
  return __y * (__x2 * __x) + __x;
}

template <typename _Abi>
_GLIBCXX_SIMD_ALWAYS_INLINE static simd<double, _Abi>
  __sinSeries(const simd<double, _Abi>& __x)
{
    // __x  = [0, 0.7854 = pi/4]
    // __x² = [0, 0.6169 = pi²/8]
    const simd<double, _Abi> __x2 = __x * __x;
    simd<double, _Abi> __y;
    __y = -0x1.ACF0000000000p-41;              // -1/15!
    __y = __y * __x2 + 0x1.6124400000000p-33;  //  1/13!
    __y = __y * __x2 - 0x1.AE64567000000p-26;  // -1/11!
    __y = __y * __x2 + 0x1.71DE3A5540000p-19;  //  1/9!
    __y = __y * __x2 - 0x1.A01A01A01A000p-13;  // -1/7!
    __y = __y * __x2 + 0x1.1111111111110p-7;   //  1/5!
    __y = __y * __x2 - 0x1.5555555555555p-3;   // -1/3!
    return __y * (__x2 * __x) + __x;
}

// }}}
// __bit_cast {{{
template <typename _To, typename _From>
_GLIBCXX_SIMD_INTRINSIC _To __bit_cast(const _From __x)
{
  static_assert(sizeof(_To) == sizeof(_From));
  _To __r;
  std::memcpy(&__r, &__x, sizeof(_To));
  return __r;
}

// }}}
// __zero_low_bits {{{
template <int _Bits, typename _Tp, typename _Abi>
_GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> __zero_low_bits(simd<_Tp, _Abi> __x)
{
  const simd<_Tp, _Abi> __bitmask =
    __bit_cast<_Tp>(~__int_for_sizeof_t<_Tp>() << _Bits);
  return {__private_init, __get_impl_t<simd<_Tp, _Abi>>::bit_and(
			    __data(__x), __data(__bitmask))};
}

// }}}
// __fold_input {{{

/**\internal
 * Fold \p x into [-¼π, ¼π] and remember the quadrant it came from:
 * quadrant 0: [-¼π,  ¼π]
 * quadrant 1: [ ¼π,  ¾π]
 * quadrant 2: [ ¾π, 1¼π]
 * quadrant 3: [1¼π, 1¾π]
 *
 * The algorithm determines `y` as the multiple `x - y * ¼π = [-¼π, ¼π]`. Using a bitmask,
 * `y` is reduced to `quadrant`. `y` can be calculated as
 * ```
 * y = trunc(x / ¼π);
 * y += fmod(y, 2);
 * ```
 * This can be simplified by moving the (implicit) division by 2 into the truncation
 * expression. The `+= fmod` effect can the be achieved by using rounding instead of
 * truncation:
 * `y = round(x / ½π) * 2`.
 * If precision allows, `2/π * x` is better (faster).
 */
template <class _Tp, class _Abi>
struct __folded
{
  simd<_Tp, _Abi> _M_x;
  rebind_simd_t<int, simd<_Tp, _Abi>> _M_quadrant;
};

namespace __math_float
{
inline constexpr float __pi_over_4 = 0x1.921FB6p-1f; // π/4
inline constexpr float __2_over_pi = 0x1.45F306p-1f; // 2/π
inline constexpr float __pi_2_5bits0 =
  0x1.921fc0p0f; // π/2, 5 0-bits (least significant)
inline constexpr float __pi_2_5bits0_rem =
  -0x1.5777a6p-21f; // π/2 - __pi_2_5bits0
}
namespace __math_double
{
inline constexpr double __pi_over_4 = 0x1.921fb54442d18p-1; // π/4
inline constexpr double __2_over_pi = 0x1.45F306DC9C883p-1; // 2/π
inline constexpr double __pi_2      = 0x1.921fb54442d18p0;  // π/2
}

template <class _Abi>
_GLIBCXX_SIMD_ALWAYS_INLINE __folded<float, _Abi>
			    __fold_input(const simd<float, _Abi>& __x)
{
  using _V  = simd<float, _Abi>;
  using _IV = rebind_simd_t<int, _V>;
  using namespace __math_float;
  __folded<float, _Abi> __r;
  __r._M_x = abs(__x);
  if (_GLIBCXX_SIMD_IS_UNLIKELY(all_of(__r._M_x < __pi_over_4)))
    {
      __r._M_quadrant = 0;
    }
  else if (_GLIBCXX_SIMD_IS_LIKELY(all_of(__r._M_x < 33 * __pi_over_4)))
    {
      const _V __y    = round(__r._M_x * __2_over_pi);
      __r._M_quadrant = static_simd_cast<_IV>(__y) & 3; // __y mod 4
      __r._M_x -= __y * __pi_2_5bits0;
      __r._M_x -= __y * __pi_2_5bits0_rem;
    }
  else
    {
      using __math_double::__2_over_pi;
      using __math_double::__pi_2;
      using _VD       = rebind_simd_t<double, _V>;
      _VD __xd        = static_simd_cast<_VD>(__r._M_x);
      _VD __y         = round(__xd * __2_over_pi);
      __r._M_quadrant = static_simd_cast<_IV>(__y) & 3; // = __y mod 4
      __r._M_x = static_simd_cast<_V>(__xd - __y * __pi_2);
    }
  return __r;
}

template <typename _Abi>
_GLIBCXX_SIMD_ALWAYS_INLINE __folded<double, _Abi>
			    __fold_input(const simd<double, _Abi>& __x)
{
    using _V = simd<double, _Abi>;
    using _IV = rebind_simd_t<int, _V>;
    using namespace __math_double;

    __folded<double, _Abi> __r;
    __r._M_x = abs(__x);
    if (_GLIBCXX_SIMD_IS_UNLIKELY(all_of(__r._M_x < __pi_over_4))) {
        __r._M_quadrant = 0;
        return __r;
    }
    const _V __y = round(__r._M_x / (2 * __pi_over_4));
    __r._M_quadrant = static_simd_cast<_IV>(__y) & 3;

    if (_GLIBCXX_SIMD_IS_LIKELY(all_of(__r._M_x < 1025 * __pi_over_4)))
      {
	// x - y * pi/2, y uses no more than 11 mantissa bits
        __r._M_x -= __y *  0x1.921FB54443000p0;
        __r._M_x -= __y * -0x1.73DCB3B39A000p-43;
        __r._M_x -= __y *  0x1.45C06E0E68948p-86;
      }
    else if (_GLIBCXX_SIMD_IS_LIKELY(all_of(__y <= 0x1.0p30)))
      {
	// x - y * pi/2, y uses no more than 29 mantissa bits
	__r._M_x -= __y * 0x1.921FB40000000p0;
	__r._M_x -= __y * 0x1.4442D00000000p-24;
	__r._M_x -= __y * 0x1.8469898CC5170p-48;
      }
    else
      {
	// x - y * pi/2, y may require all mantissa bits
	const _V __y_hi = __zero_low_bits<26>(__y);
	const _V __y_lo = __y - __y_hi;
        const auto __pi_2_1 = 0x1.921FB50000000p0;
        const auto __pi_2_2 = 0x1.110B460000000p-26;
        const auto __pi_2_3 = 0x1.1A62630000000p-54;
        const auto __pi_2_4 = 0x1.8A2E03707344Ap-81;
        __r._M_x = __r._M_x
            - __y_hi * __pi_2_1
            - max(__y_hi * __pi_2_2, __y_lo * __pi_2_1)
            - min(__y_hi * __pi_2_2, __y_lo * __pi_2_1)
            - max(__y_hi * __pi_2_3, __y_lo * __pi_2_2)
            - min(__y_hi * __pi_2_3, __y_lo * __pi_2_2)
            - max(__y    * __pi_2_4, __y_lo * __pi_2_3)
            - min(__y    * __pi_2_4, __y_lo * __pi_2_3);
      }
    return __r;
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
template <class ImplFun, class FallbackFun, class... _Args>
_GLIBCXX_SIMD_INTRINSIC auto __impl_or_fallback_dispatch(int, ImplFun&& __impl_fun,
                                                         FallbackFun&&, _Args&&... __args)
    -> decltype(__impl_fun(std::forward<_Args>(__args)...))
{
    return __impl_fun(std::forward<_Args>(__args)...);
}

template <class ImplFun, class FallbackFun, class... _Args>
inline auto __impl_or_fallback_dispatch(float, ImplFun&&, FallbackFun&& __fallback_fun,
                                        _Args&&... __args)
    -> decltype(__fallback_fun(std::forward<_Args>(__args)...))
{
    return __fallback_fun(std::forward<_Args>(__args)...);
}

template <class... _Args> _GLIBCXX_SIMD_INTRINSIC auto __impl_or_fallback(_Args&&... __args)
{
    return __impl_or_fallback_dispatch(int(), std::forward<_Args>(__args)...);
}  //}}}

// trigonometric functions {{{
_GLIBCXX_SIMD_MATH_CALL_(acos)
_GLIBCXX_SIMD_MATH_CALL_(asin)
_GLIBCXX_SIMD_MATH_CALL_(atan)
_GLIBCXX_SIMD_MATH_CALL2_(atan2, _Tp)

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
template <class _Tp, class _Abi>
enable_if_t<std::is_floating_point_v<_Tp>, simd<_Tp, _Abi>>
  cos(const simd<_Tp, _Abi>& __x)
{
  using _V = simd<_Tp, _Abi>;
  if constexpr (__is_abi<_Abi, simd_abi::scalar>() ||
		__is_fixed_size_abi_v<_Abi>)
    {
      return {__private_init, __get_impl_t<_V>::__cos(__data(__x))};
    }
  else
    {
      if constexpr (is_same_v<_Tp, float>)
	if (_GLIBCXX_SIMD_IS_UNLIKELY(any_of(abs(__x) >= 393382)))
	  return static_simd_cast<_V>(
	    cos(static_simd_cast<rebind_simd_t<double, _V>>(__x)));

      using _Impl    = __get_impl_t<_V>;
      const auto __f = __fold_input(__x);
      // quadrant | effect
      //        0 | cosSeries, +
      //        1 | sinSeries, -
      //        2 | cosSeries, -
      //        3 | sinSeries, +
      using namespace std::experimental::__proposed::float_bitwise_operators;
      const _V __sign_flip =
	_V(-0.f) & static_simd_cast<_V>((1 + __f._M_quadrant) << 30);

      const auto __need_cos = (__f._M_quadrant & 1) == 0;
      if (_GLIBCXX_SIMD_IS_UNLIKELY(all_of(__need_cos)))
	{
	  return __sign_flip ^ __cosSeries(__f._M_x);
	}
      else if (_GLIBCXX_SIMD_IS_UNLIKELY(none_of(__need_cos)))
	{
	  return __sign_flip ^ __sinSeries(__f._M_x);
	}
      else // some_of(__need_cos)
	{
	  _V __r                         = __sinSeries(__f._M_x);
	  where(__need_cos.__cvt(), __r) = __cosSeries(__f._M_x);
	  return __r ^ __sign_flip;
	}
    }
}

template <class _Tp>
_GLIBCXX_SIMD_ALWAYS_INLINE
    enable_if_t<std::is_floating_point<_Tp>::value, simd<_Tp, simd_abi::scalar>>
    cos(simd<_Tp, simd_abi::scalar> __x)
{
    return std::cos(__data(__x));
}
//}}}
//sin{{{
template <class _Tp, class _Abi>
enable_if_t<std::is_floating_point_v<_Tp>, simd<_Tp, _Abi>>
  sin(const simd<_Tp, _Abi>& __x)
{
  using _V = simd<_Tp, _Abi>;
  if constexpr (__is_abi<_Abi, simd_abi::scalar>() ||
		__is_fixed_size_abi_v<_Abi>)
    {
      return {__private_init, __get_impl_t<_V>::__sin(__data(__x))};
    }
  else
    {
      if constexpr (is_same_v<_Tp, float>)
	if (_GLIBCXX_SIMD_IS_UNLIKELY(any_of(abs(__x) >= 527449)))
	  return static_simd_cast<_V>(
	    sin(static_simd_cast<rebind_simd_t<double, _V>>(__x)));

      const auto __f = __fold_input(__x);
      // quadrant | effect
      //        0 | sinSeries
      //        1 | cosSeries
      //        2 | sinSeries, sign flip
      //        3 | cosSeries, sign flip
      using namespace std::experimental::__proposed::float_bitwise_operators;
      const auto __sign_flip =
	(__x ^ static_simd_cast<_V>(1 - __f._M_quadrant)) & _V(_Tp(-0.));

      const auto __need_sin = (__f._M_quadrant & 1) == 0;
      if (_GLIBCXX_SIMD_IS_UNLIKELY(all_of(__need_sin)))
	{
	  return __sign_flip ^ __sinSeries(__f._M_x);
	}
      else if (_GLIBCXX_SIMD_IS_UNLIKELY(none_of(__need_sin)))
	{
	  return __sign_flip ^ __cosSeries(__f._M_x);
	}
      else // some_of(__need_sin)
	{
	  _V __r                         = __cosSeries(__f._M_x);
	  where(__need_sin.__cvt(), __r) = __sinSeries(__f._M_x);
	  return __sign_flip ^ __r;
	}
    }
}

template <class _Tp>
_GLIBCXX_SIMD_ALWAYS_INLINE
    enable_if_t<std::is_floating_point<_Tp>::value, simd<_Tp, simd_abi::scalar>>
    sin(simd<_Tp, simd_abi::scalar> __x)
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
template <class _Tp, size_t _N> __storage<_Tp, _N> __getexp(__storage<_Tp, _N> __x)
{
    if constexpr (__have_avx512vl && __is_sse_ps<_Tp, _N>()) {
        return _mm_getexp_ps(__x);
    } else if constexpr (__have_avx512f && __is_sse_ps<_Tp, _N>()) {
        return __lo128(_mm512_getexp_ps(__auto_bitcast(__x)));
    } else if constexpr (__have_avx512vl && __is_sse_pd<_Tp, _N>()) {
        return _mm_getexp_pd(__x);
    } else if constexpr (__have_avx512f && __is_sse_pd<_Tp, _N>()) {
        return __lo128(_mm512_getexp_pd(__auto_bitcast(__x)));
    } else if constexpr (__have_avx512vl && __is_avx_ps<_Tp, _N>()) {
        return _mm256_getexp_ps(__x);
    } else if constexpr (__have_avx512f && __is_avx_ps<_Tp, _N>()) {
        return __lo256(_mm512_getexp_ps(__auto_bitcast(__x)));
    } else if constexpr (__have_avx512vl && __is_avx_pd<_Tp, _N>()) {
        return _mm256_getexp_pd(__x);
    } else if constexpr (__have_avx512f && __is_avx_pd<_Tp, _N>()) {
        return __lo256(_mm512_getexp_pd(__auto_bitcast(__x)));
    } else if constexpr (__is_avx512_ps<_Tp, _N>()) {
        return _mm512_getexp_ps(__x);
    } else if constexpr (__is_avx512_pd<_Tp, _N>()) {
        return _mm512_getexp_pd(__x);
    } else {
        __assert_unreachable<_Tp>();
    }
}

template <class _Tp, size_t _N> __storage<_Tp, _N> __getmant(__storage<_Tp, _N> __x)
{
    if constexpr (__have_avx512vl && __is_sse_ps<_Tp, _N>()) {
        return _mm_getmant_ps(__x, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src);
    } else if constexpr (__have_avx512f && __is_sse_ps<_Tp, _N>()) {
        return __lo128(
            _mm512_getmant_ps(__auto_bitcast(__x), _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src));
    } else if constexpr (__have_avx512vl && __is_sse_pd<_Tp, _N>()) {
        return _mm_getmant_pd(__x, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src);
    } else if constexpr (__have_avx512f && __is_sse_pd<_Tp, _N>()) {
        return __lo128(
            _mm512_getmant_pd(__auto_bitcast(__x), _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src));
    } else if constexpr (__have_avx512vl && __is_avx_ps<_Tp, _N>()) {
        return _mm256_getmant_ps(__x, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src);
    } else if constexpr (__have_avx512f && __is_avx_ps<_Tp, _N>()) {
        return __lo256(
            _mm512_getmant_ps(__auto_bitcast(__x), _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src));
    } else if constexpr (__have_avx512vl && __is_avx_pd<_Tp, _N>()) {
        return _mm256_getmant_pd(__x, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src);
    } else if constexpr (__have_avx512f && __is_avx_pd<_Tp, _N>()) {
        return __lo256(
            _mm512_getmant_pd(__auto_bitcast(__x), _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src));
    } else if constexpr (__is_avx512_ps<_Tp, _N>()) {
        return _mm512_getmant_ps(__x, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src);
    } else if constexpr (__is_avx512_pd<_Tp, _N>()) {
        return _mm512_getmant_pd(__x, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src);
    } else {
        __assert_unreachable<_Tp>();
    }
}

/**
 * splits \p __v into exponent and mantissa, the sign is kept with the mantissa
 *
 * The return value will be in the range [0.5, 1.0[
 * The \p __e value will be an integer defining the power-of-two exponent
 */
template <class _Tp, class _Abi>
enable_if_t<std::is_floating_point_v<_Tp>, simd<_Tp, _Abi>> frexp(
    const simd<_Tp, _Abi> &__x, __samesize<int, simd<_Tp, _Abi>> *__exp)
{
    if constexpr (simd_size_v<_Tp, _Abi> == 1) {
        int __tmp;
        const auto __r = std::frexp(__x[0], &__tmp);
        (*__exp)[0] = __tmp;
        return __r;
    } else if constexpr (__is_fixed_size_abi_v<_Abi>) {
        return {__private_init, __get_impl_t<simd<_Tp, _Abi>>::__frexp(__data(__x), __data(*__exp))};
    } else if constexpr (__have_avx512f) {
        using _IV = __samesize<int, simd<_Tp, _Abi>>;
        constexpr size_t _N = simd_size_v<_Tp, _Abi>;
        constexpr size_t NI = _N < 4 ? 4 : _N;
        const auto __v = __data(__x);
        const auto isnonzero = __get_impl_t<simd<_Tp, _Abi>>::isnonzerovalue_mask(__v._M_data);
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
        static_assert(sizeof(_Tp) == 4 || sizeof(_Tp) == 8);
        using _V = simd<_Tp, _Abi>;
        using _IV = rebind_simd_t<int, _V>;
        using _IM = typename _IV::mask_type;
        using _Limits = std::numeric_limits<_Tp>;
        using namespace std::experimental::__proposed;
        using namespace std::experimental::__proposed::float_bitwise_operators;

        constexpr int __exp_shift = sizeof(_Tp) == 4 ? 23 : 20;
        constexpr int __exp_adjust = sizeof(_Tp) == 4 ? 0x7e : 0x3fe;
        constexpr int __exp_offset = sizeof(_Tp) == 4 ? 0x70 : 0x200;
        constexpr _Tp __subnorm_scale =
            __double_const<1, 0, __exp_offset>;  // double->float converts as intended
        constexpr _V __exponent_mask =
            _Limits::infinity();  // 0x7f800000 or 0x7ff0000000000000
        constexpr _V __p5_1_exponent =
            _Tp(sizeof(_Tp) == 4 ? __float_const<-1, 0x007fffffu, -1>
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
template <class _Tp, class _Abi>
enable_if_t<std::is_floating_point<_Tp>::value, simd<_Tp, _Abi>> logb(
    const simd<_Tp, _Abi> &__x)
{
    constexpr size_t _N = simd_size_v<_Tp, _Abi>;
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
      }
    else if constexpr (__have_avx512vl && __is_sse_ps<_Tp, _N>())
      {
	return {__private_init, _mm_getexp_ps(__data(__x))};
      }
    else if constexpr (__have_avx512vl && __is_sse_pd<_Tp, _N>())
      {
	return {__private_init, _mm_getexp_pd(__data(__x))};
      }
    else if constexpr (__have_avx512vl && __is_avx_ps<_Tp, _N>())
      {
	return {__private_init, _mm256_getexp_ps(__data(__x))};
      }
    else if constexpr (__have_avx512vl && __is_avx_pd<_Tp, _N>())
      {
	return {__private_init, _mm256_getexp_pd(__data(__x))};
      }
    else if constexpr (__have_avx512f && __is_avx_ps<_Tp, _N>())
      {
	return {__private_init,
		__lo256(_mm512_getexp_ps(__auto_bitcast(__data(__x))))};
      }
    else if constexpr (__have_avx512f && __is_avx_pd<_Tp, _N>())
      {
	return {__private_init,
		__lo256(_mm512_getexp_pd(__auto_bitcast(__data(__x))))};
      }
    else if constexpr (__is_avx512_ps<_Tp, _N>())
      {
	return {__private_init, _mm512_getexp_ps(__data(__x))};
      }
    else if constexpr (__is_avx512_pd<_Tp, _N>())
      {
	return {__private_init, _mm512_getexp_pd(__data(__x))};
      }
    else
      {
	using _V = simd<_Tp, _Abi>;
	using namespace std::experimental::__proposed;
	auto __is_normal = isnormal(__x);

	// work on __abs(__x) to reflect the return value on Linux for negative
	// inputs (domain-error => implementation-defined value is returned)
	const _V abs_x = abs(__x);

	// __exponent(__x) returns the exponent value (bias removed) as simd<_U>
	// with integral _U
	auto&& __exponent = [](const _V& __v) {
	  using namespace std::experimental::__proposed;
	  using _IV = rebind_simd_t<
	    std::conditional_t<sizeof(_Tp) == sizeof(__llong), __llong, int>,
	    _V>;
	  return (simd_reinterpret_cast<_IV>(__v) >>
		  (std::numeric_limits<_Tp>::digits - 1)) -
		 (std::numeric_limits<_Tp>::max_exponent - 1);
	};
	_V __r = static_simd_cast<_V>(__exponent(abs_x));
	if (_GLIBCXX_SIMD_IS_LIKELY(all_of(__is_normal)))
	  {
	    // without corner cases (nan, inf, subnormal, zero) we have our
	    // answer:
	    return __r;
	  }
	const auto __is_zero  = __x == 0;
	const auto __is_nan   = isnan(__x);
	const auto __is_inf   = isinf(__x);
	where(__is_zero, __r) = -std::numeric_limits<_Tp>::infinity();
	where(__is_nan, __r)  = __x;
	where(__is_inf, __r)  = std::numeric_limits<_Tp>::infinity();
	__is_normal |= __is_zero || __is_nan || __is_inf;
	if (all_of(__is_normal))
	  {
	    // at this point everything but subnormals is handled
	    return __r;
	  }
	// subnormals repeat the exponent extraction after multiplication of the
	// input with __a floating point value that has 0x70 in its exponent
	// (not too big for sp and large enough for dp)
	const _V __scaled = abs_x * _Tp(std::is_same<_Tp, float>::value
					  ? __float_const<1, 0, 0x70>
					  : __double_const<1, 0, 0x70>);
	_V __scaled_exp   = static_simd_cast<_V>(__exponent(__scaled) - 0x70);
	_GLIBCXX_SIMD_DEBUG(_Logarithm)
	(__x, __scaled)(__is_normal)(__r, __scaled_exp);
	where(__is_normal, __scaled_exp) = __r;
	return __scaled_exp;
      }
}
//}}}
_GLIBCXX_SIMD_MATH_CALL2_(modf, _Tp *)
_GLIBCXX_SIMD_MATH_CALL2_(scalbn, int)
_GLIBCXX_SIMD_MATH_CALL2_(scalbln, long)

_GLIBCXX_SIMD_MATH_CALL_(cbrt)

_GLIBCXX_SIMD_MATH_CALL_(abs)
_GLIBCXX_SIMD_MATH_CALL_(fabs)

// [parallel.simd.math] only asks for is_floating_point_v<_Tp> and forgot to allow
// signed integral _Tp
template <class _Tp, class _Abi>
enable_if_t<!std::is_floating_point_v<_Tp> && std::is_signed_v<_Tp>, simd<_Tp, _Abi>> abs(
    const simd<_Tp, _Abi> &__x)
{
    return {__private_init, _Abi::_Simd_impl_type::__abs(__data(__x))};
}
template <class _Tp, class _Abi>
enable_if_t<!std::is_floating_point_v<_Tp> && std::is_signed_v<_Tp>, simd<_Tp, _Abi>> fabs(
    const simd<_Tp, _Abi> &__x)
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

#define _GLIBCXX_SIMD_CVTING2(_NAME)                                           \
  template <typename _Tp, typename _Abi>                                       \
  _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> _NAME(                               \
    const simd<_Tp, _Abi>& __x, const __id<simd<_Tp, _Abi>>& __y)              \
  {                                                                            \
    return _NAME(__x, __y);                                                    \
  }                                                                            \
  template <typename _Tp, typename _Abi>                                       \
  _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> _NAME(                               \
    const __id<simd<_Tp, _Abi>>& __x, const simd<_Tp, _Abi>& __y)              \
  {                                                                            \
    return _NAME(__x, __y);                                                    \
  }

#define _GLIBCXX_SIMD_CVTING3(_NAME)                                           \
  template <typename _Tp, typename _Abi>                                       \
  _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> _NAME(                               \
    const __id<simd<_Tp, _Abi>>& __x, const simd<_Tp, _Abi>& __y,              \
    const simd<_Tp, _Abi>& __z)                                                \
  {                                                                            \
    return _NAME(__x, __y, __z);                                               \
  }                                                                            \
  template <typename _Tp, typename _Abi>                                       \
  _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> _NAME(                               \
    const simd<_Tp, _Abi>& __x, const __id<simd<_Tp, _Abi>>& __y,              \
    const simd<_Tp, _Abi>& __z)                                                \
  {                                                                            \
    return _NAME(__x, __y, __z);                                               \
  }                                                                            \
  template <typename _Tp, typename _Abi>                                       \
  _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> _NAME(                               \
    const simd<_Tp, _Abi>& __x, const simd<_Tp, _Abi>& __y,                    \
    const __id<simd<_Tp, _Abi>>& __z)                                          \
  {                                                                            \
    return _NAME(__x, __y, __z);                                               \
  }                                                                            \
  template <typename _Tp, typename _Abi>                                       \
  _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> _NAME(                               \
    const simd<_Tp, _Abi>& __x, const __id<simd<_Tp, _Abi>>& __y,              \
    const __id<simd<_Tp, _Abi>>& __z)                                          \
  {                                                                            \
    return _NAME(__x, __y, __z);                                               \
  }                                                                            \
  template <typename _Tp, typename _Abi>                                       \
  _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> _NAME(                               \
    const __id<simd<_Tp, _Abi>>& __x, const simd<_Tp, _Abi>& __y,              \
    const __id<simd<_Tp, _Abi>>& __z)                                          \
  {                                                                            \
    return _NAME(__x, __y, __z);                                               \
  }                                                                            \
  template <typename _Tp, typename _Abi>                                       \
  _GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi> _NAME(                               \
    const __id<simd<_Tp, _Abi>>& __x, const __id<simd<_Tp, _Abi>>& __y,        \
    const simd<_Tp, _Abi>& __z)                                                \
  {                                                                            \
    return _NAME(__x, __y, __z);                                               \
  }

template <typename _R, typename _ToApply, typename... _Tps>
_GLIBCXX_SIMD_INTRINSIC _R __fixed_size_apply(_ToApply&& __apply,
					      const _Tps&... __args)
{
  return {__private_init, __simd_tuple_apply(
			    [](auto __impl, const auto&... __inner) {
			      using _V = typename decltype(__impl)::simd_type;
			      return __data(hypot(_V(__private_init, __inner)...));
			    },
			    __data(__args)...)};
}

template <typename _V> _V __hypot(_V __x, _V __y)
{
  using _Tp = typename _V::value_type;
  if constexpr (_V::size() == 1)
    {
      return std::hypot(_Tp(__x[0]), _Tp(__y[0]));
    }
  else if constexpr (__is_fixed_size_abi_v<typename _V::abi_type>)
    {
      return __fixed_size_apply<_V>(
	[](auto __a, auto __b) { return hypot(__a, __b); }, __x, __y);
    }
  else
    {
      // A simple solution for _Tp == float would be to cast to double and simply calculate
      // sqrt(x²+y²) as it can't over-/underflow anymore with dp. It still needs the Annex F fixups
      // though and isn't faster on Skylake-AVX512 (not even for SSE and AVX vectors, and really bad
      // for AVX-512).
      using namespace __proposed::float_bitwise_operators;
      using _Limits = std::numeric_limits<_Tp>;
      _V __absx     = abs(__x);            // no error
      _V __absy     = abs(__y);            // no error
      _V __hi       = max(__absx, __absy); // no error
      _V __lo       = min(__absy, __absx); // no error

      // round __hi down to the next power-of-2:
      constexpr _V __inf(_Limits::infinity());

#if 1
      // if __hi is subnormal, avoid scaling by inf & final mul by 0 (which
      // yields NaN) by using min()
      _V __scale = _V(1 / _Limits::min());
      // invert exponent w/o error and w/o using the slow divider unit:
      // xor inverts the exponent but off by 1. 2*__hi adjusts for the
      // discrepancy. Note that overflow into infinity produces the correct
      // result. The potential trap is non-conforming, though. The trap can be
      // avoided by multiplying with .5 after xor. The multiplication increases
      // the latency of the function by 2 cycles, though.
      // slower but avoids trap: ((__hi & __inf) ^ __inf) * .5f;
      where(__hi > _Limits::min(), __scale) = ((__hi + __hi) & __inf) ^ __inf;
      // adjust final exponent for subnormal inputs
      _V __hi_exp                            = _Limits::min();
      where(__hi > _Limits::min(), __hi_exp) = __hi & __inf; // no error
#else
      _V           __hi_exp  = __hi & __inf;          // no error
      where(__hi < _Limits::min(), __hi_exp) = std::numeric_limits<_Tp>::min();
      _V __scale = 1 / __hi_exp;   // no error
#endif
      constexpr _V __mant_mask = _Limits::min() - _Limits::lowest();
      _V           __h1                     = (__hi & __mant_mask) | _V(1);
      //_V __h1 = __hi * __scale; // no error
      _V __l1 = __lo * __scale; // no error

      // sqrt(x²+y²) = e*sqrt((x/e)²+(y/e)²):
      // this ensures no overflow in the argument to sqrt
      _V __r = __hi_exp * sqrt(__h1 * __h1 + __l1 * __l1);

#ifdef __STDC_IEC_559__
      // fixup for Annex F requirements
#if 1
      _V __fixup                                     = __hi; // __lo == 0
      where(isunordered(__x, __y), __fixup)          = _Limits::quiet_NaN();
      where(isinf(__absx) || isinf(__absy), __fixup) = __inf;
      where(
	!(__lo == 0 || isunordered(__x, __y) || isinf(__absx) || isinf(__absy)),
	__fixup) = __r;
      __r = __fixup;
#else
      where(__l1 == 0, __r) = __hi;
      where(isunordered(__x, __y), __r)          = _Limits::quiet_NaN();
      where(isinf(__absx) || isinf(__absy), __r) = __inf;
#endif
#endif
      return __r;
    }
}

template <typename _Tp, typename _Abi>
_GLIBCXX_SIMD_INTRINSIC simd<_Tp, _Abi>
			hypot(const simd<_Tp, _Abi>& __x, const simd<_Tp, _Abi>& __y)
{
  return __hypot<conditional_t<__is_fixed_size_abi_v<_Abi>,
			       const simd<_Tp, _Abi>&, simd<_Tp, _Abi>>>(__x,
									 __y);
}
_GLIBCXX_SIMD_CVTING2(hypot)

template <typename _Tp, typename _Abi>
simd<_Tp, _Abi> hypot(const simd<_Tp, _Abi>& __x,
		      const simd<_Tp, _Abi>& __y,
		      const simd<_Tp, _Abi>& __z)
{
  using _V  = simd<_Tp, _Abi>;
  /* FIXME: enable after PR77776 is resolved
  if constexpr (_V::size() == 1)
    {
      return std::hypot(_Tp(__x[0]), _Tp(__y[0]), _Tp(__z[0]));
    }
  else
  */
  if constexpr (__is_fixed_size_abi_v<_Abi> && _V::size() > 1)
    {
      return __fixed_size_apply<simd<_Tp, _Abi>>(
	[](auto __a, auto __b, auto __c) { return hypot(__a, __b, __c); }, __x,
	__y, __z);
    }
  else
    {
      using namespace __proposed::float_bitwise_operators;
      using _Limits = std::numeric_limits<_Tp>;
      _V __absx     = abs(__x);                         // no error
      _V __absy     = abs(__y);                         // no error
      _V __absz     = abs(__z);                         // no error
      _V __hi       = max(max(__absx, __absy), __absz); // no error
      _V __l0       = min(__absz, max(__absx, __absy)); // no error
      _V __l1       = min(__absy, __absx);              // no error

      // round __hi down to the next power-of-2:
      constexpr _V __inf(_Limits::infinity());
      _V           __hi_exp = __hi & __inf; // no error

      // if __hi is subnormal, avoid scaling by inf & final mul by 0 (which
      // yields NaN) by using min()
      _V __scale = _V(1 / _Limits::min());
      // invert exponent w/o error and w/o using the slow divider unit
      where(__hi >= _Limits::min(), __scale) = .5f * (__hi_exp ^ __inf);
      // adjust final exponent for subnormal inputs
      where(!(__hi >= _Limits::min()), __hi_exp) = _Limits::min();

      // scale __hi to 0x1.???p0 and __l[01] by the same factor
      _V __h1 = __hi * __scale;            // no error
      __l0 *= __scale;                     // no error
      __l1 *= __scale;                     // no error
      _V __lo = __l0 * __l0 + __l1 * __l1; // add the two smaller values first
      _V __r  = __hi_exp * sqrt(__lo + __h1 * __h1);
#ifdef __STDC_IEC_559__
      // fixup for Annex F requirements
      _V __fixup                                  = __hi; // __lo == 0
      //where(__lo == 0, __fixup)                   = __hi;
      where(isunordered(__x, __y + __z), __fixup) = _Limits::quiet_NaN();
      where(isinf(__absx) || isinf(__absy) || isinf(__absz), __fixup) = __inf;
      where(!(__lo == 0 || isunordered(__x, __y + __z) || isinf(__absx) ||
	      isinf(__absy) || isinf(__absz)),
	    __fixup)                                                  = __r;
      __r = __fixup;
#endif
      return __r;
    }
}
_GLIBCXX_SIMD_CVTING3(hypot)

_GLIBCXX_SIMD_MATH_CALL2_(pow, _Tp)

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

_GLIBCXX_SIMD_MATH_CALL2_(fmod, _Tp)
_GLIBCXX_SIMD_MATH_CALL2_(remainder, _Tp)
_GLIBCXX_SIMD_MATH_CALL3_(remquo, _Tp, int *)
_GLIBCXX_SIMD_MATH_CALL2_(copysign, _Tp)

_GLIBCXX_SIMD_MATH_CALL2_(nextafter, _Tp)
// not covered in [parallel.simd.math]:
// _GLIBCXX_SIMD_MATH_CALL2_(nexttoward, long double)
_GLIBCXX_SIMD_MATH_CALL2_(fdim, _Tp)
_GLIBCXX_SIMD_MATH_CALL2_(fmax, _Tp)
_GLIBCXX_SIMD_MATH_CALL2_(fmin, _Tp)

_GLIBCXX_SIMD_MATH_CALL3_(fma, _Tp, _Tp)
_GLIBCXX_SIMD_MATH_CALL_(fpclassify)
_GLIBCXX_SIMD_MATH_CALL_(isfinite)
_GLIBCXX_SIMD_MATH_CALL_(isinf)
_GLIBCXX_SIMD_MATH_CALL_(isnan)
_GLIBCXX_SIMD_MATH_CALL_(isnormal)
_GLIBCXX_SIMD_MATH_CALL_(signbit)

_GLIBCXX_SIMD_MATH_CALL2_(isgreater, _Tp)
_GLIBCXX_SIMD_MATH_CALL2_(isgreaterequal, _Tp)
_GLIBCXX_SIMD_MATH_CALL2_(isless, _Tp)
_GLIBCXX_SIMD_MATH_CALL2_(islessequal, _Tp)
_GLIBCXX_SIMD_MATH_CALL2_(islessgreater, _Tp)
_GLIBCXX_SIMD_MATH_CALL2_(isunordered, _Tp)

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
template <class _Tp, class _Abi>
enable_if_t<std::is_floating_point_v<_Tp>, simd<_Tp, _Abi>> assoc_laguerre(
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<_Tp, _Abi>> &__n,
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<_Tp, _Abi>> &__m,
    const std::experimental::simd<_Tp, _Abi> &__x)
{
    return std::experimental::simd<_Tp, _Abi>([&](auto __i) { return std::assoc_laguerre(__n[__i], __m[__i], __x[__i]); });
}

template <class _Tp, class _Abi>
enable_if_t<std::is_floating_point_v<_Tp>, simd<_Tp, _Abi>> assoc_legendre(
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<_Tp, _Abi>> &__n,
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<_Tp, _Abi>> &__m,
    const std::experimental::simd<_Tp, _Abi> &__x)
{
    return std::experimental::simd<_Tp, _Abi>([&](auto __i) { return std::assoc_legendre(__n[__i], __m[__i], __x[__i]); });
}

_GLIBCXX_SIMD_MATH_CALL2_(beta, _Tp)
_GLIBCXX_SIMD_MATH_CALL_(comp_ellint_1)
_GLIBCXX_SIMD_MATH_CALL_(comp_ellint_2)
_GLIBCXX_SIMD_MATH_CALL2_(comp_ellint_3, _Tp)
_GLIBCXX_SIMD_MATH_CALL2_(cyl_bessel_i, _Tp)
_GLIBCXX_SIMD_MATH_CALL2_(cyl_bessel_j, _Tp)
_GLIBCXX_SIMD_MATH_CALL2_(cyl_bessel_k, _Tp)
_GLIBCXX_SIMD_MATH_CALL2_(cyl_neumann, _Tp)
_GLIBCXX_SIMD_MATH_CALL2_(ellint_1, _Tp)
_GLIBCXX_SIMD_MATH_CALL2_(ellint_2, _Tp)
_GLIBCXX_SIMD_MATH_CALL3_(ellint_3, _Tp, _Tp)
_GLIBCXX_SIMD_MATH_CALL_(expint)

template <class _Tp, class _Abi>
enable_if_t<std::is_floating_point_v<_Tp>, simd<_Tp, _Abi>> hermite(
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<_Tp, _Abi>> &__n,
    const std::experimental::simd<_Tp, _Abi> &__x)
{
    return std::experimental::simd<_Tp, _Abi>([&](auto __i) { return std::hermite(__n[__i], __x[__i]); });
}

template <class _Tp, class _Abi>
enable_if_t<std::is_floating_point_v<_Tp>, simd<_Tp, _Abi>> laguerre(
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<_Tp, _Abi>> &__n,
    const std::experimental::simd<_Tp, _Abi> &__x)
{
    return std::experimental::simd<_Tp, _Abi>([&](auto __i) { return std::laguerre(__n[__i], __x[__i]); });
}

template <class _Tp, class _Abi>
enable_if_t<std::is_floating_point_v<_Tp>, simd<_Tp, _Abi>> legendre(
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<_Tp, _Abi>> &__n,
    const std::experimental::simd<_Tp, _Abi> &__x)
{
    return std::experimental::simd<_Tp, _Abi>([&](auto __i) { return std::legendre(__n[__i], __x[__i]); });
}

_GLIBCXX_SIMD_MATH_CALL_(riemann_zeta)

template <class _Tp, class _Abi>
enable_if_t<std::is_floating_point_v<_Tp>, simd<_Tp, _Abi>> sph_bessel(
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<_Tp, _Abi>> &__n,
    const std::experimental::simd<_Tp, _Abi> &__x)
{
    return std::experimental::simd<_Tp, _Abi>([&](auto __i) { return std::sph_bessel(__n[__i], __x[__i]); });
}

template <class _Tp, class _Abi>
enable_if_t<std::is_floating_point_v<_Tp>, simd<_Tp, _Abi>> sph_legendre(
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<_Tp, _Abi>> &__l,
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<_Tp, _Abi>> &__m,
    const std::experimental::simd<_Tp, _Abi> &theta)
{
    return std::experimental::simd<_Tp, _Abi>([&](auto __i) { return std::assoc_legendre(__l[__i], __m[__i], theta[__i]); });
}

template <class _Tp, class _Abi>
enable_if_t<std::is_floating_point_v<_Tp>, simd<_Tp, _Abi>> sph_neumann(
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<_Tp, _Abi>> &__n,
    const std::experimental::simd<_Tp, _Abi> &__x)
{
    return std::experimental::simd<_Tp, _Abi>([&](auto __i) { return std::sph_neumann(__n[__i], __x[__i]); });
}
// }}}

#undef _GLIBCXX_SIMD_MATH_CALL_
#undef _GLIBCXX_SIMD_MATH_CALL2_
#undef _GLIBCXX_SIMD_MATH_CALL3_

_GLIBCXX_SIMD_END_NAMESPACE

#endif  // __cplusplus >= 201703L
#endif  // _GLIBCXX_EXPERIMENTAL_SIMD_MATH_H_
// vim: foldmethod=marker sw=2 ts=8 noet sts=2
