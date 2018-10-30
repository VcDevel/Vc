#ifndef _GLIBCXX_EXPERIMENTAL_SIMD_MATH_H_
#define _GLIBCXX_EXPERIMENTAL_SIMD_MATH_H_

#pragma GCC system_header

#if __cplusplus >= 201703L

#include "simd_abis.h"
#include <utility>
#include <iomanip>

static_assert(std::is_same_v<bool, decltype(std::isnan(double()))>);
static_assert(std::is_same_v<bool, decltype(std::isinf(double()))>);

_GLIBCXX_SIMD_BEGIN_NAMESPACE
namespace detail
{
// *v and samesize types {{{
template <class Abi> using scharv = simd<signed char, Abi>;    // exposition only
template <class Abi> using shortv = simd<short, Abi>;          // exposition only
template <class Abi> using intv = simd<int, Abi>;              // exposition only
template <class Abi> using longv = simd<long int, Abi>;        // exposition only
template <class Abi> using llongv = simd<long long int, Abi>;  // exposition only
template <class Abi> using floatv = simd<float, Abi>;          // exposition only
template <class Abi> using doublev = simd<double, Abi>;        // exposition only
template <class Abi> using ldoublev = simd<long double, Abi>;  // exposition only
template <class T, class V>
using samesize = fixed_size_simd<T, V::size()>;  // exposition only

// }}}
// math_return_type {{{
template <class DoubleR, class T, class Abi> struct math_return_type;
template <class DoubleR, class T, class Abi>
using math_return_type_t = typename math_return_type<DoubleR, T, Abi>::type;

template <class T, class Abi> struct math_return_type<double, T, Abi> {
    using type = std::experimental::simd<T, Abi>;
};
template <class T, class Abi> struct math_return_type<bool, T, Abi> {
    using type = std::experimental::simd_mask<T, Abi>;
};
template <class DoubleR, class T, class Abi> struct math_return_type {
    using type = std::experimental::fixed_size_simd<DoubleR, simd_size_v<T, Abi>>;
};
//}}}
// _GLIBCXX_SIMD_MATH_CALL_ {{{
#define _GLIBCXX_SIMD_MATH_CALL_(name_)                                                             \
    template <class T, class Abi, class...,                                              \
              class R = std::experimental::detail::math_return_type_t<                                  \
                  decltype(std::name_(std::declval<double>())), T, Abi>>                 \
    std::enable_if_t<std::is_floating_point_v<T>, R> name_(std::experimental::simd<T, Abi> x)           \
    {                                                                                    \
        using V = std::experimental::simd<T, Abi>;                                                      \
        return std::experimental::detail::impl_or_fallback(                                             \
            [](const auto &xx) -> decltype(                                              \
                                   R(std::experimental::detail::private_init,                           \
                                     std::experimental::detail::get_impl_t<decltype(xx)>::__##name_(    \
                                         std::experimental::detail::data(xx)))) {                       \
                return {std::experimental::detail::private_init,                                        \
                        std::experimental::detail::get_impl_t<decltype(xx)>::__##name_(                 \
                            std::experimental::detail::data(xx))};                                      \
            },                                                                           \
            [](const V &xx) {                                                            \
                if constexpr (std::experimental::is_simd_mask_v<R>) {                                   \
                    return R(std::experimental::detail::private_init,                                   \
                             [&](auto i) { return std::name_(xx[i]); });                 \
                } else {                                                                 \
                    return R([&](auto i) { return std::name_(xx[i]); });                 \
                }                                                                        \
            },                                                                           \
            x);                                                                          \
    }

// }}}
//extra_argument_type{{{
template <class U, class T, class Abi> struct extra_argument_type;

template <class T, class Abi> struct extra_argument_type<T *, T, Abi> {
    using type = std::experimental::simd<T, Abi> *;
    static constexpr double *declval();
    _GLIBCXX_SIMD_INTRINSIC static constexpr auto data(type x) { return &std::experimental::detail::data(*x); }
    static constexpr bool needs_temporary_scalar = true;
};
template <class U, class T, class Abi> struct extra_argument_type<U *, T, Abi> {
    static_assert(std::is_integral_v<U>);
    using type = std::experimental::fixed_size_simd<U, std::experimental::simd_size_v<T, Abi>> *;
    static constexpr U *declval();
    _GLIBCXX_SIMD_INTRINSIC static constexpr auto data(type x) { return &std::experimental::detail::data(*x); }
    static constexpr bool needs_temporary_scalar = true;
};
template <class T, class Abi> struct extra_argument_type<T, T, Abi> {
    using type = std::experimental::simd<T, Abi>;
    static constexpr double declval();
    _GLIBCXX_SIMD_INTRINSIC static constexpr decltype(auto) data(const type &x)
    {
        return std::experimental::detail::data(x);
    }
    static constexpr bool needs_temporary_scalar = false;
};
template <class U, class T, class Abi> struct extra_argument_type {
    static_assert(std::is_integral_v<U>);
    using type = std::experimental::fixed_size_simd<U, std::experimental::simd_size_v<T, Abi>>;
    static constexpr U declval();
    _GLIBCXX_SIMD_INTRINSIC static constexpr decltype(auto) data(const type &x)
    {
        return std::experimental::detail::data(x);
    }
    static constexpr bool needs_temporary_scalar = false;
};
//}}}
// _GLIBCXX_SIMD_MATH_CALL2_ {{{
#define _GLIBCXX_SIMD_MATH_CALL2_(name_, arg2_)                                                     \
    template <                                                                           \
        class T, class Abi, class...,                                                    \
        class Arg2 = std::experimental::detail::extra_argument_type<arg2_, T, Abi>,                     \
        class R = std::experimental::detail::math_return_type_t<                                        \
            decltype(std::name_(std::declval<double>(), Arg2::declval())), T, Abi>>      \
    std::enable_if_t<std::is_floating_point_v<T>, R> name_(                              \
        const std::experimental::simd<T, Abi> &x_, const typename Arg2::type &y_)                       \
    {                                                                                    \
        using V = std::experimental::simd<T, Abi>;                                                      \
        return std::experimental::detail::impl_or_fallback(                                             \
            [](const auto &x, const auto &y)                                             \
                -> decltype(R(std::experimental::detail::private_init,                                  \
                              std::experimental::detail::get_impl_t<decltype(x)>::__##name_(            \
                                  std::experimental::detail::data(x), Arg2::data(y)))) {                \
                return {std::experimental::detail::private_init,                                        \
                        std::experimental::detail::get_impl_t<decltype(x)>::__##name_(                  \
                            std::experimental::detail::data(x), Arg2::data(y))};                        \
            },                                                                           \
            [](const V &x, const auto &y) {                                              \
                auto &&gen = [&](auto i) {                                               \
                    if constexpr (Arg2::needs_temporary_scalar) {                        \
                        const auto &yy = *y;                                             \
                        auto tmp = yy[i];                                                \
                        auto ret = std::name_(x[i], &tmp);                               \
                        (*y)[i] = tmp;                                                   \
                        return ret;                                                      \
                    } else {                                                             \
                        return std::name_(x[i], y[i]);                                   \
                    }                                                                    \
                };                                                                       \
                if constexpr (std::experimental::is_simd_mask_v<R>) {                                   \
                    return R(std::experimental::detail::private_init, gen);                             \
                } else {                                                                 \
                    return R(gen);                                                       \
                }                                                                        \
            },                                                                           \
            x_, y_);                                                                     \
    }                                                                                    \
    template <class U, class T, class Abi>                                               \
    _GLIBCXX_SIMD_INTRINSIC std::experimental::detail::math_return_type_t<                                         \
        decltype(std::name_(                                                             \
            std::declval<double>(),                                                      \
            std::declval<std::enable_if_t<                                               \
                std::conjunction_v<                                                      \
                    std::is_same<arg2_, T>,                                              \
                    std::negation<std::is_same<std::decay_t<U>, std::experimental::simd<T, Abi>>>,      \
                    std::is_convertible<U, std::experimental::simd<T, Abi>>,                            \
                    std::is_floating_point<T>>,                                          \
                double>>())),                                                            \
        T, Abi>                                                                          \
    name_(U &&x_, const std::experimental::simd<T, Abi> &y_)                                            \
    {                                                                                    \
        return std::experimental::name_(std::experimental::simd<T, Abi>(std::forward<U>(x_)), y_);                     \
    }

// }}}
// _GLIBCXX_SIMD_MATH_CALL3_ {{{
#define _GLIBCXX_SIMD_MATH_CALL3_(name_, arg2_, arg3_)                                              \
    template <class T, class Abi, class...,                                              \
              class Arg2 = std::experimental::detail::extra_argument_type<arg2_, T, Abi>,               \
              class Arg3 = std::experimental::detail::extra_argument_type<arg3_, T, Abi>,               \
              class R = std::experimental::detail::math_return_type_t<                                  \
                  decltype(std::name_(std::declval<double>(), Arg2::declval(),           \
                                      Arg3::declval())),                                 \
                  T, Abi>>                                                               \
    std::enable_if_t<std::is_floating_point_v<T>, R> name_(                              \
        std::experimental::simd<T, Abi> x_, typename Arg2::type y_, typename Arg3::type z_)             \
    {                                                                                    \
        using V = std::experimental::simd<T, Abi>;                                                      \
        return std::experimental::detail::impl_or_fallback(                                             \
            [](const auto &x, const auto &y, const auto &z)                              \
                -> decltype(R(std::experimental::detail::private_init,                                  \
                              std::experimental::detail::get_impl_t<decltype(x)>::__##name_(            \
                                  std::experimental::detail::data(x), Arg2::data(y), Arg3::data(z)))) { \
                return {std::experimental::detail::private_init,                                        \
                        std::experimental::detail::get_impl_t<decltype(x)>::__##name_(                  \
                            std::experimental::detail::data(x), Arg2::data(y), Arg3::data(z))};         \
            },                                                                           \
            [](const V &x, const auto &y, const auto &z) {                               \
                return R([&](auto i) {                                                   \
                    if constexpr (Arg3::needs_temporary_scalar) {                        \
                        const auto &zz = *z;                                             \
                        auto tmp = zz[i];                                                \
                        auto ret = std::name_(x[i], y[i], &tmp);                         \
                        (*z)[i] = tmp;                                                   \
                        return ret;                                                      \
                    } else {                                                             \
                        return std::name_(x[i], y[i], z[i]);                             \
                    }                                                                    \
                });                                                                      \
            },                                                                           \
            x_, y_, z_);                                                                 \
    }                                                                                    \
    template <class T, class U, class V, class..., class TT = std::decay_t<T>,           \
              class UU = std::decay_t<U>, class VV = std::decay_t<V>,                    \
              class Simd = std::conditional_t<std::experimental::is_simd_v<UU>, UU, VV>>                \
    _GLIBCXX_SIMD_INTRINSIC decltype(std::experimental::name_(Simd(std::declval<T>()), Simd(std::declval<U>()),    \
                                    Simd(std::declval<V>())))                            \
    name_(T &&x_, U &&y_, V &&z_)                                                        \
    {                                                                                    \
        return std::experimental::name_(Simd(std::forward<T>(x_)), Simd(std::forward<U>(y_)),           \
                         Simd(std::forward<V>(z_)));                                     \
    }

// }}}
// cosSeries {{{
template < typename Abi>
_GLIBCXX_SIMD_ALWAYS_INLINE static floatv<Abi> cosSeries(const floatv<Abi> &x)
{
    using C = detail::trig<float>;
    const floatv<Abi> x2 = x * x;
    return ((C::cos_c2  * x2 +
             C::cos_c1) * x2 +
             C::cos_c0) * (x2 * x2)
        - .5f * x2 + 1.f;
}
template <typename Abi>
_GLIBCXX_SIMD_ALWAYS_INLINE static doublev<Abi> cosSeries(const doublev<Abi> &x)
{
    using C = detail::trig<double>;
    const doublev<Abi> x2 = x * x;
    return (((((C::cos_c5  * x2 +
                C::cos_c4) * x2 +
                C::cos_c3) * x2 +
                C::cos_c2) * x2 +
                C::cos_c1) * x2 +
                C::cos_c0) * (x2 * x2)
        - .5 * x2 + 1.;
}

// }}}
// sinSeries {{{
template <typename Abi>
_GLIBCXX_SIMD_ALWAYS_INLINE static floatv<Abi> sinSeries(const floatv<Abi>& x)
{
    using C = detail::trig<float>;
    const floatv<Abi> x2 = x * x;
    return ((C::sin_c2  * x2 +
             C::sin_c1) * x2 +
             C::sin_c0) * (x2 * x)
        + x;
}

template <typename Abi>
_GLIBCXX_SIMD_ALWAYS_INLINE static doublev<Abi> sinSeries(const doublev<Abi> &x)
{
    using C = detail::trig<double>;
    const doublev<Abi> x2 = x * x;
    return (((((C::sin_c5  * x2 +
                C::sin_c4) * x2 +
                C::sin_c3) * x2 +
                C::sin_c2) * x2 +
                C::sin_c1) * x2 +
                C::sin_c0) * (x2 * x)
        + x;
}

// }}}
// foldInput {{{
template <class Abi>
_GLIBCXX_SIMD_ALWAYS_INLINE std::pair<floatv<Abi>, rebind_simd_t<int, floatv<Abi>>> foldInput(
    floatv<Abi> x)
{
    using V = floatv<Abi>;
    using C = detail::trig<float>;
    using IV = rebind_simd_t<int, V>;

    x = abs(x);
#if defined(_GLIBCXX_SIMD_HAVE_FMA4) || defined(_GLIBCXX_SIMD_HAVE_FMA)
    rebind_simd_t<int, V> quadrant =
        static_simd_cast<IV>(x * C::_4_pi + 1.f);  // prefer the fma here
    quadrant &= ~1;
#else
    rebind_simd_t<int, V> quadrant = static_simd_cast<IV>(x * C::_4_pi);
    quadrant += quadrant & 1;
#endif
    const V y = static_simd_cast<V>(quadrant);
    quadrant &= 7;

    return {((x - y * C::pi_4_hi) - y * C::pi_4_rem1) - y * C::pi_4_rem2, quadrant};
}

template <typename Abi>
_GLIBCXX_SIMD_ALWAYS_INLINE static std::pair<doublev<Abi>, rebind_simd_t<int, doublev<Abi>>>
foldInput(doublev<Abi> x)
{
    using V = doublev<Abi>;
    using C = detail::trig<double>;
    using IV = rebind_simd_t<int, V>;

    x = abs(x);
    V y = trunc(x / C::pi_4);  // * C::4_pi would work, but is >twice as imprecise
    V z = y - trunc(y * C::_1_16) * C::_16;  // y modulo 16
    IV quadrant = static_simd_cast<IV>(z);
    const auto mask = (quadrant & 1) != 0;
    ++where(mask, quadrant);
    where(static_simd_cast<typename V::mask_type>(mask), y) += V(1);
    quadrant &= 7;

    // since y is an integer we don't need to split y into low and high parts until the
    // integer
    // requires more bits than there are zero bits at the end of _pi_4_hi (30 bits -> 1e9)
    return {((x - y * C::pi_4_hi) - y * C::pi_4_rem1) - y * C::pi_4_rem2, quadrant};
}

// }}}
// extract_exponent_bits {{{
template <class Abi>
rebind_simd_t<int, floatv<Abi>> extract_exponent_bits(const floatv<Abi> &v)
{
    using namespace std::experimental::__proposed;
    using namespace std::experimental::__proposed::float_bitwise_operators;
    constexpr floatv<Abi> exponent_mask =
        std::numeric_limits<float>::infinity();  // 0x7f800000
    return simd_reinterpret_cast<rebind_simd_t<int, floatv<Abi>>>(v & exponent_mask);
}

template <class Abi>
rebind_simd_t<int, doublev<Abi>> extract_exponent_bits(const doublev<Abi> &v)
{
    using namespace std::experimental::__proposed;
    using namespace std::experimental::__proposed::float_bitwise_operators;
    const doublev<Abi> exponent_mask =
        std::numeric_limits<double>::infinity();  // 0x7ff0000000000000
    constexpr auto N = simd_size_v<double, Abi> * 2;
    constexpr auto Max = simd_abi::max_fixed_size<int>;
    if constexpr (N > Max) {
        const auto tup = split<Max / 2, (N - Max) / 2>(v & exponent_mask);
        return concat(
            shuffle<strided<2, 1>>(
                simd_reinterpret_cast<simd<int, simd_abi::deduce_t<int, Max>>>(
                    std::get<0>(tup))),
            shuffle<strided<2, 1>>(
                simd_reinterpret_cast<simd<int, simd_abi::deduce_t<int, N - Max>>>(
                    std::get<1>(tup))));
    } else {
        return shuffle<strided<2, 1>>(
            simd_reinterpret_cast<simd<int, simd_abi::deduce_t<int, N>>>(v &
                                                                         exponent_mask));
    }
}

// }}}
// impl_or_fallback {{{
template <class ImplFun, class FallbackFun, class... Args>
_GLIBCXX_SIMD_INTRINSIC auto impl_or_fallback_dispatch(int, ImplFun &&impl_fun, FallbackFun &&,
                                            Args &&... args)
    -> decltype(impl_fun(std::forward<Args>(args)...))
{
    return impl_fun(std::forward<Args>(args)...);
}

template <class ImplFun, class FallbackFun, class... Args>
inline auto impl_or_fallback_dispatch(float, ImplFun &&, FallbackFun &&fallback_fun,
                                      Args &&... args)
    -> decltype(fallback_fun(std::forward<Args>(args)...))
{
    return fallback_fun(std::forward<Args>(args)...);
}

template <class... Args> _GLIBCXX_SIMD_INTRINSIC auto impl_or_fallback(Args &&... args)
{
    return impl_or_fallback_dispatch(int(), std::forward<Args>(args)...);
}  //}}}
}  // namespace detail

// trigonometric functions {{{
_GLIBCXX_SIMD_MATH_CALL_(acos)
_GLIBCXX_SIMD_MATH_CALL_(asin)
_GLIBCXX_SIMD_MATH_CALL_(atan)
_GLIBCXX_SIMD_MATH_CALL2_(atan2, T)

/*
 * algorithm for sine and cosine:
 *
 * The result can be calculated with sine or cosine depending on the π/4 section the input
 * is in.
 * sine   ≈ x + x³
 * cosine ≈ 1 - x²
 *
 * sine:
 * Map -x to x and invert the output
 * Extend precision of x - n * π/4 by calculating
 * ((x - n * p1) - n * p2) - n * p3 (p1 + p2 + p3 = π/4)
 *
 * Calculate Taylor series with tuned coefficients.
 * Fix sign.
 */
//cos{{{
template <class T, class Abi>
std::enable_if_t<std::is_floating_point<T>::value, simd<T, Abi>> cos(simd<T, Abi> x)
{
    using V = simd<T, Abi>;
    using M = typename V::mask_type;

    auto folded = foldInput(x);
    const V &z = folded.first;
    auto &quadrant = folded.second;
    M sign = static_simd_cast<M>(quadrant > 3);
    where(quadrant > 3, quadrant) -= 4;
    sign ^= static_simd_cast<M>(quadrant > 1);

    V y = cosSeries(z);
    where(static_simd_cast<M>(quadrant == 1 || quadrant == 2), y) = sinSeries(z);
    where(sign, y) = -y;
    _GLIBCXX_SIMD_DEBUG(cosine)
        (_GLIBCXX_SIMD_PRETTY_PRINT(x))
        (_GLIBCXX_SIMD_PRETTY_PRINT(sign))
        (_GLIBCXX_SIMD_PRETTY_PRINT(z))
        (_GLIBCXX_SIMD_PRETTY_PRINT(folded.second))
        (_GLIBCXX_SIMD_PRETTY_PRINT(quadrant))
        (_GLIBCXX_SIMD_PRETTY_PRINT(y));
    return y;
}

template <class T>
_GLIBCXX_SIMD_ALWAYS_INLINE
    std::enable_if_t<std::is_floating_point<T>::value, simd<T, simd_abi::scalar>>
    cos(simd<T, simd_abi::scalar> x)
{
    return std::cos(detail::data(x));
}
//}}}
//sin{{{
template <class T, class Abi>
std::enable_if_t<std::is_floating_point<T>::value, simd<T, Abi>> sin(simd<T, Abi> x)
{
    using V = simd<T, Abi>;
    using M = typename V::mask_type;

    auto folded = foldInput(x);
    const V &z = folded.first;
    auto &quadrant = folded.second;
    const M sign = (x < 0) ^ static_simd_cast<M>(quadrant > 3);
    where(quadrant > 3, quadrant) -= 4;

    V y = sinSeries(z);
    where(static_simd_cast<M>(quadrant == 1 || quadrant == 2), y) = cosSeries(z);
    where(sign, y) = -y;
    _GLIBCXX_SIMD_DEBUG(sine)
        (_GLIBCXX_SIMD_PRETTY_PRINT(x))
        (_GLIBCXX_SIMD_PRETTY_PRINT(sign))
        (_GLIBCXX_SIMD_PRETTY_PRINT(z))
        (_GLIBCXX_SIMD_PRETTY_PRINT(folded.second))
        (_GLIBCXX_SIMD_PRETTY_PRINT(quadrant))
        (_GLIBCXX_SIMD_PRETTY_PRINT(y));
    return y;
}

template <class T>
_GLIBCXX_SIMD_ALWAYS_INLINE
    std::enable_if_t<std::is_floating_point<T>::value, simd<T, simd_abi::scalar>>
    sin(simd<T, simd_abi::scalar> x)
{
    return std::sin(detail::data(x));
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
namespace detail
{
template <class T, size_t N> Storage<T, N> getexp(Storage<T, N> x)
{
    if constexpr (have_avx512vl && is_sse_ps<T, N>()) {
        return _mm_getexp_ps(x);
    } else if constexpr (have_avx512f && is_sse_ps<T, N>()) {
        return lo128(_mm512_getexp_ps(auto_cast(x)));
    } else if constexpr (have_avx512vl && is_sse_pd<T, N>()) {
        return _mm_getexp_pd(x);
    } else if constexpr (have_avx512f && is_sse_pd<T, N>()) {
        return lo128(_mm512_getexp_pd(auto_cast(x)));
    } else if constexpr (have_avx512vl && is_avx_ps<T, N>()) {
        return _mm256_getexp_ps(x);
    } else if constexpr (have_avx512f && is_avx_ps<T, N>()) {
        return lo256(_mm512_getexp_ps(auto_cast(x)));
    } else if constexpr (have_avx512vl && is_avx_pd<T, N>()) {
        return _mm256_getexp_pd(x);
    } else if constexpr (have_avx512f && is_avx_pd<T, N>()) {
        return lo256(_mm512_getexp_pd(auto_cast(x)));
    } else if constexpr (is_avx512_ps<T, N>()) {
        return _mm512_getexp_ps(x);
    } else if constexpr (is_avx512_pd<T, N>()) {
        return _mm512_getexp_pd(x);
    } else {
        assert_unreachable<T>();
    }
}

template <class T, size_t N> Storage<T, N> getmant(Storage<T, N> x)
{
    if constexpr (have_avx512vl && is_sse_ps<T, N>()) {
        return _mm_getmant_ps(x, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src);
    } else if constexpr (have_avx512f && is_sse_ps<T, N>()) {
        return lo128(
            _mm512_getmant_ps(auto_cast(x), _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src));
    } else if constexpr (have_avx512vl && is_sse_pd<T, N>()) {
        return _mm_getmant_pd(x, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src);
    } else if constexpr (have_avx512f && is_sse_pd<T, N>()) {
        return lo128(
            _mm512_getmant_pd(auto_cast(x), _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src));
    } else if constexpr (have_avx512vl && is_avx_ps<T, N>()) {
        return _mm256_getmant_ps(x, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src);
    } else if constexpr (have_avx512f && is_avx_ps<T, N>()) {
        return lo256(
            _mm512_getmant_ps(auto_cast(x), _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src));
    } else if constexpr (have_avx512vl && is_avx_pd<T, N>()) {
        return _mm256_getmant_pd(x, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src);
    } else if constexpr (have_avx512f && is_avx_pd<T, N>()) {
        return lo256(
            _mm512_getmant_pd(auto_cast(x), _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src));
    } else if constexpr (is_avx512_ps<T, N>()) {
        return _mm512_getmant_ps(x, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src);
    } else if constexpr (is_avx512_pd<T, N>()) {
        return _mm512_getmant_pd(x, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_src);
    } else {
        assert_unreachable<T>();
    }
}
}  // namespace detail

/**
 * splits \p v into exponent and mantissa, the sign is kept with the mantissa
 *
 * The return value will be in the range [0.5, 1.0[
 * The \p e value will be an integer defining the power-of-two exponent
 */
template <class T, class Abi>
std::enable_if_t<std::is_floating_point_v<T>, simd<T, Abi>> frexp(
    const simd<T, Abi> &x, detail::samesize<int, simd<T, Abi>> *exp)
{
    using namespace std::experimental::detail;
    if constexpr (simd_size_v<T, Abi> == 1) {
        int tmp;
        const auto r = std::frexp(x[0], &tmp);
        (*exp)[0] = tmp;
        return r;
    } else if constexpr (is_fixed_size_abi_v<Abi>) {
        return {private_init, get_impl_t<simd<T, Abi>>::__frexp(data(x), data(*exp))};
    } else if constexpr (have_avx512f) {
        using IV = detail::samesize<int, simd<T, Abi>>;
        constexpr size_t N = simd_size_v<T, Abi>;
        constexpr size_t NI = N < 4 ? 4 : N;
        const auto v = data(x);
        const auto isnonzero = get_impl_t<simd<T, Abi>>::isnonzerovalue_mask(v.d);
        const auto e =
            to_intrin(detail::x86::blend(isnonzero, builtin_type_t<int, NI>(),
                                         1 + convert<Storage<int, NI>>(getexp(v)).d));
        _GLIBCXX_SIMD_DEBUG(frexp)(
            std::hex, _GLIBCXX_SIMD_PRETTY_PRINT(int(isnonzero)), std::dec, _GLIBCXX_SIMD_PRETTY_PRINT(e),
            _GLIBCXX_SIMD_PRETTY_PRINT(getexp(v)),
            _GLIBCXX_SIMD_PRETTY_PRINT(to_intrin(1 + convert<Storage<int, NI>>(getexp(v)).d)));
        builtin_store<N * sizeof(int)>(e, exp, overaligned<alignof(IV)>);
        return {private_init, detail::x86::blend(isnonzero, v, getmant(v))};
    } else {
        // fallback implementation
        static_assert(sizeof(T) == 4 || sizeof(T) == 8);
        using V = simd<T, Abi>;
        using IV = rebind_simd_t<int, V>;
        using IM = typename IV::mask_type;
        using limits = std::numeric_limits<T>;
        using namespace std::experimental::__proposed;
        using namespace std::experimental::__proposed::float_bitwise_operators;

        constexpr int exp_shift = sizeof(T) == 4 ? 23 : 20;
        constexpr int exp_adjust = sizeof(T) == 4 ? 0x7e : 0x3fe;
        constexpr int exp_offset = sizeof(T) == 4 ? 0x70 : 0x200;
        constexpr T subnorm_scale =
            detail::double_const<1, 0, exp_offset>;  // double->float converts as intended
        constexpr V exponent_mask =
            limits::infinity();  // 0x7f800000 or 0x7ff0000000000000
        constexpr V p5_1_exponent =
            T(sizeof(T) == 4 ? detail::float_const<-1, 0x007fffffu, -1>
                             : detail::double_const<-1, 0x000fffffffffffffull, -1>);

        V mant = p5_1_exponent & (exponent_mask | x);
        const IV exponent_bits = extract_exponent_bits(x);
        if (_GLIBCXX_SIMD_IS_LIKELY(all_of(isnormal(x)))) {
            *exp = simd_cast<detail::samesize<int, V>>((exponent_bits >> exp_shift) -
                                                       exp_adjust);
            return mant;
        }
        const auto iszero_inf_nan = isunordered(x * limits::infinity(), x * V());
        const V scaled_subnormal = x * subnorm_scale;
        const V mant_subnormal = p5_1_exponent & (exponent_mask | scaled_subnormal);
        where(!isnormal(x), mant) = mant_subnormal;
        where(iszero_inf_nan, mant) = x;
        IV e = extract_exponent_bits(scaled_subnormal);
        const IM value_isnormal = static_simd_cast<IM>(isnormal(x));
        where(value_isnormal, e) = exponent_bits;
        const IV offset = (simd_reinterpret_cast<IV>(value_isnormal) & IV(exp_adjust)) |
                          (simd_reinterpret_cast<IV>((exponent_bits == 0) &
                                                     (static_simd_cast<IM>(x != 0))) &
                           IV(exp_adjust + exp_offset));
        *exp = simd_cast<detail::samesize<int, V>>((e >> exp_shift) - offset);
        return mant;
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
template <class T, class Abi>
std::enable_if_t<std::is_floating_point<T>::value, simd<T, Abi>> logb(
    const simd<T, Abi> &x)
{
    using namespace std::experimental::detail;
    constexpr size_t N = simd_size_v<T, Abi>;
    if constexpr (N == 1) {
        return std::logb(x[0]);
    } else if constexpr (is_fixed_size_abi_v<Abi>) {
        return {private_init,
                simd_tuple_apply(
                    [](auto impl, auto xx) {
                        using V = typename decltype(impl)::simd_type;
                        return detail::data(std::experimental::logb(V(detail::private_init, xx)));
                    },
                    data(x))};
    } else if constexpr (detail::have_avx512vl && detail::is_sse_ps<T, N>()) {
        return {private_init, _mm_fixupimm_ps(_mm_getexp_ps(detail::x86::abs(data(x))),
                                              data(x), auto_broadcast(0x00550433), 0x00)};
    } else if constexpr (detail::have_avx512vl && detail::is_sse_pd<T, N>()) {
        return {private_init, _mm_fixupimm_pd(_mm_getexp_pd(detail::x86::abs(data(x))),
                                              data(x), auto_broadcast(0x00550433), 0x00)};
    } else if constexpr (detail::have_avx512vl && detail::is_avx_ps<T, N>()) {
        return {private_init,
                _mm256_fixupimm_ps(_mm256_getexp_ps(detail::x86::abs(data(x))), data(x),
                                   auto_broadcast(0x00550433), 0x00)};
    } else if constexpr (detail::have_avx512vl && detail::is_avx_pd<T, N>()) {
        return {private_init,
                _mm256_fixupimm_pd(_mm256_getexp_pd(detail::x86::abs(data(x))), data(x),
                                   auto_broadcast(0x00550433), 0x00)};
    } else if constexpr (detail::have_avx512f && detail::is_avx_ps<T, N>()) {
        const __m512 v = auto_cast(data(x));
        return {private_init,
                lo256(_mm512_fixupimm_ps(_mm512_getexp_ps(_mm512_abs_ps(v)), v,
                                         auto_broadcast(0x00550433), 0x00))};
    } else if constexpr (detail::have_avx512f && detail::is_avx_pd<T, N>()) {
        return {private_init, lo256(_mm512_fixupimm_pd(
                                  _mm512_getexp_pd(_mm512_abs_pd(auto_cast(data(x)))),
                                  auto_cast(data(x)), auto_broadcast(0x00550433), 0x00))};
    } else if constexpr (detail::is_avx512_ps<T, N>()) {
        return {private_init, _mm512_fixupimm_ps(_mm512_getexp_ps(abs(data(x))), data(x),
                                                 auto_broadcast(0x00550433), 0x00)};
    } else if constexpr (detail::is_avx512_pd<T, N>()) {
        return {private_init, _mm512_fixupimm_pd(_mm512_getexp_pd(abs(data(x))), data(x),
                                                 auto_broadcast(0x00550433), 0x00)};
    } else {
        using V = simd<T, Abi>;
        using namespace std::experimental::__proposed;
        using namespace std::experimental::detail;
        auto is_normal = isnormal(x);

        // work on abs(x) to reflect the return value on Linux for negative inputs
        // (domain-error => implementation-defined value is returned)
        const V abs_x = abs(x);

        // exponent(x) returns the exponent value (bias removed) as simd<U> with
        // integral U
        auto &&exponent =
            [](const V &v) {
                using namespace std::experimental::__proposed;
                using IV = rebind_simd_t<
                    std::conditional_t<sizeof(T) == sizeof(llong), llong, int>, V>;
                return (simd_reinterpret_cast<IV>(v) >>
                        (std::numeric_limits<T>::digits - 1)) -
                       (std::numeric_limits<T>::max_exponent - 1);
            };
        V r = static_simd_cast<V>(exponent(abs_x));
        if (_GLIBCXX_SIMD_IS_LIKELY(all_of(is_normal))) {
            // without corner cases (nan, inf, subnormal, zero) we have our answer:
            return r;
        }
        const auto is_zero = x == 0;
        const auto is_nan = isnan(x);
        const auto is_inf = isinf(x);
        where(is_zero, r) = -std::numeric_limits<T>::infinity();
        where(is_nan, r) = x;
        where(is_inf, r) = std::numeric_limits<T>::infinity();
        is_normal |= is_zero || is_nan || is_inf;
        if (all_of(is_normal)) {
            // at this point everything but subnormals is handled
            return r;
        }
        // subnormals repeat the exponent extraction after multiplication of the input
        // with a floating point value that has 0x70 in its exponent (not too big for
        // sp and large enough for dp)
        const V scaled =
            abs_x * T(std::is_same<T, float>::value ? detail::float_const<1, 0, 0x70>
                                                    : detail::double_const<1, 0, 0x70>);
        V scaled_exp = static_simd_cast<V>(exponent(scaled) - 0x70);
        _GLIBCXX_SIMD_DEBUG(logarithm)(x, scaled)(is_normal)(r, scaled_exp);
        where(is_normal, scaled_exp) = r;
        return scaled_exp;
    }
}
//}}}
_GLIBCXX_SIMD_MATH_CALL2_(modf, T *)
_GLIBCXX_SIMD_MATH_CALL2_(scalbn, int)
_GLIBCXX_SIMD_MATH_CALL2_(scalbln, long)

_GLIBCXX_SIMD_MATH_CALL_(cbrt)

_GLIBCXX_SIMD_MATH_CALL_(abs)
_GLIBCXX_SIMD_MATH_CALL_(fabs)

// [parallel.simd.math] only asks for is_floating_point_v<T> and forgot to allow
// signed integral T
template <class T, class Abi>
std::enable_if_t<!std::is_floating_point_v<T> && std::is_signed_v<T>, simd<T, Abi>> abs(
    const simd<T, Abi> &x)
{
    return {detail::private_init, Abi::simd_impl_type::__abs(detail::data(x))};
}
template <class T, class Abi>
std::enable_if_t<!std::is_floating_point_v<T> && std::is_signed_v<T>, simd<T, Abi>> fabs(
    const simd<T, Abi> &x)
{
    return {detail::private_init, Abi::simd_impl_type::__abs(detail::data(x))};
}

// the following are overloads for functions in <cstdlib> and not covered by
// [parallel.simd.math]. I don't see much value in making them work, though
/*
template <class Abi> simd<long, Abi> labs(const simd<long, Abi> &x)
{
    return {detail::private_init, Abi::simd_impl_type::abs(detail::data(x))};
}
template <class Abi> simd<long long, Abi> llabs(const simd<long long, Abi> &x)
{
    return {detail::private_init, Abi::simd_impl_type::abs(detail::data(x))};
}
*/

template <class T, class Abi>
simd<T, Abi> hypot(const simd<T, Abi> &x, const simd<T, Abi> &y)
{
    using namespace __proposed::float_bitwise_operators;
    auto hi = max(abs(x), abs(y));                                    // no error
    auto lo = min(abs(y), abs(x));                                    // no error
    auto he = hi & simd<T, Abi>(std::numeric_limits<T>::infinity());  // no error
    where(he == 0, he) = std::numeric_limits<T>::min();
    auto h1 = hi / he;                                                     // no error
    auto l1 = lo / he;                                                     // no error
    auto r = he * sqrt(h1 * h1 + l1 * l1);
    where(l1 == 0, r) = hi;
    where(isinf(x) || isinf(y), r) = std::numeric_limits<T>::infinity();
    return r;
}

template <class T, class Abi>
simd<T, Abi> hypot(simd<T, Abi> x, simd<T, Abi> y, simd<T, Abi> z)
{
    using namespace __proposed::float_bitwise_operators;
    x = abs(x);                                                       // no error
    y = abs(y);                                                       // no error
    z = abs(z);                                                       // no error
    auto hi = max(max(x, y), z);                                      // no error
    auto l0 = min(z, max(x, y));                                      // no error
    auto l1 = min(y, x);                                              // no error
    auto he = hi & simd<T, Abi>(std::numeric_limits<T>::infinity());  // no error
    where(he == 0, he) = std::numeric_limits<T>::min();
    auto h1 = hi / he;                                                // no error
    l0 *= 1 / he;                                                     // no error
    l1 *= 1 / he;                                                     // no error
    auto lo = l0 * l0 + l1 * l1;  // add the two smaller values first
    auto r = he * sqrt(lo + h1 * h1);
    where(l0 + l1 == 0, r) = hi;
    where(isinf(x) || isinf(y) || isinf(z), r) = std::numeric_limits<T>::infinity();
    return r;
}

_GLIBCXX_SIMD_MATH_CALL2_(pow, T)

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

_GLIBCXX_SIMD_MATH_CALL2_(fmod, T)
_GLIBCXX_SIMD_MATH_CALL2_(remainder, T)
_GLIBCXX_SIMD_MATH_CALL3_(remquo, T, int *)
_GLIBCXX_SIMD_MATH_CALL2_(copysign, T)

_GLIBCXX_SIMD_MATH_CALL2_(nextafter, T)
// not covered in [parallel.simd.math]:
// _GLIBCXX_SIMD_MATH_CALL2_(nexttoward, long double)
_GLIBCXX_SIMD_MATH_CALL2_(fdim, T)
_GLIBCXX_SIMD_MATH_CALL2_(fmax, T)
_GLIBCXX_SIMD_MATH_CALL2_(fmin, T)

_GLIBCXX_SIMD_MATH_CALL3_(fma, T, T)
_GLIBCXX_SIMD_MATH_CALL_(fpclassify)
_GLIBCXX_SIMD_MATH_CALL_(isfinite)
_GLIBCXX_SIMD_MATH_CALL_(isinf)
_GLIBCXX_SIMD_MATH_CALL_(isnan)
_GLIBCXX_SIMD_MATH_CALL_(isnormal)
_GLIBCXX_SIMD_MATH_CALL_(signbit)

_GLIBCXX_SIMD_MATH_CALL2_(isgreater, T)
_GLIBCXX_SIMD_MATH_CALL2_(isgreaterequal, T)
_GLIBCXX_SIMD_MATH_CALL2_(isless, T)
_GLIBCXX_SIMD_MATH_CALL2_(islessequal, T)
_GLIBCXX_SIMD_MATH_CALL2_(islessgreater, T)
_GLIBCXX_SIMD_MATH_CALL2_(isunordered, T)

/* not covered in [parallel.simd.math]
template <class Abi> detail::doublev<Abi> nan(const char* tagp);
template <class Abi> detail::floatv<Abi> nanf(const char* tagp);
template <class Abi> detail::ldoublev<Abi> nanl(const char* tagp);

template <class V> struct simd_div_t {
    V quot, rem;
};
template <class Abi>
simd_div_t<detail::scharv<Abi>> div(detail::scharv<Abi> numer,
                                         detail::scharv<Abi> denom);
template <class Abi>
simd_div_t<detail::shortv<Abi>> div(detail::shortv<Abi> numer,
                                         detail::shortv<Abi> denom);
template <class Abi>
simd_div_t<detail::intv<Abi>> div(detail::intv<Abi> numer, detail::intv<Abi> denom);
template <class Abi>
simd_div_t<detail::longv<Abi>> div(detail::longv<Abi> numer,
                                        detail::longv<Abi> denom);
template <class Abi>
simd_div_t<detail::llongv<Abi>> div(detail::llongv<Abi> numer,
                                         detail::llongv<Abi> denom);
*/

// special math {{{
template <class T, class Abi>
std::enable_if_t<std::is_floating_point_v<T>, simd<T, Abi>> assoc_laguerre(
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<T, Abi>> &n,
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<T, Abi>> &m,
    const std::experimental::simd<T, Abi> &x)
{
    return std::experimental::simd<T, Abi>([&](auto i) { return std::assoc_laguerre(n[i], m[i], x[i]); });
}

template <class T, class Abi>
std::enable_if_t<std::is_floating_point_v<T>, simd<T, Abi>> assoc_legendre(
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<T, Abi>> &n,
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<T, Abi>> &m,
    const std::experimental::simd<T, Abi> &x)
{
    return std::experimental::simd<T, Abi>([&](auto i) { return std::assoc_legendre(n[i], m[i], x[i]); });
}

_GLIBCXX_SIMD_MATH_CALL2_(beta, T)
_GLIBCXX_SIMD_MATH_CALL_(comp_ellint_1)
_GLIBCXX_SIMD_MATH_CALL_(comp_ellint_2)
_GLIBCXX_SIMD_MATH_CALL2_(comp_ellint_3, T)
_GLIBCXX_SIMD_MATH_CALL2_(cyl_bessel_i, T)
_GLIBCXX_SIMD_MATH_CALL2_(cyl_bessel_j, T)
_GLIBCXX_SIMD_MATH_CALL2_(cyl_bessel_k, T)
_GLIBCXX_SIMD_MATH_CALL2_(cyl_neumann, T)
_GLIBCXX_SIMD_MATH_CALL2_(ellint_1, T)
_GLIBCXX_SIMD_MATH_CALL2_(ellint_2, T)
_GLIBCXX_SIMD_MATH_CALL3_(ellint_3, T, T)
_GLIBCXX_SIMD_MATH_CALL_(expint)

template <class T, class Abi>
std::enable_if_t<std::is_floating_point_v<T>, simd<T, Abi>> hermite(
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<T, Abi>> &n,
    const std::experimental::simd<T, Abi> &x)
{
    return std::experimental::simd<T, Abi>([&](auto i) { return std::hermite(n[i], x[i]); });
}

template <class T, class Abi>
std::enable_if_t<std::is_floating_point_v<T>, simd<T, Abi>> laguerre(
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<T, Abi>> &n,
    const std::experimental::simd<T, Abi> &x)
{
    return std::experimental::simd<T, Abi>([&](auto i) { return std::laguerre(n[i], x[i]); });
}

template <class T, class Abi>
std::enable_if_t<std::is_floating_point_v<T>, simd<T, Abi>> legendre(
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<T, Abi>> &n,
    const std::experimental::simd<T, Abi> &x)
{
    return std::experimental::simd<T, Abi>([&](auto i) { return std::legendre(n[i], x[i]); });
}

_GLIBCXX_SIMD_MATH_CALL_(riemann_zeta)

template <class T, class Abi>
std::enable_if_t<std::is_floating_point_v<T>, simd<T, Abi>> sph_bessel(
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<T, Abi>> &n,
    const std::experimental::simd<T, Abi> &x)
{
    return std::experimental::simd<T, Abi>([&](auto i) { return std::sph_bessel(n[i], x[i]); });
}

template <class T, class Abi>
std::enable_if_t<std::is_floating_point_v<T>, simd<T, Abi>> sph_legendre(
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<T, Abi>> &l,
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<T, Abi>> &m,
    const std::experimental::simd<T, Abi> &theta)
{
    return std::experimental::simd<T, Abi>([&](auto i) { return std::assoc_legendre(l[i], m[i], theta[i]); });
}

template <class T, class Abi>
std::enable_if_t<std::is_floating_point_v<T>, simd<T, Abi>> sph_neumann(
    const std::experimental::fixed_size_simd<unsigned, std::experimental::simd_size_v<T, Abi>> &n,
    const std::experimental::simd<T, Abi> &x)
{
    return std::experimental::simd<T, Abi>([&](auto i) { return std::sph_neumann(n[i], x[i]); });
}
// }}}

#undef _GLIBCXX_SIMD_MATH_CALL_
#undef _GLIBCXX_SIMD_MATH_CALL2_
#undef _GLIBCXX_SIMD_MATH_CALL3_

_GLIBCXX_SIMD_END_NAMESPACE

#endif  // __cplusplus >= 201703L
#endif  // _GLIBCXX_EXPERIMENTAL_SIMD_MATH_H_
