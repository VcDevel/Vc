/*  This file is part of the Vc library. {{{
Copyright © 2017 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_DETAIL_MATH_H_
#define VC_DETAIL_MATH_H_

#include "simd.h"
#include "const.h"
#include "debug.h"
#include <utility>

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
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

#define Vc_MATH_CALL_(name_)                                                             \
    template <class Abi> Vc::detail::floatv<Abi> name_(Vc::detail::floatv<Abi> x)        \
    {                                                                                    \
        return Vc::detail::impl_or_fallback(                                             \
            [](const auto &xx) -> decltype(Vc::detail::floatv<Abi>(                      \
                Vc::detail::private_init,                                                \
                Vc::detail::get_impl_t<decltype(xx)>::name_(Vc::detail::data(xx)))) {    \
                return {                                                                 \
                    Vc::detail::private_init,                                            \
                    Vc::detail::get_impl_t<decltype(xx)>::name_(Vc::detail::data(xx))};  \
            },                                                                           \
            [](const Vc::detail::floatv<Abi> &xx) {                                      \
                return Vc::detail::floatv<Abi>(                                          \
                    [&](auto i) { return std::name_(xx[i]); });                          \
            },                                                                           \
            x);                                                                          \
    }                                                                                    \
    template <class Abi> Vc::detail::doublev<Abi> name_(Vc::detail::doublev<Abi> x)      \
    {                                                                                    \
        return Vc::detail::impl_or_fallback(                                             \
            [](const auto &xx) -> decltype(Vc::detail::doublev<Abi>(                     \
                Vc::detail::private_init,                                                \
                Vc::detail::get_impl_t<decltype(xx)>::name_(Vc::detail::data(xx)))) {    \
                return {                                                                 \
                    Vc::detail::private_init,                                            \
                    Vc::detail::get_impl_t<decltype(xx)>::name_(Vc::detail::data(xx))};  \
            },                                                                           \
            [](const Vc::detail::doublev<Abi> &xx) {                                     \
                return Vc::detail::doublev<Abi>(                                         \
                    [&](auto i) { return std::name_(xx[i]); });                          \
            },                                                                           \
            x);                                                                          \
    }                                                                                    \
    template <class Abi> Vc::detail::ldoublev<Abi> name_(Vc::detail::ldoublev<Abi> x)    \
    {                                                                                    \
        return Vc::detail::impl_or_fallback(                                             \
            [](const auto &xx) -> decltype(Vc::detail::ldoublev<Abi>(                    \
                Vc::detail::private_init,                                                \
                Vc::detail::get_impl_t<decltype(xx)>::name_(Vc::detail::data(xx)))) {    \
                return {                                                                 \
                    Vc::detail::private_init,                                            \
                    Vc::detail::get_impl_t<decltype(xx)>::name_(Vc::detail::data(xx))};  \
            },                                                                           \
            [](const Vc::detail::ldoublev<Abi> &xx) {                                    \
                return Vc::detail::ldoublev<Abi>(                                        \
                    [&](auto i) { return std::name_(xx[i]); });                          \
            },                                                                           \
            x);                                                                          \
    }

#define Vc_ARG_SAMESIZE_INT(arg1_) Vc::detail::samesize<int, arg1_>
#define Vc_ARG_SAMESIZE_LONG(arg1_) Vc::detail::samesize<long, arg1_>
#define Vc_ARG_AS_ARG1(arg1_) arg1_

#define Vc_MATH_CALL2_(name_, arg2_)                                                     \
    template <class Abi>                                                                 \
    detail::floatv<Abi> name_(Vc::detail::floatv<Abi> x_,                                \
                              arg2_(Vc::detail::floatv<Abi>) y_)                         \
    {                                                                                    \
        return Vc::detail::impl_or_fallback(                                             \
            [](const auto &x, const auto &y) -> decltype(Vc::detail::floatv<Abi>(        \
                Vc::detail::private_init,                                                \
                Vc::detail::get_impl_t<decltype(x)>::name_(Vc::detail::data(x),          \
                                                           Vc::detail::data(y)))) {      \
                return {Vc::detail::private_init,                                        \
                        Vc::detail::get_impl_t<decltype(x)>::name_(                      \
                            Vc::detail::data(x), Vc::detail::data(y))};                  \
            },                                                                           \
            [](const Vc::detail::floatv<Abi> &x, const auto &y) {                        \
                return Vc::detail::floatv<Abi>(                                          \
                    [&](auto i) { return std::name_(x[i], y[i]); });                     \
            },                                                                           \
            x_, y_);                                                                     \
    }                                                                                    \
    template <class Abi>                                                                 \
    detail::doublev<Abi> name_(Vc::detail::doublev<Abi> x_,                              \
                               arg2_(Vc::detail::doublev<Abi>) y_)                       \
    {                                                                                    \
        return Vc::detail::impl_or_fallback(                                             \
            [](const auto &x, const auto &y) -> decltype(Vc::detail::doublev<Abi>(       \
                Vc::detail::private_init,                                                \
                Vc::detail::get_impl_t<decltype(x)>::name_(Vc::detail::data(x),          \
                                                           Vc::detail::data(y)))) {      \
                return {Vc::detail::private_init,                                        \
                        Vc::detail::get_impl_t<decltype(x)>::name_(                      \
                            Vc::detail::data(x), Vc::detail::data(y))};                  \
            },                                                                           \
            [](const Vc::detail::doublev<Abi> &x, const auto &y) {                       \
                return Vc::detail::doublev<Abi>(                                         \
                    [&](auto i) { return std::name_(x[i], y[i]); });                     \
            },                                                                           \
            x_, y_);                                                                     \
    }                                                                                    \
    template <class Abi>                                                                 \
    detail::ldoublev<Abi> name_(Vc::detail::ldoublev<Abi> x_,                            \
                                arg2_(Vc::detail::ldoublev<Abi>) y_)                     \
    {                                                                                    \
        return Vc::detail::impl_or_fallback(                                             \
            [](const auto &x, const auto &y) -> decltype(Vc::detail::ldoublev<Abi>(      \
                Vc::detail::private_init,                                                \
                Vc::detail::get_impl_t<decltype(x)>::name_(Vc::detail::data(x),          \
                                                           Vc::detail::data(y)))) {      \
                return {Vc::detail::private_init,                                        \
                        Vc::detail::get_impl_t<decltype(x)>::name_(                      \
                            Vc::detail::data(x), Vc::detail::data(y))};                  \
            },                                                                           \
            [](const Vc::detail::ldoublev<Abi> &x, const auto &y) {                      \
                return Vc::detail::ldoublev<Abi>(                                        \
                    [&](auto i) { return std::name_(x[i], y[i]); });                     \
            },                                                                           \
            x_, y_);                                                                     \
    }

template < typename Abi>
static Vc_ALWAYS_INLINE floatv<Abi> cosSeries(const floatv<Abi> &x)
{
    using C = detail::trig<Abi, float>;
    const floatv<Abi> x2 = x * x;
    return ((C::cos_c2()  * x2 +
             C::cos_c1()) * x2 +
             C::cos_c0()) * (x2 * x2)
        - .5f * x2 + 1.f;
}
template <typename Abi>
static Vc_ALWAYS_INLINE doublev<Abi> cosSeries(const doublev<Abi> &x)
{
    using C = detail::trig<Abi, double>;
    const doublev<Abi> x2 = x * x;
    return (((((C::cos_c5()  * x2 +
                C::cos_c4()) * x2 +
                C::cos_c3()) * x2 +
                C::cos_c2()) * x2 +
                C::cos_c1()) * x2 +
                C::cos_c0()) * (x2 * x2)
        - .5 * x2 + 1.;
}

template <typename Abi>
static Vc_ALWAYS_INLINE floatv<Abi> sinSeries(const floatv<Abi>& x)
{
    using C = detail::trig<Abi, float>;
    const floatv<Abi> x2 = x * x;
    return ((C::sin_c2()  * x2 +
             C::sin_c1()) * x2 +
             C::sin_c0()) * (x2 * x)
        + x;
}

template <typename Abi>
static Vc_ALWAYS_INLINE doublev<Abi> sinSeries(const doublev<Abi> &x)
{
    using C = detail::trig<Abi, double>;
    const doublev<Abi> x2 = x * x;
    return (((((C::sin_c5()  * x2 +
                C::sin_c4()) * x2 +
                C::sin_c3()) * x2 +
                C::sin_c2()) * x2 +
                C::sin_c1()) * x2 +
                C::sin_c0()) * (x2 * x)
        + x;
}

template <class Abi>
Vc_ALWAYS_INLINE std::pair<floatv<Abi>, rebind_simd<int, floatv<Abi>>> foldInput(
    floatv<Abi> x)
{
    using V = floatv<Abi>;
    using C = detail::trig<Abi, float>;
    using IV = rebind_simd<int, V>;

    x = abs(x);
#if defined(Vc_HAVE_FMA4) || defined(Vc_HAVE_FMA)
    rebind_simd<int, V> quadrant =
        static_simd_cast<IV>(x * C::_4_pi() + 1.f);  // prefer the fma here
    quadrant &= ~1;
#else
    rebind_simd<int, V> quadrant = static_simd_cast<IV>(x * C::_4_pi());
    quadrant += quadrant & 1;
#endif
    const V y = static_simd_cast<V>(quadrant);
    quadrant &= 7;

    return {((x - y * C::pi_4_hi()) - y * C::pi_4_rem1()) - y * C::pi_4_rem2(), quadrant};
}

template <typename Abi>
static Vc_ALWAYS_INLINE std::pair<doublev<Abi>, rebind_simd<int, doublev<Abi>>> foldInput(
    doublev<Abi> x)
{
    using V = doublev<Abi>;
    using C = detail::trig<Abi, double>;
    using IV = rebind_simd<int, V>;

    x = abs(x);
    V y = trunc(x / C::pi_4());  // * C::4_pi() would work, but is >twice as imprecise
    V z = y - trunc(y * C::_1_16()) * C::_16();  // y modulo 16
    IV quadrant = static_simd_cast<IV>(z);
    const auto mask = (quadrant & 1) != 0;
    ++where(mask, quadrant);
    where(static_simd_cast<typename V::mask_type>(mask), y) += V(1);
    quadrant &= 7;

    // since y is an integer we don't need to split y into low and high parts until the
    // integer
    // requires more bits than there are zero bits at the end of _pi_4_hi (30 bits -> 1e9)
    return {((x - y * C::pi_4_hi()) - y * C::pi_4_rem1()) - y * C::pi_4_rem2(), quadrant};
}

// extract_exponent_bits {{{
template <class Abi>
samesize<int, doublev<Abi>> extract_exponent_bits(const doublev<Abi> &v)
{
    using namespace Vc::experimental;
    using namespace Vc::experimental::float_bitwise_operators;
    const doublev<Abi> exponent_mask =
        std::numeric_limits<double>::infinity();  // 0x7ff0000000000000
    constexpr auto N = simd_size_v<double, Abi> * 2;
    constexpr auto Max = simd_abi::max_fixed_size;
    if constexpr (N > Max) {
        const auto tup = split<Max / 2, (N - Max) / 2>(v & exponent_mask);
        return concat(
            shuffle<strided<2, 1>>(
                simd_reinterpret_cast<fixed_size_simd<int, Max>>(std::get<0>(tup))),
            shuffle<strided<2, 1>>(
                simd_reinterpret_cast<fixed_size_simd<int, N - Max>>(std::get<1>(tup))));
    } else {
        return shuffle<strided<2, 1>>(
            simd_reinterpret_cast<fixed_size_simd<int, 2 * simd_size_v<double, Abi>>>(
                v & exponent_mask));
    }
}

// }}}
// impl_or_fallback {{{
template <class ImplFun, class FallbackFun, class... Args>
Vc_INTRINSIC auto impl_or_fallback_dispatch(int, ImplFun &&impl_fun, FallbackFun &&,
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

template <class... Args> Vc_INTRINSIC auto impl_or_fallback(Args &&... args)
{
    return impl_or_fallback_dispatch(int(), std::forward<Args>(args)...);
}  //}}}
}  // namespace detail

Vc_MATH_CALL_(acos)
Vc_MATH_CALL_(asin)
Vc_MATH_CALL_(atan)
Vc_MATH_CALL2_(atan2, Vc_ARG_AS_ARG1)

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
template <class Abi> detail::floatv<Abi> cos(detail::floatv<Abi> x)
{
    using V = detail::floatv<Abi>;
    using M = typename V::mask_type;
    using IV = detail::rebind_simd<int, V>;

    IV quadrant;
    const V z = detail::foldInput(x, quadrant);
    const M sign = (x < V::Zero()) ^ simd_cast<M>(quadrant > 3);
    quadrant(quadrant > 3) -= 4;

    V y = sinSeries(z);
    y(simd_cast<M>(quadrant == IV::One() || quadrant == 2)) = cosSeries(z);
    y(sign) = -y;
    return y;
}
template <class Abi> detail::doublev<Abi> cos(detail::doublev<Abi> x);
template <class Abi> detail::ldoublev<Abi> cos(detail::ldoublev<Abi> x);

template <class Abi> detail::floatv<Abi> sin(detail::floatv<Abi> x)
{
    using V = detail::floatv<Abi>;
    using M = typename V::mask_type;

    auto folded = foldInput(x);
    const V &z = folded.first;
    auto &quadrant = folded.second;
    const M sign = (x < 0) ^ static_simd_cast<M>(quadrant > 3);
    where(quadrant > 3, quadrant) -= 4;

    V y = sinSeries(z);
    where(static_simd_cast<M>(quadrant == 1 || quadrant == 2), y) = cosSeries(z);
    where(sign, y) = -y;
    Vc_DEBUG(sine)
        (Vc_PRETTY_PRINT(x))
        (Vc_PRETTY_PRINT(sign))
        (Vc_PRETTY_PRINT(z))
        (Vc_PRETTY_PRINT(folded.second))
        (Vc_PRETTY_PRINT(quadrant))
        (Vc_PRETTY_PRINT(y));
    return y;
}

template <class Abi> detail::doublev<Abi> sin(detail::doublev<Abi> x)
{
    using V = detail::doublev<Abi>;
    using M = typename V::mask_type;

    M sign = x < 0;
    auto tmp = foldInput(x);
    const V &z = tmp.first;
    auto &quadrant = tmp.second;
    sign ^= static_simd_cast<M>(quadrant > 3);
    where(quadrant > 3, quadrant) -= 4;

    V y = sinSeries(z);
    where(static_simd_cast<M>(quadrant == 1 || quadrant == 2), y) = cosSeries(x);
    where(sign, y) = -y;
    Vc_DEBUG(sine)
        (Vc_PRETTY_PRINT(x))
        (Vc_PRETTY_PRINT(sign))
        (Vc_PRETTY_PRINT(z))
        (Vc_PRETTY_PRINT(tmp.second))
        (Vc_PRETTY_PRINT(quadrant))
        (Vc_PRETTY_PRINT(y));
    return y;
}

template <class Abi> detail::ldoublev<Abi> sin(detail::ldoublev<Abi> x);

template <> detail::floatv<simd_abi::scalar> sin(detail::floatv<simd_abi::scalar> x)
{
    return std::sin(detail::data(x));
}

template <> detail::doublev<simd_abi::scalar> sin(detail::doublev<simd_abi::scalar> x)
{
    return std::sin(detail::data(x));
}

template <> detail::ldoublev<simd_abi::scalar> sin(detail::ldoublev<simd_abi::scalar> x)
{
    return std::sin(detail::data(x));
}

Vc_MATH_CALL_(tan)
Vc_MATH_CALL_(acosh)
Vc_MATH_CALL_(asinh)
Vc_MATH_CALL_(atanh)
Vc_MATH_CALL_(cosh)
Vc_MATH_CALL_(sinh)
Vc_MATH_CALL_(tanh)
Vc_MATH_CALL_(exp)
Vc_MATH_CALL_(exp2)
Vc_MATH_CALL_(expm1)

template <class Abi>
detail::floatv<Abi> frexp(detail::floatv<Abi> value_,
                          detail::samesize<int, detail::floatv<Abi>> * exp_)
{
    return {detail::private_init,
            detail::impl_or_fallback(
                [](auto value, auto *exp) -> decltype(
                    detail::get_impl_t<decltype(value)>::frexp(detail::data(value),
                                                               detail::data(*exp))) {
                    return detail::get_impl_t<decltype(value)>::frexp(detail::data(value),
                                                                      detail::data(*exp));
                },
                [](auto value, auto *exp) {
                    using FV = detail::floatv<Abi>;
                    using IV = detail::samesize<int, FV>;
                    using IM = typename IV::mask_type;
                    using limits = std::numeric_limits<float>;
                    using namespace Vc::experimental;
                    using namespace Vc::experimental::float_bitwise_operators;

                    const FV exponent_mask = limits::infinity();  // 0x7f800000
                    const FV p5_1_exponent = detail::float_const<-1, 0x007fffffu, -1>;
                    FV mant = p5_1_exponent & (exponent_mask | value);
                    const IV exponent_bits =
                        simd_reinterpret_cast<IV>(value & exponent_mask);
                    if (Vc_IS_LIKELY(all_of(isnormal(value)))) {
                        *exp = (exponent_bits >> 23) - 0x7e;
                        return detail::data(mant);
                    }
                    const auto iszero_inf_nan =
                        isunordered(value * limits::infinity(), value * FV());
                    const FV scaled_subnormal = value * detail::float_const<1, 0, 0x70>;
                    const FV mant_subnormal =
                        p5_1_exponent & (exponent_mask | scaled_subnormal);
                    where(!isnormal(value), mant) = mant_subnormal;
                    where(iszero_inf_nan, mant) = value;
                    IV e = simd_reinterpret_cast<IV>(scaled_subnormal & exponent_mask);
                    const IM value_isnormal = static_simd_cast<IM>(isnormal(value));
                    where(value_isnormal, e) = exponent_bits;
                    const IV offset =
                        (simd_reinterpret_cast<IV>(value_isnormal) & IV(0x7e)) |
                        (simd_reinterpret_cast<IV>((exponent_bits == 0) &
                                                   (static_simd_cast<IM>(value != 0))) &
                         IV(0xee));
                    *exp = (e >> 23) - offset;
                    return detail::data(mant);
                },
                value_, exp_)};
}
template <class Abi>
detail::doublev<Abi> frexp(detail::doublev<Abi> value_,
                           detail::samesize<int, detail::doublev<Abi>> * exp_)
{
    *exp_ = 0xdead;
    return {detail::private_init,
            detail::impl_or_fallback(
                [](auto value, auto *exp) -> decltype(
                    detail::get_impl_t<decltype(value)>::frexp(detail::data(value),
                                                               detail::data(*exp))) {
                    //Vc_DEBUG("native");
                    //Vc_DEBUG_DEFERRED("*exp = ", *exp);
                    return detail::get_impl_t<decltype(value)>::frexp(detail::data(value),
                                                                      detail::data(*exp));
                },
                [](auto value, auto *exp) {
                    //Vc_DEBUG("fallback");
                    //Vc_DEBUG_DEFERRED("*exp = ", *exp);
                    using DV = detail::doublev<Abi>;
                    using IV = detail::samesize<int, DV>;
                    using IM = typename IV::mask_type;
                    using limits = std::numeric_limits<double>;
                    using namespace Vc::experimental;
                    using namespace Vc::experimental::float_bitwise_operators;

                    const DV exponent_mask = limits::infinity();  // 0x7ff0000000000000
                    const DV p5_1_exponent =
                        detail::double_const<-1, 0x000fffffffffffffull, -1>;
                    DV mant = p5_1_exponent & (exponent_mask | value);
                    const IV exponent_bits = extract_exponent_bits(value);
                    if (Vc_IS_LIKELY(all_of(isnormal(value)))) {
                        *exp = (exponent_bits >> 20) - 0x3fe;
                        return detail::data(mant);
                    }
                    const auto iszero_inf_nan =
                        isunordered(value * limits::infinity(), value * DV());
                    const DV scaled_subnormal = value * detail::double_const<1, 0, 0x200>;
                    const DV mant_subnormal =
                        p5_1_exponent & (exponent_mask | scaled_subnormal);
                    where(!isnormal(value), mant) = mant_subnormal;
                    where(iszero_inf_nan, mant) = value;
                    IV e = extract_exponent_bits(scaled_subnormal);
                    const IM value_isnormal = static_simd_cast<IM>(isnormal(value));
                    where(value_isnormal, e) = exponent_bits;
                    const IV offset =
                        (simd_reinterpret_cast<IV>(value_isnormal) & IV(0x3fe)) |
                        (simd_reinterpret_cast<IV>((exponent_bits == 0) &
                                                   (static_simd_cast<IM>(value != 0))) &
                         IV(0x5fe));
                    *exp = (e >> 20) - offset;
                    return detail::data(mant);
                },
                value_, exp_)};
}
template <class Abi>
detail::ldoublev<Abi> frexp(detail::ldoublev<Abi> value,
                            detail::samesize<int, detail::ldoublev<Abi>> * exp)
{
    return {detail::private_init, detail::get_impl_t<decltype(value)>::frexp(
                                      detail::data(value), detail::data(*exp))};
}

template <class Abi>
detail::samesize<int, detail::floatv<Abi>> ilogb(detail::floatv<Abi> x);
template <class Abi>
detail::samesize<int, detail::doublev<Abi>> ilogb(detail::doublev<Abi> x);
template <class Abi>
detail::samesize<int, detail::ldoublev<Abi>> ilogb(detail::ldoublev<Abi> x);

Vc_MATH_CALL2_(ldexp, Vc_ARG_SAMESIZE_INT)

Vc_MATH_CALL_(log)
Vc_MATH_CALL_(log10)
Vc_MATH_CALL_(log1p)
Vc_MATH_CALL_(log2)
Vc_MATH_CALL_(logb)

template <class Abi>
detail::floatv<Abi> modf(detail::floatv<Abi> value, detail::floatv<Abi> * iptr);
template <class Abi>
detail::doublev<Abi> modf(detail::doublev<Abi> value, detail::doublev<Abi> * iptr);
template <class Abi>
detail::ldoublev<Abi> modf(detail::ldoublev<Abi> value, detail::ldoublev<Abi> * iptr);

Vc_MATH_CALL2_(scalbn, Vc_ARG_SAMESIZE_INT)
Vc_MATH_CALL2_(scalbln, Vc_ARG_SAMESIZE_LONG)

Vc_MATH_CALL_(cbrt)

template <class Abi> detail::scharv<Abi> abs(detail::scharv<Abi> j)
{
    return {detail::private_init, detail::get_impl_t<decltype(j)>::abs(detail::data(j))};
}
template <class Abi> detail::shortv<Abi> abs(detail::shortv<Abi> j)
{
    return {detail::private_init, detail::get_impl_t<decltype(j)>::abs(detail::data(j))};
}
template <class Abi> detail::intv<Abi> abs(detail::intv<Abi> j)
{
    return {detail::private_init, detail::get_impl_t<decltype(j)>::abs(detail::data(j))};
}
template <class Abi> detail::longv<Abi> abs(detail::longv<Abi> j)
{
    return {detail::private_init, detail::get_impl_t<decltype(j)>::abs(detail::data(j))};
}
template <class Abi> detail::llongv<Abi> abs(detail::llongv<Abi> j)
{
    return {detail::private_init, detail::get_impl_t<decltype(j)>::abs(detail::data(j))};
}
template <class Abi> detail::floatv<Abi> abs(detail::floatv<Abi> j)
{
    return {detail::private_init, detail::get_impl_t<decltype(j)>::abs(detail::data(j))};
}
template <class Abi> detail::doublev<Abi> abs(detail::doublev<Abi> j)
{
    return {detail::private_init, detail::get_impl_t<decltype(j)>::abs(detail::data(j))};
}
template <class Abi> detail::ldoublev<Abi> abs(detail::ldoublev<Abi> j)
{
    return {detail::private_init, detail::get_impl_t<decltype(j)>::abs(detail::data(j))};
}

Vc_MATH_CALL2_(hypot, Vc_ARG_AS_ARG1)

template <class Abi>
detail::floatv<Abi> hypot(detail::floatv<Abi> x, detail::floatv<Abi> y,
                          detail::floatv<Abi> z);
template <class Abi>
detail::doublev<Abi> hypot(detail::doublev<Abi> x, detail::doublev<Abi> y,
                           detail::doublev<Abi> z);
template <class Abi>
detail::ldoublev<Abi> hypot(detail::ldoublev<Abi> x, detail::ldoublev<Abi> y,
                            detail::ldoublev<Abi> z);

Vc_MATH_CALL2_(pow, Vc_ARG_AS_ARG1)

Vc_MATH_CALL_(sqrt)
Vc_MATH_CALL_(erf)
Vc_MATH_CALL_(erfc)
Vc_MATH_CALL_(lgamma)
Vc_MATH_CALL_(tgamma)
Vc_MATH_CALL_(ceil)
Vc_MATH_CALL_(floor)
Vc_MATH_CALL_(nearbyint)
Vc_MATH_CALL_(rint)

template <class Abi>
detail::samesize<long int, detail::floatv<Abi>> lrint(detail::floatv<Abi> x);
template <class Abi>
detail::samesize<long int, detail::doublev<Abi>> lrint(detail::doublev<Abi> x);
template <class Abi>
detail::samesize<long int, detail::ldoublev<Abi>> lrint(detail::ldoublev<Abi> x);

template <class Abi>
detail::samesize<long long int, detail::floatv<Abi>> llrint(detail::floatv<Abi> x);
template <class Abi>
detail::samesize<long long int, detail::doublev<Abi>> llrint(detail::doublev<Abi> x);
template <class Abi>
detail::samesize<long long int, detail::ldoublev<Abi>> llrint(detail::ldoublev<Abi> x);

Vc_MATH_CALL_(round)

template <class Abi>
detail::samesize<long int, detail::floatv<Abi>> lround(detail::floatv<Abi> x);
template <class Abi>
detail::samesize<long int, detail::doublev<Abi>> lround(detail::doublev<Abi> x);
template <class Abi>
detail::samesize<long int, detail::ldoublev<Abi>> lround(detail::ldoublev<Abi> x);

template <class Abi>
detail::samesize<long long int, detail::floatv<Abi>> llround(detail::floatv<Abi> x);
template <class Abi>
detail::samesize<long long int, detail::doublev<Abi>> llround(detail::doublev<Abi> x);
template <class Abi>
detail::samesize<long long int, detail::ldoublev<Abi>> llround(detail::ldoublev<Abi> x);

Vc_MATH_CALL_(trunc)

Vc_MATH_CALL2_(fmod, Vc_ARG_AS_ARG1)
Vc_MATH_CALL2_(remainder, Vc_ARG_AS_ARG1)

template <class Abi>
detail::floatv<Abi> remquo(detail::floatv<Abi> x, detail::floatv<Abi> y,
                           detail::samesize<int, detail::floatv<Abi>> * quo);
template <class Abi>
detail::doublev<Abi> remquo(detail::doublev<Abi> x, detail::doublev<Abi> y,
                            detail::samesize<int, detail::doublev<Abi>> * quo);
template <class Abi>
detail::ldoublev<Abi> remquo(detail::ldoublev<Abi> x, detail::ldoublev<Abi> y,
                             detail::samesize<int, detail::ldoublev<Abi>> * quo);

Vc_MATH_CALL2_(copysign, Vc_ARG_AS_ARG1)

template <class Abi> detail::doublev<Abi> nan(const char* tagp);
template <class Abi> detail::floatv<Abi> nanf(const char* tagp);
template <class Abi> detail::ldoublev<Abi> nanl(const char* tagp);

Vc_MATH_CALL2_(nextafter, Vc_ARG_AS_ARG1)
Vc_MATH_CALL2_(nexttoward, Vc_ARG_AS_ARG1)
Vc_MATH_CALL2_(fdim, Vc_ARG_AS_ARG1)
Vc_MATH_CALL2_(fmax, Vc_ARG_AS_ARG1)
Vc_MATH_CALL2_(fmin, Vc_ARG_AS_ARG1)

template <class Abi>
detail::floatv<Abi> fma(detail::floatv<Abi> x, detail::floatv<Abi> y,
                        detail::floatv<Abi> z);
template <class Abi>
detail::doublev<Abi> fma(detail::doublev<Abi> x, detail::doublev<Abi> y,
                         detail::doublev<Abi> z);
template <class Abi>
detail::ldoublev<Abi> fma(detail::ldoublev<Abi> x, detail::ldoublev<Abi> y,
                          detail::ldoublev<Abi> z);

template <class Abi>
detail::samesize<int, detail::floatv<Abi>> fpclassify(detail::floatv<Abi> x)
{
    return {detail::private_init,
            detail::get_impl_t<decltype(x)>::fpclassify(detail::data(x))};
}
template <class Abi>
detail::samesize<int, detail::doublev<Abi>> fpclassify(detail::doublev<Abi> x)
{
    return {detail::private_init,
            detail::get_impl_t<decltype(x)>::fpclassify(detail::data(x))};
}
template <class Abi>
detail::samesize<int, detail::ldoublev<Abi>> fpclassify(detail::ldoublev<Abi> x)
{
    return {detail::private_init,
            detail::get_impl_t<decltype(x)>::fpclassify(detail::data(x))};
}

#define Vc_MATH_CLASS_(name_)                                                            \
    template <class Abi> simd_mask<float, Abi> name_(detail::floatv<Abi> x)              \
    {                                                                                    \
        return {detail::private_init,                                                    \
                detail::get_impl_t<decltype(x)>::name_(detail::data(x))};                \
    }                                                                                    \
    template <class Abi> simd_mask<double, Abi> name_(detail::doublev<Abi> x)            \
    {                                                                                    \
        return {detail::private_init,                                                    \
                detail::get_impl_t<decltype(x)>::name_(detail::data(x))};                \
    }                                                                                    \
    template <class Abi> simd_mask<long double, Abi> name_(detail::ldoublev<Abi> x)      \
    {                                                                                    \
        return {detail::private_init,                                                    \
                detail::get_impl_t<decltype(x)>::name_(detail::data(x))};                \
    }

Vc_MATH_CLASS_(isfinite)
Vc_MATH_CLASS_(isinf)
Vc_MATH_CLASS_(isnan)
Vc_MATH_CLASS_(isnormal)
Vc_MATH_CLASS_(signbit)
#undef Vc_MATH_CLASS_

#define Vc_MATH_CMP_(name_)                                                              \
    template <class Abi>                                                                 \
    simd_mask<float, Abi> name_(detail::floatv<Abi> x, detail::floatv<Abi> y)            \
    {                                                                                    \
        return {detail::private_init, detail::get_impl_t<decltype(x)>::name_(            \
                                          detail::data(x), detail::data(y))};            \
    }                                                                                    \
    template <class Abi>                                                                 \
    simd_mask<double, Abi> name_(detail::doublev<Abi> x, detail::doublev<Abi> y)         \
    {                                                                                    \
        return {detail::private_init, detail::get_impl_t<decltype(x)>::name_(            \
                                          detail::data(x), detail::data(y))};            \
    }                                                                                    \
    template <class Abi>                                                                 \
    simd_mask<long double, Abi> name_(detail::ldoublev<Abi> x, detail::ldoublev<Abi> y)  \
    {                                                                                    \
        return {detail::private_init, detail::get_impl_t<decltype(x)>::name_(            \
                                          detail::data(x), detail::data(y))};            \
    }

Vc_MATH_CMP_(isgreater)
Vc_MATH_CMP_(isgreaterequal)
Vc_MATH_CMP_(isless)
Vc_MATH_CMP_(islessequal)
Vc_MATH_CMP_(islessgreater)
Vc_MATH_CMP_(isunordered)
#undef Vc_MATH_CMP_

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

#undef Vc_MATH_CALL_

namespace detail
{
template <class T, bool = std::is_arithmetic<std::decay_t<T>>::value>
struct autocvt_to_simd {
    T d;
    using TT = std::decay_t<T>;
    operator TT() { return d; }
    operator TT &()
    {
        static_assert(std::is_lvalue_reference<T>::value, "");
        static_assert(!std::is_const<T>::value, "");
        return d;
    }
    operator TT *()
    {
        static_assert(std::is_lvalue_reference<T>::value, "");
        static_assert(!std::is_const<T>::value, "");
        return &d;
    }

    template <class Abi> operator simd<typename TT::value_type, Abi>()
    {
        return {detail::private_init, d};
    }

    template <class Abi> operator simd<typename TT::value_type, Abi> *()
    {
        return reinterpret_cast<simd<typename TT::value_type, Abi> *>(&d);
    }
};

template <class T> struct autocvt_to_simd<T, true> {
    using TT = std::decay_t<T>;
    T d;
    fixed_size_simd<TT, 1> fd;

    autocvt_to_simd(T dd) : d(dd), fd(d) {}
    ~autocvt_to_simd()
    {
        //Vc_DEBUG("fd = ", detail::data(fd).first);
        d = detail::data(fd).first;
    }

    operator fixed_size_simd<TT, 1>()
    {
        return fd;
    }
    operator fixed_size_simd<TT, 1> *()
    {
        static_assert(std::is_lvalue_reference<T>::value, "");
        static_assert(!std::is_const<T>::value, "");
        return &fd;
    }
};
#define Vc_FIXED_SIZE_FWD_(name_)                                                        \
    template <class Impl, class Arg0, class... Args>                                     \
    Vc_INTRINSIC auto name_##_fwd::operator()(Impl, Arg0 &&arg0,                         \
                                              Args &&... args) noexcept                  \
    {                                                                                    \
        auto ret = detail::data(Vc::name_<typename Impl::abi_type>(                      \
            typename Impl::simd_type{detail::private_init, std::forward<Arg0>(arg0)},    \
            autocvt_to_simd<Args>{std::forward<Args>(args)}...));                        \
        /*Vc_DEBUG(args...);*/                                                           \
        return ret;                                                                      \
    }
Vc_FIXED_SIZE_FWD_(frexp)
#undef Vc_FIXED_SIZE_FWD_
}  // namespace detail

Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DETAIL_MATH_H_

// vim: foldmethod=marker
