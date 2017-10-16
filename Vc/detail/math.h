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
    template <class Abi> detail::floatv<Abi> name_(detail::floatv<Abi> x)                \
    {                                                                                    \
        return {detail::private_init,                                                    \
                detail::get_impl_t<decltype(x)>::name_(detail::data(x))};                \
    }                                                                                    \
    template <class Abi> detail::doublev<Abi> name_(detail::doublev<Abi> x)              \
    {                                                                                    \
        return {detail::private_init,                                                    \
                detail::get_impl_t<decltype(x)>::name_(detail::data(x))};                \
    }                                                                                    \
    template <class Abi> detail::ldoublev<Abi> name_(detail::ldoublev<Abi> x)            \
    {                                                                                    \
        return {detail::private_init,                                                    \
                detail::get_impl_t<decltype(x)>::name_(detail::data(x))};                \
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

}  // namespace detail

template <class Abi> detail::floatv<Abi> acos(detail::floatv<Abi> x);
template <class Abi> detail::doublev<Abi> acos(detail::doublev<Abi> x);
template <class Abi> detail::ldoublev<Abi> acos(detail::ldoublev<Abi> x);

template <class Abi> detail::floatv<Abi> asin(detail::floatv<Abi> x);
template <class Abi> detail::doublev<Abi> asin(detail::doublev<Abi> x);
template <class Abi> detail::ldoublev<Abi> asin(detail::ldoublev<Abi> x);

template <class Abi> detail::floatv<Abi> atan(detail::floatv<Abi> x);
template <class Abi> detail::doublev<Abi> atan(detail::doublev<Abi> x);
template <class Abi> detail::ldoublev<Abi> atan(detail::ldoublev<Abi> x);

template <class Abi>
detail::floatv<Abi> atan2(detail::floatv<Abi> y, detail::floatv<Abi> x);
template <class Abi>
detail::doublev<Abi> atan2(detail::doublev<Abi> y, detail::doublev<Abi> x);
template <class Abi>
detail::ldoublev<Abi> atan2(detail::ldoublev<Abi> y, detail::ldoublev<Abi> x);

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

template <class Abi> detail::floatv<Abi> tan(detail::floatv<Abi> x);
template <class Abi> detail::doublev<Abi> tan(detail::doublev<Abi> x);
template <class Abi> detail::ldoublev<Abi> tan(detail::ldoublev<Abi> x);

template <class Abi> detail::floatv<Abi> acosh(detail::floatv<Abi> x);
template <class Abi> detail::doublev<Abi> acosh(detail::doublev<Abi> x);
template <class Abi> detail::ldoublev<Abi> acosh(detail::ldoublev<Abi> x);

template <class Abi> detail::floatv<Abi> asinh(detail::floatv<Abi> x);
template <class Abi> detail::doublev<Abi> asinh(detail::doublev<Abi> x);
template <class Abi> detail::ldoublev<Abi> asinh(detail::ldoublev<Abi> x);

template <class Abi> detail::floatv<Abi> atanh(detail::floatv<Abi> x);
template <class Abi> detail::doublev<Abi> atanh(detail::doublev<Abi> x);
template <class Abi> detail::ldoublev<Abi> atanh(detail::ldoublev<Abi> x);

template <class Abi> detail::floatv<Abi> cosh(detail::floatv<Abi> x);
template <class Abi> detail::doublev<Abi> cosh(detail::doublev<Abi> x);
template <class Abi> detail::ldoublev<Abi> cosh(detail::ldoublev<Abi> x);

template <class Abi> detail::floatv<Abi> sinh(detail::floatv<Abi> x);
template <class Abi> detail::doublev<Abi> sinh(detail::doublev<Abi> x);
template <class Abi> detail::ldoublev<Abi> sinh(detail::ldoublev<Abi> x);

template <class Abi> detail::floatv<Abi> tanh(detail::floatv<Abi> x);
template <class Abi> detail::doublev<Abi> tanh(detail::doublev<Abi> x);
template <class Abi> detail::ldoublev<Abi> tanh(detail::ldoublev<Abi> x);

template <class Abi> detail::floatv<Abi> exp(detail::floatv<Abi> x);
template <class Abi> detail::doublev<Abi> exp(detail::doublev<Abi> x);
template <class Abi> detail::ldoublev<Abi> exp(detail::ldoublev<Abi> x);

template <class Abi> detail::floatv<Abi> exp2(detail::floatv<Abi> x);
template <class Abi> detail::doublev<Abi> exp2(detail::doublev<Abi> x);
template <class Abi> detail::ldoublev<Abi> exp2(detail::ldoublev<Abi> x);

template <class Abi> detail::floatv<Abi> expm1(detail::floatv<Abi> x);
template <class Abi> detail::doublev<Abi> expm1(detail::doublev<Abi> x);
template <class Abi> detail::ldoublev<Abi> expm1(detail::ldoublev<Abi> x);

template <class Abi>
detail::floatv<Abi> frexp(detail::floatv<Abi> value,
                          detail::samesize<int, detail::floatv<Abi>> * exp);
template <class Abi>
detail::doublev<Abi> frexp(detail::doublev<Abi> value,
                           detail::samesize<int, detail::doublev<Abi>> * exp);
template <class Abi>
detail::ldoublev<Abi> frexp(detail::ldoublev<Abi> value,
                            detail::samesize<int, detail::ldoublev<Abi>> * exp);

template <class Abi>
detail::samesize<int, detail::floatv<Abi>> ilogb(detail::floatv<Abi> x);
template <class Abi>
detail::samesize<int, detail::doublev<Abi>> ilogb(detail::doublev<Abi> x);
template <class Abi>
detail::samesize<int, detail::ldoublev<Abi>> ilogb(detail::ldoublev<Abi> x);

template <class Abi>
detail::floatv<Abi> ldexp(detail::floatv<Abi> x,
                          detail::samesize<int, detail::floatv<Abi>> exp);
template <class Abi>
detail::doublev<Abi> ldexp(detail::doublev<Abi> x,
                           detail::samesize<int, detail::doublev<Abi>> exp);
template <class Abi>
detail::ldoublev<Abi> ldexp(detail::ldoublev<Abi> x,
                            detail::samesize<int, detail::ldoublev<Abi>> exp);

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

template <class Abi>
detail::floatv<Abi> scalbn(detail::floatv<Abi> x,
                           detail::samesize<int, detail::floatv<Abi>> n);
template <class Abi>
detail::doublev<Abi> scalbn(detail::doublev<Abi> x,
                            detail::samesize<int, detail::doublev<Abi>> n);
template <class Abi>
detail::ldoublev<Abi> scalbn(detail::ldoublev<Abi> x,
                             detail::samesize<int, detail::ldoublev<Abi>> n);

template <class Abi>
detail::floatv<Abi> scalbln(detail::floatv<Abi> x,
                            detail::samesize<long int, detail::floatv<Abi>> n);
template <class Abi>
detail::doublev<Abi> scalbln(detail::doublev<Abi> x,
                             detail::samesize<long int, detail::doublev<Abi>> n);
template <class Abi>
detail::ldoublev<Abi> scalbln(detail::ldoublev<Abi> x,
                              detail::samesize<long int, detail::ldoublev<Abi>> n);

template <class Abi> detail::floatv<Abi> cbrt(detail::floatv<Abi> x);
template <class Abi> detail::doublev<Abi> cbrt(detail::doublev<Abi> x);
template <class Abi> detail::ldoublev<Abi> cbrt(detail::ldoublev<Abi> x);

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

template <class Abi>
detail::floatv<Abi> hypot(detail::floatv<Abi> x, detail::floatv<Abi> y);
template <class Abi>
detail::doublev<Abi> hypot(detail::doublev<Abi> x, detail::doublev<Abi> y);
template <class Abi>
detail::ldoublev<Abi> hypot(detail::doublev<Abi> x, detail::doublev<Abi> y);

template <class Abi>
detail::floatv<Abi> hypot(detail::floatv<Abi> x, detail::floatv<Abi> y,
                          detail::floatv<Abi> z);
template <class Abi>
detail::doublev<Abi> hypot(detail::doublev<Abi> x, detail::doublev<Abi> y,
                           detail::doublev<Abi> z);
template <class Abi>
detail::ldoublev<Abi> hypot(detail::ldoublev<Abi> x, detail::ldoublev<Abi> y,
                            detail::ldoublev<Abi> z);

template <class Abi>
detail::floatv<Abi> pow(detail::floatv<Abi> x, detail::floatv<Abi> y);
template <class Abi>
detail::doublev<Abi> pow(detail::doublev<Abi> x, detail::doublev<Abi> y);
template <class Abi>
detail::ldoublev<Abi> pow(detail::ldoublev<Abi> x, detail::ldoublev<Abi> y);

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

template <class Abi> detail::floatv<Abi> round(detail::floatv<Abi> x);
template <class Abi> detail::doublev<Abi> round(detail::doublev<Abi> x);
template <class Abi> detail::ldoublev<Abi> round(detail::ldoublev<Abi> x);

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

template <class Abi>
detail::floatv<Abi> fmod(detail::floatv<Abi> x, detail::floatv<Abi> y);
template <class Abi>
detail::doublev<Abi> fmod(detail::doublev<Abi> x, detail::doublev<Abi> y);
template <class Abi>
detail::ldoublev<Abi> fmod(detail::ldoublev<Abi> x, detail::ldoublev<Abi> y);

template <class Abi>
detail::floatv<Abi> remainder(detail::floatv<Abi> x, detail::floatv<Abi> y);
template <class Abi>
detail::doublev<Abi> remainder(detail::doublev<Abi> x, detail::doublev<Abi> y);
template <class Abi>
detail::ldoublev<Abi> remainder(detail::ldoublev<Abi> x, detail::ldoublev<Abi> y);

template <class Abi>
detail::floatv<Abi> remquo(detail::floatv<Abi> x, detail::floatv<Abi> y,
                           detail::samesize<int, detail::floatv<Abi>> * quo);
template <class Abi>
detail::doublev<Abi> remquo(detail::doublev<Abi> x, detail::doublev<Abi> y,
                            detail::samesize<int, detail::doublev<Abi>> * quo);
template <class Abi>
detail::ldoublev<Abi> remquo(detail::ldoublev<Abi> x, detail::ldoublev<Abi> y,
                             detail::samesize<int, detail::ldoublev<Abi>> * quo);

template <class Abi>
detail::floatv<Abi> copysign(detail::floatv<Abi> x, detail::floatv<Abi> y);
template <class Abi>
detail::doublev<Abi> copysign(detail::doublev<Abi> x, detail::doublev<Abi> y);
template <class Abi>
detail::ldoublev<Abi> copysign(detail::ldoublev<Abi> x, detail::ldoublev<Abi> y);

template <class Abi> detail::doublev<Abi> nan(const char* tagp);
template <class Abi> detail::floatv<Abi> nanf(const char* tagp);
template <class Abi> detail::ldoublev<Abi> nanl(const char* tagp);

template <class Abi>
detail::floatv<Abi> nextafter(detail::floatv<Abi> x, detail::floatv<Abi> y);
template <class Abi>
detail::doublev<Abi> nextafter(detail::doublev<Abi> x, detail::doublev<Abi> y);
template <class Abi>
detail::ldoublev<Abi> nextafter(detail::ldoublev<Abi> x, detail::ldoublev<Abi> y);

template <class Abi>
detail::floatv<Abi> nexttoward(detail::floatv<Abi> x, detail::ldoublev<Abi> y);
template <class Abi>
detail::doublev<Abi> nexttoward(detail::doublev<Abi> x, detail::ldoublev<Abi> y);
template <class Abi>
detail::ldoublev<Abi> nexttoward(detail::ldoublev<Abi> x, detail::ldoublev<Abi> y);

template <class Abi>
detail::floatv<Abi> fdim(detail::floatv<Abi> x, detail::floatv<Abi> y);
template <class Abi>
detail::doublev<Abi> fdim(detail::doublev<Abi> x, detail::doublev<Abi> y);
template <class Abi>
detail::ldoublev<Abi> fdim(detail::ldoublev<Abi> x, detail::ldoublev<Abi> y);

template <class Abi>
detail::floatv<Abi> fmax(detail::floatv<Abi> x, detail::floatv<Abi> y);
template <class Abi>
detail::doublev<Abi> fmax(detail::doublev<Abi> x, detail::doublev<Abi> y);
template <class Abi>
detail::ldoublev<Abi> fmax(detail::ldoublev<Abi> x, detail::ldoublev<Abi> y);

template <class Abi>
detail::floatv<Abi> fmin(detail::floatv<Abi> x, detail::floatv<Abi> y);
template <class Abi>
detail::doublev<Abi> fmin(detail::doublev<Abi> x, detail::doublev<Abi> y);
template <class Abi>
detail::ldoublev<Abi> fmin(detail::ldoublev<Abi> x, detail::ldoublev<Abi> y);

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
detail::samesize<int, detail::floatv<Abi>> fpclassify(detail::floatv<Abi> x);
template <class Abi>
detail::samesize<int, detail::doublev<Abi>> fpclassify(detail::doublev<Abi> x);
template <class Abi>
detail::samesize<int, detail::ldoublev<Abi>> fpclassify(detail::ldoublev<Abi> x);

template <class Abi> simd_mask<float, Abi> isfinite(detail::floatv<Abi> x);
template <class Abi> simd_mask<double, Abi> isfinite(detail::doublev<Abi> x);
template <class Abi> simd_mask<long double, Abi> isfinite(detail::ldoublev<Abi> x);

template <class Abi>
detail::samesize<int, detail::floatv<Abi>> isinf(detail::floatv<Abi> x);
template <class Abi>
detail::samesize<int, detail::doublev<Abi>> isinf(detail::doublev<Abi> x);
template <class Abi>
detail::samesize<int, detail::ldoublev<Abi>> isinf(detail::ldoublev<Abi> x);

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

Vc_MATH_CLASS_(isnan)
Vc_MATH_CLASS_(isnormal)
Vc_MATH_CLASS_(signbit)

template <class Abi>
simd_mask<float, Abi> isgreater(detail::floatv<Abi> x, detail::floatv<Abi> y);
template <class Abi>
simd_mask<double, Abi> isgreater(detail::doublev<Abi> x, detail::doublev<Abi> y);
template <class Abi>
simd_mask<long double, Abi> isgreater(detail::ldoublev<Abi> x, detail::ldoublev<Abi> y);

template <class Abi>
simd_mask<float, Abi> isgreaterequal(detail::floatv<Abi> x, detail::floatv<Abi> y);
template <class Abi>
simd_mask<double, Abi> isgreaterequal(detail::doublev<Abi> x, detail::doublev<Abi> y);
template <class Abi>
simd_mask<long double, Abi> isgreaterequal(detail::ldoublev<Abi> x,
                                           detail::ldoublev<Abi> y);

template <class Abi>
simd_mask<float, Abi> isless(detail::floatv<Abi> x, detail::floatv<Abi> y);
template <class Abi>
simd_mask<double, Abi> isless(detail::doublev<Abi> x, detail::doublev<Abi> y);
template <class Abi>
simd_mask<long double, Abi> isless(detail::ldoublev<Abi> x, detail::ldoublev<Abi> y);

template <class Abi>
simd_mask<float, Abi> islessequal(detail::floatv<Abi> x, detail::floatv<Abi> y);
template <class Abi>
simd_mask<double, Abi> islessequal(detail::doublev<Abi> x, detail::doublev<Abi> y);
template <class Abi>
simd_mask<long double, Abi> islessequal(detail::ldoublev<Abi> x, detail::ldoublev<Abi> y);

template <class Abi>
simd_mask<float, Abi> islessgreater(detail::floatv<Abi> x, detail::floatv<Abi> y);
template <class Abi>
simd_mask<double, Abi> islessgreater(detail::doublev<Abi> x, detail::doublev<Abi> y);
template <class Abi>
simd_mask<long double, Abi> islessgreater(detail::ldoublev<Abi> x,
                                          detail::ldoublev<Abi> y);

template <class Abi>
simd_mask<float, Abi> isunordered(detail::floatv<Abi> x, detail::floatv<Abi> y);
template <class Abi>
simd_mask<double, Abi> isunordered(detail::doublev<Abi> x, detail::doublev<Abi> y);
template <class Abi>
simd_mask<long double, Abi> isunordered(detail::ldoublev<Abi> x, detail::ldoublev<Abi> y);

template <class V> struct simd_div_t {
    V quot, rem;
};
template <class Abi>
simd_div_t<detail::scharv<Abi>> simd_div(detail::scharv<Abi> numer,
                                         detail::scharv<Abi> denom);
template <class Abi>
simd_div_t<detail::shortv<Abi>> simd_div(detail::shortv<Abi> numer,
                                         detail::shortv<Abi> denom);
template <class Abi>
simd_div_t<detail::intv<Abi>> simd_div(detail::intv<Abi> numer, detail::intv<Abi> denom);
template <class Abi>
simd_div_t<detail::longv<Abi>> simd_div(detail::longv<Abi> numer,
                                        detail::longv<Abi> denom);
template <class Abi>
simd_div_t<detail::llongv<Abi>> simd_div(detail::llongv<Abi> numer,
                                         detail::llongv<Abi> denom);

#undef Vc_MATH_CALL_
#undef Vc_MATH_CLASS_
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DETAIL_MATH_H_

// vim: foldmethod=marker
