/*  This file is part of the Vc library. {{{
Copyright © 2012-2015 Matthias Kretz <kretz@kde.org>

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

// enable bit operators for easier portable bit manipulation on floats
#define Vc_ENABLE_FLOAT_BIT_OPERATORS 1

#include <Vc/vector.h>
#if defined(Vc_IMPL_SSE) || defined(Vc_IMPL_AVX)
#include <common/macros.h>

namespace Vc_VERSIONED_NAMESPACE
{
namespace Common
{

namespace
{
using Vc::Vector;
template <typename Abi> using float_v_for = Vc::Vector<float, Abi>;
template <typename Abi> using double_v_for = Vc::Vector<double, Abi>;
template <typename T, typename Abi>
using Const = typename std::conditional<std::is_same<Abi, VectorAbi::Avx>::value,
                                        AVX::Const<T>, SSE::Const<T>>::type;

template <typename V>
using best_int_v_for =
    typename std::conditional<(V::size() <= Vector<int, VectorAbi::Best<int>>::size()),
                              Vector<int, VectorAbi::Best<int>>,
                              SimdArray<int, V::size()>>::type;
template <typename Abi> using float_int_v = best_int_v_for<float_v_for<Abi>>;
template <typename Abi> using double_int_v = best_int_v_for<double_v_for<Abi>>;

template <typename T, typename Abi>
static Vc_ALWAYS_INLINE Vector<T, Abi> cosSeries(const Vector<T, Abi> &x)
{
    typedef Const<T, Abi> C;
    const Vector<T, Abi> x2 = x * x;
    return ((C::cosCoeff(2)  * x2 +
             C::cosCoeff(1)) * x2 +
             C::cosCoeff(0)) * (x2 * x2)
        - C::_1_2() * x2 + Vector<T, Abi>::One();
}
template <typename Abi>
static Vc_ALWAYS_INLINE double_v_for<Abi> cosSeries(const double_v_for<Abi> &x)
{
    typedef Const<double, Abi> C;
    const double_v_for<Abi> x2 = x * x;
    return (((((C::cosCoeff(5)  * x2 +
                C::cosCoeff(4)) * x2 +
                C::cosCoeff(3)) * x2 +
                C::cosCoeff(2)) * x2 +
                C::cosCoeff(1)) * x2 +
                C::cosCoeff(0)) * (x2 * x2)
        - C::_1_2() * x2 + double_v_for<Abi>::One();
}
template <typename T, typename Abi>
static Vc_ALWAYS_INLINE Vector<T, Abi> sinSeries(const Vector<T, Abi> &x)
{
    typedef Const<T, Abi> C;
    const Vector<T, Abi> x2 = x * x;
    return ((C::sinCoeff(2)  * x2 +
             C::sinCoeff(1)) * x2 +
             C::sinCoeff(0)) * (x2 * x)
        + x;
}
template <typename Abi>
static Vc_ALWAYS_INLINE double_v_for<Abi> sinSeries(const double_v_for<Abi> &x)
{
    typedef Const<double, Abi> C;
    const double_v_for<Abi> x2 = x * x;
    return (((((C::sinCoeff(5)  * x2 +
                C::sinCoeff(4)) * x2 +
                C::sinCoeff(3)) * x2 +
                C::sinCoeff(2)) * x2 +
                C::sinCoeff(1)) * x2 +
                C::sinCoeff(0)) * (x2 * x)
        + x;
}

template <typename Abi>
static Vc_ALWAYS_INLINE float_v_for<Abi> foldInput(float_v_for<Abi> x, float_int_v<Abi> &quadrant)
{
    typedef float_v_for<Abi> V;
    typedef Const<float, Abi> C;
    typedef float_int_v<Abi> IV;

    x = abs(x);
#if defined(Vc_IMPL_FMA4) || defined(Vc_IMPL_FMA)
        quadrant = simd_cast<IV>(x * C::_4_pi() + V::One()); // prefer the fma here
        quadrant &= ~IV::One();
#else
        quadrant = simd_cast<IV>(x * C::_4_pi());
        quadrant += quadrant & IV::One();
#endif
        const V y = simd_cast<V>(quadrant);
        quadrant &= 7;

        return ((x - y * C::_pi_4_hi()) - y * C::_pi_4_rem1()) - y * C::_pi_4_rem2();
    }
template <typename Abi>
static Vc_ALWAYS_INLINE double_v_for<Abi> foldInput(double_v_for<Abi> x,
                                                double_int_v<Abi> &quadrant)
{
    typedef double_v_for<Abi> V;
    typedef Const<double, Abi> C;

    x = abs(x);
        V y = trunc(x / C::_pi_4()); // * C::_4_pi() would work, but is >twice as imprecise
        V z = y - trunc(y * C::_1_16()) * C::_16(); // y modulo 16
        quadrant = simd_cast<double_int_v<Abi>>(z);
        int_m mask = (quadrant & double_int_v<Abi>::One()) != double_int_v<Abi>::Zero();
        ++quadrant(mask);
        y(simd_cast<double_m>(mask)) += V::One();
        quadrant &= 7;

        // since y is an integer we don't need to split y into low and high parts until the integer
        // requires more bits than there are zero bits at the end of _pi_4_hi (30 bits -> 1e9)
        return ((x - y * C::_pi_4_hi()) - y * C::_pi_4_rem1()) - y * C::_pi_4_rem2();
    }
} // anonymous namespace

/*
 * algorithm for sine and cosine:
 *
 * The result can be calculated with sine or cosine depending on the π/4 section the input is
 * in.
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
template<> template<typename V> V Trigonometric<Vc::Detail::TrigonometricImplementation<Vc::CurrentImplementation::current()>>::sin(const V &_x)
{
    typedef typename V::Mask M;
    using IV = best_int_v_for<V>;

    IV quadrant;
    const V z = foldInput(_x, quadrant);
    const M sign = (_x < V::Zero()) ^ simd_cast<M>(quadrant > 3);
    quadrant(quadrant > 3) -= 4;

    V y = sinSeries(z);
    y(simd_cast<M>(quadrant == IV::One() || quadrant == 2)) = cosSeries(z);
    y(sign) = -y;
    return y;
}

template<> template<> Vc::double_v Trigonometric<Vc::Detail::TrigonometricImplementation<Vc::CurrentImplementation::current()>>::sin(const Vc::double_v &_x)
{
    typedef Vc::double_v V;
    typedef V::Mask M;

    double_int_v<V::abi> quadrant;
    M sign = _x < V::Zero();
    const V x = foldInput(_x, quadrant);
    sign ^= simd_cast<M>(quadrant > 3);
    quadrant(quadrant > 3) -= 4;

    V y = sinSeries(x);
    y(simd_cast<M>(quadrant == double_int_v<V::abi>::One() || quadrant == 2)) = cosSeries(x);
    y(sign) = -y;
    return y;
}
template<> template<typename V> V Trigonometric<Vc::Detail::TrigonometricImplementation<Vc::CurrentImplementation::current()>>::cos(const V &_x) {
    typedef typename V::Mask M;
    using IV = best_int_v_for<V>;

    IV quadrant;
    const V x = foldInput(_x, quadrant);
    M sign = simd_cast<M>(quadrant > 3);
    quadrant(quadrant > 3) -= 4;
    sign ^= simd_cast<M>(quadrant > IV::One());

    V y = cosSeries(x);
    y(simd_cast<M>(quadrant == IV::One() || quadrant == 2)) = sinSeries(x);
    y(sign) = -y;
    return y;
}
template<> template<> Vc::double_v Trigonometric<Vc::Detail::TrigonometricImplementation<Vc::CurrentImplementation::current()>>::cos(const Vc::double_v &_x)
{
    typedef Vc::double_v V;
    typedef V::Mask M;

    double_int_v<V::abi> quadrant;
    const V x = foldInput(_x, quadrant);
    M sign = simd_cast<M>(quadrant > 3);
    quadrant(quadrant > 3) -= 4;
    sign ^= simd_cast<M>(quadrant > double_int_v<V::abi>::One());

    V y = cosSeries(x);
    y(simd_cast<M>(quadrant == double_int_v<V::abi>::One() || quadrant == 2)) = sinSeries(x);
    y(sign) = -y;
    return y;
}
template<> template<typename V> void Trigonometric<Vc::Detail::TrigonometricImplementation<Vc::CurrentImplementation::current()>>::sincos(const V &_x, V *_sin, V *_cos) {
    typedef typename V::Mask M;
    using IV = best_int_v_for<V>;

    IV quadrant;
    const V x = foldInput(_x, quadrant);
    M sign = simd_cast<M>(quadrant > 3);
    quadrant(quadrant > 3) -= 4;

    const V cos_s = cosSeries(x);
    const V sin_s = sinSeries(x);

    V c = cos_s;
    c(simd_cast<M>(quadrant == IV::One() || quadrant == 2)) = sin_s;
    c(sign ^ simd_cast<M>(quadrant > IV::One())) = -c;
    *_cos = c;

    V s = sin_s;
    s(simd_cast<M>(quadrant == IV::One() || quadrant == 2)) = cos_s;
    s(sign ^ simd_cast<M>(_x < V::Zero())) = -s;
    *_sin = s;
}
template<> template<> void Trigonometric<Vc::Detail::TrigonometricImplementation<Vc::CurrentImplementation::current()>>::sincos(const Vc::double_v &_x, Vc::double_v *_sin, Vc::double_v *_cos) {
    typedef Vc::double_v V;
    typedef V::Mask M;

    double_int_v<V::abi> quadrant;
    const V x = foldInput(_x, quadrant);
    M sign = simd_cast<M>(quadrant > 3);
    quadrant(quadrant > 3) -= 4;

    const V cos_s = cosSeries(x);
    const V sin_s = sinSeries(x);

    V c = cos_s;
    c(simd_cast<M>(quadrant == double_int_v<V::abi>::One() || quadrant == 2)) = sin_s;
    c(sign ^ simd_cast<M>(quadrant > double_int_v<V::abi>::One())) = -c;
    *_cos = c;

    V s = sin_s;
    s(simd_cast<M>(quadrant == double_int_v<V::abi>::One() || quadrant == 2)) = cos_s;
    s(sign ^ simd_cast<M>(_x < V::Zero())) = -s;
    *_sin = s;
}
template<> template<typename V> V Trigonometric<Vc::Detail::TrigonometricImplementation<Vc::CurrentImplementation::current()>>::asin (const V &_x) {
    typedef typename V::EntryType T;
    typedef Const<T, typename V::abi> C;
    typedef typename V::Mask M;

    const M &negative = _x < V::Zero();

    const V &a = abs(_x);
    const M outOfRange = a > V::One();
    const M &small = a < C::smallAsinInput();
    const M &gt_0_5 = a > C::_1_2();
    V x = a;
    V z = a * a;
    z(gt_0_5) = (V::One() - a) * C::_1_2();
    x(gt_0_5) = sqrt(z);
    z = ((((C::asinCoeff0(0)  * z
          + C::asinCoeff0(1)) * z
          + C::asinCoeff0(2)) * z
          + C::asinCoeff0(3)) * z
          + C::asinCoeff0(4)) * z * x
          + x;
    z(gt_0_5) = C::_pi_2() - (z + z);
    z(small) = a;
    z(negative) = -z;
    z.setQnan(outOfRange);

    return z;
}
template<> template<> Vc::double_v Trigonometric<Vc::Detail::TrigonometricImplementation<Vc::CurrentImplementation::current()>>::asin (const Vc::double_v &_x) {
    typedef Vc::double_v V;
    typedef Const<double, V::abi> C;
    typedef V::Mask M;

    const M negative = _x < V::Zero();

    const V a = abs(_x);
    const M outOfRange = a > V::One();
    const M small = a < C::smallAsinInput();
    const M large = a > C::largeAsinInput();

    V zz = V::One() - a;
    const V r = (((C::asinCoeff0(0) * zz + C::asinCoeff0(1)) * zz + C::asinCoeff0(2)) * zz +
            C::asinCoeff0(3)) * zz + C::asinCoeff0(4);
    const V s = (((zz + C::asinCoeff1(0)) * zz + C::asinCoeff1(1)) * zz +
            C::asinCoeff1(2)) * zz + C::asinCoeff1(3);
    V sqrtzz = sqrt(zz + zz);
    V z = C::_pi_4() - sqrtzz;
    z -= sqrtzz * (zz * r / s) - C::_pi_2_rem();
    z += C::_pi_4();

    V a2 = a * a;
    const V p = ((((C::asinCoeff2(0) * a2 + C::asinCoeff2(1)) * a2 + C::asinCoeff2(2)) * a2 +
                C::asinCoeff2(3)) * a2 + C::asinCoeff2(4)) * a2 + C::asinCoeff2(5);
    const V q = ((((a2 + C::asinCoeff3(0)) * a2 + C::asinCoeff3(1)) * a2 +
                C::asinCoeff3(2)) * a2 + C::asinCoeff3(3)) * a2 + C::asinCoeff3(4);
    z(!large) = a * (a2 * p / q) + a;

    z(negative) = -z;
    z(small) = _x;
    z.setQnan(outOfRange);

    return z;
}
template<> template<typename V> V Trigonometric<Vc::Detail::TrigonometricImplementation<Vc::CurrentImplementation::current()>>::atan (const V &_x) {
    typedef typename V::EntryType T;
    typedef Const<T, typename V::abi> C;
    typedef typename V::Mask M;
    V x = abs(_x);
    const M &gt_tan_3pi_8 = x > C::atanThrsHi();
    const M &gt_tan_pi_8  = x > C::atanThrsLo() && !gt_tan_3pi_8;
    V y = V::Zero();
    y(gt_tan_3pi_8) = C::_pi_2();
    y(gt_tan_pi_8)  = C::_pi_4();
    x(gt_tan_3pi_8) = -V::One() / x;
    x(gt_tan_pi_8)  = (x - V::One()) / (x + V::One());
    const V &x2 = x * x;
    y += (((C::atanP(0)  * x2
          - C::atanP(1)) * x2
          + C::atanP(2)) * x2
          - C::atanP(3)) * x2 * x
          + x;
    y(_x < V::Zero()) = -y;
    y.setQnan(isnan(_x));
    return y;
}
template<> template<> Vc::double_v Trigonometric<Vc::Detail::TrigonometricImplementation<Vc::CurrentImplementation::current()>>::atan (const Vc::double_v &_x) {
    typedef Vc::double_v V;
    typedef Const<double, V::abi> C;
    typedef V::Mask M;

    M sign = _x < V::Zero();
    V x = abs(_x);
    M finite = isfinite(_x);
    V ret = C::_pi_2();
    V y = V::Zero();
    const M large = x > C::atanThrsHi();
    const M gt_06 = x > C::atanThrsLo();
    V tmp = (x - V::One()) / (x + V::One());
    tmp(large) = -V::One() / x;
    x(gt_06) = tmp;
    y(gt_06) = C::_pi_4();
    y(large) = C::_pi_2();
    V z = x * x;
    const V p = (((C::atanP(0) * z + C::atanP(1)) * z + C::atanP(2)) * z + C::atanP(3)) * z + C::atanP(4);
    const V q = ((((z + C::atanQ(0)) * z + C::atanQ(1)) * z + C::atanQ(2)) * z + C::atanQ(3)) * z + C::atanQ(4);
    z = z * p / q;
    z = x * z + x;
    V morebits = C::_pi_2_rem();
    morebits(!large) *= C::_1_2();
    z(gt_06) += morebits;
    ret(finite) = y + z;
    ret(sign) = -ret;
    ret.setQnan(isnan(_x));
    return ret;
}
template<> template<typename V> V Trigonometric<Vc::Detail::TrigonometricImplementation<Vc::CurrentImplementation::current()>>::atan2(const V &y, const V &x) {
    typedef typename V::EntryType T;
    typedef Const<T, typename V::abi> C;
    typedef typename V::Mask M;

    const M xZero = x == V::Zero();
    const M yZero = y == V::Zero();
    const M xMinusZero = xZero && isnegative(x);
    const M yNeg = y < V::Zero();
    const M xInf = !isfinite(x);
    const M yInf = !isfinite(y);

    V a = copysign(C::_pi(), y);
    a.setZero(x >= V::Zero());

    // setting x to any finite value will have atan(y/x) return sign(y/x)*pi/2, just in case x is inf
    V _x = x;
    _x(yInf) = copysign(V::One(), x);

    a += atan(y / _x);

    // if x is +0 and y is +/-0 the result is +0
    a.setZero(xZero && yZero);

    // for x = -0 we add/subtract pi to get the correct result
    a(xMinusZero) += copysign(C::_pi(), y);

    // atan2(-Y, +/-0) = -pi/2
    a(xZero && yNeg) = -C::_pi_2();

    // if both inputs are inf the output is +/- (3)pi/4
    a(xInf && yInf) += copysign(C::_pi_4(), x ^ ~y);

    // correct the sign of y if the result is 0
    a(a == V::Zero()) = copysign(a, y);

    // any NaN input will lead to NaN output
    a.setQnan(isnan(y) || isnan(x));

    return a;
}
template<> template<> Vc::double_v Trigonometric<Vc::Detail::TrigonometricImplementation<Vc::CurrentImplementation::current()>>::atan2 (const Vc::double_v &y, const Vc::double_v &x) {
    typedef Vc::double_v V;
    typedef Const<double, V::abi> C;
    typedef V::Mask M;

    const M xZero = x == V::Zero();
    const M yZero = y == V::Zero();
    const M xMinusZero = xZero && isnegative(x);
    const M yNeg = y < V::Zero();
    const M xInf = !isfinite(x);
    const M yInf = !isfinite(y);

    V a = copysign(V(C::_pi()), y);
    a.setZero(x >= V::Zero());

    // setting x to any finite value will have atan(y/x) return sign(y/x)*pi/2, just in case x is inf
    V _x = x;
    _x(yInf) = copysign(V::One(), x);

    a += atan(y / _x);

    // if x is +0 and y is +/-0 the result is +0
    a.setZero(xZero && yZero);

    // for x = -0 we add/subtract pi to get the correct result
    a(xMinusZero) += copysign(C::_pi(), y);

    // atan2(-Y, +/-0) = -pi/2
    a(xZero && yNeg) = -C::_pi_2();

    // if both inputs are inf the output is +/- (3)pi/4
    a(xInf && yInf) += copysign(C::_pi_4(), x ^ ~y);

    // correct the sign of y if the result is 0
    a(a == V::Zero()) = copysign(a, y);

    // any NaN input will lead to NaN output
    a.setQnan(isnan(y) || isnan(x));

    return a;
}

}
}

// instantiate the non-specialized template functions above
template Vc::float_v Vc::Common::Trigonometric<Vc::Detail::TrigonometricImplementation<Vc::CurrentImplementation::current()>>::sin(const Vc::float_v &);
template Vc::float_v Vc::Common::Trigonometric<Vc::Detail::TrigonometricImplementation<Vc::CurrentImplementation::current()>>::cos(const Vc::float_v &);
template Vc::float_v Vc::Common::Trigonometric<Vc::Detail::TrigonometricImplementation<Vc::CurrentImplementation::current()>>::asin(const Vc::float_v &);
template Vc::float_v Vc::Common::Trigonometric<Vc::Detail::TrigonometricImplementation<Vc::CurrentImplementation::current()>>::atan(const Vc::float_v &);
template Vc::float_v Vc::Common::Trigonometric<Vc::Detail::TrigonometricImplementation<Vc::CurrentImplementation::current()>>::atan2(const Vc::float_v &, const Vc::float_v &);
template void Vc::Common::Trigonometric<Vc::Detail::TrigonometricImplementation<Vc::CurrentImplementation::current()>>::sincos(const Vc::float_v &, Vc::float_v *, Vc::float_v *);
#endif
