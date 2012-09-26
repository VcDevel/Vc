/*  This file is part of the Vc library.

    Copyright (C) 2009-2012 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with Vc.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifndef VC_COMMON_TRIGONOMETRIC_H
#define VC_COMMON_TRIGONOMETRIC_H

//#define VC_DEBUG_TRIGONOMETRIC
#ifdef VC_DEBUG_TRIGONOMETRIC
#include <iostream>
#include <iomanip>
#endif
#include "const.h"
#include "macros.h"

namespace Vc
{
namespace Common
{
#ifdef VC__USE_NAMESPACE
    using Vc::VC__USE_NAMESPACE::Const;
    using Vc::VC__USE_NAMESPACE::Vector;
#endif
    typedef Vector<float> float_v;
    typedef Vector<double> double_v;
    typedef Vector<sfloat> sfloat_v;
    namespace {
        template<typename T> static inline ALWAYS_INLINE CONST Vector<T> _foldMinusPiToPi(const Vector<T> &x) {
            typedef Const<T> C;
            // put the input in the range [-π, π]
            // 'f(x) = 2π * round(x/2π)' is the offset:
            // ⇒ f(x) = 0 ∀ x ∈ ]-π, π[ ;  f(x) = 2π ∀ x ∈ [π, 3π[
            return x - round(x * C::_1_2pi()) * C::_2pi();
        }
    }

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
    template<typename _T> static inline Vector<_T> sin(const Vector<_T> &_x)
    {
        typedef Vector<_T> V;
        typedef Const<_T> C;
        typedef typename V::EntryType T;
        typedef typename V::Mask M;
        typedef typename V::IndexType IV;

        const T DP1 = 0.78515625f;
        const T DP2 = 2.4187564849853515625e-4f;
        const T DP3 = 3.77489497744594108e-8f;
        const T PIO4F = 0.7853981633974483096f;
        const T lossth = 8192.f;

        V x = abs(_x);
        M sign = _x < 0;
        IV j = static_cast<IV>(x * 1.27323954473516f);
        typename IV::Mask mask = (j & 1) != 0;
        j(mask) += 1;
        V y = static_cast<V>(j);
        j &= 7;
        sign ^= j > 3;
        j(j > 3) -= 4;

        M lossMask = x > lossth;
        x(lossMask) = x - y * PIO4F;
        x(!lossMask) = ((x - y * DP1) - y * DP2) - y * DP3;

        V z = x * x;
        V cos_s = ((2.443315711809948E-005f * z + -1.388731625493765E-003f)
                * z + 4.166664568298827E-002f) * (z * z)
            - 0.5f * z + 1.0f;
        V sin_s = ((-1.9515295891E-4f * z + 8.3321608736E-3f)
                * z + -1.6666654611E-1f) * (z * x)
            + x;
        y = sin_s;
        y(j == 1 || j == 2) = cos_s;
        y(sign) = -y;
        return y;
    }

    template<> inline double_v sin(const double_v &_x)
    {
        typedef double_v V;
        typedef Const<double> C;
        typedef V::EntryType T;
        typedef V::Mask M;
        typedef V::IndexType IV;
        const double PIO4 = Vc_buildDouble(1, 0x921fb54442d18, -1);
        const double DP1  = Vc_buildDouble(1, 0x921fb40000000, -1);
        const double DP2  = Vc_buildDouble(1, 0x4442d00000000, -25);
        const double DP3  = Vc_buildDouble(1, 0x8469898cc5170, -49);

        V x = abs(_x);
        M sign = _x < 0;
        V y = floor(x / PIO4);
        V z = y - floor(y * 0.0625) * 16.;
        IV j = static_cast<IV>(z);
        IV::Mask mask = (j & 1) != 0;
        j(mask) += 1;
        y(static_cast<M>(mask)) += 1.;
        j &= 7;
        sign ^= static_cast<M>(j > 3);
        j(j > 3) -= 4;

        // since y is an integer we don't need to split y into low and high parts until the integer
        // requires more bits than there are zero bits at the end of DP1 (30 bits -> 1e9)
        z = ((x - y * DP1) - y * DP2) - y * DP3;

        V zz = z * z;
        V cos_s = (((((Vc_buildDouble(-1, 0x8fa49a0861a9b, -37)  * zz +
                       Vc_buildDouble( 1, 0x1ee9d7b4e3f05, -29)) * zz +
                       Vc_buildDouble(-1, 0x27e4f7eac4bc6, -22)) * zz +
                       Vc_buildDouble( 1, 0xa01a019c844f5, -16)) * zz +
                       Vc_buildDouble(-1, 0x6c16c16c14f91, -10)) * zz +
                       Vc_buildDouble( 1, 0x555555555554b,  -5)) * (zz * zz)
                  - 0.5 * zz + 1.0;
        V sin_s = (((((Vc_buildDouble( 1, 0x5d8fd1fd19ccd, -33)  * zz +
                       Vc_buildDouble(-1, 0xae5e5a9291f5d, -26)) * zz +
                       Vc_buildDouble( 1, 0x71de3567d48a1, -19)) * zz +
                       Vc_buildDouble(-1, 0xa01a019bfdf03, -13)) * zz +
                       Vc_buildDouble( 1, 0x111111110f7d0,  -7)) * zz +
                       Vc_buildDouble(-1, 0x5555555555548,  -3)) * (zz * z)
                  + z;
        y = sin_s;
        y(static_cast<M>(j == 1 || j == 2)) = cos_s;
        y(sign) = -y;
        return y;
    }
    template<typename _T> static inline Vector<_T> cos(const Vector<_T> &_x) {
        typedef Vector<_T> V;
        typedef Const<_T> C;
        typedef typename V::EntryType T;
        typedef typename V::Mask M;
        typedef typename V::IndexType IV;

        const T DP1 = 0.78515625f;
        const T DP2 = 2.4187564849853515625e-4f;
        const T DP3 = 3.77489497744594108e-8f;
        const T PIO4F = 0.7853981633974483096f;
        const T lossth = 8192.f;

        V x = abs(_x);
        IV j = static_cast<IV>(x * 1.27323954473516f);
        typename IV::Mask mask = (j & 1) != 0;
        j(mask) += 1;
        V y = static_cast<V>(j);
        j &= 7;
        M sign = j > 3;
        j(j > 3) -= 4;
        sign ^= j > 1;

        M lossMask = x > lossth;
        x(lossMask) = x - y * PIO4F;
        x(!lossMask) = ((x - y * DP1) - y * DP2) - y * DP3;

        V z = x * x;
        V cos_s = ((2.443315711809948E-005f * z + -1.388731625493765E-003f)
                * z + 4.166664568298827E-002f) * (z * z)
            - 0.5f * z + 1.0f;
        V sin_s = ((-1.9515295891E-4f * z + 8.3321608736E-3f)
                * z + -1.6666654611E-1f) * (z * x)
            + x;
        y = cos_s;
        y(j == 1 || j == 2) = sin_s;
        y(sign) = -y;
        return y;
    }
    template<> inline double_v cos(const double_v &_x)
    {
        typedef double_v V;
        typedef Const<double> C;
        typedef V::EntryType T;
        typedef V::Mask M;
        typedef V::IndexType IV;
        const double PIO4 = Vc_buildDouble(1, 0x921fb54442d18, -1);
        const double DP1  = Vc_buildDouble(1, 0x921fb40000000, -1);
        const double DP2  = Vc_buildDouble(1, 0x4442d00000000, -25);
        const double DP3  = Vc_buildDouble(1, 0x8469898cc5170, -49);

        V x = abs(_x);
        V y = floor(x / PIO4);
        V z = y - floor(y * 0.0625) * 16.;
        IV j = static_cast<IV>(z);
        IV::Mask mask = (j & 1) != 0;
        j(mask) += 1;
        y(static_cast<M>(mask)) += 1.;
        j &= 7;
        M sign = static_cast<M>(j > 3);
        j(j > 3) -= 4;
        sign ^= static_cast<M>(j > 1);

        // since y is an integer we don't need to split y into low and high parts until the integer
        // requires more bits than there are zero bits at the end of DP1 (30 bits -> 1e9)
        z = ((x - y * DP1) - y * DP2) - y * DP3;

        V zz = z * z;
        V cos_s = (((((Vc_buildDouble(-1, 0x8fa49a0861a9b, -37)  * zz +
                       Vc_buildDouble( 1, 0x1ee9d7b4e3f05, -29)) * zz +
                       Vc_buildDouble(-1, 0x27e4f7eac4bc6, -22)) * zz +
                       Vc_buildDouble( 1, 0xa01a019c844f5, -16)) * zz +
                       Vc_buildDouble(-1, 0x6c16c16c14f91, -10)) * zz +
                       Vc_buildDouble( 1, 0x555555555554b,  -5)) * (zz * zz)
                  - 0.5 * zz + 1.0;
        V sin_s = (((((Vc_buildDouble( 1, 0x5d8fd1fd19ccd, -33)  * zz +
                       Vc_buildDouble(-1, 0xae5e5a9291f5d, -26)) * zz +
                       Vc_buildDouble( 1, 0x71de3567d48a1, -19)) * zz +
                       Vc_buildDouble(-1, 0xa01a019bfdf03, -13)) * zz +
                       Vc_buildDouble( 1, 0x111111110f7d0,  -7)) * zz +
                       Vc_buildDouble(-1, 0x5555555555548,  -3)) * (zz * z)
                  + z;
        y = cos_s;
        y(static_cast<M>(j == 1 || j == 2)) = sin_s;
        y(sign) = -y;
        return y;
    }
    template<typename T> static inline void sincos(const Vector<T> &_x, Vector<T> *_sin, Vector<T> *_cos) {
        typedef Vector<T> V;
        typedef Const<T> C;
        // I did a short test how the results would look if I make use of 1=s²+c². There seems to be
        // no easy way to keep the results in an acceptable precision.

        V sin_x = _foldMinusPiToPi(_x); // [-π, π]
        V cos_x = sin_x + C::_pi_2(); // [-½π, ¾π]
        cos_x(cos_x > C::_pi_2()) = C::_pi() - cos_x; // [-½π, ½π]

        // fold the left and right fourths in to reduce the range to [-½π, ½π]
        sin_x(sin_x >  C::_pi_2()) =  C::_pi() - sin_x;
        sin_x(sin_x < -C::_pi_2()) = -C::_pi() - sin_x;

        const V &sin_x2 = sin_x * sin_x;
        const V &cos_x2 = cos_x * cos_x;

        *_sin = sin_x * (V::One() - sin_x2 * (C::_1_3fac() - sin_x2 * (C::_1_5fac() - sin_x2 * (C::_1_7fac() - sin_x2 * C::_1_9fac()))));
        *_cos = cos_x * (V::One() - cos_x2 * (C::_1_3fac() - cos_x2 * (C::_1_5fac() - cos_x2 * (C::_1_7fac() - cos_x2 * C::_1_9fac()))));
    }
    template<typename _T> static inline Vector<_T> asin (const Vector<_T> &_x) {
        typedef Vector<_T> V;
        typedef typename V::EntryType T;
        typedef typename V::Mask M;

        const V pi_2(Math<T>::pi_2());
        const M &negative = _x < V::Zero();

        const V &a = abs(_x);
        //const M &outOfRange = a > V::One();
        const M &small = a < V(T(1.e-4));
        const M &gt_0_5 = a > V(T(0.5));
        V x = a;
        V z = a * a;
        z(gt_0_5) = (V::One() - a) * V(T(0.5));
        x(gt_0_5) = sqrt(z);
        z = ((((T(4.2163199048e-2)  * z
              + T(2.4181311049e-2)) * z
              + T(4.5470025998e-2)) * z
              + T(7.4953002686e-2)) * z
              + T(1.6666752422e-1)) * z * x
              + x;
        z(gt_0_5) = pi_2 - (z + z);
        z(small) = a;
        z(negative) = -z;
        //z(outOfRange) = nan;

        return z;
    }
    template<typename _T> static inline Vector<_T> atan (const Vector<_T> &_x) {
        typedef Vector<_T> V;
        typedef typename V::EntryType T;
        typedef typename V::Mask M;
        V x = abs(_x);
        const V pi_2(Math<T>::pi_2());
        const V pi_4(Math<T>::pi_4());
        const M &gt_tan_3pi_8 = x > V(T(2.414213562373095));
        const M &gt_tan_pi_8  = x > V(T(0.4142135623730950)) && !gt_tan_3pi_8;
        const V minusOne(-1);
        V y = V::Zero();
        y(gt_tan_3pi_8) = pi_2;
        y(gt_tan_pi_8)  = pi_4;
        x(gt_tan_3pi_8) = minusOne / x;
        x(gt_tan_pi_8)  = (x - V::One()) / (x + V::One());
        const V &x2 = x * x;
        y += (((T(8.05374449538e-2) * x2
              - T(1.38776856032E-1)) * x2
              + T(1.99777106478E-1)) * x2
              - T(3.33329491539E-1)) * x2 * x
              + x;
        y(_x < V::Zero()) = -y;
        return y;
    }
    template<typename _T> static inline Vector<_T> atan2(const Vector<_T> &y, const Vector<_T> &x) {
        typedef Vector<_T> V;
        typedef typename V::EntryType T;
        typedef typename V::Mask M;
        const V pi(Math<T>::pi());
        const V pi_2(Math<T>::pi_2());

        const M &xZero = x == V::Zero();
        const M &yZero = y == V::Zero();
        const M &xNeg = x < V::Zero();
        const M &yNeg = y < V::Zero();

        const V &absX = abs(x);
        const V &absY = abs(y);

        V a = absY / absX;
        const V pi_4(Math<T>::pi_4());
        const M &gt_tan_3pi_8 = a > V(T(2.414213562373095));
        const M &gt_tan_pi_8  = a > V(T(0.4142135623730950)) && !gt_tan_3pi_8;
        const V minusOne(-1);
        V b = V::Zero();
        b(gt_tan_3pi_8) = pi_2;
        b(gt_tan_pi_8)  = pi_4;
        a(gt_tan_3pi_8) = minusOne / a;
        a(gt_tan_pi_8)  = (absY - absX) / (absY + absX);
        const V &a2 = a * a;
        b += (((T(8.05374449538e-2) * a2
              - T(1.38776856032E-1)) * a2
              + T(1.99777106478E-1)) * a2
              - T(3.33329491539E-1)) * a2 * a
              + a;
        b(xNeg ^ yNeg) = -b;

        b(xNeg && !yNeg) += pi;
        b(xNeg &&  yNeg) -= pi;
        //b(xZero) = pi_2;
        b.setZero(xZero && yZero);
        b(xZero && yNeg) = -pi_2;
        //b(yZero && xNeg) = pi;
        return b;
    }
} // namespace Common
#ifdef VC__USE_NAMESPACE
namespace VC__USE_NAMESPACE
{
    using Vc::Common::sin;
    using Vc::Common::cos;
    using Vc::Common::sincos;
    using Vc::Common::asin;
    using Vc::Common::atan;
    using Vc::Common::atan2;
} // namespace VC__USE_NAMESPACE
#undef VC__USE_NAMESPACE
#endif
} // namespace Vc

#include "undomacros.h"

#endif // VC_COMMON_TRIGONOMETRIC_H
