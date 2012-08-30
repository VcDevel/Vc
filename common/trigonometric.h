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
     * Goodie: Π=3450066π in single-precision approximates the real number very closely. Only a
     * relative error of 7e-15 remains. For comparison, the relative error of π or 2π is 3e-8.
     * Additionally, 3450066 can be represented as sp number without loss.
     *
     * Since trig(x + 2π) = trig(x) we can add or subtract n*Π from the input to bring the
     * input into range.
     *
     * This requires to find n = round(x / Π)
     * 1 / Π = Vc_buildFloat(1, 0x46218E, -24); // 9.22619705079341656528413295745849609375e-8.
     * Calculation of n * Π will lose precision unless the last 22 bits of n are 0. Thus it is best
     * to split Π and n into two 12-bit numbers and calculate:
     *   x' = x - n₁Π₁ - (n₂Π₁ + n₁Π₂) - n₂Π₂
     */

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
    template<typename _T> static inline Vector<_T> sin(const Vector<_T> &_x) {
        typedef Vector<_T> V;
        typedef Const<_T> C;
        typedef typename V::EntryType T;
        typedef typename V::Mask M;

        V x = abs(_x);
        V correction = V::Zero();

        if (VC_IS_UNLIKELY(!(x <= Vc_buildFloat(1, 0x2562AE, 22)).isFull())) {
            const V somePi1(Vc_buildFloat(1, 0x256000, 23)); // Π₁
            const V somePi2(Vc_buildFloat(1, 0x2B8000,  9)); // Π₂
            const V somePi3(Vc_buildFloat(-1, 0x2411DE, -24)); // Π₃
            const V n = round(x * Vc_buildFloat(1, 0x46218E, -24));
            const V n1 = n & C::highMask();
            const V n2 = n - n1;
            x = x - n1 * somePi1 - (n1 * somePi2 + n2 * somePi1) - n2 * somePi2;
            correction = n * somePi3;
        }

        // ñ = ⌊4x/π⌋
        //  input value  ┃ ñ ┃ substitute
        // ━━━━━━━━━━━━━━╋━━━╋━━━━━━━━━━━━
        // [    0,1/4·π[ ┃ 0 ┃  sin_s(x)
        // [1/4·π,2/4·π[ ┃ 1 ┃  cos_s(x-1/2·π)
        // [2/4·π,3/4·π[ ┃ 2 ┃  cos_s(x-1/2·π)
        // [3/4·π,π    [ ┃ 3 ┃ -sin_s(x-π)
        // [    π,5/4·π[ ┃ 4 ┃ -sin_s(x-π)
        // [5/4·π,6/4·π[ ┃ 5 ┃ -cos_s(x-3/2·π)
        // [6/4·π,7/4·π[ ┃ 6 ┃ -cos_s(x-3/2·π)
        // [7/4·π,2π   [ ┃ 7 ┃  sin_s(x-2π)
        // [   2π,9/4·π[ ┃ 8 ┃  sin_s(x-2π)
        // ...
        // improved n:
        // n = ⌊(4x/π+1)*0.5⌋ = ⌊(ñ + 1)/2⌋
        //   input value   ┃ n ┃    substitute
        // ━━━━━━━━━━━━━━━━╋━━━╋━━━━━━━━━━━━━━━━━
        // [     0, 1/4·π[ ┃ 0 ┃  sin_s(x)
        // [ 1/4·π, 3/4·π[ ┃ 1 ┃  cos_s(x-1/2·π)
        // [ 3/4·π, 5/4·π[ ┃ 2 ┃ -sin_s(x-π)
        // [ 5/4·π, 7/4·π[ ┃ 3 ┃ -cos_s(x-3/2·π)
        // [ 7/4·π, 9/4·π[ ┃ 4 ┃  sin_s(x-2π)
        // [ 9/4·π,11/4·π[ ┃ 5 ┃  cos_s(x-5/2·π)
        // [11/4·π,13/4·π[ ┃ 6 ┃ -sin_s(x-3π)
        // ...

        //V n = floor(x * Vc_buildFloat(1, 0x22f983, 0));
        V n = floor((x * Vc_buildFloat(1, 0x22f983, 0) + V::One()) * 0.5f);
        // c1+c2+c3 = π/2
        // for large n, n should be split into parts,
        // i.e. c1 has 12 significant bits. If n has more than 12 significant bits the multiplication
        // n * c1 may lose information.
        const V c1(Vc_buildFloat(1, 0x490000,   0)); // 1.5703125
        const V c2(Vc_buildFloat(1, 0x7DA000, -12)); // .0004837512969970703125
        const V c3(Vc_buildFloat(1, 0x222169, -24)); // .00000007549790126404332113452255725860595703125

        //const V c1(Vc_buildFloat( 1, 0x490000,   0));
        //const V c2(Vc_buildFloat( 1, 0x7da000, -12));
        //const V c3(Vc_buildFloat( 1, 0x222169, -24));
        if (VC_IS_LIKELY((abs(n) <= 4096.f).isFull())) {
        } else {
            // use another magic value:
            // 5152*π/2 can be split into
            const V cc1(Vc_buildFloat( 1, 0x7CE000,  12)); // 16184
            const V cc2(Vc_buildFloat( 1, 0x3E2000,  -1)); // 1.4853515625
            const V cc3(Vc_buildFloat(-1, 0x0FD1DE, -23)); // -.00000026788524110088474117219448089599609375
            //n / 5152:
            const V nn = round(n * Vc_buildFloat(1, 0x4B8728, -13));
            x = (x - nn * cc1) - nn * cc2;
            correction += nn * cc3;
            n -= nn * T(5152);

            /*
            const V c3(Vc_buildFloat(1, 0x222000, -24)); // .0000000754953362047672271728515625
            const V c4(Vc_buildFloat(1, 0x34611A, -39)); // .00000000000256334406825708960298015881562602636540
            const V nh = n & C::highMask();
            const V nl = n - nh;
            x = (x - nh * c1) - (nl * c1 + nh * c2) - (nl * c2 + nh * c3) - (nl * c3 + nh * c4) - nl * c4;
            */
        }
        x = ((x - n * c1) - n * c2) - (correction + n * c3);
        const V n_2 = floor(n * 0.5f);
        const M evenN = n_2 == n * 0.5f;
        const M negative = (_x < V::Zero()) ^ (floor(n_2 * 0.5f) != n_2 * 0.5f);

        const V xh = x & C::highMask(18);
        const V xl = x - xh;

        const V xh2 = xh * xh;
        const V x2 = x * x;

        const V cc2(-0.5f);
        const V cc4(Vc_buildFloat( 1, 0x2aaa80,  -5)); // 0x2aaaa4
        const V cc6(Vc_buildFloat(-1, 0x360596, -10)); // 0x360596
        const V cc8(Vc_buildFloat( 1, 0x500000, -16)); // 0x4cd525
        //const V cos_s = (((x2 * cc8 + cc6) * x2 + cc4) * x2 + cc2) * x2 + V::One();
        V cos_s = V::One() + xh2 * cc2; // precise
        // remainder: (x² - xh²) * cc2 = xl(x + xh) * cc2
        cos_s += ((x2 * cc8 + cc6) * x2 + cc4) * (x2 * x2) + (xl * cc2) * (x + xh);

        //const V sc3(Vc_buildFloat(-1, 0x2aaaa2,  -3));
        const V sc3h(Vc_buildFloat(-1, 0x2c0000,  -3));
        const V sc3l(Vc_buildFloat( 1, 0x2aa900,  -10)); // 0x2ab376
        const V sc5(Vc_buildFloat( 1, 0x088a00,  -7)); // 0x08838c
        const V sc7(Vc_buildFloat(-1, 0x4ca200, -13)); // 0x4ca140
        // optimize calculation order for instruction count:
        //const V sin_s = ((x2 * sc7 + sc5) * x2 + sc3) * (x2 * x) + x;

        // optimize calculation order for precision:
        //    |x|  ≤ 0.79 ~ 2^-1
        // => |x²| ≤ 0.62 ~ 2^-1
        // => |x³| ≤ 0.48 ~ 2^-2
        // |sin(x) - x| < 0.08 ~ 2^-4
        V sin_s = x + xh2 * (xh * sc3h); // precise
        sin_s += (sc3h * xl) * (x2 + x * xh + xh2) + (sc3l * x) * x2 + ((x2 * sc7 + sc5) * x2) * (x2 * x);

        x = cos_s;
        x(evenN) = sin_s;
        x(negative) = -x;
        return x;
    }
    template<typename T> static inline Vector<T> cos(const Vector<T> &_x) {
        typedef Vector<T> V;
        typedef Const<T> C;

        V x = _foldMinusPiToPi(_x) + C::_pi_2(); // [-½π, ¾π[
        x(x > C::_pi_2()) = C::_pi() - x; // [-½π, ½π]

        const V &x2 = x * x;
        return x * (V::One() - x2 * (C::_1_3fac() - x2 * (C::_1_5fac() - x2 * (C::_1_7fac() - x2 * C::_1_9fac()))));
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
