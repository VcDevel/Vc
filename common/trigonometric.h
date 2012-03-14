/*  This file is part of the Vc library.

    Copyright (C) 2009-2011 Matthias Kretz <kretz@kde.org>

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
#include "macros.h"

#if !defined M_PI
# define M_PI 3.14159265358979323846
#endif
#if !defined M_PI_2
# define M_PI_2 1.57079632679489661923
#endif
#if !defined M_PI_4
# define M_PI_4 0.785398163397448309616
#endif

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

    template<typename T> static inline Vector<T> sin(const Vector<T> &_x) {
        typedef Vector<T> V;
        typedef Const<T> C;

        // x - x**3/3! + x**5/5! - x**7/7! + x**9/9! - x**11/11! for [-½π, ½π]

        V x = _foldMinusPiToPi(_x); // [-π, π[
        // fold the left and right fourths in to reduce the range to [-½π, ½π]
        x(x >  C::_pi_2()) =  C::_pi() - x;
        x(x < -C::_pi_2()) = -C::_pi() - x;

        const V &x2 = x * x;
        return x * (V::One() - x2 * (C::_1_3fac() - x2 * (C::_1_5fac() - x2 * (C::_1_7fac() - x2 * C::_1_9fac()))));
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

        const V pi_2(T(M_PI_2));
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
        const V pi_2(T(M_PI_2));
        const V pi_4(T(M_PI_4));
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
        const V pi(T(M_PI));
        const V pi_2(T(M_PI_2));

        const M &xZero = x == V::Zero();
        const M &yZero = y == V::Zero();
        const M &xNeg = x < V::Zero();
        const M &yNeg = y < V::Zero();

        const V &absX = abs(x);
        const V &absY = abs(y);

        V a = absY / absX;
        const V pi_4(T(M_PI_4));
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
