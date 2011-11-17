/*  This file is part of the Vc library.

    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_SSE_MATH_H
#define VC_SSE_MATH_H

#include "const.h"

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
namespace SSE
{
    template<typename T> inline Vector<T> c_sin<T>::_1_2pi()  { return Vector<T>(&_data[0 * Size]); }
    template<typename T> inline Vector<T> c_sin<T>::_2pi()    { return Vector<T>(&_data[1 * Size]); }
    template<typename T> inline Vector<T> c_sin<T>::_pi_2()   { return Vector<T>(&_data[2 * Size]); }
    template<typename T> inline Vector<T> c_sin<T>::_pi()     { return Vector<T>(&_data[3 * Size]); }

    template<typename T> inline Vector<T> c_sin<T>::_1_3fac() { return Vector<T>(&_data[4 * Size]); }
    template<typename T> inline Vector<T> c_sin<T>::_1_5fac() { return Vector<T>(&_data[5 * Size]); }
    template<typename T> inline Vector<T> c_sin<T>::_1_7fac() { return Vector<T>(&_data[6 * Size]); }
    template<typename T> inline Vector<T> c_sin<T>::_1_9fac() { return Vector<T>(&_data[7 * Size]); }

    template<> inline Vector<float8> c_sin<float8>::_1_2pi()  { return Vector<float8>::broadcast4(&c_sin<float>::_data[ 0]); }
    template<> inline Vector<float8> c_sin<float8>::_2pi()    { return Vector<float8>::broadcast4(&c_sin<float>::_data[ 4]); }
    template<> inline Vector<float8> c_sin<float8>::_pi_2()   { return Vector<float8>::broadcast4(&c_sin<float>::_data[ 8]); }
    template<> inline Vector<float8> c_sin<float8>::_pi()     { return Vector<float8>::broadcast4(&c_sin<float>::_data[12]); }

    template<> inline Vector<float8> c_sin<float8>::_1_3fac() { return Vector<float8>::broadcast4(&c_sin<float>::_data[16]); }
    template<> inline Vector<float8> c_sin<float8>::_1_5fac() { return Vector<float8>::broadcast4(&c_sin<float>::_data[20]); }
    template<> inline Vector<float8> c_sin<float8>::_1_7fac() { return Vector<float8>::broadcast4(&c_sin<float>::_data[24]); }
    template<> inline Vector<float8> c_sin<float8>::_1_9fac() { return Vector<float8>::broadcast4(&c_sin<float>::_data[28]); }

    template<typename T> static inline Vector<T> sin(const Vector<T> &_x) {
        typedef Vector<T> V;
        typedef typename V::Mask M;
        typedef c_sin<T> C;
        using namespace VectorSpecialInitializerOne;

        // x - x**3/3! + x**5/5! - x**7/7! + x**9/9! - x**11/11! for [-pi/2:pi/2]

        V x = _x - round(_x * C::_1_2pi()) * C::_2pi();
        const M &gt_pi_2 = x >  C::_pi_2();
        const M &lt_pi_2 = x < -C::_pi_2();
        const V &foldRight =  C::_pi() - x;
        const V &foldLeft  = -C::_pi() - x;
        x(gt_pi_2) = foldRight;
        x(lt_pi_2) = foldLeft;

        const V &x2 = x * x;
        return x * (V(One) - x2 * (C::_1_3fac() - x2 * (C::_1_5fac() - x2 * (C::_1_7fac() - x2 * C::_1_9fac()))));
    }
    template<typename T> static inline Vector<T> cos(const Vector<T> &_x) {
        typedef Vector<T> V;
        typedef c_sin<T> C;
        using namespace VectorSpecialInitializerOne;

        V x = _x - round(_x * C::_1_2pi()) * C::_2pi() + C::_pi_2();
        x(x > C::_pi_2()) = C::_pi() - x;

        const V &x2 = x * x;
        return x * (V(One) - x2 * (C::_1_3fac() - x2 * (C::_1_5fac() - x2 * (C::_1_7fac() - x2 * C::_1_9fac()))));
    }
    template<typename _T> static inline Vector<_T> asin (const Vector<_T> &_x) {
        typedef Vector<_T> V;
        typedef typename V::EntryType T;
        typedef typename V::Mask M;
        using namespace VectorSpecialInitializerZero;
        using namespace VectorSpecialInitializerOne;

        const V pi_2(T(M_PI_2));
        const M &negative = _x < V(Zero);

        const V &a = abs(_x);
        //const M &outOfRange = a > V(One);
        const M &small = a < V(T(1.e-4));
        const M &gt_0_5 = a > V(T(0.5));
        V x = a;
        V z = a * a;
        z(gt_0_5) = (V(One) - a) * V(T(0.5));
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
        using namespace VectorSpecialInitializerZero;
        using namespace VectorSpecialInitializerOne;
        V x = abs(_x);
        const V pi_2(T(M_PI_2));
        const V pi_4(T(M_PI_4));
        const M &gt_tan_3pi_8 = x > V(T(2.414213562373095));
        const M &gt_tan_pi_8  = x > V(T(0.4142135623730950)) && !gt_tan_3pi_8;
        const V minusOne(-1);
        V y(Zero);
        y(gt_tan_3pi_8) = pi_2;
        y(gt_tan_pi_8)  = pi_4;
        x(gt_tan_3pi_8) = minusOne / x;
        x(gt_tan_pi_8)  = (x - V(One)) / (x + V(One));
        const V &x2 = x * x;
        y += (((T(8.05374449538e-2) * x2
              - T(1.38776856032E-1)) * x2
              + T(1.99777106478E-1)) * x2
              - T(3.33329491539E-1)) * x2 * x
              + x;
        y(_x < V(Zero)) = -y;
        return y;
    }
    template<typename _T> static inline Vector<_T> atan2(const Vector<_T> &y, const Vector<_T> &x) {
        typedef Vector<_T> V;
        typedef typename V::EntryType T;
        typedef typename V::Mask M;
        using namespace VectorSpecialInitializerZero;
        const V pi(T(M_PI));
        const V pi_2(T(M_PI_2));

        const M &xZero = x == V(Zero);
        const M &yZero = y == V(Zero);
        const M &xNeg = x < V(Zero);
        const M &yNeg = y < V(Zero);

        const V &absX = abs(x);
        const V &absY = abs(y);

        V a = absY / absX;
        const V pi_4(T(M_PI_4));
        const M &gt_tan_3pi_8 = a > V(T(2.414213562373095));
        const M &gt_tan_pi_8  = a > V(T(0.4142135623730950)) && !gt_tan_3pi_8;
        const V minusOne(-1);
        V b(Zero);
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
} // namespace SSE
} // namespace Vc

#endif // VC_SSE_MATH_H
