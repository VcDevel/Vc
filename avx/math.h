/*  This file is part of the Vc library.

    Copyright (C) 2009-2010 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_AVX_MATH_H
#define VC_AVX_MATH_H

#include "const.h"
#include "limits.h"
#include "macros.h"

namespace Vc
{
namespace AVX
{
    template<typename T> inline Vector<T> c_sin<T>::_1_2pi()  { return Vector<T>(_data[0]); }
    template<typename T> inline Vector<T> c_sin<T>::_2pi()    { return Vector<T>(_data[1]); }
    template<typename T> inline Vector<T> c_sin<T>::_pi_2()   { return Vector<T>(_data[2]); }
    template<typename T> inline Vector<T> c_sin<T>::_pi()     { return Vector<T>(_data[3]); }

    template<typename T> inline Vector<T> c_sin<T>::_1_3fac() { return Vector<T>(_data[4]); }
    template<typename T> inline Vector<T> c_sin<T>::_1_5fac() { return Vector<T>(_data[5]); }
    template<typename T> inline Vector<T> c_sin<T>::_1_7fac() { return Vector<T>(_data[6]); }
    template<typename T> inline Vector<T> c_sin<T>::_1_9fac() { return Vector<T>(_data[7]); }

    class M128iDummy
    {
        __m128i d;
        public:
            M128iDummy(__m128i dd) : d(dd) {}
            __m128i &operator=(__m128i dd) { d = dd; return d; }
            operator __m128i &() { return d; }
            operator __m128i () const { return d; }
    };

    template<typename T, typename M> inline M128iDummy c_log<T, M>::bias() { return avx_cast<__m128i>(_mm_broadcast_ss(f(0))); }

    typedef Vector<double> double_v;
    typedef Vector<float> float_v;
    typedef Vector<double>::Mask double_m;
    typedef Vector<float >::Mask float_m;

    template<> inline double_m c_log<double, double_m>::exponentMask() { return _mm256_broadcast_sd(d(1)); }
    template<> inline double_v c_log<double, double_m>::_1_2()         { return _mm256_broadcast_sd(&_dataT[3]); }
    template<> inline double_v c_log<double, double_m>::_1_sqrt2()     { return _mm256_broadcast_sd(&_dataT[0]); }
    template<> inline double_v c_log<double, double_m>::P(int i)       { return _mm256_broadcast_sd(d(2 + i)); }
    template<> inline double_v c_log<double, double_m>::Q(int i)       { return _mm256_broadcast_sd(d(8 + i)); }
    template<> inline double_v c_log<double, double_m>::min()          { return _mm256_broadcast_sd(d(14)); }
    template<> inline double_v c_log<double, double_m>::ln2_small()    { return _mm256_broadcast_sd(&_dataT[1]); }
    template<> inline double_v c_log<double, double_m>::ln2_large()    { return _mm256_broadcast_sd(&_dataT[2]); }
    template<> inline double_v c_log<double, double_m>::neginf()       { return _mm256_broadcast_sd(d(13)); }
    template<> inline double_v c_log<double, double_m>::log10_e()      { return _mm256_broadcast_sd(&_dataT[4]); }
    template<> inline double_v c_log<double, double_m>::log2_e()       { return _mm256_broadcast_sd(&_dataT[5]); }
    template<> inline float_m c_log<float, float_m>::exponentMask() { return _mm256_broadcast_ss(f(1)); }
    template<> inline float_v c_log<float, float_m>::_1_2()         { return _mm256_broadcast_ss(&_dataT[3]); }
    template<> inline float_v c_log<float, float_m>::_1_sqrt2()     { return _mm256_broadcast_ss(&_dataT[0]); }
    template<> inline float_v c_log<float, float_m>::P(int i)       { return _mm256_broadcast_ss(f(2 + i)); }
    template<> inline float_v c_log<float, float_m>::Q(int i)       { return _mm256_broadcast_ss(f(8 + i)); }
    template<> inline float_v c_log<float, float_m>::min()          { return _mm256_broadcast_ss(f(14)); }
    template<> inline float_v c_log<float, float_m>::ln2_small()    { return _mm256_broadcast_ss(&_dataT[1]); }
    template<> inline float_v c_log<float, float_m>::ln2_large()    { return _mm256_broadcast_ss(&_dataT[2]); }
    template<> inline float_v c_log<float, float_m>::neginf()       { return _mm256_broadcast_ss(f(13)); }
    template<> inline float_v c_log<float, float_m>::log10_e()      { return _mm256_broadcast_ss(&_dataT[4]); }
    template<> inline float_v c_log<float, float_m>::log2_e()       { return _mm256_broadcast_ss(&_dataT[5]); }

    template<typename T> static inline Vector<T> sin(const Vector<T> &_x) {
        typedef Vector<T> V;
        typedef c_sin<T> C;

        // x - x**3/3! + x**5/5! - x**7/7! + x**9/9! - x**11/11! for [-pi/2:pi/2]

        V x = _x - round(_x * C::_1_2pi()) * C::_2pi();
        x(x >  C::_pi_2()) =  C::_pi() - x;
        x(x < -C::_pi_2()) = -C::_pi() - x;

        const V &x2 = x * x;
        return x * (V::One() - x2 * (C::_1_3fac() - x2 * (C::_1_5fac() - x2 * (C::_1_7fac() - x2 * C::_1_9fac()))));
    }
    template<typename T> static inline Vector<T> cos(const Vector<T> &_x) {
        typedef Vector<T> V;
        typedef c_sin<T> C;

        V x = _x - round(_x * C::_1_2pi()) * C::_2pi() + C::_pi_2();
        x(x > C::_pi_2()) = C::_pi() - x;

        const V &x2 = x * x;
        return x * (V::One() - x2 * (C::_1_3fac() - x2 * (C::_1_5fac() - x2 * (C::_1_7fac() - x2 * C::_1_9fac()))));
    }
    template<typename T> static inline Vector<T> asin (const Vector<T> &_x) {
        typedef Vector<T> V;
        typedef typename V::Mask M;
        using namespace VectorSpecialInitializerZero;
        using namespace VectorSpecialInitializerOne;

        const V pi_2(M_PI / 2);
        const M &negative = _x < V(Zero);

        const V &a = abs(_x);
        //const M &outOfRange = a > V(One);
        const M &small = a < V(1.e-4);
        const M &gt_0_5 = a > V(0.5);
        V x = a;
        V z = a * a;
        z(gt_0_5) = (V(One) - a) * V(0.5);
        x(gt_0_5) = sqrt(z);
        z = ((((4.2163199048e-2  * z
              + 2.4181311049e-2) * z
              + 4.5470025998e-2) * z
              + 7.4953002686e-2) * z
              + 1.6666752422e-1) * z * x
              + x;
        z(gt_0_5) = pi_2 - (z + z);
        z(small) = a;
        z(negative) = -z;
        //z(outOfRange) = nan;

        return z;
    }
    template<typename T> static inline Vector<T> atan (const Vector<T> &_x) {
        typedef Vector<T> V;
        typedef typename V::Mask M;
        using namespace VectorSpecialInitializerZero;
        using namespace VectorSpecialInitializerOne;
        V x = abs(_x);
        const V pi_2(M_PI / 2);
        const V pi_4(M_PI / 4);
        const M &gt_tan_3pi_8 = x > V(2.414213562373095);
        const M &gt_tan_pi_8  = x > V(0.4142135623730950) && !gt_tan_3pi_8;
        const V minusOne(-1);
        V y(Zero);
        y(gt_tan_3pi_8) = pi_2;
        y(gt_tan_pi_8)  = pi_4;
        x(gt_tan_3pi_8) = minusOne / x;
        x(gt_tan_pi_8)  = (x - V(One)) / (x + V(One));
        const V &x2 = x * x;
        y += (((8.05374449538e-2 * x2
              - 1.38776856032E-1) * x2
              + 1.99777106478E-1) * x2
              - 3.33329491539E-1) * x2 * x
              + x;
        y(_x < V(Zero)) = -y;
        return y;
    }
    template<typename T> static inline Vector<T> atan2(const Vector<T> &y, const Vector<T> &x) {
        typedef Vector<T> V;
        typedef typename V::Mask M;
        using namespace VectorSpecialInitializerZero;
        const V pi(M_PI);
        const V pi_2(M_PI / 2);

        const M &xZero = x == V(Zero);
        const M &yZero = y == V(Zero);
        const M &xNeg = x < V(Zero);
        const M &yNeg = y < V(Zero);

        const V &absX = abs(x);
        const V &absY = abs(y);

        V a = absY / absX;
        const V pi_4(M_PI / 4);
        const M &gt_tan_3pi_8 = a > V(2.414213562373095);
        const M &gt_tan_pi_8  = a > V(0.4142135623730950) && !gt_tan_3pi_8;
        const V minusOne(-1);
        V b(Zero);
        b(gt_tan_3pi_8) = pi_2;
        b(gt_tan_pi_8)  = pi_4;
        a(gt_tan_3pi_8) = minusOne / a;
        a(gt_tan_pi_8)  = (absY - absX) / (absY + absX);
        const V &a2 = a * a;
        b += (((8.05374449538e-2 * a2
              - 1.38776856032E-1) * a2
              + 1.99777106478E-1) * a2
              - 3.33329491539E-1) * a2 * a
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
    inline __m256d INTRINSIC CONST extractExponent(__m256d x) {
        typedef c_log<double, double_m> C;
        __m128i emm0lo = _mm_srli_epi64(avx_cast<__m128i>(x), 52);
        __m128i emm0hi = _mm_srli_epi64(avx_cast<__m128i>(hi128(x)), 52);

        emm0lo = _mm_sub_epi32(emm0lo, C::bias());
        emm0hi = _mm_sub_epi32(emm0hi, C::bias());

        return _mm256_cvtepi32_pd(avx_cast<__m128i>(shuffle<X0, X2, Y0, Y2>(avx_cast<__m128>(emm0lo), avx_cast<__m128>(emm0hi))));
    }
    inline __m256 INTRINSIC CONST extractExponent(__m256 x) {
        typedef c_log<float, float_m> C;
        __m128i emm0lo = _mm_srli_epi32(avx_cast<__m128i>(x), 23);
        __m128i emm0hi = _mm_srli_epi32(avx_cast<__m128i>(hi128(x)), 23);

        emm0lo = _mm_sub_epi32(emm0lo, C::bias());
        emm0hi = _mm_sub_epi32(emm0hi, C::bias());

        return _mm256_cvtepi32_ps(concat(emm0lo, emm0hi));
    }
    inline double_v INTRINSIC CONST _or(double_v a, double_v b) {
        return _mm256_or_pd(a.data(), b.data());
    }
    inline float_v INTRINSIC CONST _or(float_v a, float_v b) {
        return _mm256_or_ps(a.data(), b.data());
    }
    template<typename T> static inline Vector<T> log(Vector<T> x) {
        typedef Vector<T> V;
        typedef typename V::Mask M;
        typedef c_log<T, M> C;

        const M invalidMask = x < V::Zero();
        const M infinityMask = x == V::Zero();

        x = max(x, C::min()); // lazy: cut off denormalized numbers

        V exponent = extractExponent(x.data());

        x.setZero(C::exponentMask()); // keep only the fractional part
        x = _or(x, C::_1_2());        // and set the exponent to 2⁻¹
        // => x ∈ [0.5, 1[

        const M smallX = x < C::_1_sqrt2();
        x(smallX) += x; // => x ∈ [1/√2,     1[ ∪ [1.5, 1 + 1/√2[
        x -= V::One();  // => x ∈ [1/√2 - 1, 0[ ∪ [0.5, 1/√2[
        exponent(!smallX) += V::One();

        const V x2 = x * x;
        V y = C::P(0);
        V y2 = C::Q(0) + x;
        unrolled_loop16(i, 1, 5,
                y = y * x + C::P(i);
                y2 = y2 * x + C::Q(i);
                );
        y2 = x / y2;
        y = y * x + C::P(5);
        y = x2 * y * y2 + exponent * C::ln2_small() - x2 * C::_1_2();
        x += y;
        x += exponent * C::ln2_large();

        x.setQnan(invalidMask);
        x(infinityMask) = C::neginf();

        return x;
    }
    template<> inline Vector<float> log(Vector<float> x) {
        typedef Vector<float> V;
        typedef V::Mask M;
        typedef c_log<float, M> C;

        const M invalidMask = x < V::Zero();
        const M infinityMask = x == V::Zero();

        x = max(x, C::min()); // lazy: cut off denormalized numbers

        V exponent = extractExponent(x.data());

        x.setZero(C::exponentMask()); // keep only the fractional part
        x = _or(x, C::_1_2());        // and set the exponent to 2⁻¹
        // => x ∈ [0.5, 1[

        const M smallX = x < C::_1_sqrt2();
        x(smallX) += x; // => x ∈ [1/√2,     1[ ∪ [1.5, 1 + 1/√2[
        x -= V::One();  // => x ∈ [1/√2 - 1, 0[ ∪ [0.5, 1/√2[
        exponent(!smallX) += V::One();

        const V x2 = x * x;
        V y = C::P(0);
        unrolled_loop16(i, 1, 9,
                y = y * x + C::P(i);
                );
        y *= x * x2;
        y += exponent * C::ln2_small();
        y -= x2 * C::_1_2();
        x += y;
        x += exponent * C::ln2_large();

        x.setQnan(invalidMask);
        x(infinityMask) = C::neginf();

        return x;
    }
    template<typename T> static inline Vector<T> log10(Vector<T> x) {
        typedef Vector<T> V;
        typedef typename V::Mask M;
        typedef c_log<T, M> C;

        return log(x) * C::log10_e();
    }
    template<typename T> static inline Vector<T> log2(Vector<T> x) {
        typedef Vector<T> V;
        typedef typename V::Mask M;
        typedef c_log<T, M> C;

        return log(x) * C::log2_e();
    }
} // namespace AVX
} // namespace Vc

#include "undomacros.h"

#endif // VC_AVX_MATH_H
