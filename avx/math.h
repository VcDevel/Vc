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
} // namespace AVX
} // namespace Vc

#include "undomacros.h"
#define VC__USE_NAMESPACE AVX
#include "../common/trigonometric.h"
#define VC__USE_NAMESPACE AVX
#include "../common/logarithm.h"

#endif // VC_AVX_MATH_H
