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
    typedef Vector<int> int_v;
    typedef Vector<short> short_v;
    typedef Vector<double>::Mask double_m;
    typedef Vector<float >::Mask float_m;
    typedef int_v::Mask int_m;
    typedef short_v::Mask short_m;

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
    ///////////////////////////////////////////////////////////////////////////

    /**
     * splits \p v into exponent and mantissa, the sign is kept with the mantissa
     *
     * The return value will be in the range [0.5, 1.0[
     * The \p e value will be an integer defining the power-of-two exponent
     */
    inline double_v frexp(const double_v &v, int_v *e) {
        const __m256d exponentBits = c_log<double, double_m>::exponentMask().dataD();
        const __m256d exponentPart = _mm256_and_pd(v.data(), exponentBits);
        e->data() = _mm256_sub_epi32(_mm256_srli_epi64(avx_cast<__m256i>(exponentPart), 52), _mm256_set1_epi32(0x3fe));
        const __m256d exponentMaximized = _mm256_or_pd(v.data(), exponentBits);
        double_v ret = _mm256_and_pd(exponentMaximized, _mm256_broadcast_sd(reinterpret_cast<const double *>(&c_general::frexpMask)));
        double_m zeroMask = v == double_v::Zero();
        ret(isnan(v) || !isfinite(v) || zeroMask) = v;
        e->setZero(zeroMask.data());
        return ret;
    }
    inline float_v frexp(const float_v &v, int_v *e) {
        const __m256 exponentBits = c_log<float, float_m>::exponentMask().data();
        const __m256 exponentPart = _mm256_and_ps(v.data(), exponentBits);
        e->data() = _mm256_sub_epi32(_mm256_srli_epi32(avx_cast<__m256i>(exponentPart), 23), _mm256_set1_epi32(0x7e));
        const __m256 exponentMaximized = _mm256_or_ps(v.data(), exponentBits);
        float_v ret = _mm256_and_ps(exponentMaximized, avx_cast<__m256>(_mm256_set1_epi32(0xbf7fffffu)));
        ret(isnan(v) || !isfinite(v) || v == float_v::Zero()) = v;
        e->setZero(v == float_v::Zero());
        return ret;
    }
    inline float_v frexp(const float_v &v, short_v *e) {
        const __m256 exponentBits = c_log<float, float_m>::exponentMask().data();
        const __m256 exponentPart = _mm256_and_ps(v.data(), exponentBits);
        e->data() = _mm_sub_epi16(_mm_packs_epi32(_mm_srli_epi32(avx_cast<__m128i>(exponentPart), 23),
                    _mm_srli_epi32(avx_cast<__m128i>(hi128(exponentPart)), 23)), _mm_set1_epi16(0x7e));
        const __m256 exponentMaximized = _mm256_or_ps(v.data(), exponentBits);
        float_v ret = _mm256_and_ps(exponentMaximized, avx_cast<__m256>(_mm256_set1_epi32(0xbf7fffffu)));
        ret(isnan(v) || !isfinite(v) || v == float_v::Zero()) = v;
        e->setZero(v == float_v::Zero());
        return ret;
    }

    /*             -> x * 2^e
     * x == NaN    -> NaN
     * x == (-)inf -> (-)inf
     */
    inline double_v ldexp(double_v v, int_v e) {
        e.setZero((v == double_v::Zero()).dataI());
        const __m256i exponentBits = _mm256_slli_epi64(e.data(), 52);
        return avx_cast<__m256d>(_mm256_add_epi64(avx_cast<__m256i>(v.data()), exponentBits));
    }
    inline float_v ldexp(float_v v, int_v e) {
        e.setZero(static_cast<int_m>(v == float_v::Zero()));
        return (v.reinterpretCast<int_v>() + (e << 23)).reinterpretCast<float_v>();
    }
    inline float_v ldexp(float_v v, short_v e) {
        e.setZero(static_cast<short_m>(v == float_v::Zero()));
        e = e << (23 - 16);
        const __m256i exponentBits = concat(_mm_unpacklo_epi16(_mm_setzero_si128(), e.data()),
                _mm_unpackhi_epi16(_mm_setzero_si128(), e.data()));
        return (v.reinterpretCast<int_v>() + exponentBits).reinterpretCast<float_v>();
    }
} // namespace AVX
} // namespace Vc

#include "undomacros.h"
#define VC__USE_NAMESPACE AVX
#include "../common/trigonometric.h"
#define VC__USE_NAMESPACE AVX
#include "../common/logarithm.h"

#endif // VC_AVX_MATH_H
