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

namespace Vc
{
namespace SSE
{
    ///////////////////////////////////////////////////////////////////////////
    // helpers for all the required constants
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

    class M128iDummy
    {
        __m128i d;
        public:
            M128iDummy(__m128i dd) : d(dd) {}
            __m128i &operator=(__m128i dd) { d = dd; return d; }
            operator __m128i &() { return d; }
            operator __m128i () const { return d; }
    };
    template<typename T, typename M> inline M128iDummy c_log<T, M>::bias() { return _mm_load_si128(reinterpret_cast<const __m128i *>(&_dataI[0])); }

    typedef Vector<double> double_v;
    typedef Vector<float> float_v;
    typedef Vector<float8> sfloat_v;
    typedef Vector<int> int_v;
    typedef Vector<short> short_v;
    typedef Vector<double>::Mask double_m;
    typedef Vector<float >::Mask float_m;
    typedef Vector<float8>::Mask sfloat_m;
    typedef int_v::Mask int_m;
    typedef short_v::Mask short_m;

    template<> inline double_m c_log<double, double_m>::exponentMask() { return _mm_load_pd(d(1)); }
    template<> inline double_v c_log<double, double_m>::_1_2()         { return _mm_load_pd(&_dataT[6]); }
    template<> inline double_v c_log<double, double_m>::_1_sqrt2()     { return _mm_load_pd(&_dataT[0]); }
    template<> inline double_v c_log<double, double_m>::P(int i)       { return _mm_load_pd(d(2 + i)); }
    template<> inline double_v c_log<double, double_m>::Q(int i)       { return _mm_load_pd(d(8 + i)); }
    template<> inline double_v c_log<double, double_m>::min()          { return _mm_load_pd(d(14)); }
    template<> inline double_v c_log<double, double_m>::ln2_small()    { return _mm_load_pd(&_dataT[2]); }
    template<> inline double_v c_log<double, double_m>::ln2_large()    { return _mm_load_pd(&_dataT[4]); }
    template<> inline double_v c_log<double, double_m>::neginf()       { return _mm_load_pd(d(13)); }
    template<> inline double_v c_log<double, double_m>::log10_e()      { return _mm_load_pd(&_dataT[8]); }
    template<> inline double_v c_log<double, double_m>::log2_e()       { return _mm_load_pd(&_dataT[10]); }
    template<> inline float_m c_log<float, float_m>::exponentMask() { return _mm_load_ps(f(1)); }
    template<> inline float_v c_log<float, float_m>::_1_2()         { return _mm_load_ps(&_dataT[12]); }
    template<> inline float_v c_log<float, float_m>::_1_sqrt2()     { return _mm_load_ps(&_dataT[0]); }
    template<> inline float_v c_log<float, float_m>::P(int i)       { return _mm_load_ps(f(2 + i)); }
    template<> inline float_v c_log<float, float_m>::Q(int i)       { return _mm_load_ps(f(8 + i)); }
    template<> inline float_v c_log<float, float_m>::min()          { return _mm_load_ps(f(14)); }
    template<> inline float_v c_log<float, float_m>::ln2_small()    { return _mm_load_ps(&_dataT[4]); }
    template<> inline float_v c_log<float, float_m>::ln2_large()    { return _mm_load_ps(&_dataT[8]); }
    template<> inline float_v c_log<float, float_m>::neginf()       { return _mm_load_ps(f(13)); }
    template<> inline float_v c_log<float, float_m>::log10_e()      { return _mm_load_ps(&_dataT[16]); }
    template<> inline float_v c_log<float, float_m>::log2_e()       { return _mm_load_ps(&_dataT[20]); }

    template<> inline sfloat_m c_log<float8, sfloat_m>::exponentMask() { return M256::dup(c_log<float, float_m>::exponentMask().data()); }
    template<> inline sfloat_v c_log<float8, sfloat_m>::_1_2()         { return M256::dup(c_log<float, float_m>::_1_2().data()); }
    template<> inline sfloat_v c_log<float8, sfloat_m>::_1_sqrt2()     { return M256::dup(c_log<float, float_m>::_1_sqrt2().data()); }
    template<> inline sfloat_v c_log<float8, sfloat_m>::P(int i)       { return M256::dup(c_log<float, float_m>::P(i).data()); }
    template<> inline sfloat_v c_log<float8, sfloat_m>::Q(int i)       { return M256::dup(c_log<float, float_m>::Q(i).data()); }
    template<> inline sfloat_v c_log<float8, sfloat_m>::min()          { return M256::dup(c_log<float, float_m>::min().data()); }
    template<> inline sfloat_v c_log<float8, sfloat_m>::ln2_small()    { return M256::dup(c_log<float, float_m>::ln2_small().data()); }
    template<> inline sfloat_v c_log<float8, sfloat_m>::ln2_large()    { return M256::dup(c_log<float, float_m>::ln2_large().data()); }
    template<> inline sfloat_v c_log<float8, sfloat_m>::neginf()       { return M256::dup(c_log<float, float_m>::neginf().data()); }
    template<> inline sfloat_v c_log<float8, sfloat_m>::log10_e()      { return M256::dup(c_log<float, float_m>::log10_e().data()); }
    template<> inline sfloat_v c_log<float8, sfloat_m>::log2_e()       { return M256::dup(c_log<float, float_m>::log2_e().data()); }
    ///////////////////////////////////////////////////////////////////////////

    /**
     * splits \p v into exponent and mantissa, the sign is kept with the mantissa
     *
     * The return value will be in the range [0.5, 1.0[
     * The \p e value will be an integer defining the power-of-two exponent
     */
    inline double_v frexp(const double_v &v, int_v *e) {
        const __m128i exponentBits = _mm_load_si128(reinterpret_cast<const __m128i *>(&c_log<double, double_m>::_dataI[2]));
        const __m128i exponentPart = _mm_and_si128(_mm_castpd_si128(v.data()), exponentBits);
        *e = _mm_sub_epi32(_mm_srli_epi64(exponentPart, 52), _mm_set1_epi32(0x3fe));
        const __m128d exponentMaximized = _mm_or_pd(v.data(), _mm_castsi128_pd(exponentBits));
        double_v ret = _mm_and_pd(exponentMaximized, _mm_load_pd(reinterpret_cast<const double *>(&c_general::frexpMask[0])));
        double_m zeroMask = v == double_v::Zero();
        ret(isnan(v) || !isfinite(v) || zeroMask) = v;
        e->setZero(zeroMask.data());
        return ret;
    }
    inline float_v frexp(const float_v &v, int_v *e) {
        const __m128i exponentBits = _mm_load_si128(reinterpret_cast<const __m128i *>(&c_log<float, float_m>::_dataI[4]));
        const __m128i exponentPart = _mm_and_si128(_mm_castps_si128(v.data()), exponentBits);
        *e = _mm_sub_epi32(_mm_srli_epi32(exponentPart, 23), _mm_set1_epi32(0x7e));
        const __m128 exponentMaximized = _mm_or_ps(v.data(), _mm_castsi128_ps(exponentBits));
        float_v ret = _mm_and_ps(exponentMaximized, _mm_castsi128_ps(_mm_set1_epi32(0xbf7fffffu)));
        ret(isnan(v) || !isfinite(v) || v == float_v::Zero()) = v;
        e->setZero(v == float_v::Zero());
        return ret;
    }
    inline sfloat_v frexp(const sfloat_v &v, short_v *e) {
        const __m128i exponentBits = _mm_load_si128(reinterpret_cast<const __m128i *>(&c_log<float, float_m>::_dataI[4]));
        const __m128i exponentPart0 = _mm_and_si128(_mm_castps_si128(v.data()[0]), exponentBits);
        const __m128i exponentPart1 = _mm_and_si128(_mm_castps_si128(v.data()[1]), exponentBits);
        *e = _mm_sub_epi16(_mm_packs_epi32(_mm_srli_epi32(exponentPart0, 23), _mm_srli_epi32(exponentPart1, 23)),
                _mm_set1_epi16(0x7e));
        const __m128 exponentMaximized0 = _mm_or_ps(v.data()[0], _mm_castsi128_ps(exponentBits));
        const __m128 exponentMaximized1 = _mm_or_ps(v.data()[1], _mm_castsi128_ps(exponentBits));
        sfloat_v ret = M256::create(
                _mm_and_ps(exponentMaximized0, _mm_castsi128_ps(_mm_set1_epi32(0xbf7fffffu))),
                _mm_and_ps(exponentMaximized1, _mm_castsi128_ps(_mm_set1_epi32(0xbf7fffffu)))
                );
        sfloat_m zeroMask = v == sfloat_v::Zero();
        ret(isnan(v) || !isfinite(v) || zeroMask) = v;
        e->setZero(static_cast<short_m>(zeroMask));
        return ret;
    }

    /*             -> x * 2^e
     * x == NaN    -> NaN
     * x == (-)inf -> (-)inf
     */
    inline double_v ldexp(double_v v, int_v e) {
        e.setZero((v == double_v::Zero()).dataI());
        const __m128i exponentBits = _mm_slli_epi64(e.data(), 52);
        return _mm_castsi128_pd(_mm_add_epi64(_mm_castpd_si128(v.data()), exponentBits));
    }
    inline float_v ldexp(float_v v, int_v e) {
        e.setZero(static_cast<int_m>(v == float_v::Zero()));
        return (v.reinterpretCast<int_v>() + (e << 23)).reinterpretCast<float_v>();
    }
    inline sfloat_v ldexp(sfloat_v v, short_v e) {
        e.setZero(static_cast<short_m>(v == sfloat_v::Zero()));
        e <<= (23 - 16);
        const __m128i exponentBits0 = _mm_unpacklo_epi16(_mm_setzero_si128(), e.data());
        const __m128i exponentBits1 = _mm_unpackhi_epi16(_mm_setzero_si128(), e.data());
        return M256::create(_mm_castsi128_ps(_mm_add_epi32(_mm_castps_si128(v.data()[0]), exponentBits0)),
                _mm_castsi128_ps(_mm_add_epi32(_mm_castps_si128(v.data()[1]), exponentBits1)));
    }

#ifdef VC_IMPL_SSE4_1
    inline double_v floor(double_v v) { return _mm_floor_pd(v.data()); }
    inline float_v floor(float_v v) { return _mm_floor_ps(v.data()); }
    inline sfloat_v floor(sfloat_v v) { return M256::create(_mm_floor_ps(v.data()[0]),
            _mm_floor_ps(v.data()[1])); }
    inline double_v ceil(double_v v) { return _mm_ceil_pd(v.data()); }
    inline float_v ceil(float_v v) { return _mm_ceil_ps(v.data()); }
    inline sfloat_v ceil(sfloat_v v) { return M256::create(_mm_ceil_ps(v.data()[0]),
            _mm_ceil_ps(v.data()[1])); }
#else
    static inline void floor_shift(float_v &v, float_v e)
    {
        int_v x = _mm_setallone_si128();
        x <<= 23;
        x >>= static_cast<int_v>(e);
        v &= x.reinterpretCast<float_v>();
    }

    static inline void floor_shift(sfloat_v &v, sfloat_v e)
    {
        int_v x = _mm_setallone_si128();
        x <<= 23;
        int_v y = x;
        x >>= _mm_cvttps_epi32(e.data()[0]);
        y >>= _mm_cvttps_epi32(e.data()[1]);
        v.data()[0] = _mm_and_ps(v.data()[0], _mm_castsi128_ps(x.data()));
        v.data()[1] = _mm_and_ps(v.data()[1], _mm_castsi128_ps(y.data()));
    }

    static inline void floor_shift(double_v &v, double_v e)
    {
        Common::VectorMemoryUnion<__m128i, long long> x = _mm_setallone_si128();
        const uint_v shifts = static_cast<uint_v>(e);
        x.v() = _mm_slli_epi64(x.v(), 52);
        x.m(0) >>= shifts[0];
        x.m(1) >>= shifts[1];
        v &= double_v(_mm_castsi128_pd(x.v()));
    }

    template<typename T>
    inline Vector<T> floor(Vector<T> _v) {
        typedef Vector<T> V;
        typedef typename V::Mask M;

        V v = _v;
        V e = abs(v).exponent();
        const M negativeExponent = e < 0;
        e.setZero(negativeExponent);
        const M negativeInput = v < V::Zero();

        floor_shift(v, e);

        v.setZero(negativeExponent);
        v(negativeInput && _v != v) -= V::One();
        return v;
    }

    template<typename T>
    inline Vector<T> ceil(Vector<T> _v) {
        typedef Vector<T> V;
        typedef typename V::Mask M;

        V v = _v;
        V e = abs(v).exponent();
        const M negativeExponent = e < 0;
        e.setZero(negativeExponent);
        const M positiveInput = v > V::Zero();

        floor_shift(v, e);

        v.setZero(negativeExponent);
        v(positiveInput && _v != v) += V::One();
        return v;
    }
#endif
} // namespace SSE
} // namespace Vc

#define VC__USE_NAMESPACE SSE
#include "../common/trigonometric.h"
#define VC__USE_NAMESPACE SSE
#include "../common/logarithm.h"

#endif // VC_SSE_MATH_H
