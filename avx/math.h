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
} // namespace AVX
} // namespace Vc

#include "undomacros.h"
#define VC__USE_NAMESPACE AVX
#include "../common/trigonometric.h"
#define VC__USE_NAMESPACE AVX
#include "../common/logarithm.h"

#endif // VC_AVX_MATH_H
