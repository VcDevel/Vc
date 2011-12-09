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
    template<typename T, typename M> inline M128iDummy c_log<T, M>::bias() { return _mm_load_si128(&_dataI[0]); }

    typedef Vector<double> double_v;
    typedef Vector<float> float_v;
    typedef Vector<float8> sfloat_v;
    typedef Vector<double>::Mask double_m;
    typedef Vector<float >::Mask float_m;
    typedef Vector<float8>::Mask sfloat_m;

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
} // namespace SSE
} // namespace Vc

#define VC__USE_NAMESPACE SSE
#include "../common/trigonometric.h"
#define VC__USE_NAMESPACE SSE
#include "../common/logarithm.h"

#endif // VC_SSE_MATH_H
