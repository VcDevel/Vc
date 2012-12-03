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

#ifndef VC_AVX_CONST_H
#define VC_AVX_CONST_H

#include <cstddef>
#include "const_data.h"
#include "macros.h"

namespace Vc
{
namespace AVX
{
    template<typename T> class Vector;

    template<typename T> struct IndexesFromZeroData;
    template<> struct IndexesFromZeroData<int> {
        static const int *address() { return reinterpret_cast<const int *>(&_IndexesFromZero32[0]); }
    };
    template<> struct IndexesFromZeroData<unsigned int> {
        static const unsigned int *address() { return &_IndexesFromZero32[0]; }
    };
    template<> struct IndexesFromZeroData<short> {
        static const short *address() { return reinterpret_cast<const short *>(&_IndexesFromZero16[0]); }
    };
    template<> struct IndexesFromZeroData<unsigned short> {
        static const unsigned short *address() { return &_IndexesFromZero16[0]; }
    };
    template<> struct IndexesFromZeroData<signed char> {
        static const signed char *address() { return reinterpret_cast<const signed char *>(&_IndexesFromZero8[0]); }
    };
    template<> struct IndexesFromZeroData<char> {
        static const char *address() { return reinterpret_cast<const char *>(&_IndexesFromZero8[0]); }
    };
    template<> struct IndexesFromZeroData<unsigned char> {
        static const unsigned char *address() { return &_IndexesFromZero8[0]; }
    };

    template<typename _T> struct Const
    {
        typedef Vector<_T> V;
        typedef typename V::EntryType T;
        typedef typename V::Mask M;

        static inline Vc_ALWAYS_INLINE Vc_CONST V _pi_4()        { return V(c_trig<T>::data[0]); }
        static inline Vc_ALWAYS_INLINE Vc_CONST V _pi_4_hi()     { return V(c_trig<T>::data[1]); }
        static inline Vc_ALWAYS_INLINE Vc_CONST V _pi_4_rem1()   { return V(c_trig<T>::data[2]); }
        static inline Vc_ALWAYS_INLINE Vc_CONST V _pi_4_rem2()   { return V(c_trig<T>::data[3]); }
        static inline Vc_ALWAYS_INLINE Vc_CONST V _1_16()        { return V(c_trig<T>::data[4]); }
        static inline Vc_ALWAYS_INLINE Vc_CONST V _16()          { return V(c_trig<T>::data[5]); }

        static inline Vc_ALWAYS_INLINE Vc_CONST V cosCoeff(int i)   { return V(c_trig<T>::data[( 8 + i)]); }
        static inline Vc_ALWAYS_INLINE Vc_CONST V sinCoeff(int i)   { return V(c_trig<T>::data[(14 + i)]); }
        static inline Vc_ALWAYS_INLINE Vc_CONST V atanP(int i)      { return V(c_trig<T>::data[(24 + i)]); }
        static inline Vc_ALWAYS_INLINE Vc_CONST V atanQ(int i)      { return V(c_trig<T>::data[(29 + i)]); }
        static inline Vc_ALWAYS_INLINE Vc_CONST V atanThrsHi()      { return V(c_trig<T>::data[34]); }
        static inline Vc_ALWAYS_INLINE Vc_CONST V atanThrsLo()      { return V(c_trig<T>::data[35]); }
        static inline Vc_ALWAYS_INLINE Vc_CONST V _pi_2_rem()       { return V(c_trig<T>::data[36]); }
        static inline Vc_ALWAYS_INLINE Vc_CONST V lossThreshold()   { return V(c_trig<T>::data[20]); }
        static inline Vc_ALWAYS_INLINE Vc_CONST V _4_pi()           { return V(c_trig<T>::data[21]); }
        static inline Vc_ALWAYS_INLINE Vc_CONST V _pi_2()           { return V(c_trig<T>::data[22]); }
        static inline Vc_ALWAYS_INLINE Vc_CONST V _pi()             { return V(c_trig<T>::data[23]); }
        static inline Vc_ALWAYS_INLINE Vc_CONST V asinCoeff0(int i) { return V(c_trig<T>::data[(40 + i)]); }
        static inline Vc_ALWAYS_INLINE Vc_CONST V asinCoeff1(int i) { return V(c_trig<T>::data[(45 + i)]); }
        static inline Vc_ALWAYS_INLINE Vc_CONST V asinCoeff2(int i) { return V(c_trig<T>::data[(49 + i)]); }
        static inline Vc_ALWAYS_INLINE Vc_CONST V asinCoeff3(int i) { return V(c_trig<T>::data[(55 + i)]); }
        static inline Vc_ALWAYS_INLINE Vc_CONST V smallAsinInput()  { return V(c_trig<T>::data[37]); }
        static inline Vc_ALWAYS_INLINE Vc_CONST V largeAsinInput()  { return V(c_trig<T>::data[38]); }

        static inline M Vc_CONST_L exponentMask() Vc_CONST_R { return M(V(c_log<T>::d(1)).data()); }
        static inline V Vc_CONST_L _1_2()         Vc_CONST_R { return V(c_log<T>::d(18)); }
        static inline V Vc_CONST_L _1_sqrt2()     Vc_CONST_R { return V(c_log<T>::d(15)); }
        static inline V Vc_CONST_L P(int i)       Vc_CONST_R { return V(c_log<T>::d(2 + i)); }
        static inline V Vc_CONST_L Q(int i)       Vc_CONST_R { return V(c_log<T>::d(8 + i)); }
        static inline V Vc_CONST_L min()          Vc_CONST_R { return V(c_log<T>::d(14)); }
        static inline V Vc_CONST_L ln2_small()    Vc_CONST_R { return V(c_log<T>::d(17)); }
        static inline V Vc_CONST_L ln2_large()    Vc_CONST_R { return V(c_log<T>::d(16)); }
        static inline V Vc_CONST_L neginf()       Vc_CONST_R { return V(c_log<T>::d(13)); }
        static inline V Vc_CONST_L log10_e()      Vc_CONST_R { return V(c_log<T>::d(19)); }
        static inline V Vc_CONST_L log2_e()       Vc_CONST_R { return V(c_log<T>::d(20)); }

        static inline V Vc_CONST_L highMask() Vc_CONST_R;
    };

    template<> inline Vector<float> Vc_CONST Const<float>::highMask() { return _mm256_broadcast_ss(reinterpret_cast<const float *>(&c_general::highMaskFloat)); }
    template<> inline Vector<double> Vc_CONST Const<double>::highMask() { return _mm256_broadcast_sd(reinterpret_cast<const double *>(&c_general::highMaskDouble)); }
    template<> inline Vector<sfloat> Vc_CONST Const<sfloat>::highMask() { return _mm256_broadcast_ss(reinterpret_cast<const float *>(&c_general::highMaskFloat)); }
} // namespace AVX
} // namespace Vc

#include "undomacros.h"

#endif // VC_AVX_CONST_H
