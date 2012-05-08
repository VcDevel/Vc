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

    template<typename T> struct Const
    {
        typedef Vector<T> V;
        typedef typename V::Mask M;

        static inline V CONST_L _1_2pi()  CONST_R { return V(c_sin<T>::data[0]); }
        static inline V CONST_L _2pi()    CONST_R { return V(c_sin<T>::data[1]); }
        static inline V CONST_L _pi_2()   CONST_R { return V(c_sin<T>::data[2]); }
        static inline V CONST_L _pi()     CONST_R { return V(c_sin<T>::data[3]); }
        static inline V CONST_L _1_3fac() CONST_R { return V(c_sin<T>::data[4]); }
        static inline V CONST_L _1_5fac() CONST_R { return V(c_sin<T>::data[5]); }
        static inline V CONST_L _1_7fac() CONST_R { return V(c_sin<T>::data[6]); }
        static inline V CONST_L _1_9fac() CONST_R { return V(c_sin<T>::data[7]); }

        static inline M CONST_L exponentMask() CONST_R { return M(V(c_log<T>::d(1)).data()); }
        static inline V CONST_L _1_2()         CONST_R { return V(c_log<T>::d(18)); }
        static inline V CONST_L _1_sqrt2()     CONST_R { return V(c_log<T>::d(15)); }
        static inline V CONST_L P(int i)       CONST_R { return V(c_log<T>::d(2 + i)); }
        static inline V CONST_L Q(int i)       CONST_R { return V(c_log<T>::d(8 + i)); }
        static inline V CONST_L min()          CONST_R { return V(c_log<T>::d(14)); }
        static inline V CONST_L ln2_small()    CONST_R { return V(c_log<T>::d(17)); }
        static inline V CONST_L ln2_large()    CONST_R { return V(c_log<T>::d(16)); }
        static inline V CONST_L neginf()       CONST_R { return V(c_log<T>::d(13)); }
        static inline V CONST_L log10_e()      CONST_R { return V(c_log<T>::d(19)); }
        static inline V CONST_L log2_e()       CONST_R { return V(c_log<T>::d(20)); }
    };

    template<> struct Const<sfloat>
    {
        typedef sfloat_v V;
        typedef V::Mask M;

        static inline V CONST_L _1_2pi()  CONST_R { return V(c_sin<float>::data[0]); }
        static inline V CONST_L _2pi()    CONST_R { return V(c_sin<float>::data[1]); }
        static inline V CONST_L _pi_2()   CONST_R { return V(c_sin<float>::data[2]); }
        static inline V CONST_L _pi()     CONST_R { return V(c_sin<float>::data[3]); }
        static inline V CONST_L _1_3fac() CONST_R { return V(c_sin<float>::data[4]); }
        static inline V CONST_L _1_5fac() CONST_R { return V(c_sin<float>::data[5]); }
        static inline V CONST_L _1_7fac() CONST_R { return V(c_sin<float>::data[6]); }
        static inline V CONST_L _1_9fac() CONST_R { return V(c_sin<float>::data[7]); }

        static inline M CONST_L exponentMask() CONST_R { return M(V(c_log<float>::d(1)).data()); }
        static inline V CONST_L _1_2()         CONST_R { return V(c_log<float>::d(18)); }
        static inline V CONST_L _1_sqrt2()     CONST_R { return V(c_log<float>::d(15)); }
        static inline V CONST_L P(int i)       CONST_R { return V(c_log<float>::d(2 + i)); }
        static inline V CONST_L Q(int i)       CONST_R { return V(c_log<float>::d(8 + i)); }
        static inline V CONST_L min()          CONST_R { return V(c_log<float>::d(14)); }
        static inline V CONST_L ln2_small()    CONST_R { return V(c_log<float>::d(17)); }
        static inline V CONST_L ln2_large()    CONST_R { return V(c_log<float>::d(16)); }
        static inline V CONST_L neginf()       CONST_R { return V(c_log<float>::d(13)); }
        static inline V CONST_L log10_e()      CONST_R { return V(c_log<float>::d(19)); }
        static inline V CONST_L log2_e()       CONST_R { return V(c_log<float>::d(20)); }
    };
} // namespace AVX
} // namespace Vc

#include "undomacros.h"

#endif // VC_AVX_CONST_H
