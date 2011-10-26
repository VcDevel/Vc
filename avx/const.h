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

#ifndef VC_AVX_CONST_H
#define VC_AVX_CONST_H

#include "macros.h"
#include <cstddef>

namespace Vc
{
namespace AVX
{
    template<typename T> class Vector;
    template<unsigned int VectorSize, size_t RegisterWidth> class Mask;
    class M128iDummy;

    ALIGN(64) extern const unsigned int   _IndexesFromZero32[8];
    ALIGN(16) extern const unsigned short _IndexesFromZero16[8];
    ALIGN(16) extern const unsigned char  _IndexesFromZero8[16];

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

    struct STRUCT_ALIGN1(64) c_general
    {
        static const float oneFloat;
        static const unsigned int absMaskFloat[2];
        static const unsigned int signMaskFloat[2];
        static const unsigned short minShort[2];
        static const unsigned short one16[2];
        static const float _2power31;
        static const double oneDouble;
    } STRUCT_ALIGN2(64);
    template<typename T> struct STRUCT_ALIGN1(64) c_sin
    {
        typedef Vector<T> V;
        static const T _data[8];

        static V CONST_L _1_2pi()  CONST_R;
        static V CONST_L _2pi()    CONST_R;
        static V CONST_L _pi_2()   CONST_R;
        static V CONST_L _pi()     CONST_R;

        static V CONST_L _1_3fac() CONST_R;
        static V CONST_L _1_5fac() CONST_R;
        static V CONST_L _1_7fac() CONST_R;
        static V CONST_L _1_9fac() CONST_R;
    } STRUCT_ALIGN2(64);

    template<typename T> struct IntForFloat { typedef unsigned int Type; };
    template<> struct IntForFloat<double> { typedef unsigned long long Type; };

    template<typename T, typename Mask> struct c_log
    {
        typedef Vector<T> Vec;
        typedef typename IntForFloat<T>::Type Int;

        static inline const double *d(int i) { return reinterpret_cast<const double *>(&_dataI[i]); }
        static inline const float *f(int i) { return reinterpret_cast<const float *>(&_dataI[i]); }
        static const Int _dataI[15];
        static const T   _dataT[6];

        static M128iDummy CONST_L bias()  CONST_R;
        static Mask CONST_L exponentMask() CONST_R;
        static Vec CONST_L _1_2() CONST_R;
        static Vec CONST_L _1_sqrt2() CONST_R;
        static Vec CONST_L P(int i) CONST_R;
        static Vec CONST_L Q(int i) CONST_R;
        static Vec CONST_L min() CONST_R;
        static Vec CONST_L ln2_small() CONST_R;
        static Vec CONST_L ln2_large() CONST_R;
        static Vec CONST_L neginf() CONST_R;
        static Vec CONST_L log10_e() CONST_R;
        static Vec CONST_L log2_e() CONST_R;
    };
} // namespace AVX
} // namespace Vc

#include "undomacros.h"

#endif // VC_AVX_CONST_H
