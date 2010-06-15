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

namespace Vc
{
namespace AVX
{
    template<typename T> class Vector;

    ALIGN(32) extern const unsigned int   _IndexesFromZero32[8];
    ALIGN(32) extern const unsigned short _IndexesFromZero16[16];
    ALIGN(32) extern const unsigned char  _IndexesFromZero8[32];

    struct c_general
    {
        ALIGN(32) static const unsigned short one16[16];
        ALIGN(32) static const unsigned int one32[4];
        ALIGN(32) static const float oneFloat[4];
        ALIGN(32) static const double oneDouble[2];
        ALIGN(32) static const int absMaskFloat[4];
        ALIGN(32) static const long long absMaskDouble[2];
        ALIGN(32) static const unsigned int signMaskFloat[4];
        ALIGN(32) static const unsigned long long signMaskDouble[2];
        ALIGN(32) static const short minShort[16];
    } ALIGN(64);
    template<typename T> struct c_sin
    {
        typedef Vector<T> V;
        enum { Size = sizeof(__m256) / sizeof(T) };
        static const T _data[Size * 8];

        static V _1_2pi()  CONST;
        static V _2pi()    CONST;
        static V _pi_2()   CONST;
        static V _pi()     CONST;

        static V _1_3fac() CONST;
        static V _1_5fac() CONST;
        static V _1_7fac() CONST;
        static V _1_9fac() CONST;
    } ALIGN(64);
} // namespace AVX
} // namespace Vc

#include "undomacros.h"

#endif // VC_AVX_CONST_H
