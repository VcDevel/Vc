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

#ifndef VC_SSE_CONST_H
#define VC_SSE_CONST_H

#include "macros.h"

namespace Vc
{
namespace SSE
{
    template<typename T> class Vector;

    ALIGN(16) extern const unsigned int   _IndexesFromZero4[4];
    ALIGN(16) extern const unsigned short _IndexesFromZero8[8];
    ALIGN(16) extern const unsigned char  _IndexesFromZero16[16];

    struct c_general
    {
        ALIGN(64) static const unsigned int allone[4];
        ALIGN(16) static const unsigned short one16[8];
        ALIGN(16) static const unsigned int one32[4];
        ALIGN(16) static const float oneFloat[4];
        ALIGN(16) static const double oneDouble[2];
        ALIGN(16) static const int absMaskFloat[4];
        ALIGN(16) static const long long absMaskDouble[2];
        ALIGN(16) static const unsigned int signMaskFloat[4];
        ALIGN(16) static const unsigned long long signMaskDouble[2];
        ALIGN(16) static const short minShort[8];
    };
    template<typename T> struct c_sin
    {
        typedef Vector<T> V;
        enum { Size = 16 / sizeof(T) };
        ALIGN(64) static const T _data[Size * 8];

        static V CONST_L _1_2pi()  CONST_R;
        static V CONST_L _2pi()    CONST_R;
        static V CONST_L _pi_2()   CONST_R;
        static V CONST_L _pi()     CONST_R;

        static V CONST_L _1_3fac() CONST_R;
        static V CONST_L _1_5fac() CONST_R;
        static V CONST_L _1_7fac() CONST_R;
        static V CONST_L _1_9fac() CONST_R;
    };
} // namespace SSE
} // namespace Vc

#include "undomacros.h"

#endif // VC_SSE_CONST_H
