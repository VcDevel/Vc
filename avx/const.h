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

    ALIGN(64) extern const unsigned int   _IndexesFromZero32[8];
    ALIGN(16) extern const unsigned short _IndexesFromZero16[8];
    ALIGN(16) extern const unsigned char  _IndexesFromZero8[16];

    struct c_general
    {
        static const float oneFloat;
        static const unsigned int absMaskFloat[2];
        static const unsigned int signMaskFloat[2];
        static const unsigned short minShort[2];
        static const unsigned short one16[2];
        static const double oneDouble;
    } ALIGN(64);
    template<typename T> struct c_sin
    {
        typedef Vector<T> V;
        static const T _data[8];

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
