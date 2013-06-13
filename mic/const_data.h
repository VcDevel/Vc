/*  This file is part of the Vc library. {{{

    Copyright (C) 2013 Matthias Kretz <kretz@kde.org>

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

}}}*/

#ifndef VC_MIC_CONST_DATA_H
#define VC_MIC_CONST_DATA_H

#include "../common/data.h"
#include "macros.h"

Vc_NAMESPACE_BEGIN(MIC)

ALIGN(16) extern const char  _IndexesFromZero8[16];

struct STRUCT_ALIGN1(64) c_general
{
    static const float oneFloat;
    static const unsigned int absMaskFloat[2];
    static const unsigned int signMaskFloat[2];
    static const unsigned int highMaskFloat;
    static const unsigned short minShort[2];
    static const unsigned short one16[2];
    static const float _2power31;
    static const double oneDouble;
    static const unsigned long long frexpMask;
    static const unsigned long long highMaskDouble;
    static const unsigned char frexpAndMask[16];
} STRUCT_ALIGN2(64);

Vc_NAMESPACE_END

#include "undomacros.h"

#endif // VC_MIC_CONST_DATA_H
