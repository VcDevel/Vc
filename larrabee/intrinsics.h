/*
    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) version 3.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Library General Public License for more details.

    You should have received a copy of the GNU Library General Public License
    along with this library; see the file COPYING.LIB.  If not, write to
    the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
    Boston, MA 02110-1301, USA.

*/

#ifndef LARRABEE_INTRINSICS_H
#define LARRABEE_INTRINSICS_H

#include "lrbni_prototypes.h"

namespace Larrabee
{
    namespace FixedIntrinsics
    {
        inline _M512 _mm512_loadd(const void *mt, _MM_FULLUP32_ENUM full_up = _MM_FULLUPC_NONE, _MM_BROADCAST32_ENUM broadcast = _MM_BROADCAST_16X16, _MM_MEM_HINT_ENUM non_temporal = _MM_HINT_NONE)
            { return ::_mm512_loadd(const_cast<void *>(mt), full_up, broadcast, non_temporal); }

        inline _M512 _mm512_loadq(const void *mt, _MM_FULLUP64_ENUM full_up = _MM_FULLUPC64_NONE, _MM_BROADCAST64_ENUM broadcast = _MM_BROADCAST_8X8, _MM_MEM_HINT_ENUM non_temporal = _MM_HINT_NONE)
            { return ::_mm512_loadq(const_cast<void *>(mt), full_up, broadcast, non_temporal); }
    } // namespace FixedIntrinsics
} // namespace Larrabee

#ifdef __GNUC__
#define LRB_ALIGN(n) __attribute__((aligned(n)))
#else
#define LRB_ALIGN(n) __declspec(align(n))
#endif

#endif // LARRABEE_INTRINSICS_H
