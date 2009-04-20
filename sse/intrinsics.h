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

#ifndef SSE_INTRINSICS_H
#define SSE_INTRINSICS_H

// MMX
#include <mmintrin.h>
// SSE
#include <xmmintrin.h>
// SSE2
#include <emmintrin.h>
// SSE3
#ifdef __SSE3__
#include <pmmintrin.h>
#endif
// SSSE3
#ifdef __SSSE3__
#include <tmmintrin.h>
#endif

// SSE4.1 (and 4.2)
#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

#endif // SSE_INTRINSICS_H
