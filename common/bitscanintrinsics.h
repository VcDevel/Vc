/*  This file is part of the Vc library.

    Copyright (C) 2011 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_COMMON_BITSCANINTRINSICS_H
#define VC_COMMON_BITSCANINTRINSICS_H

#ifdef VC_GCC
#  if VC_GCC >= 0x40500
     // GCC 4.5.0 introduced _bit_scan_forward / _bit_scan_reverse
#    include <x86intrin.h>
#  else
     // GCC 4.4 has x86intrin.h, but not the required functions
#    define _bit_scan_forward(x) __builtin_ctz(x)
#    define _bit_scan_reverse(x) __builtin_clz(x)
#  endif
#elif defined(VC_ICC)
// for all I know ICC supports the _bit_scan_* intrinsics
#elif defined(VC_OPEN64)
// TODO
#elif defined(VC_MSVC)
#    define _bit_scan_forward(x) _BitScanForward(x)
#    define _bit_scan_reverse(x) _BitScanReverse(x)
#else
// just assume the compiler can do it
#endif


#endif // VC_COMMON_BITSCANINTRINSICS_H
