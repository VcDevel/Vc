/*  This file is part of the Vc library. {{{
Copyright © 2011-2015 Matthias Kretz <kretz@kde.org>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the names of contributing organizations nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#ifndef VC_DATAPAR_BITSCAN_H_
#define VC_DATAPAR_BITSCAN_H_

#include "macros.h"
#include <x86intrin.h>

namespace Vc_VERSIONED_NAMESPACE::detail {
#if defined(Vc_ICC) || (defined(Vc_GCC) && Vc_GCC >= 0x40500)
Vc_ALWAYS_INLINE Vc_CONST auto bit_scan_forward(unsigned int x)
{
    return _bit_scan_forward(x);
}
Vc_ALWAYS_INLINE Vc_CONST auto bit_scan_reverse(unsigned int x)
{
    return _bit_scan_reverse(x);
}
#elif defined Vc_CLANG || defined Vc_APPLECLANG
// GCC <= 4.4 and clang have x86intrin.h, but not the required functions
Vc_ALWAYS_INLINE Vc_CONST auto bit_scan_forward(unsigned int x)
{
    return __builtin_ctz(x);
}
Vc_ALWAYS_INLINE Vc_CONST int bit_scan_reverse(unsigned int x)
{
    int r;
    __asm__("bsr %1,%0" : "=r"(r) : "X"(x));
    return r;
}
#elif defined(Vc_MSVC)
#pragma intrinsic(_BitScanForward)
#pragma intrinsic(_BitScanReverse)
Vc_ALWAYS_INLINE Vc_CONST unsigned long bit_scan_forward(unsigned long x)
{
    unsigned long index;
    _BitScanForward(&index, x);
    return index;
}
Vc_ALWAYS_INLINE Vc_CONST unsigned long bit_scan_reverse(unsigned long x) {
    unsigned long index;
    _BitScanReverse(&index, x);
    return index;
}
#endif
}  // namespace Vc_VERSIONED_NAMESPACE::detail

#endif  // VC_DATAPAR_BITSCAN_H_
