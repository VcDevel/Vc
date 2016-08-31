/*  This file is part of the Vc library. {{{
Copyright © 2016 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_DATAPAR_X86_COMPARES_H_
#define VC_DATAPAR_X86_COMPARES_H_

#include "storage.h"

namespace Vc_VERSIONED_NAMESPACE::detail::x86
{
#ifdef Vc_HAVE_AVX2
Vc_INTRINSIC Vc_CONST y_u64 cmpgt(y_u64 x, y_u64 y)
{
#ifdef Vc_HAVE_AVX512VL
    return _mm256_cmpgt_epu64_mask(x, y);
#else
    return _mm256_cmpgt_epi64(x, y);
#endif
}

Vc_INTRINSIC Vc_CONST y_u32 cmpgt(y_u32 x, y_u32 y)
{
#ifdef Vc_HAVE_AVX512VL
    return _mm256_cmpgt_epu32_mask(x, y);
#else
    return _mm256_cmpgt_epi32(x, y);
#endif
}

Vc_INTRINSIC Vc_CONST y_u16 cmpgt(y_u16 x, y_u16 y)
{
#ifdef Vc_HAVE_AVX512VL
    return _mm256_cmpgt_epu16_mask(x, y);
#else
    return _mm256_cmpgt_epi16(x, y);
#endif
}

Vc_INTRINSIC Vc_CONST y_u08 cmpgt(y_u08 x, y_u08 y)
{
#ifdef Vc_HAVE_AVX512VL
    return _mm256_cmpgt_epu8_mask(x, y);
#else
    return _mm256_cmpgt_epi8(x, y);
#endif
}

Vc_INTRINSIC Vc_CONST y_ulong cmpgt(y_ulong x, y_ulong y)
{
    return cmpgt(y_ulong_equiv(x), y_ulong_equiv(y)).v();
}
#endif
}  // namespace Vc_VERSIONED_NAMESPACE::detail::x86

#endif  // VC_DATAPAR_X86_COMPARES_H_
