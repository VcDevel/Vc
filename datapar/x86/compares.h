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

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
namespace x86
{
#ifdef Vc_HAVE_AVX2
Vc_INTRINSIC Vc_CONST y_u64 Vc_VDECL cmpgt(y_u64 x, y_u64 y)
{
#ifdef Vc_HAVE_AVX512VL
    return _mm256_cmpgt_epu64_mask(x, y);
#elif defined Vc_HAVE_XOP
    return concat(_mm256_comgt_epu64(lo128(x), lo128(y)),
                  _mm256_comgt_epu64(hi128(x), hi128(y)));
#else
    return _mm256_cmpgt_epi64(xor_(x, lowest32<llong>()), xor_(y, lowest32<llong>()));
#endif
}

Vc_INTRINSIC Vc_CONST y_u32 Vc_VDECL cmpgt(y_u32 x, y_u32 y)
{
#ifdef Vc_HAVE_AVX512VL
    return _mm256_cmpgt_epu32_mask(x, y);
#elif defined Vc_HAVE_XOP
    return concat(_mm256_comgt_epu32(lo128(x), lo128(y)),
                  _mm256_comgt_epu32(hi128(x), hi128(y)));
#else
    return _mm256_cmpgt_epi32(xor_(x, lowest32<int>()), xor_(y, lowest32<int>()));
#endif
}

Vc_INTRINSIC Vc_CONST y_u16 Vc_VDECL cmpgt(y_u16 x, y_u16 y)
{
#ifdef Vc_HAVE_AVX512VL
    return _mm256_cmpgt_epu16_mask(x, y);
#elif defined Vc_HAVE_XOP
    return concat(_mm256_comgt_epu16(lo128(x), lo128(y)),
                  _mm256_comgt_epu16(hi128(x), hi128(y)));
#else
    return _mm256_cmpgt_epi16(xor_(x, lowest32<short>()), xor_(y, lowest32<short>()));
#endif
}

Vc_INTRINSIC Vc_CONST y_u08 Vc_VDECL cmpgt(y_u08 x, y_u08 y)
{
#ifdef Vc_HAVE_AVX512VL
    return _mm256_cmpgt_epu8_mask(x, y);
#elif defined Vc_HAVE_XOP
    return concat(_mm256_comgt_epu8(lo128(x), lo128(y)),
                  _mm256_comgt_epu8(hi128(x), hi128(y)));
#else
    return _mm256_cmpgt_epi8(xor_(x, lowest32<schar>()), xor_(y, lowest32<schar>()));
#endif
}

Vc_INTRINSIC Vc_CONST y_ulong Vc_VDECL cmpgt(y_ulong x, y_ulong y)
{
    return cmpgt(y_ulong_equiv(x), y_ulong_equiv(y)).v();
}
#endif
}}  // namespace detail::x86
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DATAPAR_X86_COMPARES_H_
