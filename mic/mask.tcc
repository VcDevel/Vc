/*  This file is part of the Vc library. {{{
Copyright © 2013-2015 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_MIC_MASK_TCC_
#define VC_MIC_MASK_TCC_

#include "macros.h"

Vc_VERSIONED_NAMESPACE_BEGIN
template <>
template <typename Flags>
inline void MIC::double_m::load(const bool *mem, Flags)
{
    __m512i ones = _mm512_setzero_epi32();
    ones = _mm512_mask_extloadunpacklo_epi32(ones, 0xff, mem, MIC::UpDownConversion<unsigned int, unsigned char>().up(), _MM_HINT_NONE);
    ones = _mm512_mask_extloadunpackhi_epi32(ones, 0xff, mem + 64, MIC::UpDownConversion<unsigned int, unsigned char>().up(), _MM_HINT_NONE);
    //const __m512i ones = _mm512_mask_extload_epi32(_mm512_setzero_epi32(), 0xff, mem, , _MM_BROADCAST32_NONE, _MM_HINT_NONE);
    k = _mm512_cmpneq_epi32_mask(ones, _mm512_setzero_epi32());
}

template <>
template <typename Flags>
inline void MIC::double_m::store(bool *mem, Flags) const
{
    const __m512i zero = _mm512_setzero_epi32();
    const __m512i one = _mm512_set1_epi32(1);
    const __m512i tmp = MIC::_and(zero, static_cast<__mmask16>(k), one, one);
    _mm512_mask_extpackstorelo_epi32(mem, 0xff, tmp, MIC::UpDownConversion<unsigned int, unsigned char>().down(), _MM_HINT_NONE);
    _mm512_mask_extpackstorehi_epi32(mem + 64, 0xff, tmp, MIC::UpDownConversion<unsigned int, unsigned char>().down(), _MM_HINT_NONE);
}
Vc_VERSIONED_NAMESPACE_END

#endif // VC_MIC_MASK_TCC_
