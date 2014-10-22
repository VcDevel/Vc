/*  This file is part of the Vc library. {{{
Copyright © 2013-2014 Matthias Kretz <kretz@kde.org>
All rights reserved.

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

#include <common/types.h>
#include <avx/intrinsics.h>
#include <avx/casts.h>
#include <avx/sorthelper.h>
#include <avx/macros.h>

#include <src/avx_sorthelper.cpp>

namespace Vc_VERSIONED_NAMESPACE
{
namespace Vc_IMPL_NAMESPACE
{

template<> __m128i SortHelper<short>::sort(VTArg _x)
{
    m128i lo, hi, y, x = _x;
    // sort pairs
    y = _mm_shufflelo_epi16(_mm_shufflehi_epi16(x, _MM_SHUFFLE(2, 3, 0, 1)), _MM_SHUFFLE(2, 3, 0, 1));
    lo = _mm_min_epi16(x, y);
    hi = _mm_max_epi16(x, y);
    x = _mm_blend_epi16(lo, hi, 0xaa);

    // merge left and right quads
    y = _mm_shufflelo_epi16(_mm_shufflehi_epi16(x, _MM_SHUFFLE(0, 1, 2, 3)), _MM_SHUFFLE(0, 1, 2, 3));
    lo = _mm_min_epi16(x, y);
    hi = _mm_max_epi16(x, y);
    x = _mm_blend_epi16(lo, hi, 0xcc);
    y = _mm_srli_si128(x, 2);
    lo = _mm_min_epi16(x, y);
    hi = _mm_max_epi16(x, y);
    x = _mm_blend_epi16(lo, _mm_slli_si128(hi, 2), 0xaa);

    // merge quads into octs
    y = _mm_shuffle_epi32(x, _MM_SHUFFLE(1, 0, 3, 2));
    y = _mm_shufflelo_epi16(y, _MM_SHUFFLE(0, 1, 2, 3));
    lo = _mm_min_epi16(x, y);
    hi = _mm_max_epi16(x, y);

    x = _mm_unpacklo_epi16(lo, hi);
    y = _mm_srli_si128(x, 8);
    lo = _mm_min_epi16(x, y);
    hi = _mm_max_epi16(x, y);

    x = _mm_unpacklo_epi16(lo, hi);
    y = _mm_srli_si128(x, 8);
    lo = _mm_min_epi16(x, y);
    hi = _mm_max_epi16(x, y);

    return _mm_unpacklo_epi16(lo, hi);
}
template<> __m128i SortHelper<unsigned short>::sort(VTArg _x)
{
    m128i lo, hi, y, x = _x;
    // sort pairs
    y = _mm_shufflelo_epi16(_mm_shufflehi_epi16(x, _MM_SHUFFLE(2, 3, 0, 1)), _MM_SHUFFLE(2, 3, 0, 1));
    lo = _mm_min_epu16(x, y);
    hi = _mm_max_epu16(x, y);
    x = _mm_blend_epi16(lo, hi, 0xaa);

    // merge left and right quads
    y = _mm_shufflelo_epi16(_mm_shufflehi_epi16(x, _MM_SHUFFLE(0, 1, 2, 3)), _MM_SHUFFLE(0, 1, 2, 3));
    lo = _mm_min_epu16(x, y);
    hi = _mm_max_epu16(x, y);
    x = _mm_blend_epi16(lo, hi, 0xcc);
    y = _mm_srli_si128(x, 2);
    lo = _mm_min_epu16(x, y);
    hi = _mm_max_epu16(x, y);
    x = _mm_blend_epi16(lo, _mm_slli_si128(hi, 2), 0xaa);

    // merge quads into octs
    y = _mm_shuffle_epi32(x, _MM_SHUFFLE(1, 0, 3, 2));
    y = _mm_shufflelo_epi16(y, _MM_SHUFFLE(0, 1, 2, 3));
    lo = _mm_min_epu16(x, y);
    hi = _mm_max_epu16(x, y);

    x = _mm_unpacklo_epi16(lo, hi);
    y = _mm_srli_si128(x, 8);
    lo = _mm_min_epu16(x, y);
    hi = _mm_max_epu16(x, y);

    x = _mm_unpacklo_epi16(lo, hi);
    y = _mm_srli_si128(x, 8);
    lo = _mm_min_epu16(x, y);
    hi = _mm_max_epu16(x, y);

    return _mm_unpacklo_epi16(lo, hi);
}

}
}
