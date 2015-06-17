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

template<> __m256i SortHelper<int>::sort(VTArg _hgfedcba)
{
    VectorType hgfedcba = _hgfedcba;
    const m128i hgfe = hi128(hgfedcba);
    const m128i dcba = lo128(hgfedcba);
    m128i l = _mm_min_epi32(hgfe, dcba); // ↓hd ↓gc ↓fb ↓ea
    m128i h = _mm_max_epi32(hgfe, dcba); // ↑hd ↑gc ↑fb ↑ea

    m128i x = _mm_unpacklo_epi32(l, h); // ↑fb ↓fb ↑ea ↓ea
    m128i y = _mm_unpackhi_epi32(l, h); // ↑hd ↓hd ↑gc ↓gc

    l = _mm_min_epi32(x, y); // ↓(↑fb,↑hd) ↓hfdb ↓(↑ea,↑gc) ↓geca
    h = _mm_max_epi32(x, y); // ↑hfdb ↑(↓fb,↓hd) ↑geca ↑(↓ea,↓gc)

    x = _mm_min_epi32(l, Reg::permute<X2, X2, X0, X0>(h)); // 2(hfdb) 1(hfdb) 2(geca) 1(geca)
    y = _mm_max_epi32(h, Reg::permute<X3, X3, X1, X1>(l)); // 4(hfdb) 3(hfdb) 4(geca) 3(geca)

    m128i b = Reg::shuffle<Y0, Y1, X0, X1>(y, x); // b3 <= b2 <= b1 <= b0
    m128i a = _mm_unpackhi_epi64(x, y);           // a3 >= a2 >= a1 >= a0

    // _mm_extract_epi32 may return an unsigned int, breaking these comparisons.
    if (VC_IS_UNLIKELY(static_cast<int>(_mm_extract_epi32(x, 2)) >= static_cast<int>(_mm_extract_epi32(y, 1)))) {
        return concat(Reg::permute<X0, X1, X2, X3>(b), a);
    } else if (VC_IS_UNLIKELY(static_cast<int>(_mm_extract_epi32(x, 0)) >= static_cast<int>(_mm_extract_epi32(y, 3)))) {
        return concat(a, Reg::permute<X0, X1, X2, X3>(b));
    }

    // merge
    l = _mm_min_epi32(a, b); // ↓a3b3 ↓a2b2 ↓a1b1 ↓a0b0
    h = _mm_max_epi32(a, b); // ↑a3b3 ↑a2b2 ↑a1b1 ↑a0b0

    a = _mm_unpacklo_epi32(l, h); // ↑a1b1 ↓a1b1 ↑a0b0 ↓a0b0
    b = _mm_unpackhi_epi32(l, h); // ↑a3b3 ↓a3b3 ↑a2b2 ↓a2b2
    l = _mm_min_epi32(a, b);      // ↓(↑a1b1,↑a3b3) ↓a1b3 ↓(↑a0b0,↑a2b2) ↓a0b2
    h = _mm_max_epi32(a, b);      // ↑a3b1 ↑(↓a1b1,↓a3b3) ↑a2b0 ↑(↓a0b0,↓a2b2)

    a = _mm_unpacklo_epi32(l, h); // ↑a2b0 ↓(↑a0b0,↑a2b2) ↑(↓a0b0,↓a2b2) ↓a0b2
    b = _mm_unpackhi_epi32(l, h); // ↑a3b1 ↓(↑a1b1,↑a3b3) ↑(↓a1b1,↓a3b3) ↓a1b3
    l = _mm_min_epi32(a, b); // ↓(↑a2b0,↑a3b1) ↓(↑a0b0,↑a2b2,↑a1b1,↑a3b3) ↓(↑(↓a0b0,↓a2b2) ↑(↓a1b1,↓a3b3)) ↓a0b3
    h = _mm_max_epi32(a, b); // ↑a3b0 ↑(↓(↑a0b0,↑a2b2) ↓(↑a1b1,↑a3b3)) ↑(↓a0b0,↓a2b2,↓a1b1,↓a3b3) ↑(↓a0b2,↓a1b3)

    return concat(_mm_unpacklo_epi32(l, h), _mm_unpackhi_epi32(l, h));
}

template<> __m256i SortHelper<unsigned int>::sort(VTArg _hgfedcba)
{
    VectorType hgfedcba = _hgfedcba;
    const m128i hgfe = hi128(hgfedcba);
    const m128i dcba = lo128(hgfedcba);
    m128i l = _mm_min_epu32(hgfe, dcba); // ↓hd ↓gc ↓fb ↓ea
    m128i h = _mm_max_epu32(hgfe, dcba); // ↑hd ↑gc ↑fb ↑ea

    m128i x = _mm_unpacklo_epi32(l, h); // ↑fb ↓fb ↑ea ↓ea
    m128i y = _mm_unpackhi_epi32(l, h); // ↑hd ↓hd ↑gc ↓gc

    l = _mm_min_epu32(x, y); // ↓(↑fb,↑hd) ↓hfdb ↓(↑ea,↑gc) ↓geca
    h = _mm_max_epu32(x, y); // ↑hfdb ↑(↓fb,↓hd) ↑geca ↑(↓ea,↓gc)

    x = _mm_min_epu32(l, Reg::permute<X2, X2, X0, X0>(h)); // 2(hfdb) 1(hfdb) 2(geca) 1(geca)
    y = _mm_max_epu32(h, Reg::permute<X3, X3, X1, X1>(l)); // 4(hfdb) 3(hfdb) 4(geca) 3(geca)

    m128i b = Reg::shuffle<Y0, Y1, X0, X1>(y, x); // b3 <= b2 <= b1 <= b0
    m128i a = _mm_unpackhi_epi64(x, y);           // a3 >= a2 >= a1 >= a0

    if (VC_IS_UNLIKELY(extract_epu32<2>(x) >= extract_epu32<1>(y))) {
        return concat(Reg::permute<X0, X1, X2, X3>(b), a);
    } else if (VC_IS_UNLIKELY(extract_epu32<0>(x) >= extract_epu32<3>(y))) {
        return concat(a, Reg::permute<X0, X1, X2, X3>(b));
    }

    // merge
    l = _mm_min_epu32(a, b); // ↓a3b3 ↓a2b2 ↓a1b1 ↓a0b0
    h = _mm_max_epu32(a, b); // ↑a3b3 ↑a2b2 ↑a1b1 ↑a0b0

    a = _mm_unpacklo_epi32(l, h); // ↑a1b1 ↓a1b1 ↑a0b0 ↓a0b0
    b = _mm_unpackhi_epi32(l, h); // ↑a3b3 ↓a3b3 ↑a2b2 ↓a2b2
    l = _mm_min_epu32(a, b);      // ↓(↑a1b1,↑a3b3) ↓a1b3 ↓(↑a0b0,↑a2b2) ↓a0b2
    h = _mm_max_epu32(a, b);      // ↑a3b1 ↑(↓a1b1,↓a3b3) ↑a2b0 ↑(↓a0b0,↓a2b2)

    a = _mm_unpacklo_epi32(l, h); // ↑a2b0 ↓(↑a0b0,↑a2b2) ↑(↓a0b0,↓a2b2) ↓a0b2
    b = _mm_unpackhi_epi32(l, h); // ↑a3b1 ↓(↑a1b1,↑a3b3) ↑(↓a1b1,↓a3b3) ↓a1b3
    l = _mm_min_epu32(a, b); // ↓(↑a2b0,↑a3b1) ↓(↑a0b0,↑a2b2,↑a1b1,↑a3b3) ↓(↑(↓a0b0,↓a2b2) ↑(↓a1b1,↓a3b3)) ↓a0b3
    h = _mm_max_epu32(a, b); // ↑a3b0 ↑(↓(↑a0b0,↑a2b2) ↓(↑a1b1,↑a3b3)) ↑(↓a0b0,↓a2b2,↓a1b1,↓a3b3) ↑(↓a0b2,↓a1b3)

    return concat(_mm_unpacklo_epi32(l, h), _mm_unpackhi_epi32(l, h));
}

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
