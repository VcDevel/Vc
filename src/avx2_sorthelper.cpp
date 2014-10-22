/*  This file is part of the Vc library. {{{

    Copyright (C) 2013-2014 Matthias Kretz <kretz@kde.org>

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
