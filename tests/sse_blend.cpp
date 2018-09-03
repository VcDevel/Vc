/*  This file is part of the Vc library. {{{
Copyright © 2009-2014 Matthias Kretz <kretz@kde.org>

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

#include "unittest.h"
#include <Vc/sse/intrinsics.h>

TEST(blendpd)
{
    using Vc::SSE::blend_pd;
    __m128d a = _mm_set_pd(11, 10);
    __m128d b = _mm_set_pd(21, 20);

    COMPARE(_mm_movemask_pd(_mm_cmpeq_pd(blend_pd<0x0>(a, b), a)), 0x3);
    COMPARE(_mm_movemask_pd(_mm_cmpeq_pd(blend_pd<0x1>(a, b), _mm_set_pd(11, 20))), 0x3);
    COMPARE(_mm_movemask_pd(_mm_cmpeq_pd(blend_pd<0x2>(a, b), _mm_set_pd(21, 10))), 0x3);
    COMPARE(_mm_movemask_pd(_mm_cmpeq_pd(blend_pd<0x3>(a, b), b)), 0x3);
}

TEST(blendps)
{
    using Vc::SSE::blend_ps;
    __m128 a = _mm_set_ps(13, 12, 11, 10);
    __m128 b = _mm_set_ps(23, 22, 21, 20);

    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend_ps<0x0>(a, b), a)), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend_ps<0x1>(a, b), _mm_set_ps(13, 12, 11, 20))), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend_ps<0x2>(a, b), _mm_set_ps(13, 12, 21, 10))), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend_ps<0x3>(a, b), _mm_set_ps(13, 12, 21, 20))), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend_ps<0x4>(a, b), _mm_set_ps(13, 22, 11, 10))), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend_ps<0x5>(a, b), _mm_set_ps(13, 22, 11, 20))), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend_ps<0x6>(a, b), _mm_set_ps(13, 22, 21, 10))), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend_ps<0x7>(a, b), _mm_set_ps(13, 22, 21, 20))), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend_ps<0x8>(a, b), _mm_set_ps(23, 12, 11, 10))), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend_ps<0x9>(a, b), _mm_set_ps(23, 12, 11, 20))), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend_ps<0xa>(a, b), _mm_set_ps(23, 12, 21, 10))), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend_ps<0xb>(a, b), _mm_set_ps(23, 12, 21, 20))), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend_ps<0xc>(a, b), _mm_set_ps(23, 22, 11, 10))), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend_ps<0xd>(a, b), _mm_set_ps(23, 22, 11, 20))), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend_ps<0xe>(a, b), _mm_set_ps(23, 22, 21, 10))), 0xf);
    COMPARE(_mm_movemask_ps(_mm_cmpeq_ps(blend_ps<0xf>(a, b), b)), 0xf);
}

TEST(blendepi16)
{
    using Vc::SSE::blend_epi16;
    __m128i a = _mm_set_epi16(17, 16, 15, 14, 13, 12, 11, 10);
    __m128i b = _mm_set_epi16(27, 26, 25, 24, 23, 22, 21, 20);

#define CALL_2(_i, code) { enum { i = _i }; code } { enum { i = _i + 1 }; code }
#define CALL_4(_i, code) CALL_2(_i, code) CALL_2(_i + 2, code)
#define CALL_8(_i, code) CALL_4(_i, code) CALL_4(_i + 4, code)
#define CALL_16(_i, code) CALL_8(_i, code) CALL_8(_i + 8, code)
#define CALL_32(_i, code) CALL_16(_i, code) CALL_16(_i + 16, code)
#define CALL_64(_i, code) CALL_32(_i, code) CALL_32(_i + 32, code)
#define CALL_128(_i, code) CALL_64(_i, code) CALL_64(_i + 64, code)
#define CALL_256(code) CALL_128(0, code) CALL_128(128, code)
#define CALL_100(code) CALL_64(0, code) CALL_32(64, code) CALL_4(96, code)

    CALL_256(
        short r[8];
        for (int j = 0; j < 8; ++j) {
            r[j] = j + ((((i >> j) & 1) == 0) ? 10 : 20);
        }
        __m128i reference = _mm_set_epi16(r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0]);
        COMPARE(_mm_movemask_epi8(_mm_cmpeq_epi16(blend_epi16<i>(a, b), reference)), 0xffff);
    )
}
