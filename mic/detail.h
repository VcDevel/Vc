/*  This file is part of the Vc library. {{{
Copyright © 2016 Matthias Kretz <kretz@kde.org>
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

#ifndef VC_MIC_DETAIL_H_
#define VC_MIC_DETAIL_H_

#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Detail
{
// horizontal add{{{1
Vc_INTRINSIC  float add(__m512  a,  float) { return _mm512_reduce_add_ps(a); }
Vc_INTRINSIC double add(__m512d a, double) { return _mm512_reduce_add_pd(a); }
Vc_INTRINSIC    int add(__m512i a,    int) { return _mm512_reduce_add_epi32(a); }
Vc_INTRINSIC   uint add(__m512i a,   uint) { return _mm512_reduce_add_epi32(a); }
Vc_INTRINSIC  short add(__m512i a,  short) { return _mm512_reduce_add_epi32(a); }
Vc_INTRINSIC ushort add(__m512i a, ushort) { return _mm512_reduce_add_epi32(a); }

// horizontal mul{{{1
Vc_INTRINSIC  float mul(__m512  a,  float) { return _mm512_reduce_mul_ps(a); }
Vc_INTRINSIC double mul(__m512d a, double) { return _mm512_reduce_mul_pd(a); }
Vc_INTRINSIC    int mul(__m512i a,    int) { return _mm512_reduce_mul_epi32(a); }
Vc_INTRINSIC   uint mul(__m512i a,   uint) { return _mm512_reduce_mul_epi32(a); }
Vc_INTRINSIC  short mul(__m512i a,  short) { return _mm512_reduce_mul_epi32(a); }
Vc_INTRINSIC ushort mul(__m512i a, ushort) { return _mm512_reduce_mul_epi32(a); }

// horizontal min{{{1
Vc_INTRINSIC  float min(__m512  a,  float) { return _mm512_reduce_min_ps(a); }
Vc_INTRINSIC double min(__m512d a, double) { return _mm512_reduce_min_pd(a); }
Vc_INTRINSIC    int min(__m512i a,    int) { return _mm512_reduce_min_epi32(a); }
Vc_INTRINSIC   uint min(__m512i a,   uint) { return _mm512_reduce_min_epi32(a); }
Vc_INTRINSIC  short min(__m512i a,  short) { return _mm512_reduce_min_epi32(a); }
Vc_INTRINSIC ushort min(__m512i a, ushort) { return _mm512_reduce_min_epi32(a); }

// horizontal max{{{1
Vc_INTRINSIC  float max(__m512  a,  float) { return _mm512_reduce_max_ps(a); }
Vc_INTRINSIC double max(__m512d a, double) { return _mm512_reduce_max_pd(a); }
Vc_INTRINSIC    int max(__m512i a,    int) { return _mm512_reduce_max_epi32(a); }
Vc_INTRINSIC   uint max(__m512i a,   uint) { return _mm512_reduce_max_epi32(a); }
Vc_INTRINSIC  short max(__m512i a,  short) { return _mm512_reduce_max_epi32(a); }
Vc_INTRINSIC ushort max(__m512i a, ushort) { return _mm512_reduce_max_epi32(a); }

//}}}1
}  // namespace Detail
}  // namespace Vc

#endif  // VC_MIC_DETAIL_H_

// vim: foldmethod=marker
