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

#ifndef VC_MIC_DETAIL_H_
#define VC_MIC_DETAIL_H_

#include "casts.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace Detail
{
// zero {{{1
template <typename T> inline T zero();
template <> Vc_INTRINSIC Vc_CONST __m512  zero<__m512 >() { return _mm512_setzero_ps(); }
template <> Vc_INTRINSIC Vc_CONST __m512d zero<__m512d>() { return _mm512_setzero_pd(); }
template <> Vc_INTRINSIC Vc_CONST __m512i zero<__m512i>() { return _mm512_setzero_epi32(); }

// one {{{1
Vc_INTRINSIC Vc_CONST __m512  one( float) { return _mm512_set1_ps(1.f); }
Vc_INTRINSIC Vc_CONST __m512d one(double) { return _mm512_set1_pd(1.); }
Vc_INTRINSIC Vc_CONST __m512i one(   int) { return _mm512_set1_epi32(1); }
Vc_INTRINSIC Vc_CONST __m512i one(  uint) { return _mm512_set1_epi32(1); }
Vc_INTRINSIC Vc_CONST __m512i one( short) { return _mm512_set1_epi32(1); }
Vc_INTRINSIC Vc_CONST __m512i one(ushort) { return _mm512_set1_epi32(1); }
Vc_INTRINSIC Vc_CONST __m512i one( schar) { return _mm512_set1_epi32(1); }
Vc_INTRINSIC Vc_CONST __m512i one( uchar) { return _mm512_set1_epi32(1); }

// xor_ {{{1
Vc_INTRINSIC __m512  xor_(__m512  a, __m512  b) { return MIC::_xor(a, b); }
Vc_INTRINSIC __m512d xor_(__m512d a, __m512d b) { return MIC::_xor(a, b); }
Vc_INTRINSIC __m512i xor_(__m512i a, __m512i b) { return _mm512_xor_si512(a, b); }

// and_ {{{1
Vc_INTRINSIC __m512  and_(__m512  a, __m512  b) { return MIC::_and(a, b); }
Vc_INTRINSIC __m512d and_(__m512d a, __m512d b) { return MIC::_and(a, b); }
Vc_INTRINSIC __m512i and_(__m512i a, __m512i b) { return _mm512_and_si512(a, b); }

// or_ {{{1
Vc_INTRINSIC __m512  or_(__m512  a, __m512  b) { return MIC::_or(a, b); }
Vc_INTRINSIC __m512d or_(__m512d a, __m512d b) { return MIC::_or(a, b); }
Vc_INTRINSIC __m512i or_(__m512i a, __m512i b) { return _mm512_or_si512(a, b); }

// add {{{1
Vc_INTRINSIC __m512  add(__m512  a, __m512  b, float ) { return _mm512_add_ps(a, b); }
Vc_INTRINSIC __m512d add(__m512d a, __m512d b, double) { return _mm512_add_pd(a, b); }
Vc_INTRINSIC __m512i add(__m512i a, __m512i b, int   ) { return _mm512_add_epi32(a, b); }
Vc_INTRINSIC __m512i add(__m512i a, __m512i b, uint  ) { return _mm512_add_epi32(a, b); }
Vc_INTRINSIC __m512i add(__m512i a, __m512i b, short ) { return _mm512_add_epi32(a, b); }
Vc_INTRINSIC __m512i add(__m512i a, __m512i b, ushort) { return _mm512_add_epi32(a, b); }

// sub {{{1
Vc_INTRINSIC __m512  sub(__m512  a, __m512  b, float ) { return _mm512_sub_ps(a, b); }
Vc_INTRINSIC __m512d sub(__m512d a, __m512d b, double) { return _mm512_sub_pd(a, b); }
Vc_INTRINSIC __m512i sub(__m512i a, __m512i b, int   ) { return _mm512_sub_epi32(a, b); }
Vc_INTRINSIC __m512i sub(__m512i a, __m512i b, uint  ) { return _mm512_sub_epi32(a, b); }
Vc_INTRINSIC __m512i sub(__m512i a, __m512i b, short ) { return _mm512_sub_epi32(a, b); }
Vc_INTRINSIC __m512i sub(__m512i a, __m512i b, ushort) { return _mm512_sub_epi32(a, b); }

// mul {{{1
Vc_INTRINSIC __m512  mul(__m512  a, __m512  b, float ) { return _mm512_mul_ps(a, b); }
Vc_INTRINSIC __m512d mul(__m512d a, __m512d b, double) { return _mm512_mul_pd(a, b); }
Vc_INTRINSIC __m512i mul(__m512i a, __m512i b, int   ) { return _mm512_mullo_epi32(a, b); }
Vc_INTRINSIC __m512i mul(__m512i a, __m512i b, uint  ) { return _mm512_mullo_epi32(a, b); }
Vc_INTRINSIC __m512i mul(__m512i a, __m512i b, short ) { return _mm512_mullo_epi32(a, b); }
Vc_INTRINSIC __m512i mul(__m512i a, __m512i b, ushort) { return _mm512_mullo_epi32(a, b); }

// div {{{1
Vc_INTRINSIC __m512  div(__m512  a, __m512  b, float ) { return _mm512_div_ps(a, b); }
Vc_INTRINSIC __m512d div(__m512d a, __m512d b, double) { return _mm512_div_pd(a, b); }
Vc_INTRINSIC __m512i div(__m512i a, __m512i b, int   ) { return _mm512_div_epi32(a, b); }
Vc_INTRINSIC __m512i div(__m512i a, __m512i b, uint  ) { return _mm512_div_epu32(a, b); }
Vc_INTRINSIC __m512i div(__m512i a, __m512i b, short ) { return _mm512_div_epi32(a, b); }
Vc_INTRINSIC __m512i div(__m512i a, __m512i b, ushort) {
    // this specialization is required because overflow is well defined (mod 2^16) for
    // unsigned short, but a / b is not independent of the high bits (in contrast to mul,
    // add, and sub)
    return div(and_(a, MIC::_set1(0xffff)), and_(b, MIC::_set1(0xffff)), uint());
}

// rem {{{1
Vc_INTRINSIC __m512i rem(__m512i a, __m512i b, int   ) { return _mm512_rem_epi32(a, b); }
Vc_INTRINSIC __m512i rem(__m512i a, __m512i b, uint  ) { return _mm512_rem_epu32(a, b); }
Vc_INTRINSIC __m512i rem(__m512i a, __m512i b, short ) { return _mm512_rem_epi32(a, b); }
Vc_INTRINSIC __m512i rem(__m512i a, __m512i b, ushort) { return _mm512_rem_epu32(and_(a, MIC::_set1(0xffff)), and_(b, MIC::_set1(0xffff))); }

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

// abs{{{1
Vc_INTRINSIC __m512d abs(__m512d a, double)
{
    const __m512i absMask =
        _mm512_set_4to16_epi32(0xffffffff, 0x7fffffff, 0xffffffff, 0x7fffffff);
    return MIC::mic_cast<__m512d>(_mm512_and_epi32(MIC::mic_cast<__m512i>(a), absMask));
}
Vc_INTRINSIC __m512 abs(__m512 a, float)
{
    const __m512i absMask = _mm512_set_1to16_epi32(0x7fffffff);
    return MIC::mic_cast<__m512>(_mm512_and_epi32(MIC::mic_cast<__m512i>(a), absMask));
}
Vc_INTRINSIC __m512i abs(__m512i a, int)
{
    const __m512i minusOne = _mm512_set_1to16_epi32(-1);
    return _mm512_mask_mullo_epi32(a, _mm512_cmplt_epi32_mask(a, zero<__m512i>()), a,
                                   minusOne);
}
Vc_INTRINSIC __m512i abs(__m512i a, short)
{
    const __m512i minusOne = _mm512_set_1to16_epi32(-1);
    return _mm512_mask_mullo_epi32(a, _mm512_cmplt_epi32_mask(a, zero<__m512i>()), a,
                                   minusOne);
}

//InterleaveImpl{{{1
template <typename V, int Wt, size_t Sizeof> struct InterleaveImpl;

//}}}1
}  // namespace Detail
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_MIC_DETAIL_H_

// vim: foldmethod=marker
