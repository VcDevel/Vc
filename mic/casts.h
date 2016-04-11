/*  This file is part of the Vc library. {{{
Copyright © 2009-2015 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_MIC_CASTS_H_
#define VC_MIC_CASTS_H_

#include "intrinsics.h"
#include "types.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace MIC
{
// mic_cast {{{1
template <typename T> static Vc_INTRINSIC_L Vc_CONST_L T mic_cast(__m512  v) Vc_INTRINSIC_R Vc_CONST_R;
template <typename T> static Vc_INTRINSIC_L Vc_CONST_L T mic_cast(__m512i v) Vc_INTRINSIC_R Vc_CONST_R;
template <typename T> static Vc_INTRINSIC_L Vc_CONST_L T mic_cast(__m512d v) Vc_INTRINSIC_R Vc_CONST_R;

template <> Vc_INTRINSIC Vc_CONST __m512  mic_cast(__m512  v) { return v; }
template <> Vc_INTRINSIC Vc_CONST __m512  mic_cast(__m512i v) { return _mm512_castsi512_ps(v); }
template <> Vc_INTRINSIC Vc_CONST __m512  mic_cast(__m512d v) { return _mm512_castpd_ps(v); }
template <> Vc_INTRINSIC Vc_CONST __m512i mic_cast(__m512  v) { return _mm512_castps_si512(v); }
template <> Vc_INTRINSIC Vc_CONST __m512i mic_cast(__m512i v) { return v; }
template <> Vc_INTRINSIC Vc_CONST __m512i mic_cast(__m512d v) { return _mm512_castpd_si512(v); }
template <> Vc_INTRINSIC Vc_CONST __m512d mic_cast(__m512  v) { return _mm512_castps_pd(v); }
template <> Vc_INTRINSIC Vc_CONST __m512d mic_cast(__m512i v) { return _mm512_castsi512_pd(v); }
template <> Vc_INTRINSIC Vc_CONST __m512d mic_cast(__m512d v) { return v; }

// convert {{{1
template <typename From, typename To> struct ConvertTag
{
};
template <typename From, typename To>
Vc_INTRINSIC typename VectorTypeHelper<To>::Type convert(
    typename VectorTypeHelper<From>::Type v)
{
    return convert(v, ConvertTag<From, To>());
}

Vc_INTRINSIC __m512d convert(__m512  v, ConvertTag<float , double>) { return _mm512_cvtpslo_pd(v); }
Vc_INTRINSIC __m512i convert(__m512  v, ConvertTag<float , int   >) { return _mm512_cvtfxpnt_round_adjustps_epi32(v, _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE); }
Vc_INTRINSIC __m512i convert(__m512  v, ConvertTag<float , uint  >) {
    // cvtfxpntps2udq converts any negative input to 0
    // but static_cast<uint>(-1.f) == static_cast<uint>(-1)
    // => for negative input use cvtfxpntps2dq instead
    const auto negative = _mm512_cmplt_ps_mask(v, _mm512_setzero_ps());
    return _mm512_mask_cvtfxpnt_round_adjustps_epi32(
        _mm512_cvtfxpnt_round_adjustps_epu32(v, _MM_ROUND_MODE_TOWARD_ZERO,
                                             _MM_EXPADJ_NONE),
        negative, v, _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE);
}
Vc_INTRINSIC __m512i convert(__m512  v, ConvertTag<float , short >) { return _mm512_cvtfxpnt_round_adjustps_epi32(v, _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE); }
Vc_INTRINSIC __m512i convert(__m512  v, ConvertTag<float , ushort>) {
    // use conversion to epi32 on purpose here! Conversion to epu32 drops negative inputs.
    // And since we convert to 32bit ints the positive values are all covered.
    return _mm512_cvtfxpnt_round_adjustps_epi32(v, _MM_ROUND_MODE_TOWARD_ZERO,
                                                _MM_EXPADJ_NONE);
}

Vc_INTRINSIC __m512  convert(__m512d v, ConvertTag<double, float >) { return _mm512_cvtpd_pslo(v); }
Vc_INTRINSIC __m512i convert(__m512d v, ConvertTag<double, int   >) { return _mm512_cvtfxpnt_roundpd_epi32lo(v, _MM_ROUND_MODE_TOWARD_ZERO); }
Vc_INTRINSIC __m512i convert(__m512d v, ConvertTag<double, uint  >) {
    // conversion of negative inputs needs to use _mm512_cvtfxpnt_roundpd_epi32lo
    const auto negative = _mm512_cmplt_pd_mask(v, _mm512_setzero_pd());
    return _mm512_mask_cvtfxpnt_roundpd_epi32lo(
        _mm512_cvtfxpnt_roundpd_epu32lo(v, _MM_ROUND_MODE_TOWARD_ZERO), negative, v,
        _MM_ROUND_MODE_TOWARD_ZERO);
}
Vc_INTRINSIC __m512i convert(__m512d v, ConvertTag<double, short >) { return _mm512_cvtfxpnt_roundpd_epi32lo(v, _MM_ROUND_MODE_TOWARD_ZERO); }
Vc_INTRINSIC __m512i convert(__m512d v, ConvertTag<double, ushort>) {
    // use conversion to epi32 on purpose here! Conversion to epu32 drops negative inputs.
    // And since we convert to 32bit ints the positive values are all covered.
    return _mm512_cvtfxpnt_roundpd_epi32lo(v, _MM_ROUND_MODE_TOWARD_ZERO);
}

Vc_INTRINSIC __m512  convert(__m512i v, ConvertTag<int   , float >) { return _mm512_cvtfxpnt_round_adjustepi32_ps(v, _MM_FROUND_CUR_DIRECTION, _MM_EXPADJ_NONE); }
Vc_INTRINSIC __m512d convert(__m512i v, ConvertTag<int   , double>) { return _mm512_cvtepi32lo_pd(v); }
Vc_INTRINSIC __m512i convert(__m512i v, ConvertTag<int   , uint  >) { return v; }
Vc_INTRINSIC __m512i convert(__m512i v, ConvertTag<int   , short >) { return v; }
Vc_INTRINSIC __m512i convert(__m512i v, ConvertTag<int   , ushort>) { return v; }

Vc_INTRINSIC __m512  convert(__m512i v, ConvertTag<uint  , float >) { return _mm512_cvtfxpnt_round_adjustepu32_ps(v, _MM_FROUND_CUR_DIRECTION, _MM_EXPADJ_NONE); }
Vc_INTRINSIC __m512d convert(__m512i v, ConvertTag<uint  , double>) { return _mm512_cvtepu32lo_pd(v); }
Vc_INTRINSIC __m512i convert(__m512i v, ConvertTag<uint  , int   >) { return v; }
Vc_INTRINSIC __m512i convert(__m512i v, ConvertTag<uint  , short >) { return v; }
Vc_INTRINSIC __m512i convert(__m512i v, ConvertTag<uint  , ushort>) { return v; }

Vc_INTRINSIC __m512  convert(__m512i v, ConvertTag<short , float >) { return _mm512_cvtfxpnt_round_adjustepi32_ps(v, _MM_FROUND_CUR_DIRECTION, _MM_EXPADJ_NONE); }
Vc_INTRINSIC __m512d convert(__m512i v, ConvertTag<short , double>) { return _mm512_cvtepi32lo_pd(v); }
Vc_INTRINSIC __m512i convert(__m512i v, ConvertTag<short , int   >) { return v; }
Vc_INTRINSIC __m512i convert(__m512i v, ConvertTag<short , uint  >) { return v; }
Vc_INTRINSIC __m512i convert(__m512i v, ConvertTag<short , ushort>) { return v; }

Vc_INTRINSIC __m512  convert(__m512i v, ConvertTag<ushort, float >) { return _mm512_cvtfxpnt_round_adjustepu32_ps(v, _MM_FROUND_CUR_DIRECTION, _MM_EXPADJ_NONE); }
Vc_INTRINSIC __m512d convert(__m512i v, ConvertTag<ushort, double>) { return _mm512_cvtepu32lo_pd(v); }
Vc_INTRINSIC __m512i convert(__m512i v, ConvertTag<ushort, int   >) { return v; }
Vc_INTRINSIC __m512i convert(__m512i v, ConvertTag<ushort, uint  >) { return v; }
Vc_INTRINSIC __m512i convert(__m512i v, ConvertTag<ushort, short >) {
    const auto negative = _mm512_cmpgt_epu32_mask(v, _mm512_set1_epi32(0x7fff));
    return _mm512_mask_or_epi32(v, negative, v, _mm512_set1_epi32(0xffff0000u));
}

//}}}1
}  // namespace MIC
}  // namespace Vc

#endif // VC_MIC_CASTS_H_

// vim: foldmethod=marker
