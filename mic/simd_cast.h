/*  This file is part of the Vc library. {{{
Copyright © 2014 Matthias Kretz <kretz@kde.org>
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
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#ifndef VC_MIC_SIMD_CAST_H_
#define VC_MIC_SIMD_CAST_H_

#ifndef VC_MIC_VECTOR_H
#error "Vc/mic/vector.h needs to be included before Vc/mic/simd_cast.h"
#endif
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace MIC
{
// MIC <-> MIC Vector casts {{{1
// 1 MIC::Vector to 1 MIC::Vector {{{2
#define Vc_CAST__(To__)                                                                  \
    template <typename Return>                                                           \
    Vc_INTRINSIC Vc_CONST enable_if<std::is_same<Return, To__>::value, Return>
// to int_v {{{3
Vc_CAST__(   int_v) simd_cast( short_v x) { return x.data(); }
Vc_CAST__(   int_v) simd_cast(ushort_v x) { return _mm512_and_epi32(x.data(), _mm512_set1_epi32(0xffff)); }
Vc_CAST__(   int_v) simd_cast(  uint_v x) { return x.data(); }
Vc_CAST__(   int_v) simd_cast(double_v x) { return _mm512_cvtfxpnt_roundpd_epi32lo(x.data(), _MM_ROUND_MODE_TOWARD_ZERO); }
Vc_CAST__(   int_v) simd_cast( float_v x) { return _mm512_cvtfxpnt_round_adjustps_epi32(x.data(), _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE); }

// to uint_v {{{3
Vc_CAST__(  uint_v) simd_cast( short_v x) { return x.data(); }
Vc_CAST__(  uint_v) simd_cast(ushort_v x)
{ return _mm512_and_epi32(x.data(), _mm512_set1_epi32(0xffff)); }
Vc_CAST__(  uint_v) simd_cast(   int_v x) { return x.data(); }
Vc_CAST__(  uint_v) simd_cast(double_v x) {
    const auto negative = _mm512_cmplt_pd_mask(x.data(), _mm512_setzero_pd());
    return _mm512_mask_cvtfxpnt_roundpd_epi32lo(
        _mm512_cvtfxpnt_roundpd_epu32lo(x.data(), _MM_ROUND_MODE_TOWARD_ZERO), negative,
        x.data(), _MM_ROUND_MODE_TOWARD_ZERO);
}
Vc_CAST__(  uint_v) simd_cast( float_v x) {
    const auto negative = _mm512_cmplt_ps_mask(x.data(), _mm512_setzero_ps());
    return _mm512_mask_cvtfxpnt_round_adjustps_epi32(
        _mm512_cvtfxpnt_round_adjustps_epu32(x.data(), _MM_ROUND_MODE_TOWARD_ZERO,
                                             _MM_EXPADJ_NONE),
        negative, x.data(), _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE);
}

// to short_v {{{3
Vc_CAST__( short_v) simd_cast(ushort_v x) { return _mm512_srai_epi32(_mm512_slli_epi32(x.data(), 16), 16); }
Vc_CAST__( short_v) simd_cast(   int_v x) { return _mm512_srai_epi32(_mm512_slli_epi32(x.data(), 16), 16); }
Vc_CAST__( short_v) simd_cast(  uint_v x) { return _mm512_srai_epi32(_mm512_slli_epi32(x.data(), 16), 16); }
Vc_CAST__( short_v) simd_cast(double_v x) { return _mm512_cvtfxpnt_roundpd_epi32lo(x.data(), _MM_ROUND_MODE_TOWARD_ZERO); }
Vc_CAST__( short_v) simd_cast( float_v x) { return _mm512_cvtfxpnt_round_adjustps_epi32(x.data(), _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE); }

// to ushort_v {{{3
Vc_CAST__(ushort_v) simd_cast( short_v x) { return x.data(); }
Vc_CAST__(ushort_v) simd_cast(   int_v x) { return x.data(); }
Vc_CAST__(ushort_v) simd_cast(  uint_v x) { return x.data(); }
Vc_CAST__(ushort_v) simd_cast(double_v x) {
    // use conversion to epi32 on purpose here! Conversion to epu32 drops negative inputs.
    // And since we convert to 32bit ints the positive values are all covered.
    return _mm512_cvtfxpnt_roundpd_epi32lo(x.data(), _MM_ROUND_MODE_TOWARD_ZERO);
}
Vc_CAST__(ushort_v) simd_cast( float_v x) {
    // use conversion to epi32 on purpose here! Conversion to epu32 drops negative inputs.
    // And since we convert to 32bit ints the positive values are all covered.
    return _mm512_cvtfxpnt_round_adjustps_epi32(x.data(), _MM_ROUND_MODE_TOWARD_ZERO,
                                                _MM_EXPADJ_NONE);
}
// to float_v {{{3
Vc_CAST__( float_v) simd_cast(   int_v x) {
    return _mm512_cvtfxpnt_round_adjustepi32_ps(x.data(), _MM_FROUND_CUR_DIRECTION,
                                                _MM_EXPADJ_NONE);
}
Vc_CAST__( float_v) simd_cast(  uint_v x) {
    return _mm512_cvtfxpnt_round_adjustepu32_ps(x.data(), _MM_FROUND_CUR_DIRECTION,
                                                _MM_EXPADJ_NONE);
}
Vc_CAST__( float_v) simd_cast( short_v x) { return simd_cast<float_v>(simd_cast< int_v>(x)); }
Vc_CAST__( float_v) simd_cast(ushort_v x) { return simd_cast<float_v>(simd_cast<uint_v>(x)); }
Vc_CAST__( float_v) simd_cast(double_v x) { return _mm512_cvtpd_pslo(x.data()); }
// to double_v {{{3
Vc_CAST__(double_v) simd_cast( float_v x) { return _mm512_cvtpslo_pd(x.data()); }
Vc_CAST__(double_v) simd_cast(   int_v x) { return _mm512_cvtepi32lo_pd(x.data()); }
Vc_CAST__(double_v) simd_cast(  uint_v x) { return _mm512_cvtepu32lo_pd(x.data()); }
Vc_CAST__(double_v) simd_cast( short_v x) { return simd_cast<double_v>(simd_cast< int_v>(x)); }
Vc_CAST__(double_v) simd_cast(ushort_v x) { return simd_cast<double_v>(simd_cast<uint_v>(x)); }
// 2 MIC::Vector to 1 MIC::Vector {{{2
Vc_CAST__(ushort_v) simd_cast(double_v a, double_v b)
{
    return _mm512_mask_permute4f128_epi32(
        _mm512_cvtfxpnt_roundpd_epi32lo(a.data(), _MM_ROUND_MODE_TOWARD_ZERO), 0xff00,
        _mm512_cvtfxpnt_roundpd_epi32lo(b.data(), _MM_ROUND_MODE_TOWARD_ZERO),
        _MM_PERM_BABA);
}
Vc_CAST__( short_v) simd_cast(double_v a, double_v b)
{
    return _mm512_mask_permute4f128_epi32(
        _mm512_cvtfxpnt_roundpd_epi32lo(a.data(), _MM_ROUND_MODE_TOWARD_ZERO), 0xff00,
        _mm512_cvtfxpnt_roundpd_epi32lo(b.data(), _MM_ROUND_MODE_TOWARD_ZERO),
        _MM_PERM_BABA);
}
Vc_CAST__(  uint_v) simd_cast(double_v a, double_v b)
{
    return _mm512_mask_permute4f128_epi32(
        _mm512_cvtfxpnt_roundpd_epi32lo(a.data(), _MM_ROUND_MODE_TOWARD_ZERO), 0xff00,
        _mm512_cvtfxpnt_roundpd_epi32lo(b.data(), _MM_ROUND_MODE_TOWARD_ZERO),
        _MM_PERM_BABA);
}
Vc_CAST__(   int_v) simd_cast(double_v a, double_v b)
{
    return _mm512_mask_permute4f128_epi32(
        _mm512_cvtfxpnt_roundpd_epi32lo(a.data(), _MM_ROUND_MODE_TOWARD_ZERO), 0xff00,
        _mm512_cvtfxpnt_roundpd_epi32lo(b.data(), _MM_ROUND_MODE_TOWARD_ZERO),
        _MM_PERM_BABA);
}
Vc_CAST__( float_v) simd_cast(double_v a, double_v b)
{
    return _mm512_mask_permute4f128_ps(_mm512_cvtpd_pslo(a.data()), 0xff00,
                                       _mm512_cvtpd_pslo(b.data()), _MM_PERM_BABA);
}
#undef Vc_CAST__
// 1 MIC::Vector to 2 MIC::Vector {{{2
#define Vc_CAST__(To__, Offset__)                                                        \
    template <typename Return, int offset>                                               \
    Vc_INTRINSIC Vc_CONST                                                                \
        enable_if<std::is_same<Return, To__>::value&& offset == Offset__, Return>
Vc_CAST__(double_v, 1) simd_cast(ushort_v x) { return simd_cast<double_v>(ushort_v(_mm512_permute4f128_epi32(x.data(), _MM_PERM_DCDC))); }
Vc_CAST__(double_v, 1) simd_cast( short_v x) { return simd_cast<double_v>( short_v(_mm512_permute4f128_epi32(x.data(), _MM_PERM_DCDC))); }
Vc_CAST__(double_v, 1) simd_cast(  uint_v x) { return simd_cast<double_v>(  uint_v(_mm512_permute4f128_epi32(x.data(), _MM_PERM_DCDC))); }
Vc_CAST__(double_v, 1) simd_cast(   int_v x) { return simd_cast<double_v>(   int_v(_mm512_permute4f128_epi32(x.data(), _MM_PERM_DCDC))); }
Vc_CAST__(double_v, 1) simd_cast( float_v x) { return simd_cast<double_v>( float_v(_mm512_permute4f128_ps(x.data(), _MM_PERM_DCDC))); }
#undef Vc_CAST__
template <typename Return, int offset, typename T>
Vc_INTRINSIC Vc_CONST enable_if<(is_vector<Return>::value && offset == 0), Return>
    simd_cast(Vector<T> x)
{ return simd_cast<Return>(x); }
// MIC <-> MIC Mask casts {{{1
// 1 MIC::Mask to 1 MIC::Mask {{{2
template <typename Return, typename M>
Vc_INTRINSIC Vc_CONST
    enable_if<(is_mask<Return>::value&& is_mask<M>::value &&
               !std::is_same<Return, M>::value && Return::Size == M::Size),
              Return>
        simd_cast(M k)
{
    return {k.data()};
}
template <typename Return, typename M>
Vc_INTRINSIC Vc_CONST enable_if<
    (is_mask<Return>::value && is_mask<M>::value && Return::Size != M::Size), Return>
    simd_cast(M k)
{
    return {static_cast<typename Return::MaskType>(_mm512_kand(k.data(), 0xff))};
}
// 2 MIC::Mask to 1 MIC::Mask {{{2
template <typename Return, typename M>
Vc_INTRINSIC Vc_CONST enable_if<
    (is_mask<Return>::value&& is_mask<M>::value&& Return::Size == 2 * M::Size), Return>
    simd_cast(M k0, M k1)
{
    return {_mm512_kmovlhb(k0.data(), k1.data())};
}
// 1 MIC::Mask to 2 MIC::Mask {{{2
template <typename Return, int offset, typename M>
Vc_INTRINSIC Vc_CONST
    enable_if<(is_mask<Return>::value&& is_mask<M>::value&& offset == 0), Return>
        simd_cast(M k)
{
    return simd_cast<Return>(k);
}
template <typename Return, int offset, typename M>
Vc_INTRINSIC Vc_CONST
    enable_if<(is_mask<Return>::value&& is_mask<M>::value&& Return::Size * 2 ==
               M::Size&& offset == 1),
              Return>
        simd_cast(M k)
{
    return {static_cast<typename Return::MaskType>(_mm512_kswapb(k.data(), 0))};
}

}  // namespace MIC
using MIC::simd_cast;
}  // namespace Vc

#include "undomacros.h"

#endif  // VC_MIC_SIMD_CAST_H_

// vim: foldmethod=marker
