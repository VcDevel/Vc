/*  This file is part of the Vc library. {{{
Copyright © 2014-2015 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_MIC_SIMD_CAST_H_
#define VC_MIC_SIMD_CAST_H_

#ifndef VC_MIC_VECTOR_H_
#error "Vc/mic/vector.h needs to be included before Vc/mic/simd_cast.h"
#endif
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace MIC
{
// MIC <-> MIC Vector casts {{{1
// 1 MIC::Vector to 1 MIC::Vector {{{2
#define Vc_CAST_(To_)                                                                    \
    template <typename Return>                                                           \
    Vc_INTRINSIC Vc_CONST enable_if<std::is_same<Return, To_>::value, Return>
// to int_v {{{3
Vc_CAST_(   int_v) simd_cast( short_v x) { return x.data(); }
Vc_CAST_(   int_v) simd_cast(ushort_v x) { return _mm512_and_epi32(x.data(), _mm512_set1_epi32(0xffff)); }
Vc_CAST_(   int_v) simd_cast(  uint_v x) { return x.data(); }
Vc_CAST_(   int_v) simd_cast(double_v x) { return _mm512_cvtfxpnt_roundpd_epi32lo(x.data(), _MM_ROUND_MODE_TOWARD_ZERO); }
Vc_CAST_(   int_v) simd_cast( float_v x) { return _mm512_cvtfxpnt_round_adjustps_epi32(x.data(), _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE); }

// to uint_v {{{3
Vc_CAST_(  uint_v) simd_cast( short_v x) { return x.data(); }
Vc_CAST_(  uint_v) simd_cast(ushort_v x)
{ return _mm512_and_epi32(x.data(), _mm512_set1_epi32(0xffff)); }
Vc_CAST_(  uint_v) simd_cast(   int_v x) { return x.data(); }
Vc_CAST_(  uint_v) simd_cast(double_v x) {
    const auto negative = _mm512_cmplt_pd_mask(x.data(), _mm512_setzero_pd());
    return _mm512_mask_cvtfxpnt_roundpd_epi32lo(
        _mm512_cvtfxpnt_roundpd_epu32lo(x.data(), _MM_ROUND_MODE_TOWARD_ZERO), negative,
        x.data(), _MM_ROUND_MODE_TOWARD_ZERO);
}
Vc_CAST_(  uint_v) simd_cast( float_v x) {
    const auto negative = _mm512_cmplt_ps_mask(x.data(), _mm512_setzero_ps());
    return _mm512_mask_cvtfxpnt_round_adjustps_epi32(
        _mm512_cvtfxpnt_round_adjustps_epu32(x.data(), _MM_ROUND_MODE_TOWARD_ZERO,
                                             _MM_EXPADJ_NONE),
        negative, x.data(), _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE);
}

// to short_v {{{3
Vc_CAST_( short_v) simd_cast(ushort_v x) { return _mm512_srai_epi32(_mm512_slli_epi32(x.data(), 16), 16); }
Vc_CAST_( short_v) simd_cast(   int_v x) { return _mm512_srai_epi32(_mm512_slli_epi32(x.data(), 16), 16); }
Vc_CAST_( short_v) simd_cast(  uint_v x) { return _mm512_srai_epi32(_mm512_slli_epi32(x.data(), 16), 16); }
Vc_CAST_( short_v) simd_cast(double_v x) { return _mm512_cvtfxpnt_roundpd_epi32lo(x.data(), _MM_ROUND_MODE_TOWARD_ZERO); }
Vc_CAST_( short_v) simd_cast( float_v x) { return _mm512_cvtfxpnt_round_adjustps_epi32(x.data(), _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE); }

// to ushort_v {{{3
Vc_CAST_(ushort_v) simd_cast( short_v x) { return x.data(); }
Vc_CAST_(ushort_v) simd_cast(   int_v x) { return x.data(); }
Vc_CAST_(ushort_v) simd_cast(  uint_v x) { return x.data(); }
Vc_CAST_(ushort_v) simd_cast(double_v x) {
    // use conversion to epi32 on purpose here! Conversion to epu32 drops negative inputs.
    // And since we convert to 32bit ints the positive values are all covered.
    return _mm512_cvtfxpnt_roundpd_epi32lo(x.data(), _MM_ROUND_MODE_TOWARD_ZERO);
}
Vc_CAST_(ushort_v) simd_cast( float_v x) {
    // use conversion to epi32 on purpose here! Conversion to epu32 drops negative inputs.
    // And since we convert to 32bit ints the positive values are all covered.
    return _mm512_cvtfxpnt_round_adjustps_epi32(x.data(), _MM_ROUND_MODE_TOWARD_ZERO,
                                                _MM_EXPADJ_NONE);
}
// to float_v {{{3
Vc_CAST_( float_v) simd_cast(   int_v x) {
    return _mm512_cvtfxpnt_round_adjustepi32_ps(x.data(), _MM_FROUND_CUR_DIRECTION,
                                                _MM_EXPADJ_NONE);
}
Vc_CAST_( float_v) simd_cast(  uint_v x) {
    return _mm512_cvtfxpnt_round_adjustepu32_ps(x.data(), _MM_FROUND_CUR_DIRECTION,
                                                _MM_EXPADJ_NONE);
}
Vc_CAST_( float_v) simd_cast( short_v x) { return simd_cast<float_v>(simd_cast< int_v>(x)); }
Vc_CAST_( float_v) simd_cast(ushort_v x) { return simd_cast<float_v>(simd_cast<uint_v>(x)); }
Vc_CAST_( float_v) simd_cast(double_v x) { return _mm512_cvtpd_pslo(x.data()); }
// to double_v {{{3
Vc_CAST_(double_v) simd_cast( float_v x) { return _mm512_cvtpslo_pd(x.data()); }
Vc_CAST_(double_v) simd_cast(   int_v x) { return _mm512_cvtepi32lo_pd(x.data()); }
Vc_CAST_(double_v) simd_cast(  uint_v x) { return _mm512_cvtepu32lo_pd(x.data()); }
Vc_CAST_(double_v) simd_cast( short_v x) { return simd_cast<double_v>(simd_cast< int_v>(x)); }
Vc_CAST_(double_v) simd_cast(ushort_v x) { return simd_cast<double_v>(simd_cast<uint_v>(x)); }
// 2 MIC::Vector to 1 MIC::Vector {{{2
Vc_CAST_(ushort_v) simd_cast(double_v a, double_v b)
{
    return _mm512_mask_permute4f128_epi32(
        _mm512_cvtfxpnt_roundpd_epi32lo(a.data(), _MM_ROUND_MODE_TOWARD_ZERO), 0xff00,
        _mm512_cvtfxpnt_roundpd_epi32lo(b.data(), _MM_ROUND_MODE_TOWARD_ZERO),
        _MM_PERM_BABA);
}
Vc_CAST_( short_v) simd_cast(double_v a, double_v b)
{
    return _mm512_mask_permute4f128_epi32(
        _mm512_cvtfxpnt_roundpd_epi32lo(a.data(), _MM_ROUND_MODE_TOWARD_ZERO), 0xff00,
        _mm512_cvtfxpnt_roundpd_epi32lo(b.data(), _MM_ROUND_MODE_TOWARD_ZERO),
        _MM_PERM_BABA);
}
Vc_CAST_(  uint_v) simd_cast(double_v a, double_v b)
{
    return _mm512_mask_permute4f128_epi32(
        _mm512_cvtfxpnt_roundpd_epu32lo(a.data(), _MM_ROUND_MODE_TOWARD_ZERO), 0xff00,
        _mm512_cvtfxpnt_roundpd_epu32lo(b.data(), _MM_ROUND_MODE_TOWARD_ZERO),
        _MM_PERM_BABA);
}
Vc_CAST_(   int_v) simd_cast(double_v a, double_v b)
{
    return _mm512_mask_permute4f128_epi32(
        _mm512_cvtfxpnt_roundpd_epi32lo(a.data(), _MM_ROUND_MODE_TOWARD_ZERO), 0xff00,
        _mm512_cvtfxpnt_roundpd_epi32lo(b.data(), _MM_ROUND_MODE_TOWARD_ZERO),
        _MM_PERM_BABA);
}
Vc_CAST_( float_v) simd_cast(double_v a, double_v b)
{
    return _mm512_mask_permute4f128_ps(_mm512_cvtpd_pslo(a.data()), 0xff00,
                                       _mm512_cvtpd_pslo(b.data()), _MM_PERM_BABA);
}
#undef Vc_CAST_
// 1 MIC::Vector to 2 MIC::Vector {{{2
#define Vc_CAST_(To_, Offset_)                                                           \
    template <typename Return, int offset>                                               \
    Vc_INTRINSIC Vc_CONST                                                                \
        enable_if<std::is_same<Return, To_>::value && offset == Offset_, Return>
Vc_CAST_(double_v, 1) simd_cast(ushort_v x) { return simd_cast<double_v>(ushort_v(_mm512_permute4f128_epi32(x.data(), _MM_PERM_DCDC))); }
Vc_CAST_(double_v, 1) simd_cast( short_v x) { return simd_cast<double_v>( short_v(_mm512_permute4f128_epi32(x.data(), _MM_PERM_DCDC))); }
Vc_CAST_(double_v, 1) simd_cast(  uint_v x) { return simd_cast<double_v>(  uint_v(_mm512_permute4f128_epi32(x.data(), _MM_PERM_DCDC))); }
Vc_CAST_(double_v, 1) simd_cast(   int_v x) { return simd_cast<double_v>(   int_v(_mm512_permute4f128_epi32(x.data(), _MM_PERM_DCDC))); }
Vc_CAST_(double_v, 1) simd_cast( float_v x) { return simd_cast<double_v>( float_v(_mm512_permute4f128_ps(x.data(), _MM_PERM_DCDC))); }
#undef Vc_CAST_
template <typename Return, int offset, typename T>
Vc_INTRINSIC Vc_CONST enable_if<(is_vector<Return>::value && offset == 0), Return>
    simd_cast(Vector<T> x)
{ return simd_cast<Return>(x); }
// MIC <-> Scalar Vector casts {{{1
// 1 MIC::Vector to 1 Scalar::Vector {{{2
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST enable_if<(Scalar::is_vector<Return>::value), Return> simd_cast(
    MIC::Vector<T> x)
{
    return {static_cast<typename Return::value_type>(x[0])};
}
// 1 Scalar::Vector to 1 MIC::Vector {{{2
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST enable_if<(MIC::is_vector<Return>::value), Return> simd_cast(
    Scalar::Vector<T> x)
{
    Return r{};
    r[0] = static_cast<typename Return::value_type>(x[0]);
    return r;
}
// 2 Scalar::Vector to 1 MIC::Vector {{{2
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST enable_if<(MIC::is_vector<Return>::value), Return> simd_cast(
    Scalar::Vector<T> x0, Scalar::Vector<T> x1)
{
    Return r{};
    r[0] = static_cast<typename Return::value_type>(x0[0]);
    r[1] = static_cast<typename Return::value_type>(x1[0]);
    return r;
}
// 3 Scalar::Vector to 1 MIC::Vector {{{2
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST enable_if<(MIC::is_vector<Return>::value), Return> simd_cast(
    Scalar::Vector<T> x0, Scalar::Vector<T> x1, Scalar::Vector<T> x2)
{
    Return r{};
    r[0] = static_cast<typename Return::value_type>(x0[0]);
    r[1] = static_cast<typename Return::value_type>(x1[0]);
    r[2] = static_cast<typename Return::value_type>(x2[0]);
    return r;
}
// 4 Scalar::Vector to 1 MIC::Vector {{{2
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST enable_if<(MIC::is_vector<Return>::value), Return> simd_cast(
    Scalar::Vector<T> x0, Scalar::Vector<T> x1, Scalar::Vector<T> x2,
    Scalar::Vector<T> x3)
{
    Return r{};
    r[0] = static_cast<typename Return::value_type>(x0[0]);
    r[1] = static_cast<typename Return::value_type>(x1[0]);
    r[2] = static_cast<typename Return::value_type>(x2[0]);
    r[3] = static_cast<typename Return::value_type>(x3[0]);
    return r;
}
// 5 Scalar::Vector to 1 MIC::Vector {{{2
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST enable_if<(MIC::is_vector<Return>::value), Return> simd_cast(
    Scalar::Vector<T> x0, Scalar::Vector<T> x1, Scalar::Vector<T> x2,
    Scalar::Vector<T> x3, Scalar::Vector<T> x4)
{
    Return r{};
    r[0] = static_cast<typename Return::value_type>(x0[0]);
    r[1] = static_cast<typename Return::value_type>(x1[0]);
    r[2] = static_cast<typename Return::value_type>(x2[0]);
    r[3] = static_cast<typename Return::value_type>(x3[0]);
    r[4] = static_cast<typename Return::value_type>(x4[0]);
    return r;
}
// 6 Scalar::Vector to 1 MIC::Vector {{{2
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST enable_if<(MIC::is_vector<Return>::value), Return> simd_cast(
    Scalar::Vector<T> x0, Scalar::Vector<T> x1, Scalar::Vector<T> x2,
    Scalar::Vector<T> x3, Scalar::Vector<T> x4, Scalar::Vector<T> x5)
{
    Return r{};
    r[0] = static_cast<typename Return::value_type>(x0[0]);
    r[1] = static_cast<typename Return::value_type>(x1[0]);
    r[2] = static_cast<typename Return::value_type>(x2[0]);
    r[3] = static_cast<typename Return::value_type>(x3[0]);
    r[4] = static_cast<typename Return::value_type>(x4[0]);
    r[5] = static_cast<typename Return::value_type>(x5[0]);
    return r;
}
// 7 Scalar::Vector to 1 MIC::Vector {{{2
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST enable_if<(MIC::is_vector<Return>::value), Return> simd_cast(
    Scalar::Vector<T> x0, Scalar::Vector<T> x1, Scalar::Vector<T> x2,
    Scalar::Vector<T> x3, Scalar::Vector<T> x4, Scalar::Vector<T> x5,
    Scalar::Vector<T> x6)
{
    Return r{};
    r[0] = static_cast<typename Return::value_type>(x0[0]);
    r[1] = static_cast<typename Return::value_type>(x1[0]);
    r[2] = static_cast<typename Return::value_type>(x2[0]);
    r[3] = static_cast<typename Return::value_type>(x3[0]);
    r[4] = static_cast<typename Return::value_type>(x4[0]);
    r[5] = static_cast<typename Return::value_type>(x5[0]);
    r[6] = static_cast<typename Return::value_type>(x6[0]);
    return r;
}
// 8 Scalar::Vector to 1 MIC::Vector {{{2
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST enable_if<(MIC::is_vector<Return>::value), Return> simd_cast(
    Scalar::Vector<T> x0, Scalar::Vector<T> x1, Scalar::Vector<T> x2,
    Scalar::Vector<T> x3, Scalar::Vector<T> x4, Scalar::Vector<T> x5,
    Scalar::Vector<T> x6, Scalar::Vector<T> x7)
{
    Memory<Return, Return::Size> m;
    m.setZero();
    m[0] = static_cast<typename Return::value_type>(x0[0]);
    m[1] = static_cast<typename Return::value_type>(x1[0]);
    m[2] = static_cast<typename Return::value_type>(x2[0]);
    m[3] = static_cast<typename Return::value_type>(x3[0]);
    m[4] = static_cast<typename Return::value_type>(x4[0]);
    m[5] = static_cast<typename Return::value_type>(x5[0]);
    m[6] = static_cast<typename Return::value_type>(x6[0]);
    m[7] = static_cast<typename Return::value_type>(x7[0]);
    return m.firstVector();
}
// 9 Scalar::Vector to 1 MIC::Vector {{{2
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST
    enable_if<(MIC::is_vector<Return>::value && Return::Size == 16), Return>
    simd_cast(Scalar::Vector<T> x0, Scalar::Vector<T> x1, Scalar::Vector<T> x2,
              Scalar::Vector<T> x3, Scalar::Vector<T> x4, Scalar::Vector<T> x5,
              Scalar::Vector<T> x6, Scalar::Vector<T> x7, Scalar::Vector<T> x8)
{
    Memory<Return, Return::Size> m;
    m.setZero();
    m[ 0] = static_cast<typename Return::value_type>(x0[0]);
    m[ 1] = static_cast<typename Return::value_type>(x1[0]);
    m[ 2] = static_cast<typename Return::value_type>(x2[0]);
    m[ 3] = static_cast<typename Return::value_type>(x3[0]);
    m[ 4] = static_cast<typename Return::value_type>(x4[0]);
    m[ 5] = static_cast<typename Return::value_type>(x5[0]);
    m[ 6] = static_cast<typename Return::value_type>(x6[0]);
    m[ 7] = static_cast<typename Return::value_type>(x7[0]);
    m[ 8] = static_cast<typename Return::value_type>(x8[0]);
    return m.firstVector();
}
// 10 Scalar::Vector to 1 MIC::Vector {{{2
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST
    enable_if<(MIC::is_vector<Return>::value && Return::Size == 16), Return>
    simd_cast(Scalar::Vector<T> x0, Scalar::Vector<T> x1, Scalar::Vector<T> x2,
              Scalar::Vector<T> x3, Scalar::Vector<T> x4, Scalar::Vector<T> x5,
              Scalar::Vector<T> x6, Scalar::Vector<T> x7, Scalar::Vector<T> x8,
              Scalar::Vector<T> x9)
{
    Memory<Return, Return::Size> m;
    m.setZero();
    m[ 0] = static_cast<typename Return::value_type>(x0[0]);
    m[ 1] = static_cast<typename Return::value_type>(x1[0]);
    m[ 2] = static_cast<typename Return::value_type>(x2[0]);
    m[ 3] = static_cast<typename Return::value_type>(x3[0]);
    m[ 4] = static_cast<typename Return::value_type>(x4[0]);
    m[ 5] = static_cast<typename Return::value_type>(x5[0]);
    m[ 6] = static_cast<typename Return::value_type>(x6[0]);
    m[ 7] = static_cast<typename Return::value_type>(x7[0]);
    m[ 8] = static_cast<typename Return::value_type>(x8[0]);
    m[ 9] = static_cast<typename Return::value_type>(x9[0]);
    return m.firstVector();
}
// 11 Scalar::Vector to 1 MIC::Vector {{{2
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST
    enable_if<(MIC::is_vector<Return>::value && Return::Size == 16), Return>
    simd_cast(Scalar::Vector<T> x0, Scalar::Vector<T> x1, Scalar::Vector<T> x2,
              Scalar::Vector<T> x3, Scalar::Vector<T> x4, Scalar::Vector<T> x5,
              Scalar::Vector<T> x6, Scalar::Vector<T> x7, Scalar::Vector<T> x8,
              Scalar::Vector<T> x9, Scalar::Vector<T> x10)
{
    Memory<Return, Return::Size> m;
    m.setZero();
    m[ 0] = static_cast<typename Return::value_type>(x0[0]);
    m[ 1] = static_cast<typename Return::value_type>(x1[0]);
    m[ 2] = static_cast<typename Return::value_type>(x2[0]);
    m[ 3] = static_cast<typename Return::value_type>(x3[0]);
    m[ 4] = static_cast<typename Return::value_type>(x4[0]);
    m[ 5] = static_cast<typename Return::value_type>(x5[0]);
    m[ 6] = static_cast<typename Return::value_type>(x6[0]);
    m[ 7] = static_cast<typename Return::value_type>(x7[0]);
    m[ 8] = static_cast<typename Return::value_type>(x8[0]);
    m[ 9] = static_cast<typename Return::value_type>(x9[0]);
    m[10] = static_cast<typename Return::value_type>(x10[0]);
    return m.firstVector();
}
// 12 Scalar::Vector to 1 MIC::Vector {{{2
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST
    enable_if<(MIC::is_vector<Return>::value && Return::Size == 16), Return>
    simd_cast(Scalar::Vector<T> x0, Scalar::Vector<T> x1, Scalar::Vector<T> x2,
              Scalar::Vector<T> x3, Scalar::Vector<T> x4, Scalar::Vector<T> x5,
              Scalar::Vector<T> x6, Scalar::Vector<T> x7, Scalar::Vector<T> x8,
              Scalar::Vector<T> x9, Scalar::Vector<T> x10, Scalar::Vector<T> x11)
{
    Memory<Return, Return::Size> m;
    m.setZero();
    m[ 0] = static_cast<typename Return::value_type>(x0[0]);
    m[ 1] = static_cast<typename Return::value_type>(x1[0]);
    m[ 2] = static_cast<typename Return::value_type>(x2[0]);
    m[ 3] = static_cast<typename Return::value_type>(x3[0]);
    m[ 4] = static_cast<typename Return::value_type>(x4[0]);
    m[ 5] = static_cast<typename Return::value_type>(x5[0]);
    m[ 6] = static_cast<typename Return::value_type>(x6[0]);
    m[ 7] = static_cast<typename Return::value_type>(x7[0]);
    m[ 8] = static_cast<typename Return::value_type>(x8[0]);
    m[ 9] = static_cast<typename Return::value_type>(x9[0]);
    m[10] = static_cast<typename Return::value_type>(x10[0]);
    m[11] = static_cast<typename Return::value_type>(x11[0]);
    return m.firstVector();
}
// 13 Scalar::Vector to 1 MIC::Vector {{{2
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST
    enable_if<(MIC::is_vector<Return>::value && Return::Size == 16), Return>
    simd_cast(Scalar::Vector<T> x0, Scalar::Vector<T> x1, Scalar::Vector<T> x2,
              Scalar::Vector<T> x3, Scalar::Vector<T> x4, Scalar::Vector<T> x5,
              Scalar::Vector<T> x6, Scalar::Vector<T> x7, Scalar::Vector<T> x8,
              Scalar::Vector<T> x9, Scalar::Vector<T> x10, Scalar::Vector<T> x11,
              Scalar::Vector<T> x12)
{
    Memory<Return, Return::Size> m;
    m.setZero();
    m[ 0] = static_cast<typename Return::value_type>(x0[0]);
    m[ 1] = static_cast<typename Return::value_type>(x1[0]);
    m[ 2] = static_cast<typename Return::value_type>(x2[0]);
    m[ 3] = static_cast<typename Return::value_type>(x3[0]);
    m[ 4] = static_cast<typename Return::value_type>(x4[0]);
    m[ 5] = static_cast<typename Return::value_type>(x5[0]);
    m[ 6] = static_cast<typename Return::value_type>(x6[0]);
    m[ 7] = static_cast<typename Return::value_type>(x7[0]);
    m[ 8] = static_cast<typename Return::value_type>(x8[0]);
    m[ 9] = static_cast<typename Return::value_type>(x9[0]);
    m[10] = static_cast<typename Return::value_type>(x10[0]);
    m[11] = static_cast<typename Return::value_type>(x11[0]);
    m[12] = static_cast<typename Return::value_type>(x12[0]);
    return m.firstVector();
}
// 14 Scalar::Vector to 1 MIC::Vector {{{2
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST
    enable_if<(MIC::is_vector<Return>::value && Return::Size == 16), Return>
    simd_cast(Scalar::Vector<T> x0, Scalar::Vector<T> x1, Scalar::Vector<T> x2,
              Scalar::Vector<T> x3, Scalar::Vector<T> x4, Scalar::Vector<T> x5,
              Scalar::Vector<T> x6, Scalar::Vector<T> x7, Scalar::Vector<T> x8,
              Scalar::Vector<T> x9, Scalar::Vector<T> x10, Scalar::Vector<T> x11,
              Scalar::Vector<T> x12, Scalar::Vector<T> x13)
{
    Memory<Return, Return::Size> m;
    m.setZero();
    m[ 0] = static_cast<typename Return::value_type>(x0[0]);
    m[ 1] = static_cast<typename Return::value_type>(x1[0]);
    m[ 2] = static_cast<typename Return::value_type>(x2[0]);
    m[ 3] = static_cast<typename Return::value_type>(x3[0]);
    m[ 4] = static_cast<typename Return::value_type>(x4[0]);
    m[ 5] = static_cast<typename Return::value_type>(x5[0]);
    m[ 6] = static_cast<typename Return::value_type>(x6[0]);
    m[ 7] = static_cast<typename Return::value_type>(x7[0]);
    m[ 8] = static_cast<typename Return::value_type>(x8[0]);
    m[ 9] = static_cast<typename Return::value_type>(x9[0]);
    m[10] = static_cast<typename Return::value_type>(x10[0]);
    m[11] = static_cast<typename Return::value_type>(x11[0]);
    m[12] = static_cast<typename Return::value_type>(x12[0]);
    m[13] = static_cast<typename Return::value_type>(x13[0]);
    return m.firstVector();
}
// 15 Scalar::Vector to 1 MIC::Vector {{{2
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST
    enable_if<(MIC::is_vector<Return>::value && Return::Size == 16), Return>
    simd_cast(Scalar::Vector<T> x0, Scalar::Vector<T> x1, Scalar::Vector<T> x2,
              Scalar::Vector<T> x3, Scalar::Vector<T> x4, Scalar::Vector<T> x5,
              Scalar::Vector<T> x6, Scalar::Vector<T> x7, Scalar::Vector<T> x8,
              Scalar::Vector<T> x9, Scalar::Vector<T> x10, Scalar::Vector<T> x11,
              Scalar::Vector<T> x12, Scalar::Vector<T> x13, Scalar::Vector<T> x14)
{
    Memory<Return, Return::Size> m;
    m.setZero();
    m[ 0] = static_cast<typename Return::value_type>(x0[0]);
    m[ 1] = static_cast<typename Return::value_type>(x1[0]);
    m[ 2] = static_cast<typename Return::value_type>(x2[0]);
    m[ 3] = static_cast<typename Return::value_type>(x3[0]);
    m[ 4] = static_cast<typename Return::value_type>(x4[0]);
    m[ 5] = static_cast<typename Return::value_type>(x5[0]);
    m[ 6] = static_cast<typename Return::value_type>(x6[0]);
    m[ 7] = static_cast<typename Return::value_type>(x7[0]);
    m[ 8] = static_cast<typename Return::value_type>(x8[0]);
    m[ 9] = static_cast<typename Return::value_type>(x9[0]);
    m[10] = static_cast<typename Return::value_type>(x10[0]);
    m[11] = static_cast<typename Return::value_type>(x11[0]);
    m[12] = static_cast<typename Return::value_type>(x12[0]);
    m[13] = static_cast<typename Return::value_type>(x13[0]);
    m[14] = static_cast<typename Return::value_type>(x14[0]);
    return m.firstVector();
}
// 16 Scalar::Vector to 1 MIC::Vector {{{2
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST
    enable_if<(MIC::is_vector<Return>::value && Return::Size == 16), Return>
    simd_cast(Scalar::Vector<T> x0, Scalar::Vector<T> x1, Scalar::Vector<T> x2,
              Scalar::Vector<T> x3, Scalar::Vector<T> x4, Scalar::Vector<T> x5,
              Scalar::Vector<T> x6, Scalar::Vector<T> x7, Scalar::Vector<T> x8,
              Scalar::Vector<T> x9, Scalar::Vector<T> x10, Scalar::Vector<T> x11,
              Scalar::Vector<T> x12, Scalar::Vector<T> x13, Scalar::Vector<T> x14,
              Scalar::Vector<T> x15)
{
    Memory<Return, Return::Size> m;
    m.setZero();
    m[ 0] = static_cast<typename Return::value_type>(x0[0]);
    m[ 1] = static_cast<typename Return::value_type>(x1[0]);
    m[ 2] = static_cast<typename Return::value_type>(x2[0]);
    m[ 3] = static_cast<typename Return::value_type>(x3[0]);
    m[ 4] = static_cast<typename Return::value_type>(x4[0]);
    m[ 5] = static_cast<typename Return::value_type>(x5[0]);
    m[ 6] = static_cast<typename Return::value_type>(x6[0]);
    m[ 7] = static_cast<typename Return::value_type>(x7[0]);
    m[ 8] = static_cast<typename Return::value_type>(x8[0]);
    m[ 9] = static_cast<typename Return::value_type>(x9[0]);
    m[10] = static_cast<typename Return::value_type>(x10[0]);
    m[11] = static_cast<typename Return::value_type>(x11[0]);
    m[12] = static_cast<typename Return::value_type>(x12[0]);
    m[13] = static_cast<typename Return::value_type>(x13[0]);
    m[14] = static_cast<typename Return::value_type>(x14[0]);
    m[15] = static_cast<typename Return::value_type>(x15[0]);
    return m.firstVector();
}
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
// MIC <-> Scalar Mask casts {{{1
// 1 MIC::Mask to 1 Scalar::Mask {{{2
template <typename Return, typename T>
Vc_INTRINSIC enable_if<Scalar::is_mask<Return>::value, Return> simd_cast(MIC::Mask<T> k)
{
    return Return(static_cast<bool>(k[0]));
}
// 1 Scalar::Mask to 1 MIC::Mask {{{2
template <typename Return, typename T>
Vc_INTRINSIC enable_if<MIC::is_mask<Return>::value, Return> simd_cast(Scalar::Mask<T> k)
{
    Return r{};
    r[0] = k[0];
    return r;
}
// 2 Scalar::Mask to 1 MIC::Mask {{{2
template <typename Return, typename T>
Vc_INTRINSIC enable_if<MIC::is_mask<Return>::value, Return> simd_cast(Scalar::Mask<T> k0,
                                                                      Scalar::Mask<T> k1)
{
    Return r{};
    r[0] = k0[0];
    r[1] = k1[0];
    return r;
}
// 4 Scalar::Mask to 1 MIC::Mask {{{2
template <typename Return, typename T>
Vc_INTRINSIC enable_if<MIC::is_mask<Return>::value, Return> simd_cast(Scalar::Mask<T> k0,
                                                                      Scalar::Mask<T> k1,
                                                                      Scalar::Mask<T> k2,
                                                                      Scalar::Mask<T> k3)
{
    Return r{};
    r[0] = k0[0];
    r[1] = k1[0];
    r[2] = k2[0];
    r[3] = k3[0];
    return r;
}
// 8 Scalar::Mask to 1 MIC::Mask {{{2
template <typename Return, typename T>
Vc_INTRINSIC enable_if<MIC::is_mask<Return>::value, Return> simd_cast(Scalar::Mask<T> k0,
                                                                      Scalar::Mask<T> k1,
                                                                      Scalar::Mask<T> k2,
                                                                      Scalar::Mask<T> k3,
                                                                      Scalar::Mask<T> k4,
                                                                      Scalar::Mask<T> k5,
                                                                      Scalar::Mask<T> k6,
                                                                      Scalar::Mask<T> k7)
{
    Return r{};
    r[0] = k0[0];
    r[1] = k1[0];
    r[2] = k2[0];
    r[3] = k3[0];
    r[4] = k4[0];
    r[5] = k5[0];
    r[6] = k6[0];
    r[7] = k7[0];
    return r;
}
//}}}1

}  // namespace MIC
using MIC::simd_cast;
}  // namespace Vc

#endif  // VC_MIC_SIMD_CAST_H_

// vim: foldmethod=marker
