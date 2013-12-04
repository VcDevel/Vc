/*  This file is part of the Vc library. {{{

    Copyright (C) 2013 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_MIC_MATH_H
#define VC_MIC_MATH_H

#include "macros.h"

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)

// copysign {{{1
template <typename T> Vc_ALWAYS_INLINE Vector<T> copysign(Vector<T> a, Vector<T> b)
{
    return a.copySign(b);
}
// trunc {{{1
template<typename V> Vc_ALWAYS_INLINE V trunc(V v)
{
    return _mm512_trunc_ps(v.data());
}
Vc_ALWAYS_INLINE double_v trunc(double_v v)
{
    return _mm512_trunc_pd(v.data());
}
// isfinite {{{1
template<typename T> static Vc_ALWAYS_INLINE Mask<T> isfinite(Vector<T> x)
{
    return _mm512_cmpord_ps_mask(x.data(), (x * Vector<T>::Zero()).data());
}
static Vc_ALWAYS_INLINE double_m isfinite(double_v x)
{
    return _mm512_cmpord_pd_mask(x.data(), (x * double_v::Zero()).data());
}
// isnotfinite {{{1
// i.e. !isfinite(x), this is not equivalent to isinfinite because NaN also is not finite
template<typename T> static Vc_ALWAYS_INLINE Mask<T> isnotfinite(Vector<T> x)
{
    return _mm512_cmpunord_ps_mask(x.data(), (x * Vector<T>::Zero()).data());
}
static Vc_ALWAYS_INLINE double_m isnotfinite(double_v x)
{
    return _mm512_cmpunord_pd_mask(x.data(), (x * double_v::Zero()).data());
}
// isinf {{{1
Vc_ALWAYS_INLINE float_m isinf(float_v x)
{
    return _mm512_cmpeq_epi32_mask(_mm512_castps_si512(abs(x).data()), _mm512_set1_epi32(0x7f800000));
}
Vc_ALWAYS_INLINE double_m isinf(double_v x)
{
    auto mask = _mm512_cmpeq_epi32_mask(_mm512_castpd_si512(abs(x).data()), _mm512_set1_epi64(0x7ff0000000000000ull));
    // FIXME this is not efficient:
    return ((mask & 0x01) | ((mask >> 2) & 0x06) | ((mask >> 4) & 0x18) | ((mask >> 6) & 0x60) |
            ((mask >> 8) & 0x80)) &
           (((mask >> 1) & 0x03) | ((mask >> 3) & 0x0c) | ((mask >> 5) & 0x30) |
            ((mask >> 7) & 0xc0));
}
// isnan {{{1
template<typename T> static Vc_ALWAYS_INLINE Mask<T> isnan(Vector<T> x)
{
    return _mm512_cmpunord_ps_mask(x.data(), x.data());
}
static Vc_ALWAYS_INLINE double_m isnan(double_v x)
{
    return _mm512_cmpunord_pd_mask(x.data(), x.data());
}
// frexp {{{1
inline double_v frexp(double_v::AsArg v, int_v *e) {
    const __m512i vi = mic_cast<__m512i>(v.data());
    const __m512i exponentBits = _set1(0x7ff0000000000000ull);
    const __m512i exponentPart = _and(vi, exponentBits);
    const __mmask8  zeroMask  = _mm512_cmpneq_pd_mask(v.data(), _mm512_setzero_pd());
    // bit-interleave 0x00 and zeroMask:
    const __mmask16 zeroMask2 = _mm512_cmpgt_epu32_mask(_mm512_and_epi32(
                _mm512_set1_epi64(zeroMask),
                _load(c_general::frexpAndMask, _MM_UPCONV_EPI32_UINT8)),
            _mm512_setzero_epi32());
    e->data() = _mm512_mask_sub_epi32(_mm512_setzero_epi32(), zeroMask2,
            _mm512_srli_epi32(_mm512_swizzle_epi32(exponentPart, _MM_SWIZ_REG_CDAB), 20),
            _set1(0x3fe));
    const __m512i exponentMaximized = _or(vi, exponentBits);
    const __mmask8 nonzeroNumber = _mm512_kand(isfinite(v).data(),
               _mm512_cmpneq_pd_mask(v.data(), _mm512_setzero_pd()));
    double_v ret = mic_cast<__m512d>(_mm512_mask_and_epi64(vi, nonzeroNumber,
                exponentMaximized, _set1(0xbfefffffffffffffull)));
    return ret;
}
inline float_v frexp(float_v::AsArg v, int_v *e) {
    const __m512i vi = mic_cast<__m512i>(v.data());
    const __m512i exponentBits = _set1(0x7f800000u);
    const __m512i exponentPart = _and(vi, exponentBits);
    const __mmask16 zeroMask = _mm512_cmpneq_ps_mask(v.data(), _mm512_setzero_ps());
    e->data() = _mm512_mask_sub_epi32(_mm512_setzero_epi32(), zeroMask,
            _mm512_srli_epi32(exponentPart, 23), _set1(0x7e));
    const __m512i exponentMaximized = _or(vi, exponentBits);
    const __mmask16 nonzeroNumber = _mm512_kand(isfinite(v).data(),
               _mm512_cmpneq_ps_mask(v.data(), _mm512_setzero_ps()));
    float_v ret = mic_cast<__m512>(_mm512_mask_and_epi32(vi, nonzeroNumber,
                exponentMaximized, _set1(0xbf7fffffu)));
    return ret;
}
// ldexp {{{1
Vc_ALWAYS_INLINE double_v ldexp(double_v::AsArg v, int_v::AsArg _e)
{
    __m512i e = _mm512_mask_xor_epi64(_e.data(), (v == double_v::Zero()).data(),
            _e.data(), _e.data());
    const __m512i exponentBits = _mm512_mask_slli_epi32(_mm512_setzero_epi32(),
            0xaaaa, _mm512_swizzle_epi32(e, _MM_SWIZ_REG_CDAB), 20);
    return mic_cast<__m512d>(_mm512_add_epi32(mic_cast<__m512i>(v.data()), exponentBits));
}
Vc_ALWAYS_INLINE float_v ldexp(float_v::AsArg v, int_v::AsArg _e)
{
    int_v e = _e;
    e.setZero(static_cast<int_m>(v == float_v::Zero()));
    return (v.reinterpretCast<int_v>() + (e << 23)).reinterpretCast<float_v>();
}
//}}}1
Vc_NAMESPACE_END

#include "undomacros.h"

#endif // VC_MIC_MATH_H

// vim: foldmethod=marker
