/*  This file is part of the Vc library. {{{
Copyright © 2013-2015 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_MIC_MATH_H_
#define VC_MIC_MATH_H_

#include "macros.h"

Vc_VERSIONED_NAMESPACE_BEGIN
// copysign {{{1
Vc_INTRINSIC Vc_CONST MIC::float_v copysign(MIC::float_v mag, MIC::float_v sign)
{
    return MIC::_or(MIC::_and(sign.data(), MIC::_mm512_setsignmask_ps()),
                    MIC::_and(mag.data(), MIC::_mm512_setabsmask_ps()));
}
Vc_INTRINSIC Vc_CONST MIC::double_v copysign(MIC::double_v mag, MIC::double_v sign)
{
    return MIC::_or(MIC::_and(sign.data(), MIC::_mm512_setsignmask_pd()),
                    MIC::_and(mag.data(), MIC::_mm512_setabsmask_pd()));
}

// trunc {{{1
template <typename V> Vc_ALWAYS_INLINE V trunc(V v) { return _mm512_trunc_ps(v.data()); }
Vc_ALWAYS_INLINE MIC::double_v trunc(MIC::double_v v)
{
    return _mm512_trunc_pd(v.data());
}
// isfinite {{{1
template <typename T>
static Vc_ALWAYS_INLINE Mask<T> isfinite(Vector<T, VectorAbi::Mic> x)
{
    return _mm512_cmpord_ps_mask(x.data(),
                                 (x * Vector<T, VectorAbi::Mic>::Zero()).data());
}
static Vc_ALWAYS_INLINE MIC::double_m isfinite(MIC::double_v x)
{
    return _mm512_cmpord_pd_mask(x.data(), (x * MIC::double_v::Zero()).data());
}
// isnotfinite {{{1
// i.e. !isfinite(x), this is not equivalent to isinfinite because NaN also is not finite
template <typename T>
static Vc_ALWAYS_INLINE Mask<T> isnotfinite(Vector<T, VectorAbi::Mic> x)
{
    return _mm512_cmpunord_ps_mask(x.data(),
                                   (x * Vector<T, VectorAbi::Mic>::Zero()).data());
}
static Vc_ALWAYS_INLINE MIC::double_m isnotfinite(MIC::double_v x)
{
    return _mm512_cmpunord_pd_mask(x.data(), (x * MIC::double_v::Zero()).data());
}
// isinf {{{1
Vc_ALWAYS_INLINE MIC::float_m isinf(MIC::float_v x)
{
    return _mm512_cmpeq_epi32_mask(_mm512_castps_si512(abs(x).data()),
                                   _mm512_set1_epi32(0x7f800000));
}
Vc_ALWAYS_INLINE MIC::double_m isinf(MIC::double_v x)
{
    auto mask = _mm512_cmpeq_epi32_mask(_mm512_castpd_si512(abs(x).data()),
                                        _mm512_set1_epi64(0x7ff0000000000000ull));
    // FIXME this is not efficient:
    return ((mask & 0x01) | ((mask >> 2) & 0x06) | ((mask >> 4) & 0x18) |
            ((mask >> 6) & 0x60) | ((mask >> 8) & 0x80)) &
           (((mask >> 1) & 0x03) | ((mask >> 3) & 0x0c) | ((mask >> 5) & 0x30) |
            ((mask >> 7) & 0xc0));
}
// isnan {{{1
template <typename T> static Vc_ALWAYS_INLINE Mask<T> isnan(Vector<T, VectorAbi::Mic> x)
{
    return _mm512_cmpunord_ps_mask(x.data(), x.data());
}
static Vc_ALWAYS_INLINE MIC::double_m isnan(MIC::double_v x)
{
    return _mm512_cmpunord_pd_mask(x.data(), x.data());
}
// fma {{{1
Vc_ALWAYS_INLINE Vector<double, VectorAbi::Mic> fma(Vector<double, VectorAbi::Mic> a,
                                                    Vector<double, VectorAbi::Mic> b,
                                                    Vector<double, VectorAbi::Mic> c)
{
    return _mm512_fmadd_pd(a.data(), b.data(), c.data());
}
Vc_ALWAYS_INLINE Vector<float, VectorAbi::Mic> fma(Vector<float, VectorAbi::Mic> a,
                                                   Vector<float, VectorAbi::Mic> b,
                                                   Vector<float, VectorAbi::Mic> c)
{
    return _mm512_fmadd_ps(a.data(), b.data(), c.data());
}
template <typename T>
Vc_ALWAYS_INLINE Vector<T, VectorAbi::Mic> fma(Vector<T, VectorAbi::Mic> a,
                                               Vector<T, VectorAbi::Mic> b,
                                               Vector<T, VectorAbi::Mic> c)
{
    return _mm512_fmadd_epi32(a.data(), b.data(), c.data());
}

// frexp {{{1
inline MIC::double_v frexp(MIC::double_v::AsArg v, MIC::double_v::IndexType *e)
{
    using namespace MIC;
    const __m512d mantissa =
        _mm512_getmant_pd(v.data(), _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src);
    int_v exponent = int_v::Zero();
    exponent(static_cast<int_m>(v != v.Zero())) =
        int_v(convert<double, int>(_mm512_getexp_pd(v.data()))) + 1;
    *e = simd_cast<double_v::IndexType>(exponent);
    return double_v(mantissa) * 0.5;
}
inline MIC::float_v frexp(MIC::float_v::AsArg v, MIC::float_v::IndexType *e)
{
    using namespace MIC;
    const __m512 mantissa =
        _mm512_getmant_ps(v.data(), _MM_MANT_NORM_1_2, _MM_MANT_SIGN_src);
    int_v exponent = int_v::Zero();
    exponent(static_cast<int_m>(v != v.Zero())) =
        int_v(convert<float, int>(_mm512_getexp_ps(v.data()))) + 1;
    internal_data(*e) = exponent;
    return float_v(mantissa) * 0.5f;
}
// ldexp {{{1
Vc_ALWAYS_INLINE MIC::double_v ldexp(MIC::double_v::AsArg v, MIC::double_v::IndexType _e)
{
    using namespace MIC;
    static_assert(sizeof(_e) >= 32, "");
    static_assert(std::is_same<double_v::IndexType::EntryType, int>::value, "");
    const auto e_ = _mm512_extload_epi32(&_e, _MM_UPCONV_EPI32_SINT16,
                                         _MM_BROADCAST32_NONE, _MM_HINT_NONE);
    __m512i e = _mm512_mask_xor_epi64(e_, (v == double_v::Zero()).data(), e_, e_);
    const __m512i exponentBits = _mm512_mask_slli_epi32(
        _mm512_setzero_epi32(), 0xaaaa, _mm512_swizzle_epi32(e, _MM_SWIZ_REG_CDAB), 20);
    return mic_cast<__m512d>(_mm512_add_epi32(mic_cast<__m512i>(v.data()), exponentBits));
}
Vc_ALWAYS_INLINE MIC::float_v ldexp(MIC::float_v::AsArg v, MIC::float_v::IndexType _e)
{
    MIC::int_v e = internal_data(_e);
    e.setZero(static_cast<MIC::int_m>(v == MIC::float_v::Zero()));
    return (v.reinterpretCast<MIC::int_v>() + (e << 23)).reinterpretCast<MIC::float_v>();
}
//}}}1
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_MIC_MATH_H_

// vim: foldmethod=marker
