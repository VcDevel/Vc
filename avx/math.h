/*  This file is part of the Vc library. {{{
Copyright © 2009-2015 Matthias Kretz <kretz@kde.org>
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

#ifndef VC_AVX_MATH_H
#define VC_AVX_MATH_H

#include "const.h"
#include "limits.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
Vc_INTRINSIC Vc_CONST AVX2::double_v abs(AVX2::double_v x)
{
    return Detail::and_(x.data(), AVX::setabsmask_pd());
}
Vc_INTRINSIC Vc_CONST AVX2::float_v abs(AVX2::float_v x)
{
    return Detail::and_(x.data(), AVX::setabsmask_ps());
}
#ifdef VC_IMPL_AVX2
Vc_INTRINSIC Vc_CONST AVX2::int_v abs(AVX2::int_v x)
{
    return _mm256_abs_epi32(x.data());
}
Vc_INTRINSIC Vc_CONST AVX2::short_v abs(AVX2::short_v x)
{
    return _mm256_abs_epi16(x.data());
}
#endif
template <typename T>
Vc_INTRINSIC AVX2::Vector<T> copysign(AVX2::Vector<T> a, AVX2::Vector<T> b)
{
    return a.copySign(b);
}
/**
 * splits \p v into exponent and mantissa, the sign is kept with the mantissa
 *
 * The return value will be in the range [0.5, 1.0[
 * The \p e value will be an integer defining the power-of-two exponent
 */
inline AVX2::double_v frexp(AVX2::double_v::AsArg v, SimdArray<int, 4, SSE::int_v, 4> *e)
{
    const __m256d exponentBits = AVX::Const<double>::exponentMask().dataD();
    const __m256d exponentPart = _mm256_and_pd(v.data(), exponentBits);
    auto lo = AVX::avx_cast<__m128i>(AVX::lo128(exponentPart));
    auto hi = AVX::avx_cast<__m128i>(AVX::hi128(exponentPart));
    lo = _mm_sub_epi32(_mm_srli_epi64(lo, 52), _mm_set1_epi64x(0x3fe));
    hi = _mm_sub_epi32(_mm_srli_epi64(hi, 52), _mm_set1_epi64x(0x3fe));
    SSE::int_v exponent = Mem::shuffle<X0, X2, Y0, Y2>(lo, hi);
    const __m256d exponentMaximized = _mm256_or_pd(v.data(), exponentBits);
    AVX2::double_v ret =
        _mm256_and_pd(exponentMaximized,
                      _mm256_broadcast_sd(reinterpret_cast<const double *>(&AVX::c_general::frexpMask)));
    const double_m zeroMask = v == AVX2::double_v::Zero();
    ret(isnan(v) || !isfinite(v) || zeroMask) = v;
    exponent.setZero(static_cast<SSE::int_m>(zeroMask));
    internal_data(*e) = exponent;
    return ret;
}
namespace Detail
{
Vc_INTRINSIC AVX2::float_v::IndexType extractExponent(__m256 e)
{
    SimdArray<uint, float_v::size()> exponentPart;
    const auto ee = AVX::avx_cast<__m256i>(e);
#ifdef VC_IMPL_AVX2
    exponentPart = AVX2::uint_v(ee);
#else
    internal_data(internal_data0(exponentPart)) = AVX::lo128(ee);
    internal_data(internal_data1(exponentPart)) = AVX::hi128(ee);
#endif
    return (exponentPart >> 23) - 0x7e;
}
}  // namespace Detail
inline AVX2::float_v frexp(AVX2::float_v::AsArg v, AVX2::float_v::IndexType *e)
{
    using namespace Detail;
    using namespace AVX2;
    const __m256 exponentBits = Const<float>::exponentMask().data();
    *e = extractExponent(and_(v.data(), exponentBits));
    const __m256 exponentMaximized = or_(v.data(), exponentBits);
    float_v ret = _mm256_and_ps(exponentMaximized, avx_cast<__m256>(set1_epi32(0xbf7fffffu)));
    ret(isnan(v) || !isfinite(v) || v == float_v::Zero()) = v;
    e->setZero(static_cast<decltype(*e == *e)>(v == float_v::Zero()));
    return ret;
}

/*             -> x * 2^e
 * x == NaN    -> NaN
 * x == (-)inf -> (-)inf
 */
inline AVX2::double_v ldexp(AVX2::double_v::AsArg v, const SimdArray<int, 4, SSE::int_v, 4> &_e)
{
    SSE::int_v e = internal_data(_e);
    e.setZero(SSE::int_m{v == AVX2::double_v::Zero()});
    const __m256i exponentBits = AVX::concat(_mm_slli_epi64(_mm_unpacklo_epi32(e.data(), e.data()), 52),
                                      _mm_slli_epi64(_mm_unpackhi_epi32(e.data(), e.data()), 52));
    return AVX::avx_cast<__m256d>(AVX::add_epi64(AVX::avx_cast<__m256i>(v.data()), exponentBits));
}
inline AVX2::float_v ldexp(AVX2::float_v::AsArg v, SimdArray<int, 8, SSE::int_v, 4> e)
{
    e.setZero(static_cast<decltype(e == e)>(v == AVX2::float_v::Zero()));
    e <<= 23;
    return {AVX::avx_cast<__m256>(
        AVX::concat(_mm_add_epi32(AVX::avx_cast<__m128i>(AVX::lo128(v.data())), internal_data(internal_data0(e)).data()),
               _mm_add_epi32(AVX::avx_cast<__m128i>(AVX::hi128(v.data())), internal_data(internal_data1(e)).data())))};
}

static Vc_ALWAYS_INLINE AVX2::float_v trunc(AVX2::float_v::AsArg v)
{
    return _mm256_round_ps(v.data(), 0x3);
}
static Vc_ALWAYS_INLINE AVX2::double_v trunc(AVX2::double_v::AsArg v)
{
    return _mm256_round_pd(v.data(), 0x3);
}

static Vc_ALWAYS_INLINE AVX2::float_v floor(AVX2::float_v::AsArg v)
{
    return _mm256_floor_ps(v.data());
}
static Vc_ALWAYS_INLINE AVX2::double_v floor(AVX2::double_v::AsArg v)
{
    return _mm256_floor_pd(v.data());
}

static Vc_ALWAYS_INLINE AVX2::float_v ceil(AVX2::float_v::AsArg v)
{
    return _mm256_ceil_ps(v.data());
}
static Vc_ALWAYS_INLINE AVX2::double_v ceil(AVX2::double_v::AsArg v)
{
    return _mm256_ceil_pd(v.data());
}
}  // namespace Vc

#include "undomacros.h"

#endif // VC_AVX_MATH_H
