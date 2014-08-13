/*  This file is part of the Vc library.

    Copyright (C) 2009-2012 Matthias Kretz <kretz@kde.org>

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

*/

#ifndef VC_AVX_MATH_H
#define VC_AVX_MATH_H

#include "const.h"
#include "limits.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Vc_IMPL_NAMESPACE
{
template <typename T> Vc_ALWAYS_INLINE Vector<T> copysign(Vector<T> a, Vector<T> b)
{
    return a.copySign(b);
}
    /**
     * splits \p v into exponent and mantissa, the sign is kept with the mantissa
     *
     * The return value will be in the range [0.5, 1.0[
     * The \p e value will be an integer defining the power-of-two exponent
     */
inline double_v frexp(double_v::AsArg v, simdarray<int, 4, SSE::int_v, 4> *e)
{
    const m256d exponentBits = Const<double>::exponentMask().dataD();
    const m256d exponentPart = _mm256_and_pd(v.data(), exponentBits);
    auto lo = avx_cast<m128i>(lo128(exponentPart));
    auto hi = avx_cast<m128i>(hi128(exponentPart));
    lo = _mm_sub_epi32(_mm_srli_epi64(lo, 52), _mm_set1_epi64x(0x3fe));
    hi = _mm_sub_epi32(_mm_srli_epi64(hi, 52), _mm_set1_epi64x(0x3fe));
    SSE::int_v exponent = Mem::shuffle<X0, X2, Y0, Y2>(lo, hi);
    const m256d exponentMaximized = _mm256_or_pd(v.data(), exponentBits);
    double_v ret =
        _mm256_and_pd(exponentMaximized,
                      _mm256_broadcast_sd(reinterpret_cast<const double *>(&c_general::frexpMask)));
    const double_m zeroMask = v == double_v::Zero();
    ret(isnan(v) || !isfinite(v) || zeroMask) = v;
    exponent.setZero(static_cast<SSE::int_m>(zeroMask));
    internal_data(*e) = exponent;
    return ret;
}
inline float_v frexp(float_v::AsArg v, simdarray<int, 8, SSE::int_v, 4> *e)
{
    const m256 exponentBits = Const<float>::exponentMask().data();
    const m256 exponentPart = _mm256_and_ps(v.data(), exponentBits);
    auto lo = avx_cast<m128i>(lo128(exponentPart));
    auto hi = avx_cast<m128i>(hi128(exponentPart));
    internal_data(internal_data0(*e)) = _mm_sub_epi32(_mm_srli_epi32(lo, 23), _mm_set1_epi32(0x7e));
    internal_data(internal_data1(*e)) = _mm_sub_epi32(_mm_srli_epi32(hi, 23), _mm_set1_epi32(0x7e));
    const m256 exponentMaximized = _mm256_or_ps(v.data(), exponentBits);
    float_v ret = _mm256_and_ps(exponentMaximized, avx_cast<m256>(_mm256_set1_epi32(0xbf7fffffu)));
    ret(isnan(v) || !isfinite(v) || v == float_v::Zero()) = v;
    e->setZero(static_cast<decltype(*e == *e)>(v == float_v::Zero()));
    return ret;
}

    /*             -> x * 2^e
     * x == NaN    -> NaN
     * x == (-)inf -> (-)inf
     */
inline double_v ldexp(double_v::AsArg v, const simdarray<int, 4, SSE::int_v, 4> &_e)
{
    SSE::int_v e = internal_data(_e);
    e.setZero(SSE::int_m{v == double_v::Zero()});
    const m256i exponentBits = concat(_mm_slli_epi64(_mm_unpacklo_epi32(e.data(), e.data()), 52),
                                      _mm_slli_epi64(_mm_unpackhi_epi32(e.data(), e.data()), 52));
    return avx_cast<m256d>(add_epi64(avx_cast<m256i>(v.data()), exponentBits));
}
inline float_v ldexp(float_v::AsArg v, simdarray<int, 8, SSE::int_v, 4> e)
{
    e.setZero(static_cast<decltype(e == e)>(v == float_v::Zero()));
    e <<= 23;
    return {avx_cast<m256>(
        concat(_mm_add_epi32(avx_cast<m128i>(lo128(v.data())), internal_data(internal_data0(e)).data()),
               _mm_add_epi32(avx_cast<m128i>(hi128(v.data())), internal_data(internal_data1(e)).data())))};
}

    static Vc_ALWAYS_INLINE  float_v trunc( float_v::AsArg v) { return _mm256_round_ps(v.data(), 0x3); }
    static Vc_ALWAYS_INLINE double_v trunc(double_v::AsArg v) { return _mm256_round_pd(v.data(), 0x3); }

    static Vc_ALWAYS_INLINE float_v floor(float_v::AsArg v) { return _mm256_floor_ps(v.data()); }
    static Vc_ALWAYS_INLINE double_v floor(double_v::AsArg v) { return _mm256_floor_pd(v.data()); }

    static Vc_ALWAYS_INLINE float_v ceil(float_v::AsArg v) { return _mm256_ceil_ps(v.data()); }
    static Vc_ALWAYS_INLINE double_v ceil(double_v::AsArg v) { return _mm256_ceil_pd(v.data()); }
}
}

#include "undomacros.h"

#endif // VC_AVX_MATH_H
