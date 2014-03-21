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

#ifndef VC_SSE_SIMD_CAST_H
#define VC_SSE_SIMD_CAST_H

#include "../common/simd_cast.h"
#include "intrinsics.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{

#define Vc_SIMD_CAST_SSE_1(from__, to__)                                                           \
    template <typename To>                                                                         \
    Vc_INTRINSIC To                                                                                \
        simd_cast(SSE::from__ x, enable_if<std::is_same<To, SSE::to__>::value> = nullarg)

#define Vc_SIMD_CAST_SSE_2(from__, to__)                                                           \
    template <typename To>                                                                         \
    Vc_INTRINSIC To simd_cast(                                                                     \
        SSE::from__ x0, SSE::from__ x1, enable_if<std::is_same<To, SSE::to__>::value> = nullarg)

#define Vc_SIMD_CAST_SSE_4(from__, to__)                                                           \
    template <typename To>                                                                         \
    Vc_INTRINSIC To simd_cast(SSE::from__ x0,                                                      \
                              SSE::from__ x1,                                                      \
                              SSE::from__ x2,                                                      \
                              SSE::from__ x3,                                                      \
                              enable_if<std::is_same<To, SSE::to__>::value> = nullarg)

Vc_SIMD_CAST_SSE_1( float_v,    int_v) { return _mm_cvttps_epi32(x.data()); }
Vc_SIMD_CAST_SSE_1(double_v,    int_v) { return _mm_cvttpd_epi32(x.data()); }
Vc_SIMD_CAST_SSE_2(double_v,    int_v) { return _mm_unpacklo_epi64(_mm_cvttpd_epi32(x0.data()), _mm_cvttpd_epi32(x1.data())); }  // XXX: improve with AVX
Vc_SIMD_CAST_SSE_1(  uint_v,    int_v) { return x.data(); }
Vc_SIMD_CAST_SSE_1( short_v,    int_v) { return _mm_srai_epi32(_mm_unpacklo_epi16(x.data(), x.data()), 16); }
Vc_SIMD_CAST_SSE_1(ushort_v,    int_v) { return _mm_unpacklo_epi16(x.data(), _mm_setzero_si128()); }

Vc_SIMD_CAST_SSE_1( float_v,   uint_v) {
    using namespace SseIntrinsics;
    return _mm_castps_si128(
        _mm_blendv_ps(_mm_castsi128_ps(_mm_cvttps_epi32(x.data())),
                      _mm_castsi128_ps(_mm_add_epi32(
                          _mm_cvttps_epi32(_mm_sub_ps(x.data(), _mm_set1_ps(1u << 31))),
                          _mm_set1_epi32(1 << 31))),
                      _mm_cmpge_ps(x.data(), _mm_set1_ps(1u << 31))));
}
Vc_SIMD_CAST_SSE_1(double_v,   uint_v) {
    return _mm_add_epi32(_mm_cvttpd_epi32(_mm_sub_pd(x.data(), _mm_set1_pd(0x80000000u))),
                         _mm_cvtsi64_si128(0x8000000080000000ull));
}
Vc_SIMD_CAST_SSE_2(double_v,   uint_v) {  // XXX: improve with AVX
    return _mm_add_epi32(
        _mm_unpacklo_epi64(_mm_cvttpd_epi32(_mm_sub_pd(x0.data(), _mm_set1_pd(0x80000000u))),
                           _mm_cvttpd_epi32(_mm_sub_pd(x1.data(), _mm_set1_pd(0x80000000u)))),
        _mm_set1_epi32(0x80000000u));
}
Vc_SIMD_CAST_SSE_1(   int_v,   uint_v) { return x.data(); }
Vc_SIMD_CAST_SSE_1( short_v,   uint_v) {
    // the conversion rule is x mod 2^32
    // and the definition of mod here is the one that yields only positive numbers
    return _mm_srai_epi32(_mm_unpacklo_epi16(x.data(), x.data()), 16); }
Vc_SIMD_CAST_SSE_1(ushort_v,   uint_v) { return _mm_unpacklo_epi16(x.data(), _mm_setzero_si128()); }

Vc_SIMD_CAST_SSE_1(double_v,  float_v) { return _mm_cvtpd_ps(x.data()); }
Vc_SIMD_CAST_SSE_2(double_v,  float_v) { return _mm_movelh_ps(_mm_cvtpd_ps(x0.data()), _mm_cvtpd_ps(x1.data())); }  // XXX: improve with AVX
Vc_SIMD_CAST_SSE_1(   int_v,  float_v) { return _mm_cvtepi32_ps(x.data()); }
Vc_SIMD_CAST_SSE_1(  uint_v,  float_v) {
    using namespace SseIntrinsics;
    return _mm_blendv_ps(_mm_cvtepi32_ps(x.data()),
                         _mm_add_ps(_mm_cvtepi32_ps(_mm_sub_epi32(x.data(), _mm_setmin_epi32())),
                                    _mm_set1_ps(1u << 31)),
                         _mm_castsi128_ps(_mm_cmplt_epi32(x.data(), _mm_setzero_si128())));
}
Vc_SIMD_CAST_SSE_1( short_v,  float_v) { return simd_cast<SSE::float_v>(simd_cast<SSE::int_v>(x)); }
Vc_SIMD_CAST_SSE_1(ushort_v,  float_v) { return simd_cast<SSE::float_v>(simd_cast<SSE::int_v>(x)); }

Vc_SIMD_CAST_SSE_1( float_v, double_v) { return _mm_cvtps_pd(x.data()); }
Vc_SIMD_CAST_SSE_1(   int_v, double_v) { return _mm_cvtepi32_pd(x.data()); }
Vc_SIMD_CAST_SSE_1(  uint_v, double_v) {
    using namespace SseIntrinsics;
    return _mm_add_pd(_mm_cvtepi32_pd(_mm_sub_epi32(x.data(), _mm_setmin_epi32())),
                      _mm_set1_pd(1u << 31));
}
Vc_SIMD_CAST_SSE_1( short_v, double_v) { return simd_cast<SSE::double_v>(simd_cast<SSE::int_v>(x)); }
Vc_SIMD_CAST_SSE_1(ushort_v, double_v) { return simd_cast<SSE::double_v>(simd_cast<SSE::int_v>(x)); }

/*
 * §4.7 p3 (integral conversions)
 *  If the destination type is signed, the value is unchanged if it can be represented in the
 *  destination type (and bit-field width); otherwise, the value is implementation-defined.
 */
#ifdef VC_GCC
Vc_SIMD_CAST_SSE_1(   int_v,  short_v) {
    auto tmp0 = _mm_unpacklo_epi16(x.data(), _mm_setzero_si128());  // 0 4 X X 1 5 X X
    auto tmp1 = _mm_unpackhi_epi16(x.data(), _mm_setzero_si128());  // 2 6 X X 3 7 X X
    auto tmp2 = _mm_unpacklo_epi16(tmp0, tmp1);                     // 0 2 4 6 X X X X
    auto tmp3 = _mm_unpackhi_epi16(tmp0, tmp1);                     // 1 3 5 7 X X X X
    return _mm_unpacklo_epi16(tmp2, tmp3);                          // 0 1 2 3 4 5 6 7
}
Vc_SIMD_CAST_SSE_2(   int_v,  short_v) {
    auto tmp0 = _mm_unpacklo_epi16(x0.data(), x1.data());  // 0 4 X X 1 5 X X
    auto tmp1 = _mm_unpackhi_epi16(x0.data(), x1.data());  // 2 6 X X 3 7 X X
    auto tmp2 = _mm_unpacklo_epi16(tmp0, tmp1);            // 0 2 4 6 X X X X
    auto tmp3 = _mm_unpackhi_epi16(tmp0, tmp1);            // 1 3 5 7 X X X X
    return _mm_unpacklo_epi16(tmp2, tmp3);                 // 0 1 2 3 4 5 6 7
}
Vc_SIMD_CAST_SSE_1(  uint_v,  short_v) {
    auto tmp0 = _mm_unpacklo_epi16(x.data(), _mm_setzero_si128());  // 0 4 X X 1 5 X X
    auto tmp1 = _mm_unpackhi_epi16(x.data(), _mm_setzero_si128());  // 2 6 X X 3 7 X X
    auto tmp2 = _mm_unpacklo_epi16(tmp0, tmp1);                     // 0 2 4 6 X X X X
    auto tmp3 = _mm_unpackhi_epi16(tmp0, tmp1);                     // 1 3 5 7 X X X X
    return _mm_unpacklo_epi16(tmp2, tmp3);                          // 0 1 2 3 4 5 6 7
}
Vc_SIMD_CAST_SSE_2(  uint_v,  short_v) {
    auto tmp0 = _mm_unpacklo_epi16(x0.data(), x1.data());  // 0 4 X X 1 5 X X
    auto tmp1 = _mm_unpackhi_epi16(x0.data(), x1.data());  // 2 6 X X 3 7 X X
    auto tmp2 = _mm_unpacklo_epi16(tmp0, tmp1);            // 0 2 4 6 X X X X
    auto tmp3 = _mm_unpackhi_epi16(tmp0, tmp1);            // 1 3 5 7 X X X X
    return _mm_unpacklo_epi16(tmp2, tmp3);                 // 0 1 2 3 4 5 6 7
}
#else
Vc_SIMD_CAST_SSE_1(   int_v,  short_v) { return _mm_packs_epi32(x.data(), _mm_setzero_si128()); }
Vc_SIMD_CAST_SSE_2(   int_v,  short_v) { return _mm_packs_epi32(x0.data(), x1.data()); }
Vc_SIMD_CAST_SSE_1(  uint_v,  short_v) { return _mm_packs_epi32(x.data(), _mm_setzero_si128()); }
Vc_SIMD_CAST_SSE_2(  uint_v,  short_v) { return _mm_packs_epi32(x0.data(), x1.data()); }
#endif
Vc_SIMD_CAST_SSE_1( float_v,  short_v) { return _mm_packs_epi32(simd_cast<SSE::int_v>(x).data(), _mm_setzero_si128()); }
Vc_SIMD_CAST_SSE_2( float_v,  short_v) { return _mm_packs_epi32(simd_cast<SSE::int_v>(x0).data(), simd_cast<SSE::int_v>(x1).data()); }
Vc_SIMD_CAST_SSE_1(double_v,  short_v) { return _mm_packs_epi32(simd_cast<SSE::int_v>(x).data(), _mm_setzero_si128()); }
Vc_SIMD_CAST_SSE_2(double_v,  short_v) { return _mm_packs_epi32(simd_cast<SSE::int_v>(x0, x1).data(), _mm_setzero_si128()); }
Vc_SIMD_CAST_SSE_4(double_v,  short_v) { return _mm_packs_epi32(simd_cast<SSE::int_v>(x0, x1).data(), simd_cast<SSE::int_v>(x2, x3).data()); }
Vc_SIMD_CAST_SSE_1(ushort_v,  short_v) { return x.data(); }

Vc_SIMD_CAST_SSE_1(   int_v, ushort_v) {
    auto tmp0 = _mm_unpacklo_epi16(x.data(), _mm_setzero_si128());  // 0 4 X X 1 5 X X
    auto tmp1 = _mm_unpackhi_epi16(x.data(), _mm_setzero_si128());  // 2 6 X X 3 7 X X
    auto tmp2 = _mm_unpacklo_epi16(tmp0, tmp1);                     // 0 2 4 6 X X X X
    auto tmp3 = _mm_unpackhi_epi16(tmp0, tmp1);                     // 1 3 5 7 X X X X
    return _mm_unpacklo_epi16(tmp2, tmp3);                          // 0 1 2 3 4 5 6 7
}
Vc_SIMD_CAST_SSE_2(   int_v, ushort_v) {
    auto tmp0 = _mm_unpacklo_epi16(x0.data(), x1.data());  // 0 4 X X 1 5 X X
    auto tmp1 = _mm_unpackhi_epi16(x0.data(), x1.data());  // 2 6 X X 3 7 X X
    auto tmp2 = _mm_unpacklo_epi16(tmp0, tmp1);            // 0 2 4 6 X X X X
    auto tmp3 = _mm_unpackhi_epi16(tmp0, tmp1);            // 1 3 5 7 X X X X
    return _mm_unpacklo_epi16(tmp2, tmp3);                 // 0 1 2 3 4 5 6 7
}
Vc_SIMD_CAST_SSE_1( float_v, ushort_v) { return simd_cast<SSE::ushort_v>(simd_cast<SSE::int_v>(x)); }
Vc_SIMD_CAST_SSE_2( float_v, ushort_v) { return simd_cast<SSE::ushort_v>(simd_cast<SSE::int_v>(x0), simd_cast<SSE::int_v>(x1)); }
Vc_SIMD_CAST_SSE_1(double_v, ushort_v) { return simd_cast<SSE::ushort_v>(simd_cast<SSE::int_v>(x)); }
Vc_SIMD_CAST_SSE_2(double_v, ushort_v) { return simd_cast<SSE::ushort_v>(simd_cast<SSE::int_v>(x0, x1)); }
Vc_SIMD_CAST_SSE_4(double_v, ushort_v) { return simd_cast<SSE::ushort_v>(simd_cast<SSE::int_v>(x0, x1), simd_cast<SSE::int_v>(x2, x3)); }
Vc_SIMD_CAST_SSE_1(  uint_v, ushort_v) {
    auto tmp0 = _mm_unpacklo_epi16(x.data(), _mm_setzero_si128());  // 0 4 X X 1 5 X X
    auto tmp1 = _mm_unpackhi_epi16(x.data(), _mm_setzero_si128());  // 2 6 X X 3 7 X X
    auto tmp2 = _mm_unpacklo_epi16(tmp0, tmp1);                     // 0 2 4 6 X X X X
    auto tmp3 = _mm_unpackhi_epi16(tmp0, tmp1);                     // 1 3 5 7 X X X X
    return _mm_unpacklo_epi16(tmp2, tmp3);                          // 0 1 2 3 4 5 6 7
}
Vc_SIMD_CAST_SSE_2(  uint_v, ushort_v) {
    auto tmp0 = _mm_unpacklo_epi16(x0.data(), x1.data());  // 0 4 X X 1 5 X X
    auto tmp1 = _mm_unpackhi_epi16(x0.data(), x1.data());  // 2 6 X X 3 7 X X
    auto tmp2 = _mm_unpacklo_epi16(tmp0, tmp1);            // 0 2 4 6 X X X X
    auto tmp3 = _mm_unpackhi_epi16(tmp0, tmp1);            // 1 3 5 7 X X X X
    return _mm_unpacklo_epi16(tmp2, tmp3);                 // 0 1 2 3 4 5 6 7
}
Vc_SIMD_CAST_SSE_1( short_v, ushort_v) { return x.data(); }

#undef Vc_SIMD_CAST_SSE_1
#undef Vc_SIMD_CAST_SSE_2
#undef Vc_SIMD_CAST_SSE_4

template <typename Return, int offset, typename T>
Vc_INTRINSIC Return simd_cast(SSE::Vector<T> x, enable_if<offset != 0> = nullarg)
{
    using V = SSE::Vector<T>;
    constexpr int shift = sizeof(T) * offset * Return::Size;
    static_assert(shift > 0 && shift < 16, "");
    return simd_cast<Return>(V{SSE::sse_cast<typename V::VectorType>(
        _mm_srli_si128(SSE::sse_cast<__m128i>(x.data()), shift))});
}

template <typename Return, int offset, typename T>
Vc_INTRINSIC Return simd_cast(SSE::Vector<T> x, enable_if<offset == 0> = nullarg)
{
    return simd_cast<Return>(x);
}


}  // namespace Vc

#include "undomacros.h"

#endif // VC_SSE_SIMD_CAST_H
