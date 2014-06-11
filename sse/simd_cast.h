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

#ifndef VC_SSE_VECTOR_H__
#error "Vc/sse/vector.h needs to be included before Vc/sse/simd_cast.h"
#endif
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{

// helper macros Vc_SIMD_CAST_SSE_[124] & Vc_SIMD_CAST_[1248] {{{1
#define Vc_SIMD_CAST_SSE_1(from__, to__)                                                           \
    template <typename To>                                                                         \
    Vc_INTRINSIC Vc_CONST To                                                                       \
        simd_cast(SSE::from__ x, enable_if<std::is_same<To, SSE::to__>::value> = nullarg)

#define Vc_SIMD_CAST_SSE_2(from__, to__)                                                           \
    template <typename To>                                                                         \
    Vc_INTRINSIC Vc_CONST To simd_cast(                                                            \
        SSE::from__ x0, SSE::from__ x1, enable_if<std::is_same<To, SSE::to__>::value> = nullarg)

#define Vc_SIMD_CAST_SSE_4(from__, to__)                                                           \
    template <typename To>                                                                         \
    Vc_INTRINSIC Vc_CONST To simd_cast(SSE::from__ x0,                                             \
                                       SSE::from__ x1,                                             \
                                       SSE::from__ x2,                                             \
                                       SSE::from__ x3,                                             \
                                       enable_if<std::is_same<To, SSE::to__>::value> = nullarg)

#define Vc_SIMD_CAST_1(from__, to__)                                                               \
    template <typename To>                                                                         \
    Vc_INTRINSIC Vc_CONST To                                                                       \
        simd_cast(from__ x, enable_if<std::is_same<To, to__>::value> = nullarg)

#define Vc_SIMD_CAST_2(from__, to__)                                                               \
    template <typename To>                                                                         \
    Vc_INTRINSIC Vc_CONST To                                                                       \
        simd_cast(from__ x0, from__ x1, enable_if<std::is_same<To, to__>::value> = nullarg)

#define Vc_SIMD_CAST_4(from__, to__)                                                               \
    template <typename To>                                                                         \
    Vc_INTRINSIC Vc_CONST To simd_cast(from__ x0,                                                  \
                                       from__ x1,                                                  \
                                       from__ x2,                                                  \
                                       from__ x3,                                                  \
                                       enable_if<std::is_same<To, to__>::value> = nullarg)

#define Vc_SIMD_CAST_8(from__, to__)                                                               \
    template <typename To>                                                                         \
    Vc_INTRINSIC Vc_CONST To simd_cast(from__ x0,                                                  \
                                       from__ x1,                                                  \
                                       from__ x2,                                                  \
                                       from__ x3,                                                  \
                                       from__ x4,                                                  \
                                       from__ x5,                                                  \
                                       from__ x6,                                                  \
                                       from__ x7,                                                  \
                                       enable_if<std::is_same<To, to__>::value> = nullarg)

// Vector casts without offset {{{1
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
#if 1
// GCC uses wrapping
Vc_SIMD_CAST_SSE_2(   int_v,  short_v) {
    auto tmp0 = _mm_unpacklo_epi16(x0.data(), x1.data());  // 0 4 X X 1 5 X X
    auto tmp1 = _mm_unpackhi_epi16(x0.data(), x1.data());  // 2 6 X X 3 7 X X
    auto tmp2 = _mm_unpacklo_epi16(tmp0, tmp1);            // 0 2 4 6 X X X X
    auto tmp3 = _mm_unpackhi_epi16(tmp0, tmp1);            // 1 3 5 7 X X X X
    return _mm_unpacklo_epi16(tmp2, tmp3);                 // 0 1 2 3 4 5 6 7
}
Vc_SIMD_CAST_SSE_1(   int_v,  short_v) {
    return simd_cast<SSE::short_v>(x, SSE::int_v::Zero());
}
Vc_SIMD_CAST_SSE_2(  uint_v,  short_v) {
    auto tmp0 = _mm_unpacklo_epi16(x0.data(), x1.data());  // 0 4 X X 1 5 X X
    auto tmp1 = _mm_unpackhi_epi16(x0.data(), x1.data());  // 2 6 X X 3 7 X X
    auto tmp2 = _mm_unpacklo_epi16(tmp0, tmp1);            // 0 2 4 6 X X X X
    auto tmp3 = _mm_unpackhi_epi16(tmp0, tmp1);            // 1 3 5 7 X X X X
    return _mm_unpacklo_epi16(tmp2, tmp3);                 // 0 1 2 3 4 5 6 7
}
Vc_SIMD_CAST_SSE_1(  uint_v,  short_v) {
    return simd_cast<SSE::short_v>(x, SSE::uint_v::Zero());
}
#else
// the alternative, which is probably incorrect for most compilers out there.
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

Vc_SIMD_CAST_SSE_2(   int_v, ushort_v) {
    auto tmp0 = _mm_unpacklo_epi16(x0.data(), x1.data());  // 0 4 X X 1 5 X X
    auto tmp1 = _mm_unpackhi_epi16(x0.data(), x1.data());  // 2 6 X X 3 7 X X
    auto tmp2 = _mm_unpacklo_epi16(tmp0, tmp1);            // 0 2 4 6 X X X X
    auto tmp3 = _mm_unpackhi_epi16(tmp0, tmp1);            // 1 3 5 7 X X X X
    return _mm_unpacklo_epi16(tmp2, tmp3);                 // 0 1 2 3 4 5 6 7
}
Vc_SIMD_CAST_SSE_1(   int_v, ushort_v) {
    return simd_cast<SSE::ushort_v>(x, SSE::int_v::Zero());
}
Vc_SIMD_CAST_SSE_1( float_v, ushort_v) { return simd_cast<SSE::ushort_v>(simd_cast<SSE::int_v>(x)); }
Vc_SIMD_CAST_SSE_2( float_v, ushort_v) { return simd_cast<SSE::ushort_v>(simd_cast<SSE::int_v>(x0), simd_cast<SSE::int_v>(x1)); }
Vc_SIMD_CAST_SSE_1(double_v, ushort_v) { return simd_cast<SSE::ushort_v>(simd_cast<SSE::int_v>(x)); }
Vc_SIMD_CAST_SSE_2(double_v, ushort_v) { return simd_cast<SSE::ushort_v>(simd_cast<SSE::int_v>(x0, x1)); }
Vc_SIMD_CAST_SSE_4(double_v, ushort_v) { return simd_cast<SSE::ushort_v>(simd_cast<SSE::int_v>(x0, x1), simd_cast<SSE::int_v>(x2, x3)); }
Vc_SIMD_CAST_SSE_2(  uint_v, ushort_v) {
    auto tmp0 = _mm_unpacklo_epi16(x0.data(), x1.data());  // 0 4 X X 1 5 X X
    auto tmp1 = _mm_unpackhi_epi16(x0.data(), x1.data());  // 2 6 X X 3 7 X X
    auto tmp2 = _mm_unpacklo_epi16(tmp0, tmp1);            // 0 2 4 6 X X X X
    auto tmp3 = _mm_unpackhi_epi16(tmp0, tmp1);            // 1 3 5 7 X X X X
    return _mm_unpacklo_epi16(tmp2, tmp3);                 // 0 1 2 3 4 5 6 7
}
Vc_SIMD_CAST_SSE_1(  uint_v, ushort_v) {
    return simd_cast<SSE::ushort_v>(x, SSE::uint_v::Zero());
}
Vc_SIMD_CAST_SSE_1( short_v, ushort_v) { return x.data(); }

Vc_SIMD_CAST_1(Scalar::int_v, SSE::double_v) { return _mm_setr_pd(x.data(), 0); } // FIXME: register - register mov
Vc_SIMD_CAST_2(Scalar::int_v, SSE::double_v) { return _mm_setr_pd(x0.data(), x1.data()); }
Vc_SIMD_CAST_1(Scalar::int_v, SSE::float_v) { return _mm_setr_ps(x.data(), 0, 0, 0); } // FIXME
Vc_SIMD_CAST_2(Scalar::int_v, SSE::float_v) { return _mm_setr_ps(x0.data(), x1.data(), 0, 0); } // FIXME
Vc_SIMD_CAST_4(Scalar::int_v, SSE::float_v) { return _mm_setr_ps(x0.data(), x1.data(), x2.data(), x3.data()); }
Vc_SIMD_CAST_1(Scalar::int_v, SSE::int_v) { return _mm_setr_epi32(x.data(), 0, 0, 0); } // FIXME
Vc_SIMD_CAST_2(Scalar::int_v, SSE::int_v) { return _mm_setr_epi32(x0.data(), x1.data(), 0, 0); } // FIXME
Vc_SIMD_CAST_4(Scalar::int_v, SSE::int_v) { return _mm_setr_epi32(x0.data(), x1.data(), x2.data(), x3.data()); }
Vc_SIMD_CAST_1(Scalar::int_v, SSE::uint_v) { return _mm_setr_epi32(x.data(), 0, 0, 0); } // FIXME
Vc_SIMD_CAST_2(Scalar::int_v, SSE::uint_v) { return _mm_setr_epi32(x0.data(), x1.data(), 0, 0); } // FIXME
Vc_SIMD_CAST_4(Scalar::int_v, SSE::uint_v) { return _mm_setr_epi32(x0.data(), x1.data(), x2.data(), x3.data()); }
Vc_SIMD_CAST_1(Scalar::int_v, SSE::short_v) { return _mm_setr_epi16(x.data(), 0, 0, 0, 0, 0, 0, 0); } // FIXME
Vc_SIMD_CAST_2(Scalar::int_v, SSE::short_v) { return _mm_setr_epi16(x0.data(), x1.data(), 0, 0, 0, 0, 0, 0); } // FIXME
Vc_SIMD_CAST_4(Scalar::int_v, SSE::short_v) { return _mm_setr_epi16(x0.data(), x1.data(), x2.data(), x3.data(), 0, 0, 0, 0); }
Vc_SIMD_CAST_8(Scalar::int_v, SSE::short_v) { return _mm_setr_epi16(x0.data(), x1.data(), x2.data(), x3.data(), x4.data(), x5.data(), x6.data(), x7.data()); }
Vc_SIMD_CAST_1(Scalar::int_v, SSE::ushort_v) { return _mm_setr_epi16(x.data(), 0, 0, 0, 0, 0, 0, 0); } // FIXME
Vc_SIMD_CAST_2(Scalar::int_v, SSE::ushort_v) { return _mm_setr_epi16(x0.data(), x1.data(), 0, 0, 0, 0, 0, 0); } // FIXME
Vc_SIMD_CAST_4(Scalar::int_v, SSE::ushort_v) { return _mm_setr_epi16(x0.data(), x1.data(), x2.data(), x3.data(), 0, 0, 0, 0); }
Vc_SIMD_CAST_8(Scalar::int_v, SSE::ushort_v) { return _mm_setr_epi16(x0.data(), x1.data(), x2.data(), x3.data(), x4.data(), x5.data(), x6.data(), x7.data()); }

Vc_SIMD_CAST_1(Scalar::uint_v, SSE::double_v) { return _mm_setr_pd(x.data(), 0); } // FIXME: register - register mov
Vc_SIMD_CAST_2(Scalar::uint_v, SSE::double_v) { return _mm_setr_pd(x0.data(), x1.data()); }
Vc_SIMD_CAST_1(Scalar::uint_v, SSE::float_v) { return _mm_setr_ps(x.data(), 0, 0, 0); } // FIXME
Vc_SIMD_CAST_2(Scalar::uint_v, SSE::float_v) { return _mm_setr_ps(x0.data(), x1.data(), 0, 0); } // FIXME
Vc_SIMD_CAST_4(Scalar::uint_v, SSE::float_v) { return _mm_setr_ps(x0.data(), x1.data(), x2.data(), x3.data()); }
Vc_SIMD_CAST_1(Scalar::uint_v, SSE::int_v) { return _mm_setr_epi32(x.data(), 0, 0, 0); } // FIXME
Vc_SIMD_CAST_2(Scalar::uint_v, SSE::int_v) { return _mm_setr_epi32(x0.data(), x1.data(), 0, 0); } // FIXME
Vc_SIMD_CAST_4(Scalar::uint_v, SSE::int_v) { return _mm_setr_epi32(x0.data(), x1.data(), x2.data(), x3.data()); }
Vc_SIMD_CAST_1(Scalar::uint_v, SSE::uint_v) { return _mm_setr_epi32(x.data(), 0, 0, 0); } // FIXME
Vc_SIMD_CAST_2(Scalar::uint_v, SSE::uint_v) { return _mm_setr_epi32(x0.data(), x1.data(), 0, 0); } // FIXME
Vc_SIMD_CAST_4(Scalar::uint_v, SSE::uint_v) { return _mm_setr_epi32(x0.data(), x1.data(), x2.data(), x3.data()); }
Vc_SIMD_CAST_1(Scalar::uint_v, SSE::short_v) { return _mm_setr_epi16(x.data(), 0, 0, 0, 0, 0, 0, 0); } // FIXME
Vc_SIMD_CAST_2(Scalar::uint_v, SSE::short_v) { return _mm_setr_epi16(x0.data(), x1.data(), 0, 0, 0, 0, 0, 0); } // FIXME
Vc_SIMD_CAST_4(Scalar::uint_v, SSE::short_v) { return _mm_setr_epi16(x0.data(), x1.data(), x2.data(), x3.data(), 0, 0, 0, 0); }
Vc_SIMD_CAST_8(Scalar::uint_v, SSE::short_v) { return _mm_setr_epi16(x0.data(), x1.data(), x2.data(), x3.data(), x4.data(), x5.data(), x6.data(), x7.data()); }
Vc_SIMD_CAST_1(Scalar::uint_v, SSE::ushort_v) { return _mm_setr_epi16(x.data(), 0, 0, 0, 0, 0, 0, 0); } // FIXME
Vc_SIMD_CAST_2(Scalar::uint_v, SSE::ushort_v) { return _mm_setr_epi16(x0.data(), x1.data(), 0, 0, 0, 0, 0, 0); } // FIXME
Vc_SIMD_CAST_4(Scalar::uint_v, SSE::ushort_v) { return _mm_setr_epi16(x0.data(), x1.data(), x2.data(), x3.data(), 0, 0, 0, 0); }
Vc_SIMD_CAST_8(Scalar::uint_v, SSE::ushort_v) { return _mm_setr_epi16(x0.data(), x1.data(), x2.data(), x3.data(), x4.data(), x5.data(), x6.data(), x7.data()); }

Vc_SIMD_CAST_1(Scalar::short_v, SSE::double_v) { return _mm_setr_pd(x.data(), 0); } // FIXME: register - register mov
Vc_SIMD_CAST_2(Scalar::short_v, SSE::double_v) { return _mm_setr_pd(x0.data(), x1.data()); }
Vc_SIMD_CAST_1(Scalar::short_v, SSE::float_v) { return _mm_setr_ps(x.data(), 0, 0, 0); } // FIXME
Vc_SIMD_CAST_2(Scalar::short_v, SSE::float_v) { return _mm_setr_ps(x0.data(), x1.data(), 0, 0); } // FIXME
Vc_SIMD_CAST_4(Scalar::short_v, SSE::float_v) { return _mm_setr_ps(x0.data(), x1.data(), x2.data(), x3.data()); }
Vc_SIMD_CAST_1(Scalar::short_v, SSE::int_v) { return _mm_setr_epi32(x.data(), 0, 0, 0); } // FIXME
Vc_SIMD_CAST_2(Scalar::short_v, SSE::int_v) { return _mm_setr_epi32(x0.data(), x1.data(), 0, 0); } // FIXME
Vc_SIMD_CAST_4(Scalar::short_v, SSE::int_v) { return _mm_setr_epi32(x0.data(), x1.data(), x2.data(), x3.data()); }
Vc_SIMD_CAST_1(Scalar::short_v, SSE::uint_v) { return _mm_setr_epi32(x.data(), 0, 0, 0); } // FIXME
Vc_SIMD_CAST_2(Scalar::short_v, SSE::uint_v) { return _mm_setr_epi32(x0.data(), x1.data(), 0, 0); } // FIXME
Vc_SIMD_CAST_4(Scalar::short_v, SSE::uint_v) { return _mm_setr_epi32(x0.data(), x1.data(), x2.data(), x3.data()); }
Vc_SIMD_CAST_1(Scalar::short_v, SSE::short_v) { return _mm_setr_epi16(x.data(), 0, 0, 0, 0, 0, 0, 0); } // FIXME
Vc_SIMD_CAST_2(Scalar::short_v, SSE::short_v) { return _mm_setr_epi16(x0.data(), x1.data(), 0, 0, 0, 0, 0, 0); } // FIXME
Vc_SIMD_CAST_4(Scalar::short_v, SSE::short_v) { return _mm_setr_epi16(x0.data(), x1.data(), x2.data(), x3.data(), 0, 0, 0, 0); }
Vc_SIMD_CAST_8(Scalar::short_v, SSE::short_v) { return _mm_setr_epi16(x0.data(), x1.data(), x2.data(), x3.data(), x4.data(), x5.data(), x6.data(), x7.data()); }
Vc_SIMD_CAST_1(Scalar::short_v, SSE::ushort_v) { return _mm_setr_epi16(x.data(), 0, 0, 0, 0, 0, 0, 0); } // FIXME
Vc_SIMD_CAST_2(Scalar::short_v, SSE::ushort_v) { return _mm_setr_epi16(x0.data(), x1.data(), 0, 0, 0, 0, 0, 0); } // FIXME
Vc_SIMD_CAST_4(Scalar::short_v, SSE::ushort_v) { return _mm_setr_epi16(x0.data(), x1.data(), x2.data(), x3.data(), 0, 0, 0, 0); }
Vc_SIMD_CAST_8(Scalar::short_v, SSE::ushort_v) { return _mm_setr_epi16(x0.data(), x1.data(), x2.data(), x3.data(), x4.data(), x5.data(), x6.data(), x7.data()); }

Vc_SIMD_CAST_1(Scalar::ushort_v, SSE::double_v) { return _mm_setr_pd(x.data(), 0); } // FIXME: register - register mov
Vc_SIMD_CAST_2(Scalar::ushort_v, SSE::double_v) { return _mm_setr_pd(x0.data(), x1.data()); }
Vc_SIMD_CAST_1(Scalar::ushort_v, SSE::float_v) { return _mm_setr_ps(x.data(), 0, 0, 0); } // FIXME
Vc_SIMD_CAST_2(Scalar::ushort_v, SSE::float_v) { return _mm_setr_ps(x0.data(), x1.data(), 0, 0); } // FIXME
Vc_SIMD_CAST_4(Scalar::ushort_v, SSE::float_v) { return _mm_setr_ps(x0.data(), x1.data(), x2.data(), x3.data()); }
Vc_SIMD_CAST_1(Scalar::ushort_v, SSE::int_v) { return _mm_setr_epi32(x.data(), 0, 0, 0); } // FIXME
Vc_SIMD_CAST_2(Scalar::ushort_v, SSE::int_v) { return _mm_setr_epi32(x0.data(), x1.data(), 0, 0); } // FIXME
Vc_SIMD_CAST_4(Scalar::ushort_v, SSE::int_v) { return _mm_setr_epi32(x0.data(), x1.data(), x2.data(), x3.data()); }
Vc_SIMD_CAST_1(Scalar::ushort_v, SSE::uint_v) { return _mm_setr_epi32(x.data(), 0, 0, 0); } // FIXME
Vc_SIMD_CAST_2(Scalar::ushort_v, SSE::uint_v) { return _mm_setr_epi32(x0.data(), x1.data(), 0, 0); } // FIXME
Vc_SIMD_CAST_4(Scalar::ushort_v, SSE::uint_v) { return _mm_setr_epi32(x0.data(), x1.data(), x2.data(), x3.data()); }
Vc_SIMD_CAST_1(Scalar::ushort_v, SSE::short_v) { return _mm_setr_epi16(x.data(), 0, 0, 0, 0, 0, 0, 0); } // FIXME
Vc_SIMD_CAST_2(Scalar::ushort_v, SSE::short_v) { return _mm_setr_epi16(x0.data(), x1.data(), 0, 0, 0, 0, 0, 0); } // FIXME
Vc_SIMD_CAST_4(Scalar::ushort_v, SSE::short_v) { return _mm_setr_epi16(x0.data(), x1.data(), x2.data(), x3.data(), 0, 0, 0, 0); }
Vc_SIMD_CAST_8(Scalar::ushort_v, SSE::short_v) { return _mm_setr_epi16(x0.data(), x1.data(), x2.data(), x3.data(), x4.data(), x5.data(), x6.data(), x7.data()); }
Vc_SIMD_CAST_1(Scalar::ushort_v, SSE::ushort_v) { return _mm_setr_epi16(x.data(), 0, 0, 0, 0, 0, 0, 0); } // FIXME
Vc_SIMD_CAST_2(Scalar::ushort_v, SSE::ushort_v) { return _mm_setr_epi16(x0.data(), x1.data(), 0, 0, 0, 0, 0, 0); } // FIXME
Vc_SIMD_CAST_4(Scalar::ushort_v, SSE::ushort_v) { return _mm_setr_epi16(x0.data(), x1.data(), x2.data(), x3.data(), 0, 0, 0, 0); }
Vc_SIMD_CAST_8(Scalar::ushort_v, SSE::ushort_v) { return _mm_setr_epi16(x0.data(), x1.data(), x2.data(), x3.data(), x4.data(), x5.data(), x6.data(), x7.data()); }

#define Vc_SIMD_CAST_SSE_TO_SCALAR(to__)                                                 \
    template <typename To, typename FromT>                                               \
    Vc_INTRINSIC Vc_CONST To                                                             \
        simd_cast(SSE::Vector<FromT> x,                                                  \
                  enable_if<std::is_same<To, Scalar::to__>::value> = nullarg)            \
    {                                                                                    \
        return static_cast<To>(x[0]);                                                    \
    }

VC_ALL_VECTOR_TYPES(Vc_SIMD_CAST_SSE_TO_SCALAR)
#undef Vc_SIMD_CAST_SSE_TO_SCALAR

// Mask casts without offset {{{1
// any one SSE Mask to one other SSE Mask
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST Return
    simd_cast(SSE::Mask<T> x, enable_if<SSE::is_mask<Return>::value> = nullarg)
{
    using M = SSE::Mask<T>;
    return {SSE::sse_cast<__m128>(SSE::internal::mask_cast<M::Size, Return::Size>(x.dataI()))};
}
// any two SSE Masks to one other SSE Mask
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST Return simd_cast(
    SSE::Mask<T> x0,
    SSE::Mask<T> x1,
    enable_if<SSE::is_mask<Return>::value && SSE::Mask<T>::Size * 2 == Return::Size> = nullarg)
{
    return SSE::sse_cast<__m128>(_mm_packs_epi16(x0.dataI(), x1.dataI()));
}
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST Return simd_cast(
    SSE::Mask<T> x0,
    SSE::Mask<T> x1,
    enable_if<SSE::is_mask<Return>::value && SSE::Mask<T>::Size * 4 == Return::Size> = nullarg)
{
    return SSE::sse_cast<__m128>(
        _mm_packs_epi16(_mm_packs_epi16(x0.dataI(), x1.dataI()), _mm_setzero_si128()));
}
// any four SSE Masks to one other SSE Mask
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST Return simd_cast(
    SSE::Mask<T> x0,
    SSE::Mask<T> x1,
    SSE::Mask<T> x2,
    SSE::Mask<T> x3,
    enable_if<SSE::is_mask<Return>::value && SSE::Mask<T>::Size * 4 == Return::Size> = nullarg)
{
    return SSE::sse_cast<__m128>(_mm_packs_epi16(_mm_packs_epi16(x0.dataI(), x1.dataI()),
                                                 _mm_packs_epi16(x2.dataI(), x3.dataI())));
}

// 1 Scalar Mask to 1 SSE Mask
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST Return
    simd_cast(Scalar::Mask<T> x, enable_if<SSE::is_mask<Return>::value> = nullarg)
{
    Return m(false);
    m[0] = x[0];
    return m;
}
// 2 Scalar Masks to 1 SSE Mask
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST Return
    simd_cast(Scalar::Mask<T> x0, Scalar::Mask<T> x1, enable_if<SSE::is_mask<Return>::value> = nullarg)
{
    Return m(false);
    m[0] = x0[0];
    m[1] = x1[0];
    return m;
}
// 4 Scalar Masks to 1 SSE Mask
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST Return simd_cast(Scalar::Mask<T> x0,
                                       Scalar::Mask<T> x1,
                                       Scalar::Mask<T> x2,
                                       Scalar::Mask<T> x3,
                                       enable_if<SSE::is_mask<Return>::value> = nullarg)
{
    Return m(false);
    m[0] = x0[0];
    m[1] = x1[0];
    if (Return::Size >= 4) {
        m[2] = x2[0];
        m[3] = x3[0];
    }
    return m;
}
// 8 Scalar Masks to 1 SSE Mask
template <typename Return, typename T>
Vc_INTRINSIC Vc_CONST Return simd_cast(Scalar::Mask<T> x0,
                                       Scalar::Mask<T> x1,
                                       Scalar::Mask<T> x2,
                                       Scalar::Mask<T> x3,
                                       Scalar::Mask<T> x4,
                                       Scalar::Mask<T> x5,
                                       Scalar::Mask<T> x6,
                                       Scalar::Mask<T> x7,
                                       enable_if<SSE::is_mask<Return>::value> = nullarg)
{
    Return m(false);
    m[0] = x0[0];
    m[1] = x1[0];
    if (Return::Size >= 4) {
        m[2] = x2[0];
        m[3] = x3[0];
    }
    if (Return::Size >= 8) {
        m[4] = x4[0];
        m[5] = x5[0];
        m[6] = x6[0];
        m[7] = x7[0];
    }
    return m;
}

// any 1 simd_mask_array to 1 SSE::Mask
template <typename Return, typename T, std::size_t N, typename V>
Vc_INTRINSIC Vc_CONST Return simd_cast(const simd_mask_array<T, N, V, N> &x,
                                       enable_if<SSE::is_mask<Return>::value> = nullarg)
{
    return simd_cast<Return>(internal_data(x));
}
template <typename Return, typename T, std::size_t N, typename V, std::size_t M>
Vc_INTRINSIC Vc_CONST Return simd_cast(
    const simd_mask_array<T, N, V, M> &x,
    enable_if<(SSE::is_mask<Return>::value &&
               Traits::decay<decltype(internal_data0(x))>::Size >= Return::Size)> = nullarg)
{
    return simd_cast<Return>(internal_data0(x));
}
template <typename Return, typename T, std::size_t N, typename V, std::size_t M>
Vc_INTRINSIC Vc_CONST Return simd_cast(
    const simd_mask_array<T, N, V, M> &x,
    enable_if<(SSE::is_mask<Return>::value &&
               Traits::decay<decltype(internal_data0(x))>::Size < Return::Size)> = nullarg)
{
    return simd_cast<Return>(internal_data0(x), internal_data1(x));
}

// any 2 simd_mask_array to 1 SSE::Mask
template <typename Return, typename T, std::size_t N, typename V>
Vc_INTRINSIC Vc_CONST Return simd_cast(const simd_mask_array<T, N, V, N> &x0,
                                       const simd_mask_array<T, N, V, N> &x1,
                                       enable_if<SSE::is_mask<Return>::value> = nullarg)
{
    return simd_cast<Return>(internal_data(x0), internal_data(x1));
}
template <typename Return, typename T, std::size_t N, typename V, std::size_t M>
Vc_INTRINSIC Vc_CONST Return simd_cast(
    const simd_mask_array<T, N, V, M> &x0,
    const simd_mask_array<T, N, V, M> &x1,
    enable_if<(SSE::is_mask<Return>::value &&
               // make sure all four arguments to the forwarded simd_cast are *required*
               N + Traits::decay<decltype(internal_data0(x1))>::Size < Return::Size)> = nullarg)
{
    return simd_cast<Return>(
        internal_data0(x0), internal_data1(x0), internal_data0(x1), internal_data1(x1));
}

// any 4 simd_mask_array to 1 SSE::Mask
template <typename Return, typename T, std::size_t N, typename V>
Vc_INTRINSIC Vc_CONST Return simd_cast(const simd_mask_array<T, N, V, N> &x0,
                                       const simd_mask_array<T, N, V, N> &x1,
                                       const simd_mask_array<T, N, V, N> &x2,
                                       const simd_mask_array<T, N, V, N> &x3,
                                       enable_if<SSE::is_mask<Return>::value> = nullarg)
{
    return simd_cast<Return>(
        internal_data(x0), internal_data(x1), internal_data(x2), internal_data(x3));
}
template <typename Return, typename T, std::size_t N, typename V>
Vc_INTRINSIC Vc_CONST Return simd_cast(const simd_mask_array<T, 2 * N, V, N> &x0,
                                       const simd_mask_array<T, 2 * N, V, N> &x1,
                                       const simd_mask_array<T, 2 * N, V, N> &x2,
                                       const simd_mask_array<T, 2 * N, V, N> &x3,
                                       enable_if<SSE::is_mask<Return>::value> = nullarg)
{
    return simd_cast<Return>(internal_data0(x0),
                             internal_data1(x0),
                             internal_data0(x1),
                             internal_data1(x1),
                             internal_data0(x2),
                             internal_data1(x2),
                             internal_data0(x3),
                             internal_data1(x3));
}
// any 8 simd_mask_array to 1 SSE::Mask
template <typename Return, typename T, std::size_t N, typename V>
Vc_INTRINSIC Vc_CONST Return
    simd_cast(const simd_mask_array<T, N, V, N> &x0,
              const simd_mask_array<T, N, V, N> &x1,
              const simd_mask_array<T, N, V, N> &x2,
              const simd_mask_array<T, N, V, N> &x3,
              const simd_mask_array<T, N, V, N> &,
              const simd_mask_array<T, N, V, N> &,
              const simd_mask_array<T, N, V, N> &,
              const simd_mask_array<T, N, V, N> &,
              enable_if<(SSE::is_mask<Return>::value && Return::Size <= N * 4)> = nullarg)
{
    return simd_cast<Return>(internal_data(x0),
                             internal_data(x1),
                             internal_data(x2),
                             internal_data(x3));
}
template <typename Return, typename T, std::size_t N, typename V>
Vc_INTRINSIC Vc_CONST Return
    simd_cast(const simd_mask_array<T, N, V, N> &x0,
              const simd_mask_array<T, N, V, N> &x1,
              const simd_mask_array<T, N, V, N> &x2,
              const simd_mask_array<T, N, V, N> &x3,
              const simd_mask_array<T, N, V, N> &x4,
              const simd_mask_array<T, N, V, N> &x5,
              const simd_mask_array<T, N, V, N> &x6,
              const simd_mask_array<T, N, V, N> &x7,
              enable_if<(SSE::is_mask<Return>::value && Return::Size > N * 4)> = nullarg)
{
    return simd_cast<Return>(internal_data(x0),
                             internal_data(x1),
                             internal_data(x2),
                             internal_data(x3),
                             internal_data(x4),
                             internal_data(x5),
                             internal_data(x6),
                             internal_data(x7));
}

// offset == 0 | convert from SSE::Mask/Vector to SSE/Scalar::Mask/Vector {{{1
template <typename Return, int offset, typename V>
Vc_INTRINSIC Vc_CONST Return simd_cast(
    V &&x,
    enable_if<offset == 0 &&
              ((SSE::is_vector<Traits::decay<V>>::value &&
                (SSE::is_vector<Return>::value || Scalar::is_vector<Return>::value)) ||
               (SSE::is_mask<Traits::decay<V>>::value &&
                (SSE::is_mask<Return>::value || Scalar::is_mask<Return>::value)))> =
        nullarg)
{
    return simd_cast<Return>(x);
}

// Vector casts with offset {{{1
// SSE to SSE (Vector)
template <typename Return, int offset, typename V>
Vc_INTRINSIC Vc_CONST Return simd_cast(
    V x,
    enable_if<offset != 0 && (SSE::is_vector<Return>::value && SSE::is_vector<V>::value)> = nullarg)
{
    constexpr int shift = (sizeof(V) / V::Size) * offset * Return::Size;
    static_assert(shift > 0 && shift < 16, "");
    return simd_cast<Return>(V{SSE::sse_cast<typename V::VectorType>(
        _mm_srli_si128(SSE::sse_cast<__m128i>(x.data()), shift))});
}

// SSE to Scalar (Vector)
template <typename Return, int offset, typename T>
Vc_INTRINSIC Vc_CONST Return
    simd_cast(SSE::Vector<T> x,
              enable_if<offset != 0 && Scalar::is_vector<Return>::value> = nullarg)
{
    using RT = typename Return::EntryType;
    const auto tmp = simd_cast<SSE::Vector<RT>>(x);
    return tmp[offset];
}

// Mask casts with offset {{{1
// SSE to SSE (Mask)
template <typename Return, int offset, typename V>
Vc_INTRINSIC Vc_CONST Return simd_cast(
    V x,
    enable_if<offset != 0 && (SSE::is_mask<Return>::value && SSE::is_mask<V>::value)> = nullarg)
{
    constexpr int shift = (sizeof(V) / V::Size) * offset * Return::Size;
    static_assert(shift > 0 && shift < 16, "");
    return simd_cast<Return>(V{SSE::sse_cast<typename V::VectorType>(
        _mm_srli_si128(SSE::sse_cast<__m128i>(x.data()), shift))});
}

// part of a simd_mask_array to SSE::Mask
template <typename Return, int offset, typename T, std::size_t N, typename V>
Vc_INTRINSIC Vc_CONST Return simd_cast(const simd_mask_array<T, N, V, N> &x,
                                       enable_if<SSE::is_mask<Return>::value> = nullarg)
{
    return simd_cast<Return, offset>(internal_data(x));
}
template <typename Return, int offset, typename T, std::size_t N, typename V, std::size_t M>
Vc_INTRINSIC Vc_CONST Return simd_cast(
    const simd_mask_array<T, N, V, M> &x,
    enable_if<(N > M) &&
              (Return::Size * offset < Traits::decay<decltype(internal_data0(x))>::Size) &&
              (Return::Size * (offset + 1) > Traits::decay<decltype(internal_data0(x))>::Size) &&
              SSE::is_mask<Return>::value> = nullarg)
{
    Return r;
    for (std::size_t i = 0; i < Return::Size; ++i) {
        r[i] = x[i + Return::Size * offset];
    }
    return r;
}
template <typename Return, int offset, typename T, std::size_t N, typename V, std::size_t M>
Vc_INTRINSIC Vc_CONST Return simd_cast(
    const simd_mask_array<T, N, V, M> &x,
    enable_if<(N > M) &&
              (Return::Size * offset < Traits::decay<decltype(internal_data0(x))>::Size) &&
              (Return::Size * (offset + 1) <= Traits::decay<decltype(internal_data0(x))>::Size) &&
              SSE::is_mask<Return>::value> = nullarg)
{
    return simd_cast<Return, offset>(internal_data0(x));
}
template <typename Return, int offset, typename T, std::size_t N, typename V, std::size_t M>
Vc_INTRINSIC Vc_CONST Return
    simd_cast(const simd_mask_array<T, N, V, M> &x,
              enable_if<(N > M) && (Return::Size * offset >=
                                    Traits::decay<decltype(internal_data0(x))>::Size) &&
                        SSE::is_mask<Return>::value> = nullarg)
{
    return simd_cast<Return,
                     offset - Traits::decay<decltype(internal_data0(x))>::Size / Return::Size>(
        internal_data1(x));
}

// undef Vc_SIMD_CAST_SSE_[124] & Vc_SIMD_CAST_[1248] {{{1
#undef Vc_SIMD_CAST_SSE_1
#undef Vc_SIMD_CAST_SSE_2
#undef Vc_SIMD_CAST_SSE_4

#undef Vc_SIMD_CAST_1
#undef Vc_SIMD_CAST_2
#undef Vc_SIMD_CAST_4
#undef Vc_SIMD_CAST_8
// }}}1

}  // namespace Vc

#include "undomacros.h"

#endif // VC_SSE_SIMD_CAST_H
