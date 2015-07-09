/*  This file is part of the Vc library. {{{
Copyright © 2009-2014 Matthias Kretz <kretz@kde.org>
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

#ifndef AVX_CASTS_H
#define AVX_CASTS_H

#include "intrinsics.h"
#include "types.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace AVX
{
namespace Casts
{
    template<typename T> Vc_INTRINSIC_L T avx_cast(__m128  v) Vc_INTRINSIC_R;
    template<typename T> Vc_INTRINSIC_L T avx_cast(__m128i v) Vc_INTRINSIC_R;
    template<typename T> Vc_INTRINSIC_L T avx_cast(__m128d v) Vc_INTRINSIC_R;
    template<typename T> Vc_INTRINSIC_L T avx_cast(__m256  v) Vc_INTRINSIC_R;
    template<typename T> Vc_INTRINSIC_L T avx_cast(__m256i v) Vc_INTRINSIC_R;
    template<typename T> Vc_INTRINSIC_L T avx_cast(__m256d v) Vc_INTRINSIC_R;

    // 128 -> 128
    template<> Vc_INTRINSIC __m128  avx_cast(__m128  v) { return v; }
    template<> Vc_INTRINSIC __m128  avx_cast(__m128i v) { return _mm_castsi128_ps(v); }
    template<> Vc_INTRINSIC __m128  avx_cast(__m128d v) { return _mm_castpd_ps(v); }
    template<> Vc_INTRINSIC __m128i avx_cast(__m128  v) { return _mm_castps_si128(v); }
    template<> Vc_INTRINSIC __m128i avx_cast(__m128i v) { return v; }
    template<> Vc_INTRINSIC __m128i avx_cast(__m128d v) { return _mm_castpd_si128(v); }
    template<> Vc_INTRINSIC __m128d avx_cast(__m128  v) { return _mm_castps_pd(v); }
    template<> Vc_INTRINSIC __m128d avx_cast(__m128i v) { return _mm_castsi128_pd(v); }
    template<> Vc_INTRINSIC __m128d avx_cast(__m128d v) { return v; }

    // 128 -> 256
    // FIXME: the following casts leave the upper 128bits undefined. With GCC and ICC I've never
    // seen the cast not do what I want though: after a VEX-coded SSE instruction the register's
    // upper 128bits are zero. Thus using the same register as AVX register will have the upper
    // 128bits zeroed. MSVC, though, implements _mm256_castxx128_xx256 with a 128bit move to memory
    // + 256bit load. Thus the upper 128bits are really undefined. But there is no intrinsic to do
    // what I want (i.e. alias the register, disallowing the move to memory in-between). I'm stuck,
    // do we really want to rely on specific compiler behavior here?
    template<> Vc_INTRINSIC __m256  avx_cast(__m128  v) { return _mm256_castps128_ps256(v); }
    template<> Vc_INTRINSIC __m256  avx_cast(__m128i v) { return _mm256_castps128_ps256(_mm_castsi128_ps(v)); }
    template<> Vc_INTRINSIC __m256  avx_cast(__m128d v) { return _mm256_castps128_ps256(_mm_castpd_ps(v)); }
    template<> Vc_INTRINSIC __m256i avx_cast(__m128  v) { return _mm256_castsi128_si256(_mm_castps_si128(v)); }
    template<> Vc_INTRINSIC __m256i avx_cast(__m128i v) { return _mm256_castsi128_si256(v); }
    template<> Vc_INTRINSIC __m256i avx_cast(__m128d v) { return _mm256_castsi128_si256(_mm_castpd_si128(v)); }
    template<> Vc_INTRINSIC __m256d avx_cast(__m128  v) { return _mm256_castpd128_pd256(_mm_castps_pd(v)); }
    template<> Vc_INTRINSIC __m256d avx_cast(__m128i v) { return _mm256_castpd128_pd256(_mm_castsi128_pd(v)); }
    template<> Vc_INTRINSIC __m256d avx_cast(__m128d v) { return _mm256_castpd128_pd256(v); }

#if defined VC_MSVC || defined VC_CLANG
    static Vc_INTRINSIC Vc_CONST __m256  zeroExtend(__m128  v) { return _mm256_permute2f128_ps   (_mm256_castps128_ps256(v), _mm256_castps128_ps256(v), 0x80); }
    static Vc_INTRINSIC Vc_CONST __m256i zeroExtend(__m128i v) { return _mm256_permute2f128_si256(_mm256_castsi128_si256(v), _mm256_castsi128_si256(v), 0x80); }
    static Vc_INTRINSIC Vc_CONST __m256d zeroExtend(__m128d v) { return _mm256_permute2f128_pd   (_mm256_castpd128_pd256(v), _mm256_castpd128_pd256(v), 0x80); }
#else
    static Vc_INTRINSIC Vc_CONST __m256  zeroExtend(__m128  v) { return _mm256_castps128_ps256(v); }
    static Vc_INTRINSIC Vc_CONST __m256i zeroExtend(__m128i v) { return _mm256_castsi128_si256(v); }
    static Vc_INTRINSIC Vc_CONST __m256d zeroExtend(__m128d v) { return _mm256_castpd128_pd256(v); }
#endif

    // 256 -> 128
    template<> Vc_INTRINSIC __m128  avx_cast(__m256  v) { return _mm256_castps256_ps128(v); }
    template<> Vc_INTRINSIC __m128  avx_cast(__m256i v) { return _mm256_castps256_ps128(_mm256_castsi256_ps(v)); }
    template<> Vc_INTRINSIC __m128  avx_cast(__m256d v) { return _mm256_castps256_ps128(_mm256_castpd_ps(v)); }
    template<> Vc_INTRINSIC __m128i avx_cast(__m256  v) { return _mm256_castsi256_si128(_mm256_castps_si256(v)); }
    template<> Vc_INTRINSIC __m128i avx_cast(__m256i v) { return _mm256_castsi256_si128(v); }
    template<> Vc_INTRINSIC __m128i avx_cast(__m256d v) { return _mm256_castsi256_si128(_mm256_castpd_si256(v)); }
    template<> Vc_INTRINSIC __m128d avx_cast(__m256  v) { return _mm256_castpd256_pd128(_mm256_castps_pd(v)); }
    template<> Vc_INTRINSIC __m128d avx_cast(__m256i v) { return _mm256_castpd256_pd128(_mm256_castsi256_pd(v)); }
    template<> Vc_INTRINSIC __m128d avx_cast(__m256d v) { return _mm256_castpd256_pd128(v); }

    // 256 -> 256
    template<> Vc_INTRINSIC __m256  avx_cast(__m256  v) { return v; }
    template<> Vc_INTRINSIC __m256  avx_cast(__m256i v) { return _mm256_castsi256_ps(v); }
    template<> Vc_INTRINSIC __m256  avx_cast(__m256d v) { return _mm256_castpd_ps(v); }
    template<> Vc_INTRINSIC __m256i avx_cast(__m256  v) { return _mm256_castps_si256(v); }
    template<> Vc_INTRINSIC __m256i avx_cast(__m256i v) { return v; }
    template<> Vc_INTRINSIC __m256i avx_cast(__m256d v) { return _mm256_castpd_si256(v); }
    template<> Vc_INTRINSIC __m256d avx_cast(__m256  v) { return _mm256_castps_pd(v); }
    template<> Vc_INTRINSIC __m256d avx_cast(__m256i v) { return _mm256_castsi256_pd(v); }
    template<> Vc_INTRINSIC __m256d avx_cast(__m256d v) { return v; }

    // simplify splitting 256-bit registers in 128-bit registers
    Vc_INTRINSIC Vc_CONST __m128  lo128(__m256  v) { return avx_cast<__m128>(v); }
    Vc_INTRINSIC Vc_CONST __m128d lo128(__m256d v) { return avx_cast<__m128d>(v); }
    Vc_INTRINSIC Vc_CONST __m128i lo128(__m256i v) { return avx_cast<__m128i>(v); }
    Vc_INTRINSIC Vc_CONST __m128  hi128(__m256  v) { return _mm256_extractf128_ps(v, 1); }
    Vc_INTRINSIC Vc_CONST __m128d hi128(__m256d v) { return _mm256_extractf128_pd(v, 1); }
    Vc_INTRINSIC Vc_CONST __m128i hi128(__m256i v) { return _mm256_extractf128_si256(v, 1); }

    // simplify combining 128-bit registers in 256-bit registers
    Vc_INTRINSIC Vc_CONST __m256  concat(__m128  a, __m128  b) { return _mm256_insertf128_ps   (avx_cast<__m256 >(a), b, 1); }
    Vc_INTRINSIC Vc_CONST __m256d concat(__m128d a, __m128d b) { return _mm256_insertf128_pd   (avx_cast<__m256d>(a), b, 1); }
    Vc_INTRINSIC Vc_CONST __m256i concat(__m128i a, __m128i b) { return _mm256_insertf128_si256(avx_cast<__m256i>(a), b, 1); }

}  // namespace Casts
using namespace Casts;
}  // namespace AVX

namespace AVX2
{
using namespace AVX::Casts;
}  // namespace AVX2

namespace Vc_IMPL_NAMESPACE
{
    template<typename From, typename To> struct StaticCastHelper {};
    template<> struct StaticCastHelper<float         , int           > { static Vc_ALWAYS_INLINE Vc_CONST m128i  cast(__m256  v) { return _mm_cvttps_epi32(lo128(v)); } };
    template<> struct StaticCastHelper<double        , int           > { static Vc_ALWAYS_INLINE Vc_CONST m128i  cast(__m256d v) { return _mm256_cvttpd_epi32(v); } };
    template<> struct StaticCastHelper<int           , int           > { static Vc_ALWAYS_INLINE Vc_CONST m128i  cast(__m128i v) { return v; } };
    template<> struct StaticCastHelper<unsigned int  , int           > { static Vc_ALWAYS_INLINE Vc_CONST m128i  cast(__m128i v) { return v; } };
    template<> struct StaticCastHelper<short         , int           > { static Vc_ALWAYS_INLINE Vc_CONST m128i  cast(__m128i v) { return _mm_srai_epi32(_mm_unpacklo_epi16(v, v), 16); } };
    template<> struct StaticCastHelper<unsigned short, int           > { static Vc_ALWAYS_INLINE Vc_CONST m128i  cast(__m128i v) { return _mm_srli_epi32(_mm_unpacklo_epi16(v, v), 16); } };
    template<> struct StaticCastHelper<float         , unsigned int  > { static inline Vc_CONST m128i  cast(__m256  v) {
            return _mm_blendv_epi8(
                _mm_cvttps_epi32(lo128(v)),
                _mm_add_epi32(
                    _mm_cvttps_epi32(_mm_sub_ps(lo128(v), _mm_set2power31_ps())),
                    _mm_set2power31_epu32()),
                _mm_castps_si128(_mm_cmpge_ps(lo128(v), _mm_set2power31_ps())));
    } };
    template<> struct StaticCastHelper<double        , unsigned int  > { static Vc_ALWAYS_INLINE Vc_CONST m128i  cast(__m256d v) {
            return _mm_add_epi32(m128i(_mm256_cvttpd_epi32(_mm256_sub_pd(
                                     _mm256_floor_pd(v), set1_pd(0x80000000u)))),
                                 _mm_set1_epi32(0x80000000u));
    } };
    template<> struct StaticCastHelper<int           , unsigned int  > { static Vc_ALWAYS_INLINE Vc_CONST m128i  cast(__m128i v) { return v; } };
    template<> struct StaticCastHelper<unsigned int  , unsigned int  > { static Vc_ALWAYS_INLINE Vc_CONST m128i  cast(__m128i v) { return v; } };
    template<> struct StaticCastHelper<short         , unsigned int  > { static Vc_ALWAYS_INLINE Vc_CONST m128i  cast(__m128i v) { return _mm_srli_epi32(_mm_unpacklo_epi16(v, v), 16); } };
    template<> struct StaticCastHelper<unsigned short, unsigned int  > { static Vc_ALWAYS_INLINE Vc_CONST m128i  cast(__m128i v) { return _mm_srli_epi32(_mm_unpacklo_epi16(v, v), 16); } };
    template<> struct StaticCastHelper<float         , float         > { static Vc_ALWAYS_INLINE Vc_CONST m256   cast(__m256  v) { return v; } };
    template<> struct StaticCastHelper<double        , float         > { static Vc_ALWAYS_INLINE Vc_CONST m256   cast(__m256d v) { return avx_cast<m256>(_mm256_cvtpd_ps(v)); } };
    template<> struct StaticCastHelper<int           , float         > { static Vc_ALWAYS_INLINE Vc_CONST m256   cast(__m128i v) { return zeroExtend(_mm_cvtepi32_ps(v)); } };
    template<> struct StaticCastHelper<unsigned int  , float         > { static inline Vc_CONST m256   cast(__m128i v) {
            return zeroExtend(_mm_blendv_ps(
                _mm_cvtepi32_ps(v),
                _mm_add_ps(_mm_cvtepi32_ps(_mm_sub_epi32(v, _mm_set2power31_epu32())),
                           _mm_set2power31_ps()),
                _mm_castsi128_ps(_mm_cmplt_epi32(v, _mm_setzero_si128()))));
    } };
    template<> struct StaticCastHelper<short         , float         > { static Vc_ALWAYS_INLINE Vc_CONST m256  cast(__m128i v) {
            return _mm256_cvtepi32_ps(
                concat(_mm_srai_epi32(_mm_unpacklo_epi16(v, v), 16),
                       _mm_srai_epi32(_mm_unpackhi_epi16(v, v), 16)));
    } };
    template<> struct StaticCastHelper<unsigned short, float         > { static Vc_ALWAYS_INLINE Vc_CONST m256  cast(__m128i v) {
            return _mm256_cvtepi32_ps(concat(_mm_unpacklo_epi16(v, _mm_setzero_si128()),
                                             _mm_unpackhi_epi16(v, _mm_setzero_si128())));
    } }; // FIXME: needs srli instead?
    template<> struct StaticCastHelper<float         , double        > { static Vc_ALWAYS_INLINE Vc_CONST m256d cast(__m256  v) { return _mm256_cvtps_pd(avx_cast<m128>(v)); } };
    template<> struct StaticCastHelper<double        , double        > { static Vc_ALWAYS_INLINE Vc_CONST m256d cast(__m256d v) { return v; } };
    template<> struct StaticCastHelper<int           , double        > { static Vc_ALWAYS_INLINE Vc_CONST m256d cast(__m128i v) { return _mm256_cvtepi32_pd(v); } };
    template<> struct StaticCastHelper<unsigned int  , double        > { static Vc_ALWAYS_INLINE Vc_CONST m256d cast(__m128i v) {
            return _mm256_add_pd(_mm256_cvtepi32_pd(_mm_sub_epi32(v, _mm_setmin_epi32())),
                                 set1_pd(1u << 31));
    } };
    template<> struct StaticCastHelper<int           , short         > { static Vc_ALWAYS_INLINE Vc_CONST m128i cast(__m128i v) { return _mm_packs_epi32(v, _mm_setzero_si128()); } };
    template<> struct StaticCastHelper<unsigned int  , short         > { static Vc_ALWAYS_INLINE Vc_CONST m128i cast(__m128i v) {
        const auto tmp0 = _mm_unpacklo_epi16(v, _mm_setzero_si128()); // 0 0 X X 1 0 X X
        const auto tmp1 = _mm_unpackhi_epi16(v, _mm_setzero_si128()); // 2 0 X X 3 0 X X
        const auto tmp2 = _mm_unpacklo_epi16(tmp0, tmp1); // 0 2 0 0 X X X X
        const auto tmp3 = _mm_unpackhi_epi16(tmp0, tmp1); // 1 3 0 0 X X X X
        return _mm_unpacklo_epi16(tmp2, tmp3); // 0 1 2 3 0 0 0 0
    } };
    template<> struct StaticCastHelper<float         , short         > { static Vc_ALWAYS_INLINE Vc_CONST m128i cast(__m256  v) {
            auto tmpi = _mm256_cvttps_epi32(v);
            auto tmp0 = _mm_unpacklo_epi16(lo128(tmpi), hi128(tmpi));  // 0 4 X X 1 5 X X
            auto tmp1 = _mm_unpackhi_epi16(lo128(tmpi), hi128(tmpi));  // 2 6 X X 3 7 X X
            auto tmp2 = _mm_unpacklo_epi16(tmp0, tmp1);                // 0 2 4 6 X X X X
            auto tmp3 = _mm_unpackhi_epi16(tmp0, tmp1);                // 1 3 5 7 X X X X
            return _mm_unpacklo_epi16(tmp2, tmp3);                     // 0 1 2 3 4 5 6 7
    } };
    template<> struct StaticCastHelper<short         , short         > { static Vc_ALWAYS_INLINE Vc_CONST m128i cast(__m128i v) { return v; } };
    template<> struct StaticCastHelper<unsigned short, short         > { static Vc_ALWAYS_INLINE Vc_CONST m128i cast(__m128i v) { return v; } };
    template<> struct StaticCastHelper<int           , unsigned short> { static Vc_ALWAYS_INLINE Vc_CONST m128i cast(__m128i v) {
        auto tmp0 = _mm_unpacklo_epi16(v, _mm_setzero_si128()); // 0 0 X X 1 0 X X
        auto tmp1 = _mm_unpackhi_epi16(v, _mm_setzero_si128()); // 2 0 X X 3 0 X X
        auto tmp2 = _mm_unpacklo_epi16(tmp0, tmp1); // 0 2 0 0 X X X X
        auto tmp3 = _mm_unpackhi_epi16(tmp0, tmp1); // 1 3 0 0 X X X X
        return _mm_unpacklo_epi16(tmp2, tmp3); // 0 1 2 3 0 0 0 0
    } };
    template<> struct StaticCastHelper<unsigned int  , unsigned short> { static Vc_ALWAYS_INLINE Vc_CONST m128i cast(__m128i v) {
        auto tmp0 = _mm_unpacklo_epi16(v, _mm_setzero_si128()); // 0 0 X X 1 0 X X
        auto tmp1 = _mm_unpackhi_epi16(v, _mm_setzero_si128()); // 2 0 X X 3 0 X X
        auto tmp2 = _mm_unpacklo_epi16(tmp0, tmp1); // 0 2 0 0 X X X X
        auto tmp3 = _mm_unpackhi_epi16(tmp0, tmp1); // 1 3 0 0 X X X X
        return _mm_unpacklo_epi16(tmp2, tmp3); // 0 1 2 3 0 0 0 0
    } };
    template<> struct StaticCastHelper<float         , unsigned short> { static Vc_ALWAYS_INLINE Vc_CONST m128i cast(__m256  v) {
            auto tmpi = avx_cast<m256i>(_mm256_blendv_ps(
                avx_cast<m256>(_mm256_cvttps_epi32(v)),
                avx_cast<m256>(
                    add_epi32(_mm256_cvttps_epi32(_mm256_sub_ps(v, set2power31_ps())),
                              set2power31_epu32())),
                cmpge_ps(v, set2power31_ps())));
            auto tmp0 = _mm_unpacklo_epi16(lo128(tmpi), hi128(tmpi));  // 0 4 X X 1 5 X X
            auto tmp1 = _mm_unpackhi_epi16(lo128(tmpi), hi128(tmpi));  // 2 6 X X 3 7 X X
            auto tmp2 = _mm_unpacklo_epi16(tmp0, tmp1);                // 0 2 4 6 X X X X
            auto tmp3 = _mm_unpackhi_epi16(tmp0, tmp1);                // 1 3 5 7 X X X X
            return _mm_unpacklo_epi16(tmp2, tmp3);                     // 0 1 2 3 4 5 6 7
    } };
    template<> struct StaticCastHelper<short         , unsigned short> { static Vc_ALWAYS_INLINE Vc_CONST m128i cast(__m128i v) { return v; } };
    template<> struct StaticCastHelper<unsigned short, unsigned short> { static Vc_ALWAYS_INLINE Vc_CONST m128i cast(__m128i v) { return v; } };
}  // namespace AVX(2)
}  // namespace Vc

#include "undomacros.h"

#endif // AVX_CASTS_H
