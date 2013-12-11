/*  This file is part of the Vc library.

    Copyright (C) 2009-2011 Matthias Kretz <kretz@kde.org>

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

#ifndef SSE_CASTS_H
#define SSE_CASTS_H

#include "intrinsics.h"
#include "types.h"

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)
    template<typename To, typename From> static Vc_ALWAYS_INLINE To Vc_CONST mm128_reinterpret_cast(VC_ALIGNED_PARAMETER(From) v) { return v; }
    template<> Vc_ALWAYS_INLINE _M128I Vc_CONST mm128_reinterpret_cast<_M128I, _M128 >(VC_ALIGNED_PARAMETER(_M128 ) v) { return _mm_castps_si128(v); }
    template<> Vc_ALWAYS_INLINE _M128I Vc_CONST mm128_reinterpret_cast<_M128I, _M128D>(VC_ALIGNED_PARAMETER(_M128D) v) { return _mm_castpd_si128(v); }
    template<> Vc_ALWAYS_INLINE _M128  Vc_CONST mm128_reinterpret_cast<_M128 , _M128D>(VC_ALIGNED_PARAMETER(_M128D) v) { return _mm_castpd_ps(v);    }
    template<> Vc_ALWAYS_INLINE _M128  Vc_CONST mm128_reinterpret_cast<_M128 , _M128I>(VC_ALIGNED_PARAMETER(_M128I) v) { return _mm_castsi128_ps(v); }
    template<> Vc_ALWAYS_INLINE _M128D Vc_CONST mm128_reinterpret_cast<_M128D, _M128I>(VC_ALIGNED_PARAMETER(_M128I) v) { return _mm_castsi128_pd(v); }
    template<> Vc_ALWAYS_INLINE _M128D Vc_CONST mm128_reinterpret_cast<_M128D, _M128 >(VC_ALIGNED_PARAMETER(_M128 ) v) { return _mm_castps_pd(v);    }
    template<typename To, typename From> static Vc_ALWAYS_INLINE To Vc_CONST sse_cast(VC_ALIGNED_PARAMETER(From) v) { return mm128_reinterpret_cast<To, From>(v); }

    template<typename From, typename To> struct StaticCastHelper;
    template<> struct StaticCastHelper<float       , int         > { static Vc_ALWAYS_INLINE _M128I cast(const _M128  &v) { return _mm_cvttps_epi32(v); } };
    template<> struct StaticCastHelper<double      , int         > { static Vc_ALWAYS_INLINE _M128I cast(const _M128D &v) { return _mm_cvttpd_epi32(v); } };
    template<> struct StaticCastHelper<int         , int         > { static Vc_ALWAYS_INLINE _M128I cast(const _M128I &v) { return v; } };
    template<> struct StaticCastHelper<unsigned int, int         > { static Vc_ALWAYS_INLINE _M128I cast(const _M128I &v) { return v; } };
    template<> struct StaticCastHelper<float       , unsigned int> { static Vc_ALWAYS_INLINE _M128I cast(const _M128  &v) {
        return _mm_castps_si128(_mm_blendv_ps(
                _mm_castsi128_ps(_mm_cvttps_epi32(v)),
                _mm_castsi128_ps(_mm_add_epi32(_mm_cvttps_epi32(_mm_sub_ps(v, _mm_set1_ps(1u << 31))), _mm_set1_epi32(1 << 31))),
                _mm_cmpge_ps(v, _mm_set1_ps(1u << 31))
                ));

    } };
    template<> struct StaticCastHelper<double      , unsigned int> { static Vc_ALWAYS_INLINE _M128I cast(const _M128D &v) {
            return _mm_add_epi32(_mm_cvttpd_epi32(_mm_sub_pd(v, _mm_set1_pd(0x80000000u))),
                                 _mm_cvtsi64_si128(0x8000000080000000ull));
    } };
    template<> struct StaticCastHelper<int         , unsigned int> { static Vc_ALWAYS_INLINE _M128I cast(const _M128I &v) { return v; } };
    template<> struct StaticCastHelper<unsigned int, unsigned int> { static Vc_ALWAYS_INLINE _M128I cast(const _M128I &v) { return v; } };
    template<> struct StaticCastHelper<float       , float       > { static Vc_ALWAYS_INLINE _M128  cast(const _M128  &v) { return v; } };
    template<> struct StaticCastHelper<double      , float       > { static Vc_ALWAYS_INLINE _M128  cast(const _M128D &v) { return _mm_cvtpd_ps(v); } };
    template<> struct StaticCastHelper<int         , float       > { static Vc_ALWAYS_INLINE _M128  cast(const _M128I &v) { return _mm_cvtepi32_ps(v); } };
    template<> struct StaticCastHelper<unsigned int, float       > { static Vc_ALWAYS_INLINE _M128  cast(const _M128I &v) {
        return _mm_blendv_ps(
                _mm_cvtepi32_ps(v),
                _mm_add_ps(_mm_cvtepi32_ps(_mm_sub_epi32(v, _mm_setmin_epi32())), _mm_set1_ps(1u << 31)),
                _mm_castsi128_ps(_mm_cmplt_epi32(v, _mm_setzero_si128()))
                );
    } };
    template<> struct StaticCastHelper<float       , double      > { static Vc_ALWAYS_INLINE _M128D cast(const _M128  &v) { return _mm_cvtps_pd(v); } };
    template<> struct StaticCastHelper<double      , double      > { static Vc_ALWAYS_INLINE _M128D cast(const _M128D &v) { return v; } };
    template<> struct StaticCastHelper<int         , double      > { static Vc_ALWAYS_INLINE _M128D cast(const _M128I &v) { return _mm_cvtepi32_pd(v); } };
    template<> struct StaticCastHelper<unsigned int, double      > { static Vc_ALWAYS_INLINE _M128D cast(const _M128I &v) {
            return _mm_add_pd(_mm_cvtepi32_pd(_mm_sub_epi32(v, _mm_setmin_epi32())),
                              _mm_set1_pd(1u << 31));
    } };

    template<> struct StaticCastHelper<float         , short         > { static Vc_ALWAYS_INLINE _M128I cast(const _M128  &v) { return _mm_packs_epi32(_mm_cvttps_epi32(v), _mm_setzero_si128()); } };
    template<> struct StaticCastHelper<int           , short         > { static Vc_ALWAYS_INLINE _M128I cast(const _M128I &v) { return _mm_packs_epi32(v, _mm_setzero_si128()); } };
    template<> struct StaticCastHelper<unsigned int  , short         > { static Vc_ALWAYS_INLINE _M128I cast(const _M128I &v) { return _mm_packs_epi32(v, _mm_setzero_si128()); } };
    template<> struct StaticCastHelper<short         , short         > { static Vc_ALWAYS_INLINE _M128I cast(const _M128I &v) { return v; } };
    template<> struct StaticCastHelper<unsigned short, short         > { static Vc_ALWAYS_INLINE _M128I cast(const _M128I &v) { return v; } };
    template<> struct StaticCastHelper<double        , short         > { static Vc_ALWAYS_INLINE _M128I cast(const _M128D &v) { return StaticCastHelper<int, short>::cast(StaticCastHelper<double, int>::cast(v)); } };
    template<> struct StaticCastHelper<int           , unsigned short> { static Vc_ALWAYS_INLINE _M128I cast(const _M128I &v) {
        auto tmp0 = _mm_unpacklo_epi16(v, _mm_setzero_si128()); // 0 4 X X 1 5 X X
        auto tmp1 = _mm_unpackhi_epi16(v, _mm_setzero_si128()); // 2 6 X X 3 7 X X
        auto tmp2 = _mm_unpacklo_epi16(tmp0, tmp1); // 0 2 4 6 X X X X
        auto tmp3 = _mm_unpackhi_epi16(tmp0, tmp1); // 1 3 5 7 X X X X
        return _mm_unpacklo_epi16(tmp2, tmp3); // 0 1 2 3 4 5 6 7
    } };
    template<> struct StaticCastHelper<unsigned int  , unsigned short> { static Vc_ALWAYS_INLINE _M128I cast(const _M128I &v) {
        auto tmp0 = _mm_unpacklo_epi16(v, _mm_setzero_si128()); // 0 4 X X 1 5 X X
        auto tmp1 = _mm_unpackhi_epi16(v, _mm_setzero_si128()); // 2 6 X X 3 7 X X
        auto tmp2 = _mm_unpacklo_epi16(tmp0, tmp1); // 0 2 4 6 X X X X
        auto tmp3 = _mm_unpackhi_epi16(tmp0, tmp1); // 1 3 5 7 X X X X
        return _mm_unpacklo_epi16(tmp2, tmp3); // 0 1 2 3 4 5 6 7
    } };
    template<> struct StaticCastHelper<float         , unsigned short> { static Vc_ALWAYS_INLINE _M128I cast(const _M128  &v) { return StaticCastHelper<int, unsigned short>::cast(_mm_cvttps_epi32(v)); } };
    template<> struct StaticCastHelper<short         , unsigned short> { static Vc_ALWAYS_INLINE _M128I cast(const _M128I &v) { return v; } };
    template<> struct StaticCastHelper<unsigned short, unsigned short> { static Vc_ALWAYS_INLINE _M128I cast(const _M128I &v) { return v; } };
    template<> struct StaticCastHelper<double        , unsigned short> { static Vc_ALWAYS_INLINE _M128I cast(const _M128D &v) { return StaticCastHelper<int, unsigned short>::cast(StaticCastHelper<double, int>::cast(v)); } };
Vc_IMPL_NAMESPACE_END

#endif // SSE_CASTS_H
