/*  This file is part of the Vc library.

    Copyright (C) 2009-2013 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_MIC_CASTS_H
#define VC_MIC_CASTS_H

#include "intrinsics.h"
#include "types.h"
#include "macros.h"

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)
    template<typename T> static Vc_INTRINSIC_L Vc_CONST_L T mic_cast(__m512  v) Vc_INTRINSIC_R Vc_CONST_R;
    template<typename T> static Vc_INTRINSIC_L Vc_CONST_L T mic_cast(__m512i v) Vc_INTRINSIC_R Vc_CONST_R;
    template<typename T> static Vc_INTRINSIC_L Vc_CONST_L T mic_cast(__m512d v) Vc_INTRINSIC_R Vc_CONST_R;

    template<> Vc_INTRINSIC Vc_CONST __m512  mic_cast(__m512  v) { return v; }
    template<> Vc_INTRINSIC Vc_CONST __m512  mic_cast(__m512i v) { return _mm512_castsi512_ps(v); }
    template<> Vc_INTRINSIC Vc_CONST __m512  mic_cast(__m512d v) { return _mm512_castpd_ps(v); }
    template<> Vc_INTRINSIC Vc_CONST __m512i mic_cast(__m512  v) { return _mm512_castps_si512(v); }
    template<> Vc_INTRINSIC Vc_CONST __m512i mic_cast(__m512i v) { return v; }
    template<> Vc_INTRINSIC Vc_CONST __m512i mic_cast(__m512d v) { return _mm512_castpd_si512(v); }
    template<> Vc_INTRINSIC Vc_CONST __m512d mic_cast(__m512  v) { return _mm512_castps_pd(v); }
    template<> Vc_INTRINSIC Vc_CONST __m512d mic_cast(__m512i v) { return _mm512_castsi512_pd(v); }
    template<> Vc_INTRINSIC Vc_CONST __m512d mic_cast(__m512d v) { return v; }

    template<typename From, typename To> struct StaticCastHelper {};
    template<typename Both> struct StaticCastHelper<Both, Both> {
        static __m512  cast(__m512  v) { return v; }
        static __m512d cast(__m512d v) { return v; }
        static __m512i cast(__m512i v) { return v; }
    };
    template<> struct StaticCastHelper<float         , short         > { static __m512i cast(__m512  v) { return _mm512_cvtfxpnt_round_adjustps_epi32(v, _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE); } };
    template<> struct StaticCastHelper<float         , unsigned short> { static __m512i cast(__m512  v) {
        // use conversion to epi32 on purpose here! Conversion to epu32 drops negative inputs. And
        // since we convert to 32bit ints the positive values are all covered.
        return _mm512_cvtfxpnt_round_adjustps_epi32(v, _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE);
    } };
    template<> struct StaticCastHelper<float         , int           > { static __m512i cast(__m512  v) { return _mm512_cvtfxpnt_round_adjustps_epi32(v, _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE); } };
    template<> struct StaticCastHelper<float         , unsigned int  > { static __m512i cast(__m512  v) {
        // cvtfxpntps2udq converts any negative input to 0
        // but static_cast<unsigned int>(-1.f) == static_cast<unsigned int>(-1)
        // => for negative input use cvtfxpntps2dq instead
        const auto negative = _mm512_cmplt_ps_mask(v, _mm512_setzero_ps());
        return _mm512_mask_cvtfxpnt_round_adjustps_epi32(
                    _mm512_cvtfxpnt_round_adjustps_epu32(v, _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE), negative,
                    v, _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE);
    } };
    template<> struct StaticCastHelper<float         , double        > { static __m512d cast(__m512  v) { return _mm512_cvtpslo_pd(v); } };
    template<> struct StaticCastHelper<double        , short         > { static __m512i cast(__m512d v) { return _mm512_cvtfxpnt_roundpd_epi32lo(v, _MM_ROUND_MODE_TOWARD_ZERO); } };
    template<> struct StaticCastHelper<double        , unsigned short> { static __m512i cast(__m512d v) {
        // use conversion to epi32 on purpose here! Conversion to epu32 drops negative inputs. And
        // since we convert to 32bit ints the positive values are all covered.
        return _mm512_cvtfxpnt_roundpd_epi32lo(v, _MM_ROUND_MODE_TOWARD_ZERO);
    } };
    template<> struct StaticCastHelper<double        , int           > { static __m512i cast(__m512d v) { return _mm512_cvtfxpnt_roundpd_epi32lo(v, _MM_ROUND_MODE_TOWARD_ZERO); } };
    template<> struct StaticCastHelper<double        , unsigned int  > { static __m512i cast(__m512d v) {
        // conversion of negative inputs needs to use _mm512_cvtfxpnt_roundpd_epi32lo
        const auto negative = _mm512_cmplt_pd_mask(v, _mm512_setzero_pd());
        return _mm512_mask_cvtfxpnt_roundpd_epi32lo(
                    _mm512_cvtfxpnt_roundpd_epu32lo(v, _MM_ROUND_MODE_TOWARD_ZERO), negative,
                    v, _MM_ROUND_MODE_TOWARD_ZERO);
    } };
    template<> struct StaticCastHelper<double        , float         > { static __m512  cast(__m512d v) { return _mm512_cvtpd_pslo(v); } };
    template<> struct StaticCastHelper<int           , short         > { static __m512i cast(__m512i v) { return v; } };
    template<> struct StaticCastHelper<int           , unsigned short> { static __m512i cast(__m512i v) { return v; } };
    template<> struct StaticCastHelper<int           , unsigned int  > { static __m512i cast(__m512i v) { return v; } };
    template<> struct StaticCastHelper<int           , float         > { static __m512  cast(__m512i v) { return _mm512_cvtfxpnt_round_adjustepi32_ps(v, _MM_FROUND_CUR_DIRECTION, _MM_EXPADJ_NONE); } };
    template<> struct StaticCastHelper<int           , double        > { static __m512d cast(__m512i v) { return _mm512_cvtepi32lo_pd(v); } };
    template<> struct StaticCastHelper<unsigned int  , short         > { static __m512i cast(__m512i v) { return v; } };
    template<> struct StaticCastHelper<unsigned int  , unsigned short> { static __m512i cast(__m512i v) { return v; } };
    template<> struct StaticCastHelper<unsigned int  , int           > { static __m512i cast(__m512i v) { return v; } };
    template<> struct StaticCastHelper<unsigned int  , float         > { static __m512  cast(__m512i v) { return _mm512_cvtfxpnt_round_adjustepu32_ps(v, _MM_FROUND_CUR_DIRECTION, _MM_EXPADJ_NONE); } };
    template<> struct StaticCastHelper<unsigned int  , double        > { static __m512d cast(__m512i v) { return _mm512_cvtepu32lo_pd(v); } };
    template<> struct StaticCastHelper<short         , int           > { static __m512i cast(__m512i v) { return v; } };
    template<> struct StaticCastHelper<short         , unsigned int  > { static __m512i cast(__m512i v) { return v; } };
    template<> struct StaticCastHelper<short         , unsigned short> { static __m512i cast(__m512i v) { return v; } };
    template<> struct StaticCastHelper<short         , float         > { static __m512  cast(__m512i v) { return _mm512_cvtfxpnt_round_adjustepi32_ps(v, _MM_FROUND_CUR_DIRECTION, _MM_EXPADJ_NONE); } };
    template<> struct StaticCastHelper<short         , double        > { static __m512d cast(__m512i v) { return _mm512_cvtepi32lo_pd(v); } };
    template<> struct StaticCastHelper<unsigned short, unsigned int  > { static __m512i cast(__m512i v) { return v; } };
    template<> struct StaticCastHelper<unsigned short, int           > { static __m512i cast(__m512i v) { return v; } };
    template<> struct StaticCastHelper<unsigned short, short         > { static __m512i cast(__m512i v) {
        const auto negative = _mm512_cmpgt_epu32_mask(v, _mm512_set1_epi32(0x7fff));
        return _mm512_mask_or_epi32(v, negative, v, _mm512_set1_epi32(0xffff0000u));
    } };
    template<> struct StaticCastHelper<unsigned short, float         > { static __m512  cast(__m512i v) { return _mm512_cvtfxpnt_round_adjustepu32_ps(v, _MM_FROUND_CUR_DIRECTION, _MM_EXPADJ_NONE); } };
    template<> struct StaticCastHelper<unsigned short, double        > { static __m512d cast(__m512i v) { return _mm512_cvtepu32lo_pd(v); } };

    template<typename From, typename To> struct ReinterpretCastHelper {};
    template<> struct ReinterpretCastHelper<float       , int         > { static __m512i cast(__m512  v) { return _mm512_castps_si512(v); } };
    template<> struct ReinterpretCastHelper<double      , int         > { static __m512i cast(__m512d v) { return _mm512_castpd_si512(v); } };
    template<> struct ReinterpretCastHelper<int         , int         > { static __m512i cast(__m512i v) { return v; } };
    template<> struct ReinterpretCastHelper<unsigned int, int         > { static __m512i cast(__m512i v) { return v; } };
    template<> struct ReinterpretCastHelper<float       , unsigned int> { static __m512i cast(__m512  v) { return _mm512_castps_si512(v); } };
    template<> struct ReinterpretCastHelper<double      , unsigned int> { static __m512i cast(__m512d v) { return _mm512_castpd_si512(v); } };
    template<> struct ReinterpretCastHelper<int         , unsigned int> { static __m512i cast(__m512i v) { return v; } };
    template<> struct ReinterpretCastHelper<unsigned int, unsigned int> { static __m512i cast(__m512i v) { return v; } };
    template<> struct ReinterpretCastHelper<float       , float       > { static __m512  cast(__m512  v) { return v; } };
    template<> struct ReinterpretCastHelper<double      , float       > { static __m512  cast(__m512d v) { return _mm512_castpd_ps(v);    } };
    template<> struct ReinterpretCastHelper<int         , float       > { static __m512  cast(__m512i v) { return _mm512_castsi512_ps(v); } };
    template<> struct ReinterpretCastHelper<unsigned int, float       > { static __m512  cast(__m512i v) { return _mm512_castsi512_ps(v); } };
    template<> struct ReinterpretCastHelper<float       , double      > { static __m512d cast(__m512  v) { return _mm512_castps_pd(v);    } };
    template<> struct ReinterpretCastHelper<double      , double      > { static __m512d cast(__m512d v) { return v; } };
    template<> struct ReinterpretCastHelper<int         , double      > { static __m512d cast(__m512i v) { return _mm512_castsi512_pd(v); } };
    template<> struct ReinterpretCastHelper<unsigned int, double      > { static __m512d cast(__m512i v) { return _mm512_castsi512_pd(v); } };
Vc_NAMESPACE_END

#include "undomacros.h"

#endif // VC_MIC_CASTS_H
