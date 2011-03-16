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

#ifndef VC_LARRABEE_CASTS_H
#define VC_LARRABEE_CASTS_H

#include "intrinsics.h"
#include "types.h"
#include "macros.h"

namespace Vc
{
namespace LRBni
{
    template<typename T> static inline T lrb_cast(__m512  v) INTRINSIC CONST;
    template<typename T> static inline T lrb_cast(__m512i v) INTRINSIC CONST;
    template<typename T> static inline T lrb_cast(__m512d v) INTRINSIC CONST;

    template<> inline __m512  INTRINSIC CONST lrb_cast(__m512  v) { return v; }
    template<> inline __m512  INTRINSIC CONST lrb_cast(__m512i v) { return _mm512_castsi512_ps(v); }
    template<> inline __m512  INTRINSIC CONST lrb_cast(__m512d v) { return _mm512_castpd_ps(v); }
    template<> inline __m512i INTRINSIC CONST lrb_cast(__m512  v) { return _mm512_castps_si512(v); }
    template<> inline __m512i INTRINSIC CONST lrb_cast(__m512i v) { return v; }
    template<> inline __m512i INTRINSIC CONST lrb_cast(__m512d v) { return _mm512_castpd_si512(v); }
    template<> inline __m512d INTRINSIC CONST lrb_cast(__m512  v) { return _mm512_castps_pd(v); }
    template<> inline __m512d INTRINSIC CONST lrb_cast(__m512i v) { return _mm512_castsi512_pd(v); }
    template<> inline __m512d INTRINSIC CONST lrb_cast(__m512d v) { return v; }

    template<typename From, typename To> struct StaticCastHelper {};
    template<> struct StaticCastHelper<float       , int         > { static _M512I cast(const _M512  &v) { return _mm512_cvt_ps2pi(v, _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE); } };
    template<> struct StaticCastHelper<float       , unsigned int> { static _M512I cast(const _M512  &v) { return _mm512_cvt_ps2pu(v, _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE); } };
    template<> struct StaticCastHelper<float       , float       > { static _M512  cast(const _M512  &v) { return v; } };
    template<> struct StaticCastHelper<float       , double      > { static _M512D cast(const _M512  &v) { return _mm512_cvtl_ps2pd(v); } };
    template<> struct StaticCastHelper<double      , int         > { static _M512I cast(const _M512D &v) { return _mm512_cvtl_pd2pi(_M512I(), v, _MM_ROUND_MODE_TOWARD_ZERO); } };
    template<> struct StaticCastHelper<double      , unsigned int> { static _M512I cast(const _M512D &v) { return _mm512_cvtl_pd2pu(_M512I(), v, _MM_ROUND_MODE_TOWARD_ZERO); } };
    template<> struct StaticCastHelper<double      , float       > { static _M512  cast(const _M512D &v) { return _mm512_cvtl_pd2ps(_M512(), v, _MM_ROUND_MODE_NEAREST); } };
    template<> struct StaticCastHelper<double      , double      > { static _M512D cast(const _M512D &v) { return v; } };
    template<> struct StaticCastHelper<int         , int         > { static _M512I cast(const _M512I &v) { return v; } };
    template<> struct StaticCastHelper<int         , unsigned int> { static _M512I cast(const _M512I &v) { return v; } };
    template<> struct StaticCastHelper<int         , float       > { static _M512  cast(const _M512I &v) { return _mm512_cvt_pi2ps(v, _MM_EXPADJ_NONE); } };
    template<> struct StaticCastHelper<int         , double      > { static _M512D cast(const _M512I &v) { return _mm512_cvtl_pi2pd(v); } };
    template<> struct StaticCastHelper<unsigned int, int         > { static _M512I cast(const _M512I &v) { return v; } };
    template<> struct StaticCastHelper<unsigned int, unsigned int> { static _M512I cast(const _M512I &v) { return v; } };
    template<> struct StaticCastHelper<unsigned int, float       > { static _M512  cast(const _M512I &v) { return _mm512_cvt_pu2ps(v, _MM_EXPADJ_NONE); } };
    template<> struct StaticCastHelper<unsigned int, double      > { static _M512D cast(const _M512I &v) { return _mm512_cvtl_pu2pd(v); } };

    template<typename From, typename To> struct ReinterpretCastHelper {};
    template<> struct ReinterpretCastHelper<float       , int         > { static _M512I cast(const _M512  &v) { return _mm512_castps_si512(v); } };
    template<> struct ReinterpretCastHelper<double      , int         > { static _M512I cast(const _M512D &v) { return _mm512_castpd_si512(v); } };
    template<> struct ReinterpretCastHelper<int         , int         > { static _M512I cast(const _M512I &v) { return v; } };
    template<> struct ReinterpretCastHelper<unsigned int, int         > { static _M512I cast(const _M512I &v) { return v; } };
    template<> struct ReinterpretCastHelper<float       , unsigned int> { static _M512I cast(const _M512  &v) { return _mm512_castps_si512(v); } };
    template<> struct ReinterpretCastHelper<double      , unsigned int> { static _M512I cast(const _M512D &v) { return _mm512_castpd_si512(v); } };
    template<> struct ReinterpretCastHelper<int         , unsigned int> { static _M512I cast(const _M512I &v) { return v; } };
    template<> struct ReinterpretCastHelper<unsigned int, unsigned int> { static _M512I cast(const _M512I &v) { return v; } };
    template<> struct ReinterpretCastHelper<float       , float       > { static _M512  cast(const _M512  &v) { return v; } };
    template<> struct ReinterpretCastHelper<double      , float       > { static _M512  cast(const _M512D &v) { return _mm512_castpd_ps(v);    } };
    template<> struct ReinterpretCastHelper<int         , float       > { static _M512  cast(const _M512I &v) { return _mm512_castsi512_ps(v); } };
    template<> struct ReinterpretCastHelper<unsigned int, float       > { static _M512  cast(const _M512I &v) { return _mm512_castsi512_ps(v); } };
    template<> struct ReinterpretCastHelper<float       , double      > { static _M512D cast(const _M512  &v) { return _mm512_castps_pd(v);    } };
    template<> struct ReinterpretCastHelper<double      , double      > { static _M512D cast(const _M512D &v) { return v; } };
    template<> struct ReinterpretCastHelper<int         , double      > { static _M512D cast(const _M512I &v) { return _mm512_castsi512_pd(v); } };
    template<> struct ReinterpretCastHelper<unsigned int, double      > { static _M512D cast(const _M512I &v) { return _mm512_castsi512_pd(v); } };
} // namespace LRBni
} // namespace Vc

#include "undomacros.h"

#endif // VC_LARRABEE_CASTS_H
