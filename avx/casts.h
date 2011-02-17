/*  This file is part of the Vc library.

    Copyright (C) 2009-2010 Matthias Kretz <kretz@kde.org>

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

#ifndef AVX_CASTS_H
#define AVX_CASTS_H

#include "intrinsics.h"
#include "types.h"

namespace Vc
{
namespace AVX
{
    template<typename T> static inline T avx_cast(__m128  v) INTRINSIC;
    template<typename T> static inline T avx_cast(__m128i v) INTRINSIC;
    template<typename T> static inline T avx_cast(__m128d v) INTRINSIC;
    template<typename T> static inline T avx_cast(__m256  v) INTRINSIC;
    template<typename T> static inline T avx_cast(__m256i v) INTRINSIC;
    template<typename T> static inline T avx_cast(__m256d v) INTRINSIC;

    // 128 -> 128
    template<> inline __m128  INTRINSIC avx_cast(__m128  v) { return v; }
    template<> inline __m128  INTRINSIC avx_cast(__m128i v) { return _mm_castsi128_ps(v); }
    template<> inline __m128  INTRINSIC avx_cast(__m128d v) { return _mm_castpd_ps(v); }
    template<> inline __m128i INTRINSIC avx_cast(__m128  v) { return _mm_castps_si128(v); }
    template<> inline __m128i INTRINSIC avx_cast(__m128i v) { return v; }
    template<> inline __m128i INTRINSIC avx_cast(__m128d v) { return _mm_castpd_si128(v); }
    template<> inline __m128d INTRINSIC avx_cast(__m128  v) { return _mm_castps_pd(v); }
    template<> inline __m128d INTRINSIC avx_cast(__m128i v) { return _mm_castsi128_pd(v); }
    template<> inline __m128d INTRINSIC avx_cast(__m128d v) { return v; }

    // 128 -> 256
    template<> inline __m256  INTRINSIC avx_cast(__m128  v) { return _mm256_castps128_ps256(v); }
    template<> inline __m256  INTRINSIC avx_cast(__m128i v) { return _mm256_castps128_ps256(_mm_castsi128_ps(v)); }
    template<> inline __m256  INTRINSIC avx_cast(__m128d v) { return _mm256_castps128_ps256(_mm_castpd_ps(v)); }
    template<> inline __m256i INTRINSIC avx_cast(__m128  v) { return _mm256_castsi128_si256(_mm_castps_si128(v)); }
    template<> inline __m256i INTRINSIC avx_cast(__m128i v) { return _mm256_castsi128_si256(v); }
    template<> inline __m256i INTRINSIC avx_cast(__m128d v) { return _mm256_castsi128_si256(_mm_castpd_si128(v)); }
    template<> inline __m256d INTRINSIC avx_cast(__m128  v) { return _mm256_castpd128_pd256(_mm_castps_pd(v)); }
    template<> inline __m256d INTRINSIC avx_cast(__m128i v) { return _mm256_castpd128_pd256(_mm_castsi128_pd(v)); }
    template<> inline __m256d INTRINSIC avx_cast(__m128d v) { return _mm256_castpd128_pd256(v); }

    // 256 -> 128
    template<> inline __m128  INTRINSIC avx_cast(__m256  v) { return _mm256_castps256_ps128(v); }
    template<> inline __m128  INTRINSIC avx_cast(__m256i v) { return _mm256_castps256_ps128(_mm256_castsi256_ps(v)); }
    template<> inline __m128  INTRINSIC avx_cast(__m256d v) { return _mm256_castps256_ps128(_mm256_castpd_ps(v)); }
    template<> inline __m128i INTRINSIC avx_cast(__m256  v) { return _mm256_castsi256_si128(_mm256_castps_si256(v)); }
    template<> inline __m128i INTRINSIC avx_cast(__m256i v) { return _mm256_castsi256_si128(v); }
    template<> inline __m128i INTRINSIC avx_cast(__m256d v) { return _mm256_castsi256_si128(_mm256_castpd_si256(v)); }
    template<> inline __m128d INTRINSIC avx_cast(__m256  v) { return _mm256_castpd256_pd128(_mm256_castps_pd(v)); }
    template<> inline __m128d INTRINSIC avx_cast(__m256i v) { return _mm256_castpd256_pd128(_mm256_castsi256_pd(v)); }
    template<> inline __m128d INTRINSIC avx_cast(__m256d v) { return _mm256_castpd256_pd128(v); }

    // 256 -> 256
    template<> inline __m256  INTRINSIC avx_cast(__m256  v) { return v; }
    template<> inline __m256  INTRINSIC avx_cast(__m256i v) { return _mm256_castsi256_ps(v); }
    template<> inline __m256  INTRINSIC avx_cast(__m256d v) { return _mm256_castpd_ps(v); }
    template<> inline __m256i INTRINSIC avx_cast(__m256  v) { return _mm256_castps_si256(v); }
    template<> inline __m256i INTRINSIC avx_cast(__m256i v) { return v; }
    template<> inline __m256i INTRINSIC avx_cast(__m256d v) { return _mm256_castpd_si256(v); }
    template<> inline __m256d INTRINSIC avx_cast(__m256  v) { return _mm256_castps_pd(v); }
    template<> inline __m256d INTRINSIC avx_cast(__m256i v) { return _mm256_castsi256_pd(v); }
    template<> inline __m256d INTRINSIC avx_cast(__m256d v) { return v; }


    template<typename To, typename From> static inline To mm256_reinterpret_cast(From v) CONST;
    template<typename To, typename From> static inline To mm256_reinterpret_cast(From v) { return v; }
    template<> inline _M256I mm256_reinterpret_cast<_M256I, _M256 >(_M256  v) CONST;
    template<> inline _M256I mm256_reinterpret_cast<_M256I, _M256D>(_M256D v) CONST;
    template<> inline _M256  mm256_reinterpret_cast<_M256 , _M256D>(_M256D v) CONST;
    template<> inline _M256  mm256_reinterpret_cast<_M256 , _M256I>(_M256I v) CONST;
    template<> inline _M256D mm256_reinterpret_cast<_M256D, _M256I>(_M256I v) CONST;
    template<> inline _M256D mm256_reinterpret_cast<_M256D, _M256 >(_M256  v) CONST;
    template<> inline _M256I mm256_reinterpret_cast<_M256I, _M256 >(_M256  v) { return _mm256_castps_si256(v); }
    template<> inline _M256I mm256_reinterpret_cast<_M256I, _M256D>(_M256D v) { return _mm256_castpd_si256(v); }
    template<> inline _M256  mm256_reinterpret_cast<_M256 , _M256D>(_M256D v) { return _mm256_castpd_ps(v);    }
    template<> inline _M256  mm256_reinterpret_cast<_M256 , _M256I>(_M256I v) { return _mm256_castsi256_ps(v); }
    template<> inline _M256D mm256_reinterpret_cast<_M256D, _M256I>(_M256I v) { return _mm256_castsi256_pd(v); }
    template<> inline _M256D mm256_reinterpret_cast<_M256D, _M256 >(_M256  v) { return _mm256_castps_pd(v);    }

    template<typename From, typename To> struct StaticCastHelper {};
    template<> struct StaticCastHelper<float       , int         > { static _M256I cast(const _M256  &v) { return _mm256_cvttps_epi32(v); } };
    template<> struct StaticCastHelper<double      , int         > { static _M256I cast(const _M256D &v) { return avx_cast<_M256I>(_mm256_cvttpd_epi32(v)); } };
    template<> struct StaticCastHelper<int         , int         > { static _M256I cast(const _M256I &v) { return v; } };
    template<> struct StaticCastHelper<unsigned int, int         > { static _M256I cast(const _M256I &v) { return v; } };
    template<> struct StaticCastHelper<float       , unsigned int> { static _M256I cast(const _M256  &v) { return _mm256_cvttps_epi32(v); } };
    template<> struct StaticCastHelper<double      , unsigned int> { static _M256I cast(const _M256D &v) { return avx_cast<_M256I>(_mm256_cvttpd_epi32(v)); } };
    template<> struct StaticCastHelper<int         , unsigned int> { static _M256I cast(const _M256I &v) { return v; } };
    template<> struct StaticCastHelper<unsigned int, unsigned int> { static _M256I cast(const _M256I &v) { return v; } };
    template<> struct StaticCastHelper<float       , float       > { static _M256  cast(const _M256  &v) { return v; } };
    template<> struct StaticCastHelper<double      , float       > { static _M256  cast(const _M256D &v) { return avx_cast<_M256>(_mm256_cvtpd_ps(v)); } };
    template<> struct StaticCastHelper<int         , float       > { static _M256  cast(const _M256I &v) { return _mm256_cvtepi32_ps(v); } };
    template<> struct StaticCastHelper<unsigned int, float       > { static _M256  cast(const _M256I &v) { return _mm256_cvtepi32_ps(v); } };
    template<> struct StaticCastHelper<float       , double      > { static _M256D cast(const _M256  &v) { return _mm256_cvtps_pd(avx_cast<__m128>(v)); } };
    template<> struct StaticCastHelper<double      , double      > { static _M256D cast(const _M256D &v) { return v; } };
    template<> struct StaticCastHelper<int         , double      > { static _M256D cast(const _M256I &v) { return _mm256_cvtepi32_pd(avx_cast<__m128i>(v)); } };
    template<> struct StaticCastHelper<unsigned int, double      > { static _M256D cast(const _M256I &v) { return _mm256_cvtepi32_pd(avx_cast<__m128i>(v)); } };

    template<> struct StaticCastHelper<float         , short         > { static _M256I cast(const _M256  &v) { return _mm256_packs_epi32(_mm256_cvttps_epi32(v), _mm256_setzero_si256()); } };
    template<> struct StaticCastHelper<short         , short         > { static _M256I cast(const _M256I &v) { return v; } };
    template<> struct StaticCastHelper<unsigned short, short         > { static _M256I cast(const _M256I &v) { return v; } };
    template<> struct StaticCastHelper<float         , unsigned short> { static _M256I cast(const _M256  &v) { return _mm256_packs_epi32(_mm256_cvttps_epi32(v), _mm256_setzero_si256()); } };
    template<> struct StaticCastHelper<short         , unsigned short> { static _M256I cast(const _M256I &v) { return v; } };
    template<> struct StaticCastHelper<unsigned short, unsigned short> { static _M256I cast(const _M256I &v) { return v; } };

    template<typename From, typename To> struct ReinterpretCastHelper {};
    template<> struct ReinterpretCastHelper<float       , int         > { static _M256I cast(const _M256  &v) { return _mm256_castps_si256(v); } };
    template<> struct ReinterpretCastHelper<double      , int         > { static _M256I cast(const _M256D &v) { return _mm256_castpd_si256(v); } };
    template<> struct ReinterpretCastHelper<int         , int         > { static _M256I cast(const _M256I &v) { return v; } };
    template<> struct ReinterpretCastHelper<unsigned int, int         > { static _M256I cast(const _M256I &v) { return v; } };
    template<> struct ReinterpretCastHelper<float       , unsigned int> { static _M256I cast(const _M256  &v) { return _mm256_castps_si256(v); } };
    template<> struct ReinterpretCastHelper<double      , unsigned int> { static _M256I cast(const _M256D &v) { return _mm256_castpd_si256(v); } };
    template<> struct ReinterpretCastHelper<int         , unsigned int> { static _M256I cast(const _M256I &v) { return v; } };
    template<> struct ReinterpretCastHelper<unsigned int, unsigned int> { static _M256I cast(const _M256I &v) { return v; } };
    template<> struct ReinterpretCastHelper<float       , float       > { static _M256  cast(const _M256  &v) { return v; } };
    template<> struct ReinterpretCastHelper<double      , float       > { static _M256  cast(const _M256D &v) { return _mm256_castpd_ps(v); } };
    template<> struct ReinterpretCastHelper<int         , float       > { static _M256  cast(const _M256I &v) { return _mm256_castsi256_ps(v); } };
    template<> struct ReinterpretCastHelper<unsigned int, float       > { static _M256  cast(const _M256I &v) { return _mm256_castsi256_ps(v); } };
    template<> struct ReinterpretCastHelper<float       , double      > { static _M256D cast(const _M256  &v) { return _mm256_castps_pd(v); } };
    template<> struct ReinterpretCastHelper<double      , double      > { static _M256D cast(const _M256D &v) { return v; } };
    template<> struct ReinterpretCastHelper<int         , double      > { static _M256D cast(const _M256I &v) { return _mm256_castsi256_pd(v); } };
    template<> struct ReinterpretCastHelper<unsigned int, double      > { static _M256D cast(const _M256I &v) { return _mm256_castsi256_pd(v); } };

    template<> struct ReinterpretCastHelper<unsigned short, short         > { static _M256I cast(const _M256I &v) { return v; } };
    template<> struct ReinterpretCastHelper<unsigned short, unsigned short> { static _M256I cast(const _M256I &v) { return v; } };
    template<> struct ReinterpretCastHelper<short         , unsigned short> { static _M256I cast(const _M256I &v) { return v; } };
    template<> struct ReinterpretCastHelper<short         , short         > { static _M256I cast(const _M256I &v) { return v; } };
} // namespace AVX
} // namespace Vc

#endif // AVX_CASTS_H
