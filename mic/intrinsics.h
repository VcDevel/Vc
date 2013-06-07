/*  This file is part of the Vc library. {{{

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

}}}*/

#ifndef VC_MIC_INTRINSICS_H
#define VC_MIC_INTRINSICS_H

#include <immintrin.h>

#include "const_data.h"
#include "macros.h"

Vc_NAMESPACE_BEGIN(MICIntrinsics)
    using MIC::c_general;

    static Vc_INTRINSIC Vc_CONST __m512  _mm512_setallone_ps()    { return _mm512_castsi512_ps(_mm512_set_1to16_pi(~0)); }
    static Vc_INTRINSIC Vc_CONST __m512d _mm512_setallone_pd()    { return _mm512_castsi512_pd(_mm512_set_1to16_pi(~0)); }
    static Vc_INTRINSIC Vc_CONST __m512i _mm512_setallone_si512() { return _mm512_set_1to16_pi(~0); }

    template<typename T> T _setallone();
    template<> __m512i _setallone<__m512i>() { return _mm512_setallone_si512(); }
    template<> __m512d _setallone<__m512d>() { return _mm512_setallone_pd(); }
    template<> __m512  _setallone<__m512 >() { return _mm512_setallone_ps(); }

    static Vc_INTRINSIC Vc_CONST __m512d _mm512_setabsmask_pd() { return _mm512_set1_pd(reinterpret_cast<const double &>(c_general::absMaskFloat[0])); }
    static Vc_INTRINSIC Vc_CONST __m512  _mm512_setabsmask_ps() { return _mm512_set1_ps(reinterpret_cast<const float &>(c_general::absMaskFloat[1])); }
    static Vc_INTRINSIC Vc_CONST __m512d _mm512_setsignmask_pd(){ return _mm512_set1_pd(reinterpret_cast<const double &>(c_general::signMaskFloat[0])); }
    static Vc_INTRINSIC Vc_CONST __m512  _mm512_setsignmask_ps(){ return _mm512_set1_ps(reinterpret_cast<const float &>(c_general::signMaskFloat[1])); }

    static Vc_INTRINSIC __m512d _mm512_loadu_pd(const void *mt,
            _MM_UPCONV_PD_ENUM upconv = _MM_UPCONV_PD_NONE, int memHint = _MM_HINT_NONE)
    {
        __m512d r;
#pragma warning(disable: 592)
        r = _mm512_extloadunpacklo_pd(r, mt, upconv, memHint);
#pragma warning(default: 592)
        r = _mm512_extloadunpackhi_pd(r, mt + 8 * sizeof(double), upconv, memHint);
        return r;
    }

    static Vc_INTRINSIC __m512  _mm512_loadu_ps(const void *mt,
            _MM_UPCONV_PS_ENUM upconv = _MM_UPCONV_PS_NONE, int memHint = _MM_HINT_NONE)
    {
        __m512 r = _mm512_setzero_ps();
#pragma warning(disable: 592)
        r = _mm512_extloadunpacklo_ps(r, mt, upconv, memHint);
#pragma warning(default: 592)
        r = _mm512_extloadunpackhi_ps(r, mt + 16 * sizeof(float), upconv, memHint);
        return r;
    }

    static Vc_INTRINSIC __m512i _mm512_loadu_epi32(const void *mt,
            _MM_UPCONV_EPI32_ENUM upconv = _MM_UPCONV_EPI32_NONE, int memHint = _MM_HINT_NONE)
    {
        __m512i r;
#pragma warning(disable: 592)
        r = _mm512_extloadunpacklo_epi32(r, mt, upconv, memHint);
#pragma warning(default: 592)
        r = _mm512_extloadunpackhi_epi32(r, mt + 16 * sizeof(int), upconv, memHint);
        return r;
    }

    static Vc_INTRINSIC __m512i _mm512_loadu_epu32(const void *mt,
            _MM_UPCONV_EPI32_ENUM upconv = _MM_UPCONV_EPI32_NONE, int memHint = _MM_HINT_NONE)
    {
        __m512i r;
#pragma warning(disable: 592)
        r = _mm512_extloadunpacklo_epi32(r, mt, upconv, memHint);
#pragma warning(default: 592)
        r = _mm512_extloadunpackhi_epi32(r, mt + 16 * sizeof(int), upconv, memHint);
        return r;
    }

    static Vc_INTRINSIC __m512 _load(const float *m, _MM_UPCONV_PS_ENUM upconv = _MM_UPCONV_PS_NONE,
            _MM_BROADCAST32_ENUM broadcast = _MM_BROADCAST32_NONE, int memHint = _MM_HINT_NONE)
    {
        return _mm512_extload_ps(m, upconv, broadcast, memHint);
    }
    static Vc_INTRINSIC __m512 _load(__m512 v, __mmask16 k, const float *m, _MM_UPCONV_PS_ENUM upconv = _MM_UPCONV_PS_NONE,
            _MM_BROADCAST32_ENUM broadcast = _MM_BROADCAST32_NONE, int memHint = _MM_HINT_NONE)
    {
        return _mm512_mask_extload_ps(v, k, m, upconv, broadcast, memHint);
    }
    static Vc_INTRINSIC __m512d _load(const double *m, _MM_UPCONV_PD_ENUM upconv = _MM_UPCONV_PD_NONE,
            _MM_BROADCAST64_ENUM broadcast = _MM_BROADCAST64_NONE, int memHint = _MM_HINT_NONE)
    {
        return _mm512_extload_pd(m, upconv, broadcast, memHint);
    }
    static Vc_INTRINSIC __m512d _load(__m512d v, __mmask8 k, const double *m, _MM_UPCONV_PD_ENUM upconv = _MM_UPCONV_PD_NONE,
            _MM_BROADCAST64_ENUM broadcast = _MM_BROADCAST64_NONE, int memHint = _MM_HINT_NONE)
    {
        return _mm512_mask_extload_pd(v, k, m, upconv, broadcast, memHint);
    }
    template<typename T>
    static Vc_INTRINSIC __m512i _load(const T *m, _MM_UPCONV_EPI32_ENUM upconv = _MM_UPCONV_EPI32_NONE,
            _MM_BROADCAST32_ENUM broadcast = _MM_BROADCAST32_NONE, int memHint = _MM_HINT_NONE)
    {
        return _mm512_extload_epi32(m, upconv, broadcast, memHint);
    }
    template<typename T>
    static Vc_INTRINSIC __m512i _load(__m512i v, __mmask16 k, const T *m,
            _MM_UPCONV_EPI32_ENUM upconv = _MM_UPCONV_EPI32_NONE,
            _MM_BROADCAST32_ENUM broadcast = _MM_BROADCAST32_NONE, int memHint = _MM_HINT_NONE)
    {
        return _mm512_mask_extload_epi32(v, k, m, upconv, broadcast, memHint);
    }

static Vc_INTRINSIC void store_unaligned(void *m, __m512 v, _MM_DOWNCONV_PS_ENUM downconv, int memHint)
{
    _mm512_extpackstorelo_ps(m, v, downconv, memHint);
    _mm512_extpackstorehi_ps(m + 64, v, downconv, memHint);
}
static Vc_INTRINSIC void store_unaligned(void *m, __m512d v, _MM_DOWNCONV_PD_ENUM downconv, int memHint)
{
    _mm512_extpackstorelo_pd(m, v, downconv, memHint);
    _mm512_extpackstorehi_pd(m + 64, v, downconv, memHint);
}
static Vc_INTRINSIC void store_unaligned(void *m, __m512i v, _MM_DOWNCONV_EPI32_ENUM downconv, int memHint)
{
    _mm512_extpackstorelo_epi32(m, v, downconv, memHint);
    _mm512_extpackstorehi_epi32(m + 64, v, downconv, memHint);
}

static Vc_INTRINSIC void store_unaligned(__mmask16 mask, void *m, __m512 v, _MM_DOWNCONV_PS_ENUM downconv, int memHint)
{
    _mm512_mask_extpackstorelo_ps(m, mask, v, downconv, memHint);
    _mm512_mask_extpackstorehi_ps(m + 64, mask, v, downconv, memHint);
}
static Vc_INTRINSIC void store_unaligned(__mmask8 mask, void *m, __m512d v, _MM_DOWNCONV_PD_ENUM downconv, int memHint)
{
    _mm512_mask_extpackstorelo_pd(m, mask, v, downconv, memHint);
    _mm512_mask_extpackstorehi_pd(m + 64, mask, v, downconv, memHint);
}
static Vc_INTRINSIC void store_unaligned(__mmask16 mask, void *m, __m512i v, _MM_DOWNCONV_EPI32_ENUM downconv, int memHint)
{
    _mm512_mask_extpackstorelo_epi32(m, mask, v, downconv, memHint);
    _mm512_mask_extpackstorehi_epi32(m + 64, mask, v, downconv, memHint);
}

static Vc_INTRINSIC void store_aligned(void *m, __m512 v, _MM_DOWNCONV_PS_ENUM downconv, int memHint)
{
    _mm512_extstore_ps(m, v, downconv, memHint);
}
static Vc_INTRINSIC void store_aligned(void *m, __m512d v, _MM_DOWNCONV_PD_ENUM downconv, int memHint)
{
    _mm512_extstore_pd(m, v, downconv, memHint);
}
static Vc_INTRINSIC void store_aligned(void *m, __m512i v, _MM_DOWNCONV_EPI32_ENUM downconv, int memHint)
{
    _mm512_extstore_epi32(m, v, downconv, memHint);
}

static Vc_INTRINSIC void store_aligned(__mmask16 mask, void *m, __m512 v, _MM_DOWNCONV_PS_ENUM downconv, int memHint)
{
    _mm512_mask_extstore_ps(m, mask, v, downconv, memHint);
}
static Vc_INTRINSIC void store_aligned(__mmask8 mask, void *m, __m512d v, _MM_DOWNCONV_PD_ENUM downconv, int memHint)
{
    _mm512_mask_extstore_pd(m, mask, v, downconv, memHint);
}
static Vc_INTRINSIC void store_aligned(__mmask16 mask, void *m, __m512i v, _MM_DOWNCONV_EPI32_ENUM downconv, int memHint)
{
    _mm512_mask_extstore_epi32(m, mask, v, downconv, memHint);
}

template<typename V, typename DownConv> static Vc_INTRINSIC
    void store(void *m, V v, DownConv downconv, AlignedFlag) { store_aligned(m, v, downconv, _MM_HINT_NONE); }
template<typename V, typename DownConv> static Vc_INTRINSIC
    void store(void *m, V v, DownConv downconv, UnalignedFlag) { store_unaligned(m, v, downconv, _MM_HINT_NONE); }
template<typename V, typename DownConv> static Vc_INTRINSIC
    void store(void *m, V v, DownConv downconv, StreamingAndAlignedFlag) { store_aligned(m, v, downconv, _MM_HINT_NT); }
template<typename V, typename DownConv> static Vc_INTRINSIC
    void store(void *m, V v, DownConv downconv, StreamingAndUnalignedFlag) { store_unaligned(m, v, downconv, _MM_HINT_NT); }

template<typename M, typename V, typename DownConv> static Vc_INTRINSIC
    void store(M mask, void *m, V v, DownConv downconv, AlignedFlag) { store_aligned(mask, m, v, downconv, _MM_HINT_NONE); }
template<typename M, typename V, typename DownConv> static Vc_INTRINSIC
    void store(M mask, void *m, V v, DownConv downconv, UnalignedFlag) { store_unaligned(mask, m, v, downconv, _MM_HINT_NONE); }
template<typename M, typename V, typename DownConv> static Vc_INTRINSIC
    void store(M mask, void *m, V v, DownConv downconv, StreamingAndAlignedFlag) { store_aligned(mask, m, v, downconv, _MM_HINT_NT); }
template<typename M, typename V, typename DownConv> static Vc_INTRINSIC
    void store(M mask, void *m, V v, DownConv downconv, StreamingAndUnalignedFlag) { store_unaligned(mask, m, v, downconv, _MM_HINT_NT); }

#define _mm512_rsqrt_pd _mm512_invsqrt_pd
#define _mm512_mask_rsqrt_pd _mm512_mask_invsqrt_pd
#define _mm512_rsqrt_ps _mm512_invsqrt_ps
#define _mm512_mask_rsqrt_ps _mm512_mask_invsqrt_ps
#define _mm512_reduce_max_pi _mm512_reduce_max_epi32
#define _mm512_reduce_min_pi _mm512_reduce_min_epi32
#define _mm512_reduce_mul_pi _mm512_reduce_mul_epi32
#define _mm512_reduce_add_pi _mm512_reduce_add_epi32

#define Vc_INTEGER_FUN2(fun) \
    static Vc_INTRINSIC __m512  _##fun(__m512  a, __m512  b) { \
        return _mm512_castsi512_ps(_mm512_##fun##_epi32(_mm512_castps_si512(a), _mm512_castps_si512(b))); \
    } \
    static Vc_INTRINSIC __m512d _##fun(__m512d a, __m512d b) { \
        return _mm512_castsi512_pd(_mm512_##fun##_epi64(_mm512_castpd_si512(a), _mm512_castpd_si512(b))); \
    } \
    static Vc_INTRINSIC __m512i _##fun(__m512i a, __m512i b) { \
        return _mm512_##fun##_epi32(a, b); \
    } \
    static Vc_INTRINSIC __m512  _##fun(__m512  r, __mmask16 k, __m512  a, __m512  b) { \
        return _mm512_castsi512_ps(_mm512_mask_##fun##_epi32(_mm512_castps_si512(r), k, _mm512_castps_si512(a), _mm512_castps_si512(b))); \
    } \
    static Vc_INTRINSIC __m512d _##fun(__m512d r, __mmask8  k, __m512d a, __m512d b) { \
        return _mm512_castsi512_pd(_mm512_mask_##fun##_epi64(_mm512_castpd_si512(r), k, _mm512_castpd_si512(a), _mm512_castpd_si512(b))); \
    } \
    static Vc_INTRINSIC __m512i _##fun(__m512i r, __mmask16 k, __m512i a, __m512i b) { \
        return _mm512_mask_##fun##_epi32(r, k, a, b); \
    }
    Vc_INTEGER_FUN2(and)
    Vc_INTEGER_FUN2(or)
    Vc_INTEGER_FUN2(xor)
    Vc_INTEGER_FUN2(andnot)
#undef Vc_INTEGER_FUN2

#define Vc_FUN2(fun) \
    static Vc_INTRINSIC __m512  _##fun(__m512  a, __m512  b) { \
        return _mm512_##fun##_ps(a, b); \
    } \
    static Vc_INTRINSIC __m512d _##fun(__m512d a, __m512d b) { \
        return _mm512_##fun##_pd(a, b); \
    } \
    static Vc_INTRINSIC __m512i _##fun(__m512i a, __m512i b) { \
        return _mm512_##fun##_epi32(a, b); \
    } \
    static Vc_INTRINSIC __m512  _##fun(__m512  r, __mmask16 k, __m512  a, __m512  b) { \
        return _mm512_mask_##fun##_ps(r, k, a, b); \
    } \
    static Vc_INTRINSIC __m512d _##fun(__m512d r, __mmask8  k, __m512d a, __m512d b) { \
        return _mm512_mask_##fun##_pd(r, k, a, b); \
    } \
    static Vc_INTRINSIC __m512i _##fun(__m512i r, __mmask16 k, __m512i a, __m512i b) { \
        return _mm512_mask_##fun##_epi32(r, k, a, b); \
    }
    Vc_FUN2(min)
    Vc_FUN2(max)
    Vc_FUN2(add)
    Vc_FUN2(sub)
    Vc_FUN2(div)
#undef Vc_FUN2

    static Vc_INTRINSIC __m512  _mul(__m512  a, __m512  b) { return _mm512_mul_ps(a, b); }
    static Vc_INTRINSIC __m512d _mul(__m512d a, __m512d b) { return _mm512_mul_pd(a, b); }
    static Vc_INTRINSIC __m512i _mul(__m512i a, __m512i b) { return _mm512_mullo_epi32(a, b); }
    static Vc_INTRINSIC __m512  _mul(__m512  r, __mmask16 k, __m512  a, __m512  b) { return _mm512_mask_mul_ps(r, k, a, b); }
    static Vc_INTRINSIC __m512d _mul(__m512d r, __mmask8  k, __m512d a, __m512d b) { return _mm512_mask_mul_pd(r, k, a, b); }
    static Vc_INTRINSIC __m512i _mul(__m512i r, __mmask16 k, __m512i a, __m512i b) { return _mm512_mask_mullo_epi32(r, k, a, b); }

    static Vc_INTRINSIC __m512  _mask_mov(__m512  r, __mmask16 k, __m512  a) { return _mm512_mask_mov_ps(r, k, a); }
    static Vc_INTRINSIC __m512d _mask_mov(__m512d r, __mmask8  k, __m512d a) { return _mm512_mask_mov_pd(r, k, a); }
    static Vc_INTRINSIC __m512i _mask_mov(__m512i r, __mmask16 k, __m512i a) { return _mm512_mask_mov_epi32(r, k, a); }

    static Vc_INTRINSIC __m512  _set1(             float a) { return _mm512_set1_ps(a); }
    static Vc_INTRINSIC __m512d _set1(            double a) { return _mm512_set1_pd(a); }
    static Vc_INTRINSIC __m512i _set1(         long long a) { return _mm512_set1_epi64(a); }
    static Vc_INTRINSIC __m512i _set1(unsigned long long a) { return _mm512_set1_epi64(a); }
    static Vc_INTRINSIC __m512i _set1(               int a) { return _mm512_set1_epi32(a); }
    static Vc_INTRINSIC __m512i _set1(      unsigned int a) { return _mm512_set1_epi32(a); }
    static Vc_INTRINSIC __m512i _set1(             short a) { return _mm512_set1_epi32(a); }
    static Vc_INTRINSIC __m512i _set1(    unsigned short a) { return _mm512_set1_epi32(a); }
    static Vc_INTRINSIC __m512i _set1(       signed char a) { return _mm512_set1_epi32(a); }
    static Vc_INTRINSIC __m512i _set1(     unsigned char a) { return _mm512_set1_epi32(a); }

Vc_NAMESPACE_END

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)
    using namespace MICIntrinsics;
Vc_NAMESPACE_END

#include "undomacros.h"

#endif // VC_MIC_INTRINSICS_H
