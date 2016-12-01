/*  This file is part of the Vc library. {{{
Copyright © 2009-2015 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_MIC_INTRINSICS_H_
#define VC_MIC_INTRINSICS_H_

#include <immintrin.h>

#include "const_data.h"
#include "../common/loadstoreflags.h"
#include "macros.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace MicIntrinsics
{
    using MIC::c_general;

    static Vc_INTRINSIC Vc_CONST __m512  _mm512_setallone_ps()    { return _mm512_castsi512_ps(_mm512_set_1to16_pi(~0)); }
    static Vc_INTRINSIC Vc_CONST __m512d _mm512_setallone_pd()    { return _mm512_castsi512_pd(_mm512_set_1to16_pi(~0)); }
    static Vc_INTRINSIC Vc_CONST __m512i _mm512_setallone_si512() { return _mm512_set_1to16_pi(~0); }

    template<typename T> T allone();
    template<> __m512i allone<__m512i>() { return _mm512_setallone_si512(); }
    template<> __m512d allone<__m512d>() { return _mm512_setallone_pd(); }
    template<> __m512  allone<__m512 >() { return _mm512_setallone_ps(); }

    static Vc_INTRINSIC Vc_CONST __m512d _mm512_setabsmask_pd() { return _mm512_set1_pd(reinterpret_cast<const double &>(c_general::absMaskFloat[0])); }
    static Vc_INTRINSIC Vc_CONST __m512  _mm512_setabsmask_ps() { return _mm512_set1_ps(reinterpret_cast<const float &>(c_general::absMaskFloat[1])); }
    static Vc_INTRINSIC Vc_CONST __m512d _mm512_setsignmask_pd(){ return _mm512_set1_pd(reinterpret_cast<const double &>(c_general::signMaskFloat[0])); }
    static Vc_INTRINSIC Vc_CONST __m512  _mm512_setsignmask_ps(){ return _mm512_set1_ps(reinterpret_cast<const float &>(c_general::signMaskFloat[1])); }

    static Vc_INTRINSIC __m512d mm512_loadu_pd(const void *mt,
            _MM_UPCONV_PD_ENUM upconv = _MM_UPCONV_PD_NONE, int memHint = _MM_HINT_NONE)
    {
        __m512d r;
#pragma warning(disable: 592)
        r = _mm512_extloadunpacklo_pd(r, mt, upconv, memHint);
#pragma warning(default: 592)
        r = _mm512_extloadunpackhi_pd(r, static_cast<const char *>(mt) + 8 * sizeof(double), upconv, memHint);
        return r;
    }

    static Vc_INTRINSIC __m512  mm512_loadu_ps(const void *mt,
            _MM_UPCONV_PS_ENUM upconv = _MM_UPCONV_PS_NONE, int memHint = _MM_HINT_NONE)
    {
        __m512 r = _mm512_setzero_ps();
#pragma warning(disable: 592)
        r = _mm512_extloadunpacklo_ps(r, mt, upconv, memHint);
#pragma warning(default: 592)
        r = _mm512_extloadunpackhi_ps(r, static_cast<const char *>(mt) + 16 * sizeof(float), upconv, memHint);
        return r;
    }

    static Vc_INTRINSIC __m512i mm512_loadu_epi32(const void *mt,
            _MM_UPCONV_EPI32_ENUM upconv = _MM_UPCONV_EPI32_NONE, int memHint = _MM_HINT_NONE)
    {
        __m512i r;
#pragma warning(disable: 592)
        r = _mm512_extloadunpacklo_epi32(r, mt, upconv, memHint);
#pragma warning(default: 592)
        r = _mm512_extloadunpackhi_epi32(r, static_cast<const char *>(mt) + 16 * sizeof(int), upconv, memHint);
        return r;
    }

    static Vc_INTRINSIC __m512i _mm512_loadu_epu32(const void *mt,
            _MM_UPCONV_EPI32_ENUM upconv = _MM_UPCONV_EPI32_NONE, int memHint = _MM_HINT_NONE)
    {
        __m512i r;
#pragma warning(disable: 592)
        r = _mm512_extloadunpacklo_epi32(r, mt, upconv, memHint);
#pragma warning(default: 592)
        r = _mm512_extloadunpackhi_epi32(r, static_cast<const char *>(mt) + 16 * sizeof(int), upconv, memHint);
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
    _mm512_extpackstorehi_ps(static_cast<char *>(m) + 64, v, downconv, memHint);
}
static Vc_INTRINSIC void store_unaligned(void *m, __m512d v, _MM_DOWNCONV_PD_ENUM downconv, int memHint)
{
    _mm512_extpackstorelo_pd(m, v, downconv, memHint);
    _mm512_extpackstorehi_pd(static_cast<char *>(m) + 64, v, downconv, memHint);
}
static Vc_INTRINSIC void store_unaligned(void *m, __m512i v, _MM_DOWNCONV_EPI32_ENUM downconv, int memHint)
{
    _mm512_extpackstorelo_epi32(m, v, downconv, memHint);
    _mm512_extpackstorehi_epi32(static_cast<char *>(m) + 64, v, downconv, memHint);
}

/*
void store_unaligned(__mmask16 mask, void *m, __m512 v, _MM_DOWNCONV_PS_ENUM downconv, int memHint)
and friends are not that easy. MIC only provides packstore for unaligned stores. But packstore packs
the masked values consecutively, which is not what we want.
The maskstore is thus better implemented in the Vector class - where more type information is still
available.
*/

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

template<typename Flags, typename V, typename DownConv> static Vc_INTRINSIC void store(void *m, V v, DownConv downconv, typename Flags::EnableIfAligned               = nullptr) { store_aligned(m, v, downconv, _MM_HINT_NONE); }
template<typename Flags, typename V, typename DownConv> static Vc_INTRINSIC void store(void *m, V v, DownConv downconv, typename Flags::EnableIfUnalignedNotStreaming = nullptr) { store_unaligned(m, v, downconv, _MM_HINT_NONE); }
template<typename Flags, typename V, typename DownConv> static Vc_INTRINSIC void store(void *m, V v, DownConv downconv, typename Flags::EnableIfStreaming             = nullptr) { store_aligned(m, v, downconv, _MM_HINT_NT); }
template<typename Flags, typename V, typename DownConv> static Vc_INTRINSIC void store(void *m, V v, DownConv downconv, typename Flags::EnableIfUnalignedAndStreaming = nullptr) { store_unaligned(m, v, downconv, _MM_HINT_NT); }

template<typename Flags, typename M, typename V, typename DownConv> static Vc_INTRINSIC void store(M mask, void *m, V v, DownConv downconv, typename Flags::EnableIfAligned               = nullptr) { store_aligned(mask, m, v, downconv, _MM_HINT_NONE); }
template<typename Flags, typename M, typename V, typename DownConv> static Vc_INTRINSIC void store(M mask, void *m, V v, DownConv downconv, typename Flags::EnableIfUnalignedNotStreaming = nullptr) { store_unaligned(mask, m, v, downconv, _MM_HINT_NONE); }
template<typename Flags, typename M, typename V, typename DownConv> static Vc_INTRINSIC void store(M mask, void *m, V v, DownConv downconv, typename Flags::EnableIfStreaming             = nullptr) { store_aligned(mask, m, v, downconv, _MM_HINT_NT); }
template<typename Flags, typename M, typename V, typename DownConv> static Vc_INTRINSIC void store(M mask, void *m, V v, DownConv downconv, typename Flags::EnableIfUnalignedAndStreaming = nullptr) { store_unaligned(mask, m, v, downconv, _MM_HINT_NT); }

//__m512  _mm512_i32extgather_ps(__m512i, void const*, _MM_UPCONV_PS_ENUM, int, int /* mem hint */);
//__m512  _mm512_mask_i32extgather_ps(__m512, __mmask16, __m512i, void const*, _MM_UPCONV_PS_ENUM, int, int /* mem hint */);
template<typename MemT> static Vc_INTRINSIC __m512  gather(__m512i i, MemT const *mem, _MM_UPCONV_PS_ENUM    conv, int scale = sizeof(MemT)) { return _mm512_i32extgather_ps   (i, mem, conv, scale, _MM_HINT_NONE); }
template<typename MemT> static Vc_INTRINSIC __m512d gather(__m512i i, MemT const *mem, _MM_UPCONV_PD_ENUM    conv, int scale = sizeof(MemT)) { return _mm512_i32loextgather_pd (i, mem, conv, scale, _MM_HINT_NONE); }
template<typename MemT> static Vc_INTRINSIC __m512i gather(__m512i i, MemT const *mem, _MM_UPCONV_EPI32_ENUM conv, int scale = sizeof(MemT)) { return _mm512_i32extgather_epi32(i, mem, conv, scale, _MM_HINT_NONE); }

template<typename MemT> static Vc_INTRINSIC __m512  gather(__m512  old, __mmask16 mask, __m512i i, MemT const *mem, _MM_UPCONV_PS_ENUM    conv, int scale = sizeof(MemT)) { return _mm512_mask_i32extgather_ps   (old, mask, i, mem, conv, scale, _MM_HINT_NONE); }
template<typename MemT> static Vc_INTRINSIC __m512d gather(__m512d old, __mmask8  mask, __m512i i, MemT const *mem, _MM_UPCONV_PD_ENUM    conv, int scale = sizeof(MemT)) { return _mm512_mask_i32loextgather_pd (old, mask, i, mem, conv, scale, _MM_HINT_NONE); }
template<typename MemT> static Vc_INTRINSIC __m512i gather(__m512i old, __mmask16 mask, __m512i i, MemT const *mem, _MM_UPCONV_EPI32_ENUM conv, int scale = sizeof(MemT)) { return _mm512_mask_i32extgather_epi32(old, mask, i, mem, conv, scale, _MM_HINT_NONE); }

template<typename DownConv> static Vc_INTRINSIC
    void scatter(void *m, __m512i i, __m512  v, DownConv downconv, int scale)
{
    _mm512_i32extscatter_ps(m, i, v, downconv.down(), scale, _MM_HINT_NONE);
}
template<typename DownConv> static Vc_INTRINSIC
    void scatter(void *m, __m512i i, __m512d v, DownConv downconv, int scale)
{
    _mm512_i32loextscatter_pd(m, i, v, downconv.down(), scale, _MM_HINT_NONE);
}
template<typename DownConv> static Vc_INTRINSIC
    void scatter(void *m, __m512i i, __m512i v, DownConv downconv, int scale)
{
    _mm512_i32extscatter_epi32(m, i, v, downconv.down(), scale, _MM_HINT_NONE);
}

template<typename M, typename DownConv, typename MemT> static Vc_INTRINSIC
    void scatter(M mask, MemT *m, __m512i i, __m512  v, DownConv downconv, int scale = sizeof(MemT))
{
    _mm512_mask_i32extscatter_ps(m, mask, i, v, downconv.down(), scale, _MM_HINT_NONE);
}
template<typename M, typename DownConv, typename MemT> static Vc_INTRINSIC
    void scatter(M mask, MemT *m, __m512i i, __m512d v, DownConv downconv, int scale = sizeof(MemT))
{
    _mm512_mask_i32loextscatter_pd(m, mask, i, v, downconv.down(), scale, _MM_HINT_NONE);
}
template<typename M, typename DownConv, typename MemT> static Vc_INTRINSIC
    void scatter(M mask, MemT *m, __m512i i, __m512i v, DownConv downconv, int scale = sizeof(MemT))
{
    _mm512_mask_i32extscatter_epi32(m, mask, i, v, downconv.down(), scale, _MM_HINT_NONE);
}

static Vc_INTRINSIC __m512  swizzle(__m512  v, _MM_SWIZZLE_ENUM swiz) { return _mm512_swizzle_ps(v, swiz); }
static Vc_INTRINSIC __m512d swizzle(__m512d v, _MM_SWIZZLE_ENUM swiz) { return _mm512_swizzle_pd(v, swiz); }
static Vc_INTRINSIC __m512i swizzle(__m512i v, _MM_SWIZZLE_ENUM swiz) { return _mm512_swizzle_epi32(v, swiz); }

static Vc_INTRINSIC __m512  shuffle(__m512  v, _MM_PERM_ENUM perm) { return _mm512_castsi512_ps(_mm512_shuffle_epi32(_mm512_castps_si512(v), perm)); }
static Vc_INTRINSIC __m512i shuffle(__m512i v, _MM_PERM_ENUM perm) { return _mm512_shuffle_epi32(v, perm); }

static Vc_INTRINSIC __m512 permute128(__m512 v, _MM_PERM_ENUM perm)
{
    return _mm512_permute4f128_ps(v, perm);
}
static Vc_INTRINSIC __m512i permute128(__m512i v, _MM_PERM_ENUM perm)
{
    return _mm512_permute4f128_epi32(v, perm);
}
static Vc_INTRINSIC __m512d permute128(__m512d v, _MM_PERM_ENUM perm)
{
    return _mm512_castps_pd(_mm512_permute4f128_ps(_mm512_castpd_ps(v), perm));
}

#define _mm512_rsqrt_pd _mm512_invsqrt_pd
#define _mm512_mask_rsqrt_pd _mm512_mask_invsqrt_pd
#define _mm512_rsqrt_ps _mm512_invsqrt_ps
#define _mm512_mask_rsqrt_ps _mm512_mask_invsqrt_ps
#define _mm512_reduce_max_pi _mm512_reduce_max_epi32
#define _mm512_reduce_min_pi _mm512_reduce_min_epi32
#define _mm512_reduce_mul_pi _mm512_reduce_mul_epi32
#define _mm512_reduce_add_pi _mm512_reduce_add_epi32

#define Vc_INTEGER_FUN2(fun) \
    static Vc_INTRINSIC __m512  fun(__m512  a, __m512  b) { \
        return _mm512_castsi512_ps(_mm512##fun##_epi32(_mm512_castps_si512(a), _mm512_castps_si512(b))); \
    } \
    static Vc_INTRINSIC __m512d fun(__m512d a, __m512d b) { \
        return _mm512_castsi512_pd(_mm512##fun##_epi64(_mm512_castpd_si512(a), _mm512_castpd_si512(b))); \
    } \
    static Vc_INTRINSIC __m512i fun(__m512i a, __m512i b) { \
        return _mm512##fun##_epi32(a, b); \
    } \
    static Vc_INTRINSIC __m512  fun(__m512  r, __mmask16 k, __m512  a, __m512  b) { \
        return _mm512_castsi512_ps(_mm512_mask##fun##_epi32(_mm512_castps_si512(r), k, _mm512_castps_si512(a), _mm512_castps_si512(b))); \
    } \
    static Vc_INTRINSIC __m512d fun(__m512d r, __mmask8  k, __m512d a, __m512d b) { \
        return _mm512_castsi512_pd(_mm512_mask##fun##_epi64(_mm512_castpd_si512(r), k, _mm512_castpd_si512(a), _mm512_castpd_si512(b))); \
    } \
    static Vc_INTRINSIC __m512i fun(__m512i r, __mmask16 k, __m512i a, __m512i b) { \
        return _mm512_mask##fun##_epi32(r, k, a, b); \
    }
    Vc_INTEGER_FUN2(_and)
    Vc_INTEGER_FUN2(_or)
    Vc_INTEGER_FUN2(_xor)
    Vc_INTEGER_FUN2(_andnot)
#undef Vc_INTEGER_FUN2

    Vc_INTRINSIC __m512i _mm512_add_epu32(__m512i a, __m512i b) { return _mm512_add_epi32(a, b); }
    Vc_INTRINSIC __m512i _mm512_sub_epu32(__m512i a, __m512i b) { return _mm512_sub_epi32(a, b); }
    Vc_INTRINSIC __m512i _mm512_mask_add_epu32(__m512i r, __mmask16 k, __m512i a, __m512i b) { return _mm512_mask_add_epi32(r, k, a, b); }
    Vc_INTRINSIC __m512i _mm512_mask_sub_epu32(__m512i r, __mmask16 k, __m512i a, __m512i b) { return _mm512_mask_sub_epi32(r, k, a, b); }
#define Vc_FUN2(fun) \
    template<typename> Vc_INTRINSIC __m512  fun(__m512  a, __m512  b) { \
        return _mm512##fun##_ps(a, b); \
    } \
    template<typename> Vc_INTRINSIC __m512d fun(__m512d a, __m512d b) { \
        return _mm512##fun##_pd(a, b); \
    } \
    template<typename> Vc_INTRINSIC __m512i fun(__m512i a, __m512i b) { \
        return _mm512##fun##_epi32(a, b); \
    } \
    template<> Vc_INTRINSIC __m512i fun<unsigned int>(__m512i a, __m512i b) { \
        return _mm512##fun##_epu32(a, b); \
    } \
    template<typename> Vc_INTRINSIC __m512  fun(__m512  r, __mmask16 k, __m512  a, __m512  b) { \
        return _mm512_mask##fun##_ps(r, k, a, b); \
    } \
    template<typename> Vc_INTRINSIC __m512d fun(__m512d r, __mmask8  k, __m512d a, __m512d b) { \
        return _mm512_mask##fun##_pd(r, k, a, b); \
    } \
    template<typename> Vc_INTRINSIC __m512i fun(__m512i r, __mmask16 k, __m512i a, __m512i b) { \
        return _mm512_mask##fun##_epi32(r, k, a, b); \
    } \
    template<> Vc_INTRINSIC __m512i fun<unsigned int>(__m512i r, __mmask16 k, __m512i a, __m512i b) { \
        return _mm512_mask##fun##_epu32(r, k, a, b); \
    }
    Vc_FUN2(_min)
    Vc_FUN2(_max)
    Vc_FUN2(_add)
    Vc_FUN2(_sub)
    Vc_FUN2(_div)
#undef Vc_FUN2

Vc_INTRINSIC __m512  _set1(             float a) { return _mm512_set1_ps(a); }
Vc_INTRINSIC __m512d _set1(            double a) { return _mm512_set1_pd(a); }
Vc_INTRINSIC __m512i _set1(         long long a) { return _mm512_set1_epi64(a); }
Vc_INTRINSIC __m512i _set1(unsigned long long a) { return _mm512_set1_epi64(a); }
Vc_INTRINSIC __m512i _set1(               int a) { return _mm512_set1_epi32(a); }
Vc_INTRINSIC __m512i _set1(      unsigned int a) { return _mm512_set1_epi32(a); }
Vc_INTRINSIC __m512i _set1(             short a) { return _mm512_set1_epi32(a); }
Vc_INTRINSIC __m512i _set1(    unsigned short a) { return _mm512_set1_epi32(a); }
Vc_INTRINSIC __m512i _set1(       signed char a) { return _mm512_set1_epi32(a); }
Vc_INTRINSIC __m512i _set1(     unsigned char a) { return _mm512_set1_epi32(a); }

template <typename T>
Vc_INTRINSIC enable_if<std::is_signed<T>::value, __m512i> mod_(__m512i a, __m512i b)
{ return _mm512_rem_epi32(a, b); }
template <typename T>
Vc_INTRINSIC enable_if<std::is_same<T, uint>::value, __m512i> mod_(__m512i a, __m512i b)
{ return _mm512_rem_epu32(a, b); }
template <typename T>
Vc_INTRINSIC enable_if<std::is_same<T, ushort>::value, __m512i> mod_(__m512i a, __m512i b)
{
    return _mm512_rem_epu32(_and(a, _set1(0xffff)), _and(b, _set1(0xffff)));
}

    template<typename> Vc_INTRINSIC __m512  _mul(__m512  a, __m512  b) { return _mm512_mul_ps(a, b); }
    template<typename> Vc_INTRINSIC __m512d _mul(__m512d a, __m512d b) { return _mm512_mul_pd(a, b); }
    template<typename> Vc_INTRINSIC __m512i _mul(__m512i a, __m512i b) { return _mm512_mullo_epi32(a, b); }
    template<typename> Vc_INTRINSIC __m512  _mul(__m512  r, __mmask16 k, __m512  a, __m512  b) { return _mm512_mask_mul_ps(r, k, a, b); }
    template<typename> Vc_INTRINSIC __m512d _mul(__m512d r, __mmask8  k, __m512d a, __m512d b) { return _mm512_mask_mul_pd(r, k, a, b); }
    template<typename> Vc_INTRINSIC __m512i _mul(__m512i r, __mmask16 k, __m512i a, __m512i b) { return _mm512_mask_mullo_epi32(r, k, a, b); }

    static Vc_INTRINSIC __m512  mask_mov(__m512  r, __mmask16 k, __m512  a) { return _mm512_mask_mov_ps(r, k, a); }
    static Vc_INTRINSIC __m512d mask_mov(__m512d r, __mmask8  k, __m512d a) { return _mm512_mask_mov_pd(r, k, a); }
    static Vc_INTRINSIC __m512i mask_mov(__m512i r, __mmask16 k, __m512i a) { return _mm512_mask_mov_epi32(r, k, a); }

}
Vc_VERSIONED_NAMESPACE_END

Vc_VERSIONED_NAMESPACE_BEGIN
namespace MIC
{
using namespace MicIntrinsics;

template<typename T> struct DetermineVectorEntryType { typedef T Type; };
// MIC does not support epi8/epu8 operations, thus we change the EntryType to int/uint
template<> struct DetermineVectorEntryType<char> { typedef int Type; };
template<> struct DetermineVectorEntryType<signed char> { typedef int Type; };
template<> struct DetermineVectorEntryType<unsigned char> { typedef unsigned int Type; };
// MIC does not support epi16/epu16 operations, thus we change the EntryType to int/uint
template<> struct DetermineVectorEntryType<short> { typedef int Type; };
template<> struct DetermineVectorEntryType<unsigned short> { typedef unsigned int Type; };
// MIC does not support epi64/epu64 operations, thus we change the EntryType to int/uint
template<> struct DetermineVectorEntryType<long> { typedef int Type; };
template<> struct DetermineVectorEntryType<unsigned long> { typedef unsigned int Type; };
// MIC does not support epi64/epu64 operations, thus we change the EntryType to int/uint
template<> struct DetermineVectorEntryType<long long> { typedef int Type; };
template<> struct DetermineVectorEntryType<unsigned long long> { typedef unsigned int Type; };

template <typename T> struct SwizzledVector;
template <typename T> class VectorMultiplication;

alignas(16) extern const char _IndexesFromZero[16];

template <typename T> struct VectorTypeHelper;
template <> struct VectorTypeHelper<char> { typedef __m512i Type; };
template <> struct VectorTypeHelper<signed char> { typedef __m512i Type; };
template <> struct VectorTypeHelper<unsigned char> { typedef __m512i Type; };
template <> struct VectorTypeHelper<short> { typedef __m512i Type; };
template <> struct VectorTypeHelper<unsigned short> { typedef __m512i Type; };
template <> struct VectorTypeHelper<int> { typedef __m512i Type; };
template <> struct VectorTypeHelper<unsigned int> { typedef __m512i Type; };
template <> struct VectorTypeHelper<long> { typedef __m512i Type; };
template <> struct VectorTypeHelper<unsigned long> { typedef __m512i Type; };
template <> struct VectorTypeHelper<long long> { typedef __m512i Type; };
template <> struct VectorTypeHelper<unsigned long long> { typedef __m512i Type; };
template <> struct VectorTypeHelper<float> { typedef __m512 Type; };
template <> struct VectorTypeHelper<double> { typedef __m512d Type; };
template <> struct VectorTypeHelper<__m512i> { typedef __m512i Type; };
template <> struct VectorTypeHelper<__m512> { typedef __m512 Type; };
template <> struct VectorTypeHelper<__m512d> { typedef __m512d Type; };

template <typename T> struct MaskTypeHelper { typedef __mmask16 Type; };
template <> struct MaskTypeHelper<__m512d> { typedef __mmask8 Type; };
template <> struct MaskTypeHelper<double> { typedef __mmask8 Type; };

template <typename T> struct ReturnTypeHelper { typedef char Type; };
template <> struct ReturnTypeHelper<unsigned int> { typedef unsigned char Type; };
template <> struct ReturnTypeHelper<int> { typedef signed char Type; };
template <typename T> const typename ReturnTypeHelper<T>::Type *IndexesFromZeroHelper()
{
    return reinterpret_cast<const typename ReturnTypeHelper<T>::Type *>(&_IndexesFromZero[0]);
}

template <size_t Size> struct IndexScaleHelper;
template <> struct IndexScaleHelper<8> { static inline _MM_INDEX_SCALE_ENUM value() { return _MM_SCALE_4; } };
template <> struct IndexScaleHelper<4> { static inline _MM_INDEX_SCALE_ENUM value() { return _MM_SCALE_4; } };
template <> struct IndexScaleHelper<2> { static inline _MM_INDEX_SCALE_ENUM value() { return _MM_SCALE_2; } };
template <> struct IndexScaleHelper<1> { static inline _MM_INDEX_SCALE_ENUM value() { return _MM_SCALE_1; } };
template <typename T> struct IndexScale
{
    static inline _MM_INDEX_SCALE_ENUM value() { return IndexScaleHelper<sizeof(T)>::value(); }
};

template <typename EntryType, typename MemType> struct UpDownConversion;
template <> struct UpDownConversion<double, double>
{
    constexpr _MM_DOWNCONV_PD_ENUM down() const { return _MM_DOWNCONV_PD_NONE; }
    constexpr _MM_UPCONV_PD_ENUM up() const { return _MM_UPCONV_PD_NONE; }
    constexpr operator _MM_DOWNCONV_PD_ENUM() const { return _MM_DOWNCONV_PD_NONE; }
    constexpr operator _MM_UPCONV_PD_ENUM() const { return _MM_UPCONV_PD_NONE; }
};
template <> struct UpDownConversion<float, float>
{
    constexpr _MM_DOWNCONV_PS_ENUM down() const { return _MM_DOWNCONV_PS_NONE; }
    constexpr _MM_UPCONV_PS_ENUM up() const { return _MM_UPCONV_PS_NONE; }
    constexpr operator _MM_DOWNCONV_PS_ENUM() const { return _MM_DOWNCONV_PS_NONE; }
    constexpr operator _MM_UPCONV_PS_ENUM() const { return _MM_UPCONV_PS_NONE; }
};
/*template<> struct UpDownConversion<float, half_float> {
    constexpr _MM_DOWNCONV_PS_ENUM down() const { return _MM_DOWNCONV_PS_FLOAT16; }
    constexpr _MM_UPCONV_PS_ENUM up() const { return _MM_UPCONV_PS_FLOAT16; }
    constexpr operator _MM_DOWNCONV_PS_ENUM() const { return _MM_DOWNCONV_PS_FLOAT16; }
    constexpr operator _MM_UPCONV_PS_ENUM() const { return _MM_UPCONV_PS_FLOAT16; }
};*/
template <> struct UpDownConversion<float, unsigned char>
{
    constexpr _MM_DOWNCONV_PS_ENUM down() const { return _MM_DOWNCONV_PS_UINT8; }
    constexpr _MM_UPCONV_PS_ENUM up() const { return _MM_UPCONV_PS_UINT8; }
    constexpr operator _MM_DOWNCONV_PS_ENUM() const { return _MM_DOWNCONV_PS_UINT8; }
    constexpr operator _MM_UPCONV_PS_ENUM() const { return _MM_UPCONV_PS_UINT8; }
};
template <> struct UpDownConversion<float, signed char>
{
    constexpr _MM_DOWNCONV_PS_ENUM down() const { return _MM_DOWNCONV_PS_SINT8; }
    constexpr _MM_UPCONV_PS_ENUM up() const { return _MM_UPCONV_PS_SINT8; }
    constexpr operator _MM_DOWNCONV_PS_ENUM() const { return _MM_DOWNCONV_PS_SINT8; }
    constexpr operator _MM_UPCONV_PS_ENUM() const { return _MM_UPCONV_PS_SINT8; }
};
template <> struct UpDownConversion<float, unsigned short>
{
    constexpr _MM_DOWNCONV_PS_ENUM down() const { return _MM_DOWNCONV_PS_UINT16; }
    constexpr _MM_UPCONV_PS_ENUM up() const { return _MM_UPCONV_PS_UINT16; }
    constexpr operator _MM_DOWNCONV_PS_ENUM() const { return _MM_DOWNCONV_PS_UINT16; }
    constexpr operator _MM_UPCONV_PS_ENUM() const { return _MM_UPCONV_PS_UINT16; }
};
template <> struct UpDownConversion<float, signed short>
{
    constexpr _MM_DOWNCONV_PS_ENUM down() const { return _MM_DOWNCONV_PS_SINT16; }
    constexpr _MM_UPCONV_PS_ENUM up() const { return _MM_UPCONV_PS_SINT16; }
    constexpr operator _MM_DOWNCONV_PS_ENUM() const { return _MM_DOWNCONV_PS_SINT16; }
    constexpr operator _MM_UPCONV_PS_ENUM() const { return _MM_UPCONV_PS_SINT16; }
};
template <> struct UpDownConversion<unsigned int, char>
{
    constexpr _MM_DOWNCONV_EPI32_ENUM down() const { return _MM_DOWNCONV_EPI32_SINT8; }
    constexpr _MM_UPCONV_EPI32_ENUM up() const { return _MM_UPCONV_EPI32_SINT8; }
    constexpr operator _MM_DOWNCONV_EPI32_ENUM() const { return _MM_DOWNCONV_EPI32_SINT8; }
    constexpr operator _MM_UPCONV_EPI32_ENUM() const { return _MM_UPCONV_EPI32_SINT8; }
};
template <> struct UpDownConversion<unsigned int, signed char>
{
    constexpr _MM_DOWNCONV_EPI32_ENUM down() const { return _MM_DOWNCONV_EPI32_SINT8; }
    constexpr _MM_UPCONV_EPI32_ENUM up() const { return _MM_UPCONV_EPI32_SINT8; }
    constexpr operator _MM_DOWNCONV_EPI32_ENUM() const { return _MM_DOWNCONV_EPI32_SINT8; }
    constexpr operator _MM_UPCONV_EPI32_ENUM() const { return _MM_UPCONV_EPI32_SINT8; }
};
template <> struct UpDownConversion<unsigned int, unsigned char>
{
    constexpr _MM_DOWNCONV_EPI32_ENUM down() const { return _MM_DOWNCONV_EPI32_UINT8; }
    constexpr _MM_UPCONV_EPI32_ENUM up() const { return _MM_UPCONV_EPI32_UINT8; }
    constexpr operator _MM_DOWNCONV_EPI32_ENUM() const { return _MM_DOWNCONV_EPI32_UINT8; }
    constexpr operator _MM_UPCONV_EPI32_ENUM() const { return _MM_UPCONV_EPI32_UINT8; }
};
template <> struct UpDownConversion<unsigned int, short>
{
    constexpr _MM_DOWNCONV_EPI32_ENUM down() const { return _MM_DOWNCONV_EPI32_SINT16; }
    constexpr _MM_UPCONV_EPI32_ENUM up() const { return _MM_UPCONV_EPI32_SINT16; }
    constexpr operator _MM_DOWNCONV_EPI32_ENUM() const { return _MM_DOWNCONV_EPI32_SINT16; }
    constexpr operator _MM_UPCONV_EPI32_ENUM() const { return _MM_UPCONV_EPI32_SINT16; }
};
template <> struct UpDownConversion<unsigned int, unsigned short>
{
    constexpr _MM_DOWNCONV_EPI32_ENUM down() const { return _MM_DOWNCONV_EPI32_UINT16; }
    constexpr _MM_UPCONV_EPI32_ENUM up() const { return _MM_UPCONV_EPI32_UINT16; }
    constexpr operator _MM_DOWNCONV_EPI32_ENUM() const { return _MM_DOWNCONV_EPI32_UINT16; }
    constexpr operator _MM_UPCONV_EPI32_ENUM() const { return _MM_UPCONV_EPI32_UINT16; }
};
template <> struct UpDownConversion<unsigned int, int>
{
    constexpr _MM_DOWNCONV_EPI32_ENUM down() const { return _MM_DOWNCONV_EPI32_NONE; }
    constexpr _MM_UPCONV_EPI32_ENUM up() const { return _MM_UPCONV_EPI32_NONE; }
    constexpr operator _MM_DOWNCONV_EPI32_ENUM() const { return _MM_DOWNCONV_EPI32_NONE; }
    constexpr operator _MM_UPCONV_EPI32_ENUM() const { return _MM_UPCONV_EPI32_NONE; }
};
template <> struct UpDownConversion<unsigned int, unsigned int>
{
    constexpr _MM_DOWNCONV_EPI32_ENUM down() const { return _MM_DOWNCONV_EPI32_NONE; }
    constexpr _MM_UPCONV_EPI32_ENUM up() const { return _MM_UPCONV_EPI32_NONE; }
    constexpr operator _MM_DOWNCONV_EPI32_ENUM() const { return _MM_DOWNCONV_EPI32_NONE; }
    constexpr operator _MM_UPCONV_EPI32_ENUM() const { return _MM_UPCONV_EPI32_NONE; }
};
template <> struct UpDownConversion<int, char>
{
    constexpr _MM_DOWNCONV_EPI32_ENUM down() const { return _MM_DOWNCONV_EPI32_SINT8; }
    constexpr _MM_UPCONV_EPI32_ENUM up() const { return _MM_UPCONV_EPI32_SINT8; }
    constexpr operator _MM_DOWNCONV_EPI32_ENUM() const { return _MM_DOWNCONV_EPI32_SINT8; }
    constexpr operator _MM_UPCONV_EPI32_ENUM() const { return _MM_UPCONV_EPI32_SINT8; }
};
template <> struct UpDownConversion<int, signed char>
{
    constexpr _MM_DOWNCONV_EPI32_ENUM down() const { return _MM_DOWNCONV_EPI32_SINT8; }
    constexpr _MM_UPCONV_EPI32_ENUM up() const { return _MM_UPCONV_EPI32_SINT8; }
    constexpr operator _MM_DOWNCONV_EPI32_ENUM() const { return _MM_DOWNCONV_EPI32_SINT8; }
    constexpr operator _MM_UPCONV_EPI32_ENUM() const { return _MM_UPCONV_EPI32_SINT8; }
};
template <> struct UpDownConversion<int, unsigned char>
{
    constexpr _MM_DOWNCONV_EPI32_ENUM down() const { return _MM_DOWNCONV_EPI32_UINT8; }
    constexpr _MM_UPCONV_EPI32_ENUM up() const { return _MM_UPCONV_EPI32_UINT8; }
    constexpr operator _MM_DOWNCONV_EPI32_ENUM() const { return _MM_DOWNCONV_EPI32_UINT8; }
    constexpr operator _MM_UPCONV_EPI32_ENUM() const { return _MM_UPCONV_EPI32_UINT8; }
};
template <> struct UpDownConversion<int, short>
{
    constexpr _MM_DOWNCONV_EPI32_ENUM down() const { return _MM_DOWNCONV_EPI32_SINT16; }
    constexpr _MM_UPCONV_EPI32_ENUM up() const { return _MM_UPCONV_EPI32_SINT16; }
    constexpr operator _MM_DOWNCONV_EPI32_ENUM() const { return _MM_DOWNCONV_EPI32_SINT16; }
    constexpr operator _MM_UPCONV_EPI32_ENUM() const { return _MM_UPCONV_EPI32_SINT16; }
};
template <> struct UpDownConversion<int, unsigned short>
{
    constexpr _MM_DOWNCONV_EPI32_ENUM down() const { return _MM_DOWNCONV_EPI32_UINT16; }
    constexpr _MM_UPCONV_EPI32_ENUM up() const { return _MM_UPCONV_EPI32_UINT16; }
    constexpr operator _MM_DOWNCONV_EPI32_ENUM() const { return _MM_DOWNCONV_EPI32_UINT16; }
    constexpr operator _MM_UPCONV_EPI32_ENUM() const { return _MM_UPCONV_EPI32_UINT16; }
};
template <> struct UpDownConversion<int, int>
{
    constexpr _MM_DOWNCONV_EPI32_ENUM down() const { return _MM_DOWNCONV_EPI32_NONE; }
    constexpr _MM_UPCONV_EPI32_ENUM up() const { return _MM_UPCONV_EPI32_NONE; }
    constexpr operator _MM_DOWNCONV_EPI32_ENUM() const { return _MM_DOWNCONV_EPI32_NONE; }
    constexpr operator _MM_UPCONV_EPI32_ENUM() const { return _MM_UPCONV_EPI32_NONE; }
};
template <> struct UpDownConversion<int, unsigned int>
{
    constexpr _MM_DOWNCONV_EPI32_ENUM down() const { return _MM_DOWNCONV_EPI32_NONE; }
    constexpr _MM_UPCONV_EPI32_ENUM up() const { return _MM_UPCONV_EPI32_NONE; }
    constexpr operator _MM_DOWNCONV_EPI32_ENUM() const { return _MM_DOWNCONV_EPI32_NONE; }
    constexpr operator _MM_UPCONV_EPI32_ENUM() const { return _MM_UPCONV_EPI32_NONE; }
};
}  // namespace MIC
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_MIC_INTRINSICS_H_
