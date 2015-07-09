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

#ifndef VC_AVX_INTRINSICS_H
#define VC_AVX_INTRINSICS_H

#include "../common/windows_fix_intrin.h"

#include <Vc/global.h>
#include "../traits/type_traits.h"

// see comment in sse/intrinsics.h
extern "C" {
// AVX
#include <immintrin.h>

#if (defined(VC_IMPL_XOP) || defined(VC_IMPL_FMA4)) && !defined(VC_MSVC)
#include <x86intrin.h>
#endif
}

#include "../common/fix_clang_emmintrin.h"

#if defined(VC_CLANG) && VC_CLANG < 0x30100
// _mm_permute_ps is broken: http://llvm.org/bugs/show_bug.cgi?id=12401
#undef _mm_permute_ps
#define _mm_permute_ps(A, C) __extension__ ({ \
  m128 __A = (A); \
  (m128)__builtin_shufflevector((__v4sf)__A, (__v4sf) _mm_setzero_ps(), \
                                   (C) & 0x3, ((C) & 0xc) >> 2, \
                                   ((C) & 0x30) >> 4, ((C) & 0xc0) >> 6); })
#endif

#include "const_data.h"
#include "macros.h"
#include <cstdlib>

#if defined(VC_CLANG) || defined(VC_MSVC) || (defined(VC_GCC) && !defined(__OPTIMIZE__))
#define VC_REQUIRES_MACRO_FOR_IMMEDIATE_ARGUMENT
#endif

#if defined(VC_CLANG) && VC_CLANG <= 0x30000
// _mm_alignr_epi8 doesn't specify its return type, thus breaking overload resolution
#undef _mm_alignr_epi8
#define _mm_alignr_epi8(a, b, n) ((m128i)__builtin_ia32_palignr128((a), (b), (n)))
#endif

namespace Vc_VERSIONED_NAMESPACE
{
namespace AvxIntrinsics
{
    using AVX::c_general;
    using AVX::_IndexesFromZero32;
    using AVX::_IndexesFromZero16;
    using AVX::_IndexesFromZero8;

    typedef __m128  m128 ;
    typedef __m128d m128d;
    typedef __m128i m128i;
    typedef __m256  m256 ;
    typedef __m256d m256d;
    typedef __m256i m256i;

    typedef const m128  param128 ;
    typedef const m128d param128d;
    typedef const m128i param128i;
    typedef const m256  param256 ;
    typedef const m256d param256d;
    typedef const m256i param256i;

#ifdef VC_GCC
    // Redefine the mul/add/sub intrinsics to use GCC-specific operators instead of builtin
    // functions. This way the fp-contraction optimization step kicks in and creates FMAs! :)
    static Vc_INTRINSIC Vc_CONST m256d _mm256_mul_pd(m256d a, m256d b) { return static_cast<m256d>(static_cast<__v4df>(a) * static_cast<__v4df>(b)); }
    static Vc_INTRINSIC Vc_CONST m256d _mm256_add_pd(m256d a, m256d b) { return static_cast<m256d>(static_cast<__v4df>(a) + static_cast<__v4df>(b)); }
    static Vc_INTRINSIC Vc_CONST m256d _mm256_sub_pd(m256d a, m256d b) { return static_cast<m256d>(static_cast<__v4df>(a) - static_cast<__v4df>(b)); }
    static Vc_INTRINSIC Vc_CONST m256 _mm256_mul_ps(m256 a, m256 b) { return static_cast<m256>(static_cast<__v8sf>(a) * static_cast<__v8sf>(b)); }
    static Vc_INTRINSIC Vc_CONST m256 _mm256_add_ps(m256 a, m256 b) { return static_cast<m256>(static_cast<__v8sf>(a) + static_cast<__v8sf>(b)); }
    static Vc_INTRINSIC Vc_CONST m256 _mm256_sub_ps(m256 a, m256 b) { return static_cast<m256>(static_cast<__v8sf>(a) - static_cast<__v8sf>(b)); }
#endif

    static Vc_INTRINSIC m256  Vc_CONST set1_ps   (float  a) { return ::_mm256_set1_ps   (a); }
    static Vc_INTRINSIC m256d Vc_CONST set1_pd   (double a) { return ::_mm256_set1_pd   (a); }
    static Vc_INTRINSIC m256i Vc_CONST set1_epi32(int    a) { return ::_mm256_set1_epi32(a); }
    //static Vc_INTRINSIC m256i Vc_CONST _mm256_set1_epu32(unsigned int a) { return ::_mm256_set1_epu32(a); }

    static Vc_INTRINSIC Vc_CONST m128i _mm_setallone_si128() { return _mm_load_si128(reinterpret_cast<const __m128i *>(Common::AllBitsSet)); }
    static Vc_INTRINSIC Vc_CONST m128  _mm_setallone_ps() { return _mm_load_ps(reinterpret_cast<const float *>(Common::AllBitsSet)); }
    static Vc_INTRINSIC Vc_CONST m128d _mm_setallone_pd() { return _mm_load_pd(reinterpret_cast<const double *>(Common::AllBitsSet)); }

    static Vc_INTRINSIC Vc_CONST m256i setallone_si256() { return _mm256_castps_si256(_mm256_load_ps(reinterpret_cast<const float *>(Common::AllBitsSet))); }
    static Vc_INTRINSIC Vc_CONST m256d setallone_pd() { return _mm256_load_pd(reinterpret_cast<const double *>(Common::AllBitsSet)); }
    static Vc_INTRINSIC Vc_CONST m256  setallone_ps() { return _mm256_load_ps(reinterpret_cast<const float *>(Common::AllBitsSet)); }

    static Vc_INTRINSIC m128i Vc_CONST _mm_setone_epi8 ()  { return _mm_set1_epi8(1); }
    static Vc_INTRINSIC m128i Vc_CONST _mm_setone_epu8 ()  { return _mm_setone_epi8(); }
    static Vc_INTRINSIC m128i Vc_CONST _mm_setone_epi16()  { return _mm_castps_si128(_mm_broadcast_ss(reinterpret_cast<const float *>(c_general::one16))); }
    static Vc_INTRINSIC m128i Vc_CONST _mm_setone_epu16()  { return _mm_setone_epi16(); }
    static Vc_INTRINSIC m128i Vc_CONST _mm_setone_epi32()  { return _mm_castps_si128(_mm_broadcast_ss(reinterpret_cast<const float *>(&_IndexesFromZero32[1]))); }
    static Vc_INTRINSIC m128i Vc_CONST _mm_setone_epu32()  { return _mm_setone_epi32(); }

    static Vc_INTRINSIC m256i Vc_CONST setone_epi8 ()  { return _mm256_set1_epi8(1); }
    static Vc_INTRINSIC m256i Vc_CONST setone_epu8 ()  { return setone_epi8(); }
    static Vc_INTRINSIC m256i Vc_CONST setone_epi16()  { return _mm256_castps_si256(_mm256_broadcast_ss(reinterpret_cast<const float *>(c_general::one16))); }
    static Vc_INTRINSIC m256i Vc_CONST setone_epu16()  { return setone_epi16(); }
    static Vc_INTRINSIC m256i Vc_CONST setone_epi32()  { return _mm256_castps_si256(_mm256_broadcast_ss(reinterpret_cast<const float *>(&_IndexesFromZero32[1]))); }
    static Vc_INTRINSIC m256i Vc_CONST setone_epu32()  { return setone_epi32(); }

    static Vc_INTRINSIC m256  Vc_CONST setone_ps()     { return _mm256_broadcast_ss(&c_general::oneFloat); }
    static Vc_INTRINSIC m256d Vc_CONST setone_pd()     { return _mm256_broadcast_sd(&c_general::oneDouble); }

    static Vc_INTRINSIC m256d Vc_CONST setabsmask_pd() { return _mm256_broadcast_sd(reinterpret_cast<const double *>(&c_general::absMaskFloat[0])); }
    static Vc_INTRINSIC m256  Vc_CONST setabsmask_ps() { return _mm256_broadcast_ss(reinterpret_cast<const float *>(&c_general::absMaskFloat[1])); }
    static Vc_INTRINSIC m256d Vc_CONST setsignmask_pd(){ return _mm256_broadcast_sd(reinterpret_cast<const double *>(&c_general::signMaskFloat[0])); }
    static Vc_INTRINSIC m256  Vc_CONST setsignmask_ps(){ return _mm256_broadcast_ss(reinterpret_cast<const float *>(&c_general::signMaskFloat[1])); }

    static Vc_INTRINSIC m256  Vc_CONST set2power31_ps()    { return _mm256_broadcast_ss(&c_general::_2power31); }
    static Vc_INTRINSIC m128  Vc_CONST _mm_set2power31_ps()    { return _mm_broadcast_ss(&c_general::_2power31); }
    static Vc_INTRINSIC m256i Vc_CONST set2power31_epu32() { return _mm256_castps_si256(_mm256_broadcast_ss(reinterpret_cast<const float *>(&c_general::signMaskFloat[1]))); }
    static Vc_INTRINSIC m128i Vc_CONST _mm_set2power31_epu32() { return _mm_castps_si128(_mm_broadcast_ss(reinterpret_cast<const float *>(&c_general::signMaskFloat[1]))); }

    //X         static Vc_INTRINSIC m256i Vc_CONST setmin_epi8 () { return _mm256_slli_epi8 (setallone_si256(),  7); }
    static Vc_INTRINSIC m128i Vc_CONST _mm_setmin_epi16() { return _mm_castps_si128(_mm_broadcast_ss(reinterpret_cast<const float *>(c_general::minShort))); }
    static Vc_INTRINSIC m128i Vc_CONST _mm_setmin_epi32() { return _mm_castps_si128(_mm_broadcast_ss(reinterpret_cast<const float *>(&c_general::signMaskFloat[1]))); }
    static Vc_INTRINSIC m256i Vc_CONST setmin_epi16() { return _mm256_castps_si256(_mm256_broadcast_ss(reinterpret_cast<const float *>(c_general::minShort))); }
    static Vc_INTRINSIC m256i Vc_CONST setmin_epi32() { return _mm256_castps_si256(_mm256_broadcast_ss(reinterpret_cast<const float *>(&c_general::signMaskFloat[1]))); }

    template <int i>
    static Vc_INTRINSIC Vc_CONST unsigned char extract_epu8(__m128i x)
    {
        return _mm_extract_epi8(x, i);
    }
    template <int i>
    static Vc_INTRINSIC Vc_CONST unsigned short extract_epu16(__m128i x)
    {
        return _mm_extract_epi16(x, i);
    }
    template <int i>
    static Vc_INTRINSIC Vc_CONST unsigned int extract_epu32(__m128i x)
    {
        return _mm_extract_epi32(x, i);
    }

    /////////////////////// COMPARE OPS ///////////////////////
    static Vc_INTRINSIC m256d Vc_CONST cmpeq_pd   (__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_EQ_OQ); }
    static Vc_INTRINSIC m256d Vc_CONST cmpneq_pd  (__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_NEQ_UQ); }
    static Vc_INTRINSIC m256d Vc_CONST cmplt_pd   (__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_LT_OS); }
    static Vc_INTRINSIC m256d Vc_CONST cmpnlt_pd  (__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_NLT_US); }
    static Vc_INTRINSIC m256d Vc_CONST cmple_pd   (__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_LE_OS); }
    static Vc_INTRINSIC m256d Vc_CONST cmpnle_pd  (__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_NLE_US); }
    static Vc_INTRINSIC m256d Vc_CONST cmpord_pd  (__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_ORD_Q); }
    static Vc_INTRINSIC m256d Vc_CONST cmpunord_pd(__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_UNORD_Q); }

    static Vc_INTRINSIC m256  Vc_CONST cmpeq_ps   (__m256  a, __m256  b) { return _mm256_cmp_ps(a, b, _CMP_EQ_OQ); }
    static Vc_INTRINSIC m256  Vc_CONST cmpneq_ps  (__m256  a, __m256  b) { return _mm256_cmp_ps(a, b, _CMP_NEQ_UQ); }
    static Vc_INTRINSIC m256  Vc_CONST cmplt_ps   (__m256  a, __m256  b) { return _mm256_cmp_ps(a, b, _CMP_LT_OS); }
    static Vc_INTRINSIC m256  Vc_CONST cmpnlt_ps  (__m256  a, __m256  b) { return _mm256_cmp_ps(a, b, _CMP_NLT_US); }
    static Vc_INTRINSIC m256  Vc_CONST cmpge_ps   (__m256  a, __m256  b) { return _mm256_cmp_ps(a, b, _CMP_NLT_US); }
    static Vc_INTRINSIC m256  Vc_CONST cmple_ps   (__m256  a, __m256  b) { return _mm256_cmp_ps(a, b, _CMP_LE_OS); }
    static Vc_INTRINSIC m256  Vc_CONST cmpnle_ps  (__m256  a, __m256  b) { return _mm256_cmp_ps(a, b, _CMP_NLE_US); }
    static Vc_INTRINSIC m256  Vc_CONST cmpgt_ps   (__m256  a, __m256  b) { return _mm256_cmp_ps(a, b, _CMP_NLE_US); }
    static Vc_INTRINSIC m256  Vc_CONST cmpord_ps  (__m256  a, __m256  b) { return _mm256_cmp_ps(a, b, _CMP_ORD_Q); }
    static Vc_INTRINSIC m256  Vc_CONST cmpunord_ps(__m256  a, __m256  b) { return _mm256_cmp_ps(a, b, _CMP_UNORD_Q); }

#if defined(VC_IMPL_XOP) && !defined(VC_CLANG)
    static Vc_INTRINSIC m128i cmplt_epu16(__m128i a, __m128i b) {
        return _mm_comlt_epu16(a, b);
    }
    static Vc_INTRINSIC m128i cmpgt_epu16(__m128i a, __m128i b) {
        return _mm_comgt_epu16(a, b);
    }
#else
    static Vc_INTRINSIC m128i cmplt_epu16(__m128i a, __m128i b) {
        return _mm_cmplt_epi16(_mm_xor_si128(a, _mm_setmin_epi16()), _mm_xor_si128(b, _mm_setmin_epi16()));
    }
    static Vc_INTRINSIC m128i cmpgt_epu16(__m128i a, __m128i b) {
        return _mm_cmpgt_epi16(_mm_xor_si128(a, _mm_setmin_epi16()), _mm_xor_si128(b, _mm_setmin_epi16()));
    }
#endif

#ifdef VC_IMPL_AVX2
    template <int shift> Vc_INTRINSIC Vc_CONST m256i alignr(__m256i s1, __m256i s2)
    {
        return _mm256_alignr_epi8(s1, s2, shift);
    }
#else
    template <int shift> Vc_INTRINSIC Vc_CONST m256i alignr(__m256i s1, __m256i s2)
    {
        return _mm256_insertf128_si256(
            _mm256_castsi128_si256(_mm_alignr_epi8(_mm256_castsi256_si128(s1),
                                                   _mm256_castsi256_si128(s2), shift)),
            _mm_alignr_epi8(_mm256_extractf128_si256(s1, 1),
                            _mm256_extractf128_si256(s2, 1), shift),
            1);
    }
#endif

#ifdef VC_IMPL_AVX2
#define AVX_TO_SSE_2_NEW(name)                                                           \
    Vc_INTRINSIC Vc_CONST m256i name(__m256i a0, __m256i b0)                         \
    {                                                                                    \
        return _mm256_##name(a0, b0);                                                    \
    }
#define AVX_TO_SSE_256_128(name)                                                         \
    Vc_INTRINSIC Vc_CONST m256i name(__m256i a0, __m128i b0)                         \
    {                                                                                    \
        return _mm256_##name(a0, b0);                                                    \
    }
#define AVX_TO_SSE_1i(name)                                                              \
    template <int i> Vc_INTRINSIC Vc_CONST m256i name(__m256i a0)                      \
    {                                                                                    \
        return _mm256_##name(a0, i);                                                     \
    }
#define AVX_TO_SSE_1(name)                                                               \
    Vc_INTRINSIC Vc_CONST __m256i name(__m256i a0) { return _mm256_##name(a0); }
#define AVX_TO_SSE_1_128(name, shift__)                                                  \
    Vc_INTRINSIC Vc_CONST __m256i name(__m128i a0) { return _mm256_##name(a0); }
#else
/**\internal
 * Defines the function \p name, which takes to __m256i arguments and calls `_mm_##name` on the low
 * and high 128 bit halfs of the arguments.
 *
 * In case the AVX2 intrinsics are enabled, the arguments are directly passed to a single
 * `_mm256_##name` call.
 */
#define AVX_TO_SSE_1(name)                                                               \
    Vc_INTRINSIC Vc_CONST __m256i name(__m256i a0)                                       \
    {                                                                                    \
        __m128i a1 = _mm256_extractf128_si256(a0, 1);                                    \
        __m128i r0 = _mm_##name(_mm256_castsi256_si128(a0));                             \
        __m128i r1 = _mm_##name(a1);                                                     \
        return _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);               \
    }
#define AVX_TO_SSE_1_128(name, shift__)                                                  \
    Vc_INTRINSIC Vc_CONST __m256i name(__m128i a0)                                       \
    {                                                                                    \
        __m128i r0 = _mm_##name(a0);                                                     \
        __m128i r1 = _mm_##name(_mm_srli_si128(a0, shift__));                            \
        return _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);               \
    }
#define AVX_TO_SSE_2_NEW(name)                                                           \
    Vc_INTRINSIC Vc_CONST m256i name(__m256i a0, __m256i b0)                         \
    {                                                                                    \
        m128i a1 = _mm256_extractf128_si256(a0, 1);                                      \
        m128i b1 = _mm256_extractf128_si256(b0, 1);                                      \
        m128i r0 = _mm_##name(_mm256_castsi256_si128(a0), _mm256_castsi256_si128(b0));   \
        m128i r1 = _mm_##name(a1, b1);                                                   \
        return _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);               \
    }
#define AVX_TO_SSE_256_128(name)                                                         \
    Vc_INTRINSIC Vc_CONST m256i name(__m256i a0, __m128i b0)                         \
    {                                                                                    \
        m128i a1 = _mm256_extractf128_si256(a0, 1);                                      \
        m128i r0 = _mm_##name(_mm256_castsi256_si128(a0), b0);                           \
        m128i r1 = _mm_##name(a1, b0);                                                   \
        return _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);               \
    }
#define AVX_TO_SSE_1i(name)                                                              \
    template <int i> Vc_INTRINSIC Vc_CONST m256i name(__m256i a0)                      \
    {                                                                                    \
        m128i a1 = _mm256_extractf128_si256(a0, 1);                                      \
        m128i r0 = _mm_##name(_mm256_castsi256_si128(a0), i);                            \
        m128i r1 = _mm_##name(a1, i);                                                    \
        return _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);               \
    }
#endif
    Vc_INTRINSIC Vc_CONST __m128i sll_epi16(__m128i a, __m128i b) { return _mm_sll_epi16(a, b); }
    Vc_INTRINSIC Vc_CONST __m128i sll_epi32(__m128i a, __m128i b) { return _mm_sll_epi32(a, b); }
    Vc_INTRINSIC Vc_CONST __m128i sll_epi64(__m128i a, __m128i b) { return _mm_sll_epi64(a, b); }
    Vc_INTRINSIC Vc_CONST __m128i srl_epi16(__m128i a, __m128i b) { return _mm_srl_epi16(a, b); }
    Vc_INTRINSIC Vc_CONST __m128i srl_epi32(__m128i a, __m128i b) { return _mm_srl_epi32(a, b); }
    Vc_INTRINSIC Vc_CONST __m128i srl_epi64(__m128i a, __m128i b) { return _mm_srl_epi64(a, b); }
    Vc_INTRINSIC Vc_CONST __m128i sra_epi16(__m128i a, __m128i b) { return _mm_sra_epi16(a, b); }
    Vc_INTRINSIC Vc_CONST __m128i sra_epi32(__m128i a, __m128i b) { return _mm_sra_epi32(a, b); }

    AVX_TO_SSE_1i(slli_epi16)
    AVX_TO_SSE_1i(slli_epi32)
    AVX_TO_SSE_1i(slli_epi64)
    AVX_TO_SSE_1i(srai_epi16)
    AVX_TO_SSE_1i(srai_epi32)
    AVX_TO_SSE_1i(srli_epi16)
    AVX_TO_SSE_1i(srli_epi32)
    AVX_TO_SSE_1i(srli_epi64)

    AVX_TO_SSE_256_128(sll_epi16)
    AVX_TO_SSE_256_128(sll_epi32)
    AVX_TO_SSE_256_128(sll_epi64)
    AVX_TO_SSE_256_128(srl_epi16)
    AVX_TO_SSE_256_128(srl_epi32)
    AVX_TO_SSE_256_128(srl_epi64)
    AVX_TO_SSE_256_128(sra_epi16)
    AVX_TO_SSE_256_128(sra_epi32)

    AVX_TO_SSE_2_NEW(cmpeq_epi8)
    AVX_TO_SSE_2_NEW(cmpeq_epi16)
    AVX_TO_SSE_2_NEW(cmpeq_epi32)
    AVX_TO_SSE_2_NEW(cmpeq_epi64)
    AVX_TO_SSE_2_NEW(cmpgt_epi8)
    AVX_TO_SSE_2_NEW(cmpgt_epi16)
    AVX_TO_SSE_2_NEW(cmpgt_epi32)
    AVX_TO_SSE_2_NEW(cmpgt_epi64)
    AVX_TO_SSE_2_NEW(packs_epi16)
    AVX_TO_SSE_2_NEW(packs_epi32)
    AVX_TO_SSE_2_NEW(packus_epi16)
    AVX_TO_SSE_2_NEW(unpackhi_epi8)
    AVX_TO_SSE_2_NEW(unpackhi_epi16)
    AVX_TO_SSE_2_NEW(unpackhi_epi32)
    AVX_TO_SSE_2_NEW(unpackhi_epi64)
    AVX_TO_SSE_2_NEW(unpacklo_epi8)
    AVX_TO_SSE_2_NEW(unpacklo_epi16)
    AVX_TO_SSE_2_NEW(unpacklo_epi32)
    AVX_TO_SSE_2_NEW(unpacklo_epi64)
    AVX_TO_SSE_2_NEW(add_epi8)
    AVX_TO_SSE_2_NEW(add_epi16)
    AVX_TO_SSE_2_NEW(add_epi32)
    AVX_TO_SSE_2_NEW(add_epi64)
    AVX_TO_SSE_2_NEW(adds_epi8)
    AVX_TO_SSE_2_NEW(adds_epi16)
    AVX_TO_SSE_2_NEW(adds_epu8)
    AVX_TO_SSE_2_NEW(adds_epu16)
    AVX_TO_SSE_2_NEW(sub_epi8)
    AVX_TO_SSE_2_NEW(sub_epi16)
    AVX_TO_SSE_2_NEW(sub_epi32)
    AVX_TO_SSE_2_NEW(sub_epi64)
    AVX_TO_SSE_2_NEW(subs_epi8)
    AVX_TO_SSE_2_NEW(subs_epi16)
    AVX_TO_SSE_2_NEW(subs_epu8)
    AVX_TO_SSE_2_NEW(subs_epu16)
    AVX_TO_SSE_2_NEW(madd_epi16)
    AVX_TO_SSE_2_NEW(mulhi_epi16)
    AVX_TO_SSE_2_NEW(mullo_epi16)
    AVX_TO_SSE_2_NEW(mul_epu32)
    AVX_TO_SSE_2_NEW(max_epi16)
    AVX_TO_SSE_2_NEW(max_epu8)
    AVX_TO_SSE_2_NEW(min_epi16)
    AVX_TO_SSE_2_NEW(min_epu8)
    AVX_TO_SSE_2_NEW(mulhi_epu16)
    // shufflehi_epi16
    // shufflelo_epi16 (__m128i __A, const int __mask)
    // shuffle_epi32 (__m128i __A, const int __mask)
    // maskmoveu_si128 (__m128i __A, __m128i __B, char *__C)
    AVX_TO_SSE_2_NEW(avg_epu8)
    AVX_TO_SSE_2_NEW(avg_epu16)
    AVX_TO_SSE_2_NEW(sad_epu8)
    // stream_si32 (int *__A, int __B)
    // stream_si128 (__m128i *__A, __m128i __B)
    // cvtsi32_si128 (int __A)
    // cvtsi64_si128 (long long __A)
    // cvtsi64x_si128 (long long __A)
    AVX_TO_SSE_2_NEW(hadd_epi16)
    AVX_TO_SSE_2_NEW(hadd_epi32)
    AVX_TO_SSE_2_NEW(hadds_epi16)
    AVX_TO_SSE_2_NEW(hsub_epi16)
    AVX_TO_SSE_2_NEW(hsub_epi32)
    AVX_TO_SSE_2_NEW(hsubs_epi16)
    AVX_TO_SSE_2_NEW(maddubs_epi16)
    AVX_TO_SSE_2_NEW(mulhrs_epi16)
    AVX_TO_SSE_2_NEW(shuffle_epi8)
    AVX_TO_SSE_2_NEW(sign_epi8)
    AVX_TO_SSE_2_NEW(sign_epi16)
    AVX_TO_SSE_2_NEW(sign_epi32)
    AVX_TO_SSE_2_NEW(min_epi8)
    AVX_TO_SSE_2_NEW(max_epi8)
    AVX_TO_SSE_2_NEW(min_epu16)
    AVX_TO_SSE_2_NEW(max_epu16)
    AVX_TO_SSE_2_NEW(min_epi32)
    AVX_TO_SSE_2_NEW(max_epi32)
    AVX_TO_SSE_2_NEW(min_epu32)
    AVX_TO_SSE_2_NEW(max_epu32)
    AVX_TO_SSE_2_NEW(mullo_epi32)
    AVX_TO_SSE_2_NEW(mul_epi32)

    AVX_TO_SSE_1(abs_epi8)
    AVX_TO_SSE_1(abs_epi16)
    AVX_TO_SSE_1(abs_epi32)
    AVX_TO_SSE_1_128(cvtepi8_epi16, 8)
    AVX_TO_SSE_1_128(cvtepi8_epi32, 4)
    AVX_TO_SSE_1_128(cvtepi8_epi64, 2)
    AVX_TO_SSE_1_128(cvtepi16_epi32, 8)
    AVX_TO_SSE_1_128(cvtepi16_epi64, 4)
    AVX_TO_SSE_1_128(cvtepi32_epi64, 8)
    AVX_TO_SSE_1_128(cvtepu8_epi16, 8)
    AVX_TO_SSE_1_128(cvtepu8_epi32, 4)
    AVX_TO_SSE_1_128(cvtepu8_epi64, 2)
    AVX_TO_SSE_1_128(cvtepu16_epi32, 8)
    AVX_TO_SSE_1_128(cvtepu16_epi64, 4)
    AVX_TO_SSE_1_128(cvtepu32_epi64, 8)
#if !defined(VC_CLANG) || VC_CLANG > 0x30100
    // clang is missing _mm_minpos_epu16 from smmintrin.h
    // http://llvm.org/bugs/show_bug.cgi?id=12399
    //AVX_TO_SSE_1(minpos_epu16)
#endif

    AVX_TO_SSE_2_NEW(packus_epi32)

#ifndef VC_IMPL_AVX2

/////////////////////////////////////////////////////////////////////////
// implementation of the intrinsics missing in AVX
/////////////////////////////////////////////////////////////////////////

    template <int i> Vc_INTRINSIC Vc_CONST __m256i srli_si256(__m256i a0) {
        const m128i vLo = _mm256_castsi256_si128(a0);
        const m128i vHi = _mm256_extractf128_si256(a0, 1);
        switch (i) {
        case  0: return a0;
        case  1: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo,  1)), _mm_srli_si128(vHi,  1), 1);
        case  2: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo,  2)), _mm_srli_si128(vHi,  2), 1);
        case  3: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo,  3)), _mm_srli_si128(vHi,  3), 1);
        case  4: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo,  4)), _mm_srli_si128(vHi,  4), 1);
        case  5: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo,  5)), _mm_srli_si128(vHi,  5), 1);
        case  6: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo,  6)), _mm_srli_si128(vHi,  6), 1);
        case  7: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo,  7)), _mm_srli_si128(vHi,  7), 1);
        case  8: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo,  8)), _mm_srli_si128(vHi,  8), 1);
        case  9: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo,  9)), _mm_srli_si128(vHi,  9), 1);
        case 10: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo, 10)), _mm_srli_si128(vHi, 10), 1);
        case 11: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo, 11)), _mm_srli_si128(vHi, 11), 1);
        case 12: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo, 12)), _mm_srli_si128(vHi, 12), 1);
        case 13: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo, 13)), _mm_srli_si128(vHi, 13), 1);
        case 14: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo, 14)), _mm_srli_si128(vHi, 14), 1);
        case 15: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_alignr_epi8(vHi, vLo, 15)), _mm_srli_si128(vHi, 15), 1);
        case 16: return _mm256_permute2f128_si256(a0, a0, 0x81);
        case 17: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi,  1)), _mm256_castsi128_si256(_mm_srli_si128(vHi,  1)), 0x80);
        case 18: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi,  2)), _mm256_castsi128_si256(_mm_srli_si128(vHi,  2)), 0x80);
        case 19: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi,  3)), _mm256_castsi128_si256(_mm_srli_si128(vHi,  3)), 0x80);
        case 20: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi,  4)), _mm256_castsi128_si256(_mm_srli_si128(vHi,  4)), 0x80);
        case 21: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi,  5)), _mm256_castsi128_si256(_mm_srli_si128(vHi,  5)), 0x80);
        case 22: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi,  6)), _mm256_castsi128_si256(_mm_srli_si128(vHi,  6)), 0x80);
        case 23: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi,  7)), _mm256_castsi128_si256(_mm_srli_si128(vHi,  7)), 0x80);
        case 24: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi,  8)), _mm256_castsi128_si256(_mm_srli_si128(vHi,  8)), 0x80);
        case 25: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi,  9)), _mm256_castsi128_si256(_mm_srli_si128(vHi,  9)), 0x80);
        case 26: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi, 10)), _mm256_castsi128_si256(_mm_srli_si128(vHi, 10)), 0x80);
        case 27: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi, 11)), _mm256_castsi128_si256(_mm_srli_si128(vHi, 11)), 0x80);
        case 28: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi, 12)), _mm256_castsi128_si256(_mm_srli_si128(vHi, 12)), 0x80);
        case 29: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi, 13)), _mm256_castsi128_si256(_mm_srli_si128(vHi, 13)), 0x80);
        case 30: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi, 14)), _mm256_castsi128_si256(_mm_srli_si128(vHi, 14)), 0x80);
        case 31: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_srli_si128(vHi, 15)), _mm256_castsi128_si256(_mm_srli_si128(vHi, 15)), 0x80);
        }
        return _mm256_setzero_si256();
    }
    template <int i> Vc_INTRINSIC Vc_CONST m256i slli_si256(__m256i a0) {
        const m128i vLo = _mm256_castsi256_si128(a0);
        const m128i vHi = _mm256_extractf128_si256(a0, 1);
        switch (i) {
        case  0: return a0;
        case  1: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  1)), _mm_alignr_epi8(vHi, vLo, 15), 1);
        case  2: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  2)), _mm_alignr_epi8(vHi, vLo, 14), 1);
        case  3: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  3)), _mm_alignr_epi8(vHi, vLo, 13), 1);
        case  4: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  4)), _mm_alignr_epi8(vHi, vLo, 12), 1);
        case  5: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  5)), _mm_alignr_epi8(vHi, vLo, 11), 1);
        case  6: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  6)), _mm_alignr_epi8(vHi, vLo, 10), 1);
        case  7: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  7)), _mm_alignr_epi8(vHi, vLo,  9), 1);
        case  8: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  8)), _mm_alignr_epi8(vHi, vLo,  8), 1);
        case  9: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  9)), _mm_alignr_epi8(vHi, vLo,  7), 1);
        case 10: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo, 10)), _mm_alignr_epi8(vHi, vLo,  6), 1);
        case 11: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo, 11)), _mm_alignr_epi8(vHi, vLo,  5), 1);
        case 12: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo, 12)), _mm_alignr_epi8(vHi, vLo,  4), 1);
        case 13: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo, 13)), _mm_alignr_epi8(vHi, vLo,  3), 1);
        case 14: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo, 14)), _mm_alignr_epi8(vHi, vLo,  2), 1);
        case 15: return _mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo, 15)), _mm_alignr_epi8(vHi, vLo,  1), 1);
        case 16: return _mm256_permute2f128_si256(a0, a0, 0x8);
        case 17: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  1)), _mm256_castsi128_si256(_mm_slli_si128(vLo,  1)), 0x8);
        case 18: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  2)), _mm256_castsi128_si256(_mm_slli_si128(vLo,  2)), 0x8);
        case 19: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  3)), _mm256_castsi128_si256(_mm_slli_si128(vLo,  3)), 0x8);
        case 20: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  4)), _mm256_castsi128_si256(_mm_slli_si128(vLo,  4)), 0x8);
        case 21: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  5)), _mm256_castsi128_si256(_mm_slli_si128(vLo,  5)), 0x8);
        case 22: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  6)), _mm256_castsi128_si256(_mm_slli_si128(vLo,  6)), 0x8);
        case 23: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  7)), _mm256_castsi128_si256(_mm_slli_si128(vLo,  7)), 0x8);
        case 24: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  8)), _mm256_castsi128_si256(_mm_slli_si128(vLo,  8)), 0x8);
        case 25: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo,  9)), _mm256_castsi128_si256(_mm_slli_si128(vLo,  9)), 0x8);
        case 26: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo, 10)), _mm256_castsi128_si256(_mm_slli_si128(vLo, 10)), 0x8);
        case 27: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo, 11)), _mm256_castsi128_si256(_mm_slli_si128(vLo, 11)), 0x8);
        case 28: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo, 12)), _mm256_castsi128_si256(_mm_slli_si128(vLo, 12)), 0x8);
        case 29: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo, 13)), _mm256_castsi128_si256(_mm_slli_si128(vLo, 13)), 0x8);
        case 30: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo, 14)), _mm256_castsi128_si256(_mm_slli_si128(vLo, 14)), 0x8);
        case 31: return _mm256_permute2f128_si256(_mm256_castsi128_si256(_mm_slli_si128(vLo, 15)), _mm256_castsi128_si256(_mm_slli_si128(vLo, 15)), 0x8);
        }
        return _mm256_setzero_si256();
    }

    static Vc_INTRINSIC m256i Vc_CONST and_si256(__m256i x, __m256i y) {
        return _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(x), _mm256_castsi256_ps(y)));
    }
    static Vc_INTRINSIC m256i Vc_CONST andnot_si256(__m256i x, __m256i y) {
        return _mm256_castps_si256(_mm256_andnot_ps(_mm256_castsi256_ps(x), _mm256_castsi256_ps(y)));
    }
    static Vc_INTRINSIC m256i Vc_CONST or_si256(__m256i x, __m256i y) {
        return _mm256_castps_si256(_mm256_or_ps(_mm256_castsi256_ps(x), _mm256_castsi256_ps(y)));
    }
    static Vc_INTRINSIC m256i Vc_CONST xor_si256(__m256i x, __m256i y) {
        return _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(x), _mm256_castsi256_ps(y)));
    }

    Vc_INTRINSIC Vc_CONST int movemask_epi8(__m256i a0)
    {
        m128i a1 = _mm256_extractf128_si256(a0, 1);
        return (_mm_movemask_epi8(a1) << 16) | _mm_movemask_epi8(_mm256_castsi256_si128(a0));
    }
    template <int m> Vc_INTRINSIC Vc_CONST m256i blend_epi16(param256i a0, param256i b0)
    {
        m128i a1 = _mm256_extractf128_si256(a0, 1);
        m128i b1 = _mm256_extractf128_si256(b0, 1);
        m128i r0 = _mm_blend_epi16(_mm256_castsi256_si128(a0), _mm256_castsi256_si128(b0), m & 0xff);
        m128i r1 = _mm_blend_epi16(a1, b1, m >> 8);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
    }
    Vc_INTRINSIC Vc_CONST m256i blendv_epi8(param256i a0, param256i b0, param256i m0) {
        m128i a1 = _mm256_extractf128_si256(a0, 1);
        m128i b1 = _mm256_extractf128_si256(b0, 1);
        m128i m1 = _mm256_extractf128_si256(m0, 1);
        m128i r0 = _mm_blendv_epi8(_mm256_castsi256_si128(a0), _mm256_castsi256_si128(b0), _mm256_castsi256_si128(m0));
        m128i r1 = _mm_blendv_epi8(a1, b1, m1);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
    }
    // mpsadbw_epu8 (__m128i __X, __m128i __Y, const int __M)
    // stream_load_si128 (__m128i *__X)

#if defined(VC_IMPL_FMA4) && defined(VC_CLANG) && VC_CLANG < 0x30300
        // clang miscompiles _mm256_macc_ps: http://llvm.org/bugs/show_bug.cgi?id=15040
        static Vc_INTRINSIC __m256 my256_macc_ps(__m256 a, __m256 b, __m256 c) {
            __m256 r;
            // avoid loading c from memory as that would trigger the bug
            asm("vfmaddps %[c], %[b], %[a], %[r]" : [r]"=x"(r) : [a]"x"(a), [b]"x"(b), [c]"x"(c));
            return r;
        }
#ifdef _mm256_macc_ps
#undef _mm256_macc_ps
#endif
#define _mm256_macc_ps(a, b, c) Vc::AVX::my256_macc_ps(a, b, c)

        static Vc_INTRINSIC __m256d my256_macc_pd(__m256d a, __m256d b, __m256d c) {
            __m256d r;
            // avoid loading c from memory as that would trigger the bug
            asm("vfmaddpd %[c], %[b], %[a], %[r]" : [r]"=x"(r) : [a]"x"(a), [b]"x"(b), [c]"x"(c));
            return r;
        }
#ifdef _mm256_macc_pd
#undef _mm256_macc_pd
#endif
#define _mm256_macc_pd(a, b, c) Vc::AVX::my256_macc_pd(a, b, c)
#endif

#else // VC_IMPL_AVX2

static Vc_INTRINSIC Vc_CONST m256i xor_si256(__m256i x, __m256i y) { return _mm256_xor_si256(x, y); }
static Vc_INTRINSIC Vc_CONST m256i or_si256(__m256i x, __m256i y) { return _mm256_or_si256(x, y); }
static Vc_INTRINSIC Vc_CONST m256i and_si256(__m256i x, __m256i y) { return _mm256_and_si256(x, y); }
static Vc_INTRINSIC Vc_CONST m256i andnot_si256(__m256i x, __m256i y) { return _mm256_andnot_si256(x, y); }

template <int i> Vc_INTRINSIC Vc_CONST __m256i srli_si256(__m256i a0)
{
    return _mm256_srli_si256(a0, i);
}
template <int i> Vc_INTRINSIC Vc_CONST __m256i slli_si256(__m256i a0)
{
    return _mm256_slli_si256(a0, i);
}

/////////////////////////////////////////////////////////////////////////
// implementation of the intrinsics missing in AVX2
/////////////////////////////////////////////////////////////////////////
Vc_INTRINSIC Vc_CONST m256i blendv_epi8(__m256i a0, __m256i b0, __m256i m0)
{
    return _mm256_blendv_epi8(a0, b0, m0);
}
Vc_INTRINSIC Vc_CONST int movemask_epi8(__m256i a0)
{
    return _mm256_movemask_epi8(a0);
}

#endif // VC_IMPL_AVX2

static Vc_INTRINSIC m256i cmplt_epi64(__m256i a, __m256i b) {
    return cmpgt_epi64(b, a);
}
static Vc_INTRINSIC m256i cmplt_epi32(__m256i a, __m256i b) {
    return cmpgt_epi32(b, a);
}
static Vc_INTRINSIC m256i cmplt_epi16(__m256i a, __m256i b) {
    return cmpgt_epi16(b, a);
}
static Vc_INTRINSIC m256i cmplt_epi8(__m256i a, __m256i b) {
    return cmpgt_epi8(b, a);
}

/////////////////////////////////////////////////////////////////////////
// implementation of intrinsics missing in AVX and AVX2
/////////////////////////////////////////////////////////////////////////

//X     static Vc_INTRINSIC m256i cmplt_epu8 (__m256i a, __m256i b) { return cmplt_epi8 (
//X             xor_si256(a, setmin_epi8 ()), xor_si256(b, setmin_epi8 ())); }
//X     static Vc_INTRINSIC m256i cmpgt_epu8 (__m256i a, __m256i b) { return cmpgt_epi8 (
//X             xor_si256(a, setmin_epi8 ()), xor_si256(b, setmin_epi8 ())); }
#if defined(VC_IMPL_XOP) && (!defined(VC_CLANG) || VC_CLANG >= 0x30400)
    AVX_TO_SSE_2_NEW(comlt_epu32)
    AVX_TO_SSE_2_NEW(comgt_epu32)
    AVX_TO_SSE_2_NEW(comlt_epu16)
    AVX_TO_SSE_2_NEW(comgt_epu16)
    static Vc_INTRINSIC m256i Vc_CONST cmplt_epu32(__m256i a, __m256i b) { return comlt_epu32(a, b); }
    static Vc_INTRINSIC m256i Vc_CONST cmpgt_epu32(__m256i a, __m256i b) { return comgt_epu32(a, b); }
    static Vc_INTRINSIC m256i Vc_CONST cmplt_epu16(__m256i a, __m256i b) { return comlt_epu16(a, b); }
    static Vc_INTRINSIC m256i Vc_CONST cmpgt_epu16(__m256i a, __m256i b) { return comgt_epu16(a, b); }
#else
    static Vc_INTRINSIC m256i Vc_CONST cmplt_epu32(__m256i _a, __m256i _b) {
        m256i a = _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(_a), _mm256_castsi256_ps(setmin_epi32())));
        m256i b = _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(_b), _mm256_castsi256_ps(setmin_epi32())));
        return cmplt_epi32(a, b);
    }
    static Vc_INTRINSIC m256i Vc_CONST cmpgt_epu32(__m256i _a, __m256i _b) {
        m256i a = _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(_a), _mm256_castsi256_ps(setmin_epi32())));
        m256i b = _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(_b), _mm256_castsi256_ps(setmin_epi32())));
        return cmpgt_epi32(a, b);
    }
    static Vc_INTRINSIC m256i Vc_CONST cmplt_epu16(__m256i _a, __m256i _b) {
        m256i a = _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(_a), _mm256_castsi256_ps(setmin_epi32())));
        m256i b = _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(_b), _mm256_castsi256_ps(setmin_epi32())));
        return cmplt_epi16(a, b);
    }
    static Vc_INTRINSIC m256i Vc_CONST cmpgt_epu16(__m256i _a, __m256i _b) {
        m256i a = _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(_a), _mm256_castsi256_ps(setmin_epi32())));
        m256i b = _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(_b), _mm256_castsi256_ps(setmin_epi32())));
        return cmpgt_epi16(a, b);
    }
#endif

static Vc_INTRINSIC void _mm256_maskstore(float *mem, const __m256 mask, const __m256 v) {
#ifndef VC_MM256_MASKSTORE_WRONG_MASK_TYPE
    _mm256_maskstore_ps(mem, _mm256_castps_si256(mask), v);
#else
    _mm256_maskstore_ps(mem, mask, v);
#endif
}
static Vc_INTRINSIC void _mm256_maskstore(double *mem, const __m256d mask, const __m256d v) {
#ifndef VC_MM256_MASKSTORE_WRONG_MASK_TYPE
    _mm256_maskstore_pd(mem, _mm256_castpd_si256(mask), v);
#else
    _mm256_maskstore_pd(mem, mask, v);
#endif
}
static Vc_INTRINSIC void _mm256_maskstore(int *mem, const __m256i mask, const __m256i v) {
#ifdef VC_IMPL_AVX2
    _mm256_maskstore_epi32(mem, mask, v);
#elif !defined(VC_MM256_MASKSTORE_WRONG_MASK_TYPE)
    _mm256_maskstore_ps(reinterpret_cast<float *>(mem), mask, _mm256_castsi256_ps(v));
#else
    _mm256_maskstore_ps(reinterpret_cast<float *>(mem), _mm256_castsi256_ps(mask), _mm256_castsi256_ps(v));
#endif
}
static Vc_INTRINSIC void _mm256_maskstore(unsigned int *mem, const __m256i mask, const __m256i v) {
    _mm256_maskstore(reinterpret_cast<int *>(mem), mask, v);
}

#undef AVX_TO_SSE_1
#undef AVX_TO_SSE_1_128
#undef AVX_TO_SSE_2_NEW
#undef AVX_TO_SSE_256_128
#undef AVX_TO_SSE_1i

template<typename R> Vc_INTRINSIC_L R stream_load(const float *mem) Vc_INTRINSIC_R;
template<> Vc_INTRINSIC m128 stream_load<m128>(const float *mem)
{
    return _mm_castsi128_ps(_mm_stream_load_si128(reinterpret_cast<__m128i *>(const_cast<float *>(mem))));
}
template<> Vc_INTRINSIC m256 stream_load<m256>(const float *mem)
{
    return _mm256_insertf128_ps(_mm256_castps128_ps256(stream_load<m128>(mem)),
                                stream_load<m128>(mem + 4), 1);
}

template<typename R> Vc_INTRINSIC_L R stream_load(const double *mem) Vc_INTRINSIC_R;
template<> Vc_INTRINSIC m128d stream_load<m128d>(const double *mem)
{
    return _mm_castsi128_pd(_mm_stream_load_si128(reinterpret_cast<__m128i *>(const_cast<double *>(mem))));
}
template<> Vc_INTRINSIC m256d stream_load<m256d>(const double *mem)
{
    return _mm256_insertf128_pd(_mm256_castpd128_pd256(stream_load<m128d>(mem)),
                                stream_load<m128d>(mem + 2), 1);
}

template<typename R> Vc_INTRINSIC_L R stream_load(const void *mem) Vc_INTRINSIC_R;
template<> Vc_INTRINSIC m128i stream_load<m128i>(const void *mem)
{
    return _mm_stream_load_si128(reinterpret_cast<__m128i *>(const_cast<void *>(mem)));
}
template<> Vc_INTRINSIC m256i stream_load<m256i>(const void *mem)
{
    return _mm256_insertf128_si256(_mm256_castsi128_si256(stream_load<m128i>(mem)),
                                stream_load<m128i>(static_cast<const __m128i *>(mem) + 1), 1);
}

Vc_INTRINSIC void stream_store(float *mem, __m128 value, __m128 mask)
{
    _mm_maskmoveu_si128(_mm_castps_si128(value), _mm_castps_si128(mask), reinterpret_cast<char *>(mem));
}
Vc_INTRINSIC void stream_store(float *mem, __m256 value, __m256 mask)
{
    stream_store(mem, _mm256_castps256_ps128(value), _mm256_castps256_ps128(mask));
    stream_store(mem + 4, _mm256_extractf128_ps(value, 1), _mm256_extractf128_ps(mask, 1));
}
Vc_INTRINSIC void stream_store(double *mem, __m128d value, __m128d mask)
{
    _mm_maskmoveu_si128(_mm_castpd_si128(value), _mm_castpd_si128(mask), reinterpret_cast<char *>(mem));
}
Vc_INTRINSIC void stream_store(double *mem, __m256d value, __m256d mask)
{
    stream_store(mem, _mm256_castpd256_pd128(value), _mm256_castpd256_pd128(mask));
    stream_store(mem + 2, _mm256_extractf128_pd(value, 1), _mm256_extractf128_pd(mask, 1));
}
Vc_INTRINSIC void stream_store(void *mem, __m128i value, __m128i mask)
{
    _mm_maskmoveu_si128(value, mask, reinterpret_cast<char *>(mem));
}
Vc_INTRINSIC void stream_store(void *mem, __m256i value, __m256i mask)
{
    stream_store(mem, _mm256_castsi256_si128(value), _mm256_castsi256_si128(mask));
    stream_store(static_cast<__m128i *>(mem) + 1, _mm256_extractf128_si256(value, 1), _mm256_extractf128_si256(mask, 1));
}

#ifndef __x86_64__
Vc_INTRINSIC Vc_PURE __m128i _mm_cvtsi64_si128(int64_t x) {
    return _mm_castpd_si128(_mm_load_sd(reinterpret_cast<const double *>(&x)));
}
#endif

}  // namespace AvxIntrinsics
}  // namespace Vc

namespace Vc_VERSIONED_NAMESPACE
{
namespace AVX
{
    using namespace AvxIntrinsics;
}  // namespace AVX
namespace AVX2
{
    using namespace AvxIntrinsics;
}  // namespace AVX2
namespace Vc_AVX_NAMESPACE
{
    template<typename T> struct VectorTypeHelper;
#ifdef VC_IMPL_AVX2
    template<> struct VectorTypeHelper<         char > { typedef __m256i Type; };
    template<> struct VectorTypeHelper<  signed char > { typedef __m256i Type; };
    template<> struct VectorTypeHelper<unsigned char > { typedef __m256i Type; };
    template<> struct VectorTypeHelper<         short> { typedef __m256i Type; };
    template<> struct VectorTypeHelper<unsigned short> { typedef __m256i Type; };
    template<> struct VectorTypeHelper<         int  > { typedef __m256i Type; };
    template<> struct VectorTypeHelper<unsigned int  > { typedef __m256i Type; };
    template<> struct VectorTypeHelper<         long > { typedef __m256i Type; };
    template<> struct VectorTypeHelper<unsigned long > { typedef __m256i Type; };
    template<> struct VectorTypeHelper<         long long> { typedef __m256i Type; };
    template<> struct VectorTypeHelper<unsigned long long> { typedef __m256i Type; };
#else
    template<> struct VectorTypeHelper<         char > { typedef __m128i Type; };
    template<> struct VectorTypeHelper<  signed char > { typedef __m128i Type; };
    template<> struct VectorTypeHelper<unsigned char > { typedef __m128i Type; };
    template<> struct VectorTypeHelper<         short> { typedef __m128i Type; };
    template<> struct VectorTypeHelper<unsigned short> { typedef __m128i Type; };
    template<> struct VectorTypeHelper<         int  > { typedef __m128i Type; };
    template<> struct VectorTypeHelper<unsigned int  > { typedef __m128i Type; };
    template<> struct VectorTypeHelper<         long > { typedef __m128i Type; };
    template<> struct VectorTypeHelper<unsigned long > { typedef __m128i Type; };
    template<> struct VectorTypeHelper<         long long> { typedef __m128i Type; };
    template<> struct VectorTypeHelper<unsigned long long> { typedef __m128i Type; };
#endif
    template<> struct VectorTypeHelper<         float> { typedef __m256  Type; };
    template<> struct VectorTypeHelper<        double> { typedef __m256d Type; };

    template<typename T> struct SseVectorType;
    template<> struct SseVectorType<__m256 > { typedef __m128  Type; };
    template<> struct SseVectorType<__m256i> { typedef __m128i Type; };
    template<> struct SseVectorType<__m256d> { typedef __m128d Type; };
    template<> struct SseVectorType<__m128 > { typedef __m128  Type; };
    template<> struct SseVectorType<__m128i> { typedef __m128i Type; };
    template<> struct SseVectorType<__m128d> { typedef __m128d Type; };

    template <typename T>
    using IntegerVectorType =
        typename std::conditional<sizeof(T) == 16, __m128i, __m256i>::type;
    template <typename T>
    using DoubleVectorType =
        typename std::conditional<sizeof(T) == 16, __m128d, __m256d>::type;
    template <typename T>
    using FloatVectorType =
        typename std::conditional<sizeof(T) == 16, __m128, __m256>::type;

    template<typename T> struct VectorHelper {};
    template<typename T> struct GatherHelper;
    template<typename T> struct ScatterHelper;

    template<typename T> struct HasVectorDivisionHelper { enum { Value = 1 }; };
    template<typename T> struct VectorHelperSize;

    template <typename V>
    class
#ifndef VC_ICC
        alignas(alignof(V))
#endif
        VectorAlignedBaseT
    {
    public:
        FREE_STORE_OPERATORS_ALIGNED(alignof(V))
    };
}  // namespace AVX(2)
}  // namespace Vc

#include "undomacros.h"

#endif // VC_AVX_INTRINSICS_H
