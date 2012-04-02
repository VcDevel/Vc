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

#ifndef VC_AVX_INTRINSICS_H
#define VC_AVX_INTRINSICS_H

// AVX
#include <immintrin.h>

#if defined(VC_CLANG) && VC_CLANG < 0x30100
// _mm_permute_ps is broken: http://llvm.org/bugs/show_bug.cgi?id=12401
#undef _mm_permute_ps
#define _mm_permute_ps(A, C) __extension__ ({ \
  __m128 __A = (A); \
  (__m128)__builtin_shufflevector((__v4sf)__A, (__v4sf) _mm_setzero_ps(), \
                                   (C) & 0x3, ((C) & 0xc) >> 2, \
                                   ((C) & 0x30) >> 4, ((C) & 0xc0) >> 6); })
#endif

#include <Vc/global.h>
#include "const_data.h"
#include "macros.h"
#include <cstdlib>

namespace Vc
{
namespace AVX
{
#if defined(__GNUC__) && !defined(NVALGRIND)
    static inline __m128i CONST _mm_setallone() { __m128i r; __asm__("pcmpeqb %0,%0":"=x"(r)); return r; }
#else
    static inline __m128i CONST _mm_setallone() { __m128i r = _mm_setzero_si128(); return _mm_cmpeq_epi8(r, r); }
#endif
    static inline __m128i CONST _mm_setallone_si128() { return _mm_setallone(); }
    static inline __m128d CONST _mm_setallone_pd() { return _mm_castsi128_pd(_mm_setallone()); }
    static inline __m128  CONST _mm_setallone_ps() { return _mm_castsi128_ps(_mm_setallone()); }

    static inline __m128i CONST _mm_setone_epi8 ()  { return _mm_set1_epi8(1); }
    static inline __m128i CONST _mm_setone_epu8 ()  { return _mm_setone_epi8(); }
    static inline __m128i CONST _mm_setone_epi16()  { return _mm_castps_si128(_mm_broadcast_ss(reinterpret_cast<const float *>(c_general::one16))); }
    static inline __m128i CONST _mm_setone_epu16()  { return _mm_setone_epi16(); }
    static inline __m128i CONST _mm_setone_epi32()  { return _mm_castps_si128(_mm_broadcast_ss(reinterpret_cast<const float *>(&_IndexesFromZero32[1]))); }

#if defined(__GNUC__) && !defined(NVALGRIND)
    static inline __m256i CONST _mm256_setallone() { __m256i r; __asm__("vcmpps $8,%0,%0,%0":"=x"(r)); return r; }
#else
    static inline __m256i CONST _mm256_setallone() { __m256 r = _mm256_setzero_ps(); return _mm256_cmp_ps(r, r, _CMP_EQ_UQ); }
#endif
    static inline __m256i CONST _mm256_setallone_si256() { return _mm256_setallone(); }
    static inline __m256d CONST _mm256_setallone_pd() { return _mm256_castsi256_pd(_mm256_setallone()); }
    static inline __m256  CONST _mm256_setallone_ps() { return _mm256_castsi256_ps(_mm256_setallone()); }

    static inline __m256i CONST _mm256_setone_epi8 ()  { return _mm256_set1_epi8(1); }
    static inline __m256i CONST _mm256_setone_epu8 ()  { return _mm256_setone_epi8(); }
    static inline __m256i CONST _mm256_setone_epi16()  { return _mm256_castps_si256(_mm256_broadcast_ss(reinterpret_cast<const float *>(c_general::one16))); }
    static inline __m256i CONST _mm256_setone_epu16()  { return _mm256_setone_epi16(); }
    static inline __m256i CONST _mm256_setone_epi32()  { return _mm256_castps_si256(_mm256_broadcast_ss(reinterpret_cast<const float *>(&_IndexesFromZero32[1]))); }
    static inline __m256i CONST _mm256_setone_epu32()  { return _mm256_setone_epi32(); }

    static inline __m256  CONST _mm256_setone_ps()     { return _mm256_broadcast_ss(&c_general::oneFloat); }
    static inline __m256d CONST _mm256_setone_pd()     { return _mm256_broadcast_sd(&c_general::oneDouble); }

    static inline __m256d CONST _mm256_setabsmask_pd() { return _mm256_broadcast_sd(reinterpret_cast<const double *>(&c_general::absMaskFloat[0])); }
    static inline __m256  CONST _mm256_setabsmask_ps() { return _mm256_broadcast_ss(reinterpret_cast<const float *>(&c_general::absMaskFloat[1])); }
    static inline __m256d CONST _mm256_setsignmask_pd(){ return _mm256_broadcast_sd(reinterpret_cast<const double *>(&c_general::signMaskFloat[0])); }
    static inline __m256  CONST _mm256_setsignmask_ps(){ return _mm256_broadcast_ss(reinterpret_cast<const float *>(&c_general::signMaskFloat[1])); }

    static inline __m256  CONST _mm256_set2power31_ps()    { return _mm256_broadcast_ss(&c_general::_2power31); }
    static inline __m256i CONST _mm256_set2power31_epu32() { return _mm256_castps_si256(_mm256_broadcast_ss(reinterpret_cast<const float *>(&c_general::signMaskFloat[1]))); }

    //X         static inline __m256i CONST _mm256_setmin_epi8 () { return _mm256_slli_epi8 (_mm256_setallone_si256(),  7); }
    static inline __m128i CONST _mm_setmin_epi16() { return _mm_castps_si128(_mm_broadcast_ss(reinterpret_cast<const float *>(c_general::minShort))); }
    static inline __m128i CONST _mm_setmin_epi32() { return _mm_castps_si128(_mm_broadcast_ss(reinterpret_cast<const float *>(&c_general::signMaskFloat[1]))); }
    static inline __m256i CONST _mm256_setmin_epi16() { return _mm256_castps_si256(_mm256_broadcast_ss(reinterpret_cast<const float *>(c_general::minShort))); }
    static inline __m256i CONST _mm256_setmin_epi32() { return _mm256_castps_si256(_mm256_broadcast_ss(reinterpret_cast<const float *>(&c_general::signMaskFloat[1]))); }

    static inline unsigned char INTRINSIC CONST _mm_extract_epu8(__m128i x, const int i) { return _mm_extract_epi8(x, i); }
    static inline unsigned short INTRINSIC CONST _mm_extract_epu16(__m128i x, const int i) { return _mm_extract_epi16(x, i); }
    static inline unsigned int INTRINSIC CONST _mm_extract_epu32(__m128i x, const int i) { return _mm_extract_epi32(x, i); }

    /////////////////////// COMPARE OPS ///////////////////////
    static inline __m256d INTRINSIC CONST _mm256_cmpeq_pd   (__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_EQ_OQ); }
    static inline __m256d INTRINSIC CONST _mm256_cmpneq_pd  (__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_NEQ_UQ); }
    static inline __m256d INTRINSIC CONST _mm256_cmplt_pd   (__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_LT_OS); }
    static inline __m256d INTRINSIC CONST _mm256_cmpnlt_pd  (__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_NLT_US); }
    static inline __m256d INTRINSIC CONST _mm256_cmple_pd   (__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_LE_OS); }
    static inline __m256d INTRINSIC CONST _mm256_cmpnle_pd  (__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_NLE_US); }
    static inline __m256d INTRINSIC CONST _mm256_cmpord_pd  (__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_ORD_Q); }
    static inline __m256d INTRINSIC CONST _mm256_cmpunord_pd(__m256d a, __m256d b) { return _mm256_cmp_pd(a, b, _CMP_UNORD_Q); }

    static inline __m256  INTRINSIC CONST _mm256_cmpeq_ps   (__m256  a, __m256  b) { return _mm256_cmp_ps(a, b, _CMP_EQ_OQ); }
    static inline __m256  INTRINSIC CONST _mm256_cmpneq_ps  (__m256  a, __m256  b) { return _mm256_cmp_ps(a, b, _CMP_NEQ_UQ); }
    static inline __m256  INTRINSIC CONST _mm256_cmplt_ps   (__m256  a, __m256  b) { return _mm256_cmp_ps(a, b, _CMP_LT_OS); }
    static inline __m256  INTRINSIC CONST _mm256_cmpnlt_ps  (__m256  a, __m256  b) { return _mm256_cmp_ps(a, b, _CMP_NLT_US); }
    static inline __m256  INTRINSIC CONST _mm256_cmpge_ps   (__m256  a, __m256  b) { return _mm256_cmp_ps(a, b, _CMP_NLT_US); }
    static inline __m256  INTRINSIC CONST _mm256_cmple_ps   (__m256  a, __m256  b) { return _mm256_cmp_ps(a, b, _CMP_LE_OS); }
    static inline __m256  INTRINSIC CONST _mm256_cmpnle_ps  (__m256  a, __m256  b) { return _mm256_cmp_ps(a, b, _CMP_NLE_US); }
    static inline __m256  INTRINSIC CONST _mm256_cmpgt_ps   (__m256  a, __m256  b) { return _mm256_cmp_ps(a, b, _CMP_NLE_US); }
    static inline __m256  INTRINSIC CONST _mm256_cmpord_ps  (__m256  a, __m256  b) { return _mm256_cmp_ps(a, b, _CMP_ORD_Q); }
    static inline __m256  INTRINSIC CONST _mm256_cmpunord_ps(__m256  a, __m256  b) { return _mm256_cmp_ps(a, b, _CMP_UNORD_Q); }

    /////////////////////// INTEGER OPS ///////////////////////
#define AVX_TO_SSE_2(name) \
    static inline __m256i INTRINSIC CONST _mm256_##name(__m256i a0, __m256i b0) { \
        __m128i a1 = _mm256_extractf128_si256(a0, 1); \
        __m128i b1 = _mm256_extractf128_si256(b0, 1); \
        __m128i r0 = _mm_##name(_mm256_castsi256_si128(a0), _mm256_castsi256_si128(b0)); \
        __m128i r1 = _mm_##name(a1, b1); \
        return _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1); \
    }
#define AVX_TO_SSE_2_si128_si256(name) \
    static inline __m256i INTRINSIC CONST _mm256_##name##_si256(__m256i a0, __m256i b0) { \
        __m128i a1 = _mm256_extractf128_si256(a0, 1); \
        __m128i b1 = _mm256_extractf128_si256(b0, 1); \
        __m128i r0 = _mm_##name##_si128(_mm256_castsi256_si128(a0), _mm256_castsi256_si128(b0)); \
        __m128i r1 = _mm_##name##_si128(a1, b1); \
        return _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1); \
    }
#define AVX_TO_SSE_1(name) \
    static inline __m256i INTRINSIC CONST _mm256_##name(__m256i a0) { \
        __m128i a1 = _mm256_extractf128_si256(a0, 1); \
        __m128i r0 = _mm_##name(_mm256_castsi256_si128(a0)); \
        __m128i r1 = _mm_##name(a1); \
        return _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1); \
    }
#define AVX_TO_SSE_1i(name) \
    static inline __m256i INTRINSIC CONST _mm256_##name(__m256i a0, const int i) { \
        __m128i a1 = _mm256_extractf128_si256(a0, 1); \
        __m128i r0 = _mm_##name(_mm256_castsi256_si128(a0), i); \
        __m128i r1 = _mm_##name(a1, i); \
        return _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1); \
    }

#if defined(VC_CLANG) || (defined(VC_GCC) && !defined(__OPTIMIZE__))
#   define _mm256_srli_si256(a, i) \
        _mm256_insertf128_si256( \
                _mm256_castsi128_si256(_mm_srli_si128(_mm256_castsi256_si128((a)), i)), \
                _mm_srli_si128(_mm256_extractf128_si256((a), 1), i), 1);
#   define _mm256_slli_si256(a, i) \
        _mm256_insertf128_si256( \
                _mm256_castsi128_si256( _mm_slli_si128(_mm256_castsi256_si128((a)), i)), \
                _mm_slli_si128(_mm256_extractf128_si256((a), 1), i), 1);
#else
    static inline __m256i INTRINSIC CONST _mm256_srli_si256(__m256i a0, const int i) {
        const __m128i r0 = _mm_srli_si128(_mm256_castsi256_si128(a0), i);
        const __m128i r1 = _mm_srli_si128(_mm256_extractf128_si256(a0, 1), i);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
    }
    static inline __m256i INTRINSIC CONST _mm256_slli_si256(__m256i a0, const int i) {
        const __m128i r0 = _mm_slli_si128(_mm256_castsi256_si128(a0), i);
        const __m128i r1 = _mm_slli_si128(_mm256_extractf128_si256(a0, 1), i);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
    }
#endif

    static inline __m256i INTRINSIC CONST _mm256_and_si256(__m256i x, __m256i y) {
        return _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(x), _mm256_castsi256_ps(y)));
    }
    static inline __m256i INTRINSIC CONST _mm256_andnot_si256(__m256i x, __m256i y) {
        return _mm256_castps_si256(_mm256_andnot_ps(_mm256_castsi256_ps(x), _mm256_castsi256_ps(y)));
    }
    static inline __m256i INTRINSIC CONST _mm256_or_si256(__m256i x, __m256i y) {
        return _mm256_castps_si256(_mm256_or_ps(_mm256_castsi256_ps(x), _mm256_castsi256_ps(y)));
    }
    static inline __m256i INTRINSIC CONST _mm256_xor_si256(__m256i x, __m256i y) {
        return _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(x), _mm256_castsi256_ps(y)));
    }

    AVX_TO_SSE_2(packs_epi16)
    AVX_TO_SSE_2(packs_epi32)
    AVX_TO_SSE_2(packus_epi16)
    AVX_TO_SSE_2(unpackhi_epi8)
    AVX_TO_SSE_2(unpackhi_epi16)
    AVX_TO_SSE_2(unpackhi_epi32)
    AVX_TO_SSE_2(unpackhi_epi64)
    AVX_TO_SSE_2(unpacklo_epi8)
    AVX_TO_SSE_2(unpacklo_epi16)
    AVX_TO_SSE_2(unpacklo_epi32)
    AVX_TO_SSE_2(unpacklo_epi64)
    AVX_TO_SSE_2(add_epi8)
    AVX_TO_SSE_2(add_epi16)
    AVX_TO_SSE_2(add_epi32)
    AVX_TO_SSE_2(add_epi64)
    AVX_TO_SSE_2(adds_epi8)
    AVX_TO_SSE_2(adds_epi16)
    AVX_TO_SSE_2(adds_epu8)
    AVX_TO_SSE_2(adds_epu16)
    AVX_TO_SSE_2(sub_epi8)
    AVX_TO_SSE_2(sub_epi16)
    AVX_TO_SSE_2(sub_epi32)
    AVX_TO_SSE_2(sub_epi64)
    AVX_TO_SSE_2(subs_epi8)
    AVX_TO_SSE_2(subs_epi16)
    AVX_TO_SSE_2(subs_epu8)
    AVX_TO_SSE_2(subs_epu16)
    AVX_TO_SSE_2(madd_epi16)
    AVX_TO_SSE_2(mulhi_epi16)
    AVX_TO_SSE_2(mullo_epi16)
    AVX_TO_SSE_2(mul_epu32)
    AVX_TO_SSE_1i(slli_epi16)
    AVX_TO_SSE_1i(slli_epi32)
    AVX_TO_SSE_1i(slli_epi64)
    AVX_TO_SSE_1i(srai_epi16)
    AVX_TO_SSE_1i(srai_epi32)
    AVX_TO_SSE_1i(srli_epi16)
    AVX_TO_SSE_1i(srli_epi32)
    AVX_TO_SSE_1i(srli_epi64)
    AVX_TO_SSE_2(sll_epi16)
    AVX_TO_SSE_2(sll_epi32)
    AVX_TO_SSE_2(sll_epi64)
    AVX_TO_SSE_2(sra_epi16)
    AVX_TO_SSE_2(sra_epi32)
    AVX_TO_SSE_2(srl_epi16)
    AVX_TO_SSE_2(srl_epi32)
    AVX_TO_SSE_2(srl_epi64)
    AVX_TO_SSE_2(cmpeq_epi8)
    AVX_TO_SSE_2(cmpeq_epi16)
    AVX_TO_SSE_2(cmpeq_epi32)
    AVX_TO_SSE_2(cmplt_epi8)
    AVX_TO_SSE_2(cmplt_epi16)
    AVX_TO_SSE_2(cmplt_epi32)
    AVX_TO_SSE_2(cmpgt_epi8)
    AVX_TO_SSE_2(cmpgt_epi16)
    AVX_TO_SSE_2(cmpgt_epi32)
    AVX_TO_SSE_2(max_epi16)
    AVX_TO_SSE_2(max_epu8)
    AVX_TO_SSE_2(min_epi16)
    AVX_TO_SSE_2(min_epu8)
    inline int INTRINSIC CONST _mm256_movemask_epi8(__m256i a0)
    {
        __m128i a1 = _mm256_extractf128_si256(a0, 1);
        return (_mm_movemask_epi8(a1) << 16) | _mm_movemask_epi8(_mm256_castsi256_si128(a0));
    }
    AVX_TO_SSE_2(mulhi_epu16)
    // shufflehi_epi16
    // shufflelo_epi16 (__m128i __A, const int __mask)
    // shuffle_epi32 (__m128i __A, const int __mask)
    // maskmoveu_si128 (__m128i __A, __m128i __B, char *__C)
    AVX_TO_SSE_2(avg_epu8)
    AVX_TO_SSE_2(avg_epu16)
    AVX_TO_SSE_2(sad_epu8)
    // stream_si32 (int *__A, int __B)
    // stream_si128 (__m128i *__A, __m128i __B)
    // cvtsi32_si128 (int __A)
    // cvtsi64_si128 (long long __A)
    // cvtsi64x_si128 (long long __A)
    AVX_TO_SSE_2(hadd_epi16)
    AVX_TO_SSE_2(hadd_epi32)
    AVX_TO_SSE_2(hadds_epi16)
    AVX_TO_SSE_2(hsub_epi16)
    AVX_TO_SSE_2(hsub_epi32)
    AVX_TO_SSE_2(hsubs_epi16)
    AVX_TO_SSE_2(maddubs_epi16)
    AVX_TO_SSE_2(mulhrs_epi16)
    AVX_TO_SSE_2(shuffle_epi8)
    AVX_TO_SSE_2(sign_epi8)
    AVX_TO_SSE_2(sign_epi16)
    AVX_TO_SSE_2(sign_epi32)
    // alignr_epi8(__m128i __X, __m128i __Y, const int __N)
    AVX_TO_SSE_1(abs_epi8)
    AVX_TO_SSE_1(abs_epi16)
    AVX_TO_SSE_1(abs_epi32)
#if !defined(VC_CLANG) || (defined(VC_GCC) && defined(__OPTIMIZE__))
    __m256i inline INTRINSIC CONST _mm256_blend_epi16(__m256i a0, __m256i b0, const int m) {
        __m128i a1 = _mm256_extractf128_si256(a0, 1);
        __m128i b1 = _mm256_extractf128_si256(b0, 1);
        __m128i r0 = _mm_blend_epi16(_mm256_castsi256_si128(a0), _mm256_castsi256_si128(b0), m & 0xff);
        __m128i r1 = _mm_blend_epi16(a1, b1, m >> 8);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
    }
#else
#   define _mm256_blend_epi16(a0, b0, m) \
    _mm256_insertf128_si256( \
            _mm256_castsi128_si256( \
                _mm_blend_epi16( \
                    _mm256_castsi256_si128(a0), _mm256_castsi256_si128(b0), m & 0xff)), \
            _mm_blend_epi16(_mm256_extractf128_si256(a0, 1), _mm256_extractf128_si256(b0, 1), m >> 8);, 1)
#endif
    inline __m256i INTRINSIC CONST _mm256_blendv_epi8(__m256i a0, __m256i b0, __m256i m0) {
        __m128i a1 = _mm256_extractf128_si256(a0, 1);
        __m128i b1 = _mm256_extractf128_si256(b0, 1);
        __m128i m1 = _mm256_extractf128_si256(m0, 1);
        __m128i r0 = _mm_blendv_epi8(_mm256_castsi256_si128(a0), _mm256_castsi256_si128(b0), _mm256_castsi256_si128(m0));
        __m128i r1 = _mm_blendv_epi8(a1, b1, m1);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
    }
    AVX_TO_SSE_2(cmpeq_epi64)
    AVX_TO_SSE_2(min_epi8)
    AVX_TO_SSE_2(max_epi8)
    AVX_TO_SSE_2(min_epu16)
    AVX_TO_SSE_2(max_epu16)
    AVX_TO_SSE_2(min_epi32)
    AVX_TO_SSE_2(max_epi32)
    AVX_TO_SSE_2(min_epu32)
    AVX_TO_SSE_2(max_epu32)
    AVX_TO_SSE_2(mullo_epi32)
    AVX_TO_SSE_2(mul_epi32)
#if !defined(VC_CLANG) || VC_CLANG > 0x30100
    // clang is missing _mm_minpos_epu16 from smmintrin.h
    // http://llvm.org/bugs/show_bug.cgi?id=12399
    AVX_TO_SSE_1(minpos_epu16)
#endif
    AVX_TO_SSE_1(cvtepi8_epi32)
    AVX_TO_SSE_1(cvtepi16_epi32)
    AVX_TO_SSE_1(cvtepi8_epi64)
    AVX_TO_SSE_1(cvtepi32_epi64)
    AVX_TO_SSE_1(cvtepi16_epi64)
    AVX_TO_SSE_1(cvtepi8_epi16)
    AVX_TO_SSE_1(cvtepu8_epi32)
    AVX_TO_SSE_1(cvtepu16_epi32)
    AVX_TO_SSE_1(cvtepu8_epi64)
    AVX_TO_SSE_1(cvtepu32_epi64)
    AVX_TO_SSE_1(cvtepu16_epi64)
    AVX_TO_SSE_1(cvtepu8_epi16)
    AVX_TO_SSE_2(packus_epi32)
    // mpsadbw_epu8 (__m128i __X, __m128i __Y, const int __M)
    // stream_load_si128 (__m128i *__X)
    AVX_TO_SSE_2(cmpgt_epi64)

//X     static inline __m256i _mm256_cmplt_epu8 (__m256i a, __m256i b) { return _mm256_cmplt_epi8 (
//X             _mm256_xor_si256(a, _mm256_setmin_epi8 ()), _mm256_xor_si256(b, _mm256_setmin_epi8 ())); }
//X     static inline __m256i _mm256_cmpgt_epu8 (__m256i a, __m256i b) { return _mm256_cmpgt_epi8 (
//X             _mm256_xor_si256(a, _mm256_setmin_epi8 ()), _mm256_xor_si256(b, _mm256_setmin_epi8 ())); }
    static inline __m128i _mm_cmplt_epu16(__m128i a, __m128i b) {
        return _mm_cmplt_epi16(_mm_xor_si128(a, _mm_setmin_epi16()), _mm_xor_si128(b, _mm_setmin_epi16()));
    }
    static inline __m128i _mm_cmpgt_epu16(__m128i a, __m128i b) {
        return _mm_cmpgt_epi16(_mm_xor_si128(a, _mm_setmin_epi16()), _mm_xor_si128(b, _mm_setmin_epi16()));
    }
    static inline __m256i CONST _mm256_cmplt_epu32(__m256i a, __m256i b) {
        a = _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(_mm256_setmin_epi32())));
        b = _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(b), _mm256_castsi256_ps(_mm256_setmin_epi32())));
        return _mm256_insertf128_si256(_mm256_castsi128_si256(
                    _mm_cmplt_epi32(_mm256_castsi256_si128(a), _mm256_castsi256_si128(b))),
                _mm_cmplt_epi32(_mm256_extractf128_si256(a, 1), _mm256_extractf128_si256(b, 1)), 1);
    }
    static inline __m256i CONST _mm256_cmpgt_epu32(__m256i a, __m256i b) {
        a = _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(_mm256_setmin_epi32())));
        b = _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(b), _mm256_castsi256_ps(_mm256_setmin_epi32())));
        return _mm256_insertf128_si256(_mm256_castsi128_si256(
                    _mm_cmpgt_epi32(_mm256_castsi256_si128(a), _mm256_castsi256_si128(b))),
                _mm_cmpgt_epi32(_mm256_extractf128_si256(a, 1), _mm256_extractf128_si256(b, 1)), 1);
    }

#if defined(VC_GCC) && (VC_GCC > 0x40502 || (VC_GCC == 0x40502 && defined(__GNUC_UBUNTU_VERSION__) && __GNUC_UBUNTU_VERSION__ == 0xb0409) || (VC_GCC > 0x40405 && VC_GCC < 0x40500))
// GCC 4.6.0 / 4.5.3 / 4.4.6 switched to the broken interface as defined by ICC
// Ubuntu 11.04 ships a GCC 4.5.2 with the new interface
#define VC_MASKSTORE_MASK_TYPE_IS_INT 1
#endif
        static inline void INTRINSIC _mm256_maskstore(float *mem, const __m256 mask, const __m256 v) {
#ifdef VC_MASKSTORE_MASK_TYPE_IS_INT
            _mm256_maskstore_ps(mem, _mm256_castps_si256(mask), v);
#else
            _mm256_maskstore_ps(mem, mask, v);
#endif
        }
        static inline void INTRINSIC _mm256_maskstore(double *mem, const __m256d mask, const __m256d v) {
#ifdef VC_MASKSTORE_MASK_TYPE_IS_INT
            _mm256_maskstore_pd(mem, _mm256_castpd_si256(mask), v);
#else
            _mm256_maskstore_pd(mem, mask, v);
#endif
        }
        static inline void INTRINSIC _mm256_maskstore(int *mem, const __m256i mask, const __m256i v) {
#ifdef VC_MASKSTORE_MASK_TYPE_IS_INT
            _mm256_maskstore_ps(reinterpret_cast<float *>(mem), mask, _mm256_castsi256_ps(v));
#else
            _mm256_maskstore_ps(reinterpret_cast<float *>(mem), _mm256_castsi256_ps(mask), _mm256_castsi256_ps(v));
#endif
        }
        static inline void INTRINSIC _mm256_maskstore(unsigned int *mem, const __m256i mask, const __m256i v) {
            _mm256_maskstore(reinterpret_cast<int *>(mem), mask, v);
        }
} // namespace AVX
} // namespace Vc

#include "shuffle.h"

#endif // VC_AVX_INTRINSICS_H
