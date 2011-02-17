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

#ifndef AVX_INTRINSICS_H
#define AVX_INTRINSICS_H

// AVX
#include <immintrin.h>

#include "const.h"
#include "macros.h"
#include <cstdlib>

namespace Vc
{
namespace AVX
{
    static inline __m256i _mm256_setallone() CONST;
    static inline __m256i _mm256_setallone_si256() CONST;
    static inline __m256d _mm256_setallone_pd() CONST;
    static inline __m256  _mm256_setallone_ps() CONST;
    static inline __m256i _mm256_setone_epi8 () CONST;
    static inline __m256i _mm256_setone_epu8 () CONST;
    static inline __m256i _mm256_setone_epi16() CONST;
    static inline __m256i _mm256_setone_epu16() CONST;
    static inline __m256i _mm256_setone_epi32() CONST;
    static inline __m256i _mm256_setone_epu32() CONST;
    static inline __m256  _mm256_setone_ps() CONST;
    static inline __m256d _mm256_setone_pd() CONST;
    static inline __m256d _mm256_setabsmask_pd() CONST;
    static inline __m256  _mm256_setabsmask_ps() CONST;
    static inline __m256d _mm256_setsignmask_pd() CONST;
    static inline __m256  _mm256_setsignmask_ps() CONST;

    //X         static inline __m256i _mm256_setmin_epi8 () CONST;
    static inline __m256i _mm256_setmin_epi16() CONST;
    static inline __m256i _mm256_setmin_epi32() CONST;

    //X         static inline __m256i _mm256_cmplt_epu8 (__m256i a, __m256i b) CONST;
    //X         static inline __m256i _mm256_cmpgt_epu8 (__m256i a, __m256i b) CONST;
    static inline __m256i _mm256_cmplt_epu16(__m256i a, __m256i b) CONST;
    static inline __m256i _mm256_cmpgt_epu16(__m256i a, __m256i b) CONST;
    static inline __m256i _mm256_cmplt_epu32(__m256i a, __m256i b) CONST;
    static inline __m256i _mm256_cmpgt_epu32(__m256i a, __m256i b) CONST;

#if defined(__GNUC__) && !defined(NVALGRIND)
    static inline __m256i _mm256_setallone() { __m256i r; __asm__("vcmpeqb %0,%0":"=x"(r)); return r; }
#else
    static inline __m256i _mm256_setallone() { __m256i r = _mm256_setzero_si256(); return _mm256_cmpeq_epi8(r, r); }
#endif
    static inline __m256i _mm256_setallone_si256() { return _mm256_setallone(); }
    static inline __m256d _mm256_setallone_pd() { return _mm256_castsi256_pd(_mm256_setallone()); }
    static inline __m256  _mm256_setallone_ps() { return _mm256_castsi256_ps(_mm256_setallone()); }

    static inline __m256i _mm256_setone_epi8 ()  { return _mm256_set1_epi8(1); }
    static inline __m256i _mm256_setone_epu8 ()  { return _mm256_setone_epi8(); }
    static inline __m256i _mm256_setone_epi16()  { return _mm256_load_si256(reinterpret_cast<const __m256i *>(c_general::one16)); }
    static inline __m256i _mm256_setone_epu16()  { return _mm256_setone_epi16(); }
    static inline __m256i _mm256_setone_epi32()  { return _mm256_load_si256(reinterpret_cast<const __m256i *>(c_general::one32)); }
    static inline __m256i _mm256_setone_epu32()  { return _mm256_setone_epi32(); }

    static inline __m256  _mm256_setone_ps()     { return _mm256_load_ps(c_general::oneFloat); }
    static inline __m256d _mm256_setone_pd()     { return _mm256_load_pd(c_general::oneDouble); }

    static inline __m256d _mm256_setabsmask_pd() { return _mm256_load_pd(reinterpret_cast<const double *>(c_general::absMaskDouble)); }
    static inline __m256  _mm256_setabsmask_ps() { return _mm256_load_ps(reinterpret_cast<const float *>(c_general::absMaskFloat)); }
    static inline __m256d _mm256_setsignmask_pd(){ return _mm256_load_pd(reinterpret_cast<const double *>(c_general::signMaskDouble)); }
    static inline __m256  _mm256_setsignmask_ps(){ return _mm256_load_ps(reinterpret_cast<const float *>(c_general::signMaskFloat)); }

    //X         static inline __m256i _mm256_setmin_epi8 () { return _mm256_slli_epi8 (_mm256_setallone_si256(),  7); }
    static inline __m256i _mm256_setmin_epi16() { return _mm256_load_si256(reinterpret_cast<const __m256i *>(c_general::minShort)); }
    static inline __m256i _mm256_setmin_epi32() { return _mm256_load_si256(reinterpret_cast<const __m256i *>(c_general::signMaskFloat)); }

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
#define AVX_TO_SSE_1i_si128_si256(name) \
    static inline __m256i INTRINSIC CONST _mm256_##name##_si256(__m256i a0, const int i) { \
        __m128i a1 = _mm256_extractf128_si256(a0, 1); \
        __m128i r0 = _mm_##name##_si128(_mm256_castsi256_si128(a0), i); \
        __m128i r1 = _mm_##name##_si128(a1, i); \
        return _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1); \
    }

    AVX_TO_SSE_1i_si128_si256(srli)
    AVX_TO_SSE_1i_si128_si256(slli)
    AVX_TO_SSE_2_si128_si256(and)
    AVX_TO_SSE_2_si128_si256(andnot)
    AVX_TO_SSE_2_si128_si256(or)
    AVX_TO_SSE_2_si128_si256(xor)

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
    int _mm256_movemask_epi8(__m256i a0) CONST;
    int _mm256_movemask_epi8(__m256i a0)
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
    __m256i _mm256_blend_epi16(__m256i a0, __m256i b0, const int m) CONST;
    __m256i _mm256_blend_epi16(__m256i a0, __m256i b0, const int m) {
        __m128i a1 = _mm256_extractf128_si256(a0, 1);
        __m128i b1 = _mm256_extractf128_si256(b0, 1);
        __m128i r0 = _mm_blend_epi16(_mm256_castsi256_si128(a0), _mm256_castsi256_si128(b0), m & 0xff);
        __m128i r1 = _mm_blend_epi16(a1, b1, m >> 8);
        return _mm256_insertf128_si256(_mm256_castsi128_si256(r0), r1, 1);
    }
    __m256i _mm256_blendv_epi8(__m256i a0, __m256i b0, __m256i m0) CONST;
    __m256i _mm256_blendv_epi8(__m256i a0, __m256i b0, __m256i m0) {
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
    AVX_TO_SSE_1(minpos_epu16)
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
    static inline __m256i _mm256_cmplt_epu16(__m256i a, __m256i b) { return _mm256_cmplt_epi16(
            _mm256_xor_si256(a, _mm256_setmin_epi16()), _mm256_xor_si256(b, _mm256_setmin_epi16())); }
    static inline __m256i _mm256_cmpgt_epu16(__m256i a, __m256i b) { return _mm256_cmpgt_epi16(
            _mm256_xor_si256(a, _mm256_setmin_epi16()), _mm256_xor_si256(b, _mm256_setmin_epi16())); }
    static inline __m256i _mm256_cmplt_epu32(__m256i a, __m256i b) { return _mm256_cmplt_epi32(
            _mm256_xor_si256(a, _mm256_setmin_epi32()), _mm256_xor_si256(b, _mm256_setmin_epi32())); }
    static inline __m256i _mm256_cmpgt_epu32(__m256i a, __m256i b) { return _mm256_cmpgt_epi32(
            _mm256_xor_si256(a, _mm256_setmin_epi32()), _mm256_xor_si256(b, _mm256_setmin_epi32())); }

} // namespace AVX
} // namespace Vc

#endif // AVX_INTRINSICS_H
