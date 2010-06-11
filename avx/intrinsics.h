/*  This file is part of the Vc library.

    Copyright (C) 2009-2010 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Leavxr General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Leavxr General Public License for more details.

    You should have received a copy of the GNU Leavxr General Public
    License along with Vc.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifndef AVX_INTRINSICS_H
#define AVX_INTRINSICS_H

// MMX
#include <mmintrin.h>
// AVX
#include <xmmintrin.h>
// AVX2
#include <emmintrin.h>

#if defined(__GNUC__) && !defined(VC_IMPL_AVX2)
#error "AVX Vector class needs at least AVX2"
#endif

#include "const.h"
#include "macros.h"
#include <cstdlib>

namespace Vc
{
namespace AVX
{
    static inline __m256i _mm256_setallone() CONST;
    static inline __m256i _mm256_setallone_si128() CONST;
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

    // not overriding _mm256_set1_epi8 because this one should only be used for non-constants
    static inline __m256i set1_epi8(int a) CONST;

    //X         static inline __m256i _mm256_cmplt_epu8 (__m256i a, __m256i b) CONST;
    //X         static inline __m256i _mm256_cmpgt_epu8 (__m256i a, __m256i b) CONST;
    static inline __m256i _mm256_cmplt_epu16(__m256i a, __m256i b) CONST;
    static inline __m256i _mm256_cmpgt_epu16(__m256i a, __m256i b) CONST;
    static inline __m256i _mm256_cmplt_epu32(__m256i a, __m256i b) CONST;
    static inline __m256i _mm256_cmpgt_epu32(__m256i a, __m256i b) CONST;

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 6 && !defined(VC_DONT_FIX_AVX_SHIFT)
    static inline __m256i _mm256_sll_epi16(__m256i a, __m256i count) CONST;
    static inline __m256i _mm256_sll_epi16(__m256i a, __m256i count) { __asm__("psllw %1,%0" : "+x"(a) : "x"(count)); return a; }
    static inline __m256i _mm256_sll_epi32(__m256i a, __m256i count) CONST;
    static inline __m256i _mm256_sll_epi32(__m256i a, __m256i count) { __asm__("pslld %1,%0" : "+x"(a) : "x"(count)); return a; }
    static inline __m256i _mm256_sll_epi64(__m256i a, __m256i count) CONST;
    static inline __m256i _mm256_sll_epi64(__m256i a, __m256i count) { __asm__("psllq %1,%0" : "+x"(a) : "x"(count)); return a; }
    static inline __m256i _mm256_srl_epi16(__m256i a, __m256i count) CONST;
    static inline __m256i _mm256_srl_epi16(__m256i a, __m256i count) { __asm__("psrlw %1,%0" : "+x"(a) : "x"(count)); return a; }
    static inline __m256i _mm256_srl_epi32(__m256i a, __m256i count) CONST;
    static inline __m256i _mm256_srl_epi32(__m256i a, __m256i count) { __asm__("psrld %1,%0" : "+x"(a) : "x"(count)); return a; }
    static inline __m256i _mm256_srl_epi64(__m256i a, __m256i count) CONST;
    static inline __m256i _mm256_srl_epi64(__m256i a, __m256i count) { __asm__("psrlq %1,%0" : "+x"(a) : "x"(count)); return a; }
#endif

#if defined(__GNUC__) && !defined(NVALGRIND)
    static inline __m256i _mm256_setallone() { __m256i r; __asm__("pcmpeqb %0,%0":"=x"(r)); return r; }
#else
    static inline __m256i _mm256_setallone() { __m256i r = _mm256_setzero_si128(); return _mm256_cmpeq_epi8(r, r); }
#endif
    static inline __m256i _mm256_setallone_si128() { return _mm256_setallone(); }
    static inline __m256d _mm256_setallone_pd() { return _mm256_castsi128_pd(_mm256_setallone()); }
    static inline __m256  _mm256_setallone_ps() { return _mm256_castsi128_ps(_mm256_setallone()); }

    static inline __m256i _mm256_setone_epi8 ()  { return _mm256_set1_epi8(1); }
    static inline __m256i _mm256_setone_epu8 ()  { return _mm256_setone_epi8(); }
    static inline __m256i _mm256_setone_epi16()  { return _mm256_load_si128(reinterpret_cast<const __m256i *>(c_general::one16)); }
    static inline __m256i _mm256_setone_epu16()  { return _mm256_setone_epi16(); }
    static inline __m256i _mm256_setone_epi32()  { return _mm256_load_si128(reinterpret_cast<const __m256i *>(c_general::one32)); }
    static inline __m256i _mm256_setone_epu32()  { return _mm256_setone_epi32(); }

    static inline __m256  _mm256_setone_ps()     { return _mm256_load_ps(c_general::oneFloat); }
    static inline __m256d _mm256_setone_pd()     { return _mm256_load_pd(c_general::oneDouble); }

    static inline __m256d _mm256_setabsmask_pd() { return _mm256_load_pd(reinterpret_cast<const double *>(c_general::absMaskDouble)); }
    static inline __m256  _mm256_setabsmask_ps() { return _mm256_load_ps(reinterpret_cast<const float *>(c_general::absMaskFloat)); }
    static inline __m256d _mm256_setsignmask_pd(){ return _mm256_load_pd(reinterpret_cast<const double *>(c_general::signMaskDouble)); }
    static inline __m256  _mm256_setsignmask_ps(){ return _mm256_load_ps(reinterpret_cast<const float *>(c_general::signMaskFloat)); }

    //X         static inline __m256i _mm256_setmin_epi8 () { return _mm256_slli_epi8 (_mm256_setallone_si128(),  7); }
    static inline __m256i _mm256_setmin_epi16() { return _mm256_load_si128(reinterpret_cast<const __m256i *>(c_general::minShort)); }
    static inline __m256i _mm256_setmin_epi32() { return _mm256_load_si128(reinterpret_cast<const __m256i *>(c_general::signMaskFloat)); }

    //X         static inline __m256i _mm256_cmplt_epu8 (__m256i a, __m256i b) { return _mm256_cmplt_epi8 (
    //X                 _mm256_xor_si128(a, _mm256_setmin_epi8 ()), _mm256_xor_si128(b, _mm256_setmin_epi8 ())); }
    //X         static inline __m256i _mm256_cmpgt_epu8 (__m256i a, __m256i b) { return _mm256_cmpgt_epi8 (
    //X                 _mm256_xor_si128(a, _mm256_setmin_epi8 ()), _mm256_xor_si128(b, _mm256_setmin_epi8 ())); }
    static inline __m256i _mm256_cmplt_epu16(__m256i a, __m256i b) { return _mm256_cmplt_epi16(
            _mm256_xor_si128(a, _mm256_setmin_epi16()), _mm256_xor_si128(b, _mm256_setmin_epi16())); }
    static inline __m256i _mm256_cmpgt_epu16(__m256i a, __m256i b) { return _mm256_cmpgt_epi16(
            _mm256_xor_si128(a, _mm256_setmin_epi16()), _mm256_xor_si128(b, _mm256_setmin_epi16())); }
    static inline __m256i _mm256_cmplt_epu32(__m256i a, __m256i b) { return _mm256_cmplt_epi32(
            _mm256_xor_si128(a, _mm256_setmin_epi32()), _mm256_xor_si128(b, _mm256_setmin_epi32())); }
    static inline __m256i _mm256_cmpgt_epu32(__m256i a, __m256i b) { return _mm256_cmpgt_epi32(
            _mm256_xor_si128(a, _mm256_setmin_epi32()), _mm256_xor_si128(b, _mm256_setmin_epi32())); }
} // namespace AVX
} // namespace Vc

// AVX3
#ifdef VC_IMPL_AVX3
#include <pmmintrin.h>
#endif
// SAVX3
#ifdef VC_IMPL_SAVX3
#include <tmmintrin.h>
namespace Vc
{
namespace AVX
{

    static inline __m256i set1_epi8(int a) {
#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 5
        return _mm256_shuffle_epi8(_mm256_cvtsi32_si128(a), _mm256_setzero_si128());
#else
        // GCC 4.5 nows about the pshufb improvement
        return _mm256_set1_epi8(a);
#endif
    }

} // namespace AVX
} // namespace Vc
#else
namespace Vc
{
namespace AVX
{
    static inline __m256i _mm256_abs_epi8 (__m256i a) CONST;
    static inline __m256i _mm256_abs_epi16(__m256i a) CONST;
    static inline __m256i _mm256_abs_epi32(__m256i a) CONST;
    static inline __m256i _mm256_alignr_epi8(__m256i a, __m256i b, const int s) CONST;

    static inline __m256i _mm256_abs_epi8 (__m256i a) {
        __m256i negative = _mm256_cmplt_epi8 (a, _mm256_setzero_si128());
        return _mm256_add_epi8 (_mm256_xor_si128(a, negative), _mm256_and_si128(negative,  _mm256_setone_epi8()));
    }
    // positive value:
    //   negative == 0
    //   a unchanged after xor
    //   0 >> 31 -> 0
    //   a + 0 -> a
    // negative value:
    //   negative == -1
    //   a xor -1 -> -a - 1
    //   -1 >> 31 -> 1
    //   -a - 1 + 1 -> -a
    static inline __m256i _mm256_abs_epi16(__m256i a) {
        __m256i negative = _mm256_cmplt_epi16(a, _mm256_setzero_si128());
        return _mm256_add_epi16(_mm256_xor_si128(a, negative), _mm256_srli_epi16(negative, 15));
    }
    static inline __m256i _mm256_abs_epi32(__m256i a) {
        __m256i negative = _mm256_cmplt_epi32(a, _mm256_setzero_si128());
        return _mm256_add_epi32(_mm256_xor_si128(a, negative), _mm256_srli_epi32(negative, 31));
    }
    static inline __m256i set1_epi8(int a) {
        return _mm256_set1_epi8(a);
    }
    static inline __m256i _mm256_alignr_epi8(__m256i a, __m256i b, const int s) {
        switch (s) {
            case  0: return b;
            case  1: return _mm256_or_si128(_mm256_slli_si128(a, 15), _mm256_srli_si128(b,  1));
            case  2: return _mm256_or_si128(_mm256_slli_si128(a, 14), _mm256_srli_si128(b,  2));
            case  3: return _mm256_or_si128(_mm256_slli_si128(a, 13), _mm256_srli_si128(b,  3));
            case  4: return _mm256_or_si128(_mm256_slli_si128(a, 12), _mm256_srli_si128(b,  4));
            case  5: return _mm256_or_si128(_mm256_slli_si128(a, 11), _mm256_srli_si128(b,  5));
            case  6: return _mm256_or_si128(_mm256_slli_si128(a, 10), _mm256_srli_si128(b,  6));
            case  7: return _mm256_or_si128(_mm256_slli_si128(a,  9), _mm256_srli_si128(b,  7));
            case  8: return _mm256_or_si128(_mm256_slli_si128(a,  8), _mm256_srli_si128(b,  8));
            case  9: return _mm256_or_si128(_mm256_slli_si128(a,  7), _mm256_srli_si128(b,  9));
            case 10: return _mm256_or_si128(_mm256_slli_si128(a,  6), _mm256_srli_si128(b, 10));
            case 11: return _mm256_or_si128(_mm256_slli_si128(a,  5), _mm256_srli_si128(b, 11));
            case 12: return _mm256_or_si128(_mm256_slli_si128(a,  4), _mm256_srli_si128(b, 12));
            case 13: return _mm256_or_si128(_mm256_slli_si128(a,  3), _mm256_srli_si128(b, 13));
            case 14: return _mm256_or_si128(_mm256_slli_si128(a,  2), _mm256_srli_si128(b, 14));
            case 15: return _mm256_or_si128(_mm256_slli_si128(a,  1), _mm256_srli_si128(b, 15));
            case 16: return a;
            case 17: return _mm256_srli_si128(a,  1);
            case 18: return _mm256_srli_si128(a,  2);
            case 19: return _mm256_srli_si128(a,  3);
            case 20: return _mm256_srli_si128(a,  4);
            case 21: return _mm256_srli_si128(a,  5);
            case 22: return _mm256_srli_si128(a,  6);
            case 23: return _mm256_srli_si128(a,  7);
            case 24: return _mm256_srli_si128(a,  8);
            case 25: return _mm256_srli_si128(a,  9);
            case 26: return _mm256_srli_si128(a, 10);
            case 27: return _mm256_srli_si128(a, 11);
            case 28: return _mm256_srli_si128(a, 12);
            case 29: return _mm256_srli_si128(a, 13);
            case 30: return _mm256_srli_si128(a, 14);
            case 31: return _mm256_srli_si128(a, 15);
        }
        return _mm256_setzero_si128();
    }

} // namespace AVX
} // namespace Vc

#endif

// AVX4.1
#ifdef VC_IMPL_AVX4_1
#include <smmintrin.h>
#else
namespace Vc
{
namespace AVX
{
    static inline __m256d _mm256_blendv_pd(__m256d a, __m256d b, __m256d c) CONST ALWAYS_INLINE;
    static inline __m256  _mm256_blendv_ps(__m256  a, __m256  b, __m256  c) CONST ALWAYS_INLINE;
    static inline __m256i _mm256_blendv_epi8(__m256i a, __m256i b, __m256i c) CONST ALWAYS_INLINE;
    static inline __m256d _mm256_blend_pd(__m256d a, __m256d b, const int mask) CONST ALWAYS_INLINE;
    static inline __m256  _mm256_blend_ps(__m256  a, __m256  b, const int mask) CONST ALWAYS_INLINE;
    static inline __m256i _mm256_blend_epi16(__m256i a, __m256i b, const int mask) CONST ALWAYS_INLINE;
    static inline __m256i _mm256_max_epi8 (__m256i a, __m256i b) CONST;
    static inline __m256i _mm256_max_epi32(__m256i a, __m256i b) CONST;
    //static inline __m256i _mm256_max_epu8 (__m256i a, __m256i b) CONST;
    static inline __m256i _mm256_max_epu16(__m256i a, __m256i b) CONST;
    static inline __m256i _mm256_max_epu32(__m256i a, __m256i b) CONST;
    //static inline __m256i _mm256_min_epu8 (__m256i a, __m256i b) CONST;
    static inline __m256i _mm256_min_epu16(__m256i a, __m256i b) CONST;
    static inline __m256i _mm256_min_epu32(__m256i a, __m256i b) CONST;
    static inline __m256i _mm256_min_epi8 (__m256i a, __m256i b) CONST;
    static inline __m256i _mm256_min_epi32(__m256i a, __m256i b) CONST;

    static inline __m256d _mm256_blendv_pd(__m256d a, __m256d b, __m256d c) {
        return _mm256_or_pd(_mm256_andnot_pd(c, a), _mm256_and_pd(c, b));
    }
    static inline __m256  _mm256_blendv_ps(__m256  a, __m256  b, __m256  c) {
        return _mm256_or_ps(_mm256_andnot_ps(c, a), _mm256_and_ps(c, b));
    }
    static inline __m256i _mm256_blendv_epi8(__m256i a, __m256i b, __m256i c) {
        return _mm256_or_si128(_mm256_andnot_si128(c, a), _mm256_and_si128(c, b));
    }

    // only use the following blend functions with immediates as mask and, of course, compiling
    // with optimization
    static inline __m256d _mm256_blend_pd(__m256d a, __m256d b, const int mask) {
        __m256i c;
        switch (mask) {
        case 0x0:
            return a;
        case 0x1:
            c = _mm256_srli_si128(_mm256_setallone_si128(), 8);
            break;
        case 0x2:
            c = _mm256_slli_si128(_mm256_setallone_si128(), 8);
            break;
        case 0x3:
            return b;
        }
        __m256d _c = _mm256_castsi128_pd(c);
        return _mm256_or_pd(_mm256_andnot_pd(_c, a), _mm256_and_pd(_c, b));
    }
    static inline __m256  _mm256_blend_ps(__m256  a, __m256  b, const int mask) {
        __m256i c;
        switch (mask) {
        case 0x0:
            return a;
        case 0x1:
            c = _mm256_srli_si128(_mm256_setallone_si128(), 12);
            break;
        case 0x2:
            c = _mm256_slli_si128(_mm256_srli_si128(_mm256_setallone_si128(), 12), 4);
            break;
        case 0x3:
            c = _mm256_srli_si128(_mm256_setallone_si128(), 8);
            break;
        case 0x4:
            c = _mm256_slli_si128(_mm256_srli_si128(_mm256_setallone_si128(), 12), 8);
            break;
        case 0x5:
            c = _mm256_set_epi32(0, -1, 0, -1);
            break;
        case 0x6:
            c = _mm256_slli_si128(_mm256_srli_si128(_mm256_setallone_si128(), 8), 4);
            break;
        case 0x7:
            c = _mm256_srli_si128(_mm256_setallone_si128(), 4);
            break;
        case 0x8:
            c = _mm256_slli_si128(_mm256_setallone_si128(), 12);
            break;
        case 0x9:
            c = _mm256_set_epi32(-1, 0, 0, -1);
            break;
        case 0xa:
            c = _mm256_set_epi32(-1, 0, -1, 0);
            break;
        case 0xb:
            c = _mm256_set_epi32(-1, 0, -1, -1);
            break;
        case 0xc:
            c = _mm256_slli_si128(_mm256_setallone_si128(), 8);
            break;
        case 0xd:
            c = _mm256_set_epi32(-1, -1, 0, -1);
            break;
        case 0xe:
            c = _mm256_slli_si128(_mm256_setallone_si128(), 4);
            break;
        case 0xf:
            return b;
        default: // may not happen
            abort();
            c = _mm256_setzero_si128();
            break;
        }
        __m256 _c = _mm256_castsi128_ps(c);
        return _mm256_or_ps(_mm256_andnot_ps(_c, a), _mm256_and_ps(_c, b));
    }
    static inline __m256i _mm256_blend_epi16(__m256i a, __m256i b, const int mask) {
        __m256i c;
        switch (mask) {
        case 0x00:
            return a;
        case 0x01:
            c = _mm256_srli_si128(_mm256_setallone_si128(), 14);
            break;
        case 0x03:
            c = _mm256_srli_si128(_mm256_setallone_si128(), 12);
            break;
        case 0x07:
            c = _mm256_srli_si128(_mm256_setallone_si128(), 10);
            break;
        case 0x0f:
            return _mm256_unpackhi_epi64(_mm256_slli_si128(b, 8), a);
        case 0x1f:
            c = _mm256_srli_si128(_mm256_setallone_si128(), 6);
            break;
        case 0x3f:
            c = _mm256_srli_si128(_mm256_setallone_si128(), 4);
            break;
        case 0x7f:
            c = _mm256_srli_si128(_mm256_setallone_si128(), 2);
            break;
        case 0x80:
            c = _mm256_slli_si128(_mm256_setallone_si128(), 14);
            break;
        case 0xc0:
            c = _mm256_slli_si128(_mm256_setallone_si128(), 12);
            break;
        case 0xe0:
            c = _mm256_slli_si128(_mm256_setallone_si128(), 10);
            break;
        case 0xf0:
            c = _mm256_slli_si128(_mm256_setallone_si128(), 8);
            break;
        case 0xf8:
            c = _mm256_slli_si128(_mm256_setallone_si128(), 6);
            break;
        case 0xfc:
            c = _mm256_slli_si128(_mm256_setallone_si128(), 4);
            break;
        case 0xfe:
            c = _mm256_slli_si128(_mm256_setallone_si128(), 2);
            break;
        case 0xff:
            return b;
        case 0xcc:
            return _mm256_unpacklo_epi32(_mm256_shuffle_epi32(a, _MM_SHUFFLE(2, 0, 2, 0)), _mm256_shuffle_epi32(b, _MM_SHUFFLE(3, 1, 3, 1)));
        case 0x33:
            return _mm256_unpacklo_epi32(_mm256_shuffle_epi32(b, _MM_SHUFFLE(2, 0, 2, 0)), _mm256_shuffle_epi32(a, _MM_SHUFFLE(3, 1, 3, 1)));
        default:
            const __m256i shift = _mm256_set_epi16(0x0100, 0x0200, 0x0400, 0x0800, 0x1000, 0x2000, 0x4000, -0x7fff);
            c = _mm256_srai_epi16(_mm256_mullo_epi16(_mm256_set1_epi16(mask), shift), 15);
            break;
        }
        return _mm256_or_si128(_mm256_andnot_si128(c, a), _mm256_and_si128(c, b));
    }

    static inline __m256i _mm256_max_epi8 (__m256i a, __m256i b) {
        return _mm256_blendv_epi8(b, a, _mm256_cmpgt_epi8 (a, b));
    }
    static inline __m256i _mm256_max_epi32(__m256i a, __m256i b) {
        return _mm256_blendv_epi8(b, a, _mm256_cmpgt_epi32(a, b));
    }
//X         static inline __m256i _mm256_max_epu8 (__m256i a, __m256i b) {
//X             return _mm256_blendv_epi8(b, a, _mm256_cmpgt_epu8 (a, b));
//X         }
    static inline __m256i _mm256_max_epu16(__m256i a, __m256i b) {
        return _mm256_blendv_epi8(b, a, _mm256_cmpgt_epu16(a, b));
    }
    static inline __m256i _mm256_max_epu32(__m256i a, __m256i b) {
        return _mm256_blendv_epi8(b, a, _mm256_cmpgt_epu32(a, b));
    }
//X         static inline __m256i _mm256_min_epu8 (__m256i a, __m256i b) {
//X             return _mm256_blendv_epi8(a, b, _mm256_cmpgt_epu8 (a, b));
//X         }
    static inline __m256i _mm256_min_epu16(__m256i a, __m256i b) {
        return _mm256_blendv_epi8(a, b, _mm256_cmpgt_epu16(a, b));
    }
    static inline __m256i _mm256_min_epu32(__m256i a, __m256i b) {
        return _mm256_blendv_epi8(a, b, _mm256_cmpgt_epi32(a, b));
    }
    static inline __m256i _mm256_min_epi8 (__m256i a, __m256i b) {
        return _mm256_blendv_epi8(a, b, _mm256_cmpgt_epi8 (a, b));
    }
    static inline __m256i _mm256_min_epi32(__m256i a, __m256i b) {
        return _mm256_blendv_epi8(a, b, _mm256_cmpgt_epi32(a, b));
    }
} // namespace AVX
} // namespace Vc
#endif

#endif // AVX_INTRINSICS_H
