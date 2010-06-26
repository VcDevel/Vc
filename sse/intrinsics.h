/*  This file is part of the Vc library.

    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

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

#ifndef SSE_INTRINSICS_H
#define SSE_INTRINSICS_H

#if defined(_MSC_VER) && !defined(__midl)
// MSVC sucks. If you include intrin.h you get all SSE intrinsics
// declared. Something always includes intrin.h even if you don't
// do it explicitly. Therefore we try to be the first to include it
// but with __midl defined -- therefore not actually doing anything
#ifdef __INTRIN_H_
#error "intrin.h was already included, polluting the namespace. Please fix your code to include the Vc headers before anything that includes intrin.h. If you need any of the intrinsics from intrin.h declare the functions manually instead (you can copy them out of the intrin.h header."
#endif
#define __midl
#include <intrin.h>
#undef __midl
extern "C" {
#ifdef _WIN64
unsigned char _BitScanForward64(unsigned long* Index, unsigned __int64 Mask);
unsigned char _bittestandreset64(__int64 *a, __int64 b);
#pragma intrinsic(_BitScanForward64)
#pragma intrinsic(_bittestandreset64)
#endif
unsigned char _BitScanForward(unsigned long* Index, unsigned long Mask);
unsigned char _bittestandreset(long *a, long b);
#pragma intrinsic(_BitScanForward)
#pragma intrinsic(_bittestandreset)
}
#endif

// MMX
#include <mmintrin.h>
// SSE
#include <xmmintrin.h>
// SSE2
#include <emmintrin.h>

#if defined(__GNUC__) && !defined(VC_IMPL_SSE2)
#error "SSE Vector class needs at least SSE2"
#endif

#include "const.h"
#include "macros.h"
#include <cstdlib>

namespace Vc
{
namespace SSE
{
    static inline __m128i _mm_setallone() CONST;
    static inline __m128i _mm_setallone_si128() CONST;
    static inline __m128d _mm_setallone_pd() CONST;
    static inline __m128  _mm_setallone_ps() CONST;
    static inline __m128i _mm_setone_epi8 () CONST;
    static inline __m128i _mm_setone_epu8 () CONST;
    static inline __m128i _mm_setone_epi16() CONST;
    static inline __m128i _mm_setone_epu16() CONST;
    static inline __m128i _mm_setone_epi32() CONST;
    static inline __m128i _mm_setone_epu32() CONST;
    static inline __m128  _mm_setone_ps() CONST;
    static inline __m128d _mm_setone_pd() CONST;
    static inline __m128d _mm_setabsmask_pd() CONST;
    static inline __m128  _mm_setabsmask_ps() CONST;
    static inline __m128d _mm_setsignmask_pd() CONST;
    static inline __m128  _mm_setsignmask_ps() CONST;

    //X         static inline __m128i _mm_setmin_epi8 () CONST;
    static inline __m128i _mm_setmin_epi16() CONST;
    static inline __m128i _mm_setmin_epi32() CONST;

    // not overriding _mm_set1_epi8 because this one should only be used for non-constants
    static inline __m128i set1_epi8(int a) CONST;

    //X         static inline __m128i _mm_cmplt_epu8 (__m128i a, __m128i b) CONST;
    //X         static inline __m128i _mm_cmpgt_epu8 (__m128i a, __m128i b) CONST;
    static inline __m128i _mm_cmplt_epu16(__m128i a, __m128i b) CONST;
    static inline __m128i _mm_cmpgt_epu16(__m128i a, __m128i b) CONST;
    static inline __m128i _mm_cmplt_epu32(__m128i a, __m128i b) CONST;
    static inline __m128i _mm_cmpgt_epu32(__m128i a, __m128i b) CONST;

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 6 && !defined(VC_DONT_FIX_SSE_SHIFT)
    static inline __m128i _mm_sll_epi16(__m128i a, __m128i count) CONST;
    static inline __m128i _mm_sll_epi16(__m128i a, __m128i count) { __asm__("psllw %1,%0" : "+x"(a) : "x"(count)); return a; }
    static inline __m128i _mm_sll_epi32(__m128i a, __m128i count) CONST;
    static inline __m128i _mm_sll_epi32(__m128i a, __m128i count) { __asm__("pslld %1,%0" : "+x"(a) : "x"(count)); return a; }
    static inline __m128i _mm_sll_epi64(__m128i a, __m128i count) CONST;
    static inline __m128i _mm_sll_epi64(__m128i a, __m128i count) { __asm__("psllq %1,%0" : "+x"(a) : "x"(count)); return a; }
    static inline __m128i _mm_srl_epi16(__m128i a, __m128i count) CONST;
    static inline __m128i _mm_srl_epi16(__m128i a, __m128i count) { __asm__("psrlw %1,%0" : "+x"(a) : "x"(count)); return a; }
    static inline __m128i _mm_srl_epi32(__m128i a, __m128i count) CONST;
    static inline __m128i _mm_srl_epi32(__m128i a, __m128i count) { __asm__("psrld %1,%0" : "+x"(a) : "x"(count)); return a; }
    static inline __m128i _mm_srl_epi64(__m128i a, __m128i count) CONST;
    static inline __m128i _mm_srl_epi64(__m128i a, __m128i count) { __asm__("psrlq %1,%0" : "+x"(a) : "x"(count)); return a; }
#endif

#if defined(__GNUC__) && !defined(NVALGRIND)
    static inline __m128i _mm_setallone() { __m128i r; __asm__("pcmpeqb %0,%0":"=x"(r)); return r; }
#else
    static inline __m128i _mm_setallone() { __m128i r = _mm_setzero_si128(); return _mm_cmpeq_epi8(r, r); }
#endif
    static inline __m128i _mm_setallone_si128() { return _mm_setallone(); }
    static inline __m128d _mm_setallone_pd() { return _mm_castsi128_pd(_mm_setallone()); }
    static inline __m128  _mm_setallone_ps() { return _mm_castsi128_ps(_mm_setallone()); }

    static inline __m128i _mm_setone_epi8 ()  { return _mm_set1_epi8(1); }
    static inline __m128i _mm_setone_epu8 ()  { return _mm_setone_epi8(); }
    static inline __m128i _mm_setone_epi16()  { return _mm_load_si128(reinterpret_cast<const __m128i *>(c_general::one16)); }
    static inline __m128i _mm_setone_epu16()  { return _mm_setone_epi16(); }
    static inline __m128i _mm_setone_epi32()  { return _mm_load_si128(reinterpret_cast<const __m128i *>(c_general::one32)); }
    static inline __m128i _mm_setone_epu32()  { return _mm_setone_epi32(); }

    static inline __m128  _mm_setone_ps()     { return _mm_load_ps(c_general::oneFloat); }
    static inline __m128d _mm_setone_pd()     { return _mm_load_pd(c_general::oneDouble); }

    static inline __m128d _mm_setabsmask_pd() { return _mm_load_pd(reinterpret_cast<const double *>(c_general::absMaskDouble)); }
    static inline __m128  _mm_setabsmask_ps() { return _mm_load_ps(reinterpret_cast<const float *>(c_general::absMaskFloat)); }
    static inline __m128d _mm_setsignmask_pd(){ return _mm_load_pd(reinterpret_cast<const double *>(c_general::signMaskDouble)); }
    static inline __m128  _mm_setsignmask_ps(){ return _mm_load_ps(reinterpret_cast<const float *>(c_general::signMaskFloat)); }

    //X         static inline __m128i _mm_setmin_epi8 () { return _mm_slli_epi8 (_mm_setallone_si128(),  7); }
    static inline __m128i _mm_setmin_epi16() { return _mm_load_si128(reinterpret_cast<const __m128i *>(c_general::minShort)); }
    static inline __m128i _mm_setmin_epi32() { return _mm_load_si128(reinterpret_cast<const __m128i *>(c_general::signMaskFloat)); }

    //X         static inline __m128i _mm_cmplt_epu8 (__m128i a, __m128i b) { return _mm_cmplt_epi8 (
    //X                 _mm_xor_si128(a, _mm_setmin_epi8 ()), _mm_xor_si128(b, _mm_setmin_epi8 ())); }
    //X         static inline __m128i _mm_cmpgt_epu8 (__m128i a, __m128i b) { return _mm_cmpgt_epi8 (
    //X                 _mm_xor_si128(a, _mm_setmin_epi8 ()), _mm_xor_si128(b, _mm_setmin_epi8 ())); }
    static inline __m128i _mm_cmplt_epu16(__m128i a, __m128i b) { return _mm_cmplt_epi16(
            _mm_xor_si128(a, _mm_setmin_epi16()), _mm_xor_si128(b, _mm_setmin_epi16())); }
    static inline __m128i _mm_cmpgt_epu16(__m128i a, __m128i b) { return _mm_cmpgt_epi16(
            _mm_xor_si128(a, _mm_setmin_epi16()), _mm_xor_si128(b, _mm_setmin_epi16())); }
    static inline __m128i _mm_cmplt_epu32(__m128i a, __m128i b) { return _mm_cmplt_epi32(
            _mm_xor_si128(a, _mm_setmin_epi32()), _mm_xor_si128(b, _mm_setmin_epi32())); }
    static inline __m128i _mm_cmpgt_epu32(__m128i a, __m128i b) { return _mm_cmpgt_epi32(
            _mm_xor_si128(a, _mm_setmin_epi32()), _mm_xor_si128(b, _mm_setmin_epi32())); }
} // namespace SSE
} // namespace Vc

// SSE3
#ifdef VC_IMPL_SSE3
#include <pmmintrin.h>
#endif
// SSSE3
#ifdef VC_IMPL_SSSE3
#include <tmmintrin.h>
namespace Vc
{
namespace SSE
{

    static inline __m128i set1_epi8(int a) {
#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 5
        return _mm_shuffle_epi8(_mm_cvtsi32_si128(a), _mm_setzero_si128());
#else
        // GCC 4.5 nows about the pshufb improvement
        return _mm_set1_epi8(a);
#endif
    }

} // namespace SSE
} // namespace Vc
#else
namespace Vc
{
namespace SSE
{
    static inline __m128i _mm_abs_epi8 (__m128i a) CONST;
    static inline __m128i _mm_abs_epi16(__m128i a) CONST;
    static inline __m128i _mm_abs_epi32(__m128i a) CONST;
    static inline __m128i _mm_alignr_epi8(__m128i a, __m128i b, const int s) CONST;

    static inline __m128i _mm_abs_epi8 (__m128i a) {
        __m128i negative = _mm_cmplt_epi8 (a, _mm_setzero_si128());
        return _mm_add_epi8 (_mm_xor_si128(a, negative), _mm_and_si128(negative,  _mm_setone_epi8()));
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
    static inline __m128i _mm_abs_epi16(__m128i a) {
        __m128i negative = _mm_cmplt_epi16(a, _mm_setzero_si128());
        return _mm_add_epi16(_mm_xor_si128(a, negative), _mm_srli_epi16(negative, 15));
    }
    static inline __m128i _mm_abs_epi32(__m128i a) {
        __m128i negative = _mm_cmplt_epi32(a, _mm_setzero_si128());
        return _mm_add_epi32(_mm_xor_si128(a, negative), _mm_srli_epi32(negative, 31));
    }
    static inline __m128i set1_epi8(int a) {
        return _mm_set1_epi8(a);
    }
    static inline __m128i _mm_alignr_epi8(__m128i a, __m128i b, const int s) {
        switch (s) {
            case  0: return b;
            case  1: return _mm_or_si128(_mm_slli_si128(a, 15), _mm_srli_si128(b,  1));
            case  2: return _mm_or_si128(_mm_slli_si128(a, 14), _mm_srli_si128(b,  2));
            case  3: return _mm_or_si128(_mm_slli_si128(a, 13), _mm_srli_si128(b,  3));
            case  4: return _mm_or_si128(_mm_slli_si128(a, 12), _mm_srli_si128(b,  4));
            case  5: return _mm_or_si128(_mm_slli_si128(a, 11), _mm_srli_si128(b,  5));
            case  6: return _mm_or_si128(_mm_slli_si128(a, 10), _mm_srli_si128(b,  6));
            case  7: return _mm_or_si128(_mm_slli_si128(a,  9), _mm_srli_si128(b,  7));
            case  8: return _mm_or_si128(_mm_slli_si128(a,  8), _mm_srli_si128(b,  8));
            case  9: return _mm_or_si128(_mm_slli_si128(a,  7), _mm_srli_si128(b,  9));
            case 10: return _mm_or_si128(_mm_slli_si128(a,  6), _mm_srli_si128(b, 10));
            case 11: return _mm_or_si128(_mm_slli_si128(a,  5), _mm_srli_si128(b, 11));
            case 12: return _mm_or_si128(_mm_slli_si128(a,  4), _mm_srli_si128(b, 12));
            case 13: return _mm_or_si128(_mm_slli_si128(a,  3), _mm_srli_si128(b, 13));
            case 14: return _mm_or_si128(_mm_slli_si128(a,  2), _mm_srli_si128(b, 14));
            case 15: return _mm_or_si128(_mm_slli_si128(a,  1), _mm_srli_si128(b, 15));
            case 16: return a;
            case 17: return _mm_srli_si128(a,  1);
            case 18: return _mm_srli_si128(a,  2);
            case 19: return _mm_srli_si128(a,  3);
            case 20: return _mm_srli_si128(a,  4);
            case 21: return _mm_srli_si128(a,  5);
            case 22: return _mm_srli_si128(a,  6);
            case 23: return _mm_srli_si128(a,  7);
            case 24: return _mm_srli_si128(a,  8);
            case 25: return _mm_srli_si128(a,  9);
            case 26: return _mm_srli_si128(a, 10);
            case 27: return _mm_srli_si128(a, 11);
            case 28: return _mm_srli_si128(a, 12);
            case 29: return _mm_srli_si128(a, 13);
            case 30: return _mm_srli_si128(a, 14);
            case 31: return _mm_srli_si128(a, 15);
        }
        return _mm_setzero_si128();
    }

} // namespace SSE
} // namespace Vc

#endif

// SSE4.1
#ifdef VC_IMPL_SSE4_1
#include <smmintrin.h>
#else
namespace Vc
{
namespace SSE
{
    static inline __m128d _mm_blendv_pd(__m128d a, __m128d b, __m128d c) CONST ALWAYS_INLINE;
    static inline __m128  _mm_blendv_ps(__m128  a, __m128  b, __m128  c) CONST ALWAYS_INLINE;
    static inline __m128i _mm_blendv_epi8(__m128i a, __m128i b, __m128i c) CONST ALWAYS_INLINE;
    static inline __m128d _mm_blend_pd(__m128d a, __m128d b, const int mask) CONST ALWAYS_INLINE;
    static inline __m128  _mm_blend_ps(__m128  a, __m128  b, const int mask) CONST ALWAYS_INLINE;
    static inline __m128i _mm_blend_epi16(__m128i a, __m128i b, const int mask) CONST ALWAYS_INLINE;
    static inline __m128i _mm_max_epi8 (__m128i a, __m128i b) CONST;
    static inline __m128i _mm_max_epi32(__m128i a, __m128i b) CONST;
    //static inline __m128i _mm_max_epu8 (__m128i a, __m128i b) CONST;
    static inline __m128i _mm_max_epu16(__m128i a, __m128i b) CONST;
    static inline __m128i _mm_max_epu32(__m128i a, __m128i b) CONST;
    //static inline __m128i _mm_min_epu8 (__m128i a, __m128i b) CONST;
    static inline __m128i _mm_min_epu16(__m128i a, __m128i b) CONST;
    static inline __m128i _mm_min_epu32(__m128i a, __m128i b) CONST;
    static inline __m128i _mm_min_epi8 (__m128i a, __m128i b) CONST;
    static inline __m128i _mm_min_epi32(__m128i a, __m128i b) CONST;

    static inline __m128d _mm_blendv_pd(__m128d a, __m128d b, __m128d c) {
        return _mm_or_pd(_mm_andnot_pd(c, a), _mm_and_pd(c, b));
    }
    static inline __m128  _mm_blendv_ps(__m128  a, __m128  b, __m128  c) {
        return _mm_or_ps(_mm_andnot_ps(c, a), _mm_and_ps(c, b));
    }
    static inline __m128i _mm_blendv_epi8(__m128i a, __m128i b, __m128i c) {
        return _mm_or_si128(_mm_andnot_si128(c, a), _mm_and_si128(c, b));
    }

    // only use the following blend functions with immediates as mask and, of course, compiling
    // with optimization
    static inline __m128d _mm_blend_pd(__m128d a, __m128d b, const int mask) {
        __m128i c;
        switch (mask) {
        case 0x0:
            return a;
        case 0x1:
            c = _mm_srli_si128(_mm_setallone_si128(), 8);
            break;
        case 0x2:
            c = _mm_slli_si128(_mm_setallone_si128(), 8);
            break;
        case 0x3:
            return b;
        }
        __m128d _c = _mm_castsi128_pd(c);
        return _mm_or_pd(_mm_andnot_pd(_c, a), _mm_and_pd(_c, b));
    }
    static inline __m128  _mm_blend_ps(__m128  a, __m128  b, const int mask) {
        __m128i c;
        switch (mask) {
        case 0x0:
            return a;
        case 0x1:
            c = _mm_srli_si128(_mm_setallone_si128(), 12);
            break;
        case 0x2:
            c = _mm_slli_si128(_mm_srli_si128(_mm_setallone_si128(), 12), 4);
            break;
        case 0x3:
            c = _mm_srli_si128(_mm_setallone_si128(), 8);
            break;
        case 0x4:
            c = _mm_slli_si128(_mm_srli_si128(_mm_setallone_si128(), 12), 8);
            break;
        case 0x5:
            c = _mm_set_epi32(0, -1, 0, -1);
            break;
        case 0x6:
            c = _mm_slli_si128(_mm_srli_si128(_mm_setallone_si128(), 8), 4);
            break;
        case 0x7:
            c = _mm_srli_si128(_mm_setallone_si128(), 4);
            break;
        case 0x8:
            c = _mm_slli_si128(_mm_setallone_si128(), 12);
            break;
        case 0x9:
            c = _mm_set_epi32(-1, 0, 0, -1);
            break;
        case 0xa:
            c = _mm_set_epi32(-1, 0, -1, 0);
            break;
        case 0xb:
            c = _mm_set_epi32(-1, 0, -1, -1);
            break;
        case 0xc:
            c = _mm_slli_si128(_mm_setallone_si128(), 8);
            break;
        case 0xd:
            c = _mm_set_epi32(-1, -1, 0, -1);
            break;
        case 0xe:
            c = _mm_slli_si128(_mm_setallone_si128(), 4);
            break;
        case 0xf:
            return b;
        default: // may not happen
            abort();
            c = _mm_setzero_si128();
            break;
        }
        __m128 _c = _mm_castsi128_ps(c);
        return _mm_or_ps(_mm_andnot_ps(_c, a), _mm_and_ps(_c, b));
    }
    static inline __m128i _mm_blend_epi16(__m128i a, __m128i b, const int mask) {
        __m128i c;
        switch (mask) {
        case 0x00:
            return a;
        case 0x01:
            c = _mm_srli_si128(_mm_setallone_si128(), 14);
            break;
        case 0x03:
            c = _mm_srli_si128(_mm_setallone_si128(), 12);
            break;
        case 0x07:
            c = _mm_srli_si128(_mm_setallone_si128(), 10);
            break;
        case 0x0f:
            return _mm_unpackhi_epi64(_mm_slli_si128(b, 8), a);
        case 0x1f:
            c = _mm_srli_si128(_mm_setallone_si128(), 6);
            break;
        case 0x3f:
            c = _mm_srli_si128(_mm_setallone_si128(), 4);
            break;
        case 0x7f:
            c = _mm_srli_si128(_mm_setallone_si128(), 2);
            break;
        case 0x80:
            c = _mm_slli_si128(_mm_setallone_si128(), 14);
            break;
        case 0xc0:
            c = _mm_slli_si128(_mm_setallone_si128(), 12);
            break;
        case 0xe0:
            c = _mm_slli_si128(_mm_setallone_si128(), 10);
            break;
        case 0xf0:
            c = _mm_slli_si128(_mm_setallone_si128(), 8);
            break;
        case 0xf8:
            c = _mm_slli_si128(_mm_setallone_si128(), 6);
            break;
        case 0xfc:
            c = _mm_slli_si128(_mm_setallone_si128(), 4);
            break;
        case 0xfe:
            c = _mm_slli_si128(_mm_setallone_si128(), 2);
            break;
        case 0xff:
            return b;
        case 0xcc:
            return _mm_unpacklo_epi32(_mm_shuffle_epi32(a, _MM_SHUFFLE(2, 0, 2, 0)), _mm_shuffle_epi32(b, _MM_SHUFFLE(3, 1, 3, 1)));
        case 0x33:
            return _mm_unpacklo_epi32(_mm_shuffle_epi32(b, _MM_SHUFFLE(2, 0, 2, 0)), _mm_shuffle_epi32(a, _MM_SHUFFLE(3, 1, 3, 1)));
        default:
            const __m128i shift = _mm_set_epi16(0x0100, 0x0200, 0x0400, 0x0800, 0x1000, 0x2000, 0x4000, -0x7fff);
            c = _mm_srai_epi16(_mm_mullo_epi16(_mm_set1_epi16(mask), shift), 15);
            break;
        }
        return _mm_or_si128(_mm_andnot_si128(c, a), _mm_and_si128(c, b));
    }

    static inline __m128i _mm_max_epi8 (__m128i a, __m128i b) {
        return _mm_blendv_epi8(b, a, _mm_cmpgt_epi8 (a, b));
    }
    static inline __m128i _mm_max_epi32(__m128i a, __m128i b) {
        return _mm_blendv_epi8(b, a, _mm_cmpgt_epi32(a, b));
    }
//X         static inline __m128i _mm_max_epu8 (__m128i a, __m128i b) {
//X             return _mm_blendv_epi8(b, a, _mm_cmpgt_epu8 (a, b));
//X         }
    static inline __m128i _mm_max_epu16(__m128i a, __m128i b) {
        return _mm_blendv_epi8(b, a, _mm_cmpgt_epu16(a, b));
    }
    static inline __m128i _mm_max_epu32(__m128i a, __m128i b) {
        return _mm_blendv_epi8(b, a, _mm_cmpgt_epu32(a, b));
    }
//X         static inline __m128i _mm_min_epu8 (__m128i a, __m128i b) {
//X             return _mm_blendv_epi8(a, b, _mm_cmpgt_epu8 (a, b));
//X         }
    static inline __m128i _mm_min_epu16(__m128i a, __m128i b) {
        return _mm_blendv_epi8(a, b, _mm_cmpgt_epu16(a, b));
    }
    static inline __m128i _mm_min_epu32(__m128i a, __m128i b) {
        return _mm_blendv_epi8(a, b, _mm_cmpgt_epi32(a, b));
    }
    static inline __m128i _mm_min_epi8 (__m128i a, __m128i b) {
        return _mm_blendv_epi8(a, b, _mm_cmpgt_epi8 (a, b));
    }
    static inline __m128i _mm_min_epi32(__m128i a, __m128i b) {
        return _mm_blendv_epi8(a, b, _mm_cmpgt_epi32(a, b));
    }
} // namespace SSE
} // namespace Vc
#endif

#endif // SSE_INTRINSICS_H
