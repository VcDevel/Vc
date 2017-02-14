/*  This file is part of the Vc library. {{{
Copyright © 2016 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_DATAPAR_X86_ARITHMETICS_H_
#define VC_DATAPAR_X86_ARITHMETICS_H_

#include "storage.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
namespace x86
{
// plus{{{1
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <class T, size_t N> Vc_INTRINSIC auto plus(Storage<T, N> a, Storage<T, N> b)
{
    return a.builtin() + b.builtin();
}
#else   // Vc_USE_BUILTIN_VECTOR_TYPES
Vc_INTRINSIC __m128  Vc_VDECL plus(x_f32 a, x_f32 b) { return _mm_add_ps(a, b); }
Vc_INTRINSIC __m128d Vc_VDECL plus(x_f64 a, x_f64 b) { return _mm_add_pd(a, b); }
Vc_INTRINSIC __m128i Vc_VDECL plus(x_i64 a, x_i64 b) { return _mm_add_epi64(a, b); }
Vc_INTRINSIC __m128i Vc_VDECL plus(x_u64 a, x_u64 b) { return _mm_add_epi64(a, b); }
Vc_INTRINSIC __m128i Vc_VDECL plus(x_i32 a, x_i32 b) { return _mm_add_epi32(a, b); }
Vc_INTRINSIC __m128i Vc_VDECL plus(x_u32 a, x_u32 b) { return _mm_add_epi32(a, b); }
Vc_INTRINSIC __m128i Vc_VDECL plus(x_i16 a, x_i16 b) { return _mm_add_epi16(a, b); }
Vc_INTRINSIC __m128i Vc_VDECL plus(x_u16 a, x_u16 b) { return _mm_add_epi16(a, b); }
Vc_INTRINSIC __m128i Vc_VDECL plus(x_i08 a, x_i08 b) { return _mm_add_epi8 (a, b); }
Vc_INTRINSIC __m128i Vc_VDECL plus(x_u08 a, x_u08 b) { return _mm_add_epi8 (a, b); }

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC __m256  Vc_VDECL plus(y_f32 a, y_f32 b) { return _mm256_add_ps(a, b); }
Vc_INTRINSIC __m256d Vc_VDECL plus(y_f64 a, y_f64 b) { return _mm256_add_pd(a, b); }
#endif  // Vc_HAVE_AVX
#ifdef Vc_HAVE_AVX2
Vc_INTRINSIC __m256i Vc_VDECL plus(y_i64 a, y_i64 b) { return _mm256_add_epi64(a, b); }
Vc_INTRINSIC __m256i Vc_VDECL plus(y_u64 a, y_u64 b) { return _mm256_add_epi64(a, b); }
Vc_INTRINSIC __m256i Vc_VDECL plus(y_i32 a, y_i32 b) { return _mm256_add_epi32(a, b); }
Vc_INTRINSIC __m256i Vc_VDECL plus(y_u32 a, y_u32 b) { return _mm256_add_epi32(a, b); }
Vc_INTRINSIC __m256i Vc_VDECL plus(y_i16 a, y_i16 b) { return _mm256_add_epi16(a, b); }
Vc_INTRINSIC __m256i Vc_VDECL plus(y_u16 a, y_u16 b) { return _mm256_add_epi16(a, b); }
Vc_INTRINSIC __m256i Vc_VDECL plus(y_i08 a, y_i08 b) { return _mm256_add_epi8 (a, b); }
Vc_INTRINSIC __m256i Vc_VDECL plus(y_u08 a, y_u08 b) { return _mm256_add_epi8 (a, b); }
#endif  // Vc_HAVE_AVX2

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC __m512  Vc_VDECL plus(z_f32 a, z_f32 b) { return _mm512_add_ps(a, b); }
Vc_INTRINSIC __m512d Vc_VDECL plus(z_f64 a, z_f64 b) { return _mm512_add_pd(a, b); }
Vc_INTRINSIC __m512i Vc_VDECL plus(z_i64 a, z_i64 b) { return _mm512_add_epi64(a, b); }
Vc_INTRINSIC __m512i Vc_VDECL plus(z_u64 a, z_u64 b) { return _mm512_add_epi64(a, b); }
Vc_INTRINSIC __m512i Vc_VDECL plus(z_i32 a, z_i32 b) { return _mm512_add_epi32(a, b); }
Vc_INTRINSIC __m512i Vc_VDECL plus(z_u32 a, z_u32 b) { return _mm512_add_epi32(a, b); }
#ifdef Vc_HAVE_AVX512BW
Vc_INTRINSIC __m512i Vc_VDECL plus(z_i16 a, z_i16 b) { return _mm512_add_epi16(a, b); }
Vc_INTRINSIC __m512i Vc_VDECL plus(z_u16 a, z_u16 b) { return _mm512_add_epi16(a, b); }
Vc_INTRINSIC __m512i Vc_VDECL plus(z_i08 a, z_i08 b) { return _mm512_add_epi8(a, b); }
Vc_INTRINSIC __m512i Vc_VDECL plus(z_u08 a, z_u08 b) { return _mm512_add_epi8(a, b); }
#endif  // Vc_HAVE_AVX512BW
#endif  // Vc_HAVE_AVX512F
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES

// minus{{{1
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <class T, size_t N> Vc_INTRINSIC auto minus(Storage<T, N> a, Storage<T, N> b)
{
    return a.builtin() - b.builtin();
}
#else   // Vc_USE_BUILTIN_VECTOR_TYPES
Vc_INTRINSIC __m128  Vc_VDECL minus(x_f32 a, x_f32 b) { return _mm_sub_ps(a, b); }
Vc_INTRINSIC __m128d Vc_VDECL minus(x_f64 a, x_f64 b) { return _mm_sub_pd(a, b); }
Vc_INTRINSIC __m128i Vc_VDECL minus(x_i64 a, x_i64 b) { return _mm_sub_epi64(a, b); }
Vc_INTRINSIC __m128i Vc_VDECL minus(x_u64 a, x_u64 b) { return _mm_sub_epi64(a, b); }
Vc_INTRINSIC __m128i Vc_VDECL minus(x_i32 a, x_i32 b) { return _mm_sub_epi32(a, b); }
Vc_INTRINSIC __m128i Vc_VDECL minus(x_u32 a, x_u32 b) { return _mm_sub_epi32(a, b); }
Vc_INTRINSIC __m128i Vc_VDECL minus(x_i16 a, x_i16 b) { return _mm_sub_epi16(a, b); }
Vc_INTRINSIC __m128i Vc_VDECL minus(x_u16 a, x_u16 b) { return _mm_sub_epi16(a, b); }
Vc_INTRINSIC __m128i Vc_VDECL minus(x_i08 a, x_i08 b) { return _mm_sub_epi8 (a, b); }
Vc_INTRINSIC __m128i Vc_VDECL minus(x_u08 a, x_u08 b) { return _mm_sub_epi8 (a, b); }

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC __m256  Vc_VDECL minus(y_f32 a, y_f32 b) { return _mm256_sub_ps(a, b); }
Vc_INTRINSIC __m256d Vc_VDECL minus(y_f64 a, y_f64 b) { return _mm256_sub_pd(a, b); }
#endif  // Vc_HAVE_AVX
#ifdef Vc_HAVE_AVX2
Vc_INTRINSIC __m256i Vc_VDECL minus(y_i64 a, y_i64 b) { return _mm256_sub_epi64(a, b); }
Vc_INTRINSIC __m256i Vc_VDECL minus(y_u64 a, y_u64 b) { return _mm256_sub_epi64(a, b); }
Vc_INTRINSIC __m256i Vc_VDECL minus(y_i32 a, y_i32 b) { return _mm256_sub_epi32(a, b); }
Vc_INTRINSIC __m256i Vc_VDECL minus(y_u32 a, y_u32 b) { return _mm256_sub_epi32(a, b); }
Vc_INTRINSIC __m256i Vc_VDECL minus(y_i16 a, y_i16 b) { return _mm256_sub_epi16(a, b); }
Vc_INTRINSIC __m256i Vc_VDECL minus(y_u16 a, y_u16 b) { return _mm256_sub_epi16(a, b); }
Vc_INTRINSIC __m256i Vc_VDECL minus(y_i08 a, y_i08 b) { return _mm256_sub_epi8 (a, b); }
Vc_INTRINSIC __m256i Vc_VDECL minus(y_u08 a, y_u08 b) { return _mm256_sub_epi8 (a, b); }
#endif  // Vc_HAVE_AVX2

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC __m512  Vc_VDECL minus(z_f32 a, z_f32 b) { return _mm512_sub_ps(a, b); }
Vc_INTRINSIC __m512d Vc_VDECL minus(z_f64 a, z_f64 b) { return _mm512_sub_pd(a, b); }
Vc_INTRINSIC __m512i Vc_VDECL minus(z_i64 a, z_i64 b) { return _mm512_sub_epi64(a, b); }
Vc_INTRINSIC __m512i Vc_VDECL minus(z_u64 a, z_u64 b) { return _mm512_sub_epi64(a, b); }
Vc_INTRINSIC __m512i Vc_VDECL minus(z_i32 a, z_i32 b) { return _mm512_sub_epi32(a, b); }
Vc_INTRINSIC __m512i Vc_VDECL minus(z_u32 a, z_u32 b) { return _mm512_sub_epi32(a, b); }
#ifdef Vc_HAVE_AVX512BW
Vc_INTRINSIC __m512i Vc_VDECL minus(z_i16 a, z_i16 b) { return _mm512_sub_epi16(a, b); }
Vc_INTRINSIC __m512i Vc_VDECL minus(z_u16 a, z_u16 b) { return _mm512_sub_epi16(a, b); }
Vc_INTRINSIC __m512i Vc_VDECL minus(z_i08 a, z_i08 b) { return _mm512_sub_epi8(a, b); }
Vc_INTRINSIC __m512i Vc_VDECL minus(z_u08 a, z_u08 b) { return _mm512_sub_epi8(a, b); }
#endif  // Vc_HAVE_AVX512BW
#endif  // Vc_HAVE_AVX512F
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES

// multiplies{{{1
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <class T, size_t N> Vc_INTRINSIC auto Vc_VDECL multiplies(Storage<T, N> a, Storage<T, N> b)
{
    return a.builtin() * b.builtin();
}
#else   // Vc_USE_BUILTIN_VECTOR_TYPES
Vc_INTRINSIC __m128  Vc_VDECL multiplies(x_f32 a, x_f32 b) { return _mm_mul_ps(a, b); }
Vc_INTRINSIC __m128d Vc_VDECL multiplies(x_f64 a, x_f64 b) { return _mm_mul_pd(a, b); }
Vc_INTRINSIC __m128i Vc_VDECL multiplies(x_i64 a, x_i64 b) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm_mullo_epi64(a, b);
#else
    return x_i64{a[0] * b[0], a[1] * b[1]};
#endif
}
Vc_INTRINSIC __m128i Vc_VDECL multiplies(x_u64 a, x_u64 b) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm_mullo_epi64(a, b);
#else
    return x_u64{a[0] * b[0], a[1] * b[1]};
#endif
}
Vc_INTRINSIC __m128i Vc_VDECL multiplies(x_i32 a, x_i32 b) {
#ifdef Vc_HAVE_SSE4_1
    return _mm_mullo_epi32(a, b);
#else
    const __m128i aShift = _mm_srli_si128(a, 4);
    const __m128i ab02 = _mm_mul_epu32(a, b);  // [a0 * b0, a2 * b2]
    const __m128i bShift = _mm_srli_si128(b, 4);
    const __m128i ab13 = _mm_mul_epu32(aShift, bShift);  // [a1 * b1, a3 * b3]
    return _mm_unpacklo_epi32(_mm_shuffle_epi32(ab02, 8), _mm_shuffle_epi32(ab13, 8));
#endif
}
Vc_INTRINSIC __m128i Vc_VDECL multiplies(x_u32 a, x_u32 b) { return multiplies(x_i32(a), x_i32(b)); }
Vc_INTRINSIC __m128i Vc_VDECL multiplies(x_i16 a, x_i16 b) { return _mm_mullo_epi16(a, b); }
Vc_INTRINSIC __m128i Vc_VDECL multiplies(x_u16 a, x_u16 b) { return _mm_mullo_epi16(a, b); }
Vc_INTRINSIC __m128i Vc_VDECL multiplies(x_i08 a, x_i08 b) {
    return or_(
        and_(_mm_mullo_epi16(a, b), _mm_srli_epi16(allone<__m128i>(), 8)),
        _mm_slli_epi16(_mm_mullo_epi16(_mm_srli_si128(a, 1), _mm_srli_si128(b, 1)), 8));
}
Vc_INTRINSIC __m128i Vc_VDECL multiplies(x_u08 a, x_u08 b) { return multiplies(x_i08(a), x_i08(b)); }

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC __m256  Vc_VDECL multiplies(y_f32 a, y_f32 b) { return _mm256_mul_ps(a, b); }
Vc_INTRINSIC __m256d Vc_VDECL multiplies(y_f64 a, y_f64 b) { return _mm256_mul_pd(a, b); }
#endif  // Vc_HAVE_AVX
#ifdef Vc_HAVE_AVX2
#if defined Vc_HAVE_AVX512DQ && defined Vc_HAVE_AVX512VL
Vc_INTRINSIC __m256i Vc_VDECL multiplies(y_i64 a, y_i64 b) { return _mm256_mullo_epi64(a, b); }
Vc_INTRINSIC __m256i Vc_VDECL multiplies(y_u64 a, y_u64 b) { return _mm256_mullo_epi64(a, b); }
#else
Vc_INTRINSIC __m256i Vc_VDECL multiplies(y_i64 a, y_i64 b) { return y_i64{a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]}; }
Vc_INTRINSIC __m256i Vc_VDECL multiplies(y_u64 a, y_u64 b) { return y_u64{a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]}; }
#endif
Vc_INTRINSIC __m256i Vc_VDECL multiplies(y_i32 a, y_i32 b) { return _mm256_mullo_epi32(a, b); }
Vc_INTRINSIC __m256i Vc_VDECL multiplies(y_u32 a, y_u32 b) { return _mm256_mullo_epi32(a, b); }
Vc_INTRINSIC __m256i Vc_VDECL multiplies(y_i16 a, y_i16 b) { return _mm256_mullo_epi16(a, b); }
Vc_INTRINSIC __m256i Vc_VDECL multiplies(y_u16 a, y_u16 b) { return _mm256_mullo_epi16(a, b); }
Vc_INTRINSIC __m256i Vc_VDECL multiplies(y_i08 a, y_i08 b) {
    return or_(
        and_(_mm256_mullo_epi16(a, b), _mm256_srli_epi16(allone<__m256i>(), 8)),
        _mm256_slli_epi16(_mm256_mullo_epi16(_mm256_srli_si256(a, 1), _mm256_srli_si256(b, 1)), 8));
}
Vc_INTRINSIC __m256i Vc_VDECL multiplies(y_u08 a, y_u08 b) { return multiplies(y_i08(a), y_i08(b)); }
#endif  // Vc_HAVE_AVX2

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC __m512  Vc_VDECL multiplies(z_f32 a, z_f32 b) { return _mm512_mul_ps(a, b); }
Vc_INTRINSIC __m512d Vc_VDECL multiplies(z_f64 a, z_f64 b) { return _mm512_mul_pd(a, b); }
Vc_INTRINSIC __m512i Vc_VDECL multiplies(z_i64 a, z_i64 b) {
#ifdef Vc_HAVE_AVX512DQ
    return _mm512_mullo_epi64(a, b);
#else   // Vc_HAVE_AVX512DQ
    const __m512i r0 = _mm512_mul_epu32(a, b);
    const __m512i aShift = _mm512_srli_epi64(a, 32);
    const __m512i bShift = _mm512_srli_epi64(b, 32);
    const __m512i r1 = _mm512_slli_epi64(_mm512_mul_epu32(aShift, b), 32);
    const __m512i r2 = _mm512_slli_epi64(_mm512_mul_epu32(a, bShift), 32);
    return _mm512_add_epi64(_mm512_add_epi64(r0, r1), r2);
#endif  // Vc_HAVE_AVX512DQ
}
Vc_INTRINSIC __m512i Vc_VDECL multiplies(z_u64 a, z_u64 b) { return multiplies(z_i64(a),z_i64(b)); }
Vc_INTRINSIC __m512i Vc_VDECL multiplies(z_i32 a, z_i32 b) { return _mm512_mullo_epi32(a, b); }
Vc_INTRINSIC __m512i Vc_VDECL multiplies(z_u32 a, z_u32 b) { return _mm512_mullo_epi32(a, b); }
#ifdef Vc_HAVE_AVX512BW
Vc_INTRINSIC __m512i Vc_VDECL multiplies(z_i16 a, z_i16 b) { return _mm512_mullo_epi16(a, b); }
Vc_INTRINSIC __m512i Vc_VDECL multiplies(z_u16 a, z_u16 b) { return _mm512_mullo_epi16(a, b); }
Vc_INTRINSIC __m512i Vc_VDECL multiplies(z_i08 a, z_i08 b) {
    return or_(
        and_(_mm512_mullo_epi16(a, b), _mm512_srli_epi16(allone<__m512i>(), 8)),
        _mm512_slli_epi16(_mm512_mullo_epi16(_mm512_srli_epi16(a, 8), _mm512_srli_epi16(b, 8)), 8));
}
Vc_INTRINSIC __m512i Vc_VDECL multiplies(z_u08 a, z_u08 b) { return multiplies(z_i08(a), z_i08(b)); }
#endif  // Vc_HAVE_AVX512BW
#endif  // Vc_HAVE_AVX512F
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES

// divides{{{1
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
// builtin{{{2
template <class T, size_t N> Vc_INTRINSIC auto divides(Storage<T, N> a, Storage<T, N> b)
{
    return a.builtin() / b.builtin();
}
#else   // Vc_USE_BUILTIN_VECTOR_TYPES
// sse{{{2
Vc_INTRINSIC x_f32 Vc_VDECL divides(x_f32 a, x_f32 b) { return _mm_div_ps(a, b); }

Vc_INTRINSIC x_f64 Vc_VDECL divides(x_f64 a, x_f64 b) { return _mm_div_pd(a, b); }

Vc_INTRINSIC x_i64 Vc_VDECL divides(x_i64 a, x_i64 b) { return {a[0] / b[0], a[1] / b[1]}; }

Vc_INTRINSIC x_u64 Vc_VDECL divides(x_u64 a, x_u64 b) { return {a[0] / b[0], a[1] / b[1]}; }

Vc_INTRINSIC x_i32 Vc_VDECL divides(x_i32 a, x_i32 b) {
#ifdef Vc_HAVE_AVX
    return _mm256_cvttpd_epi32(
        _mm256_div_pd(_mm256_cvtepi32_pd(a), _mm256_cvtepi32_pd(b)));
#else
    return {a[0] / b[0], a[1] / b[1], a[2] / b[2], a[3] / b[3]};
#endif
}

Vc_INTRINSIC x_u32 Vc_VDECL divides(x_u32 a, x_u32 b) {
#ifdef Vc_HAVE_AVX512VL
    return _mm256_cvttpd_epu32(
        _mm256_div_pd(_mm256_cvtepu32_pd(a), _mm256_cvtepu32_pd(b)));
#else
    return {a[0] / b[0], a[1] / b[1], a[2] / b[2], a[3] / b[3]};
#endif
}

Vc_INTRINSIC x_i16 Vc_VDECL divides(x_i16 a, x_i16 b) {
    const __m128i mask = broadcast16(0x0000ffffu);
    const auto ahi = andnot_(mask, a);
    const auto bhi = andnot_(mask, b);
    const auto alo = _mm_slli_epi32(a, 16);
    const auto blo = _mm_slli_epi32(b, 16);
#ifdef Vc_HAVE_AVX
    __m256i c = _mm256_cvttps_epi32(_mm256_div_ps(_mm256_cvtepi32_ps(concat(ahi, alo)),
                                                  _mm256_cvtepi32_ps(concat(bhi, blo))));
    return or_(_mm_slli_epi32(lo128(c), 16), and_(mask, hi128(c)));
#else
    __m128i c[2] = {
        _mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(ahi), _mm_cvtepi32_ps(bhi))),
        _mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(alo), _mm_cvtepi32_ps(blo)))};
    return or_(_mm_slli_epi32(c[0], 16), and_(mask, c[1]));
#endif
}

Vc_INTRINSIC x_u16 Vc_VDECL divides(x_u16 a, x_u16 b) {
#ifdef Vc_HAVE_AVX
    return convert<y_f32, x_u16>(
        _mm256_div_ps(convert<x_u16, y_f32>(a), convert<x_u16, y_f32>(b)));
#else
    return or_(
        _mm_slli_epi32(
            _mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(_mm_srli_epi32(a, 16)),
                                        _mm_cvtepi32_ps(_mm_srli_epi32(b, 16)))),
            16),
        _mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(and_(a, broadcast16(0xffff))),
                                    _mm_cvtepi32_ps(and_(b, broadcast16(0xffff))))));
#endif
}

inline x_i08 Vc_VDECL divides(x_i08 a, x_i08 b) {
#ifdef Vc_HAVE_AVX512F
    return convert<z_f32, x_i08>(
        _mm512_div_ps(convert<x_i08, z_f32>(a), convert<x_i08, z_f32>(b)));
#else
    const __m128i maskhi = broadcast16(0xff000000u);
    const __m128i masklo = _mm_srli_epi32(maskhi, 24);
    __m128i r[4] = {
        _mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(     _mm_slli_epi32(a, 24)         ),
                                    _mm_cvtepi32_ps(     _mm_slli_epi32(b, 24)         ))),
        _mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(and_(_mm_slli_epi32(a, 16), maskhi)),
                                    _mm_cvtepi32_ps(and_(_mm_slli_epi32(b, 16), maskhi)))),
        _mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(and_(_mm_slli_epi32(a,  8), maskhi)),
                                    _mm_cvtepi32_ps(and_(_mm_slli_epi32(b,  8), maskhi)))),
        _mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(and_(               a     , maskhi)),
                                    _mm_cvtepi32_ps(and_(               b     , maskhi))))
    };
    r[0] = and_(r[0], masklo);
    r[1] = _mm_slli_epi32(and_(r[1], masklo), 8);
    r[2] = _mm_slli_epi32(and_(r[2], masklo), 16);
    r[3] = _mm_slli_epi32(r[3], 24);
    return or_(or_(r[0], r[1]), or_(r[2], r[3]));
#endif
}

inline x_u08 Vc_VDECL divides(x_u08 a, x_u08 b) {
#ifdef Vc_HAVE_AVX512F
    return convert<z_f32, x_u08>(
        _mm512_div_ps(convert<x_u08, z_f32>(a), convert<x_u08, z_f32>(b)));
#else
    const __m128i mask = broadcast16(0xff);
    return or_(
        or_(_mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(and_(a, mask)),
                                        _mm_cvtepi32_ps(and_(b, mask)))),
            _mm_slli_epi32(_mm_cvttps_epi32(_mm_div_ps(
                               _mm_cvtepi32_ps(and_(_mm_srli_epi32(a, 8), mask)),
                               _mm_cvtepi32_ps(and_(_mm_srli_epi32(b, 8), mask)))),
                           8)),
        or_(_mm_slli_epi32(_mm_cvttps_epi32(_mm_div_ps(
                               _mm_cvtepi32_ps(and_(_mm_srli_epi32(a, 16), mask)),
                               _mm_cvtepi32_ps(and_(_mm_srli_epi32(b, 16), mask)))),
                           16),
            _mm_slli_epi32(
                _mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(_mm_srli_epi32(a, 24)),
                                            _mm_cvtepi32_ps(_mm_srli_epi32(b, 24)))),
                24)));
#endif
}

// avx{{{2
#ifdef Vc_HAVE_AVX
Vc_INTRINSIC y_f32 Vc_VDECL divides(y_f32 a, y_f32 b) { return _mm256_div_ps(a, b); }
Vc_INTRINSIC y_f64 Vc_VDECL divides(y_f64 a, y_f64 b) { return _mm256_div_pd(a, b); }
#endif// Vc_HAVE_AVX
#ifdef Vc_HAVE_AVX2
Vc_INTRINSIC y_i64 Vc_VDECL divides(y_i64 a, y_i64 b) { return {a[0] / b[0], a[1] / b[1], a[2] / b[2], a[3] / b[3]}; }
Vc_INTRINSIC y_u64 Vc_VDECL divides(y_u64 a, y_u64 b) { return {a[0] / b[0], a[1] / b[1], a[2] / b[2], a[3] / b[3]}; }
Vc_INTRINSIC y_i32 Vc_VDECL divides(y_i32 a, y_i32 b) {
#ifdef Vc_HAVE_AVX512F
    return _mm512_cvttpd_epi32(
        _mm512_div_pd(_mm512_cvtepi32_pd(a), _mm512_cvtepi32_pd(b)));
#else
    return concat(_mm256_cvttpd_epi32(_mm256_div_pd(_mm256_cvtepi32_pd(lo128(a)),
                                                    _mm256_cvtepi32_pd(lo128(b)))),
                  _mm256_cvttpd_epi32(_mm256_div_pd(_mm256_cvtepi32_pd(hi128(a)),
                                                    _mm256_cvtepi32_pd(hi128(b)))));
#endif
}
Vc_INTRINSIC y_u32 Vc_VDECL divides(y_u32 a, y_u32 b) {
#ifdef Vc_HAVE_AVX512F
    return _mm512_cvttpd_epu32(
        _mm512_div_pd(_mm512_cvtepu32_pd(a), _mm512_cvtepu32_pd(b)));
#else
    return concat(convert<y_f64, x_u32>(_mm256_div_pd(convert<x_u32, y_f64>(lo128(a)),
                                                      convert<x_u32, y_f64>(lo128(b)))),
                  convert<y_f64, x_u32>(_mm256_div_pd(convert<x_u32, y_f64>(hi128(a)),
                                                      convert<x_u32, y_f64>(hi128(b)))));
#endif
}
Vc_INTRINSIC y_i16 Vc_VDECL divides(y_i16 a, y_i16 b) {
#ifdef Vc_HAVE_AVX512F
    return convert<z_f32, y_i16>(
        _mm512_div_ps(convert<y_i16, z_f32>(a), convert<y_i16, z_f32>(b)));
#else
    return convert<y_f32, y_i16>(
        _mm256_div_ps(convert<x_i16, y_f32>(lo128(a)), convert<x_i16, y_f32>(lo128(b))),
        _mm256_div_ps(convert<x_i16, y_f32>(hi128(a)), convert<x_i16, y_f32>(hi128(b))));
#endif
}
Vc_INTRINSIC y_u16 Vc_VDECL divides(y_u16 a, y_u16 b) {
#ifdef Vc_HAVE_AVX512F
    return convert<z_f32, y_u16>(
        _mm512_div_ps(convert<y_u16, z_f32>(a), convert<y_u16, z_f32>(b)));
#else
    return convert<y_f32, y_u16>(
        _mm256_div_ps(convert<x_u16, y_f32>(lo128(a)), convert<x_u16, y_f32>(lo128(b))),
        _mm256_div_ps(convert<x_u16, y_f32>(hi128(a)), convert<x_u16, y_f32>(hi128(b))));
#endif
}
Vc_INTRINSIC y_i08 Vc_VDECL divides(y_i08 a, y_i08 b) {
#ifdef Vc_HAVE_AVX512F
    return convert<z_f32, y_i08>(
        _mm512_div_ps(convert<x_i08, z_f32>(lo128(a)), convert<x_i08, z_f32>(lo128(b))),
        _mm512_div_ps(convert<x_i08, z_f32>(hi128(a)), convert<x_i08, z_f32>(hi128(b))));
#else
    return convert<y_f32, y_i08>(
        _mm256_div_ps(convert<x_i08, y_f32>(lo128(a)), convert<x_i08, y_f32>(lo128(b))),
        _mm256_div_ps(convert<x_i08, y_f32>(lo128(_mm256_permute4x64_epi64(a, 0x01))),
                      convert<x_i08, y_f32>(lo128(_mm256_permute4x64_epi64(b, 0x01)))),
        _mm256_div_ps(convert<x_i08, y_f32>(lo128(_mm256_permute4x64_epi64(a, 0x02))),
                      convert<x_i08, y_f32>(lo128(_mm256_permute4x64_epi64(b, 0x02)))),
        _mm256_div_ps(convert<x_i08, y_f32>(lo128(_mm256_permute4x64_epi64(a, 0x03))),
                      convert<x_i08, y_f32>(lo128(_mm256_permute4x64_epi64(b, 0x03)))));
#endif
}
Vc_INTRINSIC y_u08 Vc_VDECL divides(y_u08 a, y_u08 b) {
#ifdef Vc_HAVE_AVX512F
    return convert<z_f32, y_u08>(
        _mm512_div_ps(convert<x_u08, z_f32>(lo128(a)), convert<x_u08, z_f32>(lo128(b))),
        _mm512_div_ps(convert<x_u08, z_f32>(hi128(a)), convert<x_u08, z_f32>(hi128(b))));
#else
    return convert<y_f32, y_u08>(
        _mm256_div_ps(convert<x_u08, y_f32>(lo128(a)), convert<x_u08, y_f32>(lo128(b))),
        _mm256_div_ps(convert<x_u08, y_f32>(lo128(_mm256_permute4x64_epi64(a, 0x01))),
                      convert<x_u08, y_f32>(lo128(_mm256_permute4x64_epi64(b, 0x01)))),
        _mm256_div_ps(convert<x_u08, y_f32>(lo128(_mm256_permute4x64_epi64(a, 0x02))),
                      convert<x_u08, y_f32>(lo128(_mm256_permute4x64_epi64(b, 0x02)))),
        _mm256_div_ps(convert<x_u08, y_f32>(lo128(_mm256_permute4x64_epi64(a, 0x03))),
                      convert<x_u08, y_f32>(lo128(_mm256_permute4x64_epi64(b, 0x03)))));
#endif
}
#endif// Vc_HAVE_AVX2

// avx512{{{2
#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC z_f32 Vc_VDECL divides(z_f32 a, z_f32 b) { return _mm512_div_ps(a, b); }
Vc_INTRINSIC z_f64 Vc_VDECL divides(z_f64 a, z_f64 b) { return _mm512_div_pd(a, b); }
Vc_INTRINSIC z_i64 Vc_VDECL divides(z_i64 a, z_i64 b) {
    return {a[0] / b[0], a[1] / b[1], a[2] / b[2], a[3] / b[3],
            a[4] / b[4], a[5] / b[5], a[6] / b[6], a[7] / b[7]};
}
Vc_INTRINSIC z_u64 Vc_VDECL divides(z_u64 a, z_u64 b) {
    return {a[0] / b[0], a[1] / b[1], a[2] / b[2], a[3] / b[3],
            a[4] / b[4], a[5] / b[5], a[6] / b[6], a[7] / b[7]};
}
Vc_INTRINSIC z_i32 Vc_VDECL divides(z_i32 a, z_i32 b) {
    return concat(_mm512_cvttpd_epi32(_mm512_div_pd(_mm512_cvtepi32_pd(lo256(a)),
                                                    _mm512_cvtepi32_pd(lo256(b)))),
                  _mm512_cvttpd_epi32(_mm512_div_pd(_mm512_cvtepi32_pd(hi256(a)),
                                                    _mm512_cvtepi32_pd(hi256(b)))));
}
Vc_INTRINSIC z_u32 Vc_VDECL divides(z_u32 a, z_u32 b) {
    return concat(_mm512_cvttpd_epu32(_mm512_div_pd(_mm512_cvtepu32_pd(lo256(a)),
                                                    _mm512_cvtepu32_pd(lo256(b)))),
                  _mm512_cvttpd_epu32(_mm512_div_pd(_mm512_cvtepu32_pd(hi256(a)),
                                                    _mm512_cvtepu32_pd(hi256(b)))));
}
Vc_INTRINSIC z_i16 Vc_VDECL divides(z_i16 a, z_i16 b) {
    return convert<z_f32, z_i16>(
        _mm512_div_ps(convert<y_i16, z_f32>(lo256(a)), convert<y_i16, z_f32>(lo256(b))),
        _mm512_div_ps(convert<y_i16, z_f32>(hi256(a)), convert<y_i16, z_f32>(hi256(b))));
}
Vc_INTRINSIC z_u16 Vc_VDECL divides(z_u16 a, z_u16 b) {
    return convert<z_f32, z_u16>(
        _mm512_div_ps(convert<y_u16, z_f32>(lo256(a)), convert<y_u16, z_f32>(lo256(b))),
        _mm512_div_ps(convert<y_u16, z_f32>(hi256(a)), convert<y_u16, z_f32>(hi256(b))));
}
Vc_INTRINSIC z_i08 Vc_VDECL divides(z_i08 a, z_i08 b) {
    return convert<z_f32, z_i08>(
        _mm512_div_ps(convert<x_i08, z_f32>(lo128(a)), convert<x_i08, z_f32>(lo128(b))),
        _mm512_div_ps(convert<x_i08, z_f32>(extract128<1>(a)),
                      convert<x_i08, z_f32>(extract128<1>(b))),
        _mm512_div_ps(convert<x_i08, z_f32>(extract128<2>(a)),
                      convert<x_i08, z_f32>(extract128<2>(b))),
        _mm512_div_ps(convert<x_i08, z_f32>(extract128<3>(a)),
                      convert<x_i08, z_f32>(extract128<3>(b))));
}
Vc_INTRINSIC z_u08 Vc_VDECL divides(z_u08 a, z_u08 b) {
    return convert<z_f32, z_u08>(
        _mm512_div_ps(convert<x_u08, z_f32>(lo128(a)), convert<x_u08, z_f32>(lo128(b))),
        _mm512_div_ps(convert<x_u08, z_f32>(extract128<1>(a)),
                      convert<x_u08, z_f32>(extract128<1>(b))),
        _mm512_div_ps(convert<x_u08, z_f32>(extract128<2>(a)),
                      convert<x_u08, z_f32>(extract128<2>(b))),
        _mm512_div_ps(convert<x_u08, z_f32>(extract128<3>(a)),
                      convert<x_u08, z_f32>(extract128<3>(b))));
}
#endif// Vc_HAVE_AVX512F
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES

// modulus{{{1
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <class T, size_t N> Vc_INTRINSIC auto modulus(Storage<T, N> a, Storage<T, N> b)
{
    static_assert(std::is_integral<T>::value, "modulus is only supported for integral types");
    return a.builtin() % b.builtin();
}
#else   // Vc_USE_BUILTIN_VECTOR_TYPES
template <class T, size_t N> Vc_INTRINSIC auto modulus(Storage<T, N> a, Storage<T, N> b)
{
    static_assert(std::is_integral<T>::value, "modulus is only supported for integral types");
    return minus(a, multiplies(divides(a, b), b));
}
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES

// bit_and{{{1
template <class T, size_t N> Vc_INTRINSIC auto bit_and(Storage<T, N> a, Storage<T, N> b)
{
    static_assert(std::is_integral<T>::value, "bit_and is only supported for integral types");
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
    return a.builtin() & b.builtin();
#else   // Vc_USE_BUILTIN_VECTOR_TYPES
    return and_(a, b);
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES
}

// bit_or{{{1
template <class T, size_t N> Vc_INTRINSIC auto bit_or(Storage<T, N> a, Storage<T, N> b)
{
    static_assert(std::is_integral<T>::value, "bit_or is only supported for integral types");
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
    return a.builtin() | b.builtin();
#else   // Vc_USE_BUILTIN_VECTOR_TYPES
    return or_(a, b);
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES
}

// bit_xor{{{1
template <class T, size_t N> Vc_INTRINSIC auto bit_xor(Storage<T, N> a, Storage<T, N> b)
{
    static_assert(std::is_integral<T>::value, "bit_xor is only supported for integral types");
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
    return a.builtin() ^ b.builtin();
#else   // Vc_USE_BUILTIN_VECTOR_TYPES
    return xor_(a, b);
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES
}

// bit_shift_left{{{1
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <class T, size_t N> Vc_INTRINSIC auto bit_shift_left(Storage<T, N> a, Storage<T, N> b)
{
    static_assert(std::is_integral<T>::value, "bit_shift_left is only supported for integral types");
    return a.builtin() << b.builtin();
}
#else   // Vc_USE_BUILTIN_VECTOR_TYPES

// generic scalar fallback
template <class T, size_t N> Vc_INTRINSIC auto bit_shift_left(Storage<T, N> a, Storage<T, N> b)
{
    static_assert(std::is_integral<T>::value, "bit_shift_left is only supported for integral types");
    return generate_from_n_evaluations<N, Storage<T, N>>(
        [&](auto i) { return a[i] << b[i]; });
}

// improvements/specializations with newer instruction set extensions
#ifdef Vc_HAVE_AVX2
Vc_INTRINSIC x_i64 bit_shift_left(x_i64 a, x_i64 b) { return _mm_sllv_epi64(a, b); }
Vc_INTRINSIC x_u64 bit_shift_left(x_u64 a, x_u64 b) { return _mm_sllv_epi64(a, b); }
Vc_INTRINSIC x_i32 bit_shift_left(x_i32 a, x_i32 b) { return _mm_sllv_epi32(a, b); }
Vc_INTRINSIC x_u32 bit_shift_left(x_u32 a, x_u32 b) { return _mm_sllv_epi32(a, b); }

Vc_INTRINSIC y_i64 bit_shift_left(y_i64 a, y_i64 b) { return _mm256_sllv_epi64(a, b); }
Vc_INTRINSIC y_u64 bit_shift_left(y_u64 a, y_u64 b) { return _mm256_sllv_epi64(a, b); }
Vc_INTRINSIC y_i32 bit_shift_left(y_i32 a, y_i32 b) { return _mm256_sllv_epi32(a, b); }
Vc_INTRINSIC y_u32 bit_shift_left(y_u32 a, y_u32 b) { return _mm256_sllv_epi32(a, b); }

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC z_i64 bit_shift_left(z_i64 a, z_i64 b) { return _mm512_sllv_epi64(a, b); }
Vc_INTRINSIC z_u64 bit_shift_left(z_u64 a, z_u64 b) { return _mm512_sllv_epi64(a, b); }
Vc_INTRINSIC z_i32 bit_shift_left(z_i32 a, z_i32 b) { return _mm512_sllv_epi32(a, b); }
Vc_INTRINSIC z_u32 bit_shift_left(z_u32 a, z_u32 b) { return _mm512_sllv_epi32(a, b); }
#endif  // Vc_HAVE_AVX512F

#ifdef Vc_HAVE_AVX512BW
Vc_INTRINSIC z_i16 bit_shift_left(z_i16 a, z_i16 b) { return _mm512_sllv_epi16(a, b); }
Vc_INTRINSIC z_u16 bit_shift_left(z_u16 a, z_u16 b) { return _mm512_sllv_epi16(a, b); }
Vc_INTRINSIC y_i08 bit_shift_left(y_i08 a, y_i08 b) { return _mm512_cvtepi16_epi8(_mm512_sllv_epi16(_mm512_cvtepi8_epi16(a), _mm512_cvtepi8_epi16(b))); }
Vc_INTRINSIC y_u08 bit_shift_left(y_u08 a, y_u08 b) { return _mm512_cvtepi16_epi8(_mm512_sllv_epi16(_mm512_cvtepi8_epi16(a), _mm512_cvtepi8_epi16(b))); }
Vc_INTRINSIC z_i08 bit_shift_left(z_i08 a, z_i08 b) { return concat(bit_shift_left(lo256(a), lo256(b)), bit_shift_left(hi256(a), hi256(b))); }
Vc_INTRINSIC z_u08 bit_shift_left(z_u08 a, z_u08 b) { return concat(bit_shift_left(lo256(a), lo256(b)), bit_shift_left(hi256(a), hi256(b))); }
#ifdef Vc_HAVE_AVX512VL
Vc_INTRINSIC y_i16 bit_shift_left(y_i16 a, y_i16 b) { return _mm256_sllv_epi16(a, b); }
Vc_INTRINSIC y_u16 bit_shift_left(y_u16 a, y_u16 b) { return _mm256_sllv_epi16(a, b); }
Vc_INTRINSIC x_i16 bit_shift_left(x_i16 a, x_i16 b) { return _mm_sllv_epi16(a, b); }
Vc_INTRINSIC x_u16 bit_shift_left(x_u16 a, x_u16 b) { return _mm_sllv_epi16(a, b); }
Vc_INTRINSIC x_i08 bit_shift_left(x_i08 a, x_i08 b) { return _mm256_cvtepi16_epi8(_mm256_sllv_epi16(_mm256_cvtepi8_epi16(a), _mm256_cvtepi8_epi16(b))); }
Vc_INTRINSIC x_u08 bit_shift_left(x_u08 a, x_u08 b) { return _mm256_cvtepi16_epi8(_mm256_sllv_epi16(_mm256_cvtepi8_epi16(a), _mm256_cvtepi8_epi16(b))); }
#else   // Vc_HAVE_AVX512VL
Vc_INTRINSIC y_i16 bit_shift_left(y_i16 a, y_i16 b) { return lo256(_mm512_sllv_epi16(intrin_cast<__m512i>(a), intrin_cast<__m512i>(b)); }
Vc_INTRINSIC y_u16 bit_shift_left(y_u16 a, y_u16 b) { return lo256(_mm512_sllv_epi16(intrin_cast<__m512i>(a), intrin_cast<__m512i>(b)); }
Vc_INTRINSIC x_i16 bit_shift_left(x_i16 a, x_i16 b) { return lo128(_mm512_sllv_epi16(intrin_cast<__m512i>(a), intrin_cast<__m512i>(b)); }
Vc_INTRINSIC x_u16 bit_shift_left(x_u16 a, x_u16 b) { return lo128(_mm512_sllv_epi16(intrin_cast<__m512i>(a), intrin_cast<__m512i>(b)); }
Vc_INTRINSIC x_i08 bit_shift_left(x_i08 a, x_i08 b) { return lo128(bit_shift_left(y_i08(intrin_cast<__m256i>(a)), y_i08(intrin_cast<__m256i>(b)))); }
Vc_INTRINSIC x_u08 bit_shift_left(x_u08 a, x_u08 b) { return lo128(bit_shift_left(y_u08(intrin_cast<__m256i>(a)), y_u08(intrin_cast<__m256i>(b)))); }
#endif  // Vc_HAVE_AVX512VL
#else   // Vc_HAVE_AVX512BW
Vc_INTRINSIC x_i16 bit_shift_left(x_i16 a, x_i16 b)
{
    // left shift into or over the sign bit is UB => OR suffices
    return or_(_mm_sllv_epi32(a, and_(b, broadcast16(0x0000ffffu))),
               _mm_sllv_epi32(and_(a, broadcast16(0xffff0000u)), _mm_srli_epi32(b, 16)));
}
Vc_INTRINSIC x_u16 bit_shift_left(x_u16 a, x_u16 b)
{
    return _mm_blend_epi16(
        _mm_sllv_epi32(a, and_(b, broadcast16(0x0000ffffu))),
        _mm_sllv_epi32(and_(a, broadcast16(0xffff0000u)), _mm_srli_epi32(b, 16)), 0xaa);
}
Vc_INTRINSIC x_i08 bit_shift_left(x_i08 a, x_i08 b)
{
    // exploit UB: The behavior is undefined if the right operand is [...] greater than or
    // equal to the length in bits of the promoted left operand.
    const auto mask1 = broadcast16(0x00ff00ffu);
    const auto mask2 = broadcast16(0xff00ff00u);
    const auto a1 = and_(a, mask1);
    const auto a2 = and_(a, mask2);
    const auto s1 = _mm_sllv_epi32(a1,                and_(b, broadcast16(0x000000ffu))     );
    const auto s2 = _mm_sllv_epi32(a2, _mm_srli_epi32(and_(b, broadcast16(0x0000ff00u)),  8));
    const auto s3 = _mm_sllv_epi32(a1, _mm_srli_epi32(and_(b, broadcast16(0x00ff0000u)), 16));
    const auto s4 = _mm_sllv_epi32(a2, _mm_srli_epi32(     b,                            24));
    return _mm_blendv_epi8(or_(s1, s3), or_(s2, s4), mask2);
}
Vc_INTRINSIC x_u08 bit_shift_left(x_u08 a, x_u08 b)
{
    // exploit UB: The behavior is undefined if the right operand is [...] greater than or
    // equal to the length in bits of the promoted left operand.
    const auto mask1 = broadcast16(0x00ff00ffu);
    const auto mask2 = broadcast16(0xff00ff00u);
    const auto a1 = and_(a, mask1);
    const auto a2 = and_(a, mask2);
    const auto s1 = _mm_sllv_epi32(a1,                and_(b, broadcast16(0x000000ffu))     );
    const auto s2 = _mm_sllv_epi32(a2, _mm_srli_epi32(and_(b, broadcast16(0x0000ff00u)),  8));
    const auto s3 = _mm_sllv_epi32(a1, _mm_srli_epi32(and_(b, broadcast16(0x00ff0000u)), 16));
    const auto s4 = _mm_sllv_epi32(a2, _mm_srli_epi32(     b,                            24));
    return _mm_blendv_epi8(or_(s1, s3), or_(s2, s4), mask2);
}

Vc_INTRINSIC y_i16 bit_shift_left(y_i16 a, y_i16 b)
{
    // left shift into or over the sign bit is UB => OR suffices
    return or_(
        _mm256_sllv_epi32(a, and_(b, broadcast32(0x0000ffffu))),
        _mm256_sllv_epi32(and_(a, broadcast32(0xffff0000u)), _mm256_srli_epi32(b, 16)));
}
Vc_INTRINSIC y_u16 bit_shift_left(y_u16 a, y_u16 b)
{
    return _mm256_blend_epi16(
        _mm256_sllv_epi32(a, and_(b, broadcast32(0x0000ffffu))),
        _mm256_sllv_epi32(and_(a, broadcast32(0xffff0000u)), _mm256_srli_epi32(b, 16)),
        0xaa);
}
Vc_INTRINSIC y_i08 bit_shift_left(y_i08 a, y_i08 b)
{
    // exploit UB: The behavior is undefined if the right operand is [...] greater than or
    // equal to the length in bits of the promoted left operand.
    const auto mask1 = broadcast32(0x00ff00ffu);
    const auto mask2 = broadcast32(0xff00ff00u);
    const auto a1 = and_(a, mask1);
    const auto a2 = and_(a, mask2);
    const auto s1 = _mm256_sllv_epi32(a1,                   and_(b, broadcast32(0x000000ffu))     );
    const auto s2 = _mm256_sllv_epi32(a2, _mm256_srli_epi32(and_(b, broadcast32(0x0000ff00u)),  8));
    const auto s3 = _mm256_sllv_epi32(a1, _mm256_srli_epi32(and_(b, broadcast32(0x00ff0000u)), 16));
    const auto s4 = _mm256_sllv_epi32(a2, _mm256_srli_epi32(     b,                            24));
    return _mm256_blendv_epi8(or_(s1, s3), or_(s2, s4), mask2);
}
Vc_INTRINSIC y_u08 bit_shift_left(y_u08 a, y_u08 b)
{
    // exploit UB: The behavior is undefined if the right operand is [...] greater than or
    // equal to the length in bits of the promoted left operand.
    const auto mask1 = broadcast32(0x00ff00ffu);
    const auto mask2 = broadcast32(0xff00ff00u);
    const auto a1 = and_(a, mask1);
    const auto a2 = and_(a, mask2);
    const auto s1 = _mm256_sllv_epi32(a1,                   and_(b, broadcast32(0x000000ffu))     );
    const auto s2 = _mm256_sllv_epi32(a2, _mm256_srli_epi32(and_(b, broadcast32(0x0000ff00u)),  8));
    const auto s3 = _mm256_sllv_epi32(a1, _mm256_srli_epi32(and_(b, broadcast32(0x00ff0000u)), 16));
    const auto s4 = _mm256_sllv_epi32(a2, _mm256_srli_epi32(     b,                            24));
    return _mm256_blendv_epi8(or_(s1, s3), or_(s2, s4), mask2);
}

/* Not actually part of datapar_abi::avx512:
Vc_INTRINSIC z_i16 bit_shift_left(z_i16 a, z_i16 b) { return or_(_mm512_sllv_epi32(and_(a, broadcast64(0x0000ffffu)), and_(b, broadcast64(0x0000ffffu))), _mm512_sllv_epi32(and_(a, broadcast64(0xffff0000u)), _mm512_srli_epi32(b, 16))); }
Vc_INTRINSIC z_u16 bit_shift_left(z_u16 a, z_u16 b) { return or_(_mm512_sllv_epi32(and_(a, broadcast64(0x0000ffffu)), and_(b, broadcast64(0x0000ffffu))), _mm512_sllv_epi32(and_(a, broadcast64(0xffff0000u)), _mm512_srli_epi32(b, 16))); }
Vc_INTRINSIC z_i08 bit_shift_left(z_i08 a, z_i08 b)
{
    return or_(
        or_(_mm512_sllv_epi32(and_(a, broadcast64(0x000000ffu)),
                              and_(b, broadcast64(0x000000ffu))),
            _mm512_sllv_epi32(and_(a, broadcast64(0x0000ff00u)),
                              _mm512_srli_epi32(and_(b, broadcast64(0x0000ff00u)), 8))),
        or_(_mm512_sllv_epi32(and_(a, broadcast64(0x00ff0000u)),
                              _mm512_srli_epi32(and_(b, broadcast64(0x00ff0000u)), 16)),
            _mm512_sllv_epi32(and_(a, broadcast64(0xff000000u)),
                              _mm512_srli_epi32(b, 24))));
}
Vc_INTRINSIC z_u08 bit_shift_left(z_u08 a, z_u08 b) { return z_u08(bit_shift_left(z_i08(a), z_i08(b))); }
*/
#endif  // Vc_HAVE_AVX512BW
#endif  // Vc_HAVE_AVX2
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES

// bit_shift_right{{{1
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <class T, size_t N> Vc_INTRINSIC auto bit_shift_right(Storage<T, N> a, Storage<T, N> b)
{
    static_assert(std::is_integral<T>::value, "bit_shift_right is only supported for integral types");
    return a.builtin() >> b.builtin();
}
#else   // Vc_USE_BUILTIN_VECTOR_TYPES

// generic scalar fallback
template <class T, size_t N> Vc_INTRINSIC auto bit_shift_right(Storage<T, N> a, Storage<T, N> b)
{
    static_assert(std::is_integral<T>::value, "bit_shift_right is only supported for integral types");
    return generate_from_n_evaluations<N, Storage<T, N>>(
        [&](auto i) { return a[i] >> b[i]; });
}

// improvements/specializations with newer instruction set extensions
#ifdef Vc_HAVE_AVX2
Vc_INTRINSIC x_i32 bit_shift_right(x_i32 a, x_i32 b) { return _mm_srav_epi32(a, b); }
Vc_INTRINSIC x_u32 bit_shift_right(x_u32 a, x_u32 b) { return _mm_srlv_epi32(a, b); }

Vc_INTRINSIC y_i32 bit_shift_right(y_i32 a, y_i32 b) { return _mm256_srav_epi32(a, b); }
Vc_INTRINSIC y_u32 bit_shift_right(y_u32 a, y_u32 b) { return _mm256_srlv_epi32(a, b); }

#ifdef Vc_HAVE_AVX512VL
Vc_INTRINSIC x_i64 bit_shift_right(x_i64 a, x_i64 b) { return _mm_srav_epi64(a, b); }
Vc_INTRINSIC x_u64 bit_shift_right(x_u64 a, x_u64 b) { return _mm_srlv_epi64(a, b); }

Vc_INTRINSIC y_i64 bit_shift_right(y_i64 a, y_i64 b) { return _mm256_srav_epi64(a, b); }
Vc_INTRINSIC y_u64 bit_shift_right(y_u64 a, y_u64 b) { return _mm256_srlv_epi64(a, b); }
#endif  // Vc_HAVE_AVX512VL

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC z_i64 bit_shift_right(z_i64 a, z_i64 b) { return _mm512_srav_epi64(a, b); }
Vc_INTRINSIC z_u64 bit_shift_right(z_u64 a, z_u64 b) { return _mm512_srlv_epi64(a, b); }
Vc_INTRINSIC z_i32 bit_shift_right(z_i32 a, z_i32 b) { return _mm512_srav_epi32(a, b); }
Vc_INTRINSIC z_u32 bit_shift_right(z_u32 a, z_u32 b) { return _mm512_srlv_epi32(a, b); }
#endif  // Vc_HAVE_AVX512F

#ifdef Vc_HAVE_AVX512BW
Vc_INTRINSIC z_i16 bit_shift_right(z_i16 a, z_i16 b) { return _mm512_srav_epi16(a, b); }
Vc_INTRINSIC z_u16 bit_shift_right(z_u16 a, z_u16 b) { return _mm512_srlv_epi16(a, b); }
Vc_INTRINSIC y_i08 bit_shift_right(y_i08 a, y_i08 b) { return _mm512_cvtepi16_epi8(_mm512_srav_epi16(_mm512_cvtepi8_epi16(a), _mm512_cvtepi8_epi16(b))); }
Vc_INTRINSIC y_u08 bit_shift_right(y_u08 a, y_u08 b) { return _mm512_cvtepi16_epi8(_mm512_srlv_epi16(_mm512_cvtepi8_epi16(a), _mm512_cvtepi8_epi16(b))); }
Vc_INTRINSIC z_i08 bit_shift_right(z_i08 a, z_i08 b) { return concat(bit_shift_right(lo256(a), lo256(b)), bit_shift_right(hi256(a), hi256(b))); }
Vc_INTRINSIC z_u08 bit_shift_right(z_u08 a, z_u08 b) { return concat(bit_shift_right(lo256(a), lo256(b)), bit_shift_right(hi256(a), hi256(b))); }
#ifdef Vc_HAVE_AVX512VL
Vc_INTRINSIC y_i16 bit_shift_right(y_i16 a, y_i16 b) { return _mm256_srav_epi16(a, b); }
Vc_INTRINSIC y_u16 bit_shift_right(y_u16 a, y_u16 b) { return _mm256_srlv_epi16(a, b); }
Vc_INTRINSIC x_i16 bit_shift_right(x_i16 a, x_i16 b) { return _mm_srav_epi16(a, b); }
Vc_INTRINSIC x_u16 bit_shift_right(x_u16 a, x_u16 b) { return _mm_srlv_epi16(a, b); }
Vc_INTRINSIC x_i08 bit_shift_right(x_i08 a, x_i08 b) { return _mm256_cvtepi16_epi8(_mm256_srav_epi16(_mm256_cvtepi8_epi16(a), _mm256_cvtepi8_epi16(b))); }
Vc_INTRINSIC x_u08 bit_shift_right(x_u08 a, x_u08 b) { return _mm256_cvtepi16_epi8(_mm256_srlv_epi16(_mm256_cvtepi8_epi16(a), _mm256_cvtepi8_epi16(b))); }
#else   // Vc_HAVE_AVX512VL
Vc_INTRINSIC y_i16 bit_shift_right(y_i16 a, y_i16 b) { return lo256(_mm512_srav_epi16(intrin_cast<__m512i>(a), intrin_cast<__m512i>(b)); }
Vc_INTRINSIC y_u16 bit_shift_right(y_u16 a, y_u16 b) { return lo256(_mm512_srlv_epi16(intrin_cast<__m512i>(a), intrin_cast<__m512i>(b)); }
Vc_INTRINSIC x_i16 bit_shift_right(x_i16 a, x_i16 b) { return lo128(_mm512_srav_epi16(intrin_cast<__m512i>(a), intrin_cast<__m512i>(b)); }
Vc_INTRINSIC x_u16 bit_shift_right(x_u16 a, x_u16 b) { return lo128(_mm512_srlv_epi16(intrin_cast<__m512i>(a), intrin_cast<__m512i>(b)); }
Vc_INTRINSIC x_i08 bit_shift_right(x_i08 a, x_i08 b) { return lo128(bit_shift_right(y_i08(intrin_cast<__m256i>(a)), y_i08(intrin_cast<__m256i>(b)))); }
Vc_INTRINSIC x_u08 bit_shift_right(x_u08 a, x_u08 b) { return lo128(bit_shift_right(y_u08(intrin_cast<__m256i>(a)), y_u08(intrin_cast<__m256i>(b)))); }
#endif  // Vc_HAVE_AVX512VL
#else   // Vc_HAVE_AVX512BW
Vc_INTRINSIC x_i16 bit_shift_right(x_i16 a, x_i16 b)
{
    return _mm_blend_epi16(_mm_srav_epi32(and_(a, broadcast16(0x0000ffffu)),
                                          and_(b, broadcast16(0x0000ffffu))),
                           _mm_srav_epi32(a, _mm_srli_epi32(b, 16)), 0xaa);
}
Vc_INTRINSIC x_u16 bit_shift_right(x_u16 a, x_u16 b)
{
    return _mm_blend_epi16(_mm_srlv_epi32(and_(a, broadcast16(0x0000ffffu)),
                                          and_(b, broadcast16(0x0000ffffu))),
                           _mm_srlv_epi32(a, _mm_srli_epi32(b, 16)), 0xaa);
}
Vc_INTRINSIC x_i08 bit_shift_right(x_i08 a, x_i08 b)
{
    // exploit UB: The behavior is undefined if the right operand is [...] greater than or
    // equal to the length in bits of the promoted left operand.
    const auto mask1 = broadcast16(0x00ff00ffu);
    const auto mask2 = broadcast16(0xff00ff00u);
    const auto a1 = and_(a, mask1);
    const auto a2 = and_(a, mask2);
    const auto s1 = _mm_srav_epi32(a1,                and_(b, broadcast16(0x000000ffu))     );
    const auto s2 = _mm_srav_epi32(a2, _mm_srli_epi32(and_(b, broadcast16(0x0000ff00u)),  8));
    const auto s3 = _mm_srav_epi32(a1, _mm_srli_epi32(and_(b, broadcast16(0x00ff0000u)), 16));
    const auto s4 = _mm_srav_epi32(a2, _mm_srli_epi32(     b,                            24));
    return _mm_blendv_epi8(or_(s1, s3), or_(s2, s4), mask2);
}
Vc_INTRINSIC x_u08 bit_shift_right(x_u08 a, x_u08 b)
{
    // exploit UB: The behavior is undefined if the right operand is [...] greater than or
    // equal to the length in bits of the promoted left operand.
    const auto mask1 = broadcast16(0x00ff00ffu);
    const auto mask2 = broadcast16(0xff00ff00u);
    const auto a1 = and_(a, mask1);
    const auto a2 = and_(a, mask2);
    const auto s1 = _mm_srlv_epi32(a1,                and_(b, broadcast16(0x000000ffu))     );
    const auto s2 = _mm_srlv_epi32(a2, _mm_srli_epi32(and_(b, broadcast16(0x0000ff00u)),  8));
    const auto s3 = _mm_srlv_epi32(a1, _mm_srli_epi32(and_(b, broadcast16(0x00ff0000u)), 16));
    const auto s4 = _mm_srlv_epi32(a2, _mm_srli_epi32(     b,                            24));
    return _mm_blendv_epi8(or_(s1, s3), or_(s2, s4), mask2);
}

Vc_INTRINSIC y_i16 bit_shift_right(y_i16 a, y_i16 b)
{
    return _mm256_blend_epi16(_mm256_srav_epi32(and_(a, broadcast32(0x0000ffffu)),
                                                and_(b, broadcast32(0x0000ffffu))),
                              _mm256_srav_epi32(a, _mm256_srli_epi32(b, 16)), 0xaa);
}
Vc_INTRINSIC y_u16 bit_shift_right(y_u16 a, y_u16 b)
{
    return _mm256_blend_epi16(_mm256_srlv_epi32(and_(a, broadcast32(0x0000ffffu)),
                                                and_(b, broadcast32(0x0000ffffu))),
                              _mm256_srlv_epi32(a, _mm256_srli_epi32(b, 16)), 0xaa);
}
Vc_INTRINSIC y_i08 bit_shift_right(y_i08 a, y_i08 b)
{
    // exploit UB: The behavior is undefined if the right operand is [...] greater than or
    // equal to the length in bits of the promoted left operand.
    const auto mask1 = broadcast32(0x00ff00ffu);
    const auto mask2 = broadcast32(0xff00ff00u);
    const auto a1 = and_(a, mask1);
    const auto a2 = and_(a, mask2);
    const auto s1 = _mm256_srav_epi32(a1,                   and_(b, broadcast32(0x000000ffu))     );
    const auto s2 = _mm256_srav_epi32(a2, _mm256_srli_epi32(and_(b, broadcast32(0x0000ff00u)),  8));
    const auto s3 = _mm256_srav_epi32(a1, _mm256_srli_epi32(and_(b, broadcast32(0x00ff0000u)), 16));
    const auto s4 = _mm256_srav_epi32(a2, _mm256_srli_epi32(     b,                            24));
    return _mm256_blendv_epi8(or_(s1, s3), or_(s2, s4), mask2);
}
Vc_INTRINSIC y_u08 bit_shift_right(y_u08 a, y_u08 b)
{
    // exploit UB: The behavior is undefined if the right operand is [...] greater than or
    // equal to the length in bits of the promoted left operand.
    const auto mask1 = broadcast32(0x00ff00ffu);
    const auto mask2 = broadcast32(0xff00ff00u);
    const auto a1 = and_(a, mask1);
    const auto a2 = and_(a, mask2);
    const auto s1 = _mm256_srlv_epi32(a1,                   and_(b, broadcast32(0x000000ffu))     );
    const auto s2 = _mm256_srlv_epi32(a2, _mm256_srli_epi32(and_(b, broadcast32(0x0000ff00u)),  8));
    const auto s3 = _mm256_srlv_epi32(a1, _mm256_srli_epi32(and_(b, broadcast32(0x00ff0000u)), 16));
    const auto s4 = _mm256_srlv_epi32(a2, _mm256_srli_epi32(     b,                            24));
    return _mm256_blendv_epi8(or_(s1, s3), or_(s2, s4), mask2);
}

/* Not actually part of datapar_abi::avx512:
Vc_INTRINSIC z_i16 bit_shift_right(z_i16 a, z_i16 b) { return or_(_mm512_sllv_epi32(and_(a, broadcast64(0x0000ffffu)), and_(b, broadcast64(0x0000ffffu))), _mm512_sllv_epi32(and_(a, broadcast64(0xffff0000u)), _mm512_srli_epi32(b, 16))); }
Vc_INTRINSIC z_u16 bit_shift_right(z_u16 a, z_u16 b) { return or_(_mm512_sllv_epi32(and_(a, broadcast64(0x0000ffffu)), and_(b, broadcast64(0x0000ffffu))), _mm512_sllv_epi32(and_(a, broadcast64(0xffff0000u)), _mm512_srli_epi32(b, 16))); }
Vc_INTRINSIC z_i08 bit_shift_right(z_i08 a, z_i08 b)
{
    return or_(
        or_(_mm512_sllv_epi32(and_(a, broadcast64(0x000000ffu)),
                              and_(b, broadcast64(0x000000ffu))),
            _mm512_sllv_epi32(and_(a, broadcast64(0x0000ff00u)),
                              _mm512_srli_epi32(and_(b, broadcast64(0x0000ff00u)), 8))),
        or_(_mm512_sllv_epi32(and_(a, broadcast64(0x00ff0000u)),
                              _mm512_srli_epi32(and_(b, broadcast64(0x00ff0000u)), 16)),
            _mm512_sllv_epi32(and_(a, broadcast64(0xff000000u)),
                              _mm512_srli_epi32(b, 24))));
}
Vc_INTRINSIC z_u08 bit_shift_right(z_u08 a, z_u08 b) { return z_u08(bit_shift_right(z_i08(a), z_i08(b))); }
*/
#endif  // Vc_HAVE_AVX512BW
#endif  // Vc_HAVE_AVX2
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES

// complement{{{1
template <typename T> Vc_INTRINSIC auto Vc_VDECL complement(T v) {
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
    return ~v.builtin();
#else
    return not_(v);
#endif
}

//}}}1
// unary_minus{{{1
template <typename T> Vc_INTRINSIC auto Vc_VDECL unary_minus(T v) { return minus(T{}, v); }
Vc_INTRINSIC __m128  Vc_VDECL unary_minus(x_f32 v) { return xor_(v, signmask16(float())); }
#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC __m128d Vc_VDECL unary_minus(x_f64 v) { return xor_(v, signmask16(double())); }
#endif  // Vc_HAVE_SSE2
#ifdef Vc_HAVE_SSSE3
Vc_INTRINSIC __m128i Vc_VDECL unary_minus(x_i32 v) { return _mm_sign_epi32(v, allone<__m128i>()); }
Vc_INTRINSIC __m128i Vc_VDECL unary_minus(x_u32 v) { return _mm_sign_epi32(v, allone<__m128i>()); }
Vc_INTRINSIC __m128i Vc_VDECL unary_minus(x_i16 v) { return _mm_sign_epi16(v, allone<__m128i>()); }
Vc_INTRINSIC __m128i Vc_VDECL unary_minus(x_u16 v) { return _mm_sign_epi16(v, allone<__m128i>()); }
Vc_INTRINSIC __m128i Vc_VDECL unary_minus(x_i08 v) { return _mm_sign_epi8 (v, allone<__m128i>()); }
Vc_INTRINSIC __m128i Vc_VDECL unary_minus(x_u08 v) { return _mm_sign_epi8 (v, allone<__m128i>()); }
#endif  // Vc_HAVE_SSSE3

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC __m256  Vc_VDECL unary_minus(y_f32 v) { return xor_(v, signmask32(float())); }
Vc_INTRINSIC __m256d Vc_VDECL unary_minus(y_f64 v) { return xor_(v, signmask32(double())); }
#endif  // Vc_HAVE_AVX
#ifdef Vc_HAVE_AVX2
Vc_INTRINSIC __m256i Vc_VDECL unary_minus(y_i32 v) { return _mm256_sign_epi32(v, allone<__m256i>()); }
Vc_INTRINSIC __m256i Vc_VDECL unary_minus(y_u32 v) { return _mm256_sign_epi32(v, allone<__m256i>()); }
Vc_INTRINSIC __m256i Vc_VDECL unary_minus(y_i16 v) { return _mm256_sign_epi16(v, allone<__m256i>()); }
Vc_INTRINSIC __m256i Vc_VDECL unary_minus(y_u16 v) { return _mm256_sign_epi16(v, allone<__m256i>()); }
Vc_INTRINSIC __m256i Vc_VDECL unary_minus(y_i08 v) { return _mm256_sign_epi8 (v, allone<__m256i>()); }
Vc_INTRINSIC __m256i Vc_VDECL unary_minus(y_u08 v) { return _mm256_sign_epi8 (v, allone<__m256i>()); }
#endif  // Vc_HAVE_AVX2

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC __m512  Vc_VDECL unary_minus(z_f32 v) { return xor_(v, signmask64(float())); }
Vc_INTRINSIC __m512d Vc_VDECL unary_minus(z_f64 v) { return xor_(v, signmask64(double())); }
#endif  // Vc_HAVE_AVX512F

//}}}1
}}  // namespace detail::x86
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DATAPAR_X86_ARITHMETICS_H_
