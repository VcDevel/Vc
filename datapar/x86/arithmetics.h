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

namespace Vc_VERSIONED_NAMESPACE::detail::x86
{
// plus{{{1
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <class T, size_t N> Vc_INTRINSIC auto plus(Storage<T, N> a, Storage<T, N> b)
{
    return a.builtin() + b.builtin();
}
#else   // Vc_USE_BUILTIN_VECTOR_TYPES
Vc_INTRINSIC __m128  plus(x_f32 a, x_f32 b) { return _mm_add_ps(a, b); }
Vc_INTRINSIC __m128d plus(x_f64 a, x_f64 b) { return _mm_add_pd(a, b); }
Vc_INTRINSIC __m128i plus(x_i64 a, x_i64 b) { return _mm_add_epi64(a, b); }
Vc_INTRINSIC __m128i plus(x_u64 a, x_u64 b) { return _mm_add_epi64(a, b); }
Vc_INTRINSIC __m128i plus(x_i32 a, x_i32 b) { return _mm_add_epi32(a, b); }
Vc_INTRINSIC __m128i plus(x_u32 a, x_u32 b) { return _mm_add_epi32(a, b); }
Vc_INTRINSIC __m128i plus(x_i16 a, x_i16 b) { return _mm_add_epi16(a, b); }
Vc_INTRINSIC __m128i plus(x_u16 a, x_u16 b) { return _mm_add_epi16(a, b); }
Vc_INTRINSIC __m128i plus(x_i08 a, x_i08 b) { return _mm_add_epi8 (a, b); }
Vc_INTRINSIC __m128i plus(x_u08 a, x_u08 b) { return _mm_add_epi8 (a, b); }

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC __m256  plus(y_f32 a, y_f32 b) { return _mm256_add_ps(a, b); }
Vc_INTRINSIC __m256d plus(y_f64 a, y_f64 b) { return _mm256_add_pd(a, b); }
#endif  // Vc_HAVE_AVX
#ifdef Vc_HAVE_AVX2
Vc_INTRINSIC __m256i plus(y_i64 a, y_i64 b) { return _mm256_add_epi64(a, b); }
Vc_INTRINSIC __m256i plus(y_u64 a, y_u64 b) { return _mm256_add_epi64(a, b); }
Vc_INTRINSIC __m256i plus(y_i32 a, y_i32 b) { return _mm256_add_epi32(a, b); }
Vc_INTRINSIC __m256i plus(y_u32 a, y_u32 b) { return _mm256_add_epi32(a, b); }
Vc_INTRINSIC __m256i plus(y_i16 a, y_i16 b) { return _mm256_add_epi16(a, b); }
Vc_INTRINSIC __m256i plus(y_u16 a, y_u16 b) { return _mm256_add_epi16(a, b); }
Vc_INTRINSIC __m256i plus(y_i08 a, y_i08 b) { return _mm256_add_epi8 (a, b); }
Vc_INTRINSIC __m256i plus(y_u08 a, y_u08 b) { return _mm256_add_epi8 (a, b); }
#endif  // Vc_HAVE_AVX2
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES

// minus{{{1
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <class T, size_t N> Vc_INTRINSIC auto minus(Storage<T, N> a, Storage<T, N> b)
{
    return a.builtin() - b.builtin();
}
#else   // Vc_USE_BUILTIN_VECTOR_TYPES
Vc_INTRINSIC __m128  minus(x_f32 a, x_f32 b) { return _mm_sub_ps(a, b); }
Vc_INTRINSIC __m128d minus(x_f64 a, x_f64 b) { return _mm_sub_pd(a, b); }
Vc_INTRINSIC __m128i minus(x_i64 a, x_i64 b) { return _mm_sub_epi64(a, b); }
Vc_INTRINSIC __m128i minus(x_u64 a, x_u64 b) { return _mm_sub_epi64(a, b); }
Vc_INTRINSIC __m128i minus(x_i32 a, x_i32 b) { return _mm_sub_epi32(a, b); }
Vc_INTRINSIC __m128i minus(x_u32 a, x_u32 b) { return _mm_sub_epi32(a, b); }
Vc_INTRINSIC __m128i minus(x_i16 a, x_i16 b) { return _mm_sub_epi16(a, b); }
Vc_INTRINSIC __m128i minus(x_u16 a, x_u16 b) { return _mm_sub_epi16(a, b); }
Vc_INTRINSIC __m128i minus(x_i08 a, x_i08 b) { return _mm_sub_epi8 (a, b); }
Vc_INTRINSIC __m128i minus(x_u08 a, x_u08 b) { return _mm_sub_epi8 (a, b); }

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC __m256  minus(y_f32 a, y_f32 b) { return _mm256_sub_ps(a, b); }
Vc_INTRINSIC __m256d minus(y_f64 a, y_f64 b) { return _mm256_sub_pd(a, b); }
#endif  // Vc_HAVE_AVX
#ifdef Vc_HAVE_AVX2
Vc_INTRINSIC __m256i minus(y_i64 a, y_i64 b) { return _mm256_sub_epi64(a, b); }
Vc_INTRINSIC __m256i minus(y_u64 a, y_u64 b) { return _mm256_sub_epi64(a, b); }
Vc_INTRINSIC __m256i minus(y_i32 a, y_i32 b) { return _mm256_sub_epi32(a, b); }
Vc_INTRINSIC __m256i minus(y_u32 a, y_u32 b) { return _mm256_sub_epi32(a, b); }
Vc_INTRINSIC __m256i minus(y_i16 a, y_i16 b) { return _mm256_sub_epi16(a, b); }
Vc_INTRINSIC __m256i minus(y_u16 a, y_u16 b) { return _mm256_sub_epi16(a, b); }
Vc_INTRINSIC __m256i minus(y_i08 a, y_i08 b) { return _mm256_sub_epi8 (a, b); }
Vc_INTRINSIC __m256i minus(y_u08 a, y_u08 b) { return _mm256_sub_epi8 (a, b); }
#endif  // Vc_HAVE_AVX2
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES

// multiplies{{{1
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <class T, size_t N> Vc_INTRINSIC auto multiplies(Storage<T, N> a, Storage<T, N> b)
{
    return a.builtin() * b.builtin();
}
#else   // Vc_USE_BUILTIN_VECTOR_TYPES
Vc_INTRINSIC __m128  multiplies(x_f32 a, x_f32 b) { return _mm_mul_ps(a, b); }
Vc_INTRINSIC __m128d multiplies(x_f64 a, x_f64 b) { return _mm_mul_pd(a, b); }
Vc_INTRINSIC __m128i multiplies(x_i64 a, x_i64 b) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm_mullo_epi64(a, b);
#else
    return x_i64{a[0] * b[0], a[1] * b[1]};
#endif
}
Vc_INTRINSIC __m128i multiplies(x_u64 a, x_u64 b) { return multiplies(x_i64(a), x_i64(b)); }
Vc_INTRINSIC __m128i multiplies(x_i32 a, x_i32 b) {
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
Vc_INTRINSIC __m128i multiplies(x_u32 a, x_u32 b) { return multiplies(x_i32(a), x_i32(b)); }
Vc_INTRINSIC __m128i multiplies(x_i16 a, x_i16 b) { return _mm_mullo_epi16(a, b); }
Vc_INTRINSIC __m128i multiplies(x_u16 a, x_u16 b) { return _mm_mullo_epi16(a, b); }
Vc_INTRINSIC __m128i multiplies(x_i08 a, x_i08 b) {
    return or_(
        and_(_mm_mullo_epi16(a, b), _mm_slli_epi16(allone<__m128i>(), 8)),
        _mm_slli_epi16(_mm_mullo_epi16(_mm_srli_si128(a, 1), _mm_srli_si128(b, 1)), 8));
}
Vc_INTRINSIC __m128i multiplies(x_u08 a, x_u08 b) { return multiplies(x_i08(a), x_i08(b)); }
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES

// divides{{{1
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <class T, size_t N> Vc_INTRINSIC auto divides(Storage<T, N> a, Storage<T, N> b)
{
    return a.builtin() / b.builtin();
}
#else   // Vc_USE_BUILTIN_VECTOR_TYPES
Vc_INTRINSIC x_f32 divides(x_f32 a, x_f32 b) { return _mm_div_ps(a, b); }
Vc_INTRINSIC x_f64 divides(x_f64 a, x_f64 b) { return _mm_div_pd(a, b); }
Vc_INTRINSIC x_i64 divides(x_i64 a, x_i64 b) { return {a[0] / b[0], a[1] / b[1]}; }
Vc_INTRINSIC x_u64 divides(x_u64 a, x_u64 b) { return {a[0] / b[0], a[1] / b[1]}; }
Vc_INTRINSIC x_i32 divides(x_i32 a, x_i32 b) { return {a[0] / b[0], a[1] / b[1], a[2] / b[2], a[3] / b[3]}; }
Vc_INTRINSIC x_u32 divides(x_u32 a, x_u32 b) { return {a[0] / b[0], a[1] / b[1], a[2] / b[2], a[3] / b[3]}; }
Vc_INTRINSIC x_i16 divides(x_i16 a, x_i16 b) {
    return or_(_mm_slli_epi32(
                   _mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(_mm_srai_epi32(a, 16)),
                                               _mm_cvtepi32_ps(_mm_srai_epi32(b, 16)))),
                   16),
               _mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(_mm_slli_epi32(a, 16)),
                                           _mm_cvtepi32_ps(_mm_slli_epi32(b, 16)))));
}
Vc_INTRINSIC x_u16 divides(x_u16 a, x_u16 b) {
    return or_(
        _mm_slli_epi32(
            _mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(_mm_srli_epi32(a, 16)),
                                        _mm_cvtepi32_ps(_mm_srli_epi32(b, 16)))),
            16),
        _mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(and_(a, broadcast16(0xffff))),
                                    _mm_cvtepi32_ps(and_(b, broadcast16(0xffff))))));
}
Vc_INTRINSIC x_i08 divides(x_i08 a, x_i08 b) {
    const __m128i mask = broadcast16(0xff000000u);
    return or_(
        or_(_mm_slli_epi32(_mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(and_(a, mask)),
                                                       _mm_cvtepi32_ps(and_(b, mask)))),
                           24),
            _mm_slli_epi32(_mm_cvttps_epi32(_mm_div_ps(
                               _mm_cvtepi32_ps(and_(_mm_slli_epi32(a, 8), mask)),
                               _mm_cvtepi32_ps(and_(_mm_slli_epi32(b, 8), mask)))),
                           16)),
        or_(_mm_slli_epi32(_mm_cvttps_epi32(_mm_div_ps(
                               _mm_cvtepi32_ps(and_(_mm_slli_epi32(a, 16), mask)),
                               _mm_cvtepi32_ps(and_(_mm_slli_epi32(b, 16), mask)))),
                           8),
            _mm_cvttps_epi32(_mm_div_ps(_mm_cvtepi32_ps(_mm_slli_epi32(a, 24)),
                                        _mm_cvtepi32_ps(_mm_slli_epi32(b, 24))))));
}
Vc_INTRINSIC x_u08 divides(x_u08 a, x_u08 b) {
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
}
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES

// unary_minus{{{1
template <typename T> Vc_INTRINSIC auto unary_minus(T v) { return minus(T{}, v); }
Vc_INTRINSIC __m128  unary_minus(x_f32 v) { return xor_(v, signmask16(float())); }
Vc_INTRINSIC __m128d unary_minus(x_f64 v) { return xor_(v, signmask16(double())); }
#ifdef Vc_HAVE_SSSE3
Vc_INTRINSIC __m128i unary_minus(x_i32 v) { return _mm_sign_epi32(v, allone<__m128i>()); }
Vc_INTRINSIC __m128i unary_minus(x_u32 v) { return _mm_sign_epi32(v, allone<__m128i>()); }
Vc_INTRINSIC __m128i unary_minus(x_i16 v) { return _mm_sign_epi16(v, allone<__m128i>()); }
Vc_INTRINSIC __m128i unary_minus(x_u16 v) { return _mm_sign_epi16(v, allone<__m128i>()); }
Vc_INTRINSIC __m128i unary_minus(x_i08 v) { return _mm_sign_epi8 (v, allone<__m128i>()); }
Vc_INTRINSIC __m128i unary_minus(x_u08 v) { return _mm_sign_epi8 (v, allone<__m128i>()); }
#endif  // Vc_HAVE_SSSE3

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC __m256  unary_minus(y_f32 v) { return xor_(v, signmask32(float())); }
Vc_INTRINSIC __m256d unary_minus(y_f64 v) { return xor_(v, signmask32(double())); }
#endif  // Vc_HAVE_AVX
#ifdef Vc_HAVE_AVX2
Vc_INTRINSIC __m256i unary_minus(y_i32 v) { return _mm256_sign_epi32(v, allone<__m256i>()); }
Vc_INTRINSIC __m256i unary_minus(y_u32 v) { return _mm256_sign_epi32(v, allone<__m256i>()); }
Vc_INTRINSIC __m256i unary_minus(y_i16 v) { return _mm256_sign_epi16(v, allone<__m256i>()); }
Vc_INTRINSIC __m256i unary_minus(y_u16 v) { return _mm256_sign_epi16(v, allone<__m256i>()); }
Vc_INTRINSIC __m256i unary_minus(y_i08 v) { return _mm256_sign_epi8 (v, allone<__m256i>()); }
Vc_INTRINSIC __m256i unary_minus(y_u08 v) { return _mm256_sign_epi8 (v, allone<__m256i>()); }
#endif  // Vc_HAVE_AVX2

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC __m512  unary_minus(z_f32 v) { return xor_(v, signmask64(float())); }
Vc_INTRINSIC __m512d unary_minus(z_f64 v) { return xor_(v, signmask64(double())); }
#endif  // Vc_HAVE_AVX512F

//}}}1
}  // namespace Vc_VERSIONED_NAMESPACE::detail::x86

#endif  // VC_DATAPAR_X86_ARITHMETICS_H_
