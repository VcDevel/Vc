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

#ifndef VC_DATAPAR_X86_CONVERT_H_
#define VC_DATAPAR_X86_CONVERT_H_

#include "storage.h"

namespace Vc_VERSIONED_NAMESPACE::detail::x86
{
// convert_builtin{{{1
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <typename To, typename From, size_t... I>
Vc_INTRINSIC To convert_builtin(From v0, std::index_sequence<I...>)
{
    using T = typename To::EntryType;
    return typename To::Builtin{static_cast<T>(v0[I])...};
}

template <typename To, typename From, size_t... I>
Vc_INTRINSIC To convert_builtin(From v0, From v1, std::index_sequence<I...>)
{
    using T = typename To::EntryType;
    return typename To::Builtin{static_cast<T>(v0[I])..., static_cast<T>(v1[I])...};
}

template <typename To, typename From, size_t... I>
Vc_INTRINSIC To convert_builtin(From v0, From v1, From v2, From v3,
                                std::index_sequence<I...>)
{
    using T = typename To::EntryType;
    return typename To::Builtin{static_cast<T>(v0[I])..., static_cast<T>(v1[I])...,
                                static_cast<T>(v2[I])..., static_cast<T>(v3[I])...};
}

template <typename To, typename From, size_t... I>
Vc_INTRINSIC To convert_builtin(From v0, From v1, From v2, From v3, From v4, From v5,
                                From v6, From v7, std::index_sequence<I...>)
{
    using T = typename To::EntryType;
    return typename To::Builtin{static_cast<T>(v0[I])..., static_cast<T>(v1[I])...,
                                static_cast<T>(v2[I])..., static_cast<T>(v3[I])...,
                                static_cast<T>(v4[I])..., static_cast<T>(v5[I])...,
                                static_cast<T>(v6[I])..., static_cast<T>(v7[I])...};
}

template <typename To, typename From, size_t... I0, size_t... I1>
Vc_INTRINSIC To convert_builtin(From v0, From v1, std::index_sequence<I0...>,
                                std::index_sequence<I1...>)
{
    using T = typename To::EntryType;
    return typename To::Builtin{static_cast<T>(v0[I0])..., static_cast<T>(v1[I0])...,
                                (I1, T{})...};
}

template <typename To, typename From, size_t... I0, size_t... I1>
Vc_INTRINSIC To convert_builtin(From v0, From v1, From v2, From v3,
                                std::index_sequence<I0...>, std::index_sequence<I1...>)
{
    using T = typename To::EntryType;
    return typename To::Builtin{static_cast<T>(v0[I0])..., static_cast<T>(v1[I0])...,
                                static_cast<T>(v2[I0])..., static_cast<T>(v3[I0])...,
                                (I1, T{})...};
}

template <typename To, typename From, size_t... I0, size_t... I1>
Vc_INTRINSIC To convert_builtin(From v0, From v1, From v2, From v3, From v4, From v5,
                                From v6, From v7, std::index_sequence<I0...>,
                                std::index_sequence<I1...>)
{
    using T = typename To::EntryType;
    return typename To::Builtin{
        static_cast<T>(v0[I0])..., static_cast<T>(v1[I0])..., static_cast<T>(v2[I0])...,
        static_cast<T>(v3[I0])..., static_cast<T>(v4[I0])..., static_cast<T>(v5[I0])...,
        static_cast<T>(v6[I0])..., static_cast<T>(v7[I0])..., (I1, T{})...};
}
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES

// convert_to declarations{{{1
template <typename To> Vc_INTRINSIC To convert_to(x_f32 v0);
template <typename To> Vc_INTRINSIC To convert_to(x_f32 v0, x_f32 v1);
template <typename To> Vc_INTRINSIC To convert_to(x_f32 v0, x_f32 v1, x_f32 v2, x_f32 v3);
#ifdef Vc_HAVE_SSE2
template <typename To> Vc_INTRINSIC To convert_to(x_f64 v0);
template <typename To> Vc_INTRINSIC To convert_to(x_f64 v0, x_f64 v1);
template <typename To> Vc_INTRINSIC To convert_to(x_f64 v0, x_f64 v1, x_f64 v2, x_f64 v3);
template <typename To> Vc_INTRINSIC To convert_to(x_f64 v0, x_f64 v1, x_f64 v2, x_f64 v3, x_f64 v4, x_f64 v5, x_f64 v6, x_f64 v7);
template <typename To> Vc_INTRINSIC To convert_to(x_i08);
template <typename To> Vc_INTRINSIC To convert_to(x_u08);
template <typename To> Vc_INTRINSIC To convert_to(x_i16);
template <typename To> Vc_INTRINSIC To convert_to(x_i16, x_i16);
template <typename To> Vc_INTRINSIC To convert_to(x_u16);
template <typename To> Vc_INTRINSIC To convert_to(x_u16, x_u16);
template <typename To> Vc_INTRINSIC To convert_to(x_i32);
template <typename To> Vc_INTRINSIC To convert_to(x_i32, x_i32);
template <typename To> Vc_INTRINSIC To convert_to(x_i32, x_i32, x_i32, x_i32);
template <typename To> Vc_INTRINSIC To convert_to(x_u32);
template <typename To> Vc_INTRINSIC To convert_to(x_u32, x_u32);
template <typename To> Vc_INTRINSIC To convert_to(x_u32, x_u32, x_u32, x_u32);
template <typename To> Vc_INTRINSIC To convert_to(x_i64);
template <typename To> Vc_INTRINSIC To convert_to(x_i64, x_i64);
template <typename To> Vc_INTRINSIC To convert_to(x_i64, x_i64, x_i64, x_i64);
template <typename To> Vc_INTRINSIC To convert_to(x_i64, x_i64, x_i64, x_i64, x_i64, x_i64, x_i64, x_i64);
template <typename To> Vc_INTRINSIC To convert_to(x_u64);
template <typename To> Vc_INTRINSIC To convert_to(x_u64, x_u64);
template <typename To> Vc_INTRINSIC To convert_to(x_u64, x_u64, x_u64, x_u64);
template <typename To> Vc_INTRINSIC To convert_to(x_u64, x_u64, x_u64, x_u64, x_u64, x_u64, x_u64, x_u64);
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
template <typename To> Vc_INTRINSIC To convert_to(y_f32);
template <typename To> Vc_INTRINSIC To convert_to(y_f32, y_f32);
template <typename To> Vc_INTRINSIC To convert_to(y_f32, y_f32, y_f32, y_f32);
template <typename To> Vc_INTRINSIC To convert_to(y_f64);
template <typename To> Vc_INTRINSIC To convert_to(y_f64, y_f64);
template <typename To> Vc_INTRINSIC To convert_to(y_f64, y_f64, y_f64, y_f64);
template <typename To> Vc_INTRINSIC To convert_to(y_f64, y_f64, y_f64, y_f64, y_f64, y_f64, y_f64, y_f64);
template <typename To> Vc_INTRINSIC To convert_to(y_i08);
template <typename To> Vc_INTRINSIC To convert_to(y_u08);
template <typename To> Vc_INTRINSIC To convert_to(y_i16);
template <typename To> Vc_INTRINSIC To convert_to(y_i16, y_i16);
template <typename To> Vc_INTRINSIC To convert_to(y_u16);
template <typename To> Vc_INTRINSIC To convert_to(y_u16, y_u16);
template <typename To> Vc_INTRINSIC To convert_to(y_i32);
template <typename To> Vc_INTRINSIC To convert_to(y_i32, y_i32);
template <typename To> Vc_INTRINSIC To convert_to(y_i32, y_i32, y_i32, y_i32);
template <typename To> Vc_INTRINSIC To convert_to(y_u32);
template <typename To> Vc_INTRINSIC To convert_to(y_u32, y_u32);
template <typename To> Vc_INTRINSIC To convert_to(y_u32, y_u32, y_u32, y_u32);
template <typename To> Vc_INTRINSIC To convert_to(y_i64);
template <typename To> Vc_INTRINSIC To convert_to(y_i64, y_i64);
template <typename To> Vc_INTRINSIC To convert_to(y_i64, y_i64, y_i64, y_i64);
template <typename To> Vc_INTRINSIC To convert_to(y_i64, y_i64, y_i64, y_i64, y_i64, y_i64, y_i64, y_i64);
template <typename To> Vc_INTRINSIC To convert_to(y_u64);
template <typename To> Vc_INTRINSIC To convert_to(y_u64, y_u64);
template <typename To> Vc_INTRINSIC To convert_to(y_u64, y_u64, y_u64, y_u64);
template <typename To> Vc_INTRINSIC To convert_to(y_u64, y_u64, y_u64, y_u64, y_u64, y_u64, y_u64, y_u64);
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
template <typename To> Vc_INTRINSIC To convert_to(z_f32);
template <typename To> Vc_INTRINSIC To convert_to(z_f64);
template <typename To> Vc_INTRINSIC To convert_to(z_f64, z_f64);
template <typename To> Vc_INTRINSIC To convert_to(z_i08);
template <typename To> Vc_INTRINSIC To convert_to(z_u08);
template <typename To> Vc_INTRINSIC To convert_to(z_i16);
template <typename To> Vc_INTRINSIC To convert_to(z_i16, z_i16);
template <typename To> Vc_INTRINSIC To convert_to(z_u16);
template <typename To> Vc_INTRINSIC To convert_to(z_u16, z_u16);
template <typename To> Vc_INTRINSIC To convert_to(z_i32);
template <typename To> Vc_INTRINSIC To convert_to(z_i32, z_i32);
template <typename To> Vc_INTRINSIC To convert_to(z_i32, z_i32, z_i32, z_i32);
template <typename To> Vc_INTRINSIC To convert_to(z_u32);
template <typename To> Vc_INTRINSIC To convert_to(z_u32, z_u32);
template <typename To> Vc_INTRINSIC To convert_to(z_u32, z_u32, z_u32, z_u32);
template <typename To> Vc_INTRINSIC To convert_to(z_i64);
template <typename To> Vc_INTRINSIC To convert_to(z_i64, z_i64);
template <typename To> Vc_INTRINSIC To convert_to(z_i64, z_i64, z_i64, z_i64);
template <typename To> Vc_INTRINSIC To convert_to(z_i64, z_i64, z_i64, z_i64, z_i64, z_i64, z_i64, z_i64);
template <typename To> Vc_INTRINSIC To convert_to(z_u64);
template <typename To> Vc_INTRINSIC To convert_to(z_u64, z_u64);
template <typename To> Vc_INTRINSIC To convert_to(z_u64, z_u64, z_u64, z_u64);
template <typename To> Vc_INTRINSIC To convert_to(z_u64, z_u64, z_u64, z_u64, z_u64, z_u64, z_u64, z_u64);
#endif  // Vc_HAVE_AVX512F

// convert_to<x_i64>{{{1
#ifdef Vc_HAVE_SSE2
// from float{{{2
template <> Vc_INTRINSIC x_i64 convert_to<x_i64>(x_f32 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm_cvttps_epi64(v);
#else
    return {v.m(0), v.m(1)};
#endif
}

// from double{{{2
template <> Vc_INTRINSIC x_i64 convert_to<x_i64>(x_f64 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm_cvttpd_epi64(v);
#else
    return {v.m(0), v.m(1)};
#endif
}

// from ullong{{{2
template <> Vc_INTRINSIC x_i64 convert_to<x_i64>(x_u64 v) { return v.v(); }

// from llong{{{2
template <> Vc_INTRINSIC x_i64 convert_to<x_i64>(x_i64 v) { return v; }

// from int{{{2
template <> Vc_INTRINSIC x_i64 convert_to<x_i64>(x_i32 v) {
#ifdef Vc_HAVE_SSE4_1
    return _mm_cvtepi32_epi64(v);
#else
    return _mm_unpacklo_epi32(v, _mm_srai_epi32(v, 32));
#endif
}

// from uint{{{2
template <> Vc_INTRINSIC x_i64 convert_to<x_i64>(x_u32 v) {
#ifdef Vc_HAVE_SSE4_1
    return _mm_cvtepu32_epi64(v);
#else
    return _mm_unpacklo_epi32(v, zero<__m128i>());
#endif
}

// from short{{{2
template <> Vc_INTRINSIC x_i64 convert_to<x_i64>(x_i16 v) {
#ifdef Vc_HAVE_SSE4_1
    return _mm_cvtepi16_epi64(v);
#else
    auto x = _mm_srai_epi16(v, 16);
    auto y = _mm_unpacklo_epi16(v, x);
    x = _mm_unpacklo_epi16(x, x);
    return _mm_unpacklo_epi32(y, x);
#endif
}

// from ushort{{{2
template <> Vc_INTRINSIC x_i64 convert_to<x_i64>(x_u16 v) {
#ifdef Vc_HAVE_SSE4_1
    return _mm_cvtepu16_epi64(v);
#else
    return _mm_unpacklo_epi32(_mm_unpacklo_epi16(v, zero<__m128i>()), zero<__m128i>());
#endif
}

// from schar{{{2
template <> Vc_INTRINSIC x_i64 convert_to<x_i64>(x_i08 v) {
#ifdef Vc_HAVE_SSE4_1
    return _mm_cvtepi8_epi64(v);
#else
    auto x = _mm_cmplt_epi8(v, zero<__m128i>());
    auto y = _mm_unpacklo_epi8(v, x);
    x = _mm_unpacklo_epi8(x, x);
    y = _mm_unpacklo_epi16(y, x);
    x = _mm_unpacklo_epi16(x, x);
    return _mm_unpacklo_epi32(y, x);
#endif
}

// from uchar{{{2
template <> Vc_INTRINSIC x_i64 convert_to<x_i64>(x_u08 v) {
#ifdef Vc_HAVE_SSE4_1
    return _mm_cvtepu8_epi64(v);
#else
    return _mm_unpacklo_epi32(_mm_unpacklo_epi16(_mm_unpacklo_epi8(v, zero<__m128i>()), zero<__m128i>()), zero<__m128i>());
#endif
}

// convert_to<x_u64>{{{1
// from float{{{2
template <> Vc_INTRINSIC x_u64 convert_to<x_u64>(x_f32 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm_cvttps_epu64(v);
#else
    return {v.m(0), v.m(1)};
#endif
}

// from double{{{2
template <> Vc_INTRINSIC x_u64 convert_to<x_u64>(x_f64 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm_cvttpd_epu64(v);
#else
    return {v.m(0), v.m(1)};
#endif
}

// from ullong{{{2
template <> Vc_INTRINSIC x_u64 convert_to<x_u64>(x_u64 v) { return v; }

// from llong{{{2
template <> Vc_INTRINSIC x_u64 convert_to<x_u64>(x_i64 v) { return v.v(); }

// from int{{{2
template <> Vc_INTRINSIC x_u64 convert_to<x_u64>(x_i32 v) { return _mm_unpacklo_epi32(v, zero<__m128i>()); }

// from uint{{{2
template <> Vc_INTRINSIC x_u64 convert_to<x_u64>(x_u32 v) { return _mm_unpacklo_epi32(v, zero<__m128i>()); }

// from short{{{2
template <> Vc_INTRINSIC x_u64 convert_to<x_u64>(x_i16 v) { return _mm_unpacklo_epi32(_mm_unpacklo_epi16(v, zero<__m128i>()), zero<__m128i>()); }

// from ushort{{{2
template <> Vc_INTRINSIC x_u64 convert_to<x_u64>(x_u16 v) { return _mm_unpacklo_epi32(_mm_unpacklo_epi16(v, zero<__m128i>()), zero<__m128i>()); }

// from schar{{{2
template <> Vc_INTRINSIC x_u64 convert_to<x_u64>(x_i08 v) { return _mm_unpacklo_epi32(_mm_unpacklo_epi16(_mm_unpacklo_epi8(v, zero<__m128i>()), zero<__m128i>()), zero<__m128i>()); }

// from uchar{{{2
template <> Vc_INTRINSIC x_u64 convert_to<x_u64>(x_u08 v) { return _mm_unpacklo_epi32(_mm_unpacklo_epi16(_mm_unpacklo_epi8(v, zero<__m128i>()), zero<__m128i>()), zero<__m128i>()); }

// convert_to<x_i32>{{{1
// from float{{{2
template <> Vc_INTRINSIC x_i32 convert_to<x_i32>(x_f32 v) { return _mm_cvttps_epi32(v); }

// from double{{{2
template <> Vc_INTRINSIC x_i32 convert_to<x_i32>(x_f64 v) { return _mm_cvttpd_epi32(v); }

template <> Vc_INTRINSIC x_i32 convert_to<x_i32>(x_f64 v0, x_f64 v1)
{
    return _mm_unpacklo_epi64(convert_to<x_i32>(v0), convert_to<x_i32>(v1));
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_i32 convert_to<x_i32>(y_f64 v)
{
    return _mm256_cvttpd_epi32(v);
}
#endif

// from llong{{{2
template <> Vc_INTRINSIC x_i32 convert_to<x_i32>(x_i64 v) {
#ifdef Vc_HAVE_AVX512VL
    return _mm_cvtepi64_epi32(v);
#else
    return {v.m(0), v.m(1), 0, 0};
#endif
}

template <> Vc_INTRINSIC x_i32 convert_to<x_i32>(x_i64 v0, x_i64 v1)
{
#ifdef Vc_HAVE_AVX512VL
    return _mm256_cvtepi64_epi32(concat(v0, v1));
#else
    return {v0.m(0), v0.m(1), v1.m(0), v1.m(1)};
#endif
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_i32 convert_to<x_i32>(y_i64 v0)
{
#ifdef Vc_HAVE_AVX512VL
    return _mm256_cvtepi64_epi32(v0);
#else
    return {v0.m(0), v0.m(1), v0.m(2), v0.m(3)};
#endif
}
#endif  // Vc_HAVE_AVX

// from ullong{{{2
template <> Vc_INTRINSIC x_i32 convert_to<x_i32>(x_u64 v) {
#ifdef Vc_HAVE_AVX512VL
    return _mm_cvtepi64_epi32(v);
#else
    return {v.m(0), v.m(1), 0, 0};
#endif
}

template <> Vc_INTRINSIC x_i32 convert_to<x_i32>(x_u64 v0, x_u64 v1)
{
#ifdef Vc_HAVE_AVX512VL
    return _mm256_cvtepi64_epi32(concat(v0, v1));
#else
    return {v0.m(0), v0.m(1), v1.m(0), v1.m(1)};
#endif
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_i32 convert_to<x_i32>(y_u64 v0)
{
#ifdef Vc_HAVE_AVX512VL
    return _mm256_cvtepi64_epi32(v0);
#else
    return {v0.m(0), v0.m(1), v0.m(2), v0.m(3)};
#endif
}
#endif  // Vc_HAVE_AVX

// from int{{{2
template <> Vc_INTRINSIC x_i32 convert_to<x_i32>(x_i32 v) { return v; }

// from uint{{{2
template <> Vc_INTRINSIC x_i32 convert_to<x_i32>(x_u32 v) { return v.v(); }

// from short{{{2
template <> Vc_INTRINSIC x_i32 convert_to<x_i32>(x_i16 v) {
#ifdef Vc_HAVE_SSE4_1
   return _mm_cvtepi16_epi32(v);
#else
   return _mm_srai_epi32(_mm_unpacklo_epi16(v, v), 16);
#endif
}

// from ushort{{{2
template <> Vc_INTRINSIC x_i32 convert_to<x_i32>(x_u16 v) {
#ifdef Vc_HAVE_SSE4_1
    return _mm_cvtepu16_epi32(v);
#else
    return _mm_unpacklo_epi16(v, zero<__m128i>());
#endif
}

// from schar{{{2
template <> Vc_INTRINSIC x_i32 convert_to<x_i32>(x_i08 v) {
#ifdef Vc_HAVE_SSE4_1
    return _mm_cvtepi8_epi32(v);
#else
    const auto x = _mm_unpacklo_epi8(v, v);
    return _mm_srai_epi32(_mm_unpacklo_epi16(x, x), 24);
#endif
}

// from uchar{{{2
template <> Vc_INTRINSIC x_i32 convert_to<x_i32>(x_u08 v) {
#ifdef Vc_HAVE_SSE4_1
    return _mm_cvtepu8_epi32(v);
#else
    return _mm_unpacklo_epi16(_mm_unpacklo_epi8(v, zero<__m128i>()), zero<__m128i>());
#endif
}

// convert_to<y_i32>{{{1
#ifdef Vc_HAVE_AVX
// from double{{{2
template <> Vc_INTRINSIC y_i32 convert_to<y_i32>(y_f64 v0, y_f64 v1)
{
#ifdef Vc_HAVE_AVX512F
    return _mm512_cvtpd_epi32(concat(v0, v1));
#else
    return concat(_mm256_cvtpd_epi32(v0), _mm256_cvtpd_epi32(v1));
#endif
}

// from float{{{2
template <> Vc_INTRINSIC y_i32 convert_to<y_i32>(y_f32 v0)
{
    return _mm256_cvtps_epi32(v0);
}
#endif  // Vc_HAVE_AVX

// convert_to<x_u32>{{{1
// from float{{{2
template <> Vc_INTRINSIC x_u32 convert_to<x_u32>(x_f32 v) {
#ifdef Vc_HAVE_AVX512VL
    return _mm_cvttps_epu32(v);
#else
    return _mm_castps_si128(
        blendv_ps(_mm_castsi128_ps(_mm_cvttps_epi32(v)),
                  _mm_castsi128_ps(_mm_xor_si128(
                      _mm_cvttps_epi32(_mm_sub_ps(v, _mm_set1_ps(1u << 31))),
                      _mm_set1_epi32(1 << 31))),
                  _mm_cmpge_ps(v, _mm_set1_ps(1u << 31))));
#endif
}

// from double{{{2
template <> Vc_INTRINSIC x_u32 convert_to<x_u32>(x_f64 v) {
#ifdef Vc_HAVE_AVX512VL
    return _mm_cvttpd_epu32(v);
#elif defined Vc_HAVE_SSE4_1
    return _mm_xor_si128(
        _mm_cvttpd_epi32(_mm_sub_pd(_mm_floor_pd(v), _mm_set1_pd(0x80000000u))),
        _mm_cvtsi64_si128(0x8000000080000000ull));
#else
    return blendv_epi8(
        _mm_cvttpd_epi32(v),
        _mm_xor_si128(_mm_cvttpd_epi32(_mm_sub_pd(v, _mm_set1_pd(0x80000000u))),
                      _mm_cvtsi64_si128(0x8000000080000000ull)),
        _mm_castpd_si128(_mm_cmpge_pd(v, _mm_set1_pd(0x80000000u))));
#endif
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_u32 convert_to<x_u32>(y_f64 v)
{
#ifdef Vc_HAVE_AVX512VL
    return _mm256_cvttpd_epu32(v);
#else
    return xor_(_mm256_cvttpd_epi32(
                    _mm256_sub_pd(_mm256_floor_pd(v), _mm256_set1_pd(0x80000000u))),
                intrin_cast<__m128i>(signmask16(float())));
#endif
}
#endif

template <> Vc_INTRINSIC x_u32 convert_to<x_u32>(x_f64 v0, x_f64 v1)
{
#ifdef Vc_HAVE_AVX
    return convert_to<x_u32>(y_f64(concat(v0, v1)));
#else
    return _mm_unpacklo_epi64(convert_to<x_u32>(v0), convert_to<x_u32>(v1));
#endif
}

// from int{{{2
template <> Vc_INTRINSIC x_u32 convert_to<x_u32>(x_i32 v) { return v.v(); }

// from uint{{{2
template <> Vc_INTRINSIC x_u32 convert_to<x_u32>(x_u32 v) { return v; }

// from short{{{2
template <> Vc_INTRINSIC x_u32 convert_to<x_u32>(x_i16 v) { return _mm_unpacklo_epi16(v, zero<__m128i>()); }

// from ushort{{{2
template <> Vc_INTRINSIC x_u32 convert_to<x_u32>(x_u16 v) { return _mm_unpacklo_epi16(v, zero<__m128i>()); }

// from schar{{{2
template <> Vc_INTRINSIC x_u32 convert_to<x_u32>(x_i08 v) {
#ifdef Vc_HAVE_SSE4_1
    return _mm_cvtepi8_epi32(v);
#else
    return _mm_unpacklo_epi16(_mm_unpacklo_epi8(v, zero<__m128i>()), zero<__m128i>());
#endif
}

// from uchar{{{2
template <> Vc_INTRINSIC x_u32 convert_to<x_u32>(x_u08 v) {
#ifdef Vc_HAVE_SSE4_1
    return _mm_cvtepu8_epi32(v);
#else
    return _mm_unpacklo_epi16(_mm_unpacklo_epi8(v, zero<__m128i>()), zero<__m128i>());
#endif
}

// convert_to<x_i16>{{{1
// from llong{{{2
template <> Vc_INTRINSIC x_i16 convert_to<x_i16>(x_i64 v) {
#ifdef Vc_HAVE_AVX512VL
    return _mm_cvtepi64_epi16(v);
#elif defined Vc_HAVE_SSSE3
    return _mm_shuffle_epi8(
        v, _mm_setr_epi8(0, 1, 8, 9, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80,
                         -0x80, -0x80, -0x80, -0x80, -0x80));
#else
    return {v.m(0), v.m(1), 0, 0, 0, 0, 0, 0};
#endif
}

template <> Vc_INTRINSIC x_i16 convert_to<x_i16>(x_i64 v0, x_i64 v1)
{
#ifdef Vc_HAVE_AVX512VL
    return _mm256_cvtepi64_epi16(concat(v0, v1));
#elif defined Vc_HAVE_AVX512F
    return _mm512_cvtepi64_epi16(concat(concat(v0, v1), zero<__m256i>()));
#elif defined Vc_HAVE_SSE4_1
    return _mm_shuffle_epi8(_mm_blend_epi16(v0, _mm_slli_si128(v1, 4), 0x44),
                            _mm_setr_epi8(0, 1, 8, 9, 4, 5, 12, 13, -0x80, -0x80, -0x80,
                                          -0x80, -0x80, -0x80, -0x80, -0x80));
#else
    return {v0.m(0), v0.m(1), v1.m(0), v1.m(1), 0, 0, 0, 0};
#endif
}

template <>
Vc_INTRINSIC x_i16 convert_to<x_i16>(x_i64 v0, x_i64 v1, x_i64 v2, x_i64 v3)
{
#ifdef Vc_HAVE_AVX512F
    return _mm512_cvtepi64_epi16(concat(concat(v0, v1), concat(v2, v3)));
#elif defined Vc_HAVE_SSE4_1
    return _mm_shuffle_epi8(
        _mm_blend_epi16(
            _mm_blend_epi16(v0, _mm_slli_si128(v1, 2), 0x22),
            _mm_blend_epi16(_mm_slli_si128(v2, 4), _mm_slli_si128(v3, 6), 0x88),
            0xcc),
        _mm_setr_epi8(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15));
#else
    return _mm_unpacklo_epi32(convert_to<x_i16>(v0, v2), convert_to<x_i16>(v1, v3));
#endif
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_i16 convert_to<x_i16>(y_i64 v0, y_i64 v1)
{
#ifdef Vc_HAVE_AVX512F
    return _mm512_cvtepi64_epi16(concat(v0, v1));
#elif defined Vc_HAVE_AVX2
    auto a = _mm256_unpacklo_epi16(v0, v1);         // 04.. .... 26.. ....
    auto b = _mm256_unpackhi_epi16(v0, v1);         // 15.. .... 37.. ....
    auto c = _mm256_unpacklo_epi16(a, b);           // 0145 .... 2367 ....
    return _mm_unpacklo_epi32(lo128(c), hi128(c));  // 0123 4567
#else
    return convert_to<x_i16>(lo128(v0), hi128(v0), lo128(v1), hi128(v1));
#endif
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
template <> Vc_INTRINSIC x_i16 convert_to<x_i16>(z_i64 v0)
{
    return _mm512_cvtepi64_epi16(v0);
}
#endif  // Vc_HAVE_AVX512F

// from ullong{{{2
template <> Vc_INTRINSIC x_i16 convert_to<x_i16>(x_u64 v) {
    return convert_to<x_i16>(x_i64(v));
}

template <> Vc_INTRINSIC x_i16 convert_to<x_i16>(x_u64 v0, x_u64 v1)
{
    return convert_to<x_i16>(x_i64(v0), x_i64(v1));
}

template <> Vc_INTRINSIC x_i16 convert_to<x_i16>(x_u64 v0, x_u64 v1, x_u64 v2, x_u64 v3)
{
    return convert_to<x_i16>(x_i64(v0), x_i64(v1), x_i64(v2), x_i64(v3));
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_i16 convert_to<x_i16>(y_u64 v0, y_u64 v1)
{
    return convert_to<x_i16>(y_i64(v0), y_i64(v1));
}
#endif  // Vc_HAVE_AVX

// from int{{{2
template <> Vc_INTRINSIC x_i16 convert_to<x_i16>(x_i32 v) {
#ifdef Vc_HAVE_AVX512VL
    return _mm_cvtepi32_epi16(v);
#else
    auto a = _mm_unpacklo_epi16(v, zero<__m128i>()); // 0o.o 1o.o
    auto b = _mm_unpackhi_epi16(v, zero<__m128i>()); // 2o.o 3o.o
    auto c = _mm_unpacklo_epi16(a, b); // 02oo ..oo
    auto d = _mm_unpackhi_epi16(a, b); // 13oo ..oo
    return _mm_unpacklo_epi16(c, d); // 0123 oooo
#endif
}

template <> Vc_INTRINSIC x_i16 convert_to<x_i16>(x_i32 v0, x_i32 v1)
{
#ifdef Vc_HAVE_AVX512VL
    return _mm256_cvtepi32_epi16(concat(v0, v1));
#elif defined Vc_HAVE_AVX512F
    return lo128(_mm512_cvtepi32_epi16(concat(concat(v0, v1), zero<__m256i>())));
#elif defined Vc_HAVE_SSE4_1
    return _mm_shuffle_epi8(
        _mm_blend_epi16(v0, _mm_slli_si128(v1, 2), 0xaa),
        _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15));
#else
    auto a = _mm_unpacklo_epi16(v0, v1); // 04.. 15..
    auto b = _mm_unpackhi_epi16(v0, v1); // 26.. 37..
    auto c = _mm_unpacklo_epi16(a, b); // 0246 ....
    auto d = _mm_unpackhi_epi16(a, b); // 1357 ....
    return _mm_unpacklo_epi16(c, d); // 0123 4567
#endif
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_i16 convert_to<x_i16>(y_i32 v0)
{
#ifdef Vc_HAVE_AVX512VL
    return _mm256_cvtepi32_epi16(v0);
#elif defined Vc_HAVE_AVX512F
    return lo128(_mm512_cvtepi32_epi16(concat(v0, zero<__m256i>())));
#elif defined Vc_HAVE_AVX2
    auto a = _mm256_shuffle_epi8(
        v0, _mm256_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, -0x80, -0x80, -0x80, -0x80, -0x80,
                             -0x80, -0x80, -0x80, 0, 1, 4, 5, 8, 9, 12, 13, -0x80, -0x80,
                             -0x80, -0x80, -0x80, -0x80, -0x80, -0x80));
    return _mm_unpacklo_epi64(lo128(a), hi128(a));
#else
    return convert_to<x_i16>(lo128(v0), hi128(v0));
#endif
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
template <> Vc_INTRINSIC x_i16 convert_to<x_i16>(z_u64 v0)
{
    return _mm512_cvtepi64_epi16(v0);
}
#endif  // Vc_HAVE_AVX512F

// from uint{{{2
template <> Vc_INTRINSIC x_i16 convert_to<x_i16>(x_u32 v) {
    return convert_to<x_i16>(x_i32(v.v()));
}

template <> Vc_INTRINSIC x_i16 convert_to<x_i16>(x_u32 v0, x_u32 v1)
{
    return convert_to<x_i16>(x_i32(v0), x_i32(v1));
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_i16 convert_to<x_i16>(y_u32 v) {
    return convert_to<x_i16>(y_i32(v.v()));
}
#endif  // Vc_HAVE_AVX

// from short{{{2
template <> Vc_INTRINSIC x_i16 convert_to<x_i16>(x_i16 v) { return v; }

// from ushort{{{2
template <> Vc_INTRINSIC x_i16 convert_to<x_i16>(x_u16 v) { return v.v(); }

// from schar{{{2
template <> Vc_INTRINSIC x_i16 convert_to<x_i16>(x_i08 v) {
#ifdef Vc_HAVE_SSE4_1
   return _mm_cvtepi8_epi16(v);
#else
   return _mm_unpacklo_epi8(v, zero<__m128i>());
#endif
}

// from uchar{{{2
template <> Vc_INTRINSIC x_i16 convert_to<x_i16>(x_u08 v) {
#ifdef Vc_HAVE_SSE4_1
   return _mm_cvtepu8_epi16(v);
#else
   return _mm_unpacklo_epi8(v, zero<__m128i>());
#endif
}

// from double{{{2
template <> Vc_INTRINSIC x_i16 convert_to<x_i16>(x_f64 v)
{
    return convert_to<x_i16>(convert_to<x_i32>(v));
}

template <> Vc_INTRINSIC x_i16 convert_to<x_i16>(x_f64 v0, x_f64 v1)
{
    return convert_to<x_i16>(convert_to<x_i32>(v0, v1));
}

template <>
Vc_INTRINSIC x_i16 convert_to<x_i16>(x_f64 v0, x_f64 v1, x_f64 v2, x_f64 v3)
{
    return convert_to<x_i16>(convert_to<x_i32>(v0, v1), convert_to<x_i32>(v2, v3));
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_i16 convert_to<x_i16>(y_f64 v0, y_f64 v1)
{
#ifdef Vc_HAVE_AVX512F
    return convert_to<x_i16>(y_i32(_mm512_cvtpd_epi32(concat(v0, v1))));
#else
    return convert_to<x_i16>(x_i32(_mm256_cvtpd_epi32(v0)),
                             x_i32(_mm256_cvtpd_epi32(v1)));
#endif
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
template <> Vc_INTRINSIC x_i16 convert_to<x_i16>(z_f64 v0)
{
    return convert_to<x_i16>(y_i32(_mm512_cvtpd_epi32(v0)));
}
#endif  // Vc_HAVE_AVX512F

// from float{{{2
template <> Vc_INTRINSIC x_i16 convert_to<x_i16>(x_f32 v)
{
    return convert_to<x_i16>(convert_to<x_i32>(v));
}

template <> Vc_INTRINSIC x_i16 convert_to<x_i16>(x_f32 v0, x_f32 v1)
{
    return convert_to<x_i16>(convert_to<x_i32>(v0), convert_to<x_i32>(v1));
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_i16 convert_to<x_i16>(y_f32 v0)
{
    return convert_to<x_i16>(convert_to<y_i32>(v0));
}
#endif  // Vc_HAVE_AVX

// convert_to<x_u16>{{{1
// from short{{{2
template <> Vc_INTRINSIC x_u16 convert_to<x_u16>(x_i16 v) { return v.v(); }

// from ushort{{{2
template <> Vc_INTRINSIC x_u16 convert_to<x_u16>(x_u16 v) { return v; }

// from schar{{{2
template <> Vc_INTRINSIC x_u16 convert_to<x_u16>(x_i08 v) {
#ifdef Vc_HAVE_SSE4_1
    return _mm_cvtepi8_epi16(v);
#else
    return _mm_unpacklo_epi8(v, zero<__m128i>());
#endif
}

// from uchar{{{2
template <> Vc_INTRINSIC x_u16 convert_to<x_u16>(x_u08 v) {
#ifdef Vc_HAVE_SSE4_1
    return _mm_cvtepu8_epi16(v);
#else
    return _mm_unpacklo_epi8(v, zero<__m128i>());
#endif
}

// from double{{{2
template <> Vc_INTRINSIC x_u16 convert_to<x_u16>(x_f64 v)
{
    return convert_to<x_u16>(convert_to<x_i32>(v));
}

template <> Vc_INTRINSIC x_u16 convert_to<x_u16>(x_f64 v0, x_f64 v1)
{
    return convert_to<x_u16>(convert_to<x_i32>(v0, v1));
}

template <>
Vc_INTRINSIC x_u16 convert_to<x_u16>(x_f64 v0, x_f64 v1, x_f64 v2, x_f64 v3)
{
    return convert_to<x_u16>(convert_to<x_i32>(v0, v1), convert_to<x_i32>(v2, v3));
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_u16 convert_to<x_u16>(y_f64 v0, y_f64 v1)
{
    return convert_to<x_u16>(convert_to<y_i32>(v0, v1));
}
#endif  // Vc_HAVE_AVX

// from float{{{2
template <> Vc_INTRINSIC x_u16 convert_to<x_u16>(x_f32 v)
{
    return convert_to<x_u16>(convert_to<x_i32>(v));
}

template <> Vc_INTRINSIC x_u16 convert_to<x_u16>(x_f32 v0, x_f32 v1)
{
    return convert_to<x_u16>(convert_to<x_i32>(v0), convert_to<x_i32>(v1));
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_u16 convert_to<x_u16>(y_f32 v)
{
    return convert_to<x_u16>(convert_to<y_i32>(v));
}
#endif  // Vc_HAVE_AVX

// convert_to<x_i08>{{{1
// from llong{{{2
template <> Vc_INTRINSIC x_i08 convert_to<x_i08>(x_i64 v) {
#ifdef Vc_HAVE_AVX512VL
    return _mm_cvtepi64_epi8(v);
#else
    return {v.m(0), v.m(1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#endif
}

template <> Vc_INTRINSIC x_i08 convert_to<x_i08>(x_i64 v0, x_i64 v1)
{
#ifdef Vc_HAVE_AVX512VL
    return _mm256_cvtepi64_epi8(concat(v0, v1));
#elif defined Vc_HAVE_AVX512F
    return _mm512_cvtepi64_epi8(concat(concat(v0, v1), zero<__m256i>()));
#else
    return {v0.m(0), v0.m(1), v1.m(0), v1.m(1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#endif
}

template <>
Vc_INTRINSIC x_i08 convert_to<x_i08>(x_i64 v0, x_i64 v1, x_i64 v2, x_i64 v3)
{
#ifdef Vc_HAVE_AVX512F
    return _mm512_cvtepi64_epi8(concat(concat(v0, v1), concat(v2, v3)));
#else
    return {v0.m(0), v0.m(1), v1.m(0), v1.m(1), v2.m(0), v2.m(1), v3.m(0), v3.m(1),
            0,       0,       0,       0,       0,       0,       0,       0};
#endif
}

template <>
Vc_INTRINSIC x_i08 convert_to<x_i08>(x_i64 v0, x_i64 v1, x_i64 v2, x_i64 v3,
                                         x_i64 v4, x_i64 v5, x_i64 v6, x_i64 v7)
{
#ifdef Vc_HAVE_AVX512F
    return _mm_unpacklo_epi64(
        _mm512_cvtepi64_epi8(concat(concat(v0, v1), concat(v2, v3))),
        _mm512_cvtepi64_epi8(concat(concat(v4, v5), concat(v6, v7))));
#else
    return _mm_unpacklo_epi8(
        _mm_unpacklo_epi32(
            _mm_unpacklo_epi16(_mm_unpacklo_epi8(v0, v1), _mm_unpacklo_epi8(v2, v3)),
            _mm_unpacklo_epi16(_mm_unpacklo_epi8(v4, v5), _mm_unpacklo_epi8(v6, v7))),
        _mm_unpacklo_epi32(
            _mm_unpacklo_epi16(_mm_unpackhi_epi8(v0, v1), _mm_unpackhi_epi8(v2, v3)),
            _mm_unpacklo_epi16(_mm_unpackhi_epi8(v4, v5), _mm_unpackhi_epi8(v6, v7))));
#endif
}

#ifdef Vc_HAVE_AVX
template <>
Vc_INTRINSIC x_i08 convert_to<x_i08>(y_i64 v0, y_i64 v1, y_i64 v2, y_i64 v3)
{
#ifdef Vc_HAVE_AVX512F
    return _mm_unpacklo_epi64(_mm512_cvtepi64_epi8(concat(v0, v1)),
                              _mm512_cvtepi64_epi8(concat(v2, v3)));
#elif defined Vc_HAVE_AVX2
    auto a =
        or_(or_(_mm256_srli_epi32(_mm256_slli_epi32(v0, 24), 24),
                _mm256_srli_epi32(_mm256_slli_epi32(v1, 24), 16)),
            or_(_mm256_srli_epi32(_mm256_slli_epi32(v2, 24), 8),
                _mm256_slli_epi32(v3, 24)));  // 048C .... 159D .... 26AE .... 37BF ....
    auto b = _mm256_unpackhi_epi64(a, a);     // 159D .... 159D .... 37BF .... 37BF ....
    auto c = _mm256_unpacklo_epi8(a, b);      // 0145 89CD .... .... 2367 ABEF .... ....
    return _mm_unpacklo_epi16(lo128(c), hi128(c));  // 0123 4567 89AB CDEF
#else
    return convert_to<x_i08>(lo128(v0), hi128(v0), lo128(v1), hi128(v1), lo128(v2),
                             hi128(v2), lo128(v3), hi128(v3));
#endif
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
template <> Vc_INTRINSIC x_i08 convert_to<x_i08>(z_i64 v0, z_i64 v1)
{
    return _mm_unpacklo_epi64(_mm512_cvtepi64_epi8(v0), _mm512_cvtepi64_epi8(v1));
}
#endif  // Vc_HAVE_AVX512F

// from ullong{{{2
template <> Vc_INTRINSIC x_i08 convert_to<x_i08>(x_u64 v) {
    return convert_to<x_i08>(x_i64(v));
}

template <> Vc_INTRINSIC x_i08 convert_to<x_i08>(x_u64 v0, x_u64 v1)
{
    return convert_to<x_i08>(x_i64(v0), x_i64(v1));
}

template <>
Vc_INTRINSIC x_i08 convert_to<x_i08>(x_u64 v0, x_u64 v1, x_u64 v2, x_u64 v3)
{
    return convert_to<x_i08>(x_i64(v0), x_i64(v1), x_i64(v2), x_i64(v3));
}

template <>
Vc_INTRINSIC x_i08 convert_to<x_i08>(x_u64 v0, x_u64 v1, x_u64 v2, x_u64 v3,
                                         x_u64 v4, x_u64 v5, x_u64 v6, x_u64 v7)
{
    return convert_to<x_i08>(x_i64(v0), x_i64(v1), x_i64(v2), x_i64(v3), x_i64(v4),
                             x_i64(v5), x_i64(v6), x_i64(v7));
}

#ifdef Vc_HAVE_AVX
template <>
Vc_INTRINSIC x_i08 convert_to<x_i08>(y_u64 v0, y_u64 v1, y_u64 v2, y_u64 v3)
{
    return convert_to<x_i08>(y_i64(v0), y_i64(v1), y_i64(v2), y_i64(v3));
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
template <> Vc_INTRINSIC x_i08 convert_to<x_i08>(z_u64 v0, z_u64 v1)
{
    return _mm_unpacklo_epi64(_mm512_cvtepi64_epi8(v0), _mm512_cvtepi64_epi8(v1));
}
#endif  // Vc_HAVE_AVX512F

// from int{{{2
template <> Vc_INTRINSIC x_i08 convert_to<x_i08>(x_i32 v) {
#ifdef Vc_HAVE_AVX512VL
    return _mm_cvtepi32_epi8(v);
#elif defined Vc_HAVE_AVX512F
    return _mm512_cvtepi32_epi8(concat(concat(v, zero<__m128i>()), zero<__m256i>()));
#elif defined Vc_HAVE_SSSE3
    return _mm_shuffle_epi8(
        v, _mm_setr_epi8(0, 4, 8, 12, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80,
                         -0x80, -0x80, -0x80, -0x80, -0x80));
#else
    auto a = _mm_unpacklo_epi8(v, v);  // 0... .... 1... ....
    auto b = _mm_unpackhi_epi8(v, v);  // 2... .... 3... ....
    auto c = _mm_unpacklo_epi8(a, b);  // 02.. .... .... ....
    auto d = _mm_unpackhi_epi8(a, b);  // 13.. .... .... ....
    auto e = _mm_unpacklo_epi8(c, d);  // 0123 .... .... ....
    return detail::and_(e, _mm_cvtsi32_si128(-1));
#endif
}

template <> Vc_INTRINSIC x_i08 convert_to<x_i08>(x_i32 v0, x_i32 v1)
{
#ifdef Vc_HAVE_AVX512VL
    return _mm256_cvtepi32_epi8(concat(v0, v1));
#elif defined Vc_HAVE_AVX512F
    return _mm512_cvtepi32_epi8(concat(concat(v0, v1), zero<__m256i>()));
#elif defined Vc_HAVE_SSSE3
    const auto shufmask = _mm_setr_epi8(0, 4, 8, 12, -0x80, -0x80, -0x80, -0x80, -0x80,
                                        -0x80, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80);
    return _mm_unpacklo_epi32(_mm_shuffle_epi8(v0, shufmask),
                              _mm_shuffle_epi8(v1, shufmask));
#else
    auto a = _mm_unpacklo_epi8(v0, v1);  // 04.. .... 15.. ....
    auto b = _mm_unpackhi_epi8(v0, v1);  // 26.. .... 37.. ....
    auto c = _mm_unpacklo_epi8(a, b);  // 0246 .... .... ....
    auto d = _mm_unpackhi_epi8(a, b);  // 1357 .... .... ....
    auto e = _mm_unpacklo_epi8(c, d);  // 0123 4567 .... ....
    return detail::and_(e, _mm_cvtsi64x_si128(-1LL));
#endif
}

Vc_INTRINSIC x_i08 sse2_convert_to_i08(x_i32 v0, x_i32 v1, x_i32 v2, x_i32 v3)
{
    auto a = _mm_unpacklo_epi8(v0, v2);  // 08.. .... 19.. ....
    auto b = _mm_unpackhi_epi8(v0, v2);  // 2A.. .... 3B.. ....
    auto c = _mm_unpacklo_epi8(v1, v3);  // 4C.. .... 5D.. ....
    auto d = _mm_unpackhi_epi8(v1, v3);  // 6E.. .... 7F.. ....
    auto e = _mm_unpacklo_epi8(a, c);    // 048C .... .... ....
    auto f = _mm_unpackhi_epi8(a, c);    // 159D .... .... ....
    auto g = _mm_unpacklo_epi8(b, d);    // 26AE .... .... ....
    auto h = _mm_unpackhi_epi8(b, d);    // 37BF .... .... ....
    return _mm_unpacklo_epi8(_mm_unpacklo_epi8(e, g),  // 0246 8ACE .... ....
                             _mm_unpacklo_epi8(f, h)   // 1357 9BDF .... ....
                             );                        // 0123 4567 89AB CDEF
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_i08 convert_to<x_i08>(y_i32 v0, y_i32 v1)
{
#ifdef Vc_HAVE_AVX512F
    return _mm512_cvtepi32_epi8(concat(v0, v1));
#elif defined Vc_HAVE_AVX2
    auto a = _mm256_unpacklo_epi8(v0, v1);  // 08.. .... 19.. .... 4C.. .... 5D.. ....
    auto b = _mm256_unpackhi_epi8(v0, v1);  // 2A.. .... 3B.. .... 6E.. .... 7F.. ....
    auto c = _mm256_unpacklo_epi8(a, b);    // 028A .... .... .... 46CE ...
    auto d = _mm256_unpackhi_epi8(a, b);    // 139B .... .... .... 57DF ...
    auto e = _mm256_unpacklo_epi8(c, d);    // 0123 89AB .... .... 4567 CDEF ...
    return _mm_unpacklo_epi32(lo128(e), hi128(e));  // 0123 4567 89AB CDEF
#else
    return sse2_convert_to_i08(lo128(v0), hi128(v0), lo128(v1), hi128(v1));
#endif
}
#endif  // Vc_HAVE_AVX

template <>
Vc_INTRINSIC x_i08 convert_to<x_i08>(x_i32 v0, x_i32 v1, x_i32 v2, x_i32 v3)
{
#ifdef Vc_HAVE_AVX512F
    return _mm512_cvtepi32_epi8(concat(concat(v0, v1), concat(v2, v3)));
#elif defined Vc_HAVE_AVX2
    return convert_to<x_i08>(y_i32(concat(v0, v1)), y_i32(concat(v2, v3)));
#else
    return sse2_convert_to_i08(v0, v1, v2, v3);
#endif
}

#ifdef Vc_HAVE_AVX512F
template <> Vc_INTRINSIC x_i08 convert_to<x_i08>(z_i32 v0)
{
    return _mm512_cvtepi32_epi8(v0);
}
#endif  // Vc_HAVE_AVX512F

// from uint{{{2
template <> Vc_INTRINSIC x_i08 convert_to<x_i08>(x_u32 v)
{
    return convert_to<x_i08>(x_i32(v.v()));
}

template <> Vc_INTRINSIC x_i08 convert_to<x_i08>(x_u32 v0, x_u32 v1)
{
    return convert_to<x_i08>(x_i32(v0.v()), x_i32(v1.v()));
}

template <>
Vc_INTRINSIC x_i08 convert_to<x_i08>(x_u32 v0, x_u32 v1, x_u32 v2, x_u32 v3)
{
    return convert_to<x_i08>(x_i32(v0.v()), x_i32(v1.v()), x_i32(v2.v()),
                               x_i32(v3.v()));
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_i08 convert_to<x_i08>(y_u32 v0, y_u32 v1)
{
    return convert_to<x_i08>(y_i32(v0.v()), y_i32(v1.v()));
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
template <> Vc_INTRINSIC x_i08 convert_to<x_i08>(z_u32 v0)
{
    return _mm512_cvtepi32_epi8(v0);
}
#endif  // Vc_HAVE_AVX512F

// from short{{{2
template <> Vc_INTRINSIC x_i08 convert_to<x_i08>(x_i16 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512BW
    return _mm_cvtepi16_epi8(v);
#elif defined Vc_HAVE_SSSE3
    auto shuf = load16(sse_const::cvti16_i08_shuffle, flags::vector_aligned);
    return _mm_shuffle_epi8(v, shuf);
#else
    auto a = _mm_unpacklo_epi8(v, v);  // 00.. 11.. 22.. 33..
    auto b = _mm_unpackhi_epi8(v, v);  // 44.. 55.. 66.. 77..
    auto c = _mm_unpacklo_epi8(a, b);  // 0404 .... 1515 ....
    auto d = _mm_unpackhi_epi8(a, b);  // 2626 .... 3737 ....
    auto e = _mm_unpacklo_epi8(c, d);  // 0246 0246 .... ....
    auto f = _mm_unpackhi_epi8(c, d);  // 1357 1357 .... ....
    return _mm_unpacklo_epi8(e, f);
#endif
}

template <> Vc_INTRINSIC x_i08 convert_to<x_i08>(x_i16 v0, x_i16 v1)
{
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512BW
    return _mm256_cvtepi16_epi8(concat(v0, v1));
#elif defined Vc_HAVE_SSSE3
    auto shuf = load16(sse_const::cvti16_i08_shuffle, flags::vector_aligned);
    return _mm_unpacklo_epi64(_mm_shuffle_epi8(v0, shuf), _mm_shuffle_epi8(v1, shuf));
#else
    auto a = _mm_unpacklo_epi8(v0, v1);  // 08.. 19.. 2A.. 3B..
    auto b = _mm_unpackhi_epi8(v0, v1);  // 4C.. 5D.. 6E.. 7F..
    auto c = _mm_unpacklo_epi8(a, b);  // 048C .... 159D ....
    auto d = _mm_unpackhi_epi8(a, b);  // 26AE .... 37BF ....
    auto e = _mm_unpacklo_epi8(c, d);  // 0246 8ACE .... ....
    auto f = _mm_unpackhi_epi8(c, d);  // 1357 9BDF .... ....
    return _mm_unpacklo_epi8(e, f);
#endif
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_i08 convert_to<x_i08>(y_i16 v0)
{
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512BW
    return _mm256_cvtepi16_epi8(v0);
#elif defined Vc_HAVE_AVX2
    auto a = _mm256_shuffle_epi8(
        v0, _mm256_broadcastsi128_si256(
                load16(sse_const::cvti16_i08_shuffle, flags::vector_aligned)));
    return _mm_unpacklo_epi64(lo128(a), hi128(a));
#else
    auto shuf = load16(sse_const::cvti16_i08_shuffle, flags::vector_aligned);
    return _mm_unpacklo_epi64(_mm_shuffle_epi8(lo128(v0), shuf),
                              _mm_shuffle_epi8(hi128(v0), shuf));
#endif
}
#endif  // Vc_HAVE_AVX

// from ushort{{{2
template <> Vc_INTRINSIC x_i08 convert_to<x_i08>(x_u16 v) {
    return convert_to<x_i08>(x_i16(v));
}

template <> Vc_INTRINSIC x_i08 convert_to<x_i08>(x_u16 v0, x_u16 v1)
{
    return convert_to<x_i08>(x_i16(v0), x_i16(v1));
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_i08 convert_to<x_i08>(y_u16 v) {
    return convert_to<x_i08>(y_i16(v));
}
#endif  // Vc_HAVE_AVX

// from [su]char{{{2
template <> Vc_INTRINSIC x_i08 convert_to<x_i08>(x_i08 v) { return v; }
template <> Vc_INTRINSIC x_i08 convert_to<x_i08>(x_u08 v) { return v.v(); }

// from float{{{2
template <> Vc_INTRINSIC x_i08 convert_to<x_i08>(x_f32 v)
{
    return convert_to<x_i08>(convert_to<x_i32>(v));
}

template <> Vc_INTRINSIC x_i08 convert_to<x_i08>(x_f32 v0, x_f32 v1)
{
    return convert_to<x_i08>(convert_to<x_i32>(v0), convert_to<x_i32>(v1));
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_i08 convert_to<x_i08>(y_f32 v0, y_f32 v1)
{
#ifdef Vc_HAVE_AVX512F
    return _mm512_cvtepi32_epi8(_mm512_cvttps_epi32(concat(v0, v1)));
#else
    return convert_to<x_i08>(convert_to<y_i32>(v0), convert_to<y_i32>(v1));
#endif
}
#endif  // Vc_HAVE_AVX

template <>
Vc_INTRINSIC x_i08 convert_to<x_i08>(x_f32 v0, x_f32 v1, x_f32 v2, x_f32 v3)
{
#ifdef Vc_HAVE_AVX
    return convert_to<x_i08>(y_f32(concat(v0, v1)), y_f32(concat(v2, v3)));
#else
    return convert_to<x_i08>(convert_to<x_i32>(v0), convert_to<x_i32>(v1),
                             convert_to<x_i32>(v2), convert_to<x_i32>(v3));
#endif
}

#ifdef Vc_HAVE_AVX512F
template <> Vc_INTRINSIC x_i08 convert_to<x_i08>(z_f32 v0)
{
    return _mm512_cvtepi32_epi8(_mm512_cvttps_epi32(v0));
}
#endif  // Vc_HAVE_AVX512F

// from double{{{2
template <> Vc_INTRINSIC x_i08 convert_to<x_i08>(x_f64 v)
{
    return convert_to<x_i08>(convert_to<x_i32>(v));
}

template <> Vc_INTRINSIC x_i08 convert_to<x_i08>(x_f64 v0, x_f64 v1)
{
    return convert_to<x_i08>(convert_to<x_i32>(v0, v1));
}

template <>
Vc_INTRINSIC x_i08 convert_to<x_i08>(x_f64 v0, x_f64 v1, x_f64 v2, x_f64 v3)
{
    return convert_to<x_i08>(convert_to<x_i32>(v0, v1), convert_to<x_i32>(v2, v3));
}

template <>
Vc_INTRINSIC x_i08 convert_to<x_i08>(x_f64 v0, x_f64 v1, x_f64 v2, x_f64 v3,
                                         x_f64 v4, x_f64 v5, x_f64 v6, x_f64 v7)
{
#ifdef Vc_HAVE_AVX512F
    return _mm512_cvtepi32_epi8(
        concat(_mm512_cvttpd_epi32(concat(concat(v0, v1), concat(v2, v3))),
               _mm512_cvttpd_epi32(concat(concat(v4, v5), concat(v6, v7)))));
#else
    return convert_to<x_i08>(convert_to<x_i32>(v0, v1), convert_to<x_i32>(v2, v3),
                             convert_to<x_i32>(v4, v5), convert_to<x_i32>(v6, v7));
#endif
}

#ifdef Vc_HAVE_AVX
template <>
Vc_INTRINSIC x_i08 convert_to<x_i08>(y_f64 v0, y_f64 v1, y_f64 v2, y_f64 v3)
{
#ifdef Vc_HAVE_AVX512F
    return _mm512_cvtepi32_epi8(
        concat(_mm512_cvttpd_epi32(concat(v0, v1)), _mm512_cvttpd_epi32(concat(v2, v3))));
#else
    return convert_to<x_i08>(convert_to<y_i32>(v0, v1), convert_to<y_i32>(v2, v3));
#endif
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
template <>
Vc_INTRINSIC x_i08 convert_to<x_i08>(z_f64 v0, z_f64 v1)
{
    return _mm512_cvtepi32_epi8(concat(_mm512_cvttpd_epi32(v0), _mm512_cvttpd_epi32(v1)));
}
#endif  // Vc_HAVE_AVX512F

// convert_to<x_u08>{{{1
// from schar{{{2
template <> Vc_INTRINSIC x_u08 convert_to<x_u08>(x_i08 v) { return v.v(); }

// from uchar{{{2
template <> Vc_INTRINSIC x_u08 convert_to<x_u08>(x_u08 v) { return v; }

// from float{{{2
template <> Vc_INTRINSIC x_u08 convert_to<x_u08>(x_f32 v) { return convert_to<x_u08>(convert_to<x_i32>(v)); }
template <> Vc_INTRINSIC x_u08 convert_to<x_u08>(x_f32 v0, x_f32 v1)
{
    return _mm_unpacklo_epi32(convert_to<x_u08>(v0), convert_to<x_u08>(v1));
}
template <>
Vc_INTRINSIC x_u08 convert_to<x_u08>(x_f32 v0, x_f32 v1, x_f32 v2, x_f32 v3)
{
    return _mm_unpacklo_epi64(
        _mm_unpacklo_epi32(convert_to<x_u08>(v0), convert_to<x_u08>(v1)),
        _mm_unpacklo_epi32(convert_to<x_u08>(v2), convert_to<x_u08>(v3)));
}

// from double{{{2
template <> Vc_INTRINSIC x_u08 convert_to<x_u08>(x_f64 v)
{
    return convert_to<x_u08>(convert_to<x_i32>(v));
}

template <> Vc_INTRINSIC x_u08 convert_to<x_u08>(x_f64 v0, x_f64 v1)
{
    return convert_to<x_u08>(convert_to<x_i32>(v0, v1));
}

template <>
Vc_INTRINSIC x_u08 convert_to<x_u08>(x_f64 v0, x_f64 v1, x_f64 v2, x_f64 v3)
{
    return convert_to<x_u08>(convert_to<x_i32>(v0, v1), convert_to<x_i32>(v2, v3));
}

template <>
Vc_INTRINSIC x_u08 convert_to<x_u08>(x_f64 v0, x_f64 v1, x_f64 v2, x_f64 v3, x_f64 v4,
                                     x_f64 v5, x_f64 v6, x_f64 v7)
{
    return convert_to<x_u08>(convert_to<x_i32>(v0, v1), convert_to<x_i32>(v2, v3),
                             convert_to<x_i32>(v4, v5), convert_to<x_i32>(v6, v7));
}

// convert_to<x_f64>{{{1
// from float{{{2
template <> Vc_INTRINSIC x_f64 convert_to<x_f64>(x_f32 v) { return _mm_cvtps_pd(v); }

// from double{{{2
template <> Vc_INTRINSIC x_f64 convert_to<x_f64>(x_f64 v) { return v; }

// from llong{{{2
template <> Vc_INTRINSIC x_f64 convert_to<x_f64>(x_i64 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm_cvtepi64_pd(v);
#else
    return x_f64{v.m(0), v.m(1)};
#endif
}

// from ullong{{{2
template <> Vc_INTRINSIC x_f64 convert_to<x_f64>(x_u64 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm_cvtepu64_pd(v);
#else
    return x_f64{v.m(0), v.m(1)};
#endif
}

// from int{{{2
template <> Vc_INTRINSIC x_f64 convert_to<x_f64>(x_i32 v) { return _mm_cvtepi32_pd(v); }

// from uint{{{2
template <> Vc_INTRINSIC x_f64 convert_to<x_f64>(x_u32 v) { return _mm_add_pd(_mm_cvtepi32_pd(_mm_xor_si128(v, setmin_epi32())), _mm_set1_pd(1u << 31)); }

// from short{{{2
template <> Vc_INTRINSIC x_f64 convert_to<x_f64>(x_i16 v) { return convert_to<x_f64>(convert_to<x_i32>(v)); }

// from ushort{{{2
template <> Vc_INTRINSIC x_f64 convert_to<x_f64>(x_u16 v) { return convert_to<x_f64>(convert_to<x_i32>(v)); }

// from schar{{{2
template <> Vc_INTRINSIC x_f64 convert_to<x_f64>(x_i08 v) { return convert_to<x_f64>(convert_to<x_i32>(v)); }

// from uchar{{{2
template <> Vc_INTRINSIC x_f64 convert_to<x_f64>(x_u08 v) { return convert_to<x_f64>(convert_to<x_i32>(v)); }

// convert_to<x_f32>{{{1
// from float{{{2
template <> Vc_INTRINSIC x_f32 convert_to<x_f32>(x_f32 v) { return v; }

// from double{{{2
template <> Vc_INTRINSIC x_f32 convert_to<x_f32>(x_f64 v) { return _mm_cvtpd_ps(v); }

template <> Vc_INTRINSIC x_f32 convert_to<x_f32>(x_f64 v0, x_f64 v1)
{
#ifdef Vc_HAVE_AVX
    return _mm256_cvtpd_ps(concat(v0, v1));
#else
    return _mm_movelh_ps(_mm_cvtpd_ps(v0), _mm_cvtpd_ps(v1));
#endif
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_f32 convert_to<x_f32>(y_f64 v0)
{
    return _mm256_cvtpd_ps(v0);
}
#endif  // Vc_HAVE_AVX

// from llong{{{2
template <> Vc_INTRINSIC x_f32 convert_to<x_f32>(x_i64 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm_cvtepi64_ps(v);
#else
    return {v.m(0), v.m(1), 0.f, 0.f};
#endif
}

template <> Vc_INTRINSIC x_f32 convert_to<x_f32>(x_i64 v0, x_i64 v1)
{
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm256_cvtepi64_ps(concat(v0, v1));
#else
    return {v0.m(0), v0.m(1), v1.m(0), v1.m(1)};
#endif
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_f32 convert_to<x_f32>(y_i64 v0)
{
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm256_cvtepi64_ps(v0);
#else
    return {v0.m(0), v0.m(1), v0.m(2), v0.m(3)};
#endif
}
#endif  // Vc_HAVE_AVX

// from ullong{{{2
template <> Vc_INTRINSIC x_f32 convert_to<x_f32>(x_u64 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm_cvtepu64_ps(v);
#else
    return {v.m(0), v.m(1), 0.f, 0.f};
#endif
}

template <> Vc_INTRINSIC x_f32 convert_to<x_f32>(x_u64 v0, x_u64 v1)
{
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm256_cvtepu64_ps(concat(v0, v1));
#else
    return {v0.m(0), v0.m(1), v1.m(0), v1.m(1)};
#endif
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_f32 convert_to<x_f32>(y_u64 v0)
{
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm256_cvtepu64_ps(v0);
#else
    return {v0.m(0), v0.m(1), v0.m(2), v0.m(3)};
#endif
}
#endif  // Vc_HAVE_AVX

// from int{{{2
template <> Vc_INTRINSIC x_f32 convert_to<x_f32>(x_i32 v) { return _mm_cvtepi32_ps(v); }

// from uint{{{2
template <> Vc_INTRINSIC x_f32 convert_to<x_f32>(x_u32 v) {
#ifdef Vc_HAVE_AVX512VL
    return _mm_cvtepu32_ps(v);
#else
    // see AVX::convert_to<uint, float> for an explanation of the math behind the
    // implementation
    return blendv_ps(
        _mm_cvtepi32_ps(v),
        _mm_add_ps(
            _mm_cvtepi32_ps(_mm_and_si128(v, _mm_set1_epi32(0x7ffffe00))),
            _mm_add_ps(_mm_set1_ps(1u << 31),
                       _mm_cvtepi32_ps(_mm_and_si128(v, _mm_set1_epi32(0x000001ff))))),
        _mm_castsi128_ps(_mm_cmplt_epi32(v, _mm_setzero_si128())));
#endif
}

// from short{{{2
template <> Vc_INTRINSIC x_f32 convert_to<x_f32>(x_i16 v) { return convert_to<x_f32>(convert_to<x_i32>(v)); }

// from ushort{{{2
template <> Vc_INTRINSIC x_f32 convert_to<x_f32>(x_u16 v) { return convert_to<x_f32>(convert_to<x_i32>(v)); }

// from schar{{{2
template <> Vc_INTRINSIC x_f32 convert_to<x_f32>(x_i08 v) { return convert_to<x_f32>(convert_to<x_i32>(v)); }

// from uchar{{{2
template <> Vc_INTRINSIC x_f32 convert_to<x_f32>(x_u08 v) { return convert_to<x_f32>(convert_to<x_i32>(v)); }

#endif  // Vc_HAVE_SSE2

// generic forwarding for down-conversions to unsigned int{{{1
template <typename T> struct fallback_int_type {
    using type = typename std::make_signed<T>::type;
};
template <> struct fallback_int_type<long> {
    using type = equal_int_type_t<long>;
};
template <> struct fallback_int_type<ulong> {
    using type = equal_int_type_t<long>;
};

// generic (u)long forwarding to (u)(llong|int){{{1

template <typename To, size_t N> Vc_INTRINSIC To convert_to(Storage<long, N> v)
{
    return convert_to<To>(Storage<equal_int_type_t<long>, N>(v));
}
template <typename To, size_t N>
Vc_INTRINSIC To convert_to(Storage<long, N> v0, Storage<long, N> v1)
{
    return convert_to<To>(Storage<equal_int_type_t<long>, N>(v0),
                          Storage<equal_int_type_t<long>, N>(v1));
}
template <typename To, size_t N>
Vc_INTRINSIC To convert_to(Storage<long, N> v0, Storage<long, N> v1, Storage<long, N> v2,
                           Storage<long, N> v3)
{
    return convert_to<To>(
        Storage<equal_int_type_t<long>, N>(v0), Storage<equal_int_type_t<long>, N>(v1),
        Storage<equal_int_type_t<long>, N>(v2), Storage<equal_int_type_t<long>, N>(v3));
}
template <typename To, size_t N>
Vc_INTRINSIC To convert_to(Storage<long, N> v0, Storage<long, N> v1, Storage<long, N> v2,
                           Storage<long, N> v3, Storage<long, N> v4, Storage<long, N> v5,
                           Storage<long, N> v6, Storage<long, N> v7)
{
    return convert_to<To>(
        Storage<equal_int_type_t<long>, N>(v0), Storage<equal_int_type_t<long>, N>(v1),
        Storage<equal_int_type_t<long>, N>(v2), Storage<equal_int_type_t<long>, N>(v3),
        Storage<equal_int_type_t<long>, N>(v4), Storage<equal_int_type_t<long>, N>(v5),
        Storage<equal_int_type_t<long>, N>(v6), Storage<equal_int_type_t<long>, N>(v7));
}

template <typename To, size_t N> Vc_INTRINSIC To convert_to(Storage<ulong, N> v)
{
    return convert_to<To>(Storage<equal_int_type_t<ulong>, N>(v));
}
template <typename To, size_t N>
Vc_INTRINSIC To convert_to(Storage<ulong, N> v0, Storage<ulong, N> v1)
{
    return convert_to<To>(Storage<equal_int_type_t<ulong>, N>(v0),
                          Storage<equal_int_type_t<ulong>, N>(v1));
}
template <typename To, size_t N>
Vc_INTRINSIC To convert_to(Storage<ulong, N> v0, Storage<ulong, N> v1, Storage<ulong, N> v2,
                           Storage<ulong, N> v3)
{
    return convert_to<To>(
        Storage<equal_int_type_t<ulong>, N>(v0), Storage<equal_int_type_t<ulong>, N>(v1),
        Storage<equal_int_type_t<ulong>, N>(v2), Storage<equal_int_type_t<ulong>, N>(v3));
}
template <typename To, size_t N>
Vc_INTRINSIC To convert_to(Storage<ulong, N> v0, Storage<ulong, N> v1, Storage<ulong, N> v2,
                           Storage<ulong, N> v3, Storage<ulong, N> v4, Storage<ulong, N> v5,
                           Storage<ulong, N> v6, Storage<ulong, N> v7)
{
    return convert_to<To>(
        Storage<equal_int_type_t<ulong>, N>(v0), Storage<equal_int_type_t<ulong>, N>(v1),
        Storage<equal_int_type_t<ulong>, N>(v2), Storage<equal_int_type_t<ulong>, N>(v3),
        Storage<equal_int_type_t<ulong>, N>(v4), Storage<equal_int_type_t<ulong>, N>(v5),
        Storage<equal_int_type_t<ulong>, N>(v6), Storage<equal_int_type_t<ulong>, N>(v7));
}

template <typename T>
using equivalent_storage_t =
    Storage<typename fallback_int_type<typename T::EntryType>::type, T::size()>;

template <typename To> Vc_INTRINSIC To convert_to(x_f32 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
#ifdef Vc_HAVE_SSE2
template <typename To> Vc_INTRINSIC To convert_to(x_f64 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(x_f64 v0, x_f64 v1) { return convert_to<equivalent_storage_t<To>>(v0, v1).v(); }
template <typename To> Vc_INTRINSIC To convert_to(x_i08 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(x_u08 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(x_i16 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(x_i16 v0, x_i16 v1) { return convert_to<equivalent_storage_t<To>>(v0, v1).v(); }
template <typename To> Vc_INTRINSIC To convert_to(x_u16 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(x_u16 v0, x_u16 v1) { return convert_to<equivalent_storage_t<To>>(v0, v1).v(); }
template <typename To> Vc_INTRINSIC To convert_to(x_i32 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(x_i32 v0, x_i32 v1) { return convert_to<equivalent_storage_t<To>>(v0, v1).v(); }
template <typename To> Vc_INTRINSIC To convert_to(x_i32 v0, x_i32 v1, x_i32 v2, x_i32 v3) { return convert_to<equivalent_storage_t<To>>(v0, v1, v2, v3).v(); }
template <typename To> Vc_INTRINSIC To convert_to(x_u32 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(x_u32 v0, x_u32 v1) { return convert_to<equivalent_storage_t<To>>(v0, v1).v(); }
template <typename To> Vc_INTRINSIC To convert_to(x_u32 v0, x_u32 v1, x_u32 v2, x_u32 v3) { return convert_to<equivalent_storage_t<To>>(v0, v1, v2, v3).v(); }
template <typename To> Vc_INTRINSIC To convert_to(x_i64 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(x_i64 v0, x_i64 v1) { return convert_to<equivalent_storage_t<To>>(v0, v1).v(); }
template <typename To> Vc_INTRINSIC To convert_to(x_i64 v0, x_i64 v1, x_i64 v2, x_i64 v3) { return convert_to<equivalent_storage_t<To>>(v0, v1, v2, v3).v(); }
template <typename To> Vc_INTRINSIC To convert_to(x_i64 v0, x_i64 v1, x_i64 v2, x_i64 v3, x_i64 v4, x_i64 v5, x_i64 v6, x_i64 v7) { return convert_to<equivalent_storage_t<To>>(v0, v1, v2, v3, v4, v5, v6, v7).v(); }
template <typename To> Vc_INTRINSIC To convert_to(x_u64 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(x_u64 v0, x_u64 v1) { return convert_to<equivalent_storage_t<To>>(v0, v1).v(); }
template <typename To> Vc_INTRINSIC To convert_to(x_u64 v0, x_u64 v1, x_u64 v2, x_u64 v3) { return convert_to<equivalent_storage_t<To>>(v0, v1, v2, v3).v(); }
template <typename To> Vc_INTRINSIC To convert_to(x_u64 v0, x_u64 v1, x_u64 v2, x_u64 v3, x_u64 v4, x_u64 v5, x_u64 v6, x_u64 v7) { return convert_to<equivalent_storage_t<To>>(v0, v1, v2, v3, v4, v5, v6, v7).v(); }
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
template <typename To> Vc_INTRINSIC To convert_to(y_f32 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(y_f32 v0, y_f32 v1) { return convert_to<equivalent_storage_t<To>>(v0, v1).v(); }
template <typename To> Vc_INTRINSIC To convert_to(y_f64 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(y_f64 v0, y_f64 v1) { return convert_to<equivalent_storage_t<To>>(v0, v1).v(); }
template <typename To> Vc_INTRINSIC To convert_to(y_f64 v0, y_f64 v1, y_f64 v2, y_f64 v3) { return convert_to<equivalent_storage_t<To>>(v0, v1, v2, v3).v(); }
template <typename To> Vc_INTRINSIC To convert_to(y_i08 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(y_u08 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(y_i16 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(y_i16 v0, y_i16 v1) { return convert_to<equivalent_storage_t<To>>(v0, v1).v(); }
template <typename To> Vc_INTRINSIC To convert_to(y_u16 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(y_u16 v0, y_u16 v1) { return convert_to<equivalent_storage_t<To>>(v0, v1).v(); }
template <typename To> Vc_INTRINSIC To convert_to(y_i32 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(y_i32 v0, y_i32 v1) { return convert_to<equivalent_storage_t<To>>(v0, v1).v(); }
template <typename To> Vc_INTRINSIC To convert_to(y_i32 v0, y_i32 v1, y_i32 v2, y_i32 v3) { return convert_to<equivalent_storage_t<To>>(v0, v1, v2, v3).v(); }
template <typename To> Vc_INTRINSIC To convert_to(y_u32 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(y_u32 v0, y_u32 v1) { return convert_to<equivalent_storage_t<To>>(v0, v1).v(); }
template <typename To> Vc_INTRINSIC To convert_to(y_u32 v0, y_u32 v1, y_u32 v2, y_u32 v3) { return convert_to<equivalent_storage_t<To>>(v0, v1, v2, v3).v(); }
template <typename To> Vc_INTRINSIC To convert_to(y_i64 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(y_i64 v0, y_i64 v1) { return convert_to<equivalent_storage_t<To>>(v0, v1).v(); }
template <typename To> Vc_INTRINSIC To convert_to(y_i64 v0, y_i64 v1, y_i64 v2, y_i64 v3) { return convert_to<equivalent_storage_t<To>>(v0, v1, v2, v3).v(); }
template <typename To> Vc_INTRINSIC To convert_to(y_i64 v0, y_i64 v1, y_i64 v2, y_i64 v3, y_i64 v4, y_i64 v5, y_i64 v6, y_i64 v7) { return convert_to<equivalent_storage_t<To>>(v0, v1, v2, v3, v4, v5, v6, v7).v(); }
template <typename To> Vc_INTRINSIC To convert_to(y_u64 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(y_u64 v0, y_u64 v1) { return convert_to<equivalent_storage_t<To>>(v0, v1).v(); }
template <typename To> Vc_INTRINSIC To convert_to(y_u64 v0, y_u64 v1, y_u64 v2, y_u64 v3) { return convert_to<equivalent_storage_t<To>>(v0, v1, v2, v3).v(); }
template <typename To> Vc_INTRINSIC To convert_to(y_u64 v0, y_u64 v1, y_u64 v2, y_u64 v3, y_u64 v4, y_u64 v5, y_u64 v6, y_u64 v7) { return convert_to<equivalent_storage_t<To>>(v0, v1, v2, v3, v4, v5, v6, v7).v(); }
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
template <typename To> Vc_INTRINSIC To convert_to(z_f32 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(z_f64 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(z_f64 v0, z_f64 v1) { return convert_to<equivalent_storage_t<To>>(v0, v1).v(); }
template <typename To> Vc_INTRINSIC To convert_to(z_i08 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(z_u08 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(z_i16 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(z_i16 v0, z_i16 v1) { return convert_to<equivalent_storage_t<To>>(v0, v1).v(); }
template <typename To> Vc_INTRINSIC To convert_to(z_u16 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(z_u16 v0, z_u16 v1) { return convert_to<equivalent_storage_t<To>>(v0, v1).v(); }
template <typename To> Vc_INTRINSIC To convert_to(z_i32 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(z_i32 v0, z_i32 v1) { return convert_to<equivalent_storage_t<To>>(v0, v1).v(); }
template <typename To> Vc_INTRINSIC To convert_to(z_i32 v0, z_i32 v1, z_i32 v2, z_i32 v3) { return convert_to<equivalent_storage_t<To>>(v0, v1, v2, v3).v(); }
template <typename To> Vc_INTRINSIC To convert_to(z_u32 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(z_u32 v0, z_u32 v1) { return convert_to<equivalent_storage_t<To>>(v0, v1).v(); }
template <typename To> Vc_INTRINSIC To convert_to(z_u32 v0, z_u32 v1, z_u32 v2, z_u32 v3) { return convert_to<equivalent_storage_t<To>>(v0, v1, v2, v3).v(); }
template <typename To> Vc_INTRINSIC To convert_to(z_i64 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(z_i64 v0, z_i64 v1) { return convert_to<equivalent_storage_t<To>>(v0, v1).v(); }
template <typename To> Vc_INTRINSIC To convert_to(z_i64 v0, z_i64 v1, z_i64 v2, z_i64 v3) { return convert_to<equivalent_storage_t<To>>(v0, v1, v2, v3).v(); }
template <typename To> Vc_INTRINSIC To convert_to(z_i64 v0, z_i64 v1, z_i64 v2, z_i64 v3, z_i64 v4, z_i64 v5, z_i64 v6, z_i64 v7) { return convert_to<equivalent_storage_t<To>>(v0, v1, v2, v3, v4, v5, v6, v7).v(); }
template <typename To> Vc_INTRINSIC To convert_to(z_u64 v0) { return convert_to<equivalent_storage_t<To>>(v0).v(); }
template <typename To> Vc_INTRINSIC To convert_to(z_u64 v0, z_u64 v1) { return convert_to<equivalent_storage_t<To>>(v0, v1).v(); }
template <typename To> Vc_INTRINSIC To convert_to(z_u64 v0, z_u64 v1, z_u64 v2, z_u64 v3) { return convert_to<equivalent_storage_t<To>>(v0, v1, v2, v3).v(); }
template <typename To> Vc_INTRINSIC To convert_to(z_u64 v0, z_u64 v1, z_u64 v2, z_u64 v3, z_u64 v4, z_u64 v5, z_u64 v6, z_u64 v7) { return convert_to<equivalent_storage_t<To>>(v0, v1, v2, v3, v4, v5, v6, v7).v(); }
#endif  // Vc_HAVE_AVX512F

// convert function{{{1
template <typename From, typename To> Vc_INTRINSIC To convert(From v)
{
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
    constexpr auto N = From::size() < To::size() ? From::size() : To::size();
    return convert_builtin<To>(v.builtin(), std::make_index_sequence<N>());
#else
    return convert_to<To>(v);
#endif
}

template <typename From, typename To> Vc_INTRINSIC To convert(From v0, From v1)
{
    static_assert(To::size() >= 2 * From::size(),
                  "convert(v0, v1) requires the input to fit into the output");
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
    return convert_builtin<To>(
        v0.builtin(), v1.builtin(), std::make_index_sequence<From::size()>(),
        std::make_index_sequence<To::size() - 2 * From::size()>());
#else
    return convert_to<To>(v0, v1);
#endif
}

template <typename From, typename To>
Vc_INTRINSIC To convert(From v0, From v1, From v2, From v3)
{
    static_assert(To::size() >= 4 * From::size(),
                  "convert(v0, v1, v2, v3) requires the input to fit into the output");
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
    return convert_builtin<To>(
        v0.builtin(), v1.builtin(), v2.builtin(), v3.builtin(),
        std::make_index_sequence<From::size()>(),
        std::make_index_sequence<To::size() - 4 * From::size()>());
#else
    return convert_to<To>(v0, v1, v2, v3);
#endif
}

template <typename From, typename To>
Vc_INTRINSIC To convert(From v0, From v1, From v2, From v3, From v4, From v5, From v6,
                        From v7)
{
    static_assert(To::size() >= 8 * From::size(),
                  "convert(v0, v1, v2, v3, v4, v5, v6, v7) "
                  "requires the input to fit into the output");
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
    return convert_builtin<To>(
        v0.builtin(), v1.builtin(), v2.builtin(), v3.builtin(), v4.builtin(),
        v5.builtin(), v6.builtin(), v7.builtin(),
        std::make_index_sequence<From::size()>(),
        std::make_index_sequence<To::size() - 8 * From::size()>());
#else
    return convert_to<To>(v0, v1, v2, v3, v4, v5, v6, v7);
#endif
}

// }}}1
}  // namespace Vc_VERSIONED_NAMESPACE::detail::x86

#endif  // VC_DATAPAR_X86_CONVERT_H_
