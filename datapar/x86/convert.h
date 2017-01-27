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

#include <iostream>
#include <iomanip>
#include "storage.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
namespace x86
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
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_f32 v0);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_f32 v0, x_f32 v1);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_f32 v0, x_f32 v1, x_f32 v2, x_f32 v3);
#ifdef Vc_HAVE_SSE2
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_f64 v0);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_f64 v0, x_f64 v1);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_f64 v0, x_f64 v1, x_f64 v2, x_f64 v3);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_f64 v0, x_f64 v1, x_f64 v2, x_f64 v3, x_f64 v4, x_f64 v5, x_f64 v6, x_f64 v7);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_i08);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u08);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_i16);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_i16, x_i16);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u16);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u16, x_u16);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_i32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_i32, x_i32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_i32, x_i32, x_i32, x_i32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u32, x_u32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u32, x_u32, x_u32, x_u32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_i64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_i64, x_i64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_i64, x_i64, x_i64, x_i64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_i64, x_i64, x_i64, x_i64, x_i64, x_i64, x_i64, x_i64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u64, x_u64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u64, x_u64, x_u64, x_u64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u64, x_u64, x_u64, x_u64, x_u64, x_u64, x_u64, x_u64);
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_f32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_f32, y_f32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_f32, y_f32, y_f32, y_f32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_f64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_f64, y_f64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_f64, y_f64, y_f64, y_f64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_f64, y_f64, y_f64, y_f64, y_f64, y_f64, y_f64, y_f64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_i08);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_u08);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_i16);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_i16, y_i16);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_u16);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_u16, y_u16);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_i32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_i32, y_i32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_i32, y_i32, y_i32, y_i32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_u32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_u32, y_u32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_u32, y_u32, y_u32, y_u32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_i64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_i64, y_i64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_i64, y_i64, y_i64, y_i64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_i64, y_i64, y_i64, y_i64, y_i64, y_i64, y_i64, y_i64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_u64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_u64, y_u64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_u64, y_u64, y_u64, y_u64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_u64, y_u64, y_u64, y_u64, y_u64, y_u64, y_u64, y_u64);
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_f32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_f32, z_f32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_f32, z_f32, z_f32, z_f32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_f64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_f64, z_f64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_f64, z_f64, z_f64, z_f64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_f64, z_f64, z_f64, z_f64, z_f64, z_f64, z_f64, z_f64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_i08);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_u08);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_i16);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_i16, z_i16);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_u16);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_u16, z_u16);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_i32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_i32, z_i32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_i32, z_i32, z_i32, z_i32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_u32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_u32, z_u32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_u32, z_u32, z_u32, z_u32);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_i64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_i64, z_i64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_i64, z_i64, z_i64, z_i64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_i64, z_i64, z_i64, z_i64, z_i64, z_i64, z_i64, z_i64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_u64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_u64, z_u64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_u64, z_u64, z_u64, z_u64);
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_u64, z_u64, z_u64, z_u64, z_u64, z_u64, z_u64, z_u64);
#endif  // Vc_HAVE_AVX512F

//}}}1

#ifdef Vc_HAVE_SSE2
//--------------------llong & ullong{{{1
//
// convert_to<x_i64> (long long, 2){{{1
// from float{{{2
template <> Vc_INTRINSIC x_i64 Vc_VDECL convert_to<x_i64>(x_f32 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm_cvttps_epi64(v);
#else
    return {v.m(0), v.m(1)};
#endif
}

// from double{{{2
template <> Vc_INTRINSIC x_i64 Vc_VDECL convert_to<x_i64>(x_f64 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm_cvttpd_epi64(v);
#else
    return {v.m(0), v.m(1)};
#endif
}

// from llong{{{2
template <> Vc_INTRINSIC x_i64 Vc_VDECL convert_to<x_i64>(x_i64 v) { return v; }

// from int{{{2
template <> Vc_INTRINSIC x_i64 Vc_VDECL convert_to<x_i64>(x_i32 v) {
#ifdef Vc_HAVE_SSE4_1
    return _mm_cvtepi32_epi64(v);
#else
    return _mm_unpacklo_epi32(v, _mm_srai_epi32(v, 32));
#endif
}

// from uint{{{2
template <> Vc_INTRINSIC x_i64 Vc_VDECL convert_to<x_i64>(x_u32 v) {
#ifdef Vc_HAVE_SSE4_1
    return _mm_cvtepu32_epi64(v);
#else
    return _mm_unpacklo_epi32(v, zero<__m128i>());
#endif
}

// from short{{{2
template <> Vc_INTRINSIC x_i64 Vc_VDECL convert_to<x_i64>(x_i16 v) {
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
template <> Vc_INTRINSIC x_i64 Vc_VDECL convert_to<x_i64>(x_u16 v) {
#ifdef Vc_HAVE_SSE4_1
    return _mm_cvtepu16_epi64(v);
#else
    return _mm_unpacklo_epi32(_mm_unpacklo_epi16(v, zero<__m128i>()), zero<__m128i>());
#endif
}

// from schar{{{2
template <> Vc_INTRINSIC x_i64 Vc_VDECL convert_to<x_i64>(x_i08 v) {
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
template <> Vc_INTRINSIC x_i64 Vc_VDECL convert_to<x_i64>(x_u08 v) {
#ifdef Vc_HAVE_SSE4_1
    return _mm_cvtepu8_epi64(v);
#else
    return _mm_unpacklo_epi32(_mm_unpacklo_epi16(_mm_unpacklo_epi8(v, zero<__m128i>()), zero<__m128i>()), zero<__m128i>());
#endif
}

// convert_to<y_i64> (long long, 4){{{1
#ifdef Vc_HAVE_AVX
// from float{{{2
template <> Vc_INTRINSIC y_i64 Vc_VDECL convert_to<y_i64>(x_f32 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm256_cvttps_epi64(v);
#else
    return {v.m(0), v.m(1), v.m(2), v.m(3)};
#endif
}

// from double{{{2
template <> Vc_INTRINSIC y_i64 Vc_VDECL convert_to<y_i64>(x_f64 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return zeroExtend(_mm_cvttpd_epi64(v));
#else
    return {v.m(0), v.m(1), 0.f, 0.f};
#endif
}

template <> Vc_INTRINSIC y_i64 Vc_VDECL convert_to<y_i64>(y_f64 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm256_cvttpd_epi64(v);
#else
    return {v.m(0), v.m(1), v.m(2), v.m(3)};
#endif
}

// from ullong{{{2
template <> Vc_INTRINSIC y_i64 Vc_VDECL convert_to<y_i64>(x_u64 v) { return zeroExtend(v.v()); }

// from llong{{{2
template <> Vc_INTRINSIC y_i64 Vc_VDECL convert_to<y_i64>(x_i64 v) { return zeroExtend(v.v()); }
template <> Vc_INTRINSIC y_i64 Vc_VDECL convert_to<y_i64>(y_i64 v) { return v; }

// from int{{{2
template <> Vc_INTRINSIC y_i64 Vc_VDECL convert_to<y_i64>(x_i32 v) {
#ifdef Vc_HAVE_AVX2
    return _mm256_cvtepi32_epi64(v);
#else
    return concat(_mm_cvtepi32_epi64(v), _mm_cvtepi32_epi64(_mm_unpackhi_epi64(v, v)));
#endif
}

// from uint{{{2
template <> Vc_INTRINSIC y_i64 Vc_VDECL convert_to<y_i64>(x_u32 v) {
#ifdef Vc_HAVE_AVX2
    return _mm256_cvtepu32_epi64(v);
#else
    return concat(_mm_cvtepu32_epi64(v), _mm_cvtepu32_epi64(_mm_unpackhi_epi64(v, v)));
#endif
}

// from short{{{2
template <> Vc_INTRINSIC y_i64 Vc_VDECL convert_to<y_i64>(x_i16 v) {
#ifdef Vc_HAVE_AVX2
    return _mm256_cvtepi16_epi64(v);
#else
    return concat(_mm_cvtepi16_epi64(v), _mm_cvtepi16_epi64(_mm_srli_si128(v, 4)));
#endif
}

// from ushort{{{2
template <> Vc_INTRINSIC y_i64 Vc_VDECL convert_to<y_i64>(x_u16 v) {
#ifdef Vc_HAVE_AVX2
    return _mm256_cvtepu16_epi64(v);
#else
    return concat(_mm_cvtepu16_epi64(v), _mm_cvtepu16_epi64(_mm_srli_si128(v, 4)));
#endif
}

// from schar{{{2
template <> Vc_INTRINSIC y_i64 Vc_VDECL convert_to<y_i64>(x_i08 v) {
#ifdef Vc_HAVE_AVX2
    return _mm256_cvtepi8_epi64(v);
#else
    return concat(_mm_cvtepi8_epi64(v), _mm_cvtepi8_epi64(_mm_srli_si128(v, 2)));
#endif
}

// from uchar{{{2
template <> Vc_INTRINSIC y_i64 Vc_VDECL convert_to<y_i64>(x_u08 v) {
#ifdef Vc_HAVE_AVX2
    return _mm256_cvtepu8_epi64(v);
#else
    return concat(_mm_cvtepu8_epi64(v), _mm_cvtepu8_epi64(_mm_srli_si128(v, 2)));
#endif
}

//}}}2
#endif  // Vc_HAVE_AVX

// convert_to<z_i64> (long long, 8){{{1
#ifdef Vc_HAVE_AVX512F
// from float{{{2
template <> Vc_INTRINSIC z_i64 Vc_VDECL convert_to<z_i64>(x_f32 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return zeroExtend(_mm256_cvttps_epi64(v));
#elif defined Vc_HAVE_AVX512DQ
    return _mm512_cvttps_epi64(zeroExtend(v));
#else
    return {v.m(0), v.m(1), v.m(2), v.m(3), 0, 0, 0, 0};
#endif
}

template <> Vc_INTRINSIC z_i64 Vc_VDECL convert_to<z_i64>(y_f32 v)
{
#ifdef Vc_HAVE_AVX512DQ
    return _mm512_cvttps_epi64(v);
#else
    return {v.m(0), v.m(1), v.m(2), v.m(3), v.m(4), v.m(5), v.m(6), v.m(7)};
#endif
}

template <> Vc_INTRINSIC z_i64 Vc_VDECL convert_to<z_i64>(x_f32 v0, x_f32 v1)
{
    return convert_to<z_i64>(concat(v0, v1));
}

// from double{{{2
template <> Vc_INTRINSIC z_i64 Vc_VDECL convert_to<z_i64>(x_f64 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return zeroExtend(zeroExtend(_mm_cvttpd_epi64(v)));
#else
    return {v.m(0), v.m(1), 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#endif
}

template <> Vc_INTRINSIC z_i64 Vc_VDECL convert_to<z_i64>(y_f64 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return zeroExtend(_mm256_cvttpd_epi64(v));
#elif defined Vc_HAVE_AVX512DQ
    return _mm512_cvttpd_epi64(zeroExtend(v));
#else
    return {v.m(0), v.m(1), v.m(2), v.m(3), 0.f, 0.f, 0.f, 0.f};
#endif
}

template <> Vc_INTRINSIC z_i64 Vc_VDECL convert_to<z_i64>(z_f64 v) {
#if defined Vc_HAVE_AVX512DQ
    return _mm512_cvttpd_epi64(v);
#else
    return {v.m(0), v.m(1), v.m(2), v.m(3), v.m(4), v.m(5), v.m(6), v.m(7)};
#endif
}

// from ullong{{{2
template <> Vc_INTRINSIC z_i64 Vc_VDECL convert_to<z_i64>(y_u64 v) { return zeroExtend(v.v()); }

// from llong{{{2
template <> Vc_INTRINSIC z_i64 Vc_VDECL convert_to<z_i64>(y_i64 v) { return zeroExtend(v.v()); }
template <> Vc_INTRINSIC z_i64 Vc_VDECL convert_to<z_i64>(z_i64 v) { return v; }

// from int{{{2
template <> Vc_INTRINSIC z_i64 Vc_VDECL convert_to<z_i64>(y_i32 v) {
    return _mm512_cvtepi32_epi64(v);
}

// from uint{{{2
template <> Vc_INTRINSIC z_i64 Vc_VDECL convert_to<z_i64>(y_u32 v) {
    return _mm512_cvtepu32_epi64(v);
}

// from short{{{2
template <> Vc_INTRINSIC z_i64 Vc_VDECL convert_to<z_i64>(x_i16 v) {
    return _mm512_cvtepi16_epi64(v);
}

// from ushort{{{2
template <> Vc_INTRINSIC z_i64 Vc_VDECL convert_to<z_i64>(x_u16 v) {
    return _mm512_cvtepu16_epi64(v);
}

// from schar{{{2
template <> Vc_INTRINSIC z_i64 Vc_VDECL convert_to<z_i64>(x_i08 v) {
    return _mm512_cvtepi8_epi64(v);
}

// from uchar{{{2
template <> Vc_INTRINSIC z_i64 Vc_VDECL convert_to<z_i64>(x_u08 v) {
    return _mm512_cvtepu8_epi64(v);
}

//}}}2
#endif  // Vc_HAVE_AVX512F

// convert_to<x_u64>{{{1
// from float{{{2
template <> Vc_INTRINSIC x_u64 Vc_VDECL convert_to<x_u64>(x_f32 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm_cvttps_epu64(v);
#else
    return {v.m(0), v.m(1)};
#endif
}

// from double{{{2
template <> Vc_INTRINSIC x_u64 Vc_VDECL convert_to<x_u64>(x_f64 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm_cvttpd_epu64(v);
#else
    return {v.m(0), v.m(1)};
#endif
}

// convert_to<y_u64>{{{1
#ifdef Vc_HAVE_AVX
// from float{{{2
template <> Vc_INTRINSIC y_u64 Vc_VDECL convert_to<y_u64>(x_f32 v0) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm256_cvttps_epu64(v0);
#else
    return {v0.m(0), v0.m(1), v0.m(2), v0.m(3)};
#endif
}

// from double{{{2
template <> Vc_INTRINSIC y_u64 Vc_VDECL convert_to<y_u64>(x_f64 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return zeroExtend(_mm_cvttpd_epu64(v));
#else
    return {v.m(0), v.m(1), 0.f, 0.f};
#endif
}

template <> Vc_INTRINSIC y_u64 Vc_VDECL convert_to<y_u64>(y_f64 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm256_cvttpd_epu64(v);
#elif defined Vc_HAVE_AVX512DQ
    return lo256(_mm512_cvttpd_epu64(intrin_cast<__m512d>(v));
#else
    return {v.m(0), v.m(1), v.m(2), v.m(3)};
#endif
}

//}}}2
#endif  // Vc_HAVE_AVX

// convert_to<z_u64>{{{1
#ifdef Vc_HAVE_AVX512F
// from float{{{2
template <> Vc_INTRINSIC z_u64 Vc_VDECL convert_to<z_u64>(y_f32 v0) {
#if defined Vc_HAVE_AVX512DQ
    return _mm512_cvttps_epu64(v0);
#else
    return {v0.m(0), v0.m(1), v0.m(2), v0.m(3), v0.m(4), v0.m(5), v0.m(6), v0.m(7)};
#endif
}

// from double{{{2
template <> Vc_INTRINSIC z_u64 Vc_VDECL convert_to<z_u64>(z_f64 v0) {
#if defined Vc_HAVE_AVX512DQ
    return _mm512_cvttpd_epu64(v0);
#else
    return {v0.m(0), v0.m(1), v0.m(2), v0.m(3), v0.m(4), v0.m(5), v0.m(6), v0.m(7)};
#endif
}

//}}}2
#endif  // Vc_HAVE_AVX512F

//--------------------int & uint{{{1
//
// convert_to<x_i32> (int, 4){{{1
// from float{{{2
template <> Vc_INTRINSIC x_i32 Vc_VDECL convert_to<x_i32>(x_f32 v) { return _mm_cvttps_epi32(v); }

// from double{{{2
template <> Vc_INTRINSIC x_i32 Vc_VDECL convert_to<x_i32>(x_f64 v) { return _mm_cvttpd_epi32(v); }

template <> Vc_INTRINSIC x_i32 Vc_VDECL convert_to<x_i32>(x_f64 v0, x_f64 v1)
{
    return _mm_unpacklo_epi64(convert_to<x_i32>(v0), convert_to<x_i32>(v1));
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_i32 Vc_VDECL convert_to<x_i32>(y_f64 v)
{
    return _mm256_cvttpd_epi32(v);
}
#endif

// from llong{{{2
template <> Vc_INTRINSIC x_i32 Vc_VDECL convert_to<x_i32>(x_i64 v) {
#ifdef Vc_HAVE_AVX512VL
    return _mm_cvtepi64_epi32(v);
#else
    return _mm_unpacklo_epi64(_mm_shuffle_epi32(v, 8), _mm_setzero_si128());
    //return {v.m(0), v.m(1), 0, 0};
#endif
}

template <> Vc_INTRINSIC x_i32 Vc_VDECL convert_to<x_i32>(x_i64 v0, x_i64 v1)
{
#ifdef Vc_HAVE_AVX512VL
    return _mm256_cvtepi64_epi32(concat(v0, v1));
#elif defined Vc_HAVE_AVX512F
    return lo128(_mm512_cvtepi64_epi32(intrin_cast<__m512i>(concat(v0, v1))));
#else
    return _mm_unpacklo_epi64(_mm_shuffle_epi32(v0, 8), _mm_shuffle_epi32(v1, 8));
    //return {v0.m(0), v0.m(1), v1.m(0), v1.m(1)};
#endif
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_i32 Vc_VDECL convert_to<x_i32>(y_i64 v0)
{
#ifdef Vc_HAVE_AVX512VL
    return _mm256_cvtepi64_epi32(v0);
#elif defined Vc_HAVE_AVX512F
    return lo128(_mm512_cvtepi64_epi32(intrin_cast<__m512i>(v0)));
#elif defined Vc_HAVE_AVX2
    return lo128(_mm256_permute4x64_epi64(_mm256_shuffle_epi32(v0, 8), 0 + 4 * 2));
#else
    return _mm_unpacklo_epi64(_mm_shuffle_epi32(lo128(v0), 8),
                              _mm_shuffle_epi32(hi128(v0), 8));
    //return {v0.m(0), v0.m(1), v0.m(2), v0.m(3)};
#endif
}
#endif  // Vc_HAVE_AVX

// from int{{{2
template <> Vc_INTRINSIC x_i32 Vc_VDECL convert_to<x_i32>(x_i32 v) { return v; }

// from short{{{2
template <> Vc_INTRINSIC x_i32 Vc_VDECL convert_to<x_i32>(x_i16 v) {
#ifdef Vc_HAVE_SSE4_1
   return _mm_cvtepi16_epi32(v);
#else
   return _mm_srai_epi32(_mm_unpacklo_epi16(v, v), 16);
#endif
}

// from ushort{{{2
template <> Vc_INTRINSIC x_i32 Vc_VDECL convert_to<x_i32>(x_u16 v) {
#ifdef Vc_HAVE_SSE4_1
    return _mm_cvtepu16_epi32(v);
#else
    return _mm_unpacklo_epi16(v, zero<__m128i>());
#endif
}

// from schar{{{2
template <> Vc_INTRINSIC x_i32 Vc_VDECL convert_to<x_i32>(x_i08 v) {
#ifdef Vc_HAVE_SSE4_1
    return _mm_cvtepi8_epi32(v);
#else
    const auto x = _mm_unpacklo_epi8(v, v);
    return _mm_srai_epi32(_mm_unpacklo_epi16(x, x), 24);
#endif
}

// from uchar{{{2
template <> Vc_INTRINSIC x_i32 Vc_VDECL convert_to<x_i32>(x_u08 v) {
#ifdef Vc_HAVE_SSE4_1
    return _mm_cvtepu8_epi32(v);
#else
    return _mm_unpacklo_epi16(_mm_unpacklo_epi8(v, zero<__m128i>()), zero<__m128i>());
#endif
}

// convert_to<y_i32> (int, 8){{{1
#ifdef Vc_HAVE_AVX
// from float{{{2
template <> Vc_INTRINSIC y_i32 Vc_VDECL convert_to<y_i32>(x_f32 v)
{
    return zeroExtend(_mm_cvttps_epi32(v));
}

template <> Vc_INTRINSIC y_i32 Vc_VDECL convert_to<y_i32>(y_f32 v0)
{
    return _mm256_cvttps_epi32(v0);
}

// from double{{{2
template <> Vc_INTRINSIC y_i32 Vc_VDECL convert_to<y_i32>(x_f64 v)
{
    return zeroExtend(_mm_cvttpd_epi32(v));
}

template <> Vc_INTRINSIC y_i32 Vc_VDECL convert_to<y_i32>(y_f64 v)
{
    return zeroExtend(_mm256_cvttpd_epi32(v));
}

template <> Vc_INTRINSIC y_i32 Vc_VDECL convert_to<y_i32>(y_f64 v0, y_f64 v1)
{
#ifdef Vc_HAVE_AVX512F
    return _mm512_cvttpd_epi32(concat(v0, v1));
#else
    return concat(_mm256_cvttpd_epi32(v0), _mm256_cvttpd_epi32(v1));
#endif
}

#ifdef Vc_HAVE_AVX512F
template <> Vc_INTRINSIC y_i32 Vc_VDECL convert_to<y_i32>(z_f64 v0)
{
    return _mm512_cvttpd_epi32(v0);
}
#endif

// from llong{{{2
template <> Vc_INTRINSIC y_i32 Vc_VDECL convert_to<y_i32>(x_i64 v) {
    return zeroExtend(convert_to<x_i32>(v));
}

template <> Vc_INTRINSIC y_i32 Vc_VDECL convert_to<y_i32>(y_i64 v0)
{
#ifdef Vc_HAVE_AVX512VL
    return zeroExtend(_mm256_cvtepi64_epi32(v0));
#elif defined Vc_HAVE_AVX512F
    return _mm512_cvtepi64_epi32(zeroExtend(v0));
#elif defined Vc_HAVE_AVX2
    const auto vabxxcdxx = _mm256_shuffle_epi32(v0, 8);
    const auto v00ab00cd = _mm256_srli_si256(vabxxcdxx, 8);
    return _mm256_permute4x64_epi64(v00ab00cd, 1 + 4 * 3); // abcd0000
#else
    return intrin_cast<__m256i>(zeroExtend(
        _mm_shuffle_ps(_mm_castsi128_ps(lo128(v0)), _mm_castsi128_ps(hi128(v0)),
                       0x01 * 0 + 0x04 * 2 + 0x10 * 0 + 0x40 * 2)));
#endif
}

template <> Vc_INTRINSIC y_i32 Vc_VDECL convert_to<y_i32>(y_i64 v0, y_i64 v1)
{
#ifdef Vc_HAVE_AVX512F
    return _mm512_cvtepi64_epi32(concat(v0, v1));
#elif defined Vc_HAVE_AVX2
    const auto v0_abxxcdxx = _mm256_shuffle_epi32(v0, 8);
    const auto v1_efxxghxx = _mm256_shuffle_epi32(v1, 8);
    const auto v_abefcdgh = _mm256_unpacklo_epi64(v0_abxxcdxx, v1_efxxghxx);
    return _mm256_permute4x64_epi64(v_abefcdgh, 0x01 * 0 + 0x04 * 2 + 0x10 * 1 + 0x40 * 3); // abcdefgh
#else
    return intrin_cast<__m256i>(
        concat(_mm_shuffle_ps(intrin_cast<__m128>(v0), hi128(intrin_cast<__m256>(v0)),
                              0x01 * 0 + 0x04 * 2 + 0x10 * 0 + 0x40 * 2),
               _mm_shuffle_ps(intrin_cast<__m128>(v1), hi128(intrin_cast<__m256>(v1)),
                              0x01 * 0 + 0x04 * 2 + 0x10 * 0 + 0x40 * 2)));
#endif
}

#ifdef Vc_HAVE_AVX512F
template <> Vc_INTRINSIC y_i32 Vc_VDECL convert_to<y_i32>(z_i64 v0)
{
    return _mm512_cvtepi64_epi32(v0);
}
#endif

// from ullong{{{2

// from int{{{2
template <> Vc_INTRINSIC y_i32 Vc_VDECL convert_to<y_i32>(x_i32 v) { return zeroExtend(v); }
template <> Vc_INTRINSIC y_i32 Vc_VDECL convert_to<y_i32>(y_i32 v) { return v; }

// from uint{{{2
template <> Vc_INTRINSIC y_i32 Vc_VDECL convert_to<y_i32>(x_u32 v) { return zeroExtend(v); }
template <> Vc_INTRINSIC y_i32 Vc_VDECL convert_to<y_i32>(y_u32 v) { return v.v(); }

// from short{{{2
template <> Vc_INTRINSIC y_i32 Vc_VDECL convert_to<y_i32>(x_i16 v) {
#ifdef Vc_HAVE_AVX2
    return _mm256_cvtepi16_epi32(v);
#else
    return concat(_mm_cvtepi16_epi32(v), _mm_cvtepi16_epi32(shift_right<8>(v)));
#endif
}

template <> Vc_INTRINSIC y_i32 Vc_VDECL convert_to<y_i32>(y_i16 v) {
    return convert_to<y_i32>(lo128(v));
}

// from ushort{{{2
template <> Vc_INTRINSIC y_i32 Vc_VDECL convert_to<y_i32>(x_u16 v) {
#ifdef Vc_HAVE_AVX2
    return _mm256_cvtepu16_epi32(v);
#else
    return concat(_mm_cvtepu16_epi32(v), _mm_cvtepu16_epi32(shift_right<8>(v)));
#endif
}

// from schar{{{2
template <> Vc_INTRINSIC y_i32 Vc_VDECL convert_to<y_i32>(x_i08 v) {
#ifdef Vc_HAVE_AVX2
    return _mm256_cvtepi8_epi32(v);
#else
    return concat(_mm_cvtepi8_epi32(v), _mm_cvtepi8_epi32(shift_right<4>(v)));
#endif
}

template <> Vc_INTRINSIC y_i32 Vc_VDECL convert_to<y_i32>(y_i08 v) {
    return convert_to<y_i32>(lo128(v));
}

// from uchar{{{2
template <> Vc_INTRINSIC y_i32 Vc_VDECL convert_to<y_i32>(x_u08 v) {
#ifdef Vc_HAVE_AVX2
    return _mm256_cvtepu8_epi32(v);
#else
    return concat(_mm_cvtepu8_epi32(v), _mm_cvtepu8_epi32(shift_right<4>(v)));
#endif
}

template <> Vc_INTRINSIC y_i32 Vc_VDECL convert_to<y_i32>(y_u08 v) {
    return convert_to<y_i32>(lo128(v));
}

//}}2
#endif  // Vc_HAVE_AVX

//convert_to<z_i32> (int, 16){{{1
#ifdef Vc_HAVE_AVX512F
//from llong{{{2
template <> Vc_INTRINSIC z_i32 Vc_VDECL convert_to<z_i32>(z_i64 v0)
{
    return zeroExtend(_mm512_cvtepi64_epi32(v0));
}

template <> Vc_INTRINSIC z_i32 Vc_VDECL convert_to<z_i32>(z_i64 v0, z_i64 v1)
{
    return concat(_mm512_cvtepi64_epi32(v0), _mm512_cvtepi64_epi32(v1));
}

//from int{{{2
template <> Vc_INTRINSIC z_i32 Vc_VDECL convert_to<z_i32>(z_i32 v0) { return v0; }

//from short{{{2
template <> Vc_INTRINSIC z_i32 Vc_VDECL convert_to<z_i32>(y_i16 v0)
{
    return _mm512_cvtepi16_epi32(v0);
}

//from ushort{{{2
template <> Vc_INTRINSIC z_i32 Vc_VDECL convert_to<z_i32>(y_u16 v0)
{
    return _mm512_cvtepu16_epi32(v0);
}

//from schar{{{2
template <> Vc_INTRINSIC z_i32 Vc_VDECL convert_to<z_i32>(x_i08 v0)
{
    return _mm512_cvtepi8_epi32(v0);
}

// from uchar{{{2
template <> Vc_INTRINSIC z_i32 Vc_VDECL convert_to<z_i32>(x_u08 v0)
{
    return _mm512_cvtepu8_epi32(v0);
}

//from double{{{2
template <> Vc_INTRINSIC z_i32 Vc_VDECL convert_to<z_i32>(z_f64 v0, z_f64 v1)
{
    return concat(_mm512_cvttpd_epi32(v0), _mm512_cvttpd_epi32(v1));
}

//from float{{{2
template <> Vc_INTRINSIC z_i32 Vc_VDECL convert_to<z_i32>(z_f32 v0)
{
    return _mm512_cvttps_epi32(v0);
}

//}}}2
#endif  // Vc_HAVE_AVX512F

// convert_to<x_u32>{{{1
// from float{{{2
template <> Vc_INTRINSIC x_u32 Vc_VDECL convert_to<x_u32>(x_f32 v) {
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
template <> Vc_INTRINSIC x_u32 Vc_VDECL convert_to<x_u32>(x_f64 v) {
#ifdef Vc_HAVE_AVX512VL
    return _mm_cvttpd_epu32(v);
#elif defined Vc_HAVE_SSE4_1
    return _mm_xor_si128(
        _mm_cvttpd_epi32(_mm_sub_pd(_mm_floor_pd(v), _mm_set1_pd(0x80000000u))),
        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(sse_const::signMaskFloat)));
#else
    return {v[0], v[1], 0, 0};
#endif
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_u32 Vc_VDECL convert_to<x_u32>(y_f64 v)
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

template <> Vc_INTRINSIC x_u32 Vc_VDECL convert_to<x_u32>(x_f64 v0, x_f64 v1)
{
#ifdef Vc_HAVE_AVX
    return convert_to<x_u32>(y_f64(concat(v0, v1)));
#else
    return _mm_unpacklo_epi64(convert_to<x_u32>(v0), convert_to<x_u32>(v1));
#endif
}

//convert_to<y_u32>{{{1
#ifdef Vc_HAVE_AVX
//from double{{{2
template <> Vc_INTRINSIC y_u32 Vc_VDECL convert_to<y_u32>(y_f64 v0, y_f64 v1)
{
#ifdef Vc_HAVE_AVX512F
    return _mm512_cvttpd_epu32(concat(v0, v1));
#else
    return xor_(concat(_mm256_cvttpd_epi32(
                           _mm256_sub_pd(_mm256_floor_pd(v0), avx_2_pow_31<double>())),
                       _mm256_cvttpd_epi32(
                           _mm256_sub_pd(_mm256_floor_pd(v1), avx_2_pow_31<double>()))),
                avx_2_pow_31<uint>());
#endif
}

#ifdef Vc_HAVE_AVX512F
template <> Vc_INTRINSIC y_u32 Vc_VDECL convert_to<y_u32>(z_f64 v0)
{
    return _mm512_cvttpd_epu32(v0);
}
#endif  // Vc_HAVE_AVX512F

//from float{{{2
template <> Vc_INTRINSIC y_u32 Vc_VDECL convert_to<y_u32>(x_f32 v0)
{
    return zeroExtend(convert_to<x_u32>(v0));
}

template <> Vc_INTRINSIC y_u32 Vc_VDECL convert_to<y_u32>(y_f32 v0)
{
#if defined Vc_HAVE_AVX512F && defined Vc_HAVE_AVX512VL
    return _mm256_cvttps_epu32(v0);
#elif defined Vc_HAVE_AVX512F
    return lo256(_mm512_cvttps_epu32(intrin_cast<__m512>(v0)));
#else
    return _mm256_blendv_epi8(
        _mm256_cvttps_epi32(v0),
        _mm256_add_epi32(_mm256_cvttps_epi32(_mm256_sub_ps(v0, avx_2_pow_31<float>())),
                         avx_2_pow_31<uint>()),
        _mm256_castps_si256(_mm256_cmp_ps(v0, avx_2_pow_31<float>(), _CMP_NLT_US)));
#endif
}

//}}}2
#endif  // Vc_HAVE_AVX512F

//convert_to<z_u32>{{{1
#ifdef Vc_HAVE_AVX512F
//from double{{{2
template <> Vc_INTRINSIC z_u32 Vc_VDECL convert_to<z_u32>(z_f64 v0, z_f64 v1)
{
    return concat(_mm512_cvttpd_epu32(v0), _mm512_cvttpd_epu32(v1));
}

//from float{{{2
template <> Vc_INTRINSIC z_u32 Vc_VDECL convert_to<z_u32>(z_f32 v0)
{
    return _mm512_cvttps_epu32(v0);
}

//}}}2
#endif  // Vc_HAVE_AVX512F

//--------------------short & ushort{{{1
//
// convert_to<x_i16> (short, 8){{{1
// from llong{{{2
template <> Vc_INTRINSIC x_i16 Vc_VDECL convert_to<x_i16>(x_i64 v) {
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

template <> Vc_INTRINSIC x_i16 Vc_VDECL convert_to<x_i16>(x_i64 v0, x_i64 v1)
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
Vc_INTRINSIC x_i16 Vc_VDECL convert_to<x_i16>(x_i64 v0, x_i64 v1, x_i64 v2, x_i64 v3)
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
template <> Vc_INTRINSIC x_i16 Vc_VDECL convert_to<x_i16>(y_i64 v0, y_i64 v1)
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
template <> Vc_INTRINSIC x_i16 Vc_VDECL convert_to<x_i16>(z_i64 v0)
{
    return _mm512_cvtepi64_epi16(v0);
}
#endif  // Vc_HAVE_AVX512F

// from int{{{2
template <> Vc_INTRINSIC x_i16 Vc_VDECL convert_to<x_i16>(x_i32 v) {
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

template <> Vc_INTRINSIC x_i16 Vc_VDECL convert_to<x_i16>(x_i32 v0, x_i32 v1)
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
template <> Vc_INTRINSIC x_i16 Vc_VDECL convert_to<x_i16>(y_i32 v0)
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
template <> Vc_INTRINSIC x_i16 Vc_VDECL convert_to<x_i16>(z_u64 v0)
{
    return _mm512_cvtepi64_epi16(v0);
}
#endif  // Vc_HAVE_AVX512F

// from short{{{2
template <> Vc_INTRINSIC x_i16 Vc_VDECL convert_to<x_i16>(x_i16 v) { return v; }

// from schar{{{2
template <> Vc_INTRINSIC x_i16 Vc_VDECL convert_to<x_i16>(x_i08 v) {
#ifdef Vc_HAVE_SSE4_1
   return _mm_cvtepi8_epi16(v);
#else
    return _mm_srai_epi16(_mm_unpacklo_epi8(v, v), 8);
#endif
}

// from uchar{{{2
template <> Vc_INTRINSIC x_i16 Vc_VDECL convert_to<x_i16>(x_u08 v) {
#ifdef Vc_HAVE_SSE4_1
   return _mm_cvtepu8_epi16(v);
#else
   return _mm_unpacklo_epi8(v, zero<__m128i>());
#endif
}

// from double{{{2
template <> Vc_INTRINSIC x_i16 Vc_VDECL convert_to<x_i16>(x_f64 v)
{
    return convert_to<x_i16>(convert_to<x_i32>(v));
}

template <> Vc_INTRINSIC x_i16 Vc_VDECL convert_to<x_i16>(x_f64 v0, x_f64 v1)
{
    return convert_to<x_i16>(convert_to<x_i32>(v0, v1));
}

template <>
Vc_INTRINSIC x_i16 Vc_VDECL convert_to<x_i16>(x_f64 v0, x_f64 v1, x_f64 v2, x_f64 v3)
{
    return convert_to<x_i16>(convert_to<x_i32>(v0, v1), convert_to<x_i32>(v2, v3));
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_i16 Vc_VDECL convert_to<x_i16>(y_f64 v0, y_f64 v1)
{
#ifdef Vc_HAVE_AVX512F
    return convert_to<x_i16>(y_i32(_mm512_cvttpd_epi32(concat(v0, v1))));
#else
    return convert_to<x_i16>(x_i32(_mm256_cvttpd_epi32(v0)),
                             x_i32(_mm256_cvttpd_epi32(v1)));
#endif
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
template <> Vc_INTRINSIC x_i16 Vc_VDECL convert_to<x_i16>(z_f64 v0)
{
    return convert_to<x_i16>(y_i32(_mm512_cvttpd_epi32(v0)));
}
#endif  // Vc_HAVE_AVX512F

// from float{{{2
template <> Vc_INTRINSIC x_i16 Vc_VDECL convert_to<x_i16>(x_f32 v)
{
    return convert_to<x_i16>(convert_to<x_i32>(v));
}

template <> Vc_INTRINSIC x_i16 Vc_VDECL convert_to<x_i16>(x_f32 v0, x_f32 v1)
{
    return convert_to<x_i16>(convert_to<x_i32>(v0), convert_to<x_i32>(v1));
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_i16 Vc_VDECL convert_to<x_i16>(y_f32 v0)
{
    return convert_to<x_i16>(convert_to<y_i32>(v0));
}
#endif  // Vc_HAVE_AVX

// convert_to<y_i16> (short, 16){{{1
#ifdef Vc_HAVE_AVX
// from llong{{{2
template <> Vc_INTRINSIC y_i16 Vc_VDECL convert_to<y_i16>(x_i64 v) {
#ifdef Vc_HAVE_AVX512VL
    return zeroExtend(_mm_cvtepi64_epi16(v));
#else
    return zeroExtend(_mm_shuffle_epi8(
        v, _mm_setr_epi8(0, 1, 8, 9, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80,
                         -0x80, -0x80, -0x80, -0x80, -0x80)));
#endif
}

#ifdef Vc_HAVE_AVX2
template <> Vc_INTRINSIC y_i16 Vc_VDECL convert_to<y_i16>(y_i64 v0)
{
#ifdef Vc_HAVE_AVX512F
    return zeroExtend(_mm512_cvtepi64_epi16(concat(v0, zero<__m256i>())));
#else
    auto a = _mm256_unpacklo_epi16(v0, zero<__m256i>());        // 04.. .... 26.. ....
    auto b = _mm256_unpackhi_epi16(v0, zero<__m256i>());        // 15.. .... 37.. ....
    auto c = _mm256_unpacklo_epi16(a, b);                       // 0145 .... 2367 ....
    return zeroExtend(_mm_unpacklo_epi32(lo128(c), hi128(c)));  // 0123 4567
#endif
}

template <> Vc_INTRINSIC y_i16 Vc_VDECL convert_to<y_i16>(y_i64 v0, y_i64 v1)
{
#ifdef Vc_HAVE_AVX512F
    return zeroExtend(_mm512_cvtepi64_epi16(concat(v0, v1)));
#else
    auto a = _mm256_unpacklo_epi16(v0, v1);                     // 04.. .... 26.. ....
    auto b = _mm256_unpackhi_epi16(v0, v1);                     // 15.. .... 37.. ....
    auto c = _mm256_unpacklo_epi16(a, b);                       // 0145 .... 2367 ....
    return zeroExtend(_mm_unpacklo_epi32(lo128(c), hi128(c)));  // 0123 4567
#endif
}

template <> Vc_INTRINSIC y_i16 Vc_VDECL convert_to<y_i16>(y_i64 v0, y_i64 v1, y_i64 v2, y_i64 v3)
{
#ifdef Vc_HAVE_AVX512F
    return concat(_mm512_cvtepi64_epi16(concat(v0, v1)),
                  _mm512_cvtepi64_epi16(concat(v2, v3)));
#else
    auto a = _mm256_unpacklo_epi16(v0, v1);                     // 04.. .... 26.. ....
    auto b = _mm256_unpackhi_epi16(v0, v1);                     // 15.. .... 37.. ....
    auto c = _mm256_unpacklo_epi16(v2, v3);                     // 8C.. .... AE.. ....
    auto d = _mm256_unpackhi_epi16(v2, v3);                     // 9D.. .... BF.. ....
    auto e = _mm256_unpacklo_epi16(a, b);                       // 0145 .... 2367 ....
    auto f = _mm256_unpacklo_epi16(c, d);                       // 89CD .... ABEF ....
    auto g = _mm256_unpacklo_epi64(e, f);                       // 0145 89CD 2367 ABEF
    return concat(_mm_unpacklo_epi32(lo128(g), hi128(g)),
                  _mm_unpackhi_epi32(lo128(g), hi128(g)));  // 0123 4567 89AB CDEF
#endif
}
#endif  // Vc_HAVE_AVX2

#ifdef Vc_HAVE_AVX512F
template <> Vc_INTRINSIC y_i16 Vc_VDECL convert_to<y_i16>(z_i64 v0)
{
    return zeroExtend(_mm512_cvtepi64_epi16(v0));
}

template <> Vc_INTRINSIC y_i16 Vc_VDECL convert_to<y_i16>(z_i64 v0, z_i64 v1)
{
    return concat(_mm512_cvtepi64_epi16(v0), _mm512_cvtepi64_epi16(v1));
}
#endif  // Vc_HAVE_AVX512F

// from int{{{2
template <> Vc_INTRINSIC y_i16 Vc_VDECL convert_to<y_i16>(x_i32 v) {
    return zeroExtend(convert_to<x_i16>(v));
}

template <> Vc_INTRINSIC y_i16 Vc_VDECL convert_to<y_i16>(y_i32 v0)
{
    return zeroExtend(convert_to<x_i16>(v0));
}

template <> Vc_INTRINSIC y_i16 Vc_VDECL convert_to<y_i16>(y_i32 v0, y_i32 v1)
{
#if defined Vc_HAVE_AVX512F
    return _mm512_cvtepi32_epi16(concat(v0, v1));
#else
    return concat(convert_to<x_i16>(v0), convert_to<x_i16>(v1));
#endif
}

#ifdef Vc_HAVE_AVX512F
template <> Vc_INTRINSIC y_i16 Vc_VDECL convert_to<y_i16>(z_i32 v0)
{
    return _mm512_cvtepi32_epi16(v0);
}
#endif  // Vc_HAVE_AVX512F

// from short{{{2
template <> Vc_INTRINSIC y_i16 Vc_VDECL convert_to<y_i16>(x_i16 v) { return zeroExtend(v); }
template <> Vc_INTRINSIC y_i16 Vc_VDECL convert_to<y_i16>(y_i16 v) { return v; }

// from ushort{{{2
template <> Vc_INTRINSIC y_i16 Vc_VDECL convert_to<y_i16>(x_u16 v) { return zeroExtend(v); }

// from schar{{{2
template <> Vc_INTRINSIC y_i16 Vc_VDECL convert_to<y_i16>(x_i08 v) {
#ifdef Vc_HAVE_AVX2
    return _mm256_cvtepi8_epi16(v);
#else   // Vc_HAVE_AVX2
    return concat(_mm_cvtepi8_epi16(v), _mm_cvtepi8_epi16(_mm_unpackhi_epi64(v, v)));
#endif  // Vc_HAVE_AVX2
}

template <> Vc_INTRINSIC y_i16 Vc_VDECL convert_to<y_i16>(y_i08 v) {
    return convert_to<y_i16>(lo128(v));
}

// from uchar{{{2
template <> Vc_INTRINSIC y_i16 Vc_VDECL convert_to<y_i16>(x_u08 v) {
#ifdef Vc_HAVE_AVX2
    return _mm256_cvtepu8_epi16(v);
#else   // Vc_HAVE_AVX2
    return concat(_mm_cvtepu8_epi16(v), _mm_cvtepu8_epi16(_mm_unpackhi_epi64(v, v)));
#endif  // Vc_HAVE_AVX2
}

template <> Vc_INTRINSIC y_i16 Vc_VDECL convert_to<y_i16>(y_u08 v) {
    return convert_to<y_i16>(lo128(v));
}

// from double{{{2
template <> Vc_INTRINSIC y_i16 Vc_VDECL convert_to<y_i16>(x_f64 v)
{
    return convert_to<y_i16>(convert_to<x_i32>(v));
}

template <> Vc_INTRINSIC y_i16 Vc_VDECL convert_to<y_i16>(y_f64 v)
{
    return convert_to<y_i16>(convert_to<x_i32>(v));
}

template <> Vc_INTRINSIC y_i16 Vc_VDECL convert_to<y_i16>(y_f64 v0, y_f64 v1)
{
    return convert_to<y_i16>(convert_to<y_i32>(v0, v1));
}

template <> Vc_INTRINSIC y_i16 Vc_VDECL convert_to<y_i16>(y_f64 v0, y_f64 v1, y_f64 v2, y_f64 v3)
{
    return convert_to<y_i16>(convert_to<y_i32>(v0, v1), convert_to<y_i32>(v2, v3));
}

#ifdef Vc_HAVE_AVX512F
template <> Vc_INTRINSIC y_i16 Vc_VDECL convert_to<y_i16>(z_f64 v0)
{
    return convert_to<y_i16>(y_i32(_mm512_cvttpd_epi32(v0)));
}

template <> Vc_INTRINSIC y_i16 Vc_VDECL convert_to<y_i16>(z_f64 v0, z_f64 v1)
{
    return _mm512_cvtepi32_epi16(concat(_mm512_cvttpd_epi32(v0), _mm512_cvttpd_epi32(v1)));
}
#endif  // Vc_HAVE_AVX512F

// from float{{{2
template <> Vc_INTRINSIC y_i16 Vc_VDECL convert_to<y_i16>(x_f32 v)
{
    return convert_to<y_i16>(convert_to<x_i32>(v));
}

template <> Vc_INTRINSIC y_i16 Vc_VDECL convert_to<y_i16>(y_f32 v0)
{
    return convert_to<y_i16>(convert_to<y_i32>(v0));
}

template <> Vc_INTRINSIC y_i16 Vc_VDECL convert_to<y_i16>(y_f32 v0, y_f32 v1)
{
    return convert_to<y_i16>(convert_to<y_i32>(v0), convert_to<y_i32>(v1));
}

#ifdef Vc_HAVE_AVX512F
template <> Vc_INTRINSIC y_i16 Vc_VDECL convert_to<y_i16>(z_f32 v0)
{
    return _mm512_cvtepi32_epi16(_mm512_cvttps_epi32(v0));
}
#endif  // Vc_HAVE_AVX512F

//}}}2
#endif  // Vc_HAVE_AVX

// convert_to<z_i16> (short, 32){{{1
#ifdef Vc_HAVE_AVX512F
//from llong{{{2
template <> Vc_INTRINSIC z_i16 Vc_VDECL convert_to<z_i16>(z_i64 v0)
{
    return zeroExtend(zeroExtend(_mm512_cvtepi64_epi16(v0)));
}

template <> Vc_INTRINSIC z_i16 Vc_VDECL convert_to<z_i16>(z_i64 v0, z_i64 v1)
{
    return zeroExtend(concat(_mm512_cvtepi64_epi16(v0), _mm512_cvtepi64_epi16(v1)));
}

template <>
Vc_INTRINSIC z_i16 Vc_VDECL convert_to<z_i16>(z_i64 v0, z_i64 v1, z_i64 v2, z_i64 v3)
{
    return concat(concat(_mm512_cvtepi64_epi16(v0), _mm512_cvtepi64_epi16(v1)),
                  concat(_mm512_cvtepi64_epi16(v2), _mm512_cvtepi64_epi16(v3)));
}

// from int{{{2
template <> Vc_INTRINSIC z_i16 Vc_VDECL convert_to<z_i16>(z_i32 v0)
{
    return zeroExtend(_mm512_cvtepi32_epi16(v0));
}

template <> Vc_INTRINSIC z_i16 Vc_VDECL convert_to<z_i16>(z_i32 v0, z_i32 v1)
{
    return concat(_mm512_cvtepi32_epi16(v0), _mm512_cvtepi32_epi16(v1));
}

//from short{{{2
template <> Vc_INTRINSIC z_i16 Vc_VDECL convert_to<z_i16>(z_i16 v0) { return v0; }

//from schar{{{2
template <> Vc_INTRINSIC z_i16 Vc_VDECL convert_to<z_i16>(y_i08 v0)
{
#ifdef Vc_HAVE_AVX512BW
    return _mm512_cvtepi8_epi16(v0);
#else
    return concat(_mm256_cvtepi8_epi16(lo128(v0)), _mm256_cvtepi8_epi16(hi128(v0)));
#endif
}

//from uchar{{{2
template <> Vc_INTRINSIC z_i16 Vc_VDECL convert_to<z_i16>(y_u08 v0)
{
#ifdef Vc_HAVE_AVX512BW
    return _mm512_cvtepu8_epi16(v0);
#else
    return concat(_mm256_cvtepu8_epi16(lo128(v0)), _mm256_cvtepu8_epi16(hi128(v0)));
#endif
}

//from double{{{2
template <> Vc_INTRINSIC z_i16 Vc_VDECL convert_to<z_i16>(z_f64 v0, z_f64 v1, z_f64 v2, z_f64 v3)
{
    return convert_to<z_i16>(convert_to<z_i32>(v0, v1), convert_to<z_i32>(v2, v3));
}

//from float{{{2
template <> Vc_INTRINSIC z_i16 Vc_VDECL convert_to<z_i16>(z_f32 v0, z_f32 v1)
{
    return convert_to<z_i16>(convert_to<z_i32>(v0), convert_to<z_i32>(v1));
}

//}}}2
#endif  // Vc_HAVE_AVX512F

// no unsigned specializations needed, conversion goes via int32{{{1
//
//--------------------schar & uchar{{{1
//
// convert_to<x_i08> (signed char, 16){{{1
// from llong{{{2
template <> Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(x_i64 v) {
#ifdef Vc_HAVE_AVX512VL
    return _mm_cvtepi64_epi8(v);
#else
    return {v.m(0), v.m(1), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#endif
}

template <> Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(x_i64 v0, x_i64 v1)
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
Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(x_i64 v0, x_i64 v1, x_i64 v2, x_i64 v3)
{
#ifdef Vc_HAVE_AVX512F
    return _mm512_cvtepi64_epi8(concat(concat(v0, v1), concat(v2, v3)));
#else
    return {v0.m(0), v0.m(1), v1.m(0), v1.m(1), v2.m(0), v2.m(1), v3.m(0), v3.m(1),
            0,       0,       0,       0,       0,       0,       0,       0};
#endif
}

template <>
Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(x_i64 v0, x_i64 v1, x_i64 v2, x_i64 v3,
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
Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(y_i64 v0, y_i64 v1, y_i64 v2, y_i64 v3)
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
template <> Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(z_i64 v0)
{
    return _mm512_cvtepi64_epi8(v0);
}

template <> Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(z_i64 v0, z_i64 v1)
{
    return _mm_unpacklo_epi64(_mm512_cvtepi64_epi8(v0), _mm512_cvtepi64_epi8(v1));
}
#endif  // Vc_HAVE_AVX512F

// from int{{{2
template <> Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(x_i32 v) {
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

template <> Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(x_i32 v0, x_i32 v1)
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
    return detail::and_(
        e, _mm_loadl_epi64(reinterpret_cast<const __m128i*>(sse_const::AllBitsSet)));
#endif
}

Vc_INTRINSIC x_i08 Vc_VDECL sse2_convert_to_i08(x_i32 v0, x_i32 v1, x_i32 v2, x_i32 v3)
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
template <> Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(y_i32 v0, y_i32 v1)
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
Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(x_i32 v0, x_i32 v1, x_i32 v2, x_i32 v3)
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
template <> Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(z_i32 v0)
{
    return _mm512_cvtepi32_epi8(v0);
}
#endif  // Vc_HAVE_AVX512F

// from short{{{2
template <> Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(x_i16 v) {
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

template <> Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(x_i16 v0, x_i16 v1)
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
template <> Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(y_i16 v0)
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

// from [su]char{{{2
template <> Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(x_i08 v) { return v; }
template <> Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(x_u08 v) { return v.v(); }

// from float{{{2
template <> Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(x_f32 v)
{
    return convert_to<x_i08>(convert_to<x_i32>(v));
}

template <> Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(x_f32 v0, x_f32 v1)
{
    return convert_to<x_i08>(convert_to<x_i32>(v0), convert_to<x_i32>(v1));
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(y_f32 v0, y_f32 v1)
{
#ifdef Vc_HAVE_AVX512F
    return _mm512_cvtepi32_epi8(_mm512_cvttps_epi32(concat(v0, v1)));
#else
    return convert_to<x_i08>(convert_to<y_i32>(v0), convert_to<y_i32>(v1));
#endif
}
#endif  // Vc_HAVE_AVX

template <>
Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(x_f32 v0, x_f32 v1, x_f32 v2, x_f32 v3)
{
#ifdef Vc_HAVE_AVX
    return convert_to<x_i08>(y_f32(concat(v0, v1)), y_f32(concat(v2, v3)));
#else
    return convert_to<x_i08>(convert_to<x_i32>(v0), convert_to<x_i32>(v1),
                             convert_to<x_i32>(v2), convert_to<x_i32>(v3));
#endif
}

#ifdef Vc_HAVE_AVX512F
template <> Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(z_f32 v0)
{
    return _mm512_cvtepi32_epi8(_mm512_cvttps_epi32(v0));
}
#endif  // Vc_HAVE_AVX512F

// from double{{{2
template <> Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(x_f64 v)
{
    return convert_to<x_i08>(convert_to<x_i32>(v));
}

template <> Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(x_f64 v0, x_f64 v1)
{
    return convert_to<x_i08>(convert_to<x_i32>(v0, v1));
}

template <>
Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(x_f64 v0, x_f64 v1, x_f64 v2, x_f64 v3)
{
    return convert_to<x_i08>(convert_to<x_i32>(v0, v1), convert_to<x_i32>(v2, v3));
}

template <>
Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(x_f64 v0, x_f64 v1, x_f64 v2, x_f64 v3,
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
Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(y_f64 v0, y_f64 v1, y_f64 v2, y_f64 v3)
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
template <> Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(z_f64 v0)
{
#ifdef Vc_HAVE_AVX512VL
    return _mm256_cvtepi32_epi8(_mm512_cvttpd_epi32(v0));
#else
    return _mm512_cvtepi32_epi8(zeroExtend(_mm512_cvttpd_epi32(v0)));
#endif
}

template <>
Vc_INTRINSIC x_i08 Vc_VDECL convert_to<x_i08>(z_f64 v0, z_f64 v1)
{
    return _mm512_cvtepi32_epi8(concat(_mm512_cvttpd_epi32(v0), _mm512_cvttpd_epi32(v1)));
}
#endif  // Vc_HAVE_AVX512F

// convert_to<y_i08> (signed char, 32){{{1
#ifdef Vc_HAVE_AVX
//from llong{{{2
#ifdef Vc_HAVE_AVX512F
template <> Vc_INTRINSIC y_i08 Vc_VDECL convert_to<y_i08>(z_i64 v0)
{
    return zeroExtend(_mm512_cvtepi64_epi8(v0));
}

template <> Vc_INTRINSIC y_i08 Vc_VDECL convert_to<y_i08>(z_i64 v0, z_i64 v1)
{
    return zeroExtend(_mm_unpacklo_epi64(_mm512_cvtepi64_epi8(v0), _mm512_cvtepi64_epi8(v1)));
}

template <>
Vc_INTRINSIC y_i08 Vc_VDECL convert_to<y_i08>(z_i64 v0, z_i64 v1, z_i64 v2, z_i64 v3)
{
    return concat(_mm_unpacklo_epi64(_mm512_cvtepi64_epi8(v0), _mm512_cvtepi64_epi8(v1)),
                  _mm_unpacklo_epi64(_mm512_cvtepi64_epi8(v2), _mm512_cvtepi64_epi8(v3)));
}
#endif

template <>
Vc_INTRINSIC y_i08 Vc_VDECL convert_to<y_i08>(y_i64 v0, y_i64 v1, y_i64 v2, y_i64 v3)
{
    return zeroExtend(convert_to<x_i08>(v0, v1, v2, v3));
}

template <>
Vc_INTRINSIC y_i08 Vc_VDECL convert_to<y_i08>(y_i64 v0, y_i64 v1, y_i64 v2, y_i64 v3,
                                              y_i64 v4, y_i64 v5, y_i64 v6, y_i64 v7)
{
#ifdef Vc_HAVE_AVX512F
    return convert_to<y_i08>(concat(v0, v1), concat(v2, v3), concat(v4, v5),
                             concat(v6, v7));
#elif defined Vc_HAVE_AVX2
    auto a = or_(
        or_(or_(_mm256_srli_epi64(_mm256_slli_epi64(v0, 56), 56),
                _mm256_srli_epi64(_mm256_slli_epi64(v1, 56), 48)),
            or_(_mm256_srli_epi64(_mm256_slli_epi64(v2, 56), 40),
                _mm256_srli_epi64(_mm256_slli_epi64(v3, 56),
                                  32))),  // 048C .... 159D .... 26AE .... 37BF ....
        or_(or_(_mm256_srli_epi64(_mm256_slli_epi64(v4, 56), 24),
                _mm256_srli_epi64(_mm256_slli_epi64(v5, 56), 16)),
            or_(_mm256_srli_epi64(_mm256_slli_epi64(v6, 56), 8),
                _mm256_slli_epi64(v7, 56)))   // .... GKOS .... HLPT .... IMQU .... JNRV
        );                                    // 048C GKOS 159D HLPT 26AE IMQU 37BF JNRV
    auto b = _mm256_unpackhi_epi64(a, a);     // 159D HLPT 159D HLPT 37BF JNRV 37BF JNRV
    auto c = _mm256_unpacklo_epi8(a, b);      // 0145 89CD GHKL OPST 2367 ABEF IJMN QRUV
    return concat(_mm_unpacklo_epi16(lo128(c), hi128(c)),   // 0123 4567 89AB CDEF
                  _mm_unpackhi_epi16(lo128(c), hi128(c)));  // GHIJ KLMN OPQR STUV
#else
    // I don't care for non-AVX2 users that convert between non-float AVX vectors
    return generate_from_n_evaluations<32, y_i08>([&](auto i) {
        switch (i / 4) {
        case 0: return static_cast<schar>(v0[i % 4]); break;
        case 1: return static_cast<schar>(v1[i % 4]); break;
        case 2: return static_cast<schar>(v2[i % 4]); break;
        case 3: return static_cast<schar>(v3[i % 4]); break;
        case 4: return static_cast<schar>(v4[i % 4]); break;
        case 5: return static_cast<schar>(v5[i % 4]); break;
        case 6: return static_cast<schar>(v6[i % 4]); break;
        case 7: return static_cast<schar>(v7[i % 4]); break;
        }
    });
#endif
}

// from int{{{2
template <> Vc_INTRINSIC y_i08 Vc_VDECL convert_to<y_i08>(y_i32 v0, y_i32 v1, y_i32 v2, y_i32 v3)
{
#ifdef Vc_HAVE_AVX512F
    return concat(_mm512_cvtepi32_epi8(concat(v0, v1)),
                  _mm512_cvtepi32_epi8(concat(v2, v3)));
#else   // Vc_HAVE_AVX512F
    return concat(convert_to<x_i08>(v0, v1), convert_to<x_i08>(v2, v3));
#endif  // Vc_HAVE_AVX512F
}

#ifdef Vc_HAVE_AVX512F
template <> Vc_INTRINSIC y_i08 Vc_VDECL convert_to<y_i08>(z_i32 v0, z_i32 v1)
{
    return concat(_mm512_cvtepi32_epi8(v0), _mm512_cvtepi32_epi8(v1));
}
#endif  // Vc_HAVE_AVX512F

//from short{{{2
#ifdef Vc_HAVE_AVX512F
template <> Vc_INTRINSIC y_i08 Vc_VDECL convert_to<y_i08>(z_i16 v0)
{
#ifdef Vc_HAVE_AVX512BW
    return _mm512_cvtepi16_epi8(v0);
#else
    const auto mask = _mm512_set1_epi32(0x00ff00ff);
    auto a = and_(v0, mask);
    auto l0 = lo256(a);                      // a0b0 c0d0 e0f0 g0h0 i0j0 k0l0 m0n0 o0p0
    auto h0 = hi256(a);                      // q0r0 s0t0 u0v0 w0x0 y0z0 A0B0 C0D0 E0F0
    auto l1 = _mm256_unpacklo_epi8(l0, h0);  // aq00 br00 cs00 dt00 iy00 jz00 kA00 lB00
    auto h1 = _mm256_unpackhi_epi8(l0, h0);  // eu00 fv00 gw00 hx00 mC00 nD00 oE00 pF00
    l0 = _mm256_unpacklo_epi8(l1, h1);       // aequ 0000 bfrv 0000 imyC 0000 jnzD 0000
    h0 = _mm256_unpackhi_epi8(l1, h1);       // cgsw 0000 dhtx 0000 koAE 0000 lpBF 0000
    l1 = _mm256_unpacklo_epi8(l0, h0);       // aceg qsuw 0000 0000 ikmo yACE 0000 0000
    h1 = _mm256_unpackhi_epi8(l0, h0);       // bdfh rtvx 0000 0000 jlnp zBDF 0000 0000
    l0 = _mm256_unpacklo_epi8(l1, h1);       // abcd efgh qrst uvwx ijkl mnop yzAB CDEF
    return _mm256_permute4x64_epi64(l0, 0xd8);
#endif  // Vc_HAVE_AVX512BW
}
#endif  // Vc_HAVE_AVX512F

template <> Vc_INTRINSIC y_i08 Vc_VDECL convert_to<y_i08>(y_i16 v0)
{
    return zeroExtend(convert_to<x_i08>(v0));
}

template <> Vc_INTRINSIC y_i08 Vc_VDECL convert_to<y_i08>(y_i16 v0, y_i16 v1)
{
#ifdef Vc_HAVE_AVX512F
    return _mm512_cvtepi16_epi8(concat(v0, v1));
#else
    return concat(convert_to<x_i08>(v0), convert_to<x_i08>(v1));
#endif
}

//from schar{{{2
template <> Vc_INTRINSIC y_i08 Vc_VDECL convert_to<y_i08>(y_i08 v0) { return v0; }

//from double{{{2
template <>
Vc_INTRINSIC y_i08 Vc_VDECL convert_to<y_i08>(y_f64 v0, y_f64 v1, y_f64 v2, y_f64 v3, y_f64 v4,
                                     y_f64 v5, y_f64 v6, y_f64 v7)
{
    return convert_to<y_i08>(convert_to<y_i32>(v0, v1), convert_to<y_i32>(v2, v3),
                             convert_to<y_i32>(v4, v5), convert_to<y_i32>(v6, v7));
}

#ifdef Vc_HAVE_AVX512F
template <> Vc_INTRINSIC y_i08 Vc_VDECL convert_to<y_i08>(z_f64 v0, z_f64 v1, z_f64 v2, z_f64 v3)
{
    return concat(
        _mm512_cvtepi32_epi8(concat(_mm512_cvttpd_epi32(v0), _mm512_cvttpd_epi32(v1))),
        _mm512_cvtepi32_epi8(concat(_mm512_cvttpd_epi32(v2), _mm512_cvttpd_epi32(v3))));
}
#endif  // Vc_HAVE_AVX512F

//from float{{{2
template <> Vc_INTRINSIC y_i08 Vc_VDECL convert_to<y_i08>(y_f32 v0, y_f32 v1, y_f32 v2, y_f32 v3)
{
    return convert_to<y_i08>(convert_to<y_i32>(v0), convert_to<y_i32>(v1),
                             convert_to<y_i32>(v2), convert_to<y_i32>(v3));
}

#ifdef Vc_HAVE_AVX512F
template <> Vc_INTRINSIC y_i08 Vc_VDECL convert_to<y_i08>(z_f32 v0, z_f32 v1)
{
    return concat(_mm512_cvtepi32_epi8(_mm512_cvttps_epi32(v0)),
                  _mm512_cvtepi32_epi8(_mm512_cvttps_epi32(v1)));
}
#endif  // Vc_HAVE_AVX512F

//}}}2
#endif  // Vc_HAVE_AVX
// convert_to<z_i08> (signed char, 64){{{1
#ifdef Vc_HAVE_AVX512F
//from llong{{{2
template <>
Vc_INTRINSIC z_i08 Vc_VDECL convert_to<z_i08>(z_i64 v0, z_i64 v1, z_i64 v2, z_i64 v3,
                                              z_i64 v4, z_i64 v5, z_i64 v6, z_i64 v7)
{
    return concat(
        concat(_mm_unpacklo_epi64(_mm512_cvtepi64_epi8(v0), _mm512_cvtepi64_epi8(v1)),
               _mm_unpacklo_epi64(_mm512_cvtepi64_epi8(v2), _mm512_cvtepi64_epi8(v3))),
        concat(_mm_unpacklo_epi64(_mm512_cvtepi64_epi8(v4), _mm512_cvtepi64_epi8(v5)),
               _mm_unpacklo_epi64(_mm512_cvtepi64_epi8(v6), _mm512_cvtepi64_epi8(v7))));
}

template <>
Vc_INTRINSIC z_i08 Vc_VDECL convert_to<z_i08>(z_i64 v0, z_i64 v1, z_i64 v2, z_i64 v3)
{
    return zeroExtend(
        concat(_mm_unpacklo_epi64(_mm512_cvtepi64_epi8(v0), _mm512_cvtepi64_epi8(v1)),
               _mm_unpacklo_epi64(_mm512_cvtepi64_epi8(v2), _mm512_cvtepi64_epi8(v3))));
}

template <> Vc_INTRINSIC z_i08 Vc_VDECL convert_to<z_i08>(z_i64 v0, z_i64 v1)
{
    return zeroExtend(zeroExtend(
        _mm_unpacklo_epi64(_mm512_cvtepi64_epi8(v0), _mm512_cvtepi64_epi8(v1))));
}

template <> Vc_INTRINSIC z_i08 Vc_VDECL convert_to<z_i08>(z_i64 v0)
{
    return zeroExtend(zeroExtend(_mm512_cvtepi64_epi8(v0)));
}

// from int{{{2
template <> Vc_INTRINSIC z_i08 Vc_VDECL convert_to<z_i08>(z_i32 v0, z_i32 v1, z_i32 v2, z_i32 v3)
{
    return concat(concat(_mm512_cvtepi32_epi8(v0), _mm512_cvtepi32_epi8(v1)),
                  concat(_mm512_cvtepi32_epi8(v2), _mm512_cvtepi32_epi8(v3)));
}

template <> Vc_INTRINSIC z_i08 Vc_VDECL convert_to<z_i08>(z_i32 v0, z_i32 v1)
{
    return zeroExtend(concat(_mm512_cvtepi32_epi8(v0), _mm512_cvtepi32_epi8(v1)));
}

template <> Vc_INTRINSIC z_i08 Vc_VDECL convert_to<z_i08>(z_i32 v0)
{
    return zeroExtend(zeroExtend(_mm512_cvtepi32_epi8(v0)));
}

//from short{{{2
template <> Vc_INTRINSIC z_i08 Vc_VDECL convert_to<z_i08>(z_i16 v0, z_i16 v1)
{
    return concat(_mm512_cvtepi16_epi8(v0), _mm512_cvtepi16_epi8(v1));
}

template <> Vc_INTRINSIC z_i08 Vc_VDECL convert_to<z_i08>(z_i16 v0)
{
    return zeroExtend(_mm512_cvtepi16_epi8(v0));
}

//from schar{{{2
template <> Vc_INTRINSIC z_i08 Vc_VDECL convert_to<z_i08>(z_i08 v0) { return v0; }

//from double{{{2
template <>
Vc_INTRINSIC z_i08 Vc_VDECL convert_to<z_i08>(z_f64 v0, z_f64 v1, z_f64 v2, z_f64 v3, z_f64 v4,
                                     z_f64 v5, z_f64 v6, z_f64 v7)
{
    return convert_to<z_i08>(convert_to<z_i32>(v0, v1), convert_to<z_i32>(v2, v3),
                             convert_to<z_i32>(v4, v5), convert_to<z_i32>(v6, v7));
}

//from float{{{2
template <> Vc_INTRINSIC z_i08 Vc_VDECL convert_to<z_i08>(z_f32 v0, z_f32 v1, z_f32 v2, z_f32 v3)
{
    return convert_to<z_i08>(convert_to<z_i32>(v0), convert_to<z_i32>(v1),
                             convert_to<z_i32>(v2), convert_to<z_i32>(v3));
}

//}}}2
#endif  // Vc_HAVE_AVX512F
// no unsigned specializations needed, conversion goes via int32{{{1
//
//--------------------double{{{1
//
// convert_to<x_f64> (double, 2){{{1
// from float{{{2
template <> Vc_INTRINSIC x_f64 Vc_VDECL convert_to<x_f64>(x_f32 v) { return _mm_cvtps_pd(v); }

// from double{{{2
template <> Vc_INTRINSIC x_f64 Vc_VDECL convert_to<x_f64>(x_f64 v) { return v; }

// from llong{{{2
template <> Vc_INTRINSIC x_f64 Vc_VDECL convert_to<x_f64>(x_i64 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm_cvtepi64_pd(v);
#else
    return x_f64{v.m(0), v.m(1)};
#endif
}

// from ullong{{{2
template <> Vc_INTRINSIC x_f64 Vc_VDECL convert_to<x_f64>(x_u64 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm_cvtepu64_pd(v);
#else
    return x_f64{v.m(0), v.m(1)};
#endif
}

// from int{{{2
template <> Vc_INTRINSIC x_f64 Vc_VDECL convert_to<x_f64>(x_i32 v) { return _mm_cvtepi32_pd(v); }

// from uint{{{2
template <> Vc_INTRINSIC x_f64 Vc_VDECL convert_to<x_f64>(x_u32 v)
{
#ifdef Vc_HAVE_AVX512VL
    return _mm_cvtepu32_pd(v);
#elif defined Vc_HAVE_AVX512F
    return lo128(_mm512_cvtepu32_pd(intrin_cast<__m256i>(v)));
#else
    return _mm_add_pd(_mm_cvtepi32_pd(_mm_xor_si128(v, lowest16<int>())),
                      _mm_set1_pd(1u << 31));
#endif
}

// from short{{{2
template <> Vc_INTRINSIC x_f64 Vc_VDECL convert_to<x_f64>(x_i16 v) { return convert_to<x_f64>(convert_to<x_i32>(v)); }

// from ushort{{{2
template <> Vc_INTRINSIC x_f64 Vc_VDECL convert_to<x_f64>(x_u16 v) { return convert_to<x_f64>(convert_to<x_i32>(v)); }

// from schar{{{2
template <> Vc_INTRINSIC x_f64 Vc_VDECL convert_to<x_f64>(x_i08 v) { return convert_to<x_f64>(convert_to<x_i32>(v)); }

// from uchar{{{2
template <> Vc_INTRINSIC x_f64 Vc_VDECL convert_to<x_f64>(x_u08 v) { return convert_to<x_f64>(convert_to<x_i32>(v)); }

// convert_to<y_f64> (double, 4){{{1
#ifdef Vc_HAVE_AVX
// from float{{{2
template <> Vc_INTRINSIC y_f64 Vc_VDECL convert_to<y_f64>(x_f32 v) { return _mm256_cvtps_pd(v); }

// from double{{{2
template <> Vc_INTRINSIC y_f64 Vc_VDECL convert_to<y_f64>(y_f64 v) { return v; }

// from llong{{{2
template <> Vc_INTRINSIC y_f64 Vc_VDECL convert_to<y_f64>(y_i64 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm256_cvtepi64_pd(v);
#else
    return y_f64{v.m(0), v.m(1), v.m(2), v.m(3)};
#endif
}

// from ullong{{{2
template <> Vc_INTRINSIC y_f64 Vc_VDECL convert_to<y_f64>(y_u64 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm256_cvtepu64_pd(v);
#else
    return y_f64{v.m(0), v.m(1), v.m(2), v.m(3)};
#endif
}

// from int{{{2
template <> Vc_INTRINSIC y_f64 Vc_VDECL convert_to<y_f64>(x_i32 v) { return _mm256_cvtepi32_pd(v); }

// from uint{{{2
template <> Vc_INTRINSIC y_f64 Vc_VDECL convert_to<y_f64>(x_u32 v)
{
#ifdef Vc_HAVE_AVX512VL
    return _mm256_cvtepu32_pd(v);
#elif defined Vc_HAVE_AVX512F
    return lo256(_mm512_cvtepu32_pd(intrin_cast<__m256i>(v)));
#else
    return _mm256_add_pd(_mm256_cvtepi32_pd(xor_(v, lowest16<int>())),
                         broadcast32(double(1u << 31)));
#endif
}

// from short{{{2
template <> Vc_INTRINSIC y_f64 Vc_VDECL convert_to<y_f64>(x_i16 v) { return convert_to<y_f64>(convert_to<x_i32>(v)); }

// from ushort{{{2
template <> Vc_INTRINSIC y_f64 Vc_VDECL convert_to<y_f64>(x_u16 v) { return convert_to<y_f64>(convert_to<x_i32>(v)); }

// from schar{{{2
template <> Vc_INTRINSIC y_f64 Vc_VDECL convert_to<y_f64>(x_i08 v) { return convert_to<y_f64>(convert_to<x_i32>(v)); }

// from uchar{{{2
template <> Vc_INTRINSIC y_f64 Vc_VDECL convert_to<y_f64>(x_u08 v) { return convert_to<y_f64>(convert_to<x_i32>(v)); }
#endif  // Vc_HAVE_AVX

// convert_to<z_f64> (double, 8){{{1
#ifdef Vc_HAVE_AVX512F
// from float{{{2
template <> Vc_INTRINSIC z_f64 Vc_VDECL convert_to<z_f64>(x_f32 v) { return zeroExtend(_mm256_cvtps_pd(v)); }

template <> Vc_INTRINSIC z_f64 Vc_VDECL convert_to<z_f64>(y_f32 v) { return _mm512_cvtps_pd(v); }

// from double{{{2
template <> Vc_INTRINSIC z_f64 Vc_VDECL convert_to<z_f64>(z_f64 v) { return v; }

// from llong{{{2
template <> Vc_INTRINSIC z_f64 Vc_VDECL convert_to<z_f64>(z_i64 v0) {
#if defined Vc_HAVE_AVX512DQ
    return _mm512_cvtepi64_pd(v0);
#else
    return _mm512_fmadd_pd(
        _mm512_cvtepi32_pd(_mm512_cvtepi64_epi32(_mm512_srai_epi64(v0, 32))),
        _mm512_set1_pd(0x100000000LL), _mm512_cvtepu32_pd(_mm512_cvtepi64_epi32(v0)));
#endif
}

// from ullong{{{2
template <> Vc_INTRINSIC z_f64 Vc_VDECL convert_to<z_f64>(z_u64 v0) {
#ifdef Vc_HAVE_AVX512DQ
    return _mm512_cvtepu64_pd(v0);
#else
    return _mm512_fmadd_pd(
        _mm512_cvtepu32_pd(_mm512_cvtepi64_epi32(_mm512_srli_epi64(v0, 32))),
        _mm512_set1_pd(0x100000000LL), _mm512_cvtepu32_pd(_mm512_cvtepi64_epi32(v0)));
#endif
}

// from int{{{2
template <> Vc_INTRINSIC z_f64 Vc_VDECL convert_to<z_f64>(x_i32 v)
{
    return zeroExtend(_mm256_cvtepi32_pd(v));
}

template <> Vc_INTRINSIC z_f64 Vc_VDECL convert_to<z_f64>(y_i32 v)
{
    return _mm512_cvtepi32_pd(v);
}

template <> Vc_INTRINSIC z_f64 Vc_VDECL convert_to<z_f64>(z_i32 v)
{
    return _mm512_cvtepi32_pd(lo256(v));
}

// from uint{{{2
template <> Vc_INTRINSIC z_f64 Vc_VDECL convert_to<z_f64>(x_u32 v)
{
#ifdef Vc_HAVE_AVX512VL
    return zeroExtend(_mm256_cvtepu32_pd(v));
#else
    return _mm512_cvtepu32_pd(zeroExtend(v));
#endif
}

template <> Vc_INTRINSIC z_f64 Vc_VDECL convert_to<z_f64>(y_u32 v)
{
    return _mm512_cvtepu32_pd(v);
}

template <> Vc_INTRINSIC z_f64 Vc_VDECL convert_to<z_f64>(z_u32 v)
{
    return _mm512_cvtepu32_pd(lo256(v));
}

// from short{{{2
template <> Vc_INTRINSIC z_f64 Vc_VDECL convert_to<z_f64>(x_i16 v)
{
    return convert_to<z_f64>(convert_to<y_i32>(v));
}

// from ushort{{{2
template <> Vc_INTRINSIC z_f64 Vc_VDECL convert_to<z_f64>(x_u16 v)
{
    return convert_to<z_f64>(convert_to<y_i32>(v));
}

// from schar{{{2
template <> Vc_INTRINSIC z_f64 Vc_VDECL convert_to<z_f64>(x_i08 v)
{
    return _mm512_cvtepi32_pd(convert_to<y_i32>(v));
}

// from uchar{{{2
template <> Vc_INTRINSIC z_f64 Vc_VDECL convert_to<z_f64>(x_u08 v)
{
    return convert_to<z_f64>(convert_to<y_i32>(v));
}

#endif  // Vc_HAVE_AVX512F

//--------------------float{{{1
//
// convert_to<x_f32> (float, 4){{{1
// from float{{{2
template <> Vc_INTRINSIC x_f32 Vc_VDECL convert_to<x_f32>(x_f32 v) { return v; }

// from double{{{2
template <> Vc_INTRINSIC x_f32 Vc_VDECL convert_to<x_f32>(x_f64 v) { return _mm_cvtpd_ps(v); }

template <> Vc_INTRINSIC x_f32 Vc_VDECL convert_to<x_f32>(x_f64 v0, x_f64 v1)
{
#ifdef Vc_HAVE_AVX
    return _mm256_cvtpd_ps(concat(v0, v1));
#else
    return _mm_movelh_ps(_mm_cvtpd_ps(v0), _mm_cvtpd_ps(v1));
#endif
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_f32 Vc_VDECL convert_to<x_f32>(y_f64 v0)
{
    return _mm256_cvtpd_ps(v0);
}
#endif  // Vc_HAVE_AVX

// from llong{{{2
template <> Vc_INTRINSIC x_f32 Vc_VDECL convert_to<x_f32>(x_i64 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm_cvtepi64_ps(v);
#else
    return {v.m(0), v.m(1), 0.f, 0.f};
#endif
}

template <> Vc_INTRINSIC x_f32 Vc_VDECL convert_to<x_f32>(x_i64 v0, x_i64 v1)
{
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm256_cvtepi64_ps(concat(v0, v1));
#else
    return {v0.m(0), v0.m(1), v1.m(0), v1.m(1)};
#endif
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_f32 Vc_VDECL convert_to<x_f32>(y_i64 v0)
{
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm256_cvtepi64_ps(v0);
#else
    return {v0.m(0), v0.m(1), v0.m(2), v0.m(3)};
#endif
}
#endif  // Vc_HAVE_AVX

// from ullong{{{2
template <> Vc_INTRINSIC x_f32 Vc_VDECL convert_to<x_f32>(x_u64 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm_cvtepu64_ps(v);
#else
    return {v.m(0), v.m(1), 0.f, 0.f};
#endif
}

template <> Vc_INTRINSIC x_f32 Vc_VDECL convert_to<x_f32>(x_u64 v0, x_u64 v1)
{
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm256_cvtepu64_ps(concat(v0, v1));
#else
    return {v0.m(0), v0.m(1), v1.m(0), v1.m(1)};
#endif
}

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC x_f32 Vc_VDECL convert_to<x_f32>(y_u64 v0)
{
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return _mm256_cvtepu64_ps(v0);
#else
    return {v0.m(0), v0.m(1), v0.m(2), v0.m(3)};
#endif
}
#endif  // Vc_HAVE_AVX

// from int{{{2
template <> Vc_INTRINSIC x_f32 Vc_VDECL convert_to<x_f32>(x_i32 v) { return _mm_cvtepi32_ps(v); }

// from uint{{{2
template <> Vc_INTRINSIC x_f32 Vc_VDECL convert_to<x_f32>(x_u32 v) {
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
template <> Vc_INTRINSIC x_f32 Vc_VDECL convert_to<x_f32>(x_i16 v) { return convert_to<x_f32>(convert_to<x_i32>(v)); }

// from ushort{{{2
template <> Vc_INTRINSIC x_f32 Vc_VDECL convert_to<x_f32>(x_u16 v) { return convert_to<x_f32>(convert_to<x_i32>(v)); }

// from schar{{{2
template <> Vc_INTRINSIC x_f32 Vc_VDECL convert_to<x_f32>(x_i08 v) { return convert_to<x_f32>(convert_to<x_i32>(v)); }

// from uchar{{{2
template <> Vc_INTRINSIC x_f32 Vc_VDECL convert_to<x_f32>(x_u08 v) { return convert_to<x_f32>(convert_to<x_i32>(v)); }

// convert_to<y_f32> (float, 8){{{1
#ifdef Vc_HAVE_AVX
// from float{{{2
template <> Vc_INTRINSIC y_f32 Vc_VDECL convert_to<y_f32>(y_f32 v) { return v; }

// from double{{{2
template <> Vc_INTRINSIC y_f32 Vc_VDECL convert_to<y_f32>(x_f64 v)
{
    return zeroExtend(_mm_cvtpd_ps(v));
}

template <> Vc_INTRINSIC y_f32 Vc_VDECL convert_to<y_f32>(y_f64 v0)
{
    return zeroExtend(_mm256_cvtpd_ps(v0));
}

template <> Vc_INTRINSIC y_f32 Vc_VDECL convert_to<y_f32>(y_f64 v0, y_f64 v1)
{
#if defined Vc_HAVE_AVX512F
    return _mm512_cvtpd_ps(concat(v0, v1));
#else
    return concat(_mm256_cvtpd_ps(v0), _mm256_cvtpd_ps(v1));
#endif
}

#ifdef Vc_HAVE_AVX512F
template <> Vc_INTRINSIC y_f32 Vc_VDECL convert_to<y_f32>(z_f64 v0)
{
    return _mm512_cvtpd_ps(v0);
}
#endif  // Vc_HAVE_AVX512F

//}}}2
// from llong{{{2
template <> Vc_INTRINSIC y_f32 Vc_VDECL convert_to<y_f32>(x_i64 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return zeroExtend(_mm_cvtepi64_ps(v));
#else
    return {v.m(0), v.m(1), 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#endif
}

template <> Vc_INTRINSIC y_f32 Vc_VDECL convert_to<y_f32>(y_i64 v0)
{
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return zeroExtend(_mm256_cvtepi64_ps(v0));
#elif defined Vc_HAVE_AVX512DQ
    return _mm512_cvtepi64_ps(zeroExtend(v0));
#else
    return {v0.m(0), v0.m(1), v0.m(2), v0.m(3), 0.f, 0.f, 0.f, 0.f};
#endif
}

#ifdef Vc_HAVE_AVX512F
template <> Vc_INTRINSIC y_f32 Vc_VDECL convert_to<y_f32>(z_i64 v0)
{
#ifdef Vc_HAVE_AVX512DQ
    return _mm512_cvtepi64_ps(v0);
#else
    return _mm512_cvtpd_ps(convert_to<z_f64>(v0));
    /* The above solution should be more efficient.
    y_f32 hi32 = _mm256_cvtepi32_ps(_mm512_cvtepi64_epi32(_mm512_srai_epi64(v0, 32)));
    y_u32 lo32 = _mm512_cvtepi64_epi32(v0);
    y_f32 hi16 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(lo32, 16)),
                               _mm256_set1_ps(0x10000));
    y_f32 lo16 = _mm256_cvtepi32_ps(and_(_mm256_set1_epi32(0xffff), lo32));
    return _mm256_add_ps(_mm256_fmadd_ps(hi32, _mm256_set1_ps(0x100000000LL), hi16),
                         lo16);
                         */
#endif
}
#endif  // Vc_HAVE_AVX512F

template <> Vc_INTRINSIC y_f32 Vc_VDECL convert_to<y_f32>(y_i64 v0, y_i64 v1)
{
#ifdef Vc_HAVE_AVX512F
    return convert_to<y_f32>(concat(v0, v1));
#elif defined Vc_HAVE_AVX2
    // v0 = aAbB cCdD
    // v1 = eEfF gGhH
    auto a = _mm256_unpacklo_epi32(v0, v1);                    // aeAE cgCG
    auto b = _mm256_unpackhi_epi32(v0, v1);                    // bfBF dhDH
    y_u32 lo32 = _mm256_unpacklo_epi32(a, b);  // abef cdgh
    y_f32 hi16 = _mm256_mul_ps(_mm256_set1_ps(0x10000),
                               _mm256_cvtepi32_ps(_mm256_srli_epi32(lo32, 16)));
    y_f32 lo16 = _mm256_cvtepi32_ps(and_(_mm256_set1_epi32(0x0000ffffu), lo32));
    y_f32 hi32 = _mm256_cvtepi32_ps(_mm256_unpackhi_epi32(a, b));  // ABEF CDGH
    const y_f32 scale = _mm256_set1_ps(0x100000000LL);
    y_f32 result = _mm256_add_ps(_mm256_fmadd_ps(hi32, scale, hi16), lo16);  // abef cdgh
    result = _mm256_castpd_ps(concat(
        _mm_unpacklo_pd(_mm_castps_pd(lo128(result)), _mm_castps_pd(hi128(result))),
        _mm_unpackhi_pd(_mm_castps_pd(lo128(result)), _mm_castps_pd(hi128(result)))));
    return result;
#else
    return {v0.m(0), v0.m(1), v0.m(2), v0.m(3), v1.m(0), v1.m(1), v1.m(2), v1.m(3)};
#endif
}

// from ullong{{{2
template <> Vc_INTRINSIC y_f32 Vc_VDECL convert_to<y_f32>(x_u64 v) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return zeroExtend(_mm_cvtepu64_ps(v));
#else
    return {v.m(0), v.m(1), 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#endif
}

template <> Vc_INTRINSIC y_f32 Vc_VDECL convert_to<y_f32>(y_u64 v0)
{
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512DQ
    return zeroExtend(_mm256_cvtepu64_ps(v0));
#elif defined Vc_HAVE_AVX512DQ
    return _mm512_cvtepu64_ps(zeroExtend(v0));
#else
    return {v0.m(0), v0.m(1), v0.m(2), v0.m(3), 0.f, 0.f, 0.f, 0.f};
#endif
}

#ifdef Vc_HAVE_AVX512F
template <> Vc_INTRINSIC y_f32 Vc_VDECL convert_to<y_f32>(z_u64 v0)
{
#ifdef Vc_HAVE_AVX512DQ
    return _mm512_cvtepu64_ps(v0);
#else
    return _mm256_fmadd_ps(
        lo256(_mm512_cvtepu32_ps(intrin_cast<__m512i>(_mm512_cvtepi64_epi32(_mm512_srai_epi64(v0, 32))))),
        _mm256_set1_ps(0x100000000LL),
        lo256(_mm512_cvtepu32_ps(intrin_cast<__m512i>(_mm512_cvtepi64_epi32(v0)))));
#endif
}
#endif  // Vc_HAVE_AVX512F

template <> Vc_INTRINSIC y_f32 Vc_VDECL convert_to<y_f32>(y_u64 v0, y_u64 v1)
{
#ifdef Vc_HAVE_AVX512F
    return convert_to<y_f32>(concat(v0, v1));
#else
    return {v0.m(0), v0.m(1), v0.m(2), v0.m(3), v1.m(0), v1.m(1), v1.m(2), v1.m(3)};
#endif
}

// from int{{{2
template <> Vc_INTRINSIC y_f32 Vc_VDECL convert_to<y_f32>(x_i32 v)
{
    return zeroExtend(_mm_cvtepi32_ps(v));
}

template <> Vc_INTRINSIC y_f32 Vc_VDECL convert_to<y_f32>(y_i32 v)
{
    return _mm256_cvtepi32_ps(v);
}

// from uint{{{2
template <> Vc_INTRINSIC y_f32 Vc_VDECL convert_to<y_f32>(x_u32 v)
{
#ifdef Vc_HAVE_AVX512VL
    return zeroExtend(_mm_cvtepu32_ps(v));
#else
    // see AVX::convert_to<uint, float> for an explanation of the math behind the
    // implementation
    return zeroExtend(blendv_ps(
        _mm_cvtepi32_ps(v),
        _mm_add_ps(
            _mm_cvtepi32_ps(_mm_and_si128(v, _mm_set1_epi32(0x7ffffe00))),
            _mm_add_ps(_mm_set1_ps(1u << 31),
                       _mm_cvtepi32_ps(_mm_and_si128(v, _mm_set1_epi32(0x000001ff))))),
        _mm_castsi128_ps(_mm_cmplt_epi32(v, _mm_setzero_si128()))));
#endif
}

template <> Vc_INTRINSIC y_f32 Vc_VDECL convert_to<y_f32>(y_u32 v)
{
#ifdef Vc_HAVE_AVX512VL
    return _mm256_cvtepu32_ps(v);
#elif defined Vc_HAVE_AVX512F
    return lo256(_mm512_cvtepu32_ps(intrin_cast<__m512i>(v)));
#else
    // this is complicated because cvtepi32_ps only supports signed input. Thus, all
    // input values with the MSB set would produce a negative result. We can reuse the
    // cvtepi32_ps instruction if we unset the MSB. But then the rounding results can be
    // different. Since float uses 24 bits for the mantissa (effectively), the 9-bit LSB
    // determines the rounding direction. (Consider the bits ...8'7654'3210. The bits [0:7]
    // need to be dropped and if > 0x80 round up, if < 0x80 round down. If [0:7] == 0x80
    // then the rounding direction is determined by bit [8] for round to even. That's why
    // the 9th bit is relevant for the rounding decision.)
    // If the MSB of the input is set to 0, the cvtepi32_ps instruction makes its rounding
    // decision on the lowest 8 bits instead. A second rounding decision is made when
    // float(0x8000'0000) is added. This will rarely fix the rounding issue.
    //
    // Here's what the standard rounding mode expects:
    // 0xc0000080 should cvt to 0xc0000000
    // 0xc0000081 should cvt to 0xc0000100
    //     --     should cvt to 0xc0000100
    // 0xc000017f should cvt to 0xc0000100
    // 0xc0000180 should cvt to 0xc0000200
    //
    // However: using float(input ^ 0x8000'0000) + float(0x8000'0000) we get:
    // 0xc0000081 would cvt to 0xc0000000
    // 0xc00000c0 would cvt to 0xc0000000
    // 0xc00000c1 would cvt to 0xc0000100
    // 0xc000013f would cvt to 0xc0000100
    // 0xc0000140 would cvt to 0xc0000200
    //
    // Solution: float(input & 0x7fff'fe00) + (float(0x8000'0000) + float(input & 0x1ff))
    // This ensures the rounding decision is made on the 9-bit LSB when 0x8000'0000 is
    // added to the float value of the low 8 bits of the input.
    return _mm256_blendv_ps(
        _mm256_cvtepi32_ps(v),
        _mm256_add_ps(
            _mm256_cvtepi32_ps(and_(v, broadcast32(0x7ffffe00))),
            _mm256_add_ps(avx_2_pow_31<float>(),
                          _mm256_cvtepi32_ps(and_(v, broadcast32(0x000001ff))))),
        _mm256_castsi256_ps(
#ifdef Vc_HAVE_AVX2
            _mm256_cmpgt_epi32(y_i32(), v)
#else   // Vc_HAVE_AVX2
            concat(_mm_cmpgt_epi32(x_i32(), lo128(v)), _mm_cmpgt_epi32(x_i32(), hi128(v)))
#endif  // Vc_HAVE_AVX2
                ));
#endif
}

// from short{{{2
template <> Vc_INTRINSIC y_f32 Vc_VDECL convert_to<y_f32>(x_i16 v)
{
    return convert_to<y_f32>(convert_to<y_i32>(v));
}

template <> Vc_INTRINSIC y_f32 Vc_VDECL convert_to<y_f32>(y_i16 v)
{
    return convert_to<y_f32>(convert_to<y_i32>(lo128(v)));
}

// from ushort{{{2
template <> Vc_INTRINSIC y_f32 Vc_VDECL convert_to<y_f32>(x_u16 v)
{
    return convert_to<y_f32>(convert_to<y_i32>(v));
}

template <> Vc_INTRINSIC y_f32 Vc_VDECL convert_to<y_f32>(y_u16 v)
{
    return convert_to<y_f32>(convert_to<y_i32>(lo128(v)));
}

// from schar{{{2
template <> Vc_INTRINSIC y_f32 Vc_VDECL convert_to<y_f32>(x_i08 v)
{
    return convert_to<y_f32>(convert_to<y_i32>(v));
}

template <> Vc_INTRINSIC y_f32 Vc_VDECL convert_to<y_f32>(y_i08 v)
{
    return convert_to<y_f32>(convert_to<y_i32>(lo128(v)));
}

// from uchar{{{2
template <> Vc_INTRINSIC y_f32 Vc_VDECL convert_to<y_f32>(x_u08 v)
{
    return convert_to<y_f32>(convert_to<y_i32>(v));
}

template <> Vc_INTRINSIC y_f32 Vc_VDECL convert_to<y_f32>(y_u08 v)
{
    return convert_to<y_f32>(convert_to<y_i32>(v));
}
//}}}2
#endif  // Vc_HAVE_AVX

//convert_to<z_f32> (float, 16){{{1
#ifdef Vc_HAVE_AVX512F
// from float{{{2
template <> Vc_INTRINSIC z_f32 Vc_VDECL convert_to<z_f32>(y_f32 v0, y_f32 v1) { return concat(v0, v1); }

template <> Vc_INTRINSIC z_f32 Vc_VDECL convert_to<z_f32>(z_f32 v) { return v; }

// from double{{{2
template <> Vc_INTRINSIC z_f32 Vc_VDECL convert_to<z_f32>(y_f64 v)
{
    return zeroExtend64(_mm256_cvtpd_ps(v));
}

template <> Vc_INTRINSIC z_f32 Vc_VDECL convert_to<z_f32>(y_f64 v0, y_f64 v1)
{
    return zeroExtend(_mm512_cvtpd_ps(concat(v0, v1)));
}

template <> Vc_INTRINSIC z_f32 Vc_VDECL convert_to<z_f32>(y_f64 v0, y_f64 v1, y_f64 v2, y_f64 v3)
{
    return concat(_mm512_cvtpd_ps(concat(v0, v1)), _mm512_cvtpd_ps(concat(v2, v3)));
}

template <> Vc_INTRINSIC z_f32 Vc_VDECL convert_to<z_f32>(z_f64 v0)
{
    return zeroExtend(_mm512_cvtpd_ps(v0));
}

template <> Vc_INTRINSIC z_f32 Vc_VDECL convert_to<z_f32>(z_f64 v0, z_f64 v1)
{
    return concat(_mm512_cvtpd_ps(v0), _mm512_cvtpd_ps(v1));
}

// from llong{{{2
template <> Vc_INTRINSIC z_f32 Vc_VDECL convert_to<z_f32>(z_i64 v0)
{
    return zeroExtend(convert_to<y_f32>(v0));
}

template <> Vc_INTRINSIC z_f32 Vc_VDECL convert_to<z_f32>(z_i64 v0, z_i64 v1)
{
#ifdef Vc_HAVE_AVX512DQ
    return concat(_mm512_cvtepi64_ps(v0), _mm512_cvtepi64_ps(v1));
#else
    z_f32 hi32 =
        _mm512_cvtepi32_ps(concat(_mm512_cvtepi64_epi32(_mm512_srai_epi64(v0, 32)),
                                  _mm512_cvtepi64_epi32(_mm512_srai_epi64(v1, 32))));
    const z_u32 lo32 = concat(_mm512_cvtepi64_epi32(v0), _mm512_cvtepi64_epi32(v1));
    // split low 32-bits, because if hi32 is a small negative number, the 24-bit mantissa may lose
    // important information if any of the high 8 bits of lo32 is set, leading to catastrophic
    // cancelation in the FMA
    z_f32 hi16 = _mm512_cvtepu32_ps(and_(_mm512_set1_epi32(0xffff0000u), lo32));
    z_f32 lo16 = _mm512_cvtepi32_ps(and_(_mm512_set1_epi32(0x0000ffffu), lo32));
    const z_f32 scale = _mm512_set1_ps(0x100000000LL);
    const z_f32 result = _mm512_add_ps(_mm512_fmadd_ps(hi32, scale, hi16), lo16);
    return result;
#endif
}

// from ullong{{{2
template <> Vc_INTRINSIC z_f32 Vc_VDECL convert_to<z_f32>(z_u64 v0)
{
    return zeroExtend(convert_to<y_f32>(v0));
}

template <> Vc_INTRINSIC z_f32 Vc_VDECL convert_to<z_f32>(z_u64 v0, z_u64 v1)
{
#ifdef Vc_HAVE_AVX512DQ
    return concat(_mm512_cvtepu64_ps(v0), _mm512_cvtepu64_ps(v1));
#else
    return _mm512_fmadd_ps(
        _mm512_cvtepu32_ps(concat(_mm512_cvtepi64_epi32(_mm512_srai_epi64(v0, 32)),
                                  _mm512_cvtepi64_epi32(_mm512_srai_epi64(v1, 32)))),
        _mm512_set1_ps(0x100000000LL),
        _mm512_cvtepu32_ps(concat(_mm512_cvtepi64_epi32(v0), _mm512_cvtepi64_epi32(v1))));
#endif  // Vc_HAVE_AVX512DQ
}

// from int{{{2
template <> Vc_INTRINSIC z_f32 Vc_VDECL convert_to<z_f32>(z_i32 v0)
{
    return _mm512_cvtepi32_ps(v0);
}

// from uint{{{2
template <> Vc_INTRINSIC z_f32 Vc_VDECL convert_to<z_f32>(z_u32 v0)
{
    return _mm512_cvtepu32_ps(v0);
}

// from short{{{2
template <> Vc_INTRINSIC z_f32 Vc_VDECL convert_to<z_f32>(y_i16 v0)
{
    return convert_to<z_f32>(convert_to<z_i32>(v0));
}

// from ushort{{{2
template <> Vc_INTRINSIC z_f32 Vc_VDECL convert_to<z_f32>(y_u16 v0)
{
    return convert_to<z_f32>(convert_to<z_i32>(v0));
}

// from schar{{{2
template <> Vc_INTRINSIC z_f32 Vc_VDECL convert_to<z_f32>(x_i08 v0)
{
    return convert_to<z_f32>(convert_to<z_i32>(v0));
}

// from uchar{{{2
template <> Vc_INTRINSIC z_f32 Vc_VDECL convert_to<z_f32>(x_u08 v0)
{
    return convert_to<z_f32>(convert_to<z_i32>(v0));
}

#endif  // Vc_HAVE_AVX512F

//}}}1
#endif  // Vc_HAVE_SSE2

// generic (u)long forwarding to (u)(llong|int){{{1

template <typename To, size_t N> Vc_INTRINSIC To Vc_VDECL convert_to(Storage<long, N> v)
{
    return convert_to<To>(Storage<equal_int_type_t<long>, N>(v));
}
template <typename To, size_t N>
Vc_INTRINSIC To Vc_VDECL convert_to(Storage<long, N> v0, Storage<long, N> v1)
{
    return convert_to<To>(Storage<equal_int_type_t<long>, N>(v0),
                          Storage<equal_int_type_t<long>, N>(v1));
}
template <typename To, size_t N>
Vc_INTRINSIC To Vc_VDECL convert_to(Storage<long, N> v0, Storage<long, N> v1, Storage<long, N> v2,
                           Storage<long, N> v3)
{
    return convert_to<To>(
        Storage<equal_int_type_t<long>, N>(v0), Storage<equal_int_type_t<long>, N>(v1),
        Storage<equal_int_type_t<long>, N>(v2), Storage<equal_int_type_t<long>, N>(v3));
}
template <typename To, size_t N>
Vc_INTRINSIC To Vc_VDECL convert_to(Storage<long, N> v0, Storage<long, N> v1, Storage<long, N> v2,
                           Storage<long, N> v3, Storage<long, N> v4, Storage<long, N> v5,
                           Storage<long, N> v6, Storage<long, N> v7)
{
    return convert_to<To>(
        Storage<equal_int_type_t<long>, N>(v0), Storage<equal_int_type_t<long>, N>(v1),
        Storage<equal_int_type_t<long>, N>(v2), Storage<equal_int_type_t<long>, N>(v3),
        Storage<equal_int_type_t<long>, N>(v4), Storage<equal_int_type_t<long>, N>(v5),
        Storage<equal_int_type_t<long>, N>(v6), Storage<equal_int_type_t<long>, N>(v7));
}

template <typename To, size_t N> Vc_INTRINSIC To Vc_VDECL convert_to(Storage<ulong, N> v)
{
    return convert_to<To>(Storage<equal_int_type_t<ulong>, N>(v));
}
template <typename To, size_t N>
Vc_INTRINSIC To Vc_VDECL convert_to(Storage<ulong, N> v0, Storage<ulong, N> v1)
{
    return convert_to<To>(Storage<equal_int_type_t<ulong>, N>(v0),
                          Storage<equal_int_type_t<ulong>, N>(v1));
}
template <typename To, size_t N>
Vc_INTRINSIC To Vc_VDECL convert_to(Storage<ulong, N> v0, Storage<ulong, N> v1, Storage<ulong, N> v2,
                           Storage<ulong, N> v3)
{
    return convert_to<To>(
        Storage<equal_int_type_t<ulong>, N>(v0), Storage<equal_int_type_t<ulong>, N>(v1),
        Storage<equal_int_type_t<ulong>, N>(v2), Storage<equal_int_type_t<ulong>, N>(v3));
}
template <typename To, size_t N>
Vc_INTRINSIC To Vc_VDECL convert_to(Storage<ulong, N> v0, Storage<ulong, N> v1, Storage<ulong, N> v2,
                           Storage<ulong, N> v3, Storage<ulong, N> v4, Storage<ulong, N> v5,
                           Storage<ulong, N> v6, Storage<ulong, N> v7)
{
    return convert_to<To>(
        Storage<equal_int_type_t<ulong>, N>(v0), Storage<equal_int_type_t<ulong>, N>(v1),
        Storage<equal_int_type_t<ulong>, N>(v2), Storage<equal_int_type_t<ulong>, N>(v3),
        Storage<equal_int_type_t<ulong>, N>(v4), Storage<equal_int_type_t<ulong>, N>(v5),
        Storage<equal_int_type_t<ulong>, N>(v6), Storage<equal_int_type_t<ulong>, N>(v7));
}

// generic forwarding for down-conversions to unsigned int{{{1
struct scalar_conversion_fallback_tag {};
template <typename T> struct fallback_int_type { using type = scalar_conversion_fallback_tag; };
template <> struct fallback_int_type< uchar> { using type = schar; };
template <> struct fallback_int_type<ushort> { using type = short; };
template <> struct fallback_int_type<  uint> { using type = int; };
template <> struct fallback_int_type<ullong> { using type = llong; };
template <> struct fallback_int_type<  long> { using type = equal_int_type_t< long>; };
template <> struct fallback_int_type< ulong> { using type = equal_int_type_t<ulong>; };

template <typename T>
using equivalent_storage_t =
    Storage<typename fallback_int_type<typename T::EntryType>::type, T::size()>;

template <typename To, typename From>
Vc_INTRINSIC std::conditional_t<
    (std::is_integral<typename To::EntryType>::value &&
     sizeof(typename To::EntryType) <= sizeof(typename From::EntryType)),
    Storage<std::make_signed_t<typename From::EntryType>, From::size()>, From>
    Vc_VDECL maybe_make_signed(From v)
{
    static_assert(
        std::is_unsigned<typename From::EntryType>::value,
        "maybe_make_signed must only be used with unsigned integral Storage types");
    return std::conditional_t<
        (std::is_integral<typename To::EntryType>::value &&
         sizeof(typename To::EntryType) <= sizeof(typename From::EntryType)),
        Storage<std::make_signed_t<typename From::EntryType>, From::size()>, From>{v};
}

template <typename To,
          typename Fallback = typename fallback_int_type<typename To::EntryType>::type>
struct equivalent_conversion {
    template <size_t N, typename... From>
    static Vc_INTRINSIC Vc_CONST To Vc_VDECL convert(Storage<uchar, N> v0, From... vs)
    {
        using S = Storage<Fallback, To::size()>;
        return convert_to<S>(maybe_make_signed<To>(v0), maybe_make_signed<To>(vs)...).v();
    }

    template <size_t N, typename... From>
    static Vc_INTRINSIC Vc_CONST To Vc_VDECL convert(Storage<ushort, N> v0, From... vs)
    {
        using S = Storage<Fallback, To::size()>;
        return convert_to<S>(maybe_make_signed<To>(v0), maybe_make_signed<To>(vs)...).v();
    }

    template <size_t N, typename... From>
    static Vc_INTRINSIC Vc_CONST To Vc_VDECL convert(Storage<uint, N> v0, From... vs)
    {
        using S = Storage<Fallback, To::size()>;
        return convert_to<S>(maybe_make_signed<To>(v0), maybe_make_signed<To>(vs)...).v();
    }

    template <size_t N, typename... From>
    static Vc_INTRINSIC Vc_CONST To Vc_VDECL convert(Storage<ulong, N> v0, From... vs)
    {
        using S = Storage<Fallback, To::size()>;
        return convert_to<S>(maybe_make_signed<To>(v0), maybe_make_signed<To>(vs)...).v();
    }

    template <size_t N, typename... From>
    static Vc_INTRINSIC Vc_CONST To Vc_VDECL convert(Storage<ullong, N> v0, From... vs)
    {
        using S = Storage<Fallback, To::size()>;
        return convert_to<S>(maybe_make_signed<To>(v0), maybe_make_signed<To>(vs)...).v();
    }

    template <typename F0, typename... From>
    static Vc_INTRINSIC Vc_CONST To Vc_VDECL convert(F0 v0, From... vs)
    {
        static_assert(!std::is_unsigned<typename F0::EntryType>::value, "overload error");
        using S = Storage<Fallback, To::size()>;
        return convert_to<S>(v0, vs...).v();
    }
};

// fallback: scalar aggregate conversion{{{1
template <typename To> struct equivalent_conversion<To, scalar_conversion_fallback_tag> {
    template <typename From, typename... Fs>
    static Vc_INTRINSIC Vc_CONST To Vc_VDECL convert(From v0, Fs... vs)
    {
        using F = typename From::value_type;
        using T = typename To::value_type;
        static_assert(sizeof(F) >= sizeof(T) && std::is_integral<T>::value &&
                          std::is_unsigned<F>::value,
                      "missing an implementation for convert<To>(From, Fs...)");
        using S = Storage<typename fallback_int_type<F>::type, From::size()>;
        return convert_to<To>(S(v0), S(vs)...);
    }
};

// convert_to implementations invoking the fallbacks{{{1
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_f32 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_f32 v0, x_f32 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_f32 v0, x_f32 v1, x_f32 v2, x_f32 v3) { return equivalent_conversion<To>::convert(v0, v1, v2, v3); }
#ifdef Vc_HAVE_SSE2
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_f64 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_f64 v0, x_f64 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_f64 v0, x_f64 v1, x_f64 v2, x_f64 v3) { return equivalent_conversion<To>::convert(v0, v1, v2, v3); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_f64 v0, x_f64 v1, x_f64 v2, x_f64 v3, x_f64 v4, x_f64 v5, x_f64 v6, x_f64 v7) { return equivalent_conversion<To>::convert(v0, v1, v2, v3, v4, v5, v6, v7); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_i08 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u08 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_i16 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_i16 v0, x_i16 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u16 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u16 v0, x_u16 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_i32 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_i32 v0, x_i32 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_i32 v0, x_i32 v1, x_i32 v2, x_i32 v3) { return equivalent_conversion<To>::convert(v0, v1, v2, v3); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u32 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u32 v0, x_u32 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u32 v0, x_u32 v1, x_u32 v2, x_u32 v3) { return equivalent_conversion<To>::convert(v0, v1, v2, v3); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_i64 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_i64 v0, x_i64 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_i64 v0, x_i64 v1, x_i64 v2, x_i64 v3) { return equivalent_conversion<To>::convert(v0, v1, v2, v3); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_i64 v0, x_i64 v1, x_i64 v2, x_i64 v3, x_i64 v4, x_i64 v5, x_i64 v6, x_i64 v7) { return equivalent_conversion<To>::convert(v0, v1, v2, v3, v4, v5, v6, v7); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u64 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u64 v0, x_u64 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u64 v0, x_u64 v1, x_u64 v2, x_u64 v3) { return equivalent_conversion<To>::convert(v0, v1, v2, v3); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(x_u64 v0, x_u64 v1, x_u64 v2, x_u64 v3, x_u64 v4, x_u64 v5, x_u64 v6, x_u64 v7) { return equivalent_conversion<To>::convert(v0, v1, v2, v3, v4, v5, v6, v7); }
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_f32 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_f32 v0, y_f32 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_f32 v0, y_f32 v1, y_f32 v2, y_f32 v3) { return equivalent_conversion<To>::convert(v0, v1, v2, v3); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_f64 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_f64 v0, y_f64 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_f64 v0, y_f64 v1, y_f64 v2, y_f64 v3) { return equivalent_conversion<To>::convert(v0, v1, v2, v3); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_f64 v0, y_f64 v1, y_f64 v2, y_f64 v3, y_f64 v4, y_f64 v5, y_f64 v6, y_f64 v7) { return equivalent_conversion<To>::convert(v0, v1, v2, v3, v4, v5, v6, v7); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_i08 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_u08 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_i16 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_i16 v0, y_i16 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_u16 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_u16 v0, y_u16 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_i32 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_i32 v0, y_i32 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_i32 v0, y_i32 v1, y_i32 v2, y_i32 v3) { return equivalent_conversion<To>::convert(v0, v1, v2, v3); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_u32 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_u32 v0, y_u32 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_u32 v0, y_u32 v1, y_u32 v2, y_u32 v3) { return equivalent_conversion<To>::convert(v0, v1, v2, v3); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_i64 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_i64 v0, y_i64 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_i64 v0, y_i64 v1, y_i64 v2, y_i64 v3) { return equivalent_conversion<To>::convert(v0, v1, v2, v3); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_i64 v0, y_i64 v1, y_i64 v2, y_i64 v3, y_i64 v4, y_i64 v5, y_i64 v6, y_i64 v7) { return equivalent_conversion<To>::convert(v0, v1, v2, v3, v4, v5, v6, v7); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_u64 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_u64 v0, y_u64 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_u64 v0, y_u64 v1, y_u64 v2, y_u64 v3) { return equivalent_conversion<To>::convert(v0, v1, v2, v3); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(y_u64 v0, y_u64 v1, y_u64 v2, y_u64 v3, y_u64 v4, y_u64 v5, y_u64 v6, y_u64 v7) { return equivalent_conversion<To>::convert(v0, v1, v2, v3, v4, v5, v6, v7); }
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_f32 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_f32 v0, z_f32 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_f32 v0, z_f32 v1, z_f32 v2, z_f32 v3) { return equivalent_conversion<To>::convert(v0, v1, v2, v3); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_f64 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_f64 v0, z_f64 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_f64 v0, z_f64 v1, z_f64 v2, z_f64 v3) { return equivalent_conversion<To>::convert(v0, v1, v2, v3); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_f64 v0, z_f64 v1, z_f64 v2, z_f64 v3, z_f64 v4, z_f64 v5, z_f64 v6, z_f64 v7) { return equivalent_conversion<To>::convert(v0, v1, v2, v3, v4, v5, v6, v7); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_i08 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_u08 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_i16 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_i16 v0, z_i16 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_u16 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_u16 v0, z_u16 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_i32 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_i32 v0, z_i32 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_i32 v0, z_i32 v1, z_i32 v2, z_i32 v3) { return equivalent_conversion<To>::convert(v0, v1, v2, v3); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_u32 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_u32 v0, z_u32 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_u32 v0, z_u32 v1, z_u32 v2, z_u32 v3) { return equivalent_conversion<To>::convert(v0, v1, v2, v3); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_i64 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_i64 v0, z_i64 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_i64 v0, z_i64 v1, z_i64 v2, z_i64 v3) { return equivalent_conversion<To>::convert(v0, v1, v2, v3); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_i64 v0, z_i64 v1, z_i64 v2, z_i64 v3, z_i64 v4, z_i64 v5, z_i64 v6, z_i64 v7) { return equivalent_conversion<To>::convert(v0, v1, v2, v3, v4, v5, v6, v7); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_u64 v0) { return equivalent_conversion<To>::convert(v0); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_u64 v0, z_u64 v1) { return equivalent_conversion<To>::convert(v0, v1); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_u64 v0, z_u64 v1, z_u64 v2, z_u64 v3) { return equivalent_conversion<To>::convert(v0, v1, v2, v3); }
template <typename To> Vc_INTRINSIC To Vc_VDECL convert_to(z_u64 v0, z_u64 v1, z_u64 v2, z_u64 v3, z_u64 v4, z_u64 v5, z_u64 v6, z_u64 v7) { return equivalent_conversion<To>::convert(v0, v1, v2, v3, v4, v5, v6, v7); }
#endif  // Vc_HAVE_AVX512F

// convert function{{{1
template <typename From, typename To> Vc_INTRINSIC To Vc_VDECL convert(From v)
{
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
    constexpr auto N = From::size() < To::size() ? From::size() : To::size();
    return convert_builtin<To>(v.builtin(), std::make_index_sequence<N>());
#else
    return convert_to<To>(v);
#endif
}

template <typename From, typename To> Vc_INTRINSIC To Vc_VDECL convert(From v0, From v1)
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
Vc_INTRINSIC To Vc_VDECL convert(From v0, From v1, From v2, From v3)
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
Vc_INTRINSIC To Vc_VDECL convert(From v0, From v1, From v2, From v3, From v4, From v5, From v6,
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

// convert_all function{{{1
template <typename To, typename From>
Vc_INTRINSIC auto Vc_VDECL convert_all_impl(From v, std::true_type)
{
    constexpr size_t N = From::size() / To::size();
    return generate_from_n_evaluations<N, std::array<To, N>>([&](auto i) {
        using namespace Vc::detail::x86;  // ICC needs this to find convert and
                                          // shift_right below.
        constexpr int shift = decltype(i)::value  // MSVC needs this instead of a simple
                                                  // `i`, apparently their conversion
                                                  // operator is not (considered)
                                                  // constexpr.
                              * To::size() * sizeof(From) / From::size();
        return convert<From, To>(shift_right<shift>(v));
    });
}

template <typename To, typename From>
Vc_INTRINSIC To Vc_VDECL convert_all_impl(From v, std::false_type)
{
    return convert<From, To>(v);
}

template <typename To, typename From> Vc_INTRINSIC auto Vc_VDECL convert_all(From v)
{
    return convert_all_impl<To, From>(
        v, std::integral_constant<bool, (From::size() > To::size())>());
}

// }}}1
}}  // namespace detail::x86
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DATAPAR_X86_CONVERT_H_

// vim: foldmethod=marker
