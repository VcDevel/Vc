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

#ifndef VC_DATAPAR_X86_H_
#define VC_DATAPAR_X86_H_

#include <limits>
#include <climits>
#include <cstring>

#include "../macros.h"
#include "../detail.h"
#include "../const.h"

#ifdef Vc_HAVE_SSE

#ifdef Vc_MSVC
#include <intrin.h>
#else   // Vc_MSVC
#include <x86intrin.h>
#endif  // Vc_MSVC

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
namespace x86
{

// builtin_type{{{1
template <typename ValueType, size_t Bytes> struct builtin_type_impl {};

#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <> struct builtin_type_impl<         double   , 16> { typedef          double    type [[gnu::vector_size(16)]]; };
template <> struct builtin_type_impl<         float    , 16> { typedef          float     type [[gnu::vector_size(16)]]; };
template <> struct builtin_type_impl<         long long, 16> { typedef          long long type [[gnu::vector_size(16)]]; };
template <> struct builtin_type_impl<unsigned long long, 16> { typedef unsigned long long type [[gnu::vector_size(16)]]; };
template <> struct builtin_type_impl<         long     , 16> { typedef          long      type [[gnu::vector_size(16)]]; };
template <> struct builtin_type_impl<unsigned long     , 16> { typedef unsigned long      type [[gnu::vector_size(16)]]; };
template <> struct builtin_type_impl<         int      , 16> { typedef          int       type [[gnu::vector_size(16)]]; };
template <> struct builtin_type_impl<unsigned int      , 16> { typedef unsigned int       type [[gnu::vector_size(16)]]; };
template <> struct builtin_type_impl<         short    , 16> { typedef          short     type [[gnu::vector_size(16)]]; };
template <> struct builtin_type_impl<unsigned short    , 16> { typedef unsigned short     type [[gnu::vector_size(16)]]; };
template <> struct builtin_type_impl<         char     , 16> { typedef          char      type [[gnu::vector_size(16)]]; };
template <> struct builtin_type_impl<unsigned char     , 16> { typedef unsigned char      type [[gnu::vector_size(16)]]; };
template <> struct builtin_type_impl<  signed char     , 16> { typedef   signed char      type [[gnu::vector_size(16)]]; };
template <> struct builtin_type_impl<         bool     , 16> { typedef unsigned char      type [[gnu::vector_size(16)]]; };

template <> struct builtin_type_impl<         double   , 32> { typedef          double    type [[gnu::vector_size(32)]]; };
template <> struct builtin_type_impl<         float    , 32> { typedef          float     type [[gnu::vector_size(32)]]; };
template <> struct builtin_type_impl<         long long, 32> { typedef          long long type [[gnu::vector_size(32)]]; };
template <> struct builtin_type_impl<unsigned long long, 32> { typedef unsigned long long type [[gnu::vector_size(32)]]; };
template <> struct builtin_type_impl<         long     , 32> { typedef          long      type [[gnu::vector_size(32)]]; };
template <> struct builtin_type_impl<unsigned long     , 32> { typedef unsigned long      type [[gnu::vector_size(32)]]; };
template <> struct builtin_type_impl<         int      , 32> { typedef          int       type [[gnu::vector_size(32)]]; };
template <> struct builtin_type_impl<unsigned int      , 32> { typedef unsigned int       type [[gnu::vector_size(32)]]; };
template <> struct builtin_type_impl<         short    , 32> { typedef          short     type [[gnu::vector_size(32)]]; };
template <> struct builtin_type_impl<unsigned short    , 32> { typedef unsigned short     type [[gnu::vector_size(32)]]; };
template <> struct builtin_type_impl<         char     , 32> { typedef          char      type [[gnu::vector_size(32)]]; };
template <> struct builtin_type_impl<unsigned char     , 32> { typedef unsigned char      type [[gnu::vector_size(32)]]; };
template <> struct builtin_type_impl<  signed char     , 32> { typedef   signed char      type [[gnu::vector_size(32)]]; };
template <> struct builtin_type_impl<         bool     , 32> { typedef unsigned char      type [[gnu::vector_size(32)]]; };

template <> struct builtin_type_impl<         double   , 64> { typedef          double    type [[gnu::vector_size(64)]]; };
template <> struct builtin_type_impl<         float    , 64> { typedef          float     type [[gnu::vector_size(64)]]; };
template <> struct builtin_type_impl<         long long, 64> { typedef          long long type [[gnu::vector_size(64)]]; };
template <> struct builtin_type_impl<unsigned long long, 64> { typedef unsigned long long type [[gnu::vector_size(64)]]; };
template <> struct builtin_type_impl<         long     , 64> { typedef          long      type [[gnu::vector_size(64)]]; };
template <> struct builtin_type_impl<unsigned long     , 64> { typedef unsigned long      type [[gnu::vector_size(64)]]; };
template <> struct builtin_type_impl<         int      , 64> { typedef          int       type [[gnu::vector_size(64)]]; };
template <> struct builtin_type_impl<unsigned int      , 64> { typedef unsigned int       type [[gnu::vector_size(64)]]; };
template <> struct builtin_type_impl<         short    , 64> { typedef          short     type [[gnu::vector_size(64)]]; };
template <> struct builtin_type_impl<unsigned short    , 64> { typedef unsigned short     type [[gnu::vector_size(64)]]; };
template <> struct builtin_type_impl<         char     , 64> { typedef          char      type [[gnu::vector_size(64)]]; };
template <> struct builtin_type_impl<unsigned char     , 64> { typedef unsigned char      type [[gnu::vector_size(64)]]; };
template <> struct builtin_type_impl<  signed char     , 64> { typedef   signed char      type [[gnu::vector_size(64)]]; };
template <> struct builtin_type_impl<         bool     , 64> { typedef unsigned char      type [[gnu::vector_size(64)]]; };
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES

template <typename T, size_t Size>
using builtin_type = typename builtin_type_impl<T, Size * sizeof(T)>::type;

// intrinsic_type{{{1
template <typename T, size_t Bytes> struct intrinsic_type_impl {
    static_assert(sizeof(T) == Bytes,
                  "intrinsic_type without SIMD target support may only have Size = 1");
    using type = T;
};

#if defined Vc_HAVE_AVX512F
template <> struct intrinsic_type_impl<double, 64> { using type = __m512d; };
template <> struct intrinsic_type_impl< float, 64> { using type = __m512; };
template <typename T> struct intrinsic_type_impl<T, 64> { using type = __m512i; };
#endif  // Vc_HAVE_AVX512F

#if defined Vc_HAVE_AVX
template <> struct intrinsic_type_impl<double, 32> { using type = __m256d; };
template <> struct intrinsic_type_impl< float, 32> { using type = __m256; };
template <typename T> struct intrinsic_type_impl<T, 32> { using type = __m256i; };
#endif  // Vc_HAVE_AVX

#if defined Vc_HAVE_SSE
template <> struct intrinsic_type_impl< float, 16> { using type = __m128; };
#endif  // Vc_HAVE_SSE
#if defined Vc_HAVE_SSE2
template <> struct intrinsic_type_impl<double, 16> { using type = __m128d; };
template <typename T> struct intrinsic_type_impl<T, 16> { using type = __m128i; };
#endif  // Vc_HAVE_SSE2

template <typename T, size_t Size>
using intrinsic_type = typename intrinsic_type_impl<T, Size * sizeof(T)>::type;

// is_intrinsic{{{1
template <class T> struct is_intrinsic : public std::false_type {};
template <> struct is_intrinsic<__m128> : public std::true_type {};
#ifdef Vc_HAVE_SSE2
template <> struct is_intrinsic<__m128d> : public std::true_type {};
template <> struct is_intrinsic<__m128i> : public std::true_type {};
#endif  // Vc_HAVE_SSE2
#ifdef Vc_HAVE_AVX
template <> struct is_intrinsic<__m256 > : public std::true_type {};
template <> struct is_intrinsic<__m256d> : public std::true_type {};
template <> struct is_intrinsic<__m256i> : public std::true_type {};
#endif  // Vc_HAVE_AVX
#ifdef Vc_HAVE_AVX512F
template <> struct is_intrinsic<__m512 > : public std::true_type {};
template <> struct is_intrinsic<__m512d> : public std::true_type {};
template <> struct is_intrinsic<__m512i> : public std::true_type {};
#endif  // Vc_HAVE_AVX512F
template <class T> constexpr bool is_intrinsic_v = is_intrinsic<T>::value;

// is_builtin_vector{{{1
template <class T> struct is_builtin_vector : public std::false_type {};
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
template <> struct is_builtin_vector<builtin_type<float, 4>> : public std::true_type {};
#ifdef Vc_HAVE_SSE2
template <> struct is_builtin_vector<builtin_type<double, 2>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type< llong, 2>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type<ullong, 2>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type<  long, 16 / sizeof( long)>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type< ulong, 16 / sizeof(ulong)>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type<   int, 4>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type<  uint, 4>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type< short, 8>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type<ushort, 8>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type< schar,16>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type< uchar,16>> : public std::true_type {};
#endif  // Vc_HAVE_SSE2
#ifdef Vc_HAVE_AVX
template <> struct is_builtin_vector<builtin_type< float, 4 * 2>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type<double, 2 * 2>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type< llong, 2 * 2>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type<ullong, 2 * 2>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type<  long, 16 / sizeof( long) * 2>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type< ulong, 16 / sizeof(ulong) * 2>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type<   int, 4 * 2>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type<  uint, 4 * 2>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type< short, 8 * 2>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type<ushort, 8 * 2>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type< schar,16 * 2>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type< uchar,16 * 2>> : public std::true_type {};
#endif  // Vc_HAVE_AVX
#ifdef Vc_HAVE_AVX512F
template <> struct is_builtin_vector<builtin_type< float, 4 * 4>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type<double, 2 * 4>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type< llong, 2 * 4>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type<ullong, 2 * 4>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type<  long, 16 / sizeof( long) * 4>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type< ulong, 16 / sizeof(ulong) * 4>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type<   int, 4 * 4>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type<  uint, 4 * 4>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type< short, 8 * 4>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type<ushort, 8 * 4>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type< schar,16 * 4>> : public std::true_type {};
template <> struct is_builtin_vector<builtin_type< uchar,16 * 4>> : public std::true_type {};
#endif  // Vc_HAVE_AVX512F
#endif  // Vc_USE_BUILTIN_VECTOR_TYPES
template <class T> constexpr bool is_builtin_vector_v = is_builtin_vector<T>::value;

// zeroExtend{{{1
#ifdef Vc_HAVE_AVX
#if defined Vc_MSVC || defined Vc_CLANG || defined Vc_APPLECLANG
Vc_INTRINSIC Vc_CONST __m256  zeroExtend(__m128  v) { return _mm256_permute2f128_ps   (_mm256_castps128_ps256(v), _mm256_castps128_ps256(v), 0x80); }
Vc_INTRINSIC Vc_CONST __m256i zeroExtend(__m128i v) { return _mm256_permute2f128_si256(_mm256_castsi128_si256(v), _mm256_castsi128_si256(v), 0x80); }
Vc_INTRINSIC Vc_CONST __m256d zeroExtend(__m128d v) { return _mm256_permute2f128_pd   (_mm256_castpd128_pd256(v), _mm256_castpd128_pd256(v), 0x80); }

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC Vc_CONST __m512  zeroExtend(__m256  v) { return _mm512_castpd_ps(_mm512_insertf64x4(_mm512_setzero_pd(), _mm256_castps_pd(v), 0x0)); }
Vc_INTRINSIC Vc_CONST __m512d zeroExtend(__m256d v) { return _mm512_insertf64x4(_mm512_setzero_pd(), v, 0x0); }
Vc_INTRINSIC Vc_CONST __m512i zeroExtend(__m256i v) { return _mm512_inserti64x4(_mm512_setzero_si512(), v, 0x0); }

Vc_INTRINSIC Vc_CONST __m512  zeroExtend64(__m128  v) { return _mm512_insertf32x4(_mm512_setzero_ps(), v, 0x0); }
Vc_INTRINSIC Vc_CONST __m512d zeroExtend64(__m128d v) { return _mm512_castps_pd(_mm512_insertf32x4(_mm512_setzero_ps(), _mm_castpd_ps(v), 0x0)); }
Vc_INTRINSIC Vc_CONST __m512i zeroExtend64(__m128i v) { return _mm512_inserti32x4(_mm512_setzero_si512(), v, 0x0); }
#endif  // Vc_HAVE_AVX512F
#else   // defined Vc_MSVC || defined Vc_CLANG || defined Vc_APPLECLANG
Vc_INTRINSIC Vc_CONST __m256  zeroExtend(__m128  v) { return _mm256_castps128_ps256(v); }
Vc_INTRINSIC Vc_CONST __m256i zeroExtend(__m128i v) { return _mm256_castsi128_si256(v); }
Vc_INTRINSIC Vc_CONST __m256d zeroExtend(__m128d v) { return _mm256_castpd128_pd256(v); }

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC Vc_CONST __m512  zeroExtend(__m256  v) { return _mm512_castps256_ps512(v); }
Vc_INTRINSIC Vc_CONST __m512d zeroExtend(__m256d v) { return _mm512_castpd256_pd512(v); }
Vc_INTRINSIC Vc_CONST __m512i zeroExtend(__m256i v) { return _mm512_castsi256_si512(v); }

Vc_INTRINSIC Vc_CONST __m512  zeroExtend64(__m128  v) { return _mm512_castps128_ps512(v); }
Vc_INTRINSIC Vc_CONST __m512d zeroExtend64(__m128d v) { return _mm512_castpd128_pd512(v); }
Vc_INTRINSIC Vc_CONST __m512i zeroExtend64(__m128i v) { return _mm512_castsi128_si512(v); }
#endif  // Vc_HAVE_AVX512F
#endif  // defined Vc_MSVC || defined Vc_CLANG || defined Vc_APPLECLANG
#endif  // Vc_HAVE_AVX

// intrin_cast{{{1
template<typename T> Vc_INTRINSIC_L T intrin_cast(__m128  v) Vc_INTRINSIC_R;
#ifdef Vc_HAVE_SSE2
template<typename T> Vc_INTRINSIC_L T intrin_cast(__m128i v) Vc_INTRINSIC_R;
template<typename T> Vc_INTRINSIC_L T intrin_cast(__m128d v) Vc_INTRINSIC_R;
#endif  // Vc_HAVE_SSE2
#ifdef Vc_HAVE_AVX
template<typename T> Vc_INTRINSIC_L T intrin_cast(__m256  v) Vc_INTRINSIC_R;
template<typename T> Vc_INTRINSIC_L T intrin_cast(__m256i v) Vc_INTRINSIC_R;
template<typename T> Vc_INTRINSIC_L T intrin_cast(__m256d v) Vc_INTRINSIC_R;
#endif  // Vc_HAVE_AVX
#ifdef Vc_HAVE_AVX512F
template<typename T> Vc_INTRINSIC_L T intrin_cast(__m512  v) Vc_INTRINSIC_R;
template<typename T> Vc_INTRINSIC_L T intrin_cast(__m512i v) Vc_INTRINSIC_R;
template<typename T> Vc_INTRINSIC_L T intrin_cast(__m512d v) Vc_INTRINSIC_R;
#endif  // Vc_HAVE_AVX512F

template<> Vc_INTRINSIC __m128  intrin_cast(__m128  v) { return v; }
#ifdef Vc_HAVE_SSE2
// 128 -> 128
template<> Vc_INTRINSIC __m128  intrin_cast(__m128i v) { return _mm_castsi128_ps(v); }
template<> Vc_INTRINSIC __m128  intrin_cast(__m128d v) { return _mm_castpd_ps(v); }
template<> Vc_INTRINSIC __m128i intrin_cast(__m128  v) { return _mm_castps_si128(v); }
template<> Vc_INTRINSIC __m128i intrin_cast(__m128i v) { return v; }
template<> Vc_INTRINSIC __m128i intrin_cast(__m128d v) { return _mm_castpd_si128(v); }
template<> Vc_INTRINSIC __m128d intrin_cast(__m128  v) { return _mm_castps_pd(v); }
template<> Vc_INTRINSIC __m128d intrin_cast(__m128i v) { return _mm_castsi128_pd(v); }
template<> Vc_INTRINSIC __m128d intrin_cast(__m128d v) { return v; }

#ifdef Vc_HAVE_AVX
// 128 -> 256
// FIXME: the following casts leave the upper 128bits undefined. With GCC and ICC I've never
// seen the cast not do what I want though: after a VEX-coded SSE instruction the register's
// upper 128bits are zero. Thus using the same register as AVX register will have the upper
// 128bits zeroed. MSVC, though, implements _mm256_castxx128_xx256 with a 128bit move to memory
// + 256bit load. Thus the upper 128bits are really undefined. But there is no intrinsic to do
// what I want (i.e. alias the register, disallowing the move to memory in-between). I'm stuck,
// do we really want to rely on specific compiler behavior here?
template<> Vc_INTRINSIC __m256  intrin_cast(__m128  v) { return _mm256_castps128_ps256(v); }
template<> Vc_INTRINSIC __m256  intrin_cast(__m128i v) { return _mm256_castps128_ps256(_mm_castsi128_ps(v)); }
template<> Vc_INTRINSIC __m256  intrin_cast(__m128d v) { return _mm256_castps128_ps256(_mm_castpd_ps(v)); }
template<> Vc_INTRINSIC __m256i intrin_cast(__m128  v) { return _mm256_castsi128_si256(_mm_castps_si128(v)); }
template<> Vc_INTRINSIC __m256i intrin_cast(__m128i v) { return _mm256_castsi128_si256(v); }
template<> Vc_INTRINSIC __m256i intrin_cast(__m128d v) { return _mm256_castsi128_si256(_mm_castpd_si128(v)); }
template<> Vc_INTRINSIC __m256d intrin_cast(__m128  v) { return _mm256_castpd128_pd256(_mm_castps_pd(v)); }
template<> Vc_INTRINSIC __m256d intrin_cast(__m128i v) { return _mm256_castpd128_pd256(_mm_castsi128_pd(v)); }
template<> Vc_INTRINSIC __m256d intrin_cast(__m128d v) { return _mm256_castpd128_pd256(v); }

// 256 -> 128
template<> Vc_INTRINSIC __m128  intrin_cast(__m256  v) { return _mm256_castps256_ps128(v); }
template<> Vc_INTRINSIC __m128  intrin_cast(__m256i v) { return _mm256_castps256_ps128(_mm256_castsi256_ps(v)); }
template<> Vc_INTRINSIC __m128  intrin_cast(__m256d v) { return _mm256_castps256_ps128(_mm256_castpd_ps(v)); }
template<> Vc_INTRINSIC __m128i intrin_cast(__m256  v) { return _mm256_castsi256_si128(_mm256_castps_si256(v)); }
template<> Vc_INTRINSIC __m128i intrin_cast(__m256i v) { return _mm256_castsi256_si128(v); }
template<> Vc_INTRINSIC __m128i intrin_cast(__m256d v) { return _mm256_castsi256_si128(_mm256_castpd_si256(v)); }
template<> Vc_INTRINSIC __m128d intrin_cast(__m256  v) { return _mm256_castpd256_pd128(_mm256_castps_pd(v)); }
template<> Vc_INTRINSIC __m128d intrin_cast(__m256i v) { return _mm256_castpd256_pd128(_mm256_castsi256_pd(v)); }
template<> Vc_INTRINSIC __m128d intrin_cast(__m256d v) { return _mm256_castpd256_pd128(v); }

// 256 -> 256
template<> Vc_INTRINSIC __m256  intrin_cast(__m256  v) { return v; }
template<> Vc_INTRINSIC __m256  intrin_cast(__m256i v) { return _mm256_castsi256_ps(v); }
template<> Vc_INTRINSIC __m256  intrin_cast(__m256d v) { return _mm256_castpd_ps(v); }
template<> Vc_INTRINSIC __m256i intrin_cast(__m256  v) { return _mm256_castps_si256(v); }
template<> Vc_INTRINSIC __m256i intrin_cast(__m256i v) { return v; }
template<> Vc_INTRINSIC __m256i intrin_cast(__m256d v) { return _mm256_castpd_si256(v); }
template<> Vc_INTRINSIC __m256d intrin_cast(__m256  v) { return _mm256_castps_pd(v); }
template<> Vc_INTRINSIC __m256d intrin_cast(__m256i v) { return _mm256_castsi256_pd(v); }
template<> Vc_INTRINSIC __m256d intrin_cast(__m256d v) { return v; }

#ifdef Vc_HAVE_AVX512F
// 256 -> 512
template<> Vc_INTRINSIC __m512  intrin_cast(__m256  v) { return _mm512_castps256_ps512(v); }
template<> Vc_INTRINSIC __m512  intrin_cast(__m256i v) { return _mm512_castps256_ps512(intrin_cast<__m256>(v)); }
template<> Vc_INTRINSIC __m512  intrin_cast(__m256d v) { return _mm512_castps256_ps512(intrin_cast<__m256>(v)); }
template<> Vc_INTRINSIC __m512i intrin_cast(__m256  v) { return _mm512_castsi256_si512(intrin_cast<__m256i>(v)); }
template<> Vc_INTRINSIC __m512i intrin_cast(__m256i v) { return _mm512_castsi256_si512(v); }
template<> Vc_INTRINSIC __m512i intrin_cast(__m256d v) { return _mm512_castsi256_si512(intrin_cast<__m256i>(v)); }
template<> Vc_INTRINSIC __m512d intrin_cast(__m256  v) { return _mm512_castpd256_pd512(intrin_cast<__m256d>(v)); }
template<> Vc_INTRINSIC __m512d intrin_cast(__m256i v) { return _mm512_castpd256_pd512(intrin_cast<__m256d>(v)); }
template<> Vc_INTRINSIC __m512d intrin_cast(__m256d v) { return _mm512_castpd256_pd512(v); }

// 512 -> 128
template<> Vc_INTRINSIC __m128  intrin_cast(__m512  v) { return _mm512_castps512_ps128(v); }
template<> Vc_INTRINSIC __m128  intrin_cast(__m512i v) { return _mm512_castps512_ps128(_mm512_castsi512_ps(v)); }
template<> Vc_INTRINSIC __m128  intrin_cast(__m512d v) { return _mm512_castps512_ps128(_mm512_castpd_ps(v)); }
template<> Vc_INTRINSIC __m128i intrin_cast(__m512  v) { return _mm512_castsi512_si128(_mm512_castps_si512(v)); }
template<> Vc_INTRINSIC __m128i intrin_cast(__m512i v) { return _mm512_castsi512_si128(v); }
template<> Vc_INTRINSIC __m128i intrin_cast(__m512d v) { return _mm512_castsi512_si128(_mm512_castpd_si512(v)); }
template<> Vc_INTRINSIC __m128d intrin_cast(__m512  v) { return _mm512_castpd512_pd128(_mm512_castps_pd(v)); }
template<> Vc_INTRINSIC __m128d intrin_cast(__m512i v) { return _mm512_castpd512_pd128(_mm512_castsi512_pd(v)); }
template<> Vc_INTRINSIC __m128d intrin_cast(__m512d v) { return _mm512_castpd512_pd128(v); }

// 512 -> 256
template<> Vc_INTRINSIC __m256  intrin_cast(__m512  v) { return _mm512_castps512_ps256(v); }
template<> Vc_INTRINSIC __m256  intrin_cast(__m512i v) { return _mm512_castps512_ps256(_mm512_castsi512_ps(v)); }
template<> Vc_INTRINSIC __m256  intrin_cast(__m512d v) { return _mm512_castps512_ps256(_mm512_castpd_ps(v)); }
template<> Vc_INTRINSIC __m256i intrin_cast(__m512  v) { return _mm512_castsi512_si256(_mm512_castps_si512(v)); }
template<> Vc_INTRINSIC __m256i intrin_cast(__m512i v) { return _mm512_castsi512_si256(v); }
template<> Vc_INTRINSIC __m256i intrin_cast(__m512d v) { return _mm512_castsi512_si256(_mm512_castpd_si512(v)); }
template<> Vc_INTRINSIC __m256d intrin_cast(__m512  v) { return _mm512_castpd512_pd256(_mm512_castps_pd(v)); }
template<> Vc_INTRINSIC __m256d intrin_cast(__m512i v) { return _mm512_castpd512_pd256(_mm512_castsi512_pd(v)); }
template<> Vc_INTRINSIC __m256d intrin_cast(__m512d v) { return _mm512_castpd512_pd256(v); }

// 512 -> 512
template<> Vc_INTRINSIC __m512  intrin_cast(__m512  v) { return v; }
template<> Vc_INTRINSIC __m512  intrin_cast(__m512i v) { return _mm512_castsi512_ps(v); }
template<> Vc_INTRINSIC __m512  intrin_cast(__m512d v) { return _mm512_castpd_ps(v); }
template<> Vc_INTRINSIC __m512i intrin_cast(__m512  v) { return _mm512_castps_si512(v); }
template<> Vc_INTRINSIC __m512i intrin_cast(__m512i v) { return v; }
template<> Vc_INTRINSIC __m512i intrin_cast(__m512d v) { return _mm512_castpd_si512(v); }
template<> Vc_INTRINSIC __m512d intrin_cast(__m512  v) { return _mm512_castps_pd(v); }
template<> Vc_INTRINSIC __m512d intrin_cast(__m512i v) { return _mm512_castsi512_pd(v); }
template<> Vc_INTRINSIC __m512d intrin_cast(__m512d v) { return v; }
#endif  // Vc_HAVE_AVX512F
#endif  // Vc_HAVE_AVX
#endif  // Vc_HAVE_SSE2

// insert128{{{1
#ifdef Vc_HAVE_AVX
template <int offset> Vc_INTRINSIC __m256 insert128(__m256 a, __m128 b)
{
    return _mm256_insertf128_ps(a, b, offset);
}
template <int offset> Vc_INTRINSIC __m256d insert128(__m256d a, __m128d b)
{
    return _mm256_insertf128_pd(a, b, offset);
}
template <int offset> Vc_INTRINSIC __m256i insert128(__m256i a, __m128i b)
{
#ifdef Vc_HAVE_AVX2
    return _mm256_inserti128_si256(a, b, offset);
#else
    return _mm256_insertf128_si256(a, b, offset);
#endif
}
#endif  // Vc_HAVE_AVX

// insert256{{{1
#ifdef Vc_HAVE_AVX512F
template <int offset> Vc_INTRINSIC __m512 insert256(__m512 a, __m256 b)
{
    return _mm512_castpd_ps(_mm512_insertf64x4(_mm512_castps_pd(a), _mm256_castps_pd(b), offset));
}
template <int offset> Vc_INTRINSIC __m512d insert256(__m512d a, __m256d b)
{
    return _mm512_insertf64x4(a, b, offset);
}
template <int offset> Vc_INTRINSIC __m512i insert256(__m512i a, __m256i b)
{
    return _mm512_inserti64x4(a, b, offset);
}
#endif  // Vc_HAVE_AVX512F

// extract128{{{1
#ifdef Vc_HAVE_AVX
template <int offset> Vc_INTRINSIC __m128 extract128(__m256 a)
{
    return _mm256_extractf128_ps(a, offset);
}
template <int offset> Vc_INTRINSIC __m128d extract128(__m256d a)
{
    return _mm256_extractf128_pd(a, offset);
}
template <int offset> Vc_INTRINSIC __m128i extract128(__m256i a)
{
#ifdef Vc_IMPL_AVX2
    return _mm256_extracti128_si256(a, offset);
#else
    return _mm256_extractf128_si256(a, offset);
#endif
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
template <int offset> Vc_INTRINSIC __m128 extract128(__m512 a)
{
    return _mm512_extractf32x4_ps(a, offset);
}
template <int offset> Vc_INTRINSIC __m128d extract128(__m512d a)
{
#ifdef Vc_HAVE_AVX512DQ
    return _mm512_extractf64x2_pd(a, offset);
#else
    return _mm_castps_pd(_mm512_extractf32x4_ps(_mm512_castpd_ps(a), offset));
#endif
}
template <int offset> Vc_INTRINSIC __m128i extract128(__m512i a)
{
    return _mm512_extracti32x4_epi32(a, offset);
}
#endif  // Vc_HAVE_AVX512F

// extract256{{{1
#ifdef Vc_HAVE_AVX512F
template <int offset> Vc_INTRINSIC __m256 extract256(__m512 a)
{
#ifdef Vc_HAVE_AVX512DQ
    return _mm512_extractf32x8_ps(a, offset);
#else
    return _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(a), offset));
#endif
}
template <int offset> Vc_INTRINSIC __m256d extract256(__m512d a)
{
    return _mm512_extractf64x4_pd(a, offset);
}
template <int offset> Vc_INTRINSIC __m256i extract256(__m512i a)
{
    return _mm512_extracti64x4_epi64(a, offset);
}
#endif  // Vc_HAVE_AVX512F

// lo/hi128{{{1
#ifdef Vc_HAVE_AVX
Vc_INTRINSIC Vc_CONST __m128  lo128(__m256  v) { return intrin_cast<__m128 >(v); }
Vc_INTRINSIC Vc_CONST __m128d lo128(__m256d v) { return intrin_cast<__m128d>(v); }
Vc_INTRINSIC Vc_CONST __m128i lo128(__m256i v) { return intrin_cast<__m128i>(v); }
Vc_INTRINSIC Vc_CONST __m128  hi128(__m256  v) { return extract128<1>(v); }
Vc_INTRINSIC Vc_CONST __m128d hi128(__m256d v) { return extract128<1>(v); }
Vc_INTRINSIC Vc_CONST __m128i hi128(__m256i v) { return extract128<1>(v); }
#endif  // Vc_HAVE_AVX

// lo/hi256{{{1
#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC Vc_CONST __m256  lo256(__m512  v) { return intrin_cast<__m256 >(v); }
Vc_INTRINSIC Vc_CONST __m256d lo256(__m512d v) { return intrin_cast<__m256d>(v); }
Vc_INTRINSIC Vc_CONST __m256i lo256(__m512i v) { return intrin_cast<__m256i>(v); }
Vc_INTRINSIC Vc_CONST __m256  hi256(__m512  v) { return extract256<1>(v); }
Vc_INTRINSIC Vc_CONST __m256d hi256(__m512d v) { return extract256<1>(v); }
Vc_INTRINSIC Vc_CONST __m256i hi256(__m512i v) { return extract256<1>(v); }

Vc_INTRINSIC Vc_CONST __m128  lo128(__m512  v) { return intrin_cast<__m128 >(v); }
Vc_INTRINSIC Vc_CONST __m128d lo128(__m512d v) { return intrin_cast<__m128d>(v); }
Vc_INTRINSIC Vc_CONST __m128i lo128(__m512i v) { return intrin_cast<__m128i>(v); }
#endif  // Vc_HAVE_AVX

// concat{{{1
#ifdef Vc_HAVE_AVX
Vc_INTRINSIC Vc_CONST __m256 concat(__m128 a, __m128 b)
{
    return insert128<1>(intrin_cast<__m256>(a), b);
}
Vc_INTRINSIC Vc_CONST __m256d concat(__m128d a, __m128d b)
{
    return insert128<1>(intrin_cast<__m256d>(a), b);
}
Vc_INTRINSIC Vc_CONST __m256i concat(__m128i a, __m128i b)
{
    return insert128<1>(intrin_cast<__m256i>(a), b);
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC Vc_CONST __m512 concat(__m256 a, __m256 b)
{
    return insert256<1>(intrin_cast<__m512>(a), b);
}
Vc_INTRINSIC Vc_CONST __m512d concat(__m256d a, __m256d b)
{
    return insert256<1>(intrin_cast<__m512d>(a), b);
}
Vc_INTRINSIC Vc_CONST __m512i concat(__m256i a, __m256i b)
{
    return insert256<1>(intrin_cast<__m512i>(a), b);
}
#endif  // Vc_HAVE_AVX512F

// allone{{{1
template <typename V> Vc_INTRINSIC_L Vc_CONST_L V allone() Vc_INTRINSIC_R Vc_CONST_R;
template <> Vc_INTRINSIC Vc_CONST __m128 allone<__m128>()
{
    return _mm_load_ps(reinterpret_cast<const float *>(sse_const::AllBitsSet));
}
#ifdef Vc_HAVE_SSE2
template <> Vc_INTRINSIC Vc_CONST __m128i allone<__m128i>()
{
    return _mm_load_si128(reinterpret_cast<const __m128i *>(sse_const::AllBitsSet));
}
template <> Vc_INTRINSIC Vc_CONST __m128d allone<__m128d>()
{
    return _mm_load_pd(reinterpret_cast<const double *>(sse_const::AllBitsSet));
}
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC Vc_CONST __m256 allone<__m256>()
{
    return _mm256_load_ps(reinterpret_cast<const float *>(avx_const::AllBitsSet));
}
template <> Vc_INTRINSIC Vc_CONST __m256i allone<__m256i>()
{
    return _mm256_load_si256(reinterpret_cast<const __m256i *>(avx_const::AllBitsSet));
}
template <> Vc_INTRINSIC Vc_CONST __m256d allone<__m256d>()
{
    return _mm256_load_pd(reinterpret_cast<const double *>(avx_const::AllBitsSet));
}
#endif

#ifdef Vc_HAVE_AVX512F
template <> Vc_INTRINSIC Vc_CONST __m512 allone<__m512>()
{
    return _mm512_broadcast_f32x4(allone<__m128>());
}
template <> Vc_INTRINSIC Vc_CONST __m512d allone<__m512d>()
{
    return _mm512_castps_pd(allone<__m512>());
}
template <> Vc_INTRINSIC Vc_CONST __m512i allone<__m512i>()
{
    return _mm512_broadcast_i32x4(allone<__m128i>());
}
#endif  // Vc_HAVE_AVX512F

// zero{{{1
template <typename V> Vc_INTRINSIC_L Vc_CONST_L V zero() Vc_INTRINSIC_R Vc_CONST_R;
template<> Vc_INTRINSIC Vc_CONST __m128  zero<__m128 >() { return _mm_setzero_ps(); }
#ifdef Vc_HAVE_SSE2
template<> Vc_INTRINSIC Vc_CONST __m128i zero<__m128i>() { return _mm_setzero_si128(); }
template<> Vc_INTRINSIC Vc_CONST __m128d zero<__m128d>() { return _mm_setzero_pd(); }
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
template<> Vc_INTRINSIC Vc_CONST __m256  zero<__m256 >() { return _mm256_setzero_ps(); }
template<> Vc_INTRINSIC Vc_CONST __m256i zero<__m256i>() { return _mm256_setzero_si256(); }
template<> Vc_INTRINSIC Vc_CONST __m256d zero<__m256d>() { return _mm256_setzero_pd(); }
#endif

#ifdef Vc_HAVE_AVX512F
template<> Vc_INTRINSIC Vc_CONST __m512  zero<__m512 >() { return _mm512_setzero_ps(); }
template<> Vc_INTRINSIC Vc_CONST __m512i zero<__m512i>() { return _mm512_setzero_si512(); }
template<> Vc_INTRINSIC Vc_CONST __m512d zero<__m512d>() { return _mm512_setzero_pd(); }
#endif

// one16/32{{{1
Vc_INTRINSIC Vc_CONST __m128  one16( float) { return _mm_load_ps(sse_const::oneFloat); }

#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC Vc_CONST __m128d one16(double) { return _mm_load_pd(sse_const::oneDouble); }
Vc_INTRINSIC Vc_CONST __m128i one16( schar) { return _mm_load_si128(reinterpret_cast<const __m128i *>(sse_const::one8)); }
Vc_INTRINSIC Vc_CONST __m128i one16( uchar) { return one16(schar()); }
Vc_INTRINSIC Vc_CONST __m128i one16( short) { return _mm_load_si128(reinterpret_cast<const __m128i *>(sse_const::one16)); }
Vc_INTRINSIC Vc_CONST __m128i one16(ushort) { return one16(short()); }
Vc_INTRINSIC Vc_CONST __m128i one16(   int) { return _mm_load_si128(reinterpret_cast<const __m128i *>(sse_const::one32)); }
Vc_INTRINSIC Vc_CONST __m128i one16(  uint) { return one16(int()); }
Vc_INTRINSIC Vc_CONST __m128i one16( llong) { return _mm_load_si128(reinterpret_cast<const __m128i *>(sse_const::one64)); }
Vc_INTRINSIC Vc_CONST __m128i one16(ullong) { return one16(llong()); }
Vc_INTRINSIC Vc_CONST __m128i one16(  long) { return one16(equal_int_type_t<long>()); }
Vc_INTRINSIC Vc_CONST __m128i one16( ulong) { return one16(equal_int_type_t<ulong>()); }
#endif

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC Vc_CONST __m256  one32( float) { return _mm256_broadcast_ss(&avx_const::oneFloat); }
Vc_INTRINSIC Vc_CONST __m256d one32(double) { return _mm256_broadcast_sd(&avx_const::oneDouble); }
Vc_INTRINSIC Vc_CONST __m256i one32( llong) { return _mm256_castpd_si256(_mm256_broadcast_sd(reinterpret_cast<const double *>(&avx_const::IndexesFromZero64[1]))); }
Vc_INTRINSIC Vc_CONST __m256i one32(ullong) { return one32(llong()); }
Vc_INTRINSIC Vc_CONST __m256i one32(   int) { return _mm256_castps_si256(_mm256_broadcast_ss(reinterpret_cast<const float *>(&avx_const::IndexesFromZero32[1]))); }
Vc_INTRINSIC Vc_CONST __m256i one32(  uint) { return one32(int()); }
Vc_INTRINSIC Vc_CONST __m256i one32( short) { return _mm256_castps_si256(_mm256_broadcast_ss(reinterpret_cast<const float *>(avx_const::one16))); }
Vc_INTRINSIC Vc_CONST __m256i one32(ushort) { return one32(short()); }
Vc_INTRINSIC Vc_CONST __m256i one32( schar) { return _mm256_castps_si256(_mm256_broadcast_ss(reinterpret_cast<const float *>(avx_const::one8))); }
Vc_INTRINSIC Vc_CONST __m256i one32( uchar) { return one32(schar()); }
Vc_INTRINSIC Vc_CONST __m256i one32(  long) { return one32(equal_int_type_t<long>()); }
Vc_INTRINSIC Vc_CONST __m256i one32( ulong) { return one32(equal_int_type_t<ulong>()); }
#endif

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC Vc_CONST __m512  one64( float) { return _mm512_broadcastss_ps(_mm_load_ss(&avx_const::oneFloat)); }
Vc_INTRINSIC Vc_CONST __m512d one64(double) { return _mm512_broadcastsd_pd(_mm_load_sd(&avx_const::oneDouble)); }
Vc_INTRINSIC Vc_CONST __m512i one64( llong) { return _mm512_set1_epi64(1ll); }
Vc_INTRINSIC Vc_CONST __m512i one64(ullong) { return _mm512_set1_epi64(1ll); }
Vc_INTRINSIC Vc_CONST __m512i one64(   int) { return _mm512_set1_epi32(1); }
Vc_INTRINSIC Vc_CONST __m512i one64(  uint) { return _mm512_set1_epi32(1); }
Vc_INTRINSIC Vc_CONST __m512i one64( short) { return _mm512_set1_epi16(1); }
Vc_INTRINSIC Vc_CONST __m512i one64(ushort) { return _mm512_set1_epi16(1); }
Vc_INTRINSIC Vc_CONST __m512i one64( schar) { return _mm512_broadcast_i32x4(one16(schar())); }
Vc_INTRINSIC Vc_CONST __m512i one64( uchar) { return one64(schar()); }
Vc_INTRINSIC Vc_CONST __m512i one64(  long) { return one64(equal_int_type_t<long>()); }
Vc_INTRINSIC Vc_CONST __m512i one64( ulong) { return one64(equal_int_type_t<ulong>()); }
#endif  // Vc_HAVE_AVX512F

// signmask{{{1
#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC Vc_CONST __m128d signmask16(double){ return _mm_load_pd(reinterpret_cast<const double *>(sse_const::signMaskDouble)); }
#endif  // Vc_HAVE_SSE2
Vc_INTRINSIC Vc_CONST __m128  signmask16( float){ return _mm_load_ps(reinterpret_cast<const float *>(sse_const::signMaskFloat)); }

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC Vc_CONST __m256d signmask32(double){ return _mm256_broadcast_sd(reinterpret_cast<const double *>(&avx_const::signMaskFloat[0])); }
Vc_INTRINSIC Vc_CONST __m256  signmask32( float){ return _mm256_broadcast_ss(reinterpret_cast<const float *>(&avx_const::signMaskFloat[1])); }
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC Vc_CONST __m512d signmask64(double){ return _mm512_broadcast_f64x4(signmask32(double())); }
Vc_INTRINSIC Vc_CONST __m512  signmask64( float){ return _mm512_broadcast_f32x4(signmask16(float())); }
#endif  // Vc_HAVE_AVX

// set16/32/64{{{1
Vc_INTRINSIC Vc_CONST __m128 set(float x0, float x1, float x2, float x3)
{
    return _mm_set_ps(x3, x2, x1, x0);
}
#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC Vc_CONST __m128d set(double x0, double x1) { return _mm_set_pd(x1, x0); }

Vc_INTRINSIC Vc_CONST __m128i set(llong x0, llong x1) { return _mm_set_epi64x(x1, x0); }
Vc_INTRINSIC Vc_CONST __m128i set(ullong x0, ullong x1) { return _mm_set_epi64x(x1, x0); }

Vc_INTRINSIC Vc_CONST __m128i set(int x0, int x1, int x2, int x3)
{
    return _mm_set_epi32(x3, x2, x1, x0);
}
Vc_INTRINSIC Vc_CONST __m128i set(uint x0, uint x1, uint x2, uint x3)
{
    return _mm_set_epi32(x3, x2, x1, x0);
}

Vc_INTRINSIC Vc_CONST __m128i set(short x0, short x1, short x2, short x3, short x4,
                                  short x5, short x6, short x7)
{
    return _mm_set_epi16(x7, x6, x5, x4, x3, x2, x1, x0);
}
Vc_INTRINSIC Vc_CONST __m128i set(ushort x0, ushort x1, ushort x2, ushort x3, ushort x4,
                                  ushort x5, ushort x6, ushort x7)
{
    return _mm_set_epi16(x7, x6, x5, x4, x3, x2, x1, x0);
}

Vc_INTRINSIC Vc_CONST __m128i set(schar x0, schar x1, schar x2, schar x3, schar x4,
                                  schar x5, schar x6, schar x7, schar x8, schar x9,
                                  schar x10, schar x11, schar x12, schar x13, schar x14,
                                  schar x15)
{
    return _mm_set_epi8(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2, x1,
                         x0);
}
Vc_INTRINSIC Vc_CONST __m128i set(uchar x0, uchar x1, uchar x2, uchar x3, uchar x4,
                                  uchar x5, uchar x6, uchar x7, uchar x8, uchar x9,
                                  uchar x10, uchar x11, uchar x12, uchar x13, uchar x14,
                                  uchar x15)
{
    return _mm_set_epi8(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2, x1,
                         x0);
}
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC Vc_CONST __m256 set(float x0, float x1, float x2, float x3, float x4,
                                 float x5, float x6, float x7)
{
    return _mm256_set_ps(x7, x6, x5, x4, x3, x2, x1, x0);
}

Vc_INTRINSIC Vc_CONST __m256d set(double x0, double x1, double x2, double x3)
{
    return _mm256_set_pd(x3, x2, x1, x0);
}

Vc_INTRINSIC Vc_CONST __m256i set(llong x0, llong x1, llong x2, llong x3)
{
    return _mm256_set_epi64x(x3, x2, x1, x0);
}
Vc_INTRINSIC Vc_CONST __m256i set(ullong x0, ullong x1, ullong x2, ullong x3)
{
    return _mm256_set_epi64x(x3, x2, x1, x0);
}

Vc_INTRINSIC Vc_CONST __m256i set(int x0, int x1, int x2, int x3, int x4, int x5, int x6,
                                  int x7)
{
    return _mm256_set_epi32(x7, x6, x5, x4, x3, x2, x1, x0);
}
Vc_INTRINSIC Vc_CONST __m256i set(uint x0, uint x1, uint x2, uint x3, uint x4, uint x5,
                                  uint x6, uint x7)
{
    return _mm256_set_epi32(x7, x6, x5, x4, x3, x2, x1, x0);
}

Vc_INTRINSIC Vc_CONST __m256i set(short x0, short x1, short x2, short x3, short x4,
                                  short x5, short x6, short x7, short x8, short x9,
                                  short x10, short x11, short x12, short x13, short x14,
                                  short x15)
{
    return _mm256_set_epi16(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2,
                            x1, x0);
}
Vc_INTRINSIC Vc_CONST __m256i set(ushort x0, ushort x1, ushort x2, ushort x3, ushort x4,
                                  ushort x5, ushort x6, ushort x7, ushort x8, ushort x9,
                                  ushort x10, ushort x11, ushort x12, ushort x13,
                                  ushort x14, ushort x15)
{
    return _mm256_set_epi16(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2,
                            x1, x0);
}

Vc_INTRINSIC Vc_CONST __m256i set(schar x0, schar x1, schar x2, schar x3, schar x4,
                                  schar x5, schar x6, schar x7, schar x8, schar x9,
                                  schar x10, schar x11, schar x12, schar x13, schar x14,
                                  schar x15, schar x16, schar x17, schar x18, schar x19,
                                  schar x20, schar x21, schar x22, schar x23, schar x24,
                                  schar x25, schar x26, schar x27, schar x28, schar x29,
                                  schar x30, schar x31)
{
    return _mm256_set_epi8(x31, x30, x29, x28, x27, x26, x25, x24, x23, x22, x21, x20,
                           x19, x18, x17, x16, x15, x14, x13, x12, x11, x10, x9, x8, x7,
                           x6, x5, x4, x3, x2, x1, x0);
}
Vc_INTRINSIC Vc_CONST __m256i set(uchar x0, uchar x1, uchar x2, uchar x3, uchar x4,
                                  uchar x5, uchar x6, uchar x7, uchar x8, uchar x9,
                                  uchar x10, uchar x11, uchar x12, uchar x13, uchar x14,
                                  uchar x15, uchar x16, uchar x17, uchar x18, uchar x19,
                                  uchar x20, uchar x21, uchar x22, uchar x23, uchar x24,
                                  uchar x25, uchar x26, uchar x27, uchar x28, uchar x29,
                                  uchar x30, uchar x31)
{
    return _mm256_set_epi8(x31, x30, x29, x28, x27, x26, x25, x24, x23, x22, x21, x20,
                           x19, x18, x17, x16, x15, x14, x13, x12, x11, x10, x9, x8, x7,
                           x6, x5, x4, x3, x2, x1, x0);
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC Vc_CONST __m512d set(double x0, double x1, double x2, double x3, double x4,
                                  double x5, double x6, double x7)
{
    return _mm512_set_pd(x7, x6, x5, x4, x3, x2, x1, x0);
}

Vc_INTRINSIC Vc_CONST __m512 set(float x0, float x1, float x2, float x3, float x4,
                                 float x5, float x6, float x7, float x8, float x9,
                                 float x10, float x11, float x12, float x13, float x14,
                                 float x15)
{
    return _mm512_set_ps(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2, x1,
                         x0);
}

Vc_INTRINSIC Vc_CONST __m512i set(llong x0, llong x1, llong x2, llong x3, llong x4,
                                  llong x5, llong x6, llong x7)
{
    return _mm512_set_epi64(x7, x6, x5, x4, x3, x2, x1, x0);
}
Vc_INTRINSIC Vc_CONST __m512i set(ullong x0, ullong x1, ullong x2, ullong x3, ullong x4,
                                  ullong x5, ullong x6, ullong x7)
{
    return _mm512_set_epi64(x7, x6, x5, x4, x3, x2, x1, x0);
}

Vc_INTRINSIC Vc_CONST __m512i set(int x0, int x1, int x2, int x3, int x4, int x5, int x6,
                                  int x7, int x8, int x9, int x10, int x11, int x12,
                                  int x13, int x14, int x15)
{
    return _mm512_set_epi32(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2,
                            x1, x0);
}
Vc_INTRINSIC Vc_CONST __m512i set(uint x0, uint x1, uint x2, uint x3, uint x4, uint x5,
                                  uint x6, uint x7, uint x8, uint x9, uint x10, uint x11,
                                  uint x12, uint x13, uint x14, uint x15)
{
    return _mm512_set_epi32(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2,
                            x1, x0);
}

Vc_INTRINSIC Vc_CONST __m512i set(short x0, short x1, short x2, short x3, short x4,
                                  short x5, short x6, short x7, short x8, short x9,
                                  short x10, short x11, short x12, short x13, short x14,
                                  short x15, short x16, short x17, short x18, short x19,
                                  short x20, short x21, short x22, short x23, short x24,
                                  short x25, short x26, short x27, short x28, short x29,
                                  short x30, short x31)
{
    return concat(_mm256_set_epi16(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4,
                                   x3, x2, x1, x0),
                  _mm256_set_epi16(x31, x30, x29, x28, x27, x26, x25, x24, x23, x22, x21,
                                   x20, x19, x18, x17, x16));
}

Vc_INTRINSIC Vc_CONST __m512i set(ushort x0, ushort x1, ushort x2, ushort x3, ushort x4,
                                  ushort x5, ushort x6, ushort x7, ushort x8, ushort x9,
                                  ushort x10, ushort x11, ushort x12, ushort x13, ushort x14,
                                  ushort x15, ushort x16, ushort x17, ushort x18, ushort x19,
                                  ushort x20, ushort x21, ushort x22, ushort x23, ushort x24,
                                  ushort x25, ushort x26, ushort x27, ushort x28, ushort x29,
                                  ushort x30, ushort x31)
{
    return concat(_mm256_set_epi16(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4,
                                   x3, x2, x1, x0),
                  _mm256_set_epi16(x31, x30, x29, x28, x27, x26, x25, x24, x23, x22, x21,
                                   x20, x19, x18, x17, x16));
}

Vc_INTRINSIC Vc_CONST __m512i
set(schar x0, schar x1, schar x2, schar x3, schar x4, schar x5, schar x6, schar x7,
    schar x8, schar x9, schar x10, schar x11, schar x12, schar x13, schar x14, schar x15,
    schar x16, schar x17, schar x18, schar x19, schar x20, schar x21, schar x22,
    schar x23, schar x24, schar x25, schar x26, schar x27, schar x28, schar x29,
    schar x30, schar x31, schar x32, schar x33, schar x34, schar x35, schar x36,
    schar x37, schar x38, schar x39, schar x40, schar x41, schar x42, schar x43,
    schar x44, schar x45, schar x46, schar x47, schar x48, schar x49, schar x50,
    schar x51, schar x52, schar x53, schar x54, schar x55, schar x56, schar x57,
    schar x58, schar x59, schar x60, schar x61, schar x62, schar x63)
{
    return concat(_mm256_set_epi8(x31, x30, x29, x28, x27, x26, x25, x24, x23, x22, x21,
                                  x20, x19, x18, x17, x16, x15, x14, x13, x12, x11, x10,
                                  x9, x8, x7, x6, x5, x4, x3, x2, x1, x0),
                  _mm256_set_epi8(x63, x62, x61, x60, x59, x58, x57, x56, x55, x54, x53,
                                  x52, x51, x50, x49, x48, x47, x46, x45, x44, x43, x42,
                                  x41, x40, x39, x38, x37, x36, x35, x34, x33, x32));
}

Vc_INTRINSIC Vc_CONST __m512i
set(uchar x0, uchar x1, uchar x2, uchar x3, uchar x4, uchar x5, uchar x6, uchar x7,
    uchar x8, uchar x9, uchar x10, uchar x11, uchar x12, uchar x13, uchar x14, uchar x15,
    uchar x16, uchar x17, uchar x18, uchar x19, uchar x20, uchar x21, uchar x22,
    uchar x23, uchar x24, uchar x25, uchar x26, uchar x27, uchar x28, uchar x29,
    uchar x30, uchar x31, uchar x32, uchar x33, uchar x34, uchar x35, uchar x36,
    uchar x37, uchar x38, uchar x39, uchar x40, uchar x41, uchar x42, uchar x43,
    uchar x44, uchar x45, uchar x46, uchar x47, uchar x48, uchar x49, uchar x50,
    uchar x51, uchar x52, uchar x53, uchar x54, uchar x55, uchar x56, uchar x57,
    uchar x58, uchar x59, uchar x60, uchar x61, uchar x62, uchar x63)
{
    return concat(_mm256_set_epi8(x31, x30, x29, x28, x27, x26, x25, x24, x23, x22, x21,
                                  x20, x19, x18, x17, x16, x15, x14, x13, x12, x11, x10,
                                  x9, x8, x7, x6, x5, x4, x3, x2, x1, x0),
                  _mm256_set_epi8(x63, x62, x61, x60, x59, x58, x57, x56, x55, x54, x53,
                                  x52, x51, x50, x49, x48, x47, x46, x45, x44, x43, x42,
                                  x41, x40, x39, x38, x37, x36, x35, x34, x33, x32));
}

#endif  // Vc_HAVE_AVX512F

// generic forward for (u)long to (u)int or (u)llong
template <typename... Ts> Vc_INTRINSIC Vc_CONST auto set(Ts... args)
{
    return set(static_cast<equal_int_type_t<Ts>>(args)...);
}

// broadcast16/32/64{{{1
Vc_INTRINSIC __m128  broadcast16( float x) { return _mm_set1_ps(x); }
#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC __m128d broadcast16(double x) { return _mm_set1_pd(x); }
Vc_INTRINSIC __m128i broadcast16( schar x) { return _mm_set1_epi8(x); }
Vc_INTRINSIC __m128i broadcast16( uchar x) { return _mm_set1_epi8(x); }
Vc_INTRINSIC __m128i broadcast16( short x) { return _mm_set1_epi16(x); }
Vc_INTRINSIC __m128i broadcast16(ushort x) { return _mm_set1_epi16(x); }
Vc_INTRINSIC __m128i broadcast16(   int x) { return _mm_set1_epi32(x); }
Vc_INTRINSIC __m128i broadcast16(  uint x) { return _mm_set1_epi32(x); }
Vc_INTRINSIC __m128i broadcast16(  long x) { return sizeof( long) == 4 ? _mm_set1_epi32(x) : _mm_set1_epi64x(x); }
Vc_INTRINSIC __m128i broadcast16( ulong x) { return sizeof(ulong) == 4 ? _mm_set1_epi32(x) : _mm_set1_epi64x(x); }
Vc_INTRINSIC __m128i broadcast16( llong x) { return _mm_set1_epi64x(x); }
Vc_INTRINSIC __m128i broadcast16(ullong x) { return _mm_set1_epi64x(x); }
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC __m256  broadcast32( float x) { return _mm256_set1_ps(x); }
Vc_INTRINSIC __m256d broadcast32(double x) { return _mm256_set1_pd(x); }
Vc_INTRINSIC __m256i broadcast32( schar x) { return _mm256_set1_epi8(x); }
Vc_INTRINSIC __m256i broadcast32( uchar x) { return _mm256_set1_epi8(x); }
Vc_INTRINSIC __m256i broadcast32( short x) { return _mm256_set1_epi16(x); }
Vc_INTRINSIC __m256i broadcast32(ushort x) { return _mm256_set1_epi16(x); }
Vc_INTRINSIC __m256i broadcast32(   int x) { return _mm256_set1_epi32(x); }
Vc_INTRINSIC __m256i broadcast32(  uint x) { return _mm256_set1_epi32(x); }
Vc_INTRINSIC __m256i broadcast32(  long x) { return sizeof( long) == 4 ? _mm256_set1_epi32(x) : _mm256_set1_epi64x(x); }
Vc_INTRINSIC __m256i broadcast32( ulong x) { return sizeof(ulong) == 4 ? _mm256_set1_epi32(x) : _mm256_set1_epi64x(x); }
Vc_INTRINSIC __m256i broadcast32( llong x) { return _mm256_set1_epi64x(x); }
Vc_INTRINSIC __m256i broadcast32(ullong x) { return _mm256_set1_epi64x(x); }
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC __m512  broadcast64( float x) { return _mm512_set1_ps(x); }
Vc_INTRINSIC __m512d broadcast64(double x) { return _mm512_set1_pd(x); }
Vc_INTRINSIC __m512i broadcast64( schar x) { return _mm512_set1_epi8(x); }
Vc_INTRINSIC __m512i broadcast64( uchar x) { return _mm512_set1_epi8(x); }
Vc_INTRINSIC __m512i broadcast64( short x) { return _mm512_set1_epi16(x); }
Vc_INTRINSIC __m512i broadcast64(ushort x) { return _mm512_set1_epi16(x); }
Vc_INTRINSIC __m512i broadcast64(   int x) { return _mm512_set1_epi32(x); }
Vc_INTRINSIC __m512i broadcast64(  uint x) { return _mm512_set1_epi32(x); }
Vc_INTRINSIC __m512i broadcast64(  long x) { return sizeof( long) == 4 ? _mm512_set1_epi32(x) : _mm512_set1_epi64(x); }
Vc_INTRINSIC __m512i broadcast64( ulong x) { return sizeof(ulong) == 4 ? _mm512_set1_epi32(x) : _mm512_set1_epi64(x); }
Vc_INTRINSIC __m512i broadcast64( llong x) { return _mm512_set1_epi64(x); }
Vc_INTRINSIC __m512i broadcast64(ullong x) { return _mm512_set1_epi64(x); }
#endif  // Vc_HAVE_AVX512F

// lowest16/32/64{{{1
template <class T>
Vc_INTRINSIC Vc_CONST typename intrinsic_type_impl<T, 16>::type lowest16()
{
    return broadcast16(std::numeric_limits<T>::lowest());
}

#ifdef Vc_HAVE_SSE2
template <> Vc_INTRINSIC Vc_CONST __m128i lowest16< short>() { return _mm_load_si128(reinterpret_cast<const __m128i *>(sse_const::minShort)); }
template <> Vc_INTRINSIC Vc_CONST __m128i lowest16<   int>() { return _mm_load_si128(reinterpret_cast<const __m128i *>(sse_const::signMaskFloat)); }
template <> Vc_INTRINSIC Vc_CONST __m128i lowest16< llong>() { return _mm_load_si128(reinterpret_cast<const __m128i *>(sse_const::signMaskDouble)); }
template <> Vc_INTRINSIC Vc_CONST __m128i lowest16<  long>() { return lowest16<equal_int_type_t<long>>(); }

template <> Vc_INTRINSIC Vc_CONST __m128i lowest16< uchar>() { return _mm_setzero_si128(); }
template <> Vc_INTRINSIC Vc_CONST __m128i lowest16<ushort>() { return _mm_setzero_si128(); }
template <> Vc_INTRINSIC Vc_CONST __m128i lowest16<  uint>() { return _mm_setzero_si128(); }
template <> Vc_INTRINSIC Vc_CONST __m128i lowest16< ulong>() { return _mm_setzero_si128(); }
template <> Vc_INTRINSIC Vc_CONST __m128i lowest16<ullong>() { return _mm_setzero_si128(); }
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
template <class T>
Vc_INTRINSIC Vc_CONST typename intrinsic_type_impl<T, 32>::type lowest32()
{
    return broadcast32(std::numeric_limits<T>::lowest());
}

template <> Vc_INTRINSIC Vc_CONST __m256i lowest32<short>() { return _mm256_castps_si256(_mm256_broadcast_ss(reinterpret_cast<const float *>(sse_const::minShort))); }
template <> Vc_INTRINSIC Vc_CONST __m256i lowest32<  int>() { return _mm256_castps_si256(signmask32(float())); }
template <> Vc_INTRINSIC Vc_CONST __m256i lowest32<llong>() { return _mm256_castpd_si256(signmask32(double())); }
template <> Vc_INTRINSIC Vc_CONST __m256i lowest32<  long>() { return lowest32<equal_int_type_t<long>>(); }

template <> Vc_INTRINSIC Vc_CONST __m256i lowest32< uchar>() { return _mm256_setzero_si256(); }
template <> Vc_INTRINSIC Vc_CONST __m256i lowest32<ushort>() { return _mm256_setzero_si256(); }
template <> Vc_INTRINSIC Vc_CONST __m256i lowest32<  uint>() { return _mm256_setzero_si256(); }
template <> Vc_INTRINSIC Vc_CONST __m256i lowest32< ulong>() { return _mm256_setzero_si256(); }
template <> Vc_INTRINSIC Vc_CONST __m256i lowest32<ullong>() { return _mm256_setzero_si256(); }
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
template <class T>
Vc_INTRINSIC Vc_CONST typename intrinsic_type_impl<T, 64>::type lowest64()
{
    return broadcast64(std::numeric_limits<T>::lowest());
}

template <> Vc_INTRINSIC Vc_CONST __m512i lowest64<short>() { return _mm512_broadcast_i32x4(lowest16<short>()); }
template <> Vc_INTRINSIC Vc_CONST __m512i lowest64<  int>() { return _mm512_broadcast_i32x4(lowest16<  int>()); }
template <> Vc_INTRINSIC Vc_CONST __m512i lowest64<llong>() { return _mm512_broadcast_i32x4(lowest16<llong>()); }
template <> Vc_INTRINSIC Vc_CONST __m512i lowest64< long>() { return _mm512_broadcast_i32x4(lowest16< long>()); }

template <> Vc_INTRINSIC Vc_CONST __m512i lowest64< uchar>() { return _mm512_setzero_si512(); }
template <> Vc_INTRINSIC Vc_CONST __m512i lowest64<ushort>() { return _mm512_setzero_si512(); }
template <> Vc_INTRINSIC Vc_CONST __m512i lowest64<  uint>() { return _mm512_setzero_si512(); }
template <> Vc_INTRINSIC Vc_CONST __m512i lowest64< ulong>() { return _mm512_setzero_si512(); }
template <> Vc_INTRINSIC Vc_CONST __m512i lowest64<ullong>() { return _mm512_setzero_si512(); }
#endif  // Vc_HAVE_AVX512F

// _2_pow_31{{{1
template <class T> inline typename intrinsic_type_impl<T, 16>::type sse_2_pow_31();
template <> Vc_INTRINSIC Vc_CONST __m128  sse_2_pow_31< float>() { return broadcast16( float(1u << 31)); }
#ifdef Vc_HAVE_SSE2
template <> Vc_INTRINSIC Vc_CONST __m128d sse_2_pow_31<double>() { return broadcast16(double(1u << 31)); }
template <> Vc_INTRINSIC Vc_CONST __m128i sse_2_pow_31<  uint>() { return lowest16<int>(); }
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
template <class T> inline typename intrinsic_type_impl<T, 32>::type avx_2_pow_31();
template <> Vc_INTRINSIC Vc_CONST __m256  avx_2_pow_31< float>() { return _mm256_broadcast_ss(&avx_const::_2_pow_31); }
template <> Vc_INTRINSIC Vc_CONST __m256d avx_2_pow_31<double>() { return broadcast32(double(1u << 31)); }
template <> Vc_INTRINSIC Vc_CONST __m256i avx_2_pow_31<  uint>() { return lowest32<int>(); }
#endif  // Vc_HAVE_AVX

// SSE intrinsics emulation{{{1
Vc_INTRINSIC __m128  setone_ps()     { return _mm_load_ps(sse_const::oneFloat); }
Vc_INTRINSIC __m128  setabsmask_ps() { return _mm_load_ps(reinterpret_cast<const float *>(sse_const::absMaskFloat)); }

#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC __m128i setone_epi8 ()  { return _mm_set1_epi8(1); }
Vc_INTRINSIC __m128i setone_epu8 ()  { return setone_epi8(); }
Vc_INTRINSIC __m128i setone_epi16()  { return _mm_load_si128(reinterpret_cast<const __m128i *>(sse_const::one16)); }
Vc_INTRINSIC __m128i setone_epu16()  { return setone_epi16(); }
Vc_INTRINSIC __m128i setone_epi32()  { return _mm_load_si128(reinterpret_cast<const __m128i *>(sse_const::one32)); }
Vc_INTRINSIC __m128i setone_epu32()  { return setone_epi32(); }

Vc_INTRINSIC __m128d setone_pd()     { return _mm_load_pd(sse_const::oneDouble); }

Vc_INTRINSIC __m128d setabsmask_pd() { return _mm_load_pd(reinterpret_cast<const double *>(sse_const::absMaskDouble)); }
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_SSE2
#if defined(Vc_IMPL_XOP)
Vc_INTRINSIC __m128i cmplt_epu8 (__m128i a, __m128i b) { return _mm_comlt_epu8 (a, b); }
Vc_INTRINSIC __m128i cmpgt_epu8 (__m128i a, __m128i b) { return _mm_comgt_epu8 (a, b); }
Vc_INTRINSIC __m128i cmplt_epu16(__m128i a, __m128i b) { return _mm_comlt_epu16(a, b); }
Vc_INTRINSIC __m128i cmpgt_epu16(__m128i a, __m128i b) { return _mm_comgt_epu16(a, b); }
Vc_INTRINSIC __m128i cmplt_epu32(__m128i a, __m128i b) { return _mm_comlt_epu32(a, b); }
Vc_INTRINSIC __m128i cmpgt_epu32(__m128i a, __m128i b) { return _mm_comgt_epu32(a, b); }
Vc_INTRINSIC __m128i cmplt_epu64(__m128i a, __m128i b) { return _mm_comlt_epu64(a, b); }
Vc_INTRINSIC __m128i cmpgt_epu64(__m128i a, __m128i b) { return _mm_comgt_epu64(a, b); }
#else
Vc_INTRINSIC __m128i Vc_CONST cmplt_epu8(__m128i a, __m128i b)
{
    return _mm_cmplt_epi8(_mm_xor_si128(a, lowest16<schar>()),
                          _mm_xor_si128(b, lowest16<schar>()));
}
Vc_INTRINSIC __m128i Vc_CONST cmpgt_epu8(__m128i a, __m128i b)
{
    return _mm_cmpgt_epi8(_mm_xor_si128(a, lowest16<schar>()),
                          _mm_xor_si128(b, lowest16<schar>()));
}
Vc_INTRINSIC __m128i Vc_CONST cmplt_epu16(__m128i a, __m128i b)
{
    return _mm_cmplt_epi16(_mm_xor_si128(a, lowest16<short>()),
                           _mm_xor_si128(b, lowest16<short>()));
}
Vc_INTRINSIC __m128i Vc_CONST cmpgt_epu16(__m128i a, __m128i b)
{
    return _mm_cmpgt_epi16(_mm_xor_si128(a, lowest16<short>()),
                           _mm_xor_si128(b, lowest16<short>()));
}
Vc_INTRINSIC __m128i Vc_CONST cmplt_epu32(__m128i a, __m128i b)
{
    return _mm_cmplt_epi32(_mm_xor_si128(a, lowest16<int>()),
                           _mm_xor_si128(b, lowest16<int>()));
}
Vc_INTRINSIC __m128i Vc_CONST cmpgt_epu32(__m128i a, __m128i b)
{
    return _mm_cmpgt_epi32(_mm_xor_si128(a, lowest16<int>()),
                           _mm_xor_si128(b, lowest16<int>()));
}
Vc_INTRINSIC __m128i Vc_CONST cmpgt_epi64(__m128i a, __m128i b)
{
#ifdef Vc_IMPL_SSE4_2
    return _mm_cmpgt_epi64(a, b);
#else
    const auto aa = _mm_xor_si128(a, _mm_srli_epi64(lowest16<int>(), 32));
    const auto bb = _mm_xor_si128(b, _mm_srli_epi64(lowest16<int>(), 32));
    const auto gt = _mm_cmpgt_epi32(aa, bb);
    const auto eq = _mm_cmpeq_epi32(aa, bb);
    // Algorithm:
    // 1. if the high 32 bits of gt are true, make the full 64 bits true
    // 2. if the high 32 bits of gt are false and the high 32 bits of eq are true,
    //    duplicate the low 32 bits of gt to the high 32 bits (note that this requires
    //    unsigned compare on the lower 32 bits, which is the reason for the xors
    //    above)
    // 3. else make the full 64 bits false

    const auto gt2 =
        _mm_shuffle_epi32(gt, 0xf5);  // dup the high 32 bits to the low 32 bits
    const auto lo = _mm_shuffle_epi32(_mm_and_si128(_mm_srli_epi64(eq, 32), gt), 0xa0);
    return _mm_or_si128(gt2, lo);
#endif
}
Vc_INTRINSIC __m128i Vc_CONST cmpgt_epu64(__m128i a, __m128i b)
{
    return cmpgt_epi64(_mm_xor_si128(a, lowest16<llong>()),
                       _mm_xor_si128(b, lowest16<llong>()));
}
#endif

Vc_INTRINSIC Vc_CONST __m128i abs_epi8(__m128i a) {
#ifdef Vc_HAVE_SSSE3
    return _mm_abs_epi8(a);
#else
    __m128i negative = _mm_cmplt_epi8(a, _mm_setzero_si128());
    return _mm_add_epi8(_mm_xor_si128(a, negative),
                        _mm_and_si128(negative, setone_epi8()));
#endif
}

Vc_INTRINSIC Vc_CONST __m128i abs_epi16(__m128i a) {
#ifdef Vc_HAVE_SSSE3
    return _mm_abs_epi16(a);
#else
    __m128i negative = _mm_cmplt_epi16(a, _mm_setzero_si128());
    return _mm_add_epi16(_mm_xor_si128(a, negative), _mm_srli_epi16(negative, 15));
#endif
}

Vc_INTRINSIC Vc_CONST __m128i abs_epi32(__m128i a) {
#ifdef Vc_HAVE_SSSE3
    return _mm_abs_epi32(a);
#else
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
    __m128i negative = _mm_cmplt_epi32(a, _mm_setzero_si128());
    return _mm_add_epi32(_mm_xor_si128(a, negative), _mm_srli_epi32(negative, 31));
#endif
}

template <int s> Vc_INTRINSIC Vc_CONST __m128i alignr(__m128i a, __m128i b)
{
#ifdef Vc_HAVE_SSSE3
    return _mm_alignr_epi8(a, b, s & 0x1fu);
#else
    switch (s & 0x1fu) {
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
#endif
}
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
template <int shift> Vc_INTRINSIC Vc_CONST __m256i alignr(__m256i s1, __m256i s2)
{
#ifdef Vc_HAVE_AVX2
    return _mm256_alignr_epi8(s1, s2, shift);
#else
    return insert128<1>(
        _mm256_castsi128_si256(_mm_alignr_epi8(_mm256_castsi256_si128(s1),
                                               _mm256_castsi256_si128(s2), shift)),
        _mm_alignr_epi8(extract128<1>(s1), extract128<1>(s2), shift));
#endif  // Vc_HAVE_AVX2
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_SSE4_1
Vc_INTRINSIC Vc_CONST __m128i cmpeq_epi64(__m128i a, __m128i b)
{
    return _mm_cmpeq_epi64(a, b);
}
template <int index> Vc_INTRINSIC Vc_CONST int extract_epi32(__m128i v)
{
    return _mm_extract_epi32(v, index);
}
Vc_INTRINSIC Vc_CONST __m128d blendv_pd(__m128d a, __m128d b, __m128d c)
{
    return _mm_blendv_pd(a, b, c);
}
Vc_INTRINSIC Vc_CONST __m128 blendv_ps(__m128 a, __m128 b, __m128 c)
{
    return _mm_blendv_ps(a, b, c);
}
Vc_INTRINSIC Vc_CONST __m128i blendv_epi8(__m128i a, __m128i b, __m128i c)
{
    return _mm_blendv_epi8(a, b, c);
}
Vc_INTRINSIC Vc_CONST __m128i max_epi8(__m128i a, __m128i b)
{
    return _mm_max_epi8(a, b);
}
Vc_INTRINSIC Vc_CONST __m128i max_epi32(__m128i a, __m128i b)
{
    return _mm_max_epi32(a, b);
}
Vc_INTRINSIC Vc_CONST __m128i max_epu16(__m128i a, __m128i b)
{
    return _mm_max_epu16(a, b);
}
Vc_INTRINSIC Vc_CONST __m128i max_epu32(__m128i a, __m128i b)
{
    return _mm_max_epu32(a, b);
}
Vc_INTRINSIC Vc_CONST __m128i min_epu16(__m128i a, __m128i b)
{
    return _mm_min_epu16(a, b);
}
Vc_INTRINSIC Vc_CONST __m128i min_epu32(__m128i a, __m128i b)
{
    return _mm_min_epu32(a, b);
}
Vc_INTRINSIC Vc_CONST __m128i min_epi8(__m128i a, __m128i b)
{
    return _mm_min_epi8(a, b);
}
Vc_INTRINSIC Vc_CONST __m128i min_epi32(__m128i a, __m128i b)
{
    return _mm_min_epi32(a, b);
}
Vc_INTRINSIC Vc_CONST __m128i cvtepu8_epi16(__m128i epu8)
{
    return _mm_cvtepu8_epi16(epu8);
}
Vc_INTRINSIC Vc_CONST __m128i cvtepi8_epi16(__m128i epi8)
{
    return _mm_cvtepi8_epi16(epi8);
}
Vc_INTRINSIC Vc_CONST __m128i cvtepu16_epi32(__m128i epu16)
{
    return _mm_cvtepu16_epi32(epu16);
}
Vc_INTRINSIC Vc_CONST __m128i cvtepi16_epi32(__m128i epu16)
{
    return _mm_cvtepi16_epi32(epu16);
}
Vc_INTRINSIC Vc_CONST __m128i cvtepu8_epi32(__m128i epu8)
{
    return _mm_cvtepu8_epi32(epu8);
}
Vc_INTRINSIC Vc_CONST __m128i cvtepi8_epi32(__m128i epi8)
{
    return _mm_cvtepi8_epi32(epi8);
}
Vc_INTRINSIC Vc_PURE __m128i stream_load_si128(__m128i *mem)
{
    return _mm_stream_load_si128(mem);
}
#else  // Vc_HAVE_SSE4_1
Vc_INTRINSIC Vc_CONST __m128  blendv_ps(__m128  a, __m128  b, __m128  c) {
    return _mm_or_ps(_mm_andnot_ps(c, a), _mm_and_ps(c, b));
}

#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC Vc_CONST __m128d blendv_pd(__m128d a, __m128d b, __m128d c) {
    return _mm_or_pd(_mm_andnot_pd(c, a), _mm_and_pd(c, b));
}
Vc_INTRINSIC Vc_CONST __m128i blendv_epi8(__m128i a, __m128i b, __m128i c) {
    return _mm_or_si128(_mm_andnot_si128(c, a), _mm_and_si128(c, b));
}

Vc_INTRINSIC Vc_CONST __m128i cmpeq_epi64(__m128i a, __m128i b) {
    auto tmp = _mm_cmpeq_epi32(a, b);
    return _mm_and_si128(tmp, _mm_shuffle_epi32(tmp, 1*1 + 0*4 + 3*16 + 2*64));
}
template <int index> Vc_INTRINSIC Vc_CONST int extract_epi32(__m128i v)
{
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
    typedef int int32v4 __attribute__((__vector_size__(16)));
    return reinterpret_cast<const may_alias<int32v4> &>(v)[index];
#else
    return _mm_cvtsi128_si32(_mm_srli_si128(v, index * 4));
#endif
}

Vc_INTRINSIC Vc_CONST __m128i max_epi8 (__m128i a, __m128i b) {
    return blendv_epi8(b, a, _mm_cmpgt_epi8 (a, b));
}
Vc_INTRINSIC Vc_CONST __m128i max_epi32(__m128i a, __m128i b) {
    return blendv_epi8(b, a, _mm_cmpgt_epi32(a, b));
}
//X         Vc_INTRINSIC Vc_CONST __m128i max_epu8 (__m128i a, __m128i b) {
//X             return _mm_blendv_epi8(b, a, cmpgt_epu8 (a, b));
//X         }
Vc_INTRINSIC Vc_CONST __m128i max_epu16(__m128i a, __m128i b) {
    return blendv_epi8(b, a, cmpgt_epu16(a, b));
}
Vc_INTRINSIC Vc_CONST __m128i max_epu32(__m128i a, __m128i b) {
    return blendv_epi8(b, a, cmpgt_epu32(a, b));
}
//X         Vc_INTRINSIC Vc_CONST __m128i _mm_min_epu8 (__m128i a, __m128i b) {
//X             return _mm_blendv_epi8(a, b, cmpgt_epu8 (a, b));
//X         }
Vc_INTRINSIC Vc_CONST __m128i min_epu16(__m128i a, __m128i b) {
    return blendv_epi8(a, b, cmpgt_epu16(a, b));
}
Vc_INTRINSIC Vc_CONST __m128i min_epu32(__m128i a, __m128i b) {
    return blendv_epi8(a, b, cmpgt_epu32(a, b));
}
Vc_INTRINSIC Vc_CONST __m128i min_epi8 (__m128i a, __m128i b) {
    return blendv_epi8(a, b, _mm_cmpgt_epi8 (a, b));
}
Vc_INTRINSIC Vc_CONST __m128i min_epi32(__m128i a, __m128i b) {
    return blendv_epi8(a, b, _mm_cmpgt_epi32(a, b));
}
Vc_INTRINSIC Vc_CONST __m128i cvtepu8_epi16(__m128i epu8) {
    return _mm_unpacklo_epi8(epu8, _mm_setzero_si128());
}
Vc_INTRINSIC Vc_CONST __m128i cvtepi8_epi16(__m128i epi8) {
    return _mm_unpacklo_epi8(epi8, _mm_cmplt_epi8(epi8, _mm_setzero_si128()));
}
Vc_INTRINSIC Vc_CONST __m128i cvtepu16_epi32(__m128i epu16) {
    return _mm_unpacklo_epi16(epu16, _mm_setzero_si128());
}
Vc_INTRINSIC Vc_CONST __m128i cvtepi16_epi32(__m128i epu16) {
    return _mm_unpacklo_epi16(epu16, _mm_cmplt_epi16(epu16, _mm_setzero_si128()));
}
Vc_INTRINSIC Vc_CONST __m128i cvtepu8_epi32(__m128i epu8) {
    return cvtepu16_epi32(cvtepu8_epi16(epu8));
}
Vc_INTRINSIC Vc_CONST __m128i cvtepi8_epi32(__m128i epi8) {
    const __m128i neg = _mm_cmplt_epi8(epi8, _mm_setzero_si128());
    const __m128i epi16 = _mm_unpacklo_epi8(epi8, neg);
    return _mm_unpacklo_epi16(epi16, _mm_unpacklo_epi8(neg, neg));
}
Vc_INTRINSIC Vc_PURE __m128i stream_load_si128(__m128i *mem) {
    return _mm_load_si128(mem);
}
#endif  // Vc_HAVE_SSE2
#endif  // Vc_HAVE_SSE4_1

// blend{{{1
Vc_INTRINSIC Vc_CONST __m128 blend(__m128 mask, __m128 at0, __m128 at1)
{
    return blendv_ps(at0, at1, mask);
}

#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC Vc_CONST __m128d blend(__m128d mask, __m128d at0, __m128d at1)
{
    return blendv_pd(at0, at1, mask);
}
Vc_INTRINSIC Vc_CONST __m128i blend(__m128i mask, __m128i at0, __m128i at1)
{
    return blendv_epi8(at0, at1, mask);
}
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC Vc_CONST __m256  blend(__m256  mask, __m256  at0, __m256  at1)
{
    return _mm256_blendv_ps(at0, at1, mask);
}
Vc_INTRINSIC Vc_CONST __m256d blend(__m256d mask, __m256d at0, __m256d at1)
{
    return _mm256_blendv_pd(at0, at1, mask);
}
#ifdef Vc_HAVE_AVX2
Vc_INTRINSIC Vc_CONST __m256i blend(__m256i mask, __m256i at0, __m256i at1)
{
    return _mm256_blendv_epi8(at0, at1, mask);
}
#endif  // Vc_HAVE_AVX2
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC Vc_CONST __mmask8 blend(__mmask8 mask, __mmask8 at0, __mmask8 at1)
{
    return (mask & at1) | (~mask & at0);
}
Vc_INTRINSIC Vc_CONST __mmask16 blend(__mmask16 mask, __mmask16 at0, __mmask16 at1)
{
    return _mm512_kor(_mm512_kand(mask, at1), _mm512_kandn(mask, at0));
}
Vc_INTRINSIC Vc_CONST __m512  blend(__mmask16 mask, __m512 at0, __m512 at1)
{
    return _mm512_mask_mov_ps(at0, mask, at1);
}
Vc_INTRINSIC Vc_CONST __m512d blend(__mmask8 mask, __m512d at0, __m512d at1)
{
    return _mm512_mask_mov_pd(at0, mask, at1);
}
Vc_INTRINSIC Vc_CONST __m512i blend(__mmask8 mask, __m512i at0, __m512i at1)
{
    return _mm512_mask_mov_epi64(at0, mask, at1);
}
Vc_INTRINSIC Vc_CONST __m512i blend(__mmask16 mask, __m512i at0, __m512i at1)
{
    return _mm512_mask_mov_epi32(at0, mask, at1);
}
#ifdef Vc_HAVE_AVX512BW
Vc_INTRINSIC Vc_CONST __mmask32 blend(__mmask32 mask, __mmask32 at0, __mmask32 at1)
{
    return (mask & at1) | (~mask & at0);
}
Vc_INTRINSIC Vc_CONST __mmask64 blend(__mmask64 mask, __mmask64 at0, __mmask64 at1)
{
    return (mask & at1) | (~mask & at0);
}
Vc_INTRINSIC Vc_CONST __m512i blend(__mmask32 mask, __m512i at0, __m512i at1)
{
    return _mm512_mask_mov_epi16(at0, mask, at1);
}
Vc_INTRINSIC Vc_CONST __m512i blend(__mmask64 mask, __m512i at0, __m512i at1)
{
    return _mm512_mask_mov_epi8(at0, mask, at1);
}
#endif  // Vc_HAVE_AVX512BW
#endif  // Vc_HAVE_AVX512F

// testc{{{1
#ifdef Vc_HAVE_SSE4_1
Vc_INTRINSIC Vc_CONST int testc(__m128  a, __m128  b) { return _mm_testc_si128(_mm_castps_si128(a), _mm_castps_si128(b)); }
Vc_INTRINSIC Vc_CONST int testc(__m128d a, __m128d b) { return _mm_testc_si128(_mm_castpd_si128(a), _mm_castpd_si128(b)); }
Vc_INTRINSIC Vc_CONST int testc(__m128i a, __m128i b) { return _mm_testc_si128(a, b); }
#endif  // Vc_HAVE_SSE4_1

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC Vc_CONST int testc(__m256  a, __m256  b) { return _mm256_testc_ps(a, b); }
Vc_INTRINSIC Vc_CONST int testc(__m256d a, __m256d b) { return _mm256_testc_pd(a, b); }
Vc_INTRINSIC Vc_CONST int testc(__m256i a, __m256i b) { return _mm256_testc_si256(a, b); }
#endif  // Vc_HAVE_AVX

// testallset{{{1
#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC Vc_CONST bool testallset(__mmask8 a) { return a == 0xffU; }

Vc_INTRINSIC Vc_CONST bool testallset(__mmask16 a)
{
#ifdef Vc_GCC
    // GCC ICEs on _mm512_kortestc
    return a == 0xffffU;
#else
    return _mm512_kortestc(a, a);
#endif
}

#ifdef Vc_HAVE_AVX512BW
Vc_INTRINSIC Vc_CONST bool testallset(__mmask32 a) { return a == 0xffffffffU; }
// return _mm512_kortestc(a, a) && _mm512_kortestc(a >> 16, a >> 16); }
Vc_INTRINSIC Vc_CONST bool testallset(__mmask64 a) { return a == 0xffffffffffffffffULL; }
/*{
    return _mm512_kortestc(a, a) && _mm512_kortestc(a >> 16, a >> 16) &&
           _mm512_kortestc(a >> 32, a >> 32) && _mm512_kortestc(a >> 48, a >> 48);
}*/
#endif  // Vc_HAVE_AVX512BW
#endif  // Vc_HAVE_AVX512F

// testz{{{1
#ifdef Vc_HAVE_SSE4_1
Vc_INTRINSIC Vc_CONST int testz(__m128  a, __m128  b) { return _mm_testz_si128(_mm_castps_si128(a), _mm_castps_si128(b)); }
Vc_INTRINSIC Vc_CONST int testz(__m128d a, __m128d b) { return _mm_testz_si128(_mm_castpd_si128(a), _mm_castpd_si128(b)); }
Vc_INTRINSIC Vc_CONST int testz(__m128i a, __m128i b) { return _mm_testz_si128(a, b); }
#endif  // Vc_HAVE_SSE4_1
#ifdef Vc_HAVE_AVX
Vc_INTRINSIC Vc_CONST int testz(__m256  a, __m256  b) { return _mm256_testz_ps(a, b); }
Vc_INTRINSIC Vc_CONST int testz(__m256d a, __m256d b) { return _mm256_testz_pd(a, b); }
Vc_INTRINSIC Vc_CONST int testz(__m256i a, __m256i b) { return _mm256_testz_si256(a, b); }
#endif  // Vc_HAVE_AVX

// testnzc{{{1
#ifdef Vc_HAVE_SSE4_1
Vc_INTRINSIC Vc_CONST int testnzc(__m128  a, __m128  b) { return _mm_testnzc_si128(_mm_castps_si128(a), _mm_castps_si128(b)); }
Vc_INTRINSIC Vc_CONST int testnzc(__m128d a, __m128d b) { return _mm_testnzc_si128(_mm_castpd_si128(a), _mm_castpd_si128(b)); }
Vc_INTRINSIC Vc_CONST int testnzc(__m128i a, __m128i b) { return _mm_testnzc_si128(a, b); }
#endif  // Vc_HAVE_SSE4_1
#ifdef Vc_HAVE_AVX
Vc_INTRINSIC Vc_CONST int testnzc(__m256  a, __m256  b) { return _mm256_testnzc_ps(a, b); }
Vc_INTRINSIC Vc_CONST int testnzc(__m256d a, __m256d b) { return _mm256_testnzc_pd(a, b); }
Vc_INTRINSIC Vc_CONST int testnzc(__m256i a, __m256i b) { return _mm256_testnzc_si256(a, b); }
#endif  // Vc_HAVE_AVX

// movemask{{{1
Vc_INTRINSIC Vc_CONST int movemask(__m128  a) { return _mm_movemask_ps(a); }
#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC Vc_CONST int movemask(__m128d a) { return _mm_movemask_pd(a); }
Vc_INTRINSIC Vc_CONST int movemask(__m128i a) { return _mm_movemask_epi8(a); }
#endif  // Vc_HAVE_SSE2
#ifdef Vc_HAVE_AVX
Vc_INTRINSIC Vc_CONST int movemask(__m256i a) {
#ifdef Vc_HAVE_AVX2
    return _mm256_movemask_epi8(a);
#else
    return _mm_movemask_epi8(lo128(a)) | (_mm_movemask_epi8(hi128(a)) << 16);
#endif
}
Vc_INTRINSIC Vc_CONST int movemask(__m256d a) { return _mm256_movemask_pd(a); }
Vc_INTRINSIC Vc_CONST int movemask(__m256  a) { return _mm256_movemask_ps(a); }
#endif  // Vc_HAVE_AVX

// AVX512: convert_mask{{{1
#ifdef Vc_HAVE_AVX512F
template <size_t VectorSize> struct convert_mask_return_type;
template <> struct convert_mask_return_type<16> { using type = __m128i; };
template <> struct convert_mask_return_type<32> { using type = __m256i; };
template <> struct convert_mask_return_type<64> { using type = __m512i; };

template <size_t EntrySize, size_t VectorSize>
inline typename convert_mask_return_type<VectorSize>::type convert_mask(__mmask8);
template <size_t EntrySize, size_t VectorSize>
inline typename convert_mask_return_type<VectorSize>::type convert_mask(__mmask16);
template <size_t EntrySize, size_t VectorSize>
inline typename convert_mask_return_type<VectorSize>::type convert_mask(__mmask32);
template <size_t EntrySize, size_t VectorSize>
inline typename convert_mask_return_type<VectorSize>::type convert_mask(__mmask64);

#ifdef Vc_HAVE_AVX512VL
#ifdef Vc_HAVE_AVX512BW
template <> Vc_INTRINSIC __m128i convert_mask<1, 16>(__mmask16 k) { return _mm_movm_epi8 (k); }
template <> Vc_INTRINSIC __m128i convert_mask<2, 16>(__mmask8  k) { return _mm_movm_epi16(k); }
template <> Vc_INTRINSIC __m256i convert_mask<1, 32>(__mmask32 k) { return _mm256_movm_epi8 (k); }
template <> Vc_INTRINSIC __m256i convert_mask<2, 32>(__mmask16 k) { return _mm256_movm_epi16(k); }
#endif  // Vc_HAVE_AVX512BW

#ifdef Vc_HAVE_AVX512DQ
template <> Vc_INTRINSIC __m128i convert_mask<4, 16>(__mmask8  k) { return _mm_movm_epi32(k); }
template <> Vc_INTRINSIC __m128i convert_mask<8, 16>(__mmask8  k) { return _mm_movm_epi64(k); }
template <> Vc_INTRINSIC __m256i convert_mask<4, 32>(__mmask8  k) { return _mm256_movm_epi32(k); }
template <> Vc_INTRINSIC __m256i convert_mask<8, 32>(__mmask8  k) { return _mm256_movm_epi64(k); }
#endif  // Vc_HAVE_AVX512DQ
#endif  // Vc_HAVE_AVX512VL

#ifdef Vc_HAVE_AVX512BW
template <> Vc_INTRINSIC __m512i convert_mask<1, 64>(__mmask64 k) { return _mm512_movm_epi8 (k); }
template <> Vc_INTRINSIC __m512i convert_mask<2, 64>(__mmask32 k) { return _mm512_movm_epi16(k); }
#endif  // Vc_HAVE_AVX512BW

#ifdef Vc_HAVE_AVX512DQ
template <> Vc_INTRINSIC __m512i convert_mask<4, 64>(__mmask16 k) { return _mm512_movm_epi32(k); }
template <> Vc_INTRINSIC __m512i convert_mask<8, 64>(__mmask8  k) { return _mm512_movm_epi64(k); }
#endif  // Vc_HAVE_AVX512DQ
#endif  // Vc_HAVE_AVX512F

// negate{{{1
Vc_ALWAYS_INLINE Vc_CONST __m128 negate(__m128 v, std::integral_constant<std::size_t, 4>)
{
    return _mm_xor_ps(v, signmask16(float()));
}
#ifdef Vc_HAVE_SSE2
Vc_ALWAYS_INLINE Vc_CONST __m128d negate(__m128d v, std::integral_constant<std::size_t, 8>)
{
    return _mm_xor_pd(v, signmask16(double()));
}
Vc_ALWAYS_INLINE Vc_CONST __m128i negate(__m128i v, std::integral_constant<std::size_t, 4>)
{
#ifdef Vc_IMPL_SSSE3
    return _mm_sign_epi32(v, allone<__m128i>());
#else
    return _mm_sub_epi32(zero<__m128i>(), v);
#endif
}
Vc_ALWAYS_INLINE Vc_CONST __m128i negate(__m128i v, std::integral_constant<std::size_t, 2>)
{
#ifdef Vc_IMPL_SSSE3
    return _mm_sign_epi16(v, allone<__m128i>());
#else
    return _mm_sub_epi16(zero<__m128i>(), v);
#endif
}
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
Vc_ALWAYS_INLINE Vc_CONST __m256 negate(__m256 v, std::integral_constant<std::size_t, 4>)
{
    return _mm256_xor_ps(v, signmask32(float()));
}
Vc_ALWAYS_INLINE Vc_CONST __m256d negate(__m256d v, std::integral_constant<std::size_t, 8>)
{
    return _mm256_xor_pd(v, signmask32(double()));
}
#ifdef Vc_HAVE_AVX2
Vc_ALWAYS_INLINE Vc_CONST __m256i negate(__m256i v, std::integral_constant<std::size_t, 4>)
{
    return _mm256_sign_epi32(v, allone<__m256i>());
}
Vc_ALWAYS_INLINE Vc_CONST __m256i negate(__m256i v, std::integral_constant<std::size_t, 2>)
{
    return _mm256_sign_epi16(v, allone<__m256i>());
}
Vc_ALWAYS_INLINE Vc_CONST __m256i negate(__m256i v, std::integral_constant<std::size_t, 1>)
{
    return _mm256_sign_epi8(v, allone<__m256i>());
}
#endif  // Vc_HAVE_AVX2
#endif  // Vc_HAVE_AVX

// xor_{{{1
Vc_INTRINSIC __m128  xor_(__m128  a, __m128  b) { return _mm_xor_ps(a, b); }
#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC __m128d xor_(__m128d a, __m128d b) { return _mm_xor_pd(a, b); }
Vc_INTRINSIC __m128i xor_(__m128i a, __m128i b) { return _mm_xor_si128(a, b); }
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC __m256  xor_(__m256  a, __m256  b) { return _mm256_xor_ps(a, b); }
Vc_INTRINSIC __m256d xor_(__m256d a, __m256d b) { return _mm256_xor_pd(a, b); }
Vc_INTRINSIC __m256i xor_(__m256i a, __m256i b) {
#ifdef Vc_HAVE_AVX2
    return _mm256_xor_si256(a, b);
#else
    return _mm256_castps_si256(xor_(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
#endif
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
#ifdef Vc_HAVE_AVX512DQ
Vc_INTRINSIC __m512  xor_(__m512  a, __m512  b) { return _mm512_xor_ps(a, b); }
Vc_INTRINSIC __m512d xor_(__m512d a, __m512d b) { return _mm512_xor_pd(a, b); }
#else   // Vc_HAVE_AVX512DQ
Vc_INTRINSIC __m512 xor_(__m512 a, __m512 b)
{
    return intrin_cast<__m512>(
        _mm512_xor_epi32(intrin_cast<__m512i>(a), intrin_cast<__m512i>(b)));
}
Vc_INTRINSIC __m512d xor_(__m512d a, __m512d b)
{
    return intrin_cast<__m512d>(
        _mm512_xor_epi64(intrin_cast<__m512i>(a), intrin_cast<__m512i>(b)));
}
#endif  // Vc_HAVE_AVX512DQ
Vc_INTRINSIC __m512i xor_(__m512i a, __m512i b) { return _mm512_xor_epi32(a, b); }
#endif  // Vc_HAVE_AVX512F

// or_{{{1
Vc_INTRINSIC __m128 or_(__m128 a, __m128 b) { return _mm_or_ps(a, b); }
#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC __m128d or_(__m128d a, __m128d b) { return _mm_or_pd(a, b); }
Vc_INTRINSIC __m128i or_(__m128i a, __m128i b) { return _mm_or_si128(a, b); }
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC __m256  or_(__m256  a, __m256  b) { return _mm256_or_ps(a, b); }
Vc_INTRINSIC __m256d or_(__m256d a, __m256d b) { return _mm256_or_pd(a, b); }
Vc_INTRINSIC __m256i or_(__m256i a, __m256i b) {
#ifdef Vc_HAVE_AVX2
    return _mm256_or_si256(a, b);
#else
    return _mm256_castps_si256(or_(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
#endif
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC __m512  or_(__m512  a, __m512  b) { return _mm512_or_ps(a, b); }
Vc_INTRINSIC __m512d or_(__m512d a, __m512d b) { return _mm512_or_pd(a, b); }
Vc_INTRINSIC __m512i or_(__m512i a, __m512i b) { return _mm512_or_epi32(a, b); }
#endif  // Vc_HAVE_AVX512F

// and_{{{1
Vc_INTRINSIC __m128 and_(__m128 a, __m128 b) { return _mm_and_ps(a, b); }
#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC __m128d and_(__m128d a, __m128d b) { return _mm_and_pd(a, b); }
Vc_INTRINSIC __m128i and_(__m128i a, __m128i b) { return _mm_and_si128(a, b); }
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC __m256  and_(__m256  a, __m256  b) { return _mm256_and_ps(a, b); }
Vc_INTRINSIC __m256d and_(__m256d a, __m256d b) { return _mm256_and_pd(a, b); }
Vc_INTRINSIC __m256i and_(__m256i a, __m256i b) {
#ifdef Vc_HAVE_AVX2
    return _mm256_and_si256(a, b);
#else
    return _mm256_castps_si256(and_(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
#endif
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC __m512  and_(__m512  a, __m512  b) { return _mm512_and_ps(a, b); }
Vc_INTRINSIC __m512d and_(__m512d a, __m512d b) { return _mm512_and_pd(a, b); }
Vc_INTRINSIC __m512i and_(__m512i a, __m512i b) { return _mm512_and_epi32(a, b); }
#endif  // Vc_HAVE_AVX512F

// andnot_{{{1
Vc_INTRINSIC __m128 andnot_(__m128 a, __m128 b) { return _mm_andnot_ps(a, b); }
#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC __m128d andnot_(__m128d a, __m128d b) { return _mm_andnot_pd(a, b); }
Vc_INTRINSIC __m128i andnot_(__m128i a, __m128i b) { return _mm_andnot_si128(a, b); }
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC __m256  andnot_(__m256  a, __m256  b) { return _mm256_andnot_ps(a, b); }
Vc_INTRINSIC __m256d andnot_(__m256d a, __m256d b) { return _mm256_andnot_pd(a, b); }
Vc_INTRINSIC __m256i andnot_(__m256i a, __m256i b) {
#ifdef Vc_HAVE_AVX2
    return _mm256_andnot_si256(a, b);
#else
    return _mm256_castps_si256(andnot_(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
#endif
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC __m512  andnot_(__m512  a, __m512  b) { return _mm512_andnot_ps(a, b); }
Vc_INTRINSIC __m512d andnot_(__m512d a, __m512d b) { return _mm512_andnot_pd(a, b); }
Vc_INTRINSIC __m512i andnot_(__m512i a, __m512i b) { return _mm512_andnot_epi32(a, b); }
#endif  // Vc_HAVE_AVX512F

// not_{{{1
Vc_INTRINSIC __m128  not_(__m128  a) { return andnot_(a, allone<__m128 >()); }
#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC __m128d not_(__m128d a) { return andnot_(a, allone<__m128d>()); }
Vc_INTRINSIC __m128i not_(__m128i a) { return andnot_(a, allone<__m128i>()); }
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC __m256  not_(__m256  a) { return andnot_(a, allone<__m256 >()); }
Vc_INTRINSIC __m256d not_(__m256d a) { return andnot_(a, allone<__m256d>()); }
Vc_INTRINSIC __m256i not_(__m256i a) { return andnot_(a, allone<__m256i>()); }
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC __m512  not_(__m512  a) { return andnot_(a, allone<__m512 >()); }
Vc_INTRINSIC __m512d not_(__m512d a) { return andnot_(a, allone<__m512d>()); }
Vc_INTRINSIC __m512i not_(__m512i a) { return andnot_(a, allone<__m512i>()); }

Vc_INTRINSIC __mmask8  not_(__mmask8  a) { return ~a; }
Vc_INTRINSIC __mmask16 not_(__mmask16 a) { return ~a; }
#ifdef Vc_HAVE_AVX512BW
Vc_INTRINSIC __mmask32 not_(__mmask32 a) { return ~a; }
Vc_INTRINSIC __mmask64 not_(__mmask64 a) { return ~a; }
#endif  // Vc_HAVE_AVX512BW
#endif  // Vc_HAVE_AVX512F

// shift_left{{{1
#ifdef Vc_HAVE_SSE2
template <int n> Vc_INTRINSIC __m128  shift_left(__m128  v) { return _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(v), n)); }
template <int n> Vc_INTRINSIC __m128d shift_left(__m128d v) { return _mm_castsi128_pd(_mm_slli_si128(_mm_castpd_si128(v), n)); }
template <int n> Vc_INTRINSIC __m128i shift_left(__m128i v) { return _mm_slli_si128(v, n); }
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX2
template <int n> Vc_INTRINSIC __m256 shift_left(__m256 v)
{
    __m256i vi = _mm256_castps_si256(v);
    return _mm256_castsi256_ps(
        n < 16 ? _mm256_slli_si256(vi, n)
               : _mm256_slli_si256(_mm256_permute2x128_si256(vi, vi, 0x08), n));
}
template <int n> Vc_INTRINSIC __m256d shift_left(__m256d v)
{
    __m256i vi = _mm256_castpd_si256(v);
    return _mm256_castsi256_pd(
        n < 16 ? _mm256_slli_si256(vi, n)
               : _mm256_slli_si256(_mm256_permute2x128_si256(vi, vi, 0x08), n));
}
template <int n> Vc_INTRINSIC __m256i shift_left(__m256i v)
{
    return _mm256_castsi256_pd(
        n < 16 ? _mm256_slli_si256(v, n)
               : _mm256_slli_si256(_mm256_permute2x128_si256(v, v, 0x08), n));
}
#endif

// shift_right{{{1
template <int n> Vc_INTRINSIC __m128  shift_right(__m128  v);
template <> Vc_INTRINSIC __m128  shift_right< 0>(__m128  v) { return v; }
template <> Vc_INTRINSIC __m128  shift_right<16>(__m128   ) { return _mm_setzero_ps(); }

#ifdef Vc_HAVE_SSE2
template <int n> Vc_INTRINSIC __m128  shift_right(__m128  v) { return _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v), n)); }
template <int n> Vc_INTRINSIC __m128d shift_right(__m128d v) { return _mm_castsi128_pd(_mm_srli_si128(_mm_castpd_si128(v), n)); }
template <int n> Vc_INTRINSIC __m128i shift_right(__m128i v) { return _mm_srli_si128(v, n); }

template <> Vc_INTRINSIC __m128  shift_right< 8>(__m128  v) { return _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(v), _mm_setzero_pd())); }
template <> Vc_INTRINSIC __m128d shift_right< 0>(__m128d v) { return v; }
template <> Vc_INTRINSIC __m128d shift_right< 8>(__m128d v) { return _mm_unpackhi_pd(v, _mm_setzero_pd()); }
template <> Vc_INTRINSIC __m128d shift_right<16>(__m128d  ) { return _mm_setzero_pd(); }
template <> Vc_INTRINSIC __m128i shift_right< 0>(__m128i v) { return v; }
template <> Vc_INTRINSIC __m128i shift_right<16>(__m128i  ) { return _mm_setzero_si128(); }
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX2
template <int n> Vc_INTRINSIC __m256 shift_right(__m256 v)
{
    __m256i vi = _mm256_castps_si256(v);
    return _mm256_castsi256_ps(
        n < 16 ? _mm256_srli_si256(vi, n)
               : _mm256_srli_si256(_mm256_permute2x128_si256(vi, vi, 0x81), n));
}
template <> Vc_INTRINSIC __m256 shift_right<0>(__m256 v) { return v; }
template <> Vc_INTRINSIC __m256 shift_right<16>(__m256 v) { return intrin_cast<__m256>(lo128(v)); }
template <int n> Vc_INTRINSIC __m256d shift_right(__m256d v)
{
    __m256i vi = _mm256_castpd_si256(v);
    return _mm256_castsi256_pd(
        n < 16 ? _mm256_srli_si256(vi, n)
               : _mm256_srli_si256(_mm256_permute2x128_si256(vi, vi, 0x81), n));
}
template <> Vc_INTRINSIC __m256d shift_right<0>(__m256d v) { return v; }
template <> Vc_INTRINSIC __m256d shift_right<16>(__m256d v) { return intrin_cast<__m256d>(lo128(v)); }
template <int n> Vc_INTRINSIC __m256i shift_right(__m256i v)
{
    return n < 16 ? _mm256_srli_si256(v, n)
                  : _mm256_srli_si256(_mm256_permute2x128_si256(v, v, 0x81), n);
}
template <> Vc_INTRINSIC __m256i shift_right<0>(__m256i v) { return v; }
template <> Vc_INTRINSIC __m256i shift_right<16>(__m256i v) { return _mm256_permute2x128_si256(v, v, 0x81); }
#endif

// popcnt{{{1
Vc_INTRINSIC Vc_CONST unsigned int popcnt4(unsigned int n)
{
#ifdef Vc_IMPL_POPCNT
    return _mm_popcnt_u32(n);
#else
    n = (n & 0x5U) + ((n >> 1) & 0x5U);
    n = (n & 0x3U) + ((n >> 2) & 0x3U);
    return n;
#endif
}
Vc_INTRINSIC Vc_CONST unsigned int popcnt8(unsigned int n)
{
#ifdef Vc_IMPL_POPCNT
    return _mm_popcnt_u32(n);
#else
    n = (n & 0x55U) + ((n >> 1) & 0x55U);
    n = (n & 0x33U) + ((n >> 2) & 0x33U);
    n = (n & 0x0fU) + ((n >> 4) & 0x0fU);
    return n;
#endif
}
Vc_INTRINSIC Vc_CONST unsigned int popcnt16(unsigned int n)
{
#ifdef Vc_IMPL_POPCNT
    return _mm_popcnt_u32(n);
#else
    n = (n & 0x5555U) + ((n >> 1) & 0x5555U);
    n = (n & 0x3333U) + ((n >> 2) & 0x3333U);
    n = (n & 0x0f0fU) + ((n >> 4) & 0x0f0fU);
    n = (n & 0x00ffU) + ((n >> 8) & 0x00ffU);
    return n;
#endif
}
Vc_INTRINSIC Vc_CONST unsigned int popcnt32(unsigned int n)
{
#ifdef Vc_IMPL_POPCNT
    return _mm_popcnt_u32(n);
#else
    n = (n & 0x55555555U) + ((n >> 1) & 0x55555555U);
    n = (n & 0x33333333U) + ((n >> 2) & 0x33333333U);
    n = (n & 0x0f0f0f0fU) + ((n >> 4) & 0x0f0f0f0fU);
    n = (n & 0x00ff00ffU) + ((n >> 8) & 0x00ff00ffU);
    n = (n & 0x0000ffffU) + ((n >>16) & 0x0000ffffU);
    return n;
#endif
}
Vc_INTRINSIC Vc_CONST unsigned int popcnt64(ullong n)
{
#ifdef Vc_IMPL_POPCNT
#ifdef Vc_IS_AMD64
    return _mm_popcnt_u64(n);
#else   // Vc_IS_AMD64
    return _mm_popcnt_u32(n) + _mm_popcnt_u32(n >> 32u);
#endif  // Vc_IS_AMD64
#else
    n = (n & 0x5555555555555555ULL) + ((n >> 1) & 0x5555555555555555ULL);
    n = (n & 0x3333333333333333ULL) + ((n >> 2) & 0x3333333333333333ULL);
    n = (n & 0x0f0f0f0f0f0f0f0fULL) + ((n >> 4) & 0x0f0f0f0f0f0f0f0fULL);
    n = (n & 0x00ff00ff00ff00ffULL) + ((n >> 8) & 0x00ff00ff00ff00ffULL);
    n = (n & 0x0000ffff0000ffffULL) + ((n >>16) & 0x0000ffff0000ffffULL);
    n = (n & 0x00000000ffffffffULL) + ((n >>32) & 0x00000000ffffffffULL);
    return n;
#endif
}

// firstbit{{{1
Vc_INTRINSIC Vc_CONST int firstbit(ullong bits)
{
#ifdef Vc_HAVE_BMI1
#ifdef Vc_IS_AMD64
    return _tzcnt_u64(bits);
#else
    uint lo = bits;
    uint hi = bits >> 32u;
    if (lo == 0u) {
        return 32u + _tzcnt_u32(hi);
    } else {
        return _tzcnt_u32(lo);
    }
#endif
#else   // Vc_HAVE_BMI1
    return __builtin_ctzll(bits);
#endif  // Vc_HAVE_BMI1
}

#ifdef Vc_MSVC
#pragma intrinsic(_BitScanForward)
#pragma intrinsic(_BitScanReverse)
#endif

Vc_INTRINSIC Vc_CONST auto firstbit(uint x)
{
#if defined Vc_ICC || defined Vc_GCC
    return _bit_scan_forward(x);
#elif defined Vc_CLANG || defined Vc_APPLECLANG
    return __builtin_ctz(x);
#elif defined Vc_MSVC
    unsigned long index;
    _BitScanForward(&index, x);
    return index;
#else
#error "Implementation for firstbit(uint) is missing"
#endif
}

Vc_INTRINSIC Vc_CONST auto firstbit(llong bits) { return firstbit(ullong(bits)); }
#if LONG_MAX == LLONG_MAX
Vc_INTRINSIC Vc_CONST auto firstbit(ulong bits) { return firstbit(ullong(bits)); }
Vc_INTRINSIC Vc_CONST auto firstbit(long bits) { return firstbit(ullong(bits)); }
#endif  // long uses 64 bits
template <class T> Vc_INTRINSIC Vc_CONST auto firstbit(T bits)
{
    static_assert(sizeof(T) <= sizeof(uint),
                  "there's a missing overload to call the 64-bit variant");
    return firstbit(uint(bits));
}

// lastbit{{{1
Vc_INTRINSIC Vc_CONST int lastbit(ullong bits)
{
#ifdef Vc_HAVE_BMI1
#ifdef Vc_IS_AMD64
    return 63u - _lzcnt_u64(bits);
#else
    uint lo = bits;
    uint hi = bits >> 32u;
    if (hi == 0u) {
        return 31u - _lzcnt_u32(lo);
    } else {
        return 63u - _lzcnt_u32(hi);
    }
#endif
#else   // Vc_HAVE_BMI1
    return 63 - __builtin_clzll(bits);
#endif  // Vc_HAVE_BMI1
}

Vc_INTRINSIC Vc_CONST auto lastbit(uint x)
{
#if defined Vc_ICC || defined Vc_GCC
    return _bit_scan_reverse(x);
#elif defined Vc_CLANG || defined Vc_APPLECLANG
    return 31 - __builtin_clz(x);
#elif defined(Vc_MSVC)
    unsigned long index;
    _BitScanReverse(&index, x);
    return index;
#else
#error "Implementation for lastbit(uint) is missing"
#endif
}

Vc_INTRINSIC Vc_CONST auto lastbit(llong bits) { return lastbit(ullong(bits)); }
#if LONG_MAX == LLONG_MAX
Vc_INTRINSIC Vc_CONST auto lastbit(ulong bits) { return lastbit(ullong(bits)); }
Vc_INTRINSIC Vc_CONST auto lastbit(long bits) { return lastbit(ullong(bits)); }
#endif  // long uses 64 bits
template <class T> Vc_INTRINSIC Vc_CONST auto lastbit(T bits)
{
    static_assert(sizeof(T) <= sizeof(uint),
                  "there's a missing overload to call the 64-bit variant");
    return lastbit(uint(bits));
}

// mask_count{{{1
template <size_t Size> int mask_count(__m128 );
template<> Vc_INTRINSIC Vc_CONST int mask_count<4>(__m128  k)
{
#ifdef Vc_IMPL_POPCNT
    return _mm_popcnt_u32(_mm_movemask_ps(k));
#elif defined Vc_HAVE_SSE2
    auto x = _mm_srli_epi32(_mm_castps_si128(k), 31);
    x = _mm_add_epi32(x, _mm_shuffle_epi32(x, _MM_SHUFFLE(0, 1, 2, 3)));
    x = _mm_add_epi32(x, _mm_shufflelo_epi16(x, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si32(x);
#else
    return popcnt4(_mm_movemask_ps(k));
#endif
}

#ifdef Vc_HAVE_SSE2
template <size_t Size> int mask_count(__m128i);
template <size_t Size> int mask_count(__m128d);
template<> Vc_INTRINSIC Vc_CONST int mask_count<2>(__m128d k)
{
    int mask = _mm_movemask_pd(k);
    return (mask & 1) + (mask >> 1);
}
template<> Vc_INTRINSIC Vc_CONST int mask_count<2>(__m128i k)
{
    return mask_count<2>(_mm_castsi128_pd(k));
}

template<> Vc_INTRINSIC Vc_CONST int mask_count<4>(__m128i k)
{
    return mask_count<4>(_mm_castsi128_ps(k));
}

template<> Vc_INTRINSIC Vc_CONST int mask_count<8>(__m128i k)
{
#ifdef Vc_IMPL_POPCNT
    return _mm_popcnt_u32(_mm_movemask_epi8(k)) / 2;
#else
    auto x = _mm_srli_epi16(k, 15);
    x = _mm_add_epi16(x, _mm_shuffle_epi32(x, _MM_SHUFFLE(0, 1, 2, 3)));
    x = _mm_add_epi16(x, _mm_shufflelo_epi16(x, _MM_SHUFFLE(0, 1, 2, 3)));
    x = _mm_add_epi16(x, _mm_shufflelo_epi16(x, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_extract_epi16(x, 0);
#endif
}

template<> Vc_INTRINSIC Vc_CONST int mask_count<16>(__m128i k)
{
    return popcnt16(_mm_movemask_epi8(k));
}
#endif  // Vc_HAVE_SSE2

// mask_to_int{{{1
template <size_t Size> inline int mask_to_int(__m128 ) { static_assert(Size == Size, "Size value not implemented"); return 0; }
#ifdef Vc_HAVE_SSE2
template <size_t Size> inline int mask_to_int(__m128d) { static_assert(Size == Size, "Size value not implemented"); return 0; }
template <size_t Size> inline int mask_to_int(__m128i) { static_assert(Size == Size, "Size value not implemented"); return 0; }
#endif  // Vc_HAVE_SSE2
#ifdef Vc_HAVE_AVX
template <size_t Size> inline int mask_to_int(__m256 ) { static_assert(Size == Size, "Size value not implemented"); return 0; }
template <size_t Size> inline int mask_to_int(__m256d) { static_assert(Size == Size, "Size value not implemented"); return 0; }
template <size_t Size> inline int mask_to_int(__m256i) { static_assert(Size == Size, "Size value not implemented"); return 0; }
#endif  // Vc_HAVE_AVX
#ifdef Vc_HAVE_AVX512F
template <size_t Size> inline uint mask_to_int(__mmask8  k) { return k; }
template <size_t Size> inline uint mask_to_int(__mmask16 k) { return k; }
template <size_t Size> inline uint mask_to_int(__mmask32 k) { return k; }
template <size_t Size> inline ullong mask_to_int(__mmask64 k) { return k; }
#endif  // Vc_HAVE_AVX512F

template<> Vc_INTRINSIC Vc_CONST int mask_to_int<4>(__m128 k)
{
    return _mm_movemask_ps(k);
}
#ifdef Vc_HAVE_SSE2
template<> Vc_INTRINSIC Vc_CONST int mask_to_int<2>(__m128d k)
{
    return _mm_movemask_pd(k);
}
template<> Vc_INTRINSIC Vc_CONST int mask_to_int<2>(__m128i k)
{
    return _mm_movemask_pd(_mm_castsi128_pd(k));
}
template<> Vc_INTRINSIC Vc_CONST int mask_to_int<4>(__m128i k)
{
    return _mm_movemask_ps(_mm_castsi128_ps(k));
}
template<> Vc_INTRINSIC Vc_CONST int mask_to_int<8>(__m128i k)
{
    return _mm_movemask_epi8(_mm_packs_epi16(k, _mm_setzero_si128()));
}
template<> Vc_INTRINSIC Vc_CONST int mask_to_int<16>(__m128i k)
{
    return _mm_movemask_epi8(k);
}
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC Vc_CONST int mask_to_int<4>(__m256d k)
{
    return _mm256_movemask_pd(k);
}
template <> Vc_INTRINSIC Vc_CONST int mask_to_int<4>(__m256i k)
{
    return mask_to_int<4>(_mm256_castsi256_pd(k));
}

template <> Vc_INTRINSIC Vc_CONST int mask_to_int<8>(__m256  k)
{
    return _mm256_movemask_ps(k);
}
template <> Vc_INTRINSIC Vc_CONST int mask_to_int<8>(__m256i k)
{
    return mask_to_int<8>(_mm256_castsi256_ps(k));
}

#ifdef Vc_HAVE_AVX2
template <> Vc_INTRINSIC Vc_CONST int mask_to_int<16>(__m256i k)
{
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512BW
    return _mm256_movepi16_mask(k);
#else
    return _mm256_movemask_epi8(_mm256_packs_epi16(k, intrin_cast<__m256i>(hi128(k)))) &
           0xffff;
#endif
}

template <> Vc_INTRINSIC Vc_CONST int mask_to_int<32>(__m256i k)
{
    return _mm256_movemask_epi8(k);
}
#endif  // Vc_HAVE_AVX2
#endif  // Vc_HAVE_AVX

// is_equal{{{1
template <size_t> inline bool is_equal(__m128, __m128);
template <> Vc_INTRINSIC Vc_CONST bool is_equal<4>(__m128 k1, __m128 k2)
{
    return _mm_movemask_ps(k1) == _mm_movemask_ps(k2);
}

#ifdef Vc_HAVE_SSE2
template <size_t> inline bool is_equal(__m128d, __m128d);
template <size_t> inline bool is_equal(__m128i, __m128i);
template <> Vc_INTRINSIC Vc_CONST bool is_equal<2>(__m128d k1, __m128d k2)
{
    return _mm_movemask_pd(k1) == _mm_movemask_pd(k2);
}

template <> Vc_INTRINSIC Vc_CONST bool is_equal<2>(__m128i k1, __m128i k2)
{
    return _mm_movemask_pd(_mm_castsi128_pd(k1)) == _mm_movemask_pd(_mm_castsi128_pd(k2));
}

template <> Vc_INTRINSIC Vc_CONST bool is_equal<4>(__m128i k1, __m128i k2)
{
    return _mm_movemask_ps(_mm_castsi128_ps(k1)) == _mm_movemask_ps(_mm_castsi128_ps(k2));
}

template <> Vc_INTRINSIC Vc_CONST bool is_equal<8>(__m128i k1, __m128i k2)
{
    return _mm_movemask_epi8(k1) == _mm_movemask_epi8(k2);
}

template <> Vc_INTRINSIC Vc_CONST bool is_equal<16>(__m128i k1, __m128i k2)
{
    return _mm_movemask_epi8(k1) == _mm_movemask_epi8(k2);
}
#endif  // Vc_HAVE_SSE2

// long cmp{{{1
#ifdef Vc_HAVE_AVX512F
template <int = sizeof(long)> Vc_INTRINSIC auto cmpeq_long_mask(__m512i x, __m512i y);
template <> Vc_INTRINSIC auto cmpeq_long_mask<8>(__m512i x, __m512i y)
{
    return _mm512_cmpeq_epi64_mask(x, y);
}
template <> Vc_INTRINSIC auto cmpeq_long_mask<4>(__m512i x, __m512i y)
{
    return _mm512_cmpeq_epi32_mask(x, y);
}

template <int = sizeof(long)> Vc_INTRINSIC auto cmplt_long_mask(__m512i x, __m512i y);
template <> Vc_INTRINSIC auto cmplt_long_mask<8>(__m512i x, __m512i y)
{
    return _mm512_cmplt_epi64_mask(x, y);
}
template <> Vc_INTRINSIC auto cmplt_long_mask<4>(__m512i x, __m512i y)
{
    return _mm512_cmplt_epi32_mask(x, y);
}

template <int = sizeof(long)> Vc_INTRINSIC auto cmplt_ulong_mask(__m512i x, __m512i y);
template <> Vc_INTRINSIC auto cmplt_ulong_mask<8>(__m512i x, __m512i y)
{
    return _mm512_cmplt_epu64_mask(x, y);
}
template <> Vc_INTRINSIC auto cmplt_ulong_mask<4>(__m512i x, __m512i y)
{
    return _mm512_cmplt_epu32_mask(x, y);
}

template <int = sizeof(long)> Vc_INTRINSIC auto cmple_long_mask(__m512i x, __m512i y);
template <> Vc_INTRINSIC auto cmple_long_mask<8>(__m512i x, __m512i y)
{
    return _mm512_cmple_epi64_mask(x, y);
}
template <> Vc_INTRINSIC auto cmple_long_mask<4>(__m512i x, __m512i y)
{
    return _mm512_cmple_epi32_mask(x, y);
}

template <int = sizeof(long)> Vc_INTRINSIC auto cmple_ulong_mask(__m512i x, __m512i y);
template <> Vc_INTRINSIC auto cmple_ulong_mask<8>(__m512i x, __m512i y)
{
    return _mm512_cmple_epu64_mask(x, y);
}
template <> Vc_INTRINSIC auto cmple_ulong_mask<4>(__m512i x, __m512i y)
{
    return _mm512_cmple_epu32_mask(x, y);
}
#endif  // Vc_HAVE_AVX512F

// loads{{{1
#ifndef Vc_CHECK_ALIGNMENT
template <class V, class T>
static Vc_ALWAYS_INLINE void assertCorrectAlignment(const T *)
{
}
#else
template <class V, class T>
static Vc_ALWAYS_INLINE void assertCorrectAlignment(const T *ptr)
{
    constexpr size_t s = alignof(V);
    static_assert((s & (s - 1)) == 0, "");
    if ((reinterpret_cast<size_t>(ptr) & (s - 1)) != 0) {
        std::fprintf(stderr, "A load with incorrect alignment has just been called. Look at the stacktrace to find the guilty load.\n");
        std::abort();
    }
}
#endif
/**
 * \internal
 * Abstraction for simplifying load operations in the SSE/AVX/AVX512 implementations
 *
 * \note The number in the suffix signifies the number of Bytes
 */
#ifdef Vc_HAVE_SSE2
template <class T> Vc_INTRINSIC __m128i load2(const T *mem, when_aligned<2>)
{
    assertCorrectAlignment<unsigned short>(mem);
    static_assert(sizeof(T) == 1, "expected argument with sizeof == 1");
    return _mm_cvtsi32_si128(*reinterpret_cast<const unsigned short *>(mem));
}
template <class T> Vc_INTRINSIC __m128i load2(const T *mem, when_unaligned<2>)
{
    static_assert(sizeof(T) == 1, "expected argument with sizeof == 1");
    short tmp;
    std::memcpy(&tmp, mem, 2);
    return _mm_cvtsi32_si128(tmp);
}
#endif  // Vc_HAVE_SSE2

template <class F> Vc_INTRINSIC __m128 load4(const float *mem, F)
{
    assertCorrectAlignment<float>(mem);
    return _mm_load_ss(mem);
}

#ifdef Vc_HAVE_SSE2
template <class F> Vc_INTRINSIC __m128i load4(const int *mem, F)
{
    assertCorrectAlignment<int>(mem);
    return _mm_cvtsi32_si128(mem[0]);
}
template <class F> Vc_INTRINSIC __m128i load4(const unsigned int *mem, F)
{
    assertCorrectAlignment<unsigned int>(mem);
    return _mm_cvtsi32_si128(mem[0]);
}
template <class T, class F> Vc_INTRINSIC __m128i load4(const T *mem, F)
{
    static_assert(sizeof(T) <= 2, "expected argument with sizeof <= 2");
    int tmp;
    std::memcpy(&tmp, mem, 4);
    return _mm_cvtsi32_si128(tmp);
}
#endif  // Vc_HAVE_SSE2

template <class F> Vc_INTRINSIC __m128 load8(const float *mem, F)
{
#ifdef Vc_HAVE_SSE2
    assertCorrectAlignment<double>(mem);
    return _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(mem)));
#else
    assertCorrectAlignment<__m64>(mem);
    return _mm_loadl_pi(_mm_undefined_ps(), reinterpret_cast<const __m64 *>(mem));
#endif
}

#ifdef Vc_HAVE_SSE2
template <class F> Vc_INTRINSIC __m128d load8(const double *mem, F)
{
    assertCorrectAlignment<double>(mem);
    return _mm_load_sd(mem);
}
template <class T, class F> Vc_INTRINSIC __m128i load8(const T *mem, F)
{
    assertCorrectAlignment<T>(mem);
    static_assert(std::is_integral<T>::value, "load8<T> is only intended for integral T");
    return _mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem));
}
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_SSE
Vc_INTRINSIC __m128 load16(const float *mem, when_aligned<16>)
{
    assertCorrectAlignment<__m128>(mem);
    return _mm_load_ps(mem);
}
Vc_INTRINSIC __m128 load16(const float *mem, when_unaligned<16>)
{
    return _mm_loadu_ps(mem);
}
#endif  // Vc_HAVE_SSE

#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC __m128d load16(const double *mem, when_aligned<16>)
{
    assertCorrectAlignment<__m128d>(mem);
    return _mm_load_pd(mem);
}
Vc_INTRINSIC __m128d load16(const double *mem, when_unaligned<16>)
{
    return _mm_loadu_pd(mem);
}
template <class T> Vc_INTRINSIC __m128i load16(const T *mem, when_aligned<16>)
{
    assertCorrectAlignment<__m128i>(mem);
    static_assert(std::is_integral<T>::value, "load16<T> is only intended for integral T");
    return _mm_load_si128(reinterpret_cast<const __m128i *>(mem));
}
template <class T> Vc_INTRINSIC __m128i load16(const T *mem, when_unaligned<16>)
{
    static_assert(std::is_integral<T>::value, "load16<T> is only intended for integral T");
    return _mm_loadu_si128(reinterpret_cast<const __m128i *>(mem));
}
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC __m256 load32(const float *mem, when_aligned<32>)
{
    assertCorrectAlignment<__m256>(mem);
    return _mm256_load_ps(mem);
}
Vc_INTRINSIC __m256 load32(const float *mem, when_unaligned<32>)
{
    return _mm256_loadu_ps(mem);
}
Vc_INTRINSIC __m256d load32(const double *mem, when_aligned<32>)
{
    assertCorrectAlignment<__m256d>(mem);
    return _mm256_load_pd(mem);
}
Vc_INTRINSIC __m256d load32(const double *mem, when_unaligned<32>)
{
    return _mm256_loadu_pd(mem);
}
template <class T> Vc_INTRINSIC __m256i load32(const T *mem, when_aligned<32>)
{
    assertCorrectAlignment<__m256i>(mem);
    static_assert(std::is_integral<T>::value, "load32<T> is only intended for integral T");
    return _mm256_load_si256(reinterpret_cast<const __m256i *>(mem));
}
template <class T> Vc_INTRINSIC __m256i load32(const T *mem, when_unaligned<32>)
{
    static_assert(std::is_integral<T>::value, "load32<T> is only intended for integral T");
    return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(mem));
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC __m512 load64(const float *mem, when_aligned<64>)
{
    assertCorrectAlignment<__m512>(mem);
    return _mm512_load_ps(mem);
}
Vc_INTRINSIC __m512 load64(const float *mem, when_unaligned<64>)
{
    return _mm512_loadu_ps(mem);
}
Vc_INTRINSIC __m512d load64(const double *mem, when_aligned<64>)
{
    assertCorrectAlignment<__m512d>(mem);
    return _mm512_load_pd(mem);
}
Vc_INTRINSIC __m512d load64(const double *mem, when_unaligned<64>)
{
    return _mm512_loadu_pd(mem);
}
template <class T>
Vc_INTRINSIC __m512i load64(const T *mem, when_aligned<64>)
{
    assertCorrectAlignment<__m512i>(mem);
    static_assert(std::is_integral<T>::value, "load64<T> is only intended for integral T");
    return _mm512_load_si512(mem);
}
template <class T>
Vc_INTRINSIC __m512i load64(const T *mem, when_unaligned<64>)
{
    static_assert(std::is_integral<T>::value, "load64<T> is only intended for integral T");
    return _mm512_loadu_si512(mem);
}
#endif

// stores{{{1
#ifdef Vc_HAVE_SSE
Vc_INTRINSIC void store4(__m128 v, float *mem, when_aligned<alignof(float)>)
{
    *mem = _mm_cvtss_f32(v);
}

Vc_INTRINSIC void store4(__m128 v, float *mem, when_unaligned<alignof(float)>)
{
    *mem = _mm_cvtss_f32(v);
}

Vc_INTRINSIC void store8(__m128 v, float *mem, when_aligned<alignof(__m64)>)
{
    _mm_storel_pi(reinterpret_cast<__m64 *>(mem), v);
}

Vc_INTRINSIC void store8(__m128 v, float *mem, when_unaligned<alignof(__m64)>)
{
    _mm_storel_pi(reinterpret_cast<__m64 *>(mem), v);
}

Vc_INTRINSIC void store16(__m128 v, float *mem, when_aligned<16>)
{
    _mm_store_ps(mem, v);
}
Vc_INTRINSIC void store16(__m128 v, float *mem, when_unaligned<16>)
{
    _mm_storeu_ps(mem, v);
}
#endif  // Vc_HAVE_SSE

#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC void store8(__m128d v, double *mem, when_aligned<alignof(double)>)
{
    *mem = _mm_cvtsd_f64(v);
}

Vc_INTRINSIC void store8(__m128d v, double *mem, when_unaligned<alignof(double)>)
{
    *mem = _mm_cvtsd_f64(v);
}

Vc_INTRINSIC void store16(__m128d v, double *mem, when_aligned<16>)
{
    _mm_store_pd(mem, v);
}
Vc_INTRINSIC void store16(__m128d v, double *mem, when_unaligned<16>)
{
    _mm_storeu_pd(mem, v);
}

template <class T> Vc_INTRINSIC void store2(__m128i v, T *mem, when_aligned<alignof(ushort)>)
{
    static_assert(std::is_integral<T>::value && sizeof(T) <= 2,
                  "store4<T> is only intended for integral T with sizeof(T) <= 2");
    *reinterpret_cast<may_alias<ushort> *>(mem) = uint(_mm_cvtsi128_si32(v));
}

template <class T> Vc_INTRINSIC void store2(__m128i v, T *mem, when_unaligned<alignof(ushort)>)
{
    static_assert(std::is_integral<T>::value && sizeof(T) <= 2,
                  "store4<T> is only intended for integral T with sizeof(T) <= 2");
    const uint tmp(_mm_cvtsi128_si32(v));
    std::memcpy(mem, &tmp, 2);
}

template <class T> Vc_INTRINSIC void store4(__m128i v, T *mem, when_aligned<alignof(int)>)
{
    static_assert(std::is_integral<T>::value && sizeof(T) <= 4,
                  "store4<T> is only intended for integral T with sizeof(T) <= 4");
    *reinterpret_cast<may_alias<int> *>(mem) = _mm_cvtsi128_si32(v);
}

template <class T> Vc_INTRINSIC void store4(__m128i v, T *mem, when_unaligned<alignof(int)>)
{
    static_assert(std::is_integral<T>::value && sizeof(T) <= 4,
                  "store4<T> is only intended for integral T with sizeof(T) <= 4");
    const int tmp = _mm_cvtsi128_si32(v);
    std::memcpy(mem, &tmp, 4);
}

template <class T> Vc_INTRINSIC void store8(__m128i v, T *mem, when_aligned<8>)
{
    static_assert(std::is_integral<T>::value, "store8<T> is only intended for integral T");
    _mm_storel_epi64(reinterpret_cast<__m128i *>(mem), v);
}

template <class T> Vc_INTRINSIC void store8(__m128i v, T *mem, when_unaligned<8>)
{
    static_assert(std::is_integral<T>::value, "store8<T> is only intended for integral T");
    _mm_storel_epi64(reinterpret_cast<__m128i *>(mem), v);
}

template <class T> Vc_INTRINSIC void store16(__m128i v, T *mem, when_aligned<16>)
{
    static_assert(std::is_integral<T>::value, "store16<T> is only intended for integral T");
    _mm_store_si128(reinterpret_cast<__m128i *>(mem), v);
}
template <class T> Vc_INTRINSIC void store16(__m128i v, T *mem, when_unaligned<16>)
{
    static_assert(std::is_integral<T>::value, "store16<T> is only intended for integral T");
    _mm_storeu_si128(reinterpret_cast<__m128i *>(mem), v);
}
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC void store32(__m256 v, float *mem, when_aligned<32>)
{
    _mm256_store_ps(mem, v);
}
Vc_INTRINSIC void store32(__m256 v, float *mem, when_unaligned<32>)
{
    _mm256_storeu_ps(mem, v);
}
Vc_INTRINSIC void store32(__m256d v, double *mem, when_aligned<32>)
{
    _mm256_store_pd(mem, v);
}
Vc_INTRINSIC void store32(__m256d v, double *mem, when_unaligned<32>)
{
    _mm256_storeu_pd(mem, v);
}
template <class T> Vc_INTRINSIC void store32(__m256i v, T *mem, when_aligned<32>)
{
    static_assert(std::is_integral<T>::value, "store32<T> is only intended for integral T");
    _mm256_store_si256(reinterpret_cast<__m256i *>(mem), v);
}
template <class T> Vc_INTRINSIC void store32(__m256i v, T *mem, when_unaligned<32>)
{
    static_assert(std::is_integral<T>::value, "store32<T> is only intended for integral T");
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(mem), v);
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC void store64(__m512 v, float *mem, when_aligned<64>)
{
    assertCorrectAlignment<__m512>(mem);
    _mm512_store_ps(mem, v);
}
Vc_INTRINSIC void store64(__m512 v, float *mem, when_unaligned<64>)
{
    _mm512_storeu_ps(mem, v);
}
Vc_INTRINSIC void store64(__m512d v, double *mem, when_aligned<64>)
{
    assertCorrectAlignment<__m512d>(mem);
    _mm512_store_pd(mem, v);
}
Vc_INTRINSIC void store64(__m512d v, double *mem, when_unaligned<64>)
{
    _mm512_storeu_pd(mem, v);
}
template <class T>
Vc_INTRINSIC void store64(__m512i v, T *mem, when_aligned<64>)
{
    assertCorrectAlignment<__m512i>(mem);
    static_assert(std::is_integral<T>::value, "store64<T> is only intended for integral T");
    _mm512_store_si512(mem, v);
}
template <class T>
Vc_INTRINSIC void store64(__m512i v, T *mem, when_unaligned<64>)
{
    static_assert(std::is_integral<T>::value, "store64<T> is only intended for integral T");
    _mm512_storeu_si512(mem, v);
}
#endif

// }}}1
}  // namespace x86
using namespace x86;
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END
#endif  // Vc_HAVE_SSE

#endif  // VC_DATAPAR_X86_H_
