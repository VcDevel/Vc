/*  This file is part of the Vc library. {{{
Copyright © 2018 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_DETAIL_X86_TYPES_H_
#define VC_DETAIL_X86_TYPES_H_

#include "../macros.h"
#include "../builtins.h"

#include <x86intrin.h>
#include <climits>

namespace Vc_VERSIONED_NAMESPACE
{
namespace detail
{

// bool_storage_member_type{{{1
#ifdef Vc_HAVE_AVX512F
template <> struct bool_storage_member_type< 2> { using type = __mmask8 ; };
template <> struct bool_storage_member_type< 4> { using type = __mmask8 ; };
template <> struct bool_storage_member_type< 8> { using type = __mmask8 ; };
template <> struct bool_storage_member_type<16> { using type = __mmask16; };
template <> struct bool_storage_member_type<32> { using type = __mmask32; };
template <> struct bool_storage_member_type<64> { using type = __mmask64; };
#endif  // Vc_HAVE_AVX512F

// intrinsic_type{{{1
// the following excludes bool via is_vectorizable
template <class T>
using void_if_integral_t = detail::void_t<std::enable_if_t<
    detail::all<std::is_integral<T>, detail::is_vectorizable<T>>::value>>;
#if defined Vc_HAVE_AVX512F
template <> struct intrinsic_type<double, 64, void> { using type = __m512d; };
template <> struct intrinsic_type< float, 64, void> { using type = __m512; };
template <typename T> struct intrinsic_type<T, 64, void_if_integral_t<T>> { using type = __m512i; };
#endif  // Vc_HAVE_AVX512F

#if defined Vc_HAVE_AVX
template <> struct intrinsic_type<double, 32, void> { using type = __m256d; };
template <> struct intrinsic_type< float, 32, void> { using type = __m256; };
template <typename T> struct intrinsic_type<T, 32, void_if_integral_t<T>> { using type = __m256i; };
#endif  // Vc_HAVE_AVX

#if defined Vc_HAVE_SSE
template <> struct intrinsic_type< float, 16, void> { using type = __m128; };
template <> struct intrinsic_type< float,  8, void> { using type = __m128; };
template <> struct intrinsic_type< float,  4, void> { using type = __m128; };
#endif  // Vc_HAVE_SSE
#if defined Vc_HAVE_SSE2
template <> struct intrinsic_type<double, 16, void> { using type = __m128d; };
template <> struct intrinsic_type<double,  8, void> { using type = __m128d; };
template <typename T> struct intrinsic_type<T, 16, void_if_integral_t<T>> { using type = __m128i; };
template <typename T> struct intrinsic_type<T,  8, void_if_integral_t<T>> { using type = __m128i; };
template <typename T> struct intrinsic_type<T,  4, void_if_integral_t<T>> { using type = __m128i; };
template <typename T> struct intrinsic_type<T,  2, void_if_integral_t<T>> { using type = __m128i; };
template <typename T> struct intrinsic_type<T,  1, void_if_integral_t<T>> { using type = __m128i; };
#endif  // Vc_HAVE_SSE2

// is_intrinsic{{{1
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


// (sse|avx|avx512)_(simd|mask)_member_type{{{1
template <class T> using sse_simd_member_type = storage16_t<T>;
template <class T> using sse_mask_member_type = storage16_t<T>;

template <class T> using avx_simd_member_type = storage32_t<T>;
template <class T> using avx_mask_member_type = storage32_t<T>;

template <class T> using avx512_simd_member_type = storage64_t<T>;
template <class T> using avx512_mask_member_type = Storage<bool, 64 / sizeof(T)>;
template <size_t N> using avx512_mask_member_type_n = Storage<bool, N>;

//}}}1

// x_ aliases {{{
#ifdef Vc_HAVE_SSE
using x_f32 = Storage< float,  4>;
#ifdef Vc_HAVE_SSE2
using x_f64 = Storage<double,  2>;
using x_i08 = Storage< schar, 16>;
using x_u08 = Storage< uchar, 16>;
using x_i16 = Storage< short,  8>;
using x_u16 = Storage<ushort,  8>;
using x_i32 = Storage<   int,  4>;
using x_u32 = Storage<  uint,  4>;
using x_i64 = Storage< llong,  2>;
using x_u64 = Storage<ullong,  2>;
using x_long = Storage<long,   16 / sizeof(long)>;
using x_ulong = Storage<ulong, 16 / sizeof(ulong)>;
using x_long_equiv = Storage<equal_int_type_t<long>, 16 / sizeof(long)>;
using x_ulong_equiv = Storage<equal_int_type_t<ulong>, 16 / sizeof(ulong)>;
using x_chr = Storage<    char, 16>;
using x_c16 = Storage<char16_t,  8>;
using x_c32 = Storage<char32_t,  4>;
using x_wch = Storage< wchar_t, 16 / sizeof(wchar_t)>;
#endif  // Vc_HAVE_SSE2
#endif  // Vc_HAVE_SSE

//}}}
// y_ aliases {{{
using y_f32 = Storage< float,  8>;
using y_f64 = Storage<double,  4>;
using y_i08 = Storage< schar, 32>;
using y_u08 = Storage< uchar, 32>;
using y_i16 = Storage< short, 16>;
using y_u16 = Storage<ushort, 16>;
using y_i32 = Storage<   int,  8>;
using y_u32 = Storage<  uint,  8>;
using y_i64 = Storage< llong,  4>;
using y_u64 = Storage<ullong,  4>;
using y_long = Storage<long,   32 / sizeof(long)>;
using y_ulong = Storage<ulong, 32 / sizeof(ulong)>;
using y_long_equiv = Storage<equal_int_type_t<long>, 32 / sizeof(long)>;
using y_ulong_equiv = Storage<equal_int_type_t<ulong>, 32 / sizeof(ulong)>;
using y_chr = Storage<    char, 32>;
using y_c16 = Storage<char16_t, 16>;
using y_c32 = Storage<char32_t,  8>;
using y_wch = Storage< wchar_t, 32 / sizeof(wchar_t)>;

//}}}
// z_ aliases {{{
using z_f32 = Storage< float, 16>;
using z_f64 = Storage<double,  8>;
using z_i32 = Storage<   int, 16>;
using z_u32 = Storage<  uint, 16>;
using z_i64 = Storage< llong,  8>;
using z_u64 = Storage<ullong,  8>;
using z_long = Storage<long,   64 / sizeof(long)>;
using z_ulong = Storage<ulong, 64 / sizeof(ulong)>;
using z_i08 = Storage< schar, 64>;
using z_u08 = Storage< uchar, 64>;
using z_i16 = Storage< short, 32>;
using z_u16 = Storage<ushort, 32>;
using z_long_equiv = Storage<equal_int_type_t<long>, 64 / sizeof(long)>;
using z_ulong_equiv = Storage<equal_int_type_t<ulong>, 64 / sizeof(ulong)>;
using z_chr = Storage<    char, 64>;
using z_c16 = Storage<char16_t, 32>;
using z_c32 = Storage<char32_t, 16>;
using z_wch = Storage< wchar_t, 64 / sizeof(wchar_t)>;

//}}}

#define Vc_BINARY_OVERLOAD_SAME_VALUE_REP_SSE2_(fun_)                                    \
    Vc_INTRINSIC x_chr fun_(x_chr a, x_chr b) { return fun_(a.equiv(), b.equiv()); }     \
    Vc_INTRINSIC x_c16 fun_(x_c16 a, x_c16 b) { return fun_(a.equiv(), b.equiv()); }     \
    Vc_INTRINSIC x_c32 fun_(x_c32 a, x_c32 b) { return fun_(a.equiv(), b.equiv()); }     \
    Vc_INTRINSIC x_wch fun_(x_wch a, x_wch b) { return fun_(a.equiv(), b.equiv()); }     \
    Vc_INTRINSIC x_long fun_(x_long a, x_long b) { return fun_(a.equiv(), b.equiv()); }  \
    Vc_INTRINSIC x_ulong fun_(x_ulong a, x_ulong b) { return fun_(a.equiv(), b.equiv()); }

#define Vc_BINARY_OVERLOAD_SAME_VALUE_REP_AVX2_(fun_)                                    \
    Vc_BINARY_OVERLOAD_SAME_VALUE_REP_SSE2_(fun_)                                        \
    Vc_INTRINSIC y_chr fun_(y_chr a, y_chr b) { return fun_(a.equiv(), b.equiv()); }     \
    Vc_INTRINSIC y_c16 fun_(y_c16 a, y_c16 b) { return fun_(a.equiv(), b.equiv()); }     \
    Vc_INTRINSIC y_c32 fun_(y_c32 a, y_c32 b) { return fun_(a.equiv(), b.equiv()); }     \
    Vc_INTRINSIC y_wch fun_(y_wch a, y_wch b) { return fun_(a.equiv(), b.equiv()); }     \
    Vc_INTRINSIC y_long fun_(y_long a, y_long b) { return fun_(a.equiv(), b.equiv()); }  \
    Vc_INTRINSIC y_ulong fun_(y_ulong a, y_ulong b) { return fun_(a.equiv(), b.equiv()); }

#define Vc_BINARY_OVERLOAD_SAME_VALUE_REP_AVX512_(fun_)                                  \
    Vc_BINARY_OVERLOAD_SAME_VALUE_REP_AVX2_(fun_)                                        \
    Vc_INTRINSIC z_chr fun_(z_chr a, z_chr b) { return fun_(a.equiv(), b.equiv()); }     \
    Vc_INTRINSIC z_c16 fun_(z_c16 a, z_c16 b) { return fun_(a.equiv(), b.equiv()); }     \
    Vc_INTRINSIC z_c32 fun_(z_c32 a, z_c32 b) { return fun_(a.equiv(), b.equiv()); }     \
    Vc_INTRINSIC z_wch fun_(z_wch a, z_wch b) { return fun_(a.equiv(), b.equiv()); }     \
    Vc_INTRINSIC z_long fun_(z_long a, z_long b) { return fun_(a.equiv(), b.equiv()); }  \
    Vc_INTRINSIC z_ulong fun_(z_ulong a, z_ulong b) { return fun_(a.equiv(), b.equiv()); }

#ifdef Vc_HAVE_AVX512F
#define Vc_BINARY_OVERLOAD_SAME_VALUE_REP_(fun_)                                         \
    Vc_BINARY_OVERLOAD_SAME_VALUE_REP_AVX512_(fun_)
#elif defined Vc_HAVE_AVX2
#define Vc_BINARY_OVERLOAD_SAME_VALUE_REP_(fun_)                                         \
    Vc_BINARY_OVERLOAD_SAME_VALUE_REP_AVX2_(fun_)
#elif defined Vc_HAVE_SSE2
#define Vc_BINARY_OVERLOAD_SAME_VALUE_REP_(fun_)                                         \
    Vc_BINARY_OVERLOAD_SAME_VALUE_REP_SSE2_(fun_)
#else
#define Vc_BINARY_OVERLOAD_SAME_VALUE_REP_(fun_)
#endif

}  // namespace detail
}  // namespace Vc

#endif  // VC_DETAIL_X86_TYPES_H_

// vim: foldmethod=marker
