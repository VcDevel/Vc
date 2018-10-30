/*  This file is part of the Vc library. {{{
Copyright © 2016-2017 Matthias Kretz <kretz@kde.org>

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

#ifndef TESTS_TESTTYPES_H_
#define TESTS_TESTTYPES_H_

#include <vir/typelist.h>
#include <experimental/simd>

using schar = signed char;
using uchar = unsigned char;
using ushort = unsigned short;
using uint = unsigned int;
using ulong = unsigned long;
using llong = long long;
using ullong = unsigned long long;
using ldouble = long double;
using wchar = wchar_t;
using char16 = char16_t;
using char32 = char32_t;

using all_native_abis =
    vir::Typelist<std::experimental::simd_abi::scalar, std::experimental::simd_abi::__sse, std::experimental::simd_abi::__avx,
                  std::experimental::simd_abi::__avx512, std::experimental::simd_abi::__neon>;

using testtypes = vir::Typelist<
#ifdef TESTTYPES
    TESTTYPES
#else
    ldouble, double, float, ullong, llong, ulong, long, uint, int, ushort, short, uchar,
    schar, char, wchar_t, char16_t, char32_t
#endif
    >;
using testtypes_wo_ldouble = typename vir::filter_list<long double, testtypes>::type;
using testtypes_64_32 =
    typename vir::filter_list<vir::Typelist<ushort, short, uchar, schar, char, wchar_t, char16_t>,
                              testtypes_wo_ldouble>::type;
using testtypes_fp =
    typename vir::filter_list<vir::Typelist<ullong, llong, ulong, long, uint, int, char32_t>,
                              testtypes_64_32>::type;
using testtypes_float = typename vir::filter_list<double, testtypes_fp>::type;
static_assert(vir::list_size<testtypes_fp>::value <= 2, "filtering the list failed");
static_assert(vir::list_size<testtypes_float>::value <= 1, "filtering the list failed");

// (all_)arithmetic_types {{{1
using all_arithmetic_types =
    vir::Typelist<long double, double, float, long long, unsigned long, int,
                  unsigned short, signed char, unsigned long long, long, unsigned int,
                  short, unsigned char, char32_t, char16_t, char, wchar_t>;
#ifdef ONE_RANDOM_ARITHMETIC_TYPE
using arithmetic_types = VIR_CHOOSE_ONE_RANDOMLY(all_arithmetic_types);
#else
using arithmetic_types = all_arithmetic_types;
#endif

// vT {{{1
using vschar = std::experimental::native_simd<schar>;
using vuchar = std::experimental::native_simd<uchar>;
using vshort = std::experimental::native_simd<short>;
using vushort = std::experimental::native_simd<ushort>;
using vint = std::experimental::native_simd<int>;
using vuint = std::experimental::native_simd<uint>;
using vlong = std::experimental::native_simd<long>;
using vulong = std::experimental::native_simd<ulong>;
using vllong = std::experimental::native_simd<llong>;
using vullong = std::experimental::native_simd<ullong>;
using vfloat = std::experimental::native_simd<float>;
using vdouble = std::experimental::native_simd<double>;
using vldouble = std::experimental::native_simd<long double>;

using vchar = std::experimental::native_simd<char>;
using vwchar = std::experimental::native_simd<wchar_t>;
using vchar16 = std::experimental::native_simd<char16_t>;
using vchar32 = std::experimental::native_simd<char32_t>;

// viN/vfN {{{1
template <typename T> using vi8  = std::experimental::fixed_size_simd<T, vschar::size()>;
template <typename T> using vi16 = std::experimental::fixed_size_simd<T, vshort::size()>;
template <typename T> using vf32 = std::experimental::fixed_size_simd<T, vfloat::size()>;
template <typename T> using vi32 = std::experimental::fixed_size_simd<T, vint::size()>;
template <typename T> using vf64 = std::experimental::fixed_size_simd<T, vdouble::size()>;
template <typename T> using vi64 = std::experimental::fixed_size_simd<T, vllong::size()>;
template <typename T>
using vl = typename std::conditional<sizeof(long) == sizeof(llong), vi64<T>, vi32<T>>::type;

// current_native_test_types {{{1
using current_native_test_types =
    vir::expand_one<vir::Template1<std::experimental::native_simd>, testtypes>;
using current_native_mask_test_types =
    vir::expand_one<vir::Template1<std::experimental::native_simd_mask>, testtypes>;

// native_test_types {{{1
typedef vir::concat<
#if defined _GLIBCXX_SIMD_HAVE_AVX512_ABI && !defined _GLIBCXX_SIMD_HAVE_FULL_AVX512_ABI
    vir::expand_one<vir::Template<base_template, std::experimental::simd_abi::__avx512>,
                    testtypes_64_32>,
#endif
#if defined _GLIBCXX_SIMD_HAVE_AVX_ABI && !defined _GLIBCXX_SIMD_HAVE_FULL_AVX_ABI
    vir::expand_one<vir::Template<base_template, std::experimental::simd_abi::__avx>, testtypes_fp>,
#endif
#if defined _GLIBCXX_SIMD_HAVE_SSE_ABI && !defined _GLIBCXX_SIMD_HAVE_FULL_SSE_ABI
    vir::expand_one<vir::Template<base_template, std::experimental::simd_abi::__sse>, testtypes_float>,
#endif
    vir::expand_list<vir::concat<
#ifdef _GLIBCXX_SIMD_HAVE_FULL_AVX512_ABI
                         vir::Template<base_template, std::experimental::simd_abi::__avx512>,
#endif
#ifdef _GLIBCXX_SIMD_HAVE_FULL_AVX_ABI
                         vir::Template<base_template, std::experimental::simd_abi::__avx>,
#endif
#ifdef _GLIBCXX_SIMD_HAVE_FULL_SSE_ABI
                         vir::Template<base_template, std::experimental::simd_abi::__sse>,
#endif
                         vir::Typelist<>>,
                     testtypes_wo_ldouble>> native_test_types;

// native_real_test_types {{{1
using native_real_test_types = vir::concat<
#if defined _GLIBCXX_SIMD_HAVE_AVX512_ABI
    vir::expand_one<vir::Template<base_template, std::experimental::simd_abi::__avx512>, testtypes_fp>,
#endif
#if defined _GLIBCXX_SIMD_HAVE_AVX_ABI
    vir::expand_one<vir::Template<base_template, std::experimental::simd_abi::__avx>, testtypes_fp>,
#endif
#if defined _GLIBCXX_SIMD_HAVE_SSE_ABI
#if defined _GLIBCXX_SIMD_HAVE_FULL_SSE_ABI
    vir::expand_one<vir::Template<base_template, std::experimental::simd_abi::__sse>, testtypes_fp>,
#else
    vir::expand_one<vir::Template<base_template, std::experimental::simd_abi::__sse>, testtypes_float>,
#endif
#endif
    vir::Typelist<>>;

// two_fixed_size_abi {{{1
using two_fixed_size_abi = vir::concat<
    VIR_CHOOSE_ONE_RANDOMLY(
        vir::Typelist<vir::Template<base_template, std::experimental::simd_abi::fixed_size<1>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<2>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<3>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<4>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<5>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<6>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<7>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<8>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<9>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<10>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<11>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<12>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<13>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<14>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<15>>>),
    VIR_CHOOSE_ONE_RANDOMLY(
        vir::Typelist<
            vir::Template<base_template, std::experimental::simd_abi::fixed_size<16>>,
            vir::Template<base_template, std::experimental::simd_abi::fixed_size<
                                             std::experimental::simd_abi::max_fixed_size<double> - 1>>,
            vir::Template<base_template, std::experimental::simd_abi::fixed_size<
                                             std::experimental::simd_abi::max_fixed_size<double>>>>)>;

// three_fixed_size_abi {{{1
using three_fixed_size_abi = vir::concat<
    VIR_CHOOSE_ONE_RANDOMLY(
        vir::Typelist<vir::Template<base_template, std::experimental::simd_abi::fixed_size<3>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<6>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<8>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<12>>>),
    VIR_CHOOSE_ONE_RANDOMLY(
        vir::Typelist<vir::Template<base_template, std::experimental::simd_abi::fixed_size<1>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<2>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<4>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<5>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<7>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<9>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<10>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<11>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<13>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<14>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<15>>>),
    VIR_CHOOSE_ONE_RANDOMLY(
        vir::Typelist<vir::Template<base_template, std::experimental::simd_abi::fixed_size<16>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<31>>,
                      vir::Template<base_template, std::experimental::simd_abi::fixed_size<32>>>)>;

// all_test_types {{{1
using all_test_types = vir::concat<
    native_test_types,
    vir::expand_list<vir::concat<vir::Template<base_template, std::experimental::simd_abi::scalar>,
                                 three_fixed_size_abi>,
                     testtypes>>;

// real_test_types {{{1
using real_test_types = vir::concat<
    native_real_test_types,
    vir::expand_list<vir::concat<vir::Template<base_template, std::experimental::simd_abi::scalar>,
                                 three_fixed_size_abi>,
                     testtypes_fp>>;

// many_fixed_size_types {{{1
using many_fixed_size_types = vir::expand_list<
    vir::Typelist<vir::Template<base_template, std::experimental::simd_abi::fixed_size<3>>,
                  vir::Template<base_template, std::experimental::simd_abi::fixed_size<4>>,
                  vir::Template<base_template, std::experimental::simd_abi::fixed_size<5>>,
                  vir::Template<base_template, std::experimental::simd_abi::fixed_size<6>>,
                  vir::Template<base_template, std::experimental::simd_abi::fixed_size<7>>,
                  vir::Template<base_template, std::experimental::simd_abi::fixed_size<8>>,
                  vir::Template<base_template, std::experimental::simd_abi::fixed_size<9>>,
                  vir::Template<base_template, std::experimental::simd_abi::fixed_size<10>>,
                  vir::Template<base_template, std::experimental::simd_abi::fixed_size<11>>,
                  vir::Template<base_template, std::experimental::simd_abi::fixed_size<12>>,
                  vir::Template<base_template, std::experimental::simd_abi::fixed_size<13>>,
                  vir::Template<base_template, std::experimental::simd_abi::fixed_size<14>>,
                  vir::Template<base_template, std::experimental::simd_abi::fixed_size<15>>,
                  vir::Template<base_template, std::experimental::simd_abi::fixed_size<17>>>,
    testtypes_float>;
// reduced_test_types {{{1
#ifdef _GLIBCXX_SIMD_HAVE_AVX512F
// reduce compile times when AVX512 is available
using reduced_test_types = native_test_types;
#else   // _GLIBCXX_SIMD_HAVE_AVX512F
typedef vir::concat<
    native_test_types,
    vir::expand_list<vir::Typelist<vir::Template<base_template, std::experimental::simd_abi::scalar>>,
                     testtypes>> reduced_test_types;
#endif  // _GLIBCXX_SIMD_HAVE_AVX512F

//}}}1
#endif  // TESTS_TESTTYPES_H_
