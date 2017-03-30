/*  This file is part of the Vc library. {{{
Copyright © 2017 Matthias Kretz <kretz@kde.org>

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

//#define UNITTEST_ONLY_XTEST 1
#include <vir/test.h>
#include <Vc/datapar>
#include "metahelpers.h"

#define TESTTYPES                                                                        \
    long double, double, float, unsigned long long, long long, unsigned long, long,      \
        unsigned int, int, unsigned short, short, unsigned char, signed char, char
template <class... Ts> using base_template = Vc::datapar<Ts...>;
#include "testtypes.h"

using vir::Typelist;
using vir::Template;
using vir::Template1;
using vir::expand_list;
using vir::filter_list;
using vir::concat;
using Vc::datapar;
using Vc::mask;

using all_valid_scalars = expand_list<Typelist<Template<datapar, Vc::datapar_abi::scalar>,
                                               Template<mask, Vc::datapar_abi::scalar>>,
                                      testtypes>;

using all_valid_fixed_size =
    expand_list<Typelist<Template<datapar, Vc::datapar_abi::fixed_size<1>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<2>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<3>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<4>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<5>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<6>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<7>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<8>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<9>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<10>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<11>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<12>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<13>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<14>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<15>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<16>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<17>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<18>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<19>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<20>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<21>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<22>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<23>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<24>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<25>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<26>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<27>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<28>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<29>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<30>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<31>>,
                         Template<datapar, Vc::datapar_abi::fixed_size<32>>,
                         Template<mask, Vc::datapar_abi::fixed_size<1>>,
                         Template<mask, Vc::datapar_abi::fixed_size<2>>,
                         Template<mask, Vc::datapar_abi::fixed_size<3>>,
                         Template<mask, Vc::datapar_abi::fixed_size<4>>,
                         Template<mask, Vc::datapar_abi::fixed_size<5>>,
                         Template<mask, Vc::datapar_abi::fixed_size<6>>,
                         Template<mask, Vc::datapar_abi::fixed_size<7>>,
                         Template<mask, Vc::datapar_abi::fixed_size<8>>,
                         Template<mask, Vc::datapar_abi::fixed_size<9>>,
                         Template<mask, Vc::datapar_abi::fixed_size<10>>,
                         Template<mask, Vc::datapar_abi::fixed_size<11>>,
                         Template<mask, Vc::datapar_abi::fixed_size<12>>,
                         Template<mask, Vc::datapar_abi::fixed_size<13>>,
                         Template<mask, Vc::datapar_abi::fixed_size<14>>,
                         Template<mask, Vc::datapar_abi::fixed_size<15>>,
                         Template<mask, Vc::datapar_abi::fixed_size<16>>,
                         Template<mask, Vc::datapar_abi::fixed_size<17>>,
                         Template<mask, Vc::datapar_abi::fixed_size<18>>,
                         Template<mask, Vc::datapar_abi::fixed_size<19>>,
                         Template<mask, Vc::datapar_abi::fixed_size<20>>,
                         Template<mask, Vc::datapar_abi::fixed_size<21>>,
                         Template<mask, Vc::datapar_abi::fixed_size<22>>,
                         Template<mask, Vc::datapar_abi::fixed_size<23>>,
                         Template<mask, Vc::datapar_abi::fixed_size<24>>,
                         Template<mask, Vc::datapar_abi::fixed_size<25>>,
                         Template<mask, Vc::datapar_abi::fixed_size<26>>,
                         Template<mask, Vc::datapar_abi::fixed_size<27>>,
                         Template<mask, Vc::datapar_abi::fixed_size<28>>,
                         Template<mask, Vc::datapar_abi::fixed_size<29>>,
                         Template<mask, Vc::datapar_abi::fixed_size<30>>,
                         Template<mask, Vc::datapar_abi::fixed_size<31>>,
                         Template<mask, Vc::datapar_abi::fixed_size<32>>>,
                testtypes>;

using all_valid_simd = concat<
#if defined Vc_HAVE_FULL_SSE_ABI
    expand_list<Typelist<Template<datapar, Vc::datapar_abi::sse>,
                         Template<mask, Vc::datapar_abi::sse>>,
                testtypes_wo_ldouble>,
#elif defined Vc_HAVE_SSE_ABI
    expand_list<Typelist<Template<datapar, Vc::datapar_abi::sse>,
                         Template<mask, Vc::datapar_abi::sse>>,
                testtypes_float>,
#endif
#if defined Vc_HAVE_FULL_AVX_ABI
    expand_list<Typelist<Template<datapar, Vc::datapar_abi::avx>,
                         Template<mask, Vc::datapar_abi::avx>>,
                testtypes_wo_ldouble>,
#elif defined Vc_HAVE_AVX_ABI
    expand_list<Typelist<Template<datapar, Vc::datapar_abi::avx>,
                         Template<mask, Vc::datapar_abi::avx>>,
                testtypes_fp>,
#endif
#if defined Vc_HAVE_FULL_AVX512_ABI
    expand_list<Typelist<Template<datapar, Vc::datapar_abi::avx512>,
                         Template<mask, Vc::datapar_abi::avx512>>,
                testtypes_wo_ldouble>,
#elif defined Vc_HAVE_AVX512_ABI
    expand_list<Typelist<Template<datapar, Vc::datapar_abi::avx512>,
                         Template<mask, Vc::datapar_abi::avx512>>,
                testtypes_64_32>,
#endif
#if defined Vc_HAVE_FULL_NEON_ABI
    expand_list<Typelist<Template<datapar, Vc::datapar_abi::neon>,
                         Template<mask, Vc::datapar_abi::neon>>,
                testtypes_wo_ldouble>,
#endif
    Typelist<>>;

TEST_TYPES(V, is_usable, concat<all_valid_scalars, all_valid_simd, all_valid_fixed_size>)
{
    VERIFY(std::is_destructible<V>::value);
    VERIFY(std::is_copy_constructible<V>::value);
    VERIFY(std::is_copy_assignable<V>::value);
}

struct dummy {};
template <class A> using dummy_datapar = datapar<dummy, A>;
template <class A> using dummy_mask = mask<dummy, A>;
template <class A> using bool_datapar = datapar<bool, A>;
template <class A> using bool_mask = mask<bool, A>;

using unusable_abis = Typelist<
#if !defined Vc_HAVE_SSE_ABI
    Template<datapar, Vc::datapar_abi::sse>, Template<mask, Vc::datapar_abi::sse>,
#endif
#if !defined Vc_HAVE_AVX_ABI
    Template<datapar, Vc::datapar_abi::avx>, Template<mask, Vc::datapar_abi::avx>,
#endif
#if !defined Vc_HAVE_AVX512_ABI
    Template<datapar, Vc::datapar_abi::avx512>, Template<mask, Vc::datapar_abi::avx512>,
#endif
#if !defined Vc_HAVE_NEON_ABI
    Template<datapar, Vc::datapar_abi::neon>, Template<mask, Vc::datapar_abi::neon>,
#endif
    Template<datapar, int>, Template<mask, int>>;

using unusable_fixed_size =
    expand_list<Typelist<Template<datapar, Vc::datapar_abi::fixed_size<33>>,
                         Template<mask, Vc::datapar_abi::fixed_size<33>>>,
                testtypes>;

using unusable_simd_types =
    concat<expand_list<Typelist<Template<datapar, Vc::datapar_abi::sse>,
                                Template<mask, Vc::datapar_abi::sse>>,
#if defined Vc_HAVE_SSE_ABI && !defined Vc_HAVE_FULL_SSE_ABI
                       typename filter_list<float, testtypes>::type
#else
                       Typelist<long double>
#endif
                       >,
           expand_list<Typelist<Template<datapar, Vc::datapar_abi::avx>,
                                Template<mask, Vc::datapar_abi::avx>>,
#if defined Vc_HAVE_AVX_ABI && !defined Vc_HAVE_FULL_AVX_ABI
                       typename filter_list<Typelist<float, double>, testtypes>::type
#else
                       Typelist<long double>
#endif
                       >,
           expand_list<Typelist<Template<datapar, Vc::datapar_abi::neon>,
                                Template<mask, Vc::datapar_abi::neon>>,
                       Typelist<long double>>,
           expand_list<Typelist<Template<datapar, Vc::datapar_abi::avx512>,
                                Template<mask, Vc::datapar_abi::avx512>>,
#if defined Vc_HAVE_AVX512_ABI && !defined Vc_HAVE_FULL_AVX512_ABI
                       typename filter_list<
                           Typelist<float, double, ullong, llong, ulong, long, uint, int>,
                           testtypes>::type
#else
                       Typelist<long double>
#endif
                       >>;

TEST_TYPES(V, is_unusable,
           concat<expand_list<unusable_abis, testtypes_wo_ldouble>, unusable_simd_types,
                  unusable_fixed_size,
                  expand_list<Typelist<Template1<dummy_datapar>, Template1<dummy_mask>,
                                       Template1<bool_datapar>, Template1<bool_mask>>,
                              all_native_abis>>)
{
    VERIFY(!std::is_destructible<V>::value);
    VERIFY(!std::is_copy_constructible<V>::value);
    VERIFY(!std::is_copy_assignable<V>::value);
}
