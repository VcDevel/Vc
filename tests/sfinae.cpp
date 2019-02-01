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
#include "unittest.h"
#include "metahelpers.h"

#define TESTTYPES                                                                        \
    long double, double, float, unsigned long long, long long, unsigned long, long,      \
        unsigned int, int, unsigned short, short, unsigned char, signed char, char,      \
        wchar_t, char16_t, char32_t
template <class... Ts> using base_template = std::experimental::simd<Ts...>;
#include "testtypes.h"

using vir::Typelist;
using vir::Template;
using vir::Template1;
using vir::expand_list;
using vir::filter_list;
using vir::concat;
using std::experimental::simd;
using std::experimental::simd_mask;

struct dummy {};
template <class A> using dummy_simd = simd<dummy, A>;
template <class A> using dummy_mask = simd_mask<dummy, A>;
template <class A> using bool_simd = simd<bool, A>;
template <class A> using bool_mask = simd_mask<bool, A>;

namespace assertions
{
using std::experimental::simd_abi::scalar;
using std::experimental::simd_abi::__sse;
using std::experimental::simd_abi::__avx;
using std::experimental::simd_abi::__avx512;
using std::experimental::__fixed_size_storage_t;
using std::experimental::_SimdTuple;

static_assert(std::is_same_v<__fixed_size_storage_t<float, 1>, _SimdTuple<float, scalar>>);
static_assert(std::is_same_v<__fixed_size_storage_t<int, 1>, _SimdTuple<int, scalar>>);
static_assert(std::is_same_v<__fixed_size_storage_t<char16_t, 1>, _SimdTuple<char16_t, scalar>>);

static_assert(std::is_same_v<__fixed_size_storage_t<float, 2>, _SimdTuple<float, scalar, scalar>>);
static_assert(std::is_same_v<__fixed_size_storage_t<float, 3>, _SimdTuple<float, scalar, scalar, scalar>>);
#if _GLIBCXX_SIMD_HAVE_SSE_ABI
static_assert(std::is_same_v<__fixed_size_storage_t<float, 4>, _SimdTuple<float, __sse>>);
static_assert(std::is_same_v<__fixed_size_storage_t<float, 5>, _SimdTuple<float, __sse, scalar>>);
#endif  // _GLIBCXX_SIMD_HAVE_SSE_ABI
#if _GLIBCXX_SIMD_HAVE_AVX_ABI
static_assert(std::is_same_v<__fixed_size_storage_t<float,  8>, _SimdTuple<float, __avx>>);
static_assert(std::is_same_v<__fixed_size_storage_t<float, 12>, _SimdTuple<float, __avx, __sse>>);
static_assert(std::is_same_v<__fixed_size_storage_t<float, 13>, _SimdTuple<float, __avx, __sse, scalar>>);
#endif
#if _GLIBCXX_SIMD_HAVE_AVX512_ABI
static_assert(std::is_same_v<__fixed_size_storage_t<float, 16>, _SimdTuple<float, __avx512>>);
static_assert(std::is_same_v<__fixed_size_storage_t<float, 20>, _SimdTuple<float, __avx512, __sse>>);
static_assert(std::is_same_v<__fixed_size_storage_t<float, 24>, _SimdTuple<float, __avx512, __avx>>);
static_assert(std::is_same_v<__fixed_size_storage_t<float, 28>, _SimdTuple<float, __avx512, __avx, __sse>>);
static_assert(std::is_same_v<__fixed_size_storage_t<float, 29>, _SimdTuple<float, __avx512, __avx, __sse, scalar>>);
#endif
}  // namespace assertions

// type lists {{{1
using all_valid_scalars = expand_list<Typelist<Template<simd, std::experimental::simd_abi::scalar>,
                                               Template<simd_mask, std::experimental::simd_abi::scalar>>,
                                      testtypes>;

using all_valid_fixed_size = expand_list<
    concat<
        VIR_CHOOSE_ONE_RANDOMLY(Typelist<Template<simd, std::experimental::simd_abi::fixed_size<1>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<2>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<3>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<4>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<5>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<6>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<7>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<8>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<9>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<10>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<11>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<12>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<13>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<14>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<15>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<16>>>),
        VIR_CHOOSE_ONE_RANDOMLY(Typelist<Template<simd, std::experimental::simd_abi::fixed_size<17>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<18>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<19>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<20>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<21>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<22>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<23>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<24>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<25>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<26>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<27>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<28>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<29>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<30>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<31>>,
                                         Template<simd, std::experimental::simd_abi::fixed_size<32>>>),
        VIR_CHOOSE_ONE_RANDOMLY(
            Typelist<Template<simd_mask, std::experimental::simd_abi::fixed_size<1>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<2>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<3>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<4>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<5>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<6>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<7>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<8>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<9>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<10>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<11>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<12>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<13>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<14>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<15>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<16>>>),
        VIR_CHOOSE_ONE_RANDOMLY(
            Typelist<Template<simd_mask, std::experimental::simd_abi::fixed_size<17>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<18>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<19>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<20>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<21>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<22>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<23>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<24>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<25>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<26>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<27>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<28>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<29>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<30>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<31>>,
                     Template<simd_mask, std::experimental::simd_abi::fixed_size<32>>>)>,
    testtypes>;

using all_valid_simd = concat<
#if _GLIBCXX_SIMD_HAVE_FULL_SSE_ABI
    expand_list<Typelist<Template<simd, std::experimental::simd_abi::__sse>,
                         Template<simd_mask, std::experimental::simd_abi::__sse>>,
                testtypes_wo_ldouble>,
#elif _GLIBCXX_SIMD_HAVE_SSE_ABI
    expand_list<Typelist<Template<simd, std::experimental::simd_abi::__sse>,
                         Template<simd_mask, std::experimental::simd_abi::__sse>>,
                testtypes_float>,
#endif
#if _GLIBCXX_SIMD_HAVE_FULL_AVX_ABI
    expand_list<Typelist<Template<simd, std::experimental::simd_abi::__avx>,
                         Template<simd_mask, std::experimental::simd_abi::__avx>>,
                testtypes_wo_ldouble>,
#elif _GLIBCXX_SIMD_HAVE_AVX_ABI
    expand_list<Typelist<Template<simd, std::experimental::simd_abi::__avx>,
                         Template<simd_mask, std::experimental::simd_abi::__avx>>,
                testtypes_fp>,
#endif
#if _GLIBCXX_SIMD_HAVE_FULL_AVX512_ABI
    expand_list<Typelist<Template<simd, std::experimental::simd_abi::__avx512>,
                         Template<simd_mask, std::experimental::simd_abi::__avx512>>,
                testtypes_wo_ldouble>,
#elif _GLIBCXX_SIMD_HAVE_AVX512_ABI
    expand_list<Typelist<Template<simd, std::experimental::simd_abi::__avx512>,
                         Template<simd_mask, std::experimental::simd_abi::__avx512>>,
                testtypes_64_32>,
#endif
#if _GLIBCXX_SIMD_HAVE_FULL_NEON_ABI
    expand_list<Typelist<Template<simd, std::experimental::simd_abi::__neon>,
                         Template<simd_mask, std::experimental::simd_abi::__neon>>,
                testtypes_wo_ldouble>,
#endif
    Typelist<>>;

using all_native_simd_types = expand_list<
    Typelist<Template<simd, std::experimental::simd_abi::__sse>, Template<simd, std::experimental::simd_abi::__avx>,
             Template<simd, std::experimental::simd_abi::__avx512>, Template<simd, std::experimental::simd_abi::__neon>>,
    testtypes>;

TEST_TYPES(V, has_size, all_native_simd_types)  //{{{1
{
    VERIFY((std::experimental::simd_size_v<typename V::value_type, typename V::abi_type>) > 0);
}

TEST_TYPES(Tup, has_no_size,  //{{{1
    concat<outer_product<concat<testtypes, nullptr_t, dummy>, Typelist<int, dummy>>,
           outer_product<Typelist<nullptr_t, dummy>,
                         Typelist<std::experimental::simd_abi::scalar, std::experimental::simd_abi::fixed_size<4>,
                                  std::experimental::simd_abi::__sse, std::experimental::simd_abi::__avx,
                                  std::experimental::simd_abi::__avx512, std::experimental::simd_abi::__neon>>>)
{
    VERIFY(
        !(sfinae_is_callable<typename Tup::template at<0>, typename Tup::template at<1>>(
            [](auto a, auto b) -> decltype(
                std::experimental::simd_size<decltype(a), decltype(b)>::type) { return {}; })));
}

template <class T> constexpr bool is_fixed_size_mask(T) { return false; }
template <class T, int N> constexpr bool is_fixed_size_mask(std::experimental::fixed_size_simd_mask<T, N>)
{
    return true;
}

TEST_TYPES(V, is_usable,  //{{{1
           concat<all_valid_scalars, all_valid_simd, all_valid_fixed_size>)
{
    if (!is_fixed_size_mask(V())) {
        // fixed_size_simd_mask uses std::bitset for storage, which is not trivially
        // constructible
        // Actually, is_trivially_constructible is not a hard requirement by the spec, but
        // something we want to support AFAIP.
        VERIFY(std::is_trivially_constructible<V>::value);
    }
    VERIFY(std::is_destructible<V>::value);
    VERIFY(std::is_copy_constructible<V>::value);
    VERIFY(std::is_copy_assignable<V>::value);
}

using unusable_abis = Typelist<
#if !_GLIBCXX_SIMD_HAVE_SSE_ABI
    Template<simd, std::experimental::simd_abi::__sse>, Template<simd_mask, std::experimental::simd_abi::__sse>,
#endif
#if !_GLIBCXX_SIMD_HAVE_AVX_ABI
    Template<simd, std::experimental::simd_abi::__avx>, Template<simd_mask, std::experimental::simd_abi::__avx>,
#endif
#if !_GLIBCXX_SIMD_HAVE_AVX512_ABI
    Template<simd, std::experimental::simd_abi::__avx512>, Template<simd_mask, std::experimental::simd_abi::__avx512>,
#endif
#if !_GLIBCXX_SIMD_HAVE_NEON_ABI
    Template<simd, std::experimental::simd_abi::__neon>, Template<simd_mask, std::experimental::simd_abi::__neon>,
#endif
    Template<simd, int>, Template<simd_mask, int>>;

using unusable_fixed_size =
    expand_list<Typelist<Template<simd, std::experimental::simd_abi::fixed_size<33>>,
                         Template<simd_mask, std::experimental::simd_abi::fixed_size<33>>>,
                testtypes>;

using unusable_simd_types =
    concat<expand_list<Typelist<Template<simd, std::experimental::simd_abi::__sse>,
                                Template<simd_mask, std::experimental::simd_abi::__sse>>,
#if _GLIBCXX_SIMD_HAVE_SSE_ABI && !_GLIBCXX_SIMD_HAVE_FULL_SSE_ABI
                       typename filter_list<float, testtypes>::type
#else
                       Typelist<long double>
#endif
                       >,
           expand_list<Typelist<Template<simd, std::experimental::simd_abi::__avx>,
                                Template<simd_mask, std::experimental::simd_abi::__avx>>,
#if _GLIBCXX_SIMD_HAVE_AVX_ABI && !_GLIBCXX_SIMD_HAVE_FULL_AVX_ABI
                       typename filter_list<Typelist<float, double>, testtypes>::type
#else
                       Typelist<long double>
#endif
                       >,
           expand_list<Typelist<Template<simd, std::experimental::simd_abi::__neon>,
                                Template<simd_mask, std::experimental::simd_abi::__neon>>,
                       Typelist<long double>>,
           expand_list<Typelist<Template<simd, std::experimental::simd_abi::__avx512>,
                                Template<simd_mask, std::experimental::simd_abi::__avx512>>,
#if _GLIBCXX_SIMD_HAVE_AVX512_ABI && !_GLIBCXX_SIMD_HAVE_FULL_AVX512_ABI
                       typename filter_list<
                           Typelist<float, double, ullong, llong, ulong, long, uint, int,
#if WCHAR_MAX > 0xffff
                                    wchar_t,
#endif
                                    char32_t>,
                           testtypes>::type
#else
                       Typelist<long double>
#endif
                       >>;

TEST_TYPES(V, is_unusable,  //{{{1
           concat<expand_list<unusable_abis, testtypes_wo_ldouble>, unusable_simd_types,
                  unusable_fixed_size,
                  expand_list<Typelist<Template1<dummy_simd>, Template1<dummy_mask>,
                                       Template1<bool_simd>, Template1<bool_mask>>,
                              all_native_abis>>)
{
    VERIFY(!std::is_constructible<V>::value);
    VERIFY(!std::is_destructible<V>::value);
    VERIFY(!std::is_copy_constructible<V>::value);
    VERIFY(!std::is_copy_assignable<V>::value);
}

// loadstore_pointer_types {{{1
struct call_memload {
    template <class V, class T>
    auto operator()(V &&v, const T *mem)
        -> decltype(v.copy_from(mem, std::experimental::element_aligned));
};
struct call_masked_memload {
    template <class M, class V, class T>
    auto operator()(const M &k, V &&v, const T *mem)
        -> decltype(std::experimental::where(k, v).copy_from(mem, std::experimental::element_aligned));
};
struct call_memstore {
    template <class V, class T>
    auto operator()(V &&v, T *mem)
        -> decltype(v.copy_to(mem, std::experimental::element_aligned));
};
struct call_masked_memstore {
    template <class M, class V, class T>
    auto operator()(const M &k, V &&v, T *mem)
        -> decltype(std::experimental::where(k, v).copy_to(mem, std::experimental::element_aligned));
};
TEST_TYPES(V, loadstore_pointer_types, all_test_types)
{
    using vir::test::sfinae_is_callable;
    using M = typename V::mask_type;
    struct Foo {
    };
    VERIFY( (sfinae_is_callable<V &, const int *>(call_memload())));
    VERIFY( (sfinae_is_callable<V &, const float *>(call_memload())));
    VERIFY(!(sfinae_is_callable<V &, const bool *>(call_memload())));
    VERIFY(!(sfinae_is_callable<V &, const Foo *>(call_memload())));
    VERIFY( (sfinae_is_callable<const V &, int *>(call_memstore())));
    VERIFY( (sfinae_is_callable<const V &, float *>(call_memstore())));
    VERIFY(!(sfinae_is_callable<const V &, bool *>(call_memstore())));
    VERIFY(!(sfinae_is_callable<const V &, Foo *>(call_memstore())));

    VERIFY( (sfinae_is_callable<M, const V &, const int *>(call_masked_memload())));
    VERIFY( (sfinae_is_callable<M, const V &, const float *>(call_masked_memload())));
    VERIFY(!(sfinae_is_callable<M, const V &, const Foo *>(call_masked_memload())));
    VERIFY( (sfinae_is_callable<M, const V &, int *>(call_masked_memstore())));
    VERIFY( (sfinae_is_callable<M, const V &, float *>(call_masked_memstore())));
    VERIFY(!(sfinae_is_callable<M, const V &, Foo *>(call_masked_memstore())));

    VERIFY( (sfinae_is_callable<M, V &, const int *>(call_masked_memload())));
    VERIFY( (sfinae_is_callable<M, V &, const float *>(call_masked_memload())));
    VERIFY(!(sfinae_is_callable<M, V &, const Foo *>(call_masked_memload())));

    VERIFY( (sfinae_is_callable<M &, const bool *>(call_memload())));
    VERIFY(!(sfinae_is_callable<M &, const int *>(call_memload())));
    VERIFY(!(sfinae_is_callable<M &, const Foo *>(call_memload())));
    VERIFY( (sfinae_is_callable<M &, bool *>(call_memstore())));
    VERIFY(!(sfinae_is_callable<M &, int *>(call_memstore())));
    VERIFY(!(sfinae_is_callable<M &, Foo *>(call_memstore())));

    VERIFY( (sfinae_is_callable<M, M &, const bool *>(call_masked_memload())));
    VERIFY(!(sfinae_is_callable<M, M &, const int *>(call_masked_memload())));
    VERIFY(!(sfinae_is_callable<M, M &, const Foo *>(call_masked_memload())));
    VERIFY( (sfinae_is_callable<M, M &, bool *>(call_masked_memstore())));
    VERIFY(!(sfinae_is_callable<M, M &, int *>(call_masked_memstore())));
    VERIFY(!(sfinae_is_callable<M, M &, Foo *>(call_masked_memstore())));

    VERIFY( (sfinae_is_callable<M, const M &, const bool *>(call_masked_memload())));
    VERIFY(!(sfinae_is_callable<M, const M &, const int *>(call_masked_memload())));
    VERIFY(!(sfinae_is_callable<M, const M &, const Foo *>(call_masked_memload())));
    VERIFY( (sfinae_is_callable<M, const M &, bool *>(call_masked_memstore())));
    VERIFY(!(sfinae_is_callable<M, const M &, int *>(call_masked_memstore())));
    VERIFY(!(sfinae_is_callable<M, const M &, Foo *>(call_masked_memstore())));
}

TEST(masked_loadstore_builtin) {
    VERIFY( (sfinae_is_callable<bool, const int &, const int *>(call_masked_memload())));
    VERIFY( (sfinae_is_callable<bool, const int &, const float *>(call_masked_memload())));
    VERIFY( (sfinae_is_callable<bool, const bool &, const bool *>(call_masked_memload())));
    VERIFY(!(sfinae_is_callable<bool, const bool &, const int *>(call_masked_memload())));
    VERIFY( (sfinae_is_callable<bool, const int &, int *>(call_masked_memstore())));
    VERIFY( (sfinae_is_callable<bool, const int &, float *>(call_masked_memstore())));
    VERIFY( (sfinae_is_callable<bool, const bool &, bool *>(call_masked_memstore())));
    VERIFY(!(sfinae_is_callable<bool, const bool &, int *>(call_masked_memstore())));

    VERIFY( (sfinae_is_callable<bool, int &, const int *>(call_masked_memload())));
    VERIFY( (sfinae_is_callable<bool, int &, const float *>(call_masked_memload())));
    VERIFY( (sfinae_is_callable<bool, bool &, const bool *>(call_masked_memload())));
    VERIFY(!(sfinae_is_callable<bool, bool &, const int *>(call_masked_memload())));
}

TEST(deduce_broken)
{
    VERIFY(!(sfinae_is_callable<bool>([](auto a) -> std::experimental::simd_abi::deduce_t<decltype(a), 1> { return {}; })));
    VERIFY(!(sfinae_is_callable<bool>([](auto a) -> std::experimental::simd_abi::deduce_t<decltype(a), 2> { return {}; })));
    VERIFY(!(sfinae_is_callable<bool>([](auto a) -> std::experimental::simd_abi::deduce_t<decltype(a), 4> { return {}; })));
    VERIFY(!(sfinae_is_callable<bool>([](auto a) -> std::experimental::simd_abi::deduce_t<decltype(a), 8> { return {}; })));
    enum Foo {};
    VERIFY(!(sfinae_is_callable<Foo>([](auto a) -> std::experimental::simd_abi::deduce_t<decltype(a), 1> { return {}; })));
    VERIFY(!(sfinae_is_callable<Foo>([](auto a) -> std::experimental::simd_abi::deduce_t<decltype(a), 8> { return {}; })));
}

TEST_TYPES(V, deduce_from_list, all_test_types)
{
    using T = typename V::value_type;
    using A = typename V::abi_type;
    VERIFY( (sfinae_is_callable<T>([](auto a) -> std::experimental::simd_abi::deduce_t<decltype(a), 1> { return {}; })));
    VERIFY( (sfinae_is_callable<T>([](auto a) -> std::experimental::simd_abi::deduce_t<decltype(a), 2> { return {}; })));
    VERIFY( (sfinae_is_callable<T>([](auto a) -> std::experimental::simd_abi::deduce_t<decltype(a), 4> { return {}; })));
    VERIFY( (sfinae_is_callable<T>([](auto a) -> std::experimental::simd_abi::deduce_t<decltype(a), 8> { return {}; })));
    using W = std::experimental::simd_abi::deduce_t<T, V::size(), typename V::abi_type>;
    VERIFY((sfinae_is_callable<W>([](auto a) -> std::experimental::simd<T, W> { return {}; })));
    if constexpr (std::experimental::__is_fixed_size_abi_v<A>) {
        VERIFY((V::size() == std::experimental::simd_size_v<T, W>)) << vir::typeToString<W>();
    } else {
        VERIFY((std::is_same_v<A, W>)) << vir::typeToString<W>();
    }
}

// vim: foldmethod=marker
