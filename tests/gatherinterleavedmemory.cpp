/*  This file is part of the Vc library. {{{
Copyright © 2015 Matthias Kretz <kretz@kde.org>

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


#include "unittest.h"

using namespace Vc;

template<typename T, size_t N> struct SomeStruct
{
    T d[N];
};

template<typename V, size_t StructSize, bool Random = true> struct Types
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;
    typedef typename V::AsArg VArg;
    typedef const I &IArg;
    typedef SomeStruct<T, StructSize> S;
    typedef const Vc::InterleavedMemoryWrapper<S, V> &Wrapper;
};
template<typename V, size_t StructSize> struct Types<V, StructSize, false>
{
    typedef typename V::EntryType T;
    typedef int I;
    typedef typename V::AsArg VArg;
    typedef I IArg;
    typedef SomeStruct<T, StructSize> S;
    typedef const Vc::InterleavedMemoryWrapper<S, V> &Wrapper;
};
template<typename V, size_t StructSize, bool Random, size_t N = StructSize> struct TestDeinterleaveGatherCompare;
template<typename V, size_t StructSize, bool Random> struct TestDeinterleaveGatherCompare<V, StructSize, Random, 8> {
    static void test(typename Types<V, StructSize, Random>::Wrapper data_v, typename Types<V, StructSize, Random>::IArg indexes, const typename V::AsArg reference)
    {
        V v0, v1, v2, v3, v4, v5, v6, v7;
        tie(v0, v1, v2, v3, v4, v5, v6, v7) = data_v[indexes];
        COMPARE(v0, reference + 0) << "N = 8";
        COMPARE(v1, reference + 1) << "N = 8";
        COMPARE(v2, reference + 2) << "N = 8";
        COMPARE(v3, reference + 3) << "N = 8";
        COMPARE(v4, reference + 4) << "N = 8";
        COMPARE(v5, reference + 5) << "N = 8";
        COMPARE(v6, reference + 6) << "N = 8";
        COMPARE(v7, reference + 7) << "N = 8";
        TestDeinterleaveGatherCompare<V, StructSize, Random, 7>::test(data_v, indexes, reference);
    }
};
template<typename V, size_t StructSize, bool Random> struct TestDeinterleaveGatherCompare<V, StructSize, Random, 7> {
    static void test(typename Types<V, StructSize, Random>::Wrapper data_v, typename Types<V, StructSize, Random>::IArg indexes, const typename V::AsArg reference)
    {
        V v0, v1, v2, v3, v4, v5, v6;
        tie(v0, v1, v2, v3, v4, v5, v6) = data_v[indexes];
        COMPARE(v0, reference + 0) << "N = 7";
        COMPARE(v1, reference + 1) << "N = 7";
        COMPARE(v2, reference + 2) << "N = 7";
        COMPARE(v3, reference + 3) << "N = 7";
        COMPARE(v4, reference + 4) << "N = 7";
        COMPARE(v5, reference + 5) << "N = 7";
        COMPARE(v6, reference + 6) << "N = 7";
        TestDeinterleaveGatherCompare<V, StructSize, Random, 6>::test(data_v, indexes, reference);
    }
};
template<typename V, size_t StructSize, bool Random> struct TestDeinterleaveGatherCompare<V, StructSize, Random, 6> {
    static void test(typename Types<V, StructSize, Random>::Wrapper data_v, typename Types<V, StructSize, Random>::IArg indexes, const typename V::AsArg reference)
    {
        V v0, v1, v2, v3, v4, v5;
        tie(v0, v1, v2, v3, v4, v5) = data_v[indexes];
        COMPARE(v0, reference + 0) << "N = 6";
        COMPARE(v1, reference + 1) << "N = 6";
        COMPARE(v2, reference + 2) << "N = 6";
        COMPARE(v3, reference + 3) << "N = 6";
        COMPARE(v4, reference + 4) << "N = 6";
        COMPARE(v5, reference + 5) << "N = 6";
        TestDeinterleaveGatherCompare<V, StructSize, Random, 5>::test(data_v, indexes, reference);
    }
};
template<typename V, size_t StructSize, bool Random> struct TestDeinterleaveGatherCompare<V, StructSize, Random, 5> {
    static void test(typename Types<V, StructSize, Random>::Wrapper data_v, typename Types<V, StructSize, Random>::IArg indexes, const typename V::AsArg reference)
    {
        V v0, v1, v2, v3, v4;
        tie(v0, v1, v2, v3, v4) = data_v[indexes];
        COMPARE(v0, reference + 0) << "N = 5";
        COMPARE(v1, reference + 1) << "N = 5";
        COMPARE(v2, reference + 2) << "N = 5";
        COMPARE(v3, reference + 3) << "N = 5";
        COMPARE(v4, reference + 4) << "N = 5";
        TestDeinterleaveGatherCompare<V, StructSize, Random, 4>::test(data_v, indexes, reference);
    }
};
template<typename V, size_t StructSize, bool Random> struct TestDeinterleaveGatherCompare<V, StructSize, Random, 4> {
    static void test(typename Types<V, StructSize, Random>::Wrapper data_v, typename Types<V, StructSize, Random>::IArg indexes, const typename V::AsArg reference)
    {
        V a, b, c, d;
        tie(a, b, c, d) = data_v[indexes];
        COMPARE(a, reference + 0) << "N = 4";
        COMPARE(b, reference + 1) << "N = 4";
        COMPARE(c, reference + 2) << "N = 4";
        COMPARE(d, reference + 3) << "N = 4";
        TestDeinterleaveGatherCompare<V, StructSize, Random, 3>::test(data_v, indexes, reference);
    }
};
template<typename V, size_t StructSize, bool Random> struct TestDeinterleaveGatherCompare<V, StructSize, Random, 3> {
    static void test(typename Types<V, StructSize, Random>::Wrapper data_v, typename Types<V, StructSize, Random>::IArg indexes, const typename V::AsArg reference)
    {
        V a, b, c;
        tie(a, b, c) = data_v[indexes];
        COMPARE(a, reference + 0) << "N = 3";
        COMPARE(b, reference + 1) << "N = 3";
        COMPARE(c, reference + 2) << "N = 3";
        TestDeinterleaveGatherCompare<V, StructSize, Random, 2>::test(data_v, indexes, reference);
    }
};
template<typename V, size_t StructSize, bool Random> struct TestDeinterleaveGatherCompare<V, StructSize, Random, 2> {
    static void test(typename Types<V, StructSize, Random>::Wrapper data_v, typename Types<V, StructSize, Random>::IArg indexes, const typename V::AsArg reference)
    {
        V a, b;
        tie(a, b) = data_v[indexes];
        COMPARE(a, reference + 0) << "N = 2, indexes: " << indexes;
        COMPARE(b, reference + 1) << "N = 2, indexes: " << indexes;
    }
};

size_t createNMask(size_t N)
{
    size_t NMask = (N >> 1) | (N >> 2);
    for (size_t shift = 2; shift < sizeof(size_t) * 8; shift *= 2) {
        NMask |= NMask >> shift;
    }
    return NMask;
}

#if !defined __GNUC__ || defined __OPTIMIZE__
static constexpr int TotalRetests = 10000;
#else
static constexpr int TotalRetests = 1000;
#endif

TEST_TYPES(Param, testDeinterleaveGather,
           outer_product<AllVectors, Typelist<std::integral_constant<std::size_t, 2>,
                                              std::integral_constant<std::size_t, 3>,
                                              std::integral_constant<std::size_t, 4>,
                                              std::integral_constant<std::size_t, 5>,
                                              std::integral_constant<std::size_t, 6>,
                                              std::integral_constant<std::size_t, 7>,
                                              std::integral_constant<std::size_t, 8>>>)
{
    typedef typename Param::template at<0> V;
    constexpr auto StructSize = Param::template at<1>::value;
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;
    typedef SomeStruct<T, StructSize> S;
    typedef Vc::InterleavedMemoryWrapper<S, V> Wrapper;
    const size_t N = std::min(
        // indexes * StructSize may not overflow for signed integral types. That would be
        // UB and MIC::short_v will happily use it for more performance.
        std::is_integral<T>::value && std::is_signed<T>::value
            ? static_cast<size_t>(std::numeric_limits<T>::max()) / StructSize
            : std::numeric_limits<size_t>::max(),
        std::min(static_cast<size_t>(std::numeric_limits<typename I::EntryType>::max()),
                 1024u * 1024u / sizeof(S)));
    const size_t NMask = createNMask(N);

    S *data = Vc::malloc<S, Vc::AlignOnVector>(N);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < StructSize; ++j) {
            data[i].d[j] = i * StructSize + j;
        }
    }
    const Wrapper data_v(data);

    for (int retest = 0; retest < TotalRetests; ++retest) {
        I indexes = (I::Random() >> 10) & I(NMask);
        VERIFY(all_of(indexes >= 0));
        VERIFY(all_of(indexes < int(N)));
        const V reference = Vc::simd_cast<V>(indexes) * V(StructSize);

        TestDeinterleaveGatherCompare<V, StructSize, true>::test(data_v, indexes, reference);
    }

    for (int i = 0; i < int(N - V::Size); ++i) {
        const V reference([&](int n) { return (n + i) * StructSize; });
        TestDeinterleaveGatherCompare<V, StructSize, false>::test(data_v, i, reference);
    }
}

// vim: foldmethod=marker
