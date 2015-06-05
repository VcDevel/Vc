/*  This file is part of the Vc library. {{{
Copyright © 2010-2014 Matthias Kretz <kretz@kde.org>
All rights reserved.

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
#include <iostream>
#include <limits>

using Vc::float_v;
using Vc::double_v;
using Vc::int_v;
using Vc::uint_v;
using Vc::short_v;
using Vc::ushort_v;

/*
 *   V \  M | float | double | ushort | short | uint | int
 * ---------+----------------------------------------------
 *  float_v |   X   |        |    X   |   X   |      |
 * double_v |       |    X   |        |       |      |
 *    int_v |       |        |        |   X   |      |  X
 *   uint_v |       |        |    X   |       |   X  |
 *  short_v |       |        |        |   X   |      |
 * ushort_v |       |        |    X   |       |      |
 */
typedef Typelist<float_v, float> float_float;
typedef Typelist<float_v, unsigned short> float_ushort;
typedef Typelist<float_v, short> float_short;

typedef Typelist<double_v, double> double_double;
typedef Typelist<short_v, short> short_short;
typedef Typelist<ushort_v, unsigned short> ushort_ushort;

typedef Typelist<int_v, int> int_int;
typedef Typelist<int_v, short> int_short;

typedef Typelist<uint_v, unsigned int> uint_uint;
typedef Typelist<uint_v, unsigned short> uint_ushort;

TEST_TYPES(Pair, testDeinterleave,
           (float_float, float_ushort, float_short, double_double, int_int, int_short,
            uint_uint, uint_ushort, short_short, ushort_ushort))
{
    typedef typename Pair::template at<0> V;
    typedef typename Pair::template at<1> M;
    typedef typename V::IndexType I;

    const bool isSigned = std::numeric_limits<M>::is_signed;

    const typename V::EntryType offset = isSigned ? -512 : 0;
    const V _0246 = Vc::simd_cast<V>(I::IndexesFromZero()) * 2 + offset;

    M memory[1024];
    for (int i = 0; i < 1024; ++i) {
        memory[i] = static_cast<M>(i + offset);
    }

    V a, b;

    for (size_t i = 0; i < 1024 - 2 * V::Size; ++i) {
        // note that a 32 bit integer is certainly enough to decide on alignment...
        // ... but uintptr_t is C99 but not C++ yet
        // ... and GCC refuses to do the cast, even if I know what I'm doing
        if (reinterpret_cast<unsigned long>(&memory[i]) & (Vc::VectorAlignment - 1)) {
            Vc::deinterleave(&a, &b, &memory[i], Vc::Unaligned);
        } else {
            Vc::deinterleave(&a, &b, &memory[i]);
        }
        COMPARE(_0246 + i,     a);
        COMPARE(_0246 + i + 1, b);
    }
}

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
        COMPARE(a, reference + 0) << "N = 2";
        COMPARE(b, reference + 1) << "N = 2";
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

TEST_TYPES(Param, testDeinterleaveGather,
           (outer_product<Typelist<ALL_VECTORS>,
                          Typelist<std::integral_constant<std::size_t, 2>,
                                   std::integral_constant<std::size_t, 3>,
                                   std::integral_constant<std::size_t, 4>,
                                   std::integral_constant<std::size_t, 5>,
                                   std::integral_constant<std::size_t, 6>,
                                   std::integral_constant<std::size_t, 7>,
                                   std::integral_constant<std::size_t, 8>>>))
{
    typedef typename Param::template at<0> V;
    constexpr auto StructSize = Param::template at<1>::value;
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;
    typedef SomeStruct<T, StructSize> S;
    typedef Vc::InterleavedMemoryWrapper<S, V> Wrapper;
    const size_t N = std::min<size_t>(std::numeric_limits<typename I::EntryType>::max(), 1024 * 1024 / sizeof(S));
    const size_t NMask = createNMask(N);

    S *data = Vc::malloc<S, Vc::AlignOnVector>(N);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < StructSize; ++j) {
            data[i].d[j] = i * StructSize + j;
        }
    }
    const Wrapper data_v(data);

    for (int retest = 0; retest < 10000; ++retest) {
        I indexes = (I::Random() >> 10) & I(NMask);
        VERIFY(all_of(indexes >= 0));
        VERIFY(all_of(indexes < N));
        const V reference = Vc::simd_cast<V>(indexes) * V(StructSize);

        TestDeinterleaveGatherCompare<V, StructSize, true>::test(data_v, indexes, reference);
    }

    for (int i = 0; i < int(N - V::Size); ++i) {
        const V reference = Vc::simd_cast<V>(i + I::IndexesFromZero()) * T(StructSize);
        TestDeinterleaveGatherCompare<V, StructSize, false>::test(data_v, i, reference);
    }
}

template <typename T> T rotate(T x)
{
    return x.rotated(1);
}
template <typename T, std::size_t N> Vc::simdarray<T, N> rotate(const Vc::simdarray<T, N> &x)
{
    Vc::simdarray<T, N> r;
    r[0] = x[N - 1];
    for (std::size_t i = 0; i < N - 1; ++i) {
        r[i + 1] = x[i];
    }
    return r;
}

namespace std
{
template <typename T, std::size_t N>
inline std::ostream &operator<<(std::ostream &out, const std::array<T, N> &a)
{
    out << "\narray{" << a[0];
    for (std::size_t i = 1; i < N; ++i) {
        out << ", " << a[i];
    }
    return out << '}';
}
template <typename T, std::size_t N>
inline bool operator==(const std::array<Vc::Vector<T>, N> &a, const std::array<Vc::Vector<T>, N> &b)
{
    for (std::size_t i = 0; i < N; ++i) {
        if (!Vc::all_of(a[i] == b[i])) {
            return false;
        }
    }
    return true;
}
}  // namespace std

template <typename V, typename Wrapper, typename IndexType, std::size_t... Indexes>
void testInterleavingScatterCompare(Wrapper &data, const IndexType &i,
                                    Vc::index_sequence<Indexes...>)
{
    const std::array<V, sizeof...(Indexes)> reference = {
        {(V::Random() + (Indexes - Indexes))...}};

    data[i] = tie(reference[Indexes]...);
    std::array<V, sizeof...(Indexes)> t = data[i];
    COMPARE(t, reference);

    for (auto &x : t) {
        x.setZero();
    }
    tie(t[Indexes]...) = data[i];
    COMPARE(t, reference);

    if (sizeof...(Indexes) > 2) {
        testInterleavingScatterCompare<V>(
            data, i, Vc::make_index_sequence<
                         (sizeof...(Indexes) > 2 ? sizeof...(Indexes) - 1 : 2)>());
    }
}

TEST_TYPES(Param, testInterleavingScatter,
           (outer_product<Typelist<ALL_VECTORS>,
                          Typelist<std::integral_constant<std::size_t, 2>,
                                   std::integral_constant<std::size_t, 3>,
                                   std::integral_constant<std::size_t, 4>,
                                   std::integral_constant<std::size_t, 5>,
                                   std::integral_constant<std::size_t, 6>,
                                   std::integral_constant<std::size_t, 7>,
                                   std::integral_constant<std::size_t, 8>>>))
{
    typedef typename Param::template at<0> V;
    constexpr auto StructSize = Param::template at<1>::value;
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;
    typedef SomeStruct<T, StructSize> S;
    typedef Vc::InterleavedMemoryWrapper<S, V> Wrapper;
    const size_t N = std::min<size_t>(std::numeric_limits<typename I::EntryType>::max(), 1024 * 1024 / sizeof(S));
    const size_t NMask = createNMask(N);

    S *data = Vc::malloc<S, Vc::AlignOnVector>(N);
    std::memset(data, 0, sizeof(S) * N);
    Wrapper data_v(data);

    for (int retest = 0; retest < 10000; ++retest) {
        I indexes = (I::Random() >> 10) & I(NMask);
        if (I::Size != 1) {
            // ensure the indexes are unique
            while(any_of(indexes.sorted() == rotate(indexes.sorted()))) {
                indexes = (I::Random() >> 10) & I(NMask);
            }
        }
        VERIFY(all_of(indexes >= 0));
        VERIFY(all_of(indexes < N));

        testInterleavingScatterCompare<V>(data_v, indexes,
                                          Vc::make_index_sequence<StructSize>());
    }

    for (size_t i = 0; i < N - V::Size; ++i) {
        testInterleavingScatterCompare<V>(data_v, i,
                                          Vc::make_index_sequence<StructSize>());
    }
}
