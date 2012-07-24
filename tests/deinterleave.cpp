/*  This file is part of the Vc library.

    Copyright (C) 2010-2011 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with Vc.  If not, see <http://www.gnu.org/licenses/>.

*/

#include <Vc/Vc>
#include "unittest.h"
#include <iostream>
#include <limits>

using namespace Vc;


/*
 *   V \  M | float | double | ushort | short | uint | int
 * ---------+----------------------------------------------
 *  float_v |   X   |        |    X   |   X   |      |
 * sfloat_v |   X   |        |    X   |   X   |      |
 * double_v |       |    X   |        |       |      |
 *    int_v |       |        |        |   X   |      |  X
 *   uint_v |       |        |    X   |       |   X  |
 *  short_v |       |        |        |   X   |      |
 * ushort_v |       |        |    X   |       |      |
 */
template<typename A, typename B> struct TPair { typedef A V; typedef B M; };

typedef TPair<float_v, float> float_float;
typedef TPair<float_v, unsigned short> float_ushort;
typedef TPair<float_v, short> float_short;

typedef TPair<sfloat_v, float> sfloat_float;
typedef TPair<sfloat_v, unsigned short> sfloat_ushort;
typedef TPair<sfloat_v, short> sfloat_short;

typedef TPair<double_v, double> double_double;
typedef TPair<short_v, short> short_short;
typedef TPair<ushort_v, unsigned short> ushort_ushort;

typedef TPair<int_v, int> int_int;
typedef TPair<int_v, short> int_short;

typedef TPair<uint_v, unsigned int> uint_uint;
typedef TPair<uint_v, unsigned short> uint_ushort;

template<typename Pair> void testDeinterleave()
{
    typedef typename Pair::V V;
    typedef typename Pair::M M;
    typedef typename V::IndexType I;

    const bool isSigned = std::numeric_limits<M>::is_signed;

    const typename V::EntryType offset = isSigned ? -512 : 0;
    const V _0246 = static_cast<V>(I::IndexesFromZero()) * 2 + offset;

    M memory[1024];
    for (int i = 0; i < 1024; ++i) {
        memory[i] = static_cast<M>(i + offset);
    }

    V a, b;

    for (int i = 0; i < 1024 - 2 * V::Size; ++i) {
        // note that a 32 bit integer is certainly enough to decide on alignment...
        // ... but uintptr_t is C99 but not C++ yet
        // ... and GCC refuses to do the cast, even if I know what I'm doing
        if (reinterpret_cast<unsigned long>(&memory[i]) & (VectorAlignment - 1)) {
            Vc::deinterleave(&a, &b, &memory[i], Unaligned);
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

template<typename V, size_t StructSize> struct Types
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;
    typedef SomeStruct<T, StructSize> S;
    typedef const Vc::InterleavedMemoryWrapper<S, V> &Wrapper;
};
template<typename V, size_t StructSize, size_t N = StructSize> struct TestDeinterleaveGatherCompare;
template<typename V, size_t StructSize> struct TestDeinterleaveGatherCompare<V, StructSize, 8> {
    static void test(typename Types<V, StructSize>::Wrapper data_v, typename Types<V, StructSize>::I indexes, const V reference)
    {
        V v0, v1, v2, v3, v4, v5, v6, v7;
        (v0, v1, v2, v3, v4, v5, v6, v7) = data_v[indexes];
        COMPARE(v0, reference + 0) << "N = 8";
        COMPARE(v1, reference + 1) << "N = 8";
        COMPARE(v2, reference + 2) << "N = 8";
        COMPARE(v3, reference + 3) << "N = 8";
        COMPARE(v4, reference + 4) << "N = 8";
        COMPARE(v5, reference + 5) << "N = 8";
        COMPARE(v6, reference + 6) << "N = 8";
        COMPARE(v7, reference + 7) << "N = 8";
        TestDeinterleaveGatherCompare<V, StructSize, 7>::test(data_v, indexes, reference);
    }
};
template<typename V, size_t StructSize> struct TestDeinterleaveGatherCompare<V, StructSize, 7> {
    static void test(typename Types<V, StructSize>::Wrapper data_v, typename Types<V, StructSize>::I indexes, const V reference)
    {
        V v0, v1, v2, v3, v4, v5, v6;
        (v0, v1, v2, v3, v4, v5, v6) = data_v[indexes];
        COMPARE(v0, reference + 0) << "N = 7";
        COMPARE(v1, reference + 1) << "N = 7";
        COMPARE(v2, reference + 2) << "N = 7";
        COMPARE(v3, reference + 3) << "N = 7";
        COMPARE(v4, reference + 4) << "N = 7";
        COMPARE(v5, reference + 5) << "N = 7";
        COMPARE(v6, reference + 6) << "N = 7";
        TestDeinterleaveGatherCompare<V, StructSize, 6>::test(data_v, indexes, reference);
    }
};
template<typename V, size_t StructSize> struct TestDeinterleaveGatherCompare<V, StructSize, 6> {
    static void test(typename Types<V, StructSize>::Wrapper data_v, typename Types<V, StructSize>::I indexes, const V reference)
    {
        V v0, v1, v2, v3, v4, v5;
        (v0, v1, v2, v3, v4, v5) = data_v[indexes];
        COMPARE(v0, reference + 0) << "N = 6";
        COMPARE(v1, reference + 1) << "N = 6";
        COMPARE(v2, reference + 2) << "N = 6";
        COMPARE(v3, reference + 3) << "N = 6";
        COMPARE(v4, reference + 4) << "N = 6";
        COMPARE(v5, reference + 5) << "N = 6";
        TestDeinterleaveGatherCompare<V, StructSize, 5>::test(data_v, indexes, reference);
    }
};
template<typename V, size_t StructSize> struct TestDeinterleaveGatherCompare<V, StructSize, 5> {
    static void test(typename Types<V, StructSize>::Wrapper data_v, typename Types<V, StructSize>::I indexes, const V reference)
    {
        V v0, v1, v2, v3, v4;
        (v0, v1, v2, v3, v4) = data_v[indexes];
        COMPARE(v0, reference + 0) << "N = 5";
        COMPARE(v1, reference + 1) << "N = 5";
        COMPARE(v2, reference + 2) << "N = 5";
        COMPARE(v3, reference + 3) << "N = 5";
        COMPARE(v4, reference + 4) << "N = 5";
        TestDeinterleaveGatherCompare<V, StructSize, 4>::test(data_v, indexes, reference);
    }
};
template<typename V, size_t StructSize> struct TestDeinterleaveGatherCompare<V, StructSize, 4> {
    static void test(typename Types<V, StructSize>::Wrapper data_v, typename Types<V, StructSize>::I indexes, const V reference)
    {
        V a, b, c, d;
        (a, b, c, d) = data_v[indexes];
        COMPARE(a, reference + 0) << "N = 4";
        COMPARE(b, reference + 1) << "N = 4";
        COMPARE(c, reference + 2) << "N = 4";
        COMPARE(d, reference + 3) << "N = 4";
        TestDeinterleaveGatherCompare<V, StructSize, 3>::test(data_v, indexes, reference);
    }
};
template<typename V, size_t StructSize> struct TestDeinterleaveGatherCompare<V, StructSize, 3> {
    static void test(typename Types<V, StructSize>::Wrapper data_v, typename Types<V, StructSize>::I indexes, const V reference)
    {
        V a, b, c;
        (a, b, c) = data_v[indexes];
        COMPARE(a, reference + 0) << "N = 3";
        COMPARE(b, reference + 1) << "N = 3";
        COMPARE(c, reference + 2) << "N = 3";
        TestDeinterleaveGatherCompare<V, StructSize, 2>::test(data_v, indexes, reference);
    }
};
template<typename V, size_t StructSize> struct TestDeinterleaveGatherCompare<V, StructSize, 2> {
    static void test(typename Types<V, StructSize>::Wrapper data_v, typename Types<V, StructSize>::I indexes, const V reference)
    {
        V a, b;
        (a, b) = data_v[indexes];
        COMPARE(a, reference + 0) << "N = 2";
        COMPARE(b, reference + 1) << "N = 2";
    }
};

template<typename V, size_t StructSize> void testDeinterleaveGatherImpl()
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;
    typedef SomeStruct<T, StructSize> S;
    typedef Vc::InterleavedMemoryWrapper<S, V> Wrapper;
    const size_t N = std::min<size_t>(std::numeric_limits<typename I::EntryType>::max(), 1024 * 1024 / sizeof(S));

    S *data = Vc::malloc<S, Vc::AlignOnVector>(N);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < StructSize; ++j) {
            data[i].d[j] = i * StructSize + j;
        }
    }
    const Wrapper data_v(data);

    for (int retest = 0; retest < 10000; ++retest) {
        I indexes = (I::Random() >> 10) & I(N - 1);
        VERIFY(indexes >= 0);
        VERIFY(indexes < N);
        const V reference = static_cast<V>(indexes) * V(StructSize);

        TestDeinterleaveGatherCompare<V, StructSize>::test(data_v, indexes, reference);
    }
}

template<typename V> void testDeinterleaveGather()
{
    testDeinterleaveGatherImpl<V, 2>();
    testDeinterleaveGatherImpl<V, 3>();
    testDeinterleaveGatherImpl<V, 4>();
    testDeinterleaveGatherImpl<V, 5>();
    testDeinterleaveGatherImpl<V, 6>();
    testDeinterleaveGatherImpl<V, 7>();
    testDeinterleaveGatherImpl<V, 8>();
}

int main()
{
    runTest(testDeinterleave<float_float>);
    runTest(testDeinterleave<float_ushort>);
    runTest(testDeinterleave<float_short>);
    runTest(testDeinterleave<sfloat_float>);
    runTest(testDeinterleave<sfloat_ushort>);
    runTest(testDeinterleave<sfloat_short>);
    runTest(testDeinterleave<double_double>);
    runTest(testDeinterleave<int_int>);
    runTest(testDeinterleave<int_short>);
    runTest(testDeinterleave<uint_uint>);
    runTest(testDeinterleave<uint_ushort>);
    runTest(testDeinterleave<short_short>);
    runTest(testDeinterleave<ushort_ushort>);

    testAllTypes(testDeinterleaveGather);
}
