/*  This file is part of the Vc library.

    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

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
#include "vectormemoryhelper.h"
#include <cmath>
#include <algorithm>

using namespace Vc;

#ifdef isfinite
#undef isfinite
#endif
#ifdef isnan
#undef isnan
#endif

template<typename Vec> void testAbs()
{
    for (int i = 0; i < 0x7fff; ++i) {
        Vec a(i);
        Vec b(-i);
        COMPARE(a, Vc::abs(a));
        COMPARE(a, Vc::abs(b));
    }
}

template<typename Vec> void testLog()
{
    setFuzzyness<float>(1.2e-7f);
    setFuzzyness<double>(3e-16);
    typedef typename Vec::EntryType T;
    typedef typename Vec::IndexType I;
    const I indexesFromZero(IndexesFromZero);
    Vec a(indexesFromZero);
    a *= T(0.1);
    const Vec end(1000);
    for (; a < end; a += Vec::Size) {
        Vec b = Vc::log(a);

        const typename Vec::EntryType two = 2.;

        for (int i = 0; i < Vec::Size; ++i) {
            FUZZY_COMPARE(static_cast<T>(b[i]), static_cast<T>(std::log(a[i])));
        }

        const Vec a2 = a * a;
        FUZZY_COMPARE(Vc::log(a2), two * Vc::log(a));
    }
    setFuzzyness<float>(0.f);
    setFuzzyness<double>(0.);
}

template<typename Vec>
void testMax()
{
    typedef typename Vec::EntryType T;
    VectorMemoryHelper<Vec> mem(3);
    T *data = mem;
    for (int i = 0; i < Vec::Size; ++i) {
        data[i] = i;
        data[i + Vec::Size] = Vec::Size + 1 - i;
        data[i + 2 * Vec::Size] = std::max(data[i], data[i + Vec::Size]);
    }
    Vec a(&data[0]);
    Vec b(&data[Vec::Size]);
    Vec c(&data[2 * Vec::Size]);

    COMPARE(Vc::max(a, b), c);
}

#define FillHelperMemory(code) \
    typename Vec::Memory data; \
    typename Vec::Memory reference; \
    for (int ii = 0; ii < Vec::Size; ++ii) { \
        const T i = static_cast<T>(ii); \
        data[ii] = i; \
        reference[ii] = code; \
    } do {} while (false)

template<typename Vec> void testSqrt()
{
    typedef typename Vec::EntryType T;
    FillHelperMemory(std::sqrt(i));
    Vec a(data);
    Vec b(reference);

    FUZZY_COMPARE(Vc::sqrt(a), b);
}

template<typename Vec> void testRSqrt()
{
    typedef typename Vec::EntryType T;
    const T one = 1;
    FillHelperMemory(one / std::sqrt(i));
    Vec a(data);
    Vec b(reference);

    // RSQRTPS is documented as having a relative error <= 1.5 * 2^-12
    setFuzzyness<float>(0.0003662109375f);
    FUZZY_COMPARE(Vc::rsqrt(a), b);
    setFuzzyness<float>(0.f);
}

template<typename T> void my_sincos(T x, T *sin, T *cos);
template<> void my_sincos<double>(double x, double *sin, double *cos)
{
    sincos(x, sin, cos);
}
template<> void my_sincos<float>(float x, float *sin, float *cos)
{
    sincosf(x, sin, cos);
}

template<typename Vec> void testSincos()
{
    typedef typename Vec::EntryType T;
    setFuzzyness<double>(4e-6);
    typename Vec::Memory base;
    for (int i = 0; i < Vec::Size; ++i) {
        base[i] = static_cast<T>(i);
    }
    for (int offset = -1000; offset < 1000 - Vec::Size; offset += Vec::Size) {
        const T scale = T(0.01);
        Vec sin, cos;
        Vc::sincos((base.vector(0) + offset) * scale, &sin, &cos);

        for (int i = 0; i < Vec::Size; ++i) {
            T scalarSin, scalarCos;
            my_sincos((i + offset) * scale, &scalarSin, &scalarCos);
            setFuzzyness<float>(6e-5f);
            FUZZY_COMPARE(T(sin[i]), scalarSin);
            setFuzzyness<float>(2.1e-4f);
            FUZZY_COMPARE(T(cos[i]), scalarCos);
        }
    }
}

template<typename Vec> void testSin()
{
    typedef typename Vec::EntryType T;
    setFuzzyness<float>(6e-5f);
    setFuzzyness<double>(4e-6);
    for (int offset = -1000; offset < 1000 - Vec::Size; offset += Vec::Size) {
        const T scale = T(0.01);
        FillHelperMemory(std::sin((i + offset) * scale));
        Vec a(data);
        Vec b(reference);

        FUZZY_COMPARE(Vc::sin((a + offset) * scale), b);
    }
}

template<typename Vec> void testCos()
{
    typedef typename Vec::EntryType T;
    setFuzzyness<float>(2.1e-4f);
    setFuzzyness<double>(4e-6);
    for (int offset = -1000; offset < 1000 - Vec::Size; offset += Vec::Size) {
        const T scale = T(0.01);
        FillHelperMemory(std::cos((i + offset) * scale));
        Vec a(data);
        Vec b(reference);

        FUZZY_COMPARE(Vc::cos((a + offset) * scale), b);
    }
}

template<typename Vec> void testAsin()
{
    typedef typename Vec::EntryType T;
    setFuzzyness<float>(1.1e-6f);
    setFuzzyness<double>(8.8e-9);
    for (int offset = -1000; offset < 1000 - Vec::Size; offset += Vec::Size) {
        const T scale = T(0.001);
        FillHelperMemory(std::asin((i + offset) * scale));
        Vec a(data);
        Vec b(reference);

        FUZZY_COMPARE(Vc::asin((a + offset) * scale), b);
    }
}

template<typename Vec> void testAtan()
{
    typedef typename Vec::EntryType T;
    setFuzzyness<float>(1e-7f);
    setFuzzyness<double>(2e-8);
    for (int offset = -1000; offset < 1000; offset += 10) {
        const T scale = T(0.1);
        FillHelperMemory(std::atan((i + offset) * scale));
        Vec a(data);
        Vec b(reference);

        FUZZY_COMPARE(Vc::atan((a + offset) * scale), b);
    }
}

template<typename Vec> void testAtan2()
{
    typedef typename Vec::EntryType T;
    setFuzzyness<float>(3e-7f);
    setFuzzyness<double>(3e-8);
    for (int xoffset = -100; xoffset < 1000; xoffset += 10) {
        for (int yoffset = -100; yoffset < 1000; yoffset += 10) {
            FillHelperMemory(std::atan2((i + xoffset) * 0.15, (i + yoffset) * 0.15));
            Vec a(data);
            Vec b(reference);

            //std::cout << (a + xoffset) * 0.15 << (a + yoffset) * 0.15 << std::endl;
            FUZZY_COMPARE(Vc::atan2((a + xoffset) * T(0.15), (a + yoffset) * T(0.15)), b);
        }
    }
}

template<typename Vec> void testReciprocal()
{
    typedef typename Vec::EntryType T;
    setFuzzyness<float>(3.e-4f);
    setFuzzyness<double>(0);
    const T one = 1;
    for (int offset = -1000; offset < 1000; offset += 10) {
        const T scale = T(0.1);
        typename Vec::Memory data;
        typename Vec::Memory reference;
        for (int ii = 0; ii < Vec::Size; ++ii) {
            const T i = static_cast<T>(ii);
            data[ii] = i;
            T tmp = (i + offset) * scale;
            reference[ii] = one / tmp;
        }
        Vec a(data);
        Vec b(reference);

        FUZZY_COMPARE(Vc::reciprocal((a + offset) * scale), b);
    }
}

template<typename Vec> void testInf()
{
    typedef typename Vec::EntryType T;
    const T one = 1;
    const Vec zero(Zero);
    VERIFY(Vc::isfinite(zero));
    VERIFY(Vc::isfinite(Vec(one)));
    VERIFY(!Vc::isfinite(one / zero));
}

template<typename Vec> void testNaN()
{
    typedef typename Vec::EntryType T;
    const T one = 1;
    const Vec zero(Zero);
    VERIFY(!Vc::isnan(zero));
    VERIFY(!Vc::isnan(Vec(one)));
    const Vec inf = one / zero;
    VERIFY(Vc::isnan(Vec(inf * zero)));
}

template<typename Vec> void testRound()
{
    typedef typename Vec::EntryType T;
    enum {
        Count = (16 + Vec::Size) / Vec::Size
    };
    VectorMemoryHelper<Vec> mem1(Count);
    VectorMemoryHelper<Vec> mem2(Count);
    T *data = mem1;
    T *reference = mem2;
    for (int i = 0; i < Count * Vec::Size; ++i) {
        data[i] = i * 0.25 - 2.0;
        reference[i] = std::floor(i * 0.25 - 2.0 + 0.5);
        if (i % 8 == 2) {
            reference[i] -= 1.;
        }
        //std::cout << reference[i] << " ";
    }
    //std::cout << std::endl;
    for (int i = 0; i < Count; ++i) {
        const Vec a(&data[i * Vec::Size]);
        const Vec ref(&reference[i * Vec::Size]);
        //std::cout << a << ref << std::endl;
        COMPARE(Vc::round(a), ref);
    }
}

template<typename Vec> void testReduceMin()
{
    typedef typename Vec::EntryType T;
    const T one = 1;
    VectorMemoryHelper<Vec> mem(Vec::Size);
    T *data = mem;
    for (int i = 0; i < Vec::Size * Vec::Size; ++i) {
        data[i] = i % (Vec::Size + 1) + one;
    }
    for (int i = 0; i < Vec::Size; ++i, data += Vec::Size) {
        const Vec a(&data[0]);
        //std::cout << a << std::endl;
        COMPARE(a.min(), one);
    }
}

template<typename Vec> void testReduceMax()
{
    typedef typename Vec::EntryType T;
    const T max = Vec::Size + 1;
    VectorMemoryHelper<Vec> mem(Vec::Size);
    T *data = mem;
    for (int i = 0; i < Vec::Size * Vec::Size; ++i) {
        data[i] = (i + Vec::Size) % (Vec::Size + 1) + 1;
    }
    for (int i = 0; i < Vec::Size; ++i, data += Vec::Size) {
        const Vec a(&data[0]);
        //std::cout << a << std::endl;
        COMPARE(a.max(), max);
    }
}

template<typename Vec> void testReduceProduct()
{
    enum {
        Max = Vec::Size > 8 ? Vec::Size / 2 : Vec::Size
    };
    typedef typename Vec::EntryType T;
    int _product = 1;
    for (int i = 1; i < Vec::Size; ++i) {
        _product *= (i % Max) + 1;
    }
    const T product = _product;
    VectorMemoryHelper<Vec> mem(Vec::Size);
    T *data = mem;
    for (int i = 0; i < Vec::Size * Vec::Size; ++i) {
        data[i] = ((i + (i / Vec::Size)) % Max) + 1;
    }
    for (int i = 0; i < Vec::Size; ++i, data += Vec::Size) {
        const Vec a(&data[0]);
        //std::cout << a << std::endl;
        COMPARE(a.product(), product);
    }
}

template<typename Vec> void testReduceSum()
{
    typedef typename Vec::EntryType T;
    int _sum = 1;
    for (int i = 2; i <= Vec::Size; ++i) {
        _sum += i;
    }
    const T sum = _sum;
    VectorMemoryHelper<Vec> mem(Vec::Size);
    T *data = mem;
    for (int i = 0; i < Vec::Size * Vec::Size; ++i) {
        data[i] = (i + i / Vec::Size) % Vec::Size + 1;
    }
    for (int i = 0; i < Vec::Size; ++i, data += Vec::Size) {
        const Vec a(&data[0]);
        //std::cout << a << std::endl;
        COMPARE(a.sum(), sum);
    }
}

template<typename V> void testExponent()
{
    typedef typename V::EntryType T;
    Vc::Memory<V, 32> input;
    Vc::Memory<V, 32> expected;
    input[ 0] = T(0.25); expected[ 0] = T(-2);
    input[ 1] = T(   1); expected[ 1] = T( 0);
    input[ 2] = T(   2); expected[ 2] = T( 1);
    input[ 3] = T(   3); expected[ 3] = T( 1);
    input[ 4] = T(   4); expected[ 4] = T( 2);
    input[ 5] = T( 0.5); expected[ 5] = T(-1);
    input[ 6] = T(   6); expected[ 6] = T( 2);
    input[ 7] = T(   7); expected[ 7] = T( 2);
    input[ 8] = T(   8); expected[ 8] = T( 3);
    input[ 9] = T(   9); expected[ 9] = T( 3);
    input[10] = T(  10); expected[10] = T( 3);
    input[11] = T(  11); expected[11] = T( 3);
    input[12] = T(  12); expected[12] = T( 3);
    input[13] = T(  13); expected[13] = T( 3);
    input[14] = T(  14); expected[14] = T( 3);
    input[15] = T(  15); expected[15] = T( 3);
    input[16] = T(  16); expected[16] = T( 4);
    input[17] = T(  17); expected[17] = T( 4);
    input[18] = T(  18); expected[18] = T( 4);
    input[19] = T(  19); expected[19] = T( 4);
    input[20] = T(  20); expected[20] = T( 4);
    input[21] = T(  21); expected[21] = T( 4);
    input[22] = T(  22); expected[22] = T( 4);
    input[23] = T(  23); expected[23] = T( 4);
    input[24] = T(  24); expected[24] = T( 4);
    input[25] = T(  25); expected[25] = T( 4);
    input[26] = T(  26); expected[26] = T( 4);
    input[27] = T(  27); expected[27] = T( 4);
    input[28] = T(  28); expected[28] = T( 4);
    input[29] = T(  29); expected[29] = T( 4);
    input[30] = T(  32); expected[30] = T( 5);
    input[31] = T(  31); expected[31] = T( 4);
    for (size_t i = 0; i < input.vectorsCount(); ++i) {
        COMPARE(V(input.vector(i)).exponent(), V(expected.vector(i)));
    }
}

int main(int argc, char **argv)
{
    initTest(argc, argv);

    runTest(testAbs<int_v>);
    runTest(testAbs<float_v>);
    runTest(testAbs<double_v>);
    runTest(testAbs<short_v>);
    runTest(testAbs<sfloat_v>);

    runTest(testLog<float_v>);
    runTest(testLog<double_v>);
    runTest(testLog<sfloat_v>);

    runTest(testMax<int_v>);
    runTest(testMax<uint_v>);
    runTest(testMax<float_v>);
    runTest(testMax<double_v>);
    runTest(testMax<short_v>);
    runTest(testMax<ushort_v>);
    runTest(testMax<sfloat_v>);

    runTest(testSqrt<float_v>);
    runTest(testSqrt<double_v>);
    runTest(testSqrt<sfloat_v>);

    runTest(testRSqrt<float_v>);
    runTest(testRSqrt<double_v>);
    runTest(testRSqrt<sfloat_v>);

    runTest(testSin<float_v>);
    runTest(testSin<sfloat_v>);
    runTest(testSin<double_v>);

    runTest(testCos<float_v>);
    runTest(testCos<sfloat_v>);
    runTest(testCos<double_v>);

    runTest(testAsin<float_v>);
    runTest(testAsin<sfloat_v>);
    runTest(testAsin<double_v>);

    runTest(testAtan<float_v>);
    runTest(testAtan<sfloat_v>);
    runTest(testAtan<double_v>);

    runTest(testAtan2<float_v>);
    runTest(testAtan2<sfloat_v>);
    runTest(testAtan2<double_v>);

    runTest(testReciprocal<float_v>);
    runTest(testReciprocal<sfloat_v>);
    runTest(testReciprocal<double_v>);

    runTest(testInf<float_v>);
    runTest(testInf<double_v>);
    runTest(testInf<sfloat_v>);

    runTest(testNaN<float_v>);
    runTest(testNaN<double_v>);
    runTest(testNaN<sfloat_v>);

    runTest(testRound<float_v>);
    runTest(testRound<double_v>);
    runTest(testRound<sfloat_v>);

    runTest(testReduceMin<float_v>);
    runTest(testReduceMin<sfloat_v>);
    runTest(testReduceMin<double_v>);
    runTest(testReduceMin<int_v>);
    runTest(testReduceMin<uint_v>);
    runTest(testReduceMin<short_v>);
    runTest(testReduceMin<ushort_v>);

    runTest(testReduceMax<float_v>);
    runTest(testReduceMax<sfloat_v>);
    runTest(testReduceMax<double_v>);
    runTest(testReduceMax<int_v>);
    runTest(testReduceMax<uint_v>);
    runTest(testReduceMax<short_v>);
    runTest(testReduceMax<ushort_v>);

    runTest(testReduceProduct<float_v>);
    runTest(testReduceProduct<sfloat_v>);
    runTest(testReduceProduct<double_v>);
    runTest(testReduceProduct<int_v>);
    runTest(testReduceProduct<uint_v>);
    runTest(testReduceProduct<short_v>);
    runTest(testReduceProduct<ushort_v>);

    runTest(testReduceSum<float_v>);
    runTest(testReduceSum<sfloat_v>);
    runTest(testReduceSum<double_v>);
    runTest(testReduceSum<int_v>);
    runTest(testReduceSum<uint_v>);
    runTest(testReduceSum<short_v>);
    runTest(testReduceSum<ushort_v>);

    runTest(testSincos<float_v>);
    runTest(testSincos<sfloat_v>);
    runTest(testSincos<double_v>);

    runTest(testExponent<float_v>);
    runTest(testExponent<sfloat_v>);
    runTest(testExponent<double_v>);

    return 0;
}
