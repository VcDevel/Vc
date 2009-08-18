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
    typedef typename Vec::IndexType I;
    const I indexesFromZero(IndexesFromZero);
    Vec a(indexesFromZero);
    a *= 0.1;
    const Vec end(1000);
    for (; a < end; a += Vec::Size) {
        Vec b = Vc::log(a);

        const typename Vec::EntryType two = 2.;

        for (int i = 0; i < Vec::Size; ++i) {
            FUZZY_COMPARE(b[i], static_cast<typename Vec::EntryType>(std::log(a[i])));
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
    VectorMemoryHelper<Vec> mem(2); \
    T *data = mem; \
    T *reference = &data[Vec::Size]; \
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

template<typename Vec> void testAtan()
{
    typedef typename Vec::EntryType T;
    setFuzzyness<float>(8e-5f);
    setFuzzyness<double>(2e-8);
    for (int offset = -1000; offset < 1000; offset += 10) {
        FillHelperMemory(std::atan((i + offset) * 0.1));
        Vec a(data);
        Vec b(reference);

        FUZZY_COMPARE(Vc::atan((a + offset) * 0.1), b);
    }
}

template<typename Vec> void testAtan2()
{
    typedef typename Vec::EntryType T;
    setFuzzyness<float>(8e-5f);
    setFuzzyness<double>(3e-8);
    for (int xoffset = -100; xoffset < 1000; xoffset += 10) {
        for (int yoffset = -100; yoffset < 1000; yoffset += 10) {
            FillHelperMemory(std::atan2((i + xoffset) * 0.15, (i + yoffset) * 0.15));
            Vec a(data);
            Vec b(reference);

            //std::cout << (a + xoffset) * 0.15 << (a + yoffset) * 0.15 << std::endl;
            FUZZY_COMPARE(Vc::atan2((a + xoffset) * 0.15, (a + yoffset) * 0.15), b);
        }
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

int main()
{
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

    runTest(testAtan<float_v>);
    runTest(testAtan<sfloat_v>);
    runTest(testAtan<double_v>);

    runTest(testAtan2<float_v>);
    runTest(testAtan2<sfloat_v>);
    runTest(testAtan2<double_v>);

    runTest(testInf<float_v>);
    runTest(testInf<double_v>);
    runTest(testInf<sfloat_v>);

    runTest(testNaN<float_v>);
    runTest(testNaN<double_v>);
    runTest(testNaN<sfloat_v>);

    runTest(testReduceMin<float_v>);
    runTest(testReduceMin<sfloat_v>);
    runTest(testReduceMin<double_v>);
    runTest(testReduceMin<int_v>);
    runTest(testReduceMin<uint_v>);
    runTest(testReduceMin<short_v>);
    runTest(testReduceMin<ushort_v>);

    return 0;
}
