/*
    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation; either version 2 of
    the License, or (at your option) version 3.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
    02110-1301, USA.

*/

#include "../vector.h"
#include "unittest.h"
#include <iostream>
#include "vecio.h"
#include "vectormemoryhelper.h"
#include <cmath>
#include <algorithm>

using namespace Vc;

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
    Vec a(int_v(IndexesFromZero).staticCast<typename Vec::EntryType>());
    a *= 0.1;
    const Vec end(1000);
    for (; a < end; a += Vec::Size) {
        Vec b = Vc::log(a);

        const typename Vec::EntryType two = 2.;

        for (int i = 0; i < Vec::Size; ++i) {
            FUZZY_COMPARE(b[i], std::log(a[i]));
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
    setFuzzyness<float>(0.0003662109375);
    FUZZY_COMPARE(Vc::rsqrt(a), b);
    setFuzzyness<float>(0.f);
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

int main()
{
    runTest(testAbs<int_v>);
    runTest(testAbs<float_v>);
    runTest(testAbs<double_v>);
    runTest(testAbs<short_v>);

    runTest(testLog<float_v>);
    runTest(testLog<double_v>);

    runTest(testMax<int_v>);
    runTest(testMax<uint_v>);
    runTest(testMax<float_v>);
    runTest(testMax<double_v>);
    runTest(testMax<short_v>);
    runTest(testMax<ushort_v>);

    runTest(testSqrt<float_v>);
    runTest(testSqrt<double_v>);

    runTest(testRSqrt<float_v>);
    runTest(testRSqrt<double_v>);

    runTest(testInf<float_v>);
    runTest(testInf<double_v>);

    return 0;
}
