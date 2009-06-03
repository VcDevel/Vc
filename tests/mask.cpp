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

using namespace Vc;

template<typename Vec> void testInc()
{
    VectorMemoryHelper<Vec> mem(2);
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;
    T *data = mem;
    for (int borderI = 0; borderI < Vec::Size; ++borderI) {
        const T border = static_cast<T>(borderI);
        for (int i = 0; i < Vec::Size; ++i) {
            data[i] = static_cast<T>(i);
            data[i + Vec::Size] = data[i] + static_cast<T>(data[i] < border ? 1 : 0);
        }
        Vec a(&data[0]);
        Vec b(&data[Vec::Size]);
        Mask m = a < border;
        Vec aa(a);
        COMPARE(aa(m)++, a);
        COMPARE(aa, b);
        COMPARE(++a(m), b);
        COMPARE(a, b);
    }
}

template<typename Vec> void testDec()
{
    VectorMemoryHelper<Vec> mem(2);
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;
    T *data = mem;
    for (int borderI = 0; borderI < Vec::Size; ++borderI) {
        const T border = static_cast<T>(borderI);
        for (int i = 0; i < Vec::Size; ++i) {
            data[i] = static_cast<T>(i + 1);
            data[i + Vec::Size] = data[i] - static_cast<T>(data[i] < border ? 1 : 0);
        }
        Vec a(&data[0]);
        Vec b(&data[Vec::Size]);
        Mask m = a < border;
        Vec aa(a);
        COMPARE(aa(m)--, a);
        COMPARE(--a(m), b);
        COMPARE(a, b);
        COMPARE(aa, b);
    }
}

template<typename Vec> void testPlusEq()
{
    VectorMemoryHelper<Vec> mem(2);
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;
    T *data = mem;
    for (int borderI = 0; borderI < Vec::Size; ++borderI) {
        const T border = static_cast<T>(borderI);
        for (int i = 0; i < Vec::Size; ++i) {
            data[i] = static_cast<T>(i + 1);
            data[i + Vec::Size] = data[i] + static_cast<T>(data[i] < border ? 2 : 0);
        }
        Vec a(&data[0]);
        Vec b(&data[Vec::Size]);
        Mask m = a < border;
        COMPARE(a(m) += static_cast<T>(2), b);
        COMPARE(a, b);
    }
}

template<typename Vec> void testMinusEq()
{
    VectorMemoryHelper<Vec> mem(2);
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;
    T *data = mem;
    for (int borderI = 0; borderI < Vec::Size; ++borderI) {
        const T border = static_cast<T>(borderI);
        for (int i = 0; i < Vec::Size; ++i) {
            data[i] = static_cast<T>(i + 2);
            data[i + Vec::Size] = data[i] - static_cast<T>(data[i] < border ? 2 : 0);
        }
        Vec a(&data[0]);
        Vec b(&data[Vec::Size]);
        Mask m = a < border;
        COMPARE(a(m) -= static_cast<T>(2), b);
        COMPARE(a, b);
    }
}

template<typename Vec> void testTimesEq()
{
    VectorMemoryHelper<Vec> mem(2);
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;
    T *data = mem;
    for (int borderI = 0; borderI < Vec::Size; ++borderI) {
        const T border = static_cast<T>(borderI);
        for (int i = 0; i < Vec::Size; ++i) {
            data[i] = static_cast<T>(i);
            data[i + Vec::Size] = data[i] * static_cast<T>(data[i] < border ? 2 : 1);
        }
        Vec a(&data[0]);
        Vec b(&data[Vec::Size]);
        Mask m = a < border;
        COMPARE(a(m) *= static_cast<T>(2), b);
        COMPARE(a, b);
    }
}

template<typename Vec> void testDivEq()
{
    VectorMemoryHelper<Vec> mem(2);
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;
    T *data = mem;
    for (int borderI = 0; borderI < Vec::Size; ++borderI) {
        const T border = static_cast<T>(borderI);
        for (int i = 0; i < Vec::Size; ++i) {
            data[i] = static_cast<T>(5 * i);
            data[i + Vec::Size] = data[i] / static_cast<T>(data[i] < border ? 3 : 1);
        }
        Vec a(&data[0]);
        Vec b(&data[Vec::Size]);
        Mask m = a < border;
        COMPARE(a(m) /= static_cast<T>(3), b);
        COMPARE(a, b);
    }
}

template<typename Vec> void testAssign()
{
    VectorMemoryHelper<Vec> mem(2);
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;
    T *data = mem;
    for (int borderI = 0; borderI < Vec::Size; ++borderI) {
        const T border = static_cast<T>(borderI);
        for (int i = 0; i < Vec::Size; ++i) {
            data[i] = static_cast<T>(i);
            data[i + Vec::Size] = data[i] + static_cast<T>(data[i] < border ? 2 : 0);
        }
        Vec a(&data[0]);
        Vec b(&data[Vec::Size]);
        Mask m = a < border;
        COMPARE(a(m) = b, b);
        COMPARE(a, b);
    }
}

template<typename Vec> void testZero()
{
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;
    typedef typename Vec::IndexType I;

    for (int cut = 0; cut < Vec::Size; ++cut) {
        const Mask mask(I(Vc::IndexesFromZero) < cut);
        std::cout << mask << std::endl;

        const T aa = 4;
        Vec a(aa);
        Vec b(Vc::Zero);

        b(!mask) = a;
        a.makeZero(mask);

        COMPARE(a, b);
    }
}

int main()
{
    runTest(testInc<int_v>);
    runTest(testInc<uint_v>);
    runTest(testInc<float_v>);
    runTest(testInc<double_v>);
    runTest(testInc<short_v>);
    runTest(testInc<ushort_v>);
    runTest(testInc<sfloat_v>);

    runTest(testDec<int_v>);
    runTest(testDec<uint_v>);
    runTest(testDec<float_v>);
    runTest(testDec<double_v>);
    runTest(testDec<short_v>);
    runTest(testDec<ushort_v>);
    runTest(testDec<sfloat_v>);

    runTest(testPlusEq<int_v>);
    runTest(testPlusEq<uint_v>);
    runTest(testPlusEq<float_v>);
    runTest(testPlusEq<double_v>);
    runTest(testPlusEq<short_v>);
    runTest(testPlusEq<ushort_v>);
    runTest(testPlusEq<sfloat_v>);

    runTest(testMinusEq<int_v>);
    runTest(testMinusEq<uint_v>);
    runTest(testMinusEq<float_v>);
    runTest(testMinusEq<double_v>);
    runTest(testMinusEq<short_v>);
    runTest(testMinusEq<ushort_v>);
    runTest(testMinusEq<sfloat_v>);

    runTest(testTimesEq<int_v>);
    runTest(testTimesEq<uint_v>);
    runTest(testTimesEq<float_v>);
    runTest(testTimesEq<double_v>);
    runTest(testTimesEq<short_v>);
    runTest(testTimesEq<ushort_v>);
    runTest(testTimesEq<sfloat_v>);

    runTest(testDivEq<int_v>);
    runTest(testDivEq<uint_v>);
    runTest(testDivEq<float_v>);
    runTest(testDivEq<double_v>);
    runTest(testDivEq<short_v>);
    runTest(testDivEq<ushort_v>);
    runTest(testDivEq<sfloat_v>);

    runTest(testAssign<int_v>);
    runTest(testAssign<uint_v>);
    runTest(testAssign<float_v>);
    runTest(testAssign<double_v>);
    runTest(testAssign<short_v>);
    runTest(testAssign<ushort_v>);
    runTest(testAssign<sfloat_v>);

    runTest(testZero<int_v>);
    runTest(testZero<uint_v>);
    runTest(testZero<float_v>);
    runTest(testZero<double_v>);
    runTest(testZero<short_v>);
    runTest(testZero<ushort_v>);
    runTest(testZero<sfloat_v>);

    return 0;
}
