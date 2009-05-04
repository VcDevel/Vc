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
    Vec a(int_v(IndexesFromZero).staticCast<typename Vec::Type>());
    a *= 0.1;
    const Vec end(1000);
    for (; a < end; a += Vec::Size) {
        Vec b = Vc::log(a);

        const typename Vec::Type two = 2.;

        for (int i = 0; i < Vec::Size; ++i) {
            FUZZY_COMPARE(b[i], std::log(a[i]));
        }

        const Vec a2 = a * a;
        FUZZY_COMPARE(Vc::log(a2), two * Vc::log(a));
    }
}

template<typename Vec>
void testMax()
{
    typedef typename Vec::Type T;
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

int main()
{
    runTest(testAbs<int_v>);
    runTest(testAbs<float_v>);
    runTest(testAbs<double_v>);
    runTest(testAbs<short_v>);

    setFuzzyness<float>(1e-7);
    setFuzzyness<double>(1e-15);
    runTest(testLog<float_v>);
    runTest(testLog<double_v>);

    runTest(testMax<int_v>);
    runTest(testMax<uint_v>);
    runTest(testMax<float_v>);
    runTest(testMax<double_v>);
    runTest(testMax<short_v>);
    runTest(testMax<ushort_v>);

    return 0;
}
