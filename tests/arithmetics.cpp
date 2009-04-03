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

using namespace Vc;

template<typename Vec> void testZero()
{
    Vec a(Zero), b(Zero);
    VERIFY(a == b);
    Vec c, d(1);
    c.makeZero();
    VERIFY(a == c);
    d.makeZero();
    VERIFY(a == d);
    VERIFY(a == 0);
    VERIFY(b == 0);
    VERIFY(c == 0);
    VERIFY(d == 0);
}

template<typename Vec> void testCmp()
{
    Vec a(Zero), b(Zero);
    VERIFY(a == b);
    if (a != b) {
        std::cerr << a << " != " << b << ", (a != b) = " << (a != b) << ", (a == b) = " << (a == b) << std::endl;
    }
    VERIFY(!(a != b));

    Vec c(1);
    VERIFY(a < c);
    VERIFY(c > a);
    VERIFY(a <= b);
    VERIFY(a <= c);
    VERIFY(b >= a);
    VERIFY(c >= a);
}

template<typename Vec> void testAdd()
{
    Vec a(Zero), b(Zero);
    VERIFY(a == b);

    a += 1;
    Vec c(1);
    VERIFY(a == c);

    VERIFY(a == b + 1);
    VERIFY(a == b + c);
    Vec x(Zero);
}

template<typename Vec> void testSub()
{
    Vec a(2), b(2);
    VERIFY(a == b);

    a -= 1;
    Vec c(1);
    VERIFY(a == c);

    VERIFY(a == b - 1);
    VERIFY(a == b - c);
}

template<typename Vec> void testMul()
{
    for (unsigned int i = 0; i < 0xffff; ++i) {
        const Vec i2 = i * i;
        Vec a(i);

        COMPARE(a * a, i2);
    }
}

template<typename Vec> void testDiv()
{
    for (unsigned int i = 0; i < 0x7fff / 3; ++i) {
        Vec a(i * 3);

        COMPARE(a / 3, Vec(i));
    }
}

template<typename Vec> void testAnd()
{
    Vec a(0x7fff);
    Vec b(0xf);
    VERIFY(FullMask == ((a & 0xf) == b));
    Vec c(Vec::IndexesFromZero);
    VERIFY(FullMask == (c == (c & 0xf)));
    VERIFY(FullMask == ((c & 0xfff0) == 0));
}

int main()
{
    runTest(testZero<int_v>);
    runTest(testZero<uint_v>);
    runTest(testZero<float_v>);
    runTest(testZero<double_v>);
    runTest(testZero<short_v>);
    runTest(testZero<ushort_v>);
    runTest(testCmp<int_v>);
    runTest(testCmp<uint_v>);
    runTest(testCmp<float_v>);
    runTest(testCmp<double_v>);
    runTest(testCmp<short_v>);
    runTest(testCmp<ushort_v>);
    runTest(testAdd<int_v>);
    runTest(testAdd<uint_v>);
    runTest(testAdd<float_v>);
    runTest(testAdd<double_v>);
    runTest(testAdd<short_v>);
    runTest(testAdd<ushort_v>);
    runTest(testSub<int_v>);
    runTest(testSub<uint_v>);
    runTest(testSub<float_v>);
    runTest(testSub<double_v>);
    runTest(testSub<short_v>);
    runTest(testSub<ushort_v>);
    runTest(testMul<int_v>);
    runTest(testMul<uint_v>);
    runTest(testMul<float_v>);
    runTest(testMul<double_v>);
    runTest(testMul<short_v>);
    runTest(testMul<ushort_v>);
    runTest(testDiv<int_v>);
    runTest(testDiv<uint_v>);
    runTest(testDiv<float_v>);
    runTest(testDiv<double_v>);
    runTest(testDiv<short_v>);
    runTest(testDiv<ushort_v>);
    runTest(testAnd<int_v>);
    runTest(testAnd<uint_v>);
    runTest(testAnd<short_v>);
    runTest(testAnd<ushort_v>);
    // no operator& for float/double
    return 0;
}
