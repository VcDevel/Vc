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
#include <limits>

using namespace Vc;

template<typename Vec> void testZero()
{
    Vec a(Zero), b(Zero);
    COMPARE(a, b);
    Vec c, d(1);
    c.makeZero();
    COMPARE(a, c);
    d.makeZero();
    COMPARE(a, d);
    d = 0.;
    COMPARE(a, d);
    const typename Vec::EntryType zero = 0;
    COMPARE(a, Vec(zero));
    COMPARE(b, Vec(zero));
    COMPARE(c, Vec(zero));
    COMPARE(d, Vec(zero));
}

template<typename Vec> void testCmp()
{
    typedef typename Vec::EntryType T;
    Vec a(Zero), b(Zero);
    COMPARE(a, b);
    if (!(a != b).isEmpty()) {
        std::cerr << a << " != " << b << ", (a != b) = " << (a != b) << ", (a == b) = " << (a == b) << std::endl;
    }
    VERIFY((a != b).isEmpty());

    Vec c(1);
    VERIFY((a < c).isFull());
    VERIFY((c > a).isFull());
    VERIFY((a <= b).isFull());
    VERIFY((a <= c).isFull());
    VERIFY((b >= a).isFull());
    VERIFY((c >= a).isFull());

    {
        const T max = static_cast<T>(std::numeric_limits<T>::max() * 0.95);
        const T min = 0;
        const T step = max / 200;
        T j = min;
        VERIFY(Vec(Zero) == Vec(j));
        VERIFY(!(Vec(Zero) < Vec(j)));
        VERIFY(!(Vec(Zero) > Vec(j)));
        VERIFY(!(Vec(Zero) != Vec(j)));
        j += step;
        for (int i = 0; i < 200; ++i, j += step) {
            if(Vec(Zero) >= Vec(j)) {
                std::cout << j << " " << Vec(j) << " " << (Vec(Zero) >= Vec(j)) << std::endl;
            }
            VERIFY(Vec(Zero) < Vec(j));
            VERIFY(Vec(j) > Vec(Zero));
            VERIFY(!(Vec(Zero) >= Vec(j)));
            VERIFY(!(Vec(j) <= Vec(Zero)));
            VERIFY(!static_cast<bool>(Vec(Zero) >= Vec(j)));
            VERIFY(!static_cast<bool>(Vec(j) <= Vec(Zero)));
        }
    }
    if (std::numeric_limits<T>::min() <= 0) {
        const T min = static_cast<T>(std::numeric_limits<T>::min() * 0.95);
        if (min == 0) {
            return;
        }
        const T step = min / -201;
        T j = min;
        for (int i = 0; i < 200; ++i, j += step) {
            VERIFY(Vec(j) < Vec(Zero));
            VERIFY(Vec(Zero) > Vec(j));
            VERIFY(!(Vec(Zero) <= Vec(j)));
            VERIFY(!(Vec(j) >= Vec(Zero)));
        }
    }
}

template<typename Vec> void testIsMix()
{
    Vec a(IndexesFromZero);
    Vec b(Zero);
    Vec c(One);
    if (Vec::Size > 1) {
        VERIFY((a == b).isMix());
        VERIFY((a != b).isMix());
        VERIFY((a == c).isMix());
        VERIFY((a != c).isMix());
        VERIFY(!(a == a).isMix());
        VERIFY(!(a != a).isMix());
    } else { // masks of size 1 can never be a mix of 0 and 1
        VERIFY(!(a == b).isMix());
        VERIFY(!(a != b).isMix());
        VERIFY(!(a == c).isMix());
        VERIFY(!(a != c).isMix());
        VERIFY(!(a == a).isMix());
        VERIFY(!(a != a).isMix());
    }
}

template<typename Vec> void testAdd()
{
    Vec a(Zero), b(Zero);
    COMPARE(a, b);

    a += 1;
    Vec c(1);
    COMPARE(a, c);

    COMPARE(a, b + 1);
    COMPARE(a, b + c);
    Vec x(Zero);
}

template<typename Vec> void testSub()
{
    Vec a(2), b(2);
    COMPARE(a, b);

    a -= 1;
    Vec c(1);
    COMPARE(a, c);

    COMPARE(a, b - 1);
    COMPARE(a, b - c);
}

template<typename T> struct MulRangeHelper
{
    typedef int Iterator;
    static const Iterator Start;
    static const Iterator End;
};
template<> struct MulRangeHelper<unsigned int> {
    typedef unsigned int Iterator;
    static const Iterator Start;
    static const Iterator End;
};
template<> const int MulRangeHelper<float>::Start = -0xffffff;
template<> const int MulRangeHelper<float>::End   =  0xffffff - 133;
template<> const int MulRangeHelper<double>::Start = -0xffffff;
template<> const int MulRangeHelper<double>::End   =  0xffffff - 133;
template<> const int MulRangeHelper<int>::Start = -0x80000000;
template<> const int MulRangeHelper<int>::End   = 0x7fffffff - 110;
const unsigned int MulRangeHelper<unsigned int>::Start = 0;
const unsigned int MulRangeHelper<unsigned int>::End = 0xffffffff - 110;
template<> const int MulRangeHelper<short>::Start = -0x8000;
template<> const int MulRangeHelper<short>::End = 0x7fff - 50;
template<> const int MulRangeHelper<unsigned short>::Start = 0;
template<> const int MulRangeHelper<unsigned short>::End = 0xffff - 50;

template<typename Vec> void testMul()
{
    typedef typename Vec::EntryType T;
    typedef MulRangeHelper<T> Range;
    for (typename Range::Iterator i = Range::Start; i < Range::End; i += 0xef) {
        T i2 = static_cast<T>(i);
        Vec a(i2);
        i2 *= i2 - 9;

        COMPARE(a * (a - 9), Vec(i2));
    }
}

template<typename Vec> void testMulAdd()
{
    for (unsigned int i = 0; i < 0xffff; ++i) {
        const Vec i2(i * i + 1);
        Vec a(i);

        FUZZY_COMPARE(a * a + 1, i2);
    }
}

template<typename Vec> void testMulSub()
{
    for (unsigned int i = 0; i < 0xffff; ++i) {
        const Vec i2(i * i - i);
        Vec a(i);

        FUZZY_COMPARE(a * a - i, i2);
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
    COMPARE((a & 0xf), b);
    Vec c(IndexesFromZero);
    COMPARE(c, (c & 0xf));
    const typename Vec::EntryType zero = 0;
    COMPARE((c & 0x7ff0), Vec(zero));
}

template<typename Vec> void testShift()
{
    Vec a(1);
    Vec b(2);

    // left shifts
    COMPARE((a << 1), b);
    COMPARE((a << 2), (a << 2));
    COMPARE((a << 2), (b << 1));

    Vec shifts(IndexesFromZero);
    a <<= shifts;
    for (typename Vec::EntryType i = 0, x = 1; i < Vec::Size; ++i, x <<= 1) {
        COMPARE(a[i], x);
    }

    // right shifts
    a = Vec(4);
    COMPARE((a >> 1), b);
    COMPARE((a >> 2), (a >> 2));
    COMPARE((a >> 2), (b >> 1));

    a = Vec(16);
    a >>= shifts;
    for (typename Vec::EntryType i = 0, x = 16; i < Vec::Size; ++i, x >>= 1) {
        COMPARE(a[i], x);
    }
}

template<typename Vec> void testOnesComplement()
{
    Vec a(One);
    Vec b = ~a;
    COMPARE(~a, b);
    COMPARE(~b, a);
    COMPARE(~(a + b), Vec(Zero));
}

int main()
{
    runTest(testZero<int_v>);
    runTest(testZero<uint_v>);
    runTest(testZero<float_v>);
    runTest(testZero<double_v>);
    runTest(testZero<short_v>);
    runTest(testZero<ushort_v>);
    runTest(testZero<sfloat_v>);

    runTest(testCmp<int_v>);
    runTest(testCmp<uint_v>);
    runTest(testCmp<float_v>);
    runTest(testCmp<double_v>);
    runTest(testCmp<short_v>);
    runTest(testCmp<ushort_v>);
    runTest(testCmp<sfloat_v>);

    runTest(testIsMix<int_v>);
    runTest(testIsMix<uint_v>);
    //runTest(testIsMix<float_v>);
    //runTest(testIsMix<double_v>);
    runTest(testIsMix<short_v>);
    runTest(testIsMix<ushort_v>);
    //runTest(testIsMix<sfloat_v>);

    runTest(testAdd<int_v>);
    runTest(testAdd<uint_v>);
    runTest(testAdd<float_v>);
    runTest(testAdd<double_v>);
    runTest(testAdd<short_v>);
    runTest(testAdd<ushort_v>);
    runTest(testAdd<sfloat_v>);

    runTest(testSub<int_v>);
    runTest(testSub<uint_v>);
    runTest(testSub<float_v>);
    runTest(testSub<double_v>);
    runTest(testSub<short_v>);
    runTest(testSub<ushort_v>);
    runTest(testSub<sfloat_v>);

    runTest(testMul<int_v>);
    runTest(testMul<uint_v>);
    runTest(testMul<float_v>);
    runTest(testMul<double_v>);
    runTest(testMul<short_v>);
    runTest(testMul<ushort_v>);
    runTest(testMul<sfloat_v>);

    runTest(testDiv<int_v>);
    runTest(testDiv<uint_v>);
    runTest(testDiv<float_v>);
    runTest(testDiv<double_v>);
    runTest(testDiv<short_v>);
    runTest(testDiv<ushort_v>);
    runTest(testDiv<sfloat_v>);

    runTest(testAnd<int_v>);
    runTest(testAnd<uint_v>);
    runTest(testAnd<short_v>);
    runTest(testAnd<ushort_v>);
    // no operator& for float/double

    runTest(testShift<int_v>);
    runTest(testShift<uint_v>);
    runTest(testShift<short_v>);
    runTest(testShift<ushort_v>);

    runTest(testMulAdd<int_v>);
    runTest(testMulAdd<uint_v>);
    runTest(testMulAdd<float_v>);
    runTest(testMulAdd<double_v>);
    runTest(testMulAdd<short_v>);
    runTest(testMulAdd<ushort_v>);
    runTest(testMulAdd<sfloat_v>);

    runTest(testMulSub<int_v>);
    runTest(testMulSub<uint_v>);
    runTest(testMulSub<float_v>);
    runTest(testMulSub<double_v>);
    runTest(testMulSub<short_v>);
    runTest(testMulSub<ushort_v>);
    runTest(testMulSub<sfloat_v>);

    runTest(testOnesComplement<int_v>);
    runTest(testOnesComplement<uint_v>);
    runTest(testOnesComplement<short_v>);
    runTest(testOnesComplement<ushort_v>);

    return 0;
}
