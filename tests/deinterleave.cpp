/*  This file is part of the Vc library.

    Copyright (C) 2010 Matthias Kretz <kretz@kde.org>

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

template<typename Pair> void testDeinterlace()
{
    typedef typename Pair::V V;
    typedef typename Pair::M M;
    typedef typename V::IndexType I;

    const bool isSigned = std::numeric_limits<M>::is_signed;

    const typename V::EntryType offset = isSigned ? -512 : 0;
    const V _0246 = static_cast<V>(I::IndexesFromZero()) * 2 + offset;

    M memory[1024];
    for (int i = 0; i < 1024; ++i) {
        memory[i] = i;
        if (isSigned) {
            memory[i] -= 512;
        }
    }

    V a, b;

    for (int i = 0; i < 1024 - 2 * V::Size; ++i) {
        if (reinterpret_cast<uintptr_t>(&memory[i]) & (VectorAlignment - 1)) {
            Vc::deinterleave(&a, &b, &memory[i], Unaligned);
        } else {
            Vc::deinterleave(&a, &b, &memory[i]);
        }
        COMPARE(_0246 + i,     a);
        COMPARE(_0246 + i + 1, b);
    }
}

int main()
{
    runTest(testDeinterlace<float_float>);
    runTest(testDeinterlace<float_ushort>);
    runTest(testDeinterlace<float_short>);
    runTest(testDeinterlace<sfloat_float>);
    runTest(testDeinterlace<sfloat_ushort>);
    runTest(testDeinterlace<sfloat_short>);
    runTest(testDeinterlace<double_double>);
    runTest(testDeinterlace<int_int>);
    runTest(testDeinterlace<int_short>);
    runTest(testDeinterlace<uint_uint>);
    runTest(testDeinterlace<uint_ushort>);
    runTest(testDeinterlace<short_short>);
    runTest(testDeinterlace<ushort_ushort>);
}
