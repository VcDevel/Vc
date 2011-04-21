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
}
