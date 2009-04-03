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

template<typename Vec> void gatherArray()
{
    const int count = 39999;
    typename Vec::Type array[count];
    for (int i = 0; i < count; ++i) {
        array[i] = i;
    }
    Mask mask;
    for (int_v i = IndexesFromZero; (mask = (i < count)); i += Vec::Size) {
        const Vec &ii = i.staticCast<typename Vec::Type>();
        if (FullMask == mask) {
            Vec a(array, i);
            VERIFY(FullMask == (a == ii));
            a.gather(array, i);
            VERIFY(FullMask == (a == ii));
        }
        Vec b;
        b.gather(array, i, mask);
        if (sizeof(typename Vec::Type) == 8) {
            // mask is for a 32bit entries vector whereas with double_v (b == ii)
            // returns a mask for a 64bit entries vector => half the mask size.
            // Therefore we need to use cmpeq32_64
            VERIFY(cmpeq32_64(mask, b == ii));
        } else {
            VERIFY(mask == (b == ii));
        }
    }
}

template<typename T> struct Struct
{
    T a;
    char x;
    T b;
    short y;
    T c;
};

template<typename Vec> void gatherStruct()
{
    typedef Struct<typename Vec::Type> S;
    const int count = 3999;
    S array[count];
    for (int i = 0; i < count; ++i) {
        array[i].a = i;
        array[i].b = i + 1;
        array[i].c = i + 2;
    }
    Mask mask;
    for (int_v i = IndexesFromZero; (mask = (i < count)); i += Vec::Size) {
        // if Vec is double_v the staticCast keeps only the lower two values, which is why the ==
        // comparison works
        const Vec &ii = i.staticCast<typename Vec::Type>();
        if (FullMask == mask) {
            Vec a(array, &S::a, i);
            VERIFY(FullMask == (a == ii));
            a.gather(array, &S::a, i);
            VERIFY(FullMask == (a == ii));
        }
        Vec b;
        b.gather(array, &S::a, i, mask);
        if (sizeof(typename Vec::Type) == 8) {
            // mask is for a 32bit entries vector whereas with double_v (b == ii)
            // returns a mask for a 64bit entries vector => half the mask size.
            // Therefore we need to use cmpeq32_64
            VERIFY(cmpeq32_64(mask, b == ii));
        } else {
            VERIFY(mask == (b == ii));
        }
    }
}

int main()
{
    runTest(gatherArray<int_v>);
    runTest(gatherArray<uint_v>);
    runTest(gatherArray<float_v>);
    runTest(gatherArray<double_v>);
    runTest(gatherStruct<int_v>);
    runTest(gatherStruct<uint_v>);
    runTest(gatherStruct<float_v>);
    runTest(gatherStruct<double_v>);
    return 0;
}
