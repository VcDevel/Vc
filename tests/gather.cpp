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
    typedef uint_v It;
    typedef typename Vec::Type T;

    const int count = 39999;
    T array[count];
    for (int i = 0; i < count; ++i) {
        array[i] = i;
    }
    Mask mask;
    for (It i = It::IndexesFromZero; (mask = (i < count)); i += Vec::Size) {
        const Vec &ii = i.staticCast<T>();
        if (FullMask == mask) {
            Vec a(array, i);
            VERIFY(FullMask == (a == ii));
            Vec b(Zero);
            b.gather(array, i);
            VERIFY(FullMask == (b == ii));
        }
        Vec b(Zero);
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

template<typename Vec> void gatherArray16()
{
    typedef ushort_v It;
    typedef typename Vec::Type T;

    const int count = 39999;
    T array[count];
    for (int i = 0; i < count; ++i) {
        array[i] = i;
    }
    Mask mask;
    for (It i = It::IndexesFromZero; (mask = (i < count)); i += Vec::Size) {
        const Vec &ii = i.staticCast<T>();
        if (FullMask == mask) {
            Vec a(array, i);
            VERIFY(FullMask == (a == ii));
            Vec b(Zero);
            b.gather(array, i);
            VERIFY(FullMask == (b == ii));
        }
        Vec b(Zero);
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
    char z;
};

template<typename Vec> void gatherStruct()
{
    typedef uint_v It;
    typedef typename Vec::Type T;
    typedef Struct<T> S;
    const int count = 3999;
    S array[count];
    for (int i = 0; i < count; ++i) {
        array[i].a = i;
        array[i].b = i + 1;
        array[i].c = i + 2;
    }
    Mask mask;
    for (It i = It::IndexesFromZero; (mask = (i < count)); i += Vec::Size) {
        // if Vec is double_v the staticCast keeps only the lower two values, which is why the ==
        // comparison works
        const Vec &i0 = i.staticCast<T>();
        const Vec &i1 = (i + 1).staticCast<T>();
        const Vec &i2 = (i + 2).staticCast<T>();
        if (FullMask == mask) {
            Vec a(array, &S::a, i);
            VERIFY(FullMask == (a == i0));
            a.gather(array, &S::b, i);
            VERIFY(FullMask == (a == i1));
            a.gather(array, &S::c, i);
            VERIFY(FullMask == (a == i2));
        }

        // mask is for a 32bit entries vector whereas with double_v (b == ii)
        // returns a mask for a 64bit entries vector => half the mask size.
        // Therefore we need to use cmpeq32_64
        Vec b;
        b.gather(array, &S::a, i, mask);
        if (sizeof(typename Vec::Type) == 8) {
            VERIFY(cmpeq32_64(mask, b == i0));
        } else {
            VERIFY(mask == (b == i0));
        }
        b.gather(array, &S::b, i, mask);
        if (sizeof(typename Vec::Type) == 8) {
            VERIFY(cmpeq32_64(mask, b == i1));
        } else {
            VERIFY(mask == (b == i1));
        }
        b.gather(array, &S::c, i, mask);
        if (sizeof(typename Vec::Type) == 8) {
            VERIFY(cmpeq32_64(mask, b == i2));
        } else {
            VERIFY(mask == (b == i2));
        }
    }
}

template<typename Vec> void gatherStruct16()
{
    typedef ushort_v It;
    typedef typename Vec::Type T;
    typedef Struct<T> S;
    const int count = 3999;
    S array[count];
    for (int i = 0; i < count; ++i) {
        array[i].a = i;
        array[i].b = i + 1;
        array[i].c = i + 2;
    }
    Mask mask;
    for (It i = It::IndexesFromZero; (mask = (i < count)); i += Vec::Size) {
        // if Vec is double_v the staticCast keeps only the lower two values, which is why the ==
        // comparison works
        const Vec &i0 = i.staticCast<T>();
        const Vec &i1 = (i + 1).staticCast<T>();
        const Vec &i2 = (i + 2).staticCast<T>();
        if (FullMask == mask) {
            Vec a(array, &S::a, i);
            VERIFY(FullMask == (a == i0));
            a.gather(array, &S::b, i);
            VERIFY(FullMask == (a == i1));
            a.gather(array, &S::c, i);
            VERIFY(FullMask == (a == i2));
        }

        Vec b;
        b.gather(array, &S::a, i, mask);
        VERIFY(mask == (b == i0));
        b.gather(array, &S::b, i, mask);
        VERIFY(mask == (b == i1));
        b.gather(array, &S::c, i, mask);
        VERIFY(mask == (b == i2));
    }
}

int main()
{
    runTest(gatherArray<int_v>);
    runTest(gatherArray<uint_v>);
    runTest(gatherArray<float_v>);
    runTest(gatherArray<double_v>);
    runTest(gatherArray16<short_v>);
    runTest(gatherArray16<ushort_v>);
    runTest(gatherStruct<int_v>);
    runTest(gatherStruct<uint_v>);
    runTest(gatherStruct<float_v>);
    runTest(gatherStruct<double_v>);
    runTest(gatherStruct16<short_v>);
    runTest(gatherStruct16<ushort_v>);
    return 0;
}
