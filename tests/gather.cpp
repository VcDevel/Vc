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

using namespace Vc;

template<typename Vec> void gatherArray()
{
    typedef typename Vec::IndexType It;
    typedef typename Vec::EntryType T;
    typedef typename It::Mask M;

    const int count = 39999;
    T array[count];
    for (int i = 0; i < count; ++i) {
        array[i] = i;
    }
    M mask;
    for (It i = It(IndexesFromZero); !(mask = (i < count)).isEmpty(); i += Vec::Size) {
        const Vec ii(i);
        const typename Vec::Mask castedMask(mask);
        if (castedMask.isFull()) {
            Vec a(array, i);
            COMPARE(a, ii);
            Vec b(Zero);
            b.gather(array, i);
            COMPARE(b, ii);
            COMPARE(a, b);
        }
        Vec b(Zero);
        b.gather(array, i, castedMask);
        COMPARE(castedMask, b == ii);
        if (!castedMask.isFull()) {
            COMPARE(!castedMask, b == Vec(Zero));
        }
    }

    const typename Vec::Mask k(Zero);
    Vec a(One);
    a.gather(array, It(IndexesFromZero), k);
    COMPARE(a, Vec(One));
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
    typedef typename Vec::IndexType It;
    typedef typename Vec::EntryType T;
    typedef Struct<T> S;
    const int count = 3999;
    S array[count];
    for (int i = 0; i < count; ++i) {
        array[i].a = i;
        array[i].b = i + 1;
        array[i].c = i + 2;
    }
    typename It::Mask mask;
    for (It i = It(IndexesFromZero); !(mask = (i < count)).isEmpty(); i += Vec::Size) {
        // if Vec is double_v the cast keeps only the lower two values, which is why the ==
        // comparison works
        const Vec i0(i);
        const Vec i1(i + 1);
        const Vec i2(i + 2);
        const typename Vec::Mask castedMask(mask);

        if (castedMask.isFull()) {
            Vec a(array, &S::a, i);
            COMPARE(a, i0);
            a.gather(array, &S::b, i);
            COMPARE(a, i1);
            a.gather(array, &S::c, i);
            COMPARE(a, i2);
        }

        Vec b(Zero);
        b.gather(array, &S::a, i, castedMask);
        COMPARE(castedMask, (b == i0));
        if (!castedMask.isFull()) {
            COMPARE(!castedMask, b == Vec(Zero));
        }
        b.gather(array, &S::b, i, castedMask);
        COMPARE(castedMask, (b == i1));
        if (!castedMask.isFull()) {
            COMPARE(!castedMask, b == Vec(Zero));
        }
        b.gather(array, &S::c, i, castedMask);
        COMPARE(castedMask, (b == i2));
        if (!castedMask.isFull()) {
            COMPARE(!castedMask, b == Vec(Zero));
        }
    }
}

template<typename T> struct Row
{
    T *data;
};

template<typename Vec> void gather2dim()
{
    typedef typename Vec::IndexType It;
    typedef typename Vec::EntryType T;
    const int count = 3999;
    typedef Row<T> S;
    S array[count];
    for (int i = 0; i < count; ++i) {
        array[i].data = new T[count];
        for (int j = 0; j < count; ++j) {
            array[i].data[j] = 2 * i + j;
        }
    }

    typename It::Mask mask;
    for (It i = It(IndexesFromZero); !(mask = (i < count)).isEmpty(); i += Vec::Size) {
        for (It j = It(IndexesFromZero); !(mask &= (j < count)).isEmpty(); j += Vec::Size) {
            const Vec i0(i * 2 + j);
            const typename Vec::Mask castedMask(mask);

            Vec a(array, &S::data, i, j, castedMask);
            COMPARE(castedMask, castedMask && (a == i0));

            Vec b(Zero);
            b.gather(array, &S::data, i, j, castedMask);
            COMPARE(castedMask, (b == i0));
            if (!castedMask.isFull()) {
                COMPARE(!castedMask, b == Vec(Zero));
            }
        }
    }
    for (int i = 0; i < count; ++i) {
        delete[] array[i].data;
    }
}

int main()
{
    runTest(gatherArray<int_v>);
    runTest(gatherArray<uint_v>);
    runTest(gatherArray<float_v>);
    runTest(gatherArray<double_v>);
    runTest(gatherArray<short_v>);
    runTest(gatherArray<ushort_v>);
    runTest(gatherArray<sfloat_v>);

    runTest(gatherStruct<int_v>);
    runTest(gatherStruct<uint_v>);
    runTest(gatherStruct<float_v>);
    runTest(gatherStruct<double_v>);
    runTest(gatherStruct<short_v>);
    runTest(gatherStruct<ushort_v>);
    runTest(gatherStruct<sfloat_v>);

    runTest(gather2dim<int_v>);
    runTest(gather2dim<uint_v>);
    runTest(gather2dim<short_v>);
    runTest(gather2dim<ushort_v>);
    runTest(gather2dim<float_v>);
    runTest(gather2dim<sfloat_v>);
    runTest(gather2dim<double_v>);

    return 0;
}
