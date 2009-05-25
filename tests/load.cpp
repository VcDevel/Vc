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

template<typename Vec> void checkAlignment()
{
    unsigned char i = 1;
    Vec a[10];
    unsigned long mask = VectorAlignment - 1;
    if (Vec::Size == 1 && sizeof(typename Vec::EntryType) != VectorAlignment) {
        mask = sizeof(typename Vec::EntryType) - 1;
    }
    for (i = 0; i < 10; ++i) {
        VERIFY((reinterpret_cast<unsigned long>(&a[i]) & mask) == 0);
    }
    const char *data = reinterpret_cast<const char *>(&a[0]);
    for (i = 0; i < 10; ++i) {
        VERIFY(&data[i * Vec::Size * sizeof(typename Vec::EntryType)] == reinterpret_cast<const char *>(&a[i]));
    }
}

template<typename Vec> void loadArray()
{
    typedef typename Vec::EntryType T;

    const int count = 256 * 1024 / sizeof(T);
    T array[count];
    for (int i = 0; i < count; ++i) {
        array[i] = i;
    }

    const Vec &offsets = int_v(IndexesFromZero).staticCast<T>();
    for (int i = 0; i < count; i += Vec::Size) {
        const T *const addr = &array[i];
        Vec ii(i);
        ii += offsets;

        Vec a(addr);
        COMPARE(a, ii);

        Vec b(Zero);
        b.load(addr);
        COMPARE(b, ii);
    }
}

template<typename Vec> void loadArrayShort()
{
    typedef typename Vec::EntryType T;

    const int count = 32 * 1024;
    T array[count];
    for (int i = 0; i < count; ++i) {
        array[i] = i;
    }

    const Vec &offsets = ushort_v(IndexesFromZero).staticCast<T>();
    for (int i = 0; i < count; i += Vec::Size) {
        const T *const addr = &array[i];
        Vec ii(i);
        ii += offsets;

        Vec a(addr);
        COMPARE(a, ii);

        Vec b(Zero);
        b.load(addr);
        COMPARE(b, ii);
    }
}

int main()
{
#if !defined(ENABLE_LARRABEE) && !defined(__LRB__)
    runTest(checkAlignment<int_v>);
    runTest(checkAlignment<uint_v>);
    runTest(checkAlignment<float_v>);
    runTest(checkAlignment<double_v>);
    runTest(checkAlignment<short_v>);
    runTest(checkAlignment<ushort_v>);
#endif
    runTest(loadArray<int_v>);
    runTest(loadArray<uint_v>);
    runTest(loadArray<float_v>);
    runTest(loadArray<double_v>);
    runTest(loadArrayShort<short_v>);
    runTest(loadArrayShort<ushort_v>);
    return 0;
}
