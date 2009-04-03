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

template<typename Vec> void loadArray()
{
    typedef typename Vec::Type T;

    const int count = 256 * 1024 / sizeof(T);
    T array[count];
    for (int i = 0; i < count; ++i) {
        array[i] = i;
    }

    const Vec &offsets = int_v(int_v::IndexesFromZero).staticCast<T>();
    for (int i = 0; i < count; i += Vec::Size) {
        const T *const addr = &array[i];
        Vec ii(i);
        ii += offsets;

        Vec a(addr);
        VERIFY(FullMask == (a == ii));

        Vec b(Zero);
        b.load(addr);
        VERIFY(FullMask == (b == ii));
    }
}

template<typename Vec> void loadArrayShort()
{
    typedef typename Vec::Type T;

    const int count = 32 * 1024;
    T array[count];
    for (int i = 0; i < count; ++i) {
        array[i] = i;
    }

    const Vec &offsets = ushort_v(ushort_v::IndexesFromZero).staticCast<T>();
    for (int i = 0; i < count; i += Vec::Size) {
        const T *const addr = &array[i];
        Vec ii(i);
        ii += offsets;

        Vec a(addr);
        VERIFY(FullMask == (a == ii));

        Vec b(Zero);
        b.load(addr);
        VERIFY(FullMask == (b == ii));
    }
}

int main()
{
    runTest(loadArray<int_v>);
    runTest(loadArray<uint_v>);
    runTest(loadArray<float_v>);
    runTest(loadArray<double_v>);
    runTest(loadArrayShort<short_v>);
    runTest(loadArrayShort<ushort_v>);
    return 0;
}
