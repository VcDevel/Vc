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
#include <cstring>
#include "vecio.h"

using namespace Vc;

template<typename Vec> void storeArray()
{
    typedef typename Vec::EntryType T;

    const int count = 256 * 1024 / sizeof(T);
    T array[count];
    // do the memset to make sure the array doesn't have the old data from a previous call which
    // would mask a real problem
    std::memset(array, 0xff, count * sizeof(T));
    T xValue = 1;
    const Vec x(xValue);
    for (int i = 0; i < count; i += Vec::Size) {
        x.store(&array[i]);
    }

    for (int i = 0; i < count; ++i) {
        COMPARE(array[i], xValue);
    }
}

int main()
{
    runTest(storeArray<int_v>);
    runTest(storeArray<uint_v>);
    runTest(storeArray<float_v>);
    runTest(storeArray<double_v>);
    runTest(storeArray<short_v>);
    runTest(storeArray<ushort_v>);
    return 0;
}
