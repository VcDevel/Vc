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
#include <cstring>

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

template<typename Vec> void maskedStore()
{
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask M;
    M mask;
    {
        typedef typename Vec::IndexType I;
        const I tmp(IndexesFromZero);
        const typename I::Mask k = (tmp & I(One)) > 0;
        mask = M(k);
    }

    const int count = 256 * 1024 / sizeof(T);
    T array[count];
    T nullValue = 0;
    std::memset(array, 0, count * sizeof(T));
    T setValue = 170;
    const Vec x(setValue);
    for (int i = 0; i < count; i += Vec::Size) {
        x.store(&array[i], mask);
    }

    for (int i = 1; i < count; i += 2) {
        COMPARE(array[i], setValue);
    }
    for (int i = 0; i < count; i += 2) {
        COMPARE(array[i], nullValue);
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
    runTest(storeArray<sfloat_v>);

    if (float_v::Size > 1) {
        runTest(maskedStore<int_v>);
        runTest(maskedStore<uint_v>);
        runTest(maskedStore<float_v>);
        runTest(maskedStore<double_v>);
        runTest(maskedStore<short_v>);
        runTest(maskedStore<ushort_v>);
        runTest(maskedStore<sfloat_v>);
    }
    return 0;
}
