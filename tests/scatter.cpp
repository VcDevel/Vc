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

template<typename Vec> void scatterArray()
{
    const int count = 39999;
    typename Vec::EntryType array[count], out[count];
    for (int i = 0; i < count; ++i) {
        array[i] = i;
    }
    typename uint_v::Mask mask;
    for (uint_v i(IndexesFromZero); !(mask = (i < count)).isEmpty(); i += Vec::Size) {
        Vec a(array, i, mask.cast<Vec::Size>());
        a.scatter(out, i, mask.cast<Vec::Size>());
    }
    COMPARE(0, std::memcmp(array, out, count * sizeof(typename Vec::EntryType)));
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

template<typename Vec> void scatterStruct()
{
    typedef Struct<typename Vec::EntryType> S;
    const int count = 3999;
    S array[count], out[count];
    memset(array, 0, count * sizeof(S));
    memset(out, 0, count * sizeof(S));
    for (int i = 0; i < count; ++i) {
        array[i].a = i;
        array[i].b = i + 1;
        array[i].c = i + 2;
    }
    typename uint_v::Mask mask;
    for (uint_v i(IndexesFromZero); !(mask = (i < count)).isEmpty(); i += Vec::Size) {
        Vec a(array, &S::a, i, mask.cast<Vec::Size>());
        Vec b(array, &S::b, i, mask.cast<Vec::Size>());
        Vec c(array, &S::c, i, mask.cast<Vec::Size>());
        a.scatter(out, &S::a, i, mask.cast<Vec::Size>());
        b.scatter(out, &S::b, i, mask.cast<Vec::Size>());
        c.scatter(out, &S::c, i, mask.cast<Vec::Size>());
    }
    VERIFY(0 == memcmp(array, out, count * sizeof(S)));
}

int main()
{
    runTest(scatterArray<int_v>);
    runTest(scatterArray<uint_v>);
    runTest(scatterArray<float_v>);
    runTest(scatterArray<double_v>);
    runTest(scatterStruct<int_v>);
    runTest(scatterStruct<uint_v>);
    runTest(scatterStruct<float_v>);
    runTest(scatterStruct<double_v>);
    return 0;
}
