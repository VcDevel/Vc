/*  This file is part of the Vc library.

    Copyright (C) 2009-2012 Matthias Kretz <kretz@kde.org>

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

#include "unittest.h"
#include <iostream>
#include <Vc/array>

#define ALL_TYPES (ALL_VECTORS)

using namespace Vc;

TEST_TYPES(Vec, maskedGatherArray, ALL_TYPES)
{
    typedef typename Vec::IndexType It;
    typedef typename Vec::EntryType T;

    T mem[Vec::Size];
    for (size_t i = 0; i < Vec::Size; ++i) {
        mem[i] = i + 1;
    }

    It indexes = It::IndexesFromZero();
    for_all_masks(Vec, m) {
        const Vec a(mem, indexes, m);
        for (size_t i = 0; i < Vec::Size; ++i) {
            COMPARE(a[i], m[i] ? mem[i] : 0) << " i = " << i << ", m = " << m;
        }

        T x = Vec::Size + 1;
        Vec b = x;
        b.gather(mem, indexes, m);
        for (size_t i = 0; i < Vec::Size; ++i) {
            COMPARE(b[i], m[i] ? mem[i] : x) << " i = " << i << ", m = " << m;
        }

        // test with array of indexes instead of index-vector:
        const Vec c(mem, &indexes[0], m);
        for (size_t i = 0; i < Vec::Size; ++i) {
            COMPARE(a[i], m[i] ? mem[i] : 0) << " i = " << i << ", m = " << m;
        }

        b = x;
        b.gather(mem, &indexes[0], m);
        for (size_t i = 0; i < Vec::Size; ++i) {
            COMPARE(b[i], m[i] ? mem[i] : x) << " i = " << i << ", m = " << m;
        }
    }
}

template <typename Vec>
Vec incrementIndex(
    const typename Vec::IndexType &i,
    typename std::enable_if<!(Vc::is_integral<Vec>::value &&Vc::is_signed<Vec>::value),
                            void *>::type = nullptr)
{
    return static_cast<Vec>(i + i.One());
}

template <typename Vec>
Vec incrementIndex(const typename Vec::IndexType &i,
                   typename std::enable_if<Vc::is_integral<Vec>::value &&Vc::is_signed<Vec>::value,
                                           void *>::type = nullptr)
{
    using IT = typename Vec::IndexType;
    using T = typename Vec::EntryType;
    // if (i + 1) > std::numeric_limits<Vec>::max() it will overflow, which results in
    // undefined behavior for signed integers
    const typename Vec::Mask overflowing{i >= static_cast<IT>(std::numeric_limits<T>::max())};
    Vec r(i + IT::One());
    where(overflowing) | r = static_cast<Vec>(i - std::numeric_limits<T>::max() + std::numeric_limits<T>::min());
    return r;
}

TEST_TYPES(Vec, gatherArray, ALL_TYPES)
{
    typedef typename Vec::IndexType It;
    typedef typename Vec::EntryType T;
    typedef typename It::Mask M;

    const int count = 39999;
    T array[count];
    for (int i = 0; i < count; ++i) {
        array[i] = i + 1;
    }
    M mask;
    for (It i = It(IndexesFromZero); !(mask = (i < count)).isEmpty(); i += Vec::Size) {
        const Vec ii = incrementIndex<Vec>(i);
        const typename Vec::Mask castedMask = static_cast<typename Vec::Mask>(mask);
        if (all_of(castedMask)) {
            Vec a(array, i);
            COMPARE(a, ii) << "\n       i: " << i;
            Vec b(Zero);
            b.gather(array, i);
            COMPARE(b, ii);
            COMPARE(a, b);
        }
        Vec b(Zero);
        b.gather(array, i, castedMask);
        COMPARE(castedMask, (b == ii)) << ", b = " << b << ", ii = " << ii << ", i = " << i;
        if (!all_of(castedMask)) {
            COMPARE(!castedMask, b == Vec(Zero)) << "\nb: " << b << "\ncastedMask: " << castedMask << !castedMask;
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

TEST_TYPES(Vec, gatherStruct, ALL_TYPES)
{
    typedef typename Vec::IndexType It;
    typedef typename Vec::EntryType T;
    typedef Struct<T> S;
    constexpr int count = 3999;
    Vc::array<S, count> array;
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
            Vec a = array[i][&S::a];
            COMPARE(a, i0) << "\ni: " << i;
            a = array[i][&S::b];
            COMPARE(a, i1);
            a = array[i][&S::c];
            COMPARE(a, i2);
        }

        Vec b(Zero);
        where(castedMask) | b = array[i][&S::a];
        COMPARE(castedMask, (b == i0));
        if (!castedMask.isFull()) {
            COMPARE(!castedMask, b == Vec(Zero));
        }
        where(castedMask) | b = array[i][&S::b];
        COMPARE(castedMask, (b == i1));
        if (!castedMask.isFull()) {
            COMPARE(!castedMask, b == Vec(Zero));
        }
        where(castedMask) | b = array[i][&S::c];
        COMPARE(castedMask, (b == i2));
        if (!castedMask.isFull()) {
            COMPARE(!castedMask, b == Vec(Zero));
        }
    }
}

template<typename T, int N> struct Row
{
    T data[N];
};

TEST_TYPES(Vec, gather2dim, ALL_TYPES)
{
    typedef typename Vec::IndexType It;
    typedef typename Vec::EntryType T;
    constexpr int count = 19;
    typedef Row<T, count> S;
    Vc::array<S, count> array;
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < count; ++j) {
            array[i].data[j] = 2 * i + j + 1;
        }
    }

    typename It::Mask mask;
    for (It i = It(IndexesFromZero); !(mask = (i < count)).isEmpty(); i += Vec::Size) {
        for (It j = It(IndexesFromZero); !(mask &= (j < count)).isEmpty(); j += Vec::Size) {
            const Vec i0(i * 2 + j + 1);
            const typename Vec::Mask castedMask(mask);

            Vec a;
            where(castedMask) | a = array[i][&S::data][j];
            COMPARE(castedMask, castedMask && (a == i0)) << ", a = " << a << ", i0 = " << i0 << ", i = " << i << ", j = " << j;

            Vec b(Zero);
            where(castedMask) | b = array[i][&S::data][j];
            COMPARE(castedMask, (b == i0));
            if (!castedMask.isFull()) {
                COMPARE(!castedMask, b == Vec(Zero));
            } else {
                Vec c;
                c = array[i][&S::data][j];
                COMPARE(c, i0) << "i: " << i << ", j: " << j;
                VERIFY((c == i0).isFull());

                Vec d(Zero);
                d = array[i][&S::data][j];
                VERIFY((d == i0).isFull());
            }
        }
    }
}
