/*  This file is part of the Vc library.

    Copyright (C) 2009-2014 Matthias Kretz <kretz@kde.org>

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
// includes {{{1
#include "unittest.h"
#include <iostream>
#include <cstring>
#include <Vc/array>

using namespace Vc;

#define ALL_TYPES /*SIMD_ARRAYS(32), SIMD_ARRAYS(16), SIMD_ARRAYS(8), SIMD_ARRAYS(4), SIMD_ARRAYS(2), SIMD_ARRAYS(1),*/ ALL_VECTORS

TEST_TYPES(Vec, scatterArray, (ALL_TYPES)) //{{{1
{
    typedef typename Vec::EntryType T;
    typedef typename Vec::IndexType It;
    constexpr int count = 31999;
    Vc::array<T, count> array, out;
    for (int i = 0; i < count; ++i) {
        array[i] = i;
        if (!std::is_integral<T>::value || !std::is_unsigned<T>::value) {
            array[i] -= 100;
        }
    }
    typename It::Mask mask;
    for (It i(IndexesFromZero); !(mask = (i < count)).isEmpty(); i += Vec::Size) {
        typename Vec::Mask castedMask(mask);
        if (all_of(castedMask)) {
            Vec a(&array[0], i);
            a += 1;
            a.scatter(&out[0], i);
        } else {
            Vec a(&array[0], i, castedMask);
            a += 1;
            a.scatter(&out[0], i, castedMask);
        }
    }
    for (int i = 0; i < count; ++i) {
        array[i] += 1;
        COMPARE(array[i], out[i]);
    }
    COMPARE(0, std::memcmp(&array[0], &out[0], count * sizeof(typename Vec::EntryType)));

    for (It i(IndexesFromZero); !(mask = (i < count)).isEmpty(); i += Vec::Size) {
        typename Vec::Mask castedMask(mask);
        if (all_of(castedMask)) {
            Vec a = array[i];
            out[i] = a + 1;
        } else {
            Vec a;
            where(castedMask) | a = array[i];
            where(castedMask) | out[i] = a + 1;
        }
    }
    for (int i = 0; i < count; ++i) {
        array[i] += 1;
        COMPARE(array[i], out[i]);
    }
    COMPARE(0, std::memcmp(&array[0], &out[0], count * sizeof(typename Vec::EntryType)));
}

TEST_TYPES(Vec, maskedScatterArray, (ALL_TYPES)) //{{{1
{
    typedef typename Vec::IndexType It;
    typedef typename Vec::EntryType T;

    Vc::array<T, Vec::Size> mem;
    const Vec v = Vec::IndexesFromZero() + 1;

    for_all_masks(Vec, m) {
        Vec::Zero().store(&mem[0], Vc::Unaligned);
        where(m) | mem[It::IndexesFromZero()] = v;

        Vec reference = v;
        reference.setZeroInverted(m);

        COMPARE(Vec(&mem[0], Vc::Unaligned), reference) << "m = " << m;
    }
}

template<typename T> struct Struct //{{{1
{
    T a;
    char x;
    T b;
    short y;
    T c;
    char z;
};

TEST_TYPES(Vec, scatterStruct, (ALL_TYPES)) //{{{1
{
    typedef typename Vec::IndexType It;
    typedef Struct<typename Vec::EntryType> S;
    constexpr int count = 3999;
    Vc::array<S, count> array, out;
    memset(&array[0], 0, count * sizeof(S));
    memset(&out[0], 0, count * sizeof(S));
    for (int i = 0; i < count; ++i) {
        array[i].a = i;
        array[i].b = i + 1;
        array[i].c = i + 2;
    }
    typename It::Mask mask;
    for (It i(IndexesFromZero); !(mask = (i < count)).isEmpty(); i += Vec::Size) {
        typename Vec::Mask castedMask(mask);
        Vec a; a(castedMask) = array[i][&S::a];
        where(castedMask) | out[i][&S::a] = a;
        Vec b; b(castedMask) = array[i][&S::b];
        where(castedMask) | out[i][&S::b] = b;
        Vec c; c(castedMask) = array[i][&S::c];
        where(castedMask) | out[i][&S::c] = c;
    }
    VERIFY(0 == memcmp(&array[0], &out[0], count * sizeof(S)));
}

template<typename T> struct Struct2 //{{{1
{
    char x;
    Struct<T> b;
    short y;
};

constexpr int scatterStruct2Count = 97;

template<typename T>
static std::ostream &operator<<(std::ostream &out, const Struct2<T> &s)
{
    return out << '{' << s.b.a << ' ' << s.b.b << ' ' << s.b.c << '}';
}

template<typename T>
static std::ostream &operator<<(std::ostream &out, const Struct2<T> *s)
{
    for (int i = 0; i < scatterStruct2Count; ++i) {
        out << s[i];
    }
    return out;
}

template <typename T, std::size_t N>
static std::ostream &operator<<(std::ostream &out, const Vc::array<T, N> &x)
{
    out << x[0];
    for (std::size_t i = 1; i < N; ++i) {
        out << ' ' << x[i];
    }
    return out;
}

template<typename V> V makeReference(V v, typename V::Mask m)
{
    v.setZero(!m);
    return v;
}
TEST_TYPES(Vec, scatterStruct2, (ALL_TYPES)) //{{{1
{
    typedef typename Vec::IndexType It;
    typedef Struct2<typename Vec::EntryType> S1;
    typedef Struct<typename Vec::EntryType> S2;
    Vc::array<S1, scatterStruct2Count> array, out;
    memset(&array[0], 0, scatterStruct2Count * sizeof(S1));
    memset(&out[0], 0, scatterStruct2Count * sizeof(S1));
    for (int i = 0; i < scatterStruct2Count; ++i) {
        array[i].b.a = i + 0;
        array[i].b.b = i + 1;
        array[i].b.c = i + 2;
    }
    typename It::Mask mask;
    typename Vec::Mask castedMask;
    for (It i(IndexesFromZero); !(mask = (i < scatterStruct2Count)).isEmpty(); i += Vec::Size) {
        castedMask = static_cast<decltype(castedMask)>(mask);
        Vec a = Vec(); a(castedMask) = array[i][&S1::b][&S2::a];
        Vec b = Vec(); b(castedMask) = array[i][&S1::b][&S2::b];
        Vec c = Vec(); c(castedMask) = array[i][&S1::b][&S2::c];
        COMPARE(a, Vc::simd_cast<Vec>(makeReference(i, mask)));
        COMPARE(b, Vc::simd_cast<Vec>(makeReference(i + 1, mask)));
        COMPARE(c, Vc::simd_cast<Vec>(makeReference(i + 2, mask)));
        where(castedMask) | out[i][&S1::b][&S2::a] = a;
        where(castedMask) | out[i][&S1::b][&S2::b] = b;
        where(castedMask) | out[i][&S1::b][&S2::c] = c;
    }
    // castedmask != mask here because mask is changed in the for loop, but castedmask has the value
    // from the previous iteration
    VERIFY(0 == memcmp(&array[0], &out[0], scatterStruct2Count * sizeof(S1))) << mask << ' ' << castedMask << '\n'
        << array << '\n' << out;
}

// vim: foldmethod=marker
