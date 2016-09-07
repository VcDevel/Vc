/*  This file is part of the Vc library. {{{
Copyright © 2009-2015 Matthias Kretz <kretz@kde.org>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the names of contributing organizations nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#include "unittest.h"
#include <iostream>
#include <Vc/array>

#define ALL_TYPES (ALL_VECTORS, SimdArray<int, 7>)

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
    alignas(static_cast<std::size_t>(It::MemoryAlignment))
        std::array<typename It::EntryType, It::size()> indexArray;
    indexes.store(&indexArray[0], Vc::Aligned);
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
        const Vec c(mem, indexArray, m);
        for (size_t i = 0; i < Vec::Size; ++i) {
            COMPARE(a[i], m[i] ? mem[i] : 0) << " i = " << i << ", m = " << m;
        }

        b = x;
        b.gather(mem, indexArray, m);
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
    return Vc::simd_cast<Vec>(i + i.One());
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
    const auto overflowing =
        Vc::simd_cast<typename Vec::Mask>(i >= IT((std::numeric_limits<T>::max)()));
    Vec r = Vc::simd_cast<Vec>(i + IT::One());
    where(overflowing) | r = Vc::simd_cast<Vec>(i - (std::numeric_limits<T>::max)() + (std::numeric_limits<T>::min)());
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
        const typename Vec::Mask castedMask = simd_cast<typename Vec::Mask>(mask);
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

template<typename T, std::size_t Align> struct Struct
{
    alignas(Align) T a;
    char x;
    alignas(Align) T b;
    short y;
    alignas(Align) T c;
    char z;
};

TEST_TYPES(Vec, gatherStruct, ALL_TYPES)
{
    typedef typename Vec::IndexType It;
    typedef typename Vec::EntryType T;
    typedef Struct<T, alignof(T)> S;
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
        const Vec i0 = Vc::simd_cast<Vec>(i);
        const Vec i1 = Vc::simd_cast<Vec>(i + 1);
        const Vec i2 = Vc::simd_cast<Vec>(i + 2);
        const auto castedMask = simd_cast<typename Vec::Mask>(mask);

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
#ifdef Vc_GCC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
            array[i].data[j] = 2 * i + j + 1;
#ifdef Vc_GCC
#pragma GCC diagnostic pop
#endif
        }
    }

    typename It::Mask mask;
    for (It i = It(IndexesFromZero); !(mask = (i < count)).isEmpty(); i += Vec::Size) {
        for (It j = It(IndexesFromZero); !(mask &= (j < count)).isEmpty(); j += Vec::Size) {
            const Vec i0 = Vc::simd_cast<Vec>(i * 2 + j + 1);
            const auto castedMask = simd_cast<typename Vec::Mask>(mask);

            Vec a;
            where(castedMask) | a = array[i][&S::data][j];
            COMPARE(castedMask, castedMask && (a == i0)) << ", a = " << a << ", i0 = " << i0 << ", i = " << i << ", j = " << j;

            Vec b(Zero);
            where(castedMask) | b = array[i][&S::data][j];
            COMPARE(castedMask, (b == i0));

            b.setZero();
            b(castedMask) = array[i][&S::data][j];
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
