/*  This file is part of the Vc library. {{{
Copyright © 2009-2014 Matthias Kretz <kretz@kde.org>

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
// includes {{{1
#include "unittest.h"
#include <iostream>
#include <cstring>
#include <Vc/array>

using namespace Vc;

TEST_TYPES(Vec, scatterArray, AllTypes) //{{{1
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
        auto castedMask = simd_cast<typename Vec::Mask>(mask);
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
        auto castedMask = simd_cast<typename Vec::Mask>(mask);
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

TEST_TYPES(Vec, maskedScatterArray, AllTypes) //{{{1
{
    typedef typename Vec::IndexType It;
    typedef typename Vec::EntryType T;

    Vc::array<T, Vec::Size> mem;
    const Vec v = Vec([](T n) { return n + 1; });

    withRandomMask<Vec>([&](typename Vec::mask_type m) {
        Vec(0).store(&mem[0], Vc::Unaligned);
        where(m) | mem[It([](int n) { return n; })] = v;

        Vec reference = v;
        reference.setZeroInverted(m);

        COMPARE(Vec(&mem[0], Vc::Unaligned), reference) << "m = " << m;
    });
}

//struct Struct {{{1
template <typename T, size_t Align = std::is_arithmetic<T>::value ? sizeof(T) : alignof(T)>
struct alignas(Align > alignof(short) ? Align : alignof(short)) Struct
{
    T a;
    char x;
    T b;
    short y;
    T c;
    char z;
};

TEST_TYPES(Vec, scatterStruct, AllTypes) //{{{1
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
        auto castedMask = simd_cast<typename Vec::Mask>(mask);
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
TEST_TYPES(Vec, scatterStruct2, AllTypes) //{{{1
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
        castedMask = simd_cast<decltype(castedMask)>(mask);
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
