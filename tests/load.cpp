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

using namespace Vc;

#define ALL_TYPES (ALL_VECTORS)

template<typename T> static constexpr T min(T a, T b) { return a < b ? a : b; }

template<typename Vec> constexpr unsigned long alignmentMask()
{
    return Vec::Size == 1 ? (
            // on 32bit the maximal alignment is 4 Bytes, even for 8-Byte doubles.
            min(sizeof(void*), sizeof(typename Vec::EntryType)) - 1
        ) : (
            // AVX::VectorAlignment is too large
            min(sizeof(Vec), alignof(Vec)) - 1
        );
}

TEST_TYPES(Vec, checkAlignment, ALL_TYPES)
{
    unsigned char i = 1;
    Vec a[10];
    unsigned long mask = alignmentMask<Vec>();
    for (i = 0; i < 10; ++i) {
        VERIFY((reinterpret_cast<size_t>(&a[i]) & mask) == 0) << "a = " << a << ", mask = " << mask;
    }
    const char *data = reinterpret_cast<const char *>(&a[0]);
    for (i = 0; i < 10; ++i) {
        VERIFY(&data[i * sizeof(Vec)] == reinterpret_cast<const char *>(&a[i]));
    }
}

void *hack_to_put_b_on_the_stack = 0;

TEST_TYPES(Vec, checkMemoryAlignment, ALL_TYPES)
{
    typedef typename Vec::EntryType T;
    const T *b = 0;
    Vc::Memory<Vec, 10> a;
    b = a;
    hack_to_put_b_on_the_stack = &b;
    size_t mask = memory_alignment<Vec>::value - 1;
    for (int i = 0; i < 10; ++i) {
        VERIFY((reinterpret_cast<size_t>(&b[i * Vec::Size]) & mask) == 0) << "b = " << b << ", mask = " << mask;
    }
}

enum Enum {
    loadArrayShortCount = 32 * 1024,
    streamingLoadCount = 1024
};
TEST_TYPES(Vec, loadArrayShort, (short_v, ushort_v, simdarray<short, 32>, simdarray<unsigned short, 32>))
{
    typedef typename Vec::EntryType T;

    Vc::Memory<Vec, loadArrayShortCount> array;
    for (int i = 0; i < loadArrayShortCount; ++i) {
        array[i] = i;
    }

    const Vec offsets(IndexesFromZero);
    for (int i = 0; i < loadArrayShortCount; i += Vec::Size) {
        const T *const addr = &array[i];
        Vec ii(i);
        ii += offsets;

        Vec a(addr, Vc::Aligned);
        COMPARE(a, ii);

        Vec b = Vec::Zero();
        b.load(addr, Vc::Aligned);
        COMPARE(b, ii);
    }
}

TEST_TYPES(Vec, loadArray, ALL_TYPES)
{
    typedef typename Vec::EntryType T;
    if (sizeof(T) < 4) {
        return;
    }

    enum loadArrayEnum { count = 256 * 1024 / sizeof(T) };
    Vc::Memory<Vec, count> array;
    for (int i = 0; i < count; ++i) {
        array[i] = i;
    }

    const Vec offsets(IndexesFromZero);
    for (int i = 0; i < count; i += Vec::Size) {
        const T *const addr = &array[i];
        Vec ii(i);
        ii += offsets;

        Vec a(addr, Vc::Aligned);
        COMPARE(a, ii);

        Vec b = Vec::Zero();
        b.load(addr, Vc::Aligned);
        COMPARE(b, ii);
    }

    // check that Vc allows construction from objects that auto-convert to T*
    Vec tmp0(array, Vc::Aligned);
    tmp0.load(array, Vc::Aligned);
}

TEST_TYPES(Vec, streamingLoad, ALL_TYPES)
{
    typedef typename Vec::EntryType T;

    Vc::Memory<Vec, streamingLoadCount> data;
    data[0] = static_cast<T>(-streamingLoadCount/2);
    for (int i = 1; i < streamingLoadCount; ++i) {
        data[i] = data[i - 1];
        ++data[i];
    }

    Vec ref = data.firstVector();
    for (size_t i = 0; i < streamingLoadCount - Vec::Size; ++i, ++ref) {
        Vec v1, v2;
        if (0 == i % Vec::Size) {
            v1 = Vec(&data[i], Vc::Aligned | Vc::Streaming);
            v2.load (&data[i], Vc::Aligned | Vc::Streaming);
        } else {
            v1 = Vec(&data[i], Vc::Streaming | Vc::Unaligned);
            v2.load (&data[i], Vc::Unaligned | Vc::Streaming);
        }
        COMPARE(v1, ref) << ", i = " << i;
        COMPARE(v2, ref) << ", i = " << i;
    }
}

template<typename T, typename Current = void> struct SupportedConversions { typedef void Next; };
template<> struct SupportedConversions<float, void>           { typedef double         Next; };
template<> struct SupportedConversions<float, double>         { typedef int            Next; };
template<> struct SupportedConversions<float, int>            { typedef unsigned int   Next; };
template<> struct SupportedConversions<float, unsigned int>   { typedef short          Next; };
template<> struct SupportedConversions<float, short>          { typedef unsigned short Next; };
template<> struct SupportedConversions<float, unsigned short> { typedef signed char    Next; };
template<> struct SupportedConversions<float, signed char>    { typedef unsigned char  Next; };
template<> struct SupportedConversions<float, unsigned char>  { typedef void           Next; };
template<> struct SupportedConversions<int  , void          > { typedef unsigned int   Next; };
template<> struct SupportedConversions<int  , unsigned int  > { typedef short          Next; };
template<> struct SupportedConversions<int  , short         > { typedef unsigned short Next; };
template<> struct SupportedConversions<int  , unsigned short> { typedef signed char    Next; };
template<> struct SupportedConversions<int  , signed char   > { typedef unsigned char  Next; };
template<> struct SupportedConversions<int  , unsigned char > { typedef void           Next; };
template<> struct SupportedConversions<unsigned int, void          > { typedef unsigned short Next; };
template<> struct SupportedConversions<unsigned int, unsigned short> { typedef unsigned char  Next; };
template<> struct SupportedConversions<unsigned int, unsigned char > { typedef void           Next; };
template<> struct SupportedConversions<unsigned short, void          > { typedef unsigned char  Next; };
template<> struct SupportedConversions<unsigned short, unsigned char > { typedef void           Next; };
template<> struct SupportedConversions<         short, void          > { typedef unsigned char  Next; };
template<> struct SupportedConversions<         short, unsigned char > { typedef signed char    Next; };
template<> struct SupportedConversions<         short,   signed char > { typedef void           Next; };

template<typename Vec, typename MemT> struct LoadCvt {
    static void test() {
        typedef typename Vec::EntryType VecT;
        MemT *data = Vc::malloc<MemT, Vc::AlignOnCacheline>(128);
        for (size_t i = 0; i < 128; ++i) {
            data[i] = static_cast<MemT>(i - 64);
        }

        for (size_t i = 0; i < 128 - Vec::Size + 1; ++i) {
            Vec v;
            if (i % (2 * Vec::Size) == 0) {
                v = Vec(&data[i]);
            } else if (i % Vec::Size == 0) {
                v = Vec(&data[i], Vc::Aligned);
            } else {
                v = Vec(&data[i], Vc::Unaligned);
            }
            for (size_t j = 0; j < Vec::Size; ++j) {
                COMPARE(v[j], static_cast<VecT>(data[i + j])) << " " << UnitTest::typeToString<MemT>();
            }
        }
        for (size_t i = 0; i < 128 - Vec::Size + 1; ++i) {
            Vec v;
            if (i % (2 * Vec::Size) == 0) {
                v.load(&data[i]);
            } else if (i % Vec::Size == 0) {
                v.load(&data[i], Vc::Aligned);
            } else {
                v.load(&data[i], Vc::Unaligned);
            }
            for (size_t j = 0; j < Vec::Size; ++j) {
                COMPARE(v[j], static_cast<VecT>(data[i + j])) << " " << UnitTest::typeToString<MemT>();
            }
        }
        for (size_t i = 0; i < 128 - Vec::Size + 1; ++i) {
            Vec v;
            if (i % (2 * Vec::Size) == 0) {
                v = Vec(&data[i], Vc::Streaming);
            } else if (i % Vec::Size == 0) {
                v = Vec(&data[i], Vc::Streaming | Vc::Aligned);
            } else {
                v = Vec(&data[i], Vc::Streaming | Vc::Unaligned);
            }
            for (size_t j = 0; j < Vec::Size; ++j) {
                COMPARE(v[j], static_cast<VecT>(data[i + j])) << " " << UnitTest::typeToString<MemT>();
            }
        }

        UnitTest::ADD_PASS() << "loadCvt: load " << UnitTest::typeToString<MemT>() << "* as " << UnitTest::typeToString<Vec>();
        LoadCvt<Vec, typename SupportedConversions<VecT, MemT>::Next>::test();
    }
};
template<typename Vec> struct LoadCvt<Vec, void> { static void test() {} };

TEST_TYPES(Vec, loadCvt, ALL_TYPES)
{
    typedef typename Vec::EntryType T;
    LoadCvt<Vec, typename SupportedConversions<T>::Next>::test();
}
