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

using namespace Vc;

#define ALL_TYPES AllVectors

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
TEST_TYPES(Vec, loadArrayShort, short_v, ushort_v, SimdArray<short, 32>,
           SimdArray<unsigned short, 32>)
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

        Vec b = Vec(0);
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

        Vec b = Vec(0);
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

TEST_TYPES(
    Pair, loadCvt,
    concat<outer_product<Typelist<float>,
                         Typelist<double, int, unsigned int, short, unsigned short,
                                  signed char, unsigned char>>,
           outer_product<Typelist<int>, Typelist<unsigned int, short, unsigned short,
                                                 signed char, unsigned char>>,
           outer_product<Typelist<unsigned int>, Typelist<unsigned short, unsigned char>>,
           outer_product<Typelist<short>, Typelist<unsigned char, unsigned char>>,
           outer_product<Typelist<unsigned short>, Typelist<unsigned char>>>)
{
    using Vec = Vector<typename Pair::template at<0>>;
    using MemT = typename Pair::template at<1>;

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
            COMPARE(v[j], static_cast<VecT>(data[i + j])) << "j: " << j;
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
            COMPARE(v[j], static_cast<VecT>(data[i + j])) << "j: " << j;
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
            COMPARE(v[j], static_cast<VecT>(data[i + j])) << "j: " << j;
        }
    }
}
