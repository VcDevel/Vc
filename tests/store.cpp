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

#include "unittest.h"
#include <iostream>
#include <cstring>

using namespace Vc;

TEST_TYPES(Vec, alignedStore, AllVectors)
{
    typedef typename Vec::EntryType T;
    enum {
        Count = 256 * 1024 / sizeof(T)
    };

    Memory<Vec, Count> array;
    // do the memset to make sure the array doesn't have the old data from a previous call which
    // would mask a real problem
    std::memset(array, 0xff, Count * sizeof(T));
    T xValue = 1;
    const Vec x(xValue);
    for (int i = 0; i < Count; i += Vec::Size) {
        x.store(&array[i], Vc::Aligned);
    }

    for (int i = 0; i < Count; ++i) {
        COMPARE(array[i], xValue);
    }

    // make sure store can be used with parameters that auto-convert to T*
    x.store(array, Vc::Aligned);

    if (std::is_integral<T>::value && std::is_unsigned<T>::value) {
        // ensure that over-/underflowed values are stored correctly.
        Vec v = Vec(0) - Vec(1); // underflow
        v.store(array, Vc::Aligned);
        for (size_t i = 0; i < Vec::Size; ++i) {
            COMPARE(array[i], v[i]);
        }

        v = std::numeric_limits<T>::max() + Vec(1); // overflow
        v.store(array, Vc::Aligned);
        for (size_t i = 0; i < Vec::Size; ++i) {
            COMPARE(array[i], v[i]);
        }
    }
}

TEST_TYPES(Vec, unalignedStore, AllVectors)
{
    typedef typename Vec::EntryType T;
    enum {
        Count = 256 * 1024 / sizeof(T)
    };

    Memory<Vec, Count> array;
    // do the memset to make sure the array doesn't have the old data from a previous call which
    // would mask a real problem
    std::memset(array, 0xff, Count * sizeof(T));
    T xValue = 1;
    const Vec x(xValue);
    for (size_t i = 1; i < Count - Vec::Size + 1; i += Vec::Size) {
        x.store(&array[i], Unaligned);
    }

    for (size_t i = 1; i < Count - Vec::Size + 1; ++i) {
        COMPARE(array[i], xValue);
    }
}

TEST_TYPES(Vec, streamingAndAlignedStore, AllVectors)
{
    typedef typename Vec::EntryType T;
    enum {
        Count = 256 * 1024 / sizeof(T)
    };

    Memory<Vec, Count> array;
    // do the memset to make sure the array doesn't have the old data from a previous call which
    // would mask a real problem
    std::memset(array, 0xff, Count * sizeof(T));
    T xValue = 1;
    const Vec x(xValue);
    for (int i = 0; i < Count; i += Vec::Size) {
        x.store(&array[i], Streaming | Aligned);
    }

    for (int i = 0; i < Count; ++i) {
        COMPARE(array[i], xValue);
    }
}

TEST_TYPES(Vec, streamingAndUnalignedStore, AllVectors)
{
    typedef typename Vec::EntryType T;
    enum {
        Count = 256 * 1024 / sizeof(T)
    };

    Memory<Vec, Count> array;
    // do the memset to make sure the array doesn't have the old data from a previous call which
    // would mask a real problem
    std::memset(array, 0xff, Count * sizeof(T));
    T xValue = 1;
    const Vec x(xValue);
    for (size_t i = 1; i < Count - Vec::Size + 1; i += Vec::Size) {
        x.store(&array[i], Streaming | Unaligned);
    }

    for (size_t i = 1; i < Count - Vec::Size + 1; ++i) {
        COMPARE(array[i], xValue);
    }
}

TEST_TYPES(Vec, maskedStore, AllVectors)
{
    if ((Vec::size() & 1) == 1) {
        // only works with an even number of vector entries
        // a) masked store with 01010101 will use only 0 for size 1
        // b) the random masks will not be random, because mean == value
        return;
    }
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask M;
    M mask;
    {
        typedef typename Vec::IndexType I;
        const I tmp(IndexesFromZero);
        const typename I::Mask k = (tmp & I(One)) > 0;
        mask = simd_cast<M>(k);
    }

    const int count = 256 * 1024 / sizeof(T);
    Vc::Memory<Vec> array(count);
    array.setZero();
    const T nullValue = 0;
    const T setValue = 170;
    const Vec x(setValue);
    for (int i = 0; i < count; i += Vec::Size) {
        x.store(&array[i], mask, Vc::Aligned);
    }

    for (int i = 1; i < count; i += 2) {
        COMPARE(array[i], setValue) << ", i: " << i << ", count: " << count << ", mask: " << mask << ", array:\n" << array;
    }
    for (int i = 0; i < count; i += 2) {
        COMPARE(array[i], nullValue) << ", i: " << i << ", count: " << count << ", mask: " << mask << ", array:\n" << array;
    }

    for (int offset = 0; offset < count - int(Vec::Size); ++offset) {
        auto mem = &array[offset];
        Vec(0).store(mem, Vc::Unaligned);

        constexpr std::ptrdiff_t alignment = sizeof(T) * Vec::Size;
        constexpr std::ptrdiff_t alignmentMask = ~(alignment - 1);

        const auto loAddress = reinterpret_cast<T *>(
            (reinterpret_cast<char *>(mem) - static_cast<const char *>(0)) & alignmentMask);
        const auto offset2 = mem - loAddress;

        const Vec rand = Vec::Random();
        const auto mean = (rand / T(Vec::Size)).sum();
        mask = rand < mean;
        rand.store(mem, mask, Vc::Unaligned);
        for (std::size_t i = 0; i < Vec::Size; ++i) {
            COMPARE(mem[i], mask[i] ? rand[i] : T(0))
                << ", i: " << i << ", mask: " << mask << "\nrand: " << rand << "\nmean: " << mean
                << ", offset: " << offset << ", offset2: " << offset2 << ", mem: " << mem
                << ", loAddress: " << loAddress;
        }
    }
}
