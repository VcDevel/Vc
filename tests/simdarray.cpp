/*  This file is part of the Vc library. {{{
Copyright © 2013-2015 Matthias Kretz <kretz@kde.org>
All rights reserved.

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
#include "../common/simdarray.h"

using namespace Vc;

#define SIMD_ARRAY_LIST                                                                            \
    (SIMD_ARRAYS(32),                                                                              \
     SIMD_ARRAYS(17),                                                                              \
     SIMD_ARRAYS(16),                                                                              \
     SIMD_ARRAYS(9),                                                                               \
     SIMD_ARRAYS(8),                                                                               \
     SIMD_ARRAYS(5),                                                                               \
     SIMD_ARRAYS(4),                                                                               \
     SIMD_ARRAYS(3),                                                                               \
     SIMD_ARRAYS(2),                                                                               \
     SIMD_ARRAYS(1))

template <typename T, size_t N> constexpr size_t captureN(const SimdArray<T, N> &)
{
    return N;
}

TEST_TYPES(V, createArray, SIMD_ARRAY_LIST)
{
    typedef typename V::VectorEntryType VT;
    V array;

    COMPARE(array.size(), captureN(V()));
    VERIFY(sizeof(array) >= array.size() * sizeof(VT));
    VERIFY(sizeof(array) <= 2 * array.size() * sizeof(VT));
}

template <typename T, typename U> constexpr T bound(T x, U max)
{
    return x > max ? max : x;
}

TEST_TYPES(V, checkArrayAlignment, SIMD_ARRAY_LIST)
{
    using T = typename V::value_type;
    using M = typename V::mask_type;

    COMPARE(alignof(V), bound(sizeof(V), 128u));  // sizeof must be at least as large as alignof to
                                     // ensure proper padding in arrays. alignof is
                                     // supposed to be at least as large as the actual
                                     // data size requirements
    COMPARE(alignof(M), bound(sizeof(M), 128u));
    VERIFY(alignof(V) >= bound(V::Size * sizeof(T), 128u));
    if (V::Size > 1) {
        using V2 = SimdArray<T, Vc::Common::left_size(V::Size)>;
        using M2 = typename V2::mask_type;
        VERIFY(alignof(V) >= bound(2 * alignof(V2), 128u));
        VERIFY(alignof(M) >= bound(2 * alignof(M2), 128u));
    }
}

TEST_TYPES(V, broadcast, SIMD_ARRAY_LIST)
{
    typedef typename V::EntryType T;
    V array = 0;
    array = 1;
    V v0(T(1));
    V v1 = T(2);
    v0 = 2;
    v1 = 3;
    COMPARE(V(), V(0));
}

TEST_TYPES(V, broadcast_equal, SIMD_ARRAY_LIST)
{
    V a = 0;
    V b = 0;
    COMPARE(a, b);
    a = 1;
    b = 1;
    COMPARE(a, b);
}

TEST_TYPES(V, broadcast_not_equal, SIMD_ARRAY_LIST)
{
    V a = 0;
    V b = 1;
    VERIFY(all_of(a != b));
    VERIFY(all_of(a < b));
    VERIFY(all_of(a <= b));
    VERIFY(none_of(a > b));
    VERIFY(none_of(a >= b));
    a = 1;
    VERIFY(all_of(a <= b));
    VERIFY(all_of(a >= b));
}

TEST_TYPES(V, arithmetics, SIMD_ARRAY_LIST)
{
    V a = 0;
    V b = 1;
    V c = 10;
    COMPARE(a + b, b);
    COMPARE(c + b, V(11));
    COMPARE(c - b, V(9));
    COMPARE(b - a, b);
    COMPARE(b * a, a);
    COMPARE(b * b, b);
    COMPARE(c * b, c);
    COMPARE(c * c, V(100));
    COMPARE(c / c, b);
    COMPARE(a / c, a);
    COMPARE(c / b, c);
}

TEST_TYPES(V, indexesFromZero, SIMD_ARRAY_LIST)
{
    typedef typename V::EntryType T;
    V a(Vc::IndexesFromZero);
#if defined Vc_IMPL_MIC && Vc_VERSION_NUMBER < Vc_VERSION_CHECK(1, 99, 0)
// see https://github.com/VcDevel/Vc/issues/47:
// The subscript access to a[i] with i >= V::N0 might access Scalar::Vector<(u)short>
// storing 2-Byte data, whereas the MIC::Vector<(u)short> stores 4-Byte data.
// The plan is to have it fixed in Vc 2.0.
    if (sizeof(T) < 4) {
        alignas(V::MemoryAlignment) T mem[V::size()];
        a.store(mem, Vc::Aligned);
        for (std::size_t i = 0; i < a.size(); ++i) {
            COMPARE(mem[i], T(i));
        }
        return;
    }
#endif
    for (std::size_t i = 0; i < a.size(); ++i) {
        COMPARE(a[i], T(i));
    }
}

TEST_TYPES(V, load, SIMD_ARRAY_LIST)
{
    typedef typename V::EntryType T;
    alignas(static_cast<size_t>(V::MemoryAlignment)) T data[V::Size + 2];
    {
        int n = 0;
        for (auto &entry : data) {
            entry = T(n++);
        }
    }

    V a{ &data[0] };
    V b(Vc::IndexesFromZero);
    COMPARE(a, b);

    b.load(&data[0]);
    COMPARE(a, b);

    a.load(&data[1], Vc::Unaligned);
    COMPARE(a, b + 1);

    b = decltype(b)(&data[2], Vc::Unaligned | Vc::Streaming);
    COMPARE(a, b - 1);
}

TEST_TYPES(A,
           load_converting,
           (SimdArray<float, 32>,
            SimdArray<float, 17>,
            SimdArray<float, 16>,
            SimdArray<float, 8>,
            SimdArray<float, 4>,
            SimdArray<float, 3>,
            SimdArray<float, 2>,
            SimdArray<float, 1>))
{
    Vc::Memory<double_v, 34> data;
    for (size_t i = 0; i < data.entriesCount(); ++i) {
        data[i] = double(i);
    }

    A a{ &data[0] };
    A b(Vc::IndexesFromZero);
    COMPARE(a, b);

    b.load(&data[1], Vc::Unaligned);
    COMPARE(a + 1, b);

    a = A(&data[2], Vc::Unaligned | Vc::Streaming);
    COMPARE(a, b + 1);
}

TEST_TYPES(V, store, SIMD_ARRAY_LIST)
{
    using T = typename V::EntryType;
    alignas(static_cast<size_t>(V::MemoryAlignment)) T data[34] = {};

    V a(Vc::IndexesFromZero);
    a.store(&data[0], Vc::Aligned);
    for (size_t i = 0; i < V::Size; ++i) COMPARE(data[i], T(i));
    a.store(&data[1], Vc::Unaligned);
    for (size_t i = 0; i < V::Size; ++i) COMPARE(data[i + 1], T(i));
    a.store(&data[0], Vc::Aligned | Vc::Streaming);
    for (size_t i = 0; i < V::Size; ++i) COMPARE(data[i], T(i));
    a.store(&data[1], Vc::Unaligned | Vc::Streaming);
    for (size_t i = 0; i < V::Size; ++i) COMPARE(data[i + 1], T(i));
}
