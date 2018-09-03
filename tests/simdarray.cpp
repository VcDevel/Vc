/*  This file is part of the Vc library. {{{
Copyright © 2013-2015 Matthias Kretz <kretz@kde.org>

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
#include <Vc/common/simdarray.h>

using namespace Vc;

template <typename T, size_t N> constexpr size_t captureN(const SimdArray<T, N> &)
{
    return N;
}

TEST_TYPES(V, createArray, SimdArrayList)
{
    typedef typename V::VectorEntryType VT;
    V array;

    COMPARE(array.size(), captureN(V()));
    VERIFY(sizeof(array) >= array.size() * sizeof(VT));
    VERIFY(sizeof(array) <= 2 * array.size() * sizeof(VT));
}

TEST_TYPES(V, checkArrayAlignment, SimdArrayList)
{
    using T = typename V::value_type;
    using M = typename V::mask_type;

    auto &&bound_to_max_alignment = [](std::size_t n) {
#ifdef Vc_GCC
#ifdef __AVX__
        return n > 256u ? 256u : n;
#else
        return n > 128u ? 128u : n;
#endif
#elif defined Vc_MSVC
        return n > 32u ? 32u : n;
#else
        return n;
#endif
    };

    COMPARE(alignof(V),
            bound_to_max_alignment(
                sizeof(V)));  // sizeof must be at least as large as alignof to
                              // ensure proper padding in arrays. alignof is
                              // supposed to be at least as large as the actual
                              // data size requirements
    COMPARE(alignof(M), bound_to_max_alignment(sizeof(M)));
    VERIFY(alignof(V) >= bound_to_max_alignment(V::Size * sizeof(T)));
    if (V::Size > 1) {
        using V2 = SimdArray<T, Vc::Common::left_size<V::Size>()>;
        using M2 = typename V2::mask_type;
        VERIFY(alignof(V) >= bound_to_max_alignment(2 * alignof(V2)));
        VERIFY(alignof(M) >= bound_to_max_alignment(2 * alignof(M2)));
    }
}

TEST_TYPES(V, broadcast, SimdArrayList)
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

TEST_TYPES(V, broadcast_equal, SimdArrayList)
{
    V a = 0;
    V b = 0;
    COMPARE(a, b);
    a = 1;
    b = 1;
    COMPARE(a, b);
}

TEST_TYPES(V, broadcast_not_equal, SimdArrayList)
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

TEST_TYPES(V, arithmetics, SimdArrayList)
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

TEST_TYPES(V, generator, SimdArrayList) {
    using T = typename V::value_type;
    const V seq([](int n) { return n + 1; });
    T mem[V::size()];
    seq.store(mem, Vc::Unaligned);
    for (std::size_t i = 0; i < V::size(); ++i) {
        COMPARE(mem[i], T(i + 1));
    }
}

TEST_TYPES(V, indexesFromZero, SimdArrayList)
{
    typedef typename V::EntryType T;
    V a(Vc::IndexesFromZero);
    for (std::size_t i = 0; i < a.size(); ++i) {
        COMPARE(a[i], T(i));
    }
}

TEST_TYPES(V, load, SimdArrayList)
{
    typedef typename V::EntryType T;
    alignas(static_cast<size_t>(V::MemoryAlignment)) T data[V::Size + 2];
    {
        int n = 0;
        for (auto &entry : data) {
            entry = T(n++);
        }
    }

    V a(&data[0], Vc::Aligned);
    V b(Vc::IndexesFromZero);
    COMPARE(a, b);

    a = V(data, Vc::Aligned);  // this is different to the constructor call above since
                               // the type is T[N] (vs. T *)
    COMPARE(a, b);

    b.load(&data[0], Vc::Aligned);
    COMPARE(a, b);

    a.load(&data[1], Vc::Unaligned);
    COMPARE(a, b + 1);

    b = decltype(b)(&data[2], Vc::Unaligned | Vc::Streaming);
    COMPARE(a, b - 1);
}

TEST_TYPES(A,
           load_converting,
           Vc::fixed_size_simd<float, 32>,
           Vc::fixed_size_simd<float, 17>,
           Vc::fixed_size_simd<float, 16>,
           Vc::fixed_size_simd<float, 8>,
           Vc::fixed_size_simd<float, 4>,
           Vc::fixed_size_simd<float, 3>,
           Vc::fixed_size_simd<float, 2>,
           Vc::fixed_size_simd<float, 1>)
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

TEST_TYPES(V, store, SimdArrayList)
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
