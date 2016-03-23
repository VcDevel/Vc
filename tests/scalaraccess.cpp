/*  This file is part of the Vc library. {{{
Copyright © 2010-2016 Matthias Kretz <kretz@kde.org>
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

#include "generators.h"
#include "unittest.h"

using namespace Vc;

TEST_TYPES(V, reads, (ALL_VECTORS, SIMD_ARRAY_LIST))
{
    typedef typename V::EntryType T;

    V a = V::Zero();
    const T zero = 0;
    for (size_t i = 0; i < V::Size; ++i) {
        const T x = a[i];
        COMPARE(x, zero);
    }
    a = V::IndexesFromZero();
    for (size_t i = 0; i < V::Size; ++i) {
        const T x = a[i];
        const T y = i;
        COMPARE(x, y);
    }
    // The non-const operator[] returns a smart reference that mimics an lvalue reference.
    // But since it's not an actual lvalue reference the proxy is supposed to live as
    // short as possible. To achieve this all move & copy operations must be disabled.
    VERIFY(!std::is_move_constructible<decltype(a[0])>::value);
    VERIFY(!std::is_copy_constructible<decltype(a[0])>::value);
    VERIFY(!std::is_move_assignable<decltype(a[0])>::value);
    VERIFY(!std::is_copy_assignable<decltype(a[0])>::value);
    VERIFY(!std::is_reference<decltype(a[0])>::value);
    COMPARE(typeid(decltype(a[0])), typeid(typename V::reference));

    // the operator[] const overload returns an rvalue T and is therefore not constrained
    // in the same way as the non-const operator[]
    const V b = a;
    VERIFY(std::is_move_constructible<decltype(b[0])>::value);
    VERIFY(std::is_copy_constructible<decltype(b[0])>::value);
    VERIFY(std::is_move_assignable<decltype(b[0])>::value);
    VERIFY(std::is_copy_assignable<decltype(b[0])>::value);
    VERIFY(!std::is_reference<decltype(b[0])>::value);
    COMPARE(typeid(decltype(b[0])), typeid(typename V::value_type));
}

template<typename V, size_t Index>
inline void readsConstantIndexTest(Vc_ALIGNED_PARAMETER(V) a, Vc_ALIGNED_PARAMETER(V) b)
{
    typedef typename V::EntryType T;
    {
        const T x = a[Index];
        const T zero = 0;
        COMPARE(x, zero) << Index;
    }{
        const T x = b[Index];
        const T y = Index;
        COMPARE(x, y) << Index;
    }
}

template<typename V, size_t Index>
struct ReadsConstantIndex
{
    ReadsConstantIndex(Vc_ALIGNED_PARAMETER(V) a, Vc_ALIGNED_PARAMETER(V) b)
    {
        readsConstantIndexTest<V, Index>(a, b);
        ReadsConstantIndex<V, Index - 1>(a, b);
    }
};


template<typename V>
struct ReadsConstantIndex<V, 0>
{
    ReadsConstantIndex(Vc_ALIGNED_PARAMETER(V) a, Vc_ALIGNED_PARAMETER(V) b)
    {
        readsConstantIndexTest<V, 0>(a, b);
    }
};

TEST_TYPES(V, readsConstantIndex, (ALL_VECTORS, SIMD_ARRAY_LIST))
{
    V a = V::Zero();
    V b = V::IndexesFromZero();
    ReadsConstantIndex<V, V::Size - 1>(a, b);
}

TEST_TYPES(V, writes, (ALL_VECTORS, SIMD_ARRAY_LIST))
{
    typedef typename V::EntryType T;

    V a = 0;
    std::array<T, V::size()> reference = {0};
    int i = 0;
    iterateNumericRange<T>([&](T x) {
        reference[i] = x;
        a[i] = x;
        COMPARE(a, V(&reference[0]));
        i = (i + 1) % V::size();
    });
}

#define INT_OP(op, name)                                                                  \
    template <typename V> bool test_##name##_assign(V &, float, int) { return false; }   \
    template <typename V, typename = decltype(std::declval<V &>()[0] op## =              \
                                                  typename V::EntryType())>              \
    bool test_##name##_assign(V &a, int x, int y)                                        \
    {                                                                                    \
        using T = typename V::EntryType;                                                 \
        COMPARE(a[0] op## = T(x), T(y));                                                 \
        COMPARE(a[0], T(y));                                                             \
        return true;                                                                     \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON
INT_OP(%, percent);
INT_OP(<<, lshift);
INT_OP(>>, rshift);
INT_OP(|, bor);
INT_OP(&, band);
INT_OP(^, bxor);
#undef INT_OP

TEST_TYPES(V, operators, (ALL_VECTORS, SIMD_ARRAY_LIST))
{
    using T = typename V::EntryType;
    V a = 10;
    COMPARE(a[0]  += 1, T(11)); COMPARE(a[0], T(11));
    COMPARE(a[0]  -= 1, T(10)); COMPARE(a[0], T(10));
    COMPARE(a[0]  *= 2, T(20)); COMPARE(a[0], T(20));
    COMPARE(a[0]  /= 2, T(10)); COMPARE(a[0], T(10));
    COMPARE( --a[0]   , T( 9)); COMPARE(a[0], T( 9));
    COMPARE(   a[0]-- , T( 9)); COMPARE(a[0], T( 8));
    COMPARE( ++a[0]   , T( 9)); COMPARE(a[0], T( 9));
    COMPARE(   a[0]++ , T( 9)); COMPARE(a[0], T(10));
    COMPARE(test_percent_assign<V>(a, 6, 4), std::is_integral<T>::value);
    COMPARE(test_lshift_assign<V>(a, 1, 8), std::is_integral<T>::value);
    COMPARE(test_rshift_assign<V>(a, 2, 2), std::is_integral<T>::value);
    COMPARE(test_bor_assign<V>(a, 9, 11), std::is_integral<T>::value);
    COMPARE(test_band_assign<V>(a, 13, 9), std::is_integral<T>::value);
    COMPARE(test_bxor_assign<V>(a, 1, 8), std::is_integral<T>::value);
}
