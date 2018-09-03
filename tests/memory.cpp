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

template<typename V, unsigned int Size, template<typename V2, unsigned int Size2> class TestClass> struct TestWrapper
{
    static inline void run()
    {
        TestWrapper<V, Size/2, TestClass>::run();
        TestClass<V, Size>::test();
        TestClass<V, Size - 1>::test();
    }
};
template<typename V, template<typename V2, unsigned int Size> class TestClass> struct TestWrapper<V, 1, TestClass> {
    static inline void run() {}
};

template<typename V, unsigned int Size> struct TestEntries { static void test() {
    typedef typename V::EntryType T;
    const T x = Size;
    Memory<V, Size> m;
    const Memory<V, Size> &m2 = m;
    Memory<V> m3(Size);
    for (unsigned int i = 0; i < Size; ++i) {
        m[i] = x;
        m3[i] = x;
    }
    for (unsigned int i = 0; i < Size; ++i) {
        COMPARE(m[i], x);
        COMPARE(m2[i], x);
        COMPARE(m3[i], x);
    }
    for (unsigned int i = 0; i < Size; ++i) {
        COMPARE(m.entries()[i], x);
        COMPARE(m2.entries()[i], x);
        COMPARE(m3.entries()[i], x);
    }
    const T *ptr = m2;
    for (unsigned int i = 0; i < Size; ++i) {
        COMPARE(ptr[i], x);
    }
    ptr = m3;
    for (unsigned int i = 0; i < Size; ++i) {
        COMPARE(ptr[i], x);
    }
}};

template<typename V, unsigned int Size> struct TestEntries2D { static void test() {
    typedef typename V::EntryType T;
    const T x = Size;
    Memory<V, Size, Size> m;
    const Memory<V, Size, Size> &m2 = m;

    for (size_t i = 0; i < Size; ++i) {
        for (size_t j = 0; j < Size; ++j) {
            m[i][j] = x + i + j;
        }
    }
    for (size_t i = 0; i < Size; ++i) {
        for (size_t j = 0; j < Size; ++j) {
            COMPARE(m[i][j], T(x + i + j)) << i << ", j = " << j;
            COMPARE(m2[i][j], T(x + i + j)) << i << ", j = " << j;
        }
    }
    for (size_t i = 0; i < Size; ++i) {
        for (size_t j = 0; j < Size; ++j) {
            COMPARE(m[i].entries()[j], T(x + i + j));
            COMPARE(m2[i].entries()[j], T(x + i + j));
        }
    }
    for (size_t i = 0; i < Size; ++i) {
        const T *ptr = m2[i];
        for (size_t j = 0; j < Size; ++j) {
            COMPARE(ptr[j], T(x + i + j));
        }
    }
}};

template<typename V, unsigned int Size> struct TestVectors { static void test()
{
    const V startX = V([](int n) { return n + Size; });
    Memory<V, Size> m;
    const Memory<V, Size> &m2 = m;
    Memory<V> m3(Size);
    V x = startX;
    for (unsigned int i = 0; i < m.vectorsCount(); ++i, x += int(V::size())) {
        m.vector(i) = x;
        m3.vector(i) = x;
    }
    x = startX;
    unsigned int i;
    for (i = 0; i + 1 < m.vectorsCount(); ++i) {
        COMPARE(V(m.vector(i)), x);
        COMPARE(V(m2.vector(i)), x);
        COMPARE(V(m3.vector(i)), x);
        for (size_t shift = 0; shift < V::Size; ++shift, ++x) {
            COMPARE(V(m.vector(i, shift)), x);
            COMPARE(V(m2.vector(i, shift)), x);
            COMPARE(V(m3.vector(i, shift)), x);
        }
    }
    COMPARE(V(m.vector(i)), x);
    COMPARE(V(m2.vector(i)), x);
    COMPARE(V(m3.vector(i)), x);
}};

template<typename V, unsigned int Size> struct TestVectors2D { static void test()
{
    const V startX = V([](int n) { return n + Size; });
    Memory<V, Size, Size> m;
    const Memory<V, Size, Size> &m2 = m;
    V x = startX;
    for (size_t i = 0; i < m.rowsCount(); ++i, x += int(V::size())) {
        auto &mrow = m[i];
        for (size_t j = 0; j < mrow.vectorsCount(); ++j, x += int(V::size())) {
            mrow.vector(j) = x;
        }
    }
    x = startX;
    for (size_t i = 0; i < m.rowsCount(); ++i, x += int(V::size())) {
        auto &mrow = m[i];
        const auto &m2row = m2[i];
        size_t j;
        for (j = 0; j < mrow.vectorsCount() - 1; ++j) {
            COMPARE(V(mrow.vector(j)), x);
            COMPARE(V(m2row.vector(j)), x);
            for (size_t shift = 0; shift < V::Size; ++shift, ++x) {
                COMPARE(V(mrow.vector(j, shift)), x);
                COMPARE(V(m2row.vector(j, shift)), x);
            }
        }
        COMPARE(V(mrow.vector(j)), x) << i << " " << j;
        COMPARE(V(m2row.vector(j)), x);
        x += int(V::size());
    }
}};

template<typename V, unsigned int Size> struct TestVectorReorganization { static void test()
{
    Vc::Memory<V, V::Size> init;
    for (unsigned int i = 0; i < V::Size; ++i) {
        init[i] = i;
    }
    V x(init);
    Memory<V, Size> m;
    Memory<V> m3(Size);
    for (unsigned int i = 0; i < m.vectorsCount(); ++i) {
        m.vector(i) = x;
        m3.vector(i) = x;
        x += int(V::size());
    }
    ///////////////////////////////////////////////////////////////////////////
    x = V(init);
    for (unsigned int i = 0; i < m.vectorsCount(); ++i) {
        COMPARE(V(m.vector(i)), x);
        COMPARE(V(m3.vector(i)), x);
        x += int(V::size());
    }
    ///////////////////////////////////////////////////////////////////////////
    x = V(init);
    unsigned int indexes[Size];
    for (unsigned int i = 0; i < Size; ++i) {
        indexes[i] = i;
    }
    for (unsigned int i = 0; i + V::Size < Size; ++i) {
        COMPARE(m.gather(&indexes[i]), x);
        COMPARE(m3.gather(&indexes[i]), x);
        x += 1;
    }
    ///////////////////////////////////////////////////////////////////////////
    for (unsigned int i = 0; i < V::Size; ++i) {
        init[i] = i * 2;
    }
    x = V(init);
    for (unsigned int i = 0; i < Size; ++i) {
        indexes[i] = (i * 2) % Size;
    }
    for (unsigned int i = 0; i + V::Size < Size; ++i) {
        COMPARE(m.gather(&indexes[i]), x);
        COMPARE(m3.gather(&indexes[i]), x);
        x += 2;
        x(x >= Size) -= Size;
    }
}};

TEST_TYPES(V, testEntries, AllVectors) { TestWrapper<V, 128, TestEntries>::run(); }

TEST_TYPES(V, testEntries2D, AllVectors) { TestWrapper<V, 32, TestEntries2D>::run(); }

TEST_TYPES(V, testVectors, AllVectors) { TestWrapper<V, 128, TestVectors>::run(); }

TEST_TYPES(V, testVectors2D, AllVectors) { TestWrapper<V, 32, TestVectors2D>::run(); }

TEST_TYPES(V, testVectorReorganization, AllVectors)
{
    TestWrapper<V, 128, TestVectorReorganization>::run();
}

TEST_TYPES(V, memoryOperators, AllVectors)
{
    Memory<V, 129> m1, m2;
    m1.setZero();
    m2.setZero();
    VERIFY(m1 == m2);
    VERIFY(!(m1 != m2));
    VERIFY(!(m1 < m2));
    VERIFY(!(m1 > m2));
    m1 += m2;
    VERIFY(m1 == m2);
    VERIFY(m1 <= m2);
    VERIFY(m1 >= m2);
    m1 += 1;
    VERIFY(m1 != m2);
    VERIFY(m1 > m2);
    VERIFY(m1 >= m2);
    VERIFY(m2 < m1);
    VERIFY(m2 <= m1);
    VERIFY(!(m1 == m2));
    VERIFY(!(m1 <= m2));
    VERIFY(!(m2 >= m1));
    m2 += m1;
    VERIFY(m1 == m2);
    m2 *= 2;
    m1 += 1;
    VERIFY(m1 == m2);
    m2 /= 2;
    m1 -= 1;
    VERIFY(m1 == m2);
    m1 *= m2;
    VERIFY(m1 == m2);
    m1 /= m2;
    VERIFY(m1 == m2);
    m1 -= m2;
    m2 -= m2;
    VERIFY(m1 == m2);
}

TEST_TYPES(V, testCCtor, AllVectors)
{
    Memory<V> m1(5);
    for (size_t i = 0; i < m1.entriesCount(); ++i) {
        m1[i] = i;
    }
    Memory<V> m2(m1);
    for (size_t i = 0; i < m1.entriesCount(); ++i) {
        m1[i] += 1;
    }
    for (size_t i = 0; i < m1.entriesCount(); ++i) {
        COMPARE(m1[i], m2[i] + 1);
    }
}

void *hackToStoreToStack = 0;

TEST_TYPES(V, paddingMustBeZero, AllVectors)
{
    typedef typename V::EntryType T;
    { // poison the stack
        V v = V::Random();
        hackToStoreToStack = &v;
    }
    Memory<V, 1> m;
    m[0] = T(0);
    V x = m.vector(0);
    COMPARE(x, V(0));
}

#ifndef Vc_ICC
TEST_TYPES(V, initializerList, AllVectors)
{
    typedef typename V::EntryType T;
    Memory<V, 3> m = { T(1), T(2), T(3) };
    for (int i = 0; i < 3; ++i) {
        COMPARE(m[i], T(i + 1));
    }
}
#endif

TEST_TYPES(V, testCopyAssignment, AllVectors)
{
    using T = typename V::EntryType;
    Memory<V, 99> m1;
    m1.setZero();

    Memory<V, 99> m2(m1);
    for (size_t i = 0; i < m2.entriesCount(); ++i) {
        COMPARE(m2[i], T(0));
        m2[i] += 1;
    }
    m1 = m2;
    for (size_t i = 0; i < m2.entriesCount(); ++i) {
        COMPARE(m1[i], T(1));
    }
}
