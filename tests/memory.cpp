/*  This file is part of the Vc library.

    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

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

#include <Vc/Vc>
#include "unittest.h"

using namespace Vc;

template<typename V, unsigned int Size> struct TestEntries {
    static inline void run()
    {
        TestEntries<V, Size/2>::run();
        TestEntries<V, Size>::test();
        TestEntries<V, Size - 1>::test();
    }
    static void test();
};
template<typename V> struct TestEntries<V, 0> { static void run() {} static void test() {} };

template<typename V, unsigned int Size> struct TestVectors {
    static inline void run()
    {
        TestVectors<V, Size/2>::run();
        TestVectors<V, Size>::test();
        TestVectors<V, Size - 1>::test();
    }
    static void test();
};
template<typename V> struct TestVectors<V, 0> { static void run() {} static void test() {} };

template<typename V, unsigned int Size> struct TestVectorReorganization {
    static inline void run()
    {
        TestVectorReorganization<V, Size/2>::run();
        TestVectorReorganization<V, Size>::test();
        TestVectorReorganization<V, Size - 1>::test();
    }
    static void test();
};
template<typename V> struct TestVectorReorganization<V, 0> { static void run() {} static void test() {} };

template<typename V, unsigned int Size> void TestEntries<V, Size>::test()
{
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
}

template<typename V, unsigned int Size> void TestVectors<V, Size>::test()
{
    typedef typename V::EntryType T;
    const V startX(V::IndexType::IndexesFromZero() + Size);
    Memory<V, Size> m;
    const Memory<V, Size> &m2 = m;
    Memory<V> m3(Size);
    V x = startX;
    for (unsigned int i = 0; i < m.vectorsCount(); ++i, x += V::Size) {
        m.vector(i) = x;
        m3.vector(i) = x;
    }
    x = startX;
    unsigned int i;
    for (i = 0; i < m.vectorsCount() - 1; ++i) {
        COMPARE(V(m.vector(i)), x);
        COMPARE(V(m2.vector(i)), x);
        COMPARE(V(m3.vector(i)), x);
        for (int shift = 0; shift < V::Size; ++shift, ++x) {
            COMPARE(V(m.vector(i, shift)), x);
            COMPARE(V(m2.vector(i, shift)), x);
            COMPARE(V(m3.vector(i, shift)), x);
        }
    }
    COMPARE(V(m.vector(i)), x);
    COMPARE(V(m2.vector(i)), x);
    COMPARE(V(m3.vector(i)), x);
}

template<typename V, unsigned int Size> void TestVectorReorganization<V, Size>::test()
{
    typedef typename V::EntryType T;
    typename V::Memory init;
    for (unsigned int i = 0; i < V::Size; ++i) {
        init[i] = i;
    }
    V x(init);
    Memory<V, Size> m;
    Memory<V> m3(Size);
    for (unsigned int i = 0; i < m.vectorsCount(); ++i) {
        m.vector(i) = x;
        m3.vector(i) = x;
        x += V::Size;
    }
    ///////////////////////////////////////////////////////////////////////////
    x = V(init);
    for (unsigned int i = 0; i < m.vectorsCount(); ++i) {
        COMPARE(V(m.vector(i)), x);
        COMPARE(V(m3.vector(i)), x);
        x += V::Size;
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
}

template<typename V> void testEntries()
{
    TestEntries<V, 128>::run();
}

template<typename V> void testVectors()
{
    TestVectors<V, 128>::run();
}

template<typename V> void testVectorReorganization()
{
    TestVectorReorganization<V, 128>::run();
}

template<typename V> void memoryOperators()
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

template<typename V> void testCCtor()
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

int main()
{
    testAllTypes(testEntries);
    testAllTypes(testVectors);
    testAllTypes(testVectorReorganization);
    testAllTypes(memoryOperators);
    testAllTypes(testCCtor);

    return 0;
}
