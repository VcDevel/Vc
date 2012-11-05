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
#include <iostream>
#include "vectormemoryhelper.h"
#include "../cpuid.h"

using namespace Vc;

template<typename Vec> void testSort()
{
    typedef typename Vec::EntryType EntryType;
    typedef typename Vec::IndexType IndexType;

    typename Vec::Memory _a;
    const IndexType _ref(IndexesFromZero);
    Vec ref(_ref);
    Vec a;
    int maxPerm = 1;
    for (int x = Vec::Size; x > 0; --x) {
        maxPerm *= x;
    }
    for (int perm = 0; perm < maxPerm; ++perm) {
        int rest = perm;
        for (int i = 0; i < Vec::Size; ++i) {
            _a[i] = 0;
            for (int j = 0; j < i; ++j) {
                if (_a[i] == _a[j]) {
                    ++_a[i];
                    j = -1;
                }
            }
            _a[i] += rest % (Vec::Size - i);
            rest /= (Vec::Size - i);
            for (int j = 0; j < i; ++j) {
                if (_a[i] == _a[j]) {
                    ++_a[i];
                    j = -1;
                }
            }
        }
        a.load(_a);
        //std::cout << a << a.sorted() << std::endl;
        COMPARE(ref, a.sorted()) << ", a: " << a;
    }
}

template<typename T, typename Mem> struct Foo
{
    Foo() : i(0) {}
    void reset() { i = 0; }
    void operator()(T v) { d[i++] = v; }
    Mem d;
    int i;
};

template<typename V> void testCall()
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;
    typedef typename V::Mask M;
    typedef typename I::Mask MI;
    const I _indexes(IndexesFromZero);
    const MI _odd = (_indexes & I(One)) > 0;
    const M odd(_odd);
    V a(_indexes);
    Foo<T, typename V::Memory> f;
    a.callWithValuesSorted(f);
    V b(f.d);
    COMPARE(b, a);

    f.reset();
    a(odd) -= 1;
    a.callWithValuesSorted(f);
    V c(f.d);
    for (int i = 0; i < V::Size / 2; ++i) {
        COMPARE(a[i * 2], c[i]);
    }
    for (int i = V::Size / 2; i < V::Size; ++i) {
        COMPARE(b[i], c[i]);
    }
}

template<typename V> void testForeachBit()
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;
    typedef typename V::Mask M;
    typedef typename I::Mask MI;
    const I indexes(IndexesFromZero);
    for (int i = 0; i <= V::Size; ++i) {
        const M mask(indexes < i);
        int ref = 0;
        foreach_bit(int j, mask) {
            ref += (1 << j);
        }
        COMPARE(ref, (1 << i) - 1);
    }
}

void testMallocAlignment()
{
    int_v *a = Vc::malloc<int_v, Vc::AlignOnVector>(10);

    unsigned long mask = VectorAlignment - 1;
    for (int i = 0; i < 10; ++i) {
        VERIFY((reinterpret_cast<unsigned long>(&a[i]) & mask) == 0);
    }
    const char *data = reinterpret_cast<const char *>(&a[0]);
    for (int i = 0; i < 10; ++i) {
        VERIFY(&data[i * int_v::Size * sizeof(int_v::EntryType)] == reinterpret_cast<const char *>(&a[i]));
    }

    a = Vc::malloc<int_v, Vc::AlignOnCacheline>(10);
    mask = CpuId::cacheLineSize() - 1;
    COMPARE((reinterpret_cast<unsigned long>(&a[0]) & mask), 0ul);

    // I don't know how to properly check page alignment. So we check for 4 KiB alignment as this is
    // the minimum page size on x86
    a = Vc::malloc<int_v, Vc::AlignOnPage>(10);
    mask = 4096 - 1;
    COMPARE((reinterpret_cast<unsigned long>(&a[0]) & mask), 0ul);
}

int main()
{
    runTest(testCall<int_v>);
    runTest(testCall<uint_v>);
    runTest(testCall<short_v>);
    runTest(testCall<ushort_v>);
    runTest(testCall<float_v>);
    runTest(testCall<sfloat_v>);
    runTest(testCall<double_v>);

    runTest(testForeachBit<int_v>);
    runTest(testForeachBit<uint_v>);
    runTest(testForeachBit<short_v>);
    runTest(testForeachBit<ushort_v>);
    runTest(testForeachBit<float_v>);
    runTest(testForeachBit<sfloat_v>);
    runTest(testForeachBit<double_v>);

    runTest(testSort<int_v>);
    runTest(testSort<uint_v>);
    runTest(testSort<float_v>);
    runTest(testSort<double_v>);
    runTest(testSort<sfloat_v>);
    runTest(testSort<short_v>);
    runTest(testSort<ushort_v>);

    runTest(testMallocAlignment);

    return 0;
}
