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

#include <Vc/Vc>
#include "unittest.h"
#include <iostream>
#include "vectormemoryhelper.h"

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

template<typename V> void copySign()
{
    typedef typename V::EntryType T;
    V v(One);
    V positive(One);
    V negative = -positive;
    COMPARE(v, v.copySign(positive));
    COMPARE(-v, v.copySign(negative));
}

#include <strings.h>

template<typename V> void Random()
{
    typedef typename V::EntryType T;
    enum {
        NBits = 3,
        NBins = 1 << NBits,                        // short int
        TotalBits = sizeof(T) * 8,                 //    16  32
        RightShift = TotalBits - NBits,            //    13  29
        NHistograms = TotalBits - NBits + 1,       //    14  30
        LeftShift = (RightShift + 1) / NHistograms,//     1   1
        Mean = 135791,
        MinGood = Mean - Mean/10,
        MaxGood = Mean + Mean/10
    };
    const V mask((1 << NBits) - 1);
    int histogram[NHistograms][NBins];
    bzero(&histogram[0][0], sizeof(histogram));
    for (size_t i = 0; i < NBins * Mean / V::Size; ++i) {
        const V rand = V::Random();
        for (size_t hist = 0; hist < NHistograms; ++hist) {
            const V bin = ((rand << (hist * LeftShift)) >> RightShift) & mask;
            for (size_t k = 0; k < V::Size; ++k) {
                ++histogram[hist][bin[k]];
            }
        }
    }
//#define PRINT_RANDOM_HISTOGRAM
#ifdef PRINT_RANDOM_HISTOGRAM
    for (size_t hist = 0; hist < NHistograms; ++hist) {
        std::cout << "histogram[" << std::setw(2) << hist << "]: ";
        for (size_t bin = 0; bin < NBins; ++bin) {
            std::cout << std::setw(3) << (histogram[hist][bin] - Mean) * 1000 / Mean << "|";
        }
        std::cout << std::endl;
    }
#endif
    for (size_t hist = 0; hist < NHistograms; ++hist) {
        for (size_t bin = 0; bin < NBins; ++bin) {
            VERIFY(histogram[hist][bin] > MinGood)
                << " bin = " << bin << " is " << histogram[0][bin];
            VERIFY(histogram[hist][bin] < MaxGood)
                << " bin = " << bin << " is " << histogram[0][bin];
        }
    }
}

template<typename V, typename I> void FloatRandom()
{
    typedef typename V::EntryType T;
    enum {
        NBins = 64,
        NHistograms = 1,
        Mean = 135791,
        MinGood = Mean - Mean/10,
        MaxGood = Mean + Mean/10
    };
    int histogram[NHistograms][NBins];
    bzero(&histogram[0][0], sizeof(histogram));
    for (size_t i = 0; i < NBins * Mean / V::Size; ++i) {
        const V rand = V::Random();
        const I bin = static_cast<I>(rand * T(NBins));
        for (size_t k = 0; k < V::Size; ++k) {
            ++histogram[0][bin[k]];
        }
    }
#ifdef PRINT_RANDOM_HISTOGRAM
    for (size_t hist = 0; hist < NHistograms; ++hist) {
        std::cout << "histogram[" << std::setw(2) << hist << "]: ";
        for (size_t bin = 0; bin < NBins; ++bin) {
            std::cout << std::setw(3) << (histogram[hist][bin] - Mean) * 1000 / Mean << "|";
        }
        std::cout << std::endl;
    }
#endif
    for (size_t hist = 0; hist < NHistograms; ++hist) {
        for (size_t bin = 0; bin < NBins; ++bin) {
            VERIFY(histogram[hist][bin] > MinGood)
                << " bin = " << bin << " is " << histogram[0][bin];
            VERIFY(histogram[hist][bin] < MaxGood)
                << " bin = " << bin << " is " << histogram[0][bin];
        }
    }
}

template<> void Random<float_v>() { FloatRandom<float_v, int_v>(); }
template<> void Random<double_v>() { FloatRandom<double_v, int_v>(); }
#ifdef VC_IMPL_SSE
template<> void Random<sfloat_v>() { FloatRandom<sfloat_v, short_v>(); }
#endif

template<typename T> T add2(T x) { return x + T(2); }

template<typename V>
void apply()
{
    typedef typename V::EntryType T;

    const V two(T(2));
    for (int i = 0; i < 1000; ++i) {
        const V rand = V::Random();
        COMPARE(rand.apply(add2<T>), rand + two);
        for_all_masks(V, mask) {
            V copy1 = rand;
            V copy2 = rand;
            copy1(mask) += two;
            COMPARE(copy2(mask).apply(add2<T>), copy1) << mask;
            COMPARE(rand.apply(add2<T>, mask), copy1) << mask;
        }
    }
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

    runTest(copySign<float_v>);
    runTest(copySign<sfloat_v>);
    runTest(copySign<double_v>);

    testAllTypes(Random);

    testAllTypes(apply);

    return 0;
}
