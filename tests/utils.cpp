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
#include <iostream>
#include "vectormemoryhelper.h"
#include <Vc/cpuid.h>
#include <Vc/iterators>

using namespace Vc;

TEST_TYPES(V, reversed, (ALL_VECTORS, SIMD_ARRAYS(2), SIMD_ARRAYS(3), SIMD_ARRAYS(15)))
{
    const V x = V::IndexesFromZero() + 1;
    const V reference = V::generate([](int i) { return V::Size - i; });
    COMPARE(x.reversed(), reference);
}

TEST_TYPES(Vec, testSort, (ALL_VECTORS))
{
    Vec ref(Vc::IndexesFromZero);
    Vec a;
//X     for (int i = 0; i < Vec::Size; ++i) {
//X         a[i] = Vec::Size - i - 1;
//X     }
//X     COMPARE(ref, a.sorted()) << ", a: " << a;

    int maxPerm = 1;
    for (int x = Vec::Size; x > 0 && maxPerm < 400000; --x) {
        maxPerm *= x;
    }
    for (int perm = 0; perm < maxPerm; ++perm) {
        int rest = perm;
        for (size_t i = 0; i < Vec::Size; ++i) {
            a[i] = 0;
            for (size_t j = 0; j < i; ++j) {
                if (a[i] == a[j]) {
                    ++a[i];
                    j = -1;
                }
            }
            a[i] += rest % (Vec::Size - i);
            rest /= (Vec::Size - i);
            for (size_t j = 0; j < i; ++j) {
                if (a[i] == a[j]) {
                    ++a[i];
                    j = -1;
                }
            }
        }
        //std::cout << a << a.sorted() << std::endl;
        COMPARE(ref, a.sorted()) << ", a: " << a;
    }

    for (int repetition = 0; repetition < 1000; ++repetition) {
        Vec test = Vec::Random();
        Vc::Memory<Vec, Vec::Size> reference;
        reference.vector(0) = test;
        std::sort(&reference[0], &reference[Vec::Size]);
        ref = reference.vector(0);
        COMPARE(ref, test.sorted());
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

TEST_TYPES(V, testCall, (ALL_VECTORS))
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;
    typedef typename V::Mask M;
    const I _indexes(Vc::IndexesFromZero);
    const M odd = Vc::simd_cast<M>((_indexes & I(One)) > 0);
    V a(Vc::IndexesFromZero);
    Foo<T, Vc::Memory<V, V::Size>> f;
    a.callWithValuesSorted(f);
    V b(f.d);
    COMPARE(b, a);

    f.reset();
    a(odd) -= 1;
    a.callWithValuesSorted(f);
    V c(f.d);
#ifndef VC_IMPL_Scalar // avoid -Wtautological-compare warnings because of V::Size == 1
    for (size_t i = 0; i < V::Size / 2; ++i) {
        COMPARE(a[i * 2], c[i]);
    }
#endif
    for (size_t i = V::Size / 2; i < V::Size; ++i) {
        COMPARE(b[i], c[i]);
    }
}

TEST_TYPES(V, testForeachBit, (ALL_VECTORS))
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;
    const I indexes(IndexesFromZero);
    for_all_masks(V, mask) {
        V tmp = V::Zero();
        for (int j : where(mask)) {
            tmp[j] = T(1);
        }
        COMPARE(tmp == V::One(), mask);

        unsigned int count = 0;
        for (int j : where(mask)) {
            ++count;
            if (j >= 0) {
                continue;
            }
        }
        COMPARE(count, mask.count());

        count = 0;
        for (int j : where(mask)) {
            if (j >= 0) {
                break;
            }
            ++count;
        }
        COMPARE(count, 0U);
    }
}

TEST_TYPES(V, copySign, (Vc::float_v, Vc::double_v))
{
    V v(One);
    V positive(One);
    V negative = -positive;
    COMPARE(v, v.copySign(positive));
    COMPARE(-v, v.copySign(negative));
}

#ifdef _WIN32
void bzero(void *p, size_t n) { memset(p, 0, n); }
#else
#include <strings.h>
#endif

TEST_TYPES(V, testRandom, (ALL_VECTORS))
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

template<> void testRandom<float_v>::operator()() { FloatRandom<float_v, int_v>(); }
template<> void testRandom<double_v>::operator()() { FloatRandom<double_v, int_v>(); }

template<typename T> T add2(T x) { return x + T(2); }

template<typename T, typename V>
class CallTester
{
    public:
        CallTester() : v(Vc::Zero), i(0) {}

        void operator()(T x) {
            v[i] = x;
            ++i;
        }

        void reset() { v.setZero(); i = 0; }

        unsigned int callCount() const { return i; }
        V callValues() const { return v; }

    private:
        V v;
        unsigned int i;
};

TEST_TYPES(V, applyAndCall, (ALL_VECTORS))
{
    typedef typename V::EntryType T;

    const V two(T(2));
    for (int i = 0; i < 10; ++i) {
        const V rand = V::Random();
        auto add2Reference = static_cast<T (*)(T)>(add2);
        COMPARE(rand.apply(add2Reference), rand + two);
        COMPARE(rand.apply([](T x) { return x + T(2); }), rand + two);

        CallTester<T, V> callTester;
        rand.call(callTester);
        COMPARE(callTester.callCount(), static_cast<unsigned int>(V::Size));
        COMPARE(callTester.callValues(), rand);

        for_all_masks(V, mask) {
            V copy1 = rand;
            V copy2 = rand;
            copy1(mask) += two;

            COMPARE(copy2(mask).apply(add2Reference), copy1) << mask;
            COMPARE(rand.apply(add2Reference, mask), copy1) << mask;
            COMPARE(copy2(mask).apply([](T x) { return x + T(2); }), copy1) << mask;
            COMPARE(rand.apply([](T x) { return x + T(2); }, mask), copy1) << mask;

            callTester.reset();
            copy2(mask).call(callTester);
            COMPARE(callTester.callCount(), mask.count());

            callTester.reset();
            rand.call(callTester, mask);
            COMPARE(callTester.callCount(), mask.count());
        }
    }
}

template<typename T, int value> T returnConstant() { return T(value); }
template<typename T, int value> T returnConstantOffset(int i) { return T(value) + T(i); }
template<typename T, int value> T returnConstantOffset2(unsigned short i) { return T(value) + T(i); }

TEST_TYPES(V, fill, (ALL_VECTORS))
{
    typedef typename V::EntryType T;
    V test = V::Random();
    test.fill(returnConstant<T, 2>);
    COMPARE(test, V(T(2)));

    test = V::Random();
    test.fill(returnConstantOffset<T, 0>);
    COMPARE(test, static_cast<V>(V::IndexesFromZero()));

    test = V::Random();
    test.fill(returnConstantOffset2<T, 0>);
    COMPARE(test, static_cast<V>(V::IndexesFromZero()));
}

TEST_TYPES(V, shifted, (ALL_VECTORS))
{
    typedef typename V::EntryType T;
    constexpr int Size = V::Size;
    for (int shift = -2 * Size; shift <= 2 * Size; ++shift) {
        const V reference = V::Random();
        const V test = reference.shifted(shift);
        for (int i = 0; i < Size; ++i) {
            if (i + shift >= 0 && i + shift < Size) {
                COMPARE(test[i], reference[i + shift]) << "shift: " << shift << ", i: " << i << ", test: " << test << ", reference: " << reference;
            } else {
                COMPARE(test[i], T(0)) << "shift: " << shift << ", i: " << i << ", test: " << test << ", reference: " << reference;
            }
        }
    }
}

TEST_TYPES(V, rotated, (ALL_VECTORS))
{
    constexpr int Size = V::Size;
    for (int shift = -2 * Size; shift <= 2 * Size; ++shift) {
        //std::cout << "amount = " << shift % Size << std::endl;
        const V reference = V::Random();
        const V test = reference.rotated(shift);
        for (int i = 0; i < Size; ++i) {
            unsigned int refShift = i + shift;
            COMPARE(test[i], reference[refShift % V::Size]) << "shift: " << shift << ", i: " << i << ", test: " << test << ", reference: " << reference;
        }
    }
}

template <typename V> V shiftReference(const V &data, int shift)
{
    constexpr int Size = V::Size;
    using T = typename V::value_type;
    return V::generate([&](int i) -> T {
        if (shift < 0) {
            i += shift;
            if (i >= 0) {
                return data[i];
            }
            i += Size;
            if (i >= 0) {
                return data[i] + 1;
            }
        } else {
            i += shift;
            if (i < Size) {
                return data[i];
            }
            i -= Size;
            if (i < Size) {
                return data[i] + 1;
            }
        }
        return 0;
    });
}

template <typename V>
void shiftedInConstant(const V &, std::integral_constant<int, 2 * V::Size>)
{
}

template <typename V, typename Shift> void shiftedInConstant(const V &data, Shift)
{
    const V reference = shiftReference(data, Shift::value);
    const V test = data.shifted(Shift::value, data + V::One());
    COMPARE(test, reference) << "shift = " << Shift::value;
    if ((Shift::value + 1) % V::Size != 0) {
        shiftedInConstant(
            data, std::integral_constant<
                      int, ((Shift::value + 1) % V::Size == 0 ? 2 * int(V::Size)
                                                              : Shift::value + 1)>());
    }
}

TEST_TYPES(V, shiftedIn, (ALL_VECTORS, SIMD_ARRAYS(1), SIMD_ARRAYS(31), SIMD_ARRAYS(32), SIMD_ARRAYS(33)))
{
    constexpr int Size = V::Size;
    const V data = V::Random();
    for (int shift = -2 * Size + 1; shift <= 2 * Size -1; ++shift) {
        const V reference = shiftReference(data, shift);
        const V test = data.shifted(shift, data + V::One());
        COMPARE(test, reference) << "\nshift = " << shift << "\ndata = " << data;
    }
    shiftedInConstant(V::Random(), std::integral_constant<int, -2 * Size + 1>());
    shiftedInConstant(V::Random(), std::integral_constant<int, -Size>());
    shiftedInConstant(V::Random(), std::integral_constant<int, 0>());
    shiftedInConstant(V::Random(), std::integral_constant<int, Size>());
}

#if 0
TEST(testMallocAlignment)
{
    int_v *a = Vc::malloc<int_v, Vc::AlignOnVector>(10);

    size_t mask = VectorAlignment - 1;
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

TEST_TYPES(V, testIif, (ALL_VECTORS))
{
    typedef typename V::EntryType T;
    const T one = T(1);
    const T two = T(2);

    for (int i = 0; i < 10000; ++i) {
        const V x = V::Random();
        const V y = V::Random();
        V reference = y;
        V reference2 = two;
        for (size_t j = 0; j < V::Size; ++j) {
            if (x[j] > y[j]) {
                reference[j] = x[j];
                reference2[j] = one;
            }
        }
        COMPARE(iif (x > y, x, y), reference);
        COMPARE(iif (x > y, V(one), V(two)), reference2);
    }
}

TEST_TYPES(V, rangeFor, (ALL_VECTORS))
{
    typedef typename V::EntryType T;
    typedef typename V::Mask M;

    {
        V x = V::Zero();
        for (auto i : x) {
            COMPARE(i, T(0));
        }
        int n = 0;
        for (auto &i : x) {
            i = T(++n);
        }
        n = 0;
        for (auto i : x) {
            COMPARE(i, T(++n));
            i = T(0);
        }
        n = 0;
        for (auto i : static_cast<const V &>(x)) {
            COMPARE(i, T(++n));
        }
    }

    {
        M m(Vc::One);
        for (auto i : m) {
            VERIFY(i);
            i = false;
            VERIFY(!i);
        }
        for (auto i : m) {
            VERIFY(i);
        }
        for (auto i : static_cast<const M &>(m)) {
            VERIFY(i);
        }
    }

    for_all_masks(V, mask) {
        unsigned int count = 0;
        V test = V::Zero();
        for (size_t i : where(mask)) {
            VERIFY(i < V::Size);
            test[i] = T(1);
            ++count;
        }
        COMPARE(test == V::One(), mask);
        COMPARE(count, mask.count());
    }
}

TEST_TYPES(V, testNonMemberInterleave, (ALL_VECTORS, SIMD_ARRAYS(1), SIMD_ARRAYS(2), SIMD_ARRAYS(3), SIMD_ARRAYS(9), SIMD_ARRAYS(8)))
{
    for (int repeat = 0; repeat < 10; ++repeat) {
        std::array<V, 2> testValues = {V::IndexesFromZero(), V::IndexesFromZero() + V::Size};
        std::array<V, 2> references;
        for (size_t i = 0; i < 2 * V::Size; ++i) {
            references[i / V::Size][i % V::Size] = testValues[i & 1][i >> 1];
        }
        std::tie(testValues[0], testValues[1]) = interleave(testValues[0], testValues[1]);
        COMPARE(testValues[0], references[0]);
        COMPARE(testValues[1], references[1]);
    }
}
