/*  This file is part of the Vc library. {{{
Copyright © 2009-2015 Matthias Kretz <kretz@kde.org>
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
#ifndef Vc_IMPL_Scalar // avoid -Wtautological-compare warnings because of V::Size == 1
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

        int count = 0;
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
        COMPARE(count, 0);
    }
}

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

        int callCount() const { return i; }
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
        COMPARE(callTester.callCount(), static_cast<int>(V::Size));
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

TEST_TYPES(V, rotated, (ALL_VECTORS, SIMD_ARRAYS(16), SIMD_ARRAYS(15), SIMD_ARRAYS(11),
                        SIMD_ARRAYS(9), SIMD_ARRAYS(8), SIMD_ARRAYS(7), SIMD_ARRAYS(3)))
{
    constexpr int Size = V::Size;
    for (int shift = 2 * Size; shift >= -2 * Size; --shift) {
        //std::cout << "amount = " << shift % Size << std::endl;
        const V reference = V::Random();
        const V test = reference.rotated(shift);
        for (int i = 0; i < Size; ++i) {
            int refShift = (i + shift) % int(V::size());
            if (refShift < 0) {
                refShift += V::size();
            }
            COMPARE(test[i], reference[refShift])
                << "shift: " << shift << ", refShift: " << refShift << ", i: " << i
                << ", test: " << test << ", reference: " << reference;
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

TEST_TYPES(V, shiftedIn, (ALL_VECTORS, SIMD_ARRAYS(1), SIMD_ARRAYS(16), SIMD_ARRAYS(17)))
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

TEST(testMallocAlignment)
{
    int_v *a = Vc::malloc<int_v, Vc::AlignOnVector>(10);

    std::uintptr_t mask = int_v::MemoryAlignment - 1;
    for (int i = 0; i < 10; ++i) {
        VERIFY((reinterpret_cast<std::uintptr_t>(&a[i]) & mask) == 0);
    }
    const char *data = reinterpret_cast<const char *>(&a[0]);
    for (int i = 0; i < 10; ++i) {
        VERIFY(&data[i * int_v::Size * sizeof(int_v::EntryType)] == reinterpret_cast<const char *>(&a[i]));
    }

    a = Vc::malloc<int_v, Vc::AlignOnCacheline>(10);
    mask = CpuId::cacheLineSize() - 1;
    COMPARE((reinterpret_cast<std::uintptr_t>(&a[0]) & mask), 0ul);

    // I don't know how to properly check page alignment. So we check for 4 KiB alignment as this is
    // the minimum page size on x86
    a = Vc::malloc<int_v, Vc::AlignOnPage>(10);
    mask = 4096 - 1;
    COMPARE((reinterpret_cast<std::uintptr_t>(&a[0]) & mask), 0ul);
}

template <typename A, typename B, typename C,
          typename = decltype(Vc::iif(std::declval<A>(), std::declval<B>(),
                                      std::declval<C>()))>
inline void sfinaeIifIsNotCallable(A &&, B &&, C &&, int)
{
    FAIL() << "iif(" << UnitTest::typeToString<A>() << UnitTest::typeToString<B>()
           << UnitTest::typeToString<C>() << ") appears to be callable.";
}

template <typename A, typename B, typename C>
inline void sfinaeIifIsNotCallable(A &&, B &&, C &&, ...)
{
    // passes
}

TEST_TYPES(V, testIif,
           (ALL_VECTORS, SIMD_ARRAYS(31), Vc::SimdArray<float, 8, Vc::Scalar::float_v>))
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
        sfinaeIifIsNotCallable(
            x > y, int(), float(),
            int());  // mismatching true/false value types should never work
        sfinaeIifIsNotCallable(
            x > y, int(), int(),
            int());  // iif(mask, scalar, scalar) should not appear usable
    }
}

TEST(testIifBuiltin)
{
    COMPARE(Vc::iif(true, 1, 2), true ? 1 : 2);
    COMPARE(Vc::iif(false, 1, 2), false ? 1 : 2);
    sfinaeIifIsNotCallable(bool(), int(), float(), int());
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
        int count = 0;
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
