/*  This file is part of the Vc library. {{{
Copyright © 2009-2014 Matthias Kretz <kretz@kde.org>
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
#include <iostream>
#include <limits>
#include <Vc/limits>
#include <common/const.h>
#include <common/macros.h>

using namespace Vc;

#define ALL_TYPES (ALL_VECTORS)
#if 0
#define ALL_TYPES                                                                                  \
    (SIMD_ARRAYS(32),                                                                              \
     SIMD_ARRAYS(31),                                                                              \
     SIMD_ARRAYS(16),                                                                              \
     SIMD_ARRAYS(8),                                                                               \
     SIMD_ARRAYS(4),                                                                               \
     SIMD_ARRAYS(2),                                                                               \
     SIMD_ARRAYS(1),                                                                               \
     ALL_VECTORS)
#endif

std::default_random_engine randomEngine;

TEST_TYPES(Vec, testZero, ALL_TYPES)
{
    Vec a(Zero), b(Zero);
    COMPARE(a, b);
    Vec c, d(1);
    c.setZero();
    COMPARE(a, c);
    d.setZero();
    COMPARE(a, d);
    d = static_cast<typename Vec::EntryType>(0);
    COMPARE(a, d);
    const typename Vec::EntryType zero = 0;
    COMPARE(a, Vec(zero));
    COMPARE(b, Vec(zero));
    COMPARE(c, Vec(zero));
    COMPARE(d, Vec(zero));
}

TEST_TYPES(Vec, testCmp, ALL_TYPES)
{
    typedef typename Vec::EntryType T;
    Vec a(Zero), b(Zero);
    COMPARE(a, b);
    if (!(a != b).isEmpty()) {
        std::cerr << a << " != " << b << ", (a != b) = " << (a != b) << ", (a == b) = " << (a == b) << std::endl;
    }
    VERIFY((a != b).isEmpty());

    Vec c(1);
    VERIFY((a < c).isFull());
    VERIFY((c > a).isFull());
    VERIFY((a <= b).isFull());
    VERIFY((a <= c).isFull());
    VERIFY((b >= a).isFull());
    VERIFY((c >= a).isFull());

    {
        const T max = static_cast<T>(std::numeric_limits<T>::max() * 0.95);
        const T min = 0;
        const T step = max / 200;
        T j = min;
        VERIFY(all_of(Vec(Zero) == Vec(j)));
        VERIFY(none_of(Vec(Zero) < Vec(j)));
        VERIFY(none_of(Vec(Zero) > Vec(j)));
        VERIFY(none_of(Vec(Zero) != Vec(j)));
        j += step;
        for (int i = 0; i < 200; ++i, j += step) {
            if(all_of(Vec(Zero) >= Vec(j))) {
                std::cout << j << " " << Vec(j) << " " << (Vec(Zero) >= Vec(j)) << std::endl;
            }
            VERIFY(all_of(Vec(Zero) < Vec(j))) << (Vec(Zero) < Vec(j)) << ", j = " << j << ", step = " << step;
            VERIFY(all_of(Vec(j) > Vec(Zero)));
            VERIFY(none_of(Vec(Zero) >= Vec(j)));
            VERIFY(none_of(Vec(j) <= Vec(Zero)));
        }
    }
    if (std::numeric_limits<T>::min() <= 0) {
        const T min = static_cast<T>(std::numeric_limits<T>::min() * 0.95);
        if (min == 0) {
            return;
        }
        const T step = min / T(-201);
        T j = min;
        for (int i = 0; i < 200; ++i, j += step) {
            VERIFY(all_of(Vec(j) < Vec(Zero)));
            VERIFY(all_of(Vec(Zero) > Vec(j)));
            VERIFY(none_of(Vec(Zero) <= Vec(j)));
            VERIFY(none_of(Vec(j) >= Vec(Zero)));
        }
    }
}

TEST_TYPES(Vec, testIsMix, ALL_TYPES)
{
    Vec a = Vec::IndexesFromZero();
    Vec b(Zero);
    Vec c(One);
    if (Vec::Size > 1) {
        VERIFY((a == b).isMix()) << "a == b: " << (a == b);
        VERIFY((a != b).isMix());
        VERIFY((a == c).isMix());
        VERIFY((a != c).isMix());
        VERIFY(!(a == a).isMix());
        VERIFY(!(a != a).isMix());
    } else { // masks of size 1 can never be a mix of 0 and 1
        VERIFY(!(a == b).isMix());
        VERIFY(!(a != b).isMix());
        VERIFY(!(a == c).isMix());
        VERIFY(!(a != c).isMix());
        VERIFY(!(a == a).isMix());
        VERIFY(!(a != a).isMix());
    }
}

TEST_TYPES(Vec, testAdd, ALL_TYPES)
{
    Vec a(Zero), b(Zero);
    COMPARE(a, b);

    a += 1;
    Vec c(1);
    COMPARE(a, c);

    COMPARE(a, b + 1);
    COMPARE(a, b + c);

    for (int repetition = 0; repetition < 10000; ++repetition) {
        const Vec x = Vec::Random();
        const Vec y = Vec::Random();
        Vec reference;
        for (size_t i = 0; i < Vec::Size; ++i) {
            reference[i] = x[i] + y[i];
        }
        COMPARE(x + y, reference) << '\n' << x << " + " << y;
    }
}

TEST_TYPES(Vec, testSub, ALL_TYPES)
{
    Vec a(2), b(2);
    COMPARE(a, b);

    a -= 1;
    Vec c(1);
    COMPARE(a, c);

    COMPARE(a, b - 1);
    COMPARE(a, b - c);

    for (int repetition = 0; repetition < 10000; ++repetition) {
        const Vec x = Vec::Random();
        const Vec y = Vec::Random();
        Vec reference;
        for (size_t i = 0; i < Vec::Size; ++i) {
            reference[i] = x[i] - y[i];
        }
        COMPARE(x - y, reference) << '\n' << x << " - " << y;
    }
}

TEST_TYPES(V, testMul, ALL_TYPES)
{
    for (int i = 0; i < 10000; ++i) {
        V a = V::Random();
        V b = V::Random();
        V reference = a;
        for (size_t j = 0; j < V::Size; ++j) {
            // this could overflow - but at least the compiler can't know about it so it doesn't
            // matter that it's undefined behavior in C++. The only thing that matters is what the
            // hardware does...
            reference[j] *= b[j];
        }
        COMPARE(a * b, reference) << a << " * " << b;
    }
}

TEST_TYPES(Vec, testMulAdd, ALL_TYPES)
{
    typedef typename Vec::EntryType T;
    static_assert(std::is_arithmetic<T>::value, "The EntryType is not a builtin arithmetic type");
    for (std::size_t rep = 0; rep < 10000 / Vec::Size; ++rep) {
        Vec a = Vec::Random();
        if (std::is_floating_point<T>::value) {
            a *= std::sqrt(std::numeric_limits<T>::max());
        } else if (std::is_signed<T>::value) {
            a /= std::sqrt(std::numeric_limits<T>::max());
        }
        using ReferenceVector = decltype(a * a);
        ReferenceVector reference = a;
        reference = reference.apply([](T x) { return x * x + 1; });
        FUZZY_COMPARE(a * a + T(1), reference) << "a: " << a;
    }
}

TEST_TYPES(Vec, testMulSub, ALL_TYPES)
{
    typedef typename Vec::EntryType T;
    const unsigned int minI = sizeof(T) < 4 ? -0xb4 : 0;
    const unsigned int maxI = sizeof(T) < 4 ? 0xb4 : 0xffff;
    for (unsigned int i = minI; i < maxI; ++i) {
        const T j = static_cast<T>(i);
        const Vec test(j);

        FUZZY_COMPARE(test * test - test, Vec(j * j - j));
    }
}

TEST_TYPES(Vec, testDiv, ALL_TYPES)
{
    for (int repetition = 0; repetition < 10000; ++repetition) {
        const Vec a = Vec::Random();
        const Vec b = Vec::Random();
        if (none_of(b == Vec::Zero())) {
            Vec reference;
            for (size_t i = 0; i < Vec::Size; ++i) {
                reference[i] = a[i] / b[i];
            }
            COMPARE(a / b, reference) << '\n' << a << " / " << b;
        }
    }
    typedef typename Vec::EntryType T;
#if defined(VC_ICC) && !defined(__x86_64__) && VC_ICC <= 20131008
    // http://software.intel.com/en-us/forums/topic/488995
    if (isEqualType<short, T>()) {
        EXPECT_FAILURE();
    }
#endif

    const T stepsize = std::max(T(1), T(std::numeric_limits<T>::max() / 1024));
    for (T divisor = 1; divisor < 5; ++divisor) {
        for (T scalar = std::numeric_limits<T>::min(); scalar < std::numeric_limits<T>::max() - stepsize + 1; scalar += stepsize) {
            Vec vector(scalar);
            Vec reference(scalar / divisor);

            COMPARE(vector / divisor, reference) << '\n' << vector << " / " << divisor
                << ", reference: " << scalar << " / " << divisor << " = " << scalar / divisor;
            vector /= divisor;
            COMPARE(vector, reference);
        }
    }
}

TEST_TYPES(V,
           testModulo,
           (simdarray<unsigned int, 31>,
            simdarray<unsigned short, 31>,
            simdarray<unsigned int, 32>,
            simdarray<unsigned short, 32>,
            simdarray<int, 31>,
            simdarray<short, 31>,
            simdarray<int, 32>,
            simdarray<short, 32>,
            int_v,
            ushort_v,
            uint_v,
            short_v))
{
    for (int repetition = 0; repetition < 1000; ++repetition) {
        V x = V::Random();
        V y = (V::Random() & 2047) - 1023;
        y(y == 0) = -1024;
        const V z = x % y;

        V reference;
        for (size_t i = 0; i < V::Size; ++i) {
            reference[i] = x[i] % y[i];
        }

        COMPARE(z, reference) << ", x: " << x << ", y: " << y;

        COMPARE(V::Zero() % y, V::Zero());
        COMPARE(y % y, V::Zero());
    }
}

TEST_TYPES(Vec, testAnd, (int_v, ushort_v, uint_v, short_v))
{
    Vec a(0x7fff);
    Vec b(0xf);
    COMPARE((a & 0xf), b);
    Vec c(IndexesFromZero);
    COMPARE(c, (c & 0xf));
    const typename Vec::EntryType zero = 0;
    COMPARE((c & 0x7ff0), Vec(zero));
}

TEST_TYPES(Vec, testShift, (int_v, ushort_v, uint_v, short_v))
{
    typedef typename Vec::EntryType T;
    const T step = std::max<T>(1, std::numeric_limits<T>::max() / 1000);
    enum {
        NShifts = sizeof(T) * 8
    };
    for (Vec x = std::numeric_limits<Vec>::min() + Vec::IndexesFromZero();
         all_of(x < std::numeric_limits<Vec>::max() - step);
         x += step) {
        for (size_t shift = 0; shift < NShifts; ++shift) {
            const Vec rightShift = x >> shift;
            const Vec leftShift  = x << shift;
            for (size_t k = 0; k < Vec::Size; ++k) {
                COMPARE(rightShift[k], T(x[k] >> shift)) << ", x[k] = " << x[k] << ", shift = " << shift;
                COMPARE(leftShift [k], T(x[k] << shift)) << ", x[k] = " << x[k] << ", shift = " << shift;
            }
        }
    }

    Vec a(1);
    Vec b(2);

    // left shifts
    COMPARE((a << 1), b);
    COMPARE((a << 2), (a << 2));
    COMPARE((a << 2), (b << 1));

    Vec shifts(IndexesFromZero);
    a <<= shifts;
    for (T i = 0, x = 1; i < T(Vc::is_signed<Vec>::value ? Vec::Size - 1 : Vec::Size); ++i, x <<= 1) {
        COMPARE(a[i], x);
    }

    // right shifts
    a = Vec(4);
    COMPARE((a >> 1), b);
    COMPARE((a >> 2), (a >> 2));
    COMPARE((a >> 2), (b >> 1));

    a = Vec(16);
    a >>= shifts;
    for (T i = 0, x = 16; i < T(Vec::Size); ++i, x >>= 1) {
        COMPARE(a[i], x);
    }
}

TEST_TYPES(Vec, testOnesComplement, (int_v, ushort_v, uint_v, short_v))
{
    Vec a(One);
    Vec b = ~a;
    COMPARE(~a, b);
    COMPARE(~b, a);
    COMPARE(~(a + b), Vec(Zero));
}

TEST_TYPES(V, logicalNegation, ALL_TYPES)
{
    V a = V::Random();
    COMPARE(!a, a == 0) << "a = " << a;
    COMPARE(!V::Zero(), V() == V()) << "a = " << a;
}

template<typename T> struct NegateRangeHelper
{
    typedef int Iterator;
    static const Iterator Start;
    static const Iterator End;
};
template<> struct NegateRangeHelper<unsigned int> {
    typedef unsigned int Iterator;
    static const Iterator Start;
    static const Iterator End;
};
template<> const int NegateRangeHelper<float>::Start = -0xffffff;
template<> const int NegateRangeHelper<float>::End   =  0xffffff - 133;
template<> const int NegateRangeHelper<double>::Start = -0xffffff;
template<> const int NegateRangeHelper<double>::End   =  0xffffff - 133;
template<> const int NegateRangeHelper<int>::Start = -0x7fffffff;
template<> const int NegateRangeHelper<int>::End   = 0x7fffffff - 0xee;
const unsigned int NegateRangeHelper<unsigned int>::Start = 0;
const unsigned int NegateRangeHelper<unsigned int>::End = 0xffffffff - 0xee;
template<> const int NegateRangeHelper<short>::Start = -0x7fff;
template<> const int NegateRangeHelper<short>::End = 0x7fff - 0xee;
template<> const int NegateRangeHelper<unsigned short>::Start = 0;
template<> const int NegateRangeHelper<unsigned short>::End = 0xffff - 0xee;

TEST_TYPES(Vec, testNegate, ALL_TYPES)
{
    typedef typename Vec::EntryType T;

    for (int i = 0; i < 1000; ++i) {
        const Vec x = Vec::Random();
        const auto negX = -x;
        for (size_t j = 0; j < x.Size; ++j) {
            const T reference = -x[j];
            COMPARE(negX[j], reference) << "x = " << x;
        }
    }

    typedef NegateRangeHelper<T> Range;
    for (typename Range::Iterator i = Range::Start; i < Range::End; i += 0xef) {
        T i2 = static_cast<T>(i);
        Vec a(i2);

        COMPARE(-a, Vec(-i2)) << " i2: " << i2;
    }
}

TEST_TYPES(Vec, testMin, ALL_TYPES)
{
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;

    Vec v = Vec::IndexesFromZero();

    COMPARE(v.min(), static_cast<T>(0));
    COMPARE((T(Vec::Size) - v).min(), static_cast<T>(1));

    const size_t max = (size_t(1) << Vec::Size) - 1;
    std::uniform_int_distribution<size_t> dist(0, max);
    for (int rep = 0; rep < 100000; ++rep) {
        const size_t j = dist(randomEngine);
        Mask m = UnitTest::allMasks<Vec>(j);
        if (any_of(m)) {
            COMPARE(v.min(m), static_cast<T>(m.firstOne())) << m << v;
        }
    }
}

TEST_TYPES(Vec, testMax, ALL_TYPES)
{
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;

    Vec v = Vec::IndexesFromZero();

    COMPARE(v.max(), static_cast<T>(Vec::Size - 1));
    v = T(Vec::Size) - v;
    COMPARE(v.max(), static_cast<T>(Vec::Size));

    const size_t max = (size_t(1) << Vec::Size) - 1;
    std::uniform_int_distribution<size_t> dist(0, max);
    for (int rep = 0; rep < 100000; ++rep) {
        const size_t j = dist(randomEngine);
        Mask m = UnitTest::allMasks<Vec>(j);
        if (any_of(m)) {
            COMPARE(v.max(m), static_cast<T>(Vec::Size - m.firstOne())) << m << v;
        }
    }
}

TEST_TYPES(Vec, testProduct, ALL_TYPES)
{
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;

    for (int i = 0; i < 10; ++i) {
        T x = static_cast<T>(i);
        Vec v(x);
        T x2 = x;
        if (std::numeric_limits<T>::is_exact) {
            for (int k = Vec::Size; k > 1; --k) {
                x2 *= x;
            }
            COMPARE(v.product(), x2) << v;
        } else {
            x2 = std::round(std::pow(x, static_cast<int>(Vec::Size)));
            FUZZY_COMPARE(v.product(), x2) << v;
        }

        const size_t max = (size_t(1) << Vec::Size) - 1;
        std::uniform_int_distribution<size_t> dist(0, max);
        for (int rep = 0; rep < 10000; ++rep) {
            const size_t j = dist(randomEngine);
            Mask m = UnitTest::allMasks<Vec>(j);
            if (any_of(m)) {
                if (std::numeric_limits<T>::is_exact) {
                    x2 = x;
                    for (int k = m.count(); k > 1; --k) {
                        x2 *= x;
                    }
                    COMPARE(v.product(m), x2) << v << ".product(" << m << ')';
                } else {
                    x2 = std::round(std::pow(x, static_cast<int>(m.count())));
                    FUZZY_COMPARE(v.product(m), x2) << v << ".product(" << m << ')';
                }
            } else {
                COMPARE(v.product(m), T(1)) << v << ".product(" << m << ')';
            }
        }
    }
}

TEST_TYPES(Vec, testSum, ALL_TYPES)
{
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;

    for (int i = 0; i < 10; ++i) {
        T x = static_cast<T>(i);
        Vec v(x);
        COMPARE(v.sum(), T(x * Vec::Size)) << v;

        const size_t max = (size_t(1) << Vec::Size) - 1;
        std::uniform_int_distribution<size_t> dist(0, max);
        for (int rep = 0; rep < 10000; ++rep) {
            const size_t j = dist(randomEngine);
            Mask m = UnitTest::allMasks<Vec>(j);
            if (any_of(m)) {
                COMPARE(v.sum(m), static_cast<T>(x * m.count())) << m << v;
            } else {
                COMPARE(v.sum(m), T(0)) << m << v;
            }
        }
    }
}

TEST_TYPES(V, testPartialSum, ALL_TYPES)
{
    V reference = V::IndexesFromZero() + 1;
    COMPARE(V(1).partialSum(), reference);
    /* disabled until correct masking is implemented

    typedef typename V::IndexType I;
    reference = simd_cast<V>(I(2) << I::IndexesFromZero());
    COMPARE(V(2).partialSum([](const V &a, const V &b) { return a * b; }), reference);
    */
}

template <typename V, typename T> void testFmaDispatch(T)
{
    for (int i = 0; i < 1000; ++i) {
        V a = V::Random();
        const V b = V::Random();
        const V c = V::Random();
        const V reference = a * b + c;
        a.fusedMultiplyAdd(b, c);
        COMPARE(a, reference) << ", a = " << a << ", b = " << b << ", c = " << c;
    }
}

template <typename V> void testFmaDispatch(float)
{
    using Vc::Internal::floatConstant;
    V b = floatConstant<1, 0x000001, 0>();
    V c = floatConstant<1, 0x000000, -24>();
    V a = b;
    /*a *= b;
    a += c;
    COMPARE(a, V(floatConstant<1, 0x000002, 0>()));
    a = b;*/
    a.fusedMultiplyAdd(b, c);
    COMPARE(a, V(floatConstant<1, 0x000003, 0>()));

    a = floatConstant<1, 0x000002, 0>();
    b = floatConstant<1, 0x000002, 0>();
    c = floatConstant<-1, 0x000000, 0>();
    /*a *= b;
    a += c;
    COMPARE(a, V(floatConstant<1, 0x000000, -21>()));
    a = b;*/
    a.fusedMultiplyAdd(b, c); // 1 + 2^-21 + 2^-44 - 1 == (1 + 2^-20)*2^-18
    COMPARE(a, V(floatConstant<1, 0x000001, -21>()));
}

template <typename V> void testFmaDispatch(double)
{
    using Vc::Internal::doubleConstant;
    V b = doubleConstant<1, 0x0000000000001, 0>();
    V c = doubleConstant<1, 0x0000000000000, -53>();
    V a = b;
    a.fusedMultiplyAdd(b, c);
    COMPARE(a, V(doubleConstant<1, 0x0000000000003, 0>()));

    a = doubleConstant<1, 0x0000000000002, 0>();
    b = doubleConstant<1, 0x0000000000002, 0>();
    c = doubleConstant<-1, 0x0000000000000, 0>();
    a.fusedMultiplyAdd(b, c); // 1 + 2^-50 + 2^-102 - 1
    COMPARE(a, V(doubleConstant<1, 0x0000000000001, -50>()));
}

TEST_TYPES(V, testFma, ALL_TYPES)
{
    using T = typename V::EntryType;
    testFmaDispatch<V>(T());
}
