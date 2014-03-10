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

#define VC_NEWTEST
#include "unittest.h"
#include <iostream>
#include <limits>
#include <Vc/limits>
#include <common/const.h>
#include <common/macros.h>

using namespace Vc;

#define ALL_TYPES                                                                                  \
    (SIMD_ARRAYS(32),                                                                              \
     SIMD_ARRAYS(31),                                                                              \
     SIMD_ARRAYS(16),                                                                              \
     SIMD_ARRAYS(8),                                                                               \
     SIMD_ARRAYS(4),                                                                               \
     SIMD_ARRAYS(2),                                                                               \
     SIMD_ARRAYS(1),                                                                               \
     ALL_VECTORS)

TEST_BEGIN(Vec, testZero, ALL_TYPES)
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
TEST_END

TEST_ALL_V(Vec, testCmp)
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
        VERIFY(Vec(Zero) == Vec(j));
        VERIFY(!(Vec(Zero) < Vec(j)));
        VERIFY(!(Vec(Zero) > Vec(j)));
        VERIFY(!(Vec(Zero) != Vec(j)));
        j += step;
        for (int i = 0; i < 200; ++i, j += step) {
            if(Vec(Zero) >= Vec(j)) {
                std::cout << j << " " << Vec(j) << " " << (Vec(Zero) >= Vec(j)) << std::endl;
            }
            VERIFY(Vec(Zero) < Vec(j));
            VERIFY(Vec(j) > Vec(Zero));
            VERIFY(!(Vec(Zero) >= Vec(j)));
            VERIFY(!(Vec(j) <= Vec(Zero)));
            VERIFY(!static_cast<bool>(Vec(Zero) >= Vec(j)));
            VERIFY(!static_cast<bool>(Vec(j) <= Vec(Zero)));
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
            VERIFY(Vec(j) < Vec(Zero));
            VERIFY(Vec(Zero) > Vec(j));
            VERIFY(!(Vec(Zero) <= Vec(j)));
            VERIFY(!(Vec(j) >= Vec(Zero)));
        }
    }
}

TEST_ALL_V(Vec, testIsMix)
{
    typedef typename Vec::IndexType IT;
    Vec a(IT::IndexesFromZero());
    Vec b(Zero);
    Vec c(One);
    if (Vec::Size > 1) {
        VERIFY((a == b).isMix());
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

TEST_ALL_V(Vec, testAdd)
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

TEST_ALL_V(Vec, testSub)
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

TEST_ALL_V(V, testMul)
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

template<typename> void testMulAdd();
template<> void testMulAdd<short_v>()
{ // short_v over-/underflow results in undefined behavior
    for (unsigned int i = -0xb4; i < 0xb4; ++i) {
        const short_v i2(i * i + 1);
        short_v a(i);

        COMPARE(a * a + 1, i2);
    }
}

TEST_ALL_V(Vec, testMulAdd)
{
    for (unsigned int i = 0; i < 0xffff; ++i) {
        const Vec i2(i * i + 1);
        Vec a(i);

        FUZZY_COMPARE(a * a + 1, i2);
    }
}

template<typename> void testMulSub();
template<> void testMulSub<short_v>()
{ // short_v over-/underflow results in undefined behavior
    for (unsigned int i = -0xb4; i < 0xb4; ++i) {
        const short j = static_cast<short>(i);
        const short_v test(j);

        COMPARE(test * test - test, short_v(j * j - j));
    }
}

TEST_ALL_V(Vec, testMulSub)
{
    typedef typename Vec::EntryType T;
    for (unsigned int i = 0; i < 0xffff; ++i) {
        const T j = static_cast<T>(i);
        const Vec test(j);

        FUZZY_COMPARE(test * test - test, Vec(j * j - j));
    }
}

TEST_ALL_V(Vec, testDiv)
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

TEST_BEGIN(V, testModulo, (simd_array<int, 32>, simd_array<short, 32>, int_v, ushort_v, uint_v, short_v))
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
TEST_END

TEST_BEGIN(Vec, testAnd, (int_v, ushort_v, uint_v, short_v))
    Vec a(0x7fff);
    Vec b(0xf);
    COMPARE((a & 0xf), b);
    Vec c(IndexesFromZero);
    COMPARE(c, (c & 0xf));
    const typename Vec::EntryType zero = 0;
    COMPARE((c & 0x7ff0), Vec(zero));
TEST_END

TEST_BEGIN(Vec, testShift, (int_v, ushort_v, uint_v, short_v))
    typedef typename Vec::EntryType T;
    const T step = std::max<T>(1, std::numeric_limits<T>::max() / 1000);
    enum {
        NShifts = sizeof(T) * 8
    };
    for (Vec x = std::numeric_limits<Vec>::min() + Vec::IndexesFromZero();
            x <  std::numeric_limits<Vec>::max() - step;
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
TEST_END

TEST_BEGIN(Vec, testOnesComplement, (int_v, ushort_v, uint_v, short_v))
    Vec a(One);
    Vec b = ~a;
    COMPARE(~a, b);
    COMPARE(~b, a);
    COMPARE(~(a + b), Vec(Zero));
TEST_END

TEST_BEGIN(V, logicalNegation, ALL_TYPES)
    V a = V::Random();
    COMPARE(!a, a == 0) << "a = " << a;
    COMPARE(!V::Zero(), V() == V()) << "a = " << a;
TEST_END

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

TEST_BEGIN(Vec, testNegate, ALL_TYPES)
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

        COMPARE(-a, -i2) << " i2: " << i2;
    }
TEST_END

TEST_BEGIN(Vec, testMin, (ALL_VECTORS))
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;
    typedef typename Vec::IndexType I;

    Vec v(I::IndexesFromZero());

    COMPARE(v.min(), static_cast<T>(0));
    COMPARE((T(Vec::Size) - v).min(), static_cast<T>(1));

    int j = 0;
    Mask m;
    do {
        m = allMasks<Vec>(j++);
        if (m.isEmpty()) {
            break;
        }
        COMPARE(v.min(m), static_cast<T>(m.firstOne())) << m << v;
    } while (true);
TEST_END

TEST_BEGIN(Vec, testMax, (ALL_VECTORS))
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;
    typedef typename Vec::IndexType I;

    Vec v(I::IndexesFromZero());

    COMPARE(v.max(), static_cast<T>(Vec::Size - 1));
    v = T(Vec::Size) - v;
    COMPARE(v.max(), static_cast<T>(Vec::Size));

    int j = 0;
    Mask m;
    do {
        m = allMasks<Vec>(j++);
        if (m.isEmpty()) {
            break;
        }
        COMPARE(v.max(m), static_cast<T>(Vec::Size - m.firstOne())) << m << v;
    } while (true);
TEST_END

TEST_BEGIN(Vec, testProduct, (ALL_VECTORS))
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;

    for (int i = 0; i < 10; ++i) {
        T x = static_cast<T>(i);
        Vec v(x);
        T x2 = x;
        for (int k = Vec::Size; k > 1; k /= 2) {
            x2 *= x2;
        }
        COMPARE(v.product(), x2);

        int j = 0;
        Mask m = allMasks<Vec>(j++);
        COMPARE(v.product(m), x2) << m << v;
        do {
            m = allMasks<Vec>(j++);
            if (m.isEmpty()) {
                break;
            }
            if (std::numeric_limits<T>::is_exact) {
                x2 = x;
                for (int k = m.count(); k > 1; --k) {
                    x2 *= x;
                }
                COMPARE(v.product(m), x2) << m << v;
            } else {
                x2 = std::round(std::pow(x, static_cast<int>(m.count())));
                //x2 = static_cast<T>(pow(static_cast<double>(x), static_cast<int>(m.count())));
                FUZZY_COMPARE(v.product(m), x2) << m << v;
            }
        } while (true);
    }
TEST_END

TEST_BEGIN(Vec, testSum, (ALL_VECTORS))
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;

    for (int i = 0; i < 10; ++i) {
        T x = static_cast<T>(i);
        Vec v(x);
        COMPARE(v.sum(), T(x * Vec::Size));

        int j = 0;
        Mask m;
        do {
            m = allMasks<Vec>(j++);
            COMPARE(v.sum(m), static_cast<T>(x * m.count())) << m << v;
        } while (!m.isEmpty());
    }
TEST_END

TEST_BEGIN(V, testPartialSum, (ALL_VECTORS))
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;

    V reference(I::IndexesFromZero() + 1);
    COMPARE(V(1).partialSum(), reference);
    /* disabled until correct masking is implemented

    reference = V(I(2) << I::IndexesFromZero());
    COMPARE(V(2).partialSum([](const V &a, const V &b) { return a * b; }), reference);
    */
TEST_END

TEST_BEGIN(V, testFma, (ALL_VECTORS))
    for (int i = 0; i < 1000; ++i) {
        V a = V::Random();
        const V b = V::Random();
        const V c = V::Random();
        const V reference = a * b + c;
        a.fusedMultiplyAdd(b, c);
        COMPARE(a, reference) << ", a = " << a << ", b = " << b << ", c = " << c;
    }
TEST_END

template <> struct testFma<float_v>
{
void operator()()
{
    using Vc::Internal::floatConstant;
    float_v b = floatConstant<1, 0x000001, 0>();
    float_v c = floatConstant<1, 0x000000, -24>();
    float_v a = b;
    /*a *= b;
    a += c;
    COMPARE(a, float_v(floatConstant<1, 0x000002, 0>()));
    a = b;*/
    a.fusedMultiplyAdd(b, c);
    COMPARE(a, float_v(floatConstant<1, 0x000003, 0>()));

    a = floatConstant<1, 0x000002, 0>();
    b = floatConstant<1, 0x000002, 0>();
    c = floatConstant<-1, 0x000000, 0>();
    /*a *= b;
    a += c;
    COMPARE(a, float_v(floatConstant<1, 0x000000, -21>()));
    a = b;*/
    a.fusedMultiplyAdd(b, c); // 1 + 2^-21 + 2^-44 - 1 == (1 + 2^-20)*2^-18
    COMPARE(a, float_v(floatConstant<1, 0x000001, -21>()));
}};

template<> struct testFma<double_v>
{
void operator()()
{
    using Vc::Internal::doubleConstant;
    double_v b = doubleConstant<1, 0x0000000000001, 0>();
    double_v c = doubleConstant<1, 0x0000000000000, -53>();
    double_v a = b;
    a.fusedMultiplyAdd(b, c);
    COMPARE(a, double_v(doubleConstant<1, 0x0000000000003, 0>()));

    a = doubleConstant<1, 0x0000000000002, 0>();
    b = doubleConstant<1, 0x0000000000002, 0>();
    c = doubleConstant<-1, 0x0000000000000, 0>();
    a.fusedMultiplyAdd(b, c); // 1 + 2^-50 + 2^-102 - 1
    COMPARE(a, double_v(doubleConstant<1, 0x0000000000001, -50>()));
}};
