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
/*includes {{{*/
#include "unittest.h"
#include <Vc/array>
#include "vectormemoryhelper.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <Vc/common/const.h>
/*}}}*/
using namespace Vc;

// fix isfinite and isnan {{{1
#ifdef isfinite
#undef isfinite
#endif
#ifdef isnan
#undef isnan
#endif

// abs {{{1
TEST_TYPES(V, testAbs,
           concat<RealTypes, int_v, short_v, SimdArray<int, 8>, SimdArray<int, 2>,
                  SimdArray<int, 7>>)
{
    for (int i = 0; i < 0x7fff - int(V::size()); ++i) {
        const V a([&](int n) { return n + i; });
        const V b = -a;
        COMPARE(a, Vc::abs(a));
        COMPARE(a, Vc::abs(b));
    }
    using T = typename V::value_type;
    if (std::is_same<T, short>::value)
    {
        const V a = std::numeric_limits<T>::lowest();
        COMPARE(abs(a), a);
    }
    for (int i = 0; i < 1000; ++i) {
        V a = V::Random();
        if (std::is_integral<T>::value) {
            // Avoid most negative value which doesn't have an absolute value.
            a = max(a, V(std::numeric_limits<T>::min() + 1));
        }
        const V ref = V::generate([&](int j) { return std::abs(a[j]); });
        COMPARE(abs(a), ref) << "a : " << a;
    }
}

TEST_TYPES(V, testTrunc, RealTypes) //{{{1
{
    typedef typename V::EntryType T;
    for (size_t i = 0; i < 100000 / V::Size; ++i) {
        V x = (V::Random() - T(0.5)) * T(100);
        V reference = x.apply([](T _x) { return std::trunc(_x); });
        COMPARE(Vc::trunc(x), reference) << ", x = " << x << ", i = " << i;
    }
    V x([](T n) { return n; });
    V reference = x.apply([](T _x) { return std::trunc(_x); });
    COMPARE(Vc::trunc(x), reference) << ", x = " << x;
}

TEST_TYPES(V, testFloor, RealTypes) //{{{1
{
    typedef typename V::EntryType T;
    for (size_t i = 0; i < 100000 / V::Size; ++i) {
        V x = (V::Random() - T(0.5)) * T(100);
        V reference = x.apply([](T _x) { return std::floor(_x); });
        COMPARE(Vc::floor(x), reference) << ", x = " << x << ", i = " << i;
    }
    V x([](T n) { return n; });
    V reference = x.apply([](T _x) { return std::floor(_x); });
    COMPARE(Vc::floor(x), reference) << ", x = " << x;
}

TEST_TYPES(V, testCeil, RealTypes) //{{{1
{
    typedef typename V::EntryType T;
    for (size_t i = 0; i < 100000 / V::Size; ++i) {
        V x = (V::Random() - T(0.5)) * T(100);
        V reference = x.apply([](T _x) { return std::ceil(_x); });
        COMPARE(Vc::ceil(x), reference) << ", x = " << x << ", i = " << i;
    }
    V x([](T n) { return n; });
    V reference = x.apply([](T _x) { return std::ceil(_x); });
    COMPARE(Vc::ceil(x), reference) << ", x = " << x;
}

TEST_TYPES(V, testExp, RealTypes) //{{{1
{
    setFuzzyness<float>(1);
    setFuzzyness<double>(2);
    typedef typename V::EntryType T;
    for (size_t i = 0; i < 100000 / V::Size; ++i) {
        V x = (V::Random() - T(0.5)) * T(20);
        V reference = x.apply([](T _x) { return std::exp(_x); });
        FUZZY_COMPARE(Vc::exp(x), reference) << ", x = " << x << ", i = " << i;
    }
    COMPARE(Vc::exp(V(0)), V(1));
}

TEST_TYPES(V, testMax, AllTypes) //{{{1
{
    typedef typename V::EntryType T;
    VectorMemoryHelper<V> mem(3);
    T *data = mem;
    for (size_t i = 0; i < V::Size; ++i) {
        data[i] = i;
        data[i + V::Size] = V::Size + 1 - i;
        data[i + 2 * V::Size] = std::max(data[i], data[i + V::Size]);
    }
    V a(&data[0]);
    V b(&data[V::Size]);
    V c(&data[2 * V::Size]);

    COMPARE(Vc::max(a, b), c);
}

TEST_TYPES(V, testSqrt, RealTypes) //{{{1
{
    V data([](int n) { return n; });
    V reference = V::generate([&](int i) { return std::sqrt(data[i]); });

    FUZZY_COMPARE(Vc::sqrt(data), reference);
}

TEST_TYPES(V, testRSqrt, RealTypes) //{{{1
{
    typedef typename V::EntryType T;
    for (size_t i = 0; i < 1024 / V::Size; ++i) {
        const V x = V::Random() * T(1000);
        // RSQRTPS is documented as having a relative error <= 1.5 * 2^-12
        VERIFY(all_of(Vc::abs(Vc::rsqrt(x) * Vc::sqrt(x) - V(1)) < static_cast<T>(std::ldexp(1.5, -12))));
    }
}

TEST_TYPES(V, testReciprocal, RealTypes) //{{{1
{
    typedef typename V::EntryType T;
    setFuzzyness<float>(1.258295e+07);
    setFuzzyness<double>(0);
    const T one = 1;
    for (int offset = -1000; offset < 1000; offset += 10) {
        const T scale = T(0.1);
        V data;
        V reference;
        for (size_t ii = 0; ii < V::Size; ++ii) {
            const T i = static_cast<T>(ii);
            data[ii] = i;
            T tmp = (i + offset) * scale;
            reference[ii] = one / tmp;
        }

        FUZZY_COMPARE(Vc::reciprocal((data + offset) * scale), reference);
    }
}

TEST_TYPES(V, isNegative, RealTypes) //{{{1
{
    typedef typename V::EntryType T;
    VERIFY(isnegative(V(1)).isEmpty());
    VERIFY(isnegative(V(0)).isEmpty());
    VERIFY(isnegative((-V(1))).isFull());
    VERIFY(isnegative(V(T(-0.))).isFull());
}

TEST_TYPES(V, testInf, RealTypes) //{{{1
{
    typedef typename V::EntryType T;
    const T one = 1;
    const V zero(Zero);
    const V inf = one / zero;
    V nan;
    nan.setQnan();

    VERIFY(all_of(Vc::isfinite(zero)));
    VERIFY(all_of(Vc::isfinite(V(one))));
    VERIFY(none_of(Vc::isfinite(inf)));
    VERIFY(none_of(Vc::isfinite(nan)));

    VERIFY(none_of(Vc::isinf(zero)));
    VERIFY(none_of(Vc::isinf(V(one))));
    VERIFY(all_of(Vc::isinf(inf)));
    VERIFY(none_of(Vc::isinf(nan)));
}

TEST_TYPES(V, testNaN, RealTypes) //{{{1
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;
    typedef typename V::Mask M;
    const T one = 1;
    const V zero(Zero);
    VERIFY(none_of(Vc::isnan(zero)));
    VERIFY(none_of(Vc::isnan(V(one))));
    const V inf = one / zero;
    VERIFY(all_of(Vc::isnan(V(inf * zero))));
    V nan = V(0);
    const M mask = simd_cast<M>(I([](int n) { return n; }) == I(0));
    nan.setQnan(mask);
    COMPARE(Vc::isnan(nan), mask);
    nan.setQnan();
    VERIFY(all_of(Vc::isnan(nan)));
}

TEST_TYPES(V, testRound, RealTypes) //{{{1
{
    typedef typename V::EntryType T;
    enum {
        Count = (16 + V::Size) / V::Size
    };
    VectorMemoryHelper<V> mem1(Count);
    VectorMemoryHelper<V> mem2(Count);
    T *data = mem1;
    T *reference = mem2;
    for (size_t i = 0; i < Count * V::Size; ++i) {
        data[i] = i * 0.25 - 2.0;
        reference[i] = std::floor(i * 0.25 - 2.0 + 0.5);
        if (i % 8 == 2) {
            reference[i] -= 1.;
        }
        //std::cout << reference[i] << " ";
    }
    //std::cout << std::endl;
    for (int i = 0; i < Count; ++i) {
        const V a(&data[i * V::Size]);
        const V ref(&reference[i * V::Size]);
        //std::cout << a << ref << std::endl;
        COMPARE(Vc::round(a), ref);
    }
}

TEST_TYPES(V, testExponent, RealTypes) //{{{1
{
    typedef typename V::EntryType T;
    Vc::array<T, 32> input;
    Vc::array<T, 32> expected;
    input[ 0] = T(0.25); expected[ 0] = T(-2);
    input[ 1] = T(   1); expected[ 1] = T( 0);
    input[ 2] = T(   2); expected[ 2] = T( 1);
    input[ 3] = T(   3); expected[ 3] = T( 1);
    input[ 4] = T(   4); expected[ 4] = T( 2);
    input[ 5] = T( 0.5); expected[ 5] = T(-1);
    input[ 6] = T(   6); expected[ 6] = T( 2);
    input[ 7] = T(   7); expected[ 7] = T( 2);
    input[ 8] = T(   8); expected[ 8] = T( 3);
    input[ 9] = T(   9); expected[ 9] = T( 3);
    input[10] = T(  10); expected[10] = T( 3);
    input[11] = T(  11); expected[11] = T( 3);
    input[12] = T(  12); expected[12] = T( 3);
    input[13] = T(  13); expected[13] = T( 3);
    input[14] = T(  14); expected[14] = T( 3);
    input[15] = T(  15); expected[15] = T( 3);
    input[16] = T(  16); expected[16] = T( 4);
    input[17] = T(  17); expected[17] = T( 4);
    input[18] = T(  18); expected[18] = T( 4);
    input[19] = T(  19); expected[19] = T( 4);
    input[20] = T(  20); expected[20] = T( 4);
    input[21] = T(  21); expected[21] = T( 4);
    input[22] = T(  22); expected[22] = T( 4);
    input[23] = T(  23); expected[23] = T( 4);
    input[24] = T(  24); expected[24] = T( 4);
    input[25] = T(  25); expected[25] = T( 4);
    input[26] = T(  26); expected[26] = T( 4);
    input[27] = T(  27); expected[27] = T( 4);
    input[28] = T(  28); expected[28] = T( 4);
    input[29] = T(  29); expected[29] = T( 4);
    input[30] = T(  32); expected[30] = T( 5);
    input[31] = T(  31); expected[31] = T( 4);
    for (size_t i = 0; i <= input.size() - V::size(); ++i) {
        COMPARE(exponent(V(&input[i])), V(&expected[i]));
    }
}

template<typename T> struct _ExponentVector { typedef int_v Type; };

TEST_TYPES(V, testFrexp, RealTypes) //{{{1
{
    typedef typename V::EntryType T;
    using ExpV = typename V::IndexType;
    Vc::array<T, 33> input;
    Vc::array<T, 33> expectedFraction;
    Vc::array<int, 33> expectedExponent;
    input[ 0] = T(0.25); expectedFraction[ 0] = T(.5     ); expectedExponent[ 0] = -1;
    input[ 1] = T(   1); expectedFraction[ 1] = T(.5     ); expectedExponent[ 1] =  1;
    input[ 2] = T(   0); expectedFraction[ 2] = T(0.     ); expectedExponent[ 2] =  0;
    input[ 3] = T(   3); expectedFraction[ 3] = T(.75    ); expectedExponent[ 3] =  2;
    input[ 4] = T(   4); expectedFraction[ 4] = T(.5     ); expectedExponent[ 4] =  3;
    input[ 5] = T( 0.5); expectedFraction[ 5] = T(.5     ); expectedExponent[ 5] =  0;
    input[ 6] = T(   6); expectedFraction[ 6] = T( 6./8. ); expectedExponent[ 6] =  3;
    input[ 7] = T(   7); expectedFraction[ 7] = T( 7./8. ); expectedExponent[ 7] =  3;
    input[ 8] = T(   8); expectedFraction[ 8] = T( 8./16.); expectedExponent[ 8] =  4;
    input[ 9] = T(   9); expectedFraction[ 9] = T( 9./16.); expectedExponent[ 9] =  4;
    input[10] = T(  10); expectedFraction[10] = T(10./16.); expectedExponent[10] =  4;
    input[11] = T(  11); expectedFraction[11] = T(11./16.); expectedExponent[11] =  4;
    input[12] = T(  12); expectedFraction[12] = T(12./16.); expectedExponent[12] =  4;
    input[13] = T(  13); expectedFraction[13] = T(13./16.); expectedExponent[13] =  4;
    input[14] = T(  14); expectedFraction[14] = T(14./16.); expectedExponent[14] =  4;
    input[15] = T(  15); expectedFraction[15] = T(15./16.); expectedExponent[15] =  4;
    input[16] = T(  16); expectedFraction[16] = T(16./32.); expectedExponent[16] =  5;
    input[17] = T(  17); expectedFraction[17] = T(17./32.); expectedExponent[17] =  5;
    input[18] = T(  18); expectedFraction[18] = T(18./32.); expectedExponent[18] =  5;
    input[19] = T(  19); expectedFraction[19] = T(19./32.); expectedExponent[19] =  5;
    input[20] = T(  20); expectedFraction[20] = T(20./32.); expectedExponent[20] =  5;
    input[21] = T(  21); expectedFraction[21] = T(21./32.); expectedExponent[21] =  5;
    input[22] = T(  22); expectedFraction[22] = T(22./32.); expectedExponent[22] =  5;
    input[23] = T(  23); expectedFraction[23] = T(23./32.); expectedExponent[23] =  5;
    input[24] = T(  24); expectedFraction[24] = T(24./32.); expectedExponent[24] =  5;
    input[25] = T(  25); expectedFraction[25] = T(25./32.); expectedExponent[25] =  5;
    input[26] = T(  26); expectedFraction[26] = T(26./32.); expectedExponent[26] =  5;
    input[27] = T(  27); expectedFraction[27] = T(27./32.); expectedExponent[27] =  5;
    input[28] = T(  28); expectedFraction[28] = T(28./32.); expectedExponent[28] =  5;
    input[29] = T(  29); expectedFraction[29] = T(29./32.); expectedExponent[29] =  5;
    input[30] = T(  32); expectedFraction[30] = T(32./64.); expectedExponent[30] =  6;
    input[31] = T(  31); expectedFraction[31] = T(31./32.); expectedExponent[31] =  5;
    input[32] = T( -0.); expectedFraction[32] = T(-0.    ); expectedExponent[32] =  0;
    for (size_t i = 0; i <= 33 - V::size(); ++i) {
        const V v(&input[i]);
        ExpV exp;
        const V fraction = frexp(v, &exp);
        COMPARE(fraction, V(&expectedFraction[i]))
            << ", v = " << v << ", delta: " << fraction - V(&expectedFraction[i]);
        COMPARE(exp, ExpV(&expectedExponent[i]))
            << "\ninput: " << v << ", fraction: " << fraction << ", i: " << i;
    }
}

TEST_TYPES(V, testLdexp, RealTypes) //{{{1
{
    typedef typename V::EntryType T;
    using ExpV = typename V::IndexType;
    for (size_t i = 0; i < 1024 / V::Size; ++i) {
        const V v = (V::Random() - T(0.5)) * T(1000);
        ExpV e;
        const V m = frexp(v, &e);
        COMPARE(ldexp(m, e), v) << ", m = " << m << ", e = " << e;
    }
}

#include "ulp.h"
// copysign {{{1
TEST_TYPES(V, testCopysign, RealTypes)
{
    const V x = V::Random();
    const V y = -x;
    const V z = copysign(x, y);
    COMPARE(abs(x), abs(z));
    COMPARE(y > 0, z > 0);
    COMPARE(z, V::generate([&](int i) { return std::copysign(x[i], y[i]); }));
}

//}}}1

// vim: foldmethod=marker
