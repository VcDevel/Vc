/*  This file is part of the Vc library. {{{
Copyright © 2009-2016 Matthias Kretz <kretz@kde.org>

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

#define WITH_DATAPAR 1
//#define UNITTEST_ONLY_XTEST 1
#include "unittest.h"
#include <Vc/datapar>

template <class... Ts> using base_template = Vc::datapar<Ts...>;
#include "testtypes.h"

// datapar generator function {{{1
template <class V>
inline V make_vec(const std::initializer_list<typename V::value_type> &init,
                  typename V::value_type inc = 0)
{
    std::size_t i = 0;
    V r;
    typename V::value_type base = 0;
    for (;;) {
        for (auto x : init) {
            r[i] = base + x;
            if (++i == V::size()) {
                return r;
            }
        }
        base += inc;
    }
}

TEST_TYPES(V, broadcast, ALL_TYPES)  //{{{1
{
    using T = typename V::value_type;
    VERIFY(Vc::is_datapar_v<V>);

    {
        V x;      // default broadcasts 0
        x = V{};  // default broadcasts 0
        x = V();  // default broadcasts 0
        x = x;
        for (std::size_t i = 0; i < V::size(); ++i) {
            COMPARE(x[i], T(0));
        }
    }

    V x = 3;
    V y = 0;
    for (std::size_t i = 0; i < V::size(); ++i) {
        COMPARE(x[i], T(3));
        COMPARE(y[i], T(0));
    }
    y = 3;
    COMPARE(x, y);
}

//operators helpers  //{{{1
template <class T> constexpr T genHalfBits()
{
    return std::numeric_limits<T>::max() >> (std::numeric_limits<T>::digits / 2);
}
template <> constexpr long double genHalfBits<long double>() { return 0; }
template <> constexpr double genHalfBits<double>() { return 0; }
template <> constexpr float genHalfBits<float>() { return 0; }

TEST_TYPES(V, operators, ALL_TYPES)  //{{{1
{
    using M = typename V::mask_type;
    using T = typename V::value_type;
    constexpr auto min = std::numeric_limits<T>::min();
    constexpr auto max = std::numeric_limits<T>::max();
    {  // compares{{{2
        constexpr T half = genHalfBits<T>();
        for (T lo_ : {min, T(min + 1), T(-1), T(0), T(1), T(half - 1), half, T(half + 1),
                      T(max - 1)}) {
            for (T hi_ : {T(min + 1), T(-1), T(0), T(1), T(half - 1), half, T(half + 1),
                          T(max - 1), max}) {
                if (hi_ <= lo_) {
                    continue;
                }
                for (std::size_t pos = 0; pos < V::size(); ++pos) {
                    V lo = lo_;
                    V hi = hi_;
                    lo[pos] = 0;  // have a different value in the vector in case
                    hi[pos] = 1;  // this affects neighbors
                    COMPARE(hi, hi);
                    VERIFY(all_of(hi != lo)) << "hi: " << hi << ", lo: " << lo;
                    VERIFY(all_of(lo != hi)) << "hi: " << hi << ", lo: " << lo;
                    VERIFY(none_of(hi != hi)) << "hi: " << hi << ", lo: " << lo;
                    VERIFY(none_of(hi == lo)) << "hi: " << hi << ", lo: " << lo;
                    VERIFY(none_of(lo == hi)) << "hi: " << hi << ", lo: " << lo;
                    VERIFY(all_of(lo < hi)) << "hi: " << hi << ", lo: " << lo;
                    VERIFY(none_of(hi < lo)) << "hi: " << hi << ", lo: " << lo;
                    VERIFY(none_of(hi <= lo)) << "hi: " << hi << ", lo: " << lo;
                    VERIFY(all_of(hi <= hi)) << "hi: " << hi << ", lo: " << lo;
                    VERIFY(all_of(hi > lo)) << "hi: " << hi << ", lo: " << lo;
                    VERIFY(none_of(lo > hi)) << "hi: " << hi << ", lo: " << lo;
                    VERIFY(all_of(hi >= lo)) << "hi: " << hi << ", lo: " << lo;
                    VERIFY(all_of(hi >= hi)) << "hi: " << hi << ", lo: " << lo;
                }
            }
        }
    }
    {  // subscripting{{{2
        V x = max;
        for (std::size_t i = 0; i < V::size(); ++i) {
            COMPARE(x[i], max);
            x[i] = 0;
        }
        COMPARE(x, V{0});
        for (std::size_t i = 0; i < V::size(); ++i) {
            COMPARE(x[i], T(0));
            x[i] = max;
        }
        COMPARE(x, V{max});
    }
    {  // not{{{2
        V x = 0;
        COMPARE(!x, M{true});
        V y = 1;
        COMPARE(!y, M{false});
    }

    {  // unary minus{{{2
        V x = 0;
        COMPARE(-x, V(T(-T(0))));
        V y = 1;
        COMPARE(-y, V(T(-T(1))));
    }

    {  // plus{{{2
        V x = 0;
        V y = 0;
        COMPARE(x + y, x);
        COMPARE(x = x + T(1), V(1));
        COMPARE(x + x, V(2));
        y = make_vec<V>({1, 2, 3, 4, 5, 6, 7});
        COMPARE(x = x + y, make_vec<V>({2, 3, 4, 5, 6, 7, 8}));
        COMPARE(x = x + -y, V(1));
    }

    {  // minus{{{2
        V x = 1;
        V y = 0;
        COMPARE(x - y, x);
        COMPARE(x - T(1), y);
        COMPARE(y, x - T(1));
        COMPARE(x - x, y);
        y = make_vec<V>({1, 2, 3, 4, 5, 6, 7});
        COMPARE(x = y - x, make_vec<V>({0, 1, 2, 3, 4, 5, 6}));
        COMPARE(x = y - x, V(1));
    }

    {  // multiplies{{{2
        V x = 1;
        V y = 0;
        COMPARE(x * y, y);
        COMPARE(x = x * T(2), V(2));
        COMPARE(x * x, V(4));
        y = make_vec<V>({1, 2, 3, 4, 5, 6, 7});
        COMPARE(x = x * y, make_vec<V>({2, 4, 6, 8, 10, 12, 14}));
        y = 2;
        for (T n : {T(std::numeric_limits<T>::max() - 1), std::numeric_limits<T>::min()}) {
            x = n / 2;
            COMPARE(x * y, V(n));
        }
        if (std::is_integral<T>::value && std::is_unsigned<T>::value) {
            // test modulo arithmetics
            T n = std::numeric_limits<T>::max();
            x = n;
            for (T m : {T(2), T(7), T(std::numeric_limits<T>::max() / 127), std::numeric_limits<T>::max()}) {
                y = m;
                COMPARE(x * y, V(T(n * m)));
            }
        }
    }

    {  // divides{{{2
        V x = 2;
        COMPARE(x / x, V(1));
        COMPARE(T(3) / x, V(T(3) / T(2)));
        COMPARE(x / T(3), V(T(2) / T(3)));
        V y = make_vec<V>({1, 2, 3, 4, 5, 6, 7});
        COMPARE(y / x, make_vec<V>({T(.5), T(1), T(1.5), T(2), T(2.5), T(3), T(3.5)}));

        y = make_vec<V>({std::numeric_limits<T>::max(), std::numeric_limits<T>::min()});
        V ref = make_vec<V>(
            {T(std::numeric_limits<T>::max() / 2), T(std::numeric_limits<T>::min() / 2)});
        COMPARE(y / x, ref);

        y = make_vec<V>({std::numeric_limits<T>::min(), std::numeric_limits<T>::max()});
        ref = make_vec<V>(
            {T(std::numeric_limits<T>::min() / 2), T(std::numeric_limits<T>::max() / 2)});
        COMPARE(y / x, ref);

        y = make_vec<V>(
            {std::numeric_limits<T>::max(), T(std::numeric_limits<T>::min() + 1)});
        COMPARE(y / y, V(1));

        ref = make_vec<V>({T(2 / std::numeric_limits<T>::max()),
                           T(2 / (std::numeric_limits<T>::min() + 1))});
        COMPARE(x / y, ref);
    }

    {  // increment & decrement {{{2
        const V from0 = make_vec<V>({0, 1, 2, 3}, 4);
        V x = from0;
        COMPARE(x++, from0);
        COMPARE(x, from0 + 1);
        COMPARE(++x, from0 + 2);
        COMPARE(x, from0 + 2);

        COMPARE(x--, from0 + 2);
        COMPARE(x, from0 + 1);
        COMPARE(--x, from0);
        COMPARE(x, from0);
    }
}

template <class A, class B, class Expected> void binary_op_return_type()  //{{{1
{
    const auto name = typeToString<A>() + " + " + typeToString<B>();
    COMPARE(typeid(A() + B()), typeid(Expected)) << name;
    COMPARE(typeid(B() + A()), typeid(Expected)) << name;
    UnitTest::ADD_PASS() << name;
}

TEST(operator_conversions)  //{{{1
{
    // float{{{2
    binary_op_return_type<vfloat, vfloat, vfloat>();
    binary_op_return_type<vfloat, schar, vfloat>();
    binary_op_return_type<vfloat, uchar, vfloat>();
    binary_op_return_type<vfloat, short, vfloat>();
    binary_op_return_type<vfloat, ushort, vfloat>();
    binary_op_return_type<vfloat, int, vfloat>();
    binary_op_return_type<vfloat, uint, vfloat>();
    binary_op_return_type<vfloat, long, vfloat>();
    binary_op_return_type<vfloat, ulong, vfloat>();
    binary_op_return_type<vfloat, llong, vfloat>();
    binary_op_return_type<vfloat, ullong, vfloat>();
    binary_op_return_type<vfloat, float, vfloat>();
    binary_op_return_type<vfloat, double, vf32<double>>();
    binary_op_return_type<vfloat, vf32<schar>, vfloat>();
    binary_op_return_type<vfloat, vf32<uchar>, vfloat>();
    binary_op_return_type<vfloat, vf32<short>, vfloat>();
    binary_op_return_type<vfloat, vf32<ushort>, vfloat>();
    binary_op_return_type<vfloat, vf32<int>, vfloat>();
    binary_op_return_type<vfloat, vf32<uint>, vfloat>();
    binary_op_return_type<vfloat, vf32<long>, vfloat>();
    binary_op_return_type<vfloat, vf32<ulong>, vfloat>();
    binary_op_return_type<vfloat, vf32<llong>, vfloat>();
    binary_op_return_type<vfloat, vf32<ullong>, vfloat>();
    binary_op_return_type<vfloat, vf32<float>, vfloat>();
    binary_op_return_type<vfloat, vf32<double>, vf32<double>>();

    binary_op_return_type<vf32<float>, vfloat, vfloat>();
    binary_op_return_type<vf32<float>, vf32<float>, vf32<float>>();
    binary_op_return_type<vf32<float>, schar, vf32<float>>();
    binary_op_return_type<vf32<float>, uchar, vf32<float>>();
    binary_op_return_type<vf32<float>, short, vf32<float>>();
    binary_op_return_type<vf32<float>, ushort, vf32<float>>();
    binary_op_return_type<vf32<float>, int, vf32<float>>();
    binary_op_return_type<vf32<float>, uint, vf32<float>>();
    binary_op_return_type<vf32<float>, long, vf32<float>>();
    binary_op_return_type<vf32<float>, ulong, vf32<float>>();
    binary_op_return_type<vf32<float>, llong, vf32<float>>();
    binary_op_return_type<vf32<float>, ullong, vf32<float>>();
    binary_op_return_type<vf32<float>, float, vf32<float>>();
    binary_op_return_type<vf32<float>, double, vf32<double>>();
    binary_op_return_type<vf32<float>, vf32<schar>, vf32<float>>();
    binary_op_return_type<vf32<float>, vf32<uchar>, vf32<float>>();
    binary_op_return_type<vf32<float>, vf32<short>, vf32<float>>();
    binary_op_return_type<vf32<float>, vf32<ushort>, vf32<float>>();
    binary_op_return_type<vf32<float>, vf32<int>, vf32<float>>();
    binary_op_return_type<vf32<float>, vf32<uint>, vf32<float>>();
    binary_op_return_type<vf32<float>, vf32<long>, vf32<float>>();
    binary_op_return_type<vf32<float>, vf32<ulong>, vf32<float>>();
    binary_op_return_type<vf32<float>, vf32<llong>, vf32<float>>();
    binary_op_return_type<vf32<float>, vf32<ullong>, vf32<float>>();
    binary_op_return_type<vf32<float>, vf32<float>, vf32<float>>();
    binary_op_return_type<vf32<float>, vf32<double>, vf32<double>>();

    // double{{{2
    binary_op_return_type<vdouble, vdouble, vdouble>();
    binary_op_return_type<vdouble, schar, vdouble>();
    binary_op_return_type<vdouble, uchar, vdouble>();
    binary_op_return_type<vdouble, short, vdouble>();
    binary_op_return_type<vdouble, ushort, vdouble>();
    binary_op_return_type<vdouble, int, vdouble>();
    binary_op_return_type<vdouble, uint, vdouble>();
    binary_op_return_type<vdouble, long, vdouble>();
    binary_op_return_type<vdouble, ulong, vdouble>();
    binary_op_return_type<vdouble, llong, vdouble>();
    binary_op_return_type<vdouble, ullong, vdouble>();
    binary_op_return_type<vdouble, float, vdouble>();
    binary_op_return_type<vdouble, double, vdouble>();
    binary_op_return_type<vdouble, vf64<schar>, vdouble>();
    binary_op_return_type<vdouble, vf64<uchar>, vdouble>();
    binary_op_return_type<vdouble, vf64<short>, vdouble>();
    binary_op_return_type<vdouble, vf64<ushort>, vdouble>();
    binary_op_return_type<vdouble, vf64<int>, vdouble>();
    binary_op_return_type<vdouble, vf64<uint>, vdouble>();
    binary_op_return_type<vdouble, vf64<long>, vdouble>();
    binary_op_return_type<vdouble, vf64<ulong>, vdouble>();
    binary_op_return_type<vdouble, vf64<llong>, vdouble>();
    binary_op_return_type<vdouble, vf64<ullong>, vdouble>();
    binary_op_return_type<vdouble, vf64<float>, vdouble>();
    binary_op_return_type<vdouble, vf64<double>, vdouble>();

    binary_op_return_type<vf64<double>, vdouble, vdouble>();
    binary_op_return_type<vf64<double>, vf64<float>, vf64<double>>();
    binary_op_return_type<vf64<double>, schar, vf64<double>>();
    binary_op_return_type<vf64<double>, uchar, vf64<double>>();
    binary_op_return_type<vf64<double>, short, vf64<double>>();
    binary_op_return_type<vf64<double>, ushort, vf64<double>>();
    binary_op_return_type<vf64<double>, int, vf64<double>>();
    binary_op_return_type<vf64<double>, uint, vf64<double>>();
    binary_op_return_type<vf64<double>, long, vf64<double>>();
    binary_op_return_type<vf64<double>, ulong, vf64<double>>();
    binary_op_return_type<vf64<double>, llong, vf64<double>>();
    binary_op_return_type<vf64<double>, ullong, vf64<double>>();
    binary_op_return_type<vf64<double>, float, vf64<double>>();
    binary_op_return_type<vf64<double>, double, vf64<double>>();
    binary_op_return_type<vf64<double>, vf64<schar>, vf64<double>>();
    binary_op_return_type<vf64<double>, vf64<uchar>, vf64<double>>();
    binary_op_return_type<vf64<double>, vf64<short>, vf64<double>>();
    binary_op_return_type<vf64<double>, vf64<ushort>, vf64<double>>();
    binary_op_return_type<vf64<double>, vf64<int>, vf64<double>>();
    binary_op_return_type<vf64<double>, vf64<uint>, vf64<double>>();
    binary_op_return_type<vf64<double>, vf64<long>, vf64<double>>();
    binary_op_return_type<vf64<double>, vf64<ulong>, vf64<double>>();
    binary_op_return_type<vf64<double>, vf64<llong>, vf64<double>>();
    binary_op_return_type<vf64<double>, vf64<ullong>, vf64<double>>();
    binary_op_return_type<vf64<double>, vf64<float>, vf64<double>>();
    binary_op_return_type<vf64<double>, vf64<double>, vf64<double>>();

    binary_op_return_type<vf32<double>, schar, vf32<double>>();
    binary_op_return_type<vf32<double>, uchar, vf32<double>>();
    binary_op_return_type<vf32<double>, short, vf32<double>>();
    binary_op_return_type<vf32<double>, ushort, vf32<double>>();
    binary_op_return_type<vf32<double>, int, vf32<double>>();
    binary_op_return_type<vf32<double>, uint, vf32<double>>();
    binary_op_return_type<vf32<double>, long, vf32<double>>();
    binary_op_return_type<vf32<double>, ulong, vf32<double>>();
    binary_op_return_type<vf32<double>, llong, vf32<double>>();
    binary_op_return_type<vf32<double>, ullong, vf32<double>>();
    binary_op_return_type<vf32<double>, float, vf32<double>>();
    binary_op_return_type<vf32<double>, double, vf32<double>>();

    // long{{{2
    binary_op_return_type<vlong, vlong, vlong>();
    binary_op_return_type<vlong, vulong, vulong>();
    binary_op_return_type<vlong, schar, vlong>();
    binary_op_return_type<vlong, uchar, vlong>();
    binary_op_return_type<vlong, short, vlong>();
    binary_op_return_type<vlong, ushort, vlong>();
    binary_op_return_type<vlong, int, vlong>();
    binary_op_return_type<vlong, uint, Vc::native_datapar<decltype(long() + uint())>>();
    binary_op_return_type<vlong, long, vlong>();
    binary_op_return_type<vlong, ulong, vulong>();
    binary_op_return_type<vlong, llong, vl<llong>>();
    binary_op_return_type<vlong, ullong, vl<ullong>>();
    binary_op_return_type<vlong, float, vl<float>>();
    binary_op_return_type<vlong, double, vl<double>>();
    binary_op_return_type<vlong, vl<schar>, vlong>();
    binary_op_return_type<vlong, vl<uchar>, vlong>();
    binary_op_return_type<vlong, vl<short>, vlong>();
    binary_op_return_type<vlong, vl<ushort>, vlong>();
    binary_op_return_type<vlong, vl<int>, vlong>();
    binary_op_return_type<vlong, vl<uint>, Vc::native_datapar<decltype(long() + uint())>>();
    binary_op_return_type<vlong, vl<long>, vlong>();
    binary_op_return_type<vlong, vl<ulong>, vulong>();
    binary_op_return_type<vlong, vl<llong>, vl<llong>>();
    binary_op_return_type<vlong, vl<ullong>, vl<ullong>>();
    binary_op_return_type<vlong, vl<float>, vl<float>>();
    binary_op_return_type<vlong, vl<double>, vl<double>>();

    binary_op_return_type<vl<long>, vlong, vlong>();
    binary_op_return_type<vl<long>, vulong, vulong>();
    binary_op_return_type<vi32<long>, schar, vi32<long>>();
    binary_op_return_type<vi32<long>, uchar, vi32<long>>();
    binary_op_return_type<vi32<long>, short, vi32<long>>();
    binary_op_return_type<vi32<long>, ushort, vi32<long>>();
    binary_op_return_type<vi32<long>, int, vi32<long>>();
    binary_op_return_type<vi32<long>, uint, vi32<decltype(long() + uint())>>();
    binary_op_return_type<vi32<long>, long, vi32<long>>();
    binary_op_return_type<vi32<long>, ulong, vi32<ulong>>();
    binary_op_return_type<vi32<long>, llong, vi32<llong>>();
    binary_op_return_type<vi32<long>, ullong, vi32<ullong>>();
    binary_op_return_type<vi32<long>, float, vi32<float>>();
    binary_op_return_type<vi32<long>, double, vi32<double>>();
    binary_op_return_type<vi32<long>, vi32<schar>, vi32<long>>();
    binary_op_return_type<vi32<long>, vi32<uchar>, vi32<long>>();
    binary_op_return_type<vi32<long>, vi32<short>, vi32<long>>();
    binary_op_return_type<vi32<long>, vi32<ushort>, vi32<long>>();
    binary_op_return_type<vi32<long>, vi32<int>, vi32<long>>();
    binary_op_return_type<vi32<long>, vi32<uint>, vi32<decltype(long() + uint())>>();
    binary_op_return_type<vi32<long>, vi32<long>, vi32<long>>();
    binary_op_return_type<vi32<long>, vi32<ulong>, vi32<ulong>>();
    binary_op_return_type<vi32<long>, vi32<llong>, vi32<llong>>();
    binary_op_return_type<vi32<long>, vi32<ullong>, vi32<ullong>>();
    binary_op_return_type<vi32<long>, vi32<float>, vi32<float>>();
    binary_op_return_type<vi32<long>, vi32<double>, vi32<double>>();

    binary_op_return_type<vi64<long>, schar, vi64<long>>();
    binary_op_return_type<vi64<long>, uchar, vi64<long>>();
    binary_op_return_type<vi64<long>, short, vi64<long>>();
    binary_op_return_type<vi64<long>, ushort, vi64<long>>();
    binary_op_return_type<vi64<long>, int, vi64<long>>();
    binary_op_return_type<vi64<long>, uint, vi64<decltype(long() + uint())>>();
    binary_op_return_type<vi64<long>, long, vi64<long>>();
    binary_op_return_type<vi64<long>, ulong, vi64<ulong>>();
    binary_op_return_type<vi64<long>, llong, vi64<llong>>();
    binary_op_return_type<vi64<long>, ullong, vi64<ullong>>();
    binary_op_return_type<vi64<long>, float, vi64<float>>();
    binary_op_return_type<vi64<long>, double, vi64<double>>();
    binary_op_return_type<vi64<long>, vi64<schar>, vi64<long>>();
    binary_op_return_type<vi64<long>, vi64<uchar>, vi64<long>>();
    binary_op_return_type<vi64<long>, vi64<short>, vi64<long>>();
    binary_op_return_type<vi64<long>, vi64<ushort>, vi64<long>>();
    binary_op_return_type<vi64<long>, vi64<int>, vi64<long>>();
    binary_op_return_type<vi64<long>, vi64<uint>, vi64<decltype(long() + uint())>>();
    binary_op_return_type<vi64<long>, vi64<long>, vi64<long>>();
    binary_op_return_type<vi64<long>, vi64<ulong>, vi64<ulong>>();
    binary_op_return_type<vi64<long>, vi64<llong>, vi64<llong>>();
    binary_op_return_type<vi64<long>, vi64<ullong>, vi64<ullong>>();
    binary_op_return_type<vi64<long>, vi64<float>, vi64<float>>();
    binary_op_return_type<vi64<long>, vi64<double>, vi64<double>>();

    // ulong{{{2
    binary_op_return_type<vulong, vlong, vulong>();
    binary_op_return_type<vulong, vulong, vulong>();
    binary_op_return_type<vulong, schar, vulong>();
    binary_op_return_type<vulong, uchar, vulong>();
    binary_op_return_type<vulong, short, vulong>();
    binary_op_return_type<vulong, ushort, vulong>();
    binary_op_return_type<vulong, int, vulong>();
    binary_op_return_type<vulong, uint, vulong>();
    binary_op_return_type<vulong, long, vulong>();
    binary_op_return_type<vulong, ulong, vulong>();
    binary_op_return_type<vulong, llong, vl<decltype(ulong() + llong())>>();
    binary_op_return_type<vulong, ullong, vl<ullong>>();
    binary_op_return_type<vulong, float, vl<float>>();
    binary_op_return_type<vulong, double, vl<double>>();
    binary_op_return_type<vulong, vl<schar>, vulong>();
    binary_op_return_type<vulong, vl<uchar>, vulong>();
    binary_op_return_type<vulong, vl<short>, vulong>();
    binary_op_return_type<vulong, vl<ushort>, vulong>();
    binary_op_return_type<vulong, vl<int>, vulong>();
    binary_op_return_type<vulong, vl<uint>, vulong>();
    binary_op_return_type<vulong, vl<long>, vulong>();
    binary_op_return_type<vulong, vl<ulong>, vulong>();
    binary_op_return_type<vulong, vl<llong>, vl<decltype(ulong() + llong())>>();
    binary_op_return_type<vulong, vl<ullong>, vl<ullong>>();
    binary_op_return_type<vulong, vl<float>, vl<float>>();
    binary_op_return_type<vulong, vl<double>, vl<double>>();

    binary_op_return_type<vl<ulong>, vlong, vulong>();
    binary_op_return_type<vl<ulong>, vulong, vulong>();
    binary_op_return_type<vi32<ulong>, schar, vi32<ulong>>();
    binary_op_return_type<vi32<ulong>, uchar, vi32<ulong>>();
    binary_op_return_type<vi32<ulong>, short, vi32<ulong>>();
    binary_op_return_type<vi32<ulong>, ushort, vi32<ulong>>();
    binary_op_return_type<vi32<ulong>, int, vi32<ulong>>();
    binary_op_return_type<vi32<ulong>, uint, vi32<ulong>>();
    binary_op_return_type<vi32<ulong>, long, vi32<ulong>>();
    binary_op_return_type<vi32<ulong>, ulong, vi32<ulong>>();
    binary_op_return_type<vi32<ulong>, llong, vi32<decltype(ulong() + llong())>>();
    binary_op_return_type<vi32<ulong>, ullong, vi32<ullong>>();
    binary_op_return_type<vi32<ulong>, float, vi32<float>>();
    binary_op_return_type<vi32<ulong>, double, vi32<double>>();
    binary_op_return_type<vi32<ulong>, vi32<schar>, vi32<ulong>>();
    binary_op_return_type<vi32<ulong>, vi32<uchar>, vi32<ulong>>();
    binary_op_return_type<vi32<ulong>, vi32<short>, vi32<ulong>>();
    binary_op_return_type<vi32<ulong>, vi32<ushort>, vi32<ulong>>();
    binary_op_return_type<vi32<ulong>, vi32<int>, vi32<ulong>>();
    binary_op_return_type<vi32<ulong>, vi32<uint>, vi32<ulong>>();
    binary_op_return_type<vi32<ulong>, vi32<long>, vi32<ulong>>();
    binary_op_return_type<vi32<ulong>, vi32<ulong>, vi32<ulong>>();
    binary_op_return_type<vi32<ulong>, vi32<llong>, vi32<decltype(ulong() + llong())>>();
    binary_op_return_type<vi32<ulong>, vi32<ullong>, vi32<ullong>>();
    binary_op_return_type<vi32<ulong>, vi32<float>, vi32<float>>();
    binary_op_return_type<vi32<ulong>, vi32<double>, vi32<double>>();

    binary_op_return_type<vi64<ulong>, schar, vi64<ulong>>();
    binary_op_return_type<vi64<ulong>, uchar, vi64<ulong>>();
    binary_op_return_type<vi64<ulong>, short, vi64<ulong>>();
    binary_op_return_type<vi64<ulong>, ushort, vi64<ulong>>();
    binary_op_return_type<vi64<ulong>, int, vi64<ulong>>();
    binary_op_return_type<vi64<ulong>, uint, vi64<ulong>>();
    binary_op_return_type<vi64<ulong>, long, vi64<ulong>>();
    binary_op_return_type<vi64<ulong>, ulong, vi64<ulong>>();
    binary_op_return_type<vi64<ulong>, llong, vi64<decltype(ulong() + llong())>>();
    binary_op_return_type<vi64<ulong>, ullong, vi64<ullong>>();
    binary_op_return_type<vi64<ulong>, float, vi64<float>>();
    binary_op_return_type<vi64<ulong>, double, vi64<double>>();
    binary_op_return_type<vi64<ulong>, vi64<schar>, vi64<ulong>>();
    binary_op_return_type<vi64<ulong>, vi64<uchar>, vi64<ulong>>();
    binary_op_return_type<vi64<ulong>, vi64<short>, vi64<ulong>>();
    binary_op_return_type<vi64<ulong>, vi64<ushort>, vi64<ulong>>();
    binary_op_return_type<vi64<ulong>, vi64<int>, vi64<ulong>>();
    binary_op_return_type<vi64<ulong>, vi64<uint>, vi64<ulong>>();
    binary_op_return_type<vi64<ulong>, vi64<long>, vi64<ulong>>();
    binary_op_return_type<vi64<ulong>, vi64<ulong>, vi64<ulong>>();
    binary_op_return_type<vi64<ulong>, vi64<llong>, vi64<decltype(ulong() + llong())>>();
    binary_op_return_type<vi64<ulong>, vi64<ullong>, vi64<ullong>>();
    binary_op_return_type<vi64<ulong>, vi64<float>, vi64<float>>();
    binary_op_return_type<vi64<ulong>, vi64<double>, vi64<double>>();

    // llong{{{2
    binary_op_return_type<vllong, vllong, vllong>();
    binary_op_return_type<vllong, vullong, vullong>();
    binary_op_return_type<vllong, schar, vllong>();
    binary_op_return_type<vllong, uchar, vllong>();
    binary_op_return_type<vllong, short, vllong>();
    binary_op_return_type<vllong, ushort, vllong>();
    binary_op_return_type<vllong, int, vllong>();
    binary_op_return_type<vllong, uint, Vc::native_datapar<decltype(llong() + uint())>>();
    binary_op_return_type<vllong, long, vllong>();
    binary_op_return_type<vllong, ulong, Vc::native_datapar<decltype(llong() + ulong())>>();
    binary_op_return_type<vllong, llong, vllong>();
    binary_op_return_type<vllong, ullong, vullong>();
    binary_op_return_type<vllong, float, vi64<float>>();
    binary_op_return_type<vllong, double, vi64<double>>();
    binary_op_return_type<vllong, vi64<schar>, vllong>();
    binary_op_return_type<vllong, vi64<uchar>, vllong>();
    binary_op_return_type<vllong, vi64<short>, vllong>();
    binary_op_return_type<vllong, vi64<ushort>, vllong>();
    binary_op_return_type<vllong, vi64<int>, vllong>();
    binary_op_return_type<vllong, vi64<uint>, Vc::native_datapar<decltype(llong() + uint())>>();
    binary_op_return_type<vllong, vi64<long>, vllong>();
    binary_op_return_type<vllong, vi64<ulong>, Vc::native_datapar<decltype(llong() + ulong())>>();
    binary_op_return_type<vllong, vi64<llong>, vllong>();
    binary_op_return_type<vllong, vi64<ullong>, vullong>();
    binary_op_return_type<vllong, vi64<float>, vi64<float>>();
    binary_op_return_type<vllong, vi64<double>, vi64<double>>();

    binary_op_return_type<vi32<llong>, schar, vi32<llong>>();
    binary_op_return_type<vi32<llong>, uchar, vi32<llong>>();
    binary_op_return_type<vi32<llong>, short, vi32<llong>>();
    binary_op_return_type<vi32<llong>, ushort, vi32<llong>>();
    binary_op_return_type<vi32<llong>, int, vi32<llong>>();
    binary_op_return_type<vi32<llong>, uint, vi32<decltype(llong() + uint())>>();
    binary_op_return_type<vi32<llong>, long, vi32<llong>>();
    binary_op_return_type<vi32<llong>, ulong, vi32<decltype(llong() + ulong())>>();
    binary_op_return_type<vi32<llong>, llong, vi32<llong>>();
    binary_op_return_type<vi32<llong>, ullong, vi32<ullong>>();
    binary_op_return_type<vi32<llong>, float, vi32<float>>();
    binary_op_return_type<vi32<llong>, double, vi32<double>>();
    binary_op_return_type<vi32<llong>, vi32<schar>, vi32<llong>>();
    binary_op_return_type<vi32<llong>, vi32<uchar>, vi32<llong>>();
    binary_op_return_type<vi32<llong>, vi32<short>, vi32<llong>>();
    binary_op_return_type<vi32<llong>, vi32<ushort>, vi32<llong>>();
    binary_op_return_type<vi32<llong>, vi32<int>, vi32<llong>>();
    binary_op_return_type<vi32<llong>, vi32<uint>, vi32<decltype(llong() + uint())>>();
    binary_op_return_type<vi32<llong>, vi32<long>, vi32<llong>>();
    binary_op_return_type<vi32<llong>, vi32<ulong>, vi32<decltype(llong() + ulong())>>();
    binary_op_return_type<vi32<llong>, vi32<llong>, vi32<llong>>();
    binary_op_return_type<vi32<llong>, vi32<ullong>, vi32<ullong>>();
    binary_op_return_type<vi32<llong>, vi32<float>, vi32<float>>();
    binary_op_return_type<vi32<llong>, vi32<double>, vi32<double>>();

    binary_op_return_type<vi64<llong>, vllong, vllong>();
    binary_op_return_type<vi64<llong>, vullong, vullong>();
    binary_op_return_type<vi64<llong>, schar, vi64<llong>>();
    binary_op_return_type<vi64<llong>, uchar, vi64<llong>>();
    binary_op_return_type<vi64<llong>, short, vi64<llong>>();
    binary_op_return_type<vi64<llong>, ushort, vi64<llong>>();
    binary_op_return_type<vi64<llong>, int, vi64<llong>>();
    binary_op_return_type<vi64<llong>, uint, vi64<decltype(llong() + uint())>>();
    binary_op_return_type<vi64<llong>, long, vi64<llong>>();
    binary_op_return_type<vi64<llong>, ulong, vi64<decltype(llong() + ulong())>>();
    binary_op_return_type<vi64<llong>, llong, vi64<llong>>();
    binary_op_return_type<vi64<llong>, ullong, vi64<ullong>>();
    binary_op_return_type<vi64<llong>, float, vi64<float>>();
    binary_op_return_type<vi64<llong>, double, vi64<double>>();
    binary_op_return_type<vi64<llong>, vi64<schar>, vi64<llong>>();
    binary_op_return_type<vi64<llong>, vi64<uchar>, vi64<llong>>();
    binary_op_return_type<vi64<llong>, vi64<short>, vi64<llong>>();
    binary_op_return_type<vi64<llong>, vi64<ushort>, vi64<llong>>();
    binary_op_return_type<vi64<llong>, vi64<int>, vi64<llong>>();
    binary_op_return_type<vi64<llong>, vi64<uint>, vi64<decltype(llong() + uint())>>();
    binary_op_return_type<vi64<llong>, vi64<long>, vi64<llong>>();
    binary_op_return_type<vi64<llong>, vi64<ulong>, vi64<decltype(llong() + ulong())>>();
    binary_op_return_type<vi64<llong>, vi64<llong>, vi64<llong>>();
    binary_op_return_type<vi64<llong>, vi64<ullong>, vi64<ullong>>();
    binary_op_return_type<vi64<llong>, vi64<float>, vi64<float>>();
    binary_op_return_type<vi64<llong>, vi64<double>, vi64<double>>();

    // ullong{{{2
    binary_op_return_type<vullong, vllong, vullong>();
    binary_op_return_type<vullong, vullong, vullong>();
    binary_op_return_type<vullong, schar, vullong>();
    binary_op_return_type<vullong, uchar, vullong>();
    binary_op_return_type<vullong, short, vullong>();
    binary_op_return_type<vullong, ushort, vullong>();
    binary_op_return_type<vullong, int, vullong>();
    binary_op_return_type<vullong, uint, vullong>();
    binary_op_return_type<vullong, long, vullong>();
    binary_op_return_type<vullong, ulong, vullong>();
    binary_op_return_type<vullong, llong, vullong>();
    binary_op_return_type<vullong, ullong, vullong>();
    binary_op_return_type<vullong, float, vi64<float>>();
    binary_op_return_type<vullong, double, vi64<double>>();
    binary_op_return_type<vullong, vi64<schar>, vullong>();
    binary_op_return_type<vullong, vi64<uchar>, vullong>();
    binary_op_return_type<vullong, vi64<short>, vullong>();
    binary_op_return_type<vullong, vi64<ushort>, vullong>();
    binary_op_return_type<vullong, vi64<int>, vullong>();
    binary_op_return_type<vullong, vi64<uint>, vullong>();
    binary_op_return_type<vullong, vi64<long>, vullong>();
    binary_op_return_type<vullong, vi64<ulong>, vullong>();
    binary_op_return_type<vullong, vi64<llong>, vullong>();
    binary_op_return_type<vullong, vi64<ullong>, vullong>();
    binary_op_return_type<vullong, vi64<float>, vi64<float>>();
    binary_op_return_type<vullong, vi64<double>, vi64<double>>();

    binary_op_return_type<vi32<ullong>, schar, vi32<ullong>>();
    binary_op_return_type<vi32<ullong>, uchar, vi32<ullong>>();
    binary_op_return_type<vi32<ullong>, short, vi32<ullong>>();
    binary_op_return_type<vi32<ullong>, ushort, vi32<ullong>>();
    binary_op_return_type<vi32<ullong>, int, vi32<ullong>>();
    binary_op_return_type<vi32<ullong>, uint, vi32<ullong>>();
    binary_op_return_type<vi32<ullong>, long, vi32<ullong>>();
    binary_op_return_type<vi32<ullong>, ulong, vi32<ullong>>();
    binary_op_return_type<vi32<ullong>, llong, vi32<ullong>>();
    binary_op_return_type<vi32<ullong>, ullong, vi32<ullong>>();
    binary_op_return_type<vi32<ullong>, float, vi32<float>>();
    binary_op_return_type<vi32<ullong>, double, vi32<double>>();
    binary_op_return_type<vi32<ullong>, vi32<schar>, vi32<ullong>>();
    binary_op_return_type<vi32<ullong>, vi32<uchar>, vi32<ullong>>();
    binary_op_return_type<vi32<ullong>, vi32<short>, vi32<ullong>>();
    binary_op_return_type<vi32<ullong>, vi32<ushort>, vi32<ullong>>();
    binary_op_return_type<vi32<ullong>, vi32<int>, vi32<ullong>>();
    binary_op_return_type<vi32<ullong>, vi32<uint>, vi32<ullong>>();
    binary_op_return_type<vi32<ullong>, vi32<long>, vi32<ullong>>();
    binary_op_return_type<vi32<ullong>, vi32<ulong>, vi32<ullong>>();
    binary_op_return_type<vi32<ullong>, vi32<llong>, vi32<ullong>>();
    binary_op_return_type<vi32<ullong>, vi32<ullong>, vi32<ullong>>();
    binary_op_return_type<vi32<ullong>, vi32<float>, vi32<float>>();
    binary_op_return_type<vi32<ullong>, vi32<double>, vi32<double>>();

    binary_op_return_type<vi64<ullong>, vllong, vullong>();
    binary_op_return_type<vi64<ullong>, vullong, vullong>();
    binary_op_return_type<vi64<ullong>, schar, vi64<ullong>>();
    binary_op_return_type<vi64<ullong>, uchar, vi64<ullong>>();
    binary_op_return_type<vi64<ullong>, short, vi64<ullong>>();
    binary_op_return_type<vi64<ullong>, ushort, vi64<ullong>>();
    binary_op_return_type<vi64<ullong>, int, vi64<ullong>>();
    binary_op_return_type<vi64<ullong>, uint, vi64<ullong>>();
    binary_op_return_type<vi64<ullong>, long, vi64<ullong>>();
    binary_op_return_type<vi64<ullong>, ulong, vi64<ullong>>();
    binary_op_return_type<vi64<ullong>, llong, vi64<ullong>>();
    binary_op_return_type<vi64<ullong>, ullong, vi64<ullong>>();
    binary_op_return_type<vi64<ullong>, float, vi64<float>>();
    binary_op_return_type<vi64<ullong>, double, vi64<double>>();
    binary_op_return_type<vi64<ullong>, vi64<schar>, vi64<ullong>>();
    binary_op_return_type<vi64<ullong>, vi64<uchar>, vi64<ullong>>();
    binary_op_return_type<vi64<ullong>, vi64<short>, vi64<ullong>>();
    binary_op_return_type<vi64<ullong>, vi64<ushort>, vi64<ullong>>();
    binary_op_return_type<vi64<ullong>, vi64<int>, vi64<ullong>>();
    binary_op_return_type<vi64<ullong>, vi64<uint>, vi64<ullong>>();
    binary_op_return_type<vi64<ullong>, vi64<long>, vi64<ullong>>();
    binary_op_return_type<vi64<ullong>, vi64<ulong>, vi64<ullong>>();
    binary_op_return_type<vi64<ullong>, vi64<llong>, vi64<ullong>>();
    binary_op_return_type<vi64<ullong>, vi64<ullong>, vi64<ullong>>();
    binary_op_return_type<vi64<ullong>, vi64<float>, vi64<float>>();
    binary_op_return_type<vi64<ullong>, vi64<double>, vi64<double>>();

    // int{{{2
    binary_op_return_type<vint, vint, vint>();
    binary_op_return_type<vint, vuint, vuint>();
    binary_op_return_type<vint, schar, vint>();
    binary_op_return_type<vint, uchar, vint>();
    binary_op_return_type<vint, short, vint>();
    binary_op_return_type<vint, ushort, vint>();
    binary_op_return_type<vint, int, vint>();
    binary_op_return_type<vint, uint, vuint>();
    binary_op_return_type<vint, long, vi32<long>>();
    binary_op_return_type<vint, ulong, vi32<ulong>>();
    binary_op_return_type<vint, llong, vi32<llong>>();
    binary_op_return_type<vint, ullong, vi32<ullong>>();
    binary_op_return_type<vint, float, vi32<float>>();
    binary_op_return_type<vint, double, vi32<double>>();
    binary_op_return_type<vint, vi32<schar>, vint>();
    binary_op_return_type<vint, vi32<uchar>, vint>();
    binary_op_return_type<vint, vi32<short>, vint>();
    binary_op_return_type<vint, vi32<ushort>, vint>();
    binary_op_return_type<vint, vi32<int>, vint>();
    binary_op_return_type<vint, vi32<uint>, vuint>();
    binary_op_return_type<vint, vi32<long>, vi32<long>>();
    binary_op_return_type<vint, vi32<ulong>, vi32<ulong>>();
    binary_op_return_type<vint, vi32<llong>, vi32<llong>>();
    binary_op_return_type<vint, vi32<ullong>, vi32<ullong>>();
    binary_op_return_type<vint, vi32<float>, vi32<float>>();
    binary_op_return_type<vint, vi32<double>, vi32<double>>();

    binary_op_return_type<vi32<int>, vint, vint>();
    binary_op_return_type<vi32<int>, vuint, vuint>();
    binary_op_return_type<vi32<int>, schar, vi32<int>>();
    binary_op_return_type<vi32<int>, uchar, vi32<int>>();
    binary_op_return_type<vi32<int>, short, vi32<int>>();
    binary_op_return_type<vi32<int>, ushort, vi32<int>>();
    binary_op_return_type<vi32<int>, int, vi32<int>>();
    binary_op_return_type<vi32<int>, uint, vi32<uint>>();
    binary_op_return_type<vi32<int>, long, vi32<long>>();
    binary_op_return_type<vi32<int>, ulong, vi32<ulong>>();
    binary_op_return_type<vi32<int>, llong, vi32<llong>>();
    binary_op_return_type<vi32<int>, ullong, vi32<ullong>>();
    binary_op_return_type<vi32<int>, float, vi32<float>>();
    binary_op_return_type<vi32<int>, double, vi32<double>>();
    binary_op_return_type<vi32<int>, vi32<schar>, vi32<int>>();
    binary_op_return_type<vi32<int>, vi32<uchar>, vi32<int>>();
    binary_op_return_type<vi32<int>, vi32<short>, vi32<int>>();
    binary_op_return_type<vi32<int>, vi32<ushort>, vi32<int>>();
    binary_op_return_type<vi32<int>, vi32<int>, vi32<int>>();
    binary_op_return_type<vi32<int>, vi32<uint>, vi32<uint>>();
    binary_op_return_type<vi32<int>, vi32<long>, vi32<long>>();
    binary_op_return_type<vi32<int>, vi32<ulong>, vi32<ulong>>();
    binary_op_return_type<vi32<int>, vi32<llong>, vi32<llong>>();
    binary_op_return_type<vi32<int>, vi32<ullong>, vi32<ullong>>();
    binary_op_return_type<vi32<int>, vi32<float>, vi32<float>>();
    binary_op_return_type<vi32<int>, vi32<double>, vi32<double>>();

    // uint{{{2
    binary_op_return_type<vuint, vint, vuint>();
    binary_op_return_type<vuint, vuint, vuint>();
    binary_op_return_type<vuint, schar, vuint>();
    binary_op_return_type<vuint, uchar, vuint>();
    binary_op_return_type<vuint, short, vuint>();
    binary_op_return_type<vuint, ushort, vuint>();
    binary_op_return_type<vuint, int, vuint>();
    binary_op_return_type<vuint, uint, vuint>();
    binary_op_return_type<vuint, long, vi32<decltype(uint() + long())>>();
    binary_op_return_type<vuint, ulong, vi32<ulong>>();
    binary_op_return_type<vuint, llong, vi32<llong>>();
    binary_op_return_type<vuint, ullong, vi32<ullong>>();
    binary_op_return_type<vuint, float, vi32<float>>();
    binary_op_return_type<vuint, double, vi32<double>>();
    binary_op_return_type<vuint, vi32<schar>, vuint>();
    binary_op_return_type<vuint, vi32<uchar>, vuint>();
    binary_op_return_type<vuint, vi32<short>, vuint>();
    binary_op_return_type<vuint, vi32<ushort>, vuint>();
    binary_op_return_type<vuint, vi32<int>, vuint>();
    binary_op_return_type<vuint, vi32<uint>, vuint>();
    binary_op_return_type<vuint, vi32<long>, vi32<decltype(uint() + long())>>();
    binary_op_return_type<vuint, vi32<ulong>, vi32<ulong>>();
    binary_op_return_type<vuint, vi32<llong>, vi32<llong>>();
    binary_op_return_type<vuint, vi32<ullong>, vi32<ullong>>();
    binary_op_return_type<vuint, vi32<float>, vi32<float>>();
    binary_op_return_type<vuint, vi32<double>, vi32<double>>();

    binary_op_return_type<vi32<uint>, vint, vuint>();
    binary_op_return_type<vi32<uint>, vuint, vuint>();
    binary_op_return_type<vi32<uint>, schar, vi32<uint>>();
    binary_op_return_type<vi32<uint>, uchar, vi32<uint>>();
    binary_op_return_type<vi32<uint>, short, vi32<uint>>();
    binary_op_return_type<vi32<uint>, ushort, vi32<uint>>();
    binary_op_return_type<vi32<uint>, int, vi32<uint>>();
    binary_op_return_type<vi32<uint>, uint, vi32<uint>>();
    binary_op_return_type<vi32<uint>, long, vi32<decltype(uint() + long())>>();
    binary_op_return_type<vi32<uint>, ulong, vi32<ulong>>();
    binary_op_return_type<vi32<uint>, llong, vi32<llong>>();
    binary_op_return_type<vi32<uint>, ullong, vi32<ullong>>();
    binary_op_return_type<vi32<uint>, float, vi32<float>>();
    binary_op_return_type<vi32<uint>, double, vi32<double>>();
    binary_op_return_type<vi32<uint>, vi32<schar>, vi32<uint>>();
    binary_op_return_type<vi32<uint>, vi32<uchar>, vi32<uint>>();
    binary_op_return_type<vi32<uint>, vi32<short>, vi32<uint>>();
    binary_op_return_type<vi32<uint>, vi32<ushort>, vi32<uint>>();
    binary_op_return_type<vi32<uint>, vi32<int>, vi32<uint>>();
    binary_op_return_type<vi32<uint>, vi32<uint>, vi32<uint>>();
    binary_op_return_type<vi32<uint>, vi32<long>, vi32<decltype(uint() + long())>>();
    binary_op_return_type<vi32<uint>, vi32<ulong>, vi32<ulong>>();
    binary_op_return_type<vi32<uint>, vi32<llong>, vi32<llong>>();
    binary_op_return_type<vi32<uint>, vi32<ullong>, vi32<ullong>>();
    binary_op_return_type<vi32<uint>, vi32<float>, vi32<float>>();
    binary_op_return_type<vi32<uint>, vi32<double>, vi32<double>>();

    // short{{{2
    binary_op_return_type<vshort, vshort, vshort>();
    binary_op_return_type<vshort, vushort, vushort>();
    binary_op_return_type<vshort, schar, vshort>();
    binary_op_return_type<vshort, uchar, vshort>();
    binary_op_return_type<vshort, short, vshort>();
    binary_op_return_type<vshort, ushort, vushort>();
    binary_op_return_type<vshort, int, vshort>();
    binary_op_return_type<vshort, uint, vushort>();
    binary_op_return_type<vshort, long, vi16<long>>();
    binary_op_return_type<vshort, ulong, vi16<ulong>>();
    binary_op_return_type<vshort, llong, vi16<llong>>();
    binary_op_return_type<vshort, ullong, vi16<ullong>>();
    binary_op_return_type<vshort, float, vi16<float>>();
    binary_op_return_type<vshort, double, vi16<double>>();
    binary_op_return_type<vshort, vi16<schar>, vshort>();
    binary_op_return_type<vshort, vi16<uchar>, vshort>();
    binary_op_return_type<vshort, vi16<short>, vshort>();
    binary_op_return_type<vshort, vi16<ushort>, vushort>();
    binary_op_return_type<vshort, vi16<int>, vi16<int>>();
    binary_op_return_type<vshort, vi16<uint>, vi16<uint>>();
    binary_op_return_type<vshort, vi16<long>, vi16<long>>();
    binary_op_return_type<vshort, vi16<ulong>, vi16<ulong>>();
    binary_op_return_type<vshort, vi16<llong>, vi16<llong>>();
    binary_op_return_type<vshort, vi16<ullong>, vi16<ullong>>();
    binary_op_return_type<vshort, vi16<float>, vi16<float>>();
    binary_op_return_type<vshort, vi16<double>, vi16<double>>();

    binary_op_return_type<vi16<short>, vshort, vshort>();
    binary_op_return_type<vi16<short>, vushort, vushort>();
    binary_op_return_type<vi16<short>, schar, vi16<short>>();
    binary_op_return_type<vi16<short>, uchar, vi16<short>>();
    binary_op_return_type<vi16<short>, short, vi16<short>>();
    binary_op_return_type<vi16<short>, ushort, vi16<ushort>>();
    binary_op_return_type<vi16<short>, int, vi16<short>>();
    binary_op_return_type<vi16<short>, uint, vi16<ushort>>();
    binary_op_return_type<vi16<short>, long, vi16<long>>();
    binary_op_return_type<vi16<short>, ulong, vi16<ulong>>();
    binary_op_return_type<vi16<short>, llong, vi16<llong>>();
    binary_op_return_type<vi16<short>, ullong, vi16<ullong>>();
    binary_op_return_type<vi16<short>, float, vi16<float>>();
    binary_op_return_type<vi16<short>, double, vi16<double>>();
    binary_op_return_type<vi16<short>, vi16<schar>, vi16<short>>();
    binary_op_return_type<vi16<short>, vi16<uchar>, vi16<short>>();
    binary_op_return_type<vi16<short>, vi16<short>, vi16<short>>();
    binary_op_return_type<vi16<short>, vi16<ushort>, vi16<ushort>>();
    binary_op_return_type<vi16<short>, vi16<int>, vi16<int>>();
    binary_op_return_type<vi16<short>, vi16<uint>, vi16<uint>>();
    binary_op_return_type<vi16<short>, vi16<long>, vi16<long>>();
    binary_op_return_type<vi16<short>, vi16<ulong>, vi16<ulong>>();
    binary_op_return_type<vi16<short>, vi16<llong>, vi16<llong>>();
    binary_op_return_type<vi16<short>, vi16<ullong>, vi16<ullong>>();
    binary_op_return_type<vi16<short>, vi16<float>, vi16<float>>();
    binary_op_return_type<vi16<short>, vi16<double>, vi16<double>>();

    // ushort{{{2
    binary_op_return_type<vushort, vshort, vushort>();
    binary_op_return_type<vushort, vushort, vushort>();
    binary_op_return_type<vushort, schar, vushort>();
    binary_op_return_type<vushort, uchar, vushort>();
    binary_op_return_type<vushort, short, vushort>();
    binary_op_return_type<vushort, ushort, vushort>();
    binary_op_return_type<vushort, int, vushort>();
    binary_op_return_type<vushort, uint, vushort>();
    binary_op_return_type<vushort, long, vi16<long>>();
    binary_op_return_type<vushort, ulong, vi16<ulong>>();
    binary_op_return_type<vushort, llong, vi16<llong>>();
    binary_op_return_type<vushort, ullong, vi16<ullong>>();
    binary_op_return_type<vushort, float, vi16<float>>();
    binary_op_return_type<vushort, double, vi16<double>>();
    binary_op_return_type<vushort, vi16<schar>, vushort>();
    binary_op_return_type<vushort, vi16<uchar>, vushort>();
    binary_op_return_type<vushort, vi16<short>, vushort>();
    binary_op_return_type<vushort, vi16<ushort>, vushort>();
    binary_op_return_type<vushort, vi16<int>, vi16<int>>();
    binary_op_return_type<vushort, vi16<uint>, vi16<uint>>();
    binary_op_return_type<vushort, vi16<long>, vi16<long>>();
    binary_op_return_type<vushort, vi16<ulong>, vi16<ulong>>();
    binary_op_return_type<vushort, vi16<llong>, vi16<llong>>();
    binary_op_return_type<vushort, vi16<ullong>, vi16<ullong>>();
    binary_op_return_type<vushort, vi16<float>, vi16<float>>();
    binary_op_return_type<vushort, vi16<double>, vi16<double>>();

    binary_op_return_type<vi16<ushort>, vshort, vushort>();
    binary_op_return_type<vi16<ushort>, vushort, vushort>();
    binary_op_return_type<vi16<ushort>, schar, vi16<ushort>>();
    binary_op_return_type<vi16<ushort>, uchar, vi16<ushort>>();
    binary_op_return_type<vi16<ushort>, short, vi16<ushort>>();
    binary_op_return_type<vi16<ushort>, ushort, vi16<ushort>>();
    binary_op_return_type<vi16<ushort>, int, vi16<ushort>>();
    binary_op_return_type<vi16<ushort>, uint, vi16<ushort>>();
    binary_op_return_type<vi16<ushort>, long, vi16<long>>();
    binary_op_return_type<vi16<ushort>, ulong, vi16<ulong>>();
    binary_op_return_type<vi16<ushort>, llong, vi16<llong>>();
    binary_op_return_type<vi16<ushort>, ullong, vi16<ullong>>();
    binary_op_return_type<vi16<ushort>, float, vi16<float>>();
    binary_op_return_type<vi16<ushort>, double, vi16<double>>();
    binary_op_return_type<vi16<ushort>, vi16<schar>, vi16<ushort>>();
    binary_op_return_type<vi16<ushort>, vi16<uchar>, vi16<ushort>>();
    binary_op_return_type<vi16<ushort>, vi16<short>, vi16<ushort>>();
    binary_op_return_type<vi16<ushort>, vi16<ushort>, vi16<ushort>>();
    binary_op_return_type<vi16<ushort>, vi16<int>, vi16<int>>();
    binary_op_return_type<vi16<ushort>, vi16<uint>, vi16<uint>>();
    binary_op_return_type<vi16<ushort>, vi16<long>, vi16<long>>();
    binary_op_return_type<vi16<ushort>, vi16<ulong>, vi16<ulong>>();
    binary_op_return_type<vi16<ushort>, vi16<llong>, vi16<llong>>();
    binary_op_return_type<vi16<ushort>, vi16<ullong>, vi16<ullong>>();
    binary_op_return_type<vi16<ushort>, vi16<float>, vi16<float>>();
    binary_op_return_type<vi16<ushort>, vi16<double>, vi16<double>>();

    // schar{{{2
    binary_op_return_type<vschar, vschar, vschar>();
    binary_op_return_type<vschar, vuchar, vuchar>();
    binary_op_return_type<vschar, schar, vschar>();
    binary_op_return_type<vschar, uchar, vuchar>();
    // the following 4 are possibly surprising:
    binary_op_return_type<vschar, short, vi8<short>>();
    binary_op_return_type<vschar, ushort, vi8<ushort>>();
    binary_op_return_type<vschar, int, vschar>();
    binary_op_return_type<vschar, uint, vuchar>();
    binary_op_return_type<vschar, long, vi8<long>>();
    binary_op_return_type<vschar, ulong, vi8<ulong>>();
    binary_op_return_type<vschar, llong, vi8<llong>>();
    binary_op_return_type<vschar, ullong, vi8<ullong>>();
    binary_op_return_type<vschar, float, vi8<float>>();
    binary_op_return_type<vschar, double, vi8<double>>();
    binary_op_return_type<vschar, vi8<schar>, vschar>();
    binary_op_return_type<vschar, vi8<uchar>, vuchar>();
    binary_op_return_type<vschar, vi8<short>, vi8<short>>();
    binary_op_return_type<vschar, vi8<ushort>, vi8<ushort>>();
    binary_op_return_type<vschar, vi8<int>, vi8<int>>();
    binary_op_return_type<vschar, vi8<uint>, vi8<uint>>();
    binary_op_return_type<vschar, vi8<long>, vi8<long>>();
    binary_op_return_type<vschar, vi8<ulong>, vi8<ulong>>();
    binary_op_return_type<vschar, vi8<llong>, vi8<llong>>();
    binary_op_return_type<vschar, vi8<ullong>, vi8<ullong>>();
    binary_op_return_type<vschar, vi8<float>, vi8<float>>();
    binary_op_return_type<vschar, vi8<double>, vi8<double>>();

    binary_op_return_type<vi8<schar>, vschar, vschar>();
    binary_op_return_type<vi8<schar>, vuchar, vuchar>();
    binary_op_return_type<vi8<schar>, schar, vi8<schar>>();
    binary_op_return_type<vi8<schar>, uchar, vi8<uchar>>();
    // the following 4 are possibly surprising:
    binary_op_return_type<vi8<schar>, short, vi8<short>>();
    binary_op_return_type<vi8<schar>, ushort, vi8<ushort>>();
    binary_op_return_type<vi8<schar>, int, vi8<schar>>();
    binary_op_return_type<vi8<schar>, uint, vi8<uchar>>();
    binary_op_return_type<vi8<schar>, long, vi8<long>>();
    binary_op_return_type<vi8<schar>, ulong, vi8<ulong>>();
    binary_op_return_type<vi8<schar>, llong, vi8<llong>>();
    binary_op_return_type<vi8<schar>, ullong, vi8<ullong>>();
    binary_op_return_type<vi8<schar>, float, vi8<float>>();
    binary_op_return_type<vi8<schar>, double, vi8<double>>();
    binary_op_return_type<vi8<schar>, vi8<schar>, vi8<schar>>();
    binary_op_return_type<vi8<schar>, vi8<uchar>, vi8<uchar>>();
    binary_op_return_type<vi8<schar>, vi8<short>, vi8<short>>();
    binary_op_return_type<vi8<schar>, vi8<ushort>, vi8<ushort>>();
    binary_op_return_type<vi8<schar>, vi8<int>, vi8<int>>();
    binary_op_return_type<vi8<schar>, vi8<uint>, vi8<uint>>();
    binary_op_return_type<vi8<schar>, vi8<long>, vi8<long>>();
    binary_op_return_type<vi8<schar>, vi8<ulong>, vi8<ulong>>();
    binary_op_return_type<vi8<schar>, vi8<llong>, vi8<llong>>();
    binary_op_return_type<vi8<schar>, vi8<ullong>, vi8<ullong>>();
    binary_op_return_type<vi8<schar>, vi8<float>, vi8<float>>();
    binary_op_return_type<vi8<schar>, vi8<double>, vi8<double>>();

    // uchar{{{2
    binary_op_return_type<vuchar, vschar, vuchar>();
    binary_op_return_type<vuchar, vuchar, vuchar>();
    binary_op_return_type<vuchar, schar, vuchar>();
    binary_op_return_type<vuchar, uchar, vuchar>();
    // the following 4 are possibly surprising:
    binary_op_return_type<vuchar, short, vi8<short>>();
    binary_op_return_type<vuchar, ushort, vi8<ushort>>();
    binary_op_return_type<vuchar, int, vuchar>();
    binary_op_return_type<vuchar, uint, vuchar>();
    binary_op_return_type<vuchar, long, vi8<long>>();
    binary_op_return_type<vuchar, ulong, vi8<ulong>>();
    binary_op_return_type<vuchar, llong, vi8<llong>>();
    binary_op_return_type<vuchar, ullong, vi8<ullong>>();
    binary_op_return_type<vuchar, float, vi8<float>>();
    binary_op_return_type<vuchar, double, vi8<double>>();
    binary_op_return_type<vuchar, vi8<schar>, vuchar>();
    binary_op_return_type<vuchar, vi8<uchar>, vuchar>();
    binary_op_return_type<vuchar, vi8<short>, vi8<short>>();
    binary_op_return_type<vuchar, vi8<ushort>, vi8<ushort>>();
    binary_op_return_type<vuchar, vi8<int>, vi8<int>>();
    binary_op_return_type<vuchar, vi8<uint>, vi8<uint>>();
    binary_op_return_type<vuchar, vi8<long>, vi8<long>>();
    binary_op_return_type<vuchar, vi8<ulong>, vi8<ulong>>();
    binary_op_return_type<vuchar, vi8<llong>, vi8<llong>>();
    binary_op_return_type<vuchar, vi8<ullong>, vi8<ullong>>();
    binary_op_return_type<vuchar, vi8<float>, vi8<float>>();
    binary_op_return_type<vuchar, vi8<double>, vi8<double>>();

    binary_op_return_type<vi8<uchar>, vschar, vuchar>();
    binary_op_return_type<vi8<uchar>, vuchar, vuchar>();
    binary_op_return_type<vi8<uchar>, schar, vi8<uchar>>();
    binary_op_return_type<vi8<uchar>, uchar, vi8<uchar>>();
    // the following 4 are possibly surprising:
    binary_op_return_type<vi8<uchar>, short, vi8<short>>();
    binary_op_return_type<vi8<uchar>, ushort, vi8<ushort>>();
    binary_op_return_type<vi8<uchar>, int, vi8<uchar>>();
    binary_op_return_type<vi8<uchar>, uint, vi8<uchar>>();
    binary_op_return_type<vi8<uchar>, long, vi8<long>>();
    binary_op_return_type<vi8<uchar>, ulong, vi8<ulong>>();
    binary_op_return_type<vi8<uchar>, llong, vi8<llong>>();
    binary_op_return_type<vi8<uchar>, ullong, vi8<ullong>>();
    binary_op_return_type<vi8<uchar>, float, vi8<float>>();
    binary_op_return_type<vi8<uchar>, double, vi8<double>>();
    binary_op_return_type<vi8<uchar>, vi8<schar>, vi8<uchar>>();
    binary_op_return_type<vi8<uchar>, vi8<uchar>, vi8<uchar>>();
    binary_op_return_type<vi8<uchar>, vi8<short>, vi8<short>>();
    binary_op_return_type<vi8<uchar>, vi8<ushort>, vi8<ushort>>();
    binary_op_return_type<vi8<uchar>, vi8<int>, vi8<int>>();
    binary_op_return_type<vi8<uchar>, vi8<uint>, vi8<uint>>();
    binary_op_return_type<vi8<uchar>, vi8<long>, vi8<long>>();
    binary_op_return_type<vi8<uchar>, vi8<ulong>, vi8<ulong>>();
    binary_op_return_type<vi8<uchar>, vi8<llong>, vi8<llong>>();
    binary_op_return_type<vi8<uchar>, vi8<ullong>, vi8<ullong>>();
    binary_op_return_type<vi8<uchar>, vi8<float>, vi8<float>>();
    binary_op_return_type<vi8<uchar>, vi8<double>, vi8<double>>();

    // misc{{{2
    binary_op_return_type<int, vi32<long>, vi32<long>>();
    binary_op_return_type<int, vi32<ulong>, vi32<ulong>>();
    binary_op_return_type<int, vi32<llong>, vi32<llong>>();
    binary_op_return_type<int, vi32<ullong>, vi32<ullong>>();
    binary_op_return_type<int, vi32<ushort>, vi32<ushort>>();
    binary_op_return_type<int, vi32<schar>, vi32<schar>>();
    binary_op_return_type<int, vi32<uchar>, vi32<uchar>>();
}

// vim: foldmethod=marker
