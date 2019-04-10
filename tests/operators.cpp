/*  This file is part of the Vc library. {{{
Copyright © 2009-2019 Matthias Kretz <kretz@kde.org>

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

//#define UNITTEST_ONLY_XTEST 1
#include "unittest.h"
#include "make_vec.h"
#include "metahelpers.h"

template <class... Ts> using base_template = std::experimental::simd<Ts...>;
#include "testtypes.h"

//operators helpers  //{{{1
template <class T> constexpr T genHalfBits()
{
    return std::numeric_limits<T>::max() >> (std::numeric_limits<T>::digits / 2);
}
template <> constexpr long double genHalfBits<long double>() { return 0; }
template <> constexpr double genHalfBits<double>() { return 0; }
template <> constexpr float genHalfBits<float>() { return 0; }

TEST_TYPES(V, operators, all_test_types)  //{{{1
{
    using M = typename V::mask_type;
    using T = typename V::value_type;
    constexpr auto min = std::numeric_limits<T>::min();
    constexpr auto max = std::numeric_limits<T>::max();
    {  // compares{{{2
        COMPARE(V(0) == make_vec<V>({0, 1}, 0), make_mask<M>({1, 0}));
        COMPARE(V(0) == make_vec<V>({0, 1, 2}, 0), make_mask<M>({1, 0, 0}));
        COMPARE(V(1) == make_vec<V>({0, 1, 2}, 0), make_mask<M>({0, 1, 0}));
        COMPARE(V(2) == make_vec<V>({0, 1, 2}, 0), make_mask<M>({0, 0, 1}));
        COMPARE(V(0) <  make_vec<V>({0, 1, 2}, 0), make_mask<M>({0, 1, 1}));

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
                    VERIFY(all_of(lo < hi)) << "hi: " << hi << ", lo: " << lo << ", lo < hi: " << (lo < hi);
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
        COMPARE(typeid(x[0] * x[0]), typeid(T() * T()));
        COMPARE(typeid(x[0] * T()), typeid(T() * T()));
        COMPARE(typeid(T() * x[0]), typeid(T() * T()));
        COMPARE(typeid(x * x[0]), typeid(x));
        COMPARE(typeid(x[0] * x), typeid(x));

        x = V([](auto i) -> T { return i; });
        for (std::size_t i = 0; i < V::size(); ++i) {
            COMPARE(x[i], T(i));
        }
        for (std::size_t i = 0; i + 1 < V::size(); i += 2) {
            using std::swap;
            swap(x[i], x[i + 1]);
        }
        for (std::size_t i = 0; i + 1 < V::size(); i += 2) {
            COMPARE(x[i], T(i + 1));
            COMPARE(x[i + 1], T(i));
        }
        x = 1;
        V y = 0;
        COMPARE(x[0], T(1));
        x[0] = y[0];  // make sure non-const smart_reference assignment works
        COMPARE(x[0], T(0));
        x = 1;
        x[0] = x[0];  // self-assignment on smart_reference
        COMPARE(x[0], T(1));

        std::experimental::simd<typename V::value_type, std::experimental::simd_abi::scalar> z = 2;
        x[0] = z[0];
        COMPARE(x[0], T(2));
        x = 3;
        z[0] = x[0];
        COMPARE(z[0], T(3));

        //TODO: check that only value-preserving conversions happen on subscript
        //assignment
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
        COMPARE(x += y, make_vec<V>({2, 3, 4, 5, 6, 7, 8}));
        COMPARE(x, make_vec<V>({2, 3, 4, 5, 6, 7, 8}));
        COMPARE(x += -y, V(1));
        COMPARE(x, V(1));
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
        COMPARE(y -= x, make_vec<V>({0, 1, 2, 3, 4, 5, 6}));
        COMPARE(y, make_vec<V>({0, 1, 2, 3, 4, 5, 6}));
        COMPARE(y -= y, V(0));
        COMPARE(y, V(0));
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
                // if T is of lower rank than int, `n * m` will promote to int before executing the
                // multiplication. In this case an overflow will be UB (and ubsan will
                // warn about it). The solution is to cast to uint in that case.
                using U = std::conditional_t<(sizeof(T) < sizeof(int)), unsigned, T>;
                COMPARE(x * y, V(T(U(n) * U(m))));
            }
        }
        x = 2;
        COMPARE(x *= make_vec<V>({1, 2, 3}), make_vec<V>({2, 4, 6}));
        COMPARE(x, make_vec<V>({2, 4, 6}));
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
        COMPARE(x /= y, ref);
        COMPARE(x, ref);
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


// vim: foldmethod=marker
