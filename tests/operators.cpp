/*  This file is part of the Vc library. {{{
Copyright © 2009-2017 Matthias Kretz <kretz@kde.org>

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
#include <vir/test.h>
#include <Vc/datapar>
#include "make_vec.h"
#include "metahelpers.h"

template <class... Ts> using base_template = Vc::datapar<Ts...>;
#include "testtypes.h"

//operators helpers  //{{{1
template <class T> constexpr T genHalfBits()
{
    return std::numeric_limits<T>::max() >> (std::numeric_limits<T>::digits / 2);
}
template <> constexpr long double genHalfBits<long double>() { return 0; }
template <> constexpr double genHalfBits<double>() { return 0; }
template <> constexpr float genHalfBits<float>() { return 0; }

// integral_operators {{{1
template <class V>
std::enable_if_t<std::is_integral<typename V::value_type>::value, void>
integral_operators()
{
    using T = typename V::value_type;
    {  // complement{{{2
        COMPARE(~V(), V(~T()));
        COMPARE(~V(~T()), V());
    }

    {  // modulus{{{2
        V x = make_vec<V>({3, 4}, 2);
        COMPARE(x % x, V(0));
        V y = x - 1;
        COMPARE(x % y, V(1));
        y = x + 1;
        COMPARE(x % y, x);
        if (std::is_signed<T>::value) {
            x = -x;
            COMPARE(x % y, x);
            x = -y;
            COMPARE(x % y, V(0));
            x = x - 1;
            COMPARE(x % y, V(-1));
            x %= y;
            COMPARE(x, V(-1));
        }
    }

    {  // bit_and{{{2
        V x = make_vec<V>({3, 4, 5}, 8);
        COMPARE(x & x, x);
        COMPARE(x & ~x, V());
        COMPARE(x & V(), V());
        COMPARE(V() & x, V());
        V y = make_vec<V>({1, 5, 3}, 8);
        COMPARE(x & y, make_vec<V>({1, 4, 1}, 8));
        x &= y;
        COMPARE(x, make_vec<V>({1, 4, 1}, 8));
    }

    {  // bit_or{{{2
        V x = make_vec<V>({3, 4, 5}, 8);
        COMPARE(x | x, x);
        COMPARE(x | ~x, ~V());
        COMPARE(x | V(), x);
        COMPARE(V() | x, x);
        V y = make_vec<V>({1, 5, 3}, 8);
        COMPARE(x | y, make_vec<V>({3, 5, 7}, 8));
        x |= y;
        COMPARE(x, make_vec<V>({3, 5, 7}, 8));
    }

    {  // bit_xor{{{2
        V x = make_vec<V>({3, 4, 5}, 8);
        COMPARE(x ^ x, V());
        COMPARE(x ^ ~x, ~V());
        COMPARE(x ^ V(), x);
        COMPARE(V() ^ x, x);
        V y = make_vec<V>({1, 5, 3}, 8);
        COMPARE(x ^ y, make_vec<V>({2, 1, 6}, 0));
        x ^= y;
        COMPARE(x, make_vec<V>({2, 1, 6}, 0));
    }

    {  // bit_shift_left{{{2
        COMPARE(V() << 1, V());
        // Note:
        // - negative RHS or RHS >= #bits is UB
        // - negative LHS is UB
        // - shifting into (or over) the sign bit is UB
        // - unsigned LHS overflow is modulo arithmetic
        constexpr int nbits(sizeof(T) * CHAR_BIT);
        {
            V seq = make_vec<V>({0, 1}, 2);
            seq %= nbits - 1;
            COMPARE(make_vec<V>({0, 1}, 0) << seq,
                    V([&](auto i) { return T(T(i & 1) << seq[i]); }));
            COMPARE(make_vec<V>({1, 0}, 0) << seq,
                    V([&](auto i) { return T(T(~i & 1) << seq[i]); }));
            COMPARE(V(1) << seq, V([&](auto i) { return T(T(1) << seq[i]); }));
        }
        for (int i = 0; i < nbits - 1; ++i) {
            COMPARE(V(1) << i, V(T(1) << i));
        }
        if (std::is_unsigned<T>::value) {
            constexpr int shift_count = nbits - 1;
            COMPARE(V(1) << shift_count, V(T(1) << shift_count));
            constexpr T max =  // avoid overflow warning in the last COMPARE
                std::is_unsigned<T>::value ? std::numeric_limits<T>::max() : T(1);
            COMPARE(V(max) << shift_count, V(max << shift_count)) << "shift_count: " << shift_count;
        }
    }

    {  // bit_shift_right{{{2
        constexpr int nbits(sizeof(T) * CHAR_BIT);
        // Note:
        // - negative LHS is implementation defined
        // - negative RHS or RHS >= #bits is UB
        // - no other UB
        COMPARE(V(~T()) >> V(0), V(~T()));
        for (int s = 1; s < nbits; ++s) {
            COMPARE(V(~T()) >> V(s), V(T(~T()) >> s)) << "s: " << s;
        }
        for (int s = 1; s < nbits; ++s) {
            COMPARE(V(~T(1)) >> V(s), V(T(~T(1)) >> s)) << "s: " << s;
        }
        COMPARE(V(0) >> V(1), V(0));
        COMPARE(V(1) >> V(1), V(0));
        COMPARE(V(2) >> V(1), V(1));
        COMPARE(V(3) >> V(1), V(1));
        COMPARE(V(7) >> V(2), V(1));
        {
            V seq = make_vec<V>({0, 1}, 2);
            seq %= nbits - 1;
            COMPARE(V(1) >> seq, V([&](auto i) { return T(T(1) >> seq[i]); }));
        }
    }

    //}}}2
}

template <class V>
std::enable_if_t<!std::is_integral<typename V::value_type>::value, void>
integral_operators()
{
}

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

    integral_operators<V>();
    COMPARE(!std::is_integral<T>::value, (is_substitution_failure<V, V, std::modulus<>>));
    COMPARE(!std::is_integral<T>::value, (is_substitution_failure<V, V, std::bit_and<>>));
    COMPARE(!std::is_integral<T>::value, (is_substitution_failure<V, V, std::bit_or<>>));
    COMPARE(!std::is_integral<T>::value, (is_substitution_failure<V, V, std::bit_xor<>>));
    COMPARE(!std::is_integral<T>::value, (is_substitution_failure<V, V, bit_shift_left>));
    COMPARE(!std::is_integral<T>::value, (is_substitution_failure<V, V, bit_shift_right>));

    COMPARE(!std::is_integral<T>::value, (is_substitution_failure<V &, V, assign_modulus>));
    COMPARE(!std::is_integral<T>::value, (is_substitution_failure<V &, V, assign_bit_and>));
    COMPARE(!std::is_integral<T>::value, (is_substitution_failure<V &, V, assign_bit_or>));
    COMPARE(!std::is_integral<T>::value, (is_substitution_failure<V &, V, assign_bit_xor>));
    COMPARE(!std::is_integral<T>::value, (is_substitution_failure<V &, V, assign_bit_shift_left>));
    COMPARE(!std::is_integral<T>::value, (is_substitution_failure<V &, V, assign_bit_shift_right>));
}


// vim: foldmethod=marker
