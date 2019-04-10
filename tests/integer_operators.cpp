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

// make_value_unknown {{{1
template <class T> T make_value_unknown(const T& x)
{
    if constexpr (std::is_constructible_v<T, const volatile T&>) {
        const volatile T& y = x;
        return y;
    } else {
        T y = x;
        asm("" : "+m"(y));
        return y;
    }
}

// for_constexpr {{{1
template <typename T, T Begin, T End, T Stride = 1, typename F>
void for_constexpr(F&& fun)
{
    if constexpr (Begin <= End) {
        fun(std::integral_constant<T, Begin>());
        if constexpr (Begin < End) {
            for_constexpr<T, Begin + Stride, End, Stride>(std::forward<F>(fun));
        }
    }
}

TEST_TYPES(V, operators, all_test_types)  //{{{1
{
    using T = typename V::value_type;
    if constexpr (std::is_integral_v<T>) {
        constexpr int nbits(sizeof(T) * CHAR_BIT);
        constexpr int n_promo_bits = std::max(nbits, int(sizeof(int) * CHAR_BIT));

        // complement{{{2
        COMPARE(~V(), V(~T()));
        COMPARE(~V(~T()), V());

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
            // Note:
            // - negative RHS or RHS >= max(#bits(T), #bits(int)) is UB
            // - negative LHS is UB
            // - shifting into (or over) the sign bit is UB
            // - unsigned LHS overflow is modulo arithmetic
            COMPARE(V() << 1, V());
            for (int i = 0; i < nbits - 1; ++i) {
                COMPARE(V(1) << i, V(T(1) << i)) << "i: " << i;
            }
            for_constexpr<int, 0, n_promo_bits - 1>([](auto shift_ic) {
                constexpr int shift = shift_ic;
                const V seq         = make_value_unknown(V([&](T i) {
                    if constexpr (std::is_signed_v<T>) {
                        const T max = std::numeric_limits<T>::max() >> shift;
                        return max == 0 ? 1 : (std::abs(max - i) % max) + 1;
                    } else {
                        return ~T() - i;
                    }
                }));
                const V ref([&](T i) { return T(seq[i] << shift); });
                COMPARE(seq << shift, ref) << "seq: " << seq << ", shift: " << shift;
                COMPARE(seq << make_value_unknown(shift), ref)
                    << "seq: " << seq << ", shift: " << shift;
            });
            {
                V seq = make_vec<V>({0, 1}, nbits - 2);
                seq %= nbits - 1;
                COMPARE(make_vec<V>({0, 1}, 0) << seq,
                        V([&](auto i) { return T(T(i & 1) << seq[i]); }))
                    << "seq = " << seq;
                COMPARE(make_vec<V>({1, 0}, 0) << seq,
                        V([&](auto i) { return T(T(~i & 1) << seq[i]); }));
                COMPARE(V(1) << seq, V([&](auto i) { return T(T(1) << seq[i]); }));
            }
            if (std::is_unsigned<T>::value) {
                constexpr int shift_count = nbits - 1;
                COMPARE(V(1) << shift_count, V(T(1) << shift_count));
                constexpr T max =  // avoid overflow warning in the last COMPARE
                    std::is_unsigned<T>::value ? std::numeric_limits<T>::max() : T(1);
                COMPARE(V(max) << shift_count, V(max << shift_count))
                    << "shift_count: " << shift_count;
            }
        }

        {  // bit_shift_right{{{2
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
            for (int j = 0; j < 100; ++j) {
                const V seq([&](auto i) -> T { return (j + i) % n_promo_bits; });
                COMPARE(V(1) >> seq, V([&](auto i) { return T(T(1) >> seq[i]); }))
                    << "seq = " << seq;
                COMPARE(make_value_unknown(V(1)) >> make_value_unknown(seq),
                        V([&](auto i) { return T(T(1) >> seq[i]); }))
                    << "seq = " << seq;
            }
            for_constexpr<int, 0, n_promo_bits - 1>([](auto shift_ic) {
                constexpr int shift = shift_ic;
                const V seq         = make_value_unknown(V([&](int i) {
                    using U = std::make_unsigned_t<T>;
                    return T(~U() >> (i % 32));
                }));
                const V ref([&](T i) { return T(seq[i] >> shift); });
                COMPARE(seq >> shift, ref) << "seq: " << seq << ", shift: " << shift;
                COMPARE(seq >> make_value_unknown(shift), ref)
                    << "seq: " << seq << ", shift: " << shift;
            });
        }

        //}}}2
    } else {
        VERIFY((is_substitution_failure<V, V, std::modulus<>>));
        VERIFY((is_substitution_failure<V, V, std::bit_and<>>));
        VERIFY((is_substitution_failure<V, V, std::bit_or<>>));
        VERIFY((is_substitution_failure<V, V, std::bit_xor<>>));
        VERIFY((is_substitution_failure<V, V, bit_shift_left>));
        VERIFY((is_substitution_failure<V, V, bit_shift_right>));

        VERIFY((is_substitution_failure<V&, V, assign_modulus>));
        VERIFY((is_substitution_failure<V&, V, assign_bit_and>));
        VERIFY((is_substitution_failure<V&, V, assign_bit_or>));
        VERIFY((is_substitution_failure<V&, V, assign_bit_xor>));
        VERIFY((is_substitution_failure<V&, V, assign_bit_shift_left>));
        VERIFY((is_substitution_failure<V&, V, assign_bit_shift_right>));
    }
}
// }}}1

// vim: foldmethod=marker
