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
template <class M> inline M make_mask(const std::initializer_list<bool> &init)
{
    std::size_t i = 0;
    M r;
    for (;;) {
        for (bool x : init) {
            r[i] = x;
            if (++i == M::size()) {
                return r;
            }
        }
    }
}

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
//}}}1

template <class V> struct Convertible {
    operator V() const { return V(4); }
};

TEST_TYPES(V, where, (all_test_types))
{
    using M = typename V::mask_type;
    using T = typename V::value_type;
    const V indexes = make_vec<V>({1, 2, 3, 4}, 4);
    const M alternating_mask = make_mask<M>({true, false});
    V x = 0;
    where(alternating_mask, x) = indexes;
    COMPARE(alternating_mask, x == indexes);

    where(!alternating_mask, x) = T(2);
    COMPARE(!alternating_mask, x == T(2));

    where(!alternating_mask, x) = Convertible<V>();
    COMPARE(!alternating_mask, x == T(4));

    x = 0;
    COMPARE(x, T(0));
    where(alternating_mask, x) += indexes;
    COMPARE(alternating_mask, x == indexes);

    x = 10;
    COMPARE(x, T(10));
    where(!alternating_mask, x) += T(1);
    COMPARE(!alternating_mask, x == T(11));
    where(alternating_mask, x) -= Convertible<V>();
    COMPARE(alternating_mask, x == T(6));
    where(alternating_mask, x) /= T(2);
    COMPARE(alternating_mask, x == T(3));
    where(alternating_mask, x) *= T(3);
    COMPARE(alternating_mask, x == T(9));

    x = 10;
    where(alternating_mask, x)++;
    COMPARE(alternating_mask, x == T(11));
    ++where(alternating_mask, x);
    COMPARE(alternating_mask, x == T(12));
    where(alternating_mask, x)--;
    COMPARE(alternating_mask, x == T(11));
    --where(alternating_mask, x);
    --where(alternating_mask, x);
    COMPARE(alternating_mask, x == T(9));
    COMPARE(alternating_mask, -where(alternating_mask, x) == T(-T(9)));

    x = 10;
    where(alternating_mask, x) = 0;
    COMPARE(!x, alternating_mask);
    COMPARE(!where(alternating_mask, x), alternating_mask);
    COMPARE(!where(!alternating_mask, x), M(false));
}

TEST_TYPES(T, where_fundamental, (int, float, double, short))
{
    using Vc::where;
    T x = T();
    where(true, x) = x + 1;
    COMPARE(x, T(1));
    where(false, x) = x - 1;
    COMPARE(x, T(1));
    where(true, x) += T(1);
    COMPARE(x, T(2));
}
