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
#include <Vc/simd>
#include <Vc/ostream>
#include "make_vec.h"

template <class... Ts> using base_template = Vc::simd<Ts...>;
#include "testtypes.h"

template <class V> struct Convertible {
    operator V() const { return V(4); }
};

template <class M, class T> constexpr bool where_is_ill_formed_impl(M, const T &, float)
{
    return true;
}
template <class M, class T>
constexpr auto where_is_ill_formed_impl(M m, const T &v, int)
    -> std::conditional_t<true, bool, decltype(Vc::where(m, v))>
{
    return false;
}

template <class M, class T> constexpr bool where_is_ill_formed(M m, const T &v)
{
    return where_is_ill_formed_impl(m, v, int());
}

TEST_TYPES(V, where, all_test_types)
{
    using M = typename V::mask_type;
    using T = typename V::value_type;
    const V indexes = make_vec<V>({1, 2, 3, 4}, 4);
    const M alternating_mask = make_mask<M>({true, false});
    V x = 0;
    where(alternating_mask, x) = indexes;
    COMPARE(alternating_mask, x == indexes);

    where(!alternating_mask, x) = T(2);
    COMPARE(!alternating_mask, x == T(2)) << x;

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

    const auto y = x;
    VERIFY(where_is_ill_formed(true, y));
    VERIFY(where_is_ill_formed(true, x));
    VERIFY(where_is_ill_formed(true, V(x)));

    M test = alternating_mask;
    where(alternating_mask, test) = M(true);
    COMPARE(test, alternating_mask);
    where(alternating_mask, test) = M(false);
    COMPARE(test, M(false));
    where(alternating_mask, test) = M(true);
    COMPARE(test, alternating_mask);
}

TEST_TYPES(T, where_fundamental, int, float, double, short)
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
