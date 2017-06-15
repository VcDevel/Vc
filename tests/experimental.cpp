/*  This file is part of the Vc library. {{{
Copyright © 2017 Matthias Kretz <kretz@kde.org>

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
#define Vc_EXPERIMENTAL 1
#include <vir/test.h>
#include <Vc/Vc>
#include "make_vec.h"

template <class... Ts> using base_template = Vc::datapar<Ts...>;
#include "testtypes.h"

TEST_TYPES(V, where_apply, all_test_types)
{
    using T = typename V::value_type;

    for (int split = 0; split <= int(V::size());++split) {
        V a([](T i) { return i; });
        where(a > split, a)
            .apply([](auto &&masked) { masked = 1; })
            .apply_inv([](auto &&masked) { masked = 2; });
        COMPARE(a, V([split](int i) { return i > split ? T(1) : T(2); }));

        V b = 0;
        where(a == 1, a, b)
            .apply([](auto &&a_, auto &&b_) { a_ = b_; })
            .apply_inv([](auto &&a_, auto &&b_) { b_ = a_; });
        COMPARE(a, V([split](int i) { return i > split ? T(0) : T(2); }));
        COMPARE(b, V([split](int i) { return i > split ? T(0) : T(2); }));
    }
}

TEST_TYPES(V, generators, all_test_types)
{
    COMPARE(V::seq(), make_vec<V>({0, 1}, 2));
}

#ifdef __cpp_fold_expressions
template <class V> void concat_small(std::false_type) {}
template <class V> void concat_small(std::true_type)
{
    using T = typename V::value_type;
    V a(0), b(1), c(2);
    auto x = concat(a, b, c);
    COMPARE(x.size(), a.size() * 3);
    std::size_t i = 0;
    for (; i < a.size(); ++i) {
        COMPARE(x[i], T(0));
    }
    for (; i < 2 * a.size(); ++i) {
        COMPARE(x[i], T(1));
    }
    for (; i < 3 * a.size(); ++i) {
        COMPARE(x[i], T(2));
    }
}

template <class V> void concat_ge4(std::false_type) {}
template <class V> void concat_ge4(std::true_type)
{
    using T = typename V::value_type;
    V a([](auto i) -> T { return i; });
    constexpr auto N0 = V::size() / 4u;
    constexpr auto N1 = V::size() - 2 * N0;
    using V0 = Vc::datapar<T, Vc::abi_for_size_t<T, N0>>;
    using V1 = Vc::datapar<T, Vc::abi_for_size_t<T, N1>>;
    auto x = Vc::split<N0, N0, N1>(a);
    COMPARE(std::tuple_size<decltype(x)>::value, 3u);
    COMPARE(std::get<0>(x), V0([](auto i) -> T { return i; }));
    COMPARE(std::get<1>(x), V0([](auto i) -> T { return i + N0; }));
    COMPARE(std::get<2>(x), V1([](auto i) -> T { return i + 2 * N0; }));
    auto b = concat(std::get<1>(x), std::get<2>(x), std::get<0>(x));
    // a and b may have different types if a was fixed_size<N> such that another ABI tag exists with
    // equal N, then b will have the non-fixed-size ABI tag.
    COMPARE(a.size(), b.size());
    COMPARE(b, decltype(b)([](auto i) -> T { return (N0 + i) % V::size(); }));
}

template <class V> void concat_even(std::false_type) {}
template <class V> void concat_even(std::true_type)
{
    using T = typename V::value_type;
    using V2 = Vc::datapar<T, Vc::abi_for_size_t<T, 2>>;
    using V3 = Vc::datapar<T, Vc::abi_for_size_t<T, V::size() / 2>>;

    V a([](auto i) -> T { return i; });

    std::array<V2, V::size() / 2> v2s = Vc::split<V2>(a);
    int offset = 0;
    for (V2 test : v2s) {
        COMPARE(test, V2([&](auto i) -> T { return i + offset; }));
        offset += 2;
    }

    std::array<V3, 2> v3s = Vc::split<V3>(a);
    COMPARE(v3s[0], V3([](auto i) -> T { return i; }));
    COMPARE(v3s[1], V3([](auto i) -> T { return i + V3::size(); }));
}


TEST_TYPES(V, split_concat, all_test_types)
{
    concat_small<V>(
        std::integral_constant<bool, V::size() * 3 <= Vc::datapar_abi::max_fixed_size>());
    concat_ge4<V>(std::integral_constant<bool, (V::size() >= 4)>());
    concat_even<V>(std::integral_constant<bool, ((V::size() & 1) == 0)>());
}
#endif  // __cpp_fold_expressions
