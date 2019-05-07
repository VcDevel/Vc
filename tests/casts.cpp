/*{{{
Copyright © 2010-2019 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH
                      Matthias Kretz <m.kretz@gsi.de>

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
#include <vir/metahelpers.h>

template <class... Ts> using base_template = std::experimental::simd<Ts...>;
#include "testtypes.h"
#include "conversions.h"

using std::experimental::simd_cast;
using std::experimental::static_simd_cast;

TEST_TYPES(V, split_concat, all_test_types)
{
    using T = typename V::value_type;
    if constexpr (V::size() * 3 <= std::experimental::simd_abi::max_fixed_size<T>) {
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

    if constexpr (V::size() >= 4) {
        const V a([](auto i) -> T { return i; });
        constexpr auto N0 = V::size() / 4u;
        constexpr auto N1 = V::size() - 2 * N0;
        using V0 = std::experimental::simd<T, std::experimental::simd_abi::deduce_t<T, N0>>;
        using V1 = std::experimental::simd<T, std::experimental::simd_abi::deduce_t<T, N1>>;
        {
            auto x = std::experimental::split<N0, N0, N1>(a);
            COMPARE(std::tuple_size<decltype(x)>::value, 3u);
            COMPARE(std::get<0>(x), V0([](auto i) -> T { return i; }));
            COMPARE(std::get<1>(x), V0([](auto i) -> T { return i + N0; }));
            COMPARE(std::get<2>(x), V1([](auto i) -> T { return i + 2 * N0; }));
            auto b = concat(std::get<1>(x), std::get<2>(x), std::get<0>(x));
            // a and b may have different types if a was fixed_size<N> such that another
            // ABI tag exists with equal N, then b will have the non-fixed-size ABI tag.
            COMPARE(a.size(), b.size());
            COMPARE(b, decltype(b)([](auto i) -> T { return (N0 + i) % V::size(); }));
        }
        {
            auto x = std::experimental::split<N0, N1, N0>(a);
            COMPARE(std::tuple_size<decltype(x)>::value, 3u);
            COMPARE(std::get<0>(x), V0([](auto i) -> T { return i; }));
            COMPARE(std::get<1>(x), V1([](auto i) -> T { return i + N0; }));
            COMPARE(std::get<2>(x), V0([](auto i) -> T { return i + N0 + N1; }));
            auto b = concat(std::get<1>(x), std::get<2>(x), std::get<0>(x));
            // a and b may have different types if a was fixed_size<N> such that another
            // ABI tag exists with equal N, then b will have the non-fixed-size ABI tag.
            COMPARE(a.size(), b.size());
            COMPARE(b, decltype(b)([](auto i) -> T { return (N0 + i) % V::size(); }));
        }
        {
            auto x = std::experimental::split<N1, N0, N0>(a);
            COMPARE(std::tuple_size<decltype(x)>::value, 3u);
            COMPARE(std::get<0>(x), V1([](auto i) -> T { return i; }));
            COMPARE(std::get<1>(x), V0([](auto i) -> T { return i + N1; }));
            COMPARE(std::get<2>(x), V0([](auto i) -> T { return i + N0 + N1; }));
            auto b = concat(std::get<1>(x), std::get<2>(x), std::get<0>(x));
            // a and b may have different types if a was fixed_size<N> such that another
            // ABI tag exists with equal N, then b will have the non-fixed-size ABI tag.
            COMPARE(a.size(), b.size());
            COMPARE(b, decltype(b)([](auto i) -> T { return (N1 + i) % V::size(); }));
        }
    }

    if constexpr (V::size() % 3 == 0) {
        const V a([](auto i) -> T { return i; });
        constexpr auto N0 = V::size() / 3;
        using V0 = std::experimental::simd<T, std::experimental::simd_abi::deduce_t<T, N0>>;
        using V1 = std::experimental::simd<T, std::experimental::simd_abi::deduce_t<T, 2 * N0>>;
        {
            auto [x, y, z] = std::experimental::split<N0, N0, N0>(a);
            COMPARE(x, V0([](auto i) -> T { return i; }));
            COMPARE(y, V0([](auto i) -> T { return i + N0; }));
            COMPARE(z, V0([](auto i) -> T { return i + N0 * 2; }));
            auto b = concat(x, y, z);
            COMPARE(a.size(), b.size());
            COMPARE(b, simd_cast<decltype(b)>(a));
            COMPARE(simd_cast<V>(b), a);
        }
        {
            auto [x, y] = std::experimental::split<N0, 2 * N0>(a);
            COMPARE(x, V0([](auto i) -> T { return i; }));
            COMPARE(y, V1([](auto i) -> T { return i + N0; }));
            auto b = concat(x, y);
            COMPARE(a.size(), b.size());
            COMPARE(b, simd_cast<decltype(b)>(a));
            COMPARE(simd_cast<V>(b), a);
        }
        {
            auto [x, y] = std::experimental::split<2 * N0, N0>(a);
            COMPARE(x, V1([](auto i) -> T { return i; }));
            COMPARE(y, V0([](auto i) -> T { return i + 2 * N0; }));
            auto b = concat(x, y);
            COMPARE(a.size(), b.size());
            COMPARE(b, simd_cast<decltype(b)>(a));
            COMPARE(simd_cast<V>(b), a);
        }
    }

    if constexpr ((V::size() & 1) == 0) {
        using V2 = std::experimental::simd<T, std::experimental::simd_abi::deduce_t<T, 2>>;
        using V3 = std::experimental::simd<T, std::experimental::simd_abi::deduce_t<T, V::size() / 2>>;

        V a([](auto i) -> T { return i; });

        std::array<V2, V::size() / 2> v2s = std::experimental::split<V2>(a);
        int offset = 0;
        for (V2 test : v2s) {
            COMPARE(test, V2([&](auto i) -> T { return i + offset; }));
            offset += 2;
        }

        std::array<V3, 2> v3s = std::experimental::split<V3>(a);
        COMPARE(v3s[0], V3([](auto i) -> T { return i; }));
        COMPARE(v3s[1], V3([](auto i) -> T { return i + V3::size(); }));
    }
}

template <class T, size_t N> struct gen_cast {
    std::array<T, N> data;
    template <class V> gen_cast(const V &v)
    {
        for (size_t i = 0; i < V::size(); ++i) {
            data[i] = static_cast<T>(v[i]);
        }
    }
    template <class I> constexpr T operator()(I) { return data[I::value]; }
};

template <class V, class To> struct gen_seq_t {
    using From = typename V::value_type;
    const size_t N = cvt_input_data<From, To>.size();
    size_t offset = 0;
    void operator++() { offset += V::size(); }
    explicit operator bool() const { return offset < N; }
    From operator()(size_t i) const
    {
        i += offset;
        return i < N ? cvt_input_data<From, To>[i] : From(i);
    }
};

template <class To> struct foo {
    template <class T> auto operator()(const T &v) -> decltype(simd_cast<To>(v));
};

XTEST_TYPES(From, specific_conversions, all_test_types)
{
    using To = float;
    using W = std::experimental::native_simd<To>;

    using V = std::experimental::rebind_simd_t<From, W>;

    for (gen_seq_t<V, To> gen_seq; gen_seq; ++gen_seq) {
        const V seq(gen_seq);
        COMPARE(static_simd_cast<W>(seq), W(gen_cast<To, V::size()>(seq)))
            << "seq: " << seq;
    }
}

TEST_TYPES(V_To, casts, outer_product<all_test_types, arithmetic_types>)
{
    using V = typename V_To::template at<0>;
    using To = typename V_To::template at<1>;
    using From = typename V::value_type;
    constexpr auto N = V::size();
    using W = std::experimental::fixed_size_simd<To, N>;

    if constexpr (std::is_integral_v<From>) {
        using A = typename V::abi_type;
        using TU = std::make_unsigned_t<From>;
        using TS = std::make_signed_t<From>;
        COMPARE(typeid(static_simd_cast<TU>(V())), typeid(std::experimental::simd<TU, A>));
        COMPARE(typeid(static_simd_cast<TS>(V())), typeid(std::experimental::simd<TS, A>));
    }

    using is_simd_cast_allowed =
        decltype(vir::test::sfinae_is_callable_t<const V &>(foo<To>()));

    COMPARE(is_simd_cast_allowed::value,
            std::numeric_limits<From>::digits <= std::numeric_limits<To>::digits &&
                std::numeric_limits<From>::max() <= std::numeric_limits<To>::max() &&
                !(std::is_signed<From>::value && std::is_unsigned<To>::value));

    if constexpr (is_simd_cast_allowed::value) {
        for (gen_seq_t<V, To> gen_seq; gen_seq; ++gen_seq) {
            const V seq(gen_seq);
            COMPARE(simd_cast<V>(seq), seq);
            COMPARE(simd_cast<W>(seq), W(gen_cast<To, N>(seq)));
            auto test = simd_cast<To>(seq);
            // decltype(test) is not W if
            // a) V::abi_type is not fixed_size and
            // b.1) V::value_type and To are integral and of equal rank or
            // b.2) V::value_type and To are equal
            COMPARE(test, decltype(test)(gen_cast<To, N>(seq)));
            if (std::is_same<To, From>::value) {
                COMPARE(typeid(decltype(test)), typeid(V));
            }
        }
    }

    for (gen_seq_t<V, To> gen_seq; gen_seq; ++gen_seq) {
        const V seq(gen_seq);
        COMPARE(static_simd_cast<V>(seq), seq);
        COMPARE(static_simd_cast<W>(seq), W(gen_cast<To, N>(seq))) << '\n' << seq;
        auto test = static_simd_cast<To>(seq);
        // decltype(test) is not W if
        // a) V::abi_type is not fixed_size and
        // b.1) V::value_type and To are integral and of equal rank or
        // b.2) V::value_type and To are equal
        COMPARE(test, decltype(test)(gen_cast<To, N>(seq)));
        if (std::is_same<To, From>::value) {
            COMPARE(typeid(decltype(test)), typeid(V));
        }
    }
}

TEST(splits)
{
    using namespace std::experimental::parallelism_v2;
    native_simd_mask<float> k(true);
    VERIFY(all_of(k)) << k;
    const auto parts = split<simd_mask<float>>(k);
    for (auto k2 : parts) {
        VERIFY(all_of(k2)) << k2;
        COMPARE(typeid(k2), typeid(simd_mask<float>));
    }
}
