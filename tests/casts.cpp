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
#include <vir/test.h>
#include <Vc/Vc>
#include "make_vec.h"
#include <vir/metahelpers.h>

template <class... Ts> using base_template = Vc::simd<Ts...>;
#include "testtypes.h"
#include "conversions.h"

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
    using V0 = Vc::simd<T, Vc::abi_for_size_t<T, N0>>;
    using V1 = Vc::simd<T, Vc::abi_for_size_t<T, N1>>;
    {
        auto x = Vc::split<N0, N0, N1>(a);
        COMPARE(std::tuple_size<decltype(x)>::value, 3u);
        COMPARE(std::get<0>(x), V0([](auto i) -> T { return i; }));
        COMPARE(std::get<1>(x), V0([](auto i) -> T { return i + N0; }));
        COMPARE(std::get<2>(x), V1([](auto i) -> T { return i + 2 * N0; }));
        auto b = concat(std::get<1>(x), std::get<2>(x), std::get<0>(x));
        // a and b may have different types if a was fixed_size<N> such that another ABI
        // tag exists with equal N, then b will have the non-fixed-size ABI tag.
        COMPARE(a.size(), b.size());
        COMPARE(b, decltype(b)([](auto i) -> T { return (N0 + i) % V::size(); }));
    }{
        auto x = Vc::split<N0, N1, N0>(a);
        COMPARE(std::tuple_size<decltype(x)>::value, 3u);
        COMPARE(std::get<0>(x), V0([](auto i) -> T { return i; }));
        COMPARE(std::get<1>(x), V1([](auto i) -> T { return i + N0; }));
        COMPARE(std::get<2>(x), V0([](auto i) -> T { return i + N0 + N1; }));
        auto b = concat(std::get<1>(x), std::get<2>(x), std::get<0>(x));
        // a and b may have different types if a was fixed_size<N> such that another ABI
        // tag exists with equal N, then b will have the non-fixed-size ABI tag.
        COMPARE(a.size(), b.size());
        COMPARE(b, decltype(b)([](auto i) -> T { return (N0 + i) % V::size(); }));
    }{
        auto x = Vc::split<N1, N0, N0>(a);
        COMPARE(std::tuple_size<decltype(x)>::value, 3u);
        COMPARE(std::get<0>(x), V1([](auto i) -> T { return i; }));
        COMPARE(std::get<1>(x), V0([](auto i) -> T { return i + N1; }));
        COMPARE(std::get<2>(x), V0([](auto i) -> T { return i + N0 + N1; }));
        auto b = concat(std::get<1>(x), std::get<2>(x), std::get<0>(x));
        // a and b may have different types if a was fixed_size<N> such that another ABI
        // tag exists with equal N, then b will have the non-fixed-size ABI tag.
        COMPARE(a.size(), b.size());
        COMPARE(b, decltype(b)([](auto i) -> T { return (N1 + i) % V::size(); }));
    }
}

template <class V> void concat_even(std::false_type) {}
template <class V> void concat_even(std::true_type)
{
    using T = typename V::value_type;
    using V2 = Vc::simd<T, Vc::abi_for_size_t<T, 2>>;
    using V3 = Vc::simd<T, Vc::abi_for_size_t<T, V::size() / 2>>;

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
    /*
    concat_small<V>(
        std::integral_constant<bool, V::size() * 3 <= Vc::simd_abi::max_fixed_size>());
    concat_ge4<V>(std::integral_constant<bool, (V::size() >= 4)>());
    concat_even<V>(std::integral_constant<bool, ((V::size() & 1) == 0)>());
    */
}
#endif  // __cpp_fold_expressions

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

template <class V> void casts_integral(std::true_type) {
    using T = typename V::value_type;
    using A = typename V::abi_type;
    using TU = std::make_unsigned_t<T>;
    using TS = std::make_signed_t<T>;
    COMPARE(typeid(Vc::static_simd_cast<TU>(V())), typeid(Vc::simd<TU, A>));
    COMPARE(typeid(Vc::static_simd_cast<TS>(V())), typeid(Vc::simd<TS, A>));
}

template <class V> void casts_integral(std::false_type) {}

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

template <class V, class To> void casts_widen(std::true_type) {
    constexpr auto N = V::size();
    using From = typename V::value_type;
    using W = Vc::fixed_size_simd<To, N>;

    for (gen_seq_t<V, To> gen_seq; gen_seq; ++gen_seq) {
        const V seq(gen_seq);
        COMPARE(Vc::simd_cast<V>(seq), seq);
        COMPARE(Vc::simd_cast<W>(seq), W(gen_cast<To, N>(seq)));
        auto test = Vc::simd_cast<To>(seq);
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
template <class V, class To> void casts_widen(std::false_type) {}

template <class To> struct foo {
    template <class T> auto operator()(const T &v) -> decltype(Vc::simd_cast<To>(v));
};

TEST_TYPES(V_To, casts, outer_product<all_test_types, arithmetic_types>)
{
    using V = typename V_To::template at<0>;
    using To = typename V_To::template at<1>;
    using From = typename V::value_type;

    casts_integral<V>(std::is_integral<From>());

    constexpr auto N = V::size();
    using W = Vc::fixed_size_simd<To, N>;

    using is_simd_cast_allowed =
        decltype(vir::test::sfinae_is_callable_t<const V &>(foo<To>()));

    COMPARE(is_simd_cast_allowed::value,
            std::numeric_limits<From>::digits <= std::numeric_limits<To>::digits &&
                std::numeric_limits<From>::max() <= std::numeric_limits<To>::max() &&
                !(std::is_signed<From>::value && std::is_unsigned<To>::value));
    casts_widen<V, To>(is_simd_cast_allowed());

    for (gen_seq_t<V, To> gen_seq; gen_seq; ++gen_seq) {
        const V seq(gen_seq);
        COMPARE(Vc::static_simd_cast<V>(seq), seq);
        COMPARE(Vc::static_simd_cast<W>(seq), W(gen_cast<To, N>(seq)));
        auto test = Vc::static_simd_cast<To>(seq);
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
