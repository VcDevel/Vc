/*  This file is part of the Vc library. {{{
Copyright © 2014-2015 Matthias Kretz <kretz@kde.org>

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
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

//#define UNITTEST_ONLY_XTEST 1
#include "unittest.h"
#include <list>

using Vc::simdize;
using Vc::float_v;
using Vc::int_v;
using Vc::int_m;

template <typename Scalar, typename Base, std::size_t N>
using SimdizeAdapter = Vc::SimdizeDetail::Adapter<Scalar, Base, N>;

TEST(homogeneous_sizeof)
{
    using Vc::SimdizeDetail::homogeneous_sizeof;
    VERIFY((homogeneous_sizeof<int, float, unsigned>::value == 4));
    VERIFY((homogeneous_sizeof<std::tuple<int, float, unsigned>, float>::value == 4));
    VERIFY((homogeneous_sizeof<unsigned short, std::array<short, 4>>::value == 2));
    VERIFY((homogeneous_sizeof<int, short>::value == 0));
    VERIFY((homogeneous_sizeof<std::tuple<int, short>>::value == 0));
    VERIFY((homogeneous_sizeof<std::array<int, 3>, double>::value == 0));
}

TEST(test_simdize)
{
    using namespace std;
    using namespace Vc;
    using Test0 = simdize<float>;
    using Test1 = simdize<tuple<float, double>>;
    using Test1_0 = typename std::decay<decltype(get<0>(Test1()))>::type;
    using Test1_1 = typename tuple_element<1, typename Test1::base_type>::type;
    using Test2 = simdize<tuple<double, float>>;
    using Test2_0 = typename tuple_element<0, typename Test2::base_type>::type;
    using Test2_1 = typename tuple_element<1, typename Test2::base_type>::type;
    using Test3 = simdize<tuple<std::string, float>>;
    using Test3_1 = typename tuple_element<1, typename Test3::base_type>::type;
    using Test4 = simdize<tuple<double, std::string, float>>;
    using Test4_0 = typename tuple_element<0, typename Test4::base_type>::type;
    using Test4_1 = typename tuple_element<1, typename Test4::base_type>::type;
    using Test4_2 = typename tuple_element<2, typename Test4::base_type>::type;
    COMPARE(Test0::size(), float_v::size());
    COMPARE(Test1_0::size(), float_v::size());
    COMPARE(Test1_1::size(), float_v::size());
    COMPARE(Test1::size(), float_v::size());
    COMPARE(Test2_0::size(), double_v::size());
    COMPARE(Test2_1::size(), double_v::size());
    COMPARE(Test2::size(), double_v::size());
    COMPARE(Test3_1::size(), float_v::size());
    COMPARE(Test3::size(), float_v::size());
    COMPARE(Test4_0::size(), double_v::size());
    COMPARE(Test4_2::size(), double_v::size());
    COMPARE(Test4::size(), double_v::size());
    COMPARE(typeid(Test4_0), typeid(double_v));
    COMPARE(typeid(Test4_1), typeid(std::string));
    COMPARE(typeid(Test4_2), typeid(simdize<float, double_v::size()>));
    COMPARE(typeid(simdize<tuple<std::string>>), typeid(tuple<std::string>));
    COMPARE(typeid(simdize<float>), typeid(float_v));
    COMPARE(typeid(Test1),
            typeid(SimdizeAdapter<tuple<float, double>,
                                  tuple<float_v, simdize<double, float_v::size()>>,
                                  float_v::size()>));
}

TEST(simdize_bools)
{
    using namespace std;
    using namespace Vc;
    COMPARE(typeid(simdize<bool>), typeid(float_m));
    COMPARE(typeid(simdize<bool, float_m::size()>), typeid(float_m));
    COMPARE(typeid(simdize<bool, 0, int>), typeid(int_m));
    COMPARE(typeid(simdize<bool, int_m::size(), int>), typeid(int_m));
    COMPARE(typeid(simdize<bool, float_m::size() + 1>),
            typeid(fixed_size_simd_mask<float, float_m::size() + 1>));
    COMPARE(typeid(simdize<bool, int_m::size() + 1, int>),
            typeid(fixed_size_simd_mask<int, int_m::size() + 1>));

    COMPARE(typeid(simdize<tuple<bool>>),
            typeid(SimdizeAdapter<tuple<bool>, tuple<float_m>, float_m::size()>));
    COMPARE(typeid(simdize<tuple<bool>, float_m::size()>),
            typeid(SimdizeAdapter<tuple<bool>, tuple<float_m>, float_m::size()>));
    COMPARE(typeid(simdize<tuple<bool>, float_m::size() + 1>),
            typeid(SimdizeAdapter<tuple<bool>,
                                  tuple<fixed_size_simd_mask<float, float_m::size() + 1>>,
                                  float_m::size() + 1>));

    COMPARE(typeid(simdize<tuple<int, bool>>),
            typeid(SimdizeAdapter<tuple<int, bool>, tuple<int_v, int_m>, int_v::size()>));
    COMPARE(
        typeid(simdize<tuple<int, bool>, 3>),
        typeid(
            SimdizeAdapter<tuple<int, bool>,
                           tuple<SimdArray<int, 3>, fixed_size_simd_mask<float, 3>>, 3>));

    COMPARE(
        typeid(simdize<tuple<bool, double, bool>>),
        typeid(SimdizeAdapter<
            tuple<bool, double, bool>,
            tuple<float_m,
                  typename std::conditional<float_m::size() == double_v::size(), double_v,
                                            SimdArray<double, float_m::size()>>::type,
                  float_m>,
            float_m::size()>));
    COMPARE(typeid(simdize<tuple<bool, double, bool>, float_m::size() + 1>),
            typeid(SimdizeAdapter<tuple<bool, double, bool>,
                                  tuple<fixed_size_simd_mask<float, float_m::size() + 1>,
                                        SimdArray<double, float_m::size() + 1>,
                                        fixed_size_simd_mask<float, float_m::size() + 1>>,
                                  float_m::size() + 1>));
    COMPARE(typeid(simdize<tuple<bool, double, bool>, 0, double>),
            typeid(SimdizeAdapter<tuple<bool, double, bool>,
                                  tuple<double_m, double_v, double_m>, double_m::size()>));

    COMPARE(typeid(simdize<tuple<int, double, bool>, 0, double>),
            typeid(SimdizeAdapter<tuple<int, double, bool>,
                                  tuple<int_v, simdize<double, int_v::size()>,
                                        simdize<bool, int_v::size(), double>>,
                                  int_v::size()>));
}

template <typename, int...> struct Foo1
{
    static constexpr std::size_t tuple_size = 1;
};
template <typename, typename, unsigned...> struct Foo2
{
    static constexpr std::size_t tuple_size = 2;
};
template <typename, typename, typename, std::size_t...> struct Foo3
{
    static constexpr std::size_t tuple_size = 3;
};

// ICC and MSVC do not support packs of values
#ifndef Vc_VALUE_PACK_EXPANSION_IS_BROKEN
TEST(nontype_template_parameters)
{
    using namespace std;
    using namespace Vc;
    using std::array;

    using float_intsize = typename std::conditional<
        int_v::size() == float_v::size(), float_v,
        typename std::conditional<std::is_same<float_v::abi, VectorAbi::Avx>::value,
                                  Vector<float, VectorAbi::Sse>,
                                  SimdArray<float, int_v::size()>>::type>::type;

    static_assert(SimdizeDetail::is_class_template<array<float, 3>>::value, "");
    static_assert(SimdizeDetail::is_class_template<Foo1<float, 3, 5, 6>>::value, "");
    static_assert(SimdizeDetail::is_class_template<Foo2<int, float, 3, 5, 6>>::value, "");

    COMPARE(typeid(simdize<array<float, 3>>),
            typeid(SimdizeAdapter<array<float, 3>, array<float_v, 3>, float_v::size()>));
    COMPARE(typeid(simdize<array<bool, 3>>),
            typeid(SimdizeAdapter<array<bool, 3>, array<float_m, 3>, float_m::size()>));
    COMPARE(
        typeid(simdize<Foo1<float, 3, 5, 6>>),
        typeid(
            SimdizeAdapter<Foo1<float, 3, 5, 6>, Foo1<float_v, 3, 5, 6>, float_v::size()>));
    COMPARE(typeid(simdize<Foo2<int, float, 3, 5, 6>>),
            typeid(SimdizeAdapter<Foo2<int, float, 3, 5, 6>,
                                  Foo2<int_v, float_intsize, 3, 5, 6>, int_v::size()>));
    COMPARE(
        typeid(simdize<Foo3<int, int, float, 3, 5, 6>>),
        typeid(
            SimdizeAdapter<Foo3<int, int, float, 3, 5, 6>,
                           Foo3<int_v, int_v, float_intsize, 3, 5, 6>, int_v::size()>));
}
#endif  // Vc_VALUE_PACK_EXPANSION_IS_BROKEN

TEST(tuple_interface)
{
    using V0 = simdize<std::tuple<int, bool>>;
    COMPARE(std::tuple_size<V0>::value, 2u);
    COMPARE(typeid(typename std::tuple_element<0, V0>::type), typeid(int_v));
    COMPARE(typeid(typename std::tuple_element<1, V0>::type), typeid(int_m));

    V0 v;
    COMPARE(typeid(decltype(std::get<0>(v))), typeid(int_v));
    COMPARE(typeid(decltype(std::get<1>(v))), typeid(int_m));
    std::get<0>(v) = int_v([](int n) { return n; });
    COMPARE(std::get<0>(v), int_v([](int n) { return n; }));
    std::get<0>(v) += 1;
    COMPARE(std::get<0>(v), int_v([](int n) { return n + 1; }));
}

TEST(assign)
{
    using T = std::tuple<float, unsigned>;
    using V = simdize<T>;
    V v;
    for (unsigned i = 0; i < v.size(); ++i) {
        assign(v, i, T{1.f * i, i});
        COMPARE(std::get<0>(v)[i], 1.f * i);
        COMPARE(std::get<1>(v)[i], i);
    }
    for (unsigned i = 0; i < v.size(); ++i) {
        COMPARE(std::get<0>(v)[i], 1.f * i);
        COMPARE(std::get<1>(v)[i], i);
    }
}

template <typename T> T copy_by_value(T x) { return T(x); }

TEST(copy)
{
    using T = std::tuple<float, int>;
    using V = simdize<T>;
    V v = {1.f, 1};
    for (unsigned i = 0; i < v.size(); ++i) {
        COMPARE(std::get<0>(v)[i], 1.f);
        COMPARE(std::get<1>(v)[i], 1);
    }
    V v2 = copy_by_value(v);
    for (unsigned i = 0; i < v2.size(); ++i) {
        COMPARE(std::get<0>(v2)[i], 1.f);
        COMPARE(std::get<1>(v2)[i], 1);
    }
    v = {2.f, 2};
    v2 = copy_by_value(v);
    for (unsigned i = 0; i < v2.size(); ++i) {
        COMPARE(std::get<0>(v2)[i], 2.f);
        COMPARE(std::get<1>(v2)[i], 2);
    }
}

TEST(extract)
{
    using T = std::tuple<float, unsigned>;
    using V = simdize<T>;
    V v;
    for (unsigned i = 0; i < v.size(); ++i) {
        assign(v, i, T{1.f * i, i});
        COMPARE(std::get<0>(v)[i], 1.f * i);
        COMPARE(std::get<1>(v)[i], i);
    }
    for (unsigned i = 0; i < v.size(); ++i) {
        COMPARE(std::get<0>(v)[i], 1.f * i);
        COMPARE(std::get<1>(v)[i], i);
    }
    for (unsigned i = 0; i < v.size(); ++i) {
        COMPARE(extract(v, i), T(1.f * i, i));
    }
}

TEST(decorate)
{
    using T = std::tuple<float, unsigned>;
    using V = simdize<T>;
    V v;
    auto vv = decorate(v);
    for (unsigned i = 0; i < v.size(); ++i) {
        vv[i] = T{1.f * i, i};
        COMPARE(std::get<0>(v)[i], 1.f * i);
        COMPARE(std::get<1>(v)[i], i);
    }
    for (unsigned i = 0; i < v.size(); ++i) {
        COMPARE(std::get<0>(v)[i], 1.f * i);
        COMPARE(std::get<1>(v)[i], i);
    }
    for (unsigned i = 0; i < v.size(); ++i) {
        T x = vv[i];
        COMPARE(x, T(1.f * i, i));
    }
    const V &v2 = v;
    for (unsigned i = 0; i < v.size(); ++i) {
        T x = decorate(v2)[i];
        COMPARE(x, T(1.f * i, i));
    }
}

TEST(broadcast)
{
    {
        using T = std::tuple<float, int>;
        using V = simdize<T>;

        T scalar(2.f, 3);
        V vector(scalar);
        COMPARE(std::get<0>(vector), float_v(2.f));
        COMPARE(std::get<1>(vector), (simdize<int, V::size()>(3)));
    }
    {
        using T = std::array<int, 3>;
        using V = simdize<T>;
        T scalar{{1, 2, 3}};
        V vector(scalar);
        COMPARE(vector[0], int_v(1));
        COMPARE(vector[1], int_v(2));
        COMPARE(vector[2], int_v(3));
    }
}

TEST(list_iterator_vectorization)
{
    {
        using L = std::list<float>;
        using LIV = simdize<L::iterator>;
        L list;
        for (auto i = 1024; i; --i) {
            list.push_back(i);
        }
        LIV b = list.begin();
        LIV e = list.end();
        float_v reference = list.size() - float_v([](int n) { return n; });
        for (; b != e; ++b, reference -= float_v::size()) {
            float_v x = *b;
            COMPARE(x, reference);
            COMPARE(*b, reference);
            *b = x + 1;
            COMPARE(*b, reference + 1);
            auto &&ref = *b;
            ref = x + 2;
            COMPARE(*b, reference + 2);
            COMPARE(ref, reference + 2);
            ref = x + 1;
            COMPARE(*b, reference + 1);
            COMPARE(ref, reference + 1);
        }
        reference = list.size() - float_v([](int n) { return n; }) + 1;
        for (b = list.begin(); b != e; ++b, reference -= float_v::size()) {
            float_v x = *b;
            COMPARE(x, reference);
            COMPARE(*b, reference);
        }
    }
    {
        using T = std::tuple<float, unsigned, float>;
        using V = simdize<T>;
        using L = std::list<T>;
        using LIV = simdize<L::iterator>;
        L list;
        for (auto i = 1024; i; --i) {
            list.push_back(T(i, i * 2, i * 3));
        }
        LIV b = list.begin();
        LIV e = list.end();
        auto reference1 = list.size() - float_v([](int n) { return n; });
        auto reference2 =
            unsigned(list.size()) - simdize<unsigned, V::size()>([](int n) { return n; });
        for (; b != e; ++b, reference1 -= int(V::size()), reference2 -= int(V::size())) {
            V x = *b;
            COMPARE(x, V(reference1, reference2 * 2, reference1 * 3));
            COMPARE(std::get<0>(*b), reference1);
        }
    }
}

TEST_TYPES(L, cast_simdized_iterator_to_scalar, std::vector<float>, std::list<float>,
           std::vector<std::tuple<float, int>>)
{
    using T = typename L::value_type;
    using LI = typename L::iterator;
    using LIV = simdize<LI>;
    L list;
    for (auto i = 13; i; --i) {
        list.push_back(T());
    }
    LIV b = list.begin();
    LIV e = list.end();
    COMPARE(static_cast<LI>(b), list.begin());
    COMPARE(static_cast<LI>(e), list.end());
}

TEST(vector_iterator_vectorization)
{
    {
        using L = std::vector<float>;
        using LIV = simdize<L::iterator>;
        L list;
        for (auto i = 1024; i; --i) {
            list.push_back(i);
        }
        LIV b = list.begin();
        LIV e = list.end();
        const auto &bconst = b;

        COMPARE(b->sum(), (1024 - float_v([](int n) { return n; })).sum());
        COMPARE(bconst->sum(), (1024 - float_v([](int n) { return n; })).sum());
        COMPARE((*b).sum(), (1024 - float_v([](int n) { return n; })).sum());
        COMPARE((*bconst).sum(), (1024 - float_v([](int n) { return n; })).sum());
        COMPARE((e - 1)->sum(), (1 + float_v([](int n) { return n; })).sum());
        COMPARE((e + -1)->sum(), (1 + float_v([](int n) { return n; })).sum());
        COMPARE((-1 + e)->sum(), (1 + float_v([](int n) { return n; })).sum());
        COMPARE((b + 1)->sum(), (1024 - b.size() - float_v([](int n) { return n; })).sum());
        COMPARE((1 + b)->sum(), (1024 - b.size() - float_v([](int n) { return n; })).sum());
        COMPARE(e - b, static_cast<LIV::difference_type>(1024 / b.size()));
        COMPARE(b - e, -static_cast<LIV::difference_type>(1024 / b.size()));

        VERIFY(b < e);
        VERIFY(!(b > e));
        VERIFY(e > b);
        VERIFY(!(e < b));
        VERIFY(b <= e);
        VERIFY(!(b >= e));
        VERIFY(e >= b);
        VERIFY(!(e <= b));
        VERIFY(b != e);
        VERIFY(!(b == e));

        VERIFY(!(b < b));
        VERIFY(!(b > b));
        if (b.size() > 1) {
            VERIFY(!(b <= b));
            VERIFY(!(b >= b));
        } else {
            VERIFY(b <= b);
            VERIFY(b >= b);
        }
        VERIFY(b == b);
        VERIFY(!(b != b));

        auto next = b + 1;
        VERIFY(next > b);
        VERIFY(!(b > next));
        VERIFY(!(next < b));
        VERIFY(b < next);
        VERIFY(next >= b);
        VERIFY(!(b >= next));
        VERIFY(!(next <= b));
        VERIFY(b <= next);
        VERIFY(b != next);
        VERIFY(!(b == next));

        next--;
        COMPARE(next, b);
        COMPARE(*next, *b);

        float_v reference = list.size() - float_v([](int n) { return n; });
        for (; b != e; ++b, reference -= float_v::size()) {
            float_v x = *b;
            COMPARE(x, reference);
            COMPARE(*b, reference);
            *b = x + 1;
            COMPARE(*b, reference + 1);
        }
        reference = list.size() - float_v([](int n) { return n; }) + 1;
        for (b = list.begin(); b != e; ++b, reference -= float_v::size()) {
            float_v x = *b;
            COMPARE(x, reference);
            COMPARE(*b, reference);
        }

        // also test const_iterator
        reference = list.size() - float_v::IndexesFromZero() + 1;
        using LCIV = simdize<L::const_iterator>;
        LCIV it = list.cbegin();
        const LCIV ce = list.cend();
        for (; it != ce; ++it, reference -= float_v::size()) {
            float_v x = *it;
            COMPARE(x, reference);
            COMPARE(*it, reference);
        }
    }
    {
        using L = std::vector<std::tuple<short, float>>;
        using LIV = simdize<L::iterator>;
        L list;
        for (auto i = 1024; i; --i) {
            list.emplace_back(i, i);
        }
        const LIV end_it = list.end();
        for (LIV it = list.begin(); it != end_it; ++it) {
            simdize<std::tuple<short, float>> x = *it;
            COMPARE(std::get<0>(x), std::get<0>(*it));
            COMPARE(std::get<1>(x), std::get<1>(*it));
        }
    }
}

TEST(shifted)
{
    using T = std::tuple<float, int>;
    using V = simdize<T>;

    using V0 = float_v;
    using V1 = simdize<int, V::size()>;

    V v;
    std::get<0>(v) = V0([](int n) { return n; }) + 1;
    std::get<1>(v) = V1([](int n) { return n; }) + 1;

    for (int shift = -int(V::size()); shift <= int(V::size()); ++shift) {
        V test = shifted(v, shift);
        COMPARE(std::get<0>(test), (V0([](int n) { return n; }) + 1).shifted(shift));
        COMPARE(std::get<1>(test), (V1([](int n) { return n; }) + 1).shifted(shift));
        V test2 = decorate(v).shifted(shift);
        COMPARE(test2, test);
    }

    V test = shifted(v, int(V::size()));
    COMPARE(test, V{});

    test = shifted(v, -int(V::size()));
    COMPARE(test, V{});

    V reference{};
    assign(reference, 0, T(V::size(), V::size()));
    test = shifted(v, int(V::size()) - 1);
    COMPARE(test, reference);

    reference = {};
    assign(reference, V::size() - 1, T(1, 1));
    test = shifted(v, -int(V::size()) + 1);
    COMPARE(test, reference);
}

TEST(swap)
{
    using T = std::tuple<float, int>;
    using V = simdize<T>;

    using V0 = float_v;
    using V1 = simdize<int, V::size()>;

    V v;
    std::get<0>(v) = V0([](int n) { return n; }) + 1;
    std::get<1>(v) = V1([](int n) { return n; }) + 1;
    for (std::size_t i = 0; i < V::size(); ++i) {
        const T atI(1 + i, 1 + i);

        T scalar(V::size() + 1, V::size() + 1);
        T copy = scalar;

        COMPARE(extract(v, i), atI);
        swap(v, i, scalar);
        COMPARE(scalar, atI);
        COMPARE(extract(v, i), copy);

        for (std::size_t j = 0; j < V::size(); ++j) {
            if (j != i) {
                COMPARE(extract(v, j), T(j + 1, j + 1));
            }
        }

        std::swap(scalar, decorate(v)[i]);
        COMPARE(scalar, copy);
        COMPARE(extract(v, i), atI);

        for (std::size_t j = 0; j < V::size(); ++j) {
            if (j != i) {
                COMPARE(extract(v, j), T(j + 1, j + 1));
            }
        }
    }

    if (V::size() > 1) {
        auto vv = decorate(v);
        COMPARE(T(vv[0]), T(1, 1));
        COMPARE(T(vv[1]), T(2, 2));
        std::swap(vv[0], vv[1]);
        COMPARE(T(vv[0]), T(2, 2));
        COMPARE(T(vv[1]), T(1, 1));
        swap(v, 0, v, 1);
        COMPARE(T(vv[0]), T(1, 1));
        COMPARE(T(vv[1]), T(2, 2));
    }
}

TEST(conditional_assignment)
{
    using T = std::tuple<float, int, double>;
    using V = simdize<T>;

    using V0 = float_v;
    using V1 = simdize<int, V::size()>;
    using V2 = simdize<double, V::size()>;
    using M0 = typename V0::mask_type;
    using M1 = typename V1::mask_type;
    using M2 = typename V2::mask_type;

    V v;
    withRandomMask<V0, 1000>([&](M0 m) {
        using Vc::simd_cast;
        std::get<0>(v) = V0([](int n) { return n; }) + 1;
        std::get<1>(v) = V1([](int n) { return n; }) + 1;
        std::get<2>(v) = V2([](int n) { return n; }) + 1;

        where(m) | v = V{};
        COMPARE(std::get<0>(v) == V0(0), m) << std::get<0>(v);
        COMPARE(std::get<1>(v) == V1(0), simd_cast<M1>(m)) << std::get<1>(v);
        COMPARE(std::get<2>(v) == V2(0), simd_cast<M2>(m)) << std::get<2>(v);

        where(m) | v += V{T{V::size() + 2, V::size() + 2, V::size() + 2}};
        COMPARE(std::get<0>(v) == V0(V::size() + 2), m) << std::get<0>(v);
        COMPARE(std::get<1>(v) == V1(V::size() + 2), simd_cast<M1>(m)) << std::get<1>(v);
        COMPARE(std::get<2>(v) == V2(V::size() + 2), simd_cast<M2>(m)) << std::get<2>(v);

        where(m) | v *= V{T{2, 2, 2}};
        COMPARE(std::get<0>(v) == V0(2 * (V::size() + 2)), m) << std::get<0>(v);
        COMPARE(std::get<1>(v) == V1(2 * (V::size() + 2)), simd_cast<M1>(m)) << std::get<1>(v);
        COMPARE(std::get<2>(v) == V2(2 * (V::size() + 2)), simd_cast<M2>(m)) << std::get<2>(v);

        where(m) | v -= V{T{V::size() + 2, V::size() + 2, V::size() + 2}};
        COMPARE(std::get<0>(v) == V0(V::size() + 2), m) << std::get<0>(v);
        COMPARE(std::get<1>(v) == V1(V::size() + 2), simd_cast<M1>(m)) << std::get<1>(v);
        COMPARE(std::get<2>(v) == V2(V::size() + 2), simd_cast<M2>(m)) << std::get<2>(v);
    });
}

TEST(copy_simdized_objects)
{
    using T = std::tuple<float, double>;
    using V = simdize<T>;

    using V0 = typename std::tuple_element<0, V>::type;
    using V1 = typename std::tuple_element<1, V>::type;

    V v;
    V v2 = v;
    v = v2;
    v2 = std::move(v);
    V v3 = std::move(v2);
    COMPARE(std::get<0>(v3), V0(0));
    COMPARE(std::get<1>(v3), V1(0));
}

template <class T> T create(int x) { return x; }
template <> std::array<float, 3> create<std::array<float, 3>>(int _x)
{
    float x = _x;
    return {{x, x + 1, x + 2}};
}
template <> std::tuple<double, int> create<std::tuple<double, int>>(int x)
{
    return std::tuple<double, int>(x, x + 1);
}

TEST_TYPES(T, generator, float, short, std::array<float, 3>, std::tuple<double, int>)
{
    using V = simdize<T>;

    {
        const V test([](int) { return T(); });
        const auto &testv = Vc::decorate(test);
        for (std::size_t i = 0; i < V::size(); ++i) {
            COMPARE(testv[i], T());
        }
    }
    {
        const V test([](int n) { return create<T>(n); });
        const auto &testv = Vc::decorate(test);
        for (std::size_t i = 0; i < V::size(); ++i) {
            COMPARE(testv[i], create<T>(i));
        }
    }
}

// vim: foldmethod=marker
