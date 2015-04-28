/*  This file is part of the Vc library. {{{
Copyright © 2014 Matthias Kretz <kretz@kde.org>
All rights reserved.

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

#include "unittest.h"
#include <list>

using Vc::simdize;
using Vc::float_v;
using Vc::int_v;
using Vc::int_m;

template <typename Scalar, typename Base, std::size_t N>
using SimdizeAdapter = Vc::SimdizeDetail::Adapter<Scalar, Base, N>;

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
            typeid(simd_mask_array<float, float_m::size() + 1>));
    COMPARE(typeid(simdize<bool, int_m::size() + 1, int>),
            typeid(simd_mask_array<int, int_m::size() + 1>));

    COMPARE(typeid(simdize<tuple<bool>>),
            typeid(SimdizeAdapter<tuple<bool>, tuple<float_m>, float_m::size()>));
    COMPARE(typeid(simdize<tuple<bool>, float_m::size()>),
            typeid(SimdizeAdapter<tuple<bool>, tuple<float_m>, float_m::size()>));
    COMPARE(
        typeid(simdize<tuple<bool>, float_m::size() + 1>),
        typeid(
            SimdizeAdapter<tuple<bool>, tuple<simd_mask_array<float, float_m::size() + 1>>,
                           float_m::size() + 1>));

    COMPARE(typeid(simdize<tuple<int, bool>>),
            typeid(SimdizeAdapter<tuple<int, bool>, tuple<int_v, int_m>, int_v::size()>));
    COMPARE(
        typeid(simdize<tuple<int, bool>, 3>),
        typeid(SimdizeAdapter<tuple<int, bool>,
                              tuple<simdarray<int, 3>, simd_mask_array<float, 3>>, 3>));

    COMPARE(
        typeid(simdize<tuple<bool, double, bool>>),
        typeid(SimdizeAdapter<
            tuple<bool, double, bool>,
            tuple<float_m,
                  typename std::conditional<float_m::size() == double_v::size(), double_v,
                                            simdarray<double, float_m::size()>>::type,
                  float_m>,
            float_m::size()>));
    COMPARE(typeid(simdize<tuple<bool, double, bool>, float_m::size() + 1>),
            typeid(SimdizeAdapter<tuple<bool, double, bool>,
                                  tuple<simd_mask_array<float, float_m::size() + 1>,
                                        simdarray<double, float_m::size() + 1>,
                                        simd_mask_array<float, float_m::size() + 1>>,
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

template <typename, int...> struct Foo1 {};
template <typename, typename, unsigned...> struct Foo2 {};
template <typename, typename, typename, std::size_t...> struct Foo3 {};

TEST(nontype_template_parameters)
{
    using namespace std;
    using namespace Vc;
    using std::array;

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
                                  Foo2<int_v, float_v, 3, 5, 6>, int_v::size()>));
    COMPARE(typeid(simdize<Foo3<int, int, float, 3, 5, 6>>),
            typeid(SimdizeAdapter<Foo3<int, int, float, 3, 5, 6>,
                                  Foo3<int_v, int_v, float_v, 3, 5, 6>, int_v::size()>));
}

TEST(tuple_interface)
{
    using V0 = simdize<std::tuple<int, bool>>;
    COMPARE(std::tuple_size<V0>::value, 2u);
    COMPARE(typeid(typename std::tuple_element<0, V0>::type), typeid(int_v));
    COMPARE(typeid(typename std::tuple_element<1, V0>::type), typeid(int_m));

    V0 v;
    COMPARE(typeid(decltype(std::get<0>(v))), typeid(int_v));
    COMPARE(typeid(decltype(std::get<1>(v))), typeid(int_m));
    std::get<0>(v) = int_v::IndexesFromZero();
    COMPARE(std::get<0>(v), int_v::IndexesFromZero());
    std::get<0>(v) += 1;
    COMPARE(std::get<0>(v), int_v::IndexesFromZero() + 1);
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
        T scalar{1, 2, 3};
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
        float_v reference = list.size() - float_v::IndexesFromZero();
        for (; b != e; ++b, reference -= float_v::size()) {
            float_v x = *b;
            COMPARE(x, reference);
            COMPARE(*b, reference);
            *b = x + 1;
            COMPARE(*b, reference + 1);
        }
        reference = list.size() - float_v::IndexesFromZero() + 1;
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
        auto reference1 = list.size() - float_v::IndexesFromZero();
        auto reference2 = list.size() - simdize<unsigned, V::size()>::IndexesFromZero();
        for (; b != e; ++b, reference1 -= V::size(), reference2 -= V::size()) {
            V x = *b;
            COMPARE(x, V(reference1, reference2 * 2, reference1 * 3));
            COMPARE(std::get<0>(*b), reference1);
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
    std::get<0>(v) = V0::IndexesFromZero() + 1;
    std::get<1>(v) = V1::IndexesFromZero() + 1;

    for (int shift = -int(V::size()); shift <= int(V::size()); ++shift) {
        V test = shifted(v, shift);
        COMPARE(std::get<0>(test), (V0::IndexesFromZero() + 1).shifted(shift));
        COMPARE(std::get<1>(test), (V1::IndexesFromZero() + 1).shifted(shift));
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

// vim: foldmethod=marker
