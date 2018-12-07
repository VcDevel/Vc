/*  This file is part of the Vc library. {{{
Copyright © 2018 Matthias Kretz <kretz@kde.org>

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

#include <experimental/simd>
#include <initializer_list>
#include <random>

template <class T, class A>
std::experimental::simd<T, A> iif(std::experimental::simd_mask<T, A> k,
                   const typename std::experimental::simd_mask<T, A>::simd_type &t,
                   const std::experimental::simd<T, A> &f)
{
    auto r = f;
    where(k, r) = t;
    return r;
}

template <class V>
V epilogue_load(const typename V::value_type *mem, const std::size_t size)
{
    const int rem = size % V::size();
    return where(V([](int i) { return i; }) < rem, V(0))
        .copy_from(mem + size / V::size() * V::size(), std::experimental::element_aligned);
}

template <class V, class... F>
void test_values(const std::initializer_list<typename V::value_type> &inputs,
                 F &&... fun_pack)
{
    for (auto it = inputs.begin(); it + V::size() <= inputs.end(); it += V::size()) {
        [](auto...) {}((fun_pack(V(&it[0], std::experimental::element_aligned)), 0)...);
    }
    [](auto...) {}((fun_pack(epilogue_load<V>(inputs.begin(), inputs.size())), 0)...);
}

template <class V> struct RandomValues {
    const std::size_t count;
    const typename V::value_type min;
    const typename V::value_type max;
};

static std::mt19937 g_mt_gen{0};

template <class V, class... F>
void test_values(const std::initializer_list<typename V::value_type> &inputs,
                 const RandomValues<V> &random, F &&... fun_pack)
{
    test_values<V>(inputs, fun_pack...);

    using T = typename V::value_type;
    std::conditional_t<std::is_floating_point_v<T>, std::uniform_real_distribution<T>,
                       std::uniform_int_distribution<T>>
        dist(random.min, random.max);
    auto &&rnd_v = [&]() {
        V tmp;
        for (std::size_t i = 0; i < V::size(); ++i) {
            tmp[i] = dist(g_mt_gen);
        }
        return tmp;
    };

    for (size_t i = 0; i < (random.count + V::size() - 1) / V::size(); ++i) {
        [](auto...) {}((fun_pack(rnd_v()), 0)...);
    }
}

template <class V, class... F>
void test_values_2arg(const std::initializer_list<typename V::value_type> &inputs,
                      F &&... fun_pack)
{
    for (auto scalar_it = inputs.begin(); scalar_it != inputs.end(); ++scalar_it) {
        for (auto it = inputs.begin(); it + V::size() <= inputs.end(); it += V::size()) {
            [](auto...) {
            }((fun_pack(V(&it[0], std::experimental::element_aligned), V(*scalar_it)), 0)...);
        }
        [](auto...) {
        }((fun_pack(epilogue_load<V>(inputs.begin(), inputs.size()), V(*scalar_it)),
           0)...);
    }
}

template <class V, class... F>
void test_values_2arg(const std::initializer_list<typename V::value_type> &inputs,
                      const RandomValues<V> &random, F &&... fun_pack)
{
    test_values_2arg<V>(inputs, fun_pack...);

    using T = typename V::value_type;
    std::conditional_t<std::is_floating_point_v<T>, std::uniform_real_distribution<T>,
                       std::uniform_int_distribution<T>>
        dist(random.min, random.max);
    auto &&rnd_v = [&]() {
        V tmp;
        for (std::size_t i = 0; i < V::size(); ++i) {
            tmp[i] = dist(g_mt_gen);
        }
        return tmp;
    };

    for (size_t i = 0; i < (random.count + V::size() - 1) / V::size(); ++i) {
        [](auto...) {}((fun_pack(rnd_v(), rnd_v()), 0)...);
    }
}

template <class V, class... F>
void test_values_3arg(const std::initializer_list<typename V::value_type> &inputs,
                      F &&... fun_pack)
{
    for (auto scalar_it1 = inputs.begin(); scalar_it1 != inputs.end(); ++scalar_it1) {
        for (auto scalar_it2 = inputs.begin(); scalar_it2 != inputs.end(); ++scalar_it2) {
            for (auto it = inputs.begin(); it + V::size() <= inputs.end();
                 it += V::size()) {
                [](auto...) {}((fun_pack(V(&it[0], std::experimental::element_aligned), V(*scalar_it1),
                                         V(*scalar_it2)),
                                0)...);
            }
            [](auto...) {}((fun_pack(epilogue_load<V>(inputs.begin(), inputs.size()),
                                     V(*scalar_it1), V(*scalar_it2)),
                            0)...);
        }
    }
}

template <class V, class... F>
void test_values_3arg(const std::initializer_list<typename V::value_type> &inputs,
                      const RandomValues<V> &random, F &&... fun_pack)
{
    test_values_3arg<V>(inputs, fun_pack...);

    using T = typename V::value_type;
    std::conditional_t<std::is_floating_point_v<T>, std::uniform_real_distribution<T>,
                       std::uniform_int_distribution<T>>
        dist(random.min, random.max);
    auto &&rnd_v = [&]() {
        V tmp;
        for (std::size_t i = 0; i < V::size(); ++i) {
            tmp[i] = dist(g_mt_gen);
        }
        return tmp;
    };

    for (size_t i = 0; i < (random.count + V::size() - 1) / V::size(); ++i) {
        [](auto...) {}((fun_pack(rnd_v(), rnd_v(), rnd_v()), 0)...);
    }
}

#define MAKE_TESTER_2(name_, reference_)                                                 \
    [&](const auto... inputs) {                                                          \
        /*_GLIBCXX_SIMD_DEBUG()("testing " #name_ "(", input, ")");*/                    \
        const auto totest = name_(inputs...);                                            \
        using R           = std::remove_const_t<decltype(totest)>;                       \
        auto&& expected   = [&](const auto&... vs) -> const R {                          \
            R tmp = {};                                                                  \
            for (std::size_t i = 0; i < R::size(); ++i) {                                \
                tmp[i] = reference_(vs[i]...);                                           \
            }                                                                            \
            return tmp;                                                                  \
        };                                                                               \
        const R expect1 = expected(inputs...);                                           \
        if constexpr (std::is_floating_point_v<typename R::value_type>) {                \
            ((COMPARE(isnan(totest), isnan(expect1)) << #name_ "(") << ... << inputs)    \
                << ") = " << totest << " != " << expect1;                                \
            ((FUZZY_COMPARE(name_(iif(isnan(expect1), 0, inputs)...),                    \
                            expected(iif(isnan(expect1), 0, inputs)...))                 \
              << "\nclean = ")                                                           \
             << ... << iif(isnan(expect1), 0, inputs));                                  \
        } else {                                                                         \
            ((COMPARE(name_(inputs...), expect1) << "\ninputs = ") << ... << inputs);    \
        }                                                                                \
    }

#define MAKE_TESTER(name_) MAKE_TESTER_2(name_, std::name_)

// vim: foldmethod=marker
