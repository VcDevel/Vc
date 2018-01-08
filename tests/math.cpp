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
#include <Vc/simd>
#include <Vc/math>
#include "metahelpers.h"
#include <cmath>    // abs & sqrt
#include <cstdlib>  // integer abs
#include "mathreference.h"

template <class... Ts> using base_template = Vc::simd<Ts...>;
#include "testtypes.h"

template <class V>
V epilogue_load(const typename V::value_type *mem, const std::size_t size)
{
    const int rem = size % V::size();
    return where(V([](int i) { return i; }) < rem, V(0))
        .copy_from(mem + size / V::size() * V::size(), Vc::flags::element_aligned);
}

template <class V, class... F>
void test_values(const std::initializer_list<typename V::value_type> &inputs, F &&... fun_pack)
{
    for (auto it = inputs.begin(); it + V::size() <= inputs.end(); it += V::size()) {
        auto &&tmp = {(fun_pack(V(&it[0], Vc::flags::element_aligned)), 0)...};
        Vc::detail::unused(tmp);
    }
    auto &&tmp = {(fun_pack(epilogue_load<V>(inputs.begin(), inputs.size())), 0)...};
    Vc::detail::unused(tmp);
}

template <class V> void test_abs(std::false_type)
{
    //VERIFY(!(sfinae_is_callable<V &, const int *>(call_memload())));
}
template <class V> void test_abs(std::true_type)
{
    using std::abs;
    using T = typename V::value_type;
    V input([](int i) { return T(-i); });
    V expected([](int i) { return T(std::abs(T(-i))); });
    COMPARE(abs(input), expected);
}

TEST_TYPES(V, abs, all_test_types)  //{{{1
{
    test_abs<V>(std::is_signed<typename V::value_type>());
}

TEST_TYPES(V, testSqrt, real_test_types)  //{{{1
{
    using std::sqrt;
    using T = typename V::value_type;
    V input([](auto i) { return T(i); });
    V expected([](auto i) { return std::sqrt(T(i)); });
    COMPARE(sqrt(input), expected);
}

TEST_TYPES(V, logb, real_test_types)  //{{{1
{
    using std::logb;
    using T = typename V::value_type;
    const V input([](auto i) { return T(i); });
    const V expected([&input](auto i) { return std::logb(input[i]); });
    COMPARE(logb(input), expected);
}

TEST_TYPES(V, fpclassify, real_test_types)  //{{{1
{
    using limits = std::numeric_limits<typename V::value_type>;
    test_values<V>(
        {0., -0., 1., -1., limits::infinity(), -limits::infinity(), limits::max(),
         -limits::max(), limits::min(), limits::min() * 0.9, -limits::min(),
         -limits::min() * 0.9, limits::denorm_min(), -limits::denorm_min(),
         limits::quiet_NaN(), limits::signaling_NaN()},
        [](V input) {
            using intv = Vc::fixed_size_simd<int, V::size()>;
            COMPARE(isfinite(input), !V([&](auto i) { return std::isfinite(input[i]) ? 0 : 1; })) << input;
            COMPARE(isinf(input), !V([&](auto i) { return std::isinf(input[i]) ? 0 : 1; })) << input;
            COMPARE(isnan(input), !V([&](auto i) { return std::isnan(input[i]) ? 0 : 1; })) << input;
            COMPARE(isnormal(input), !V([&](auto i) { return std::isnormal(input[i]) ? 0 : 1; })) << input;
            COMPARE(signbit(input), !V([&](auto i) { return std::signbit(input[i]) ? 0 : 1; })) << input;
            COMPARE((isunordered(input, V())), !V([&](auto i) { return std::isunordered(input[i], 0) ? 0 : 1; })) << input;
            COMPARE((isunordered(V(), input)), !V([&](auto i) { return std::isunordered(0, input[i]) ? 0 : 1; })) << input;
            COMPARE(fpclassify(input), intv([&](auto i) { return std::fpclassify(input[i]); })) << input;
        });
}

TEST_TYPES(V, trunc_ceil_floor, real_test_types)  //{{{1
{
    using limits = std::numeric_limits<typename V::value_type>;
    test_values<V>({2.1,
                    2.0,
                    2.9,
                    2.5,
                    2.499,
                    1.5,
                    1.499,
                    1.99,
                    0.99,
                    0.5,
                    0.499,
                    0.,
                    -2.1,
                    -2.0,
                    -2.9,
                    -2.5,
                    -2.499,
                    -1.5,
                    -1.499,
                    -1.99,
                    -0.99,
                    -0.5,
                    -0.499,
                    -0.,
                    3 << 21,
                    3 << 22,
                    3 << 23,
                    -(3 << 21),
                    -(3 << 22),
                    -(3 << 23),
                    limits::infinity(),
                    -limits::infinity(),
                    limits::denorm_min(),
                    limits::max(),
                    limits::min(),
                    limits::min() * 0.9,
                    limits::lowest(),
                    -limits::denorm_min(),
                    -limits::max(),
                    -limits::min(),
                    -limits::min() * 0.9,
                    -limits::lowest()},
                   [](V input) {
                       const V expected([&](auto i) { return std::trunc(input[i]); });
                       COMPARE(trunc(input), expected) << input;
                   },
                   [](V input) {
                       const V expected([&](auto i) { return std::ceil(input[i]); });
                       COMPARE(ceil(input), expected) << input;
                   },
                   [](V input) {
                       const V expected([&](auto i) { return std::floor(input[i]); });
                       COMPARE(floor(input), expected) << input;
                   });

    test_values<V>({limits::quiet_NaN(), limits::signaling_NaN()},
                   [](V input) {
                       const V expected([&](auto i) { return std::trunc(input[i]); });
                       COMPARE(isnan(trunc(input)), isnan(expected)) << input;
                   },
                   [](V input) {
                       const V expected([&](auto i) { return std::ceil(input[i]); });
                       COMPARE(isnan(ceil(input)), isnan(expected)) << input;
                   },
                   [](V input) {
                       const V expected([&](auto i) { return std::floor(input[i]); });
                       COMPARE(isnan(floor(input)), isnan(expected)) << input;
                   });
}

TEST_TYPES(V, frexp, real_test_types)  //{{{1
{
    using int_v = Vc::fixed_size_simd<int, V::size()>;
    using limits = std::numeric_limits<typename V::value_type>;
    test_values<V>(
        {0,   0.25, 0.5, 1,   3,   4,   6,   7,     8,    9,   10,  11,  12,
         13,  14,   15,  16,  17,  18,  19,  20,    21,   22,  23,  24,  25,
         26,  27,   28,  29,  32,  31,  -0., -0.25, -0.5, -1,  -3,  -4,  -6,
         -7,  -8,   -9,  -10, -11, -12, -13, -14,   -15,  -16, -17, -18, -19,
         -20, -21,  -22, -23, -24, -25, -26, -27,   -28,  -29, -32, -31,
         limits::max(), -limits::max(),
         limits::max() * 0.123, -limits::max() * 0.123,
         limits::denorm_min(), -limits::denorm_min(),
         limits::min() / 2, -limits::min() / 2},
        [](V input) {
            V expectedFraction;
            const int_v expectedExponent([&](auto i) {
                int exp;
                expectedFraction[i] = std::frexp(input[i], &exp);
                return exp;
            });
            int_v exponent;
            const V fraction = frexp(input, &exponent);
            COMPARE(fraction, expectedFraction)
                << ", input = " << input << ", delta: " << fraction - expectedFraction;
            COMPARE(exponent, expectedExponent)
                << "\ninput: " << input << ", fraction: " << fraction;
        });
    test_values<V>(
        // If x is a NaN, a NaN is returned, and the value of *exp is unspecified.
        //
        // If x is positive  infinity  (negative  infinity),  positive  infinity (negative
        // infinity) is returned, and the value of *exp is unspecified.
        {limits::quiet_NaN(),
         limits::infinity(),
         -limits::infinity(),
         limits::quiet_NaN(),
         limits::infinity(),
         -limits::infinity(),
         limits::quiet_NaN(),
         limits::infinity(),
         -limits::infinity(),
         limits::quiet_NaN(),
         limits::infinity(),
         -limits::infinity(),
         limits::quiet_NaN(),
         limits::infinity(),
         -limits::infinity(),
         limits::denorm_min(),
         limits::denorm_min() * 1.72,
         -limits::denorm_min(),
         -limits::denorm_min() * 1.72,
         0.,
         -0.,
         1,
         -1},
        [](V input) {
            const V expectedFraction([&](auto i) {
                int exp;
                return std::frexp(input[i], &exp);
            });
            int_v exponent;
            const V fraction = frexp(input, &exponent);
            COMPARE(isnan(fraction), isnan(expectedFraction))
                << fraction << ", input = " << input
                << ", delta: " << fraction - expectedFraction;
            COMPARE(isinf(fraction), isinf(expectedFraction))
                << fraction << ", input = " << input
                << ", delta: " << fraction - expectedFraction;
            COMPARE(signbit(fraction), signbit(expectedFraction))
                << fraction << ", input = " << input
                << ", delta: " << fraction - expectedFraction;
        });
}

TEST_TYPES(V, testSin, real_test_types)  //{{{1
{
    using std::sin;
    using T = typename V::value_type;

    vir::test::setFuzzyness<float>(2);
    vir::test::setFuzzyness<double>(1e7);

    const auto &testdata = referenceData<function::sincos, T>();
    V input, expected;
    unsigned i = 0;
    for (const auto &entry : testdata) {
        input[i % V::size()] = entry.x;
        expected[i % V::size()] = entry.s;
        ++i;
        if (i % V::size() == 0) {
            COMPARE(sin(input), expected)
                << " input = " << input << ", i = " << i;
            COMPARE(sin(-input), -expected)
                << " input = " << input << ", i = " << i;
        }
    }
}
