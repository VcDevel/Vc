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

//#define Vc_DEBUG logarithm
//#define UNITTEST_ONLY_XTEST 1
#include <vir/test.h>
#include <Vc/simd>
#include <Vc/math>
#include "metahelpers.h"
#include <cmath>    // abs & sqrt
#include <cstdlib>  // integer abs
#include <random>
#include "mathreference.h"
#include "simd_view.h"

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
void test_values(const std::initializer_list<typename V::value_type> &inputs,
                 F &&... fun_pack)
{
    for (auto it = inputs.begin(); it + V::size() <= inputs.end(); it += V::size()) {
        [](auto...) {}((fun_pack(V(&it[0], Vc::flags::element_aligned)), 0)...);
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

    // the below noise could be written as a small lambda:
    // [&](int) { return dist(g_mt_gen); }
    // but GCC miscompiles it
    class my_lambda_t {
        std::uniform_real_distribution<typename V::value_type> dist;

    public:
        typename V::value_type operator()(int) { return dist(g_mt_gen); }
        my_lambda_t(typename V::value_type min, typename V::value_type max)
            : dist(min, max)
        {
        }
    } rnd_gen(random.min, random.max);

    for (size_t i = 0; i < (random.count + V::size() - 1) / V::size(); ++i) {
        [](auto...) {}((fun_pack(V(rnd_gen)), 0)...);
    }
}

#define MAKE_TESTER(name_)                                                               \
    [](V input) {                                                                        \
        /*Vc_DEBUG()("testing " #name_ "(", input, ")");*/                               \
        const V expected([&](auto i) {                                                   \
            using std::name_;                                                            \
            return name_(input[i]);                                                      \
        });                                                                              \
        const V totest = name_(input);                                                   \
        COMPARE(isnan(totest), isnan(expected))                                          \
            << #name_ "(" << input << ") = " << totest << " != " << expected;            \
        where(isnan(expected), input) = 0;                                               \
        FUZZY_COMPARE(name_(input), V([&](auto i) {                                      \
                          using std::name_;                                              \
                          return name_(input[i]);                                        \
                      }))                                                                \
            << "\ninput = " << input;                                                    \
    }

template <size_t Offset, class V, class Iterator> V test_tuples_gather(const Iterator &it)
{
    return V([&](auto i) { return it[i][Offset]; });
}

template <size_t Offset, class V, class Iterator>
V test_tuples_gather_epilogue(const Iterator &it, const size_t remaining)
{
    return V([&](auto i) { return it[i < remaining ? i : 0][Offset]; });
}

template <class V, size_t N, size_t... Indexes, class... F>
void test_tuples_impl(
    std::index_sequence<Indexes...>,
    const std::initializer_list<std::array<typename V::value_type, N>> &data,
    F &&... fun_pack)
{
    auto it = data.begin();
    for (; it + V::size() <= data.end(); it += V::size()) {
        [](auto...) {}((fun_pack(test_tuples_gather<Indexes, V>(it)...), 0)...);
    }
    const auto remaining = data.size() % V::size();
    if (remaining > 0) {
        [](auto...) {}((fun_pack(test_tuples_gather_epilogue<Indexes, V>(
                            it, data.size() % V::size())...),
                        0)...);
    }
}

template <class V, size_t N, class... F>
void test_tuples(const std::initializer_list<std::array<typename V::value_type, N>> &data,
                 F &&... fun_pack)
{
    test_tuples_impl<V, N>(std::make_index_sequence<N>(), data,
                           std::forward<F>(fun_pack)...);
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
    Vc::experimental::simd_view<V>(testdata).for_each(
        [&](const V input, const V expected, const V) {
            FUZZY_COMPARE(sin(input), expected) << " input = " << input;
            FUZZY_COMPARE(sin(-input), -expected) << " input = " << input;
        });
}

TEST_TYPES(V, testCos, real_test_types)  //{{{1
{
    using std::cos;
    using T = typename V::value_type;

    vir::test::setFuzzyness<float>(2);
    vir::test::setFuzzyness<double>(1e7);

    const auto &testdata = referenceData<function::sincos, T>();
    Vc::experimental::simd_view<V>(testdata).for_each(
        [&](const V input, const V, const V expected) {
            FUZZY_COMPARE(cos(input), expected) << " input = " << input;
            FUZZY_COMPARE(cos(-input), expected) << " input = " << input;
        });
}

TEST_TYPES(V, testAsin, real_test_types)  //{{{1
{
    using std::asin;
    using T = typename V::value_type;

    vir::test::setFuzzyness<float>(2);
    vir::test::setFuzzyness<double>(36);

    const auto &testdata = referenceData<function::asin, T>();
    Vc::experimental::simd_view<V>(testdata).for_each(
        [&](const V input, const V expected) {
            FUZZY_COMPARE(asin(input), expected) << " input = " << input;
            FUZZY_COMPARE(asin(-input), -expected) << " input = " << input;
        });
}

TEST_TYPES(V, testAtan, real_test_types)  //{{{1
{
    using std::atan;
    using T = typename V::value_type;

    vir::test::setFuzzyness<float>(1);
    vir::test::setFuzzyness<double>(1);

    using limits = std::numeric_limits<typename V::value_type>;
    test_values<V>({limits::quiet_NaN(), limits::infinity(), -limits::infinity()},
                   [](V input) {
                       const V expected([&](auto i) { return std::atan(input[i]); });
                       COMPARE(isnan(atan(input)), isnan(expected));
                       where(isnan(input), input) = 0;
                       COMPARE(atan(input),
                               V([&](auto i) { return std::atan(input[i]); }));
                   });

    const auto &testdata = referenceData<function::atan, T>();
    Vc::experimental::simd_view<V>(testdata).for_each(
        [&](const V input, const V expected) {
            FUZZY_COMPARE(atan(input), expected) << " input = " << input;
            FUZZY_COMPARE(atan(-input), -expected) << " input = " << input;
        });
}

TEST_TYPES(V, testAtan2, real_test_types)  //{{{1
{
    using std::atan2;
    using T = typename V::value_type;

    vir::test::setFuzzyness<float>(3);
    vir::test::setFuzzyness<double>(2);

    using limits = std::numeric_limits<typename V::value_type>;
    const T Pi   = Vc::detail::double_const<1, 0x921fb54442d18ull,  1>;
    const T inf  = limits::infinity();
    test_tuples<V, 3>(
        {
            // If y is +0 (-0) and x is less than 0, +pi (-pi) is returned.
            {+0., -3., +Pi},
            {-0., -3., -Pi},
            // If y is +0 (-0) and x is greater than 0, +0 (-0) is returned.
            {+0., +3., +0.},
            {-0., +3., -0.},
            // If y is less than 0 and x is +0 or -0, -pi/2 is returned.
            {-3., +0., -Pi / 2},
            {-3., -0., -Pi / 2},
            // If y is greater than 0 and x is +0 or -0, pi/2 is returned.
            {+3., +0., +Pi / 2},
            {+3., -0., +Pi / 2},
            // If y is +0 (-0 and x is -0, +pi (-pi is returned.
            {+0., -0., +Pi},
            {-0., -0., -Pi},
            // If y is +0 (-0 and x is +0, +0 (-0 is returned.
            {+0., +0., +0.},
            {-0., +0., -0.},
            // If y is a finite value greater (less than 0, and x is negative infinity,
            // +pi
            // (-pi is returned.
            {+1., -inf, +Pi},
            {-1., -inf, -Pi},
            // If y is a finite value greater (less than 0, and x is positive infinity, +0
            // (-0 is returned.
            {+3., +inf, +0.},
            {-3., +inf, -0.},
            // If y is positive infinity (negative infinity, and x is finite, pi/2 (-pi/2
            // is
            // returned.
            {+inf, +3., +Pi / 2},
            {-inf, +3., -Pi / 2},
            {+inf, -3., +Pi / 2},
            {-inf, -3., -Pi / 2},
#ifndef _WIN32  // the Microsoft implementation of atan2 fails this test
            // If y is positive infinity (negative infinity) and x is negative infinity,
            // +3*pi/4 (-3*pi/4) is returned.
            {+inf, -inf, T(+3. * (Pi / 4))},
            {-inf, -inf, T(-3. * (Pi / 4))},
            // If y is positive infinity (negative infinity) and x is positive infinity,
            // +pi/4 (-pi/4) is returned.
            {+inf, +inf, +Pi / 4},
            {-inf, +inf, -Pi / 4},
#endif
            // If either x or y is NaN, a NaN is returned.
            {limits::quiet_NaN(), 3., limits::quiet_NaN()},
            {3., limits::quiet_NaN(), limits::quiet_NaN()},
            {limits::quiet_NaN(), limits::quiet_NaN(), limits::quiet_NaN()},
        },
        [](V x, V y, V expected) {
            COMPARE(isnan(atan2(x, y)), isnan(expected)) << x << y;
            where(isnan(expected), x) = 0;
            where(isnan(expected), y) = 0;
            where(isnan(expected), expected) = 0;
            COMPARE(atan2(x, y), expected) << x << y;
            COMPARE(signbit(atan2(x, y)), signbit(expected)) << x << y;
        });

    for (int xoffset = -100; xoffset < 54613; xoffset += 47 * int(V::size())) {
        for (int yoffset = -100; yoffset < 54613; yoffset += 47 * int(V::size())) {
            const V data = V([](T i) { return i; });
            const V reference = V([&](auto i) {
                return std::atan2((data[i] + xoffset) * T(0.15),
                                  (data[i] + yoffset) * T(0.15));
            });

            const V x = (data + xoffset) * T(0.15);
            const V y = (data + yoffset) * T(0.15);
            FUZZY_COMPARE(atan2(x, y), reference) << ", x = " << x << ", y = " << y;
        }
    }
}

TEST_TYPES(V, testTrig, real_test_types)  //{{{1
{
    vir::test::setFuzzyness<float>(1);
    vir::test::setFuzzyness<double>(1);

    using limits = std::numeric_limits<typename V::value_type>;
    test_values<V>(
        {limits::quiet_NaN(), limits::infinity(), -limits::infinity(), +0., -0.,
         limits::denorm_min(), limits::min(), limits::max(), limits::min() / 3},
        {10000, -limits::max()/2, limits::max()/2},
        MAKE_TESTER(acos),
        MAKE_TESTER(tan),
        MAKE_TESTER(acosh),
        MAKE_TESTER(asinh),
        MAKE_TESTER(atanh),
        MAKE_TESTER(cosh),
        MAKE_TESTER(sinh),
        MAKE_TESTER(tanh)
        );
}

TEST_TYPES(V, testLog, real_test_types)  //{{{1
{
    vir::test::setFuzzyness<float>(1);
    vir::test::setFuzzyness<double>(1);

    using limits = std::numeric_limits<typename V::value_type>;
    test_values<V>(
        {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
         3, 5, 7, 15, 17, 31, 33, 63, 65,
         limits::quiet_NaN(), limits::infinity(), -limits::infinity(), +0., -0.,
         limits::denorm_min(), limits::min(), limits::max(), limits::min() / 3,
         -limits::denorm_min(), -limits::min(), -limits::max(), -limits::min() / 3},
        {10000, -limits::max() / 2, limits::max() / 2},
        MAKE_TESTER(log),
        MAKE_TESTER(log10),
        MAKE_TESTER(log1p),
        MAKE_TESTER(log2),
        MAKE_TESTER(logb)
        );
}

TEST_TYPES(V, testExp, real_test_types)  //{{{1
{
    vir::test::setFuzzyness<float>(1);
    vir::test::setFuzzyness<double>(1);

    using limits = std::numeric_limits<typename V::value_type>;
    test_values<V>(
        {limits::quiet_NaN(), limits::infinity(), -limits::infinity(), +0., -0.,
         limits::denorm_min(), limits::min(), limits::max(), limits::min() / 3},
        {10000, -limits::max()/2, limits::max()/2},
        MAKE_TESTER(exp),
        MAKE_TESTER(exp2),
        MAKE_TESTER(expm1)
        );
}

