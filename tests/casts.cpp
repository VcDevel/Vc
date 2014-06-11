/*  This file is part of the Vc library. {{{
Copyright © 2010-2014 Matthias Kretz <kretz@kde.org>
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
#include <limits>
#include <algorithm>

using namespace Vc;

/* implementation-defined
 * ======================
 * §4.7 p3 (integral conversions)
 *  If the destination type is signed, the value is unchanged if it can be represented in the
 *  destination type (and bit-field width); otherwise, the value is implementation-defined.
 *
 * undefined
 * =========
 * §4.9 p1 (floating-integral conversions)
 *  floating point type can be converted to integer type.
 *  The behavior is undefined if the truncated value cannot be
 *  represented in the destination type.
 *      p2
 *  integer can be converted to floating point type.
 *  If the value being converted is outside the range of values that can be represented, the
 *  behavior is undefined.
 */
template <typename To, typename From> bool is_conversion_undefined(From x)
{
    if (std::is_floating_point<From>::value && std::is_integral<To>::value) {
        if (x > std::numeric_limits<To>::max() || x < std::numeric_limits<To>::min()) {
            return true;
        }
    }
    return false;
}

template <typename To, typename From> void simd_cast_1_impl(const From x)
{
    using T = typename To::EntryType;
    if (is_conversion_undefined<T>(x[0])) {
        return;
    }
    /*
    std::cout << "conversion from " << UnitTest::typeToString<From>() << " to "
              << UnitTest::typeToString<To>() << " value " << x[0] << " -> " << T(x[0]) << " vs. "
              << simd_cast<To>(x)[0] << '\n';
              */
    const To reference = static_cast<T>(x[0]);
    if (To::size() > From::size()) {
        COMPARE(simd_cast<To>(x), reference.shifted(To::size() - From::size()))
            << "casted to " << UnitTest::typeToString<To>()
            << ", reference = " << reference << ", x[0] = " << x[0]
            << ", T(x[0]) = " << T(x[0]);
        ;
    } else {
        COMPARE(simd_cast<To>(x), reference)
            << "casted to " << UnitTest::typeToString<To>() << ", x[0] = " << x[0]
            << ", T(x[0]) = " << T(x[0]);
    }
}

template <typename To, typename From>
void simd_cast_2_impl(const From x,
                      Vc::enable_if<(To::size() > From::size())> = Vc::nullarg)
{
    using T = typename To::EntryType;
    if (is_conversion_undefined<T>(x[0])) {
        return;
    }
    /*
    std::cout << "conversion from " << UnitTest::typeToString<From>() << " to "
              << UnitTest::typeToString<To>() << " value " << x[0] << " -> " << T(x[0]) << " vs. "
              << simd_cast<To>(x)[0] << '\n';
              */
    const To reference = static_cast<T>(x[0]);
    if (To::size() > 2 * From::size()) {
        COMPARE(simd_cast<To>(x, x), reference.shifted(To::size() - 2 * From::size()))
            << "casted to " << UnitTest::typeToString<To>()
            << ", reference = " << reference << ", x[0] = " << x[0]
            << ", T(x[0]) = " << T(x[0]);
        ;
    } else {
        COMPARE(simd_cast<To>(x, x), reference)
            << "casted to " << UnitTest::typeToString<To>() << ", x[0] = " << x[0]
            << ", T(x[0]) = " << T(x[0]);
    }
}

template <typename To, typename From>
void simd_cast_2_impl(const From,
                      Vc::enable_if<(To::size() <= From::size())> = Vc::nullarg)
{
}

template <typename To, typename From>
void simd_cast_4_impl(const From x,
                      Vc::enable_if<(To::size() > 2 * From::size())> = Vc::nullarg)
{
    using T = typename To::EntryType;
    if (is_conversion_undefined<T>(x[0])) {
        return;
    }
    /*
    std::cout << "conversion from " << UnitTest::typeToString<From>() << " to "
              << UnitTest::typeToString<To>() << " value " << x[0] << " -> " << T(x[0]) << " vs. "
              << simd_cast<To>(x)[0] << '\n';
              */
    const To reference = static_cast<T>(x[0]);
    if (To::size() > 4 * From::size()) {
        COMPARE(simd_cast<To>(x, x, x, x),
                reference.shifted(To::size() - 2 * From::size()))
            << "casted to " << UnitTest::typeToString<To>()
            << ", reference = " << reference << ", x[0] = " << x[0]
            << ", T(x[0]) = " << T(x[0]);
        ;
    } else {
        COMPARE(simd_cast<To>(x, x, x, x), reference)
            << "casted to " << UnitTest::typeToString<To>() << ", x[0] = " << x[0]
            << ", T(x[0]) = " << T(x[0]);
    }
}

template <typename To, typename From>
void simd_cast_4_impl(const From,
                      Vc::enable_if<(To::size() <= 2 * From::size())> = Vc::nullarg)
{
}

template <typename To, typename From>
void simd_cast_to2_impl(const From x,
                        Vc::enable_if<(2 * To::size() <= From::size())> = Vc::nullarg)
{
    using T = typename To::EntryType;
    if (is_conversion_undefined<T>(x[0])) {
        return;
    }
    /*
    std::cout << "conversion from " << UnitTest::typeToString<From>() << " to "
              << UnitTest::typeToString<To>() << " value " << x[0] << " -> " << T(x[0]) << " vs. "
              << simd_cast<To>(x)[0] << '\n';
              */
    const To reference = static_cast<T>(x[0]);
    COMPARE((simd_cast<To, 0>(x)), reference)
        << "casted to " << UnitTest::typeToString<To>() << ", x[0] = " << x[0]
        << ", T(x[0]) = " << T(x[0]);
    COMPARE((simd_cast<To, 1>(x)), reference)
        << "casted to " << UnitTest::typeToString<To>() << ", x[0] = " << x[0]
        << ", T(x[0]) = " << T(x[0]);
}

template <typename To, typename From>
void simd_cast_to2_impl(const From,
                        Vc::enable_if<(2 * To::size() > From::size())> = Vc::nullarg)
{
}

template <typename To, typename From>
void simd_cast_to4_impl(const From x,
                        Vc::enable_if<(4 * To::size() <= From::size())> = Vc::nullarg)
{
    using T = typename To::EntryType;
    if (is_conversion_undefined<T>(x[0])) {
        return;
    }
    /*
    std::cout << "conversion from " << UnitTest::typeToString<From>() << " to "
              << UnitTest::typeToString<To>() << " value " << x[0] << " -> " << T(x[0]) << " vs. "
              << simd_cast<To>(x)[0] << '\n';
              */
    const To reference = static_cast<T>(x[0]);
    COMPARE((simd_cast<To, 0>(x)), reference)
        << "casted to " << UnitTest::typeToString<To>() << ", x[0] = " << x[0]
        << ", T(x[0]) = " << T(x[0]);
    COMPARE((simd_cast<To, 1>(x)), reference)
        << "casted to " << UnitTest::typeToString<To>() << ", x[0] = " << x[0]
        << ", T(x[0]) = " << T(x[0]);
    COMPARE((simd_cast<To, 2>(x)), reference)
        << "casted to " << UnitTest::typeToString<To>() << ", x[0] = " << x[0]
        << ", T(x[0]) = " << T(x[0]);
    COMPARE((simd_cast<To, 3>(x)), reference)
        << "casted to " << UnitTest::typeToString<To>() << ", x[0] = " << x[0]
        << ", T(x[0]) = " << T(x[0]);
}

template <typename To, typename From>
void simd_cast_to4_impl(const From,
                        Vc::enable_if<(4 * To::size() > From::size())> = Vc::nullarg)
{
}

TEST_ALL_V(V, cast_vector)
{
    using T = typename V::EntryType;
    for (T x : {std::numeric_limits<T>::min(),
                T(0),
                T(-1),
                T(1),
                std::numeric_limits<T>::max(),
                T(std::numeric_limits<T>::max() - 1),
                T(std::numeric_limits<T>::max() - 0xff),
                T(std::numeric_limits<T>::max() / std::pow(2., sizeof(T) * 6 - 1)),
                T(-std::numeric_limits<T>::max() / std::pow(2., sizeof(T) * 6 - 1)),
                T(std::numeric_limits<T>::max() / std::pow(2., sizeof(T) * 4 - 1)),
                T(-std::numeric_limits<T>::max() / std::pow(2., sizeof(T) * 4 - 1)),
                T(std::numeric_limits<T>::max() / std::pow(2., sizeof(T) * 2 - 1)),
                T(-std::numeric_limits<T>::max() / std::pow(2., sizeof(T) * 2 - 1)),
                T(std::numeric_limits<T>::max() - 0xff),
                T(std::numeric_limits<T>::max() - 0x55),
                T(-std::numeric_limits<T>::min()),
                T(-std::numeric_limits<T>::max())}) {
        const V v = x;

        simd_cast_1_impl<int_v>(v);
        simd_cast_1_impl<uint_v>(v);
        simd_cast_1_impl<short_v>(v);
        simd_cast_1_impl<ushort_v>(v);
        simd_cast_1_impl<float_v>(v);
        simd_cast_1_impl<double_v>(v);

        simd_cast_2_impl<int_v>(v);
        simd_cast_2_impl<uint_v>(v);
        simd_cast_2_impl<short_v>(v);
        simd_cast_2_impl<ushort_v>(v);
        simd_cast_2_impl<float_v>(v);
        simd_cast_2_impl<double_v>(v);

        simd_cast_4_impl<int_v>(v);
        simd_cast_4_impl<uint_v>(v);
        simd_cast_4_impl<short_v>(v);
        simd_cast_4_impl<ushort_v>(v);
        simd_cast_4_impl<float_v>(v);
        simd_cast_4_impl<double_v>(v);

        simd_cast_to2_impl<int_v>(v);
        simd_cast_to2_impl<uint_v>(v);
        simd_cast_to2_impl<short_v>(v);
        simd_cast_to2_impl<ushort_v>(v);
        simd_cast_to2_impl<float_v>(v);
        simd_cast_to2_impl<double_v>(v);

        simd_cast_to4_impl<int_v>(v);
        simd_cast_to4_impl<uint_v>(v);
        simd_cast_to4_impl<short_v>(v);
        simd_cast_to4_impl<ushort_v>(v);
        simd_cast_to4_impl<float_v>(v);
        simd_cast_to4_impl<double_v>(v);
    }
}

template <typename To, typename From> void mask_cast_1(From mask)
{
    To casted = simd_cast<To>(mask);
    std::size_t i = 0;
    for (; i < std::min(To::Size, From::Size); ++i) {
        COMPARE(casted[i], mask[i]) << "i: " << i << ", " << mask << " got converted to "
                                    << UnitTest::typeToString<To>() << ": " << casted;
    }
    for (; i < To::Size; ++i) {
        COMPARE(casted[i], false) << "i: " << i << ", " << mask << " got converted to "
                                  << UnitTest::typeToString<To>() << ": " << casted;
    }
}
template <typename To, typename From>
void mask_cast_2(const From mask0,
                 const From mask1,
                 Vc::enable_if<(To::size() > From::size())> = Vc::nullarg)
{
    To casted = simd_cast<To>(mask0, mask1);
    std::size_t i = 0;
    for (; i < From::Size; ++i) {
        COMPARE(casted[i], mask0[i]) << "i: " << i << mask0 << mask1
                                     << " were converted to "
                                     << UnitTest::typeToString<To>() << ": " << casted;
    }
    for (; i < std::min(To::Size, 2 * From::Size); ++i) {
        COMPARE(casted[i], mask1[i - From::Size])
            << "i: " << i << mask0 << mask1 << " were converted to "
            << UnitTest::typeToString<To>() << ": " << casted;
    }
    for (; i < To::Size; ++i) {
        COMPARE(casted[i], false) << "i: " << i << mask0 << mask1 << " were converted to "
                                  << UnitTest::typeToString<To>() << ": " << casted;
    }
}
template <typename To, typename From>
void mask_cast_2(const From,
                 const From,
                 Vc::enable_if<!(To::size() > From::size())> = Vc::nullarg)
{
}
template <typename To, typename From>
void mask_cast_4(const From mask0,
                 const From mask1,
                 const From mask2,
                 const From mask3,
                 Vc::enable_if<(To::size() > 2 * From::size())> = Vc::nullarg)
{
    To casted = simd_cast<To>(mask0, mask1, mask2, mask3);
    std::size_t i = 0;
    for (; i < From::Size; ++i) {
        COMPARE(casted[i], mask0[i]) << "i: " << i << mask0 << mask1 << mask2 << mask3
                                     << " were converted to "
                                     << UnitTest::typeToString<To>() << ": " << casted;
    }
    for (; i < std::min(To::Size, 2 * From::Size); ++i) {
        COMPARE(casted[i], mask1[i - From::Size])
            << "i: " << i << mask0 << mask1 << mask2 << mask3 << " were converted to "
            << UnitTest::typeToString<To>() << ": " << casted;
    }
    for (; i < std::min(To::Size, 3 * From::Size); ++i) {
        COMPARE(casted[i], mask2[i - 2 * From::Size])
            << "i: " << i << mask0 << mask1 << mask2 << mask3 << " were converted to "
            << UnitTest::typeToString<To>() << ": " << casted;
    }
    for (; i < std::min(To::Size, 4 * From::Size); ++i) {
        COMPARE(casted[i], mask3[i - 3 * From::Size])
            << "i: " << i << mask0 << mask1 << mask2 << mask3 << " were converted to "
            << UnitTest::typeToString<To>() << ": " << casted;
    }
    for (; i < To::Size; ++i) {
        COMPARE(casted[i], false) << "i: " << i << mask0 << mask1 << mask2 << mask3
                                  << " were converted to " << UnitTest::typeToString<To>()
                                  << ": " << casted;
    }
}
template <typename To, typename From>
void mask_cast_4(const From,
                 const From,
                 const From,
                 const From,
                 Vc::enable_if<!(To::size() > 2 * From::size())> = Vc::nullarg)
{
}
template <typename To, typename From>
void mask_cast_0_5(const From mask,
                   Vc::enable_if<(To::size() < From::size())> = Vc::nullarg)
{
    const To casted0 = simd_cast<To, 0>(mask);
    const To casted1 = simd_cast<To, 1>(mask);
    std::size_t i = 0;
    for (; i < To::Size; ++i) {
        COMPARE(casted0[i], mask[i]) << "i: " << i << ", " << mask << " got converted to "
                                     << UnitTest::typeToString<To>() << ": " << casted0
                                     << casted1;
    }
    for (; i < std::min(2 * To::Size, From::Size); ++i) {
        COMPARE(casted1[i - To::Size], mask[i])
            << "i: " << i << ", " << mask << " got converted to "
            << UnitTest::typeToString<To>() << ": " << casted0 << casted1;
    }
    for (; i < 2 * To::Size; ++i) {
        COMPARE(casted1[i - To::Size], false)
            << "i: " << i << ", " << mask << " got converted to "
            << UnitTest::typeToString<To>() << ": " << casted0 << casted1;
    }
}
template <typename To, typename From>
void mask_cast_0_5(const From, Vc::enable_if<!(To::size() < From::size())> = Vc::nullarg)
{
}

template <typename To, typename From> void mask_cast(const std::vector<From> &masks)
{
    mask_cast_1<To>(masks[0]);
    mask_cast_2<To>(masks[0], masks[1]);
    mask_cast_4<To>(masks[0], masks[1], masks[2], masks[3]);
    mask_cast_0_5<To>(masks[0]);
}

TEST_TYPES(V,
           cast_mask,
           (ALL_VECTORS,
            SIMD_ARRAYS(1),
            SIMD_ARRAYS(2),
            SIMD_ARRAYS(4),
            SIMD_ARRAYS(8),
            SIMD_ARRAYS(16),
            SIMD_ARRAYS(17)))
{
    using M = typename V::Mask;
    std::vector<M> randomMasks(4, M{false});

    UnitTest::withRandomMask<V>([&](M mask) {
        std::rotate(randomMasks.begin(), randomMasks.begin() + 1, randomMasks.end());
        randomMasks[0] = mask;
        mask_cast<int_m>(randomMasks);
        mask_cast<uint_m>(randomMasks);
        mask_cast<short_m>(randomMasks);
        mask_cast<ushort_m>(randomMasks);
        mask_cast<float_m>(randomMasks);
        mask_cast<double_m>(randomMasks);
    });
}
#if 0

template<typename T> constexpr bool may_overflow() { return std::is_integral<T>::value && std::is_unsigned<T>::value; }

template<typename T1, typename T2> struct is_conversion_exact
{
    static constexpr bool is_T2_integer = std::is_integral<T2>::value;
    static constexpr bool is_T2_signed = is_T2_integer && std::is_signed<T2>::value;
    static constexpr bool is_float_int_conversion = std::is_floating_point<T1>::value && is_T2_integer;

    template <typename U, typename V> static constexpr bool can_represent(V x) {
        return x <= std::numeric_limits<U>::max() && x >= std::numeric_limits<U>::min();
    }

    template<typename U> static constexpr U max() { return std::numeric_limits<U>::max() - U(1); }
    template<typename U> static constexpr U min() { return std::numeric_limits<U>::min() + U(1); }

    static constexpr bool for_value(T1 v) {
        return (!is_float_int_conversion && !is_T2_signed) || can_represent<T2>(v);
    }
    static constexpr bool for_plus_one(T1 v) {
        return (v <= max<T1>() || may_overflow<T1>()) && (v <= max<T2>() || may_overflow<T2>()) &&
               for_value(v + 1);
    }
    static constexpr bool for_minus_one(T1 v) {
        return (v >= min<T1>() || may_overflow<T1>()) && (v >= min<T2>() || may_overflow<T2>()) &&
               for_value(v - 1);
    }
};

template<typename V1, typename V2> V2 makeReference(V2 reference)
{
    reference.setZero(V2::IndexesFromZero() >= V1::Size);
    return reference;
}

template<typename V1, typename V2> void testNumber(double n)
{
    typedef typename V1::EntryType T1;
    typedef typename V2::EntryType T2;

    constexpr T1 One = T1(1);

    // compare casts from T1 -> T2 with casts from V1 -> V2

    const T1 n1 = static_cast<T1>(n);
    //std::cerr << "n1 = " << n1 << ", static_cast<T2>(n1) = " << static_cast<T2>(n1) << std::endl;

    if (is_conversion_exact<T1, T2>::for_value(n1)) {
        COMPARE(static_cast<V2>(V1(n1)), makeReference<V1>(V2(static_cast<T1>(n1))))
            << "\n       n1: " << n1
            << "\n   V1(n1): " << V1(n1)
            << "\n   T2(n1): " << T2(n1)
            ;
    }
    if (is_conversion_exact<T1, T2>::for_plus_one(n1)) {
        COMPARE(static_cast<V2>(V1(n1) + One), makeReference<V1>(V2(static_cast<T2>(n1 + One)))) << "\n       n1: " << n1;
    }
    if (is_conversion_exact<T1, T2>::for_minus_one(n1)) {
        COMPARE(static_cast<V2>(V1(n1) - One), makeReference<V1>(V2(static_cast<T2>(n1 - One)))) << "\n       n1: " << n1;
    }
}

template<typename T> double maxHelper()
{
    return static_cast<double>(std::numeric_limits<T>::max());
}

template<> double maxHelper<int>()
{
    const int intDigits = std::numeric_limits<int>::digits;
    const int floatDigits = std::numeric_limits<float>::digits;
    return static_cast<double>(((int(1) << floatDigits) - 1) << (intDigits - floatDigits));
}

template<> double maxHelper<unsigned int>()
{
    const int intDigits = std::numeric_limits<unsigned int>::digits;
    const int floatDigits = std::numeric_limits<float>::digits;
    return static_cast<double>(((unsigned(1) << floatDigits) - 1) << (intDigits - floatDigits));
}

template<typename V1, typename V2> void testCast2()
{
    typedef typename V1::EntryType T1;
    typedef typename V2::EntryType T2;

    const double max = std::min(maxHelper<T1>(), maxHelper<T2>());
    const double min = std::max(
            std::numeric_limits<T1>::is_integer ?
                static_cast<double>(std::numeric_limits<T1>::min()) :
                static_cast<double>(-std::numeric_limits<T1>::max()),
            std::numeric_limits<T2>::is_integer ?
                static_cast<double>(std::numeric_limits<T2>::min()) :
                static_cast<double>(-std::numeric_limits<T2>::max())
                );

    testNumber<V1, V2>(-1.);
    testNumber<V1, V2>(0.);
    testNumber<V1, V2>(0.5);
    testNumber<V1, V2>(1.);
    testNumber<V1, V2>(2.);
    testNumber<V1, V2>(max);
    testNumber<V1, V2>(max / 4 + max / 2);
    testNumber<V1, V2>(max / 2);
    testNumber<V1, V2>(max / 4);
    testNumber<V1, V2>(min);

    V1 test(IndexesFromZero);
    COMPARE(static_cast<V2>(test), makeReference<V1>(V2::IndexesFromZero()));
}

template<typename T> void testCast()
{
    testCast2<typename T::V1, typename T::V2>();
}

#define _CONCAT(A, B) A ## _ ## B
#define CONCAT(A, B) _CONCAT(A, B)
template<typename T1, typename T2>
struct T2Helper
{
    typedef T1 V1;
    typedef T2 V2;
};

void fullConversion()
{
    float_v x = float_v::Random();
    float_v r;
    for (size_t i = 0; i < float_v::Size; i += double_v::Size) {
        float_v tmp = static_cast<float_v>(0.1 * static_cast<double_v>(x.shifted(i)));
        r = r.shifted(double_v::Size, tmp);
    }
    for (size_t i = 0; i < float_v::Size; ++i) {
        COMPARE(r[i], static_cast<float>(x[i] * 0.1)) << "i = " << i;
    }
}

void testmain()
{
#define TEST_CAST(v1, v2) \
    typedef T2Helper<v1, v2> CONCAT(v1, v2); \
    runTest(testCast<CONCAT(v1, v2)>)

    TEST_CAST(double_v, double_v);
    TEST_CAST(double_v,  float_v);
    TEST_CAST(double_v,    int_v);
    TEST_CAST(double_v,   uint_v);
    //TEST_CAST(double_v,  short_v);
    //TEST_CAST(double_v, ushort_v);

    TEST_CAST(float_v, double_v);
    TEST_CAST(float_v,  float_v);
    TEST_CAST(float_v,    int_v);
    TEST_CAST(float_v,   uint_v);
    TEST_CAST(float_v,  short_v);
    TEST_CAST(float_v, ushort_v);

    TEST_CAST(int_v, double_v);
    TEST_CAST(int_v, float_v);
    TEST_CAST(int_v, int_v);
    TEST_CAST(int_v, uint_v);
    TEST_CAST(int_v, short_v);
    TEST_CAST(int_v, ushort_v);

    TEST_CAST(uint_v, double_v);
    TEST_CAST(uint_v, float_v);
    TEST_CAST(uint_v, int_v);
    TEST_CAST(uint_v, uint_v);
    TEST_CAST(uint_v, short_v);
    TEST_CAST(uint_v, ushort_v);

    TEST_CAST(ushort_v, short_v);
    TEST_CAST(ushort_v, ushort_v);

    TEST_CAST(short_v, short_v);
    TEST_CAST(short_v, ushort_v);
#undef TEST_CAST
    runTest(fullConversion);
}
#endif
