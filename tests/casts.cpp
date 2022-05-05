/*  This file is part of the Vc library. {{{
Copyright © 2010-2015 Matthias Kretz <kretz@kde.org>

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

#include "Vc/vector.h"
#include "unittest.h"
#include <limits>
#include <algorithm>

using namespace Vc;

// ExtraImplVectors {{{1
using ExtraImplVectors = vir::Typelist<
#ifdef Vc_IMPL_Scalar
#elif defined Vc_IMPL_AVX2
    Vc::Scalar::int_v, Vc::Scalar::ushort_v, Vc::Scalar::double_v, Vc::Scalar::uint_v,
    Vc::Scalar::short_v, Vc::Scalar::float_v, Vc::SSE::int_v, Vc::SSE::ushort_v,
    Vc::SSE::double_v, Vc::SSE::uint_v, Vc::SSE::short_v, Vc::SSE::float_v,
    Vc::AVX::int_v, Vc::AVX::ushort_v, Vc::AVX::double_v, Vc::AVX::uint_v,
    Vc::AVX::short_v, Vc::AVX::float_v
#elif defined Vc_IMPL_AVX
    Vc::Scalar::int_v, Vc::Scalar::ushort_v, Vc::Scalar::double_v, Vc::Scalar::uint_v,
    Vc::Scalar::short_v, Vc::Scalar::float_v, Vc::SSE::int_v, Vc::SSE::ushort_v,
    Vc::SSE::double_v, Vc::SSE::uint_v, Vc::SSE::short_v, Vc::SSE::float_v
#else
    Vc::Scalar::int_v, Vc::Scalar::ushort_v, Vc::Scalar::double_v, Vc::Scalar::uint_v,
    Vc::Scalar::short_v, Vc::Scalar::float_v
#endif
    >;

// AllTestTypes {{{1
#ifdef Vc_DEFAULT_TYPES
using AllTestTypes = vir::outer_product<AllVectors, AllVectors>;
#elif defined Vc_EXTRA_TYPES
using AllTestTypes = vir::concat<vir::outer_product<AllVectors, ExtraImplVectors>,
                                 vir::outer_product<ExtraImplVectors, AllVectors>>;
#elif defined Vc_FROM_N
#ifdef Vc_TO_N
using AllTestTypes = vir::outer_product<SimdArrays<Vc_FROM_N>, SimdArrays<Vc_TO_N>>;
#else
using AllTestTypes = vir::outer_product<SimdArrays<Vc_FROM_N>, AllVectors>;
#endif
#elif defined Vc_TO_N
using AllTestTypes = vir::outer_product<AllVectors, SimdArrays<Vc_TO_N>>;
#endif

// is_conversion_undefined {{{1
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
template <typename To, typename From>
typename std::enable_if<
    (std::is_arithmetic<From>::value && std::is_floating_point<From>::value &&
     std::is_integral<To>::value),
    bool>::type
is_conversion_undefined(From x)
{
    if (x > static_cast<From>(std::numeric_limits<To>::max()) ||
        x < static_cast<From>(std::numeric_limits<To>::min())) {
        return true;
    }
    return false;
}
template <typename To, typename From>
typename std::enable_if<
    (std::is_arithmetic<From>::value &&
     !(std::is_floating_point<From>::value && std::is_integral<To>::value)),
    bool>::type
is_conversion_undefined(From)
{
    return false;
}

template <typename To, typename From>
typename std::enable_if<Vc::is_simd_vector<From>::value, typename From::Mask>::type
    is_conversion_undefined(const From x)
{
    typename From::Mask k = false;
    for (std::size_t i = 0; i < From::Size; ++i) {
        k[i] = is_conversion_undefined(x[i]);
    }
}

// ith_scalar {{{1
template <typename V> inline typename V::EntryType ith_scalar(std::size_t i, const V &x)
{
    return x[i];
}
template <typename V, typename... Vs, typename = Vc::enable_if<(sizeof...(Vs) > 0)>>
inline typename V::EntryType ith_scalar(std::size_t i, const V &x, const Vs &... xs)
{
    return i < V::Size ? x[i] : ith_scalar(i - V::Size, xs...);
}
// extraInformation {{{1
static void doNothing(const std::initializer_list<void *> &) {}
template <typename To, typename V0, typename... Vs>
std::string extraInformation(const V0 &arg0, const Vs &... args)
{
    std::stringstream s;
    s << "\nsimd_cast<" << vir::typeToString<To>() << ">(" << std::setprecision(20) << arg0;
    doNothing({&(s << ", " << args)...});
    s << ')';
    return s.str();
}
// rnd {{{1
template <typename To, typename From>
static Vc::enable_if<(Vc::Traits::is_floating_point<From>::value), From> rnd()
{
    using T = typename From::value_type;
    auto r = (From::Random() - T(0.5)) *
             T(std::numeric_limits<typename To::value_type>::max());
    r.setZero(isnan(r));
    return r;
}
template <typename To, typename From>
static Vc::enable_if<(!Vc::Traits::is_floating_point<From>::value), From> rnd()
{
    return From::Random();
}
// cast_vector_impl {{{1
template <typename To, typename From, typename... Froms>
Vc::enable_if<(To::Size <= sizeof...(Froms) * From::Size), void> cast_vector_impl(
    const From &, const Froms &...)
{
}

template <typename To, typename From, typename... Froms>
Vc::enable_if<(To::Size > sizeof...(Froms) * From::Size), void> cast_vector_impl(
    const From &x0, const Froms &... xs)
{
    using T = typename To::EntryType;
    auto result = simd_cast<To>(x0, xs...);
#ifdef Vc_GCC
    // workaround for https://gcc.gnu.org/bugzilla/show_bug.cgi?id=47226
    // parameter pack expansion does not work inside lambda
    typename From::EntryType input[(1 + sizeof...(Froms)) * From::Size];
    for (std::size_t i = 0; i < (1 + sizeof...(Froms)) * From::Size; ++i) {
        input[i] = ith_scalar(i, x0, xs...);
    }
#endif
    const To reference = To::generate([&](std::size_t i) {
        if (i >= (1 + sizeof...(Froms)) * From::Size) {
            return T(0);
        }
#ifdef Vc_GCC
        if (is_conversion_undefined<T>(input[i])) {
            result[i] = 0;
            return T(0);
        }
        return static_cast<T>(input[i]);
#else
        const auto input = ith_scalar(i, x0, xs...);
        if (is_conversion_undefined<T>(input)) {
            result[i] = 0;
            return T(0);
        }
        return static_cast<T>(input);
#endif
    });

    COMPARE(result, reference) << extraInformation<To>(x0, xs...);
    cast_vector_impl<To>(x0, rnd<To, From>(), xs...);
}
// cast_vector_split {{{1
template <typename To, typename From, std::size_t Index = 0>
Vc::enable_if<!(Index * To::Size < From::Size && To::Size < From::Size), void>
    cast_vector_split(const From &)
{
}
template <typename To, typename From, std::size_t Index = 0>
Vc::enable_if<(Index * To::Size < From::Size && To::Size < From::Size), void>
    cast_vector_split(const From &x)
{
    using T = typename To::EntryType;
    const auto result = simd_cast<To, Index>(x);
    const To reference = To::generate([&](std::size_t i) {
        if (i + Index * To::Size >= From::Size) {
            return T(0);
        }
        const auto input = x[i + Index * To::Size];
        return is_conversion_undefined<T>(input) ? result[i] : static_cast<T>(input);
    });

    COMPARE(result, reference) << "simd_cast<" << vir::typeToString<To>() << ", "
                               << Index << ">(" << x << ')';

    cast_vector_split<To, From, Index + 1>(x);
}
TEST_TYPES(TList, cast_vector, AllTestTypes)  // {{{1
{
    using From = typename TList::template at<0>;
    using To = typename TList::template at<1>;
    using T = typename From::EntryType;

    alignas(From) T testData[21 + 2 * From::Size] = {
        T(0xc0000080u),
        T(0xc0000081u),
        T(0xc000017fu),
        T(0xc0000180u),
        std::numeric_limits<T>::min(),
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
        T(-std::numeric_limits<T>::max())};
    rnd<To, From>().store(&testData[21], Vc::Unaligned);
    for (std::size_t i = 0; i < 21 + From::Size; i += From::Size) {
        const From v(&testData[i],
                     Vc::Unaligned);  // Unaligned because From can be SimdArray<T, Odd>
        cast_vector_impl<To>(v);
        cast_vector_split<To>(v);
    }
    cast_vector_split<To>(rnd<To, From>());
}
// mask_cast_1 {{{1
template <typename To, typename From> void mask_cast_1(const From &mask)
{
    To casted = simd_cast<To>(mask);
    std::size_t i = 0;
    for (; i < std::min(To::Size, From::Size); ++i) {
        COMPARE(casted[i], mask[i]) << "i: " << i << ", " << mask << " got converted to "
                                    << vir::typeToString<To>() << ": " << casted;
    }
    for (; i < To::Size; ++i) {
        COMPARE(casted[i], false) << "i: " << i << ", " << mask << " got converted to "
                                  << vir::typeToString<To>() << ": " << casted;
    }
}
// mask_cast_2 {{{1
template <typename To, typename From>
void mask_cast_2(const From &mask0, const From &mask1,
                 Vc::enable_if<(To::Size > From::Size)> = Vc::nullarg)
{
    To casted = simd_cast<To>(mask0, mask1);
    std::size_t i = 0;
    for (; i < From::Size; ++i) {
        COMPARE(casted[i], mask0[i]) << "i: " << i << mask0 << mask1
                                     << " were converted to "
                                     << vir::typeToString<To>() << ": " << casted;
    }
    for (; i < std::min(To::Size, 2 * From::Size); ++i) {
        COMPARE(casted[i], mask1[i - From::Size])
            << "i: " << i << mask0 << mask1 << " were converted to "
            << vir::typeToString<To>() << ": " << casted;
    }
    for (; i < To::Size; ++i) {
        COMPARE(casted[i], false) << "i: " << i << mask0 << mask1 << " were converted to "
                                  << vir::typeToString<To>() << ": " << casted;
    }
}
template <typename To, typename From>
void mask_cast_2(const From &, const From &,
                 Vc::enable_if<!(To::Size > From::Size)> = Vc::nullarg)
{
}
// mask_cast_4 {{{1
template <typename To, typename From>
void mask_cast_4(const From &mask0, const From &mask1, const From &mask2,
                 const From &mask3,
                 Vc::enable_if<(To::Size > 2 * From::Size)> = Vc::nullarg)
{
    To casted = simd_cast<To>(mask0, mask1, mask2, mask3);
    std::size_t i = 0;
    for (; i < From::Size; ++i) {
        COMPARE(casted[i], mask0[i]) << "i: " << i << mask0 << mask1 << mask2 << mask3
                                     << " were converted to "
                                     << vir::typeToString<To>() << ": " << casted;
    }
    for (; i < std::min(To::Size, 2 * From::Size); ++i) {
        COMPARE(casted[i], mask1[i - From::Size])
            << "i: " << i << mask0 << mask1 << mask2 << mask3 << " were converted to "
            << vir::typeToString<To>() << ": " << casted;
    }
    for (; i < std::min(To::Size, 3 * From::Size); ++i) {
        COMPARE(casted[i], mask2[i - 2 * From::Size])
            << "i: " << i << mask0 << mask1 << mask2 << mask3 << " were converted to "
            << vir::typeToString<To>() << ": " << casted;
    }
    for (; i < std::min(To::Size, 4 * From::Size); ++i) {
        COMPARE(casted[i], mask3[i - 3 * From::Size])
            << "i: " << i << mask0 << mask1 << mask2 << mask3 << " were converted to "
            << vir::typeToString<To>() << ": " << casted;
    }
    for (; i < To::Size; ++i) {
        COMPARE(casted[i], false) << "i: " << i << mask0 << mask1 << mask2 << mask3
                                  << " were converted to " << vir::typeToString<To>()
                                  << ": " << casted;
    }
}
template <typename To, typename From>
void mask_cast_4(const From &, const From &, const From &, const From &,
                 Vc::enable_if<!(To::Size > 2 * From::Size)> = Vc::nullarg)
{
}
// cast_mask_split {{{1
template <typename To, typename From, std::size_t Index = 0>
Vc::enable_if<!(Index * To::Size < From::Size && To::Size < From::Size), void>
    cast_mask_split(const From &)
{
}
template <typename To, typename From, std::size_t Index = 0>
Vc::enable_if<(Index * To::Size < From::Size && To::Size < From::Size), void>
    cast_mask_split(const From &x)
{
    const auto result = simd_cast<To, Index>(x);
    const To reference = To::generate([&](std::size_t i) {
        return i + Index * To::Size >= From::Size ? false : x[i + Index * To::Size];
    });

    COMPARE(result, reference) << "simd_cast<" << vir::typeToString<To>() << ", "
                               << Index << ">(" << x << ')';

    cast_mask_split<To, From, Index + 1>(x);
}
TEST_TYPES(TList, cast_mask, AllTestTypes)  // {{{1
{
    using FromV = typename TList::template at<0>;
    using ToV = typename TList::template at<1>;
    using From = typename FromV::Mask;
    using To = typename ToV::Mask;
    std::vector<From> randomMasks(4, From{false});

    withRandomMask<FromV>([&](const From &mask) {
        std::rotate(randomMasks.begin(), randomMasks.begin() + 1, randomMasks.end());
        randomMasks[0] = mask;
        mask_cast_1<To>(randomMasks[0]);
        mask_cast_2<To>(randomMasks[0], randomMasks[1]);
        mask_cast_4<To>(randomMasks[0], randomMasks[1], randomMasks[2], randomMasks[3]);
        cast_mask_split<To>(randomMasks[0]);
    });
}
// }}}1
#ifdef Vc_DEFAULT_TYPES
TEST(fullConversion)/*{{{*/
{
    float_v x = float_v::Random();
    float_v r;
    for (size_t i = 0; i < float_v::Size; i += double_v::Size) {
        float_v tmp = simd_cast<float_v>(0.1 * simd_cast<double_v>(x.shifted(i)));
        r = r.shifted(double_v::Size, tmp);
    }
    for (size_t i = 0; i < float_v::Size; ++i) {
        COMPARE(r[i], static_cast<float>(x[i] * 0.1)) << "i = " << i;
    }
}/*}}}*/
#endif // Vc_DEFAULT_TYPES

TEST_TYPES(V, referenceConstruction, AllTypes)
{
    V a = V::Random();
    V r(a[0]);

    for (size_t i = 0; i < V::Size; ++i) {
        COMPARE(r[i], static_cast<float>(a[0])) << "i = " << i;
    }
}

#if 0
/*{{{*/
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
/*}}}*/
#endif

// vim: foldmethod=marker
