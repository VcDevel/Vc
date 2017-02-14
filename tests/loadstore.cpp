/*  This file is part of the Vc library. {{{
Copyright © 2009-2016 Matthias Kretz <kretz@kde.org>

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

#define WITH_DATAPAR 1
//#define UNITTEST_ONLY_XTEST 1
#include "unittest.h"
#include <Vc/datapar>
#include "make_vec.h"

template <class... Ts> using base_template = Vc::datapar<Ts...>;
#include "testtypes.h"

//operators helpers  //{{{1
template <class T> constexpr T genHalfBits()
{
    return std::numeric_limits<T>::max() >> (std::numeric_limits<T>::digits / 2);
}
template <> constexpr long double genHalfBits<long double>() { return 0; }
template <> constexpr double genHalfBits<double>() { return 0; }
template <> constexpr float genHalfBits<float>() { return 0; }

// is_conversion_undefined {{{1
/* implementation-defined
 * ======================
 * §4.7 p3 (integral conversions)
 *  If the destination type is signed, the value is unchanged if it can be represented in the
 *  destination type (and bit-field width); otherwise, the value is implementation-defined.
 *
 * undefined
 * =========
 * §4.9/1  (floating-point conversions)
 *   If the source value is neither exactly represented in the destination type nor between
 *   two adjacent destination values the result is undefined.
 *
 * §4.10/1 (floating-integral conversions)
 *  floating point type can be converted to integer type.
 *  The behavior is undefined if the truncated value cannot be
 *  represented in the destination type.
 *
 * §4.10/2
 *  integer can be converted to floating point type.
 *  If the value being converted is outside the range of values that can be represented, the
 *  behavior is undefined.
 */
template <typename To, typename From>
constexpr bool is_conversion_undefined_impl(From x, std::true_type)
{
    return x > static_cast<long double>(std::numeric_limits<To>::max()) ||
           x < static_cast<long double>(std::numeric_limits<To>::min());
}

template <typename To, typename From>
constexpr bool is_conversion_undefined_impl(From, std::false_type)
{
    return false;
}

template <typename To, typename From> constexpr bool is_conversion_undefined(From x)
{
    static_assert(std::is_arithmetic<From>::value,
                  "this overload is only meant for builtin arithmetic types");
    return is_conversion_undefined_impl<To, From>(
        x, std::integral_constant<bool, (std::is_floating_point<From>::value &&
                                         (std::is_integral<To>::value ||
                                          (std::is_floating_point<To>::value &&
                                           sizeof(From) > sizeof(To))))>());
}

static_assert(is_conversion_undefined<uint>(float(0x100000000LL)),
              "testing my expectations of is_conversion_undefined");
static_assert(!is_conversion_undefined<float>(llong(0x100000000LL)),
              "testing my expectations of is_conversion_undefined");

template <typename To, typename T, typename A>
inline Vc::mask<T, A> is_conversion_undefined(const Vc::datapar<T, A> &x)
{
    Vc::mask<T, A> k = false;
    for (std::size_t i = 0; i < x.size(); ++i) {
        k[i] = is_conversion_undefined(x[i]);
    }
    return k;
}

// loads & stores {{{1
TEST_TYPES(VU, load_store,
           (outer_product<reduced_test_types,
                          Typelist<long double, double, float, long long, unsigned long,
                                   int, unsigned short, signed char, unsigned long long,
                                   long, unsigned int, short, unsigned char>>))
{
#ifdef Vc_MSVC
// disable "warning C4756: overflow in constant arithmetic" - I don't care.
#pragma warning(disable : 4756)
#endif

    // types, tags, and constants {{{2
    using V = typename VU::template at<0>;
    using U = typename VU::template at<1>;
    using T = typename V::value_type;
    using M = typename V::mask_type;
    auto &&gen = make_vec<V>;
    using Vc::flags::element_aligned;
    using Vc::flags::vector_aligned;
    constexpr size_t alignment = 2 * Vc::memory_alignment_v<V, U>;
#ifdef Vc_MSVC
    using TT = Vc::flags::overaligned_tag<alignment>;
    constexpr TT overaligned = {};
#else
    constexpr auto overaligned = Vc::flags::overaligned<alignment>;
#endif
    const V indexes_from_0 = gen({0, 1, 2, 3}, 4);
    for (std::size_t i = 0; i < V::size(); ++i) {
        COMPARE(indexes_from_0[i], T(i));
    }
    const V indexes_from_1 = gen({1, 2, 3, 4}, 4);
    const V indexes_from_size = gen({T(V::size())}, 1);
    const M alternating_mask = make_mask<M>({0, 1});

    // loads {{{2
    constexpr U min = std::numeric_limits<U>::min();
    constexpr U max = std::numeric_limits<U>::max();
    constexpr U half = genHalfBits<U>();

    auto &&avoid_ub = [](auto x) { return is_conversion_undefined<U>(x) ? U(0) : U(x); };

    const U test_values[] = {U(0xc0000080U),
                             U(0xc0000081U),
                             U(0xc0000082U),
                             U(0xc0000084U),
                             U(0xc0000088U),
                             U(0xc0000090U),
                             U(0xc00000A0U),
                             U(0xc00000C0U),
                             U(0xc000017fU),
                             U(0xc0000180U),
                             U(0x100000001LL),
                             U(0x100000011LL),
                             U(0x100000111LL),
                             U(0x100001111LL),
                             U(0x100011111LL),
                             U(0x100111111LL),
                             U(0x101111111LL),
                             U(-0x100000001LL),
                             U(-0x100000011LL),
                             U(-0x100000111LL),
                             U(-0x100001111LL),
                             U(-0x100011111LL),
                             U(-0x100111111LL),
                             U(-0x101111111LL),
                             U(std::numeric_limits<T>::max() - 1),
                             U(std::numeric_limits<T>::max() * 0.75),
                             min,
                             U(min + 1),
                             U(-1),
                             U(-10),
                             U(-100),
                             U(-1000),
                             U(-10000),
                             U(0),
                             U(1),
                             U(half - 1),
                             half,
                             U(half + 1),
                             U(max - 1),
                             max,
                             U(max - 0xff),
                             U(max / std::pow(2., sizeof(T) * 6 - 1)),
                             avoid_ub(-max / std::pow(2., sizeof(T) * 6 - 1)),
                             U(max / std::pow(2., sizeof(T) * 4 - 1)),
                             avoid_ub(-max / std::pow(2., sizeof(T) * 4 - 1)),
                             U(max / std::pow(2., sizeof(T) * 2 - 1)),
                             avoid_ub(-max / std::pow(2., sizeof(T) * 2 - 1)),
                             U(max - 0xff),
                             U(max - 0x55),
                             U(-(min + 1)),
                             U(-max)};
    constexpr auto test_values_size = sizeof(test_values) / sizeof(U);

    constexpr auto mem_size =
        test_values_size > 3 * V::size() ? test_values_size : 3 * V::size();
    alignas(Vc::memory_alignment_v<V, U> * 2) U mem[mem_size] = {};
    alignas(Vc::memory_alignment_v<V, T> * 2) T reference[mem_size] = {};
    for (std::size_t i = 0; i < test_values_size; ++i) {
        const U value = test_values[i];
        if (is_conversion_undefined<T>(value)) {
            mem[i] = 0;
            reference[i] = 0;
        } else {
            mem[i] = value;
            reference[i] = static_cast<T>(value);
        }
    }
    for (std::size_t i = test_values_size; i < mem_size; ++i) {
        mem[i] = U(i);
        reference[i] = mem[i];
    }

    V x(&mem[V::size()], vector_aligned);
    auto &&compare = [&](const std::size_t offset) {
        static int n = 0;
        const V ref(&reference[offset], element_aligned);
        for (auto i = 0ul; i < V::size(); ++i) {
            if (is_conversion_undefined<T>(mem[i + offset])) {
                continue;
            }
            COMPARE(x[i], reference[i + offset])
                << "\nbefore conversion: " << mem[i + offset]
                << "\n   offset = " << offset
                << "\n        x = " << UnitTest::asBytes(x) << " = " << x
                << "\nreference = " << UnitTest::asBytes(ref) << " = " << ref
                << "\nx == ref  = " << UnitTest::asBytes(x == ref) << " = " << (x == ref)
                << "\ncall no. " << n;
        }
        ++n;
    };
    compare(V::size());
    x = V{mem, overaligned};
    compare(0);
    x = {&mem[1], element_aligned};
    compare(1);

    x.memload(&mem[V::size()], vector_aligned);
    compare(V::size());
    x.memload(&mem[1], element_aligned);
    compare(1);
    x.memload(mem, overaligned);
    compare(0);

    for (std::size_t i = 0; i < mem_size - V::size(); ++i) {
        x.memload(&mem[i], element_aligned);
        compare(i);
    }

    for (std::size_t i = 0; i < test_values_size; ++i) {
        mem[i] = U(i);
    }
    x = indexes_from_0;
    x.memload(&mem[V::size()], alternating_mask, vector_aligned);
    COMPARE(x == indexes_from_size, alternating_mask);
    COMPARE(x == indexes_from_0, !alternating_mask);
    x.memload(&mem[1], alternating_mask, element_aligned);
    COMPARE(x == indexes_from_1, alternating_mask);
    COMPARE(x == indexes_from_0, !alternating_mask);
    x.memload(mem, !alternating_mask, overaligned);
    COMPARE(x == indexes_from_0, !alternating_mask);
    COMPARE(x == indexes_from_1, alternating_mask);

    // stores {{{2
    memset(mem, 0, sizeof(mem));
    x = indexes_from_1;
    x.memstore(&mem[V::size()], vector_aligned);
    std::size_t i = 0;
    for (; i < V::size(); ++i) {
        COMPARE(mem[i], U(0)) << "i: " << i;
    }
    for (; i < 2 * V::size(); ++i) {
        COMPARE(mem[i], U(i - V::size() + 1)) << "i: " << i;
    }
    for (; i < 3 * V::size(); ++i) {
        COMPARE(mem[i], U(0)) << "i: " << i;
    }

    memset(mem, 0, sizeof(mem));
    x.memstore(&mem[1], element_aligned);
    COMPARE(mem[0], U(0));
    for (i = 1; i <= V::size(); ++i) {
        COMPARE(mem[i], U(i));
    }
    for (; i < 3 * V::size(); ++i) {
        COMPARE(mem[i], U(0));
    }

    memset(mem, 0, sizeof(mem));
    x.memstore(mem, overaligned);
    for (i = 0; i < V::size(); ++i) {
        COMPARE(mem[i], U(i + 1));
    }
    for (; i < 3 * V::size(); ++i) {
        COMPARE(mem[i], U(0));
    }

    memset(mem, 0, sizeof(mem));
    indexes_from_0.memstore(&mem[V::size()], alternating_mask, vector_aligned);
    for (i = 0; i < V::size() + 1; ++i) {
        COMPARE(mem[i], U(0));
    }
    for (; i < 2 * V::size(); i += 2) {
        COMPARE(mem[i], U(i - V::size()));
    }
    for (i = V::size() + 2; i < 2 * V::size(); i += 2) {
        COMPARE(mem[i], U(0));
    }
    for (; i < 3 * V::size(); ++i) {
        COMPARE(mem[i], U(0));
    }
}

