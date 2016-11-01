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
#include "unittest.h"
#include <Vc/datapar>

using schar = signed char;
using uchar = unsigned char;
using llong = long long;
using ullong = unsigned long long;

using vschar = Vc::datapar<schar>;
using vuchar = Vc::datapar<uchar>;
using vshort = Vc::datapar<short>;
using vushort = Vc::datapar<ushort>;
using vint = Vc::datapar<int>;
using vuint = Vc::datapar<uint>;
using vlong = Vc::datapar<long>;
using vulong = Vc::datapar<ulong>;
using vllong = Vc::datapar<llong>;
using vullong = Vc::datapar<ullong>;
using vfloat = Vc::datapar<float>;
using vdouble = Vc::datapar<double>;
using vldouble = Vc::datapar<long double>;

template <typename T>
using v8 = Vc::datapar<T, Vc::datapar_abi::fixed_size<vschar::size()>>;
template <typename T>
using v16 = Vc::datapar<T, Vc::datapar_abi::fixed_size<vshort::size()>>;
template <typename T>
using v32 = Vc::datapar<T, Vc::datapar_abi::fixed_size<vint::size()>>;
template <typename T>
using v64 = Vc::datapar<T, Vc::datapar_abi::fixed_size<vllong::size()>>;
template <typename T>
using vl = typename std::conditional<sizeof(long) == sizeof(llong), v64<T>, v32<T>>::type;

// all_test_types / ALL_TYPES {{{1
typedef expand_list<Typelist<
#ifdef Vc_HAVE_FULL_AVX_ABI
                        Template<Vc::datapar, Vc::datapar_abi::avx>,
#endif
#ifdef Vc_HAVE_FULL_SSE_ABI
                        Template<Vc::datapar, Vc::datapar_abi::sse>,
#endif
                        Template<Vc::datapar, Vc::datapar_abi::scalar>,
                        Template<Vc::datapar, Vc::datapar_abi::fixed_size<2>>,
                        Template<Vc::datapar, Vc::datapar_abi::fixed_size<3>>,
                        Template<Vc::datapar, Vc::datapar_abi::fixed_size<4>>,
                        Template<Vc::datapar, Vc::datapar_abi::fixed_size<8>>,
                        Template<Vc::datapar, Vc::datapar_abi::fixed_size<12>>,
                        Template<Vc::datapar, Vc::datapar_abi::fixed_size<16>>,
                        Template<Vc::datapar, Vc::datapar_abi::fixed_size<32>>>,
                    Typelist<long double, double, float, long long, unsigned long, int,
                             unsigned short, signed char, unsigned long long, long,
                             unsigned int, short, unsigned char>> all_test_types;

#define ALL_TYPES (all_test_types)

// reduced_test_types {{{1
typedef expand_list<Typelist<
#ifdef Vc_HAVE_FULL_AVX_ABI
                        Template<Vc::datapar, Vc::datapar_abi::avx>,
#endif
#ifdef Vc_HAVE_FULL_SSE_ABI
                        Template<Vc::datapar, Vc::datapar_abi::sse>,
#endif
                        Template<Vc::datapar, Vc::datapar_abi::scalar>>,
                    Typelist<long double, double, float, long long, unsigned long, int,
                             unsigned short, signed char, unsigned long long, long,
                             unsigned int, short, unsigned char>> reduced_test_types;

// datapar generator function {{{1
template <class M> inline M make_mask(const std::initializer_list<bool> &init)
{
    std::size_t i = 0;
    M r;
    for (;;) {
        for (bool x : init) {
            r[i] = x;
            if (++i == M::size()) {
                return r;
            }
        }
    }
}

template <class V>
inline V make_vec(const std::initializer_list<typename V::value_type> &init,
                  typename V::value_type inc = 0)
{
    std::size_t i = 0;
    V r;
    typename V::value_type base = 0;
    for (;;) {
        for (auto x : init) {
            r[i] = base + x;
            if (++i == V::size()) {
                return r;
            }
        }
        base += inc;
    }
}

TEST_TYPES(V, broadcast, ALL_TYPES)  //{{{1
{
    using T = typename V::value_type;
    VERIFY(Vc::is_datapar_v<V>);

    {
        V x;      // default broadcasts 0
        x = V{};  // default broadcasts 0
        x = V();  // default broadcasts 0
        x = x;
        for (std::size_t i = 0; i < V::size(); ++i) {
            COMPARE(x[i], T(0));
        }
    }

    V x = 3;
    V y = 0;
    for (std::size_t i = 0; i < V::size(); ++i) {
        COMPARE(x[i], T(3));
        COMPARE(y[i], T(0));
    }
    y = 3;
    COMPARE(x, y);
}

//operators helpers  //{{{1
template <class T> constexpr T genHalfBits()
{
    return std::numeric_limits<T>::max() >> (std::numeric_limits<T>::digits / 2);
}
template <> constexpr long double genHalfBits<long double>() { return 0; }
template <> constexpr double genHalfBits<double>() { return 0; }
template <> constexpr float genHalfBits<float>() { return 0; }

TEST_TYPES(V, operators, ALL_TYPES)  //{{{1
{
    using M = typename V::mask_type;
    using T = typename V::value_type;
    constexpr auto min = std::numeric_limits<T>::min();
    constexpr auto max = std::numeric_limits<T>::max();
    {  // compares{{{2
        constexpr T half = genHalfBits<T>();
        for (T lo_ : {min, T(min + 1), T(-1), T(0), T(1), T(half - 1), half, T(half + 1),
                      T(max - 1)}) {
            for (T hi_ : {T(min + 1), T(-1), T(0), T(1), T(half - 1), half, T(half + 1),
                          T(max - 1), max}) {
                if (hi_ <= lo_) {
                    continue;
                }
                for (std::size_t pos = 0; pos < V::size(); ++pos) {
                    V lo = lo_;
                    V hi = hi_;
                    lo[pos] = 0;  // have a different value in the vector in case
                    hi[pos] = 1;  // this affects neighbors
                    COMPARE(hi, hi);
                    VERIFY(all_of(hi != lo)) << "hi: " << hi << ", lo: " << lo;
                    VERIFY(all_of(lo != hi)) << "hi: " << hi << ", lo: " << lo;
                    VERIFY(none_of(hi != hi)) << "hi: " << hi << ", lo: " << lo;
                    VERIFY(none_of(hi == lo)) << "hi: " << hi << ", lo: " << lo;
                    VERIFY(none_of(lo == hi)) << "hi: " << hi << ", lo: " << lo;
                    VERIFY(all_of(lo < hi)) << "hi: " << hi << ", lo: " << lo;
                    VERIFY(none_of(hi < lo)) << "hi: " << hi << ", lo: " << lo;
                    VERIFY(none_of(hi <= lo)) << "hi: " << hi << ", lo: " << lo;
                    VERIFY(all_of(hi <= hi)) << "hi: " << hi << ", lo: " << lo;
                    VERIFY(all_of(hi > lo)) << "hi: " << hi << ", lo: " << lo;
                    VERIFY(none_of(lo > hi)) << "hi: " << hi << ", lo: " << lo;
                    VERIFY(all_of(hi >= lo)) << "hi: " << hi << ", lo: " << lo;
                    VERIFY(all_of(hi >= hi)) << "hi: " << hi << ", lo: " << lo;
                }
            }
        }
    }
    {  // subscripting{{{2
        V x = max;
        for (std::size_t i = 0; i < V::size(); ++i) {
            COMPARE(x[i], max);
            x[i] = 0;
        }
        COMPARE(x, V{0});
        for (std::size_t i = 0; i < V::size(); ++i) {
            COMPARE(x[i], T(0));
            x[i] = max;
        }
        COMPARE(x, V{max});
    }
    {  // negation{{{2
        V x = 0;
        COMPARE(!x, M{true});
        V y = 1;
        COMPARE(!y, M{false});
    }
}

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
inline typename std::enable_if<(std::is_arithmetic<From>::value &&
                                std::is_floating_point<From>::value &&
                                std::is_integral<To>::value),
                               bool>::type
is_conversion_undefined(From x)
{
    return x > static_cast<From>(std::numeric_limits<To>::max()) ||
           x < static_cast<From>(std::numeric_limits<To>::min());
}
template <typename To, typename From>
inline typename std::enable_if<(std::is_arithmetic<From>::value &&
                                !(std::is_floating_point<From>::value &&
                                  std::is_integral<To>::value)),
                               bool>::type is_conversion_undefined(From)
{
    return false;
}

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
    // types, tags, and constants {{{2
    using V = typename VU::template at<0>;
    using U = typename VU::template at<1>;
    using T = typename V::value_type;
    using M = typename V::mask_type;
    auto &&gen = make_vec<V>;
    using Vc::flags::element_aligned;
    using Vc::flags::vector_aligned;
    constexpr auto overaligned = Vc::flags::overaligned<Vc::memory_alignment<V, U> * 2>;
    const V indexes_from_0 = gen({0, 1, 2, 3}, 4);
    for (std::size_t i = 0; i < V::size(); ++i) {
        COMPARE(indexes_from_0[i], T(i));
    }
    const V indexes_from_1 = gen({1, 2, 3, 4}, 4);
    const V indexes_from_size = gen({V::size()}, 1);
    const M alternating_mask = make_mask<M>({0, 1});

    // loads {{{2
    constexpr U min = std::numeric_limits<U>::min();
    constexpr U max = std::numeric_limits<U>::max();
    constexpr U half = genHalfBits<U>();

    const U test_values[] = {U(0xc0000080u),
                             U(0xc0000081u),
                             U(0xc000017fu),
                             U(0xc0000180u),
                             min,
                             U(min + 1),
                             U(-1),
                             U(0),
                             U(1),
                             U(half - 1),
                             half,
                             U(half + 1),
                             U(max - 1),
                             max,
                             U(max - 0xff),
                             U(max / std::pow(2., sizeof(T) * 6 - 1)),
                             U(-max / std::pow(2., sizeof(T) * 6 - 1)),
                             U(max / std::pow(2., sizeof(T) * 4 - 1)),
                             U(-max / std::pow(2., sizeof(T) * 4 - 1)),
                             U(max / std::pow(2., sizeof(T) * 2 - 1)),
                             U(-max / std::pow(2., sizeof(T) * 2 - 1)),
                             U(max - 0xff),
                             U(max - 0x55),
                             U(-min),
                             U(-max)};
    constexpr auto test_values_size = sizeof(test_values) / sizeof(U);

    constexpr auto mem_size =
        test_values_size > 3 * V::size() ? test_values_size : 3 * V::size();
    alignas(Vc::memory_alignment<V, U> * 2) U mem[mem_size] = {};
    alignas(Vc::memory_alignment<V, T> * 2) T reference[mem_size] = {};
    for (std::size_t i = 0; i < test_values_size; ++i) {
        mem[i] = test_values[i];
        reference[i] = static_cast<T>(mem[i]);
    }
    for (std::size_t i = test_values_size; i < mem_size; ++i) {
        mem[i] = U(i);
        reference[i] = mem[i];
    }

    V x(&mem[V::size()], vector_aligned);
    auto &&compare = [&](const std::size_t offset) {
        for (auto i = 0ul; i < V::size(); ++i) {
            if (is_conversion_undefined<T>(mem[i + offset])) {
                continue;
            }
            V ref(&reference[offset], element_aligned);
            COMPARE(x[i], reference[i + offset])
                << "\nbefore conversion: " << mem[i + offset]
                << "\n   offset = " << offset
                << "\n        x = " << UnitTest::asBytes(x) << " = " << x
                << "\nreference = " << UnitTest::asBytes(ref) << " = " << ref;
        }
    };
    compare(V::size());
    x = {&mem[1], element_aligned};
    compare(1);
    x = V{mem, overaligned};
    compare(0);

    x.copy_from(&mem[V::size()], vector_aligned);
    compare(V::size());
    x.copy_from(&mem[1], element_aligned);
    compare(1);
    x.copy_from(mem, overaligned);
    compare(0);

    for (std::size_t i = 0; i < mem_size - V::size(); ++i) {
        x.copy_from(&mem[i], element_aligned);
        compare(i);
    }

    for (std::size_t i = 0; i < test_values_size; ++i) {
        mem[i] = U(i);
    }
    x = indexes_from_0;
    x.copy_from(&mem[V::size()], alternating_mask, vector_aligned);
    COMPARE(x == indexes_from_size, alternating_mask);
    COMPARE(x == indexes_from_0, !alternating_mask);
    x.copy_from(&mem[1], alternating_mask, element_aligned);
    COMPARE(x == indexes_from_1, alternating_mask);
    COMPARE(x == indexes_from_0, !alternating_mask);
    x.copy_from(mem, !alternating_mask, overaligned);
    COMPARE(x == indexes_from_0, !alternating_mask);
    COMPARE(x == indexes_from_1, alternating_mask);

    // stores {{{2
    memset(mem, 0, sizeof(mem));
    x = indexes_from_1;
    x.copy_to(&mem[V::size()], vector_aligned);
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

    /*
    memset(mem, 0, sizeof(mem));
    x.copy_to(&mem[1], element_aligned);
    COMPARE(mem[0], U(0));
    for (i = 1; i <= V::size(); ++i) {
        COMPARE(mem[i], U(i));
    }
    for (; i < 3 * V::size(); ++i) {
        COMPARE(mem[i], U(0));
    }

    memset(mem, 0, sizeof(mem));
    x.copy_to(mem, overaligned);
    for (i = 0; i < V::size(); ++i) {
        COMPARE(mem[i], U(i + 1));
    }
    for (; i < 3 * V::size(); ++i) {
        COMPARE(mem[i], U(0));
    }

    memset(mem, 0, sizeof(mem));
    indexes_from_0.copy_to(&mem[V::size()], alternating_mask, vector_aligned);
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
    */
}

template <class A, class B, class Expected> void binary_op_return_type()  //{{{1
{
    const auto name = typeToString<A>() + " + " + typeToString<B>();
    COMPARE(typeid(A() + B()), typeid(Expected)) << name;
    COMPARE(typeid(B() + A()), typeid(Expected)) << name;
    UnitTest::ADD_PASS() << name;
}

XTEST(operator_conversions)  //{{{1
{
    // float{{{2
    binary_op_return_type<vfloat, vfloat, vfloat>();
    binary_op_return_type<vfloat, schar, vfloat>();
    binary_op_return_type<vfloat, uchar, vfloat>();
    binary_op_return_type<vfloat, short, vfloat>();
    binary_op_return_type<vfloat, ushort, vfloat>();
    binary_op_return_type<vfloat, int, vfloat>();
    binary_op_return_type<vfloat, uint, vfloat>();
    binary_op_return_type<vfloat, long, vfloat>();
    binary_op_return_type<vfloat, ulong, vfloat>();
    binary_op_return_type<vfloat, llong, vfloat>();
    binary_op_return_type<vfloat, ullong, vfloat>();
    binary_op_return_type<vfloat, float, vfloat>();
    binary_op_return_type<vfloat, double, v32<double>>();
    binary_op_return_type<vfloat, v32<schar>, vfloat>();
    binary_op_return_type<vfloat, v32<uchar>, vfloat>();
    binary_op_return_type<vfloat, v32<short>, vfloat>();
    binary_op_return_type<vfloat, v32<ushort>, vfloat>();
    binary_op_return_type<vfloat, v32<int>, vfloat>();
    binary_op_return_type<vfloat, v32<uint>, vfloat>();
    binary_op_return_type<vfloat, v32<long>, vfloat>();
    binary_op_return_type<vfloat, v32<ulong>, vfloat>();
    binary_op_return_type<vfloat, v32<llong>, vfloat>();
    binary_op_return_type<vfloat, v32<ullong>, vfloat>();
    binary_op_return_type<vfloat, v32<float>, vfloat>();
    binary_op_return_type<vfloat, v32<double>, v32<double>>();

    binary_op_return_type<v32<float>, vfloat, vfloat>();
    binary_op_return_type<v32<float>, v32<float>, v32<float>>();
    binary_op_return_type<v32<float>, schar, v32<float>>();
    binary_op_return_type<v32<float>, uchar, v32<float>>();
    binary_op_return_type<v32<float>, short, v32<float>>();
    binary_op_return_type<v32<float>, ushort, v32<float>>();
    binary_op_return_type<v32<float>, int, v32<float>>();
    binary_op_return_type<v32<float>, uint, v32<float>>();
    binary_op_return_type<v32<float>, long, v32<float>>();
    binary_op_return_type<v32<float>, ulong, v32<float>>();
    binary_op_return_type<v32<float>, llong, v32<float>>();
    binary_op_return_type<v32<float>, ullong, v32<float>>();
    binary_op_return_type<v32<float>, float, v32<float>>();
    binary_op_return_type<v32<float>, double, v32<double>>();
    binary_op_return_type<v32<float>, v32<schar>, v32<float>>();
    binary_op_return_type<v32<float>, v32<uchar>, v32<float>>();
    binary_op_return_type<v32<float>, v32<short>, v32<float>>();
    binary_op_return_type<v32<float>, v32<ushort>, v32<float>>();
    binary_op_return_type<v32<float>, v32<int>, v32<float>>();
    binary_op_return_type<v32<float>, v32<uint>, v32<float>>();
    binary_op_return_type<v32<float>, v32<long>, v32<float>>();
    binary_op_return_type<v32<float>, v32<ulong>, v32<float>>();
    binary_op_return_type<v32<float>, v32<llong>, v32<float>>();
    binary_op_return_type<v32<float>, v32<ullong>, v32<float>>();
    binary_op_return_type<v32<float>, v32<float>, v32<float>>();
    binary_op_return_type<v32<float>, v32<double>, v32<double>>();

    // double{{{2
    binary_op_return_type<vdouble, vdouble, vdouble>();
    binary_op_return_type<vdouble, schar, vdouble>();
    binary_op_return_type<vdouble, uchar, vdouble>();
    binary_op_return_type<vdouble, short, vdouble>();
    binary_op_return_type<vdouble, ushort, vdouble>();
    binary_op_return_type<vdouble, int, vdouble>();
    binary_op_return_type<vdouble, uint, vdouble>();
    binary_op_return_type<vdouble, long, vdouble>();
    binary_op_return_type<vdouble, ulong, vdouble>();
    binary_op_return_type<vdouble, llong, vdouble>();
    binary_op_return_type<vdouble, ullong, vdouble>();
    binary_op_return_type<vdouble, float, vdouble>();
    binary_op_return_type<vdouble, double, vdouble>();
    binary_op_return_type<vdouble, v64<schar>, vdouble>();
    binary_op_return_type<vdouble, v64<uchar>, vdouble>();
    binary_op_return_type<vdouble, v64<short>, vdouble>();
    binary_op_return_type<vdouble, v64<ushort>, vdouble>();
    binary_op_return_type<vdouble, v64<int>, vdouble>();
    binary_op_return_type<vdouble, v64<uint>, vdouble>();
    binary_op_return_type<vdouble, v64<long>, vdouble>();
    binary_op_return_type<vdouble, v64<ulong>, vdouble>();
    binary_op_return_type<vdouble, v64<llong>, vdouble>();
    binary_op_return_type<vdouble, v64<ullong>, vdouble>();
    binary_op_return_type<vdouble, v64<float>, vdouble>();
    binary_op_return_type<vdouble, v64<double>, vdouble>();

    binary_op_return_type<v64<double>, vdouble, vdouble>();
    binary_op_return_type<v64<double>, v64<float>, v64<double>>();
    binary_op_return_type<v64<double>, schar, v64<double>>();
    binary_op_return_type<v64<double>, uchar, v64<double>>();
    binary_op_return_type<v64<double>, short, v64<double>>();
    binary_op_return_type<v64<double>, ushort, v64<double>>();
    binary_op_return_type<v64<double>, int, v64<double>>();
    binary_op_return_type<v64<double>, uint, v64<double>>();
    binary_op_return_type<v64<double>, long, v64<double>>();
    binary_op_return_type<v64<double>, ulong, v64<double>>();
    binary_op_return_type<v64<double>, llong, v64<double>>();
    binary_op_return_type<v64<double>, ullong, v64<double>>();
    binary_op_return_type<v64<double>, float, v64<double>>();
    binary_op_return_type<v64<double>, double, v64<double>>();
    binary_op_return_type<v64<double>, v64<schar>, v64<double>>();
    binary_op_return_type<v64<double>, v64<uchar>, v64<double>>();
    binary_op_return_type<v64<double>, v64<short>, v64<double>>();
    binary_op_return_type<v64<double>, v64<ushort>, v64<double>>();
    binary_op_return_type<v64<double>, v64<int>, v64<double>>();
    binary_op_return_type<v64<double>, v64<uint>, v64<double>>();
    binary_op_return_type<v64<double>, v64<long>, v64<double>>();
    binary_op_return_type<v64<double>, v64<ulong>, v64<double>>();
    binary_op_return_type<v64<double>, v64<llong>, v64<double>>();
    binary_op_return_type<v64<double>, v64<ullong>, v64<double>>();
    binary_op_return_type<v64<double>, v64<float>, v64<double>>();
    binary_op_return_type<v64<double>, v64<double>, v64<double>>();

    binary_op_return_type<v32<double>, schar, v32<double>>();
    binary_op_return_type<v32<double>, uchar, v32<double>>();
    binary_op_return_type<v32<double>, short, v32<double>>();
    binary_op_return_type<v32<double>, ushort, v32<double>>();
    binary_op_return_type<v32<double>, int, v32<double>>();
    binary_op_return_type<v32<double>, uint, v32<double>>();
    binary_op_return_type<v32<double>, long, v32<double>>();
    binary_op_return_type<v32<double>, ulong, v32<double>>();
    binary_op_return_type<v32<double>, llong, v32<double>>();
    binary_op_return_type<v32<double>, ullong, v32<double>>();
    binary_op_return_type<v32<double>, float, v32<double>>();
    binary_op_return_type<v32<double>, double, v32<double>>();

    // long{{{2
    binary_op_return_type<vlong, vlong, vlong>();
    binary_op_return_type<vlong, vulong, vulong>();
    binary_op_return_type<vlong, schar, vlong>();
    binary_op_return_type<vlong, uchar, vlong>();
    binary_op_return_type<vlong, short, vlong>();
    binary_op_return_type<vlong, ushort, vlong>();
    binary_op_return_type<vlong, int, vlong>();
    binary_op_return_type<vlong, uint, Vc::datapar<decltype(long() + uint())>>();
    binary_op_return_type<vlong, long, vlong>();
    binary_op_return_type<vlong, ulong, vulong>();
    binary_op_return_type<vlong, llong, vl<llong>>();
    binary_op_return_type<vlong, ullong, vl<ullong>>();
    binary_op_return_type<vlong, float, vl<float>>();
    binary_op_return_type<vlong, double, vl<double>>();
    binary_op_return_type<vlong, vl<schar>, vlong>();
    binary_op_return_type<vlong, vl<uchar>, vlong>();
    binary_op_return_type<vlong, vl<short>, vlong>();
    binary_op_return_type<vlong, vl<ushort>, vlong>();
    binary_op_return_type<vlong, vl<int>, vlong>();
    binary_op_return_type<vlong, vl<uint>, Vc::datapar<decltype(long() + uint())>>();
    binary_op_return_type<vlong, vl<long>, vlong>();
    binary_op_return_type<vlong, vl<ulong>, vulong>();
    binary_op_return_type<vlong, vl<llong>, vl<llong>>();
    binary_op_return_type<vlong, vl<ullong>, vl<ullong>>();
    binary_op_return_type<vlong, vl<float>, vl<float>>();
    binary_op_return_type<vlong, vl<double>, vl<double>>();

    binary_op_return_type<vl<long>, vlong, vlong>();
    binary_op_return_type<vl<long>, vulong, vulong>();
    binary_op_return_type<v32<long>, schar, v32<long>>();
    binary_op_return_type<v32<long>, uchar, v32<long>>();
    binary_op_return_type<v32<long>, short, v32<long>>();
    binary_op_return_type<v32<long>, ushort, v32<long>>();
    binary_op_return_type<v32<long>, int, v32<long>>();
    binary_op_return_type<v32<long>, uint, v32<decltype(long() + uint())>>();
    binary_op_return_type<v32<long>, long, v32<long>>();
    binary_op_return_type<v32<long>, ulong, v32<ulong>>();
    binary_op_return_type<v32<long>, llong, v32<llong>>();
    binary_op_return_type<v32<long>, ullong, v32<ullong>>();
    binary_op_return_type<v32<long>, float, v32<float>>();
    binary_op_return_type<v32<long>, double, v32<double>>();
    binary_op_return_type<v32<long>, v32<schar>, v32<long>>();
    binary_op_return_type<v32<long>, v32<uchar>, v32<long>>();
    binary_op_return_type<v32<long>, v32<short>, v32<long>>();
    binary_op_return_type<v32<long>, v32<ushort>, v32<long>>();
    binary_op_return_type<v32<long>, v32<int>, v32<long>>();
    binary_op_return_type<v32<long>, v32<uint>, v32<decltype(long() + uint())>>();
    binary_op_return_type<v32<long>, v32<long>, v32<long>>();
    binary_op_return_type<v32<long>, v32<ulong>, v32<ulong>>();
    binary_op_return_type<v32<long>, v32<llong>, v32<llong>>();
    binary_op_return_type<v32<long>, v32<ullong>, v32<ullong>>();
    binary_op_return_type<v32<long>, v32<float>, v32<float>>();
    binary_op_return_type<v32<long>, v32<double>, v32<double>>();

    binary_op_return_type<v64<long>, schar, v64<long>>();
    binary_op_return_type<v64<long>, uchar, v64<long>>();
    binary_op_return_type<v64<long>, short, v64<long>>();
    binary_op_return_type<v64<long>, ushort, v64<long>>();
    binary_op_return_type<v64<long>, int, v64<long>>();
    binary_op_return_type<v64<long>, uint, v64<decltype(long() + uint())>>();
    binary_op_return_type<v64<long>, long, v64<long>>();
    binary_op_return_type<v64<long>, ulong, v64<ulong>>();
    binary_op_return_type<v64<long>, llong, v64<llong>>();
    binary_op_return_type<v64<long>, ullong, v64<ullong>>();
    binary_op_return_type<v64<long>, float, v64<float>>();
    binary_op_return_type<v64<long>, double, v64<double>>();
    binary_op_return_type<v64<long>, v64<schar>, v64<long>>();
    binary_op_return_type<v64<long>, v64<uchar>, v64<long>>();
    binary_op_return_type<v64<long>, v64<short>, v64<long>>();
    binary_op_return_type<v64<long>, v64<ushort>, v64<long>>();
    binary_op_return_type<v64<long>, v64<int>, v64<long>>();
    binary_op_return_type<v64<long>, v64<uint>, v64<decltype(long() + uint())>>();
    binary_op_return_type<v64<long>, v64<long>, v64<long>>();
    binary_op_return_type<v64<long>, v64<ulong>, v64<ulong>>();
    binary_op_return_type<v64<long>, v64<llong>, v64<llong>>();
    binary_op_return_type<v64<long>, v64<ullong>, v64<ullong>>();
    binary_op_return_type<v64<long>, v64<float>, v64<float>>();
    binary_op_return_type<v64<long>, v64<double>, v64<double>>();

    // ulong{{{2
    binary_op_return_type<vulong, vlong, vulong>();
    binary_op_return_type<vulong, vulong, vulong>();
    binary_op_return_type<vulong, schar, vulong>();
    binary_op_return_type<vulong, uchar, vulong>();
    binary_op_return_type<vulong, short, vulong>();
    binary_op_return_type<vulong, ushort, vulong>();
    binary_op_return_type<vulong, int, vulong>();
    binary_op_return_type<vulong, uint, vulong>();
    binary_op_return_type<vulong, long, vulong>();
    binary_op_return_type<vulong, ulong, vulong>();
    binary_op_return_type<vulong, llong, vl<decltype(ulong() + llong())>>();
    binary_op_return_type<vulong, ullong, vl<ullong>>();
    binary_op_return_type<vulong, float, vl<float>>();
    binary_op_return_type<vulong, double, vl<double>>();
    binary_op_return_type<vulong, vl<schar>, vulong>();
    binary_op_return_type<vulong, vl<uchar>, vulong>();
    binary_op_return_type<vulong, vl<short>, vulong>();
    binary_op_return_type<vulong, vl<ushort>, vulong>();
    binary_op_return_type<vulong, vl<int>, vulong>();
    binary_op_return_type<vulong, vl<uint>, vulong>();
    binary_op_return_type<vulong, vl<long>, vulong>();
    binary_op_return_type<vulong, vl<ulong>, vulong>();
    binary_op_return_type<vulong, vl<llong>, vl<decltype(ulong() + llong())>>();
    binary_op_return_type<vulong, vl<ullong>, vl<ullong>>();
    binary_op_return_type<vulong, vl<float>, vl<float>>();
    binary_op_return_type<vulong, vl<double>, vl<double>>();

    binary_op_return_type<vl<ulong>, vlong, vulong>();
    binary_op_return_type<vl<ulong>, vulong, vulong>();
    binary_op_return_type<v32<ulong>, schar, v32<ulong>>();
    binary_op_return_type<v32<ulong>, uchar, v32<ulong>>();
    binary_op_return_type<v32<ulong>, short, v32<ulong>>();
    binary_op_return_type<v32<ulong>, ushort, v32<ulong>>();
    binary_op_return_type<v32<ulong>, int, v32<ulong>>();
    binary_op_return_type<v32<ulong>, uint, v32<ulong>>();
    binary_op_return_type<v32<ulong>, long, v32<ulong>>();
    binary_op_return_type<v32<ulong>, ulong, v32<ulong>>();
    binary_op_return_type<v32<ulong>, llong, v32<decltype(ulong() + llong())>>();
    binary_op_return_type<v32<ulong>, ullong, v32<ullong>>();
    binary_op_return_type<v32<ulong>, float, v32<float>>();
    binary_op_return_type<v32<ulong>, double, v32<double>>();
    binary_op_return_type<v32<ulong>, v32<schar>, v32<ulong>>();
    binary_op_return_type<v32<ulong>, v32<uchar>, v32<ulong>>();
    binary_op_return_type<v32<ulong>, v32<short>, v32<ulong>>();
    binary_op_return_type<v32<ulong>, v32<ushort>, v32<ulong>>();
    binary_op_return_type<v32<ulong>, v32<int>, v32<ulong>>();
    binary_op_return_type<v32<ulong>, v32<uint>, v32<ulong>>();
    binary_op_return_type<v32<ulong>, v32<long>, v32<ulong>>();
    binary_op_return_type<v32<ulong>, v32<ulong>, v32<ulong>>();
    binary_op_return_type<v32<ulong>, v32<llong>, v32<decltype(ulong() + llong())>>();
    binary_op_return_type<v32<ulong>, v32<ullong>, v32<ullong>>();
    binary_op_return_type<v32<ulong>, v32<float>, v32<float>>();
    binary_op_return_type<v32<ulong>, v32<double>, v32<double>>();

    binary_op_return_type<v64<ulong>, schar, v64<ulong>>();
    binary_op_return_type<v64<ulong>, uchar, v64<ulong>>();
    binary_op_return_type<v64<ulong>, short, v64<ulong>>();
    binary_op_return_type<v64<ulong>, ushort, v64<ulong>>();
    binary_op_return_type<v64<ulong>, int, v64<ulong>>();
    binary_op_return_type<v64<ulong>, uint, v64<ulong>>();
    binary_op_return_type<v64<ulong>, long, v64<ulong>>();
    binary_op_return_type<v64<ulong>, ulong, v64<ulong>>();
    binary_op_return_type<v64<ulong>, llong, v64<decltype(ulong() + llong())>>();
    binary_op_return_type<v64<ulong>, ullong, v64<ullong>>();
    binary_op_return_type<v64<ulong>, float, v64<float>>();
    binary_op_return_type<v64<ulong>, double, v64<double>>();
    binary_op_return_type<v64<ulong>, v64<schar>, v64<ulong>>();
    binary_op_return_type<v64<ulong>, v64<uchar>, v64<ulong>>();
    binary_op_return_type<v64<ulong>, v64<short>, v64<ulong>>();
    binary_op_return_type<v64<ulong>, v64<ushort>, v64<ulong>>();
    binary_op_return_type<v64<ulong>, v64<int>, v64<ulong>>();
    binary_op_return_type<v64<ulong>, v64<uint>, v64<ulong>>();
    binary_op_return_type<v64<ulong>, v64<long>, v64<ulong>>();
    binary_op_return_type<v64<ulong>, v64<ulong>, v64<ulong>>();
    binary_op_return_type<v64<ulong>, v64<llong>, v64<decltype(ulong() + llong())>>();
    binary_op_return_type<v64<ulong>, v64<ullong>, v64<ullong>>();
    binary_op_return_type<v64<ulong>, v64<float>, v64<float>>();
    binary_op_return_type<v64<ulong>, v64<double>, v64<double>>();

    // llong{{{2
    binary_op_return_type<vllong, vllong, vllong>();
    binary_op_return_type<vllong, vullong, vullong>();
    binary_op_return_type<vllong, schar, vllong>();
    binary_op_return_type<vllong, uchar, vllong>();
    binary_op_return_type<vllong, short, vllong>();
    binary_op_return_type<vllong, ushort, vllong>();
    binary_op_return_type<vllong, int, vllong>();
    binary_op_return_type<vllong, uint, Vc::datapar<decltype(llong() + uint())>>();
    binary_op_return_type<vllong, long, vllong>();
    binary_op_return_type<vllong, ulong, Vc::datapar<decltype(llong() + ulong())>>();
    binary_op_return_type<vllong, llong, vllong>();
    binary_op_return_type<vllong, ullong, vullong>();
    binary_op_return_type<vllong, float, v64<float>>();
    binary_op_return_type<vllong, double, v64<double>>();
    binary_op_return_type<vllong, v64<schar>, vllong>();
    binary_op_return_type<vllong, v64<uchar>, vllong>();
    binary_op_return_type<vllong, v64<short>, vllong>();
    binary_op_return_type<vllong, v64<ushort>, vllong>();
    binary_op_return_type<vllong, v64<int>, vllong>();
    binary_op_return_type<vllong, v64<uint>, Vc::datapar<decltype(llong() + uint())>>();
    binary_op_return_type<vllong, v64<long>, vllong>();
    binary_op_return_type<vllong, v64<ulong>, Vc::datapar<decltype(llong() + ulong())>>();
    binary_op_return_type<vllong, v64<llong>, vllong>();
    binary_op_return_type<vllong, v64<ullong>, vullong>();
    binary_op_return_type<vllong, v64<float>, v64<float>>();
    binary_op_return_type<vllong, v64<double>, v64<double>>();

    binary_op_return_type<v32<llong>, schar, v32<llong>>();
    binary_op_return_type<v32<llong>, uchar, v32<llong>>();
    binary_op_return_type<v32<llong>, short, v32<llong>>();
    binary_op_return_type<v32<llong>, ushort, v32<llong>>();
    binary_op_return_type<v32<llong>, int, v32<llong>>();
    binary_op_return_type<v32<llong>, uint, v32<decltype(llong() + uint())>>();
    binary_op_return_type<v32<llong>, long, v32<llong>>();
    binary_op_return_type<v32<llong>, ulong, v32<decltype(llong() + ulong())>>();
    binary_op_return_type<v32<llong>, llong, v32<llong>>();
    binary_op_return_type<v32<llong>, ullong, v32<ullong>>();
    binary_op_return_type<v32<llong>, float, v32<float>>();
    binary_op_return_type<v32<llong>, double, v32<double>>();
    binary_op_return_type<v32<llong>, v32<schar>, v32<llong>>();
    binary_op_return_type<v32<llong>, v32<uchar>, v32<llong>>();
    binary_op_return_type<v32<llong>, v32<short>, v32<llong>>();
    binary_op_return_type<v32<llong>, v32<ushort>, v32<llong>>();
    binary_op_return_type<v32<llong>, v32<int>, v32<llong>>();
    binary_op_return_type<v32<llong>, v32<uint>, v32<decltype(llong() + uint())>>();
    binary_op_return_type<v32<llong>, v32<long>, v32<llong>>();
    binary_op_return_type<v32<llong>, v32<ulong>, v32<decltype(llong() + ulong())>>();
    binary_op_return_type<v32<llong>, v32<llong>, v32<llong>>();
    binary_op_return_type<v32<llong>, v32<ullong>, v32<ullong>>();
    binary_op_return_type<v32<llong>, v32<float>, v32<float>>();
    binary_op_return_type<v32<llong>, v32<double>, v32<double>>();

    binary_op_return_type<v64<llong>, vllong, vllong>();
    binary_op_return_type<v64<llong>, vullong, vullong>();
    binary_op_return_type<v64<llong>, schar, v64<llong>>();
    binary_op_return_type<v64<llong>, uchar, v64<llong>>();
    binary_op_return_type<v64<llong>, short, v64<llong>>();
    binary_op_return_type<v64<llong>, ushort, v64<llong>>();
    binary_op_return_type<v64<llong>, int, v64<llong>>();
    binary_op_return_type<v64<llong>, uint, v64<decltype(llong() + uint())>>();
    binary_op_return_type<v64<llong>, long, v64<llong>>();
    binary_op_return_type<v64<llong>, ulong, v64<decltype(llong() + ulong())>>();
    binary_op_return_type<v64<llong>, llong, v64<llong>>();
    binary_op_return_type<v64<llong>, ullong, v64<ullong>>();
    binary_op_return_type<v64<llong>, float, v64<float>>();
    binary_op_return_type<v64<llong>, double, v64<double>>();
    binary_op_return_type<v64<llong>, v64<schar>, v64<llong>>();
    binary_op_return_type<v64<llong>, v64<uchar>, v64<llong>>();
    binary_op_return_type<v64<llong>, v64<short>, v64<llong>>();
    binary_op_return_type<v64<llong>, v64<ushort>, v64<llong>>();
    binary_op_return_type<v64<llong>, v64<int>, v64<llong>>();
    binary_op_return_type<v64<llong>, v64<uint>, v64<decltype(llong() + uint())>>();
    binary_op_return_type<v64<llong>, v64<long>, v64<llong>>();
    binary_op_return_type<v64<llong>, v64<ulong>, v64<decltype(llong() + ulong())>>();
    binary_op_return_type<v64<llong>, v64<llong>, v64<llong>>();
    binary_op_return_type<v64<llong>, v64<ullong>, v64<ullong>>();
    binary_op_return_type<v64<llong>, v64<float>, v64<float>>();
    binary_op_return_type<v64<llong>, v64<double>, v64<double>>();

    // int{{{2
    binary_op_return_type<vint, vint, vint>();
    binary_op_return_type<vint, vuint, vuint>();
    binary_op_return_type<vint, schar, vint>();
    binary_op_return_type<vint, uchar, vint>();
    binary_op_return_type<vint, short, vint>();
    binary_op_return_type<vint, ushort, vint>();
    binary_op_return_type<vint, int, vint>();
    binary_op_return_type<vint, uint, vuint>();
    binary_op_return_type<vint, long, v32<long>>();
    binary_op_return_type<vint, ulong, v32<ulong>>();
    binary_op_return_type<vint, llong, v32<llong>>();
    binary_op_return_type<vint, ullong, v32<ullong>>();
    binary_op_return_type<vint, float, v32<float>>();
    binary_op_return_type<vint, double, v32<double>>();
    binary_op_return_type<vint, v32<schar>, vint>();
    binary_op_return_type<vint, v32<uchar>, vint>();
    binary_op_return_type<vint, v32<short>, vint>();
    binary_op_return_type<vint, v32<ushort>, vint>();
    binary_op_return_type<vint, v32<int>, vint>();
    binary_op_return_type<vint, v32<uint>, vuint>();
    binary_op_return_type<vint, v32<long>, v32<long>>();
    binary_op_return_type<vint, v32<ulong>, v32<ulong>>();
    binary_op_return_type<vint, v32<llong>, v32<llong>>();
    binary_op_return_type<vint, v32<ullong>, v32<ullong>>();
    binary_op_return_type<vint, v32<float>, v32<float>>();
    binary_op_return_type<vint, v32<double>, v32<double>>();

    binary_op_return_type<v32<int>, vint, vint>();
    binary_op_return_type<v32<int>, vuint, vuint>();
    binary_op_return_type<v32<int>, schar, v32<int>>();
    binary_op_return_type<v32<int>, uchar, v32<int>>();
    binary_op_return_type<v32<int>, short, v32<int>>();
    binary_op_return_type<v32<int>, ushort, v32<int>>();
    binary_op_return_type<v32<int>, int, v32<int>>();
    binary_op_return_type<v32<int>, uint, v32<uint>>();
    binary_op_return_type<v32<int>, long, v32<long>>();
    binary_op_return_type<v32<int>, ulong, v32<ulong>>();
    binary_op_return_type<v32<int>, llong, v32<llong>>();
    binary_op_return_type<v32<int>, ullong, v32<ullong>>();
    binary_op_return_type<v32<int>, float, v32<float>>();
    binary_op_return_type<v32<int>, double, v32<double>>();
    binary_op_return_type<v32<int>, v32<schar>, v32<int>>();
    binary_op_return_type<v32<int>, v32<uchar>, v32<int>>();
    binary_op_return_type<v32<int>, v32<short>, v32<int>>();
    binary_op_return_type<v32<int>, v32<ushort>, v32<int>>();
    binary_op_return_type<v32<int>, v32<int>, v32<int>>();
    binary_op_return_type<v32<int>, v32<uint>, v32<uint>>();
    binary_op_return_type<v32<int>, v32<long>, v32<long>>();
    binary_op_return_type<v32<int>, v32<ulong>, v32<ulong>>();
    binary_op_return_type<v32<int>, v32<llong>, v32<llong>>();
    binary_op_return_type<v32<int>, v32<ullong>, v32<ullong>>();
    binary_op_return_type<v32<int>, v32<float>, v32<float>>();
    binary_op_return_type<v32<int>, v32<double>, v32<double>>();

    // uint{{{2
    binary_op_return_type<vuint, vint, vuint>();
    binary_op_return_type<vuint, vuint, vuint>();
    binary_op_return_type<vuint, schar, vuint>();
    binary_op_return_type<vuint, uchar, vuint>();
    binary_op_return_type<vuint, short, vuint>();
    binary_op_return_type<vuint, ushort, vuint>();
    binary_op_return_type<vuint, int, vuint>();
    binary_op_return_type<vuint, uint, vuint>();
    binary_op_return_type<vuint, long, v32<long>>();
    binary_op_return_type<vuint, ulong, v32<ulong>>();
    binary_op_return_type<vuint, llong, v32<llong>>();
    binary_op_return_type<vuint, ullong, v32<ullong>>();
    binary_op_return_type<vuint, float, v32<float>>();
    binary_op_return_type<vuint, double, v32<double>>();
    binary_op_return_type<vuint, v32<schar>, vuint>();
    binary_op_return_type<vuint, v32<uchar>, vuint>();
    binary_op_return_type<vuint, v32<short>, vuint>();
    binary_op_return_type<vuint, v32<ushort>, vuint>();
    binary_op_return_type<vuint, v32<int>, vuint>();
    binary_op_return_type<vuint, v32<uint>, vuint>();
    binary_op_return_type<vuint, v32<long>, v32<long>>();
    binary_op_return_type<vuint, v32<ulong>, v32<ulong>>();
    binary_op_return_type<vuint, v32<llong>, v32<llong>>();
    binary_op_return_type<vuint, v32<ullong>, v32<ullong>>();
    binary_op_return_type<vuint, v32<float>, v32<float>>();
    binary_op_return_type<vuint, v32<double>, v32<double>>();

    binary_op_return_type<v32<uint>, vint, vuint>();
    binary_op_return_type<v32<uint>, vuint, vuint>();
    binary_op_return_type<v32<uint>, schar, v32<uint>>();
    binary_op_return_type<v32<uint>, uchar, v32<uint>>();
    binary_op_return_type<v32<uint>, short, v32<uint>>();
    binary_op_return_type<v32<uint>, ushort, v32<uint>>();
    binary_op_return_type<v32<uint>, int, v32<uint>>();
    binary_op_return_type<v32<uint>, uint, v32<uint>>();
    binary_op_return_type<v32<uint>, long, v32<long>>();
    binary_op_return_type<v32<uint>, ulong, v32<ulong>>();
    binary_op_return_type<v32<uint>, llong, v32<llong>>();
    binary_op_return_type<v32<uint>, ullong, v32<ullong>>();
    binary_op_return_type<v32<uint>, float, v32<float>>();
    binary_op_return_type<v32<uint>, double, v32<double>>();
    binary_op_return_type<v32<uint>, v32<schar>, v32<uint>>();
    binary_op_return_type<v32<uint>, v32<uchar>, v32<uint>>();
    binary_op_return_type<v32<uint>, v32<short>, v32<uint>>();
    binary_op_return_type<v32<uint>, v32<ushort>, v32<uint>>();
    binary_op_return_type<v32<uint>, v32<int>, v32<uint>>();
    binary_op_return_type<v32<uint>, v32<uint>, v32<uint>>();
    binary_op_return_type<v32<uint>, v32<long>, v32<long>>();
    binary_op_return_type<v32<uint>, v32<ulong>, v32<ulong>>();
    binary_op_return_type<v32<uint>, v32<llong>, v32<llong>>();
    binary_op_return_type<v32<uint>, v32<ullong>, v32<ullong>>();
    binary_op_return_type<v32<uint>, v32<float>, v32<float>>();
    binary_op_return_type<v32<uint>, v32<double>, v32<double>>();

    // short{{{2
    binary_op_return_type<vshort, vshort, vshort>();
    binary_op_return_type<vshort, vushort, vushort>();
    binary_op_return_type<vshort, schar, vshort>();
    binary_op_return_type<vshort, uchar, vshort>();
    binary_op_return_type<vshort, short, vshort>();
    binary_op_return_type<vshort, ushort, vushort>();
    binary_op_return_type<vshort, int, vshort>();
    binary_op_return_type<vshort, uint, vushort>();
    binary_op_return_type<vshort, long, v16<long>>();
    binary_op_return_type<vshort, ulong, v16<ulong>>();
    binary_op_return_type<vshort, llong, v16<llong>>();
    binary_op_return_type<vshort, ullong, v16<ullong>>();
    binary_op_return_type<vshort, float, v16<float>>();
    binary_op_return_type<vshort, double, v16<double>>();
    binary_op_return_type<vshort, v16<schar>, vshort>();
    binary_op_return_type<vshort, v16<uchar>, vshort>();
    binary_op_return_type<vshort, v16<short>, vshort>();
    binary_op_return_type<vshort, v16<ushort>, vushort>();
    binary_op_return_type<vshort, v16<int>, v16<int>>();
    binary_op_return_type<vshort, v16<uint>, v16<uint>>();
    binary_op_return_type<vshort, v16<long>, v16<long>>();
    binary_op_return_type<vshort, v16<ulong>, v16<ulong>>();
    binary_op_return_type<vshort, v16<llong>, v16<llong>>();
    binary_op_return_type<vshort, v16<ullong>, v16<ullong>>();
    binary_op_return_type<vshort, v16<float>, v16<float>>();
    binary_op_return_type<vshort, v16<double>, v16<double>>();

    binary_op_return_type<v16<short>, vshort, vshort>();
    binary_op_return_type<v16<short>, vushort, vushort>();
    binary_op_return_type<v16<short>, schar, v16<short>>();
    binary_op_return_type<v16<short>, uchar, v16<short>>();
    binary_op_return_type<v16<short>, short, v16<short>>();
    binary_op_return_type<v16<short>, ushort, v16<ushort>>();
    binary_op_return_type<v16<short>, int, v16<short>>();
    binary_op_return_type<v16<short>, uint, v16<ushort>>();
    binary_op_return_type<v16<short>, long, v16<long>>();
    binary_op_return_type<v16<short>, ulong, v16<ulong>>();
    binary_op_return_type<v16<short>, llong, v16<llong>>();
    binary_op_return_type<v16<short>, ullong, v16<ullong>>();
    binary_op_return_type<v16<short>, float, v16<float>>();
    binary_op_return_type<v16<short>, double, v16<double>>();
    binary_op_return_type<v16<short>, v16<schar>, v16<short>>();
    binary_op_return_type<v16<short>, v16<uchar>, v16<short>>();
    binary_op_return_type<v16<short>, v16<short>, v16<short>>();
    binary_op_return_type<v16<short>, v16<ushort>, v16<ushort>>();
    binary_op_return_type<v16<short>, v16<int>, v16<int>>();
    binary_op_return_type<v16<short>, v16<uint>, v16<uint>>();
    binary_op_return_type<v16<short>, v16<long>, v16<long>>();
    binary_op_return_type<v16<short>, v16<ulong>, v16<ulong>>();
    binary_op_return_type<v16<short>, v16<llong>, v16<llong>>();
    binary_op_return_type<v16<short>, v16<ullong>, v16<ullong>>();
    binary_op_return_type<v16<short>, v16<float>, v16<float>>();
    binary_op_return_type<v16<short>, v16<double>, v16<double>>();

    // ushort{{{2
    binary_op_return_type<vushort, vshort, vushort>();
    binary_op_return_type<vushort, vushort, vushort>();
    binary_op_return_type<vushort, schar, vushort>();
    binary_op_return_type<vushort, uchar, vushort>();
    binary_op_return_type<vushort, short, vushort>();
    binary_op_return_type<vushort, ushort, vushort>();
    binary_op_return_type<vushort, int, vushort>();
    binary_op_return_type<vushort, uint, vushort>();
    binary_op_return_type<vushort, long, v16<long>>();
    binary_op_return_type<vushort, ulong, v16<ulong>>();
    binary_op_return_type<vushort, llong, v16<llong>>();
    binary_op_return_type<vushort, ullong, v16<ullong>>();
    binary_op_return_type<vushort, float, v16<float>>();
    binary_op_return_type<vushort, double, v16<double>>();
    binary_op_return_type<vushort, v16<schar>, vushort>();
    binary_op_return_type<vushort, v16<uchar>, vushort>();
    binary_op_return_type<vushort, v16<short>, vushort>();
    binary_op_return_type<vushort, v16<ushort>, vushort>();
    binary_op_return_type<vushort, v16<int>, v16<int>>();
    binary_op_return_type<vushort, v16<uint>, v16<uint>>();
    binary_op_return_type<vushort, v16<long>, v16<long>>();
    binary_op_return_type<vushort, v16<ulong>, v16<ulong>>();
    binary_op_return_type<vushort, v16<llong>, v16<llong>>();
    binary_op_return_type<vushort, v16<ullong>, v16<ullong>>();
    binary_op_return_type<vushort, v16<float>, v16<float>>();
    binary_op_return_type<vushort, v16<double>, v16<double>>();

    binary_op_return_type<v16<ushort>, vshort, vushort>();
    binary_op_return_type<v16<ushort>, vushort, vushort>();
    binary_op_return_type<v16<ushort>, schar, v16<ushort>>();
    binary_op_return_type<v16<ushort>, uchar, v16<ushort>>();
    binary_op_return_type<v16<ushort>, short, v16<ushort>>();
    binary_op_return_type<v16<ushort>, ushort, v16<ushort>>();
    binary_op_return_type<v16<ushort>, int, v16<ushort>>();
    binary_op_return_type<v16<ushort>, uint, v16<ushort>>();
    binary_op_return_type<v16<ushort>, long, v16<long>>();
    binary_op_return_type<v16<ushort>, ulong, v16<ulong>>();
    binary_op_return_type<v16<ushort>, llong, v16<llong>>();
    binary_op_return_type<v16<ushort>, ullong, v16<ullong>>();
    binary_op_return_type<v16<ushort>, float, v16<float>>();
    binary_op_return_type<v16<ushort>, double, v16<double>>();
    binary_op_return_type<v16<ushort>, v16<schar>, v16<ushort>>();
    binary_op_return_type<v16<ushort>, v16<uchar>, v16<ushort>>();
    binary_op_return_type<v16<ushort>, v16<short>, v16<ushort>>();
    binary_op_return_type<v16<ushort>, v16<ushort>, v16<ushort>>();
    binary_op_return_type<v16<ushort>, v16<int>, v16<int>>();
    binary_op_return_type<v16<ushort>, v16<uint>, v16<uint>>();
    binary_op_return_type<v16<ushort>, v16<long>, v16<long>>();
    binary_op_return_type<v16<ushort>, v16<ulong>, v16<ulong>>();
    binary_op_return_type<v16<ushort>, v16<llong>, v16<llong>>();
    binary_op_return_type<v16<ushort>, v16<ullong>, v16<ullong>>();
    binary_op_return_type<v16<ushort>, v16<float>, v16<float>>();
    binary_op_return_type<v16<ushort>, v16<double>, v16<double>>();

    // schar{{{2
    binary_op_return_type<vschar, vschar, vschar>();
    binary_op_return_type<vschar, vuchar, vuchar>();
    binary_op_return_type<vschar, schar, vschar>();
    binary_op_return_type<vschar, uchar, vuchar>();
    // the following 4 are possibly surprising:
    binary_op_return_type<vschar, short, v8<short>>();
    binary_op_return_type<vschar, ushort, v8<ushort>>();
    binary_op_return_type<vschar, int, vschar>();
    binary_op_return_type<vschar, uint, vuchar>();
    binary_op_return_type<vschar, long, v8<long>>();
    binary_op_return_type<vschar, ulong, v8<ulong>>();
    binary_op_return_type<vschar, llong, v8<llong>>();
    binary_op_return_type<vschar, ullong, v8<ullong>>();
    binary_op_return_type<vschar, float, v8<float>>();
    binary_op_return_type<vschar, double, v8<double>>();
    binary_op_return_type<vschar, v8<schar>, vschar>();
    binary_op_return_type<vschar, v8<uchar>, vuchar>();
    binary_op_return_type<vschar, v8<short>, v8<short>>();
    binary_op_return_type<vschar, v8<ushort>, v8<ushort>>();
    binary_op_return_type<vschar, v8<int>, v8<int>>();
    binary_op_return_type<vschar, v8<uint>, v8<uint>>();
    binary_op_return_type<vschar, v8<long>, v8<long>>();
    binary_op_return_type<vschar, v8<ulong>, v8<ulong>>();
    binary_op_return_type<vschar, v8<llong>, v8<llong>>();
    binary_op_return_type<vschar, v8<ullong>, v8<ullong>>();
    binary_op_return_type<vschar, v8<float>, v8<float>>();
    binary_op_return_type<vschar, v8<double>, v8<double>>();

    binary_op_return_type<v8<schar>, vschar, vschar>();
    binary_op_return_type<v8<schar>, vuchar, vuchar>();
    binary_op_return_type<v8<schar>, schar, v8<schar>>();
    binary_op_return_type<v8<schar>, uchar, v8<uchar>>();
    // the following 4 are possibly surprising:
    binary_op_return_type<v8<schar>, short, v8<short>>();
    binary_op_return_type<v8<schar>, ushort, v8<ushort>>();
    binary_op_return_type<v8<schar>, int, v8<schar>>();
    binary_op_return_type<v8<schar>, uint, v8<uchar>>();
    binary_op_return_type<v8<schar>, long, v8<long>>();
    binary_op_return_type<v8<schar>, ulong, v8<ulong>>();
    binary_op_return_type<v8<schar>, llong, v8<llong>>();
    binary_op_return_type<v8<schar>, ullong, v8<ullong>>();
    binary_op_return_type<v8<schar>, float, v8<float>>();
    binary_op_return_type<v8<schar>, double, v8<double>>();
    binary_op_return_type<v8<schar>, v8<schar>, v8<schar>>();
    binary_op_return_type<v8<schar>, v8<uchar>, v8<uchar>>();
    binary_op_return_type<v8<schar>, v8<short>, v8<short>>();
    binary_op_return_type<v8<schar>, v8<ushort>, v8<ushort>>();
    binary_op_return_type<v8<schar>, v8<int>, v8<int>>();
    binary_op_return_type<v8<schar>, v8<uint>, v8<uint>>();
    binary_op_return_type<v8<schar>, v8<long>, v8<long>>();
    binary_op_return_type<v8<schar>, v8<ulong>, v8<ulong>>();
    binary_op_return_type<v8<schar>, v8<llong>, v8<llong>>();
    binary_op_return_type<v8<schar>, v8<ullong>, v8<ullong>>();
    binary_op_return_type<v8<schar>, v8<float>, v8<float>>();
    binary_op_return_type<v8<schar>, v8<double>, v8<double>>();

    // misc{{{2
    binary_op_return_type<int, v32<long>, v32<long>>();
    binary_op_return_type<int, v32<ulong>, v32<ulong>>();
    binary_op_return_type<int, v32<llong>, v32<llong>>();
    binary_op_return_type<int, v32<ullong>, v32<ullong>>();
    binary_op_return_type<int, v32<ushort>, v32<ushort>>();
    binary_op_return_type<int, v32<schar>, v32<schar>>();
    binary_op_return_type<int, v32<uchar>, v32<uchar>>();
}
// vim: foldmethod=marker
