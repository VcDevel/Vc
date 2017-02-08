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
#include "metahelpers.h"

template <class... Ts> using base_template = Vc::mask<Ts...>;
#include "testtypes.h"

// mask generator functions {{{1
template <class M> M make_mask(const std::initializer_list<bool> &init)
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

template <class M> M make_alternating_mask()
{
    return make_mask<M>({false, true});
}

TEST_TYPES(M, broadcast, ALL_TYPES)  //{{{1
{
    static_assert(std::is_convertible<typename M::reference, bool>::value,
                  "A smart_reference<mask> must be convertible to bool.");
    static_assert(std::is_same<bool, decltype(std::declval<const typename M::reference &>() == true)>::value,
                  "A smart_reference<mask> must be comparable against bool.");
    static_assert(Vc::Traits::has_equality_operator<typename M::reference, bool>::value,
                  "A smart_reference<mask> must be comparable against bool.");
    VERIFY(Vc::is_mask_v<M>);

    {
        M x;      // default broadcasts 0
        x = M{};  // default broadcasts 0
        x = M();  // default broadcasts 0
        x = x;
        for (std::size_t i = 0; i < M::size(); ++i) {
            COMPARE(x[i], false);
        }
    }

    M x(true);
    M y(false);
    for (std::size_t i = 0; i < M::size(); ++i) {
        COMPARE(x[i], true);
        COMPARE(y[i], false);
    }
    y = M(true);
    COMPARE(x, y);
}

TEST_TYPES(M, operators, ALL_TYPES)  //{{{1
{
    {  // compares{{{2
        M x(true), y(false);
        VERIFY(x == x);
        VERIFY(x != y);
        VERIFY(y != x);
        VERIFY(!(x != x));
        VERIFY(!(x == y));
        VERIFY(!(y == x));
    }
    {  // subscripting{{{2
        M x(true);
        for (std::size_t i = 0; i < M::size(); ++i) {
            COMPARE(x[i], true) << "\nx: " << x << ", i: " << i;
            x[i] = !x[i];
        }
        COMPARE(x, M{false});
        for (std::size_t i = 0; i < M::size(); ++i) {
            COMPARE(x[i], false) << "\nx: " << x << ", i: " << i;
            x[i] = !x[i];
        }
        COMPARE(x, M{true});
    }
    {  // negation{{{2
        M x(false);
        M y = !x;
        COMPARE(y, M{true});
        COMPARE(!y, x);
    }
}

// convert {{{1
template <typename M, int SizeofT = sizeof(typename M::value_type)> struct ConvertType {
    using type0 = Vc::fixed_size_mask<float, M::size()>;
    using type1 = Vc::fixed_size_mask<signed short, M::size()>;
};
#ifdef Vc_HAVE_FULL_SSE_ABI
template <typename T> struct ConvertType<Vc::mask<T, Vc::datapar_abi::fixed_size<2>>, 8> {
    using type0 = Vc::mask<double, Vc::datapar_abi::sse>;
    using type1 = Vc::mask<std::uint64_t, Vc::datapar_abi::sse>;
};
template <typename T> struct ConvertType<Vc::mask<T, Vc::datapar_abi::fixed_size<4>>, 4> {
    using type0 = Vc::mask<float, Vc::datapar_abi::sse>;
    using type1 = Vc::mask<std::uint32_t, Vc::datapar_abi::sse>;
};
template <typename T> struct ConvertType<Vc::mask<T, Vc::datapar_abi::fixed_size<8>>, 2> {
    using type0 = Vc::mask<std::int16_t, Vc::datapar_abi::sse>;
    using type1 = Vc::mask<std::uint16_t, Vc::datapar_abi::sse>;
};
template <typename T> struct ConvertType<Vc::mask<T, Vc::datapar_abi::fixed_size<16>>, 1> {
    using type0 = Vc::mask<std::int8_t, Vc::datapar_abi::sse>;
    using type1 = Vc::mask<std::uint8_t, Vc::datapar_abi::sse>;
};
#endif
#ifdef Vc_HAVE_FULL_AVX_ABI
template <typename T> struct ConvertType<Vc::mask<T, Vc::datapar_abi::fixed_size<4>>, 8> {
    using type0 = Vc::mask<double, Vc::datapar_abi::avx>;
    using type1 = Vc::mask<std::uint64_t, Vc::datapar_abi::avx>;
};
template <typename T> struct ConvertType<Vc::mask<T, Vc::datapar_abi::fixed_size<8>>, 4> {
    using type0 = Vc::mask<float, Vc::datapar_abi::avx>;
    using type1 = Vc::mask<std::uint32_t, Vc::datapar_abi::avx>;
};
template <typename T> struct ConvertType<Vc::mask<T, Vc::datapar_abi::fixed_size<16>>, 2> {
    using type0 = Vc::mask<std::int16_t, Vc::datapar_abi::avx>;
    using type1 = Vc::mask<std::uint16_t, Vc::datapar_abi::avx>;
};
template <typename T> struct ConvertType<Vc::mask<T, Vc::datapar_abi::fixed_size<32>>, 1> {
    using type0 = Vc::mask<std::int8_t, Vc::datapar_abi::avx>;
    using type1 = Vc::mask<std::uint8_t, Vc::datapar_abi::avx>;
};
#endif
#ifdef Vc_HAVE_AVX512_ABI
template <typename T> struct ConvertType<Vc::mask<T, Vc::datapar_abi::fixed_size<8>>, 8> {
    using type0 = Vc::mask<double, Vc::datapar_abi::avx>;
    using type1 = Vc::mask<std::uint64_t, Vc::datapar_abi::avx>;
};
template <typename T> struct ConvertType<Vc::mask<T, Vc::datapar_abi::fixed_size<16>>, 4> {
    using type0 = Vc::mask<float, Vc::datapar_abi::avx>;
    using type1 = Vc::mask<std::uint32_t, Vc::datapar_abi::avx>;
};
#endif
#ifdef Vc_HAVE_FULL_AVX512_ABI
template <typename T> struct ConvertType<Vc::mask<T, Vc::datapar_abi::fixed_size<32>>, 2> {
    using type0 = Vc::mask<std::int16_t, Vc::datapar_abi::avx>;
    using type1 = Vc::mask<std::uint16_t, Vc::datapar_abi::avx>;
};
template <typename T> struct ConvertType<Vc::mask<T, Vc::datapar_abi::fixed_size<64>>, 1> {
    using type0 = Vc::mask<std::int8_t, Vc::datapar_abi::avx>;
    using type1 = Vc::mask<std::uint8_t, Vc::datapar_abi::avx>;
};
#endif
TEST_TYPES(M, convert, ALL_TYPES)
{
    {
        using M2 = typename ConvertType<M>::type0;
        M2 x ( true);
        M y = x;
        COMPARE(y, M{true});
        x[0] = false;
        COMPARE(x[0], false);
        y = x;
        COMPARE(y[0], false);
        for (std::size_t i = 1; i < M::size(); ++i) {
            COMPARE(y[i], true);
        }
        M2 z = y;
        COMPARE(z, x);
    }
    {
        using M2 = typename ConvertType<M>::type1;
        M2 x(true);
        M y = x;
        COMPARE(y, M{true});
        x[0] = false;
        COMPARE(x[0], false);
        y = x;
        COMPARE(y[0], false);
        for (std::size_t i = 1; i < M::size(); ++i) {
            COMPARE(y[i], true);
        }
        M2 z = y;
        COMPARE(z, x);
    }
}

TEST_TYPES(M, load_store, ALL_TYPES)  //{{{1
{
    // loads {{{2
    constexpr size_t alignment = 2 * Vc::memory_alignment_v<M
#ifdef Vc_MSVC
                                                            ,
                                                            bool
#endif
                                                            >;
    alignas(alignment) bool mem[3 * M::size()];
    std::memset(mem, 0, sizeof(mem));
    for (std::size_t i = 1; i < sizeof(mem) / sizeof(*mem); i += 2) {
        COMPARE(mem[i - 1], false);
        mem[i] = true;
    }
    using Vc::flags::element_aligned;
    using Vc::flags::vector_aligned;
#ifdef Vc_MSVC
    using TT = Vc::flags::overaligned_tag<alignment>;
    constexpr TT overaligned = {};
#else
    constexpr auto overaligned = Vc::flags::overaligned<alignment>;
#endif

    const M alternating_mask = make_alternating_mask<M>();

    M x(&mem[M::size()], vector_aligned);
    COMPARE(x, M::size() % 2 == 1 ? !alternating_mask : alternating_mask);
    x = {&mem[1], element_aligned};
    COMPARE(x, !alternating_mask);
    x = M{mem, overaligned};
    COMPARE(x, alternating_mask);

    x.copy_from(&mem[M::size()], vector_aligned);
    COMPARE(x, M::size() % 2 == 1 ? !alternating_mask : alternating_mask);
    x.copy_from(&mem[1], element_aligned);
    COMPARE(x, !alternating_mask);
    x.copy_from(mem, overaligned);
    COMPARE(x, alternating_mask);

    x = !alternating_mask;
    x.copy_from(&mem[M::size()], alternating_mask, vector_aligned);
    COMPARE(x, M::size() % 2 == 1 ? !alternating_mask : M{true});
    x = M(true);                                              // 1111
    x.copy_from(&mem[1], alternating_mask, element_aligned);  // load .0.0
    COMPARE(x, !alternating_mask);                            // 1010
    x.copy_from(mem, alternating_mask, overaligned);          // load .1.1
    COMPARE(x, M{true});                                      // 1111

    // stores {{{2
    memset(mem, 0, sizeof(mem));
    x = M(true);
    x.copy_to(&mem[M::size()], vector_aligned);
    std::size_t i = 0;
    for (; i < M::size(); ++i) {
        COMPARE(mem[i], false);
    }
    for (; i < 2 * M::size(); ++i) {
        COMPARE(mem[i], true);
    }
    for (; i < 3 * M::size(); ++i) {
        COMPARE(mem[i], false);
    }
    memset(mem, 0, sizeof(mem));
    x.copy_to(&mem[1], element_aligned);
    COMPARE(mem[0], false);
    for (i = 1; i <= M::size(); ++i) {
        COMPARE(mem[i], true);
    }
    for (; i < 3 * M::size(); ++i) {
        COMPARE(mem[i], false);
    }
    memset(mem, 0, sizeof(mem));
    x.copy_to(mem, overaligned);
    for (i = 0; i < M::size(); ++i) {
        COMPARE(mem[i], true);
    }
    for (; i < 3 * M::size(); ++i) {
        COMPARE(mem[i], false);
    }
    (!x).copy_to(mem, alternating_mask, overaligned);
    for (i = 0; i < M::size(); ++i) {
        COMPARE(mem[i], i % 2 == 0);
    }
    for (; i < 3 * M::size(); ++i) {
        COMPARE(mem[i], false);
    }
}

template <class A, class B, class Expected = A> void binary_op_return_type()  //{{{1
{
    static_assert(std::is_same<A, Expected>::value, "");
    const auto name = typeToString<A>() + " + " + typeToString<B>();
    COMPARE(typeid(A() & B()), typeid(Expected)) << name;
    COMPARE(typeid(B() & A()), typeid(Expected)) << name;
    UnitTest::ADD_PASS() << name;
}

TEST_TYPES(M, operator_conversions, (current_native_mask_test_types))  //{{{1
{
    // binary ops without conversions work
    binary_op_return_type<M, M>();

    // nothing else works: no implicit conv. or ambiguous
    using Vc::mask;
    using Vc::native_mask;
    using Vc::fixed_size_mask;
    auto &&is = [](auto x) { return std::is_same<M, native_mask<decltype(x)>>::value; };
    auto &&sfinae_test = [](auto x) {
        return operator_is_substitution_failure<M, decltype(x), std::bit_and<>>;
    };
    using ldouble = long double;
    if (!is(ldouble())) VERIFY(sfinae_test(native_mask<ldouble>()));
    if (!is(double ())) VERIFY(sfinae_test(native_mask<double >()));
    if (!is(float  ())) VERIFY(sfinae_test(native_mask<float  >()));
    if (!is(ullong ())) VERIFY(sfinae_test(native_mask<ullong >()));
    if (!is(llong  ())) VERIFY(sfinae_test(native_mask<llong  >()));
    if (!is(ulong  ())) VERIFY(sfinae_test(native_mask<ulong  >()));
    if (!is(long   ())) VERIFY(sfinae_test(native_mask<long   >()));
    if (!is(uint   ())) VERIFY(sfinae_test(native_mask<uint   >()));
    if (!is(int    ())) VERIFY(sfinae_test(native_mask<int    >()));
    if (!is(ushort ())) VERIFY(sfinae_test(native_mask<ushort >()));
    if (!is(short  ())) VERIFY(sfinae_test(native_mask<short  >()));
    if (!is(uchar  ())) VERIFY(sfinae_test(native_mask<uchar  >()));
    if (!is(schar  ())) VERIFY(sfinae_test(native_mask<schar  >()));

    VERIFY(sfinae_test(bool()));

    VERIFY(sfinae_test(mask<ldouble>()));
    VERIFY(sfinae_test(mask<double >()));
    VERIFY(sfinae_test(mask<float  >()));
    VERIFY(sfinae_test(mask<ullong >()));
    VERIFY(sfinae_test(mask<llong  >()));
    VERIFY(sfinae_test(mask<ulong  >()));
    VERIFY(sfinae_test(mask<long   >()));
    VERIFY(sfinae_test(mask<uint   >()));
    VERIFY(sfinae_test(mask<int    >()));
    VERIFY(sfinae_test(mask<ushort >()));
    VERIFY(sfinae_test(mask<short  >()));
    VERIFY(sfinae_test(mask<uchar  >()));
    VERIFY(sfinae_test(mask<schar  >()));

    VERIFY(sfinae_test(fixed_size_mask<ldouble, 2>()));
    VERIFY(sfinae_test(fixed_size_mask<double , 2>()));
    VERIFY(sfinae_test(fixed_size_mask<float  , 2>()));
    VERIFY(sfinae_test(fixed_size_mask<ullong , 2>()));
    VERIFY(sfinae_test(fixed_size_mask<llong  , 2>()));
    VERIFY(sfinae_test(fixed_size_mask<ulong  , 2>()));
    VERIFY(sfinae_test(fixed_size_mask<long   , 2>()));
    VERIFY(sfinae_test(fixed_size_mask<uint   , 2>()));
    VERIFY(sfinae_test(fixed_size_mask<int    , 2>()));
    VERIFY(sfinae_test(fixed_size_mask<ushort , 2>()));
    VERIFY(sfinae_test(fixed_size_mask<short  , 2>()));
    VERIFY(sfinae_test(fixed_size_mask<uchar  , 2>()));
    VERIFY(sfinae_test(fixed_size_mask<schar  , 2>()));
}

TEST_TYPES(M, reductions, ALL_TYPES)  //{{{1
{
    const M alternating_mask = make_alternating_mask<M>();
    COMPARE(alternating_mask[0], false);  // assumption below
    auto &&gen = make_mask<M>;

    // all_of
    VERIFY( all_of(M{true}));
    VERIFY(!all_of(alternating_mask));
    VERIFY(!all_of(M{false}));

    // any_of
    VERIFY( any_of(M{true}));
    COMPARE(any_of(alternating_mask), M::size() > 1);
    VERIFY(!any_of(M{false}));

    // none_of
    VERIFY(!none_of(M{true}));
    COMPARE(none_of(alternating_mask), M::size() == 1);
    VERIFY( none_of(M{false}));

    // some_of
    VERIFY(!some_of(M{true}));
    VERIFY(!some_of(M{false}));
    if (M::size() > 1) {
        VERIFY(some_of(gen({true, false})));
        VERIFY(some_of(gen({false, true})));
        if (M::size() > 3) {
            VERIFY(some_of(gen({0, 0, 0, 1})));
        }
    }

    // popcount
    COMPARE(popcount(M{true}), int(M::size()));
    COMPARE(popcount(alternating_mask), int(M::size()) / 2);
    COMPARE(popcount(M{false}), 0);
    COMPARE(popcount(gen({0, 0, 1})), int(M::size()) / 3);
    COMPARE(popcount(gen({0, 0, 0, 1})), int(M::size()) / 4);
    COMPARE(popcount(gen({0, 0, 0, 0, 1})), int(M::size()) / 5);

    // find_first_set
    {
        M x(false);
        for (int i = int(M::size() / 2 - 1); i >= 0; --i) {
            x[i] = true;
            COMPARE(find_first_set(x), i) << x;
        }
        x = M(false);
        for (int i = int(M::size() - 1); i >= 0; --i) {
            x[i] = true;
            COMPARE(find_first_set(x), i) << x;
        }
    }
    COMPARE(find_first_set(M{true}), 0);
    if (M::size() > 1) {
        COMPARE(find_first_set(gen({0, 1})), 1);
    }
    if (M::size() > 2) {
        COMPARE(find_first_set(gen({0, 0, 1})), 2);
    }

    // find_last_set
    {
        M x(false);
        for (int i = 0; i < int(M::size()); ++i) {
            x[i] = true;
            COMPARE(find_last_set(x), i) << x;
        }
    }
    COMPARE(find_last_set(M{true}), int(M::size()) - 1);
    if (M::size() > 1) {
        COMPARE(find_last_set(gen({1, 0})), int(M::size()) - 2 + int(M::size() & 1));
    }
    if (M::size() > 3 && (M::size() & 3) == 0) {
        COMPARE(find_last_set(gen({1, 0, 0, 0})), int(M::size()) - 4 - int(M::size() & 3));
    }
}

// vim: foldmethod=marker
