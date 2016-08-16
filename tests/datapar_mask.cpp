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

template<typename T, int N>
using fixed_mask = Vc::mask<T, Vc::datapar_abi::fixed_size<N>>;

// all_test_types / ALL_TYPES {{{1
typedef expand_list<
    Typelist<
#ifdef Vc_HAVE_FULL_AVX_ABI
        Template<Vc::mask, Vc::datapar_abi::avx>,
#endif
#ifdef Vc_HAVE_FULL_SSE_ABI
        Template<Vc::mask, Vc::datapar_abi::sse>,
#endif
        Template<Vc::mask, Vc::datapar_abi::scalar>,
        Template<Vc::mask, Vc::datapar_abi::fixed_size<2>>,
        Template<Vc::mask, Vc::datapar_abi::fixed_size<3>>,
        Template<Vc::mask, Vc::datapar_abi::fixed_size<4>>,
        Template<Vc::mask, Vc::datapar_abi::fixed_size<8>>,
        Template<Vc::mask, Vc::datapar_abi::fixed_size<12>>,
        Template<Vc::mask, Vc::datapar_abi::fixed_size<16>>,
        Template<Vc::mask, Vc::datapar_abi::fixed_size<32>>>,
    Typelist<long double, double, float, unsigned long, int, unsigned short, signed char,
             long, unsigned int, short, unsigned char>> all_test_types;

#define ALL_TYPES (all_test_types)

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
    {
        M x;      // default broadcasts 0
        x = M{};  // default broadcasts 0
        x = M();  // default broadcasts 0
        x = x;
        for (std::size_t i = 0; i < M::size(); ++i) {
            COMPARE(x[i], false);
        }
    }

    M x = true;
    M y = false;
    for (std::size_t i = 0; i < M::size(); ++i) {
        COMPARE(x[i], true);
        COMPARE(y[i], false);
    }
    y = true;
    COMPARE(x, y);
}

TEST_TYPES(M, operators, ALL_TYPES)  //{{{1
{
    {  // compares{{{2
        M x = true, y = false;
        VERIFY(x == x);
        VERIFY(x != y);
        VERIFY(y != x);
        VERIFY(!(x != x));
        VERIFY(!(x == y));
        VERIFY(!(y == x));
    }
    {  // subscripting{{{2
        M x = true;
        for (std::size_t i = 0; i < M::size(); ++i) {
            COMPARE(x[i], true);
            x[i] = !x[i];
        }
        COMPARE(x, M{false});
        for (std::size_t i = 0; i < M::size(); ++i) {
            COMPARE(x[i], false);
            x[i] = !x[i];
        }
        COMPARE(x, M{true});
    }
    {  // negation{{{2
        M x = false;
        M y = !x;
        COMPARE(y, M{true});
        COMPARE(!y, x);
    }
}

// convert {{{1
template <typename M, int SizeofT = sizeof(typename M::value_type)> struct ConvertType {
    using type0 = fixed_mask<float, M::size()>;
    using type1 = fixed_mask<signed short, M::size()>;
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
        M2 x = true;
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
        M2 x = true;
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
    alignas(Vc::memory_alignment<M> * 2) bool mem[3 * M::size()] = {};
    for (std::size_t i = 1; i < sizeof(mem) / sizeof(*mem); i += 2) {
        COMPARE(mem[i - 1], false);
        mem[i] = true;
    }
    using Vc::flags::element_aligned;
    using Vc::flags::vector_aligned;
    constexpr auto overaligned = Vc::flags::overaligned<Vc::memory_alignment<M> * 2>;

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
    x = true;                                                 // 1111
    x.copy_from(&mem[1], alternating_mask, element_aligned);  // load .0.0
    COMPARE(x, !alternating_mask);                            // 1010
    x.copy_from(mem, alternating_mask, overaligned);          // load .1.1
    COMPARE(x, M{true});                                      // 1111

    // stores {{{2
    memset(mem, 0, sizeof(mem));
    x = true;
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
    COMPARE(find_first_set(M{true}), 0);

    // find_last_set
    COMPARE(find_last_set(M{true}), int(M::size()) - 1);
}

// vim: foldmethod=marker
