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
#include <Vc/Vc>
#include "make_vec.h"

template <class... Ts> using base_template = Vc::simd<Ts...>;
#include "testtypes.h"
#include "conversions.h"

template <class T, size_t N> struct gen_cast {
    std::array<T, N> data;
    template <class V> gen_cast(const V &v)
    {
        for (size_t i = 0; i < V::size(); ++i) {
            data[i] = static_cast<T>(v[i]);
        }
    }
    template <class I> constexpr T operator()(I) { return data[I::value]; }
};

template <class V> void casts_integral(std::true_type) {
    using T = typename V::value_type;
    using A = typename V::abi_type;
    using TU = std::make_unsigned_t<T>;
    using TS = std::make_signed_t<T>;
    COMPARE(typeid(Vc::static_simd_cast<TU>(V())), typeid(Vc::simd<TU, A>));
    COMPARE(typeid(Vc::static_simd_cast<TS>(V())), typeid(Vc::simd<TS, A>));
}

template <class V> void casts_integral(std::false_type) {}

TEST_TYPES(V_To, casts, outer_product<all_test_types, arithmetic_types>)
{
    using V = typename V_To::template at<0>;
    using To = typename V_To::template at<1>;
    using From = typename V::value_type;

    casts_integral<V>(std::is_integral<From>());

    constexpr auto N = V::size();
    using W = Vc::fixed_size_simd<To, N>;

    struct {
        const size_t N = cvt_input_data<From, To>.size();
        size_t offset = 0;
        void operator++() { offset += V::size(); }
        explicit operator bool() const { return offset < N; }
        From operator()(size_t i) const {
            i += offset;
            return i < N ? cvt_input_data<From, To>[i] : From(i);
        }
    } gen_seq;

    for (; gen_seq; ++gen_seq) {
        const V seq(gen_seq);
        COMPARE(Vc::static_simd_cast<V>(seq), seq);
        COMPARE(Vc::static_simd_cast<W>(seq), W(gen_cast<To, N>(seq)));
        auto test = Vc::static_simd_cast<To>(seq);
        // decltype(test) is not W if
        // a) V::abi_type is not fixed_size and
        // b.1) V::value_type and To are integral and of equal rank or
        // b.2) V::value_type and To are equal
        COMPARE(test, decltype(test)(gen_cast<To, N>(seq)));
        if (std::is_same<To, From>::value) {
            COMPARE(typeid(decltype(test)), typeid(V));
        }
    }
}
