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

//#define UNITTEST_ONLY_XTEST 1
#include <vir/test.h>
#include <Vc/simd>

template <class... Ts> using base_template = Vc::simd_mask<Ts...>;
#include "testtypes.h"

// type iteration in a function {{{1
template <class... Ts, class F> void call_with_types(F &&f, vir::Typelist<Ts...> = {})
{
    [](std::initializer_list<int>) {}({(f(Ts()), 0)...});
}

template <class List, class F> void call_with_typelist(F &&f)
{
    call_with_types(std::forward<F>(f), List());
}

TEST_TYPES(FromTo, conversions,  //{{{1
           outer_product<all_arithmetic_types, all_arithmetic_types>)
{
    using From = typename FromTo::template at<0>;
    using FromM = Vc::native_simd_mask<From>;
    using To = typename FromTo::template at<1>;
    call_with_typelist<vir::make_unique_typelist<
        Vc::rebind_simd_t<To, FromM>, Vc::native_simd_mask<To>,
        Vc::simd_mask<To>, Vc::simd_mask<To, Vc::simd_abi::scalar>>>([](auto _b) {
        using ToM = decltype(_b);
        using ToV = typename ToM::simd_type;

        using Vc::static_simd_cast;
        using Vc::simd_cast;
        using Vc::__proposed::resizing_simd_cast;

        auto x = resizing_simd_cast<ToM>(FromM());
        COMPARE(typeid(x), typeid(ToM));
        COMPARE(x, ToM());

        x = resizing_simd_cast<ToM>(FromM(true));
        const ToM ref = ToV([](auto i) { return i; }) < int(FromM::size());
        COMPARE(x, ref) << "converted from: " << FromM(true) << '\n'
                        << vir::typeToString<FromM>() << " -> "
                        << vir::typeToString<ToM>();

        const ullong all_bits = ~ullong() >> (64 - FromM::size());
        for (ullong bit_pos = 1; bit_pos /*until overflow*/; bit_pos *= 2) {
            for (ullong bits : {bit_pos & all_bits, ~bit_pos & all_bits}) {
                const auto from = FromM::from_bitset(bits);
                const auto to = resizing_simd_cast<ToM>(from);
                COMPARE(to, ToM::from_bitset(bits))
                    << "\nfrom: " << from << "\nbits: " << std::hex << bits << std::dec
                    << '\n'
                    << vir::typeToString<FromM>() << " -> " << vir::typeToString<ToM>();
                for (std::size_t i = 0; i < ToM::size(); ++i) {
                    COMPARE(to[i], (bits >> i) & 1)
                        << "\nfrom: " << from << "\nto: " << to << "\nbits: " << std::hex
                        << bits << std::dec << "\ni: " << i << '\n'
                        << vir::typeToString<FromM>() << " -> "
                        << vir::typeToString<ToM>();
                }
            }
        }
        ADD_PASS() << vir::typeToString<FromM>() << " -> " << vir::typeToString<ToM>();
    });
}
