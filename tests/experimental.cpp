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

#define Vc_EXPERIMENTAL 1
#include <vir/test.h>
#include <Vc/Vc>

template <class... Ts> using base_template = Vc::datapar<Ts...>;
#include "testtypes.h"

TEST_TYPES(V, where_apply, all_test_types)
{
    using T = typename V::value_type;

    for (int split = 0; split <= int(V::size());++split) {
        V a([](T i) { return i; });
        where(a > split, a)
            .apply([](auto &&masked) { masked = 1; })
            .apply_inv([](auto &&masked) { masked = 2; });
        COMPARE(a, V([split](int i) { return i > split ? T(1) : T(2); }));

        V b = 0;
        where(a == 1, a, b)
            .apply([](auto &&a_, auto &&b_) { a_ = b_; })
            .apply_inv([](auto &&a_, auto &&b_) { b_ = a_; });
        COMPARE(a, V([split](int i) { return i > split ? T(0) : T(2); }));
        COMPARE(b, V([split](int i) { return i > split ? T(0) : T(2); }));
    }
}

TEST_TYPES(V, generators, ALL_TYPES)
{
    COMPARE(V::seq(), make_vec<V>({0, 1}, 2));
}
