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
#include <Vc/simd>
#include "metahelpers.h"
#include <cmath>    // abs & sqrt
#include <cstdlib>  // integer abs

template <class... Ts> using base_template = Vc::simd<Ts...>;
#include "testtypes.h"

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
