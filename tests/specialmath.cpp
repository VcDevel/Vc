/*{{{
Copyright © 2018-2019 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH
                      Matthias Kretz <m.kretz@gsi.de>

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

//#define _GLIBCXX_SIMD_DEBUG frexp
//#define UNITTEST_ONLY_XTEST 1
#include "unittest.h"
#include "metahelpers.h"
#include <cmath>    // abs & sqrt
#include "test_values.h"

template <class... Ts> using base_template = std::experimental::simd<Ts...>;
#include "testtypes.h"

template <class T> void nan_cleanup(T &totest, T &expect)
{
    COMPARE(isnan(totest), isnan(expect));
    where(isnan(totest), expect) = 0;
    where(isnan(totest), totest) = 0;
}

TEST_TYPES(V, laguerre, real_test_types)  //{{{1
{
    using limits = std::numeric_limits<typename V::value_type>;
    test_values<V>(
        {limits::quiet_NaN(), limits::infinity(), +0., -0., limits::denorm_min(),
         limits::min(), limits::max(), limits::min() / 3},
        {10000, 0, limits::max() / 2}, [](const V &v) {
            for (unsigned int n : {0, 1, 4, 7, 10, 100}) {
                auto totest = laguerre(n, v);
                using R = decltype(totest);
                R expect([&](auto i) { return std::laguerre(n, v[i]); });
                nan_cleanup(totest, expect);
                FUZZY_COMPARE(totest, expect);
                for (unsigned int m : {0, 1, 4, 7, 10, 100}) {
                    totest = assoc_laguerre(n, m, v);
                    expect = R([&](auto i) { return std::assoc_laguerre(n, m, v[i]); });
                    nan_cleanup(totest, expect);
                    FUZZY_COMPARE(totest, expect);
                }
            }
        });
}

TEST_TYPES(V, legendre, real_test_types)  //{{{1
{
    using limits = std::numeric_limits<typename V::value_type>;
    test_values<V>(
        {limits::quiet_NaN(), +0., -0., limits::denorm_min(), limits::min(),
         limits::min() / 3},
        {10000, -1, 1}, [](const V &v) {
            for (unsigned int l : {0, 1, 4, 7, 10, 100}) {
                auto totest = legendre(l, v);
                using R = decltype(totest);
                R expect([&](auto i) { return std::legendre(l, v[i]); });
                nan_cleanup(totest, expect);
                FUZZY_COMPARE(totest, expect);
                for (unsigned int m : {0, 1, 4, 7, 10, 100, 123, 345, 1234}) {
                    if (m > l) break;  // TODO: not according to the standard
                    totest = assoc_legendre(l, m, v);
                    expect = R([&](auto i) { return std::assoc_legendre(l, m, v[i]); });
                    nan_cleanup(totest, expect);
                    FUZZY_COMPARE(totest, expect);
                }
            }
        });
}

