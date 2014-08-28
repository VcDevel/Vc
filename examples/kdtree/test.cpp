/*{{{
Copyright © 2014 Matthias Kretz <kretz@kde.org>
All rights reserved.

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
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#include "simdize.h"

#include <array>
#include <bitset>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

int main()
{
    using namespace std;
    using namespace Vc;
    using std::array;
    static_assert(is_convertible<tuple<float_m>, simdize<tuple<bool>>>::value, "");
    static_assert(is_convertible<tuple<float_v, float_m>, simdize<tuple<float, bool>>>::value, "");
    static_assert(is_convertible<tuple<double_v, double_m>, simdize<tuple<double, bool>>>::value, "");
    static_assert(is_same<
                  typename std::decay<decltype(std::get<0>(simdize<tuple<double, bool>>()))>::type,
                  double_v
                  >::value, "");
    static_assert(is_same<
                  typename std::decay<decltype(std::get<1>(simdize<tuple<double, bool>>()))>::type,
                  double_m
                  >::value, "");

    static_assert(is_convertible<simdize<array<float, 3>>, array<float_v, 3>>::value, "");
    static_assert(is_convertible<array<float_v, 3>, simdize<array<float, 3>>>::value, "");
    static_assert(is_convertible<simdize<tuple<float>>, tuple<float_v>>::value, "");
    static_assert(is_convertible<tuple<float_v>, simdize<tuple<float>>>::value, "");
    static_assert(is_convertible<simdize<tuple<array<float, 3>>>, tuple<array<float_v, 3>>>::value, "");
    static_assert(is_convertible<tuple<array<float_v, 3>>, simdize<tuple<array<float, 3>>>>::value, "");
    static_assert(is_convertible<simdize<array<tuple<float>, 3>>, array<simdize<tuple<float>>, 3>>::value, "");
    static_assert(is_convertible<array<tuple<float_v>, 3>, simdize<array<tuple<float>, 3>>>::value, "");
    static_assert(is_convertible<vector<tuple<float_v>>, simdize<vector<tuple<float>>>>::value, "");
#if VC_DOUBLE_V_SIZE != VC_FLOAT_V_SIZE
    static_assert(is_convertible<
                  tuple<float_v, array<pair<float_v, simdarray<double, float_v::Size>>, 3>>,
                  simdize<tuple<float, array<pair<float, double>, 3>>>
                  >::value, "");
#else
    static_assert(
        is_convertible<tuple<float_v, array<pair<float_v, double_v>, 3>>,
                       simdize<tuple<float, array<pair<float, double>, 3>>>>::value,
        "");
#endif

    static_assert(is_same<float_v, simdize<float>>::value, "");
    static_assert(is_same<simdize<float>, float_v>::value, "");
    static_assert(is_same<simdize<string>, string>::value, "");
    static_assert(is_same<string, simdize<string>>::value, "");
    static_assert(is_same<bitset<8>, simdize<bitset<8>>>::value, "");

    simdize<string> s = "Hallo Welt";
    static_assert(is_convertible<tuple<float_v, string>, simdize<tuple<float, string>>>::value, "");

    simdize<tuple<float, int, double>> x = {float_v::IndexesFromZero(), 1, 1.};
    static_assert(
        std::is_same<decltype(simdize_get(x, 0)), tuple<float, int, double>>::value, "");
    static_assert(std::is_same<decltype(get<0>(x)), float_v &>::value, "");
    static_assert(std::is_same<decltype(get<1>(x)), int_v &>::value, "");
    static_assert(
        std::is_same<typename tuple_element<0, decltype(x)>::type, float_v>::value, "");
#if VC_DOUBLE_V_SIZE != VC_FLOAT_V_SIZE
    static_assert(
        std::is_same<decltype(get<2>(x)), simdarray<double, float_v::Size> &>::value, "");
    static_assert(std::is_same<typename tuple_element<2, decltype(x)>::type,
                               simdarray<double, float_v::Size>>::value,
                  "");
#else
    static_assert(std::is_same<decltype(get<2>(x)), double_v &>::value, "");
    static_assert(
        std::is_same<typename tuple_element<2, decltype(x)>::type, double_v>::value, "");
#endif

    return 0;
}
