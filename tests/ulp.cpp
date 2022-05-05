/*  This file is part of the Vc library. {{{
Copyright © 2015 Matthias Kretz <kretz@kde.org>

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

#include "unittest.h"

TEST_TYPES(V, testUlpDiff, concat<RealVectors, RealSimdArrayList>)  //{{{1
{
    // MSVC takes too long in debug mode
#if defined _MSC_VER && defined _DEBUG
    const auto range = 1000;
#else
    const auto range = 10000;
#endif

    typedef typename V::EntryType T;

    using vir::detail::ulpDiffToReference;
    COMPARE(ulpDiffToReference(V(0), V(0)), V(0));
    COMPARE(ulpDiffToReference(V(std::numeric_limits<T>::min()), V(0)), V(1));
    COMPARE(ulpDiffToReference(V(0), V(std::numeric_limits<T>::min())), V(1));
    for (size_t count = 0; count < 1024 / V::Size; ++count) {
        const V base = (V::Random() - T(0.5)) * T(1000);
        typename V::IndexType exp;
        frexp(base, &exp);
        const V eps = ldexp(V(std::numeric_limits<T>::epsilon()), exp - 1);
        //std::cout << base << ", " << exp << ", " << eps << std::endl;
        for (int i = -range; i <= range; ++i) {
            const V i_v = V(T(i));
            const V diff = base + i_v * eps;

            // if diff and base have a different exponent then ulpDiffToReference has an uncertainty
            // of +/-1
            const V ulpDifference = ulpDiffToReference(diff, base);
            const V expectedDifference = abs(i_v);
            const V maxUncertainty = abs(exponent(abs(diff)) - exponent(abs(base)));

            VERIFY(all_of(abs(ulpDifference - expectedDifference) <= maxUncertainty))
                << ", base = " << base << ", epsilon = " << eps << ", diff = " << diff;
            for (size_t k = 0; k < V::Size; ++k) {
                VERIFY(std::abs(ulpDifference[k] - expectedDifference[k]) <= maxUncertainty[k]);
            }
        }
    }
}

//}}}1
// vim: foldmethod=marker
