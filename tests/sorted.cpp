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

TEST_TYPES(Vec, testSort, concat<AllVectors, SimdArrays<15>, SimdArrays<8>, SimdArrays<3>, SimdArrays<1>>)
{
// On GCC/clang (i.e. __GNUC__ compatible) __OPTIMIZE__ is not defined on -O0.
// We use this information to make the test complete in a sane timeframe on debug
// builds.
#if !defined __GNUC__ || defined __OPTIMIZE__
    Vec ref(Vc::IndexesFromZero);
    Vec a;
    using limits = std::numeric_limits<typename Vec::value_type>;
    if (limits::has_infinity) {
        ref[0] = -std::numeric_limits<typename Vec::value_type>::infinity();
        if (Vec::size() > 2) {
            ref[Vec::Size - 1] =
                std::numeric_limits<typename Vec::value_type>::infinity();
        }
    }

    int maxPerm = 1;
    for (int x = Vec::Size; x > 0 && maxPerm < 200000; --x) {
        maxPerm *= x;
    }
    for (int perm = 0; perm < maxPerm; ++perm) {
        int rest = perm;
        for (size_t i = 0; i < Vec::Size; ++i) {
            a[i] = 0;
            for (size_t j = 0; j < i; ++j) {
                if (a[i] == a[j]) {
                    ++a[i];
                    j = -1;
                }
            }
            a[i] += rest % (Vec::Size - i);
            rest /= (Vec::Size - i);
            for (size_t j = 0; j < i; ++j) {
                if (a[i] == a[j]) {
                    ++a[i];
                    j = -1;
                }
            }
        }
        if (limits::has_infinity) {
            where(a == 0) | a =
                -std::numeric_limits<typename Vec::value_type>::infinity();
            if (Vec::size() > 2) {
                where(a == int(Vec::Size) - 1) | a =
                    std::numeric_limits<typename Vec::value_type>::infinity();
            }
        }
        COMPARE(a.sorted(), ref) << ", a: " << a;
    }
#endif

    for (int repetition = 0; repetition < 1000; ++repetition) {
        Vec test = Vec::Random();
        alignas(static_cast<size_t>(
            Vec::MemoryAlignment)) typename Vec::EntryType reference[Vec::Size] = {};
        test.store(&reference[0], Vc::Aligned);
        std::sort(std::begin(reference), std::end(reference));
        COMPARE(test.sorted(), Vec(&reference[0], Vc::Aligned));
    }
}

// vim: foldmethod=marker
