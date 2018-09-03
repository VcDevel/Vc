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

#ifdef _WIN32
void bzero(void *p, size_t n) { memset(p, 0, n); }
#else
#include <strings.h>
#endif

using Vc::float_v;
using Vc::double_v;
using Vc::SimdArray;

TEST_TYPES(V, Random, AllVectors)
{
    typedef typename V::EntryType T;
    enum {
        NBits = 3,
        NBins = 1 << NBits,                        // short int
        TotalBits = sizeof(T) * 8,                 //    16  32
        RightShift = TotalBits - NBits,            //    13  29
        NHistograms = TotalBits - NBits + 1,       //    14  30
        LeftShift = (RightShift + 1) / NHistograms,//     1   1
        Mean = 135791,
        MinGood = Mean - Mean/10,
        MaxGood = Mean + Mean/10
    };
    const V mask((1 << NBits) - 1);
    int histogram[NHistograms][NBins];
    bzero(&histogram[0][0], sizeof(histogram));
    for (size_t i = 0; i < NBins * Mean / V::Size; ++i) {
        const V rand = V::Random();
        for (size_t hist = 0; hist < NHistograms; ++hist) {
            const V bin = ((rand << (hist * LeftShift)) >> RightShift) & mask;
            for (size_t k = 0; k < V::Size; ++k) {
                ++histogram[hist][bin[k]];
            }
        }
    }
//#define PRINT_RANDOM_HISTOGRAM
#ifdef PRINT_RANDOM_HISTOGRAM
    for (size_t hist = 0; hist < NHistograms; ++hist) {
        std::cout << "histogram[" << std::setw(2) << hist << "]: ";
        for (size_t bin = 0; bin < NBins; ++bin) {
            std::cout << std::setw(3) << (histogram[hist][bin] - Mean) * 1000 / Mean << "|";
        }
        std::cout << std::endl;
    }
#endif
    for (size_t hist = 0; hist < NHistograms; ++hist) {
        for (size_t bin = 0; bin < NBins; ++bin) {
            VERIFY(histogram[hist][bin] > MinGood)
                << " bin = " << bin << " is " << histogram[0][bin];
            VERIFY(histogram[hist][bin] < MaxGood)
                << " bin = " << bin << " is " << histogram[0][bin];
        }
    }
}

template<typename V, typename I> void FloatRandom()
{
    typedef typename V::EntryType T;
    enum {
        NBins = 64,
        NHistograms = 1,
        Mean = 135791,
        MinGood = Mean - Mean/10,
        MaxGood = Mean + Mean/10
    };
    int histogram[NHistograms][NBins];
    bzero(&histogram[0][0], sizeof(histogram));
    for (size_t i = 0; i < NBins * Mean / V::Size; ++i) {
        const V rand = V::Random();
        const I bin = static_cast<I>(rand * T(NBins));
        for (size_t k = 0; k < V::Size; ++k) {
            ++histogram[0][bin[k]];
        }
    }
#ifdef PRINT_RANDOM_HISTOGRAM
    for (size_t hist = 0; hist < NHistograms; ++hist) {
        std::cout << "histogram[" << std::setw(2) << hist << "]: ";
        for (size_t bin = 0; bin < NBins; ++bin) {
            std::cout << std::setw(3) << (histogram[hist][bin] - Mean) * 1000 / Mean << "|";
        }
        std::cout << std::endl;
    }
#endif
    for (size_t hist = 0; hist < NHistograms; ++hist) {
        for (size_t bin = 0; bin < NBins; ++bin) {
            VERIFY(histogram[hist][bin] > MinGood)
                << " bin = " << bin << " is " << histogram[0][bin];
            VERIFY(histogram[hist][bin] < MaxGood)
                << " bin = " << bin << " is " << histogram[0][bin];
        }
    }
}

namespace Tests
{
template <> void Random_<float_v>::run()
{
    FloatRandom<float_v, SimdArray<int, float_v::size()>>();
}
template <> void Random_<double_v>::run()
{
    FloatRandom<double_v, SimdArray<int, double_v::size()>>();
}
}  // namespace Tests

// vim: foldmethod=marker
