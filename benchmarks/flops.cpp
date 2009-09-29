/*  This file is part of the Vc library.

    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with Vc.  If not, see <http://www.gnu.org/licenses/>.

*/

#include <Vc/float_v>
#include "benchmark.h"
#include <cstdio>
#include <cstdlib>

using namespace Vc;

enum {
    Factor = 2000000 / float_v::Size
};

static float randomF(float min, float max)
{
    const float delta = max - min;
    return min + delta * rand() / RAND_MAX;
}

static float randomF12() { return randomF(1.f, 2.f); }

int bmain(Benchmark::OutputMode out)
{
    const int Repetitions = out == Benchmark::Stdout ? 10 : 100;

    int blackHole = true;
    {
        Benchmark timer("class", 2 * 7 * float_v::Size * Factor, "FLOP");
        for (int repetitions = 0; repetitions < Repetitions; ++repetitions) {
            const float_v alpha(repetitions - randomF(.1f, .2f));
            const float_v y = randomF12();
            float_v x[7] = { randomF12(), randomF12(), randomF12(), randomF12(), randomF12(), randomF12(), randomF12() };

            // force the x vectors to registers, otherwise GCC decides to work on the stack and
            // lose half of the performance
            forceToRegisters(x[0], x[1], x[2], x[3], x[4], x[5], x[6]);

            timer.Start();
            ///////////////////////////////////////

            for (int i = 0; i < Factor; ++i) {
                    x[0] = y * x[0] + y;
                    x[1] = y * x[1] + y;
                    x[2] = y * x[2] + y;
                    x[3] = y * x[3] + y;
                    x[4] = y * x[4] + y;
                    x[5] = y * x[5] + y;
                    x[6] = y * x[6] + y;
            }

            ///////////////////////////////////////
            timer.Stop();

            const int k = (x[0] < x[1]) && (x[2] < x[3]) && (x[4] < x[5]) && (x[0] < x[6]);
            blackHole &= k;
        }
        timer.Print(Benchmark::PrintAverage);
    }

    // asm reference
#ifdef __GNUC__
    {
        Benchmark timer("asm reference", 2 * 7 * float_v::Size * Factor, "FLOP");
        for (int repetitions = 0; repetitions < Repetitions; ++repetitions) {
#if VC_IMPL_SSE
            __m128 x[7] = { _mm_set1_ps(randomF12()), _mm_set1_ps(randomF12()), _mm_set1_ps(randomF12()), _mm_set1_ps(randomF12()), _mm_set1_ps(randomF12()), _mm_set1_ps(randomF12()), _mm_set1_ps(randomF12()) };
            const __m128 y = _mm_set1_ps(randomF12());

            timer.Start();
            ///////////////////////////////////////
            int i = Factor;
            __asm__(
                    ".align 16\n\t0: "
                    "mulps  %8,%0"   "\n\t"
                    "sub    $1,%7"   "\n\t"
                    "mulps  %8,%1"   "\n\t"
                    "mulps  %8,%2"   "\n\t"
                    "addps  %8,%0"   "\n\t"
                    "mulps  %8,%3"   "\n\t"
                    "addps  %8,%1"   "\n\t"
                    "mulps  %8,%4"   "\n\t"
                    "addps  %8,%2"   "\n\t"
                    "mulps  %8,%5"   "\n\t"
                    "addps  %8,%3"   "\n\t"
                    "addps  %8,%4"   "\n\t"
                    "mulps  %8,%6"   "\n\t"
                    "addps  %8,%5"   "\n\t"
                    "addps  %8,%6"   "\n\t"
                    "jne 0b"         "\n\t"
                    : "+x"(x[0]), "+x"(x[1]), "+x"(x[2]), "+x"(x[3]), "+x"(x[4]), "+x"(x[5]), "+x"(x[6]), "+r"(i)
                    : "x"(y));
            ///////////////////////////////////////
            timer.Stop();

            const int k = _mm_movemask_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(x[0], x[1]), _mm_add_ps(x[2], x[3])), _mm_add_ps(_mm_add_ps(x[4], x[5]), x[6])));
            blackHole &= k;
#elif VC_IMPL_LRBni
            __m512 x[7] = { _mm512_set_1to16_ps(randomF12()), _mm512_set_1to16_ps(randomF12()), _mm512_set_1to16_ps(randomF12()), _mm512_set_1to16_ps(randomF12()), _mm512_set_1to16_ps(randomF12()), _mm512_set_1to16_ps(randomF12()), _mm512_set_1to16_ps(randomF12()) };
            const __m512 y = _mm512_set_1to16_ps(randomF12());

            timer.Start();
            ///////////////////////////////////////

            // TODO
            for (int i = 0; i < Factor; ++i) {
                    x[0] = _mm512_madd132_ps(x[0], y, y);
                    x[1] = _mm512_madd132_ps(x[1], y, y);
                    x[2] = _mm512_madd132_ps(x[2], y, y);
                    x[3] = _mm512_madd132_ps(x[3], y, y);
                    x[4] = _mm512_madd132_ps(x[4], y, y);
                    x[5] = _mm512_madd132_ps(x[5], y, y);
                    x[6] = _mm512_madd132_ps(x[6], y, y);
            }

            ///////////////////////////////////////
            timer.Stop();

            const int k = _mm512_cmpeq_ps(_mm512_add_ps(_mm512_add_ps(x[4], x[5]), _mm512_add_ps(x[0], x[1])), _mm512_add_ps(x[6], _mm512_add_ps(x[2], x[3])));
            blackHole &= k;
#else
            float x[7] = { randomF12(), randomF12(), randomF12(), randomF12(), randomF12(), randomF12(), randomF12() };
            const float y = randomF12();

            timer.Start();
            ///////////////////////////////////////
            int i = Factor;
            __asm__(
                    ".align 16\n\t0: "
                    "mulss  %8,%0"   "\n\t"
                    "sub    $1,%7"   "\n\t"
                    "mulss  %8,%1"   "\n\t"
                    "mulss  %8,%2"   "\n\t"
                    "addss  %8,%0"   "\n\t"
                    "mulss  %8,%3"   "\n\t"
                    "addss  %8,%1"   "\n\t"
                    "mulss  %8,%4"   "\n\t"
                    "addss  %8,%2"   "\n\t"
                    "mulss  %8,%5"   "\n\t"
                    "addss  %8,%3"   "\n\t"
                    "addss  %8,%4"   "\n\t"
                    "mulss  %8,%6"   "\n\t"
                    "addss  %8,%5"   "\n\t"
                    "addss  %8,%6"   "\n\t"
                    "jne 0b"         "\n\t"
                    : "+x"(x[0]), "+x"(x[1]), "+x"(x[2]), "+x"(x[3]), "+x"(x[4]), "+x"(x[5]), "+x"(x[6]), "+r"(i)
                    : "x"(y));
            ///////////////////////////////////////
            timer.Stop();

            const int k = (x[0] < x[1]) && (x[2] < x[3]) && (x[4] < x[5]) && (x[2] < x[6]);
            blackHole &= k;
#endif
        }
        timer.Print(Benchmark::PrintAverage);
    }
#endif

    // intrinsics reference
    {
        Benchmark timer("intrinsics reference", 2 * 7 * float_v::Size * Factor, "FLOP");
        for (int repetitions = 0; repetitions < Repetitions; ++repetitions) {
#if VC_IMPL_SSE
            __m128 x[7] = { _mm_set1_ps(randomF12()), _mm_set1_ps(randomF12()), _mm_set1_ps(randomF12()), _mm_set1_ps(randomF12()), _mm_set1_ps(randomF12()), _mm_set1_ps(randomF12()), _mm_set1_ps(randomF12()) };
            const __m128 y = _mm_set1_ps(randomF12());

            timer.Start();
            ///////////////////////////////////////

            for (int i = 0; i < Factor; ++i) {
                    x[0] = _mm_add_ps(_mm_mul_ps(y, x[0]), y);
                    x[1] = _mm_add_ps(_mm_mul_ps(y, x[1]), y);
                    x[2] = _mm_add_ps(_mm_mul_ps(y, x[2]), y);
                    x[3] = _mm_add_ps(_mm_mul_ps(y, x[3]), y);
                    x[4] = _mm_add_ps(_mm_mul_ps(y, x[4]), y);
                    x[5] = _mm_add_ps(_mm_mul_ps(y, x[5]), y);
                    x[6] = _mm_add_ps(_mm_mul_ps(y, x[6]), y);
            }

            ///////////////////////////////////////
            timer.Stop();

            const int k = _mm_movemask_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(x[0], x[1]), _mm_add_ps(x[2], x[3])), _mm_add_ps(_mm_add_ps(x[4], x[5]), x[6])));
            blackHole &= k;
#elif VC_IMPL_LRBni
            __m512 x[7] = { _mm512_set_1to16_ps(randomF12()), _mm512_set_1to16_ps(randomF12()), _mm512_set_1to16_ps(randomF12()), _mm512_set_1to16_ps(randomF12()), _mm512_set_1to16_ps(randomF12()), _mm512_set_1to16_ps(randomF12()), _mm512_set_1to16_ps(randomF12()) };
            const __m512 y = _mm512_set_1to16_ps(randomF12());

            timer.Start();
            ///////////////////////////////////////

            for (int i = 0; i < Factor; ++i) {
                    x[0] = _mm512_madd132_ps(x[0], y, y);
                    x[1] = _mm512_madd132_ps(x[1], y, y);
                    x[2] = _mm512_madd132_ps(x[2], y, y);
                    x[3] = _mm512_madd132_ps(x[3], y, y);
                    x[4] = _mm512_madd132_ps(x[4], y, y);
                    x[5] = _mm512_madd132_ps(x[5], y, y);
                    x[6] = _mm512_madd132_ps(x[6], y, y);
            }

            ///////////////////////////////////////
            timer.Stop();

            const int k = _mm512_cmpeq_ps(_mm512_add_ps(_mm512_add_ps(x[4], x[5]), _mm512_add_ps(x[0], x[1])), _mm512_add_ps(x[6], _mm512_add_ps(x[2], x[3])));
            blackHole &= k;
#else
            float x[7] = { randomF12(), randomF12(), randomF12(), randomF12(), randomF12(), randomF12(), randomF12() };
            const float y = randomF12();

            timer.Start();
            ///////////////////////////////////////

            for (int i = 0; i < Factor; ++i) {
                    x[0] = y * x[0] + y;
                    x[1] = y * x[1] + y;
                    x[2] = y * x[2] + y;
                    x[3] = y * x[3] + y;
                    x[4] = y * x[4] + y;
                    x[5] = y * x[5] + y;
                    x[6] = y * x[6] + y;
            }

            ///////////////////////////////////////
            timer.Stop();

            const int k = (x[0] < x[1]) && (x[2] < x[3]) && (x[4] < x[5]) && (x[2] < x[6]);
            blackHole &= k;
#endif
        }
        timer.Print(Benchmark::PrintAverage);
    }
    if (blackHole != 0) {
        std::cout << std::endl;
    }
    return 0;
}
