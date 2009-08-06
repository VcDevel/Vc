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
        Benchmark timer("class", 8. * float_v::Size * Factor, "FLOP");
        for (int repetitions = 0; repetitions < Repetitions; ++repetitions) {
            const float_v alpha[4] = {
                float_v(repetitions + randomF(.1f, .2f)),
                float_v(repetitions - randomF(.1f, .2f)),
                float_v(repetitions + randomF(.1f, .2f)),
                float_v(repetitions - randomF(.1f, .2f))
            };
            float_v x[4] = { randomF12(), randomF12(), randomF12(), randomF12() };
            const float_v y[4] = { randomF12(), randomF12(), randomF12(), randomF12() };

            // force the x vectors to registers, otherwise GCC decides to work on the stack and
            // lose half of the performance
            forceToRegisters(x[0], x[1], x[2], x[3]);

            timer.Start();
            ///////////////////////////////////////

            for (int i = 0; i < Factor; ++i) {
                    x[0] = alpha[0] * x[0] + y[0];
                    x[1] = alpha[1] * x[1] + y[1];
                    x[2] = alpha[2] * x[2] + y[2];
                    x[3] = alpha[3] * x[3] + y[3];
            }

            ///////////////////////////////////////
            timer.Stop();

            const int k = (x[0] < x[1]) && (x[2] < x[3]);
            blackHole &= k;
        }
        timer.Print(Benchmark::PrintAverage);
    }

    // reference
    {
        Benchmark timer("reference", 8. * float_v::Size * Factor, "FLOP");
        for (int repetitions = 0; repetitions < Repetitions; ++repetitions) {
#ifdef USE_SSE
            __m128 tmp = _mm_set1_ps(static_cast<float>(repetitions));
            const __m128 oPoint2 = _mm_set1_ps(randomF(.1f, .2f));
            const __m128 oPoint1 = _mm_set1_ps(randomF(.1f, .2f));
            const __m128 alpha[4] = {
                _mm_add_ps(tmp, oPoint2),
                _mm_sub_ps(tmp, oPoint2),
                _mm_add_ps(tmp, oPoint1),
                _mm_sub_ps(tmp, oPoint1)
            };
            __m128 x[4] = { _mm_set1_ps(randomF12()), _mm_set1_ps(randomF12()), _mm_set1_ps(randomF12()), _mm_set1_ps(randomF12()) };
            const __m128 y[4] = { _mm_set1_ps(randomF12()), _mm_set1_ps(randomF12()), _mm_set1_ps(randomF12()), _mm_set1_ps(randomF12()) };

            timer.Start();
            ///////////////////////////////////////

            for (int i = 0; i < Factor; ++i) {
                    x[0] = _mm_add_ps(_mm_mul_ps(alpha[0], x[0]), y[0]);
                    x[1] = _mm_add_ps(_mm_mul_ps(alpha[1], x[1]), y[1]);
                    x[2] = _mm_add_ps(_mm_mul_ps(alpha[2], x[2]), y[2]);
                    x[3] = _mm_add_ps(_mm_mul_ps(alpha[3], x[3]), y[3]);
            }

            ///////////////////////////////////////
            timer.Stop();

            const int k = _mm_movemask_ps(_mm_add_ps(_mm_add_ps(x[0], x[1]), _mm_add_ps(x[2], x[3])));
            blackHole &= k;
#elif defined(ENABLE_LARRABEE)
            __m512 tmp = _mm512_set_1to16_ps(static_cast<float>(repetitions));
            const __m512 oPoint2 = _mm512_set_1to16_ps(randomF(.1f, .2f));
            const __m512 oPoint1 = _mm512_set_1to16_ps(randomF(.1f, .2f));
            const __m512 alpha[4] = {
                _mm512_add_ps(tmp, oPoint2),
                _mm512_sub_ps(tmp, oPoint2),
                _mm512_add_ps(tmp, oPoint1),
                _mm512_sub_ps(tmp, oPoint1)
            };
            __m512 x[4] = { _mm512_set_1to16_ps(randomF12()), _mm512_set_1to16_ps(randomF12()), _mm512_set_1to16_ps(randomF12()), _mm512_set_1to16_ps(randomF12()) };
            const __m512 y[4] = { _mm512_set_1to16_ps(randomF12()), _mm512_set_1to16_ps(randomF12()), _mm512_set_1to16_ps(randomF12()), _mm512_set_1to16_ps(randomF12()) };

            timer.Start();
            ///////////////////////////////////////

            for (int i = 0; i < Factor; ++i) {
                    x[0] = _mm512_madd132_ps(x[0], y[0], alpha[0]);
                    x[1] = _mm512_madd132_ps(x[1], y[1], alpha[1]);
                    x[2] = _mm512_madd132_ps(x[2], y[2], alpha[2]);
                    x[3] = _mm512_madd132_ps(x[3], y[3], alpha[3]);
            }

            ///////////////////////////////////////
            timer.Stop();

            const int k = _mm512_cmpeq_ps(_mm512_add_ps(x[0], x[1]), _mm512_add_ps(x[2], x[3]));
            blackHole &= k;
#else
            const float alpha[4] = {
                float(repetitions + randomF(.1f, .2f)),
                float(repetitions - randomF(.1f, .2f)),
                float(repetitions + randomF(.1f, .2f)),
                float(repetitions - randomF(.1f, .2f))
            };
            float x[4] = { randomF12(), randomF12(), randomF12(), randomF12() };
            const float y[4] = { randomF12(), randomF12(), randomF12(), randomF12() };

            timer.Start();
            ///////////////////////////////////////

            for (int i = 0; i < Factor; ++i) {
                    x[0] = alpha[0] * x[0] + y[0];
                    x[1] = alpha[1] * x[1] + y[1];
                    x[2] = alpha[2] * x[2] + y[2];
                    x[3] = alpha[3] * x[3] + y[3];
            }

            ///////////////////////////////////////
            timer.Stop();

            const int k = (x[0] < x[1]) && (x[2] < x[3]);
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
