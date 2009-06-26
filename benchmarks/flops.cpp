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

using namespace Vc;

static const int factor = 1000000;

int main()
{
    int blackHole = true;
    {
        Benchmark timer("SAXPY", 8. * float_v::Size * factor, "FLOP");
        for (int repetitions = 0; repetitions < 10; ++repetitions) {
            float_v alpha[4] = {
                float_v(repetitions + 0.2f),
                float_v(repetitions - 0.2f),
                float_v(repetitions + 0.1f),
                float_v(repetitions - 0.1f)
            };
            float_v x[4] = { 2.9f, 3.2f, 1.4f, 2.1f };
            float_v y[4] = { 1.2f, 0.2f, -1.4f, 4.3f };

            timer.Start();
            ///////////////////////////////////////

            for (int i = 0; i < factor; ++i) {
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
        Benchmark timer("SAXPY (reference)", 8. * float_v::Size * factor, "FLOP");
        for (int repetitions = 0; repetitions < 10; ++repetitions) {
#ifdef USE_SSE
            __m128 tmp = _mm_set1_ps(static_cast<float>(repetitions));
            const __m128 oPoint2 = _mm_set1_ps(0.2f);
            const __m128 oPoint1 = _mm_set1_ps(0.1f);
            __m128 alpha[4] = {
                _mm_add_ps(tmp, oPoint2),
                _mm_sub_ps(tmp, oPoint2),
                _mm_add_ps(tmp, oPoint1),
                _mm_sub_ps(tmp, oPoint1)
            };
            __m128 x[4] = { _mm_set1_ps(2.9f), _mm_set1_ps(3.2f), _mm_set1_ps(1.4f), _mm_set1_ps(2.1f) };
            __m128 y[4] = { _mm_set1_ps(1.2f), _mm_set1_ps(0.2f), _mm_set1_ps(-1.4f), _mm_set1_ps(4.3f) };

            timer.Start();
            ///////////////////////////////////////

            for (int i = 0; i < factor; ++i) {
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
            const __m512 oPoint2 = _mm512_set_1to16_ps(0.2f);
            const __m512 oPoint1 = _mm512_set_1to16_ps(0.1f);
            __m512 alpha[4] = {
                _mm512_add_ps(tmp, oPoint2),
                _mm512_sub_ps(tmp, oPoint2),
                _mm512_add_ps(tmp, oPoint1),
                _mm512_sub_ps(tmp, oPoint1)
            };
            __m512 x[4] = { _mm512_set_1to16_ps(2.9f), _mm512_set_1to16_ps(3.2f), _mm512_set_1to16_ps(1.4f), _mm512_set_1to16_ps(2.1f) };
            __m512 y[4] = { _mm512_set_1to16_ps(1.2f), _mm512_set_1to16_ps(0.2f), _mm512_set_1to16_ps(-1.4f), _mm512_set_1to16_ps(4.3f) };

            timer.Start();
            ///////////////////////////////////////

            for (int i = 0; i < factor; ++i) {
                    x[0] = _mm512_madd132_ps(alpha[0], y[0], x[0]);
                    x[1] = _mm512_madd132_ps(alpha[1], y[1], x[1]);
                    x[2] = _mm512_madd132_ps(alpha[2], y[2], x[2]);
                    x[3] = _mm512_madd132_ps(alpha[3], y[3], x[3]);
            }

            ///////////////////////////////////////
            timer.Stop();

            const int k = _mm512_cmpeq_ps(_mm512_add_ps(x[0], x[1]), _mm512_add_ps(x[2], x[3]));
            blackHole &= k;
#else
            float alpha[4] = {
                float(repetitions + 0.2f),
                float(repetitions - 0.2f),
                float(repetitions + 0.1f),
                float(repetitions - 0.1f)
            };
            float x[4] = { 2.9f, 3.2f, 1.4f, 2.1f };
            float y[4] = { 1.2f, 0.2f, -1.4f, 4.3f };

            timer.Start();
            ///////////////////////////////////////

            for (int i = 0; i < factor; ++i) {
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
    if (blackHole == 82934) {
        std::cout << std::endl;
    }
    return 0;
}
