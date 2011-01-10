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

#include "benchmark.h"
#include <cstdio>
#include <cstdlib>

// FIXME: is this standard?
#ifdef __x86_64__
#define VC_64BIT
#else
#define VC_32BIT
#endif

enum {
    VectorSize = 4,
#define VectorAlignment 32
    Factor = 2000000 / VectorSize,
    Size = 8 * VectorSize
};

static float randomF(float min, float max)
{
    const float delta = max - min;
    return min + delta * rand() / RAND_MAX;
}

static float randomF12() { return randomF(1.f, 2.f); }

int blackHole = true;

int bmain()
{
    Benchmark timer("auto-vect reference", 2 * Size * Factor, "FLOP");
    while (timer.wantsMoreDataPoints()) {
        float x[Size] __attribute__((aligned(VectorAlignment)));
        for (int i = 0; i < Size; ++i) {
            x[i] = randomF12();
        }
        const float y = randomF12();

        timer.Start();
        ///////////////////////////////////////

        for (int i = 0; i < Factor; ++i) {
            for (int j = 0; j < Size; ++j) {
                x[j] = y * x[j] + y;
            }
        }

        ///////////////////////////////////////
        timer.Stop();

        for (int i = 0; i < Size; i += 2) {
            blackHole &= (x[i] < x[i + 1]);
        }
    }
    timer.Print();
    return 0;
}
