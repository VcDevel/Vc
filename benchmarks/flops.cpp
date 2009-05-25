/*
    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation; either version 2 of
    the License, or (at your option) version 3.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
    02110-1301, USA.

*/

#include <vector.h>
#include "benchmark.h"
#include <cstdio>

using namespace Vc;

static const int factor = 1000000;

int main()
{
    FILE *blackHole = std::fopen("/dev/null", "w");
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

        const bool k = (x[0] < x[1]) && (x[2] < x[3]);
        std::fwrite(&k, 1, 1, blackHole);
    }
    timer.Print(Benchmark::PrintAverage);
    return 0;
}
