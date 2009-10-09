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

#include <Vc/Vc>
#include <emmintrin.h>
#include "benchmark.h"

#include <cstdlib>

using namespace Vc;

static const unsigned int allone[4] = { 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff };
static const unsigned int absMask[4] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };

__m128 blackHole = _mm_setzero_ps();

enum {
    Factor = 512000
};

int bmain(Benchmark::OutputMode out)
{
    const int Repetitions = out == Benchmark::Stdout ? 3 : g_Repetitions > 0 ? g_Repetitions : 10;

    {
        Benchmark timer("constant, shuffled one", Factor, "Op");
        for (int rep = 0; rep < Repetitions; ++rep) {
            timer.Start();
            for (int i = 0; i < Factor; ++i) {
                asm volatile("mov $0xffffffff,%%eax" :::"eax");
                asm volatile("movd %%eax,%%xmm0" :::"xmm0", "eax");
                asm volatile("pshufd $0,%%xmm0,%%xmm0" :::"xmm0");
            }
            timer.Stop();
        }
        timer.Print(Benchmark::PrintAverage);
    }
    {
        Benchmark timer("load 4 bytes, shuffled one", Factor, "Op");
        for (int rep = 0; rep < Repetitions; ++rep) {
            timer.Start();
            for (int i = 0; i < Factor; ++i) {
                asm volatile("movss %0,%%xmm0" ::"m"(allone[0]):"xmm0");
                asm volatile("shufps $0,%%xmm0,%%xmm0" :::"xmm0");
            }
            timer.Stop();
        }
        timer.Print(Benchmark::PrintAverage);
    }
    {
        Benchmark timer("generated one", Factor, "Op");
        for (int rep = 0; rep < Repetitions; ++rep) {
            timer.Start();
            for (int i = 0; i < Factor; ++i) {
                asm volatile("pcmpeqw %%xmm0,%%xmm0" :::"xmm0");
            }
            timer.Stop();
        }
        timer.Print(Benchmark::PrintAverage);
    }
    {
        Benchmark timer("loaded one", Factor, "Op");
        for (int rep = 0; rep < Repetitions; ++rep) {
            timer.Start();
            for (int i = 0; i < Factor; ++i) {
                register __m128 tmp;
                asm volatile("movaps %1,%0" : "=x"(tmp) : "m"(allone[0]));
            }
            timer.Stop();
        }
        timer.Print(Benchmark::PrintAverage);
    }

    {
        Benchmark timer("constant, shuffled abs", 4 * Factor, "Op");
        for (int rep = 0; rep < Repetitions; ++rep) {
            timer.Start();
            asm volatile("xorps %%xmm1,%%xmm1" :::"xmm1");
            asm volatile("xorps %%xmm3,%%xmm3" :::"xmm3");
            asm volatile("xorps %%xmm5,%%xmm5" :::"xmm5");
            asm volatile("xorps %%xmm7,%%xmm7" :::"xmm7");
            for (int i = 0; i < Factor; ++i) {
                asm volatile("mov $0x7fffffff,%%eax" :::"eax");
                asm volatile("movd %%eax,%%xmm0" :::"xmm0", "eax");
                asm volatile("pshufd $0,%%xmm0,%%xmm0" :::"xmm0");
                asm volatile("pand %%xmm0,%%xmm1" :::"xmm0", "xmm1");

                asm volatile("mov $0x7fffffff,%%ecx" :::"ecx");
                asm volatile("movd %%ecx,%%xmm2" :::"xmm2", "ecx");
                asm volatile("pshufd $0,%%xmm2,%%xmm2" :::"xmm2");
                asm volatile("pand %%xmm2,%%xmm3" :::"xmm2", "xmm3");

                asm volatile("mov $0x7fffffff,%%edx" :::"edx");
                asm volatile("movd %%edx,%%xmm4" :::"xmm4", "edx");
                asm volatile("pshufd $0,%%xmm4,%%xmm4" :::"xmm4");
                asm volatile("pand %%xmm4,%%xmm5" :::"xmm4", "xmm5");

                asm volatile("mov $0x7fffffff,%%esi" :::"esi");
                asm volatile("movd %%esi,%%xmm6" :::"xmm6", "esi");
                asm volatile("pshufd $0,%%xmm6,%%xmm6" :::"xmm6");
                asm volatile("pand %%xmm6,%%xmm7" :::"xmm6", "xmm7");
            }
            timer.Stop();
        }
        timer.Print(Benchmark::PrintAverage);
    }
    {
        Benchmark timer("load 4 bytes, shuffled abs", 4 * Factor, "Op");
        for (int rep = 0; rep < Repetitions; ++rep) {
            timer.Start();
            asm volatile("xorps %%xmm1,%%xmm1" :::"xmm1");
            asm volatile("xorps %%xmm3,%%xmm3" :::"xmm3");
            asm volatile("xorps %%xmm5,%%xmm5" :::"xmm5");
            asm volatile("xorps %%xmm7,%%xmm7" :::"xmm7");
            for (int i = 0; i < Factor; ++i) {
                asm volatile("movss %0,%%xmm0" ::"m"(absMask[0]):"xmm0");
                asm volatile("shufps $0,%%xmm0,%%xmm0" :::"xmm0");
                asm volatile("pand %%xmm0,%%xmm1" :::"xmm0", "xmm1");

                asm volatile("movss %0,%%xmm2" ::"m"(absMask[0]):"xmm2");
                asm volatile("shufps $0,%%xmm2,%%xmm2" :::"xmm2");
                asm volatile("pand %%xmm2,%%xmm3" :::"xmm2", "xmm3");

                asm volatile("movss %0,%%xmm4" ::"m"(absMask[0]):"xmm4");
                asm volatile("shufps $0,%%xmm4,%%xmm4" :::"xmm4");
                asm volatile("pand %%xmm4,%%xmm5" :::"xmm4", "xmm5");

                asm volatile("movss %0,%%xmm6" ::"m"(absMask[0]):"xmm6");
                asm volatile("shufps $0,%%xmm6,%%xmm6" :::"xmm6");
                asm volatile("pand %%xmm6,%%xmm7" :::"xmm6", "xmm7");
            }
            timer.Stop();
        }
        timer.Print(Benchmark::PrintAverage);
    }
    {
        Benchmark timer("generated abs", 4 * Factor, "Op");
        for (int rep = 0; rep < Repetitions; ++rep) {
            timer.Start();
            asm volatile("xorps %%xmm1,%%xmm1" :::"xmm1");
            asm volatile("xorps %%xmm3,%%xmm3" :::"xmm3");
            asm volatile("xorps %%xmm5,%%xmm5" :::"xmm5");
            asm volatile("xorps %%xmm7,%%xmm7" :::"xmm7");
            for (int i = 0; i < Factor; ++i) {
                asm volatile("pcmpeqw %%xmm0,%%xmm0" :::"xmm0");
                asm volatile("psrld $1,%%xmm0" :::"xmm0");
                asm volatile("pand %%xmm0,%%xmm1" :::"xmm0", "xmm1");

                asm volatile("pcmpeqw %%xmm2,%%xmm2" :::"xmm2");
                asm volatile("psrld $1,%%xmm2" :::"xmm2");
                asm volatile("pand %%xmm2,%%xmm3" :::"xmm2", "xmm3");

                asm volatile("pcmpeqw %%xmm4,%%xmm4" :::"xmm4");
                asm volatile("psrld $1,%%xmm4" :::"xmm4");
                asm volatile("pand %%xmm4,%%xmm5" :::"xmm4", "xmm5");

                asm volatile("pcmpeqw %%xmm6,%%xmm6" :::"xmm6");
                asm volatile("psrld $1,%%xmm6" :::"xmm6");
                asm volatile("pand %%xmm6,%%xmm7" :::"xmm6", "xmm7");
            }
            timer.Stop();
        }
        timer.Print(Benchmark::PrintAverage);
    }
    {
        Benchmark timer("loaded abs", 4 * Factor, "Op");
        for (int rep = 0; rep < Repetitions; ++rep) {
            timer.Start();
            asm volatile("xorps %%xmm1,%%xmm1" :::"xmm1");
            asm volatile("xorps %%xmm2,%%xmm2" :::"xmm2");
            asm volatile("xorps %%xmm3,%%xmm3" :::"xmm3");
            asm volatile("xorps %%xmm4,%%xmm4" :::"xmm4");
            for (int i = 0; i < Factor; ++i) {
                asm volatile("pand %0,%%xmm1" :: "m"(absMask[0]) :"xmm1");
                asm volatile("pand %0,%%xmm2" :: "m"(absMask[0]) :"xmm2");
                asm volatile("pand %0,%%xmm3" :: "m"(absMask[0]) :"xmm3");
                asm volatile("pand %0,%%xmm4" :: "m"(absMask[0]) :"xmm4");
            }
            timer.Stop();
        }
        timer.Print(Benchmark::PrintAverage);
    }

    {
        Benchmark timer("constant, shuffled inversion", 4 * Factor, "Op");
        for (int rep = 0; rep < Repetitions; ++rep) {
            timer.Start();
            asm volatile("pxor %%xmm1,%%xmm1" :::"xmm1");
            asm volatile("pxor %%xmm3,%%xmm3" :::"xmm3");
            asm volatile("pxor %%xmm5,%%xmm5" :::"xmm5");
            asm volatile("pxor %%xmm7,%%xmm7" :::"xmm7");
            for (int i = 0; i < Factor; ++i) {
                asm volatile("mov $0xffffffff,%%eax" :::"eax");
                asm volatile("movd %%eax,%%xmm0" :::"xmm0", "eax");
                asm volatile("pshufd $0,%%xmm0,%%xmm0" :::"xmm0");
                asm volatile("pxor %%xmm0,%%xmm1" :::"xmm0", "xmm1");

                asm volatile("mov $0xffffffff,%%ecx" :::"ecx");
                asm volatile("movd %%ecx,%%xmm2" :::"xmm2", "ecx");
                asm volatile("pshufd $0,%%xmm2,%%xmm2" :::"xmm2");
                asm volatile("pxor %%xmm2,%%xmm3" :::"xmm2", "xmm3");

                asm volatile("mov $0xffffffff,%%edx" :::"edx");
                asm volatile("movd %%edx,%%xmm4" :::"xmm4", "edx");
                asm volatile("pshufd $0,%%xmm4,%%xmm4" :::"xmm4");
                asm volatile("pxor %%xmm4,%%xmm5" :::"xmm4", "xmm5");

                asm volatile("mov $0xffffffff,%%esi" :::"esi");
                asm volatile("movd %%esi,%%xmm6" :::"xmm6", "esi");
                asm volatile("pshufd $0,%%xmm6,%%xmm6" :::"xmm6");
                asm volatile("pxor %%xmm6,%%xmm7" :::"xmm6", "xmm7");
            }
            timer.Stop();
        }
        timer.Print(Benchmark::PrintAverage);
    }
    {
        Benchmark timer("load 4 bytes, shuffled inversion", 4 * Factor, "Op");
        for (int rep = 0; rep < Repetitions; ++rep) {
            timer.Start();
            asm volatile("xorps %%xmm1,%%xmm1" :::"xmm1");
            asm volatile("xorps %%xmm3,%%xmm3" :::"xmm3");
            asm volatile("xorps %%xmm5,%%xmm5" :::"xmm5");
            asm volatile("xorps %%xmm7,%%xmm7" :::"xmm7");
            for (int i = 0; i < Factor; ++i) {
                asm volatile("movss %0,%%xmm0" ::"m"(allone[0]):"xmm0");
                asm volatile("shufps $0,%%xmm0,%%xmm0" :::"xmm0");
                asm volatile("xorps %%xmm0,%%xmm1" :::"xmm0", "xmm1");

                asm volatile("movss %0,%%xmm2" ::"m"(allone[0]):"xmm2");
                asm volatile("shufps $0,%%xmm2,%%xmm2" :::"xmm2");
                asm volatile("xorps %%xmm2,%%xmm3" :::"xmm2", "xmm3");

                asm volatile("movss %0,%%xmm4" ::"m"(allone[0]):"xmm4");
                asm volatile("shufps $0,%%xmm4,%%xmm4" :::"xmm4");
                asm volatile("xorps %%xmm4,%%xmm5" :::"xmm4", "xmm5");

                asm volatile("movss %0,%%xmm6" ::"m"(allone[0]):"xmm6");
                asm volatile("shufps $0,%%xmm6,%%xmm6" :::"xmm6");
                asm volatile("xorps %%xmm6,%%xmm7" :::"xmm6", "xmm7");
            }
            timer.Stop();
        }
        timer.Print(Benchmark::PrintAverage);
    }
    {
        Benchmark timer("generated inversion", 4 * Factor, "Op");
        for (int rep = 0; rep < Repetitions; ++rep) {
            timer.Start();
            asm volatile("pxor %%xmm1,%%xmm1" :::"xmm1");
            asm volatile("pxor %%xmm3,%%xmm3" :::"xmm3");
            asm volatile("pxor %%xmm5,%%xmm5" :::"xmm5");
            asm volatile("pxor %%xmm7,%%xmm7" :::"xmm7");
            for (int i = 0; i < Factor; ++i) {
                asm volatile("pcmpeqw %%xmm0,%%xmm0" :::"xmm0");
                asm volatile("pxor %%xmm0,%%xmm1" :::"xmm0", "xmm1");

                asm volatile("pcmpeqw %%xmm2,%%xmm2" :::"xmm2");
                asm volatile("pxor %%xmm2,%%xmm3" :::"xmm2", "xmm3");

                asm volatile("pcmpeqw %%xmm4,%%xmm4" :::"xmm4");
                asm volatile("pxor %%xmm4,%%xmm5" :::"xmm4", "xmm5");

                asm volatile("pcmpeqw %%xmm6,%%xmm6" :::"xmm6");
                asm volatile("pxor %%xmm6,%%xmm7" :::"xmm6", "xmm7");
            }
            timer.Stop();
        }
        timer.Print(Benchmark::PrintAverage);
    }
    {
        Benchmark timer("loaded inversion", 4 * Factor, "Op");
        for (int rep = 0; rep < Repetitions; ++rep) {
            timer.Start();
            asm volatile("xorps %%xmm1,%%xmm1" :::"xmm1");
            asm volatile("xorps %%xmm2,%%xmm2" :::"xmm2");
            asm volatile("xorps %%xmm3,%%xmm3" :::"xmm3");
            asm volatile("xorps %%xmm4,%%xmm4" :::"xmm4");
            for (int i = 0; i < Factor; ++i) {
                asm volatile("xorps %0,%%xmm1" :: "m"(allone[0]) :"xmm1");
                asm volatile("xorps %0,%%xmm2" :: "m"(allone[0]) :"xmm2");
                asm volatile("xorps %0,%%xmm3" :: "m"(allone[0]) :"xmm3");
                asm volatile("xorps %0,%%xmm4" :: "m"(allone[0]) :"xmm4");
            }
            timer.Stop();
        }
        timer.Print(Benchmark::PrintAverage);
    }

    return 0;
}
