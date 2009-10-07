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
static const float oneFloat[4] = { 1.f, 1.f, 1.f, 1.f };

__m128 blackHole = _mm_setzero_ps();

enum {
    Factor = 512000
};

int bmain(Benchmark::OutputMode out)
{
    const int Repetitions = out == Benchmark::Stdout ? 3 : g_Repetitions > 0 ? g_Repetitions : 10;
    //asm volatile("prefetchnta %0" :: "m"(allone[0]));

    {
        Benchmark timer("constant, shuffled one", Factor, "Op");
        for (int rep = 0; rep < Repetitions; ++rep) {
//X             asm volatile("clflush %0" :: "m"(allone[0]));
//X             asm volatile("cpuid" :::"eax", "ebx", "ecx", "edx");
//X             asm volatile("prefetcht1 %0" :: "m"(allone[0]));
//X             asm volatile("cpuid" :::"eax", "ebx", "ecx", "edx");
            timer.Start();
//X             register __m128 a = _mm_setzero_ps();
//X             register __m128 b = _mm_setzero_ps();
//X             register __m128 c = _mm_setzero_ps();
//X             register __m128 d = _mm_setzero_ps();
            for (int i = 0; i < Factor; ++i) {
                //asm volatile("prefetchnta %0" :: "m"(oneFloat[0]));
                //asm volatile("mov %0,%%eax"::"i"(1.f):"eax");
                asm volatile("movss %0,%%xmm0" ::"m"(allone[0]):"xmm0");
                asm volatile("shufps $0,%%xmm0,%%xmm0" :::"xmm0");
//X                 asm volatile("paddw %%xmm0,%0" : "+x"(a));
//X                 asm volatile("paddw %%xmm0,%0" : "+x"(b));
//X                 asm volatile("paddw %%xmm0,%0" : "+x"(c));
//X                 asm volatile("paddw %%xmm0,%0" : "+x"(d));
            }
            timer.Stop();
        }
        timer.Print();
    }
    {
        Benchmark timer("generated one", Factor, "Op");
        for (int rep = 0; rep < Repetitions; ++rep) {
//X             asm volatile("clflush %0" :: "m"(allone[0]));
//X             asm volatile("cpuid" :::"eax", "ebx", "ecx", "edx");
//X             asm volatile("prefetcht1 %0" :: "m"(allone[0]));
//X             asm volatile("cpuid" :::"eax", "ebx", "ecx", "edx");
            timer.Start();
//X             register __m128 a = _mm_setzero_ps();
//X             register __m128 b = _mm_setzero_ps();
//X             register __m128 c = _mm_setzero_ps();
//X             register __m128 d = _mm_setzero_ps();
            for (int i = 0; i < Factor; ++i) {
                //asm volatile("prefetchnta %0" :: "m"(oneFloat[0]));
                asm volatile("xorps %%xmm0,%%xmm0" :::"xmm0");
                asm volatile("cmpeqps %%xmm0,%%xmm0" :::"xmm0");
//X                 asm volatile("paddw %%xmm0,%0" : "+x"(a));
//X                 asm volatile("paddw %%xmm0,%0" : "+x"(b));
//X                 asm volatile("paddw %%xmm0,%0" : "+x"(c));
//X                 asm volatile("paddw %%xmm0,%0" : "+x"(d));
            }
            timer.Stop();
        }
        timer.Print();
    }
    {
        Benchmark timer("loaded one", Factor, "Op");
        for (int rep = 0; rep < Repetitions; ++rep) {
//X             asm volatile("clflush %0" :: "m"(allone[0]));
//X             asm volatile("cpuid" :::"eax", "ebx", "ecx", "edx");
            asm volatile("prefetchnta %0" :: "m"(allone[0]));
//X             asm volatile("cpuid" :::"eax", "ebx", "ecx", "edx");
            timer.Start();
//X             register __m128 a = _mm_setzero_ps();
//X             register __m128 b = _mm_setzero_ps();
//X             register __m128 c = _mm_setzero_ps();
//X             register __m128 d = _mm_setzero_ps();
            for (int i = 0; i < Factor; ++i) {
                register __m128 tmp;
                //asm volatile("clflush %0" :: "m"(allone[0]));
                asm volatile("movaps %1,%0" : "=x"(tmp) : "m"(allone[0]));
//X                 asm volatile("paddw %1,%0" : "+x"(a) : "x"(tmp));
//X                 asm volatile("paddw %1,%0" : "+x"(b) : "x"(tmp));
//X                 asm volatile("paddw %1,%0" : "+x"(c) : "x"(tmp));
//X                 asm volatile("paddw %1,%0" : "+x"(d) : "x"(tmp));
            }
            timer.Stop();
        }
        timer.Print();
    }

    return 0;
}
