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
#include <Vc/uint_v>
#include <Vc/IO>
#include "benchmark.h"

using namespace Vc;

// to benchmark the performance of gathers there are some different interesting cases:
// 1) gathers from the same memory location (basically broadcasts)
// 2) gathers from the same cache line (random access but localized to 64 or 128 bytes) (64 on
//    most CPUs)
// 3) random access on memory that fits into L1 but is larger than a cache line
// 4) random access on memory that fits into L2 (per core) but is larger than L1
// 4b) random access on memory that fits into all of L2 but is larger than the per core L2
// 5) random access on memory that fits into L3 but is larger than L2
// 6) random access on memory that fits into memory but is larger than L3
// 7) random access on memory that fits into virtual memory but is larger than physical memory
//
// of those the last 3 are probably not interesting because they basically measure memory
// latency.

// Intel Core 2 Quad (Q6600) has 8MB L2

enum {
    Repetitions = 2,
    Factor = 160000 / float_v::Size,
    MaxArraySize = 16 * 1024 * 1024 / sizeof(float), //  16 MB
    L2ArraySize = 256 * 1024 / sizeof(float),        // 256 KB
    L1ArraySize = 32 * 1024 / sizeof(float),         //  32 KB
    CacheLineArraySize = 64 / sizeof(float),         //  64 B
    SingleArraySize = 1                              //   4 B
};

// this is not a random number generator
class PseudoRandom
{
    public:
        PseudoRandom() : state(IndexesFromZero) {}
        uint_v next();

    private:
        uint_v state;
};

uint_v PseudoRandom::next()
{
    state = (state * 1103515245 + 12345);
    return state >> 10;
}

void nextIndexes(uint_v &i, const unsigned int size)
{
    static PseudoRandom r;
    i = r.next() & (size - 1);
}

int doGather(const char *name, const unsigned int size, const float *data)
{
    int blackHole = 0;
    Benchmark timer(name, 2. * float_v::Size * Factor * sizeof(float), "B");
    uint_v indexes1(IndexesFromZero);
    uint_v indexes2(indexes1 + 1);
    nextIndexes(indexes1, size);
    nextIndexes(indexes2, size);
    for (int repetitions = 0; repetitions < Repetitions; ++repetitions) {
        float_v aa(Zero);
        float_v bb(Zero);
        timer.Start();
        for (int i = 0; i < Factor; ++i) {
            nextIndexes(indexes1, size);
            nextIndexes(indexes2, size);
            float_v a(data, indexes1);
            float_v b(data, indexes2);
            aa += a;
            bb += b;
        }
        timer.Stop();
        const int k = (aa < 20.f) && (bb < 20.f);
        blackHole += k;
    }
    timer.Print();
    return blackHole;
}

int main()
{
    int blackHole = 1;

    float *data = new float[MaxArraySize];
    for (int i = 0; i < MaxArraySize; ++i) {
        data[i] = static_cast<float>(static_cast<double>(rand()) / static_cast<double>(RAND_MAX));
    }

    blackHole += doGather("Broadcast Gather", SingleArraySize, data);
    blackHole += doGather("Cacheline Gather", CacheLineArraySize, data);
    blackHole += doGather("L1 Gather", L1ArraySize, data);
    blackHole += doGather("L2 Gather", L2ArraySize, data);
    blackHole += doGather("Memory Gather", MaxArraySize, data);

    if (blackHole < 10) {
        std::cout << std::endl;
    }
    return 0;
}
