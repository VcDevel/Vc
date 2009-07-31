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

#include <cstdlib>

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

// Intel Core 2 Quad (Q6600) has 8MB L2

enum {
    Repetitions = 4,
    Factor = 1600000 / float_v::Size,
    MaxArraySize = 32 * 1024 * 1024 / sizeof(float), //  32 MB
    L2ArraySize = 256 * 1024 / sizeof(float),        // 256 KB
    L1ArraySize = 32 * 1024 / sizeof(float),         //  32 KB
    CacheLineArraySize = 64 / sizeof(float),         //  64 B
    SingleArraySize = 1,                             //   4 B
    ArrayCount = 2 * MaxArraySize + 2 * L2ArraySize + 2 * L1ArraySize + 2 * CacheLineArraySize
};

// this is not a random number generator
class PseudoRandom
{
    public:
        PseudoRandom() { next(); next(); next(); next(); }
        static uint_v next();

    private:
        static uint_v state;
};

uint_v PseudoRandom::state(IndexesFromZero);

inline uint_v PseudoRandom::next()
{
    state = (state * 1103515245 + 12345);
    return (state >> 16) | (state << (32 - 16)); // rotate
}

template<class GatherImpl> class GatherBase
{
    public:
        GatherBase(const char *name, const unsigned int _size, const float *_data)
            : timer(name, 4. * float_v::Size * Factor * sizeof(float), "B"),
            indexes1(IndexesFromZero), indexes2(indexes1 + 1),
            indexes3(indexes1 + 2), indexes4(indexes1 + 3),
            aa(Zero), bb(Zero), cc(Zero), dd(Zero),
            size(_size), data(_data), blackHole(0)
        {
            nextIndexes();
        }

        operator int() {
            for (int repetitions = 0; repetitions < Repetitions; ++repetitions) {
                static_cast<GatherImpl *>(this)->run();
                const int k = (aa < 20.f) && (bb < 20.f) && (cc < 20.f) && (dd < 20.f);
                blackHole += k;
            }
            timer.Print();
            return blackHole;
        }

    protected:
        inline void nextIndexes()
        {
            indexes1 = PseudoRandom::next() & (size - 1);
            indexes2 = PseudoRandom::next() & (size - 1);
            indexes3 = PseudoRandom::next() & (size - 1);
            indexes4 = PseudoRandom::next() & (size - 1);
        }

        Benchmark timer;
        uint_v indexes1;
        uint_v indexes2;
        uint_v indexes3;
        uint_v indexes4;
        float_v aa;
        float_v bb;
        float_v cc;
        float_v dd;
        const unsigned int size;
        const float *const data;
        int blackHole;
};

class MaskedGather : public GatherBase<MaskedGather>
{
    public:
        MaskedGather(const char *name, const unsigned int _size, const float *_data)
            : GatherBase<MaskedGather>(name, _size * 2, _data)
        {}

        void run()
        {
            timer.Start();
            for (int i = 0; i < Factor; ++i) {
                nextIndexes();
                // the LSB determines the mask
                const uint_m k1 = (indexes1 & uint_v(One)) > 0;
                const uint_m k2 = (indexes2 & uint_v(One)) > 0;
                const uint_m k3 = (indexes3 & uint_v(One)) > 0;
                const uint_m k4 = (indexes4 & uint_v(One)) > 0;
                indexes1 >>= 1;
                indexes2 >>= 1;
                indexes3 >>= 1;
                indexes4 >>= 1;
                float_v a(data, indexes1, k1);
                float_v b(data, indexes2, k2);
                float_v c(data, indexes3, k3);
                float_v d(data, indexes4, k4);
                aa += a;
                bb += b;
                cc += c;
                dd += d;
            }
            timer.Stop();
        }
};

class Gather : public GatherBase<Gather>
{
    public:
        Gather(const char *name, const unsigned int _size, const float *_data)
            : GatherBase<Gather>(name, _size * 2, _data)
        {}

        void run()
        {
            timer.Start();
            for (int i = 0; i < Factor; ++i) {
                nextIndexes();
                // the following four lines are only here to make the runtime comparable to the
                // MaskedGather version
                indexes1 &= ~uint_v(One);
                indexes2 &= ~uint_v(One);
                indexes3 &= ~uint_v(One);
                indexes4 &= ~uint_v(One);
                indexes1 >>= 1;
                indexes2 >>= 1;
                indexes3 >>= 1;
                indexes4 >>= 1;
                float_v a(data, indexes1);
                float_v b(data, indexes2);
                float_v c(data, indexes3);
                float_v d(data, indexes4);
                aa += a;
                bb += b;
                cc += c;
                dd += d;
            }
            timer.Stop();
        }
};

static void randomize(float *const p, unsigned int size)
{
    static const double ONE_OVER_RAND_MAX = 1. / RAND_MAX;
    for (unsigned int i = 0; i < size; ++i) {
        p[i] = static_cast<float>(static_cast<double>(std::rand()) * ONE_OVER_RAND_MAX);
    }
}

int main()
{
    int blackHole = 1;

    float *const _data = new float[ArrayCount];
    randomize(_data, ArrayCount);

    float *data = _data + ArrayCount;

    data -= MaxArraySize;
    blackHole += MaskedGather("Masked Memory Gather", MaxArraySize, data);
    data -= L2ArraySize;
    blackHole += MaskedGather("Masked L2 Gather", L2ArraySize, data);
    data -= L1ArraySize;
    blackHole += MaskedGather("Masked L1 Gather", L1ArraySize, data);
    data -= CacheLineArraySize;
    blackHole += MaskedGather("Masked Cacheline Gather", CacheLineArraySize, data);
    blackHole += MaskedGather("Masked Broadcast Gather", SingleArraySize, data);

    data -= MaxArraySize;
    blackHole += Gather("Memory Gather", MaxArraySize, data);
    data -= L2ArraySize;
    blackHole += Gather("L2 Gather", L2ArraySize, data);
    data -= L1ArraySize;
    blackHole += Gather("L1 Gather", L1ArraySize, data);
    data -= CacheLineArraySize;
    blackHole += Gather("Cacheline Gather", CacheLineArraySize, data);
    blackHole += Gather("Broadcast Gather", SingleArraySize, data);

    if (blackHole < 10) {
        std::cout << ' ';
    }

    delete[] _data;
    return 0;
}
