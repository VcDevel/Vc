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
    return (state >> 16) | (state << 16); // rotate
}

template<class GatherImpl> class GatherBase
{
    public:
        GatherBase(const char *name, const unsigned int size, const float *_data)
            : timer(name, 4. * float_v::Size * Factor * sizeof(float), "B"),
            aa(Zero), bb(Zero), cc(Zero), dd(Zero),
            data(_data), blackHole(0),
            indexesCount(Factor * 4),
            im(new IndexAndMask[indexesCount])
        {
            const uint_v indexMask(size - 1);
            for (int i = 0; i < indexesCount; ++i) {
                im[i].index = PseudoRandom::next() & indexMask;
                im[i].mask = (PseudoRandom::next() & uint_v(One)) > 0;
            }
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
        Benchmark timer;
        float_v aa;
        float_v bb;
        float_v cc;
        float_v dd;
        const float *const data;
        int blackHole;
        const int indexesCount;
        struct IndexAndMask {
            uint_v index;
            uint_m mask;
        } *im;
};

class MaskedGather : public GatherBase<MaskedGather>
{
    public:
        MaskedGather(const char *name, const unsigned int _size, const float *_data)
            : GatherBase<MaskedGather>(name, _size, _data)
        {}

        // the gather function reads
        // 4 * 4  bytes with scalars
        // 4 * 16 bytes with SSE
        // 4 * 64 bytes with LRB
        // (divided by 2 actually for the masks are in average only half full)
        //
        // there should be no overhead other than the nicely prefetchable index/mask vector reading
        void run()
        {
            timer.Start();
            for (int i = 0; i < Factor; ++i) {
                const int ii = i * 4;
                const float_v a(data, im[ii + 0].index, im[ii + 0].mask);
                const float_v b(data, im[ii + 1].index, im[ii + 1].mask);
                const float_v c(data, im[ii + 2].index, im[ii + 2].mask);
                const float_v d(data, im[ii + 3].index, im[ii + 3].mask);
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
            : GatherBase<Gather>(name, _size, _data)
        {}

        void run()
        {
            timer.Start();
            for (int i = 0; i < Factor; ++i) {
                const int ii = i * 4;
                const float_v a(data, im[ii + 0].index);
                const float_v b(data, im[ii + 1].index);
                const float_v c(data, im[ii + 2].index);
                const float_v d(data, im[ii + 3].index);
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
