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
#include "benchmark.h"
#include "random.h"

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
    Repetitions = 4
};

template<typename Vector, class GatherImpl> class GatherBase
{
    typedef typename Vector::IndexType IndexVector;
    typedef typename Vector::IndexType::Mask IndexMask;
    typedef typename Vector::EntryType Scalar;

    enum {
        Factor = 1600000 / Vector::Size
    };

    public:
        GatherBase(const char *name, const unsigned int size, const Scalar *_data)
            : timer(name, 4. * Vector::Size * Factor * sizeof(Scalar), "B"),
            aa(Zero), bb(Zero), cc(Zero), dd(Zero),
            data(_data), blackHole(0),
            indexesCount(Factor * 4),
            im(new IndexAndMask[indexesCount])
        {
            PseudoRandom<IndexVector>::next();
            PseudoRandom<IndexVector>::next();
            const IndexVector indexMask(size - 1);
            for (int i = 0; i < indexesCount; ++i) {
                im[i].index = PseudoRandom<IndexVector>::next() & indexMask;
                im[i].mask = (PseudoRandom<IndexVector>::next() & IndexVector(One)) > 0;
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
        Vector aa;
        Vector bb;
        Vector cc;
        Vector dd;
        const Scalar *const data;
        int blackHole;
        const int indexesCount;
        struct IndexAndMask {
            IndexVector index;
            IndexMask mask;
        } *im;
};

template<typename Vector> struct GatherBenchmark
{
    typedef typename Vector::IndexType IndexVector;
    typedef typename Vector::IndexType::Mask IndexMask;
    typedef typename Vector::EntryType Scalar;

    enum {
        Factor = 1600000 / Vector::Size,
        MaxArraySize = 32 * 1024 * 1024 / sizeof(Scalar), //  32 MB
        L2ArraySize = 256 * 1024 / sizeof(Scalar),        // 256 KB
        L1ArraySize = 32 * 1024 / sizeof(Scalar),         //  32 KB
        CacheLineArraySize = 64 / sizeof(Scalar),         //  64 B
        SingleArraySize = 1,                             //   4 B
        ArrayCount = 2 * MaxArraySize + 2 * L2ArraySize + 2 * L1ArraySize + 2 * CacheLineArraySize
    };

    class MaskedGather : public GatherBase<Vector, MaskedGather>
    {
        typedef GatherBase<Vector, MaskedGather> Base;
        using Base::timer;
        using Base::data;
        using Base::im;
        using Base::aa;
        using Base::bb;
        using Base::cc;
        using Base::dd;

        public:
            MaskedGather(const char *name, const unsigned int _size, const Scalar *_data)
                : GatherBase<Vector, MaskedGather>(name, _size, _data)
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
                    const Vector a(data, im[ii + 0].index, im[ii + 0].mask);
                    const Vector b(data, im[ii + 1].index, im[ii + 1].mask);
                    const Vector c(data, im[ii + 2].index, im[ii + 2].mask);
                    const Vector d(data, im[ii + 3].index, im[ii + 3].mask);
                    aa += a;
                    bb += b;
                    cc += c;
                    dd += d;
                }
                timer.Stop();
            }
    };

    class Gather : public GatherBase<Vector, Gather>
    {
        typedef GatherBase<Vector, Gather> Base;
        using Base::timer;
        using Base::data;
        using Base::im;
        using Base::aa;
        using Base::bb;
        using Base::cc;
        using Base::dd;

        public:
            Gather(const char *name, const unsigned int _size, const Scalar *_data)
                : GatherBase<Vector, Gather>(name, _size, _data)
            {}

            void run()
            {
                timer.Start();
                for (int i = 0; i < Factor; ++i) {
                    const int ii = i * 4;
                    const Vector a(data, im[ii + 0].index);
                    const Vector b(data, im[ii + 1].index);
                    const Vector c(data, im[ii + 2].index);
                    const Vector d(data, im[ii + 3].index);
                    aa += a;
                    bb += b;
                    cc += c;
                    dd += d;
                }
                timer.Stop();
            }
    };

    static void randomize(Scalar *const p, unsigned int size)
    {
        static const double ONE_OVER_RAND_MAX = 1. / RAND_MAX;
        for (unsigned int i = 0; i < size; ++i) {
            p[i] = static_cast<Scalar>(static_cast<double>(std::rand()) * ONE_OVER_RAND_MAX);
        }
    }

    static int run()
    {
        int blackHole = 1;

        Scalar *const _data = new Scalar[ArrayCount];
        randomize(_data, ArrayCount);
        // the last parts of _data are still hot, so we start at the beginning

        Scalar *data = _data;
        blackHole += MaskedGather("Masked Memory Gather", MaxArraySize, data);

        // now the last parts of _data should be cold, let's go there
        data += ArrayCount - L2ArraySize;
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

        delete[] _data;
        return blackHole;
    }
};

int main()
{
    int blackHole = 1;

    std::cout << "Benchmarking float_v" << std::endl;
    blackHole += GatherBenchmark<float_v>::run();
    std::cout << "\nBenchmarking short_v" << std::endl;
    blackHole += GatherBenchmark<short_v>::run();
    std::cout << "\nBenchmarking sfloat_v" << std::endl;
    blackHole += GatherBenchmark<sfloat_v>::run();

    if (blackHole < 10) {
        std::cout << ' ';
    }

    return 0;
}
