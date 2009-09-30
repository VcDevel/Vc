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

template<typename Vector> class NameHelper
{
    public:
        static const char *prependTo(const char *t) {
            const int sLen = std::strlen(s);
            const int tLen = std::strlen(t);
            char *r = new char[sLen + tLen + 2];
            std::strcpy(r, s);
            r[sLen] = ' ';
            std::strcpy(r + sLen + 1, t);
            r[sLen + tLen + 1] = '\0';
            return r;
        }
        static const char *string() { return s; }
    private:
        static const char *s;
};

template<> const char *NameHelper<float_v>::s = "float_v";
template<> const char *NameHelper<short_v>::s = "short_v";
#if VC_IMPL_SSE
template<> const char *NameHelper<sfloat_v>::s = "sfloat_v";
#endif

int blackHole = 1;

float_m float_fullMask;
short_m short_fullMask;
#if VC_IMPL_SSE
sfloat_m sfloat_fullMask;
#endif

template<typename Vector> class FullMaskHelper
{
    protected:
        FullMaskHelper();
        typedef typename Vector::Mask IndexMask;
        IndexMask *fullMask;
};

template<typename Vector, class GatherImpl> class GatherBase : public FullMaskHelper<Vector>
{
	public:
		typedef typename Vector::IndexType IndexVector;
		typedef typename Vector::Mask IndexMask;
		typedef typename Vector::EntryType Scalar;

		enum {
			Factor = 1600000 / Vector::Size
		};

        GatherBase(const char *name, const int _Rep, const unsigned int size, const Scalar *_data, double multiplier = 4.)
            : timer(name, multiplier * Vector::Size * Factor, "Values"),
            aa(Zero), bb(Zero), cc(Zero), dd(Zero),
            indexesCount(Factor * 4),
            m_data(new const Scalar *[indexesCount]),
            Repetitions(_Rep),
            im(new IndexAndMask[indexesCount])
        {
            PseudoRandom<IndexVector>::next();
            PseudoRandom<IndexVector>::next();
            const IndexVector indexMask(size - 1);
            const unsigned int maxIndex = ~0u >> ((4 - sizeof(typename IndexVector::EntryType)) * 8);
            const unsigned int maxDataOffset = maxIndex > size ? 1 : size - maxIndex;
            for (int i = 0; i < indexesCount; ++i) {
                m_data[i] = _data + (rand() % maxDataOffset);
                im[i].index = PseudoRandom<IndexVector>::next() & indexMask;
                im[i].mask = (PseudoRandom<IndexVector>::next() & IndexVector(One)) > 0;
            }
            m_data[0] = _data;

            static_cast<GatherImpl *>(this)->run();
            for (int repetitions = 0; repetitions < Repetitions; ++repetitions) {
                timer.Start();
                static_cast<GatherImpl *>(this)->run();
                timer.Stop();
                const int k = (aa < 20.f) && (bb < 20.f) && (cc < 20.f) && (dd < 20.f);
                blackHole += k;
            }
            timer.Print();
        }

        ~GatherBase()
        {
            delete[] m_data;
            delete[] im;
        }

    protected:
        inline const Scalar *data(int i) {
            if (sizeof(typename IndexVector::EntryType) == 2) {
                return m_data[i];
            }
            return m_data[0];
        }
        Benchmark timer;
        Vector aa;
        Vector bb;
        Vector cc;
        Vector dd;
        const int indexesCount;
        const Scalar **const m_data;
        const int Repetitions;
        struct IndexAndMask : public VectorAlignedBase {
            IndexVector index;
            IndexMask mask;
        } *im;
};

template<> FullMaskHelper<float_v>::FullMaskHelper() : fullMask(&float_fullMask) {}
template<> FullMaskHelper<short_v>::FullMaskHelper() : fullMask(&short_fullMask) {}
#if VC_IMPL_SSE
template<> FullMaskHelper<sfloat_v>::FullMaskHelper() : fullMask(&sfloat_fullMask) {}
#endif

static int g_L1ArraySize = 0;
static int g_L2ArraySize = 0;
static int g_L3ArraySize = 0;
static int g_CacheLineArraySize = 0;
static int g_MaxArraySize = 0;

template<typename Vector> struct GatherBenchmark
{
    typedef typename Vector::EntryType Scalar;

    enum {
        Factor = 1600000 / Vector::Size
    };

    class FullMaskedGather : public GatherBase<Vector, FullMaskedGather>
    {
        typedef GatherBase<Vector, FullMaskedGather> Base;
        using Base::data;
        using Base::fullMask;
        using Base::im;
        using Base::aa;
        using Base::bb;
        using Base::cc;
        using Base::dd;

        public:
            FullMaskedGather(const char *name, const int _Rep, const unsigned int _size, const Scalar *_data)
                : Base(name, _Rep, _size, _data, 4.)
            {}

            inline void run()
            {
                for (int i = 0; i < Factor; ++i) {
                    const int ii = i * 4;
                    aa += Vector(data(ii + 0), im[ii + 0].index, *fullMask);
                    bb += Vector(data(ii + 1), im[ii + 1].index, *fullMask);
                    cc += Vector(data(ii + 2), im[ii + 2].index, *fullMask);
                    dd += Vector(data(ii + 3), im[ii + 3].index, *fullMask);
                }
            }
    };

    class MaskedGather : public GatherBase<Vector, MaskedGather>
    {
        typedef GatherBase<Vector, MaskedGather> Base;
        using Base::data;
        using Base::im;
        using Base::aa;
        using Base::bb;
        using Base::cc;
        using Base::dd;

        public:
            MaskedGather(const char *name, const int _Rep, const unsigned int _size, const Scalar *_data)
                : GatherBase<Vector, MaskedGather>(name, _Rep, _size, _data, 2.)
            {}

            inline void run()
            {
                for (int i = 0; i < Factor; ++i) {
                    const int ii = i * 4;
                    aa += Vector(data(ii + 0), im[ii + 0].index, im[ii + 0].mask);
                    bb += Vector(data(ii + 1), im[ii + 1].index, im[ii + 1].mask);
                    cc += Vector(data(ii + 2), im[ii + 2].index, im[ii + 2].mask);
                    dd += Vector(data(ii + 3), im[ii + 3].index, im[ii + 3].mask);
                }
            }
    };

    class Gather : public GatherBase<Vector, Gather>
    {
        typedef GatherBase<Vector, Gather> Base;
        using Base::data;
        using Base::im;
        using Base::aa;
        using Base::bb;
        using Base::cc;
        using Base::dd;

        public:
            Gather(const char *name, const int _Rep, const unsigned int _size, const Scalar *_data)
                : GatherBase<Vector, Gather>(name, _Rep, _size, _data)
            {}

            inline void run()
            {
                for (int i = 0; i < Factor; ++i) {
                    const int ii = i * 4;
                    aa += Vector(data(ii + 0), im[ii + 0].index);
                    bb += Vector(data(ii + 1), im[ii + 1].index);
                    cc += Vector(data(ii + 2), im[ii + 2].index);
                    dd += Vector(data(ii + 3), im[ii + 3].index);
                }
            }
    };

    static void randomize(Scalar *const p, unsigned int size)
    {
        static const double ONE_OVER_RAND_MAX = 1. / RAND_MAX;
        for (unsigned int i = 0; i < size; ++i) {
            p[i] = static_cast<Scalar>(static_cast<double>(std::rand()) * ONE_OVER_RAND_MAX);
        }
    }

    static void run(const int Repetitions)
    {
        const int MaxArraySize = g_MaxArraySize / sizeof(Scalar);
        const int L3ArraySize = g_L3ArraySize / sizeof(Scalar);
        const int L2ArraySize = g_L2ArraySize / sizeof(Scalar);
        const int L1ArraySize = g_L1ArraySize / sizeof(Scalar);
        const int CacheLineArraySize = g_CacheLineArraySize / sizeof(Scalar);
        const int ArrayCount = 2 * MaxArraySize;

        Scalar *const _data = new Scalar[ArrayCount];
        randomize(_data, ArrayCount);

        // the last parts of _data are still hot, so we start at the beginning
        Scalar *data = _data;
        Benchmark::setColumnData("mask", "random mask");
        MaskedGather("Memory", Repetitions, MaxArraySize, data);

        // now the last parts of _data should be cold, let's go there
        data += ArrayCount - MaxArraySize;
        Benchmark::setColumnData("mask", "not masked");
        Gather("Memory", Repetitions, MaxArraySize, data);

        data = _data;
        Benchmark::setColumnData("mask", "masked with one");
        FullMaskedGather("Memory", Repetitions, MaxArraySize, data);

        Benchmark::setColumnData("mask", "not masked");
        if (L3ArraySize > 0) Gather("L3", Repetitions, L3ArraySize, data);
        Gather("L2", Repetitions, L2ArraySize, data);
        Gather("L1", Repetitions, L1ArraySize, data);
        Gather("Cacheline", Repetitions, CacheLineArraySize, data);
        Gather("Broadcast", Repetitions, 1, data);
        Benchmark::setColumnData("mask", "random mask");
        if (L3ArraySize > 0) MaskedGather("L3", Repetitions, L3ArraySize, data);
        MaskedGather("L2", Repetitions, L2ArraySize, data);
        MaskedGather("L1", Repetitions, L1ArraySize, data);
        MaskedGather("Cacheline", Repetitions, CacheLineArraySize, data);
        MaskedGather("Broadcast", Repetitions, 1, data);
        Benchmark::setColumnData("mask", "masked with one");
        if (L3ArraySize > 0) FullMaskedGather("L3", Repetitions, L3ArraySize, data);
        FullMaskedGather("L2", Repetitions, L2ArraySize, data);
        FullMaskedGather("L1", Repetitions, L1ArraySize, data);
        FullMaskedGather("Cacheline", Repetitions, CacheLineArraySize, data);
        FullMaskedGather("Broadcast", Repetitions, 1, data);


        delete[] _data;
    }
};

#include "cpuid.h"

int bmain(Benchmark::OutputMode out)
{
    float_fullMask = float_m(One);
    short_fullMask = short_m(One);
#if VC_IMPL_SSE
    sfloat_fullMask = sfloat_m(One);
#endif

    Benchmark::addColumn("datatype");
    Benchmark::addColumn("mask");
    Benchmark::addColumn("L1.size");
    Benchmark::addColumn("L2.size");
    Benchmark::addColumn("L3.size");
    Benchmark::addColumn("Cacheline.size");
    const int Repetitions = out == Benchmark::Stdout ? 4 : (g_Repetitions > 0 ? g_Repetitions : 20);

    g_L1ArraySize = CpuId::L1Data();
    g_L2ArraySize = CpuId::L2Data();
    g_L3ArraySize = CpuId::L3Data();
    g_CacheLineArraySize = CpuId::L1DataLineSize();
    g_MaxArraySize = std::max(g_L2ArraySize, g_L3ArraySize) * 8;
    {
        std::ostringstream str;
        str << g_L1ArraySize;
        Benchmark::setColumnData("L1.size", str.str());
    }
    {
        std::ostringstream str;
        str << g_L2ArraySize;
        Benchmark::setColumnData("L2.size", str.str());
    }
    {
        std::ostringstream str;
        str << g_L3ArraySize;
        Benchmark::setColumnData("L3.size", str.str());
    }
    {
        std::ostringstream str;
        str << g_CacheLineArraySize;
        Benchmark::setColumnData("Cacheline.size", str.str());
    }

    // divide by 2 since other parts of the program also affect the caches
    g_L1ArraySize /= 2;
    g_L2ArraySize /= 2;
    g_L3ArraySize /= 2;

    Benchmark::setColumnData("datatype", "float_v");
    GatherBenchmark<float_v>::run(Repetitions);
    Benchmark::setColumnData("datatype", "short_v");
    GatherBenchmark<short_v>::run(Repetitions);
#if VC_IMPL_SSE
    Benchmark::setColumnData("datatype", "sfloat_v");
    GatherBenchmark<sfloat_v>::run(Repetitions);
#endif

    return 0;
}
