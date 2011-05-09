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
#include "../cpuid.h"
#include <cstdio>
#include <cstdlib>

using namespace Vc;

template<typename Vector> class DoMemIos
{
    typedef typename Vector::EntryType T;
    public:
        static void run()
        {
            Benchmark::setColumnData("MemorySize", "half L1");
            run(CpuId::L1Data() / (sizeof(T) * 2), 128);
            Benchmark::setColumnData("MemorySize", "L1");
            run(CpuId::L1Data() / (sizeof(T) * 1), 128);
            Benchmark::setColumnData("MemorySize", "half L2");
            run(CpuId::L2Data() / (sizeof(T) * 2), 32);
            Benchmark::setColumnData("MemorySize", "L2");
            run(CpuId::L2Data() / (sizeof(T) * 1), 16);
            if (CpuId::L3Data() > 0) {
                Benchmark::setColumnData("MemorySize", "half L3");
                run(CpuId::L3Data() / (sizeof(T) * 2), 2);
                Benchmark::setColumnData("MemorySize", "L3");
                run(CpuId::L3Data() / (sizeof(T) * 1), 2);
                Benchmark::setColumnData("MemorySize", "4x L3");
                run(CpuId::L3Data() / sizeof(T) * 4, 1);
            } else {
                Benchmark::setColumnData("MemorySize", "4x L2");
                run(CpuId::L2Data() / sizeof(T) * 4, 1);
            }
        }
    private:
        /**
         * \param Factor The number of scalar elements in the memory to read/write
         * \param Factor2 How often the memory region should be read/written
         */
        static void run(const int Factor, const int Factor2)
        {
            T *data = Vc::malloc<T, Vc::AlignOnPage>(Factor + 1);
#ifndef VC_BENCHMARK_NO_MLOCK
            mlock(data, Factor * sizeof(T));
#endif
            Benchmark::setColumnData("Alignment", "aligned");
            run(data, Vc::Aligned, Factor, Factor2);
            Benchmark::setColumnData("Alignment", "aligned mem/unaligned instr");
            run(data, Vc::Unaligned, Factor, Factor2);
            Benchmark::setColumnData("Alignment", "unaligned");
            run(data + 1, Vc::Unaligned, Factor, Factor2);

            Vc::free(data);
        }

        template<typename Align>
        static void run(T *__restrict__ a, Align alignment, const int Factor, const int Factor2)
        {
            // initial loop so that the first iteration in the benchmark loop
            // has the same cache history as subsequent runs
            for (int i = 0; i < Factor; i += Vector::Size) {
                const Vector tmp(&a[i], alignment);
                keepResults(tmp);
            }

            const int numberOfBytes = sizeof(T) * Factor * Factor2;
            const Vector foo = PseudoRandom<Vector>::next();

            // start with reads so that the cache lines are not marked as dirty yet
            benchmark_loop(Benchmark("read", numberOfBytes, "Byte")) {
                for (int j = 0; j < Factor2; ++j) {
                    for (int i = 0; i < Factor; i += 8 * Vector::Size) {
                        const Vector tmp0(&a[i + 0 * Vector::Size], alignment);
                        const Vector tmp1(&a[i + 1 * Vector::Size], alignment);
                        const Vector tmp2(&a[i + 2 * Vector::Size], alignment);
                        const Vector tmp3(&a[i + 3 * Vector::Size], alignment);
                        const Vector tmp4(&a[i + 4 * Vector::Size], alignment);
                        const Vector tmp5(&a[i + 5 * Vector::Size], alignment);
                        const Vector tmp6(&a[i + 6 * Vector::Size], alignment);
                        const Vector tmp7(&a[i + 7 * Vector::Size], alignment);
                        keepResults(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7);
                    }
                }
            }
            benchmark_loop(Benchmark("write", numberOfBytes, "Byte")) {
                for (int j = 0; j < Factor2; ++j) {
                    for (int i = 0; i < Factor; i += 8 * Vector::Size) {
                        foo.store(&a[i + 0 * Vector::Size], alignment);
                        foo.store(&a[i + 1 * Vector::Size], alignment);
                        foo.store(&a[i + 2 * Vector::Size], alignment);
                        foo.store(&a[i + 3 * Vector::Size], alignment);
                        foo.store(&a[i + 4 * Vector::Size], alignment);
                        foo.store(&a[i + 5 * Vector::Size], alignment);
                        foo.store(&a[i + 6 * Vector::Size], alignment);
                        foo.store(&a[i + 7 * Vector::Size], alignment);
                    }
                }
            }
            benchmark_loop(Benchmark("r/w", numberOfBytes, "Byte")) {
                for (int j = 0; j < Factor2; ++j) {
                    for (int i = 0; i < Factor; i += 8 * Vector::Size) {
                        const Vector tmp0(&a[i + 0 * Vector::Size], alignment);
                        const Vector tmp1(&a[i + 1 * Vector::Size], alignment);
                        const Vector tmp2(&a[i + 2 * Vector::Size], alignment);
                        const Vector tmp3(&a[i + 3 * Vector::Size], alignment);
                        const Vector tmp4(&a[i + 4 * Vector::Size], alignment);
                        const Vector tmp5(&a[i + 5 * Vector::Size], alignment);
                        const Vector tmp6(&a[i + 6 * Vector::Size], alignment);
                        const Vector tmp7(&a[i + 7 * Vector::Size], alignment);
                        tmp1.store(&a[i + 0 * Vector::Size], alignment);
                        tmp0.store(&a[i + 1 * Vector::Size], alignment);
                        tmp0.store(&a[i + 2 * Vector::Size], alignment);
                        tmp0.store(&a[i + 3 * Vector::Size], alignment);
                        tmp0.store(&a[i + 4 * Vector::Size], alignment);
                        tmp0.store(&a[i + 5 * Vector::Size], alignment);
                        tmp0.store(&a[i + 6 * Vector::Size], alignment);
                        tmp0.store(&a[i + 7 * Vector::Size], alignment);
                        keepResults(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7);
                    }
                }
            }
        }
};

int bmain()
{
    Benchmark::addColumn("MemorySize");
    Benchmark::addColumn("datatype");
    Benchmark::addColumn("Alignment");

    Benchmark::setColumnData("datatype", "double_v");
    DoMemIos<double_v>::run();
    Benchmark::setColumnData("datatype", "float_v");
    DoMemIos<float_v>::run();
    Benchmark::setColumnData("datatype", "short_v");
    DoMemIos<short_v>::run();

    return 0;
}
