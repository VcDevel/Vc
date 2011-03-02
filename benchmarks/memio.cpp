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
    public:
        static void run()
        {
            Benchmark::setColumnData("MemorySize", "half L1");
            run(CpuId::L1Data() / (sizeof(Vector) * 2), 128);
            Benchmark::setColumnData("MemorySize", "L1");
            run(CpuId::L1Data() / (sizeof(Vector) * 1), 128);
            Benchmark::setColumnData("MemorySize", "half L2");
            run(CpuId::L2Data() / (sizeof(Vector) * 2), 32);
            Benchmark::setColumnData("MemorySize", "L2");
            run(CpuId::L2Data() / (sizeof(Vector) * 1), 16);
            if (CpuId::L3Data() > 0) {
                Benchmark::setColumnData("MemorySize", "half L3");
                run(CpuId::L3Data() / (sizeof(Vector) * 2), 2);
                Benchmark::setColumnData("MemorySize", "L3");
                run(CpuId::L3Data() / (sizeof(Vector) * 1), 2);
                Benchmark::setColumnData("MemorySize", "4x L3");
                run(CpuId::L3Data() / sizeof(Vector) * 4, 1);
            } else {
                Benchmark::setColumnData("MemorySize", "4x L2");
                run(CpuId::L2Data() / sizeof(Vector) * 4, 1);
            }
        }
    private:
        static void run(const int Factor, const int Factor2)
        {
            Vector *a = new Vector[Factor];
#ifndef VC_BENCHMARK_NO_MLOCK
            mlock(a, Factor * sizeof(Vector));
#endif
            // initial loop so that the first iteration in the benchmark loop
            // has the same cache history as subsequent runs
            for (int i = 0; i < Factor; ++i) {
                Vector tmp = a[i];
                keepResults(tmp);
            }

            { // start with reads so that the cache lines are not marked as dirty yet
                Benchmark timer("read", sizeof(Vector) * Factor * Factor2, "Byte");
                benchmark_loop(timer) {
                    for (int j = 0; j < Factor2; ++j) {
                        for (int i = 0; i < Factor; i += 4) {
                            const Vector &tmp0 = a[i + 0];
                            const Vector &tmp1 = a[i + 1];
                            const Vector &tmp2 = a[i + 2];
                            const Vector &tmp3 = a[i + 3];
                            keepResults(tmp0, tmp1, tmp2, tmp3);
                        }
                    }
                }
            }
            {
                Benchmark bm("write", sizeof(Vector) * Factor * Factor2, "Byte");
                const Vector foo = PseudoRandom<Vector>::next();
                keepResults(foo);
                for (int i = 0; i < Factor; i += 4) {
                    a[i + 0] = foo;
                    a[i + 1] = foo;
                    a[i + 2] = foo;
                    a[i + 3] = foo;
                }
                while (bm.wantsMoreDataPoints()) {
                    bm.Start();
                    for (int j = 0; j < Factor2; ++j) {
                        keepResults(foo);
                        for (int i = 0; i < Factor; i += 4) {
                            a[i + 0] = foo;
                            a[i + 1] = foo;
                            a[i + 2] = foo;
                            a[i + 3] = foo;
                        }
                    }
                    bm.Stop();
                }
                bm.Print();
            }
            {
                Benchmark timer("r/w", sizeof(Vector) * Factor * Factor2, "Byte");
                const Vector foo = PseudoRandom<Vector>::next();
                while (timer.wantsMoreDataPoints()) {
                    timer.Start();
                    for (int j = 0; j < Factor2; ++j) {
                        keepResults(foo);
                        for (int i = 0; i < Factor; i += 4) {
                            const Vector &tmp0 = a[i + 0];
                            const Vector &tmp1 = a[i + 1];
                            const Vector &tmp2 = a[i + 2];
                            const Vector &tmp3 = a[i + 3];
                            keepResults(tmp0, tmp1, tmp2, tmp3);
                            a[i + 0] = foo;
                            a[i + 1] = foo;
                            a[i + 2] = foo;
                            a[i + 3] = foo;
                        }
                    }
                    timer.Stop();
                }
                timer.Print();
            }
            delete[] a;
        }
};

int bmain()
{
    Benchmark::addColumn("MemorySize");
    Benchmark::addColumn("datatype");

    Benchmark::setColumnData("datatype", "double_v");
    DoMemIos<double_v>::run();
    Benchmark::setColumnData("datatype", "float_v");
    DoMemIos<float_v>::run();
    Benchmark::setColumnData("datatype", "short_v");
    DoMemIos<short_v>::run();

    return 0;
}
