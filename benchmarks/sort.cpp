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

#include <cstdlib>

using namespace Vc;

template<typename T> static inline void keepResults(const T &tmp0)
{
#if VC_IMPL_SSE
    asm volatile(""::"x"(reinterpret_cast<const __m128 &>(tmp0)));
    if (sizeof(T) == 32) {
        asm volatile(""::"x"(reinterpret_cast<const __m128 *>(&tmp0)[1]));
    }
#else
    asm volatile(""::"r"(tmp0));
#endif
}

template<typename Vector> struct Helper
{
    typedef typename Vector::Mask Mask;
    typedef typename Vector::EntryType Scalar;

    static void run()
    {
        const int Factor = CpuId::L1Data() / sizeof(Vector);
        const int opPerSecondFactor = Factor * Vector::Size;

        union {
            Vector *v;
            typename Vector::EntryType *m;
        } data = { new Vector[Factor] };
#ifndef VC_BENCHMARK_NO_MLOCK
        mlock(&data, Factor * sizeof(Vector));
#endif
        for (int i = 0; i < Factor; ++i) {
            data.v[i] = PseudoRandom<Vector>::next();
        }

        Vector tmp;

        {
            Benchmark timer("SSE sort", opPerSecondFactor, "Op");
            while (timer.wantsMoreDataPoints()) {
                timer.Start();
                for (int i = 0; i < Factor; ++i) {
                    tmp = data.v[i].sorted();
                    keepResults(tmp);
                }
                timer.Stop();
            }
            timer.Print();
        }
        {
            Benchmark timer("std::sort", opPerSecondFactor, "Op");
            while (timer.wantsMoreDataPoints()) {
                timer.Start();
                for (int i = 0; i < Factor * Vector::Size; i += Vector::Size) {
                    std::sort(&data.m[i], &data.m[i + Vector::Size]);
                    tmp = data.m[i];
                    keepResults(tmp);
                }
                timer.Stop();
                for (int i = 0; i < Factor; ++i) {
                    data.v[i] = PseudoRandom<Vector>::next();
                }
            }
            timer.Print();
        }
        delete[] data.v;
    }
};

int bmain()
{
    Benchmark::addColumn("datatype");
    Benchmark::setColumnData("datatype", "float_v" ); Helper<float_v >::run();
    Benchmark::setColumnData("datatype", "sfloat_v"); Helper<sfloat_v>::run();
    //Benchmark::setColumnData("datatype", "double_v"); Helper<double_v>::run();
    Benchmark::setColumnData("datatype", "int_v"   ); Helper<int_v   >::run();
    Benchmark::setColumnData("datatype", "uint_v"  ); Helper<uint_v  >::run();
    Benchmark::setColumnData("datatype", "short_v" ); Helper<short_v >::run();
    Benchmark::setColumnData("datatype", "ushort_v"); Helper<ushort_v>::run();
    return 0;
}
