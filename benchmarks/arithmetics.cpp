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
#include <Vc/limits>
#include "benchmark.h"
#include "random.h"
#include "../cpuid.h"

#include <cstdlib>

using namespace Vc;

template<typename Vector> struct Arithmetics
{
    typedef typename Vector::EntryType Scalar;

    static void run()
    {
        const int Factor = CpuId::L1Data() / sizeof(Vector);

        const double valuesPerSecondFactor = Factor * Vector::Size;

        Vector *data = new Vector[Factor + 1];
#ifndef VC_BENCHMARK_NO_MLOCK
        mlock(data, (Factor + 1) * sizeof(Vector));
#endif
        for (int i = 0; i < Factor + 1; ++i) {
            data[i] = PseudoRandom<Vector>::next();
            data[i](data[i] == Vector(Zero)) += Vector(One);
        }

        Benchmark::setColumnData("unrolling", "not unrolled");
        const Vector *VC_RESTRICT const end = &data[Factor];
        benchmark_loop(Benchmark("add", valuesPerSecondFactor, "Op")) {
            for (const Vector *VC_RESTRICT ptr = &data[0]; ptr < end; ++ptr) {
                Vector tmp = ptr[0] + ptr[1];
                Vc::forceToRegisters(tmp);
            }
        }
        benchmark_loop(Benchmark("sub", valuesPerSecondFactor, "Op")) {
            for (const Vector *VC_RESTRICT ptr = &data[0]; ptr < end; ++ptr) {
                Vector tmp = ptr[0] - ptr[1];
                Vc::forceToRegisters(tmp);
            }
        }
        benchmark_loop(Benchmark("mul", valuesPerSecondFactor, "Op")) {
            for (const Vector *VC_RESTRICT ptr = &data[0]; ptr < end; ++ptr) {
                Vector tmp = ptr[0] * ptr[1];
                Vc::forceToRegisters(tmp);
            }
        }
        benchmark_loop(Benchmark("div", valuesPerSecondFactor, "Op")) {
            for (const Vector *VC_RESTRICT ptr = &data[0]; ptr < end; ++ptr) {
                Vector tmp = ptr[0] / ptr[1];
                Vc::forceToRegisters(tmp);
            }
        }

        Benchmark::setColumnData("unrolling", "2x unrolled");
        benchmark_loop(Benchmark("add", valuesPerSecondFactor, "Op")) {
            for (const Vector *VC_RESTRICT ptr = &data[0]; ptr < end; ptr += 2) {
                Vector tmp0 = ptr[0] + ptr[1];
                Vector tmp1 = ptr[1] + ptr[2];
                keepResults(tmp0, tmp1);
            }
        }
        benchmark_loop(Benchmark("sub", valuesPerSecondFactor, "Op")) {
            for (const Vector *VC_RESTRICT ptr = &data[0]; ptr < end; ptr += 2) {
                Vector tmp0 = ptr[0] - ptr[1];
                Vector tmp1 = ptr[1] - ptr[2];
                keepResults(tmp0, tmp1);
            }
        }
        benchmark_loop(Benchmark("mul", valuesPerSecondFactor, "Op")) {
            for (const Vector *VC_RESTRICT ptr = &data[0]; ptr < end; ptr += 2) {
                Vector tmp0 = ptr[0] * ptr[1];
                Vector tmp1 = ptr[1] * ptr[2];
                keepResults(tmp0, tmp1);
            }
        }
        benchmark_loop(Benchmark("div", valuesPerSecondFactor, "Op")) {
            for (const Vector *VC_RESTRICT ptr = &data[0]; ptr < end; ptr += 2) {
                Vector tmp0 = ptr[0] / ptr[1];
                Vector tmp1 = ptr[1] / ptr[2];
                keepResults(tmp0, tmp1);
            }
        }

        Benchmark::setColumnData("unrolling", "4x unrolled");
        benchmark_loop(Benchmark("add", valuesPerSecondFactor, "Op")) {
            for (const Vector *VC_RESTRICT ptr = &data[0]; ptr < end; ptr += 4) {
                Vector tmp0 = ptr[0] + ptr[1];
                Vector tmp1 = ptr[1] + ptr[2];
                Vector tmp2 = ptr[2] + ptr[3];
                Vector tmp3 = ptr[3] + ptr[4];
                keepResults(tmp0, tmp1, tmp2, tmp3);
            }
        }
        benchmark_loop(Benchmark("sub", valuesPerSecondFactor, "Op")) {
            for (const Vector *VC_RESTRICT ptr = &data[0]; ptr < end; ptr += 4) {
                Vector tmp0 = ptr[0] - ptr[1];
                Vector tmp1 = ptr[1] - ptr[2];
                Vector tmp2 = ptr[2] - ptr[3];
                Vector tmp3 = ptr[3] - ptr[4];
                keepResults(tmp0, tmp1, tmp2, tmp3);
            }
        }
        benchmark_loop(Benchmark("mul", valuesPerSecondFactor, "Op")) {
            for (const Vector *VC_RESTRICT ptr = &data[0]; ptr < end; ptr += 4) {
                Vector tmp0 = ptr[0] * ptr[1];
                Vector tmp1 = ptr[1] * ptr[2];
                Vector tmp2 = ptr[2] * ptr[3];
                Vector tmp3 = ptr[3] * ptr[4];
                keepResults(tmp0, tmp1, tmp2, tmp3);
            }
        }
        benchmark_loop(Benchmark("div", valuesPerSecondFactor, "Op")) {
            for (const Vector *VC_RESTRICT ptr = &data[0]; ptr < end; ptr += 4) {
                Vector tmp0 = ptr[0] / ptr[1];
                Vector tmp1 = ptr[1] / ptr[2];
                Vector tmp2 = ptr[2] / ptr[3];
                Vector tmp3 = ptr[3] / ptr[4];
                keepResults(tmp0, tmp1, tmp2, tmp3);
            }
        }

        Benchmark::setColumnData("unrolling", "8x unrolled");
        benchmark_loop(Benchmark("add", valuesPerSecondFactor, "Op")) {
            for (const Vector *VC_RESTRICT ptr = &data[0]; ptr < end; ptr += 8) {
                Vector tmp0 = ptr[0] + ptr[1];
                Vector tmp1 = ptr[1] + ptr[2];
                Vector tmp2 = ptr[2] + ptr[3];
                Vector tmp3 = ptr[3] + ptr[4];
                Vector tmp4 = ptr[4] + ptr[5];
                Vector tmp5 = ptr[5] + ptr[6];
                Vector tmp6 = ptr[6] + ptr[7];
                Vector tmp7 = ptr[7] + ptr[8];
                keepResults(tmp0, tmp1, tmp2, tmp3);
                keepResults(tmp4, tmp5, tmp6, tmp7);
            }
        }
        benchmark_loop(Benchmark("sub", valuesPerSecondFactor, "Op")) {
            for (const Vector *VC_RESTRICT ptr = &data[0]; ptr < end; ptr += 8) {
                Vector tmp0 = ptr[0] - ptr[1];
                Vector tmp1 = ptr[1] - ptr[2];
                Vector tmp2 = ptr[2] - ptr[3];
                Vector tmp3 = ptr[3] - ptr[4];
                Vector tmp4 = ptr[4] - ptr[5];
                Vector tmp5 = ptr[5] - ptr[6];
                Vector tmp6 = ptr[6] - ptr[7];
                Vector tmp7 = ptr[7] - ptr[8];
                keepResults(tmp0, tmp1, tmp2, tmp3);
                keepResults(tmp4, tmp5, tmp6, tmp7);
            }
        }
        benchmark_loop(Benchmark("mul", valuesPerSecondFactor, "Op")) {
            for (const Vector *VC_RESTRICT ptr = &data[0]; ptr < end; ptr += 8) {
                Vector tmp0 = ptr[0] * ptr[1];
                Vector tmp1 = ptr[1] * ptr[2];
                Vector tmp2 = ptr[2] * ptr[3];
                Vector tmp3 = ptr[3] * ptr[4];
                Vector tmp4 = ptr[4] * ptr[5];
                Vector tmp5 = ptr[5] * ptr[6];
                Vector tmp6 = ptr[6] * ptr[7];
                Vector tmp7 = ptr[7] * ptr[8];
                keepResults(tmp0, tmp1, tmp2, tmp3);
                keepResults(tmp4, tmp5, tmp6, tmp7);
            }
        }
        benchmark_loop(Benchmark("div", valuesPerSecondFactor, "Op")) {
            for (const Vector *VC_RESTRICT ptr = &data[0]; ptr < end; ptr += 8) {
                Vector tmp0 = ptr[0] / ptr[1];
                Vector tmp1 = ptr[1] / ptr[2];
                Vector tmp2 = ptr[2] / ptr[3];
                Vector tmp3 = ptr[3] / ptr[4];
                Vector tmp4 = ptr[4] / ptr[5];
                Vector tmp5 = ptr[5] / ptr[6];
                Vector tmp6 = ptr[6] / ptr[7];
                Vector tmp7 = ptr[7] / ptr[8];
                keepResults(tmp0, tmp1, tmp2, tmp3);
                keepResults(tmp4, tmp5, tmp6, tmp7);
            }
        }

        delete[] data;
    }
};

int bmain()
{
    Benchmark::addColumn("datatype");
    Benchmark::addColumn("unrolling");

    Benchmark::setColumnData("datatype", "float_v");
    Arithmetics<float_v>::run();

    Benchmark::setColumnData("datatype", "double_v");
    Arithmetics<double_v>::run();

    Benchmark::setColumnData("datatype", "int_v");
    Arithmetics<int_v>::run();

    Benchmark::setColumnData("datatype", "uint_v");
    Arithmetics<uint_v>::run();

    Benchmark::setColumnData("datatype", "short_v");
    Arithmetics<short_v>::run();

    Benchmark::setColumnData("datatype", "ushort_v");
    Arithmetics<ushort_v>::run();

    Benchmark::setColumnData("datatype", "sfloat_v");
    Arithmetics<sfloat_v>::run();

    return 0;
}
