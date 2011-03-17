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

        const Vector *__restrict__ const end = &data[Factor];
        benchmark_loop(Benchmark("add", valuesPerSecondFactor, "Op")) {
            for (const Vector *__restrict__ ptr = &data[0]; ptr < end; ++ptr) {
                Vector tmp = ptr[0] + ptr[1];
                Vc::forceToRegisters(tmp);
            }
        }
        benchmark_loop(Benchmark("sub", valuesPerSecondFactor, "Op")) {
            for (const Vector *__restrict__ ptr = &data[0]; ptr < end; ++ptr) {
                Vector tmp = ptr[0] - ptr[1];
                Vc::forceToRegisters(tmp);
            }
        }
        benchmark_loop(Benchmark("mul", valuesPerSecondFactor, "Op")) {
            for (const Vector *__restrict__ ptr = &data[0]; ptr < end; ++ptr) {
                Vector tmp = ptr[0] * ptr[1];
                Vc::forceToRegisters(tmp);
            }
        }
        benchmark_loop(Benchmark("div", valuesPerSecondFactor, "Op")) {
            for (const Vector *__restrict__ ptr = &data[0]; ptr < end; ++ptr) {
                Vector tmp = ptr[0] / ptr[1];
                Vc::forceToRegisters(tmp);
            }
        }


        delete[] data;
    }
};

int bmain()
{
    Benchmark::addColumn("datatype");

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
