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
#include "cpuid.h"

#include <cstdlib>

using namespace Vc;

template<typename Vector> struct Arithmetics
{
    typedef typename Vector::EntryType Scalar;

    static Vector &_blackHole();

    static void run(const int Repetitions)
    {
        const int Factor = CpuId::L1Data() / sizeof(Vector);

        const double valuesPerSecondFactor = Factor * Vector::Size;

        Vector *data = new Vector[Factor + 1];
        for (int i = 0; i < Factor + 1; ++i) {
            data[i] = PseudoRandom<Vector>::next();
            data[i](data[i] == Vector(Zero)) += Vector(One);
        }

        {
            Benchmark timer("add", valuesPerSecondFactor, "Op");
            for (int rep = 0; rep < Repetitions; ++rep) {
                timer.Start();
                const Vector *const end = &data[Factor];
                for (const Vector *ptr = &data[0]; ptr < end; ++ptr) {
                    Vector tmp = ptr[0] + ptr[1];
                    Vc::forceToRegisters(tmp);
                }
                timer.Stop();
            }
            timer.Print();
        }
        {
            Benchmark timer("sub", valuesPerSecondFactor, "Op");
            for (int rep = 0; rep < Repetitions; ++rep) {
                timer.Start();
                const Vector *const end = &data[Factor];
                for (const Vector *ptr = &data[0]; ptr < end; ++ptr) {
                    Vector tmp = ptr[0] - ptr[1];
                    Vc::forceToRegisters(tmp);
                }
                timer.Stop();
            }
            timer.Print();
        }
        {
            Benchmark timer("mul", valuesPerSecondFactor, "Op");
            for (int rep = 0; rep < Repetitions; ++rep) {
                timer.Start();
                const Vector *const end = &data[Factor];
                for (const Vector *ptr = &data[0]; ptr < end; ++ptr) {
                    Vector tmp = ptr[0] * ptr[1];
                    Vc::forceToRegisters(tmp);
                }
                timer.Stop();
            }
            timer.Print();
        }
        {
            Benchmark timer("div", valuesPerSecondFactor, "Op");
            for (int rep = 0; rep < Repetitions; ++rep) {
                timer.Start();
                const Vector *const end = &data[Factor];
                for (const Vector *ptr = &data[0]; ptr < end; ++ptr) {
                    Vector tmp = ptr[0] / ptr[1];
                    Vc::forceToRegisters(tmp);
                }
                timer.Stop();
            }
            timer.Print();
        }

        /*
        {
            Benchmark timer("add latency", Factor, "Op");
            for (int rep = 0; rep < Repetitions; ++rep) {
                timer.Start();
                {
                    Vector tmp = data[0];
                    for (const Vector *ptr = &data[1]; ptr < &data[Factor + 1]; ptr += 4) {
                        tmp += ptr[0];
                        tmp += ptr[1];
                        tmp += ptr[2];
                        tmp += ptr[3];
                    }
                    blackHole = tmp;
                }
                timer.Stop();
            }
            timer.Print();
        }
        {
            Benchmark timer("sub latency", Factor, "Op");
            for (int rep = 0; rep < Repetitions; ++rep) {
                timer.Start();
                {
                    Vector tmp = data[0];
                    for (int i = 1; i < Factor + 1; ++i) {
                        tmp -= data[i];
                    }
                    blackHole = tmp;
                }
                timer.Stop();
            }
            timer.Print();
        }
        {
            Benchmark timer("mul latency", Factor, "Op");
            for (int rep = 0; rep < Repetitions; ++rep) {
                timer.Start();
                {
                    Vector tmp = data[0];
                    for (int i = 1; i < Factor + 1; ++i) {
                        tmp *= data[i];
                    }
                    blackHole = tmp;
                }
                timer.Stop();
            }
            timer.Print();
        }
        data[0] = std::numeric_limits<Vector>::max();
        for (int i = 1; i < Factor + 1; ++i) {
            data[i] = PseudoRandom<Vector>::next() + Vector(One);
            data[i](data[i] == Vector(Zero)) += 2;
        }
        {
            Benchmark timer("div latency", Factor, "Op");
            for (int rep = 0; rep < Repetitions; ++rep) {
                timer.Start();
                {
                    Vector tmp = data[0];
                    for (int i = 1; i < Factor + 1; ++i) {
                        tmp /= data[i];
                    }
                    blackHole = tmp;
                }
                timer.Stop();
            }
            timer.Print();
        }
        */

        delete[] data;
    }

    static Vector blackHole;
};

template<> float_v Arithmetics<float_v>::blackHole = float_v(Zero);
template<> double_v Arithmetics<double_v>::blackHole = double_v(Zero);
template<> int_v Arithmetics<int_v>::blackHole = int_v(Zero);
template<> uint_v Arithmetics<uint_v>::blackHole = uint_v(Zero);
#ifndef VC_IMPL_LRBni
template<> short_v Arithmetics<short_v>::blackHole = short_v(Zero);
template<> ushort_v Arithmetics<ushort_v>::blackHole = ushort_v(Zero);
#endif
#ifdef VC_IMPL_SSE
template<> sfloat_v Arithmetics<sfloat_v>::blackHole = sfloat_v(Zero);
#endif

int bmain(Benchmark::OutputMode out)
{
    const int Repetitions = out == Benchmark::Stdout ? 3 : g_Repetitions > 0 ? g_Repetitions : 10;
    Benchmark::addColumn("datatype");

    Benchmark::setColumnData("datatype", "float_v");
    Arithmetics<float_v>::run(Repetitions);

    Benchmark::setColumnData("datatype", "double_v");
    Arithmetics<double_v>::run(Repetitions);

    Benchmark::setColumnData("datatype", "int_v");
    Arithmetics<int_v>::run(Repetitions);

    Benchmark::setColumnData("datatype", "uint_v");
    Arithmetics<uint_v>::run(Repetitions);

    Benchmark::setColumnData("datatype", "short_v");
    Arithmetics<short_v>::run(Repetitions);

    Benchmark::setColumnData("datatype", "ushort_v");
    Arithmetics<ushort_v>::run(Repetitions);

    Benchmark::setColumnData("datatype", "sfloat_v");
    Arithmetics<sfloat_v>::run(Repetitions);

    return 0;
}
