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
#include "cpuid.h"

#include <cstdlib>

using namespace Vc;

template<typename Vector> struct Helper
{
    typedef typename Vector::Mask Mask;
    typedef typename Vector::EntryType Scalar;

    static Vector *blackHole;

    static void setBlackHole();

    static void run(const int Repetitions)
    {
        const int Factor = CpuId::L1Data() / sizeof(Vector);
        const int opPerSecondFactor = Factor * Vector::Size;

        setBlackHole();

        Vector *data = new Vector[Factor];
        for (int i = 0; i < Factor; ++i) {
            data[i] = PseudoRandom<Vector>::next();
        }

        {
            Benchmark timer("sqrt", opPerSecondFactor, "Op");
            for (int rep = 0; rep < Repetitions; ++rep) {
                timer.Start();
                for (int i = 0; i < Factor; ++i) {
                    *blackHole = sqrt(data[i]);
                }
                timer.Stop();
            }
            timer.Print();
        }
        {
            Benchmark timer("log", opPerSecondFactor, "Op");
            for (int rep = 0; rep < Repetitions; ++rep) {
                timer.Start();
                for (int i = 0; i < Factor; ++i) {
                    *blackHole = log(data[i]);
                }
                timer.Stop();
            }
            timer.Print();
        }
        {
            Benchmark timer("atan", opPerSecondFactor, "Op");
            for (int rep = 0; rep < Repetitions; ++rep) {
                timer.Start();
                for (int i = 0; i < Factor; ++i) {
                    *blackHole = atan(data[i]);
                }
                timer.Stop();
            }
            timer.Print();
        }
        {
            Benchmark timer("atan2", opPerSecondFactor, "Op");
            for (int rep = 0; rep < Repetitions; ++rep) {
                timer.Start();
                for (int i = 0; i < Factor - 1; ++i) {
                    *blackHole = atan2(data[i], data[i + 1]);
                }
                timer.Stop();
            }
            timer.Print();
        }
        {
            Benchmark timer("rsqrt", opPerSecondFactor, "Op");
            for (int rep = 0; rep < Repetitions; ++rep) {
                timer.Start();
                for (int i = 0; i < Factor; ++i) {
                    *blackHole = rsqrt(data[i]);
                }
                timer.Stop();
            }
            timer.Print();
        }
        {
            Benchmark timer("recip", opPerSecondFactor, "Op");
            for (int rep = 0; rep < Repetitions; ++rep) {
                timer.Start();
                for (int i = 0; i < Factor; ++i) {
                    *blackHole = reciprocal(data[i]);
                }
                timer.Stop();
            }
            timer.Print();
        }

        delete[] data;
    }
};

template<typename Vec> Vec *Helper<Vec>::blackHole = 0;

// global (not file-static!) variable keeps the compiler from identifying the benchmark as dead code
float_v blackHoleFloat;
template<> inline void Helper<float_v>::setBlackHole() { blackHole = &blackHoleFloat; }
#ifdef USE_SSE
sfloat_v blackHoleSFloat;
template<> inline void Helper<sfloat_v>::setBlackHole() { blackHole = &blackHoleSFloat; }
#endif

int bmain(Benchmark::OutputMode out)
{
    const int Repetitions = out == Benchmark::Stdout ? 4 : (g_Repetitions > 0 ? g_Repetitions : 100);
    Benchmark::addColumn("datatype");
    Benchmark::setColumnData("datatype", "float_v");
    Helper<float_v>::run(Repetitions);
#ifdef USE_SSE
    Benchmark::setColumnData("datatype", "sfloat_v");
    Helper<sfloat_v>::run(Repetitions);
#endif
    return 0;
}
