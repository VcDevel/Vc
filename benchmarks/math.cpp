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

template<typename Vector> struct Helper
{
    typedef typename Vector::Mask Mask;
    typedef typename Vector::EntryType Scalar;

    static void run()
    {
        const int Factor = CpuId::L1Data() / sizeof(Vector);
        const int opPerSecondFactor = Factor * Vector::Size;

        Vector *data = new Vector[Factor];
#ifndef VC_BENCHMARK_NO_MLOCK
        mlock(data, Factor * sizeof(Vector));
#endif
        for (int i = 0; i < Factor; ++i) {
            data[i] = PseudoRandom<Vector>::next();
        }

        {
            Benchmark timer("round", opPerSecondFactor, "Op");
            while (timer.wantsMoreDataPoints()) {
                timer.Start();
                for (int i = 0; i < Factor; ++i) {
                    Vector tmp = round(data[i]);
                    Vc::forceToRegisters(tmp);
                }
                timer.Stop();
            }
            timer.Print();
        }
        {
            Benchmark timer("sqrt", opPerSecondFactor, "Op");
            while (timer.wantsMoreDataPoints()) {
                timer.Start();
                for (int i = 0; i < Factor; ++i) {
                    Vector tmp = sqrt(data[i]);
                    Vc::forceToRegisters(tmp);
                }
                timer.Stop();
            }
            timer.Print();
        }
        {
            Benchmark timer("log", opPerSecondFactor, "Op");
            while (timer.wantsMoreDataPoints()) {
                timer.Start();
                for (int i = 0; i < Factor; ++i) {
                    Vector tmp = log(data[i]);
                    Vc::forceToRegisters(tmp);
                }
                timer.Stop();
            }
            timer.Print();
        }
        {
            Benchmark timer("sin", opPerSecondFactor, "Op");
            while (timer.wantsMoreDataPoints()) {
                timer.Start();
                for (int i = 0; i < Factor; ++i) {
                    Vector tmp = sin(data[i]);
                    Vc::forceToRegisters(tmp);
                }
                timer.Stop();
            }
            timer.Print();
        }
        {
            Benchmark timer("cos", opPerSecondFactor, "Op");
            while (timer.wantsMoreDataPoints()) {
                timer.Start();
                for (int i = 0; i < Factor; ++i) {
                    Vector tmp = cos(data[i]);
                    Vc::forceToRegisters(tmp);
                }
                timer.Stop();
            }
            timer.Print();
        }
        {
            Benchmark timer("asin", opPerSecondFactor, "Op");
            while (timer.wantsMoreDataPoints()) {
                timer.Start();
                for (int i = 0; i < Factor; ++i) {
                    Vector tmp = asin(data[i]);
                    Vc::forceToRegisters(tmp);
                }
                timer.Stop();
            }
            timer.Print();
        }
        {
            Benchmark timer("atan", opPerSecondFactor, "Op");
            while (timer.wantsMoreDataPoints()) {
                timer.Start();
                for (int i = 0; i < Factor; ++i) {
                    Vector tmp = atan(data[i]);
                    Vc::forceToRegisters(tmp);
                }
                timer.Stop();
            }
            timer.Print();
        }
        {
            Benchmark timer("atan2", opPerSecondFactor, "Op");
            while (timer.wantsMoreDataPoints()) {
                timer.Start();
                for (int i = 0; i < Factor - 1; ++i) {
                    Vector tmp = atan2(data[i], data[i + 1]);
                    Vc::forceToRegisters(tmp);
                }
                timer.Stop();
            }
            timer.Print();
        }
        {
            Benchmark timer("rsqrt", opPerSecondFactor, "Op");
            while (timer.wantsMoreDataPoints()) {
                timer.Start();
                for (int i = 0; i < Factor; ++i) {
                    Vector tmp = rsqrt(data[i]);
                    Vc::forceToRegisters(tmp);
                }
                timer.Stop();
            }
            timer.Print();
        }
        {
            Benchmark timer("recip", opPerSecondFactor, "Op");
            while (timer.wantsMoreDataPoints()) {
                timer.Start();
                for (int i = 0; i < Factor; ++i) {
                    Vector tmp = reciprocal(data[i]);
                    Vc::forceToRegisters(tmp);
                }
                timer.Stop();
            }
            timer.Print();
        }

        delete[] data;
    }
};

int bmain()
{
    Benchmark::addColumn("datatype");
    Benchmark::setColumnData("datatype", "float_v");
    Helper<float_v>::run();
    Benchmark::setColumnData("datatype", "sfloat_v");
    Helper<sfloat_v>::run();
    Benchmark::setColumnData("datatype", "double_v");
    Helper<double_v>::run();
    return 0;
}
