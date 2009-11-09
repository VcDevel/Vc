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
#include <cstdio>
#include <cstdlib>

using namespace Vc;

bool blackHoleBool = false;

template<typename Vector> class DoCompares
{
    public:
        static void run(const int Repetitions)
        {
            const int Factor = CpuId::L1Data() / sizeof(Vector);
            Vector *a = new Vector[Factor + 1];
            for (int i = 0; i < Factor + 1; ++i) {
                a[i] = PseudoRandom<Vector>::next();
            }

            {
                Benchmark timer("operator==", Vector::Size * Factor, "Op");
                for (int repetitions = 0; repetitions < Repetitions; ++repetitions) {
                    timer.Start();
                    for (int i = 0; i < Factor; ++i) {
                        blackHoleMask = a[i] == a[i + 1];
                    }
                    timer.Stop();
                }
                timer.Print(Benchmark::PrintAverage);
            }
            {
                Benchmark timer("operator<", Vector::Size * Factor, "Op");
                for (int repetitions = 0; repetitions < Repetitions; ++repetitions) {
                    timer.Start();
                    for (int i = 0; i < Factor; ++i) {
                        blackHoleMask = a[i] < a[i + 1];
                    }
                    timer.Stop();
                }
                timer.Print(Benchmark::PrintAverage);
            }
            {
                Benchmark timer("(operator<).isFull()", Vector::Size * Factor, "Op");
                for (int repetitions = 0; repetitions < Repetitions; ++repetitions) {
                    timer.Start();
                    const Vector one(One);
                    for (int i = 0; i < Factor; ++i) {
                        blackHoleBool = (a[i] < a[i + 1]).isFull();
                    }
                    timer.Stop();
                }
                timer.Print(Benchmark::PrintAverage);
            }
            {
                Benchmark timer("!(operator<).isEmpty()", Vector::Size * Factor, "Op");
                for (int repetitions = 0; repetitions < Repetitions; ++repetitions) {
                    timer.Start();
                    const Vector one(One);
                    for (int i = 0; i < Factor; ++i) {
                        blackHoleBool = !(a[i] < a[i + 1]).isEmpty();
                    }
                    timer.Stop();
                }
                timer.Print(Benchmark::PrintAverage);
            }
            delete[] a;
        }

    private:
        static typename Vector::Mask blackHoleMask;
        static Vector blackHoleVector;
};

template<> double_m DoCompares<double_v>::blackHoleMask = double_m();
template<> double_v DoCompares<double_v>::blackHoleVector = double_v();
template<> float_m DoCompares<float_v>::blackHoleMask = float_m();
template<> float_v DoCompares<float_v>::blackHoleVector = float_v();
template<> short_m DoCompares<short_v>::blackHoleMask = short_m();
template<> short_v DoCompares<short_v>::blackHoleVector = short_v();
template<> ushort_m DoCompares<ushort_v>::blackHoleMask = ushort_m();
template<> ushort_v DoCompares<ushort_v>::blackHoleVector = ushort_v();
#if !VC_IMPL_LRBni
template<> int_m DoCompares<int_v>::blackHoleMask = int_m();
template<> int_v DoCompares<int_v>::blackHoleVector = int_v();
template<> uint_m DoCompares<uint_v>::blackHoleMask = uint_m();
template<> uint_v DoCompares<uint_v>::blackHoleVector = uint_v();
#endif
#if VC_IMPL_SSE
template<> sfloat_m DoCompares<sfloat_v>::blackHoleMask = sfloat_m();
template<> sfloat_v DoCompares<sfloat_v>::blackHoleVector = sfloat_v();
#endif

int bmain(Benchmark::OutputMode out)
{
    const int Repetitions = out == Benchmark::Stdout ? 10 : g_Repetitions > 0 ? g_Repetitions : 100;

    Benchmark::addColumn("datatype");

    Benchmark::setColumnData("datatype", "double_v");
    DoCompares<double_v>::run(Repetitions);
    Benchmark::setColumnData("datatype", "float_v");
    DoCompares<float_v>::run(Repetitions);
    Benchmark::setColumnData("datatype", "int_v");
    DoCompares<int_v>::run(Repetitions);
    Benchmark::setColumnData("datatype", "uint_v");
    DoCompares<uint_v>::run(Repetitions);
    Benchmark::setColumnData("datatype", "short_v");
    DoCompares<short_v>::run(Repetitions);
    Benchmark::setColumnData("datatype", "ushort_v");
    DoCompares<ushort_v>::run(Repetitions);
    Benchmark::setColumnData("datatype", "sfloat_v");
    DoCompares<sfloat_v>::run(Repetitions);

    return 0;
}
