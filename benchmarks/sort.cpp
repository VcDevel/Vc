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

        union {
            Vector *v;
            typename Vector::EntryType *m;
        } data = { new Vector[Factor] };
        for (int i = 0; i < Factor; ++i) {
            data.v[i] = PseudoRandom<Vector>::next();
        }

        {
            Benchmark timer("SSE sort", opPerSecondFactor, "Op");
            for (int rep = 0; rep < Repetitions; ++rep) {
                timer.Start();
                for (int i = 0; i < Factor; ++i) {
                    *blackHole = data.v[i].sorted();
                }
                timer.Stop();
            }
            timer.Print();
        }
        {
            Benchmark timer("std::sort", opPerSecondFactor, "Op");
            for (int rep = 0; rep < Repetitions; ++rep) {
                timer.Start();
                for (int i = 0; i < Factor * Vector::Size; i += Vector::Size) {
                    std::sort(&data.m[i], &data.m[i + Vector::Size]);
                    *blackHole = data.m[i];
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

template<typename Vec> Vec *Helper<Vec>::blackHole = 0;

float_v   blackHoleFloat;  template<> inline void Helper<float_v >::setBlackHole() { blackHole = &blackHoleFloat; }
double_v  blackHoleDouble; template<> inline void Helper<double_v>::setBlackHole() { blackHole = &blackHoleDouble; }
ushort_v  blackHoleUShort; template<> inline void Helper<ushort_v>::setBlackHole() { blackHole = &blackHoleUShort; }
short_v   blackHoleShort;  template<> inline void Helper<short_v >::setBlackHole() { blackHole = &blackHoleShort; }
uint_v    blackHoleUInt;   template<> inline void Helper<uint_v  >::setBlackHole() { blackHole = &blackHoleUInt; }
int_v     blackHoleInt;    template<> inline void Helper<int_v   >::setBlackHole() { blackHole = &blackHoleInt; }
sfloat_v  blackHoleSFloat; template<> inline void Helper<sfloat_v>::setBlackHole() { blackHole = &blackHoleSFloat; }

int bmain(Benchmark::OutputMode out)
{
    const int Repetitions = out == Benchmark::Stdout ? 4 : (g_Repetitions > 0 ? g_Repetitions : 100);
    Benchmark::addColumn("datatype");
    Benchmark::setColumnData("datatype", "float_v" ); Helper<float_v >::run(Repetitions);
    Benchmark::setColumnData("datatype", "sfloat_v"); Helper<sfloat_v>::run(Repetitions);
    //Benchmark::setColumnData("datatype", "double_v"); Helper<double_v>::run(Repetitions);
    Benchmark::setColumnData("datatype", "int_v"   ); Helper<int_v   >::run(Repetitions);
    Benchmark::setColumnData("datatype", "uint_v"  ); Helper<uint_v  >::run(Repetitions);
    Benchmark::setColumnData("datatype", "short_v" ); Helper<short_v >::run(Repetitions);
    Benchmark::setColumnData("datatype", "ushort_v"); Helper<ushort_v>::run(Repetitions);
    return 0;
}
