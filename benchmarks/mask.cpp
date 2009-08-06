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

#include <cstdlib>

using namespace Vc;

// to test
// a) conditional (masked) assignment
// b) masked ops (like flops.cpp but with masks)

// global (not file-static!) variable keeps the compiler from identifying the benchmark as dead code
int blackHole = 1;

template<typename Vector> struct CondAssignment
{
    typedef typename Vector::Mask Mask;
    typedef typename Vector::EntryType Scalar;

    enum {
        Factor = 10240000 / Vector::Size
    };

    static void run(const int Repetitions)
    {
        const double valuesPerSecondFactor = Factor * Vector::Size * 4.;
        {
            Benchmark timer("Conditional Assignment (Const Mask)", valuesPerSecondFactor, "Values");

            const Vector one(One);

            Vector *data = new Vector[Factor];
            for (int i = 0; i < Factor; ++i) {
                data[i].makeZero();
            }

            for (int rep = 0; rep < Repetitions; ++rep) {
                const Mask mask0 = PseudoRandom<Vector>::next() < PseudoRandom<Vector>::next();
                const Mask mask1 = PseudoRandom<Vector>::next() < PseudoRandom<Vector>::next();
                const Mask mask2 = PseudoRandom<Vector>::next() < PseudoRandom<Vector>::next();
                const Mask mask3 = PseudoRandom<Vector>::next() < PseudoRandom<Vector>::next();
                timer.Start();
                for (int i = 0; i < Factor; i += 4) {
                    data[i + 0](mask0) = one;
                    data[i + 1](mask1) = one;
                    data[i + 2](mask2) = one;
                    data[i + 3](mask3) = one;
                }
                timer.Stop();
            }

            timer.Print();
            for (int i = 0; i < Factor; ++i) {
                blackHole &= (data[i] < 1);
            }
        }
        {
            Benchmark timer("Conditional Assignment (Random Mask)", valuesPerSecondFactor, "Values");

            const Vector one(One);

            enum {
                MaskCount = 256 / Vector::Size
            };
            Mask masks[MaskCount];
            for (int i = 0; i < MaskCount; ++i) {
                masks[i] = PseudoRandom<Vector>::next() < PseudoRandom<Vector>::next();
            }

            Vector *data = new Vector[Factor];
            for (int i = 0; i < Factor; ++i) {
                data[i].makeZero();
            }

            for (int rep = 0; rep < Repetitions; ++rep) {
                timer.Start();
                for (int i = 0; i < Factor; ++i) {
                    data[i](masks[i & (MaskCount - 1)]) = one;
                }
                timer.Stop();
            }

            timer.Print();
            for (int i = 0; i < Factor; ++i) {
                blackHole &= (data[i] < 1);
            }
        }
    }
};

int bmain(Benchmark::OutputMode out)
{
    const int Repetitions = out == Benchmark::Stdout ? 4 : 50;
    Benchmark::addColumn("datatype");
    Benchmark::setColumnData("datatype", "float_v");
    CondAssignment<float_v>::run(Repetitions);
    Benchmark::setColumnData("datatype", "short_v");
    CondAssignment<short_v>::run(Repetitions);
#ifdef USE_SSE
    Benchmark::setColumnData("datatype", "sfloat_v");
    CondAssignment<sfloat_v>::run(Repetitions);
#endif
    return 0;
}
