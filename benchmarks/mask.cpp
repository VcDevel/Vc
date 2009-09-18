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
        const double valuesPerSecondFactor = Factor * Vector::Size * 0.5;

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
        const Vector one(One);

        {
            // gcc compiles the Simple::Vector version such that if all four masks are false it runs
            // 20 times faster than otherwise
            Benchmark timer("Conditional Assignment (Const Mask)", valuesPerSecondFactor, "Op");
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
        }
        {
            Benchmark timer("Conditional Assignment (Random Mask)", valuesPerSecondFactor, "Op");
            for (int rep = 0; rep < Repetitions; ++rep) {
                timer.Start();
                for (int i = 0; i < Factor; ++i) {
                    data[i](masks[i & (MaskCount - 1)]) = one;
                }
                timer.Stop();
            }
            timer.Print();
        }
        {
            Benchmark timer("Masked Pre-Increment", Factor * Vector::Size * 0.5, "Op");
            for (int rep = 0; rep < Repetitions; ++rep) {
                timer.Start();
                for (int i = 0; i < Factor; i += 4) {
                    ++data[i + 0](masks[(i + 0) & (MaskCount - 1)]);
                    ++data[i + 1](masks[(i + 1) & (MaskCount - 1)]);
                    ++data[i + 2](masks[(i + 2) & (MaskCount - 1)]);
                    ++data[i + 3](masks[(i + 3) & (MaskCount - 1)]);
                }
                timer.Stop();
            }
            timer.Print();
        }
        {
            Benchmark timer("Masked Post-Decrement", Factor * Vector::Size * 0.5, "Op");
            for (int rep = 0; rep < Repetitions; ++rep) {
                timer.Start();
                for (int i = 0; i < Factor; i += 4) {
                    data[i + 0](masks[(i + 0) & (MaskCount - 1)])--;
                    data[i + 1](masks[(i + 1) & (MaskCount - 1)])--;
                    data[i + 2](masks[(i + 2) & (MaskCount - 1)])--;
                    data[i + 3](masks[(i + 3) & (MaskCount - 1)])--;
                }
                timer.Stop();
            }
            timer.Print();
        }
        {
            const Vector x(3);
            Benchmark timer("Masked Multiply-Add", Factor * Vector::Size, "Op");
            for (int rep = 0; rep < Repetitions; ++rep) {
                timer.Start();
                for (int i = 0; i < Factor; i += 4) {
                    data[i + 0](masks[(i + 0) & (MaskCount - 1)]) *= x;
                    data[i + 1](masks[(i + 1) & (MaskCount - 1)]) *= x;
                    data[i + 2](masks[(i + 2) & (MaskCount - 1)]) *= x;
                    data[i + 3](masks[(i + 3) & (MaskCount - 1)]) *= x;
                    data[i + 0](masks[(i + 0) & (MaskCount - 1)]) += one;
                    data[i + 1](masks[(i + 1) & (MaskCount - 1)]) += one;
                    data[i + 2](masks[(i + 2) & (MaskCount - 1)]) += one;
                    data[i + 3](masks[(i + 3) & (MaskCount - 1)]) += one;
                }
                timer.Stop();
            }
            timer.Print();
        }
        {
            const Vector x(3);
            Benchmark timer("Masked Division", Factor * Vector::Size * 0.5, "Op");
            for (int rep = 0; rep < Repetitions; ++rep) {
                timer.Start();
                for (int i = 0; i < Factor; i += 4) {
                    data[i + 0](masks[(i + 0) & (MaskCount - 1)]) /= x;
                    data[i + 1](masks[(i + 1) & (MaskCount - 1)]) /= x;
                    data[i + 2](masks[(i + 2) & (MaskCount - 1)]) /= x;
                    data[i + 3](masks[(i + 3) & (MaskCount - 1)]) /= x;
                }
                timer.Stop();
            }
            timer.Print();
        }

        for (int i = 0; i < Factor; ++i) {
            blackHole &= (data[i] < 1);
        }
        delete[] data;
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
#if VC_IMPL_SSE
    Benchmark::setColumnData("datatype", "sfloat_v");
    CondAssignment<sfloat_v>::run(Repetitions);
#endif
    return 0;
}
