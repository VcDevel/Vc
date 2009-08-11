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
#include <cstdio>
#include <cstdlib>

using namespace Vc;

float_m *floatResults = new float_m[4];
short_m *shortResults = new short_m[4];
#ifdef USE_SSE
sfloat_m *sfloatResults = new sfloat_m[4];
#endif

#define unrolled_loop4(_it_, _code_) \
{ enum { _it_ = 0 }; _code_ } \
{ enum { _it_ = 1 }; _code_ } \
{ enum { _it_ = 2 }; _code_ } \
{ enum { _it_ = 3 }; _code_ }

template<typename Vector> class DoCompares
{
    enum {
        Factor = 1280000 / Vector::Size
    };

    public:
        DoCompares(const int Repetitions)
            : a(new Vector[4 * Factor]),
            b(new Vector[4 * Factor])
        {
            setResultPointer();

            {
                Benchmark timer("operator<", 4. * Vector::Size * Factor, "Op");
                for (int repetitions = 0; repetitions < Repetitions; ++repetitions) {
                    for (int i = 0; i < 4 * Factor; ++i) {
                        a[i] = PseudoRandom<Vector>::next();
                        b[i] = PseudoRandom<Vector>::next();
                    }

                    timer.Start();
                    doWork1();
                    timer.Stop();

                }
                timer.Print(Benchmark::PrintAverage);
            }
            {
                Benchmark timer("masked assign with operator==", 4. * Vector::Size * Factor, "Op");
                for (int repetitions = 0; repetitions < Repetitions; ++repetitions) {
                    for (int i = 0; i < 4 * Factor; ++i) {
                        a[i] = PseudoRandom<Vector>::next();
                        b[i] = PseudoRandom<Vector>::next();
                    }

                    timer.Start();
                    doWork2();
                    timer.Stop();

                    for (int i = 0; i < 4 * Factor; ++i) {
                        results[0] = a[i] > Vector(One);
                    }
                }
                timer.Print(Benchmark::PrintAverage);
            }
        }

        ~DoCompares()
        {
            delete[] a;
            delete[] b;
        }

    private:
        void setResultPointer();
        void doWork1();
        void doWork2();

        Vector *a;
        Vector *b;
        typename Vector::Mask *results;
};

template<> inline void DoCompares<float_v>::setResultPointer() { results = floatResults; }
template<> inline void DoCompares<short_v>::setResultPointer() { results = shortResults; }
#ifdef USE_SSE
template<> inline void DoCompares<sfloat_v>::setResultPointer() { results = sfloatResults; }
#endif

template<typename Vector> inline void DoCompares<Vector>::doWork1()
{
    for (int i = 0; i < Factor; ++i) {
        const int ii = 4 * i;
        unrolled_loop4(j,
                results[j] = a[ii + j] < b[ii + j];
                )
    }
}
template<typename Vector> inline void DoCompares<Vector>::doWork2()
{
    const Vector one(One);
    for (int i = 0; i < Factor; ++i) {
        const int ii = 4 * i;
        unrolled_loop4(j,
                a[ii + j](a[ii + j] == b[ii + j]) = one;
                )
    }
}

int bmain(Benchmark::OutputMode /*out*/)
{
    const int Repetitions = 10;// out == Benchmark::Stdout ? 10 : 20;

    Benchmark::addColumn("datatype");

    Benchmark::setColumnData("datatype", "float_v");
    DoCompares<float_v> a(Repetitions);
    Benchmark::setColumnData("datatype", "short_v");
    DoCompares<short_v> b(Repetitions);
#ifdef USE_SSE
    Benchmark::setColumnData("datatype", "sfloat_v");
    DoCompares<sfloat_v> c(Repetitions);
#endif

    return 0;
}
