/*{{{
    Copyright (C) 2013 Matthias Kretz <kretz@kde.org>

    Permission to use, copy, modify, and distribute this software
    and its documentation for any purpose and without fee is hereby
    granted, provided that the above copyright notice appear in all
    copies and that both that the copyright notice and this
    permission notice and warranty disclaimer appear in supporting
    documentation, and that the name of the author not be used in
    advertising or publicity pertaining to distribution of the
    software without specific, written prior permission.

    The author disclaim all warranties with regard to this
    software, including all implied warranties of merchantability
    and fitness.  In no event shall the author be liable for any
    special, indirect or consequential damages or any damages
    whatsoever resulting from loss of use, data or profits, whether
    in an action of contract, negligence or other tortious action,
    arising out of or in connection with the use or performance of
    this software.

}}}*/

#include <array>
#include <memory>

#include <Vc/Vc>
#include "../tsc.h"

using Vc::float_v;

/*
 * This example shows how an arbitrary problem scales depending on working-set size and FLOPs per
 * load/store. Understanding this can help to create better implementations.
 */

/*
 * The Runner is a method to generate the different scenarios with all parameters to the Work
 * available as constant expressions.
 * The idea is to have the compiler able to optimize as much as possible so that the actual workload
 * alone is benchmarked.
 *
 * The Runner recursively calls operator() on the Work template class with varying arguments for N
 * and FLOPs.
 */
template<template<std::size_t N, std::size_t M, int, int> class Work, std::size_t N = 256, std::size_t M = 4, int FLOPs = 2> struct Runner
{
    static void run() {
        Work<N, M, (N > 4096 ? 1 : 4096 / N), FLOPs>()();
        Runner<Work, N, M, int(FLOPs * 1.5)>::run();
    }
};

template<template<std::size_t N, std::size_t M, int, int> class Work, std::size_t N, std::size_t M> struct Runner<Work, N, M, 211>
{
    static void run() {
        Runner<Work, N * 2, M>::run();
    }
};
template<template<std::size_t N, std::size_t M, int, int> class Work, std::size_t M, int FLOPs> struct Runner<Work, 256 * 1024 * 1024, M, FLOPs>
{
    static void run() {
    }
};

/*
 * The Flops helper struct generates code that executes FLOPs many floating-point SIMD instructions
 * (add, sub, and mul)
 */
template<int FLOPs> struct Flops
{
    inline float_v operator()(float_v a, float_v b, float_v c)
    {
        typedef Flops<(FLOPs - 5) / 2> F1;
        typedef Flops<(FLOPs - 4) / 2> F2;
        return F1()(a + b, a * b, c) + F2()(a * c, b + c, a);
    }
};

template<> inline float_v Flops<2>::operator()(float_v a, float_v b, float_v c)
{
    return a * b + c;
}
template<> inline float_v Flops<3>::operator()(float_v a, float_v b, float_v c)
{
    return a * b + (c - a);
}
template<> inline float_v Flops<4>::operator()(float_v a, float_v b, float_v c)
{
    return (a * b + c) + a * c;
}
template<> inline float_v Flops<5>::operator()(float_v a, float_v b, float_v c)
{
    return a * b + (a + c) + a * c;
}
template<> inline float_v Flops<6>::operator()(float_v a, float_v b, float_v c)
{
    return (a * b + (a + c)) + (a * c - b);
}
template<> inline float_v Flops<7>::operator()(float_v a, float_v b, float_v c)
{
    return (a * b + (a + c)) + (a * c - (b + c));
}
template<> inline float_v Flops<8>::operator()(float_v a, float_v b, float_v c)
{
    return (a * b + (a + c) + b) + (a * c - (b + c));
}

/*
 * This is the benchmark code. It is called from Runner and uses Flops to do the work.
 */
template<std::size_t _N, std::size_t M, int Repetitions, int FLOPs>
struct ScaleWorkingSetSize
{
    void operator()()
    {
        constexpr std::size_t N = _N / sizeof(float_v) + 3 * 16 / float_v::Size;
        typedef std::array<std::array<float_v, N>, M> Cont;
        auto data = Vc::make_unique<Cont, Vc::AlignOnPage>();
        for (auto &arr : *data) {
            for (auto &value : arr) {
                value = float_v::Random();
            }
        }

        TimeStampCounter tsc;
        double throughput = 0.;
        for (std::size_t i = 0; i < 2 + 512 / N; ++i) {
            tsc.start();
            // ------------- start of the benchmarked code ---------------
            for (int repetitions = 0; repetitions < Repetitions; ++repetitions) {
                for (std::size_t m = 0; m < M; ++m) {
                    for (std::size_t n = 0; n < N; ++n) {
                        (*data)[m][n] = Flops<FLOPs>()((*data)[(m + 1) % M][n],
                                (*data)[(m + 2) % M][n],
                                (*data)[(m + 3) % M][n]);
                    }
                }
            }
            // -------------- end of the benchmarked code ----------------
            tsc.stop();

            throughput = std::max(throughput, (Repetitions * M * N * float_v::Size * FLOPs) / static_cast<double>(tsc.cycles()));
        }

        const long bytes = N * M * sizeof(float_v);
        printf("%10lu Byte | %4.2f FLOP/Byte | %4.1f FLOP/cycle\n", bytes, static_cast<double>(float_v::Size * FLOPs) / (4 * sizeof(float_v)), throughput
                );
    }
};

int main()
{
    ScaleWorkingSetSize<256, 4, 10, 2>()();
    printf("%10s | %4s | %4s\n", "Working-Set Size", "FLOPs per Byte", "Throughput (FLOPs/Cycle)");
    Runner<ScaleWorkingSetSize>::run();
    return 0;
}
