/*
    Copyright (C) 2010 Jochen Gerhard <gerhard@compeng.uni-frankfurt.de>
    Copyright (C) 2010-2015 Matthias Kretz <kretz@kde.org>

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

*/

/*!
  Finite difference method example

  We calculate central differences for a given function and
  compare it to the analytical solution.

*/

//! [includes]
#include <Vc/Vc>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "../tsc.h"

using Vc::float_v;
//! [includes]

#define USE_SCALAR_SINCOS

//! [constants]
static constexpr std::size_t N = 10240000, PrintStep = 1000000;

static constexpr float epsilon = 1e-7f;
static constexpr float lower = 0.f;
static constexpr float upper = 40000.f;
static constexpr float h = (upper - lower) / N;
//! [constants]

//! [functions]
// dfu is the derivative of fu. This is really easy for sine and cosine:
static inline float  fu(float x) { return ( std::sin(x) ); }
static inline float dfu(float x) { return ( std::cos(x) ); }

static inline Vc::float_v fu(Vc::float_v::AsArg x) {
#ifdef USE_SCALAR_SINCOS
  Vc::float_v r;
  for (size_t i = 0; i < Vc::float_v::Size; ++i) {
    r[i] = std::sin(x[i]);
  }
  return r;
#else
  return Vc::sin(x);
#endif
}

static inline Vc::float_v dfu(Vc::float_v::AsArg x) {
#ifdef USE_SCALAR_SINCOS
  Vc::float_v r;
  for (size_t i = 0; i < Vc::float_v::Size; ++i) {
    r[i] = std::cos(x[i]);
  }
  return r;
#else
  return Vc::cos(x);
#endif
}
//! [functions]

// It is important for this example that the following variables (especially dy_points) are global
// variables. Else the compiler can optimze all calculations of dy away except for the few places
// where the value is used in printResults.
Vc::Memory<float_v, N> x_points;
Vc::Memory<float_v, N> y_points;
float *Vc_RESTRICT dy_points;

void printResults()
{
    std::cout
        << "------------------------------------------------------------\n"
        << std::setw(15) << "fu(x_i)"
        << std::setw(15) << "FD fu'(x_i)"
        << std::setw(15) << "SYM fu'(x)"
        << std::setw(15) << "error %\n";
    for (std::size_t i = 0; i < N; i += PrintStep) {
        std::cout
            << std::setw(15) << y_points[i]
            << std::setw(15) << dy_points[i]
            << std::setw(15) << dfu(x_points[i])
            << std::setw(15) << std::abs((dy_points[i] - dfu(x_points[i])) / (dfu(x_points[i] + epsilon)) * 100)
            << "\n";
    }
    std::cout
        << std::setw(15) << y_points[N - 1]
        << std::setw(15) << dy_points[N - 1]
        << std::setw(15) << dfu(x_points[N - 1])
        << std::setw(15) << std::abs((dy_points[N - 1] - dfu(x_points[N - 1])) / (dfu(x_points[N - 1] + epsilon)) * 100)
        << std::endl;
}

int Vc_CDECL main()
{
    {
      float_v x_i(Vc::IndexesFromZero);
      for ( unsigned int i = 0; i < x_points.vectorsCount(); ++i, x_i += float_v::Size ) {
        const float_v x = x_i * h;
        x_points.vector(i) = x;
        y_points.vector(i) = fu(x);
      }
    }

    dy_points = Vc::malloc<float, Vc::AlignOnVector>(N + float_v::Size - 1) + (float_v::Size - 1);

    double speedup;
    TimeStampCounter timer;

    { ///////// ignore this part - it only wakes up the CPU ////////////////////////////
        const float oneOver2h = 0.5f / h;

        // set borders explicit as up- or downdifferential
        dy_points[0] = (y_points[1] - y_points[0]) / h;
        // GCC auto-vectorizes the following loop. It is interesting to see that both Vc::Scalar and
        // Vc::SSE are faster, though.
        for (std::size_t i = 1; i < N - 1; ++i) {
            dy_points[i] = (y_points[i + 1] - y_points[i - 1]) * oneOver2h;
        }
        dy_points[N - 1] = (y_points[N - 1] - y_points[N - 2]) / h;
    } //////////////////////////////////////////////////////////////////////////////////

    {
        std::cout << "\n" << std::setw(60) << "Classical finite difference method" << std::endl;
        timer.start();

        const float oneOver2h = 0.5f / h;

        // set borders explicit as up- or downdifferential
        dy_points[0] = (y_points[1] - y_points[0]) / h;
        // GCC auto-vectorizes the following loop. It is interesting to see that both Vc::Scalar and
        // Vc::SSE are faster, though.
        for (std::size_t i = 1; i < N - 1; ++i) {
            dy_points[i] = (y_points[i + 1] - y_points[i - 1]) * oneOver2h;
        }
        dy_points[N - 1] = (y_points[N - 1] - y_points[N - 2]) / h;

        timer.stop();
        printResults();
        std::cout << "cycle count: " << timer.cycles()
            << " | " << static_cast<double>(N * 2) / timer.cycles() << " FLOP/cycle"
            << " | " << static_cast<double>(N * 2 * sizeof(float)) / timer.cycles() << " Byte/cycle"
            << "\n";
    }

    speedup = timer.cycles();
    {
        std::cout << std::setw(60) << "Vectorized finite difference method" << std::endl;
        timer.start();

        // All the differentials require to calculate (r - l) / 2h, where we calculate 1/2h as a
        // constant before the loop to avoid unnecessary calculations. Note that a good compiler can
        // already do this for you.
        const float_v oneOver2h = 0.5f / h;

        // Calculate the left border
        dy_points[0] = (y_points[1] - y_points[0]) / h;

        // Calculate the differentials streaming through the y and dy memory. The picture below
        // should give an idea of what values in y get read and what values are written to dy in
        // each iteration:
        //
        // y  [...................................]
        //     00001111222233334444555566667777
        //       00001111222233334444555566667777
        // dy [...................................]
        //      00001111222233334444555566667777
        //
        // The loop is manually unrolled four times to improve instruction level parallelism and
        // prefetching on architectures where four vectors fill one cache line. (Note that this
        // unrolling breaks auto-vectorization of the Vc::Scalar implementation when compiling with
        // GCC.)

        // Use streaming stores to reduce the required memory bandwidth. Without streaming
        // stores the CPU would first have to load the cache line, where the store occurs, from
        // memory into L1, then overwrite the data, and finally write it back to memory. But
        // since we never actually need the data that the CPU fetched from memory we'd like to
        // keep that bandwidth free for real work. Streaming stores allow us to issue stores
        // which the CPU gathers in store buffers to form full cache lines, which then get
        // written back to memory directly without the costly read. Thus we make better use of
        // the available memory bandwidth.
        auto dy = Vc::makeIterator<float_v>(&dy_points[1], Vc::Streaming);

        // Prefetches make sure the data which is going to be used in the next iterations is already
        // in the L1 cache. The Vc::Prefetch<>() flag provides some sensible default for loops where no
        // elements are skipped. You can use Vc::Prefetch<L1, L2, Shared/Exclusive>() instead to set the stride of
        // L1 and L2 prefetches manually.
        auto y = y_points.begin(Vc::Prefetch<>());
        float_v y0 = *y++;
        const auto y_it_last = y_points.end();
#ifdef Vc_ICC
#pragma noprefetch
#endif
        for (; y < y_it_last; y += 4 , dy += 4) {
            // calculate float_v::Size differentials per (left - right) / 2h
            float_v y1 = y[0];
            float_v y2 = y[1];
            float_v y3 = y[2];
            static_assert(float_v::Size >= 2, "This code requires a SIMD vector with at least two entries.");
            dy[0] = (y0.shifted(2, y1) - y0) * oneOver2h;
            y0 = y[3];
            dy[1] = (y1.shifted(2, y2) - y1) * oneOver2h;
            dy[2] = (y2.shifted(2, y3) - y2) * oneOver2h;
            dy[3] = (y3.shifted(2, y0) - y3) * oneOver2h;
        }
        /* Alternative:
         * use the STL transform algorithm. We have to rely on the compiler for unrolling, though.
         * At least ICC doesn't produce the best code from this...
         *
        std::transform(Vc::makeIterator(y_points.vector(1), Vc::Prefetch<>()),
                y_points.end(Vc::Prefetch<>()),
                Vc::makeIterator<float_v>(&dy_points[1], Vc::Streaming),
                [&y0,oneOver2h](float_v y1) -> float_v {
                    const auto r = (y0.shifted(2, y1) - y0) * oneOver2h;
                    y0 = y1;
                    return r;
                });
        */

        // Process the last vector. Note that this works for any N because Vc::Memory adds padding
        // to y_points and dy_points such that the last scalar value is somewhere inside lastVector.
        // The correct right border value for dy_points is overwritten in the last step unless N is
        // a multiple of float_v::Size + 2.
        // y  [...................................]
        //                                  8888
        //                                    8888
        // dy [...................................]
        //                                   8888
        {
            const size_t i = y_points.vectorsCount() - 1;
            const float_v left = y_points.vector(i, -2);
            const float_v right = y_points.lastVector();
            ((right - left) * oneOver2h).store(&dy_points[i * float_v::Size - 1], Vc::Unaligned);
        }

        // ... and finally the right border
        dy_points[N - 1] = (y_points[N - 1] - y_points[N - 2]) / h;

        timer.stop();
        printResults();
        std::cout << "cycle count: " << timer.cycles()
            << " | " << static_cast<double>(N * 2) / timer.cycles() << " FLOP/cycle"
            << " | " << static_cast<double>(N * 2 * sizeof(float)) / timer.cycles() << " Byte/cycle"
            << "\n";
    }
    speedup /= timer.cycles();
    std::cout << "Speedup: " << speedup << "\n";
//! [cleanup]

    Vc::free(dy_points - float_v::Size + 1);
    return 0;
}
//! [cleanup]
