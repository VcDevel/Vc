/*
    Copyright (C) 2010 Jochen Gerhard <gerhard@compeng.uni-frankfurt.de>
    Copyright (C) 2010 Matthias Kretz <kretz@kde.org>

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

#include <Vc/Vc>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "../../benchmarks/tsc.h"

#define USE_SCALAR_SINCOS

enum {
  N = 10240000,
  PrintStep = 1000000
};

static const float epsilon = 1e-7;
static const float lower = 0.f;
static const float upper = 40000.f;
static const float h = (upper - lower) / N;

// dfu is the derivative of fu. This is really easy for sine and cosine:
static inline float  fu(float x) { return ( std::sin(x) ); }
static inline float dfu(float x) { return ( std::cos(x) ); }

static inline Vc::float_v fu(Vc::float_v x) {
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

static inline Vc::float_v dfu(Vc::float_v x) {
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

using Vc::float_v;

// It is important for this example that the following variables (especially dy_points) are global
// variables. Else the compiler can optimze all calculations of dy away except for the few places
// where the value is used in printResults.
Vc::Memory<float_v, N> x_points;
Vc::Memory<float_v, N> y_points;
Vc::Memory<float_v, N> dy_points;

void printResults()
{
    std::cout
        << "------------------------------------------------------------\n"
        << std::setw(15) << "fu(x_i)"
        << std::setw(15) << "FD fu'(x_i)"
        << std::setw(15) << "SYM fu'(x)"
        << std::setw(15) << "error %\n";
    for (int i = 0; i < N; i += PrintStep) {
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

int main()
{
    {
      float_v x_i(float_v::IndexType::IndexesFromZero());
      for ( unsigned int i = 0; i < x_points.vectorsCount(); ++i, x_i += float_v::Size ) {
        const float_v x = x_i * h;
        x_points.vector(i) = x;
        y_points.vector(i) = fu(x);
      }
    }

    double speedup;
    TimeStampCounter timer;
    {
        std::cout << "\n" << std::setw(60) << "Classical finite difference method" << std::endl;
        timer.Start();

        const float oneOver2h = 0.5f / h;

        // set borders explicit as up- or downdifferential
        dy_points[0] = (y_points[1] - y_points[0]) / h;
        for ( int i = 1; i < N - 1; ++i) {
            dy_points[i] = (y_points[i + 1] - y_points[i - 1]) * oneOver2h;
        }
        dy_points[N - 1] = (y_points[N - 1] - y_points[N - 2]) / h;

        timer.Stop();
        printResults();
        std::cout << "cycle count: " << timer.Cycles() << "\n";
    }
    speedup = timer.Cycles();
    {
        std::cout << std::setw(60) << "Vectorized finite difference method" << std::endl;
        timer.Start();

        const float_v oneOver2h = 0.5f / h;

        // set borders explicit as up- or downdifferential
        dy_points[0] = (y_points[1] - y_points[0]) / h;
        // y  [...................................]
        //     00001111222233334444555566667777
        //       00001111222233334444555566667777
        // dy [...................................]
        //      00001111222233334444555566667777
        for (unsigned int i = 0; i < (y_points.entriesCount() - 2) / float_v::Size; ++i) {
            const float_v left = y_points.vector(i);
            const float_v right = y_points.vector(i, 2);
            dy_points.vector(i, 1) = (right - left) * oneOver2h;
        }
        // y  [...................................]
        //                                  8888
        //                                    8888
        // dy [...................................]
        //                                   8888
        {
            const unsigned int i = y_points.vectorsCount() - 1;
            const float_v left = y_points.vector(i, -2);
            const float_v right = y_points.lastVector();
            dy_points.vector(i, -1) = (right - left) * oneOver2h;
        }
        dy_points[N - 1] = (y_points[N - 1] - y_points[N - 2]) / h;

        timer.Stop();
        printResults();
        std::cout << "cycle count: " << timer.Cycles() << "\n";
    }
    speedup /= timer.Cycles();
    std::cout << "Speedup: " << speedup << "\n";
    return 0;
}
