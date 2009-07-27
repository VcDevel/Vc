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

#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <iostream>
#include <iomanip>
#include <list>
#include <time.h>
#include <cstring>
#ifdef _MSC_VER
#include <windows.h>
#include <float.h>
#else
#include <cmath>
#endif

class Benchmark
{
public:
    explicit Benchmark(const char *name, double factor = 0., const char *X = 0)
        : fName(name), fFactor(factor), fX(X)
    {
        const bool interpret = (fFactor != 0.);
        char header[128];
        std::memset(header, 0, 128);
        std::strcpy(header, "+----------------+----------------+----------------+----------------+");
        if (!interpret) {
            header[35] = '\0';
        }
        const int titleLen = std::strlen(fName);
        const int headerLen = std::strlen(header);
        const int offset = (headerLen - titleLen) / 2;
        if (offset > 0) {
            std::memcpy(&header[offset], fName, titleLen);
            header[offset - 1] = ' ';
            header[offset + titleLen] = ' ';
            std::cout << header << std::flush;
        } else {
            std::cout << fName << std::flush;
        }
    }

    enum Flags {
        PrintValues = 0,
        PrintAverage = 1
    };

    void Start();
    void Mark();
    void Stop();
    void Print(int = PrintValues) const;

private:
    struct DataPoint
    {
        double fRealElapsed;
        double fCpuElapsed;
    };
    const char *const fName;
    const double fFactor;
    const char *const fX;
#ifdef _MSC_VER
    QWORD fRealTime;
    clock_t fCpuTime;
#else
    struct timespec fRealTime;
    struct timespec fCpuTime;
#endif
    std::list<DataPoint> fDataPoints;
};

inline void Benchmark::Start()
{
#ifdef _MSC_VER
    fRealTime = reinterpret_cast<QWORD &>(QueryPerfomanceCounter());
    fCpuTime = clock();
#else
    clock_gettime( CLOCK_MONOTONIC, &fRealTime );
    clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &fCpuTime );
#endif
}

#ifndef _MSC_VER
static inline double convertTimeSpec(const struct timespec &ts)
{
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}
#endif

static const double SECONDS_PER_CLOCK = 1. / CLOCKS_PER_SEC;
#ifdef _MSC_VER
static const double SECONDS_PER_PERFCOUNT = 1. / QueryPerfomanceFrequency();
#endif

inline void Benchmark::Stop()
{
#ifdef _MSC_VER
    QWORD real = reinterpret_cast<QWORD &>(QueryPerfomanceCounter());
    clock_t cpu = clock();
    const DataPoint p = {
        (real - fRealTime) * SECONDS_PER_PERFCOUNT,
        (cpu - fCpuTime) * SECONDS_PER_CLOCK
    };
#else
    struct timespec real, cpu;
    clock_gettime( CLOCK_MONOTONIC, &real );
    clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &cpu );

    const DataPoint p = {
        convertTimeSpec(real) - convertTimeSpec(fRealTime),
        convertTimeSpec(cpu ) - convertTimeSpec(fCpuTime )
    };
#endif
    fDataPoints.push_back(p);
}

inline void Benchmark::Mark()
{
    Stop();
    Start();
}

static inline void prettyPrintSeconds(double v)
{
    static const char prefix[] = { ' ', 'm', 'u', 'n', 'p' };
    if (v == 0.) {
        std::cout << "      0       ";
    } else if (v < 1.) {
        int i = 0;
        do {
            v *= 1000.;
            ++i;
        } while (v < 1.);
        std::cout << std::setw(11) << v << ' ' << prefix[i] << 's';
    } else if (v > 60.) {
        std::cout << std::setw(10) << v / 60. << " min";
    }
}

static inline void prettyPrintCount(double v)
{
    static const char prefix[] = { ' ', 'k', 'M', 'G', 'T', 'P', 'E' };
    int i = 0;
#ifdef _MSC_VER
    if (_finite(v)) {
#elif defined(__INTEL_COMPILER)
    if (::isfinite(v)) {
#else
    if (std::isfinite(v)) {
#endif
        while (v > 1000.) {
            v *= 0.001;
            ++i;
        }
    }
    std::cout << std::setw(12) << v << ' ' << prefix[i];
}

inline void Benchmark::Print(int f) const
{
    typedef std::list<DataPoint>::const_iterator It;
    double cpuAvg = 0.;
    double realAvg = 0.;
    const bool interpret = (fFactor != 0.);

    std::cout << "\n|    CPU time    |   Real time    |";
    if (interpret) {
        std::cout << std::setw(5) << fX << "/s [CPU]   |";
        std::cout << std::setw(5) << fX << "/s [Real]  |";
    }
    for (It i = fDataPoints.begin(); i != fDataPoints.end(); ++i) {
        std::cout << "\n| ";
        prettyPrintSeconds(i->fCpuElapsed);
        std::cout << " | ";
        prettyPrintSeconds(i->fRealElapsed);
        std::cout << " | ";
        if (interpret) {
            prettyPrintCount(fFactor / i->fCpuElapsed);
            std::cout << " | ";
            prettyPrintCount(fFactor / i->fRealElapsed);
            std::cout << " |";
        }
    }
    if (f & PrintAverage) {
        if (interpret) {
            std::cout << "\n|----------------------------- Average -----------------------------|";
        } else {
            std::cout << "\n|------------ Average ------------|";
        }

        for (It i = fDataPoints.begin(); i != fDataPoints.end(); ++i) {
            cpuAvg += i->fCpuElapsed;
            realAvg += i->fRealElapsed;
        }
        const double count = static_cast<double>(fDataPoints.size());
        cpuAvg /= count;
        realAvg /= count;

        std::cout << "\n| ";
        prettyPrintSeconds(cpuAvg);
        std::cout << " | ";
        prettyPrintSeconds(realAvg);
        std::cout << " | ";
        if (interpret) {
            prettyPrintCount(fFactor / cpuAvg);
            std::cout << " | ";
            prettyPrintCount(fFactor / realAvg);
            std::cout << " |";
        }
    }
    std::cout << "\n+----------------+----------------+"
        << (interpret ? "----------------+----------------+" : "") << std::endl;
}

#endif // BENCHMARK_H
