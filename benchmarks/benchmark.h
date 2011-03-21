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
#include <sstream>
#include <iomanip>
#include <list>
#include <vector>
#include <algorithm>
#include <time.h>
#include <cstring>
#include <string>
#include <fstream>
#ifndef VC_BENCHMARK_NO_MLOCK
#include <sys/mman.h>
#endif
#ifdef _MSC_VER
#include <windows.h>
#include <float.h>
#else
#include <cmath>
#endif
#ifdef __APPLE__
#include <mach/mach_time.h>
// method to get monotonic mac time, inspired by 
// http://www.wand.net.nz/~smr26/wordpress/2009/01/19/monotonic-time-in-mac-os-x/
#endif

#include "tsc.h"

#if !defined(WIN32) && !defined(__APPLE__)
//#define VC_USE_CPU_TIME
#endif

class Benchmark
{
    friend int main(int, char**);
    class FileWriter
    {
        public:
            FileWriter(const std::string &filename);
            ~FileWriter();

            void declareData(const std::string &name, const std::list<std::string> &header);
            void addDataLine(const std::list<std::string> &data);

            void addColumn(const std::string &name);
            void setColumnData(const std::string &name, const std::string &data);
            void finalize() { m_finalized = true; }

        private:
            std::ofstream m_file;
            bool m_finalized;
            std::string m_currentName;
            std::list<std::string> m_header;
            struct ExtraColumn
            {
                ExtraColumn(const std::string &n) : name(n) {}
                std::string name;
                std::string data;
                inline bool operator==(const ExtraColumn &rhs) const { return name == rhs.name; }
                inline bool operator==(const std::string &rhs) const { return name == rhs; }
            };
            std::list<ExtraColumn> m_extraColumns;
    };
public:
    static void addColumn(const std::string &name);
    static void setColumnData(const std::string &name, const std::string &data);
    static void finalize();

    explicit Benchmark(const std::string &name, double factor = 0., const std::string &X = std::string());
    void changeInterpretation(double factor, const char *X);

    bool wantsMoreDataPoints() const;
    bool Start();
    void Mark();
    void Stop();
    bool Print();

private:
    void printMiddleLine() const;
    void printBottomLine() const;

    const std::string fName;
    double fFactor;
    std::string fX;
#ifdef _MSC_VER
    __int64 fRealTime;
#elif defined(__APPLE__)
    uint64_t fRealTime;
#else
    struct timespec fRealTime;
    struct timespec fCpuTime;
#endif
    double m_mean[3];
    double m_stddev[3];
    TimeStampCounter fTsc;
    int m_dataPointsCount;
    static FileWriter *s_fileWriter;

    static const char greenEsc  [8];
    static const char cyanEsc   [8];
    static const char reverseEsc[5];
    static const char normalEsc [5];
};

inline bool Benchmark::Start()
{
#ifdef _MSC_VER
    QueryPerformanceCounter((LARGE_INTEGER *)&fRealTime);
#elif defined(__APPLE__)
    fRealTime = mach_absolute_time();
#else
    clock_gettime( CLOCK_MONOTONIC, &fRealTime );
    clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &fCpuTime );
#endif
    fTsc.Start();
    return true;
}

#ifndef _MSC_VER
static inline double convertTimeSpec(const struct timespec &ts)
{
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}
#endif

inline void Benchmark::Stop()
{
    fTsc.Stop();
#ifdef _MSC_VER
    __int64 real = 0, freq = 0;
    QueryPerformanceCounter((LARGE_INTEGER *)&real);
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    const double elapsedRealTime = static_cast<double>(real - fRealTime) / freq;
#elif defined(__APPLE__)
    uint64_t real = mach_absolute_time();
    static mach_timebase_info_data_t info = {0,0};  
    
    if (info.denom == 0)  
    	mach_timebase_info(&info);  
    
    uint64_t nanos = (real - fRealTime ) * (info.numer / info.denom);
    const double elapsedRealTime = nanos * 1e-9;
#else
    struct timespec real;
    clock_gettime( CLOCK_MONOTONIC, &real );
#ifdef VC_USE_CPU_TIME
    struct timespec cpu;
    clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &cpu );
    const double elapsedCpuTime = convertTimeSpec(cpu ) - convertTimeSpec(fCpuTime );
    m_mean[2] += elapsedCpuTime;
    m_stddev[2] += elapsedCpuTime * elapsedCpuTime;
#endif
    const double elapsedRealTime = convertTimeSpec(real) - convertTimeSpec(fRealTime);
#endif
    m_mean[0] += elapsedRealTime;
    m_mean[1] += fTsc.Cycles();
    m_stddev[0] += elapsedRealTime * elapsedRealTime;
    m_stddev[1] += fTsc.Cycles() * fTsc.Cycles();
    ++m_dataPointsCount;
}

int bmain();
extern const char *printHelp2;

#define SET_HELP_TEXT(str) \
    int _set_help_text_init() { \
        printHelp2 = str; \
        return 0; \
    } \
    int _set_help_text_init_ = _set_help_text_init()

#ifdef __GNUC__
#  define VC_IS_UNLIKELY(x) __builtin_expect(x, 0)
#  define VC_IS_LIKELY(x) __builtin_expect(x, 1)
#else
#  define VC_IS_UNLIKELY(x) x
#  define VC_IS_LIKELY(x) x
#endif

#define benchmark_loop(_bm_obj) \
    for (Benchmark _bm_obj_local = _bm_obj; \
            VC_IS_LIKELY(_bm_obj_local.wantsMoreDataPoints() && _bm_obj_local.Start()) || VC_IS_UNLIKELY(_bm_obj_local.Print()); \
            _bm_obj_local.Stop())

template<typename T, int S> struct KeepResultsHelper {
    static inline void keepDirty(T &tmp0) { asm volatile("":"+r"(tmp0)); }
    static inline void keep(const T &tmp0) { asm volatile(""::"r"(tmp0)); }
    static inline void keep(const T &tmp0, const T &tmp1, const T &tmp2, const T &tmp3) {
#ifdef __x86_64__
        asm volatile(""::"r"(tmp0), "r"(tmp1), "r"(tmp2), "r"(tmp3));
#else
        asm volatile(""::"r"(tmp0), "r"(tmp1));
        asm volatile(""::"r"(tmp2), "r"(tmp3));
#endif
    }
};

#if defined(VC_IMPL_AVX)
template<typename T> struct KeepResultsHelper<T, 16> {
    static inline void keepDirty(T &tmp0) { asm volatile("":"+x"(reinterpret_cast<__m128 &>(tmp0))); }
    static inline void keep(const T &tmp0) { asm volatile(""::"x"(reinterpret_cast<const __m128 &>(tmp0))); }
    static inline void keep(const T &tmp0, const T &tmp1, const T &tmp2, const T &tmp3) {
        asm volatile(""::"x"(reinterpret_cast<const __m128 &>(tmp0)), "x"(reinterpret_cast<const __m128 &>(tmp1)), "x"(reinterpret_cast<const __m128 &>(tmp2)), "x"(reinterpret_cast<const __m128 &>(tmp3)));
    }
};
template<typename T> struct KeepResultsHelper<T, 32> {
    static inline void keepDirty(T &tmp0) {
        asm volatile("":"+x"(reinterpret_cast<__m256 &>(tmp0)));
    }
    static inline void keep(const T &tmp0) {
        asm volatile(""::"x"(reinterpret_cast<const __m256 &>(tmp0)));
    }
    static inline void keep(const T &tmp0, const T &tmp1, const T &tmp2, const T &tmp3) {
        asm volatile(""::"x"(reinterpret_cast<const __m256 &>(tmp0)), "x"(reinterpret_cast<const __m256 &>(tmp1)),
                "x"(reinterpret_cast<const __m256 &>(tmp2)), "x"(reinterpret_cast<const __m256 &>(tmp3)));
    }
};
#elif defined(VC_IMPL_SSE)
template<typename T> struct KeepResultsHelper<T, 16> {
    static inline void keepDirty(T &tmp0) { asm volatile("":"+x"(reinterpret_cast<__m128 &>(tmp0))); }
    static inline void keep(const T &tmp0) { asm volatile(""::"x"(reinterpret_cast<const __m128 &>(tmp0))); }
    static inline void keep(const T &tmp0, const T &tmp1, const T &tmp2, const T &tmp3) {
        asm volatile(""::"x"(reinterpret_cast<const __m128 &>(tmp0)), "x"(reinterpret_cast<const __m128 &>(tmp1)), "x"(reinterpret_cast<const __m128 &>(tmp2)), "x"(reinterpret_cast<const __m128 &>(tmp3)));
    }
};
template<typename T> struct KeepResultsHelper<T, 32> {
    static inline void keepDirty(T &tmp0) {
        asm volatile("":"+x"(reinterpret_cast<__m128 &>(tmp0)), "+x"(reinterpret_cast<__m128 *>(&tmp0)[1]));
    }
    static inline void keep(const T &tmp0) {
        asm volatile(""::"x"(reinterpret_cast<const __m128 &>(tmp0)), "x"(reinterpret_cast<const __m128 *>(&tmp0)[1]));
    }
    static inline void keep(const T &tmp0, const T &tmp1, const T &tmp2, const T &tmp3) {
        asm volatile(""::"x"(reinterpret_cast<const __m128 &>(tmp0)), "x"(reinterpret_cast<const __m128 *>(&tmp0)[1]),
                "x"(reinterpret_cast<const __m128 &>(tmp1)), "x"(reinterpret_cast<const __m128 *>(&tmp1)[1]),
                "x"(reinterpret_cast<const __m128 &>(tmp2)), "x"(reinterpret_cast<const __m128 *>(&tmp2)[1]),
                "x"(reinterpret_cast<const __m128 &>(tmp3)), "x"(reinterpret_cast<const __m128 *>(&tmp3)[1]));
    }
};
#elif defined(VC_IMPL_LRBni)
template<typename T> struct KeepResultsHelper<T, 64> {
    static inline void keepDirty(T &tmp0) {
        asm volatile("":"+x"(reinterpret_cast<__m128 &>(tmp0)), "+x"(reinterpret_cast<__m128 *>(&tmp0)[1]),
                "+x"(reinterpret_cast<__m128 *>(&tmp0)[2]), "+x"(reinterpret_cast<__m128 *>(&tmp0)[3]));
    }
    static inline void keep(const T &tmp0) {
        asm volatile(""::"x"(reinterpret_cast<const __m128 &>(tmp0)), "x"(reinterpret_cast<const __m128 *>(&tmp0)[1]), "x"(reinterpret_cast<const __m128 *>(&tmp0)[2]), "x"(reinterpret_cast<const __m128 *>(&tmp0)[3]));
    }
    static inline void keep(const T &tmp0, const T &tmp1, const T &tmp2, const T &tmp3) {
        asm volatile(""::"x"(reinterpret_cast<const __m128 &>(tmp0)), "x"(reinterpret_cast<const __m128 *>(&tmp0)[1]), "x"(reinterpret_cast<const __m128 *>(&tmp0)[2]), "x"(reinterpret_cast<const __m128 *>(&tmp0)[3]),
                "x"(reinterpret_cast<const __m128 &>(tmp1)), "x"(reinterpret_cast<const __m128 *>(&tmp1)[1]), "x"(reinterpret_cast<const __m128 *>(&tmp1)[2]), "x"(reinterpret_cast<const __m128 *>(&tmp1)[3]));
        asm volatile(""::"x"(reinterpret_cast<const __m128 &>(tmp2)), "x"(reinterpret_cast<const __m128 *>(&tmp2)[1]), "x"(reinterpret_cast<const __m128 *>(&tmp2)[2]), "x"(reinterpret_cast<const __m128 *>(&tmp2)[3]),
                "x"(reinterpret_cast<const __m128 &>(tmp3)), "x"(reinterpret_cast<const __m128 *>(&tmp3)[1]), "x"(reinterpret_cast<const __m128 *>(&tmp3)[2]), "x"(reinterpret_cast<const __m128 *>(&tmp3)[3]));
    }
};
#endif

template<typename T> static inline void keepResultsDirty(T &tmp0)
{
    KeepResultsHelper<T, sizeof(T)>::keepDirty(tmp0);
}

template<typename T> static inline void keepResults(const T &tmp0)
{
    KeepResultsHelper<T, sizeof(T)>::keep(tmp0);
}

template<typename T> static inline void keepResults(const T &tmp0, const T &tmp1, const T &tmp2, const T &tmp3)
{
    KeepResultsHelper<T, sizeof(T)>::keep(tmp0, tmp1, tmp2, tmp3);
}

#endif // BENCHMARK_H
