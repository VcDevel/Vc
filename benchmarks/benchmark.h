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
    static inline void keep(const T &tmp0, const T &tmp1, const T &tmp2, const T &tmp3,
            const T &tmp4, const T &tmp5, const T &tmp6, const T &tmp7) {
#ifdef __x86_64__
        asm volatile(""::"r"(tmp0), "r"(tmp1), "r"(tmp2), "r"(tmp3),
                "r"(tmp4), "r"(tmp5), "r"(tmp6), "r"(tmp7));
#else
        asm volatile(""::"r"(tmp0), "r"(tmp1));
        asm volatile(""::"r"(tmp2), "r"(tmp3));
        asm volatile(""::"r"(tmp4), "r"(tmp5));
        asm volatile(""::"r"(tmp6), "r"(tmp7));
#endif
    }
};

template<typename T>
static inline void _keepXRegister(T x0, T x1, T x2, T x3, T x4, T x5, T x6, T x7)
{
    asm volatile(""::"x"(x0), "x"(x1), "x"(x2), "x"(x3), "x"(x4), "x"(x5), "x"(x6), "x"(x7));
}

#if defined(VC_IMPL_AVX)
template<typename T> struct KeepResultsHelper<Vc::Vector<T>, 16> {
    static inline void keepDirty(Vc::Vector<T> &tmp0) { asm volatile("":"+x"(tmp0.data())); }
    static inline void keep(Vc::Vector<T> tmp0, Vc::Vector<T> tmp1, Vc::Vector<T> tmp2, Vc::Vector<T> tmp3,
            Vc::Vector<T> tmp4, Vc::Vector<T> tmp5, Vc::Vector<T> tmp6, Vc::Vector<T> tmp7) {
        _keepXRegister(tmp0.data(), tmp1.data(), tmp2.data(), tmp3.data(), tmp4.data(), tmp5.data(), tmp6.data(), tmp7.data());
    }
};
template<typename T> struct KeepResultsHelper<Vc::Vector<T>, 32> {
    static inline void keepDirty(Vc::Vector<T> &tmp0) {
        asm volatile("":"+x"(tmp0.data()));
    }
    static inline void keep(Vc::Vector<T> tmp0, Vc::Vector<T> tmp1, Vc::Vector<T> tmp2, Vc::Vector<T> tmp3,
            Vc::Vector<T> tmp4, Vc::Vector<T> tmp5, Vc::Vector<T> tmp6, Vc::Vector<T> tmp7) {
        _keepXRegister(tmp0.data(), tmp1.data(), tmp2.data(), tmp3.data(), tmp4.data(), tmp5.data(), tmp6.data(), tmp7.data());
    }
};
template<unsigned int S1, size_t S2, int S3> struct KeepResultsHelper<Vc::AVX::Mask<S1, S2>, S3> {
    static inline void keep(Vc::AVX::Mask<S1, S2> tmp0, Vc::AVX::Mask<S1, S2> tmp1, Vc::AVX::Mask<S1, S2> tmp2, Vc::AVX::Mask<S1, S2> tmp3,
            Vc::AVX::Mask<S1, S2> tmp4, Vc::AVX::Mask<S1, S2> tmp5, Vc::AVX::Mask<S1, S2> tmp6, Vc::AVX::Mask<S1, S2> tmp7) {
        _keepXRegister(tmp0.data(), tmp1.data(), tmp2.data(), tmp3.data(), tmp4.data(), tmp5.data(), tmp6.data(), tmp7.data());
    }
};
#elif defined(VC_IMPL_SSE)
template<typename T> struct KeepResultsHelper<Vc::Vector<T>, 16> {
    static inline void keepDirty(Vc::Vector<T> &tmp0) { asm volatile("":"+x"(tmp0.data())); }
    static inline void keep(Vc::Vector<T> tmp0, Vc::Vector<T> tmp1, Vc::Vector<T> tmp2, Vc::Vector<T> tmp3,
            Vc::Vector<T> tmp4, Vc::Vector<T> tmp5, Vc::Vector<T> tmp6, Vc::Vector<T> tmp7) {
        _keepXRegister(tmp0.data(), tmp1.data(), tmp2.data(), tmp3.data(), tmp4.data(), tmp5.data(), tmp6.data(), tmp7.data());
    }
};
template<typename T> struct KeepResultsHelper<Vc::Vector<T>, 32> {
    static inline void keepDirty(Vc::Vector<T> &tmp0) {
        asm volatile("":"+x"(tmp0.data()[0]), "+x"(tmp0.data()[1]));
    }
    static inline void keep(Vc::Vector<T> tmp0, Vc::Vector<T> tmp1, Vc::Vector<T> tmp2, Vc::Vector<T> tmp3,
            Vc::Vector<T> tmp4, Vc::Vector<T> tmp5, Vc::Vector<T> tmp6, Vc::Vector<T> tmp7) {
        _keepXRegister(tmp0.data()[0], tmp0.data()[1], tmp1.data()[0], tmp1.data()[1], tmp2.data()[0], tmp2.data()[1], tmp3.data()[0], tmp3.data()[1]);
        _keepXRegister(tmp4.data()[0], tmp4.data()[1], tmp5.data()[0], tmp5.data()[1], tmp6.data()[0], tmp6.data()[1], tmp7.data()[0], tmp7.data()[1]);
    }
};
template<unsigned int S> struct KeepResultsHelper<Vc::SSE::Mask<S>, 16> {
    static inline void keepDirty(Vc::SSE::Mask<S> &tmp0) { asm volatile("":"+x"(tmp0.data())); }
    static inline void keep(Vc::SSE::Mask<S> tmp0, Vc::SSE::Mask<S> tmp1, Vc::SSE::Mask<S> tmp2, Vc::SSE::Mask<S> tmp3,
            Vc::SSE::Mask<S> tmp4, Vc::SSE::Mask<S> tmp5, Vc::SSE::Mask<S> tmp6, Vc::SSE::Mask<S> tmp7) {
        _keepXRegister(tmp0.data(), tmp1.data(), tmp2.data(), tmp3.data(), tmp4.data(), tmp5.data(), tmp6.data(), tmp7.data());
    }
};
template<> struct KeepResultsHelper<Vc::SSE::Float8Mask, 32> {
    static inline void keep(Vc::SSE::Float8Mask tmp0, Vc::SSE::Float8Mask tmp1, Vc::SSE::Float8Mask tmp2, Vc::SSE::Float8Mask tmp3,
            Vc::SSE::Float8Mask tmp4, Vc::SSE::Float8Mask tmp5, Vc::SSE::Float8Mask tmp6, Vc::SSE::Float8Mask tmp7) {
        _keepXRegister(tmp0.data()[0], tmp0.data()[1], tmp1.data()[0], tmp1.data()[1], tmp2.data()[0], tmp2.data()[1], tmp3.data()[0], tmp3.data()[1]);
        _keepXRegister(tmp4.data()[0], tmp4.data()[1], tmp5.data()[0], tmp5.data()[1], tmp6.data()[0], tmp6.data()[1], tmp7.data()[0], tmp7.data()[1]);
    }
};
#elif defined(VC_IMPL_LRBni) && defined(VC_LRBni_PROTOTYPE_H)
#include <xmmintrin.h>
template<typename T> struct KeepResultsHelper<Vc::Vector<T>, 64> {
    static inline void keepDirty(Vc::Vector<T> &tmp0) {
        __m128 *x = reinterpret_cast<__m128 *>(&tmp0);
        asm volatile("":"+x"(x[0]), "+x"(x[1]), "+x"(x[2]), "+x"(x[3]));
    }
    static inline void keep(Vc::Vector<T> tmp0, Vc::Vector<T> tmp1, Vc::Vector<T> tmp2, Vc::Vector<T> tmp3,
            Vc::Vector<T> tmp4, Vc::Vector<T> tmp5, Vc::Vector<T> tmp6, Vc::Vector<T> tmp7) {
        // doesn't make so much sense with so many required registers
        __m128 *x0 = reinterpret_cast<__m128 *>(&tmp0);
        __m128 *x1 = reinterpret_cast<__m128 *>(&tmp1);
        __m128 *x2 = reinterpret_cast<__m128 *>(&tmp2);
        __m128 *x3 = reinterpret_cast<__m128 *>(&tmp3);
        _keepXRegister(x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]);
        _keepXRegister(x2[0], x2[1], x2[2], x2[3], x3[0], x3[1], x3[2], x3[3]);
        __m128 *x4 = reinterpret_cast<__m128 *>(&tmp4);
        __m128 *x5 = reinterpret_cast<__m128 *>(&tmp5);
        __m128 *x6 = reinterpret_cast<__m128 *>(&tmp6);
        __m128 *x7 = reinterpret_cast<__m128 *>(&tmp7);
        _keepXRegister(x4[0], x4[1], x4[2], x4[3], x5[0], x5[1], x5[2], x5[3]);
        _keepXRegister(x6[0], x6[1], x6[2], x6[3], x7[0], x7[1], x7[2], x7[3]);
    }
};
#endif

template<typename T> static inline void keepResultsDirty(T &tmp0)
{
    KeepResultsHelper<T, sizeof(T)>::keepDirty(tmp0);
}

template<typename T> static inline void keepResults(T tmp0)
{
    KeepResultsHelper<T, sizeof(T)>::keep(tmp0, tmp0, tmp0, tmp0, tmp0, tmp0, tmp0, tmp0);
}

template<typename T> static inline void keepResults(T tmp0, T tmp1)
{
    KeepResultsHelper<T, sizeof(T)>::keep(tmp0, tmp1, tmp0, tmp1, tmp0, tmp1, tmp0, tmp1);
}

template<typename T> static inline void keepResults(T tmp0, T tmp1, T tmp2, T tmp3)
{
    KeepResultsHelper<T, sizeof(T)>::keep(tmp0, tmp1, tmp2, tmp3, tmp0, tmp1, tmp2, tmp3);
}

template<typename T> static inline void keepResults(T tmp0, T tmp1, T tmp2, T tmp3, T tmp4, T tmp5)
{
    KeepResultsHelper<T, sizeof(T)>::keep(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp0, tmp1);
}

template<typename T> static inline void keepResults(T tmp0, T tmp1, T tmp2, T tmp3, T tmp4, T tmp5, T tmp6, T tmp7)
{
    KeepResultsHelper<T, sizeof(T)>::keep(tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7);
}

#endif // BENCHMARK_H
