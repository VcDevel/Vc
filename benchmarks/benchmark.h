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

// limit to max. 10s per single benchmark
static double g_Time = 10.;

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
    static inline void addColumn(const std::string &name) { if (s_fileWriter) s_fileWriter->addColumn(name); }
    static inline void setColumnData(const std::string &name, const std::string &data) {
        if (s_fileWriter) {
            s_fileWriter->setColumnData(name, data);
        } else {
            std::cout << "Benchmarking " << name << " " << data << std::endl;
        }
    }
    static inline void finalize() { if (s_fileWriter) s_fileWriter->finalize(); }

    explicit Benchmark(const std::string &name, double factor = 0., const std::string &X = std::string());

    bool wantsMoreDataPoints() const;
    void Start();
    void Mark();
    void Stop();
    void Print();

private:
    void printMiddleLine() const;
    void printBottomLine() const;

    const std::string fName;
    const double fFactor;
    const std::string fX;
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
};

Benchmark::FileWriter::FileWriter(const std::string &filename)
    : m_finalized(false)
{
    std::string fn = filename;
    m_file.open(fn.c_str());

    if (Benchmark::s_fileWriter == 0) {
        Benchmark::s_fileWriter = this;
    }
}

Benchmark::FileWriter::~FileWriter()
{
    if (m_file.is_open()) {
        m_file.flush();
        m_file.close();
    }
    if (Benchmark::s_fileWriter == this) {
        Benchmark::s_fileWriter = 0;
    }
}

void Benchmark::FileWriter::declareData(const std::string &name, const std::list<std::string> &header)
{
    m_currentName = '"' + name + '"';
    if (m_header.empty()) {
        m_header = header;
        m_file << "Version 3\n"
            << "\"benchmark.name\"\t\"benchmark.arch\"";
        for (std::list<ExtraColumn>::const_iterator i = m_extraColumns.begin();
                i != m_extraColumns.end(); ++i) {
            m_file << "\t\"" << i->name << '"';
        }
        for (std::list<std::string>::const_iterator i = header.begin();
                i != header.end(); ++i) {
            m_file << '\t' << *i;
        }
        m_file << "\n";
    } else if (m_header != header) {
        std::cerr << "incompatible writes to FileWriter!\n"
            << std::endl;
    }
}

void Benchmark::FileWriter::addDataLine(const std::list<std::string> &data)
{
    m_file << m_currentName << '\t' <<
#if VC_IMPL_LRBni
#ifdef VC_LRBni_PROTOTYPE_H
            "\"LRB Prototype\"";
#else
            "\"LRB\"";
#endif
#elif VC_IMPL_SSE4_1
#ifdef VC_DISABLE_PTEST
            "\"SSE4.1 w/o PTEST\"";
#else
            "\"SSE4.1\"";
#endif
#elif VC_IMPL_SSSE3
            "\"SSSE3\"";
#elif VC_IMPL_SSE3
            "\"SSE3\"";
#elif VC_IMPL_SSE2
            "\"SSE2\"";
#elif VC_IMPL_Scalar
            "\"Scalar\"";
#else
            "\"non-Vc\"";
#endif
    for (std::list<ExtraColumn>::const_iterator i = m_extraColumns.begin();
            i != m_extraColumns.end(); ++i) {
        m_file << '\t' << i->data;
    }
    for (std::list<std::string>::const_iterator i = data.begin();
            i != data.end(); ++i) {
        m_file << '\t' << *i;
    }
    m_file << "\n";
}

void Benchmark::FileWriter::addColumn(const std::string &name)
{
    if (!m_finalized) {
        if (m_header.empty()) {
            if (m_extraColumns.end() == std::find(m_extraColumns.begin(), m_extraColumns.end(), name)) {
                m_extraColumns.push_back(name);
            }
        } else {
            std::cerr << "call addColumn before the first benchmark prints its data" << std::endl;
        }
    }
}

void Benchmark::FileWriter::setColumnData(const std::string &name, const std::string &data)
{
    for (std::list<ExtraColumn>::iterator i = m_extraColumns.begin();
            i != m_extraColumns.end(); ++i) {
        if (*i == name) {
            i->data = '"' + data + '"';
            break;
        }
    }
}

Benchmark::FileWriter *Benchmark::s_fileWriter = 0;

Benchmark::Benchmark(const std::string &_name, double factor, const std::string &X)
    : fName(_name), fFactor(factor), fX(X), m_dataPointsCount(0)
{
    for (int i = 0; i < 3; ++i) {
        m_mean[i] = m_stddev[i] = 0.;
    }
    enum {
        WCHARSIZE = sizeof("━") - 1
    };
    if (!s_fileWriter) {
        const bool interpret = (fFactor != 0.);
        char header[128 * WCHARSIZE];
        std::memset(header, 0, 128 * WCHARSIZE);
        std::strcpy(header,
#ifdef VC_USE_CPU_TIME
                "┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━"
#endif
                "┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓");
        if (!interpret) {
#ifdef VC_USE_CPU_TIME
            header[69 * WCHARSIZE] = '\0';
#else
            header[(69 - 17) * WCHARSIZE] = '\0';
#endif
        }
        const int titleLen = fName.length();
        const int headerLen = std::strlen(header) / WCHARSIZE;
        int offset = (headerLen - titleLen) / 2;
        if (offset > 0) {
            --offset;
            std::string name = ' ' + fName + ' ';
            char *ptr = &header[offset * WCHARSIZE];
            std::memcpy(ptr, name.c_str(), name.length());
            std::memmove(ptr + name.length(), ptr + name.length() * WCHARSIZE, (headerLen - offset - name.length()) * WCHARSIZE + 1);
            std::cout << header << std::flush;
        } else {
            std::cout << fName << std::flush;
        }
    }
}

inline void Benchmark::Start()
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
}

#ifndef _MSC_VER
static inline double convertTimeSpec(const struct timespec &ts)
{
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
}
#endif

static const double SECONDS_PER_CLOCK = 1. / CLOCKS_PER_SEC;

bool Benchmark::wantsMoreDataPoints() const
{
    if (m_dataPointsCount < 3) { // hard limit on the number of data points; otherwise talking about stddev is bogus
        return true;
    } else if (m_mean[0] > g_Time) { // limit on the time
        return false;
    } else if (m_dataPointsCount < 30) { // we want initial statistics
        return true;
    }
    return m_stddev[0] * m_dataPointsCount > 1.0004 * m_mean[0] * m_mean[0]; // stop if the relative error is below 2% already
}

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
    } else if (v < 2.) {
        int i = 0;
        do {
            v *= 1000.;
            ++i;
        } while (v < 1.);
        std::cout << std::setw(11) << v << ' ' << prefix[i] << 's';
    } else if (v > 60.) {
        std::cout << std::setw(10) << v / 60. << " min";
    } else {
        std::cout << std::setw(12) << v << " s";
    }
}

#ifdef isfinite
#undef isfinite
#endif

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
        if (v < 1000.) {
            std::cout << std::setw(14) << v;
            return;
        }
        while (v >= 1000.) {
            v *= 0.001;
            ++i;
        }
    }
    std::cout << std::setw(12) << v << ' ' << prefix[i];
}

static inline void prettyPrintError(double v)
{
    std::stringstream ss;
    ss << "± " << v << " %";
    std::cout << std::setw(15) << ss.str();
}

static inline std::list<std::string> &operator<<(std::list<std::string> &list, const std::string &data)
{
    std::ostringstream str;
    str << '"' << data << '"';
    list.push_back(str.str());
    return list;
}

static inline std::list<std::string> &operator<<(std::list<std::string> &list, const char *data)
{
    std::ostringstream str;
    str << '"' << data << '"';
    list.push_back(str.str());
    return list;
}

static inline std::list<std::string> &operator<<(std::list<std::string> &list, double data)
{
    std::ostringstream str;
    str << data;
    list.push_back(str.str());
    return list;
}

static inline std::list<std::string> &operator<<(std::list<std::string> &list, int data)
{
    std::ostringstream str;
    str << data;
    list.push_back(str.str());
    return list;
}

static inline std::list<std::string> &operator<<(std::list<std::string> &list, unsigned int data)
{
    std::ostringstream str;
    str << data;
    list.push_back(str.str());
    return list;
}

static inline std::list<std::string> &operator<<(std::list<std::string> &list, unsigned long long data)
{
    std::ostringstream str;
    str << data;
    list.push_back(str.str());
    return list;
}

static std::string centered(const std::string &s, const int size = 16)
{
    const int missing = size - s.length();
    if (missing < 0) {
        return s.substr(0, size);
    } else if (missing == 0) {
        return s;
    }
    const int left = missing - missing / 2;
    std::string r(size, ' ');
    r.replace(left, s.length(), s);
    return r;
}

inline void Benchmark::printMiddleLine() const
{
    const bool interpret = (fFactor != 0.);
    std::cout << "\n"
#ifdef VC_USE_CPU_TIME
        "┠────────────────"
#endif
        "┠────────────────╂────────────────"
        << (interpret ?
#ifdef VC_USE_CPU_TIME
                "╂────────────────"
#endif
                "╂────────────────╂────────────────╂────────────────┨" : "┨");
}
inline void Benchmark::printBottomLine() const
{
    const bool interpret = (fFactor != 0.);
    std::cout << "\n"
#ifdef VC_USE_CPU_TIME
        "┗━━━━━━━━━━━━━━━━"
#endif
        "┗━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━"
        << (interpret ?
#ifdef VC_USE_CPU_TIME
                "┻━━━━━━━━━━━━━━━━"
#endif
                "┻━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━┛" : "┛") << std::endl;
}

inline void Benchmark::Print()
{
    std::streambuf *backup = std::cout.rdbuf();
    if (s_fileWriter) {
        std::cout.rdbuf(0);
    }
    const bool interpret = (fFactor != 0.);

    std::list<std::string> header;
    header
        << "Real_time" << "Real_time_stddev"
        << "Cycles" << "Cycles_stddev"
#ifdef VC_USE_CPU_TIME
        << "CPU_time" << "CPU_time_stddev"
#endif
    ;

    // ┃ ━ ┏ ┓ ┗ ┛ ┣ ┫ ┳ ┻ ╋ ┠ ─ ╂ ┨
    std::cout << "\n"
#ifdef VC_USE_CPU_TIME
        << "┃    CPU time    "
#endif
        << "┃   Real time    ┃     Cycles     ┃";
    if (interpret) {
#ifdef VC_USE_CPU_TIME
        std::cout << centered(fX + "/s [CPU]")  << "┃";
#endif
        std::cout << centered(fX + "/s [Real]") << "┃";
        std::cout << centered(fX + "/cycle")    << "┃";
        std::cout << centered("cycles/" + fX)   << "┃";
        std::string X = fX;
        for (unsigned int i = 0; i < X.length(); ++i) {
            if (X[i] == ' ') {
                X[i] = '_';
            }
        }
        header
            << X + "/Real_time" << X + "/Real_time_stddev"
            << X + "/Cycles" << X + "/Cycles_stddev"
#ifdef VC_USE_CPU_TIME
            << X + "/CPU_time" << X + "/CPU_time_stddev"
#endif
            << "number_of_" + X;
    }
    printMiddleLine();
    if (s_fileWriter) {
        s_fileWriter->declareData(fName, header);
    }

    const double normalization = 1. / m_dataPointsCount;

    std::list<std::string> dataLine;
    m_mean[0] *= normalization;
    m_stddev[0] = std::sqrt(m_stddev[0] * normalization - m_mean[0] * m_mean[0]);
    dataLine << m_mean[0] << m_stddev[0];
    m_mean[1] *= normalization;
    m_stddev[1] = std::sqrt(m_stddev[1] * normalization - m_mean[1] * m_mean[1]);
    dataLine << m_mean[1] << m_stddev[1];
#ifdef VC_USE_CPU_TIME
    m_mean[2] *= normalization;
    m_stddev[2] = std::sqrt(m_stddev[2] * normalization - m_mean[2] * m_mean[2]);
    dataLine << m_mean[2] << m_stddev[2];
#endif
    double stddevint[3];
    stddevint[0] = fFactor * m_stddev[0] / (m_mean[0] * m_mean[0]);
    stddevint[1] = fFactor * m_stddev[1] / (m_mean[1] * m_mean[1]);
    dataLine << fFactor / m_mean[0] << stddevint[0];
    dataLine << fFactor / m_mean[1] << stddevint[1];
#ifdef VC_USE_CPU_TIME
    stddevint[2] = fFactor * m_stddev[2] / (m_mean[2] * m_mean[2]);
    dataLine << fFactor / m_mean[2] << stddevint[2];
#endif
    dataLine << fFactor;

    std::cout << "\n┃ ";
#ifdef VC_USE_CPU_TIME
    prettyPrintSeconds(m_mean[2]);
    std::cout << " ┃ ";
#endif
    prettyPrintSeconds(m_mean[0]);
    std::cout << " ┃ ";
    prettyPrintCount(m_mean[1]);
    std::cout << " ┃ ";
    if (interpret) {
#ifdef VC_USE_CPU_TIME
        prettyPrintCount(fFactor / m_mean[2]);
        std::cout << " ┃ ";
#endif
        prettyPrintCount(fFactor / m_mean[0]);
        std::cout << " ┃ ";
        prettyPrintCount(fFactor / m_mean[1]);
        std::cout << " ┃ ";
        prettyPrintCount(m_mean[1] / fFactor);
        std::cout << " ┃ ";
    }
    std::cout << "\n┃ ";
#ifdef VC_USE_CPU_TIME
    prettyPrintError(m_stddev[2] * 100. / m_mean[2]);
    std::cout << " ┃ ";
#endif
    prettyPrintError(m_stddev[0] * 100. / m_mean[0]);
    std::cout << " ┃ ";
    prettyPrintError(m_stddev[1] * 100. / m_mean[1]);
    std::cout << " ┃ ";
    if (interpret) {
#ifdef VC_USE_CPU_TIME
        prettyPrintError(m_stddev[2] * 100. / m_mean[2]);
        std::cout << " ┃ ";
#endif
        prettyPrintError(m_stddev[0] * 100. / m_mean[0]);
        std::cout << " ┃ ";
        prettyPrintError(m_stddev[1] * 100. / m_mean[1]);
        std::cout << " ┃ ";
        prettyPrintError(m_stddev[1] * 100. / m_mean[1]);
        std::cout << " ┃ ";
    }
    printBottomLine();
    if (s_fileWriter) {
        s_fileWriter->addDataLine(dataLine);
        std::cout.rdbuf(backup);
    }
}

int bmain();

#include "cpuset.h"

int main(int argc, char **argv)
{
#ifdef SCHED_FIFO_BENCHMARKS
    if (SCHED_FIFO != sched_getscheduler(0)) {
        // not realtime priority, check whether the benchmark executable exists
        execv("./benchmark", argv);
        // if the execv call works, great. If it doesn't we just continue, but without realtime prio
    }
#endif

    int i = 2;
    Benchmark::FileWriter *file = 0;
    enum {
        UseAllCpus = -2,
        UseAnyOneCpu = -1
    };
    int useCpus = UseAnyOneCpu;
    while (argc > i) {
        if (std::strcmp(argv[i - 1], "-o") == 0) {
            file = new Benchmark::FileWriter(argv[i]);
        } else if (std::strcmp(argv[i - 1], "-t") == 0) {
            g_Time = atof(argv[i]);
        } else if (std::strcmp(argv[i - 1], "-cpu") == 0) {
// On OS X there is no way to set CPU affinity
// TODO there is a way to ask the system to not move the process around
#ifndef __APPLE__
            if (std::strcmp(argv[i], "all") == 0) {
                useCpus = UseAllCpus;
            } else if (std::strcmp(argv[i], "any") == 0) {
                useCpus = UseAnyOneCpu;
            } else {
                useCpus = atoi(argv[i]);
            }
#endif
        }
        i += 2;
    }

    int r = 0;
    if (useCpus == UseAnyOneCpu) {
        r += bmain();
        Benchmark::finalize();
    } else {
        cpu_set_t cpumask;
        sched_getaffinity(0, sizeof(cpu_set_t), &cpumask);
        int cpucount = cpuCount(&cpumask);
        if (cpucount > 1) {
            Benchmark::addColumn("CPU_ID");
        }
        if (useCpus == UseAllCpus) {
            for (int cpuid = 0; cpuid < cpucount; ++cpuid) {
                if (cpucount > 1) {
                    std::ostringstream str;
                    str << cpuid;
                    Benchmark::setColumnData("CPU_ID", str.str());
                }
                cpuZero(&cpumask);
                cpuSet(cpuid, &cpumask);
                sched_setaffinity(0, sizeof(cpu_set_t), &cpumask);
                r += bmain();
                Benchmark::finalize();
            }
        } else {
            int cpuid = std::min(cpucount - 1, std::max(0, useCpus));
            if (cpucount > 1) {
                std::ostringstream str;
                str << cpuid;
                Benchmark::setColumnData("CPU_ID", str.str());
            }
            cpuZero(&cpumask);
            cpuSet(cpuid, &cpumask);
            sched_setaffinity(0, sizeof(cpu_set_t), &cpumask);
            r += bmain();
            Benchmark::finalize();
        }
    }
    delete file;
    return r;
}

#endif // BENCHMARK_H
