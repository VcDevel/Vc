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

#ifndef WIN32
#define VC_USE_CPU_TIME
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
            int m_line;
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
    enum OutputMode {
        DataFile,
        Stdout
    };

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
        unsigned long long fCycles;
    };
    const std::string fName;
    const double fFactor;
    const std::string fX;
#ifdef _MSC_VER
    __int64 fRealTime;
#else
#ifdef __APPLE__
    uint64_t fRealTime;
#else
    struct timespec fRealTime;
    struct timespec fCpuTime;
#endif
#endif
    TimeStampCounter fTsc;
    std::list<DataPoint> fDataPoints;
    static FileWriter *s_fileWriter;
};

Benchmark::FileWriter::FileWriter(const std::string &filename)
    : m_line(0), m_finalized(false)
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
        m_file << "Version 2\n"
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
    m_file << ++m_line << '\t' << m_currentName << '\t' <<
#if VC_IMPL_LRBni
#ifdef __LRB__
            "\"LRB\"";
#else
            "\"LRB Prototype\"";
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
#error "Unknown Vc implementation"
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

Benchmark::Benchmark(const std::string &name, double factor, const std::string &X)
    : fName(name), fFactor(factor), fX(X)
{
    if (!s_fileWriter) {
        const bool interpret = (fFactor != 0.);
        char header[128];
        std::memset(header, 0, 128);
        std::strcpy(header,
#ifdef VC_USE_CPU_TIME
                "+----------------+----------------"
#endif
                "+----------------+----------------+----------------+----------------+----------------+");
        if (!interpret) {
#ifdef VC_USE_CPU_TIME
            header[69] = '\0';
#else
            header[69 - 17] = '\0';
#endif
        }
        const int titleLen = fName.length();
        const int headerLen = std::strlen(header);
        const int offset = (headerLen - titleLen) / 2;
        if (offset > 0) {
            std::memcpy(&header[offset], fName.c_str(), titleLen);
            header[offset - 1] = ' ';
            header[offset + titleLen] = ' ';
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
#else
#ifdef __APPLE__
    fRealTime = mach_absolute_time();
#else
    clock_gettime( CLOCK_MONOTONIC, &fRealTime );
    clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &fCpuTime );
#endif
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

inline void Benchmark::Stop()
{
    fTsc.Stop();
#ifdef _MSC_VER
    __int64 real = 0, freq = 0;
    QueryPerformanceCounter((LARGE_INTEGER *)&real);
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    const DataPoint p = {
        static_cast<double>(real - fRealTime) / freq,
        1.0,
#else
#ifdef __APPLE__
    uint64_t real = mach_absolute_time();
    static mach_timebase_info_data_t info = {0,0};  
    
    if (info.denom == 0)  
    	mach_timebase_info(&info);  
    
    uint64_t nanos = (real - fRealTime ) * (info.numer / info.denom);
    const DataPoint p = {
        nanos * 1e-9,
        1.0,
#else
    struct timespec real, cpu;
    clock_gettime( CLOCK_MONOTONIC, &real );
    clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &cpu );

    const DataPoint p = {
        convertTimeSpec(real) - convertTimeSpec(fRealTime),
        convertTimeSpec(cpu ) - convertTimeSpec(fCpuTime ),
#endif
#endif
        fTsc.Cycles()
    };
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

inline void Benchmark::Print(int f) const
{
    std::streambuf *backup = std::cout.rdbuf();
    if (s_fileWriter) {
        std::cout.rdbuf(0);
    }
    typedef std::list<DataPoint>::const_iterator It;
    const bool interpret = (fFactor != 0.);

    std::list<std::string> header;
#ifdef VC_USE_CPU_TIME
    header << "CPU_time";
#endif
    header << "Real_time" << "Cycles";

    std::cout << "\n"
#ifdef VC_USE_CPU_TIME
        << "|    CPU time    "
#endif
        << "|   Real time    |     Cycles     |";
    if (interpret) {
#ifdef VC_USE_CPU_TIME
        std::cout << centered(fX + "/s [CPU]")  << "|";
#endif
        std::cout << centered(fX + "/s [Real]") << "|";
        std::cout << centered(fX + "/cycle")    << "|";
        std::cout << centered("cycles/" + fX)   << "|";
        std::string X = fX;
        for (unsigned int i = 0; i < X.length(); ++i) {
            if (X[i] == ' ') {
                X[i] = '_';
            }
        }
        header << "number_of_" + X;
    }
    if (s_fileWriter) {
        s_fileWriter->declareData(fName, header);
    }

    std::list<std::string> dataLine;
    for (It i = fDataPoints.begin(); i != fDataPoints.end(); ++i) {
        dataLine.clear();
        std::cout << "\n| ";
#ifdef VC_USE_CPU_TIME
        prettyPrintSeconds(i->fCpuElapsed);
        std::cout << " | ";
#endif
        prettyPrintSeconds(i->fRealElapsed);
        std::cout << " | ";
        prettyPrintCount(static_cast<double>(i->fCycles));
        std::cout << " | ";
        dataLine
#ifdef VC_USE_CPU_TIME
            << i->fCpuElapsed
#endif
            << i->fRealElapsed << i->fCycles;
        if (interpret) {
#ifdef VC_USE_CPU_TIME
            prettyPrintCount(fFactor / i->fCpuElapsed);
            std::cout << " | ";
#endif
            prettyPrintCount(fFactor / i->fRealElapsed);
            std::cout << " | ";
            prettyPrintCount(fFactor / i->fCycles);
            std::cout << " | ";
            prettyPrintCount(i->fCycles / fFactor);
            std::cout << " | ";
            dataLine << fFactor;
        }
        if (s_fileWriter) {
            s_fileWriter->addDataLine(dataLine);
        }
    }
    if (f & PrintAverage) {
        if (interpret) {
#ifdef VC_USE_CPU_TIME
            std::cout << "\n|---------------------------------------------- Average ---------------------------------------------------------------|";
#else
            std::cout << "\n|----------------------------- Average ----------------------------------------------|";
#endif
        } else {
#ifdef VC_USE_CPU_TIME
            std::cout << "\n|-------------------- Average ---------------------|";
#else
            std::cout << "\n|------------ Average ------------|";
#endif
        }

        double cpuAvg = 0.;
        double realAvg = 0.;
        double cycleAvg = 0.;

        for (It i = fDataPoints.begin(); i != fDataPoints.end(); ++i) {
            cpuAvg += i->fCpuElapsed;
            realAvg += i->fRealElapsed;
            cycleAvg += i->fCycles;
        }
        const double count = static_cast<double>(fDataPoints.size());
        cpuAvg /= count;
        realAvg /= count;
        cycleAvg /= count;

        std::cout << "\n| ";
#ifdef VC_USE_CPU_TIME
        prettyPrintSeconds(cpuAvg);
        std::cout << " | ";
#endif
        prettyPrintSeconds(realAvg);
        std::cout << " | ";
        prettyPrintCount(cycleAvg);
        std::cout << " | ";
        if (interpret) {
#ifdef VC_USE_CPU_TIME
            prettyPrintCount(fFactor / cpuAvg);
            std::cout << " | ";
#endif
            prettyPrintCount(fFactor / realAvg);
            std::cout << " | ";
            prettyPrintCount(fFactor / cycleAvg);
            std::cout << " | ";
            prettyPrintCount(cycleAvg / fFactor);
            std::cout << " | ";
        }
    }
    std::cout << "\n"
#ifdef VC_USE_CPU_TIME
        "+----------------"
#endif
        "+----------------+----------------+"
        << (interpret ?
#ifdef VC_USE_CPU_TIME
                "----------------+"
#endif
                "----------------+----------------+----------------+" : "") << std::endl;
    if (s_fileWriter) {
        std::cout.rdbuf(backup);
    }
}

int bmain(Benchmark::OutputMode);

static int g_Repetitions = 0;

#include <sched.h>

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
    Benchmark::OutputMode outputMode = Benchmark::Stdout;
    Benchmark::FileWriter *file = 0;
    while (argc > i) {
        if (std::strcmp(argv[i - 1], "-o") == 0) {
            file = new Benchmark::FileWriter(argv[i]);
            outputMode = Benchmark::DataFile;
        } else if (std::strcmp(argv[i - 1], "-r") == 0) {
            g_Repetitions = atoi(argv[i]);
        }
        i += 2;
    }

    int r = 0;
#ifdef __APPLE__
    // On OS X there is no way to set CPU affinity
    // TODO there is a way to ask the system to not move the process around
    r += bmain(outputMode);
    Benchmark::finalize();
#else
    cpu_set_t cpumask;
    sched_getaffinity(0, sizeof(cpu_set_t), &cpumask);
    int cpucount = 1;
    while (CPU_ISSET(cpucount, &cpumask)) {
        ++cpucount;
    }
    for (int cpuid = 0; cpuid < cpucount; ++cpuid) {
        if (cpucount > 1) {
            Benchmark::addColumn("CPU_ID");
            std::ostringstream str;
            str << cpuid;
            Benchmark::setColumnData("CPU_ID", str.str());
        }
        CPU_ZERO(&cpumask);
        CPU_SET(cpuid, &cpumask);
        sched_setaffinity(0, sizeof(cpu_set_t), &cpumask);
        r += bmain(outputMode);
        Benchmark::finalize();
    }
#endif
    delete file;
    return r;
}

#endif // BENCHMARK_H
