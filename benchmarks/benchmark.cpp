/*  This file is part of the Vc library.

    Copyright (C) 2009-2011 Matthias Kretz <kretz@kde.org>

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

#include "benchmark.h"
#include <Vc/Vc>
#include "../common/support.h"
#include <map>
#include <set>

// limit to max. 10s per single benchmark
static double g_Time = 10.;
static int g_skip = 0;

static std::set<std::string> g_skipReasons;
static std::map<std::string, std::set<std::string> > g_skipLists;

const char *printHelp2 =
"  -t <seconds>        maximum time to run a single benchmark (10s)\n"
"  -cpu (all|any|<id>) CPU to pin the benchmark to\n"
"                      all: test every CPU id in sequence\n"
"                      any: don't pin and let the OS schedule\n"
"                      <id>: pin to the specific CPU\n";

void Benchmark::addColumn(const std::string &name)
{
    if (s_fileWriter) {
        s_fileWriter->addColumn(name);
    }
}

void Benchmark::setColumnData(const std::string &name, const std::string &data)
{
    if (g_skip && g_skipReasons.find(name) != g_skipReasons.end()) {
        //std::cerr << "skip reason was: " << name << std::endl;
        g_skipReasons.erase(name);
        --g_skip;
    }
    std::set<std::string> set = g_skipLists[name];
    if (set.find(data) != set.end()) {
        g_skipReasons.insert(name);
        //std::cerr << "skip reason now is: " << name << std::endl;
        ++g_skip;
    }
    if (s_fileWriter) {
        s_fileWriter->setColumnData(name, data);
    } else {
        std::cout << "Benchmarking " << name << " " << data << std::endl;
    }
}

void Benchmark::finalize()
{
    if (s_fileWriter) {
        s_fileWriter->finalize();
    }
}

void Benchmark::changeInterpretation(double factor, const char *X)
{
    fFactor = factor;
    fX = X;
}

const char Benchmark::greenEsc  [8] = "\033[1;32m";
const char Benchmark::cyanEsc   [8] = "\033[1;36m";
const char Benchmark::reverseEsc[5] = "\033[7m";
const char Benchmark::normalEsc [5] = "\033[0m";

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
    if (m_header != header) {
        if (m_header.empty()) {
            m_file << "Version 4\n";
        }
        m_header = header;
        m_file << "\"benchmark.name\"\t\"benchmark.arch\"";
        for (std::list<ExtraColumn>::const_iterator i = m_extraColumns.begin();
                i != m_extraColumns.end(); ++i) {
            m_file << "\t\"" << i->name << '"';
        }
        for (std::list<std::string>::const_iterator i = header.begin();
                i != header.end(); ++i) {
            m_file << '\t' << *i;
        }
        m_file << "\n";
    }
}

void Benchmark::FileWriter::addDataLine(const std::list<std::string> &data)
{
    m_file << m_currentName << '\t' <<
#if VC_IMPL_AVX
            "\"AVX\"";
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
    : fName(_name), fFactor(factor), fX(X), m_dataPointsCount(0), m_skip(g_skip)
{
    if (m_skip) {
        return;
    }
    std::set<std::string> set = g_skipLists["benchmark.name"];
    if (set.find(_name) != set.end()) {
        m_skip = true;
        return;
    }
    for (int i = 0; i < 3; ++i) {
        m_mean[i] = m_stddev[i] = 0.;
    }
    enum {
        WCHARSIZE = sizeof("━") - 1
    };
    if (!s_fileWriter) {
        const bool interpret = (fFactor != 0.);
        char header[128 * WCHARSIZE + sizeof(reverseEsc) * 2];
        std::memset(header, 0, 128 * WCHARSIZE + sizeof(reverseEsc) * 2);
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
        int offset = (headerLen - titleLen - 1) / 2;
        if (offset >= 0) {
            std::string name = ' ' + fName + ' ';
            header[offset * WCHARSIZE] = '\0';
            std::cout << header << reverseEsc << name << normalEsc << &header[(offset + name.length()) * WCHARSIZE] << std::flush;
//X             std::memcpy(ptr, name.c_str(), name.length());
//X             std::memmove(ptr + name.length(), ptr + name.length() * WCHARSIZE, (headerLen - offset - name.length()) * WCHARSIZE + 1);
//X             std::cout << header << std::flush;
        } else {
            std::cout << fName << std::flush;
        }
    }
}

bool Benchmark::wantsMoreDataPoints() const
{
    if (m_skip) {
        return false;
    } else if (m_dataPointsCount < 3) { // hard limit on the number of data points; otherwise talking about stddev is bogus
        return true;
    } else if (m_mean[0] > g_Time) { // limit on the time
        return false;
    } else if (m_dataPointsCount < 30) { // we want initial statistics
        return true;
    }
    return m_stddev[0] * m_dataPointsCount > 1.0004 * m_mean[0] * m_mean[0]; // stop if the relative error is below 2% already
}

void Benchmark::Mark()
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

bool Benchmark::Print()
{
    if (m_skip) {
        return false;
    }
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
        std::cout << centered(fX + "s/s [CPU]")  << "┃";
#endif
        std::cout << centered(fX + "s/s [Real]") << "┃";
        std::cout << centered(fX + "s/Cycle")    << "┃";
        std::cout << centered("Cycles/" + fX)   << "┃";
        std::string X = fX;
        for (unsigned int i = 0; i < X.length(); ++i) {
            if (X[i] == ' ') {
                X[i] = '_';
            }
        }
        header
            << X + "s/Real_time" << X + "s/Real_time_stddev"
            << X + "s/Cycle" << X + "s/Cycle_stddev"
#ifdef VC_USE_CPU_TIME
            << X + "s/CPU_time" << X + "s/CPU_time_stddev"
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
        std::cout << greenEsc;
        prettyPrintCount(fFactor / m_mean[1]);
        std::cout << normalEsc;
        std::cout << " ┃ ";
        std::cout << cyanEsc;
        prettyPrintCount(m_mean[1] / fFactor);
        std::cout << normalEsc;
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
    return false;
}

void printHelp(const char *name) {
    std::cout << "Usage " << name << " [OPTION]...\n"
        << "Measure throughput and latency of memory in steps of 1GB\n\n"
        << "  -h, --help          print this message\n"
        << "  -o <filename>       output measurements to a file instead of stdout\n"
        << "  --skip <name> <value>  skip tests with the name/column set to the given value\n"
        ;
    if (printHelp2) {
        std::cout << printHelp2;
    }
    std::cout << "\nReport bugs to vc-devel@compeng.uni-frankfurt.de\n"
        << "Vc Homepage: http://compeng.uni-frankfurt.de/index.php?id=Vc\n"
        << std::flush;
}

typedef std::vector<std::string> ArgumentVector;
ArgumentVector g_arguments;

#include "cpuset.h"

int main(int argc, char **argv)
{
    if (!Vc::currentImplementationSupported()) {
        std::cerr << "CPU or OS requirements not met for the compiled in vector unit!\n";
        return -1;
    }

#ifdef SCHED_FIFO_BENCHMARKS
    if (SCHED_FIFO != sched_getscheduler(0)) {
        char *path = reinterpret_cast<char *>(malloc(strlen(argv[0] + sizeof("rtwrapper"))));
        strcpy(path, argv[0]);
        char *slash = strrchr(path, '/');
        if (slash) {
            slash[1] = '\0';
        }
        strcat(path, "rtwrapper");
        // not realtime priority, check whether the benchmark executable exists
        execv(path, argv);
        free(path);
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
            i += 2;
        } else if (std::strcmp(argv[i - 1], "-t") == 0) {
            g_Time = atof(argv[i]);
            i += 2;
        } else if (std::strcmp(argv[i - 1], "-cpu") == 0) {
// On OS X there is no way to set CPU affinity
// TODO there is a way to ask the system to not move the process around
#if !defined __APPLE__ && !defined _WIN32 && !defined _WIN64
            if (std::strcmp(argv[i], "all") == 0) {
                useCpus = UseAllCpus;
            } else if (std::strcmp(argv[i], "any") == 0) {
                useCpus = UseAnyOneCpu;
            } else {
                useCpus = atoi(argv[i]);
            }
#endif
            i += 2;
        } else if (std::strcmp(argv[i - 1], "--help") == 0 ||
                    std::strcmp(argv[i - 1], "-help") == 0 ||
                    std::strcmp(argv[i - 1], "-h") == 0) {
            printHelp(argv[0]);
            return 0;
        } else if (std::strcmp(argv[i - 1], "--skip") == 0) {
            const std::string name(argv[i]);
            const std::string value(argv[i + 1]);
            g_skipLists[name].insert(value);
            i += 3;
        } else {
            g_arguments.push_back(argv[i - 1]);
            ++i;
        }
    }
    if (argc == i) {
        if (std::strcmp(argv[i - 1], "--help") == 0 ||
                std::strcmp(argv[i - 1], "-help") == 0 ||
                std::strcmp(argv[i - 1], "-h") == 0) {
            printHelp(argv[0]);
            return 0;
        }
        g_arguments.push_back(argv[i - 1]);
    }

    int r = 0;
    if (useCpus == UseAnyOneCpu) {
        r += bmain();
        Benchmark::finalize();
#if !defined _WIN32 && !defined _WIN64
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
#endif
    }
    delete file;
    return r;
}

#ifndef __GNUC__
template<> int KeepResultsHelper<int, sizeof(int)>::blackHole[8];
template<> unsigned int KeepResultsHelper<unsigned int, sizeof(unsigned int)>::blackHole[8];
template<> short KeepResultsHelper<short, sizeof(short)>::blackHole[8];
template<> unsigned short KeepResultsHelper<unsigned short, sizeof(unsigned short)>::blackHole[8];
template<> float KeepResultsHelper<float, sizeof(float)>::blackHole[8];
template<> double KeepResultsHelper<double, sizeof(double)>::blackHole[8];

template<> Vc::Vector<int> KeepResultsHelper<Vc::Vector<int>, sizeof(Vc::Vector<int>)>::blackHole[8];
template<> Vc::Vector<unsigned int> KeepResultsHelper<Vc::Vector<unsigned int>,
    sizeof(Vc::Vector<unsigned int>)>::blackHole[8];
template<> Vc::Vector<short> KeepResultsHelper<Vc::Vector<short>, sizeof(Vc::Vector<short>)>::blackHole[8];
template<> Vc::Vector<unsigned short> KeepResultsHelper<Vc::Vector<unsigned short>,
    sizeof(Vc::Vector<unsigned short>)>::blackHole[8];
template<> Vc::Vector<float> KeepResultsHelper<Vc::Vector<float>, sizeof(Vc::Vector<float>)>::blackHole[8];
template<> Vc::Vector<double> KeepResultsHelper<Vc::Vector<double>, sizeof(Vc::Vector<double>)>::blackHole[8];

#ifdef VC_IMPL_Scalar
template<> Vc::Scalar::Mask<1> KeepResultsHelper<Vc::Scalar::Mask<1>, sizeof(Vc::Scalar::Mask<1>)>::blackHole[8];
#elif defined VC_IMPL_SSE
template<> Vc::SSE::Vector<Vc::SSE::float8>  KeepResultsHelper<Vc::SSE::Vector<Vc::SSE::float8>,
    sizeof(Vc::SSE::Vector<Vc::SSE::float8>)>::blackHole[8];
template<> Vc::SSE::Mask<2>  KeepResultsHelper<Vc::SSE::Mask<2>,  sizeof(Vc::SSE::Mask<2>)>::blackHole[8];
template<> Vc::SSE::Mask<4>  KeepResultsHelper<Vc::SSE::Mask<4>,  sizeof(Vc::SSE::Mask<4>)>::blackHole[8];
template<> Vc::SSE::Mask<8>  KeepResultsHelper<Vc::SSE::Mask<8>,  sizeof(Vc::SSE::Mask<8>)>::blackHole[8];
template<> Vc::SSE::Mask<16> KeepResultsHelper<Vc::SSE::Mask<16>, sizeof(Vc::SSE::Mask<16>)>::blackHole[8];
Vc::SSE::Float8Mask KeepResultsHelper<Vc::SSE::Float8Mask, sizeof(Vc::SSE::Float8Mask)>::blackHole[8];
#endif

template<> float const * KeepResultsHelper<float const *, sizeof(float *)>::blackHole[8];
template<> short const * KeepResultsHelper<short const *, sizeof(short *)>::blackHole[8];

#endif

// vim: sw=4 sts=4 et tw=100
