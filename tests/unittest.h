/*  This file is part of the Vc library.

    Copyright (C) 2009-2012 Matthias Kretz <kretz@kde.org>

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

#ifndef UNITTEST_H
#define UNITTEST_H

#ifdef VC_ASSERT
#error "include unittest.h before any Vc header"
#endif
static void unittest_assert(bool cond, const char *code, const char *file, int line);
#define VC_ASSERT(cond) unittest_assert(cond, #cond, __FILE__, __LINE__);

#include <Vc/Vc>
#include <Vc/type_traits>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <Vc/support.h>
#include "ulp.h"
#include <typeinfo>
#include <vector>
#include <tuple>
#include <common/macros.h>

#define _expand(name) #name
#define runTest(name) _unit_test_global.runTestInt(&name, _expand(name))
#define testAllTypes(name) \
    _unit_test_global.runTestInt(&name<float_v>, #name "<float_v>"); \
    _unit_test_global.runTestInt(&name<short_v>, #name "<short_v>"); \
    _unit_test_global.runTestInt(&name<ushort_v>, #name "<ushort_v>"); \
    _unit_test_global.runTestInt(&name<int_v>, #name "<int_v>"); \
    _unit_test_global.runTestInt(&name<double_v>, #name "<double_v>"); \
    _unit_test_global.runTestInt(&name<uint_v>, #name "<uint_v>")
#define testRealTypes(name) \
    _unit_test_global.runTestInt(&name<float_v>, #name "<float_v>"); \
    _unit_test_global.runTestInt(&name<double_v>, #name "<double_v>");

template<typename A, typename B> struct isEqualType
{
    operator bool() const { return false; }
};

template<typename T> struct isEqualType<T, T>
{
    operator bool() const { return true; }
};

static inline void printPass()
{
    std::cout << AnsiColor::green << " PASS: " << AnsiColor::normal;
}

bool _UnitTest_verify_vector_unit_supported()
{
    bool s = Vc::currentImplementationSupported();
    if (!s) {
        std::cerr << "CPU or OS requirements not met for the compiled in vector unit!\n";
        exit(-1);
    }
    return s;
}

namespace
{
bool _UnitTest_verify_vector_unit_supported_result = _UnitTest_verify_vector_unit_supported();
} // anonymous namespace

class _UnitTest_Failure
{
};

typedef void (*testFunction)(void);

class _UnitTest_Global_Object
{
    public:
        _UnitTest_Global_Object()
            : status(true),
            expect_failure(false),
            assert_failure(0),
            expect_assert_failure(false),
            float_fuzzyness( 1.f ),
            double_fuzzyness( 1. ),
            only_name(0),
            m_finalized(false),
            failedTests(0), passedTests(0),
            findMaximumDistance(false),
            maximumDistance(0),
            meanDistance(0),
            meanCount(0)
        {
        }

        ~_UnitTest_Global_Object()
        {
        }

        int finalize()
        {
            if (plotFile.is_open()) {
                plotFile.flush();
                plotFile.close();
            }
            m_finalized = true;
            std::cout << "\n Testing done. " << passedTests << " tests passed. " << failedTests << " tests failed." << std::endl;
            return failedTests;
        }

        void runTestInt(testFunction fun, const char *name);

        bool status;
        bool expect_failure;
        int assert_failure;
        bool expect_assert_failure;
        float float_fuzzyness;
        double double_fuzzyness;
        const char *only_name;
        std::fstream plotFile;
    private:
        bool m_finalized;
        int failedTests;
    public:
        int passedTests;
        bool findMaximumDistance;
        double maximumDistance;
        double meanDistance;
        int meanCount;
};

static _UnitTest_Global_Object _unit_test_global;

void EXPECT_FAILURE()
{
    _unit_test_global.expect_failure = true;
}

static const char *_unittest_fail()
{
    if (_unit_test_global.expect_failure) {
        return "XFAIL: ";
    }
    static const char *str = 0;
    if (str == 0) {
        if (mayUseColor(std::cout)) {
            static const char *fail = " \033[1;40;31mFAIL:\033[0m ";
            str = fail;
        } else {
            static const char *fail = " FAIL: ";
            str = fail;
        }
    }
    return str;
}

void initTest(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i) {
        if (0 == std::strcmp(argv[i], "--help") || 0 == std::strcmp(argv[i], "-h")) {
            std::cout <<
                "Usage: " << argv[0] << " [-h|--help] [--only <testname>] [--maxdist] [--plotdist <plot.dat>]\n";
            exit(0);
        }
        if (0 == std::strcmp(argv[i], "--only") && i + 1 < argc) {
            _unit_test_global.only_name = argv[i + 1];
        } else if (0 == std::strcmp(argv[i], "--maxdist")) {
            _unit_test_global.findMaximumDistance = true;
        } else if (0 == std::strcmp(argv[i], "--plotdist") && i + 1 < argc) {
            _unit_test_global.plotFile.open(argv[i + 1], std::ios_base::out);
            _unit_test_global.plotFile << "# reference\tdistance\n";
        }
    }
}

template<typename T> static inline void setFuzzyness( T );
template<> inline void setFuzzyness<float>( float fuzz ) { _unit_test_global.float_fuzzyness = fuzz; }
template<> inline void setFuzzyness<double>( double fuzz ) { _unit_test_global.double_fuzzyness = fuzz; }

void _UnitTest_Global_Object::runTestInt(testFunction fun, const char *name)
{
    if (_unit_test_global.only_name && 0 != std::strcmp(name, _unit_test_global.only_name)) {
        return;
    }
    _unit_test_global.status = true;
    _unit_test_global.expect_failure = false;
    try {
        setFuzzyness<float>(1);
        setFuzzyness<double>(1);
        maximumDistance = 0.;
        meanDistance = 0.;
        meanCount = 0;
        fun();
    } catch(_UnitTest_Failure) {
    }
    if (_unit_test_global.expect_failure) {
        if (!_unit_test_global.status) {
            std::cout << "XFAIL: " << name << std::endl;
        } else {
            std::cout << "unexpected PASS: " << name <<
                "\n    This test should have failed but didn't. Check the code!" << std::endl;
            ++failedTests;
        }
    } else {
        if (!_unit_test_global.status) {
            if (findMaximumDistance) {
                std::cout << _unittest_fail() << "│ with a maximal distance of " << maximumDistance << " to the reference (mean: " << meanDistance / meanCount << ").\n";
            }
            std::cout << _unittest_fail() << "┕ " << name << std::endl;
            ++failedTests;
        } else {
            printPass();
            std::cout << name;
            if (findMaximumDistance) {
                if (maximumDistance > 0.) {
                    std::cout << " with a maximal distance of " << maximumDistance << " to the reference (mean: " << meanDistance / meanCount << ").";
                } else {
                    std::cout << " all values matched the reference precisely.";
                }
            }
            std::cout << std::endl;
            ++passedTests;
        }
    }
}

template<typename T> struct is_simd_array_internal : public std::false_type {};
template<typename T, std::size_t N> struct is_simd_array_internal<Vc::simd_array<T, N>> : public std::true_type {};

template<typename T> struct is_simd_array
    : public is_simd_array_internal<typename std::remove_cv<typename std::remove_reference<T>::type>::type> {};

template<typename T>
struct is_simd : public std::integral_constant<bool,
    Vc::is_simd_vector<T>::value || is_simd_array<T>::value>
{};

template<typename T1, typename T2> Vc_ALWAYS_INLINE
typename std::enable_if<!(is_simd<T1>::value || is_simd<T2>::value), bool>::type
unittest_compareHelper(const T1 &a, const T2 &b) { return a == b; }

template<typename T1, typename T2> Vc_ALWAYS_INLINE
typename std::enable_if< (is_simd<T1>::value || is_simd<T2>::value), bool>::type
unittest_compareHelper(const T1 &a, const T2 &b) { return Vc::all_of(a == b); }

template<> Vc_ALWAYS_INLINE bool unittest_compareHelper<std::type_info, std::type_info>(const std::type_info &a, const std::type_info &b ) { return &a == &b; }

template<typename T> T ulpDiffToReferenceWrapper(T a, T b) {
    const T diff = ulpDiffToReference(a, b);
    if (VC_IS_UNLIKELY(_unit_test_global.findMaximumDistance)) {
        _unit_test_global.maximumDistance = std::max<double>(std::abs(diff), _unit_test_global.maximumDistance);
        _unit_test_global.meanDistance += std::abs(diff);
        ++_unit_test_global.meanCount;
    }
    return diff;
}
template<typename T> Vc::Vector<T> ulpDiffToReferenceWrapper(VC_ALIGNED_PARAMETER(Vc::Vector<T>) a, VC_ALIGNED_PARAMETER(Vc::Vector<T>) b) {
    const Vc::Vector<T> diff = ulpDiffToReference(a, b);
    if (VC_IS_UNLIKELY(_unit_test_global.findMaximumDistance)) {
        _unit_test_global.maximumDistance = std::max<double>(Vc::abs(diff).max(), _unit_test_global.maximumDistance);
        _unit_test_global.meanDistance += Vc::abs(diff).sum();
        _unit_test_global.meanCount += Vc::Vector<T>::Size;
    }
    return diff;
}
template<typename T> static inline bool unittest_fuzzyCompareHelper( const T &a, const T &b ) { return a == b; }
template<> inline bool unittest_fuzzyCompareHelper<float>( const float &a, const float &b ) {
    return ulpDiffToReferenceWrapper(a, b) <= _unit_test_global.float_fuzzyness;
}
template<> inline bool unittest_fuzzyCompareHelper<Vc::float_v>( const Vc::float_v &a, const Vc::float_v &b ) {
    return (ulpDiffToReferenceWrapper(a, b) <= _unit_test_global.float_fuzzyness).isFull();
}
template<> inline bool unittest_fuzzyCompareHelper<double>( const double &a, const double &b ) {
    return ulpDiffToReferenceWrapper(a, b) <= _unit_test_global.double_fuzzyness;
}
template<> inline bool unittest_fuzzyCompareHelper<Vc::double_v>( const Vc::double_v &a, const Vc::double_v &b ) {
    return (ulpDiffToReferenceWrapper(a, b) <= _unit_test_global.double_fuzzyness).isFull();
}

template<typename T1, typename T2, typename M> inline void unitttest_comparePrintHelper(const T1 &a, const T2 &b, const M &m, const char *aa, const char *bb, const char *file, int line, double fuzzyness = 0.) {
    std::cout << "       " << aa << " (" << std::setprecision(10) << a << std::setprecision(6) << ") == " << bb << " (" << std::setprecision(10) << b << std::setprecision(6) << ") -> " << m;
    if (fuzzyness > 0.) {
        std::cout << " with fuzzyness " << fuzzyness;
    }
    std::cout << " at " << file << ":" << line << " failed.\n";
}

template<typename T> inline double unittest_fuzzynessHelper(const T &) { return 0.; }
template<> inline double unittest_fuzzynessHelper<float>(const float &) { return _unit_test_global.float_fuzzyness; }
template<> inline double unittest_fuzzynessHelper<Vc::float_v>(const Vc::float_v &) { return _unit_test_global.float_fuzzyness; }
template<> inline double unittest_fuzzynessHelper<double>(const double &) { return _unit_test_global.double_fuzzyness; }
template<> inline double unittest_fuzzynessHelper<Vc::double_v>(const Vc::double_v &) { return _unit_test_global.double_fuzzyness; }

class _UnitTest_Compare
{
    public:
        enum OptionFuzzy { Fuzzy };
        enum OptionNoEq { NoEq };

        template<typename T1, typename T2>
        Vc_ALWAYS_INLINE _UnitTest_Compare(const T1 &a, const T2 &b, const char *_a, const char *_b, const char *_file, int _line)
            : m_ip(getIp()), m_failed(!unittest_compareHelper(a, b))
        {
            if (VC_IS_UNLIKELY(m_failed)) {
                printFirst();
                printPosition(_file, _line); print(":\n");
                print(_a); print(" ("); print(std::setprecision(10)); print(a); print(") == ");
                print(_b); print(" ("); print(std::setprecision(10)); print(b); print(std::setprecision(6));
                print(") -> "); print(a == b);
            }
        }

        template<typename T1, typename T2>
        Vc_ALWAYS_INLINE _UnitTest_Compare(const T1 &a, const T2 &b, const char *_a, const char *_b, const char *_file, int _line, OptionNoEq)
            : m_ip(getIp()), m_failed(!unittest_compareHelper(a, b))
        {
            if (VC_IS_UNLIKELY(m_failed)) {
                printFirst();
                printPosition(_file, _line); print(":\n");
                print(_a); print(" ("); print(std::setprecision(10)); print(a); print(") == ");
                print(_b); print(" ("); print(std::setprecision(10)); print(b); print(std::setprecision(6));
                print(')');
            }
        }

        template<typename T>
        Vc_ALWAYS_INLINE _UnitTest_Compare(const T &a, const T &b, const char *_a, const char *_b, const char *_file, int _line, OptionFuzzy)
            : m_ip(getIp()), m_failed(!unittest_fuzzyCompareHelper(a, b))
        {
            if (VC_IS_UNLIKELY(m_failed)) {
                printFirst();
                printPosition(_file, _line); print(":\n");
                print(_a); print(" ("); print(std::setprecision(10)); print(a); print(") ≈ ");
                print(_b); print(" ("); print(std::setprecision(10)); print(b); print(std::setprecision(6));
                print(") -> "); print(a == b);
                printFuzzyInfo(a, b);
            }
            if (_unit_test_global.plotFile.is_open()) {
                writePlotData(_unit_test_global.plotFile, a, b);
            }
        }

        Vc_ALWAYS_INLINE _UnitTest_Compare(bool good, const char *cond, const char *_file, int _line)
            : m_ip(getIp()), m_failed(!good)
        {
            if (VC_IS_UNLIKELY(m_failed)) {
                printFirst();
                printPosition(_file, _line);
                print(": "); print(cond);
            }
        }

        Vc_ALWAYS_INLINE _UnitTest_Compare(const char *_file, int _line)
            : m_ip(getIp()), m_failed(true)
        {
            printFirst();
            printPosition(_file, _line);
            print(":\n");
        }

        template<typename T> Vc_ALWAYS_INLINE const _UnitTest_Compare &operator<<(const T &x) const {
            if (VC_IS_UNLIKELY(m_failed)) {
                print(x);
            }
            return *this;
        }

        Vc_ALWAYS_INLINE const _UnitTest_Compare &operator<<(const char *str) const {
            if (VC_IS_UNLIKELY(m_failed)) {
                print(str);
            }
            return *this;
        }

        Vc_ALWAYS_INLINE const _UnitTest_Compare &operator<<(const char ch) const {
            if (VC_IS_UNLIKELY(m_failed)) {
                print(ch);
            }
            return *this;
        }

        Vc_ALWAYS_INLINE const _UnitTest_Compare &operator<<(bool b) const {
            if (VC_IS_UNLIKELY(m_failed)) {
                print(b);
            }
            return *this;
        }

        Vc_ALWAYS_INLINE ~_UnitTest_Compare()
#ifdef Vc__NO_NOEXCEPT
            throw(_UnitTest_Failure)
#else
            noexcept(false)
#endif
        {
            if (VC_IS_UNLIKELY(m_failed)) {
                printLast();
            }
        }

    private:
        static Vc_ALWAYS_INLINE size_t getIp() {
            size_t _ip;
#ifdef VC_GNU_ASM
#ifdef __x86_64__
            asm volatile("lea 0(%%rip),%0" : "=r"(_ip));
#else
            //asm volatile("call 1f\n\t1: pop %0" : "=r"(_ip));
            asm volatile("1: movl $1b,%0" : "=r"(_ip));
#endif
#else
            _ip = 0;
#endif
            return _ip;
        }
        static void printFirst() { std::cout << _unittest_fail() << "┍ "; }
        template<typename T> static inline void print(const T &x) { std::cout << x; }
        static void print(const std::type_info &x) { std::cout << x.name(); }
        static void print(const char *str) {
            const char *pos = 0;
            if (0 != (pos = std::strchr(str, '\n'))) {
                if (pos == str) {
                    std::cout << '\n' << _unittest_fail() << "│ " << &str[1];
                } else {
                    char *left = strdup(str);
                    left[pos - str] = '\0';
                    std::cout << left << '\n' << _unittest_fail() << "│ " << &pos[1];
                    free(left);
                }
            } else {
                std::cout << str;
            }
        }
        static void print(const char ch) {
            if (ch == '\n') {
                std::cout << '\n' << _unittest_fail() << "│ ";
            } else {
                std::cout << ch;
            }
        }
        static void print(bool b) {
            std::cout << (b ? "true" : "false");
        }
        static void printLast() {
            std::cout << std::endl;
            _unit_test_global.status = false;
            //if (!_unit_test_global.plotFile.is_open()) {
            throw _UnitTest_Failure();
            //}
        }
        void printPosition(const char *_file, int _line) {
            std::cout << "at " << _file << ':' << _line << " (0x" << std::hex << m_ip << std::dec << ')';
        }
        template<typename T> static inline void writePlotData(std::fstream &file, VC_ALIGNED_PARAMETER(T) a, VC_ALIGNED_PARAMETER(T) b);
        template<typename T> static inline void printFuzzyInfo(VC_ALIGNED_PARAMETER(T) a, VC_ALIGNED_PARAMETER(T) b);
        template<typename T> static inline void printFuzzyInfoImpl(VC_ALIGNED_PARAMETER(T) a, VC_ALIGNED_PARAMETER(T) b, double fuzzyness) {
            print("\ndistance: ");
            print(ulpDiffToReferenceSigned(a, b));
            print(", allowed distance: ");
            print(fuzzyness);
        }
        const size_t m_ip;
        const bool m_failed;
};
template<typename T> inline void _UnitTest_Compare::printFuzzyInfo(VC_ALIGNED_PARAMETER(T), VC_ALIGNED_PARAMETER(T)) {}
template<> inline void _UnitTest_Compare::printFuzzyInfo(VC_ALIGNED_PARAMETER(float) a, VC_ALIGNED_PARAMETER(float) b) {
    printFuzzyInfoImpl(a, b, _unit_test_global.float_fuzzyness);
}
template<> inline void _UnitTest_Compare::printFuzzyInfo(VC_ALIGNED_PARAMETER(double) a, VC_ALIGNED_PARAMETER(double) b) {
    printFuzzyInfoImpl(a, b, _unit_test_global.double_fuzzyness);
}
template<> inline void _UnitTest_Compare::printFuzzyInfo(VC_ALIGNED_PARAMETER(Vc::float_v) a, VC_ALIGNED_PARAMETER(Vc::float_v) b) {
    printFuzzyInfoImpl(a, b, _unit_test_global.float_fuzzyness);
}
template<> inline void _UnitTest_Compare::printFuzzyInfo(VC_ALIGNED_PARAMETER(Vc::double_v) a, VC_ALIGNED_PARAMETER(Vc::double_v) b) {
    printFuzzyInfoImpl(a, b, _unit_test_global.double_fuzzyness);
}
template<typename T> inline void _UnitTest_Compare::writePlotData(std::fstream &, VC_ALIGNED_PARAMETER(T), VC_ALIGNED_PARAMETER(T)) {}
template<> inline void _UnitTest_Compare::writePlotData(std::fstream &file, VC_ALIGNED_PARAMETER(float) a, VC_ALIGNED_PARAMETER(float) b) {
    file << std::setprecision(12) << b << "\t" << ulpDiffToReferenceSigned(a, b) << "\n";
}
template<> inline void _UnitTest_Compare::writePlotData(std::fstream &file, VC_ALIGNED_PARAMETER(double) a, VC_ALIGNED_PARAMETER(double) b) {
    file << std::setprecision(12) << b << "\t" << ulpDiffToReferenceSigned(a, b) << "\n";
}
template<> inline void _UnitTest_Compare::writePlotData(std::fstream &file, VC_ALIGNED_PARAMETER(Vc::float_v) a, VC_ALIGNED_PARAMETER(Vc::float_v) b) {
    const Vc::float_v ref = b;
    const Vc::float_v dist = ulpDiffToReferenceSigned(a, b);
    for (size_t i = 0; i < Vc::float_v::Size; ++i) {
        file << std::setprecision(12) << ref[i] << "\t" << dist[i] << "\n";
    }
}
template<> inline void _UnitTest_Compare::writePlotData(std::fstream &file, VC_ALIGNED_PARAMETER(Vc::double_v) a, VC_ALIGNED_PARAMETER(Vc::double_v) b) {
    const Vc::double_v ref = b;
    const Vc::double_v dist = ulpDiffToReferenceSigned(a, b);
    for (size_t i = 0; i < Vc::double_v::Size; ++i) {
        file << std::setprecision(12) << ref[i] << "\t" << dist[i] << "\n";
    }
}

// Workaround for clang: The "<< ' '" is only added to silence the warnings about unused return
// values.
#define FUZZY_COMPARE( a, b ) \
    _UnitTest_Compare(a, b, #a, #b, __FILE__, __LINE__, _UnitTest_Compare::Fuzzy) << ' '

#define COMPARE( a, b ) \
    _UnitTest_Compare(a, b, #a, #b, __FILE__, __LINE__) << ' '

#define COMPARE_NOEQ( a, b ) \
    _UnitTest_Compare(a, b, #a, #b, __FILE__, __LINE__, _UnitTest_Compare::NoEq) << ' '

#define VERIFY(cond) \
    _UnitTest_Compare(cond, #cond, __FILE__, __LINE__) << ' '

#define FAIL() \
    _UnitTest_Compare(__FILE__, __LINE__) << ' '

class ADD_PASS
{
    public:
        ADD_PASS() { ++_unit_test_global.passedTests; printPass(); }
        ~ADD_PASS() { std::cout << std::endl; }
        template<typename T> ADD_PASS &operator<<(const T &x) { std::cout << x; return *this; }
};

static void unittest_assert(bool cond, const char *code, const char *file, int line)
{
    if (!cond) {
        if (_unit_test_global.expect_assert_failure) {
            ++_unit_test_global.assert_failure;
        } else {
            _UnitTest_Compare(file, line) << "assert(" << code << ") failed.";
        }
    }
}
#ifdef assert
#undef assert
#endif
#define assert(cond) unittest_assert(cond, #cond, __FILE__, __LINE__)

#define EXPECT_ASSERT_FAILURE(code) \
    _unit_test_global.expect_assert_failure = true; \
    _unit_test_global.assert_failure = 0; \
    code; \
    if (_unit_test_global.assert_failure == 0) { \
        /* failure expected but it didn't fail */ \
        std::cout << "       " << #code << " at " << __FILE__ << ":" << __LINE__ << \
            " did not fail as was expected.\n"; \
        _unit_test_global.status = false; \
        throw _UnitTest_Failure(); \
        return; \
    } \
    _unit_test_global.expect_assert_failure = false

template<typename Vec> static typename Vec::Mask allMasks(size_t i)
{
#ifdef VC_IMPL_MIC
    return ((1 << Vec::Size) - 1) - i;
#else
    typedef typename Vec::IndexType I;
    typedef typename Vec::Mask M;

    if (i == 0) {
        return M(Vc::One);
    } else if (Vec::Size == 1) {
        return M(Vc::Zero);
    }
#ifndef VC_IMPL_Scalar
    --i;
    if (i < Vec::Size) {
        return M (I(Vc::IndexesFromZero) == i);
    }
    i -= Vec::Size;
    if (Vec::Size < 3) {
        return M(Vc::Zero);
    }
    for (size_t a = 0; a < Vec::Size - 1; ++a) {
        for (size_t b = a + 1; b < Vec::Size; ++b) {
            if (i == 0) {
                I indexes(Vc::IndexesFromZero);
                return M(indexes == a || indexes == b);
            }
            --i;
        }
    }
    if (Vec::Size < 4) {
        return M(Vc::Zero);
    }
    for (size_t a = 0; a < Vec::Size - 1; ++a) {
        for (size_t b = a + 1; b < Vec::Size; ++b) {
            for (size_t c = b + 1; c < Vec::Size; ++c) {
                if (i == 0) {
                    I indexes(Vc::IndexesFromZero);
                    return M(indexes == a || indexes == b || indexes == c);
                }
                --i;
            }
        }
    }
    if (Vec::Size < 5) {
        return M(Vc::Zero);
    }
    for (size_t a = 0; a < Vec::Size - 1; ++a) {
        for (size_t b = a + 1; b < Vec::Size; ++b) {
            for (size_t c = b + 1; c < Vec::Size; ++c) {
                for (size_t d = c + 1; d < Vec::Size; ++d) {
                    if (i == 0) {
                        I indexes(Vc::IndexesFromZero);
                        return M(indexes == a || indexes == b || indexes == c || indexes == d);
                    }
                    --i;
                }
            }
        }
    }
    if (Vec::Size < 6) {
        return M(Vc::Zero);
    }
    for (size_t a = 0; a < Vec::Size - 1; ++a) {
        for (size_t b = a + 1; b < Vec::Size; ++b) {
            for (size_t c = b + 1; c < Vec::Size; ++c) {
                for (size_t d = c + 1; d < Vec::Size; ++d) {
                    for (size_t e = d + 1; e < Vec::Size; ++e) {
                        if (i == 0) {
                            I indexes(Vc::IndexesFromZero);
                            return M(indexes == a || indexes == b || indexes == c || indexes == d || indexes == e);
                        }
                        --i;
                    }
                }
            }
        }
    }
    if (Vec::Size < 7) {
        return M(Vc::Zero);
    }
    for (size_t a = 0; a < Vec::Size - 1; ++a) {
        for (size_t b = a + 1; b < Vec::Size; ++b) {
            for (size_t c = b + 1; c < Vec::Size; ++c) {
                for (size_t d = c + 1; d < Vec::Size; ++d) {
                    for (size_t e = d + 1; e < Vec::Size; ++e) {
                        for (size_t f = e + 1; f < Vec::Size; ++f) {
                            if (i == 0) {
                                I indexes(Vc::IndexesFromZero);
                                return M(indexes == a || indexes == b || indexes == c || indexes == d || indexes == e || indexes == f);
                            }
                            --i;
                        }
                    }
                }
            }
        }
    }
    if (Vec::Size < 8) {
        return M(Vc::Zero);
    }
    for (size_t a = 0; a < Vec::Size - 1; ++a) {
        for (size_t b = a + 1; b < Vec::Size; ++b) {
            for (size_t c = b + 1; c < Vec::Size; ++c) {
                for (size_t d = c + 1; d < Vec::Size; ++d) {
                    for (size_t e = d + 1; e < Vec::Size; ++e) {
                        for (size_t f = e + 1; f < Vec::Size; ++f) {
                            for (size_t g = f + 1; g < Vec::Size; ++g) {
                                if (i == 0) {
                                    I indexes(Vc::IndexesFromZero);
                                    return M(indexes == a || indexes == b || indexes == c || indexes == d
                                            || indexes == e || indexes == f || indexes == g);
                                }
                                --i;
                            }
                        }
                    }
                }
            }
        }
    }
#endif
    return M(Vc::Zero);
#endif
}

#define for_all_masks(VecType, _mask_) \
    for (int _Vc_for_all_masks_i = 0; _Vc_for_all_masks_i == 0; ++_Vc_for_all_masks_i) \
        for (typename VecType::Mask _mask_ = allMasks<VecType>(_Vc_for_all_masks_i++); !_mask_.isEmpty(); _mask_ = allMasks<VecType>(_Vc_for_all_masks_i++))

namespace UnitTest
{
using std::vector;
using std::tuple;
using std::get;

template<typename T> inline std::string typeToString();

template<typename T, size_t N> inline std::string typeToString_impl(Vc::simd_array<T, N>)
{
    std::stringstream s;
    s << "simd_array<" << typeToString<T>() << ", " << N << '>';
    return s.str();
}

template<typename T> inline std::string typeToString_impl(T)
{
    return typeid(T).name();
}

template<typename T> inline std::string typeToString()
{
    return typeToString_impl(T());
}
template<> inline std::string typeToString<void>() { return ""; }

template<> inline std::string typeToString<Vc:: float_v>() { return " float_v"; }
template<> inline std::string typeToString<Vc:: short_v>() { return " short_v"; }
template<> inline std::string typeToString<Vc::  uint_v>() { return "  uint_v"; }
template<> inline std::string typeToString<Vc::double_v>() { return "double_v"; }
template<> inline std::string typeToString<Vc::ushort_v>() { return "ushort_v"; }
template<> inline std::string typeToString<Vc::   int_v>() { return "   int_v"; }

template<> inline std::string typeToString<long double>() { return "long double"; }
template<> inline std::string typeToString<double>() { return "double"; }
template<> inline std::string typeToString< float>() { return " float"; }
template<> inline std::string typeToString<         long long>() { return " long long"; }
template<> inline std::string typeToString<unsigned long long>() { return "ulong long"; }
template<> inline std::string typeToString<         long>() { return "  long"; }
template<> inline std::string typeToString<unsigned long>() { return " ulong"; }
template<> inline std::string typeToString<         int>() { return "   int"; }
template<> inline std::string typeToString<unsigned int>() { return "  uint"; }
template<> inline std::string typeToString<         short>() { return " short"; }
template<> inline std::string typeToString<unsigned short>() { return "ushort"; }
template<> inline std::string typeToString<         char>() { return "  char"; }
template<> inline std::string typeToString<unsigned char>() { return " uchar"; }
template<> inline std::string typeToString<  signed char>() { return " schar"; }

namespace
{
typedef tuple<testFunction, std::string> TestData;
vector<TestData> g_allTests;

void runAll()
{
    for (const auto &data : g_allTests) {
        _unit_test_global.runTestInt(get<0>(data), get<1>(data).c_str());
    }
}
} // anonymous namespace

template <typename T> class Test
{
public:
    Test(testFunction fun, std::string name)
    {
        name += '<' + typeToString<T>() + '>';
        g_allTests.emplace_back(fun, name);
    }
};

template<template<typename V> class TestFunctor, typename... TestTypes> class Test2;

template <template <typename V> class TestFunctor> class Test2<TestFunctor>
{
protected:
    Test2(const std::string &)
    {
    }
};

template <template <typename V> class TestFunctor, typename TestType0, typename... TestTypes>
class Test2<TestFunctor, TestType0, TestTypes...> : public Test2<TestFunctor, TestTypes...>
{
    typedef Test2<TestFunctor, TestTypes...> Base;

public:
    static void call() {
        TestFunctor<TestType0>();
    }

    Test2(std::string name) : Base(name)
    {
        name += '<' + typeToString<TestType0>() + '>';
        g_allTests.emplace_back(&call, name);
    }
};

template <template <typename V> class F, typename... Typelist>
UnitTest::Test2<F, Typelist...> hackTypelist(void (*)(Typelist...));

} // namespace UnitTest

#define TEST_ALL_V(V__, fun__)                                                                     \
    template <typename V__> void fun__();                                                          \
    static UnitTest::Test< float_v> test_##fun__##__float_v__(&fun__< float_v>, #fun__);           \
    static UnitTest::Test< short_v> test_##fun__##__short_v__(&fun__< short_v>, #fun__);           \
    static UnitTest::Test<  uint_v> test_##fun__##_ushort_v__(&fun__<  uint_v>, #fun__);           \
    static UnitTest::Test<double_v> test_##fun__##____int_v__(&fun__<double_v>, #fun__);           \
    static UnitTest::Test<ushort_v> test_##fun__##_double_v__(&fun__<ushort_v>, #fun__);           \
    static UnitTest::Test<   int_v> test_##fun__##___uint_v__(&fun__<   int_v>, #fun__);           \
    template <typename V__> void fun__()

#define TEST_BEGIN(V__, fun__, typelist__)                                                         \
    template <typename V__> struct fun__;                                                          \
    static auto test_##fun__##__ =                                                                 \
        decltype(UnitTest::hackTypelist<fun__>(std::declval<void typelist__>()))(#fun__);          \
    template <typename V__> struct fun__                                                           \
    {                                                                                              \
        void operator()()                                                                          \
        {
#define TEST_END                                                                                   \
    }                                                                                              \
    }                                                                                              \
    ;

#define ALL_VECTORS Vc::int_v, Vc::ushort_v, Vc::double_v, Vc::uint_v, Vc::short_v, Vc::float_v
#define SIMD_ARRAYS(N__)                                                                             \
    Vc::simd_array<int, N__>, Vc::simd_array<unsigned short, N__>, Vc::simd_array<double, N__>,       \
        Vc::simd_array<unsigned int, N__>, Vc::simd_array<short, N__>, Vc::simd_array<float, N__>

#define TEST(fun__)                                                                                \
    void fun__();                                                                                  \
    static UnitTest::Test<void> test_##fun__##__(&fun__, #fun__);                                  \
    void fun__()

void testmain();

int main(int argc, char **argv)
{
    initTest(argc, argv);
#ifdef VC_NEWTEST
    UnitTest::runAll();
#else
    testmain();
#endif
    return _unit_test_global.finalize();
}

#endif // UNITTEST_H
