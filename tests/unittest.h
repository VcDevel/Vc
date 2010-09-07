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

#ifndef UNITTEST_H
#define UNITTEST_H

#include <Vc/Vc>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include "../common/support.h"

#define runTest(name) _unit_test_global.runTestInt(&name, #name)
#define testAllTypes(name) \
    _unit_test_global.runTestInt(&name<float_v>, #name "<float_v>"); \
    _unit_test_global.runTestInt(&name<short_v>, #name "<short_v>"); \
    _unit_test_global.runTestInt(&name<sfloat_v>, #name "<sfloat_v>"); \
    _unit_test_global.runTestInt(&name<ushort_v>, #name "<ushort_v>"); \
    _unit_test_global.runTestInt(&name<int_v>, #name "<int_v>"); \
    _unit_test_global.runTestInt(&name<double_v>, #name "<double_v>"); \
    _unit_test_global.runTestInt(&name<uint_v>, #name "<uint_v>")

template<typename A, typename B> struct isEqualType
{
    operator bool() const { return false; }
};

template<typename T> struct isEqualType<T, T>
{
    operator bool() const { return true; }
};

bool _UnitTest_verify_vector_unit_supported()
{
    bool s = Vc::currentImplementationSupported();
    if (!s) {
        std::cerr << "CPU or OS requirements not met for the compiled in vector unit!\n";
        exit(-1);
    }
    return s;
}

static bool _UnitTest_verify_vector_unit_supported_result = _UnitTest_verify_vector_unit_supported();

class _UnitTest_Failure
{
};

typedef void (*testFunction)();
class _UnitTest_Global_Object
{
    public:
        _UnitTest_Global_Object()
            : status(true),
            expect_failure(false),
            assert_failure(0),
            expect_assert_failure(false),
            float_fuzzyness( 1e-6f ),
            double_fuzzyness( 1e-20f ),
            failedTests(0), passedTests(0)
        {
        }

        ~_UnitTest_Global_Object()
        {
            std::cout << "\n Testing done. " << passedTests << " tests passed. " << failedTests << " tests failed." << std::endl;
            std::exit(failedTests);
        }

        void runTestInt(testFunction fun, const char *name);

        bool status;
        bool expect_failure;
        int assert_failure;
        bool expect_assert_failure;
        float float_fuzzyness;
        double double_fuzzyness;
    private:
        int failedTests;
        int passedTests;
};

static _UnitTest_Global_Object _unit_test_global;

void EXPECT_FAILURE()
{
    _unit_test_global.expect_failure = true;
}

void _UnitTest_Global_Object::runTestInt(testFunction fun, const char *name)
{
    _unit_test_global.status = true;
    _unit_test_global.expect_failure = false;
    try {
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
            std::cout << " FAIL: " << name << std::endl;
            ++failedTests;
        } else {
            std::cout << " PASS: " << name << std::endl;
            ++passedTests;
        }
    }
}

template<typename T> static inline void setFuzzyness( T );

template<> inline void setFuzzyness<float>( float fuzz ) { _unit_test_global.float_fuzzyness = fuzz; }
template<> inline void setFuzzyness<double>( double fuzz ) { _unit_test_global.double_fuzzyness = fuzz; }

#define VERIFY(cond) if (cond) {} else { std::cout << "       " << #cond << " at " << __FILE__ << ":" << __LINE__ << " failed.\n"; _unit_test_global.status = false; return; }

template<typename T1, typename T2> static inline bool unittest_compareHelper( const T1 &a, const T2 &b ) { return a == b; }
template<> inline bool unittest_compareHelper<Vc::int_v, Vc::int_v>( const Vc::int_v &a, const Vc::int_v &b ) { return (a == b).isFull(); }
template<> inline bool unittest_compareHelper<Vc::uint_v, Vc::uint_v>( const Vc::uint_v &a, const Vc::uint_v &b ) { return (a == b).isFull(); }
template<> inline bool unittest_compareHelper<Vc::float_v, Vc::float_v>( const Vc::float_v &a, const Vc::float_v &b ) { return (a == b).isFull(); }
template<> inline bool unittest_compareHelper<Vc::double_v, Vc::double_v>( const Vc::double_v &a, const Vc::double_v &b ) { return (a == b).isFull(); }
#if !VC_IMPL_LRBni
template<> inline bool unittest_compareHelper<Vc::ushort_v, Vc::ushort_v>( const Vc::ushort_v &a, const Vc::ushort_v &b ) { return (a == b).isFull(); }
template<> inline bool unittest_compareHelper<Vc::short_v, Vc::short_v>( const Vc::short_v &a, const Vc::short_v &b ) { return (a == b).isFull(); }
#endif

template<typename T> static inline bool unittest_fuzzyCompareHelper( const T &a, const T &b ) { return a == b; }

template<> inline bool unittest_fuzzyCompareHelper<float>( const float &a, const float &b )
{
    return a == b || std::abs(a - b) <= _unit_test_global.float_fuzzyness * std::abs(b);
}

template<> inline bool unittest_fuzzyCompareHelper<Vc::float_v>( const Vc::float_v &a, const Vc::float_v &b )
{
    return a == b || Vc::abs(a - b) <= _unit_test_global.float_fuzzyness * Vc::abs(b);
}

#if VC_IMPL_SSE
template<> inline bool unittest_fuzzyCompareHelper<Vc::sfloat_v>( const Vc::sfloat_v &a, const Vc::sfloat_v &b )
{
    return a == b || Vc::abs(a - b) <= _unit_test_global.float_fuzzyness * Vc::abs(b);
}
#endif

template<> inline bool unittest_fuzzyCompareHelper<double>( const double &a, const double &b )
{
    return a == b || std::abs(a - b) <= _unit_test_global.double_fuzzyness * std::abs(b);
}

template<> inline bool unittest_fuzzyCompareHelper<Vc::double_v>( const Vc::double_v &a, const Vc::double_v &b )
{
    return a == b || Vc::abs(a - b) <= _unit_test_global.double_fuzzyness * Vc::abs(b);
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

#define FUZZY_COMPARE( a, b ) \
if ( unittest_fuzzyCompareHelper( a, b ) ) {} else { \
    unitttest_comparePrintHelper(a, b, (a) == (b), #a, #b, __FILE__, __LINE__, unittest_fuzzynessHelper(a)); \
    _unit_test_global.status = false; \
    throw _UnitTest_Failure(); \
    return; \
}

#define COMPARE( a, b ) \
if ( unittest_compareHelper( a, b ) ) {} else { \
    unitttest_comparePrintHelper(a, b, (a) == (b), #a, #b, __FILE__, __LINE__); \
    _unit_test_global.status = false; \
    throw _UnitTest_Failure(); \
    return; \
}

#define COMPARE_NOEQ( a, b ) \
if ( unittest_compareHelper( a, b ) ) {} else { \
    unitttest_comparePrintHelper(a, b, "", #a, #b, __FILE__, __LINE__); \
    _unit_test_global.status = false; \
    throw _UnitTest_Failure(); \
    return; \
}

static void unittest_assert(bool cond, const char *code, const char *file, int line)
{
    if (!cond) {
        if (_unit_test_global.expect_assert_failure) {
            ++_unit_test_global.assert_failure;
        } else {
            std::cout << "       " << code << " at " << file << ":" << line << " failed.\n";
            std::abort();
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

#endif // UNITTEST_H
