/*
    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) version 3.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Library General Public License for more details.

    You should have received a copy of the GNU Library General Public License
    along with this library; see the file COPYING.LIB.  If not, write to
    the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
    Boston, MA 02110-1301, USA.

*/

#ifndef UNITTEST_H
#define UNITTEST_H

#include <vector.h>

#include <iostream>
#include <cstdlib>

#define runTest(name) _unit_test_global.runTestInt(&name, #name)

typedef void (*testFunction)();
class _UnitTest_Global_Object
{
    public:
        _UnitTest_Global_Object()
            : status(true),
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
        int assert_failure;
        bool expect_assert_failure;
        float float_fuzzyness;
        double double_fuzzyness;
    private:
        int failedTests;
        int passedTests;
};

static _UnitTest_Global_Object _unit_test_global;

void _UnitTest_Global_Object::runTestInt(testFunction fun, const char *name)
{
    _unit_test_global.status = true;
    fun();
    if (!_unit_test_global.status) {
        std::cout << "FAIL:  " << name << std::endl;
        ++failedTests;
        return;
        //std::exit(1);
    }
    std::cout << "PASS:  " << name << std::endl;
    ++passedTests;
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
#ifndef ENABLE_LARRABEE
template<> inline bool unittest_compareHelper<Vc::ushort_v, Vc::ushort_v>( const Vc::ushort_v &a, const Vc::ushort_v &b ) { return (a == b).isFull(); }
template<> inline bool unittest_compareHelper<Vc::short_v, Vc::short_v>( const Vc::short_v &a, const Vc::short_v &b ) { return (a == b).isFull(); }
#endif

template<typename T> static inline bool unittest_fuzzyCompareHelper( const T &a, const T &b ) { return a == b; }

template<> inline bool unittest_fuzzyCompareHelper<float>( const float &a, const float &b )
{
    if (a < 0.f) {
        return ( a * ( 1.f + _unit_test_global.float_fuzzyness ) <= b ) && ( a * ( 1.f - _unit_test_global.float_fuzzyness ) >= b );
    }
    return ( a * ( 1.f + _unit_test_global.float_fuzzyness ) >= b ) && ( a * ( 1.f - _unit_test_global.float_fuzzyness ) <= b );
}

template<> inline bool unittest_fuzzyCompareHelper<Vc::float_v>( const Vc::float_v &a, const Vc::float_v &b )
{
    typedef Vc::float_v::Mask Mask;
    Mask m1 = a < 0.f;
    Mask neg = ( ( a * ( 1.f + _unit_test_global.float_fuzzyness ) <= b ) && ( a * ( 1.f - _unit_test_global.float_fuzzyness ) >= b ) );
    Mask pos = ( ( a * ( 1.f + _unit_test_global.float_fuzzyness ) >= b ) && ( a * ( 1.f - _unit_test_global.float_fuzzyness ) <= b ) );
    return (m1 && neg) || (!m1 && pos);
}

template<> inline bool unittest_fuzzyCompareHelper<double>( const double &a, const double &b )
{
    if (a < 0.) {
        return ( a * ( 1. + _unit_test_global.double_fuzzyness ) <= b ) && ( a * ( 1. - _unit_test_global.double_fuzzyness ) >= b );
    }
    return ( a * ( 1. + _unit_test_global.double_fuzzyness ) >= b ) && ( a * ( 1. - _unit_test_global.double_fuzzyness ) <= b );
}

template<> inline bool unittest_fuzzyCompareHelper<Vc::double_v>( const Vc::double_v &a, const Vc::double_v &b )
{
    typedef Vc::double_v::Mask Mask;
    Mask m1 = a < 0.f;
    Mask neg = ( ( a * ( 1.f + _unit_test_global.double_fuzzyness ) <= b ) && ( a * ( 1.f - _unit_test_global.double_fuzzyness ) >= b ) );
    Mask pos = ( ( a * ( 1.f + _unit_test_global.double_fuzzyness ) >= b ) && ( a * ( 1.f - _unit_test_global.double_fuzzyness ) <= b ) );
    return (m1 && neg) || (!m1 && pos);
}

template<typename T1, typename T2> inline void unitttest_comparePrintHelper(const T1 &a, const T2 &b, const char *aa, const char *bb, const char *file, int line) {
    std::cout << "       " << aa << " (" << a << ") == " << bb << " (" << b << ") at " << file << ":" << line << " failed.\n";
}

template<typename T> inline double unittest_fuzzynessHelper(T) { return 0.; }
template<> inline double unittest_fuzzynessHelper<float>(float) { return _unit_test_global.float_fuzzyness; }
template<> inline double unittest_fuzzynessHelper<Vc::float_v>(Vc::float_v) { return _unit_test_global.float_fuzzyness; }
template<> inline double unittest_fuzzynessHelper<double>(double) { return _unit_test_global.double_fuzzyness; }
template<> inline double unittest_fuzzynessHelper<Vc::double_v>(Vc::double_v) { return _unit_test_global.double_fuzzyness; }

#define FUZZY_COMPARE( a, b ) if ( unittest_fuzzyCompareHelper( a, b ) ) {} else { std::cout << "       " << #a << " (" << (a) << ") ~== " << #b << " (" << (b) << ") with fuzzyness " << unittest_fuzzynessHelper(a) << " at " << __FILE__ << ":" << __LINE__ << " failed.\n"; _unit_test_global.status = false; return; }

#define COMPARE( a, b ) if ( unittest_compareHelper( a, b ) ) {} else { unitttest_comparePrintHelper(a, b, #a, #b, __FILE__, __LINE__); _unit_test_global.status = false; return; }

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
        return; \
    } \
    _unit_test_global.expect_assert_failure = false

#endif // UNITTEST_H
