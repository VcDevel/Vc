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
static class _UnitTest_Global_Object
{
    public:
        _UnitTest_Global_Object()
            : status(true),
            assert_failure(0),
            expect_assert_failure(false),
            float_fuzzyness( 1e-6f ),
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
    private:
        int failedTests;
        int passedTests;
} _unit_test_global;

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

template<> static inline void setFuzzyness<float>( float fuzz ) { _unit_test_global.float_fuzzyness = fuzz; }

#define VERIFY(cond) if (cond) {} else { std::cout << "       " << #cond << " at " << __FILE__ << ":" << __LINE__ << " failed.\n"; _unit_test_global.status = false; return; }

template<typename T>
static inline bool unittest_compareHelper( const T &a, const T &b )
{
  return a == b;
}

template<> static inline bool unittest_compareHelper<float>( const float &a, const float &b )
{
  return ( a * ( 1.f + _unit_test_global.float_fuzzyness ) >= b ) && ( a * ( 1.f - _unit_test_global.float_fuzzyness ) <= b );
}

template<> static inline bool unittest_compareHelper<Vc::float_v>( const Vc::float_v &a, const Vc::float_v &b )
{
  return ( a * ( 1.f + _unit_test_global.float_fuzzyness ) >= b ) && ( a * ( 1.f - _unit_test_global.float_fuzzyness ) <= b );
}

template<> static inline bool unittest_compareHelper<double>( const double &a, const double &b )
{
  return ( a * ( 1. + 1.e-20 ) >= b ) && ( a * ( 1. - 1.e-20 ) <= b );
}

template<> static inline bool unittest_compareHelper<Vc::double_v>( const Vc::double_v &a, const Vc::double_v &b )
{
  return ( a * ( 1. + 1.e-20 ) >= b ) && ( a * ( 1. - 1.e-20 ) <= b );
}

#define COMPARE( a, b ) if ( unittest_compareHelper( a, b ) ) {} else { std::cout << "       " << #a << "(" << a << ") == " << #b << "(" << b << ") at " << __FILE__ << ":" << __LINE__ << " failed.\n"; _unit_test_global.status = false; return; }

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
#ifndef assert
#define assert(cond) unittest_assert(cond, #cond, __FILE__, __LINE__)
#endif

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
