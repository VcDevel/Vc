#ifndef TESTS_UNITTEST_OLD_H_
#define TESTS_UNITTEST_OLD_H_

#define Vc_expand(name) #name
#define runTest(name) UnitTest::global_unit_test_object_.runTestInt(&name, Vc_expand(name))
#define testAllTypes(name)                                                                         \
    UnitTest::global_unit_test_object_.runTestInt(&name<float_v>, #name "<float_v>");              \
    UnitTest::global_unit_test_object_.runTestInt(&name<short_v>, #name "<short_v>");              \
    UnitTest::global_unit_test_object_.runTestInt(&name<ushort_v>, #name "<ushort_v>");            \
    UnitTest::global_unit_test_object_.runTestInt(&name<int_v>, #name "<int_v>");                  \
    UnitTest::global_unit_test_object_.runTestInt(&name<double_v>, #name "<double_v>");            \
    UnitTest::global_unit_test_object_.runTestInt(&name<uint_v>, #name "<uint_v>")
#define testRealTypes(name)                                                                        \
    UnitTest::global_unit_test_object_.runTestInt(&name<float_v>, #name "<float_v>");              \
    UnitTest::global_unit_test_object_.runTestInt(&name<double_v>, #name "<double_v>");

void testmain();

#include "unittest.h"

#endif  // TESTS_UNITTEST_OLD_H_
