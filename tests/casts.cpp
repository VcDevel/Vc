/*  This file is part of the Vc library.

    Copyright (C) 2010 Matthias Kretz <kretz@kde.org>

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

#include <Vc/Vc>
#include "unittest.h"
#include <limits>
#include <algorithm>

using namespace Vc;

template<typename V1, typename V2> void testCast2()
{
    typedef typename V1::EntryType T1;
    typedef typename V2::EntryType T2;

    const double max = std::min(
            static_cast<double>(std::numeric_limits<T1>::max()),
            static_cast<double>(std::numeric_limits<T2>::max()));
    const double min = std::max(
            std::numeric_limits<T1>::is_integer ?
                static_cast<double>(std::numeric_limits<T1>::min()) :
                static_cast<double>(-std::numeric_limits<T1>::max()),
            std::numeric_limits<T2>::is_integer ?
                static_cast<double>(std::numeric_limits<T2>::min()) :
                static_cast<double>(-std::numeric_limits<T2>::max())
                );

    const T1 max1 = static_cast<T1>(max);
    const T2 max2 = static_cast<T2>(max);
    const T1 min1 = static_cast<T1>(min);
    const T2 min2 = static_cast<T2>(min);

    V1 v1;
    V2 v2;

    v1 = max1;
    v2 = static_cast<V2>(v1);
    COMPARE(static_cast<V1>(v2), v1);

    v2 = max2;
    v1 = static_cast<V1>(v2);
    COMPARE(static_cast<V2>(v1), v2);

    v1 = min1;
    v2 = static_cast<V2>(v1);
    COMPARE(static_cast<V1>(v2), v1);

    v2 = min2;
    v1 = static_cast<V1>(v2);
    COMPARE(static_cast<V2>(v1), v2);
}

template<typename T> void testCast()
{
    testCast2<typename T::V1, typename T::V2>();
}

#define _CONCAT(A, B) A ## _ ## B
#define CONCAT(A, B) _CONCAT(A, B)
template<typename T1, typename T2>
struct T2Helper
{
    typedef T1 V1;
    typedef T2 V2;
};

int main(int argc, char **argv)
{
    initTest(argc, argv);

#define TEST(v1, v2) \
    typedef T2Helper<v1, v2> CONCAT(v1, v2); \
    runTest(testCast<CONCAT(v1, v2)>)

    TEST(float_v, float_v);
    TEST(float_v, int_v);
    TEST(float_v, uint_v);
    // needs special handling for different Size:
    //TEST(float_v, double_v);
    //TEST(float_v, short_v);
    //TEST(float_v, ushort_v);

    TEST(int_v, float_v);
    TEST(int_v, int_v);
    TEST(int_v, uint_v);

    TEST(uint_v, float_v);
    TEST(uint_v, int_v);
    TEST(uint_v, uint_v);

    TEST(ushort_v, sfloat_v);
    TEST(ushort_v, short_v);
    TEST(ushort_v, ushort_v);

    TEST(short_v, sfloat_v);
    TEST(short_v, short_v);
    TEST(short_v, ushort_v);

    TEST(sfloat_v, sfloat_v);
    TEST(sfloat_v, short_v);
    TEST(sfloat_v, ushort_v);
#undef TEST

    return 0;
}
