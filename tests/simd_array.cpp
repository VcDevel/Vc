/*  This file is part of the Vc library. {{{

    Copyright (C) 2013 Matthias Kretz <kretz@kde.org>

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

}}}*/

#define VC_NEWTEST
#include "unittest.h"
#include "../common/simd_array.h"

using namespace Vc;

#define TEST_ALL_V_AND_N(V, N, fun) \
template<typename V, std::size_t N> void fun(); \
static UnitTest::Test test_##fun##__float_v_32_(&fun< float_v, 32>, #fun "< float_v, 32>"); \
static UnitTest::Test test_##fun##__short_v_32_(&fun< short_v, 32>, #fun "< short_v, 32>"); \
static UnitTest::Test test_##fun##_ushort_v_32_(&fun<ushort_v, 32>, #fun "<ushort_v, 32>"); \
static UnitTest::Test test_##fun##____int_v_32_(&fun<   int_v, 32>, #fun "<   int_v, 32>"); \
static UnitTest::Test test_##fun##_double_v_32_(&fun<double_v, 32>, #fun "<double_v, 32>"); \
static UnitTest::Test test_##fun##___uint_v_32_(&fun<  uint_v, 32>, #fun "<  uint_v, 32>"); \
static UnitTest::Test test_##fun##__float_v_16_(&fun< float_v, 16>, #fun "< float_v, 16>"); \
static UnitTest::Test test_##fun##__short_v_16_(&fun< short_v, 16>, #fun "< short_v, 16>"); \
static UnitTest::Test test_##fun##_ushort_v_16_(&fun<ushort_v, 16>, #fun "<ushort_v, 16>"); \
static UnitTest::Test test_##fun##____int_v_16_(&fun<   int_v, 16>, #fun "<   int_v, 16>"); \
static UnitTest::Test test_##fun##_double_v_16_(&fun<double_v, 16>, #fun "<double_v, 16>"); \
static UnitTest::Test test_##fun##___uint_v_16_(&fun<  uint_v, 16>, #fun "<  uint_v, 16>"); \
static UnitTest::Test test_##fun##__float_v__8_(&fun< float_v,  8>, #fun "< float_v,  8>"); \
static UnitTest::Test test_##fun##__short_v__8_(&fun< short_v,  8>, #fun "< short_v,  8>"); \
static UnitTest::Test test_##fun##_ushort_v__8_(&fun<ushort_v,  8>, #fun "<ushort_v,  8>"); \
static UnitTest::Test test_##fun##____int_v__8_(&fun<   int_v,  8>, #fun "<   int_v,  8>"); \
static UnitTest::Test test_##fun##_double_v__8_(&fun<double_v,  8>, #fun "<double_v,  8>"); \
static UnitTest::Test test_##fun##___uint_v__8_(&fun<  uint_v,  8>, #fun "<  uint_v,  8>"); \
static UnitTest::Test test_##fun##__float_v__4_(&fun< float_v,  4>, #fun "< float_v,  4>"); \
static UnitTest::Test test_##fun##__short_v__4_(&fun< short_v,  4>, #fun "< short_v,  4>"); \
static UnitTest::Test test_##fun##_ushort_v__4_(&fun<ushort_v,  4>, #fun "<ushort_v,  4>"); \
static UnitTest::Test test_##fun##____int_v__4_(&fun<   int_v,  4>, #fun "<   int_v,  4>"); \
static UnitTest::Test test_##fun##_double_v__4_(&fun<double_v,  4>, #fun "<double_v,  4>"); \
static UnitTest::Test test_##fun##___uint_v__4_(&fun<  uint_v,  4>, #fun "<  uint_v,  4>"); \
static UnitTest::Test test_##fun##__float_v__2_(&fun< float_v,  2>, #fun "< float_v,  2>"); \
static UnitTest::Test test_##fun##__short_v__2_(&fun< short_v,  2>, #fun "< short_v,  2>"); \
static UnitTest::Test test_##fun##_ushort_v__2_(&fun<ushort_v,  2>, #fun "<ushort_v,  2>"); \
static UnitTest::Test test_##fun##____int_v__2_(&fun<   int_v,  2>, #fun "<   int_v,  2>"); \
static UnitTest::Test test_##fun##_double_v__2_(&fun<double_v,  2>, #fun "<double_v,  2>"); \
static UnitTest::Test test_##fun##___uint_v__2_(&fun<  uint_v,  2>, #fun "<  uint_v,  2>"); \
static UnitTest::Test test_##fun##__float_v__1_(&fun< float_v,  1>, #fun "< float_v,  1>"); \
static UnitTest::Test test_##fun##__short_v__1_(&fun< short_v,  1>, #fun "< short_v,  1>"); \
static UnitTest::Test test_##fun##_ushort_v__1_(&fun<ushort_v,  1>, #fun "<ushort_v,  1>"); \
static UnitTest::Test test_##fun##____int_v__1_(&fun<   int_v,  1>, #fun "<   int_v,  1>"); \
static UnitTest::Test test_##fun##_double_v__1_(&fun<double_v,  1>, #fun "<double_v,  1>"); \
static UnitTest::Test test_##fun##___uint_v__1_(&fun<  uint_v,  1>, #fun "<  uint_v,  1>"); \
template<typename V, std::size_t N> void fun()

TEST_ALL_V_AND_N(V, N, createArray)
{
    typedef typename V::EntryType T;
    simd_array<T, N> array;

    COMPARE(array.size, N);
    VERIFY(array.register_count > 0);
    VERIFY(array.register_count <= N);
    VERIFY(array.register_count * V::Size >= N);
}

TEST_ALL_V_AND_N(V, N, broadcast)
{
    typedef typename V::EntryType T;
    simd_array<T, N> array = 0;
    array = 1;
}

TEST_ALL_V_AND_N(V, N, broadcast_equal)
{
    typedef typename V::EntryType T;
    simd_array<T, N> a = 0;
    simd_array<T, N> b = 0;
    COMPARE(a, b);
    a = 1;
    b = 1;
    COMPARE(a, b);
}

TEST_ALL_V(V, broadcast_not_equal)
{
    typedef typename V::EntryType T;
    simd_array<T, 32> a = 0;
    simd_array<T, 32> b = 1;
    VERIFY(all_of(a != b));
    VERIFY(all_of(a < b));
    VERIFY(all_of(a <= b));
    VERIFY(none_of(a > b));
    VERIFY(none_of(a >= b));
    a = 1;
    VERIFY(all_of(a <= b));
    VERIFY(all_of(a >= b));
}

TEST_ALL_V_AND_N(V, N, load)
{
    typedef typename V::EntryType T;
    Vc::Memory<V, N + 2> data;
    data = V::Zero();

    simd_array<T, N> a{ &data[0] };
    simd_array<T, N> b = 0;
    COMPARE(a, b);

    b.load(&data[0]);
    COMPARE(a, b);

    a.load(&data[1], Vc::Unaligned);
    COMPARE(a, b);

    b = decltype(b)(&data[2], Vc::Unaligned | Vc::Streaming);
    COMPARE(a, b);
}

TEST(load_converting)
{
    typedef simd_array<float, 32> A;

    Vc::Memory<double_v, 34> data;
    data = double_v::Zero();

    A a{ &data[0] };
    A b = 0.;
    COMPARE(a, b);

    b.load(&data[1], Vc::Unaligned);
    COMPARE(a, b);

    a = A(&data[2], Vc::Unaligned | Vc::Streaming);
    COMPARE(a, b);
}
