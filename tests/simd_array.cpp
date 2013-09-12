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

TEST_ALL_NATIVE_V(V, createArray)
{
    typedef typename V::EntryType T;
    simd_array<T, 32> array;

    COMPARE(array.size, 32);
    VERIFY(array.register_count > 0);
    VERIFY(array.register_count <= 32);
    VERIFY(array.register_count * V::Size >= 32);
}

TEST_ALL_NATIVE_V(V, broadcast)
{
    typedef typename V::EntryType T;
    simd_array<T, 32> array = 0;
    array = 1;
}

TEST_ALL_NATIVE_V(V, broadcast_equal)
{
    typedef typename V::EntryType T;
    simd_array<T, 32> a = 0;
    simd_array<T, 32> b = 0;
    COMPARE(a, b);
    a = 1;
    b = 1;
    COMPARE(a, b);
}

TEST_ALL_NATIVE_V(V, broadcast_not_equal)
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

TEST_ALL_NATIVE_V(V, load)
{
    typedef typename V::EntryType T;
    Vc::Memory<V, 34> data;
    data = V::Zero();

    simd_array<T, 32> a{ &data[0] };
    simd_array<T, 32> b = 0;
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
