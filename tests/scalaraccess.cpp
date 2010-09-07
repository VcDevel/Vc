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

using namespace Vc;

template<typename V> void reads()
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;

    V a = V::Zero();
    const T zero = 0;
    for (int i = 0; i < V::Size; ++i) {
        const T x = a[i];
        COMPARE(x, zero);
    }
    a = static_cast<V>(I::IndexesFromZero());
    for (int i = 0; i < V::Size; ++i) {
        const T x = a[i];
        const T y = i;
        COMPARE(x, y);
    }
}

template<typename V> void writes()
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;

    V a;
    for (int i = 0; i < V::Size; ++i) {
        a[i] = static_cast<T>(i);
    }
    V b = static_cast<V>(I::IndexesFromZero());
#if defined(__GNUC__) && !defined(__INTEL_COMPILER) && __GNUC__ == 4 && __GNUC_MINOR__ == 3 && __OPTIMIZE__ && VC_IMPL_SSE
    // GCC 4.3.x miscompiles. Somehow it fails to get the may_alias type right here
    if (isEqualType<V, int_v>() || isEqualType<V, short_v>() || isEqualType<V, ushort_v>()) {
        EXPECT_FAILURE();
    }
#endif
    COMPARE(a, b);

    const T one = 1;
    const T two = 2;

    if (V::Size == 1) {
        a(a == 0) += one;
        a[0] += one;
        a(a == 0) += one;
        COMPARE(a, V(2));
    } else if (V::Size == 4) {
        a(a == 1) += two;
        a[2] += one;
        a(a == 3) += one;
        b(b == 1) += one;
        b(b == 2) += one;
        b(b == 3) += one;
        COMPARE(a, b);
    } else if (V::Size == 8 || V::Size == 16) {
        a(a == 2) += two;
        a[3] += one;
        a(a == 4) += one;
        b(b == 2) += one;
        b(b == 3) += one;
        b(b == 4) += one;
        COMPARE(a, b);
    } else {
        a(a == 0) += two;
        a[1] += one;
        a(a == 2) += one;
        b(b == 0) += one;
        b(b == 1) += one;
        b(b == 2) += one;
        COMPARE(a, b);
    }
}

int main()
{
    testAllTypes(reads);
    testAllTypes(writes);

    return 0;
}
