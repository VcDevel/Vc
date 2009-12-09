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

#include <Vc/Vc>
#include "unittest.h"

using namespace Vc;

template<typename V, unsigned int Size> struct TestEntries { static void run(); };
template<typename V> struct TestEntries<V, 0> { static void run(); };

template<typename V, unsigned int Size> void TestEntries<V, Size>::run()
{
    TestEntries<V, Size - 1>::run();
    typedef typename V::EntryType T;
    T x = Size;
    SSE::FixedSizeMemory<V, Size> m;
    for (unsigned int i = 0; i < Size; ++i) {
        m[i] = x;
    }
    for (unsigned int i = 0; i < Size; ++i) {
        COMPARE(m[i], x);
    }
    for (unsigned int i = 0; i < Size; ++i) {
        COMPARE(m.entries()[i], x);
    }
    T *ptr = m;
    for (unsigned int i = 0; i < Size; ++i) {
        COMPARE(ptr[i], x);
    }
}

template<typename V> void TestEntries<V, 0>::run()
{
}

template<typename V> void testEntries()
{
    TestEntries<V, 128>::run();
}

int main()
{
    testAllTypes(testEntries);

    return 0;
}
