/*{{{
    Copyright © 2013 Matthias Kretz <kretz@kde.org>

    Permission to use, copy, modify, and distribute this software
    and its documentation for any purpose and without fee is hereby
    granted, provided that the above copyright notice appear in all
    copies and that both that the copyright notice and this
    permission notice and warranty disclaimer appear in supporting
    documentation, and that the name of the author not be used in
    advertising or publicity pertaining to distribution of the
    software without specific, written prior permission.

    The author disclaim all warranties with regard to this
    software, including all implied warranties of merchantability
    and fitness.  In no event shall the author be liable for any
    special, indirect or consequential damages or any damages
    whatsoever resulting from loss of use, data or profits, whether
    in an action of contract, negligence or other tortious action,
    arising out of or in connection with the use or performance of
    this software.

}}}*/

#define VC_NEWTEST
#include "unittest.h"
#include "../stl/array"
#include "../stl/vector"
#include "../common/subscript.h"

TEST_BEGIN(V, init, (ALL_VECTORS))
{
    typedef typename V::EntryType T;
    Vc::array<T, 256> data;
    Vc::vector<T> data2(256);
    for (int i = 0; i < data.size(); ++i) {
        data[i] = i;
        data2[i] = i;
    }
}
TEST_END

TEST_BEGIN(V, gathers, (SIMD_ARRAYS(32), ALL_VECTORS))
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType IT;
    Vc::array<T, 256> data;
    Vc::vector<T> data2(256);
    for (int i = 0; i < data.size(); ++i) {
        data[i] = i;
        data2[i] = i;
    }

    V test = data[IT::IndexesFromZero()];
    COMPARE(test, V::IndexesFromZero());
    test = data2[IT::IndexesFromZero()];
    COMPARE(test, V::IndexesFromZero());

    IT indexes = abs(IT::Random()) % 256;
    test = data[indexes];
    COMPARE(test, static_cast<V>(indexes));
    test = data2[indexes];
    COMPARE(test, static_cast<V>(indexes));
}
TEST_END

TEST_BEGIN(V, scatters, (SIMD_ARRAYS(32), ALL_VECTORS))
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType IT;
    Vc::array<T, 256> data1;
    Vc::vector<T> data2(256);
    std::fill_n(&data1[0], 256, 0);
    std::fill_n(&data2[0], 256, 0);

    data1[IT::IndexesFromZero()] = V::IndexesFromZero();
    data2[IT::IndexesFromZero()] = V::IndexesFromZero();

    for (size_t i = 0; i < V::Size; ++i) {
        COMPARE(data1[i], i);
        COMPARE(data2[i], i);
    }

    for (int repetition = 0; repetition < 1024 / V::Size; ++repetition) {
        IT indexes = abs(IT::Random()) % 256;
        data1[indexes] = V::IndexesFromZero();
        data2[indexes] = V::IndexesFromZero();

        for (size_t i = 0; i < V::Size; ++i) {
            COMPARE(data1[indexes[i]], i);
            COMPARE(data2[indexes[i]], i);
        }
    }
}
TEST_END

TEST_BEGIN(V, fixedWidthGathers, (SIMD_ARRAYS(4)))
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType IT;
    Vc::array<T, 256> data;
    Vc::vector<T> data2(256);
    for (int i = 0; i < data.size(); ++i) {
        data[i] = i;
        data2[i] = i;
    }
    V test;

    const auto indexes = { 0, 5, 8, 3 };
    test = data[indexes];
    COMPARE(test, static_cast<V>(indexes));
    test = data2[indexes];
    COMPARE(test, static_cast<V>(indexes));
}
TEST_END
