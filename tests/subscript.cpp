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
#include "../common/subscript.h"

TEST_BEGIN(V, init, (ALL_VECTORS))
{
    typedef typename V::EntryType T;
    Vc::array<T, 256> data;
    for (int i = 0; i < data.size(); ++i) {
        data[i] = i;
    }
}
TEST_END

TEST_BEGIN(V, gathers, (ALL_VECTORS))
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType IT;
    Vc::array<T, 256> data;
    for (int i = 0; i < data.size(); ++i) {
        data[i] = i;
    }

    V test = data[IT::IndexesFromZero()];
    COMPARE(test, V::IndexesFromZero());

    IT indexes = abs(IT::Random()) % 256;
    test = data[indexes];
    COMPARE(test, static_cast<V>(indexes));
}
TEST_END
