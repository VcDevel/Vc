/*  This file is part of the Vc library. {{{

    Copyright (C) 2009-2013 Matthias Kretz <kretz@kde.org>

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

#include "unittest.h"
#include <iostream>
#include "vectormemoryhelper.h"
#include <cmath>

using Vc::float_v;
using Vc::double_v;
using Vc::sfloat_v;
using Vc::int_v;
using Vc::uint_v;
using Vc::short_v;
using Vc::ushort_v;

template<typename T> T two() { return T(2); }
template<typename T> T three() { return T(3); }

template<typename Vec> void testInc()/*{{{*/
{
    VectorMemoryHelper<Vec> mem(2);
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;
    T *data = mem;
    for (int borderI = 0; borderI < Vec::Size; ++borderI) {/*{{{*/
        const T border = static_cast<T>(borderI);
        for (int i = 0; i < Vec::Size; ++i) {
            data[i] = static_cast<T>(i);
            data[i + Vec::Size] = data[i] + static_cast<T>(data[i] < border ? 1 : 0);
        }
        Vec a(&data[0]);
        Vec b(&data[Vec::Size]);
        Mask m = a < border;
        Vec aa(a);
        COMPARE(aa(m)++, a) << ", border: " << border << ", m: " << m;
        COMPARE(aa, b) << ", border: " << border << ", m: " << m;
        COMPARE(++a(m), b) << ", border: " << border << ", m: " << m;
        COMPARE(a, b) << ", border: " << border << ", m: " << m;
    }
/*}}}*/
    for (int borderI = 0; borderI < Vec::Size; ++borderI) {
        const T border = static_cast<T>(borderI);
        for (int i = 0; i < Vec::Size; ++i) {
            data[i] = static_cast<T>(i);
            data[i + Vec::Size] = data[i] + static_cast<T>(data[i] < border ? 1 : 0);
        }
        Vec a(&data[0]);
        Vec b(&data[Vec::Size]);
        Mask m = a < border;
        Vec aa(a);
        where(m)(aa)++;
        COMPARE(aa, b) << ", border: " << border << ", m: " << m;
        ++where(m)(a);
        COMPARE(a, b) << ", border: " << border << ", m: " << m;
    }
}
/*}}}*/
template<typename Vec> void testDec()/*{{{*/
{
    VectorMemoryHelper<Vec> mem(2);
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;
    T *data = mem;
    for (int borderI = 0; borderI < Vec::Size; ++borderI) {
        const T border = static_cast<T>(borderI);
        for (int i = 0; i < Vec::Size; ++i) {
            data[i] = static_cast<T>(i + 1);
            data[i + Vec::Size] = data[i] - static_cast<T>(data[i] < border ? 1 : 0);
        }
        Vec a(&data[0]);
        Vec b(&data[Vec::Size]);
        Mask m = a < border;
        Vec aa(a);
        COMPARE(aa(m)--, a);
        COMPARE(aa, b);

        aa = a;
        where(m)(aa)--;
        COMPARE(aa, b);

        aa = a;
        --where(m)(aa);
        COMPARE(aa, b);

        COMPARE(--a(m), b);
        COMPARE(a, b);
    }
}
/*}}}*/
template<typename Vec> void testPlusEq()/*{{{*/
{
    VectorMemoryHelper<Vec> mem(2);
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;
    T *data = mem;
    for (int borderI = 0; borderI < Vec::Size; ++borderI) {
        const T border = static_cast<T>(borderI);
        for (int i = 0; i < Vec::Size; ++i) {
            data[i] = static_cast<T>(i + 1);
            data[i + Vec::Size] = data[i] + static_cast<T>(data[i] < border ? 2 : 0);
        }
        Vec a(&data[0]);
        Vec b(&data[Vec::Size]);
        Vec c = a;
        Mask m = a < border;
        COMPARE(a(m) += two<T>(), b);
        COMPARE(a, b);
        where(m) | c += two<T>();
        COMPARE(c, b);
    }
}
/*}}}*/
template<typename Vec> void testMinusEq()/*{{{*/
{
    VectorMemoryHelper<Vec> mem(2);
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;
    T *data = mem;
    for (int borderI = 0; borderI < Vec::Size; ++borderI) {
        const T border = static_cast<T>(borderI);
        for (int i = 0; i < Vec::Size; ++i) {
            data[i] = static_cast<T>(i + 2);
            data[i + Vec::Size] = data[i] - static_cast<T>(data[i] < border ? 2 : 0);
        }
        Vec a(&data[0]);
        Vec b(&data[Vec::Size]);
        Mask m = a < border;

        Vec c = a;
        where(m) | c -= two<T>();
        COMPARE(c, b);

        COMPARE(a(m) -= two<T>(), b);
        COMPARE(a, b);
    }
}
/*}}}*/
template<typename Vec> void testTimesEq()/*{{{*/
{
    VectorMemoryHelper<Vec> mem(2);
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;
    T *data = mem;
    for (int borderI = 0; borderI < Vec::Size; ++borderI) {
        const T border = static_cast<T>(borderI);
        for (int i = 0; i < Vec::Size; ++i) {
            data[i] = static_cast<T>(i);
            data[i + Vec::Size] = data[i] * static_cast<T>(data[i] < border ? 2 : 1);
        }
        Vec a(&data[0]);
        Vec b(&data[Vec::Size]);
        Mask m = a < border;

        Vec c = a;
        where(m) | c *= two<T>();
        COMPARE(c, b);

        COMPARE(a(m) *= two<T>(), b);
        COMPARE(a, b);
    }
}
/*}}}*/
template<typename Vec> void testDivEq()/*{{{*/
{
    VectorMemoryHelper<Vec> mem(2);
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;
    T *data = mem;
    for (int borderI = 0; borderI < Vec::Size; ++borderI) {
        const T border = static_cast<T>(borderI);
        for (int i = 0; i < Vec::Size; ++i) {
            data[i] = static_cast<T>(5 * i);
            data[i + Vec::Size] = data[i] / static_cast<T>(data[i] < border ? 3 : 1);
        }
        Vec a(&data[0]);
        Vec b(&data[Vec::Size]);
        Mask m = a < border;

        Vec c = a;
        where(m) | c /= three<T>();
        COMPARE(c, b);

        COMPARE(a(m) /= three<T>(), b);
        COMPARE(a, b);
    }
}
/*}}}*/
template<typename Vec> void testAssign()/*{{{*/
{
    VectorMemoryHelper<Vec> mem(2);
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;
    T *data = mem;
    for (int borderI = 0; borderI < Vec::Size; ++borderI) {
        const T border = static_cast<T>(borderI);
        for (int i = 0; i < Vec::Size; ++i) {
            data[i] = static_cast<T>(i);
            data[i + Vec::Size] = data[i] + static_cast<T>(data[i] < border ? 2 : 0);
        }
        Vec a(&data[0]);
        Vec b(&data[Vec::Size]);
        Mask m = a < border;

        Vec c = a;
        where(m) | c = b;
        COMPARE(c, b);

        COMPARE(a(m) = b, b);
        COMPARE(a, b);
    }
}
/*}}}*/
template<typename Vec> void testZero()/*{{{*/
{
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;
    typedef typename Vec::IndexType I;

    for (int cut = 0; cut < Vec::Size; ++cut) {
        const Mask mask(I(Vc::IndexesFromZero) < cut);
        //std::cout << mask << std::endl;

        const T aa = 4;
        Vec a(aa);
        Vec b(Vc::Zero);

        where(!mask) | b = a;
        a.setZero(mask);

        COMPARE(a, b);
    }
}
/*}}}*/
template<typename Vec> void testCount()/*{{{*/
{
    typedef typename Vec::EntryType T;
    typedef typename Vec::IndexType I;
    typedef typename Vec::Mask M;

    for_all_masks(Vec, m) {
        unsigned int count = 0;
        for (int i = 0; i < Vec::Size; ++i) {
            if (m[i]) {
                ++count;
            }
        }
        COMPARE(m.count(), count) << ", m = " << m;
    }
}
/*}}}*/
template<typename Vec> void testFirstOne()/*{{{*/
{
    typedef typename Vec::EntryType T;
    typedef typename Vec::IndexType I;
    typedef typename Vec::Mask M;

    for (unsigned int i = 0; i < Vec::Size; ++i) {
        const M mask(I(Vc::IndexesFromZero) == i);
        COMPARE(mask.firstOne(), i);
    }
}
/*}}}*/
#ifdef VC_IMPL_SSE
void testFloat8GatherMask()/*{{{*/
{
    Vc::Memory<short_v, short_v::Size * 256> data;
    short_v::Memory andMemory;
    for (int i = 0; i < short_v::Size; ++i) {
        andMemory[i] = 1 << i;
    }
    const short_v andMask(andMemory);

    for (unsigned int i = 0; i < data.vectorsCount(); ++i) {
        data.vector(i) = andMask & i;
    }

    for (unsigned int i = 0; i < data.vectorsCount(); ++i) {
        const Vc::short_m mask = data.vector(i) == short_v::Zero();

        Vc::SSE::Float8GatherMask
            gatherMaskA(mask),
            gatherMaskB(static_cast<Vc::sfloat_m>(mask));
        COMPARE(gatherMaskA.toInt(), gatherMaskB.toInt());
    }
}/*}}}*/
#endif

template<typename V> void maskReductions()/*{{{*/
{
    for_all_masks(V, mask) {
        COMPARE(all_of(mask), mask.count() == V::Size);
        if (mask.count() > 0) {
            VERIFY(any_of(mask));
            VERIFY(!none_of(mask));
            COMPARE(some_of(mask), mask.count() < V::Size);
        } else {
            VERIFY(!any_of(mask));
            VERIFY(none_of(mask));
            VERIFY(!some_of(mask));
        }
    }
}/*}}}*/
template<typename V> void maskInit()/*{{{*/
{
    typedef typename V::Mask M;
    COMPARE(M(Vc::One), M(true));
    COMPARE(M(Vc::Zero), M(false));
}
/*}}}*/
template<typename V> void maskCompare()/*{{{*/
{
    int i = 0;
    auto m0 = allMasks<V>(i);
    auto m1 = allMasks<V>(i);
    while (any_of(m0)) {
        ++i;
        VERIFY(m0 == m1);
        m0 = allMasks<V>(i);
        VERIFY(m0 != m1);
        m1 = allMasks<V>(i);
    }
}/*}}}*/
int main(int argc, char **argv)/*{{{*/
{
    initTest(argc, argv);

    testAllTypes(maskInit);
    testAllTypes(maskCompare);
    testAllTypes(testInc);
    testAllTypes(testDec);
    testAllTypes(testPlusEq);
    testAllTypes(testMinusEq);
    testAllTypes(testTimesEq);
    testAllTypes(testDivEq);
    testAllTypes(testAssign);
    testAllTypes(testZero);
    testAllTypes(testCount);
    testAllTypes(testFirstOne);
    testAllTypes(maskReductions);

#ifdef VC_IMPL_SSE
    runTest(testFloat8GatherMask);
#endif

    return 0;
}/*}}}*/

// vim: foldmethod=marker
