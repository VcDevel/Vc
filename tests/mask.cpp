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
using Vc::int_v;
using Vc::uint_v;
using Vc::short_v;
using Vc::ushort_v;

using Vc::float_m;
using Vc::double_m;
using Vc::int_m;
using Vc::uint_m;
using Vc::short_m;
using Vc::ushort_m;

template<typename T> T two() { return T(2); }
template<typename T> T three() { return T(3); }

template<typename Vec> void testInc()/*{{{*/
{
    VectorMemoryHelper<Vec> mem(2);
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;
    T *data = mem;
    for (size_t borderI = 0; borderI < Vec::Size; ++borderI) {/*{{{*/
        const T border = static_cast<T>(borderI);
        for (size_t i = 0; i < Vec::Size; ++i) {
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
    for (size_t borderI = 0; borderI < Vec::Size; ++borderI) {
        const T border = static_cast<T>(borderI);
        for (size_t i = 0; i < Vec::Size; ++i) {
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
    for (size_t borderI = 0; borderI < Vec::Size; ++borderI) {
        const T border = static_cast<T>(borderI);
        for (size_t i = 0; i < Vec::Size; ++i) {
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
    for (size_t borderI = 0; borderI < Vec::Size; ++borderI) {
        const T border = static_cast<T>(borderI);
        for (size_t i = 0; i < Vec::Size; ++i) {
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
    for (size_t borderI = 0; borderI < Vec::Size; ++borderI) {
        const T border = static_cast<T>(borderI);
        for (size_t i = 0; i < Vec::Size; ++i) {
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
    for (size_t borderI = 0; borderI < Vec::Size; ++borderI) {
        const T border = static_cast<T>(borderI);
        for (size_t i = 0; i < Vec::Size; ++i) {
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
    for (size_t borderI = 0; borderI < Vec::Size; ++borderI) {
        const T border = static_cast<T>(borderI);
        for (size_t i = 0; i < Vec::Size; ++i) {
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
    for (size_t borderI = 0; borderI < Vec::Size; ++borderI) {
        const T border = static_cast<T>(borderI);
        for (size_t i = 0; i < Vec::Size; ++i) {
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

    for (size_t cut = 0; cut < Vec::Size; ++cut) {
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
        for (size_t i = 0; i < Vec::Size; ++i) {
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

template<typename M1, typename M2> void testLogicalOperatorsImpl()/*{{{*/
{
    VERIFY((M1(true) && M2(true)).isFull());
    VERIFY((M1(true) && M2(false)).isEmpty());
    VERIFY((M1(true) || M2(true)).isFull());
    VERIFY((M1(true) || M2(false)).isFull());
    VERIFY((M1(false) || M2(false)).isEmpty());
}
/*}}}*/
template<typename M1, typename M2> void testBinaryOperatorsImpl()/*{{{*/
{
    testLogicalOperatorsImpl<M1, M2>();

    VERIFY((M1(true) & M2(true)).isFull());
    VERIFY((M1(true) & M2(false)).isEmpty());
    VERIFY((M1(true) | M2(true)).isFull());
    VERIFY((M1(true) | M2(false)).isFull());
    VERIFY((M1(false) | M2(false)).isEmpty());
    VERIFY((M1(true) ^ M2(true)).isEmpty());
    VERIFY((M1(true) ^ M2(false)).isFull());
}
/*}}}*/
void testBinaryOperators()/*{{{*/
{
    testBinaryOperatorsImpl< short_m,  short_m>();
    testBinaryOperatorsImpl< short_m, ushort_m>();
    testBinaryOperatorsImpl<ushort_m,  short_m>();
    testBinaryOperatorsImpl<ushort_m, ushort_m>();

    testBinaryOperatorsImpl<   int_m,    int_m>();
    testBinaryOperatorsImpl<   int_m,   uint_m>();
    testBinaryOperatorsImpl<   int_m,  float_m>();
    testBinaryOperatorsImpl<  uint_m,    int_m>();
    testBinaryOperatorsImpl<  uint_m,   uint_m>();
    testBinaryOperatorsImpl<  uint_m,  float_m>();
    testBinaryOperatorsImpl< float_m,    int_m>();
    testBinaryOperatorsImpl< float_m,   uint_m>();
    testBinaryOperatorsImpl< float_m,  float_m>();

    testBinaryOperatorsImpl<double_m, double_m>();
}
/*}}}*/

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
template<typename V> void maskScalarAccess()/*{{{*/
{
    typedef typename V::Mask M;
    for_all_masks(V, mask) {
        const auto &mask2 = mask;
        for (size_t i = 0; i < V::Size; ++i) {
            COMPARE(bool(mask[i]), mask2[i]);
        }

        const auto maskInv = !mask;
        for (size_t i = 0; i < V::Size; ++i) {
            mask[i] = !mask[i];
        }
        COMPARE(mask, maskInv);

        for (size_t i = 0; i < V::Size; ++i) {
            mask[i] = true;
        }
        COMPARE(mask, M(true));
    }
}/*}}}*/
template<typename T> constexpr const char *typeName();
template<> constexpr const char *typeName< float_m>() { return "float_m"; }
template<> constexpr const char *typeName<double_m>() { return "double_m"; }
template<> constexpr const char *typeName<   int_m>() { return "int_m"; }
template<> constexpr const char *typeName<  uint_m>() { return "uint_m"; }
template<> constexpr const char *typeName< short_m>() { return "short_m"; }
template<> constexpr const char *typeName<ushort_m>() { return "ushort_m"; }
template<typename MTo, typename MFrom> void testMaskConversion(const MFrom &m)
{
    MTo test(m);
    size_t i = 0;
    for (; i < std::min(m.Size, test.Size); ++i) {
        COMPARE(test[i], m[i]) << i << " conversion from " << typeName<MFrom>() << " to " << typeName<MTo>();
    }
    for (; i < test.Size; ++i) {
        COMPARE(test[i], false) << i << " conversion from " << typeName<MFrom>() << " to " << typeName<MTo>();
    }
}
template<typename V> void maskConversions()
{
    typedef typename V::Mask M;
    for_all_masks(V, m) {
        testMaskConversion< float_m>(m);
        testMaskConversion<double_m>(m);
        testMaskConversion<   int_m>(m);
        testMaskConversion<  uint_m>(m);
        testMaskConversion< short_m>(m);
        testMaskConversion<ushort_m>(m);
    }
}

template<typename V> void testIntegerConversion()
{
    for_all_masks(V, m) {
        auto bit = m.toInt();
        for (size_t i = 0; i < m.Size; ++i) {
            COMPARE(!!((bit >> i) & 1), m[i]);
        }
    }
}

void testmain()/*{{{*/
{
    testAllTypes(maskInit);
    testAllTypes(maskScalarAccess);
    testAllTypes(maskCompare);
    testAllTypes(testInc);
    testAllTypes(testDec);
    testAllTypes(testPlusEq);
    testAllTypes(testMinusEq);
    testAllTypes(testTimesEq);
    testAllTypes(testDivEq);
    testAllTypes(testAssign);
    testAllTypes(testZero);
    testAllTypes(testIntegerConversion);
    testAllTypes(testCount);
    testAllTypes(testFirstOne);
    testAllTypes(maskReductions);
    runTest(testBinaryOperators);
    testAllTypes(maskConversions);
}/*}}}*/

// vim: foldmethod=marker
