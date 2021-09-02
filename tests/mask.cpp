/*  This file is part of the Vc library. {{{
Copyright © 2009-2015 Matthias Kretz <kretz@kde.org>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the names of contributing organizations nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#include "unittest.h"
#include <iostream>
#include "vectormemoryhelper.h"
#include <cmath>

using Vc::float_m;
using Vc::double_m;
using Vc::int_m;
using Vc::uint_m;
using Vc::short_m;
using Vc::ushort_m;

template<typename T> T two() { return T(2); }
template<typename T> T three() { return T(3); }

TEST_TYPES(Vec, testInc, AllVectors) /*{{{*/
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
TEST_TYPES(Vec, testDec, AllVectors) /*{{{*/
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
TEST_TYPES(Vec, testPlusEq, AllVectors) /*{{{*/
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
        a(m) += two<T>();
        COMPARE(a, b);
        where(m) | c += two<T>();
        COMPARE(c, b);
    }
}
/*}}}*/
TEST_TYPES(Vec, testMinusEq, AllVectors) /*{{{*/
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

        a(m) -= two<T>();
        COMPARE(a, b);
    }
}
/*}}}*/
TEST_TYPES(Vec, testTimesEq, AllVectors) /*{{{*/
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

        a(m) *= two<T>();
        COMPARE(a, b);
    }
}
/*}}}*/
TEST_TYPES(Vec, testDivEq, AllVectors) /*{{{*/
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

        a(m) /= three<T>();
        COMPARE(a, b);
    }
}
/*}}}*/
TEST_TYPES(Vec, testAssign, AllVectors) /*{{{*/
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

        a(m) = b;
        COMPARE(a, b);
    }
}
/*}}}*/
TEST_TYPES(Vec, testZero, AllVectors) /*{{{*/
{
    typedef typename Vec::EntryType T;
    typedef typename Vec::Mask Mask;
    typedef typename Vec::IndexType I;

    for (int cut = 0; cut < int(Vec::Size); ++cut) {
        const Mask mask = Vc::simd_cast<Mask>(I([](int n) { return n; }) < cut);
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
TEST_TYPES(V, testIntegerConversion, AllVectors) /*{{{*/
{
    withRandomMask<V>([](typename V::Mask m) {
        auto bit = m.toInt();
        for (size_t i = 0; i < m.Size; ++i) {
            COMPARE(!!((bit >> i) & 1), m[i]) << std::hex << bit;
        }
    });
}
/*}}}*/
TEST_TYPES(Vec, testCount, AllVectors) /*{{{*/
{
    typedef typename Vec::Mask M;

    withRandomMask<Vec>([](M m) {
        int count = 0;
        for (size_t i = 0; i < Vec::Size; ++i) {
            if (m[i]) {
                ++count;
            }
        }
        COMPARE(m.count(), count) << ", m = " << m;
    });
}
/*}}}*/
TEST_TYPES(Vec, testFirstOne, AllVectors) /*{{{*/
{
    typedef typename Vec::IndexType I;
    typedef typename Vec::Mask M;

    for (unsigned int i = 0; i < Vec::Size; ++i) {
        const M mask = Vc::simd_cast<M>(I(Vc::IndexesFromZero) == i);
        COMPARE(mask.firstOne(), int(i)) << mask << ' ' << I([](int n) { return n; })
                                         << ' ' << (I([](int n) { return n; }) == i);
    }
}
/*}}}*/
TEST_TYPES(V, shifted, concat<AllVectors, SimdArrays<16>, OddSimdArrays<31>>) /*{{{*/
{
    using M = typename V::Mask;
    withRandomMask<V>([](const M &reference) {
        constexpr int Size = V::Size;
        for (int shift = -2 * Size; shift <= 2 * Size; ++shift) {
            const M test = reference.shifted(shift);
            for (int i = 0; i < Size; ++i) {
                if (i + shift >= 0 && i + shift < Size) {
                    COMPARE(test[i], reference[i + shift])
                        << "shift: " << shift << ", i: " << i << ", test: " << test
                        << ", reference: " << reference;
                } else {
                    COMPARE(test[i], false) << "shift: " << shift << ", i: " << i
                                            << ", test: " << test
                                            << ", reference: " << reference;
                }
            }
        }
    });
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
TEST(testBinaryOperators) /*{{{*/
{
    testBinaryOperatorsImpl< short_m,  short_m>();
    testBinaryOperatorsImpl< short_m, ushort_m>();
    testBinaryOperatorsImpl<ushort_m,  short_m>();
    testBinaryOperatorsImpl<ushort_m, ushort_m>();

    testBinaryOperatorsImpl<   int_m,    int_m>();
    testBinaryOperatorsImpl<   int_m,   uint_m>();
    testBinaryOperatorsImpl<  uint_m,    int_m>();
    testBinaryOperatorsImpl<  uint_m,   uint_m>();

    testBinaryOperatorsImpl< float_m,  float_m>();

    testBinaryOperatorsImpl<double_m, double_m>();
}
/*}}}*/

TEST_TYPES(V, maskReductions, AllVectors) /*{{{*/
{
    withRandomMask<V>([](typename V::Mask mask) {
        constexpr decltype(mask.count()) size = V::Size;
        COMPARE(all_of(mask), mask.count() == size);
        if (mask.count() > 0) {
            VERIFY(any_of(mask));
            VERIFY(!none_of(mask));
            COMPARE(some_of(mask), mask.count() < size);
        } else {
            VERIFY(!any_of(mask));
            VERIFY(none_of(mask));
            VERIFY(!some_of(mask));
        }
    });
}/*}}}*/
TEST_TYPES(V, maskInit, AllVectors) /*{{{*/
{
    typedef typename V::Mask M;
    COMPARE(M(Vc::One), M(true));
    COMPARE(M(Vc::Zero), M(false));
}
/*}}}*/
TEST_TYPES(V, maskScalarAccess, AllVectors) /*{{{*/
{
    typedef typename V::Mask M;
    withRandomMask<V>([](M mask) {
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
    });
}/*}}}*/
template<typename MTo, typename MFrom> void testMaskConversion(const MFrom &m)/*{{{*/
{
    MTo test = Vc::simd_cast<MTo>(m);
    size_t i = 0;
    for (; i < std::min(m.Size, test.Size); ++i) {
        COMPARE(test[i], m[i]) << i << " conversion from " << vir::typeToString<MFrom>()
                               << " to " << vir::typeToString<MTo>();
    }
    for (; i < test.Size; ++i) {
        COMPARE(test[i], false) << i << " conversion from " << vir::typeToString<MFrom>()
                                << " to " << vir::typeToString<MTo>();
    }
}/*}}}*/
TEST_TYPES(V, maskConversions, AllVectors) /*{{{*/
{
    typedef typename V::Mask M;
    withRandomMask<V>([](M m) {
        testMaskConversion< float_m>(m);
        testMaskConversion<double_m>(m);
        testMaskConversion<   int_m>(m);
        testMaskConversion<  uint_m>(m);
        testMaskConversion< short_m>(m);
        testMaskConversion<ushort_m>(m);
    });
}
/*}}}*/
TEST_TYPES(V, boolConversion, AllVectors) /*{{{*/
{
    alignas(16) bool mem[V::Size + 64];
    withRandomMask<V>([&](typename V::Mask m) {
        bool *ptr = mem;
        m.store(ptr);
        for (size_t i = 0; i < V::Size; ++i) {
            COMPARE(ptr[i], m[i]) << "offset: " << ptr - mem;
        }

        typename V::Mask m2(ptr);
        COMPARE(m2, m) << "offset: " << ptr - mem;
        for (++ptr; ptr < &mem[64]; ++ptr) {
            m.store(ptr, Vc::Unaligned);
            for (size_t i = 0; i < V::Size; ++i) {
                COMPARE(ptr[i], m[i]) << "offset: " << ptr - mem;
            }

            typename V::Mask m3(ptr, Vc::Unaligned);
            COMPARE(m3, m) << "offset: " << ptr - mem;
        }
    });
}
/*}}}*/
TEST_TYPES(V, testCompareOperators, AllVectors) /*{{{*/
{
    typedef typename V::Mask M;
    const M a(true);
    const M b(false);
    VERIFY(!(a == b));
    VERIFY(!(b == a));
    VERIFY(a != b);
    VERIFY(b != a);

    for_all_masks(V, k)
    {
        M randomMask;
        do {
            randomMask = V::Random() < V::Random();
        } while (randomMask.isEmpty());
        const M k2 = k ^ randomMask;

        VERIFY( (k  == k )) << k;
        VERIFY(!(k2 == k )) << k << k2;
        VERIFY(!(k  == k2)) << k << k2;
        VERIFY( (k2 == k2)) << k << k2;

        VERIFY(!(k  != k )) << k;
        VERIFY( (k  != k2)) << k << k2;
        VERIFY( (k2 != k )) << k << k2;
        VERIFY(!(k2 != k2)) << k << k2;
    }
}

TEST_TYPES(V, testMaskDefined, AllVectors)
{
    V a(1);
    V b(2);
    auto r = a < b;
    auto s = a > b;
    VERIFY(r[0] == true);
    VERIFY(s[0] == false);
}

/*}}}*/

// vim: foldmethod=marker
