/*  This file is part of the Vc library. {{{
Copyright © 2010-2015 Matthias Kretz <kretz@kde.org>
All rights reserved.

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

#include "unittest-old.h"

using namespace Vc;

template<typename V> void reads()
{
    typedef typename V::EntryType T;

    V a = V::Zero();
    const T zero = 0;
    for (size_t i = 0; i < V::Size; ++i) {
        const T x = a[i];
        COMPARE(x, zero);
    }
    a = V::IndexesFromZero();
    for (size_t i = 0; i < V::Size; ++i) {
        const T x = a[i];
        const T y = i;
        COMPARE(x, y);
    }
}

template<typename V, size_t Index>
inline void readsConstantIndexTest(Vc_ALIGNED_PARAMETER(V) a, Vc_ALIGNED_PARAMETER(V) b)
{
    typedef typename V::EntryType T;
    {
        const T x = a[Index];
        const T zero = 0;
        COMPARE(x, zero) << Index;
    }{
        const T x = b[Index];
        const T y = Index;
        COMPARE(x, y) << Index;
    }
}

template<typename V, size_t Index>
struct ReadsConstantIndex
{
    ReadsConstantIndex(Vc_ALIGNED_PARAMETER(V) a, Vc_ALIGNED_PARAMETER(V) b)
    {
        readsConstantIndexTest<V, Index>(a, b);
        ReadsConstantIndex<V, Index - 1>(a, b);
    }
};


template<typename V>
struct ReadsConstantIndex<V, 0>
{
    ReadsConstantIndex(Vc_ALIGNED_PARAMETER(V) a, Vc_ALIGNED_PARAMETER(V) b)
    {
        readsConstantIndexTest<V, 0>(a, b);
    }
};

template<typename V> void readsConstantIndex()
{
    V a = V::Zero();
    V b = V::IndexesFromZero();
    ReadsConstantIndex<V, V::Size - 1>(a, b);
}

template<typename V> void writes()
{
    typedef typename V::EntryType T;

    V a;
    for (size_t i = 0; i < V::Size; ++i) {
        a[i] = static_cast<T>(i);
    }
    V b = V::IndexesFromZero();
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
        // expected: [0, 1, 5, 5, 5, 5, 6, 7]
        COMPARE(a, b);
    } else if (V::Size == 2) { // a = [0, 1]; b = [0, 1]
        a(a == 0) += two;      // a = [2, 1]
        a[1] += one;           // a = [2, 2]
        a(a == 2) += one;      // a = [3, 3]
        b(b == 0) += one;      // b = [1, 1]
        b(b == 1) += one;      // b = [2, 2]
        b(b == 2) += one;      // b = [3, 3]
        COMPARE(a, b);
    } else {
        FAIL() << "unsupported Vector::Size";
    }
}

void testmain()
{
    testAllTypes(reads);
    testAllTypes(writes);
    testAllTypes(readsConstantIndex);
    //testAllTypes(writesConstantIndex);
}
