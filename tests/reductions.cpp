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
#include "vectormemoryhelper.h"

TEST_TYPES(Vec, testReduceMin, concat<AllVectors, SimdArrayList>) //{{{1
{
    typedef typename Vec::EntryType T;
    const T one = 1;
    VectorMemoryHelper<Vec> mem(Vec::Size);
    T *data = mem;
    for (size_t i = 0; i < Vec::Size * Vec::Size; ++i) {
        data[i] = i % (Vec::Size + 1) + one;
    }
    for (size_t i = 0; i < Vec::Size; ++i, data += Vec::Size) {
        const Vec a(&data[0]);
        //std::cout << a << std::endl;
        COMPARE(a.min(), one);
    }
}

TEST_TYPES(Vec, testReduceMax, concat<AllVectors, SimdArrayList>) //{{{1
{
    typedef typename Vec::EntryType T;
    const T max = Vec::Size + 1;
    VectorMemoryHelper<Vec> mem(Vec::Size);
    T *data = mem;
    for (size_t i = 0; i < Vec::Size * Vec::Size; ++i) {
        data[i] = (i + Vec::Size) % (Vec::Size + 1) + 1;
    }
    for (size_t i = 0; i < Vec::Size; ++i, data += Vec::Size) {
        const Vec a(&data[0]);
        //std::cout << a << std::endl;
        COMPARE(a.max(), max);
    }
}

TEST_TYPES(V, testReduceProduct, concat<AllVectors, SimdArrayList>) //{{{1
{
    using T = typename V::EntryType;
    V test = 0;
    COMPARE(test.product(), T(0)) << "test = " << test;
    test = 1;
    COMPARE(test.product(), T(1)) << "test = " << test;
    test[0] = 2;
    COMPARE(test.product(), T(2)) << "test = " << test;
    test[0] = 3;
    COMPARE(test.product(), T(3)) << "test = " << test;

    for (std::size_t i = 0; i + 1 < V::size(); ++i) {
        test[i] = 1;
        test[i + 1] = 5;
        COMPARE(test.product(), T(5)) << "test = " << test << ", i = " << i;
    }
    for (std::size_t i = 0; i + 2 < V::size(); ++i) {
        test[i] = 1;
        test[i + 1] = 7;
        COMPARE(test.product(), T(5 * 7)) << "test = " << test << ", i = " << i;
    }
}

TEST_TYPES(Vec, testReduceSum, concat<AllVectors, SimdArrayList>) //{{{1
{
    typedef typename Vec::EntryType T;
    int _sum = 1;
    for (size_t i = 2; i <= Vec::Size; ++i) {
        _sum += i;
    }
    const T sum = _sum;
    VectorMemoryHelper<Vec> mem(Vec::Size);
    T *data = mem;
    for (size_t i = 0; i < Vec::Size * Vec::Size; ++i) {
        data[i] = (i + i / Vec::Size) % Vec::Size + 1;
    }
    for (size_t i = 0; i < Vec::Size; ++i, data += Vec::Size) {
        const Vec a(&data[0]);
        //std::cout << a << std::endl;
        COMPARE(a.sum(), sum);
    }
}

//}}}1
// vim: foldmethod=marker
