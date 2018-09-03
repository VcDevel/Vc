/*  This file is part of the Vc library. {{{
Copyright © 2010-2015 Matthias Kretz <kretz@kde.org>

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
#include <limits>

using Vc::float_v;
using Vc::double_v;
using Vc::int_v;
using Vc::uint_v;
using Vc::short_v;
using Vc::ushort_v;

/*
 *   V \  M | float | double | ushort | short | uint | int
 * ---------+----------------------------------------------
 *  float_v |   X   |        |    X   |   X   |      |
 * double_v |       |    X   |        |       |      |
 *    int_v |       |        |        |   X   |      |  X
 *   uint_v |       |        |    X   |       |   X  |
 *  short_v |       |        |        |   X   |      |
 * ushort_v |       |        |    X   |       |      |
 */

TEST_TYPES(Pair, testDeinterleave,
           vir::Typelist<float_v, float>,
           vir::Typelist<float_v, unsigned short>,
           vir::Typelist<float_v, short>,
           vir::Typelist<double_v, double>,
           vir::Typelist<int_v, int>,
           vir::Typelist<int_v, short>,
           vir::Typelist<uint_v, unsigned int>,
           vir::Typelist<uint_v, unsigned short>,
           vir::Typelist<short_v, short>,
           vir::Typelist<ushort_v, unsigned short>)
{
    typedef typename Pair::template at<0> V;
    typedef typename Pair::template at<1> M;

    const bool isSigned = std::numeric_limits<M>::is_signed;

    const typename V::EntryType offset = isSigned ? -512 : 0;
    const V _0246 = V([](int n) { return n * 2; }) + offset;

    M memory[1024];
    for (int i = 0; i < 1024; ++i) {
        memory[i] = static_cast<M>(i + offset);
    }

    V a, b;

    static_assert(2 * V::Size < 1024, "fix the test, please");
    for (int i = 0; i < 1024 - 2 * int(V::Size); ++i) {
        if (reinterpret_cast<std::uintptr_t>(&memory[i]) & (Vc::VectorAlignment - 1)) {
            Vc::deinterleave(&a, &b, &memory[i], Vc::Unaligned);
        } else {
            Vc::deinterleave(&a, &b, &memory[i]);
        }
        COMPARE(a, _0246 + i);
        COMPARE(b, _0246 + i + 1);
    }
}
