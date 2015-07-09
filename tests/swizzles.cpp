/*  This file is part of the Vc library. {{{
Copyright © 2012-2014 Matthias Kretz <kretz@kde.org>
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

enum Swizzle {
    BADC, CDAB, AAAA, BBBB, CCCC, DDDD, BCAD, BCDA, DABC, ACBD, DBCA, DCBA
};

template<typename V> V scalarSwizzle(VC_ALIGNED_PARAMETER(V) v, Swizzle s)
{
    V r = v;
    for (size_t i = 0; i + 4 <= V::Size; i += 4) {
        switch (s) {
        case BADC:
            r[i + 0] = v[i + 1];
            r[i + 1] = v[i + 0];
            r[i + 2] = v[i + 3];
            r[i + 3] = v[i + 2];
            break;
        case CDAB:
            r[i + 0] = v[i + 2];
            r[i + 1] = v[i + 3];
            r[i + 2] = v[i + 0];
            r[i + 3] = v[i + 1];
            break;
        case AAAA:
            r[i + 0] = v[i + 0];
            r[i + 1] = v[i + 0];
            r[i + 2] = v[i + 0];
            r[i + 3] = v[i + 0];
            break;
        case BBBB:
            r[i + 0] = v[i + 1];
            r[i + 1] = v[i + 1];
            r[i + 2] = v[i + 1];
            r[i + 3] = v[i + 1];
            break;
        case CCCC:
            r[i + 0] = v[i + 2];
            r[i + 1] = v[i + 2];
            r[i + 2] = v[i + 2];
            r[i + 3] = v[i + 2];
            break;
        case DDDD:
            r[i + 0] = v[i + 3];
            r[i + 1] = v[i + 3];
            r[i + 2] = v[i + 3];
            r[i + 3] = v[i + 3];
            break;
        case BCAD:
            r[i + 0] = v[i + 1];
            r[i + 1] = v[i + 2];
            r[i + 2] = v[i + 0];
            r[i + 3] = v[i + 3];
            break;
        case BCDA:
            r[i + 0] = v[i + 1];
            r[i + 1] = v[i + 2];
            r[i + 2] = v[i + 3];
            r[i + 3] = v[i + 0];
            break;
        case DABC:
            r[i + 0] = v[i + 3];
            r[i + 1] = v[i + 0];
            r[i + 2] = v[i + 1];
            r[i + 3] = v[i + 2];
            break;
        case ACBD:
            r[i + 0] = v[i + 0];
            r[i + 1] = v[i + 2];
            r[i + 2] = v[i + 1];
            r[i + 3] = v[i + 3];
            break;
        case DBCA:
            r[i + 0] = v[i + 3];
            r[i + 1] = v[i + 1];
            r[i + 2] = v[i + 2];
            r[i + 3] = v[i + 0];
            break;
        case DCBA:
            r[i + 0] = v[i + 3];
            r[i + 1] = v[i + 2];
            r[i + 2] = v[i + 1];
            r[i + 3] = v[i + 0];
            break;
        }
    }
    return r;
}

template<typename V> void testSwizzle()
{
    for (int i = 0; i < 100; ++i) {
        const V test = V::Random();
        COMPARE(test.abcd(), test);
        COMPARE(test.badc(), scalarSwizzle(test, BADC));
        COMPARE(test.cdab(), scalarSwizzle(test, CDAB));
        COMPARE(test.aaaa(), scalarSwizzle(test, AAAA));
        COMPARE(test.bbbb(), scalarSwizzle(test, BBBB));
        COMPARE(test.cccc(), scalarSwizzle(test, CCCC));
        COMPARE(test.dddd(), scalarSwizzle(test, DDDD));
        COMPARE(test.bcad(), scalarSwizzle(test, BCAD));
        COMPARE(test.bcda(), scalarSwizzle(test, BCDA));
        COMPARE(test.dabc(), scalarSwizzle(test, DABC));
        COMPARE(test.acbd(), scalarSwizzle(test, ACBD));
        COMPARE(test.dbca(), scalarSwizzle(test, DBCA));
        COMPARE(test.dcba(), scalarSwizzle(test, DCBA));
    }
}

void testmain()
{
#if VC_DOUBLE_V_SIZE >= 4 || VC_DOUBLE_V_SIZE == 1
    runTest(testSwizzle<double_v>);
#endif
    runTest(testSwizzle<float_v>);
    runTest(testSwizzle<int_v>);
    runTest(testSwizzle<uint_v>);
    runTest(testSwizzle<short_v>);
    runTest(testSwizzle<ushort_v>);
}
