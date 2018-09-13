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

//#define UNITTEST_ONLY_XTEST 1
#include "unittest.h"
#include "mathreference.h"

using Vc::Detail::doubleConstant;
using vir::test::setFuzzyness;

TEST_TYPES(V, testSincos, RealTypes) //{{{1
{
    typedef typename V::EntryType T;
    setFuzzyness<float>(2);
    setFuzzyness<double>(3);
    Array<SincosReference<T> > reference = sincosReference<T>();
    for (size_t i = 0; i + V::Size - 1 < reference.size; i += V::Size) {
        V x, sref, cref;
        for (size_t j = 0; j < V::Size; ++j) {
            x[j] = reference.data[i + j].x;
            sref[j] = reference.data[i + j].s;
            cref[j] = reference.data[i + j].c;
        }
        V sin, cos;
        Vc::sincos(x, &sin, &cos);
        FUZZY_COMPARE(sin, sref) << " x = " << x << ", i = " << i;
        FUZZY_COMPARE(cos, cref) << " x = " << x << ", i = " << i;
        Vc::sincos(-x, &sin, &cos);
        FUZZY_COMPARE(sin, -sref) << " x = " << -x << ", i = " << i;
        FUZZY_COMPARE(cos, cref) << " x = " << -x << ", i = " << i;
    }
}

TEST_TYPES(V, testSin, RealTypes) //{{{1
{
    typedef typename V::EntryType T;
    setFuzzyness<float>(2);
    setFuzzyness<double>(3);
    Array<SincosReference<T> > reference = sincosReference<T>();
    for (size_t i = 0; i + V::Size - 1 < reference.size; i += V::Size) {
        V x, sref;
        for (size_t j = 0; j < V::Size; ++j) {
            x[j] = reference.data[i + j].x;
            sref[j] = reference.data[i + j].s;
        }
        //std::cout << std::setprecision(30);
        //std::cout << std::defaultfloat << "testing sin(" << x << ") = " << sref << "\n";
        FUZZY_COMPARE(Vc::sin(x), sref) << " x = " << x << ", i = " << i
#if defined Vc_GCC && Vc_GCC >= 0x50100
            << std::hexfloat
            << "\n input:" << x
            << "\n    Vc:" << Vc::sin(x)
            << "\n   ref:" << sref
            << std::defaultfloat
#endif
            ;
        FUZZY_COMPARE(Vc::sin(-x), -sref) << " x = " << x << ", i = " << i;
    }
}

TEST_TYPES(V, testCos, RealTypes) //{{{1
{
    typedef typename V::EntryType T;
    setFuzzyness<float>(2);
    setFuzzyness<double>(3);
    Array<SincosReference<T> > reference = sincosReference<T>();
    for (size_t i = 0; i + V::Size - 1 < reference.size; i += V::Size) {
        V x, cref;
        for (size_t j = 0; j < V::Size; ++j) {
            x[j] = reference.data[i + j].x;
            cref[j] = reference.data[i + j].c;
        }
        FUZZY_COMPARE(Vc::cos(x), cref) << " x = " << x << ", i = " << i;
        FUZZY_COMPARE(Vc::cos(-x), cref) << " x = " << x << ", i = " << i;
    }
}

TEST_TYPES(V, testAsin, RealTypes) //{{{1
{
    typedef typename V::EntryType T;
    setFuzzyness<float>(2);
    setFuzzyness<double>(36);
    Array<Reference<T> > reference = referenceData<T, Asin>();
    for (size_t i = 0; i + V::Size - 1 < reference.size; i += V::Size) {
        V x, ref;
        for (size_t j = 0; j < V::Size; ++j) {
            x[j] = reference.data[i + j].x;
            ref[j] = reference.data[i + j].ref;
        }
        FUZZY_COMPARE(Vc::asin(x), ref) << " x = " << x << ", i = " << i;
        FUZZY_COMPARE(Vc::asin(-x), -ref) << " -x = " << -x << ", i = " << i;
    }
}

const union {
    unsigned int hex;
    float value;
} INF = { 0x7f800000 };

#if defined(__APPLE__)
// On Mac OS X, std::atan and std::atan2 don't return the special values defined in the
// Linux man pages (whether it's C99 or POSIX.1-2001 that specifies the special values is
// not mentioned in the man page). This issue is only relevant for VectorAbi::Scalar. But
// since the SimdArray types can inclue scalar Vector objects in the odd variants, the
// tests must use fuzzy compares in all cases on Mac OS X.
#define ATAN_COMPARE FUZZY_COMPARE
#else
#define ATAN_COMPARE COMPARE
#endif

TEST_TYPES(V, testAtan, RealTypes) //{{{1
{
    typedef typename V::EntryType T;
    setFuzzyness<float>(3);
    setFuzzyness<double>(2);

    {
        const V Pi_2 = T(doubleConstant<1, 0x921fb54442d18ull,  0>());
        V nan; nan.setQnan();
        const V inf = T(INF.value);

        VERIFY(all_of(Vc::isnan(Vc::atan(nan))));
        ATAN_COMPARE(Vc::atan(+inf), +Pi_2);
#ifdef Vc_MSVC
#pragma warning(suppress: 4756) // overflow in constant arithmetic
#endif
        ATAN_COMPARE(Vc::atan(-inf), -Pi_2);
    }

    Array<Reference<T> > reference = referenceData<T, Atan>();
    for (size_t i = 0; i + V::Size - 1 < reference.size; i += V::Size) {
        V x, ref;
        for (size_t j = 0; j < V::Size; ++j) {
            x[j] = reference.data[i + j].x;
            ref[j] = reference.data[i + j].ref;
        }
        FUZZY_COMPARE(Vc::atan(x), ref) << " x = " << x << ", i = " << i;
        FUZZY_COMPARE(Vc::atan(-x), -ref) << " -x = " << -x << ", i = " << i;
    }
}

TEST_TYPES(V, testAtan2, RealTypes) //{{{1
{
    typedef typename V::EntryType T;
    setFuzzyness<float>(3);
    setFuzzyness<double>(2);

    {
        const V Pi   = T(doubleConstant<1, 0x921fb54442d18ull,  1>());
        const V Pi_2 = T(doubleConstant<1, 0x921fb54442d18ull,  0>());
        V nan; nan.setQnan();
        const V inf = T(INF.value);

        // If y is +0 (-0) and x is less than 0, +pi (-pi) is returned.
        ATAN_COMPARE(Vc::atan2(V(T(+0.)), V(T(-3.))), +Pi);
        ATAN_COMPARE(Vc::atan2(V(T(-0.)), V(T(-3.))), -Pi);
        // If y is +0 (-0) and x is greater than 0, +0 (-0) is returned.
        COMPARE(Vc::atan2(V(T(+0.)), V(T(+3.))), V(T(+0.)));
        VERIFY(none_of(isnegative(Vc::atan2(V(T(+0.)), V(T(+3.))))));
        COMPARE(Vc::atan2(V(T(-0.)), V(T(+3.))), V(T(-0.)));
        VERIFY (all_of(isnegative(Vc::atan2(V(T(-0.)), V(T(+3.))))));
        // If y is less than 0 and x is +0 or -0, -pi/2 is returned.
        COMPARE(Vc::atan2(V(T(-3.)), V(T(+0.))), -Pi_2);
        COMPARE(Vc::atan2(V(T(-3.)), V(T(-0.))), -Pi_2);
        // If y is greater than 0 and x is +0 or -0, pi/2 is returned.
        COMPARE(Vc::atan2(V(T(+3.)), V(T(+0.))), +Pi_2);
        COMPARE(Vc::atan2(V(T(+3.)), V(T(-0.))), +Pi_2);
        // If either x or y is NaN, a NaN is returned.
        VERIFY(all_of(Vc::isnan(Vc::atan2(nan, V(T(3.))))));
        VERIFY(all_of(Vc::isnan(Vc::atan2(V(T(3.)), nan))));
        VERIFY(all_of(Vc::isnan(Vc::atan2(nan, nan))));
        // If y is +0 (-0) and x is -0, +pi (-pi) is returned.
        ATAN_COMPARE(Vc::atan2(V(T(+0.)), V(T(-0.))), +Pi);
        ATAN_COMPARE(Vc::atan2(V(T(-0.)), V(T(-0.))), -Pi);
        // If y is +0 (-0) and x is +0, +0 (-0) is returned.
        COMPARE(Vc::atan2(V(T(+0.)), V(T(+0.))), V(T(+0.)));
        COMPARE(Vc::atan2(V(T(-0.)), V(T(+0.))), V(T(-0.)));
        VERIFY(none_of(isnegative(Vc::atan2(V(T(+0.)), V(T(+0.))))));
        VERIFY( all_of(isnegative(Vc::atan2(V(T(-0.)), V(T(+0.))))));
        // If y is a finite value greater (less) than 0, and x is negative infinity, +pi (-pi) is returned.
        ATAN_COMPARE(Vc::atan2(V(T(+1.)), -inf), +Pi);
        ATAN_COMPARE(Vc::atan2(V(T(-1.)), -inf), -Pi);
        // If y is a finite value greater (less) than 0, and x is positive infinity, +0 (-0) is returned.
        COMPARE(Vc::atan2(V(T(+3.)), +inf), V(T(+0.)));
        VERIFY(none_of(isnegative(Vc::atan2(V(T(+3.)), +inf))));
        COMPARE(Vc::atan2(V(T(-3.)), +inf), V(T(-0.)));
        VERIFY( all_of(isnegative(Vc::atan2(V(T(-3.)), +inf))));
        // If y is positive infinity (negative infinity), and x is finite, pi/2 (-pi/2) is returned.
        COMPARE(Vc::atan2(+inf, V(T(+3.))), +Pi_2);
        COMPARE(Vc::atan2(-inf, V(T(+3.))), -Pi_2);
        COMPARE(Vc::atan2(+inf, V(T(-3.))), +Pi_2);
        COMPARE(Vc::atan2(-inf, V(T(-3.))), -Pi_2);
#ifndef _WIN32 // the Microsoft implementation of atan2 fails this test
        const V Pi_4 = T(doubleConstant<1, 0x921fb54442d18ull, -1>());
        // If y is positive infinity (negative infinity) and x is negative	infinity, +3*pi/4 (-3*pi/4) is returned.
        COMPARE(Vc::atan2(+inf, -inf), T(+3.) * Pi_4);
        COMPARE(Vc::atan2(-inf, -inf), T(-3.) * Pi_4);
        // If y is positive infinity (negative infinity) and x is positive infinity, +pi/4 (-pi/4) is returned.
        COMPARE(Vc::atan2(+inf, +inf), +Pi_4);
        COMPARE(Vc::atan2(-inf, +inf), -Pi_4);
#endif
    }

    for (int xoffset = -100; xoffset < 54613; xoffset += 47 * V::Size) {
        for (int yoffset = -100; yoffset < 54613; yoffset += 47 * V::Size) {
            const V data([](T n) { return n; });
            const V x = (data + xoffset) * T(0.15);
            const V y = (data + yoffset) * T(0.15);
            const V reference =
                V::generate([&](size_t i) { return std::atan2(x[i], y[i]); });
            FUZZY_COMPARE(Vc::atan2(x, y), reference) << ", x = " << x << ", y = " << y;
        }
    }
}

//}}}1
// vim: foldmethod=marker
