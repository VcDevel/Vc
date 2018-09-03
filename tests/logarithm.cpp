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
#include "mathreference.h"

template <typename T> struct Denormals { //{{{1
    static T *data;
};
template <> float *Denormals<float>::data = 0;
template <> double *Denormals<double>::data = 0;
enum { NDenormals = 64 };

TEST(prepareDenormals) //{{{1
{
    Denormals<float>::data = Vc::malloc<float, Vc::AlignOnVector>(NDenormals);
    Denormals<float>::data[0] = std::numeric_limits<float>::denorm_min();
    for (int i = 1; i < NDenormals; ++i) {
        Denormals<float>::data[i] = Denormals<float>::data[i - 1] * 2.173f;
    }
    Denormals<double>::data = Vc::malloc<double, Vc::AlignOnVector>(NDenormals);
    Denormals<double>::data[0] = std::numeric_limits<double>::denorm_min();
    for (int i = 1; i < NDenormals; ++i) {
        Denormals<double>::data[i] = Denormals<double>::data[i - 1] * 2.173;
    }
}

template <typename V, typename F> void testAllDenormals(F &&fun)  //{{{1
{
    using T = typename V::EntryType;
    fun(V(&Denormals<T>::data[0], Vc::Aligned));  // the first one is always aligned.
    for (auto i = V::Size; i + V::Size <= NDenormals; i += V::Size) {
        using AlignedLoadTag = typename std::conditional<
            (V::Size & (V::Size - 1)) == 0,  // if V::Size is even
            decltype(Vc::Aligned),           // use aligned loads
            decltype(Vc::Unaligned)          // otherwise use unaliged loads
            >::type;
        fun(V(&Denormals<T>::data[i], AlignedLoadTag()));
    }
    if (NDenormals % V::Size != 0) {
        fun(V(&Denormals<T>::data[NDenormals - V::Size], Vc::Unaligned));
    }
}

TEST_TYPES(V, testLog, RealTypes) //{{{1
{
    setFuzzyness<float>(1);
    typedef typename V::EntryType T;
    Array<Reference<T> > reference = referenceData<T, Log>();
    for (size_t i = 0; i + V::Size - 1 < reference.size; i += V::Size) {
        V x, ref;
        for (size_t j = 0; j < V::Size; ++j) {
            x[j] = reference.data[i + j].x;
            ref[j] = reference.data[i + j].ref;
        }
        FUZZY_COMPARE(Vc::log(x), ref) << " x = " << x << ", i = " << i;
    }

    COMPARE(Vc::log(V(0)), V(std::log(T(0))));
    testAllDenormals<V>([](const V &x) {
        V ref = x.apply([](T _x) { return std::log(_x); });
        FUZZY_COMPARE(Vc::log(x), ref) << ", x = " << x;
    });
}

TEST_TYPES(V, testLog2, RealTypes) //{{{1
{
#if defined(Vc_LOG_ILP) || defined(Vc_LOG_ILP2)
    setFuzzyness<float>(3);
#else
    setFuzzyness<float>(1);
#endif
    setFuzzyness<double>(1);
#if defined(Vc_MSVC) || defined(__APPLE__)
    if (Vc::Scalar::is_vector<V>::value || !Vc::Traits::isAtomicSimdArray<V>::value) {
        setFuzzyness<double>(2);
    }
#endif
    typedef typename V::EntryType T;
    Array<Reference<T> > reference = referenceData<T, Log2>();
    for (size_t i = 0; i + V::Size - 1 < reference.size; i += V::Size) {
        V x, ref;
        for (size_t j = 0; j < V::Size; ++j) {
            x[j] = reference.data[i + j].x;
            ref[j] = reference.data[i + j].ref;
        }
        FUZZY_COMPARE(Vc::log2(x), ref) << " x = " << x << ", i = " << i;
    }

    COMPARE(Vc::log2(V(0)), V(std::log2(T(0))));
    testAllDenormals<V>([](const V x) {
        V ref = x.apply([](T _x) { return std::log2(_x); });
        FUZZY_COMPARE(Vc::log2(x), ref) << ", x = " << x;
    });
}

TEST_TYPES(V, testLog10, RealTypes) //{{{1
{
    setFuzzyness<float>(2);
    setFuzzyness<double>(2);
    typedef typename V::EntryType T;
    Array<Reference<T> > reference = referenceData<T, Log10>();
    for (size_t i = 0; i + V::Size - 1 < reference.size; i += V::Size) {
        V x, ref;
        for (size_t j = 0; j < V::Size; ++j) {
            x[j] = reference.data[i + j].x;
            ref[j] = reference.data[i + j].ref;
        }
        FUZZY_COMPARE(Vc::log10(x), ref) << " x = " << x << ", i = " << i;
    }

    COMPARE(Vc::log10(V(0)), V(std::log10(T(0))));
    testAllDenormals<V>([](const V x) {
        V ref = x.apply([](T _x) { return std::log10(_x); });
        FUZZY_COMPARE(Vc::log10(x), ref) << ", x = " << x;
    });
}

//}}}1
// vim: foldmethod=marker
