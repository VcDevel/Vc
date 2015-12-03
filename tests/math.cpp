/*  This file is part of the Vc library. {{{
Copyright © 2009-2015 Matthias Kretz <kretz@kde.org>
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
/*includes {{{*/
#include "unittest.h"
#include <iostream>
#include <Vc/array>
#include "vectormemoryhelper.h"
#include <cmath>
#include <algorithm>
#include <common/const.h>
#include <common/macros.h>
/*}}}*/
using namespace Vc;
using Vc::Detail::floatConstant;
using Vc::Detail::doubleConstant;

#define SIMD_ARRAY_LIST                                                                            \
     SIMD_ARRAYS(32),                                                                              \
     SIMD_ARRAYS(17),                                                                              \
     SIMD_ARRAYS(16),                                                                              \
     SIMD_ARRAYS(9),                                                                               \
     SIMD_ARRAYS(8),                                                                               \
     SIMD_ARRAYS(5),                                                                               \
     SIMD_ARRAYS(4),                                                                               \
     SIMD_ARRAYS(3),                                                                               \
     SIMD_ARRAYS(2),                                                                               \
     SIMD_ARRAYS(1)
#define SIMD_REAL_ARRAY_LIST                                                                       \
     SIMD_REAL_ARRAYS(32),                                                                         \
     SIMD_REAL_ARRAYS(17),                                                                         \
     SIMD_REAL_ARRAYS(16),                                                                         \
     SIMD_REAL_ARRAYS(9),                                                                          \
     SIMD_REAL_ARRAYS(8),                                                                          \
     SIMD_REAL_ARRAYS(5),                                                                          \
     SIMD_REAL_ARRAYS(4),                                                                          \
     SIMD_REAL_ARRAYS(3),                                                                          \
     SIMD_REAL_ARRAYS(2),                                                                          \
     SIMD_REAL_ARRAYS(1)
#define SIMD_INT_ARRAY_LIST                                                                        \
     SIMD_INT_ARRAYS(32),                                                                          \
     SIMD_INT_ARRAYS(17),                                                                          \
     SIMD_INT_ARRAYS(16),                                                                          \
     SIMD_INT_ARRAYS(9),                                                                           \
     SIMD_INT_ARRAYS(8),                                                                           \
     SIMD_INT_ARRAYS(5),                                                                           \
     SIMD_INT_ARRAYS(4),                                                                           \
     SIMD_INT_ARRAYS(3),                                                                           \
     SIMD_INT_ARRAYS(2),                                                                           \
     SIMD_INT_ARRAYS(1)

// fix isfinite and isnan {{{1
#ifdef isfinite
#undef isfinite
#endif
#ifdef isnan
#undef isnan
#endif

template<typename T> struct SincosReference //{{{1
{
    T x, s, c;
};
template<typename T> struct Reference
{
    T x, ref;
};

template<typename T> struct Array
{
    size_t size;
    const T *data;
    Array() : size(0), data(0) {}
    Array(size_t s, const T *p) : size(s), data(p) {}
};
template<typename T> struct StaticDeleter
{
    const T *ptr;
    StaticDeleter(const T *p) : ptr(p) {}
    ~StaticDeleter() { delete[] ptr; }
};

enum Function {
    Sincos, Atan, Asin, Acos, Log, Log2, Log10
};
template<typename T, Function F> static inline const char *filename();
template<> inline const char *filename<float , Sincos>() { return "reference-sincos-sp.dat"; }
template<> inline const char *filename<double, Sincos>() { return "reference-sincos-dp.dat"; }
template<> inline const char *filename<float , Atan  >() { return "reference-atan-sp.dat"; }
template<> inline const char *filename<double, Atan  >() { return "reference-atan-dp.dat"; }
template<> inline const char *filename<float , Asin  >() { return "reference-asin-sp.dat"; }
template<> inline const char *filename<double, Asin  >() { return "reference-asin-dp.dat"; }
// template<> inline const char *filename<float , Acos  >() { return "reference-acos-sp.dat"; }
// template<> inline const char *filename<double, Acos  >() { return "reference-acos-dp.dat"; }
template<> inline const char *filename<float , Log   >() { return "reference-ln-sp.dat"; }
template<> inline const char *filename<double, Log   >() { return "reference-ln-dp.dat"; }
template<> inline const char *filename<float , Log2  >() { return "reference-log2-sp.dat"; }
template<> inline const char *filename<double, Log2  >() { return "reference-log2-dp.dat"; }
template<> inline const char *filename<float , Log10 >() { return "reference-log10-sp.dat"; }
template<> inline const char *filename<double, Log10 >() { return "reference-log10-dp.dat"; }

#ifdef Vc_IMPL_MIC
extern "C" {
extern const Reference<double> _binary_reference_acos_dp_dat_start;
extern const Reference<double> _binary_reference_acos_dp_dat_end;
extern const Reference<float > _binary_reference_acos_sp_dat_start;
extern const Reference<float > _binary_reference_acos_sp_dat_end;
extern const Reference<double> _binary_reference_asin_dp_dat_start;
extern const Reference<double> _binary_reference_asin_dp_dat_end;
extern const Reference<float > _binary_reference_asin_sp_dat_start;
extern const Reference<float > _binary_reference_asin_sp_dat_end;
extern const Reference<double> _binary_reference_atan_dp_dat_start;
extern const Reference<double> _binary_reference_atan_dp_dat_end;
extern const Reference<float > _binary_reference_atan_sp_dat_start;
extern const Reference<float > _binary_reference_atan_sp_dat_end;
extern const Reference<double> _binary_reference_ln_dp_dat_start;
extern const Reference<double> _binary_reference_ln_dp_dat_end;
extern const Reference<float > _binary_reference_ln_sp_dat_start;
extern const Reference<float > _binary_reference_ln_sp_dat_end;
extern const Reference<double> _binary_reference_log10_dp_dat_start;
extern const Reference<double> _binary_reference_log10_dp_dat_end;
extern const Reference<float > _binary_reference_log10_sp_dat_start;
extern const Reference<float > _binary_reference_log10_sp_dat_end;
extern const Reference<double> _binary_reference_log2_dp_dat_start;
extern const Reference<double> _binary_reference_log2_dp_dat_end;
extern const Reference<float > _binary_reference_log2_sp_dat_start;
extern const Reference<float > _binary_reference_log2_sp_dat_end;
extern const SincosReference<double> _binary_reference_sincos_dp_dat_start;
extern const SincosReference<double> _binary_reference_sincos_dp_dat_end;
extern const SincosReference<float > _binary_reference_sincos_sp_dat_start;
extern const SincosReference<float > _binary_reference_sincos_sp_dat_end;
}

template <typename T, Function F>
inline std::pair<const T *, const T *> binary();
template <>
inline std::pair<const SincosReference<float> *, const SincosReference<float> *>
binary<SincosReference<float>, Sincos>()
{
    return std::make_pair(&_binary_reference_sincos_sp_dat_start,
                          &_binary_reference_sincos_sp_dat_end);
}
template <>
inline std::pair<const SincosReference<double> *,
                 const SincosReference<double> *>
binary<SincosReference<double>, Sincos>()
{
    return std::make_pair(&_binary_reference_sincos_dp_dat_start,
                          &_binary_reference_sincos_dp_dat_end);
}
template <>
inline std::pair<const Reference<float> *, const Reference<float> *>
binary<Reference<float>, Atan>()
{
    return std::make_pair(&_binary_reference_atan_sp_dat_start,
                          &_binary_reference_atan_sp_dat_end);
}
template <>
inline std::pair<const Reference<double> *, const Reference<double> *>
binary<Reference<double>, Atan>()
{
    return std::make_pair(&_binary_reference_atan_dp_dat_start,
                          &_binary_reference_atan_dp_dat_end);
}
template <>
inline std::pair<const Reference<float> *, const Reference<float> *>
binary<Reference<float>, Asin>()
{
    return std::make_pair(&_binary_reference_asin_sp_dat_start,
                          &_binary_reference_asin_sp_dat_end);
}
template <>
inline std::pair<const Reference<double> *, const Reference<double> *>
binary<Reference<double>, Asin>()
{
    return std::make_pair(&_binary_reference_asin_dp_dat_start,
                          &_binary_reference_asin_dp_dat_end);
}
template <>
inline std::pair<const Reference<float> *, const Reference<float> *>
binary<Reference<float>, Acos>()
{
    return std::make_pair(&_binary_reference_acos_sp_dat_start,
                          &_binary_reference_acos_sp_dat_end);
}
template <>
inline std::pair<const Reference<double> *, const Reference<double> *>
binary<Reference<double>, Acos>()
{
    return std::make_pair(&_binary_reference_acos_dp_dat_start,
                          &_binary_reference_acos_dp_dat_end);
}
template <>
inline std::pair<const Reference<float> *, const Reference<float> *>
binary<Reference<float>, Log>()
{
    return std::make_pair(&_binary_reference_ln_sp_dat_start,
                          &_binary_reference_ln_sp_dat_end);
}
template <>
inline std::pair<const Reference<double> *, const Reference<double> *>
binary<Reference<double>, Log>()
{
    return std::make_pair(&_binary_reference_ln_dp_dat_start,
                          &_binary_reference_ln_dp_dat_end);
}
template <>
inline std::pair<const Reference<float> *, const Reference<float> *>
binary<Reference<float>, Log2>()
{
    return std::make_pair(&_binary_reference_log2_sp_dat_start,
                          &_binary_reference_log2_sp_dat_end);
}
template <>
inline std::pair<const Reference<double> *, const Reference<double> *>
binary<Reference<double>, Log2>()
{
    return std::make_pair(&_binary_reference_log2_dp_dat_start,
                          &_binary_reference_log2_dp_dat_end);
}
template <>
inline std::pair<const Reference<float> *, const Reference<float> *>
binary<Reference<float>, Log10>()
{
    return std::make_pair(&_binary_reference_log10_sp_dat_start,
                          &_binary_reference_log10_sp_dat_end);
}
template <>
inline std::pair<const Reference<double> *, const Reference<double> *>
binary<Reference<double>, Log10>()
{
    return std::make_pair(&_binary_reference_log10_dp_dat_start,
                          &_binary_reference_log10_dp_dat_end);
}
#endif

template<typename T>
static Array<SincosReference<T> > sincosReference()
{
#ifdef Vc_IMPL_MIC
    const auto range = binary<SincosReference<T>, Sincos>();
    return {range.second - range.first, range.first};
#else
    static Array<SincosReference<T> > data;
    if (data.data == 0) {
        FILE *file = std::fopen(filename<T, Sincos>(), "rb");
        if (file) {
            std::fseek(file, 0, SEEK_END);
            const size_t size = std::ftell(file) / sizeof(SincosReference<T>);
            std::rewind(file);
            auto mem = new SincosReference<T>[size];
            static StaticDeleter<SincosReference<T> > _cleanup(data.data);
            data.size = std::fread(mem, sizeof(SincosReference<T>), size, file);
            data.data = mem;
            std::fclose(file);
        } else {
            FAIL() << "the reference data " << filename<T, Sincos>() << " does not exist in the current working directory.";
        }
    }
    return data;
#endif
}

template<typename T, Function Fun>
static Array<Reference<T> > referenceData()
{
#ifdef Vc_IMPL_MIC
    const auto range = binary<Reference<T>, Fun>();
    return {range.second - range.first, range.first};
#else
    static Array<Reference<T> > data;
    if (data.data == 0) {
        FILE *file = std::fopen(filename<T, Fun>(), "rb");
        if (file) {
            std::fseek(file, 0, SEEK_END);
            const size_t size = std::ftell(file) / sizeof(Reference<T>);
            std::rewind(file);
            auto mem = new Reference<T>[size];
            static StaticDeleter<Reference<T> > _cleanup(data.data);
            data.size = std::fread(mem, sizeof(Reference<T>), size, file);
            data.data = mem;
            std::fclose(file);
        } else {
            FAIL() << "the reference data " << filename<T, Fun>() << " does not exist in the current working directory.";
        }
    }
    return data;
#endif
}

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

// testAbs{{{1
TEST_TYPES(Vec, testAbs, (REAL_VECTORS, SIMD_REAL_ARRAY_LIST, int_v, short_v, SimdArray<int, 8>,
                          SimdArray<int, 2>, SimdArray<int, 7>))
{
    for (int i = 0; i < 0x7fff - int(Vec::size()); ++i) {
        Vec a = Vec::IndexesFromZero() + i;
        Vec b = -a;
        COMPARE(a, Vc::abs(a));
        COMPARE(a, Vc::abs(b));
    }
    for (int i = 0; i < 1000; ++i) {
        const Vec a = Vec::Random();
        const Vec ref = Vec::generate([&](int j) { return std::abs(a[j]); });
        COMPARE(abs(a), ref) << "a : " << a;
    }
}

TEST_TYPES(V, testTrunc, (REAL_VECTORS, SIMD_REAL_ARRAY_LIST)) //{{{1
{
    typedef typename V::EntryType T;
    for (size_t i = 0; i < 100000 / V::Size; ++i) {
        V x = (V::Random() - T(0.5)) * T(100);
        V reference = x.apply([](T _x) { return std::trunc(_x); });
        COMPARE(Vc::trunc(x), reference) << ", x = " << x << ", i = " << i;
    }
    V x = V::IndexesFromZero();
    V reference = x.apply([](T _x) { return std::trunc(_x); });
    COMPARE(Vc::trunc(x), reference) << ", x = " << x;
}

TEST_TYPES(V, testFloor, (REAL_VECTORS, SIMD_REAL_ARRAY_LIST)) //{{{1
{
    typedef typename V::EntryType T;
    for (size_t i = 0; i < 100000 / V::Size; ++i) {
        V x = (V::Random() - T(0.5)) * T(100);
        V reference = x.apply([](T _x) { return std::floor(_x); });
        COMPARE(Vc::floor(x), reference) << ", x = " << x << ", i = " << i;
    }
    V x = V::IndexesFromZero();
    V reference = x.apply([](T _x) { return std::floor(_x); });
    COMPARE(Vc::floor(x), reference) << ", x = " << x;
}

TEST_TYPES(V, testCeil, (REAL_VECTORS, SIMD_REAL_ARRAY_LIST)) //{{{1
{
    typedef typename V::EntryType T;
    for (size_t i = 0; i < 100000 / V::Size; ++i) {
        V x = (V::Random() - T(0.5)) * T(100);
        V reference = x.apply([](T _x) { return std::ceil(_x); });
        COMPARE(Vc::ceil(x), reference) << ", x = " << x << ", i = " << i;
    }
    V x = V::IndexesFromZero();
    V reference = x.apply([](T _x) { return std::ceil(_x); });
    COMPARE(Vc::ceil(x), reference) << ", x = " << x;
}

TEST_TYPES(V, testExp, (REAL_VECTORS, SIMD_REAL_ARRAY_LIST)) //{{{1
{
    UnitTest::setFuzzyness<float>(1);
    UnitTest::setFuzzyness<double>(2);
    typedef typename V::EntryType T;
    for (size_t i = 0; i < 100000 / V::Size; ++i) {
        V x = (V::Random() - T(0.5)) * T(20);
        V reference = x.apply([](T _x) { return std::exp(_x); });
        FUZZY_COMPARE(Vc::exp(x), reference) << ", x = " << x << ", i = " << i;
    }
    COMPARE(Vc::exp(V::Zero()), V::One());
}

TEST_TYPES(V, testLog, (REAL_VECTORS, SIMD_REAL_ARRAY_LIST)) //{{{1
{
#ifdef Vc_IMPL_MIC
    UnitTest::setFuzzyness<float>(2);
#else
    UnitTest::setFuzzyness<float>(1);
#endif
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

    COMPARE(Vc::log(V::Zero()), V(std::log(T(0))));
    for (int i = 0; i < NDenormals; i += V::Size) {
        V x(&Denormals<T>::data[i]);
        V ref = x.apply([](T _x) { return std::log(_x); });
        FUZZY_COMPARE(Vc::log(x), ref) << ", x = " << x << ", i = " << i;
    }
}

TEST_TYPES(V, testLog2, (REAL_VECTORS, SIMD_REAL_ARRAY_LIST)) //{{{1
{
#if defined(Vc_LOG_ILP) || defined(Vc_LOG_ILP2)
    UnitTest::setFuzzyness<float>(3);
#else
    UnitTest::setFuzzyness<float>(1);
#endif
    UnitTest::setFuzzyness<double>(1);
#if defined(Vc_MSVC) || defined(__APPLE__)
    if (Scalar::is_vector<V>::value || !Traits::isAtomicSimdArray<V>::value) {
        UnitTest::setFuzzyness<double>(2);
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

    COMPARE(Vc::log2(V::Zero()), V(std::log2(T(0))));
    for (int i = 0; i < NDenormals; i += V::Size) {
        V x(&Denormals<T>::data[i]);
        V ref = x.apply([](T _x) { return std::log2(_x); });
        FUZZY_COMPARE(Vc::log2(x), ref) << ", x = " << x << ", i = " << i;
    }
}

TEST_TYPES(V, testLog10, (REAL_VECTORS, SIMD_REAL_ARRAY_LIST)) //{{{1
{
    UnitTest::setFuzzyness<float>(2);
    UnitTest::setFuzzyness<double>(2);
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

    COMPARE(Vc::log10(V::Zero()), V(std::log10(T(0))));
    for (int i = 0; i < NDenormals; i += V::Size) {
        V x(&Denormals<T>::data[i]);
        V ref = x.apply([](T _x) { return std::log10(_x); });
        FUZZY_COMPARE(Vc::log10(x), ref) << ", x = " << x << ", i = " << i;
    }
}

TEST_TYPES(Vec, testMax, (ALL_VECTORS, SIMD_ARRAY_LIST)) //{{{1
{
    typedef typename Vec::EntryType T;
    VectorMemoryHelper<Vec> mem(3);
    T *data = mem;
    for (size_t i = 0; i < Vec::Size; ++i) {
        data[i] = i;
        data[i + Vec::Size] = Vec::Size + 1 - i;
        data[i + 2 * Vec::Size] = std::max(data[i], data[i + Vec::Size]);
    }
    Vec a(&data[0]);
    Vec b(&data[Vec::Size]);
    Vec c(&data[2 * Vec::Size]);

    COMPARE(Vc::max(a, b), c);
}

// fillDataAndReference{{{1
template <typename V, typename F> void fillDataAndReference(V &data, V &reference, F f)
{
    using T = typename V::EntryType;
    for (size_t i = 0; i < V::Size; ++i) {
        data[i] = static_cast<T>(i);
        reference[i] = f(data[i]);
    }
}

TEST_TYPES(V, testSqrt, (REAL_VECTORS, SIMD_REAL_ARRAY_LIST)) //{{{1
{
    typedef typename V::EntryType T;
    V data, reference;
    fillDataAndReference(data, reference, [](T x) { return std::sqrt(x); });

    FUZZY_COMPARE(Vc::sqrt(data), reference);
}

TEST_TYPES(V, testRSqrt, (REAL_VECTORS, SIMD_REAL_ARRAY_LIST)) //{{{1
{
    typedef typename V::EntryType T;
    for (size_t i = 0; i < 1024 / V::Size; ++i) {
        const V x = V::Random() * T(1000);
        // RSQRTPS is documented as having a relative error <= 1.5 * 2^-12
        VERIFY(all_of(Vc::abs(Vc::rsqrt(x) * Vc::sqrt(x) - V::One()) < static_cast<T>(std::ldexp(1.5, -12))));
    }
}

TEST_TYPES(V, testSincos, (REAL_VECTORS, SIMD_REAL_ARRAY_LIST)) //{{{1
{
    typedef typename V::EntryType T;
    UnitTest::setFuzzyness<float>(2);
    UnitTest::setFuzzyness<double>(1e7);
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

TEST_TYPES(V, testSin, (REAL_VECTORS, SIMD_REAL_ARRAY_LIST)) //{{{1
{
    typedef typename V::EntryType T;
    UnitTest::setFuzzyness<float>(2);
    UnitTest::setFuzzyness<double>(1e7);
    Array<SincosReference<T> > reference = sincosReference<T>();
    for (size_t i = 0; i + V::Size - 1 < reference.size; i += V::Size) {
        V x, sref;
        for (size_t j = 0; j < V::Size; ++j) {
            x[j] = reference.data[i + j].x;
            sref[j] = reference.data[i + j].s;
        }
        FUZZY_COMPARE(Vc::sin(x), sref) << " x = " << x << ", i = " << i;
        FUZZY_COMPARE(Vc::sin(-x), -sref) << " x = " << x << ", i = " << i;
    }
}

TEST_TYPES(V, testCos, (REAL_VECTORS, SIMD_REAL_ARRAY_LIST)) //{{{1
{
    typedef typename V::EntryType T;
    UnitTest::setFuzzyness<float>(2);
    UnitTest::setFuzzyness<double>(1e7);
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

TEST_TYPES(V, testAsin, (REAL_VECTORS, SIMD_REAL_ARRAY_LIST)) //{{{1
{
    typedef typename V::EntryType T;
#ifdef Vc_IMPL_MIC
    UnitTest::setFuzzyness<float>(3);
#else
    UnitTest::setFuzzyness<float>(2);
#endif
    UnitTest::setFuzzyness<double>(36);
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

TEST_TYPES(V, testAtan, (REAL_VECTORS, SIMD_REAL_ARRAY_LIST)) //{{{1
{
    typedef typename V::EntryType T;
    UnitTest::setFuzzyness<float>(3);
    UnitTest::setFuzzyness<double>(2);

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

TEST_TYPES(V, testAtan2, (REAL_VECTORS, SIMD_REAL_ARRAY_LIST)) //{{{1
{
    typedef typename V::EntryType T;
    UnitTest::setFuzzyness<float>(3);
    UnitTest::setFuzzyness<double>(2);

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
            V data, reference;
            fillDataAndReference(data, reference, [&](T x) {
                return std::atan2((x + xoffset) * T(0.15), (x + yoffset) * T(0.15));
            });

            const V x = (data + xoffset) * T(0.15);
            const V y = (data + yoffset) * T(0.15);
            FUZZY_COMPARE(Vc::atan2(x, y), reference) << ", x = " << x << ", y = " << y;
        }
    }
}

TEST_TYPES(Vec, testReciprocal, (REAL_VECTORS, SIMD_REAL_ARRAY_LIST)) //{{{1
{
    typedef typename Vec::EntryType T;
    UnitTest::setFuzzyness<float>(1.258295e+07);
    UnitTest::setFuzzyness<double>(0);
    const T one = 1;
    for (int offset = -1000; offset < 1000; offset += 10) {
        const T scale = T(0.1);
        Vec data;
        Vec reference;
        for (size_t ii = 0; ii < Vec::Size; ++ii) {
            const T i = static_cast<T>(ii);
            data[ii] = i;
            T tmp = (i + offset) * scale;
            reference[ii] = one / tmp;
        }

        FUZZY_COMPARE(Vc::reciprocal((data + offset) * scale), reference);
    }
}

TEST_TYPES(V, isNegative, (REAL_VECTORS, SIMD_REAL_ARRAY_LIST)) //{{{1
{
    typedef typename V::EntryType T;
    VERIFY(isnegative(V::One()).isEmpty());
    VERIFY(isnegative(V::Zero()).isEmpty());
    VERIFY(isnegative((-V::One())).isFull());
    VERIFY(isnegative(V(T(-0.))).isFull());
}

TEST_TYPES(Vec, testInf, (REAL_VECTORS, SIMD_REAL_ARRAY_LIST)) //{{{1
{
    typedef typename Vec::EntryType T;
    const T one = 1;
    const Vec zero(Zero);
    const Vec inf = one / zero;
    Vec nan;
    nan.setQnan();

    VERIFY(all_of(Vc::isfinite(zero)));
    VERIFY(all_of(Vc::isfinite(Vec(one))));
    VERIFY(none_of(Vc::isfinite(inf)));
    VERIFY(none_of(Vc::isfinite(nan)));

    VERIFY(none_of(Vc::isinf(zero)));
    VERIFY(none_of(Vc::isinf(Vec(one))));
    VERIFY(all_of(Vc::isinf(inf)));
    VERIFY(none_of(Vc::isinf(nan)));
}

TEST_TYPES(Vec, testNaN, (REAL_VECTORS, SIMD_REAL_ARRAY_LIST)) //{{{1
{
    typedef typename Vec::EntryType T;
    typedef typename Vec::IndexType I;
    typedef typename Vec::Mask M;
    const T one = 1;
    const Vec zero(Zero);
    VERIFY(none_of(Vc::isnan(zero)));
    VERIFY(none_of(Vc::isnan(Vec(one))));
    const Vec inf = one / zero;
    VERIFY(all_of(Vc::isnan(Vec(inf * zero))));
    Vec nan = Vec::Zero();
    const M mask(I::IndexesFromZero() == I::Zero());
    nan.setQnan(mask);
    COMPARE(Vc::isnan(nan), mask);
    nan.setQnan();
    VERIFY(all_of(Vc::isnan(nan)));
}

TEST_TYPES(Vec, testRound, (REAL_VECTORS, SIMD_REAL_ARRAY_LIST)) //{{{1
{
    typedef typename Vec::EntryType T;
    enum {
        Count = (16 + Vec::Size) / Vec::Size
    };
    VectorMemoryHelper<Vec> mem1(Count);
    VectorMemoryHelper<Vec> mem2(Count);
    T *data = mem1;
    T *reference = mem2;
    for (size_t i = 0; i < Count * Vec::Size; ++i) {
        data[i] = i * 0.25 - 2.0;
        reference[i] = std::floor(i * 0.25 - 2.0 + 0.5);
        if (i % 8 == 2) {
            reference[i] -= 1.;
        }
        //std::cout << reference[i] << " ";
    }
    //std::cout << std::endl;
    for (int i = 0; i < Count; ++i) {
        const Vec a(&data[i * Vec::Size]);
        const Vec ref(&reference[i * Vec::Size]);
        //std::cout << a << ref << std::endl;
        COMPARE(Vc::round(a), ref);
    }
}

TEST_TYPES(Vec, testReduceMin, (ALL_VECTORS, SIMD_ARRAY_LIST)) //{{{1
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

TEST_TYPES(Vec, testReduceMax, (ALL_VECTORS, SIMD_ARRAY_LIST)) //{{{1
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

TEST_TYPES(V, testReduceProduct, (ALL_VECTORS, SIMD_ARRAY_LIST)) //{{{1
{
    using T = typename V::EntryType;
    V test = 0;
    COMPARE(test.product(), T(0));
    test = 1;
    COMPARE(test.product(), T(1));
    test[0] = 2;
    COMPARE(test.product(), T(2));
    test[0] = 3;
    COMPARE(test.product(), T(3));

    for (std::size_t i = 0; i + 1 < V::size(); ++i) {
        test[i] = 1;
        test[i + 1] = 5;
        COMPARE(test.product(), T(5));
    }
    for (std::size_t i = 0; i + 2 < V::size(); ++i) {
        test[i] = 1;
        test[i + 1] = 7;
        COMPARE(test.product(), T(5 * 7));
    }
}

TEST_TYPES(Vec, testReduceSum, (ALL_VECTORS, SIMD_ARRAY_LIST)) //{{{1
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

TEST_TYPES(V, testExponent, (REAL_VECTORS, SIMD_REAL_ARRAY_LIST)) //{{{1
{
    typedef typename V::EntryType T;
    Vc::array<T, 32> input;
    Vc::array<T, 32> expected;
    input[ 0] = T(0.25); expected[ 0] = T(-2);
    input[ 1] = T(   1); expected[ 1] = T( 0);
    input[ 2] = T(   2); expected[ 2] = T( 1);
    input[ 3] = T(   3); expected[ 3] = T( 1);
    input[ 4] = T(   4); expected[ 4] = T( 2);
    input[ 5] = T( 0.5); expected[ 5] = T(-1);
    input[ 6] = T(   6); expected[ 6] = T( 2);
    input[ 7] = T(   7); expected[ 7] = T( 2);
    input[ 8] = T(   8); expected[ 8] = T( 3);
    input[ 9] = T(   9); expected[ 9] = T( 3);
    input[10] = T(  10); expected[10] = T( 3);
    input[11] = T(  11); expected[11] = T( 3);
    input[12] = T(  12); expected[12] = T( 3);
    input[13] = T(  13); expected[13] = T( 3);
    input[14] = T(  14); expected[14] = T( 3);
    input[15] = T(  15); expected[15] = T( 3);
    input[16] = T(  16); expected[16] = T( 4);
    input[17] = T(  17); expected[17] = T( 4);
    input[18] = T(  18); expected[18] = T( 4);
    input[19] = T(  19); expected[19] = T( 4);
    input[20] = T(  20); expected[20] = T( 4);
    input[21] = T(  21); expected[21] = T( 4);
    input[22] = T(  22); expected[22] = T( 4);
    input[23] = T(  23); expected[23] = T( 4);
    input[24] = T(  24); expected[24] = T( 4);
    input[25] = T(  25); expected[25] = T( 4);
    input[26] = T(  26); expected[26] = T( 4);
    input[27] = T(  27); expected[27] = T( 4);
    input[28] = T(  28); expected[28] = T( 4);
    input[29] = T(  29); expected[29] = T( 4);
    input[30] = T(  32); expected[30] = T( 5);
    input[31] = T(  31); expected[31] = T( 4);
    for (size_t i = 0; i <= input.size() - V::size(); ++i) {
        COMPARE(V(&input[i]).exponent(), V(&expected[i]));
    }
}

template<typename T> struct _ExponentVector { typedef int_v Type; };

TEST_TYPES(V, testFrexp, (REAL_VECTORS, SIMD_REAL_ARRAY_LIST)) //{{{1
{
    typedef typename V::EntryType T;
    using ExpV = typename V::IndexType;
    Vc::array<T, 33> input;
    Vc::array<T, 33> expectedFraction;
    Vc::array<int, 33> expectedExponent;
    input[ 0] = T(0.25); expectedFraction[ 0] = T(.5     ); expectedExponent[ 0] = -1;
    input[ 1] = T(   1); expectedFraction[ 1] = T(.5     ); expectedExponent[ 1] =  1;
    input[ 2] = T(   0); expectedFraction[ 2] = T(0.     ); expectedExponent[ 2] =  0;
    input[ 3] = T(   3); expectedFraction[ 3] = T(.75    ); expectedExponent[ 3] =  2;
    input[ 4] = T(   4); expectedFraction[ 4] = T(.5     ); expectedExponent[ 4] =  3;
    input[ 5] = T( 0.5); expectedFraction[ 5] = T(.5     ); expectedExponent[ 5] =  0;
    input[ 6] = T(   6); expectedFraction[ 6] = T( 6./8. ); expectedExponent[ 6] =  3;
    input[ 7] = T(   7); expectedFraction[ 7] = T( 7./8. ); expectedExponent[ 7] =  3;
    input[ 8] = T(   8); expectedFraction[ 8] = T( 8./16.); expectedExponent[ 8] =  4;
    input[ 9] = T(   9); expectedFraction[ 9] = T( 9./16.); expectedExponent[ 9] =  4;
    input[10] = T(  10); expectedFraction[10] = T(10./16.); expectedExponent[10] =  4;
    input[11] = T(  11); expectedFraction[11] = T(11./16.); expectedExponent[11] =  4;
    input[12] = T(  12); expectedFraction[12] = T(12./16.); expectedExponent[12] =  4;
    input[13] = T(  13); expectedFraction[13] = T(13./16.); expectedExponent[13] =  4;
    input[14] = T(  14); expectedFraction[14] = T(14./16.); expectedExponent[14] =  4;
    input[15] = T(  15); expectedFraction[15] = T(15./16.); expectedExponent[15] =  4;
    input[16] = T(  16); expectedFraction[16] = T(16./32.); expectedExponent[16] =  5;
    input[17] = T(  17); expectedFraction[17] = T(17./32.); expectedExponent[17] =  5;
    input[18] = T(  18); expectedFraction[18] = T(18./32.); expectedExponent[18] =  5;
    input[19] = T(  19); expectedFraction[19] = T(19./32.); expectedExponent[19] =  5;
    input[20] = T(  20); expectedFraction[20] = T(20./32.); expectedExponent[20] =  5;
    input[21] = T(  21); expectedFraction[21] = T(21./32.); expectedExponent[21] =  5;
    input[22] = T(  22); expectedFraction[22] = T(22./32.); expectedExponent[22] =  5;
    input[23] = T(  23); expectedFraction[23] = T(23./32.); expectedExponent[23] =  5;
    input[24] = T(  24); expectedFraction[24] = T(24./32.); expectedExponent[24] =  5;
    input[25] = T(  25); expectedFraction[25] = T(25./32.); expectedExponent[25] =  5;
    input[26] = T(  26); expectedFraction[26] = T(26./32.); expectedExponent[26] =  5;
    input[27] = T(  27); expectedFraction[27] = T(27./32.); expectedExponent[27] =  5;
    input[28] = T(  28); expectedFraction[28] = T(28./32.); expectedExponent[28] =  5;
    input[29] = T(  29); expectedFraction[29] = T(29./32.); expectedExponent[29] =  5;
    input[30] = T(  32); expectedFraction[30] = T(32./64.); expectedExponent[30] =  6;
    input[31] = T(  31); expectedFraction[31] = T(31./32.); expectedExponent[31] =  5;
    input[32] = T( -0.); expectedFraction[32] = T(-0.    ); expectedExponent[32] =  0;
    for (size_t i = 0; i <= 33 - V::size(); ++i) {
        const V v(&input[i]);
        ExpV exp;
        const V fraction = frexp(v, &exp);
        COMPARE(fraction, V(&expectedFraction[i]))
            << ", v = " << v << ", delta: " << fraction - V(&expectedFraction[i]);
        COMPARE(exp, ExpV(&expectedExponent[i]))
            << "\ninput: " << v << ", fraction: " << fraction << ", i: " << i;
    }
}

TEST_TYPES(V, testLdexp, (REAL_VECTORS, SIMD_REAL_ARRAY_LIST)) //{{{1
{
    typedef typename V::EntryType T;
    using ExpV = typename V::IndexType;
    for (size_t i = 0; i < 1024 / V::Size; ++i) {
        const V v = (V::Random() - T(0.5)) * T(1000);
        ExpV e;
        const V m = frexp(v, &e);
        COMPARE(ldexp(m, e), v) << ", m = " << m << ", e = " << e;
    }
}

#include "ulp.h"
TEST_TYPES(V, testUlpDiff, (REAL_VECTORS, SIMD_REAL_ARRAY_LIST)) //{{{1
{
    typedef typename V::EntryType T;

    COMPARE(ulpDiffToReference(V::Zero(), V::Zero()), V::Zero());
    COMPARE(ulpDiffToReference(std::numeric_limits<V>::min(), V::Zero()), V::One());
    COMPARE(ulpDiffToReference(V::Zero(), std::numeric_limits<V>::min()), V::One());
    for (size_t count = 0; count < 1024 / V::Size; ++count) {
        const V base = (V::Random() - T(0.5)) * T(1000);
        typename V::IndexType exp;
        Vc::frexp(base, &exp);
        const V eps = ldexp(V(std::numeric_limits<T>::epsilon()), exp - 1);
        //std::cout << base << ", " << exp << ", " << eps << std::endl;
        for (int i = -10000; i <= 10000; ++i) {
            const V i_v = V(T(i));
            const V diff = base + i_v * eps;

            // if diff and base have a different exponent then ulpDiffToReference has an uncertainty
            // of +/-1
            const V ulpDifference = ulpDiffToReference(diff, base);
            const V expectedDifference = Vc::abs(i_v);
            const V maxUncertainty = Vc::abs(abs(diff).exponent() - abs(base).exponent());

            VERIFY(all_of(Vc::abs(ulpDifference - expectedDifference) <= maxUncertainty))
                << ", base = " << base << ", epsilon = " << eps << ", diff = " << diff;
            for (size_t k = 0; k < V::Size; ++k) {
                VERIFY(std::abs(ulpDifference[k] - expectedDifference[k]) <= maxUncertainty[k]);
            }
        }
    }
}

// TODO: copysign

// vim: foldmethod=marker
