/*  This file is part of the Vc library. {{{
Copyright © 2016-2017 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_SIMD_X86_H_
#define VC_SIMD_X86_H_

#include <limits>
#include <climits>
#include <cstring>

#include "../macros.h"
#include "../detail.h"
#include "../const.h"

#ifdef Vc_HAVE_SSE

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
namespace x86
{
// missing intrinsics
#if defined Vc_GCC && Vc_GCC < 0x80000
Vc_INTRINSIC void _mm_mask_cvtepi16_storeu_epi8(void *p, __mmask8 k, __m128i x)
{
    asm("vpmovwb %0,(%2)%{%1%}" :: "x"(x), "k"(k), "g"(p) : "k0");
}
Vc_INTRINSIC void _mm256_mask_cvtepi16_storeu_epi8(void *p, __mmask16 k, __m256i x)
{
    asm("vpmovwb %0,(%2)%{%1%}" :: "x"(x), "k"(k), "g"(p) : "k0");
}
Vc_INTRINSIC void _mm512_mask_cvtepi16_storeu_epi8(void *p, __mmask32 k, __m512i x)
{
    asm("vpmovwb %0,(%2)%{%1%}" :: "x"(x), "k"(k), "g"(p) : "k0");
}
#endif
// make_immediate{{{
template <unsigned Stride> constexpr unsigned make_immediate(unsigned a, unsigned b)
{
    return a + b * Stride;
}
template <unsigned Stride>
constexpr unsigned make_immediate(unsigned a, unsigned b, unsigned c, unsigned d)
{
    return a + Stride * (b + Stride * (c + Stride * d));
}

// }}}
// conversion with constprop{{{
#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC builtin_type_t<int, 4> cvt_i32(builtin_type_t<float, 4> x)
{
    return __builtin_constant_p(x)
               ? builtin_type_t<int, 4>{static_cast<int>(x[0]), static_cast<int>(x[1]),
                                        static_cast<int>(x[2]), static_cast<int>(x[3])}
               : reinterpret_cast<builtin_type_t<int, 4>>(_mm_cvttps_epi32(x));
}
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC builtin_type_t<int, 8> cvt_i32(builtin_type_t<float, 8> x)
{
    return __builtin_constant_p(x)
               ? builtin_type_t<int, 8>{static_cast<int>(x[0]), static_cast<int>(x[1]),
                                        static_cast<int>(x[2]), static_cast<int>(x[3]),
                                        static_cast<int>(x[4]), static_cast<int>(x[5]),
                                        static_cast<int>(x[6]), static_cast<int>(x[7])}
               : reinterpret_cast<builtin_type_t<int, 8>>(_mm256_cvttps_epi32(x));
}
#endif  // Vc_HAVE_AVX

// }}}
// zeroExtend{{{1
#ifdef Vc_HAVE_AVX
template <class From> struct zeroExtend {
    static_assert(is_intrinsic_v<From>);
    constexpr zeroExtend(From) {};
};
template <> struct zeroExtend<__m128> {
    __m128 d;
    constexpr Vc_INTRINSIC zeroExtend(__m128 x) : d(x) {}

    Vc_INTRINSIC operator __m256()
    {
        return _mm256_insertf128_ps(__m256(), d, 0);
    }
#ifdef Vc_HAVE_AVX512F
    Vc_INTRINSIC operator __m512()
    {
#ifdef Vc_WORKAROUND_PR85480
        asm("vmovaps %0, %0" : "+x"(d));
        return _mm512_castps128_ps512(d);
#else
        return _mm512_insertf32x4(__m512(), d, 0);
#endif
    }
#endif
};
template <> struct zeroExtend<__m128d> {
    __m128d d;
    constexpr Vc_INTRINSIC zeroExtend(__m128d x) : d(x) {}

    Vc_INTRINSIC operator __m256d()
    {
        return _mm256_insertf128_pd(__m256d(), d, 0);
    }
#ifdef Vc_HAVE_AVX512F
    Vc_INTRINSIC operator __m512d()
    {
#ifdef Vc_WORKAROUND_PR85480
        asm("vmovapd %0, %0" : "+x"(d));
        return _mm512_castpd128_pd512(d);
#else
        return _mm512_insertf64x2(__m512d(), d, 0);
#endif
    }
#endif
};
template <> struct zeroExtend<__m128i> {
    __m128i d;
    constexpr Vc_INTRINSIC zeroExtend(__m128i x) : d(x) {}

    Vc_INTRINSIC operator __m256i()
    {
        return _mm256_insertf128_si256(__m256i(), d, 0);
    }
#ifdef Vc_HAVE_AVX512F
    Vc_INTRINSIC operator __m512i()
    {
#ifdef Vc_WORKAROUND_PR85480
        asm("vmovadq %0, %0" : "+x"(d));
        return _mm512_castsi128_si512(d);
#else
        return _mm512_inserti32x4(__m512i(), d, 0);
#endif
    }
#endif
};
#endif  // Vc_HAVE_AVX
#ifdef Vc_HAVE_AVX512F
template <> struct zeroExtend<__m256> {
    __m256 d;
    constexpr Vc_INTRINSIC zeroExtend(__m256 x) : d(x) {}
    Vc_INTRINSIC operator __m512()
    {
        return _mm512_insertf32x8(__m512(), d, 0);
    }
};
template <> struct zeroExtend<__m256d> {
    __m256d d;
    constexpr Vc_INTRINSIC zeroExtend(__m256d x) : d(x) {}
    Vc_INTRINSIC operator __m512d()
    {
        return _mm512_insertf64x4(__m512d(), d, 0);
    }
};
template <> struct zeroExtend<__m256i> {
    __m256i d;
    constexpr Vc_INTRINSIC zeroExtend(__m256i x) : d(x) {}
    Vc_INTRINSIC operator __m512i()
    {
        return _mm512_inserti64x4(__m512i(), d, 0);
    }
};
#endif  // Vc_HAVE_AVX512F

// intrin_cast{{{1
template <class To, class From> constexpr Vc_INTRINSIC To intrin_cast(From v) {
    static_assert(is_builtin_vector_v<From> && is_builtin_vector_v<To>);
    if constexpr (sizeof(To) == sizeof(From)) {
        return reinterpret_cast<To>(v);
    } else if constexpr (sizeof(From) == 16 && sizeof(To) == 32) {
        return reinterpret_cast<To>(_mm256_castps128_ps256(reinterpret_cast<__m128>(v)));
    } else if constexpr (sizeof(From) == 16 && sizeof(To) == 64) {
        return reinterpret_cast<To>(_mm512_castps128_ps512(reinterpret_cast<__m128>(v)));
    } else if constexpr (sizeof(From) == 32 && sizeof(To) == 16) {
        return reinterpret_cast<To>(_mm256_castps256_ps128(reinterpret_cast<__m256>(v)));
    } else if constexpr (sizeof(From) == 32 && sizeof(To) == 64) {
        return reinterpret_cast<To>(_mm512_castps256_ps512(reinterpret_cast<__m256>(v)));
    } else if constexpr (sizeof(From) == 64 && sizeof(To) == 16) {
        return reinterpret_cast<To>(_mm512_castps512_ps128(reinterpret_cast<__m512>(v)));
    } else if constexpr (sizeof(From) == 64 && sizeof(To) == 32) {
        return reinterpret_cast<To>(_mm512_castps512_ps256(reinterpret_cast<__m512>(v)));
    } else {
        static_assert(!std::is_same_v<To, To>, "should be unreachable");
    }
}

// extract<N, By>{{{1
template <int Offset, int SplitBy, class T, class Trait = builtin_traits<T>,
          class R = builtin_type_t<typename Trait::value_type, Trait::width / SplitBy>>
constexpr Vc_INTRINSIC R extract(T x_)
{
#ifdef Vc_WORKAROUND_XXX_1
    constexpr int return_width = sizeof(R) / sizeof(double);
    using U = builtin_type_t<double, return_width>;
    const auto x =
        reinterpret_cast<builtin_type_t<double, sizeof(T) / sizeof(double)>>(x_);
#else
    constexpr int return_width = Trait::width / SplitBy;
    using U = R;
    const builtin_type_t<typename Traits::value_type, Trait::width> &x = x_;
#endif
    constexpr int O = Offset * return_width;
    if constexpr (return_width == 2) {
        return reinterpret_cast<R>(U{x[O + 0], x[O + 1]});
    } else if constexpr (return_width == 4) {
        return reinterpret_cast<R>(U{x[O + 0], x[O + 1], x[O + 2], x[O + 3]});
    } else if constexpr (return_width == 8) {
        return reinterpret_cast<R>(U{x[O + 0], x[O + 1], x[O + 2], x[O + 3], x[O + 4],
                                     x[O + 5], x[O + 6], x[O + 7]});
    } else if constexpr (return_width == 16) {
        return reinterpret_cast<R>(U{x[O + 0], x[O + 1], x[O + 2], x[O + 3], x[O + 4],
                                     x[O + 5], x[O + 6], x[O + 7], x[O + 8], x[O + 9],
                                     x[O + 10], x[O + 11], x[O + 12], x[O + 13],
                                     x[O + 14], x[O + 15]});
    } else if constexpr (return_width == 32) {
        return reinterpret_cast<R>(
            U{x[O + 0],  x[O + 1],  x[O + 2],  x[O + 3],  x[O + 4],  x[O + 5],  x[O + 6],
              x[O + 7],  x[O + 8],  x[O + 9],  x[O + 10], x[O + 11], x[O + 12], x[O + 13],
              x[O + 14], x[O + 15], x[O + 16], x[O + 17], x[O + 18], x[O + 19], x[O + 20],
              x[O + 21], x[O + 22], x[O + 23], x[O + 24], x[O + 25], x[O + 26], x[O + 27],
              x[O + 28], x[O + 29], x[O + 30], x[O + 31]});
    }
}

// extract128{{{1
template <int offset, class T>
constexpr Vc_INTRINSIC auto extract128(T a)
    -> decltype(extract<offset, sizeof(T) / 16>(a))
{
    return extract<offset, sizeof(T) / 16>(a);
}

// extract256{{{1
template <int offset, class T>
constexpr Vc_INTRINSIC auto extract256(T a)
    -> decltype(extract<offset, 2>(a))
{
    static_assert(sizeof(T) == 64);
    return extract<offset, 2>(a);
}

// extract256_center{{{1
#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC __m256 extract256_center(__m512 a)
{
    return intrin_cast<__m256>(
        _mm512_shuffle_f32x4(a, a, 1 + 2 * 0x4 + 2 * 0x10 + 3 * 0x40));
}
Vc_INTRINSIC __m256d extract256_center(__m512d a)
{
    return intrin_cast<__m256d>(
        _mm512_shuffle_f64x2(a, a, 1 + 2 * 0x4 + 2 * 0x10 + 3 * 0x40));
}
Vc_INTRINSIC __m256i extract256_center(__m512i a)
{
    return intrin_cast<__m256i>(
        _mm512_shuffle_i32x4(a, a, 1 + 2 * 0x4 + 2 * 0x10 + 3 * 0x40));
}
#endif  // Vc_HAVE_AVX512F

// lo/hi128{{{1
template <class T>
constexpr Vc_INTRINSIC auto lo128(T x) -> decltype(extract<0, sizeof(T) / 16>(x))
{
    return extract<0, sizeof(T) / 16>(x);
}
template <class T> constexpr Vc_INTRINSIC auto hi128(T x) -> decltype(extract<1, 2>(x))
{
    static_assert(sizeof(T) == 32);
    return extract<1, 2>(x);
}

// lo/hi256{{{1
template <class T> constexpr Vc_INTRINSIC auto lo256(T x) -> decltype(extract<0, 2>(x))
{
    static_assert(sizeof(T) == 64);
    return extract<0, 2>(x);
}
template <class T> constexpr Vc_INTRINSIC auto hi256(T x) -> decltype(extract<1, 2>(x))
{
    static_assert(sizeof(T) == 64);
    return extract<1, 2>(x);
}

// concat{{{1
template <class T, class Trait = builtin_traits<T>,
          class R = builtin_type_t<typename Trait::value_type, Trait::width * 2>>
constexpr R concat(T a_, T b_) {
#ifdef Vc_WORKAROUND_XXX_1
    constexpr int input_width = sizeof(T) / sizeof(double);
    using TT = builtin_type_t<double, input_width>;
    const TT a = reinterpret_cast<TT>(a_);
    const TT b = reinterpret_cast<TT>(b_);
    using U = builtin_type_t<double, sizeof(R) / sizeof(double)>;
#else
    constexpr int input_width = Trait::width;
    const T &a = a_;
    const T &b = b_;
    using U = R;
#endif
    if constexpr(input_width == 2) {
        return reinterpret_cast<R>(U{a[0], a[1], b[0], b[1]});
    } else if constexpr (input_width == 4) {
        return reinterpret_cast<R>(U{a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]});
    } else if constexpr (input_width == 8) {
        return reinterpret_cast<R>(U{a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], b[0],
                                     b[1], b[2], b[3], b[4], b[5], b[6], b[7]});
    } else if constexpr (input_width == 16) {
        return reinterpret_cast<R>(
            U{a[0],  a[1],  a[2],  a[3],  a[4],  a[5],  a[6],  a[7],  a[8],  a[9], a[10],
              a[11], a[12], a[13], a[14], a[15], b[0],  b[1],  b[2],  b[3],  b[4], b[5],
              b[6],  b[7],  b[8],  b[9],  b[10], b[11], b[12], b[13], b[14], b[15]});
    } else if constexpr (input_width == 32) {
        return reinterpret_cast<R>(
            U{a[0],  a[1],  a[2],  a[3],  a[4],  a[5],  a[6],  a[7],  a[8],  a[9],  a[10],
              a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21],
              a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], b[0],
              b[1],  b[2],  b[3],  b[4],  b[5],  b[6],  b[7],  b[8],  b[9],  b[10], b[11],
              b[12], b[13], b[14], b[15], b[16], b[17], b[18], b[19], b[20], b[21], b[22],
              b[23], b[24], b[25], b[26], b[27], b[28], b[29], b[30], b[31]});
    }
}

// allone{{{1
template <typename V> Vc_INTRINSIC_L Vc_CONST_L V allone() Vc_INTRINSIC_R Vc_CONST_R;
template <> Vc_INTRINSIC Vc_CONST __m128 allone<__m128>()
{
#ifdef Vc_HAVE_SSE2
    return _mm_castsi128_ps(~__m128i());
#else
    return reinterpret_cast<__m128>(
        __m128i{0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU});
#endif
}
#ifdef Vc_HAVE_SSE2
template <> Vc_INTRINSIC Vc_CONST __m128i allone<__m128i>()
{
    return ~__m128i();
}
template <> Vc_INTRINSIC Vc_CONST __m128d allone<__m128d>()
{
    return _mm_castsi128_pd(~__m128i());
}
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC Vc_CONST __m256 allone<__m256>()
{
    return _mm256_castsi256_ps(~__m256i());
}
template <> Vc_INTRINSIC Vc_CONST __m256i allone<__m256i>()
{
    return ~__m256i();
}
template <> Vc_INTRINSIC Vc_CONST __m256d allone<__m256d>()
{
    return _mm256_castsi256_pd(~__m256i());
}
#endif

#ifdef Vc_HAVE_AVX512F
template <> Vc_INTRINSIC Vc_CONST __m512 allone<__m512>()
{
    return _mm512_castsi512_ps(~__m512i());
}
template <> Vc_INTRINSIC Vc_CONST __m512d allone<__m512d>()
{
    return _mm512_castsi512_pd(~__m512i());
}
template <> Vc_INTRINSIC Vc_CONST __m512i allone<__m512i>()
{
    return ~__m512i();
}
#endif  // Vc_HAVE_AVX512F

// zero{{{1
template <typename V> Vc_INTRINSIC_L Vc_CONST_L V zero() Vc_INTRINSIC_R Vc_CONST_R;
template<> Vc_INTRINSIC Vc_CONST __m128  zero<__m128 >() { return _mm_setzero_ps(); }
#ifdef Vc_HAVE_SSE2
template<> Vc_INTRINSIC Vc_CONST __m128i zero<__m128i>() { return _mm_setzero_si128(); }
template<> Vc_INTRINSIC Vc_CONST __m128d zero<__m128d>() { return _mm_setzero_pd(); }
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
template<> Vc_INTRINSIC Vc_CONST __m256  zero<__m256 >() { return _mm256_setzero_ps(); }
template<> Vc_INTRINSIC Vc_CONST __m256i zero<__m256i>() { return _mm256_setzero_si256(); }
template<> Vc_INTRINSIC Vc_CONST __m256d zero<__m256d>() { return _mm256_setzero_pd(); }
#endif

#ifdef Vc_HAVE_AVX512F
template<> Vc_INTRINSIC Vc_CONST __m512  zero<__m512 >() { return _mm512_setzero_ps(); }
template<> Vc_INTRINSIC Vc_CONST __m512i zero<__m512i>() { return _mm512_setzero_si512(); }
template<> Vc_INTRINSIC Vc_CONST __m512d zero<__m512d>() { return _mm512_setzero_pd(); }
#endif

// auto_cast{{{1
template <class T> struct auto_cast_t {
    static_assert(is_builtin_vector_v<T>);
    const T x;
    template <class U> Vc_INTRINSIC operator U() const { return intrin_cast<U>(x); }
};
template <class T> auto_cast_t<T> auto_cast(const T &x) { return {x}; }
template <class T, size_t N>
auto_cast_t<typename Storage<T, N>::register_type> auto_cast(const Storage<T, N> &x)
{
    return {x.d};
}

// one16/32{{{1
Vc_INTRINSIC Vc_CONST __m128  one16( float) { return sse_const::oneFloat; }

#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC Vc_CONST __m128d one16(double) { return sse_const::oneDouble; }
Vc_INTRINSIC Vc_CONST __m128i one16( uchar) { return reinterpret_cast<__m128i>(sse_const::one8); }
Vc_INTRINSIC Vc_CONST __m128i one16( schar) { return one16(uchar()); }
Vc_INTRINSIC Vc_CONST __m128i one16(ushort) { return reinterpret_cast<__m128i>(sse_const::one16); }
Vc_INTRINSIC Vc_CONST __m128i one16( short) { return one16(ushort()); }
Vc_INTRINSIC Vc_CONST __m128i one16(  uint) { return reinterpret_cast<__m128i>(sse_const::one32); }
Vc_INTRINSIC Vc_CONST __m128i one16(   int) { return one16(uint()); }
Vc_INTRINSIC Vc_CONST __m128i one16(ullong) { return reinterpret_cast<__m128i>(sse_const::one64); }
Vc_INTRINSIC Vc_CONST __m128i one16( llong) { return one16(ullong()); }
Vc_INTRINSIC Vc_CONST __m128i one16(  long) { return one16(equal_int_type_t<long>()); }
Vc_INTRINSIC Vc_CONST __m128i one16( ulong) { return one16(equal_int_type_t<ulong>()); }
#endif

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC Vc_CONST __m256  one32( float) { return _mm256_broadcast_ss(&avx_const::oneFloat); }
Vc_INTRINSIC Vc_CONST __m256d one32(double) { return _mm256_broadcast_sd(&avx_const::oneDouble); }
Vc_INTRINSIC Vc_CONST __m256i one32( llong) { return _mm256_castpd_si256(_mm256_broadcast_sd(reinterpret_cast<const double *>(&avx_const::IndexesFromZero64[1]))); }
Vc_INTRINSIC Vc_CONST __m256i one32(ullong) { return one32(llong()); }
Vc_INTRINSIC Vc_CONST __m256i one32(   int) { return _mm256_castps_si256(_mm256_broadcast_ss(reinterpret_cast<const float *>(&avx_const::IndexesFromZero32[1]))); }
Vc_INTRINSIC Vc_CONST __m256i one32(  uint) { return one32(int()); }
Vc_INTRINSIC Vc_CONST __m256i one32( short) { return _mm256_castps_si256(_mm256_broadcast_ss(reinterpret_cast<const float *>(avx_const::one16))); }
Vc_INTRINSIC Vc_CONST __m256i one32(ushort) { return one32(short()); }
Vc_INTRINSIC Vc_CONST __m256i one32( schar) { return _mm256_castps_si256(_mm256_broadcast_ss(reinterpret_cast<const float *>(avx_const::one8))); }
Vc_INTRINSIC Vc_CONST __m256i one32( uchar) { return one32(schar()); }
Vc_INTRINSIC Vc_CONST __m256i one32(  long) { return one32(equal_int_type_t<long>()); }
Vc_INTRINSIC Vc_CONST __m256i one32( ulong) { return one32(equal_int_type_t<ulong>()); }
#endif

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC Vc_CONST __m512  one64( float) { return _mm512_broadcastss_ps(_mm_load_ss(&avx_const::oneFloat)); }
Vc_INTRINSIC Vc_CONST __m512d one64(double) { return _mm512_broadcastsd_pd(_mm_load_sd(&avx_const::oneDouble)); }
Vc_INTRINSIC Vc_CONST __m512i one64( llong) { return _mm512_set1_epi64(1ll); }
Vc_INTRINSIC Vc_CONST __m512i one64(ullong) { return _mm512_set1_epi64(1ll); }
Vc_INTRINSIC Vc_CONST __m512i one64(   int) { return _mm512_set1_epi32(1); }
Vc_INTRINSIC Vc_CONST __m512i one64(  uint) { return _mm512_set1_epi32(1); }
Vc_INTRINSIC Vc_CONST __m512i one64( short) { return _mm512_set1_epi16(1); }
Vc_INTRINSIC Vc_CONST __m512i one64(ushort) { return _mm512_set1_epi16(1); }
Vc_INTRINSIC Vc_CONST __m512i one64( schar) { return _mm512_broadcast_i32x4(one16(schar())); }
Vc_INTRINSIC Vc_CONST __m512i one64( uchar) { return one64(schar()); }
Vc_INTRINSIC Vc_CONST __m512i one64(  long) { return one64(equal_int_type_t<long>()); }
Vc_INTRINSIC Vc_CONST __m512i one64( ulong) { return one64(equal_int_type_t<ulong>()); }
#endif  // Vc_HAVE_AVX512F

// signmask{{{1
#ifdef Vc_HAVE_SSE2
constexpr Vc_INTRINSIC x_f64 signmask16(double) { return x_f64::broadcast(-0.); }
#endif  // Vc_HAVE_SSE2
constexpr Vc_INTRINSIC x_f32 signmask16( float) { return x_f32::broadcast(-0.f); }

#ifdef Vc_HAVE_AVX
constexpr Vc_INTRINSIC y_f64 signmask32(double) { return y_f64::broadcast(-0.); }
constexpr Vc_INTRINSIC y_f32 signmask32( float) { return y_f32::broadcast(-0.f); }
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
constexpr Vc_INTRINSIC z_f64 signmask64(double) { return z_f64::broadcast(-0.); }
constexpr Vc_INTRINSIC z_f32 signmask64( float) { return z_f32::broadcast(-0.f); }
#endif  // Vc_HAVE_AVX

// set16/32/64{{{1
Vc_INTRINSIC Vc_CONST __m128 set(float x0, float x1, float x2, float x3)
{
    return _mm_set_ps(x3, x2, x1, x0);
}
#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC Vc_CONST __m128d set(double x0, double x1) { return _mm_set_pd(x1, x0); }

Vc_INTRINSIC Vc_CONST __m128i set(llong x0, llong x1) { return _mm_set_epi64x(x1, x0); }
Vc_INTRINSIC Vc_CONST __m128i set(ullong x0, ullong x1) { return _mm_set_epi64x(x1, x0); }

Vc_INTRINSIC Vc_CONST __m128i set(int x0, int x1, int x2, int x3)
{
    return _mm_set_epi32(x3, x2, x1, x0);
}
Vc_INTRINSIC Vc_CONST __m128i set(uint x0, uint x1, uint x2, uint x3)
{
    return _mm_set_epi32(x3, x2, x1, x0);
}

Vc_INTRINSIC Vc_CONST __m128i set(short x0, short x1, short x2, short x3, short x4,
                                  short x5, short x6, short x7)
{
    return _mm_set_epi16(x7, x6, x5, x4, x3, x2, x1, x0);
}
Vc_INTRINSIC Vc_CONST __m128i set(ushort x0, ushort x1, ushort x2, ushort x3, ushort x4,
                                  ushort x5, ushort x6, ushort x7)
{
    return _mm_set_epi16(x7, x6, x5, x4, x3, x2, x1, x0);
}

Vc_INTRINSIC Vc_CONST __m128i set(schar x0, schar x1, schar x2, schar x3, schar x4,
                                  schar x5, schar x6, schar x7, schar x8, schar x9,
                                  schar x10, schar x11, schar x12, schar x13, schar x14,
                                  schar x15)
{
    return _mm_set_epi8(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2, x1,
                         x0);
}
Vc_INTRINSIC Vc_CONST __m128i set(uchar x0, uchar x1, uchar x2, uchar x3, uchar x4,
                                  uchar x5, uchar x6, uchar x7, uchar x8, uchar x9,
                                  uchar x10, uchar x11, uchar x12, uchar x13, uchar x14,
                                  uchar x15)
{
    return _mm_set_epi8(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2, x1,
                         x0);
}
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC Vc_CONST __m256 set(float x0, float x1, float x2, float x3, float x4,
                                 float x5, float x6, float x7)
{
    return _mm256_set_ps(x7, x6, x5, x4, x3, x2, x1, x0);
}

Vc_INTRINSIC Vc_CONST __m256d set(double x0, double x1, double x2, double x3)
{
    return _mm256_set_pd(x3, x2, x1, x0);
}

Vc_INTRINSIC Vc_CONST __m256i set(llong x0, llong x1, llong x2, llong x3)
{
    return _mm256_set_epi64x(x3, x2, x1, x0);
}
Vc_INTRINSIC Vc_CONST __m256i set(ullong x0, ullong x1, ullong x2, ullong x3)
{
    return _mm256_set_epi64x(x3, x2, x1, x0);
}

Vc_INTRINSIC Vc_CONST __m256i set(int x0, int x1, int x2, int x3, int x4, int x5, int x6,
                                  int x7)
{
    return _mm256_set_epi32(x7, x6, x5, x4, x3, x2, x1, x0);
}
Vc_INTRINSIC Vc_CONST __m256i set(uint x0, uint x1, uint x2, uint x3, uint x4, uint x5,
                                  uint x6, uint x7)
{
    return _mm256_set_epi32(x7, x6, x5, x4, x3, x2, x1, x0);
}

Vc_INTRINSIC Vc_CONST __m256i set(short x0, short x1, short x2, short x3, short x4,
                                  short x5, short x6, short x7, short x8, short x9,
                                  short x10, short x11, short x12, short x13, short x14,
                                  short x15)
{
    return _mm256_set_epi16(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2,
                            x1, x0);
}
Vc_INTRINSIC Vc_CONST __m256i set(ushort x0, ushort x1, ushort x2, ushort x3, ushort x4,
                                  ushort x5, ushort x6, ushort x7, ushort x8, ushort x9,
                                  ushort x10, ushort x11, ushort x12, ushort x13,
                                  ushort x14, ushort x15)
{
    return _mm256_set_epi16(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2,
                            x1, x0);
}

Vc_INTRINSIC Vc_CONST __m256i set(schar x0, schar x1, schar x2, schar x3, schar x4,
                                  schar x5, schar x6, schar x7, schar x8, schar x9,
                                  schar x10, schar x11, schar x12, schar x13, schar x14,
                                  schar x15, schar x16, schar x17, schar x18, schar x19,
                                  schar x20, schar x21, schar x22, schar x23, schar x24,
                                  schar x25, schar x26, schar x27, schar x28, schar x29,
                                  schar x30, schar x31)
{
    return _mm256_set_epi8(x31, x30, x29, x28, x27, x26, x25, x24, x23, x22, x21, x20,
                           x19, x18, x17, x16, x15, x14, x13, x12, x11, x10, x9, x8, x7,
                           x6, x5, x4, x3, x2, x1, x0);
}
Vc_INTRINSIC Vc_CONST __m256i set(uchar x0, uchar x1, uchar x2, uchar x3, uchar x4,
                                  uchar x5, uchar x6, uchar x7, uchar x8, uchar x9,
                                  uchar x10, uchar x11, uchar x12, uchar x13, uchar x14,
                                  uchar x15, uchar x16, uchar x17, uchar x18, uchar x19,
                                  uchar x20, uchar x21, uchar x22, uchar x23, uchar x24,
                                  uchar x25, uchar x26, uchar x27, uchar x28, uchar x29,
                                  uchar x30, uchar x31)
{
    return _mm256_set_epi8(x31, x30, x29, x28, x27, x26, x25, x24, x23, x22, x21, x20,
                           x19, x18, x17, x16, x15, x14, x13, x12, x11, x10, x9, x8, x7,
                           x6, x5, x4, x3, x2, x1, x0);
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC Vc_CONST __m512d set(double x0, double x1, double x2, double x3, double x4,
                                  double x5, double x6, double x7)
{
    return _mm512_set_pd(x7, x6, x5, x4, x3, x2, x1, x0);
}

Vc_INTRINSIC Vc_CONST __m512 set(float x0, float x1, float x2, float x3, float x4,
                                 float x5, float x6, float x7, float x8, float x9,
                                 float x10, float x11, float x12, float x13, float x14,
                                 float x15)
{
    return _mm512_set_ps(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2, x1,
                         x0);
}

Vc_INTRINSIC Vc_CONST __m512i set(llong x0, llong x1, llong x2, llong x3, llong x4,
                                  llong x5, llong x6, llong x7)
{
    return _mm512_set_epi64(x7, x6, x5, x4, x3, x2, x1, x0);
}
Vc_INTRINSIC Vc_CONST __m512i set(ullong x0, ullong x1, ullong x2, ullong x3, ullong x4,
                                  ullong x5, ullong x6, ullong x7)
{
    return _mm512_set_epi64(x7, x6, x5, x4, x3, x2, x1, x0);
}

Vc_INTRINSIC Vc_CONST __m512i set(int x0, int x1, int x2, int x3, int x4, int x5, int x6,
                                  int x7, int x8, int x9, int x10, int x11, int x12,
                                  int x13, int x14, int x15)
{
    return _mm512_set_epi32(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2,
                            x1, x0);
}
Vc_INTRINSIC Vc_CONST __m512i set(uint x0, uint x1, uint x2, uint x3, uint x4, uint x5,
                                  uint x6, uint x7, uint x8, uint x9, uint x10, uint x11,
                                  uint x12, uint x13, uint x14, uint x15)
{
    return _mm512_set_epi32(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2,
                            x1, x0);
}

Vc_INTRINSIC Vc_CONST __m512i set(short x0, short x1, short x2, short x3, short x4,
                                  short x5, short x6, short x7, short x8, short x9,
                                  short x10, short x11, short x12, short x13, short x14,
                                  short x15, short x16, short x17, short x18, short x19,
                                  short x20, short x21, short x22, short x23, short x24,
                                  short x25, short x26, short x27, short x28, short x29,
                                  short x30, short x31)
{
    return concat(_mm256_set_epi16(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4,
                                   x3, x2, x1, x0),
                  _mm256_set_epi16(x31, x30, x29, x28, x27, x26, x25, x24, x23, x22, x21,
                                   x20, x19, x18, x17, x16));
}

Vc_INTRINSIC Vc_CONST __m512i set(ushort x0, ushort x1, ushort x2, ushort x3, ushort x4,
                                  ushort x5, ushort x6, ushort x7, ushort x8, ushort x9,
                                  ushort x10, ushort x11, ushort x12, ushort x13, ushort x14,
                                  ushort x15, ushort x16, ushort x17, ushort x18, ushort x19,
                                  ushort x20, ushort x21, ushort x22, ushort x23, ushort x24,
                                  ushort x25, ushort x26, ushort x27, ushort x28, ushort x29,
                                  ushort x30, ushort x31)
{
    return concat(_mm256_set_epi16(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4,
                                   x3, x2, x1, x0),
                  _mm256_set_epi16(x31, x30, x29, x28, x27, x26, x25, x24, x23, x22, x21,
                                   x20, x19, x18, x17, x16));
}

Vc_INTRINSIC Vc_CONST __m512i
set(schar x0, schar x1, schar x2, schar x3, schar x4, schar x5, schar x6, schar x7,
    schar x8, schar x9, schar x10, schar x11, schar x12, schar x13, schar x14, schar x15,
    schar x16, schar x17, schar x18, schar x19, schar x20, schar x21, schar x22,
    schar x23, schar x24, schar x25, schar x26, schar x27, schar x28, schar x29,
    schar x30, schar x31, schar x32, schar x33, schar x34, schar x35, schar x36,
    schar x37, schar x38, schar x39, schar x40, schar x41, schar x42, schar x43,
    schar x44, schar x45, schar x46, schar x47, schar x48, schar x49, schar x50,
    schar x51, schar x52, schar x53, schar x54, schar x55, schar x56, schar x57,
    schar x58, schar x59, schar x60, schar x61, schar x62, schar x63)
{
    return concat(_mm256_set_epi8(x31, x30, x29, x28, x27, x26, x25, x24, x23, x22, x21,
                                  x20, x19, x18, x17, x16, x15, x14, x13, x12, x11, x10,
                                  x9, x8, x7, x6, x5, x4, x3, x2, x1, x0),
                  _mm256_set_epi8(x63, x62, x61, x60, x59, x58, x57, x56, x55, x54, x53,
                                  x52, x51, x50, x49, x48, x47, x46, x45, x44, x43, x42,
                                  x41, x40, x39, x38, x37, x36, x35, x34, x33, x32));
}

Vc_INTRINSIC Vc_CONST __m512i
set(uchar x0, uchar x1, uchar x2, uchar x3, uchar x4, uchar x5, uchar x6, uchar x7,
    uchar x8, uchar x9, uchar x10, uchar x11, uchar x12, uchar x13, uchar x14, uchar x15,
    uchar x16, uchar x17, uchar x18, uchar x19, uchar x20, uchar x21, uchar x22,
    uchar x23, uchar x24, uchar x25, uchar x26, uchar x27, uchar x28, uchar x29,
    uchar x30, uchar x31, uchar x32, uchar x33, uchar x34, uchar x35, uchar x36,
    uchar x37, uchar x38, uchar x39, uchar x40, uchar x41, uchar x42, uchar x43,
    uchar x44, uchar x45, uchar x46, uchar x47, uchar x48, uchar x49, uchar x50,
    uchar x51, uchar x52, uchar x53, uchar x54, uchar x55, uchar x56, uchar x57,
    uchar x58, uchar x59, uchar x60, uchar x61, uchar x62, uchar x63)
{
    return concat(_mm256_set_epi8(x31, x30, x29, x28, x27, x26, x25, x24, x23, x22, x21,
                                  x20, x19, x18, x17, x16, x15, x14, x13, x12, x11, x10,
                                  x9, x8, x7, x6, x5, x4, x3, x2, x1, x0),
                  _mm256_set_epi8(x63, x62, x61, x60, x59, x58, x57, x56, x55, x54, x53,
                                  x52, x51, x50, x49, x48, x47, x46, x45, x44, x43, x42,
                                  x41, x40, x39, x38, x37, x36, x35, x34, x33, x32));
}

#endif  // Vc_HAVE_AVX512F

// generic forward for (u)long to (u)int or (u)llong
template <typename... Ts> Vc_INTRINSIC Vc_CONST auto set(Ts... args)
{
    return set(static_cast<equal_int_type_t<Ts>>(args)...);
}

// broadcast16/32/64{{{1
Vc_INTRINSIC __m128  broadcast16( float x) { return _mm_set1_ps(x); }
#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC __m128d broadcast16(double x) { return _mm_set1_pd(x); }
Vc_INTRINSIC __m128i broadcast16( schar x) { return _mm_set1_epi8(x); }
Vc_INTRINSIC __m128i broadcast16( uchar x) { return _mm_set1_epi8(x); }
Vc_INTRINSIC __m128i broadcast16( short x) { return _mm_set1_epi16(x); }
Vc_INTRINSIC __m128i broadcast16(ushort x) { return _mm_set1_epi16(x); }
Vc_INTRINSIC __m128i broadcast16(   int x) { return _mm_set1_epi32(x); }
Vc_INTRINSIC __m128i broadcast16(  uint x) { return _mm_set1_epi32(x); }
Vc_INTRINSIC __m128i broadcast16(  long x) { return sizeof( long) == 4 ? _mm_set1_epi32(x) : _mm_set1_epi64x(x); }
Vc_INTRINSIC __m128i broadcast16( ulong x) { return sizeof(ulong) == 4 ? _mm_set1_epi32(x) : _mm_set1_epi64x(x); }
Vc_INTRINSIC __m128i broadcast16( llong x) { return _mm_set1_epi64x(x); }
Vc_INTRINSIC __m128i broadcast16(ullong x) { return _mm_set1_epi64x(x); }
#endif  // Vc_HAVE_SSE2
template <class T> Vc_INTRINSIC auto broadcast(T x, size_constant<16>)
{
    return broadcast16(x);
}

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC __m256  broadcast32( float x) { return _mm256_set1_ps(x); }
Vc_INTRINSIC __m256d broadcast32(double x) { return _mm256_set1_pd(x); }
Vc_INTRINSIC __m256i broadcast32( schar x) { return _mm256_set1_epi8(x); }
Vc_INTRINSIC __m256i broadcast32( uchar x) { return _mm256_set1_epi8(x); }
Vc_INTRINSIC __m256i broadcast32( short x) { return _mm256_set1_epi16(x); }
Vc_INTRINSIC __m256i broadcast32(ushort x) { return _mm256_set1_epi16(x); }
Vc_INTRINSIC __m256i broadcast32(   int x) { return _mm256_set1_epi32(x); }
Vc_INTRINSIC __m256i broadcast32(  uint x) { return _mm256_set1_epi32(x); }
Vc_INTRINSIC __m256i broadcast32(  long x) { return sizeof( long) == 4 ? _mm256_set1_epi32(x) : _mm256_set1_epi64x(x); }
Vc_INTRINSIC __m256i broadcast32( ulong x) { return sizeof(ulong) == 4 ? _mm256_set1_epi32(x) : _mm256_set1_epi64x(x); }
Vc_INTRINSIC __m256i broadcast32( llong x) { return _mm256_set1_epi64x(x); }
Vc_INTRINSIC __m256i broadcast32(ullong x) { return _mm256_set1_epi64x(x); }
template <class T> Vc_INTRINSIC auto broadcast(T x, size_constant<32>)
{
    return broadcast32(x);
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC __m512  broadcast64( float x) { return _mm512_set1_ps(x); }
Vc_INTRINSIC __m512d broadcast64(double x) { return _mm512_set1_pd(x); }
Vc_INTRINSIC __m512i broadcast64( schar x) { return _mm512_set1_epi8(x); }
Vc_INTRINSIC __m512i broadcast64( uchar x) { return _mm512_set1_epi8(x); }
Vc_INTRINSIC __m512i broadcast64( short x) { return _mm512_set1_epi16(x); }
Vc_INTRINSIC __m512i broadcast64(ushort x) { return _mm512_set1_epi16(x); }
Vc_INTRINSIC __m512i broadcast64(   int x) { return _mm512_set1_epi32(x); }
Vc_INTRINSIC __m512i broadcast64(  uint x) { return _mm512_set1_epi32(x); }
Vc_INTRINSIC __m512i broadcast64(  long x) { return sizeof( long) == 4 ? _mm512_set1_epi32(x) : _mm512_set1_epi64(x); }
Vc_INTRINSIC __m512i broadcast64( ulong x) { return sizeof(ulong) == 4 ? _mm512_set1_epi32(x) : _mm512_set1_epi64(x); }
Vc_INTRINSIC __m512i broadcast64( llong x) { return _mm512_set1_epi64(x); }
Vc_INTRINSIC __m512i broadcast64(ullong x) { return _mm512_set1_epi64(x); }
template <class T> Vc_INTRINSIC auto broadcast(T x, size_constant<64>)
{
    return broadcast64(x);
}
#endif  // Vc_HAVE_AVX512F

// broadcast<T> {{{1
template <class T> struct broadcast_t {
    const T scalar;
    operator __m128i() { return broadcast16(scalar); }
#ifdef Vc_HAVE_AVX
    operator __m256i() { return broadcast32(scalar); }
#endif
#ifdef Vc_HAVE_AVX512F
    operator __m512i() { return broadcast64(scalar); }
#endif
};
template <> struct broadcast_t<float> {
    const float scalar;
    operator __m128() { return broadcast16(scalar); }
#ifdef Vc_HAVE_AVX
    operator __m256() { return broadcast32(scalar); }
#endif
#ifdef Vc_HAVE_AVX512F
    operator __m512() { return broadcast64(scalar); }
#endif
};
template <> struct broadcast_t<double> {
    const double scalar;
    operator __m128d() { return broadcast16(scalar); }
#ifdef Vc_HAVE_AVX
    operator __m256d() { return broadcast32(scalar); }
#endif
#ifdef Vc_HAVE_AVX512F
    operator __m512d() { return broadcast64(scalar); }
#endif
};
template <class T> broadcast_t<T> broadcast(T scalar) { return {scalar}; }

// lowest16/32/64{{{1
template <class T> constexpr Vc_INTRINSIC storage16_t<T> lowest16()
{
    return storage16_t<T>::broadcast(std::numeric_limits<T>::lowest());
}

#ifdef Vc_HAVE_SSE2
template <> constexpr Vc_INTRINSIC storage16_t< uchar> lowest16< uchar>() { return {}; }
template <> constexpr Vc_INTRINSIC storage16_t<ushort> lowest16<ushort>() { return {}; }
template <> constexpr Vc_INTRINSIC storage16_t<  uint> lowest16<  uint>() { return {}; }
template <> constexpr Vc_INTRINSIC storage16_t< ulong> lowest16< ulong>() { return {}; }
template <> constexpr Vc_INTRINSIC storage16_t<ullong> lowest16<ullong>() { return {}; }
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
template <class T> constexpr Vc_INTRINSIC Vc_CONST storage32_t<T> lowest32()
{
    return storage32_t<T>::broadcast(std::numeric_limits<T>::lowest());
}

template <> constexpr Vc_INTRINSIC storage32_t< uchar> lowest32< uchar>() { return {}; }
template <> constexpr Vc_INTRINSIC storage32_t<ushort> lowest32<ushort>() { return {}; }
template <> constexpr Vc_INTRINSIC storage32_t<  uint> lowest32<  uint>() { return {}; }
template <> constexpr Vc_INTRINSIC storage32_t< ulong> lowest32< ulong>() { return {}; }
template <> constexpr Vc_INTRINSIC storage32_t<ullong> lowest32<ullong>() { return {}; }
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
template <class T> constexpr Vc_INTRINSIC Vc_CONST storage64_t<T> lowest64()
{
    return storage64_t<T>::broadcast(std::numeric_limits<T>::lowest());
}

template <> constexpr Vc_INTRINSIC storage64_t< uchar> lowest64< uchar>() { return {}; }
template <> constexpr Vc_INTRINSIC storage64_t<ushort> lowest64<ushort>() { return {}; }
template <> constexpr Vc_INTRINSIC storage64_t<  uint> lowest64<  uint>() { return {}; }
template <> constexpr Vc_INTRINSIC storage64_t< ulong> lowest64< ulong>() { return {}; }
template <> constexpr Vc_INTRINSIC storage64_t<ullong> lowest64<ullong>() { return {}; }
#endif  // Vc_HAVE_AVX512F

// _2_pow_31{{{1
template <class T> inline typename intrinsic_type<T, 16>::type sse_2_pow_31();
template <> Vc_INTRINSIC Vc_CONST __m128  sse_2_pow_31< float>() { return broadcast16( float(1u << 31)); }
#ifdef Vc_HAVE_SSE2
template <> Vc_INTRINSIC Vc_CONST __m128d sse_2_pow_31<double>() { return broadcast16(double(1u << 31)); }
template <> Vc_INTRINSIC Vc_CONST __m128i sse_2_pow_31<  uint>() { return lowest16<int>(); }
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
template <class T> inline typename intrinsic_type<T, 32>::type avx_2_pow_31();
template <> Vc_INTRINSIC Vc_CONST __m256  avx_2_pow_31< float>() { return _mm256_broadcast_ss(&avx_const::_2_pow_31); }
template <> Vc_INTRINSIC Vc_CONST __m256d avx_2_pow_31<double>() { return broadcast32(double(1u << 31)); }
template <> Vc_INTRINSIC Vc_CONST __m256i avx_2_pow_31<  uint>() { return lowest32<int>(); }
#endif  // Vc_HAVE_AVX

Vc_INTRINSIC __m128i shift_msb_to_lsb(__m128i v)
{
#if defined Vc_GCC && Vc_GCC < 0x60400 && defined Vc_HAVE_AVX512F &&                     \
    !defined Vc_HAVE_AVX512VL
    // GCC miscompiles to `vpsrlw xmm0, xmm0, xmm16` for KNL even though AVX512VL is
    // not available.
    asm("vpsrlw $15,%0,%0" : "+x"(v));
    return v;
#else
    return _mm_srli_epi16(v, 15);
#endif
}

#ifdef Vc_HAVE_AVX2
Vc_INTRINSIC __m256i shift_msb_to_lsb(__m256i v)
{
#if defined Vc_GCC && Vc_GCC < 0x60400 && defined Vc_HAVE_AVX512F &&                     \
    !defined Vc_HAVE_AVX512VL
    // GCC miscompiles to `vpsrlw xmm0, xmm0, xmm16` for KNL even though AVX512VL is
    // not available.
    asm("vpsrlw $15,%0,%0" : "+x"(v));
    return v;
#else
    return _mm256_srli_epi16(v, 15);
#endif
}
#endif  // Vc_HAVE_AVX2

// slli_epi16{{{1
template <int n> Vc_INTRINSIC __m128i slli_epi16(__m128i v)
{
#if defined Vc_GCC && Vc_GCC < 0x60400 && defined Vc_HAVE_AVX512F &&                     \
    !defined Vc_HAVE_AVX512VL
    // GCC miscompiles to `vpsllw xmm0, xmm0, xmm16` for KNL even though AVX512VL is
    // not available.
    asm("vpsllw %1,%0,%0" : "+x"(v) : "i"(n));
    return v;
#else
    return _mm_slli_epi16(v, n);
#endif
}
#ifdef Vc_HAVE_AVX2
template <int n> Vc_INTRINSIC __m256i slli_epi16(__m256i v)
{
#if defined Vc_GCC && Vc_GCC < 0x60400 && defined Vc_HAVE_AVX512F &&                     \
    !defined Vc_HAVE_AVX512VL
    // GCC miscompiles to `vpsllw xmm0, xmm0, xmm16` for KNL even though AVX512VL is
    // not available.
    asm("vpsllw %1,%0,%0" : "+x"(v) : "i"(n));
    return v;
#else
    return _mm256_slli_epi16(v, n);
#endif
}
#endif  // Vc_HAVE_AVX2

// srli_epi16{{{1
template <int n> Vc_INTRINSIC __m128i srli_epi16(__m128i v)
{
#if defined Vc_GCC && Vc_GCC < 0x60400 && defined Vc_HAVE_AVX512F &&                     \
    !defined Vc_HAVE_AVX512VL
    // GCC miscompiles to `vpsllw xmm0, xmm0, xmm16` for KNL even though AVX512VL is
    // not available.
    asm("vpsrlw %1,%0,%0" : "+x"(v) : "i"(n));
    return v;
#else
    return _mm_srli_epi16(v, n);
#endif
}
#ifdef Vc_HAVE_AVX2
template <int n> Vc_INTRINSIC __m256i srli_epi16(__m256i v)
{
#if defined Vc_GCC && Vc_GCC < 0x60400 && defined Vc_HAVE_AVX512F &&                     \
    !defined Vc_HAVE_AVX512VL
    // GCC miscompiles to `vpsllw xmm0, xmm0, xmm16` for KNL even though AVX512VL is
    // not available.
    asm("vpsrlw %1,%0,%0" : "+x"(v) : "i"(n));
    return v;
#else
    return _mm256_srli_epi16(v, n);
#endif
}
#endif  // Vc_HAVE_AVX2

// SSE/AVX intrinsics emulation{{{1
#ifdef Vc_HAVE_SSE2
#if defined(Vc_IMPL_XOP)
Vc_INTRINSIC __m128i cmpgt_epu16(__m128i a, __m128i b) { return _mm_comgt_epu16(a, b); }
Vc_INTRINSIC __m128i cmplt_epu32(__m128i a, __m128i b) { return _mm_comlt_epu32(a, b); }
Vc_INTRINSIC __m128i cmpgt_epu32(__m128i a, __m128i b) { return _mm_comgt_epu32(a, b); }
Vc_INTRINSIC __m128i cmplt_epu64(__m128i a, __m128i b) { return _mm_comlt_epu64(a, b); }
Vc_INTRINSIC __m128i cmpgt_epu64(__m128i a, __m128i b) { return _mm_comgt_epu64(a, b); }
#else
Vc_INTRINSIC __m128i Vc_CONST cmpgt_epu16(__m128i a, __m128i b)
{
    return _mm_cmpgt_epi16(_mm_xor_si128(a, lowest16<short>()),
                           _mm_xor_si128(b, lowest16<short>()));
}
Vc_INTRINSIC __m128i Vc_CONST cmplt_epu32(__m128i a, __m128i b)
{
    return _mm_cmplt_epi32(_mm_xor_si128(a, lowest16<int>()),
                           _mm_xor_si128(b, lowest16<int>()));
}
Vc_INTRINSIC __m128i Vc_CONST cmpgt_epu32(__m128i a, __m128i b)
{
    return _mm_cmpgt_epi32(_mm_xor_si128(a, lowest16<int>()),
                           _mm_xor_si128(b, lowest16<int>()));
}
Vc_INTRINSIC __m128i Vc_CONST cmpgt_epi64(__m128i a, __m128i b)
{
#ifdef Vc_IMPL_SSE4_2
    return _mm_cmpgt_epi64(a, b);
#else
    const auto aa = _mm_xor_si128(a, _mm_srli_epi64(lowest16<int>(), 32));
    const auto bb = _mm_xor_si128(b, _mm_srli_epi64(lowest16<int>(), 32));
    const auto gt = _mm_cmpgt_epi32(aa, bb);
    const auto eq = _mm_cmpeq_epi32(aa, bb);
    // Algorithm:
    // 1. if the high 32 bits of gt are true, make the full 64 bits true
    // 2. if the high 32 bits of gt are false and the high 32 bits of eq are true,
    //    duplicate the low 32 bits of gt to the high 32 bits (note that this requires
    //    unsigned compare on the lower 32 bits, which is the reason for the xors
    //    above)
    // 3. else make the full 64 bits false

    const auto gt2 =
        _mm_shuffle_epi32(gt, 0xf5);  // dup the high 32 bits to the low 32 bits
    const auto lo = _mm_shuffle_epi32(_mm_and_si128(_mm_srli_epi64(eq, 32), gt), 0xa0);
    return _mm_or_si128(gt2, lo);
#endif
}
Vc_INTRINSIC __m128i Vc_CONST cmpgt_epu64(__m128i a, __m128i b)
{
    return cmpgt_epi64(_mm_xor_si128(a, lowest16<llong>()),
                       _mm_xor_si128(b, lowest16<llong>()));
}
#endif

#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_SSE4_1
Vc_INTRINSIC Vc_CONST __m128i cmpeq_epi64(__m128i a, __m128i b)
{
    return _mm_cmpeq_epi64(a, b);
}
template <int index> Vc_INTRINSIC Vc_CONST int extract_epi32(__m128i v)
{
    return _mm_extract_epi32(v, index);
}
Vc_INTRINSIC Vc_CONST __m128d blendv_pd(__m128d a, __m128d b, __m128d c)
{
    return _mm_blendv_pd(a, b, c);
}
Vc_INTRINSIC Vc_CONST __m128 blendv_ps(__m128 a, __m128 b, __m128 c)
{
    return _mm_blendv_ps(a, b, c);
}
Vc_INTRINSIC Vc_CONST __m128i blendv_epi8(__m128i a, __m128i b, __m128i c)
{
    return _mm_blendv_epi8(a, b, c);
}
Vc_INTRINSIC Vc_CONST __m128i cvtepu8_epi16(__m128i epu8)
{
    return _mm_cvtepu8_epi16(epu8);
}
Vc_INTRINSIC Vc_CONST __m128i cvtepi8_epi16(__m128i epi8)
{
    return _mm_cvtepi8_epi16(epi8);
}
Vc_INTRINSIC Vc_CONST __m128i cvtepu16_epi32(__m128i epu16)
{
    return _mm_cvtepu16_epi32(epu16);
}
Vc_INTRINSIC Vc_CONST __m128i cvtepi16_epi32(__m128i epu16)
{
    return _mm_cvtepi16_epi32(epu16);
}
Vc_INTRINSIC Vc_CONST __m128i cvtepu8_epi32(__m128i epu8)
{
    return _mm_cvtepu8_epi32(epu8);
}
Vc_INTRINSIC Vc_CONST __m128i cvtepi8_epi32(__m128i epi8)
{
    return _mm_cvtepi8_epi32(epi8);
}
Vc_INTRINSIC Vc_PURE __m128i stream_load_si128(__m128i *mem)
{
    return _mm_stream_load_si128(mem);
}
#else  // Vc_HAVE_SSE4_1
Vc_INTRINSIC Vc_CONST __m128  blendv_ps(__m128  a, __m128  b, __m128  c) {
    return _mm_or_ps(_mm_andnot_ps(c, a), _mm_and_ps(c, b));
}

#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC Vc_CONST __m128d blendv_pd(__m128d a, __m128d b, __m128d c) {
    return _mm_or_pd(_mm_andnot_pd(c, a), _mm_and_pd(c, b));
}
Vc_INTRINSIC Vc_CONST __m128i blendv_epi8(__m128i a, __m128i b, __m128i c) {
    return _mm_or_si128(_mm_andnot_si128(c, a), _mm_and_si128(c, b));
}

Vc_INTRINSIC Vc_CONST __m128i cmpeq_epi64(__m128i a, __m128i b) {
    auto tmp = _mm_cmpeq_epi32(a, b);
    return _mm_and_si128(tmp, _mm_shuffle_epi32(tmp, 1*1 + 0*4 + 3*16 + 2*64));
}
template <int index> Vc_INTRINSIC Vc_CONST int extract_epi32(__m128i v)
{
#ifdef Vc_USE_BUILTIN_VECTOR_TYPES
    typedef int int32v4 __attribute__((__vector_size__(16)));
    return reinterpret_cast<const may_alias<int32v4> &>(v)[index];
#else
    return _mm_cvtsi128_si32(_mm_srli_si128(v, index * 4));
#endif
}

Vc_INTRINSIC Vc_CONST __m128i cvtepu8_epi16(__m128i epu8) {
    return _mm_unpacklo_epi8(epu8, _mm_setzero_si128());
}
Vc_INTRINSIC Vc_CONST __m128i cvtepi8_epi16(__m128i epi8) {
    return _mm_unpacklo_epi8(epi8, _mm_cmplt_epi8(epi8, _mm_setzero_si128()));
}
Vc_INTRINSIC Vc_CONST __m128i cvtepu16_epi32(__m128i epu16) {
    return _mm_unpacklo_epi16(epu16, _mm_setzero_si128());
}
Vc_INTRINSIC Vc_CONST __m128i cvtepi16_epi32(__m128i epu16) {
    return _mm_unpacklo_epi16(epu16, _mm_cmplt_epi16(epu16, _mm_setzero_si128()));
}
Vc_INTRINSIC Vc_CONST __m128i cvtepu8_epi32(__m128i epu8) {
    return cvtepu16_epi32(cvtepu8_epi16(epu8));
}
Vc_INTRINSIC Vc_CONST __m128i cvtepi8_epi32(__m128i epi8) {
    const __m128i neg = _mm_cmplt_epi8(epi8, _mm_setzero_si128());
    const __m128i epi16 = _mm_unpacklo_epi8(epi8, neg);
    return _mm_unpacklo_epi16(epi16, _mm_unpacklo_epi8(neg, neg));
}
Vc_INTRINSIC Vc_PURE __m128i stream_load_si128(__m128i *mem) {
    return _mm_load_si128(mem);
}
#endif  // Vc_HAVE_SSE2
#endif  // Vc_HAVE_SSE4_1

// blend{{{1
Vc_INTRINSIC Vc_CONST __m128 blend(__m128 mask, __m128 at0, __m128 at1)
{
    return blendv_ps(at0, at1, mask);
}

#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC Vc_CONST __m128d blend(__m128d mask, __m128d at0, __m128d at1)
{
    return blendv_pd(at0, at1, mask);
}
Vc_INTRINSIC Vc_CONST __m128i blend(__m128i mask, __m128i at0, __m128i at1)
{
    return blendv_epi8(at0, at1, mask);
}
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC Vc_CONST __m256  blend(__m256  mask, __m256  at0, __m256  at1)
{
    return _mm256_blendv_ps(at0, at1, mask);
}
Vc_INTRINSIC Vc_CONST __m256d blend(__m256d mask, __m256d at0, __m256d at1)
{
    return _mm256_blendv_pd(at0, at1, mask);
}
#ifdef Vc_HAVE_AVX2
Vc_INTRINSIC Vc_CONST __m256i blend(__m256i mask, __m256i at0, __m256i at1)
{
    return _mm256_blendv_epi8(at0, at1, mask);
}
#endif  // Vc_HAVE_AVX2
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC Vc_CONST __mmask8 blend(__mmask8 mask, __mmask8 at0, __mmask8 at1)
{
    return (mask & at1) | (~mask & at0);
}
Vc_INTRINSIC Vc_CONST __mmask16 blend(__mmask16 mask, __mmask16 at0, __mmask16 at1)
{
    return _mm512_kor(_mm512_kand(mask, at1), _mm512_kandn(mask, at0));
}
Vc_INTRINSIC Vc_CONST __m512  blend(__mmask16 mask, __m512 at0, __m512 at1)
{
    return _mm512_mask_mov_ps(at0, mask, at1);
}
Vc_INTRINSIC Vc_CONST __m512d blend(__mmask8 mask, __m512d at0, __m512d at1)
{
    return _mm512_mask_mov_pd(at0, mask, at1);
}
Vc_INTRINSIC Vc_CONST __m512i blend(__mmask8 mask, __m512i at0, __m512i at1)
{
    return _mm512_mask_mov_epi64(at0, mask, at1);
}
Vc_INTRINSIC Vc_CONST __m512i blend(__mmask16 mask, __m512i at0, __m512i at1)
{
    return _mm512_mask_mov_epi32(at0, mask, at1);
}
#ifdef Vc_HAVE_AVX512BW
Vc_INTRINSIC Vc_CONST __mmask32 blend(__mmask32 mask, __mmask32 at0, __mmask32 at1)
{
    return (mask & at1) | (~mask & at0);
}
Vc_INTRINSIC Vc_CONST __mmask64 blend(__mmask64 mask, __mmask64 at0, __mmask64 at1)
{
    return (mask & at1) | (~mask & at0);
}
Vc_INTRINSIC Vc_CONST __m512i blend(__mmask32 mask, __m512i at0, __m512i at1)
{
    return _mm512_mask_mov_epi16(at0, mask, at1);
}
Vc_INTRINSIC Vc_CONST __m512i blend(__mmask64 mask, __m512i at0, __m512i at1)
{
    return _mm512_mask_mov_epi8(at0, mask, at1);
}
#endif  // Vc_HAVE_AVX512BW
#endif  // Vc_HAVE_AVX512F

// testc{{{1
#ifdef Vc_HAVE_SSE4_1
Vc_INTRINSIC Vc_CONST int testc(__m128  a, __m128  b) { return _mm_testc_si128(_mm_castps_si128(a), _mm_castps_si128(b)); }
Vc_INTRINSIC Vc_CONST int testc(__m128d a, __m128d b) { return _mm_testc_si128(_mm_castpd_si128(a), _mm_castpd_si128(b)); }
Vc_INTRINSIC Vc_CONST int testc(__m128i a, __m128i b) { return _mm_testc_si128(a, b); }
#endif  // Vc_HAVE_SSE4_1

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC Vc_CONST int testc(__m256  a, __m256  b) { return _mm256_testc_ps(a, b); }
Vc_INTRINSIC Vc_CONST int testc(__m256d a, __m256d b) { return _mm256_testc_pd(a, b); }
Vc_INTRINSIC Vc_CONST int testc(__m256i a, __m256i b) { return _mm256_testc_si256(a, b); }
#endif  // Vc_HAVE_AVX

// testallset{{{1
Vc_INTRINSIC Vc_CONST bool testallset(__mmask8 a) {
    if constexpr (have_avx512dq) {
        return _kortestc_mask8_u8(a, a);
    } else {
        return a == 0xff;
    }
}

Vc_INTRINSIC Vc_CONST bool testallset(__mmask16 a) { return _kortestc_mask16_u8(a, a); }

#ifdef Vc_HAVE_AVX512BW
Vc_INTRINSIC Vc_CONST bool testallset(__mmask32 a) {
#ifdef Vc_WORKAROUND_PR85538
    return a == 0xffffffffU;
#else
    return _kortestc_mask32_u8(a, a);
#endif
}
Vc_INTRINSIC Vc_CONST bool testallset(__mmask64 a) {
#ifdef Vc_WORKAROUND_PR85538
    return a == 0xffffffffffffffffULL;
#else
    return _kortestc_mask64_u8(a, a);
#endif
}
#endif  // Vc_HAVE_AVX512BW

// testz{{{1
#ifdef Vc_HAVE_SSE4_1
Vc_INTRINSIC Vc_CONST int testz(__m128  a, __m128  b) { return _mm_testz_si128(_mm_castps_si128(a), _mm_castps_si128(b)); }
Vc_INTRINSIC Vc_CONST int testz(__m128d a, __m128d b) { return _mm_testz_si128(_mm_castpd_si128(a), _mm_castpd_si128(b)); }
Vc_INTRINSIC Vc_CONST int testz(__m128i a, __m128i b) { return _mm_testz_si128(a, b); }
#endif  // Vc_HAVE_SSE4_1
#ifdef Vc_HAVE_AVX
Vc_INTRINSIC Vc_CONST int testz(__m256  a, __m256  b) { return _mm256_testz_ps(a, b); }
Vc_INTRINSIC Vc_CONST int testz(__m256d a, __m256d b) { return _mm256_testz_pd(a, b); }
Vc_INTRINSIC Vc_CONST int testz(__m256i a, __m256i b) { return _mm256_testz_si256(a, b); }
#endif  // Vc_HAVE_AVX

// testnzc{{{1
#ifdef Vc_HAVE_SSE4_1
Vc_INTRINSIC Vc_CONST int testnzc(__m128  a, __m128  b) { return _mm_testnzc_si128(_mm_castps_si128(a), _mm_castps_si128(b)); }
Vc_INTRINSIC Vc_CONST int testnzc(__m128d a, __m128d b) { return _mm_testnzc_si128(_mm_castpd_si128(a), _mm_castpd_si128(b)); }
Vc_INTRINSIC Vc_CONST int testnzc(__m128i a, __m128i b) { return _mm_testnzc_si128(a, b); }
#endif  // Vc_HAVE_SSE4_1
#ifdef Vc_HAVE_AVX
Vc_INTRINSIC Vc_CONST int testnzc(__m256  a, __m256  b) { return _mm256_testnzc_ps(a, b); }
Vc_INTRINSIC Vc_CONST int testnzc(__m256d a, __m256d b) { return _mm256_testnzc_pd(a, b); }
Vc_INTRINSIC Vc_CONST int testnzc(__m256i a, __m256i b) { return _mm256_testnzc_si256(a, b); }
#endif  // Vc_HAVE_AVX

// movemask{{{1
Vc_INTRINSIC Vc_CONST int movemask(__m128  a) { return _mm_movemask_ps(a); }
#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC Vc_CONST int movemask(__m128d a) { return _mm_movemask_pd(a); }
Vc_INTRINSIC Vc_CONST int movemask(__m128i a) { return _mm_movemask_epi8(a); }

Vc_INTRINSIC Vc_CONST int movemask_epi16(__m128i a) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512BW
    return _mm_cmp_epi16_mask(a, zero<__m128i>(), _MM_CMPINT_NE);
#else
    return _mm_movemask_epi8(_mm_packs_epi16(a, zero<__m128i>()));
#endif
}
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC Vc_CONST int movemask(__m256i a) {
#ifdef Vc_HAVE_AVX2
    return _mm256_movemask_epi8(a);
#else
    return _mm_movemask_epi8(lo128(a)) | (_mm_movemask_epi8(hi128(a)) << 16);
#endif
}
Vc_INTRINSIC Vc_CONST int movemask(__m256d a) { return _mm256_movemask_pd(a); }
Vc_INTRINSIC Vc_CONST int movemask(__m256  a) { return _mm256_movemask_ps(a); }

Vc_INTRINSIC Vc_CONST int movemask_epi16(__m256i a) {
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512BW
    return _mm256_cmp_epi16_mask(a, zero<__m256i>(), _MM_CMPINT_NE);
#else
    return _mm_movemask_epi8(_mm_packs_epi16(lo128(a), hi128(a)));
#endif
}

#endif  // Vc_HAVE_AVX

// fixup_avx_xzyw{{{1
template <class T, class Traits = builtin_traits<T>> Vc_INTRINSIC T fixup_avx_xzyw(T a)
{
    using V = std::conditional_t<std::is_floating_point_v<typename Traits::value_type>,
                                 __m256d, __m256i>;
    const V x = reinterpret_cast<V>(a);
    return reinterpret_cast<T>(V{x[0], x[2], x[1], x[3]});
}

// AVX512: convert_mask{{{1
template <size_t EntrySize, size_t VectorSize> struct convert_mask_return_type;

template <size_t VectorSize> struct convert_mask_return_type<1, VectorSize> {
    using type = builtin_type_t<schar, VectorSize>;
};
template <size_t VectorSize> struct convert_mask_return_type<2, VectorSize> {
    using type = builtin_type_t<short, VectorSize / 2>;
};
template <size_t VectorSize> struct convert_mask_return_type<4, VectorSize> {
    using type = builtin_type_t<int, VectorSize / 4>;
};
template <size_t VectorSize> struct convert_mask_return_type<8, VectorSize> {
    using type = builtin_type_t<llong, VectorSize / 8>;
};
template <size_t EntrySize, size_t VectorSize>
using convert_mask_return_type_t =
    typename convert_mask_return_type<EntrySize, VectorSize>::type;

template <size_t EntrySize, size_t VectorSize>
inline convert_mask_return_type_t<EntrySize, VectorSize> convert_mask(__mmask8);
template <size_t EntrySize, size_t VectorSize>
inline convert_mask_return_type_t<EntrySize, VectorSize> convert_mask(__mmask16);
template <size_t EntrySize, size_t VectorSize>
inline convert_mask_return_type_t<EntrySize, VectorSize> convert_mask(__mmask32);
template <size_t EntrySize, size_t VectorSize>
inline convert_mask_return_type_t<EntrySize, VectorSize> convert_mask(__mmask64);

template <size_t EntrySize, size_t VectorSize, size_t N>
Vc_INTRINSIC auto convert_mask(std::bitset<N> bs)
{
    static_assert(VectorSize == N * EntrySize, "");
    return convert_mask<EntrySize, VectorSize>(__mmask8(bs.to_ulong()));
}

template <size_t EntrySize, size_t VectorSize>
Vc_INTRINSIC auto convert_mask(std::bitset<16> bs)
{
    static_assert(VectorSize / EntrySize == 16, "");
    return convert_mask<EntrySize, VectorSize>(__mmask16(bs.to_ulong()));
}

template <size_t EntrySize, size_t VectorSize>
Vc_INTRINSIC auto convert_mask(std::bitset<32> bs)
{
    static_assert(VectorSize / EntrySize == 32, "");
    return convert_mask<EntrySize, VectorSize>(__mmask32(bs.to_ulong()));
}

template <size_t EntrySize, size_t VectorSize>
Vc_INTRINSIC auto convert_mask(std::bitset<64> bs)
{
    static_assert(VectorSize / EntrySize == 64, "");
    return convert_mask<EntrySize, VectorSize>(__mmask64(bs.to_ullong()));
}

#ifdef Vc_HAVE_AVX512F
#ifdef Vc_HAVE_AVX512BW
template <> Vc_INTRINSIC builtin_type_t<schar, 64> convert_mask<1, 64>(__mmask64 k)
{
    return reinterpret_cast<builtin_type_t<schar, 64>>(_mm512_movm_epi8(k));
}
template <> Vc_INTRINSIC builtin_type_t<short, 32> convert_mask<2, 64>(__mmask32 k)
{
    return reinterpret_cast<builtin_type_t<short, 32>>(_mm512_movm_epi16(k));
}
#endif  // Vc_HAVE_AVX512BW

template <> Vc_INTRINSIC builtin_type_t<schar, 16> convert_mask<1, 16>(__mmask16 k)
{
    if constexpr (have_avx512bw_vl) {
        return reinterpret_cast<builtin_type_t<schar, 16>>(_mm_movm_epi8(k));
    } else if constexpr (have_avx512bw) {
        return reinterpret_cast<builtin_type_t<schar, 16>>(lo128(_mm512_movm_epi8(k)));
    } else {
        auto as32bits = _mm512_maskz_mov_epi32(k, ~__m512i());
        auto as16bits =
            fixup_avx_xzyw(_mm256_packs_epi32(lo256(as32bits), hi256(as32bits)));
        return reinterpret_cast<builtin_type_t<schar, 16>>(
            _mm_packs_epi16(lo128(as16bits), hi128(as16bits)));
    }
}

template <> Vc_INTRINSIC builtin_type_t<short, 8> convert_mask<2, 16>(__mmask8 k)
{
#if defined Vc_HAVE_AVX512BW && defined Vc_HAVE_AVX512VL
    return reinterpret_cast<builtin_type_t<short, 8>>(_mm_movm_epi16(k));
#elif defined Vc_HAVE_AVX512BW
    return reinterpret_cast<builtin_type_t<short, 8>>(lo128(_mm512_movm_epi16(k)));
#else
    auto as32bits =
#ifdef Vc_HAVE_AVX512VL
        _mm256_maskz_mov_epi32(k, ~__m256i());
#else
        lo256(_mm512_maskz_mov_epi32(k, ~__m512i()));
#endif
    return reinterpret_cast<builtin_type_t<short, 8>>(
        _mm_packs_epi32(lo128(as32bits), hi128(as32bits)));
#endif
}

template <> Vc_INTRINSIC builtin_type_t<schar, 32> convert_mask<1, 32>(__mmask32 k)
{
#if defined Vc_HAVE_AVX512BW && defined Vc_HAVE_AVX512VL
    return reinterpret_cast<builtin_type_t<schar, 32>>(_mm256_movm_epi8(k));
#elif defined Vc_HAVE_AVX512BW
    return reinterpret_cast<builtin_type_t<schar, 32>>(lo256(_mm512_movm_epi8(k)));
#else
    auto as16bits =  // 0 16 1 17 ... 15 31
        _mm512_srli_epi32(_mm512_maskz_mov_epi32(k, ~__m512i()), 16) |
        _mm512_slli_epi32(_mm512_maskz_mov_epi32(k >> 16, ~__m512i()), 16);
    auto _0_16_1_17 = fixup_avx_xzyw(_mm256_packs_epi16(
        lo256(as16bits), hi256(as16bits))  // 0 16 1 17 2 18 3 19 8 24 9 25 ...
    );
    // deinterleave:
    return reinterpret_cast<builtin_type_t<schar, 32>>(fixup_avx_xzyw(_mm256_shuffle_epi8(
        _0_16_1_17,  // 0 16 1 17 2 ...
        _mm256_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6,
                         8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13,
                         15))));  // 0-7 16-23 8-15 24-31 -> xzyw
    // 0-3  8-11 16-19 24-27
    // 4-7 12-15 20-23 28-31
#endif
}

template <> Vc_INTRINSIC builtin_type_t<short, 16> convert_mask<2, 32>(__mmask16 k)
{
#if defined Vc_HAVE_AVX512BW && defined Vc_HAVE_AVX512VL
    return reinterpret_cast<builtin_type_t<short, 16>>(_mm256_movm_epi16(k));
#elif defined Vc_HAVE_AVX512BW
    return reinterpret_cast<builtin_type_t<short, 16>>(lo256(_mm512_movm_epi16(k)));
#else
    auto as32bits = _mm512_maskz_mov_epi32(k, ~__m512i());
    return reinterpret_cast<builtin_type_t<short, 16>>(
        fixup_avx_xzyw(_mm256_packs_epi32(lo256(as32bits), hi256(as32bits))));
#endif
}

template <> Vc_INTRINSIC builtin_type_t<int, 16> convert_mask<4, 64>(__mmask16 k)
{
    return reinterpret_cast<builtin_type_t<int, 16>>(
#ifdef Vc_HAVE_AVX512DQ
        _mm512_movm_epi32(k)
#else
        _mm512_maskz_mov_epi32(k, ~__m512i())
#endif
    );
}

template <> Vc_INTRINSIC builtin_type_t<llong, 8> convert_mask<8, 64>(__mmask8 k)
{
    return reinterpret_cast<builtin_type_t<llong, 8>>(
#ifdef Vc_HAVE_AVX512DQ
        _mm512_movm_epi64(k)
#else
        _mm512_maskz_mov_epi64(k, ~__m512i())
#endif
    );
}

template <> Vc_INTRINSIC builtin_type_t<int, 4> convert_mask<4, 16>(__mmask8 k)
{
    return reinterpret_cast<builtin_type_t<int, 4>>(
#if defined Vc_HAVE_AVX512VL && Vc_HAVE_AVX512DQ
        _mm_movm_epi32(k)
#elif defined Vc_HAVE_AVX512DQ
        lo128(_mm512_movm_epi32(k))
#elif defined Vc_HAVE_AVX512VL
        _mm_maskz_mov_epi32(k, ~__m128i())
#else
        lo128(_mm512_maskz_mov_epi32(k, ~__m512i()))
#endif
    );
}
template <> Vc_INTRINSIC builtin_type_t<llong, 2> convert_mask<8, 16>(__mmask8 k)
{
    return reinterpret_cast<builtin_type_t<llong, 2>>(
#if defined Vc_HAVE_AVX512VL && Vc_HAVE_AVX512DQ
        _mm_movm_epi64(k)
#elif defined Vc_HAVE_AVX512DQ
        lo128(_mm512_movm_epi64(k))
#elif defined Vc_HAVE_AVX512VL
        _mm_maskz_mov_epi64(k, ~__m128i())
#else
        lo128(_mm512_maskz_mov_epi64(k, ~__m512i()))
#endif
    );
}
template <> Vc_INTRINSIC builtin_type_t<int, 8> convert_mask<4, 32>(__mmask8 k)
{
    return reinterpret_cast<builtin_type_t<int, 8>>(
#if defined Vc_HAVE_AVX512VL && Vc_HAVE_AVX512DQ
        _mm256_movm_epi32(k)
#elif defined Vc_HAVE_AVX512DQ
        lo256(_mm512_movm_epi32(k))
#elif defined Vc_HAVE_AVX512VL
        _mm256_maskz_mov_epi32(k, ~__m256i())
#else
        lo256(_mm512_maskz_mov_epi32(k, ~__m512i()))
#endif
    );
}
template <> Vc_INTRINSIC builtin_type_t<llong, 4> convert_mask<8, 32>(__mmask8 k)
{
    return reinterpret_cast<builtin_type_t<llong, 4>>(
#if defined Vc_HAVE_AVX512VL && Vc_HAVE_AVX512DQ
        _mm256_movm_epi64(k)
#elif defined Vc_HAVE_AVX512DQ
        lo256(_mm512_movm_epi64(k))
#elif defined Vc_HAVE_AVX512VL
        _mm256_maskz_mov_epi64(k, ~__m256i())
#else
        lo256(_mm512_maskz_mov_epi64(k, ~__m512i()))
#endif
    );
}

#endif  // Vc_HAVE_AVX512F

// xor_{{{1
Vc_INTRINSIC __m128  xor_(__m128  a, __m128  b) { return _mm_xor_ps(a, b); }
#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC __m128d xor_(__m128d a, __m128d b) { return _mm_xor_pd(a, b); }
Vc_INTRINSIC __m128i xor_(__m128i a, __m128i b) { return _mm_xor_si128(a, b); }
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC __m256  xor_(__m256  a, __m256  b) { return _mm256_xor_ps(a, b); }
Vc_INTRINSIC __m256d xor_(__m256d a, __m256d b) { return _mm256_xor_pd(a, b); }
Vc_INTRINSIC __m256i xor_(__m256i a, __m256i b) {
#ifdef Vc_HAVE_AVX2
    return _mm256_xor_si256(a, b);
#else
    return _mm256_castps_si256(xor_(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
#endif
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
#ifdef Vc_HAVE_AVX512DQ
Vc_INTRINSIC __m512  xor_(__m512  a, __m512  b) { return _mm512_xor_ps(a, b); }
Vc_INTRINSIC __m512d xor_(__m512d a, __m512d b) { return _mm512_xor_pd(a, b); }
#else   // Vc_HAVE_AVX512DQ
Vc_INTRINSIC __m512 xor_(__m512 a, __m512 b)
{
    return intrin_cast<__m512>(
        _mm512_xor_epi32(intrin_cast<__m512i>(a), intrin_cast<__m512i>(b)));
}
Vc_INTRINSIC __m512d xor_(__m512d a, __m512d b)
{
    return intrin_cast<__m512d>(
        _mm512_xor_epi64(intrin_cast<__m512i>(a), intrin_cast<__m512i>(b)));
}
#endif  // Vc_HAVE_AVX512DQ
Vc_INTRINSIC __m512i xor_(__m512i a, __m512i b) { return _mm512_xor_epi32(a, b); }
#endif  // Vc_HAVE_AVX512F

// or_{{{1
Vc_INTRINSIC __m128 or_(__m128 a, __m128 b) { return _mm_or_ps(a, b); }
#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC __m128d or_(__m128d a, __m128d b) { return _mm_or_pd(a, b); }
Vc_INTRINSIC __m128i or_(__m128i a, __m128i b) { return _mm_or_si128(a, b); }
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC __m256  or_(__m256  a, __m256  b) { return _mm256_or_ps(a, b); }
Vc_INTRINSIC __m256d or_(__m256d a, __m256d b) { return _mm256_or_pd(a, b); }
Vc_INTRINSIC __m256i or_(__m256i a, __m256i b) {
#ifdef Vc_HAVE_AVX2
    return _mm256_or_si256(a, b);
#else
    return _mm256_castps_si256(or_(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
#endif
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC __m512  or_(__m512  a, __m512  b) { return _mm512_or_ps(a, b); }
Vc_INTRINSIC __m512d or_(__m512d a, __m512d b) { return _mm512_or_pd(a, b); }
Vc_INTRINSIC __m512i or_(__m512i a, __m512i b) { return _mm512_or_epi32(a, b); }
#endif  // Vc_HAVE_AVX512F

// and_{{{1
Vc_INTRINSIC __m128 and_(__m128 a, __m128 b) { return _mm_and_ps(a, b); }
#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC __m128d and_(__m128d a, __m128d b) { return _mm_and_pd(a, b); }
Vc_INTRINSIC __m128i and_(__m128i a, __m128i b) { return _mm_and_si128(a, b); }
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC __m256  and_(__m256  a, __m256  b) { return _mm256_and_ps(a, b); }
Vc_INTRINSIC __m256d and_(__m256d a, __m256d b) { return _mm256_and_pd(a, b); }
Vc_INTRINSIC __m256i and_(__m256i a, __m256i b) {
#ifdef Vc_HAVE_AVX2
    return _mm256_and_si256(a, b);
#else
    return _mm256_castps_si256(and_(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
#endif
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC __m512i and_(__m512i a, __m512i b) { return _mm512_and_epi32(a, b); }
#ifdef Vc_HAVE_AVX512DQ
Vc_INTRINSIC __m512  and_(__m512  a, __m512  b) { return _mm512_and_ps(a, b); }
Vc_INTRINSIC __m512d and_(__m512d a, __m512d b) { return _mm512_and_pd(a, b); }
#else  // Vc_HAVE_AVX512DQ
Vc_INTRINSIC __m512  and_(__m512  a, __m512  b) { return _mm512_castsi512_ps(and_(_mm512_castps_si512(a),_mm512_castps_si512(b))); }
Vc_INTRINSIC __m512d and_(__m512d a, __m512d b) { return _mm512_castsi512_pd(and_(_mm512_castpd_si512(a),_mm512_castpd_si512(b))); }
#endif  // Vc_HAVE_AVX512DQ
#endif  // Vc_HAVE_AVX512F

// andnot_{{{1
Vc_INTRINSIC __m128 andnot_(__m128 a, __m128 b) { return _mm_andnot_ps(a, b); }
#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC __m128d andnot_(__m128d a, __m128d b) { return _mm_andnot_pd(a, b); }
Vc_INTRINSIC __m128i andnot_(__m128i a, __m128i b) { return _mm_andnot_si128(a, b); }
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC __m256  andnot_(__m256  a, __m256  b) { return _mm256_andnot_ps(a, b); }
Vc_INTRINSIC __m256d andnot_(__m256d a, __m256d b) { return _mm256_andnot_pd(a, b); }
Vc_INTRINSIC __m256i andnot_(__m256i a, __m256i b) {
#ifdef Vc_HAVE_AVX2
    return _mm256_andnot_si256(a, b);
#else
    return _mm256_castps_si256(andnot_(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
#endif
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC __m512  andnot_(__m512  a, __m512  b) { return _mm512_andnot_ps(a, b); }
Vc_INTRINSIC __m512d andnot_(__m512d a, __m512d b) { return _mm512_andnot_pd(a, b); }
Vc_INTRINSIC __m512i andnot_(__m512i a, __m512i b) { return _mm512_andnot_epi32(a, b); }

Vc_INTRINSIC __m512d andnot_(__mmask8  k, __m512d a) { return _mm512_maskz_mov_pd(~k, a); }
Vc_INTRINSIC __m512  andnot_(__mmask16 k, __m512  a) { return _mm512_maskz_mov_ps(~k, a); }
Vc_INTRINSIC __m512i andnot_(__mmask8  k, __m512i a) { return _mm512_maskz_mov_epi64(~k, a); }
Vc_INTRINSIC __m512i andnot_(__mmask16 k, __m512i a) { return _mm512_maskz_mov_epi32(~k, a); }
#ifdef Vc_HAVE_AVX512BW
Vc_INTRINSIC __m512i andnot_(__mmask32 k, __m512i a) { return _mm512_maskz_mov_epi16(~k, a); }
Vc_INTRINSIC __m512i andnot_(__mmask64 k, __m512i a) { return _mm512_maskz_mov_epi8 (~k, a); }
#endif  // Vc_HAVE_AVX512BW
#endif  // Vc_HAVE_AVX512F

// not_{{{1
Vc_INTRINSIC __m128  not_(__m128  a) { return andnot_(a, allone<__m128 >()); }
#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC __m128d not_(__m128d a) { return andnot_(a, allone<__m128d>()); }
Vc_INTRINSIC __m128i not_(__m128i a) { return andnot_(a, allone<__m128i>()); }
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC __m256  not_(__m256  a) { return andnot_(a, allone<__m256 >()); }
Vc_INTRINSIC __m256d not_(__m256d a) { return andnot_(a, allone<__m256d>()); }
Vc_INTRINSIC __m256i not_(__m256i a) { return andnot_(a, allone<__m256i>()); }
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC __m512  not_(__m512  a) { return andnot_(a, allone<__m512 >()); }
Vc_INTRINSIC __m512d not_(__m512d a) { return andnot_(a, allone<__m512d>()); }
Vc_INTRINSIC __m512i not_(__m512i a) { return andnot_(a, allone<__m512i>()); }

Vc_INTRINSIC __mmask8  not_(__mmask8  a) { return ~a; }
Vc_INTRINSIC __mmask16 not_(__mmask16 a) { return ~a; }
#ifdef Vc_HAVE_AVX512BW
Vc_INTRINSIC __mmask32 not_(__mmask32 a) { return ~a; }
Vc_INTRINSIC __mmask64 not_(__mmask64 a) { return ~a; }
#endif  // Vc_HAVE_AVX512BW
#endif  // Vc_HAVE_AVX512F

// shift_right{{{1
template <int n> Vc_INTRINSIC __m128  shift_right(__m128  v);
template <> Vc_INTRINSIC __m128  shift_right< 0>(__m128  v) { return v; }
template <> Vc_INTRINSIC __m128  shift_right<16>(__m128   ) { return _mm_setzero_ps(); }

#ifdef Vc_HAVE_SSE2
template <int n> Vc_INTRINSIC __m128  shift_right(__m128  v) { return _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v), n)); }
template <int n> Vc_INTRINSIC __m128d shift_right(__m128d v) { return _mm_castsi128_pd(_mm_srli_si128(_mm_castpd_si128(v), n)); }
template <int n> Vc_INTRINSIC __m128i shift_right(__m128i v) { return _mm_srli_si128(v, n); }

template <> Vc_INTRINSIC __m128  shift_right< 8>(__m128  v) { return _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(v), _mm_setzero_pd())); }
template <> Vc_INTRINSIC __m128d shift_right< 0>(__m128d v) { return v; }
template <> Vc_INTRINSIC __m128d shift_right< 8>(__m128d v) { return _mm_unpackhi_pd(v, _mm_setzero_pd()); }
template <> Vc_INTRINSIC __m128d shift_right<16>(__m128d  ) { return _mm_setzero_pd(); }
template <> Vc_INTRINSIC __m128i shift_right< 0>(__m128i v) { return v; }
template <> Vc_INTRINSIC __m128i shift_right<16>(__m128i  ) { return _mm_setzero_si128(); }
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX2
template <int n> Vc_INTRINSIC __m256 shift_right(__m256 v)
{
    __m256i vi = _mm256_castps_si256(v);
    return _mm256_castsi256_ps(
        n < 16 ? _mm256_srli_si256(vi, n)
               : _mm256_srli_si256(_mm256_permute2x128_si256(vi, vi, 0x81), n));
}
template <> Vc_INTRINSIC __m256 shift_right<0>(__m256 v) { return v; }
template <> Vc_INTRINSIC __m256 shift_right<16>(__m256 v) { return intrin_cast<__m256>(lo128(v)); }
template <int n> Vc_INTRINSIC __m256d shift_right(__m256d v)
{
    __m256i vi = _mm256_castpd_si256(v);
    return _mm256_castsi256_pd(
        n < 16 ? _mm256_srli_si256(vi, n)
               : _mm256_srli_si256(_mm256_permute2x128_si256(vi, vi, 0x81), n));
}
template <> Vc_INTRINSIC __m256d shift_right<0>(__m256d v) { return v; }
template <> Vc_INTRINSIC __m256d shift_right<16>(__m256d v) { return intrin_cast<__m256d>(lo128(v)); }
template <int n> Vc_INTRINSIC __m256i shift_right(__m256i v)
{
    return n < 16 ? _mm256_srli_si256(v, n)
                  : _mm256_srli_si256(_mm256_permute2x128_si256(v, v, 0x81), n);
}
template <> Vc_INTRINSIC __m256i shift_right<0>(__m256i v) { return v; }
template <> Vc_INTRINSIC __m256i shift_right<16>(__m256i v) { return _mm256_permute2x128_si256(v, v, 0x81); }
#endif

// popcnt{{{1
Vc_INTRINSIC Vc_CONST unsigned int popcnt4(unsigned int n)
{
#ifdef Vc_IMPL_POPCNT
    return _mm_popcnt_u32(n);
#else
    n = (n & 0x5U) + ((n >> 1) & 0x5U);
    n = (n & 0x3U) + ((n >> 2) & 0x3U);
    return n;
#endif
}
Vc_INTRINSIC Vc_CONST unsigned int popcnt8(unsigned int n)
{
#ifdef Vc_IMPL_POPCNT
    return _mm_popcnt_u32(n);
#else
    n = (n & 0x55U) + ((n >> 1) & 0x55U);
    n = (n & 0x33U) + ((n >> 2) & 0x33U);
    n = (n & 0x0fU) + ((n >> 4) & 0x0fU);
    return n;
#endif
}
Vc_INTRINSIC Vc_CONST unsigned int popcnt16(unsigned int n)
{
#ifdef Vc_IMPL_POPCNT
    return _mm_popcnt_u32(n);
#else
    n = (n & 0x5555U) + ((n >> 1) & 0x5555U);
    n = (n & 0x3333U) + ((n >> 2) & 0x3333U);
    n = (n & 0x0f0fU) + ((n >> 4) & 0x0f0fU);
    n = (n & 0x00ffU) + ((n >> 8) & 0x00ffU);
    return n;
#endif
}
Vc_INTRINSIC Vc_CONST unsigned int popcnt32(unsigned int n)
{
#ifdef Vc_IMPL_POPCNT
    return _mm_popcnt_u32(n);
#else
    n = (n & 0x55555555U) + ((n >> 1) & 0x55555555U);
    n = (n & 0x33333333U) + ((n >> 2) & 0x33333333U);
    n = (n & 0x0f0f0f0fU) + ((n >> 4) & 0x0f0f0f0fU);
    n = (n & 0x00ff00ffU) + ((n >> 8) & 0x00ff00ffU);
    n = (n & 0x0000ffffU) + ((n >>16) & 0x0000ffffU);
    return n;
#endif
}
Vc_INTRINSIC Vc_CONST unsigned int popcnt64(ullong n)
{
#ifdef Vc_IMPL_POPCNT
#ifdef __x86_64__
    return _mm_popcnt_u64(n);
#else   // __x86_64__
    return _mm_popcnt_u32(n) + _mm_popcnt_u32(n >> 32u);
#endif  // __x86_64__
#else
    n = (n & 0x5555555555555555ULL) + ((n >> 1) & 0x5555555555555555ULL);
    n = (n & 0x3333333333333333ULL) + ((n >> 2) & 0x3333333333333333ULL);
    n = (n & 0x0f0f0f0f0f0f0f0fULL) + ((n >> 4) & 0x0f0f0f0f0f0f0f0fULL);
    n = (n & 0x00ff00ff00ff00ffULL) + ((n >> 8) & 0x00ff00ff00ff00ffULL);
    n = (n & 0x0000ffff0000ffffULL) + ((n >>16) & 0x0000ffff0000ffffULL);
    n = (n & 0x00000000ffffffffULL) + ((n >>32) & 0x00000000ffffffffULL);
    return n;
#endif
}

// firstbit{{{1
Vc_INTRINSIC Vc_CONST int firstbit(ullong bits)
{
#ifdef Vc_HAVE_BMI1
#ifdef __x86_64__
    return _tzcnt_u64(bits);
#else
    uint lo = bits;
    uint hi = bits >> 32u;
    if (lo == 0u) {
        return 32u + _tzcnt_u32(hi);
    } else {
        return _tzcnt_u32(lo);
    }
#endif
#else   // Vc_HAVE_BMI1
    return __builtin_ctzll(bits);
#endif  // Vc_HAVE_BMI1
}

#ifdef Vc_MSVC
#pragma intrinsic(_BitScanForward)
#pragma intrinsic(_BitScanReverse)
#endif

Vc_INTRINSIC Vc_CONST auto firstbit(uint x)
{
#if defined Vc_ICC || defined Vc_GCC
    return _bit_scan_forward(x);
#elif defined Vc_CLANG || defined Vc_APPLECLANG
    return __builtin_ctz(x);
#elif defined Vc_MSVC
    unsigned long index;
    _BitScanForward(&index, x);
    return index;
#else
#error "Implementation for firstbit(uint) is missing"
#endif
}

Vc_INTRINSIC Vc_CONST auto firstbit(llong bits) { return firstbit(ullong(bits)); }
#if LONG_MAX == LLONG_MAX
Vc_INTRINSIC Vc_CONST auto firstbit(ulong bits) { return firstbit(ullong(bits)); }
Vc_INTRINSIC Vc_CONST auto firstbit(long bits) { return firstbit(ullong(bits)); }
#endif  // long uses 64 bits
template <class T> Vc_INTRINSIC Vc_CONST auto firstbit(T bits)
{
    static_assert(sizeof(T) <= sizeof(uint),
                  "there's a missing overload to call the 64-bit variant");
    return firstbit(uint(bits));
}

// lastbit{{{1
Vc_INTRINSIC Vc_CONST int lastbit(ullong bits)
{
#ifdef Vc_HAVE_BMI1
#ifdef __x86_64__
    return 63u - _lzcnt_u64(bits);
#else
    uint lo = bits;
    uint hi = bits >> 32u;
    if (hi == 0u) {
        return 31u - _lzcnt_u32(lo);
    } else {
        return 63u - _lzcnt_u32(hi);
    }
#endif
#else   // Vc_HAVE_BMI1
    return 63 - __builtin_clzll(bits);
#endif  // Vc_HAVE_BMI1
}

Vc_INTRINSIC Vc_CONST auto lastbit(uint x)
{
#if defined Vc_ICC || defined Vc_GCC
    return _bit_scan_reverse(x);
#elif defined Vc_CLANG || defined Vc_APPLECLANG
    return 31 - __builtin_clz(x);
#elif defined(Vc_MSVC)
    unsigned long index;
    _BitScanReverse(&index, x);
    return index;
#else
#error "Implementation for lastbit(uint) is missing"
#endif
}

Vc_INTRINSIC Vc_CONST auto lastbit(llong bits) { return lastbit(ullong(bits)); }
#if LONG_MAX == LLONG_MAX
Vc_INTRINSIC Vc_CONST auto lastbit(ulong bits) { return lastbit(ullong(bits)); }
Vc_INTRINSIC Vc_CONST auto lastbit(long bits) { return lastbit(ullong(bits)); }
#endif  // long uses 64 bits
template <class T> Vc_INTRINSIC Vc_CONST auto lastbit(T bits)
{
    static_assert(sizeof(T) <= sizeof(uint),
                  "there's a missing overload to call the 64-bit variant");
    return lastbit(uint(bits));
}

// mask_count{{{1
template <size_t Size> int mask_count(__m128 );
template<> Vc_INTRINSIC Vc_CONST int mask_count<4>(__m128  k)
{
#ifdef Vc_IMPL_POPCNT
    return _mm_popcnt_u32(_mm_movemask_ps(k));
#elif defined Vc_HAVE_SSE2
    auto x = _mm_srli_epi32(_mm_castps_si128(k), 31);
    x = _mm_add_epi32(x, _mm_shuffle_epi32(x, _MM_SHUFFLE(0, 1, 2, 3)));
    x = _mm_add_epi32(x, _mm_shufflelo_epi16(x, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si32(x);
#else
    return popcnt4(_mm_movemask_ps(k));
#endif
}

#ifdef Vc_HAVE_SSE2
template <size_t Size> int mask_count(__m128i);
template <size_t Size> int mask_count(__m128d);
template<> Vc_INTRINSIC Vc_CONST int mask_count<2>(__m128d k)
{
    int mask = _mm_movemask_pd(k);
    return (mask & 1) + (mask >> 1);
}
template<> Vc_INTRINSIC Vc_CONST int mask_count<2>(__m128i k)
{
    return mask_count<2>(_mm_castsi128_pd(k));
}

template<> Vc_INTRINSIC Vc_CONST int mask_count<4>(__m128i k)
{
    return mask_count<4>(_mm_castsi128_ps(k));
}

template<> Vc_INTRINSIC Vc_CONST int mask_count<8>(__m128i k)
{
#ifdef Vc_IMPL_POPCNT
    return _mm_popcnt_u32(_mm_movemask_epi8(k)) / 2;
#else
    auto x = srli_epi16<15>(k);
    x = _mm_add_epi16(x, _mm_shuffle_epi32(x, _MM_SHUFFLE(0, 1, 2, 3)));
    x = _mm_add_epi16(x, _mm_shufflelo_epi16(x, _MM_SHUFFLE(0, 1, 2, 3)));
    x = _mm_add_epi16(x, _mm_shufflelo_epi16(x, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_extract_epi16(x, 0);
#endif
}

template<> Vc_INTRINSIC Vc_CONST int mask_count<16>(__m128i k)
{
    return popcnt16(_mm_movemask_epi8(k));
}
#endif  // Vc_HAVE_SSE2

// mask_to_int{{{1
template <size_t Size> inline int mask_to_int(__m128 ) { static_assert(Size == Size, "Size value not implemented"); return 0; }
#ifdef Vc_HAVE_SSE2
template <size_t Size> inline int mask_to_int(__m128d) { static_assert(Size == Size, "Size value not implemented"); return 0; }
template <size_t Size> inline int mask_to_int(__m128i) { static_assert(Size == Size, "Size value not implemented"); return 0; }
#endif  // Vc_HAVE_SSE2
#ifdef Vc_HAVE_AVX
template <size_t Size> inline int mask_to_int(__m256 ) { static_assert(Size == Size, "Size value not implemented"); return 0; }
template <size_t Size> inline int mask_to_int(__m256d) { static_assert(Size == Size, "Size value not implemented"); return 0; }
template <size_t Size> inline int mask_to_int(__m256i) { static_assert(Size == Size, "Size value not implemented"); return 0; }
#endif  // Vc_HAVE_AVX
#ifdef Vc_HAVE_AVX512F
template <size_t Size> inline uint mask_to_int(__mmask8  k) { return k; }
template <size_t Size> inline uint mask_to_int(__mmask16 k) { return k; }
template <size_t Size> inline uint mask_to_int(__mmask32 k) { return k; }
template <size_t Size> inline ullong mask_to_int(__mmask64 k) { return k; }
#endif  // Vc_HAVE_AVX512F

template<> Vc_INTRINSIC Vc_CONST int mask_to_int<4>(__m128 k)
{
    return _mm_movemask_ps(k);
}
#ifdef Vc_HAVE_SSE2
template<> Vc_INTRINSIC Vc_CONST int mask_to_int<2>(__m128d k)
{
    return _mm_movemask_pd(k);
}
template<> Vc_INTRINSIC Vc_CONST int mask_to_int<2>(__m128i k)
{
    return _mm_movemask_pd(_mm_castsi128_pd(k));
}
template<> Vc_INTRINSIC Vc_CONST int mask_to_int<4>(__m128i k)
{
    return _mm_movemask_ps(_mm_castsi128_ps(k));
}
template<> Vc_INTRINSIC Vc_CONST int mask_to_int<8>(__m128i k)
{
    return _mm_movemask_epi8(_mm_packs_epi16(k, _mm_setzero_si128()));
}
template<> Vc_INTRINSIC Vc_CONST int mask_to_int<16>(__m128i k)
{
    return _mm_movemask_epi8(k);
}
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
template <> Vc_INTRINSIC Vc_CONST int mask_to_int<4>(__m256d k)
{
    return _mm256_movemask_pd(k);
}
template <> Vc_INTRINSIC Vc_CONST int mask_to_int<4>(__m256i k)
{
    return mask_to_int<4>(_mm256_castsi256_pd(k));
}

template <> Vc_INTRINSIC Vc_CONST int mask_to_int<8>(__m256  k)
{
    return _mm256_movemask_ps(k);
}
template <> Vc_INTRINSIC Vc_CONST int mask_to_int<8>(__m256i k)
{
    return mask_to_int<8>(_mm256_castsi256_ps(k));
}

#ifdef Vc_HAVE_AVX2
template <> Vc_INTRINSIC Vc_CONST int mask_to_int<16>(__m256i k)
{
#if defined Vc_HAVE_AVX512VL && defined Vc_HAVE_AVX512BW
    return _mm256_movepi16_mask(k);
#else
    return _mm256_movemask_epi8(_mm256_packs_epi16(k, intrin_cast<__m256i>(hi128(k)))) &
           0xffff;
#endif
}

template <> Vc_INTRINSIC Vc_CONST int mask_to_int<32>(__m256i k)
{
    return _mm256_movemask_epi8(k);
}
#endif  // Vc_HAVE_AVX2
#endif  // Vc_HAVE_AVX

// long cmp{{{1
#ifdef Vc_HAVE_AVX512F
template <int = sizeof(long)> Vc_INTRINSIC auto cmpeq_long_mask(__m512i x, __m512i y);
template <> Vc_INTRINSIC auto cmpeq_long_mask<8>(__m512i x, __m512i y)
{
    return _mm512_cmpeq_epi64_mask(x, y);
}
template <> Vc_INTRINSIC auto cmpeq_long_mask<4>(__m512i x, __m512i y)
{
    return _mm512_cmpeq_epi32_mask(x, y);
}

template <int = sizeof(long)> Vc_INTRINSIC auto cmplt_long_mask(__m512i x, __m512i y);
template <> Vc_INTRINSIC auto cmplt_long_mask<8>(__m512i x, __m512i y)
{
    return _mm512_cmplt_epi64_mask(x, y);
}
template <> Vc_INTRINSIC auto cmplt_long_mask<4>(__m512i x, __m512i y)
{
    return _mm512_cmplt_epi32_mask(x, y);
}

template <int = sizeof(long)> Vc_INTRINSIC auto cmplt_ulong_mask(__m512i x, __m512i y);
template <> Vc_INTRINSIC auto cmplt_ulong_mask<8>(__m512i x, __m512i y)
{
    return _mm512_cmplt_epu64_mask(x, y);
}
template <> Vc_INTRINSIC auto cmplt_ulong_mask<4>(__m512i x, __m512i y)
{
    return _mm512_cmplt_epu32_mask(x, y);
}

template <int = sizeof(long)> Vc_INTRINSIC auto cmple_long_mask(__m512i x, __m512i y);
template <> Vc_INTRINSIC auto cmple_long_mask<8>(__m512i x, __m512i y)
{
    return _mm512_cmple_epi64_mask(x, y);
}
template <> Vc_INTRINSIC auto cmple_long_mask<4>(__m512i x, __m512i y)
{
    return _mm512_cmple_epi32_mask(x, y);
}

template <int = sizeof(long)> Vc_INTRINSIC auto cmple_ulong_mask(__m512i x, __m512i y);
template <> Vc_INTRINSIC auto cmple_ulong_mask<8>(__m512i x, __m512i y)
{
    return _mm512_cmple_epu64_mask(x, y);
}
template <> Vc_INTRINSIC auto cmple_ulong_mask<4>(__m512i x, __m512i y)
{
    return _mm512_cmple_epu32_mask(x, y);
}
#endif  // Vc_HAVE_AVX512F

// loads{{{1
/**
 * \internal
 * Abstraction for simplifying load operations in the SSE/AVX/AVX512 implementations
 *
 * \note The number in the suffix signifies the number of Bytes
 */
#ifdef Vc_HAVE_SSE2
template <class T> Vc_INTRINSIC __m128i load2(const T *mem, when_aligned<2>)
{
    assertCorrectAlignment<unsigned short>(mem);
    static_assert(sizeof(T) == 1, "expected argument with sizeof == 1");
    return _mm_cvtsi32_si128(*reinterpret_cast<const unsigned short *>(mem));
}
template <class T> Vc_INTRINSIC __m128i load2(const T *mem, when_unaligned<2>)
{
    static_assert(sizeof(T) == 1, "expected argument with sizeof == 1");
    short tmp;
    std::memcpy(&tmp, mem, 2);
    return _mm_cvtsi32_si128(tmp);
}
#endif  // Vc_HAVE_SSE2

template <class F> Vc_INTRINSIC __m128 load4(const float *mem, F)
{
    assertCorrectAlignment<float>(mem);
    return _mm_load_ss(mem);
}

#ifdef Vc_HAVE_SSE2
template <class F> Vc_INTRINSIC __m128i load4(const int *mem, F)
{
    assertCorrectAlignment<int>(mem);
    return _mm_cvtsi32_si128(mem[0]);
}
template <class F> Vc_INTRINSIC __m128i load4(const unsigned int *mem, F)
{
    assertCorrectAlignment<unsigned int>(mem);
    return _mm_cvtsi32_si128(mem[0]);
}
template <class T, class F> Vc_INTRINSIC __m128i load4(const T *mem, F)
{
    static_assert(sizeof(T) <= 2, "expected argument with sizeof <= 2");
    int tmp;
    std::memcpy(&tmp, mem, 4);
    return _mm_cvtsi32_si128(tmp);
}
#endif  // Vc_HAVE_SSE2

template <class F> Vc_INTRINSIC __m128 load8(const float *mem, F)
{
    assertCorrectAlignment<float>(mem);
    if constexpr (have_sse2 && is_aligned_v<F, alignof(double)>) {
        return _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(mem)));
    } else {
        return _mm_loadl_pi(_mm_undefined_ps(), reinterpret_cast<const __m64 *>(mem));
    }
}

#ifdef Vc_HAVE_SSE2
template <class F> Vc_INTRINSIC __m128d load8(const double *mem, F)
{
    assertCorrectAlignment<double>(mem);
    return _mm_load_sd(mem);
}
template <class T, class F> Vc_INTRINSIC __m128i load8(const T *mem, F)
{
    assertCorrectAlignment<T>(mem);
    static_assert(std::is_integral<T>::value, "load8<T> is only intended for integral T");
    return _mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem));
}
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_SSE
Vc_INTRINSIC __m128 load16(const float *mem, when_aligned<16>)
{
    assertCorrectAlignment<__m128>(mem);
    return _mm_load_ps(mem);
}
Vc_INTRINSIC __m128 load16(const float *mem, when_unaligned<16>)
{
    return _mm_loadu_ps(mem);
}
#endif  // Vc_HAVE_SSE

#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC __m128d load16(const double *mem, when_aligned<16>)
{
    assertCorrectAlignment<__m128d>(mem);
    return _mm_load_pd(mem);
}
Vc_INTRINSIC __m128d load16(const double *mem, when_unaligned<16>)
{
    return _mm_loadu_pd(mem);
}
template <class T> Vc_INTRINSIC __m128i load16(const T *mem, when_aligned<16>)
{
    assertCorrectAlignment<__m128i>(mem);
    static_assert(std::is_integral<T>::value, "load16<T> is only intended for integral T");
    return _mm_load_si128(reinterpret_cast<const __m128i *>(mem));
}
template <class T> Vc_INTRINSIC __m128i load16(const T *mem, when_unaligned<16>)
{
    static_assert(std::is_integral<T>::value, "load16<T> is only intended for integral T");
    return _mm_loadu_si128(reinterpret_cast<const __m128i *>(mem));
}
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC __m256 load32(const float *mem, when_aligned<32>)
{
    assertCorrectAlignment<__m256>(mem);
    return _mm256_load_ps(mem);
}
Vc_INTRINSIC __m256 load32(const float *mem, when_unaligned<32>)
{
    return _mm256_loadu_ps(mem);
}
Vc_INTRINSIC __m256d load32(const double *mem, when_aligned<32>)
{
    assertCorrectAlignment<__m256d>(mem);
    return _mm256_load_pd(mem);
}
Vc_INTRINSIC __m256d load32(const double *mem, when_unaligned<32>)
{
    return _mm256_loadu_pd(mem);
}
template <class T> Vc_INTRINSIC __m256i load32(const T *mem, when_aligned<32>)
{
    assertCorrectAlignment<__m256i>(mem);
    static_assert(std::is_integral<T>::value, "load32<T> is only intended for integral T");
    return _mm256_load_si256(reinterpret_cast<const __m256i *>(mem));
}
template <class T> Vc_INTRINSIC __m256i load32(const T *mem, when_unaligned<32>)
{
    static_assert(std::is_integral<T>::value, "load32<T> is only intended for integral T");
    return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(mem));
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC __m512 load64(const float *mem, when_aligned<64>)
{
    assertCorrectAlignment<__m512>(mem);
    return _mm512_load_ps(mem);
}
Vc_INTRINSIC __m512 load64(const float *mem, when_unaligned<64>)
{
    return _mm512_loadu_ps(mem);
}
Vc_INTRINSIC __m512d load64(const double *mem, when_aligned<64>)
{
    assertCorrectAlignment<__m512d>(mem);
    return _mm512_load_pd(mem);
}
Vc_INTRINSIC __m512d load64(const double *mem, when_unaligned<64>)
{
    return _mm512_loadu_pd(mem);
}
template <class T>
Vc_INTRINSIC __m512i load64(const T *mem, when_aligned<64>)
{
    assertCorrectAlignment<__m512i>(mem);
    static_assert(std::is_integral<T>::value, "load64<T> is only intended for integral T");
    return _mm512_load_si512(mem);
}
template <class T>
Vc_INTRINSIC __m512i load64(const T *mem, when_unaligned<64>)
{
    static_assert(std::is_integral<T>::value, "load64<T> is only intended for integral T");
    return _mm512_loadu_si512(mem);
}
#endif

// stores{{{1
#ifdef Vc_HAVE_SSE
Vc_INTRINSIC void store4(__m128 v, float *mem, when_aligned<alignof(float)>)
{
    assertCorrectAlignment<float>(mem);
    *mem = _mm_cvtss_f32(v);
}

Vc_INTRINSIC void store4(__m128 v, float *mem, when_unaligned<alignof(float)>)
{
    *mem = _mm_cvtss_f32(v);
}

Vc_INTRINSIC void store8(__m128 v, float *mem, when_aligned<alignof(__m64)>)
{
    assertCorrectAlignment<__m64>(mem);
    _mm_storel_pi(reinterpret_cast<__m64 *>(mem), v);
}

Vc_INTRINSIC void store8(__m128 v, float *mem, when_unaligned<alignof(__m64)>)
{
    _mm_storel_pi(reinterpret_cast<__m64 *>(mem), v);
}

Vc_INTRINSIC void store16(__m128 v, float *mem, when_aligned<16>)
{
    assertCorrectAlignment<__m128>(mem);
    _mm_store_ps(mem, v);
}
Vc_INTRINSIC void store16(__m128 v, float *mem, when_unaligned<16>)
{
    _mm_storeu_ps(mem, v);
}
#endif  // Vc_HAVE_SSE

#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC void store8(__m128d v, double *mem, when_aligned<alignof(double)>)
{
    assertCorrectAlignment<double>(mem);
    *mem = _mm_cvtsd_f64(v);
}

Vc_INTRINSIC void store8(__m128d v, double *mem, when_unaligned<alignof(double)>)
{
    *mem = _mm_cvtsd_f64(v);
}

Vc_INTRINSIC void store16(__m128d v, double *mem, when_aligned<16>)
{
    assertCorrectAlignment<__m128d>(mem);
    _mm_store_pd(mem, v);
}
Vc_INTRINSIC void store16(__m128d v, double *mem, when_unaligned<16>)
{
    _mm_storeu_pd(mem, v);
}

template <class T> Vc_INTRINSIC void store2(__m128i v, T *mem, when_aligned<alignof(ushort)>)
{
    assertCorrectAlignment<ushort>(mem);
    static_assert(std::is_integral<T>::value && sizeof(T) <= 2,
                  "store4<T> is only intended for integral T with sizeof(T) <= 2");
    *reinterpret_cast<may_alias<ushort> *>(mem) = uint(_mm_cvtsi128_si32(v));
}

template <class T> Vc_INTRINSIC void store2(__m128i v, T *mem, when_unaligned<alignof(ushort)>)
{
    static_assert(std::is_integral<T>::value && sizeof(T) <= 2,
                  "store4<T> is only intended for integral T with sizeof(T) <= 2");
    const uint tmp(_mm_cvtsi128_si32(v));
    std::memcpy(mem, &tmp, 2);
}

template <class T> Vc_INTRINSIC void store4(__m128i v, T *mem, when_aligned<alignof(int)>)
{
    assertCorrectAlignment<int>(mem);
    static_assert(std::is_integral<T>::value && sizeof(T) <= 4,
                  "store4<T> is only intended for integral T with sizeof(T) <= 4");
    *reinterpret_cast<may_alias<int> *>(mem) = _mm_cvtsi128_si32(v);
}

template <class T> Vc_INTRINSIC void store4(__m128i v, T *mem, when_unaligned<alignof(int)>)
{
    static_assert(std::is_integral<T>::value && sizeof(T) <= 4,
                  "store4<T> is only intended for integral T with sizeof(T) <= 4");
    const int tmp = _mm_cvtsi128_si32(v);
    std::memcpy(mem, &tmp, 4);
}

template <class T> Vc_INTRINSIC void store8(__m128i v, T *mem, when_aligned<8>)
{
    assertCorrectAlignment<__m64>(mem);
    static_assert(std::is_integral<T>::value, "store8<T> is only intended for integral T");
    _mm_storel_epi64(reinterpret_cast<__m128i *>(mem), v);
}

template <class T> Vc_INTRINSIC void store8(__m128i v, T *mem, when_unaligned<8>)
{
    static_assert(std::is_integral<T>::value, "store8<T> is only intended for integral T");
    _mm_storel_epi64(reinterpret_cast<__m128i *>(mem), v);
}

template <class T> Vc_INTRINSIC void store16(__m128i v, T *mem, when_aligned<16>)
{
    assertCorrectAlignment<__m128i>(mem);
    static_assert(std::is_integral<T>::value, "store16<T> is only intended for integral T");
    _mm_store_si128(reinterpret_cast<__m128i *>(mem), v);
}
template <class T> Vc_INTRINSIC void store16(__m128i v, T *mem, when_unaligned<16>)
{
    static_assert(std::is_integral<T>::value, "store16<T> is only intended for integral T");
    _mm_storeu_si128(reinterpret_cast<__m128i *>(mem), v);
}
#endif  // Vc_HAVE_SSE2

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC void store32(__m256 v, float *mem, when_aligned<32>)
{
    assertCorrectAlignment<__m256>(mem);
    _mm256_store_ps(mem, v);
}
Vc_INTRINSIC void store32(__m256 v, float *mem, when_unaligned<32>)
{
    _mm256_storeu_ps(mem, v);
}
Vc_INTRINSIC void store32(__m256d v, double *mem, when_aligned<32>)
{
    assertCorrectAlignment<__m256d>(mem);
    _mm256_store_pd(mem, v);
}
Vc_INTRINSIC void store32(__m256d v, double *mem, when_unaligned<32>)
{
    _mm256_storeu_pd(mem, v);
}
template <class T> Vc_INTRINSIC void store32(__m256i v, T *mem, when_aligned<32>)
{
    assertCorrectAlignment<__m256i>(mem);
    static_assert(std::is_integral<T>::value, "store32<T> is only intended for integral T");
    _mm256_store_si256(reinterpret_cast<__m256i *>(mem), v);
}
template <class T> Vc_INTRINSIC void store32(__m256i v, T *mem, when_unaligned<32>)
{
    static_assert(std::is_integral<T>::value, "store32<T> is only intended for integral T");
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(mem), v);
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC void store64(__m512 v, float *mem, when_aligned<64>)
{
    assertCorrectAlignment<__m512>(mem);
    _mm512_store_ps(mem, v);
}
Vc_INTRINSIC void store64(__m512 v, float *mem, when_unaligned<64>)
{
    _mm512_storeu_ps(mem, v);
}
Vc_INTRINSIC void store64(__m512d v, double *mem, when_aligned<64>)
{
    assertCorrectAlignment<__m512d>(mem);
    _mm512_store_pd(mem, v);
}
Vc_INTRINSIC void store64(__m512d v, double *mem, when_unaligned<64>)
{
    _mm512_storeu_pd(mem, v);
}
template <class T>
Vc_INTRINSIC void store64(__m512i v, T *mem, when_aligned<64>)
{
    assertCorrectAlignment<__m512i>(mem);
    static_assert(std::is_integral<T>::value, "store64<T> is only intended for integral T");
    _mm512_store_si512(mem, v);
}
template <class T>
Vc_INTRINSIC void store64(__m512i v, T *mem, when_unaligned<64>)
{
    static_assert(std::is_integral<T>::value, "store64<T> is only intended for integral T");
    _mm512_storeu_si512(mem, v);
}
#endif

#ifdef Vc_HAVE_AVX512F
template <class T, class F, size_t N>
Vc_INTRINSIC void store_n_bytes(size_constant<N>, __m512i v, T *mem, F)
{
    std::memcpy(mem, &v, N);
}
template <class T, class F>
Vc_INTRINSIC void store_n_bytes(size_constant<4>, __m512i v, T *mem, F f)
{
    store4(lo128(v), mem, f);
}
template <class T, class F>
Vc_INTRINSIC void store_n_bytes(size_constant<8>, __m512i v, T *mem, F f)
{
    store8(lo128(v), mem, f);
}
template <class T, class F>
Vc_INTRINSIC void store_n_bytes(size_constant<16>, __m512i v, T *mem, F f)
{
    store16(lo128(v), mem, f);
}
template <class T, class F>
Vc_INTRINSIC void store_n_bytes(size_constant<32>, __m512i v, T *mem, F f)
{
    store32(lo256(v), mem, f);
}
template <class T, class F>
Vc_INTRINSIC void store_n_bytes(size_constant<64>, __m512i v, T *mem, F f)
{
    store64(v, mem, f);
}
#endif  // Vc_HAVE_AVX512F

#ifdef Vc_HAVE_AVX
template <class T, class F, size_t N>
Vc_INTRINSIC void store_n_bytes(size_constant<N>, __m256i v, T *mem, F)
{
    std::memcpy(mem, &v, N);
}
template <class T, class F>
Vc_INTRINSIC void store_n_bytes(size_constant<4>, __m256i v, T *mem, F f)
{
    store4(lo128(v), mem, f);
}
template <class T, class F>
Vc_INTRINSIC void store_n_bytes(size_constant<8>, __m256i v, T *mem, F f)
{
    store8(lo128(v), mem, f);
}
template <class T, class F>
Vc_INTRINSIC void store_n_bytes(size_constant<16>, __m256i v, T *mem, F f)
{
    store16(lo128(v), mem, f);
}
template <class T, class F>
Vc_INTRINSIC void store_n_bytes(size_constant<32>, __m256i v, T *mem, F f)
{
    store32(v, mem, f);
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_SSE2
template <class T, class F, size_t N>
Vc_INTRINSIC void store_n_bytes(size_constant<N>, __m128i v, T *mem, F)
{
    std::memcpy(mem, &v, N);
}
template <class T, class F>
Vc_INTRINSIC void store_n_bytes(size_constant<4>, __m128i v, T *mem, F f)
{
    store4(v, mem, f);
}
template <class T, class F>
Vc_INTRINSIC void store_n_bytes(size_constant<8>, __m128i v, T *mem, F f)
{
    store8(v, mem, f);
}
template <class T, class F>
Vc_INTRINSIC void store_n_bytes(size_constant<16>, __m128i v, T *mem, F f)
{
    store16(v, mem, f);
}
#endif  // Vc_HAVE_SSE2

// }}}1
// integer sign-extension {{{
builtin_type_t<short,  8> constexpr Vc_INTRINSIC sign_extend16(builtin_type_t<char, 16> x) { return __builtin_ia32_pmovsxbw128(x); }
builtin_type_t<  int,  4> constexpr Vc_INTRINSIC sign_extend32(builtin_type_t<char, 16> x) { return __builtin_ia32_pmovsxbd128(x); }
builtin_type_t<llong,  2> constexpr Vc_INTRINSIC sign_extend64(builtin_type_t<char, 16> x) { return __builtin_ia32_pmovsxbq128(x); }
builtin_type_t<  int,  4> constexpr Vc_INTRINSIC sign_extend32(builtin_type_t<short, 8> x) { return __builtin_ia32_pmovsxwd128(x); }
builtin_type_t<llong,  2> constexpr Vc_INTRINSIC sign_extend64(builtin_type_t<short, 8> x) { return __builtin_ia32_pmovsxwq128(x); }
builtin_type_t<llong,  2> constexpr Vc_INTRINSIC sign_extend64(builtin_type_t<  int, 4> x) { return __builtin_ia32_pmovsxdq128(x); }

// }}}
// integer zero-extension {{{
builtin_type_t<short,  8> constexpr Vc_INTRINSIC zero_extend16(builtin_type_t<char, 16> x) { return __builtin_ia32_pmovzxbw128(x); }
builtin_type_t<  int,  4> constexpr Vc_INTRINSIC zero_extend32(builtin_type_t<char, 16> x) { return __builtin_ia32_pmovzxbd128(x); }
builtin_type_t<llong,  2> constexpr Vc_INTRINSIC zero_extend64(builtin_type_t<char, 16> x) { return __builtin_ia32_pmovzxbq128(x); }
builtin_type_t<  int,  4> constexpr Vc_INTRINSIC zero_extend32(builtin_type_t<short, 8> x) { return __builtin_ia32_pmovzxwd128(x); }
builtin_type_t<llong,  2> constexpr Vc_INTRINSIC zero_extend64(builtin_type_t<short, 8> x) { return __builtin_ia32_pmovzxwq128(x); }
builtin_type_t<llong,  2> constexpr Vc_INTRINSIC zero_extend64(builtin_type_t<  int, 4> x) { return __builtin_ia32_pmovzxdq128(x); }

// }}}
// non-converting maskstore (SSE-AVX512BWVL) {{{
template <class T, class F>
Vc_INTRINSIC void maskstore(storage64_t<T> v, T *mem, F,
                            Storage<bool, storage64_t<T>::width> k)
{
    static_assert(sizeof(v) == 64 && have_avx512f);
    if constexpr (have_avx512bw && sizeof(T) == 1) {
        _mm512_mask_storeu_epi8(mem, k, v);
    } else if constexpr (have_avx512bw && sizeof(T) == 2) {
        _mm512_mask_storeu_epi16(mem, k, v);
    } else if constexpr (sizeof(T) == 4) {
        if constexpr (is_aligned_v<F, 64> && std::is_integral_v<T>) {
            _mm512_mask_store_epi32(mem, k, v);
        } else if constexpr (is_aligned_v<F, 64> && std::is_floating_point_v<T>) {
            _mm512_mask_store_ps(mem, k, v);
        } else if constexpr (std::is_integral_v<T>) {
            _mm512_mask_storeu_epi32(mem, k, v);
        } else {
            _mm512_mask_storeu_ps(mem, k, v);
        }
    } else if constexpr (sizeof(T) == 8) {
        if constexpr (is_aligned_v<F, 64> && std::is_integral_v<T>) {
            _mm512_mask_store_epi64(mem, k, v);
        } else if constexpr (is_aligned_v<F, 64> && std::is_floating_point_v<T>) {
            _mm512_mask_store_pd(mem, k, v);
        } else if constexpr (std::is_integral_v<T>) {
            _mm512_mask_storeu_epi64(mem, k, v);
        } else {
            _mm512_mask_storeu_pd(mem, k, v);
        }
    } else {
        using M = std::conditional_t<sizeof(T) == 1, __mmask16, __mmask8>;
        constexpr int N = 16 / sizeof(T);
        _mm_maskmoveu_si128(auto_cast(extract<0, 4>(v.d)),
                            auto_cast(convert_mask<sizeof(T), 16>(M(k))),
                            reinterpret_cast<char *>(mem));
        _mm_maskmoveu_si128(auto_cast(extract<1, 4>(v.d)),
                            auto_cast(convert_mask<sizeof(T), 16>(M(k >> 1 * N))),
                            reinterpret_cast<char *>(mem) + 1 * 16);
        _mm_maskmoveu_si128(auto_cast(extract<2, 4>(v.d)),
                            auto_cast(convert_mask<sizeof(T), 16>(M(k >> 2 * N))),
                            reinterpret_cast<char *>(mem) + 2 * 16);
        _mm_maskmoveu_si128(auto_cast(extract<3, 4>(v.d)),
                            auto_cast(convert_mask<sizeof(T), 16>(M(k >> 3 * N))),
                            reinterpret_cast<char *>(mem) + 3 * 16);
    }
}

template <class T, class F>
Vc_INTRINSIC void maskstore(storage32_t<T> v, T *mem, F, storage32_t<T> k)
{
    static_assert(sizeof(v) == 32 && have_avx);
    if constexpr (have_avx512bw_vl && sizeof(T) == 1) {
        _mm256_mask_storeu_epi8(mem, _mm256_movepi8_mask(k), v);
    } else if constexpr (have_avx512bw_vl && sizeof(T) == 2) {
        _mm256_mask_storeu_epi16(mem, _mm256_movepi16_mask(k), v);
    } else if constexpr (have_avx2 && sizeof(T) == 4 && std::is_integral_v<T>) {
        _mm256_maskstore_epi32(reinterpret_cast<int *>(mem), k, v);
    } else if constexpr (sizeof(T) == 4) {
        _mm256_maskstore_ps(reinterpret_cast<float *>(mem), to_m256i(k), to_m256(v));
    } else if constexpr (have_avx2 && sizeof(T) == 8 && std::is_integral_v<T>) {
        _mm256_maskstore_epi64(reinterpret_cast<llong *>(mem), k, v);
    } else if constexpr (sizeof(T) == 8) {
        _mm256_maskstore_pd(reinterpret_cast<double *>(mem), to_m256i(k), to_m256d(v));
    } else {
        _mm_maskmoveu_si128(lo128(v), lo128(k), reinterpret_cast<char *>(mem));
        _mm_maskmoveu_si128(hi128(v), hi128(k), reinterpret_cast<char *>(mem) + 16);
    }
}

template <class T, class F>
Vc_INTRINSIC void maskstore(storage32_t<T> v, T *mem, F f,
                            Storage<bool, storage32_t<T>::width> k)
{
    static_assert(sizeof(v) == 32 && have_avx512f);
    if constexpr (have_avx512bw_vl && sizeof(T) == 1) {
        _mm256_mask_storeu_epi8(mem, k, v);
    } else if constexpr (have_avx512bw_vl && sizeof(T) == 2) {
        _mm256_mask_storeu_epi16(mem, k, v);
    } else if constexpr (have_avx512vl && sizeof(T) == 4) {
        if constexpr (is_aligned_v<F, 32> && std::is_integral_v<T>) {
            _mm256_mask_store_epi32(mem, k, v);
        } else if constexpr (is_aligned_v<F, 32> && std::is_floating_point_v<T>) {
            _mm256_mask_store_ps(mem, k, v);
        } else if constexpr (std::is_integral_v<T>) {
            _mm256_mask_storeu_epi32(mem, k, v);
        } else {
            _mm256_mask_storeu_ps(mem, k, v);
        }
    } else if constexpr (have_avx512vl && sizeof(T) == 8) {
        if constexpr (is_aligned_v<F, 16> && std::is_integral_v<T>) {
            _mm256_mask_store_epi64(mem, k, v);
        } else if constexpr (is_aligned_v<F, 16> && std::is_floating_point_v<T>) {
            _mm256_mask_store_pd(mem, k, v);
        } else if constexpr (std::is_integral_v<T>) {
            _mm256_mask_storeu_epi64(mem, k, v);
        } else {
            _mm256_mask_storeu_pd(mem, k, v);
        }
    } else {
        maskstore(v, mem, f, convert_mask<sizeof(T), 32>(k));
    }
}

template <class T, class F>
Vc_INTRINSIC void maskstore(storage16_t<T> v, T *mem, F, storage16_t<T> k)
{
    if constexpr (have_avx512bw_vl && sizeof(T) == 1) {
        _mm_mask_storeu_epi8(mem, _mm_movepi8_mask(k), v);
    } else if constexpr (have_avx512bw_vl && sizeof(T) == 2) {
        _mm_mask_storeu_epi16(mem, _mm_movepi16_mask(k), v);
    } else if constexpr (have_avx2 && sizeof(T) == 4 && std::is_integral_v<T>) {
        _mm_maskstore_epi32(reinterpret_cast<int *>(mem), k, v);
    } else if constexpr (have_avx && sizeof(T) == 4) {
        _mm_maskstore_ps(reinterpret_cast<float *>(mem), to_m128i(k), to_m128(v));
    } else if constexpr (have_avx2 && sizeof(T) == 8 && std::is_integral_v<T>) {
        _mm_maskstore_epi64(reinterpret_cast<llong *>(mem), k, v);
    } else if constexpr (have_avx && sizeof(T) == 8) {
        _mm_maskstore_pd(reinterpret_cast<double *>(mem), to_m128i(k), to_m128d(v));
    } else if constexpr (have_sse2) {
        _mm_maskmoveu_si128(to_m128i(v), to_m128i(k), reinterpret_cast<char *>(mem));
    } else {
        execute_n_times<storage16_t<T>::width>([&](auto i) {
            if (k[i]) {
                mem[i] = v[i];
            }
        });
    }
}

template <class T, class F>
Vc_INTRINSIC void maskstore(storage16_t<T> v, T *mem, F f,
                            Storage<bool, storage16_t<T>::width> k)
{
    static_assert(sizeof(v) == 16 && have_avx512f);
    if constexpr (have_avx512bw_vl && sizeof(T) == 1) {
        _mm_mask_storeu_epi8(mem, k, v);
    } else if constexpr (have_avx512bw_vl && sizeof(T) == 2) {
        _mm_mask_storeu_epi16(mem, k, v);
    } else if constexpr (have_avx512vl && sizeof(T) == 4) {
        if constexpr (is_aligned_v<F, 16> && std::is_integral_v<T>) {
            _mm_mask_store_epi32(mem, k, v);
        } else if constexpr (is_aligned_v<F, 16> && std::is_floating_point_v<T>) {
            _mm_mask_store_ps(mem, k, v);
        } else if constexpr (std::is_integral_v<T>) {
            _mm_mask_storeu_epi32(mem, k, v);
        } else {
            _mm_mask_storeu_ps(mem, k, v);
        }
    } else if constexpr (have_avx512vl && sizeof(T) == 8) {
        if constexpr (is_aligned_v<F, 16> && std::is_integral_v<T>) {
            _mm_mask_store_epi64(mem, k, v);
        } else if constexpr (is_aligned_v<F, 16> && std::is_floating_point_v<T>) {
            _mm_mask_store_pd(mem, k, v);
        } else if constexpr (std::is_integral_v<T>) {
            _mm_mask_storeu_epi64(mem, k, v);
        } else {
            _mm_mask_storeu_pd(mem, k, v);
        }
    } else {
        maskstore(v, mem, f, convert_mask<sizeof(T), 16>(k));
    }
}

// }}}
}  // namespace x86
using namespace x86;
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END
#endif  // Vc_HAVE_SSE

#endif  // VC_SIMD_X86_H_
