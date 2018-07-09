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
// missing intrinsics {{{
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

// }}}
// to_<intrin> {{{
template <class T, size_t N>
constexpr Vc_INTRINSIC __m128 to_m128(detail::Storage<T, N> a)
{
    static_assert(N <= 16 / sizeof(T));
    return a.template intrin<__m128>();
}
template <class T, size_t N>
constexpr Vc_INTRINSIC __m128d to_m128d(detail::Storage<T, N> a)
{
    static_assert(N <= 16 / sizeof(T));
    return a.template intrin<__m128d>();
}
template <class T, size_t N>
constexpr Vc_INTRINSIC __m128i to_m128i(detail::Storage<T, N> a)
{
    static_assert(N <= 16 / sizeof(T));
    return a.template intrin<__m128i>();
}

template <class T, size_t N>
constexpr Vc_INTRINSIC __m256 to_m256(detail::Storage<T, N> a)
{
    static_assert(N <= 32 / sizeof(T) && N > 16 / sizeof(T));
    return a.template intrin<__m256>();
}
template <class T, size_t N>
constexpr Vc_INTRINSIC __m256d to_m256d(detail::Storage<T, N> a)
{
    static_assert(N <= 32 / sizeof(T) && N > 16 / sizeof(T));
    return a.template intrin<__m256d>();
}
template <class T, size_t N>
constexpr Vc_INTRINSIC __m256i to_m256i(detail::Storage<T, N> a)
{
    static_assert(N <= 32 / sizeof(T) && N > 16 / sizeof(T));
    return a.template intrin<__m256i>();
}

template <class T, size_t N>
constexpr Vc_INTRINSIC __m512 to_m512(detail::Storage<T, N> a)
{
    static_assert(N <= 64 / sizeof(T) && N > 32 / sizeof(T));
    return a.template intrin<__m512>();
}
template <class T, size_t N>
constexpr Vc_INTRINSIC __m512d to_m512d(detail::Storage<T, N> a)
{
    static_assert(N <= 64 / sizeof(T) && N > 32 / sizeof(T));
    return a.template intrin<__m512d>();
}
template <class T, size_t N>
constexpr Vc_INTRINSIC __m512i to_m512i(detail::Storage<T, N> a)
{
    static_assert(N <= 64 / sizeof(T) && N > 32 / sizeof(T));
    return a.template intrin<__m512i>();
}

// }}}
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
// zeroExtend{{{1
template <class From> struct zeroExtend {
    static_assert(is_intrinsic_v<From>);
    constexpr zeroExtend(From) {};
};
template <> struct zeroExtend<__m128> {
    __m128 d;
    constexpr Vc_INTRINSIC zeroExtend(__m128 x) : d(x) {}

#ifdef Vc_HAVE_AVX
    Vc_INTRINSIC operator __m256()
    {
        return _mm256_insertf128_ps(__m256(), d, 0);
    }
#endif
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

#ifdef Vc_HAVE_SSE2
    Vc_INTRINSIC operator __m256d()
    {
        return _mm256_insertf128_pd(__m256d(), d, 0);
    }
#endif
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

#ifdef Vc_HAVE_SSE2
    Vc_INTRINSIC operator __m256i()
    {
        return _mm256_insertf128_si256(__m256i(), d, 0);
    }
#endif
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

#ifdef Vc_HAVE_AVX512F
template <> struct zeroExtend<__m256> {
    __m256 d;
    constexpr Vc_INTRINSIC zeroExtend(__m256 x) : d(x) {}
    Vc_INTRINSIC operator __m512()
    {
        if constexpr (have_avx512dq) {
            return _mm512_insertf32x8(__m512(), d, 0);
        } else {
            return reinterpret_cast<__m512>(
                _mm512_insertf64x4(__m512d(), reinterpret_cast<__m256d>(d), 0));
        }
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

// extract128{{{1
template <int offset, class T>
constexpr Vc_INTRINSIC auto extract128(T a)
    -> decltype(extract<offset, sizeof(T) / 16>(a))
{
    return extract<offset, sizeof(T) / 16>(a);
}

template <int offset, typename T, size_t N>
constexpr Vc_INTRINSIC storage16_t<T> extract128(Storage<T, N> x)
{
    return extract<offset, sizeof(x) / 16>(x.d);
}

// extract256{{{1
template <int offset, class T>
constexpr Vc_INTRINSIC auto extract256(T a)
    -> decltype(extract<offset, 2>(a))
{
    static_assert(sizeof(T) == 64);
    return extract<offset, 2>(a);
}

// lo/hi128{{{1
template <class T>
constexpr Vc_INTRINSIC auto lo128(T x) -> decltype(extract<0, sizeof(T) / 16>(x))
{
    return extract<0, sizeof(T) / 16>(x);
}
template <class T> constexpr Vc_INTRINSIC auto hi128(T x) -> decltype(extract<1, 2>(x))
{
    static_assert(sizeof(x) == 32);
    return extract<1, 2>(x);
}

template <typename T, size_t N>
constexpr Vc_INTRINSIC storage16_t<T> lo128(Storage<T, N> x)
{
    return extract<0, sizeof(x) / 16>(x.d);
}
template <typename T, size_t N>
constexpr Vc_INTRINSIC storage16_t<T> hi128(Storage<T, N> x)
{
    static_assert(sizeof(x) == 32);
    return extract<1, 2>(x.d);
}

// lo/hi256{{{1
template <class T> constexpr Vc_INTRINSIC auto lo256(T x) -> decltype(extract<0, 2>(x))
{
    static_assert(sizeof(x) == 64);
    return extract<0, 2>(x);
}
template <class T> constexpr Vc_INTRINSIC auto hi256(T x) -> decltype(extract<1, 2>(x))
{
    static_assert(sizeof(x) == 64);
    return extract<1, 2>(x);
}

template <typename T, size_t N>
constexpr Vc_INTRINSIC storage32_t<T> lo256(Storage<T, N> x)
{
    static_assert(sizeof(x) == 64);
    return extract<0, 2>(x.d);
}
template <typename T, size_t N>
constexpr Vc_INTRINSIC storage32_t<T> hi256(Storage<T, N> x)
{
    static_assert(sizeof(x) == 64);
    return extract<1, 2>(x.d);
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
template <class T, size_t N>
constexpr Vc_INTRINSIC intrinsic_type_t<T, N / sizeof(T)> broadcast(T x, size_constant<N>)
{
    return reinterpret_cast<intrinsic_type_t<T, N / sizeof(T)>>(
        builtin_broadcast<N / sizeof(T)>(x));
}
template <class T> constexpr Vc_INTRINSIC auto broadcast16(T x)
{
    return broadcast(x, size_constant<16>());
}
template <class T> constexpr Vc_INTRINSIC auto broadcast32(T x)
{
    return broadcast(x, size_constant<32>());
}
template <class T> constexpr Vc_INTRINSIC auto broadcast64(T x)
{
    return broadcast(x, size_constant<64>());
}

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
#ifdef Vc_HAVE_AVX
template <class T> inline typename intrinsic_type<T, 32>::type avx_2_pow_31();
template <> Vc_INTRINSIC Vc_CONST __m256  avx_2_pow_31< float>() { return _mm256_broadcast_ss(&avx_const::_2_pow_31); }
template <> Vc_INTRINSIC Vc_CONST __m256d avx_2_pow_31<double>() { return builtin_broadcast<4>(double(1u << 31)); }
template <> Vc_INTRINSIC Vc_CONST __m256i avx_2_pow_31<  uint>() { return lowest32<int>(); }
#endif  // Vc_HAVE_AVX

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
#ifdef Vc_HAVE_SSE4_1
Vc_INTRINSIC Vc_CONST __m128i blendv_epi8(__m128i a, __m128i b, __m128i c)
{
    return _mm_blendv_epi8(a, b, c);
}
#else  // Vc_HAVE_SSE4_1

#ifdef Vc_HAVE_SSE2
Vc_INTRINSIC Vc_CONST __m128i blendv_epi8(__m128i a, __m128i b, __m128i c) {
    return _mm_or_si128(_mm_andnot_si128(c, a), _mm_and_si128(c, b));
}

#endif  // Vc_HAVE_SSE2
#endif  // Vc_HAVE_SSE4_1

// blend{{{1
template <class K, class V0, class V1>
Vc_INTRINSIC Vc_CONST auto blend(K mask, V0 at0, V1 at1)
{
    using V = V0;
    if constexpr (!std::is_same_v<V0, V1>) {
        static_assert(sizeof(V0) == sizeof(V1));
        if constexpr (is_builtin_vector_v<V0> && !is_builtin_vector_v<V1>) {
            return blend(mask, at0, reinterpret_cast<V0>(at1.d));
        } else if constexpr (!is_builtin_vector_v<V0> && is_builtin_vector_v<V1>) {
            return blend(mask, reinterpret_cast<V1>(at0.d), at1);
        } else {
            assert_unreachable<K>();
        }
    } else if constexpr (sizeof(V) < 16) {
        static_assert(sizeof(K) == sizeof(V0) && sizeof(V0) == sizeof(V1));
        return (mask & at1) | (~mask & at0);
    } else if constexpr (!is_builtin_vector_v<V>) {
        return blend(mask, at0.d, at1.d);
    } else if constexpr (sizeof(K) < 16) {
        using T = typename builtin_traits<V>::value_type;
        if constexpr (sizeof(V) == 16 && have_avx512bw_vl && sizeof(T) <= 2) {
            if constexpr (sizeof(T) == 1) {
                return _mm_mask_mov_epi8(to_intrin(at0), mask, to_intrin(at1));
            } else if constexpr (sizeof(T) == 2) {
                return _mm_mask_mov_epi16(to_intrin(at0), mask, to_intrin(at1));
            }
        } else if constexpr (sizeof(V) == 16 && have_avx512vl && sizeof(T) > 2) {
            if constexpr (std::is_integral_v<T> && sizeof(T) == 4) {
                return _mm_mask_mov_epi32(to_intrin(at0), mask, to_intrin(at1));
            } else if constexpr (std::is_integral_v<T> && sizeof(T) == 8) {
                return _mm_mask_mov_epi64(to_intrin(at0), mask, to_intrin(at1));
            } else if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4) {
                return _mm_mask_mov_ps(at0, mask, at1);
            } else if constexpr (std::is_floating_point_v<T> && sizeof(T) == 8) {
                return _mm_mask_mov_pd(at0, mask, at1);
            }
        } else if constexpr (sizeof(V) == 16 && have_avx512f && sizeof(T) > 2) {
            if constexpr (std::is_integral_v<T> && sizeof(T) == 4) {
                return lo128(_mm512_mask_mov_epi32(auto_cast(at0), mask, auto_cast(at1)));
            } else if constexpr (std::is_integral_v<T> && sizeof(T) == 8) {
                return lo128(_mm512_mask_mov_epi64(auto_cast(at0), mask, auto_cast(at1)));
            } else if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4) {
                return lo128(_mm512_mask_mov_ps(auto_cast(at0), mask, auto_cast(at1)));
            } else if constexpr (std::is_floating_point_v<T> && sizeof(T) == 8) {
                return lo128(_mm512_mask_mov_pd(auto_cast(at0), mask, auto_cast(at1)));
            }
        } else if constexpr (sizeof(V) == 32 && have_avx512bw_vl && sizeof(T) <= 2) {
            if constexpr (sizeof(T) == 1) {
                return _mm256_mask_mov_epi8(to_intrin(at0), mask, to_intrin(at1));
            } else if constexpr (sizeof(T) == 2) {
                return _mm256_mask_mov_epi16(to_intrin(at0), mask, to_intrin(at1));
            }
        } else if constexpr (sizeof(V) == 32 && have_avx512vl && sizeof(T) > 2) {
            if constexpr (std::is_integral_v<T> && sizeof(T) == 4) {
                return _mm256_mask_mov_epi32(to_intrin(at0), mask, to_intrin(at1));
            } else if constexpr (std::is_integral_v<T> && sizeof(T) == 8) {
                return _mm256_mask_mov_epi64(to_intrin(at0), mask, to_intrin(at1));
            } else if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4) {
                return _mm256_mask_mov_ps(at0, mask, at1);
            } else if constexpr (std::is_floating_point_v<T> && sizeof(T) == 8) {
                return _mm256_mask_mov_pd(at0, mask, at1);
            }
        } else if constexpr (sizeof(V) == 32 && have_avx512f && sizeof(T) > 2) {
            if constexpr (std::is_integral_v<T> && sizeof(T) == 4) {
                return lo256(_mm512_mask_mov_epi32(auto_cast(at0), mask, auto_cast(at1)));
            } else if constexpr (std::is_integral_v<T> && sizeof(T) == 8) {
                return lo256(_mm512_mask_mov_epi64(auto_cast(at0), mask, auto_cast(at1)));
            } else if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4) {
                return lo256(_mm512_mask_mov_ps(auto_cast(at0), mask, auto_cast(at1)));
            } else if constexpr (std::is_floating_point_v<T> && sizeof(T) == 8) {
                return lo256(_mm512_mask_mov_pd(auto_cast(at0), mask, auto_cast(at1)));
            }
        } else if constexpr (sizeof(V) == 64 && have_avx512bw && sizeof(T) <= 2) {
            if constexpr (sizeof(T) == 1) {
                return _mm512_mask_mov_epi8(to_intrin(at0), mask, to_intrin(at1));
            } else if constexpr (sizeof(T) == 2) {
                return _mm512_mask_mov_epi16(to_intrin(at0), mask, to_intrin(at1));
            }
        } else if constexpr (sizeof(V) == 64 && have_avx512f && sizeof(T) > 2) {
            if constexpr (std::is_integral_v<T> && sizeof(T) == 4) {
                return _mm512_mask_mov_epi32(to_intrin(at0), mask, to_intrin(at1));
            } else if constexpr (std::is_integral_v<T> && sizeof(T) == 8) {
                return _mm512_mask_mov_epi64(to_intrin(at0), mask, to_intrin(at1));
            } else if constexpr (std::is_floating_point_v<T> && sizeof(T) == 4) {
                return _mm512_mask_mov_ps(at0, mask, at1);
            } else if constexpr (std::is_floating_point_v<T> && sizeof(T) == 8) {
                return _mm512_mask_mov_pd(at0, mask, at1);
            }
        } else {
            assert_unreachable<K>();
        }
    } else {
        const V k = auto_cast(mask);
        using T = typename builtin_traits<V>::value_type;
        if constexpr (sizeof(V) == 16 && have_sse4_1) {
            if constexpr (std::is_integral_v<T>) {
                return _mm_blendv_epi8(to_intrin(at0), to_intrin(at1), to_intrin(k));
            } else if constexpr (sizeof(T) == 4) {
                return _mm_blendv_ps(at0, at1, k);
            } else if constexpr (sizeof(T) == 8) {
                return _mm_blendv_pd(at0, at1, k);
            }
        } else if constexpr (sizeof(V) == 32) {
            if constexpr (std::is_integral_v<T>) {
                return _mm256_blendv_epi8(to_intrin(at0), to_intrin(at1), to_intrin(k));
            } else if constexpr (sizeof(T) == 4) {
                return _mm256_blendv_ps(at0, at1, k);
            } else if constexpr (sizeof(T) == 8) {
                return _mm256_blendv_pd(at0, at1, k);
            }
        } else {
            return or_(andnot_(k, at0), and_(k, at1));
        }
    }
}

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
    static_assert(sizeof(T) == 32);
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

Vc_INTRINSIC Vc_CONST auto firstbit(uint x)
{
#if defined Vc_ICC || defined Vc_GCC
    return _bit_scan_forward(x);
#elif defined Vc_CLANG || defined Vc_APPLECLANG
    return __builtin_ctz(x);
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

// cmpord{{{1
Vc_INTRINSIC builtin_type16_t<float> cmpord(builtin_type16_t<float> x,
                                            builtin_type16_t<float> y)
{
    return _mm_cmpord_ps(x, y);
}
Vc_INTRINSIC builtin_type16_t<double> cmpord(builtin_type16_t<double> x,
                                             builtin_type16_t<double> y)
{
    return _mm_cmpord_pd(x, y);
}

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC builtin_type32_t<float> cmpord(builtin_type32_t<float> x,
                                            builtin_type32_t<float> y)
{
    return _mm256_cmp_ps(x, y, _CMP_ORD_Q);
}
Vc_INTRINSIC builtin_type32_t<double> cmpord(builtin_type32_t<double> x,
                                             builtin_type32_t<double> y)
{
    return _mm256_cmp_pd(x, y, _CMP_ORD_Q);
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC __mmask16 cmpord(builtin_type64_t<float> x, builtin_type64_t<float> y)
{
    return _mm512_cmp_ps_mask(x, y, _CMP_ORD_Q);
}
Vc_INTRINSIC __mmask8 cmpord(builtin_type64_t<double> x, builtin_type64_t<double> y)
{
    return _mm512_cmp_pd_mask(x, y, _CMP_ORD_Q);
}
#endif  // Vc_HAVE_AVX512F

// cmpunord{{{1
Vc_INTRINSIC builtin_type16_t<float> cmpunord(builtin_type16_t<float> x,
                                              builtin_type16_t<float> y)
{
    return _mm_cmpunord_ps(x, y);
}
Vc_INTRINSIC builtin_type16_t<double> cmpunord(builtin_type16_t<double> x,
                                               builtin_type16_t<double> y)
{
    return _mm_cmpunord_pd(x, y);
}

#ifdef Vc_HAVE_AVX
Vc_INTRINSIC builtin_type32_t<float> cmpunord(builtin_type32_t<float> x,
                                              builtin_type32_t<float> y)
{
    return _mm256_cmp_ps(x, y, _CMP_UNORD_Q);
}
Vc_INTRINSIC builtin_type32_t<double> cmpunord(builtin_type32_t<double> x,
                                               builtin_type32_t<double> y)
{
    return _mm256_cmp_pd(x, y, _CMP_UNORD_Q);
}
#endif  // Vc_HAVE_AVX

#ifdef Vc_HAVE_AVX512F
Vc_INTRINSIC __mmask16 cmpunord(builtin_type64_t<float> x, builtin_type64_t<float> y)
{
    return _mm512_cmp_ps_mask(x, y, _CMP_UNORD_Q);
}
Vc_INTRINSIC __mmask8 cmpunord(builtin_type64_t<double> x, builtin_type64_t<double> y)
{
    return _mm512_cmp_pd_mask(x, y, _CMP_UNORD_Q);
}
#endif  // Vc_HAVE_AVX512F

// }}}
// loads{{{
/**
 * \internal
 * Abstraction for simplifying load operations in the SSE/AVX/AVX512 implementations
 *
 * \note The number in the suffix signifies the number of Bytes
 */
template <class F> Vc_INTRINSIC auto load2(const float *mem, F f)
{
    return detail::builtin_load<float, 4, 2>(mem, f);
}
template <class F> Vc_INTRINSIC auto load2(const double *mem, F f)
{
    return detail::builtin_load<double, 2, 2>(mem, f);
}
template <class F> Vc_INTRINSIC auto load2(const void *mem, F f)
{
    return detail::builtin_load<long long, 2, 2>(mem, f);
}

template <class F> Vc_INTRINSIC auto load4(const float *mem, F f)
{
    return detail::builtin_load<float, 4, 4>(mem, f);
}
template <class F> Vc_INTRINSIC auto load4(const double *mem, F f)
{
    return detail::builtin_load<double, 2, 4>(mem, f);
}
template <class F> Vc_INTRINSIC auto load4(const void *mem, F f)
{
    return detail::builtin_load<long long, 2, 4>(mem, f);
}

template <class F> Vc_INTRINSIC auto load8(const float *mem, F f)
{
    return detail::builtin_load<float, 4, 8>(mem, f);
}
template <class F> Vc_INTRINSIC auto load8(const double *mem, F f)
{
    return detail::builtin_load<double, 2, 8>(mem, f);
}
template <class F> Vc_INTRINSIC auto load8(const void *mem, F f)
{
    return detail::builtin_load<long long, 2, 8>(mem, f);
}

template <class F> Vc_INTRINSIC auto load16(const float *mem, F f)
{
    return detail::builtin_load<float, 4, 16>(mem, f);
}
template <class F> Vc_INTRINSIC auto load16(const double *mem, F f)
{
    return detail::builtin_load<double, 2, 16>(mem, f);
}
template <class F> Vc_INTRINSIC auto load16(const void *mem, F f)
{
    return detail::builtin_load<long long, 2, 16>(mem, f);
}

template <class F> Vc_INTRINSIC auto load32(const float *mem, F f)
{
    return detail::builtin_load<float, 8, 32>(mem, f);
}
template <class F> Vc_INTRINSIC auto load32(const double *mem, F f)
{
    return detail::builtin_load<double, 4, 32>(mem, f);
}
template <class F> Vc_INTRINSIC auto load32(const void *mem, F f)
{
    return detail::builtin_load<long long, 4, 32>(mem, f);
}

template <class F> Vc_INTRINSIC auto load64(const float *mem, F f)
{
    return detail::builtin_load<float, 16, 64>(mem, f);
}
template <class F> Vc_INTRINSIC auto load64(const double *mem, F f)
{
    return detail::builtin_load<double, 8, 64>(mem, f);
}
template <class F> Vc_INTRINSIC auto load64(const void *mem, F f)
{
    return detail::builtin_load<long long, 8, 64>(mem, f);
}

// }}}
// stores{{{
template <class F> Vc_INTRINSIC auto store2(builtin_type16_t<float> v, float *mem, F f)
{
    return detail::builtin_store<2>(v, mem, f);
}
template <class F> Vc_INTRINSIC auto store2(builtin_type16_t<double> v, double *mem, F f)
{
    return detail::builtin_store<2>(v, mem, f);
}
template <class T, class F>
Vc_INTRINSIC auto store2(builtin_type16_t<T> v, void *mem, F f)
{
    return detail::builtin_store<2>(v, mem, f);
}

template <class F> Vc_INTRINSIC auto store4(builtin_type16_t<float> v, float *mem, F f)
{
    return detail::builtin_store<4>(v, mem, f);
}
template <class F> Vc_INTRINSIC auto store4(builtin_type16_t<double> v, double *mem, F f)
{
    return detail::builtin_store<4>(v, mem, f);
}
template <class T, class F>
Vc_INTRINSIC auto store4(builtin_type16_t<T> v, void *mem, F f)
{
    return detail::builtin_store<4>(v, mem, f);
}

template <class F> Vc_INTRINSIC auto store8(builtin_type16_t<float> v, float *mem, F f)
{
    return detail::builtin_store<8>(v, mem, f);
}
template <class F> Vc_INTRINSIC auto store8(builtin_type16_t<double> v, double *mem, F f)
{
    return detail::builtin_store<8>(v, mem, f);
}
template <class T, class F>
Vc_INTRINSIC auto store8(builtin_type16_t<T> v, void *mem, F f)
{
    return detail::builtin_store<8>(v, mem, f);
}

template <class F> Vc_INTRINSIC auto store16(builtin_type16_t<float> v, float *mem, F f)
{
    return detail::builtin_store<16>(v, mem, f);
}
template <class F> Vc_INTRINSIC auto store16(builtin_type16_t<double> v, double *mem, F f)
{
    return detail::builtin_store<16>(v, mem, f);
}
template <class T, class F>
Vc_INTRINSIC auto store16(builtin_type16_t<T> v, void *mem, F f)
{
    return detail::builtin_store<16>(v, mem, f);
}

template <class F> Vc_INTRINSIC auto store32(builtin_type32_t<float> v, float *mem, F f)
{
    return detail::builtin_store<32>(v, mem, f);
}
template <class F> Vc_INTRINSIC auto store32(builtin_type32_t<double> v, double *mem, F f)
{
    return detail::builtin_store<32>(v, mem, f);
}
template <class T, class F>
Vc_INTRINSIC auto store32(builtin_type32_t<T> v, void *mem, F f)
{
    return detail::builtin_store<32>(v, mem, f);
}

template <class F> Vc_INTRINSIC auto store64(builtin_type64_t<float> v, float *mem, F f)
{
    return detail::builtin_store<64>(v, mem, f);
}
template <class F> Vc_INTRINSIC auto store64(builtin_type64_t<double> v, double *mem, F f)
{
    return detail::builtin_store<64>(v, mem, f);
}
template <class T, class F>
Vc_INTRINSIC auto store64(builtin_type64_t<T> v, void *mem, F f)
{
    return detail::builtin_store<64>(v, mem, f);
}

// }}}
// integer sign-extension {{{
constexpr Vc_INTRINSIC builtin_type_t<short,  8> sign_extend16(builtin_type_t<char, 16> x) { return __builtin_ia32_pmovsxbw128(x); }
constexpr Vc_INTRINSIC builtin_type_t<  int,  4> sign_extend32(builtin_type_t<char, 16> x) { return __builtin_ia32_pmovsxbd128(x); }
constexpr Vc_INTRINSIC builtin_type_t<llong,  2> sign_extend64(builtin_type_t<char, 16> x) { return __builtin_ia32_pmovsxbq128(x); }
constexpr Vc_INTRINSIC builtin_type_t<  int,  4> sign_extend32(builtin_type_t<short, 8> x) { return __builtin_ia32_pmovsxwd128(x); }
constexpr Vc_INTRINSIC builtin_type_t<llong,  2> sign_extend64(builtin_type_t<short, 8> x) { return __builtin_ia32_pmovsxwq128(x); }
constexpr Vc_INTRINSIC builtin_type_t<llong,  2> sign_extend64(builtin_type_t<  int, 4> x) { return __builtin_ia32_pmovsxdq128(x); }

// }}}
// integer zero-extension {{{
constexpr Vc_INTRINSIC builtin_type_t<short,  8> zero_extend16(builtin_type_t<char, 16> x) { return __builtin_ia32_pmovzxbw128(x); }
constexpr Vc_INTRINSIC builtin_type_t<  int,  4> zero_extend32(builtin_type_t<char, 16> x) { return __builtin_ia32_pmovzxbd128(x); }
constexpr Vc_INTRINSIC builtin_type_t<llong,  2> zero_extend64(builtin_type_t<char, 16> x) { return __builtin_ia32_pmovzxbq128(x); }
constexpr Vc_INTRINSIC builtin_type_t<  int,  4> zero_extend32(builtin_type_t<short, 8> x) { return __builtin_ia32_pmovzxwd128(x); }
constexpr Vc_INTRINSIC builtin_type_t<llong,  2> zero_extend64(builtin_type_t<short, 8> x) { return __builtin_ia32_pmovzxwq128(x); }
constexpr Vc_INTRINSIC builtin_type_t<llong,  2> zero_extend64(builtin_type_t<  int, 4> x) { return __builtin_ia32_pmovzxdq128(x); }

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
        _mm_maskmoveu_si128(to_m128i(lo128(v)), to_m128i(lo128(k)),
                            reinterpret_cast<char *>(mem));
        _mm_maskmoveu_si128(to_m128i(hi128(v)), to_m128i(hi128(k)),
                            reinterpret_cast<char *>(mem) + 16);
    }
}

template <class T, class F>
Vc_INTRINSIC void maskstore(storage32_t<T> v, T *mem, F,
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
        if constexpr (is_aligned_v<F, 32> && std::is_integral_v<T>) {
            _mm256_mask_store_epi64(mem, k, v);
        } else if constexpr (is_aligned_v<F, 32> && std::is_floating_point_v<T>) {
            _mm256_mask_store_pd(mem, k, v);
        } else if constexpr (std::is_integral_v<T>) {
            _mm256_mask_storeu_epi64(mem, k, v);
        } else {
            _mm256_mask_storeu_pd(mem, k, v);
        }
    } else if constexpr (have_avx512f && (sizeof(T) >= 4 || have_avx512bw)) {
        // use a 512-bit maskstore, using zero-extension of the bitmask
        maskstore(
            detail::storage64_t<T>(
                detail::intrin_cast<detail::intrinsic_type64_t<T>>(v.d)),
            mem,
            // careful, vector_aligned has a stricter meaning in the 512-bit maskstore:
            std::conditional_t<std::is_same_v<F, vector_aligned_tag>, overaligned_tag<32>,
                               F>(),
            detail::Storage<bool, 64 / sizeof(T)>(k.d));
    } else {
        maskstore(
            v, mem, F(),
            detail::storage32_t<T>(builtin_cast<T>(convert_mask<sizeof(T), 32>(k))));
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
Vc_INTRINSIC void maskstore(storage16_t<T> v, T *mem, F,
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
    } else if constexpr (have_avx512f && (sizeof(T) >= 4 || have_avx512bw)) {
        // use a 512-bit maskstore, using zero-extension of the bitmask
        maskstore(
            detail::storage64_t<T>(
                detail::intrin_cast<detail::intrinsic_type64_t<T>>(v.d)),
            mem,
            // careful, vector_aligned has a stricter meaning in the 512-bit maskstore:
            std::conditional_t<std::is_same_v<F, vector_aligned_tag>, overaligned_tag<16>,
                               F>(),
            detail::Storage<bool, 64 / sizeof(T)>(k.d));
    } else {
        maskstore(v, mem, F(),
                  storage16_t<T>(builtin_cast<T>(convert_mask<sizeof(T), 16>(k))));
    }
}

// }}}
}  // namespace x86
using namespace x86;

// to_bitset(__mmaskXX){{{
template <class T, class = std::enable_if_t<detail::is_bitmask_v<T> && have_avx512f>>
constexpr Vc_INTRINSIC std::bitset<8 * sizeof(T)> to_bitset(T x)
{
    if constexpr (std::is_integral_v<T>) {
        return x;
    } else {
        return x.d;
    }
}

// }}}
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END
#endif  // Vc_HAVE_SSE

#endif  // VC_SIMD_X86_H_
