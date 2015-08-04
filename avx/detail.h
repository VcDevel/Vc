/*  This file is part of the Vc library. {{{
Copyright © 2015 Matthias Kretz <kretz@kde.org>
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

#ifndef VC_AVX_DETAIL_H_
#define VC_AVX_DETAIL_H_

#include "../sse/detail.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Detail
{
// (converting) load functions {{{1
template <typename Flags>
Vc_INTRINSIC Vc_PURE __m256 load(const float *x, __m256,
                                 typename Flags::EnableIfAligned = nullptr)
{
    return _mm256_load_ps(x);
}
template <typename Flags>
Vc_INTRINSIC Vc_PURE __m256 load(const float *x, __m256,
                                 typename Flags::EnableIfUnaligned = nullptr)
{
    return _mm256_loadu_ps(x);
}
template <typename Flags>
Vc_INTRINSIC Vc_PURE __m256 load(const float *x, __m256,
                                 typename Flags::EnableIfStreaming = nullptr)
{
    return AvxIntrinsics::stream_load<__m256>(x);
}

template <typename Flags>
Vc_INTRINSIC Vc_PURE __m256d load(const double *x, __m256d,
                                 typename Flags::EnableIfAligned = nullptr)
{
    return _mm256_load_pd(x);
}
template <typename Flags>
Vc_INTRINSIC Vc_PURE __m256d load(const double *x, __m256d,
                                 typename Flags::EnableIfUnaligned = nullptr)
{
    return _mm256_loadu_pd(x);
}
template <typename Flags>
Vc_INTRINSIC Vc_PURE __m256d load(const double *x, __m256d,
                                 typename Flags::EnableIfStreaming = nullptr)
{
    return AvxIntrinsics::stream_load<__m256d>(x);
}

template <typename Flags, typename T, typename = enable_if<std::is_integral<T>::value>>
Vc_INTRINSIC Vc_PURE __m256i load(const T *x, __m256i,
                                 typename Flags::EnableIfAligned = nullptr)
{
    return _mm256_load_si256(reinterpret_cast<const __m256i *>(x));
}
template <typename Flags, typename T, typename = enable_if<std::is_integral<T>::value>>
Vc_INTRINSIC Vc_PURE __m256i load(const T *x, __m256i,
                                 typename Flags::EnableIfUnaligned = nullptr)
{
    return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(x));
}
template <typename Flags, typename T, typename = enable_if<std::is_integral<T>::value>>
Vc_INTRINSIC Vc_PURE __m256i load(const T *x, __m256i,
                                 typename Flags::EnableIfStreaming = nullptr)
{
    return AvxIntrinsics::stream_load<__m256i>(x);
}

// no conversion load from any T {{{2
template <typename V, typename T, typename Flags>
Vc_INTRINSIC V
    load(const T *mem, Flags, LoadTag<V, T>, enable_if<sizeof(V) == 32> = nullarg)
{
    return load<Flags>(mem, V());
}

// short {{{2
template <typename Flags>
Vc_INTRINSIC __m256i load(const ushort *mem, Flags, LoadTag<__m256i, short>)
{
    return load<Flags>(mem, __m256i());
}
template <typename Flags>
Vc_INTRINSIC __m256i load(const uchar *mem, Flags f, LoadTag<__m256i, short>)
{
    return AVX::cvtepu8_epi16(load(mem, f, LoadTag<__m128i, uchar>()));
}
template <typename Flags>
Vc_INTRINSIC __m256i load(const schar *mem, Flags f, LoadTag<__m256i, short>)
{
    return AVX::cvtepi8_epi16(load(mem, f, LoadTag<__m128i, schar>()));
}

// ushort {{{2
template <typename Flags>
Vc_INTRINSIC __m256i load(const uchar *mem, Flags f, LoadTag<__m256i, ushort>)
{
    return AVX::cvtepu8_epi16(load(mem, f, LoadTag<__m128i, uchar>()));
}

// int {{{2
template <typename Flags>
Vc_INTRINSIC __m256i load(const uint *mem, Flags, LoadTag<__m256i, int>)
{
    return load<Flags>(mem, __m256i());
}
template <typename Flags>
Vc_INTRINSIC __m256i load(const ushort *mem, Flags f, LoadTag<__m256i, int>)
{
    return AVX::cvtepu16_epi32(load(mem, f, LoadTag<__m128i, ushort>()));
}
template <typename Flags>
Vc_INTRINSIC __m256i load(const short *mem, Flags f, LoadTag<__m256i, int>)
{
    return AVX::cvtepi16_epi32(load(mem, f, LoadTag<__m128i, short>()));
}
template <typename Flags>
Vc_INTRINSIC __m256i load(const uchar *mem, Flags, LoadTag<__m256i, int>)
{
    return AVX::cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem)));
}
template <typename Flags>
Vc_INTRINSIC __m256i load(const schar *mem, Flags, LoadTag<__m256i, int>)
{
    return AVX::cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem)));
}

// uint {{{2
template <typename Flags>
Vc_INTRINSIC __m256i load(const ushort *mem, Flags f, LoadTag<__m256i, uint>)
{
    return AVX::cvtepu16_epi32(load(mem, f, LoadTag<__m128i, ushort>()));
}
template <typename Flags>
Vc_INTRINSIC __m256i load(const uchar *mem, Flags, LoadTag<__m256i, uint>)
{
    return AVX::cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem)));
}

// double {{{2
template <typename Flags>
Vc_INTRINSIC __m256d load(const float *mem, Flags f, LoadTag<__m256d, double>)
{
    return AVX::convert<float, double>(load<__m128, float>(mem, f));
}
template <typename Flags>
Vc_INTRINSIC __m256d load(const uint *mem, Flags f, LoadTag<__m256d, double>)
{
    return AVX::convert<uint, double>(load<__m128i, uint>(mem, f));
}
template <typename Flags>
Vc_INTRINSIC __m256d load(const int *mem, Flags f, LoadTag<__m256d, double>)
{
    return AVX::convert<int, double>(load<__m128i, int>(mem, f));
}
template <typename Flags>
Vc_INTRINSIC __m256d load(const ushort *mem, Flags f, LoadTag<__m256d, double>)
{
    return AVX::convert<int, double>(load<__m128i, int>(mem, f));
}
template <typename Flags>
Vc_INTRINSIC __m256d load(const short *mem, Flags f, LoadTag<__m256d, double>)
{
    return AVX::convert<int, double>(load<__m128i, int>(mem, f));
}
template <typename Flags>
Vc_INTRINSIC __m256d load(const uchar *mem, Flags f, LoadTag<__m256d, double>)
{
    return AVX::convert<int, double>(load<__m128i, int>(mem, f));
}
template <typename Flags>
Vc_INTRINSIC __m256d load(const schar *mem, Flags f, LoadTag<__m256d, double>)
{
    return AVX::convert<int, double>(load<__m128i, int>(mem, f));
}

// float {{{2
template <typename Flags>
Vc_INTRINSIC __m256 load(const double *mem, Flags, LoadTag<__m256, float>)
{
    return AVX::concat(_mm256_cvtpd_ps(load<Flags>(&mem[0], __m256d())),
                       _mm256_cvtpd_ps(load<Flags>(&mem[4], __m256d())));
}
template <typename Flags>
Vc_INTRINSIC __m256 load(const uint *mem, Flags, LoadTag<__m256, float>)
{
    const auto v = load<Flags>(mem, __m256i());
    return _mm256_blendv_ps(
        _mm256_cvtepi32_ps(v),
        _mm256_add_ps(_mm256_cvtepi32_ps(AVX::sub_epi32(v, AVX::set2power31_epu32())),
                      AVX::set2power31_ps()),
        _mm256_castsi256_ps(AVX::cmplt_epi32(v, _mm256_setzero_si256())));
}
template <typename T, typename Flags,
          typename = enable_if<!std::is_same<T, float>::value>>
Vc_INTRINSIC __m256 load(const T *mem, Flags f, LoadTag<__m256, float>)
{
    return _mm256_cvtepi32_ps(load<__m256i, int>(mem, f));
}
template <typename Flags>
Vc_INTRINSIC __m256 load(const ushort *mem, Flags f, LoadTag<__m256, float>)
{
    return AVX::convert<ushort, float>(load<__m128i, ushort>(mem, f));
}
template <typename Flags>
Vc_INTRINSIC __m256 load(const short *mem, Flags f, LoadTag<__m256, float>)
{
    return AVX::convert<short, float>(load<__m128i, short>(mem, f));
}
/*
template<typename Flags> struct LoadHelper<float, unsigned char, Flags> {
    static __m256 load(const unsigned char *mem, Flags)
    {
        return _mm256_cvtepi32_ps(
            cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem))));
    }
};
template<typename Flags> struct LoadHelper<float, signed char, Flags> {
    static __m256 load(const signed char *mem, Flags)
    {
        return _mm256_cvtepi32_ps(
            cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem))));
    }
};
*/

// shifted{{{1
template <int amount, typename T>
Vc_INTRINSIC Vc_PURE enable_if<(sizeof(T) == 32 && amount >= 16), T> shifted(T k)
{
    return AVX::zeroExtend(_mm_srli_si128(AVX::hi128(k), amount - 16));
}
template <int amount, typename T>
Vc_INTRINSIC Vc_PURE enable_if<(sizeof(T) == 32 && amount > 0 && amount < 16), T> shifted(
    T k)
{
    return AVX::alignr<amount>(Mem::permute128<X1, Const0>(k), Mem::permute128<X0, X1>(k));
}
template <int amount, typename T>
Vc_INTRINSIC Vc_PURE enable_if<(sizeof(T) == 32 && amount <= -16), T> shifted(T k)
{
    return Mem::permute128<Const0, X0>(AVX::avx_cast<__m256i>(_mm_slli_si128(AVX::lo128(k), -16 - amount)));
}
template <int amount, typename T>
Vc_INTRINSIC Vc_PURE enable_if<(sizeof(T) == 32 && amount > -16 && amount < 0), T>
    shifted(T k)
{
    return AVX::alignr<16 + amount>(k, Mem::permute128<Const0, X0>(k));
}

// mask_cast{{{1
template<size_t From, size_t To, typename R> Vc_INTRINSIC Vc_CONST R mask_cast(__m256i k)
{
    static_assert(From == To, "Incorrect mask cast.");
    static_assert(std::is_same<R, __m256>::value, "Incorrect mask cast.");
    return AVX::avx_cast<__m256>(k);
}

// 4 -> 4
template <> Vc_INTRINSIC Vc_CONST __m128 mask_cast<4, 4, __m128>(__m256i k)
{
    return AVX::avx_cast<__m128>(_mm_packs_epi32(AVX::lo128(k), AVX::hi128(k)));
}

template <> Vc_INTRINSIC Vc_CONST __m256 mask_cast<4, 4, __m256>(__m128i k)
{
    const auto kk = _mm_castsi128_ps(k);
    return AVX::concat(_mm_unpacklo_ps(kk, kk), _mm_unpackhi_ps(kk, kk));
}

// 4 -> 8
template<> Vc_INTRINSIC Vc_CONST __m256 mask_cast<4, 8, __m256>(__m256i k)
{
    // aabb ccdd -> abcd 0000
    return AVX::avx_cast<__m256>(AVX::concat(_mm_packs_epi32(AVX::lo128(k), AVX::hi128(k)),
                                 _mm_setzero_si128()));
}

template<> Vc_INTRINSIC Vc_CONST __m128 mask_cast<4, 8, __m128>(__m256i k)
{
    // aaaa bbbb cccc dddd -> abcd 0000
    return AVX::avx_cast<__m128>(_mm_packs_epi16(_mm_packs_epi32(AVX::lo128(k), AVX::hi128(k)), _mm_setzero_si128()));
}

template <> Vc_INTRINSIC Vc_CONST __m256 mask_cast<4, 8, __m256>(__m128i k)
{
    return AVX::zeroExtend(AVX::avx_cast<__m128>(k));
}

// 8 -> 4
template<> Vc_INTRINSIC Vc_CONST __m256 mask_cast<8, 4, __m256>(__m256i k)
{
    // aabb ccdd eeff gghh -> aaaa bbbb cccc dddd
    const auto lo = AVX::lo128(AVX::avx_cast<__m256>(k));
    return AVX::concat(_mm_unpacklo_ps(lo, lo),
                  _mm_unpackhi_ps(lo, lo));
}

template<> Vc_INTRINSIC Vc_CONST __m128 mask_cast<8, 4, __m128>(__m256i k)
{
    return AVX::avx_cast<__m128>(AVX::lo128(k));
}

template<> Vc_INTRINSIC Vc_CONST __m256 mask_cast<8, 4, __m256>(__m128i k)
{
    // abcd efgh -> aaaa bbbb cccc dddd
    const auto tmp = _mm_unpacklo_epi16(k, k); // aa bb cc dd
    return AVX::avx_cast<__m256>(AVX::concat(_mm_unpacklo_epi32(tmp, tmp), // aaaa bbbb
                                 _mm_unpackhi_epi32(tmp, tmp))); // cccc dddd
}

// 8 -> 8
template<> Vc_INTRINSIC Vc_CONST __m128 mask_cast<8, 8, __m128>(__m256i k)
{
    // aabb ccdd eeff gghh -> abcd efgh
    return AVX::avx_cast<__m128>(_mm_packs_epi16(AVX::lo128(k), AVX::hi128(k)));
}

template<> Vc_INTRINSIC Vc_CONST __m256 mask_cast<8, 8, __m256>(__m128i k)
{
    return AVX::avx_cast<__m256>(AVX::concat(_mm_unpacklo_epi16(k, k),
                                 _mm_unpackhi_epi16(k, k)));
}

// allone{{{1
template<> Vc_INTRINSIC Vc_CONST __m256  allone<__m256 >() { return AVX::setallone_ps(); }
template<> Vc_INTRINSIC Vc_CONST __m256i allone<__m256i>() { return AVX::setallone_si256(); }
template<> Vc_INTRINSIC Vc_CONST __m256d allone<__m256d>() { return AVX::setallone_pd(); }

// zero{{{1
template<> Vc_INTRINSIC Vc_CONST __m256  zero<__m256 >() { return _mm256_setzero_ps(); }
template<> Vc_INTRINSIC Vc_CONST __m256i zero<__m256i>() { return _mm256_setzero_si256(); }
template<> Vc_INTRINSIC Vc_CONST __m256d zero<__m256d>() { return _mm256_setzero_pd(); }

// negate{{{1
Vc_ALWAYS_INLINE Vc_CONST __m256 negate(__m256 v, std::integral_constant<std::size_t, 4>)
{
    return _mm256_xor_ps(v, AVX::setsignmask_ps());
}
Vc_ALWAYS_INLINE Vc_CONST __m256d negate(__m256d v, std::integral_constant<std::size_t, 8>)
{
    return _mm256_xor_pd(v, AVX::setsignmask_pd());
}
Vc_ALWAYS_INLINE Vc_CONST __m256i negate(__m256i v, std::integral_constant<std::size_t, 4>)
{
    return AVX::sign_epi32(v, Detail::allone<__m256i>());
}
Vc_ALWAYS_INLINE Vc_CONST __m256i negate(__m256i v, std::integral_constant<std::size_t, 2>)
{
    return AVX::sign_epi16(v, Detail::allone<__m256i>());
}

// xor_{{{1
Vc_INTRINSIC __m256 xor_(__m256 a, __m256 b) { return _mm256_xor_ps(a, b); }
Vc_INTRINSIC __m256d xor_(__m256d a, __m256d b) { return _mm256_xor_pd(a, b); }
Vc_INTRINSIC __m256i xor_(__m256i a, __m256i b)
{
#ifdef VC_IMPL_AVX2
    return _mm256_xor_si256(a, b);
#else
    return _mm256_castps_si256(
        _mm256_xor_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
#endif
}

// or_{{{1
Vc_INTRINSIC __m256 or_(__m256 a, __m256 b) { return _mm256_or_ps(a, b); }
Vc_INTRINSIC __m256d or_(__m256d a, __m256d b) { return _mm256_or_pd(a, b); }
Vc_INTRINSIC __m256i or_(__m256i a, __m256i b)
{
#ifdef VC_IMPL_AVX2
    return _mm256_or_si256(a, b);
#else
    return _mm256_castps_si256(
        _mm256_or_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
#endif
}

// and_{{{1
Vc_INTRINSIC __m256 and_(__m256 a, __m256 b) { return _mm256_and_ps(a, b); }
Vc_INTRINSIC __m256d and_(__m256d a, __m256d b) { return _mm256_and_pd(a, b); }
Vc_INTRINSIC __m256i and_(__m256i a, __m256i b) {
#ifdef VC_IMPL_AVX2
    return _mm256_and_si256(a, b);
#else
    return _mm256_castps_si256(
        _mm256_and_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
#endif
}

// andnot_{{{1
Vc_INTRINSIC __m256 andnot_(__m256 a, __m256 b) { return _mm256_andnot_ps(a, b); }
Vc_INTRINSIC __m256d andnot_(__m256d a, __m256d b) { return _mm256_andnot_pd(a, b); }
Vc_INTRINSIC __m256i andnot_(__m256i a, __m256i b)
{
#ifdef VC_IMPL_AVX2
    return _mm256_andnot_si256(a, b);
#else
    return _mm256_castps_si256(
        _mm256_andnot_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
#endif
}

// add{{{1
Vc_INTRINSIC __m256  add(__m256  a, __m256  b,  float) { return _mm256_add_ps(a, b); }
Vc_INTRINSIC __m256d add(__m256d a, __m256d b, double) { return _mm256_add_pd(a, b); }
Vc_INTRINSIC __m256i add(__m256i a, __m256i b,    int) { return _mm256_add_epi32(a, b); }
Vc_INTRINSIC __m256i add(__m256i a, __m256i b,   uint) { return _mm256_add_epi32(a, b); }
Vc_INTRINSIC __m256i add(__m256i a, __m256i b,  short) { return _mm256_add_epi16(a, b); }
Vc_INTRINSIC __m256i add(__m256i a, __m256i b, ushort) { return _mm256_add_epi16(a, b); }

// sub{{{1
Vc_INTRINSIC __m256  sub(__m256  a, __m256  b,  float) { return _mm256_sub_ps(a, b); }
Vc_INTRINSIC __m256d sub(__m256d a, __m256d b, double) { return _mm256_sub_pd(a, b); }
Vc_INTRINSIC __m256i sub(__m256i a, __m256i b,    int) { return _mm256_sub_epi32(a, b); }
Vc_INTRINSIC __m256i sub(__m256i a, __m256i b,   uint) { return _mm256_sub_epi32(a, b); }
Vc_INTRINSIC __m256i sub(__m256i a, __m256i b,  short) { return _mm256_sub_epi16(a, b); }
Vc_INTRINSIC __m256i sub(__m256i a, __m256i b, ushort) { return _mm256_sub_epi16(a, b); }

// mul{{{1
Vc_INTRINSIC __m256  mul(__m256  a, __m256  b,  float) { return _mm256_mul_ps(a, b); }
Vc_INTRINSIC __m256d mul(__m256d a, __m256d b, double) { return _mm256_mul_pd(a, b); }
Vc_INTRINSIC __m256i mul(__m256i a, __m256i b,    int) { return _mm256_mullo_epi32(a, b); }
Vc_INTRINSIC __m256i mul(__m256i a, __m256i b,   uint) { return _mm256_mullo_epi32(a, b); }
Vc_INTRINSIC __m256i mul(__m256i a, __m256i b,  short) { return _mm256_mullo_epi16(a, b); }
Vc_INTRINSIC __m256i mul(__m256i a, __m256i b, ushort) { return _mm256_mullo_epi16(a, b); }

// zeroExtendIfNeeded{{{1
Vc_INTRINSIC __m256  zeroExtendIfNeeded(__m256  x) { return x; }
Vc_INTRINSIC __m256d zeroExtendIfNeeded(__m256d x) { return x; }
Vc_INTRINSIC __m256i zeroExtendIfNeeded(__m256i x) { return x; }
Vc_INTRINSIC __m256  zeroExtendIfNeeded(__m128  x) { return AVX::zeroExtend(x); }
Vc_INTRINSIC __m256d zeroExtendIfNeeded(__m128d x) { return AVX::zeroExtend(x); }
Vc_INTRINSIC __m256i zeroExtendIfNeeded(__m128i x) { return AVX::zeroExtend(x); }

// broadcast{{{1
Vc_INTRINSIC __m256  avx_broadcast( float x) { return _mm256_set1_ps(x); }
Vc_INTRINSIC __m256d avx_broadcast(double x) { return _mm256_set1_pd(x); }
Vc_INTRINSIC __m256i avx_broadcast(   int x) { return _mm256_set1_epi32(x); }
Vc_INTRINSIC __m256i avx_broadcast(  uint x) { return _mm256_set1_epi32(x); }
Vc_INTRINSIC __m256i avx_broadcast( short x) { return _mm256_set1_epi16(x); }
Vc_INTRINSIC __m256i avx_broadcast(ushort x) { return _mm256_set1_epi16(x); }
Vc_INTRINSIC __m256i avx_broadcast(  char x) { return _mm256_set1_epi8(x); }
Vc_INTRINSIC __m256i avx_broadcast( schar x) { return _mm256_set1_epi8(x); }
Vc_INTRINSIC __m256i avx_broadcast( uchar x) { return _mm256_set1_epi8(x); }

// sorted{{{1
template <Vc::Implementation Impl, typename T,
          typename = enable_if<(Impl >= AVXImpl && Impl <= AVX2Impl)>>
Vc_CONST_L AVX2::Vector<T> sorted(VC_ALIGNED_PARAMETER(AVX2::Vector<T>) x) Vc_CONST_R;
template <typename T>
Vc_INTRINSIC Vc_CONST AVX2::Vector<T> sorted(VC_ALIGNED_PARAMETER(AVX2::Vector<T>) x)
{
    return sorted<CurrentImplementation::current()>(x);
}
//}}}1

template <size_t Size>
Vc_INTRINSIC_L Vc_CONST_L int mask_to_int(__m256i x) Vc_INTRINSIC_R Vc_CONST_R;

Vc_INTRINSIC Vc_CONST int testc(__m128 a, __m128 b) { return _mm_testc_si128(_mm_castps_si128(a), _mm_castps_si128(b)); }
Vc_INTRINSIC Vc_CONST int testc(__m256 a, __m256 b) { return _mm256_testc_ps(a, b); }
Vc_INTRINSIC Vc_CONST int testz(__m128 a, __m128 b) { return _mm_testz_si128(_mm_castps_si128(a), _mm_castps_si128(b)); }
Vc_INTRINSIC Vc_CONST int testz(__m256 a, __m256 b) { return _mm256_testz_ps(a, b); }
Vc_INTRINSIC Vc_CONST int testnzc(__m128 a, __m128 b) { return _mm_testnzc_si128(_mm_castps_si128(a), _mm_castps_si128(b)); }
Vc_INTRINSIC Vc_CONST int testnzc(__m256 a, __m256 b) { return _mm256_testnzc_ps(a, b); }
Vc_INTRINSIC Vc_CONST int movemask(__m256i a) { return AVX::movemask_epi8(a); }
Vc_INTRINSIC Vc_CONST int movemask(__m128i a) { return _mm_movemask_epi8(a); }
Vc_INTRINSIC Vc_CONST int movemask(__m256d a) { return _mm256_movemask_pd(a); }
Vc_INTRINSIC Vc_CONST int movemask(__m128d a) { return _mm_movemask_pd(a); }
Vc_INTRINSIC Vc_CONST int movemask(__m256  a) { return _mm256_movemask_ps(a); }
Vc_INTRINSIC Vc_CONST int movemask(__m128  a) { return _mm_movemask_ps(a); }

} // namespace Detail

}  // namespace Vc

#include "undomacros.h"

#endif  // VC_AVX_DETAIL_H_

// vim: foldmethod=marker
