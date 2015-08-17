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

// 4 -> 16
template<> Vc_INTRINSIC Vc_CONST __m256 mask_cast<4, 16, __m256>(__m256i k)
{
    // aaaa bbbb cccc dddd -> abcd 0000 0000 0000
    return AVX::zeroExtend(mask_cast<4, 8, __m128>(k));
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

// 8 -> 16
template<> Vc_INTRINSIC Vc_CONST __m256 mask_cast<8, 16, __m256>(__m256i k)
{
    // aabb ccdd eeff gghh -> abcd efgh 0000 0000
    return AVX::zeroExtend(mask_cast<8, 8, __m128>(k));
}

// 16 -> 8
#ifdef VC_IMPL_AVX2
template<> Vc_INTRINSIC Vc_CONST __m256 mask_cast<16, 8, __m256>(__m256i k)
{
    // abcd efgh ijkl mnop -> aabb ccdd eeff gghh
    const auto flipped = Mem::permute4x64<X0, X2, X1, X3>(k);
    return _mm256_castsi256_ps(_mm256_unpacklo_epi16(flipped, flipped));
}
#endif

// 16 -> 4
template<> Vc_INTRINSIC Vc_CONST __m256 mask_cast<16, 4, __m256>(__m256i k)
{
    // abcd efgh ijkl mnop -> aaaa bbbb cccc dddd
    const auto tmp = _mm_unpacklo_epi16(AVX::lo128(k), AVX::lo128(k)); // aabb ccdd
    return _mm256_castsi256_ps(AVX::concat(_mm_unpacklo_epi32(tmp, tmp), _mm_unpackhi_epi32(tmp, tmp)));
}

// allone{{{1
template<> Vc_INTRINSIC Vc_CONST __m256  allone<__m256 >() { return AVX::setallone_ps(); }
template<> Vc_INTRINSIC Vc_CONST __m256i allone<__m256i>() { return AVX::setallone_si256(); }
template<> Vc_INTRINSIC Vc_CONST __m256d allone<__m256d>() { return AVX::setallone_pd(); }

// zero{{{1
template<> Vc_INTRINSIC Vc_CONST __m256  zero<__m256 >() { return _mm256_setzero_ps(); }
template<> Vc_INTRINSIC Vc_CONST __m256i zero<__m256i>() { return _mm256_setzero_si256(); }
template<> Vc_INTRINSIC Vc_CONST __m256d zero<__m256d>() { return _mm256_setzero_pd(); }

// one{{{1
Vc_INTRINSIC Vc_CONST __m256  one( float) { return AVX::setone_ps   (); }
Vc_INTRINSIC Vc_CONST __m256d one(double) { return AVX::setone_pd   (); }
Vc_INTRINSIC Vc_CONST __m256i one(   int) { return AVX::setone_epi32(); }
Vc_INTRINSIC Vc_CONST __m256i one(  uint) { return AVX::setone_epu32(); }
Vc_INTRINSIC Vc_CONST __m256i one( short) { return AVX::setone_epi16(); }
Vc_INTRINSIC Vc_CONST __m256i one(ushort) { return AVX::setone_epu16(); }
Vc_INTRINSIC Vc_CONST __m256i one( schar) { return AVX::setone_epi8 (); }
Vc_INTRINSIC Vc_CONST __m256i one( uchar) { return AVX::setone_epu8 (); }

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

// not_{{{1
Vc_INTRINSIC __m256  not_(__m256  a) { return andnot_(a, allone<__m256 >()); }
Vc_INTRINSIC __m256d not_(__m256d a) { return andnot_(a, allone<__m256d>()); }
Vc_INTRINSIC __m256i not_(__m256i a) { return andnot_(a, allone<__m256i>()); }

// abs{{{1
Vc_INTRINSIC __m256  abs(__m256  a,  float) { return and_(a, AVX::setabsmask_ps()); }
Vc_INTRINSIC __m256d abs(__m256d a, double) { return and_(a, AVX::setabsmask_pd()); }
Vc_INTRINSIC __m256i abs(__m256i a,    int) { return _mm256_abs_epi32(a); }
Vc_INTRINSIC __m256i abs(__m256i a,   uint) { return a; }
Vc_INTRINSIC __m256i abs(__m256i a,  short) { return _mm256_abs_epi16(a); }
Vc_INTRINSIC __m256i abs(__m256i a, ushort) { return a; }
Vc_INTRINSIC __m256i abs(__m256i a,  schar) { return _mm256_abs_epi8 (a); }
Vc_INTRINSIC __m256i abs(__m256i a,  uchar) { return a; }

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

// horizontal add{{{1
template <typename T> Vc_INTRINSIC T add(Common::IntrinsicType<T, 32 / sizeof(T)> a, T)
{
    return {add(add(AVX::lo128(a), AVX::hi128(a), T()), T())};
}

// horizontal mul{{{1
template <typename T> Vc_INTRINSIC T mul(Common::IntrinsicType<T, 32 / sizeof(T)> a, T)
{
    return {mul(mul(AVX::lo128(a), AVX::hi128(a), T()), T())};
}

// horizontal min{{{1
template <typename T> Vc_INTRINSIC T min(Common::IntrinsicType<T, 32 / sizeof(T)> a, T)
{
    return {min(min(AVX::lo128(a), AVX::hi128(a), T()), T())};
}

// horizontal max{{{1
template <typename T> Vc_INTRINSIC T max(Common::IntrinsicType<T, 32 / sizeof(T)> a, T)
{
    return {max(max(AVX::lo128(a), AVX::hi128(a), T()), T())};
}
// cmpeq{{{1
Vc_INTRINSIC __m256  cmpeq(__m256  a, __m256  b,  float) { return AvxIntrinsics::cmpeq_ps(a, b); }
Vc_INTRINSIC __m256d cmpeq(__m256d a, __m256d b, double) { return AvxIntrinsics::cmpeq_pd(a, b); }
Vc_INTRINSIC __m256i cmpeq(__m256i a, __m256i b,    int) { return AvxIntrinsics::cmpeq_epi32(a, b); }
Vc_INTRINSIC __m256i cmpeq(__m256i a, __m256i b,   uint) { return AvxIntrinsics::cmpeq_epi32(a, b); }
Vc_INTRINSIC __m256i cmpeq(__m256i a, __m256i b,  short) { return AvxIntrinsics::cmpeq_epi16(a, b); }
Vc_INTRINSIC __m256i cmpeq(__m256i a, __m256i b, ushort) { return AvxIntrinsics::cmpeq_epi16(a, b); }

// cmpneq{{{1
Vc_INTRINSIC __m256  cmpneq(__m256  a, __m256  b,  float) { return AvxIntrinsics::cmpneq_ps(a, b); }
Vc_INTRINSIC __m256d cmpneq(__m256d a, __m256d b, double) { return AvxIntrinsics::cmpneq_pd(a, b); }
Vc_INTRINSIC __m256i cmpneq(__m256i a, __m256i b,    int) { return not_(AvxIntrinsics::cmpeq_epi32(a, b)); }
Vc_INTRINSIC __m256i cmpneq(__m256i a, __m256i b,   uint) { return not_(AvxIntrinsics::cmpeq_epi32(a, b)); }
Vc_INTRINSIC __m256i cmpneq(__m256i a, __m256i b,  short) { return not_(AvxIntrinsics::cmpeq_epi16(a, b)); }
Vc_INTRINSIC __m256i cmpneq(__m256i a, __m256i b, ushort) { return not_(AvxIntrinsics::cmpeq_epi16(a, b)); }
Vc_INTRINSIC __m256i cmpneq(__m256i a, __m256i b,  schar) { return not_(AvxIntrinsics::cmpeq_epi8 (a, b)); }
Vc_INTRINSIC __m256i cmpneq(__m256i a, __m256i b,  uchar) { return not_(AvxIntrinsics::cmpeq_epi8 (a, b)); }

// cmpgt{{{1
Vc_INTRINSIC __m256  cmpgt(__m256  a, __m256  b,  float) { return AVX::cmpgt_ps(a, b); }
Vc_INTRINSIC __m256d cmpgt(__m256d a, __m256d b, double) { return AVX::cmpgt_pd(a, b); }
Vc_INTRINSIC __m256i cmpgt(__m256i a, __m256i b,    int) { return AVX::cmpgt_epi32(a, b); }
Vc_INTRINSIC __m256i cmpgt(__m256i a, __m256i b,   uint) { return AVX::cmpgt_epu32(a, b); }
Vc_INTRINSIC __m256i cmpgt(__m256i a, __m256i b,  short) { return AVX::cmpgt_epi16(a, b); }
Vc_INTRINSIC __m256i cmpgt(__m256i a, __m256i b, ushort) { return AVX::cmpgt_epu16(a, b); }
Vc_INTRINSIC __m256i cmpgt(__m256i a, __m256i b,  schar) { return AVX::cmpgt_epi8 (a, b); }
Vc_INTRINSIC __m256i cmpgt(__m256i a, __m256i b,  uchar) { return AVX::cmpgt_epu8 (a, b); }

// cmpge{{{1
Vc_INTRINSIC __m256  cmpge(__m256  a, __m256  b,  float) { return AVX::cmpge_ps(a, b); }
Vc_INTRINSIC __m256d cmpge(__m256d a, __m256d b, double) { return AVX::cmpge_pd(a, b); }
Vc_INTRINSIC __m256i cmpge(__m256i a, __m256i b,    int) { return not_(AVX::cmpgt_epi32(b, a)); }
Vc_INTRINSIC __m256i cmpge(__m256i a, __m256i b,   uint) { return not_(AVX::cmpgt_epu32(b, a)); }
Vc_INTRINSIC __m256i cmpge(__m256i a, __m256i b,  short) { return not_(AVX::cmpgt_epi16(b, a)); }
Vc_INTRINSIC __m256i cmpge(__m256i a, __m256i b, ushort) { return not_(AVX::cmpgt_epu16(b, a)); }
Vc_INTRINSIC __m256i cmpge(__m256i a, __m256i b,  schar) { return not_(AVX::cmpgt_epi8 (b, a)); }
Vc_INTRINSIC __m256i cmpge(__m256i a, __m256i b,  uchar) { return not_(AVX::cmpgt_epu8 (b, a)); }

// cmple{{{1
Vc_INTRINSIC __m256  cmple(__m256  a, __m256  b,  float) { return AVX::cmple_ps(a, b); }
Vc_INTRINSIC __m256d cmple(__m256d a, __m256d b, double) { return AVX::cmple_pd(a, b); }
Vc_INTRINSIC __m256i cmple(__m256i a, __m256i b,    int) { return not_(AVX::cmpgt_epi32(a, b)); }
Vc_INTRINSIC __m256i cmple(__m256i a, __m256i b,   uint) { return not_(AVX::cmpgt_epu32(a, b)); }
Vc_INTRINSIC __m256i cmple(__m256i a, __m256i b,  short) { return not_(AVX::cmpgt_epi16(a, b)); }
Vc_INTRINSIC __m256i cmple(__m256i a, __m256i b, ushort) { return not_(AVX::cmpgt_epu16(a, b)); }
Vc_INTRINSIC __m256i cmple(__m256i a, __m256i b,  schar) { return not_(AVX::cmpgt_epi8 (a, b)); }
Vc_INTRINSIC __m256i cmple(__m256i a, __m256i b,  uchar) { return not_(AVX::cmpgt_epu8 (a, b)); }

// cmplt{{{1
Vc_INTRINSIC __m256  cmplt(__m256  a, __m256  b,  float) { return AVX::cmplt_ps(a, b); }
Vc_INTRINSIC __m256d cmplt(__m256d a, __m256d b, double) { return AVX::cmplt_pd(a, b); }
Vc_INTRINSIC __m256i cmplt(__m256i a, __m256i b,    int) { return AVX::cmpgt_epi32(b, a); }
Vc_INTRINSIC __m256i cmplt(__m256i a, __m256i b,   uint) { return AVX::cmpgt_epu32(b, a); }
Vc_INTRINSIC __m256i cmplt(__m256i a, __m256i b,  short) { return AVX::cmpgt_epi16(b, a); }
Vc_INTRINSIC __m256i cmplt(__m256i a, __m256i b, ushort) { return AVX::cmpgt_epu16(b, a); }
Vc_INTRINSIC __m256i cmplt(__m256i a, __m256i b,  schar) { return AVX::cmpgt_epi8 (b, a); }
Vc_INTRINSIC __m256i cmplt(__m256i a, __m256i b,  uchar) { return AVX::cmpgt_epu8 (b, a); }

// fma{{{1
Vc_INTRINSIC void fma(__m256  &a, __m256  b, __m256  c,  float) {
#ifdef VC_IMPL_FMA4
    a = _mm256_macc_ps(a, b, c);
#elif defined VC_IMPL_FMA
    a = _mm256_fmadd_ps(a, b, c);
#else
    using namespace AVX;
    __m256d v1_0 = _mm256_cvtps_pd(lo128(a));
    __m256d v1_1 = _mm256_cvtps_pd(hi128(a));
    __m256d v2_0 = _mm256_cvtps_pd(lo128(b));
    __m256d v2_1 = _mm256_cvtps_pd(hi128(b));
    __m256d v3_0 = _mm256_cvtps_pd(lo128(c));
    __m256d v3_1 = _mm256_cvtps_pd(hi128(c));
    a = concat(_mm256_cvtpd_ps(_mm256_add_pd(_mm256_mul_pd(v1_0, v2_0), v3_0)),
               _mm256_cvtpd_ps(_mm256_add_pd(_mm256_mul_pd(v1_1, v2_1), v3_1)));
#endif
}
Vc_INTRINSIC void fma(__m256d &a, __m256d b, __m256d c, double) {
#ifdef VC_IMPL_FMA4
    a = _mm256_macc_pd(a, b, c);
#elif defined VC_IMPL_FMA
    a = _mm256_fmadd_pd(a, b, c);
#else
    using namespace AVX;
    __m256d h1 = and_(a, _mm256_broadcast_sd(reinterpret_cast<const double *>(
                             &c_general::highMaskDouble)));
    __m256d h2 = and_(b, _mm256_broadcast_sd(reinterpret_cast<const double *>(
                             &c_general::highMaskDouble)));
    const __m256d l1 = _mm256_sub_pd(a, h1);
    const __m256d l2 = _mm256_sub_pd(b, h2);
    const __m256d ll = mul(l1, l2, double());
    const __m256d lh = add(mul(l1, h2, double()), mul(h1, l2, double()), double());
    const __m256d hh = mul(h1, h2, double());
    // ll < lh < hh for all entries is certain
    const __m256d lh_lt_v3 = cmplt(abs(lh, double()), abs(c, double()), double());  // |lh| < |c|
    const __m256d x = _mm256_blendv_pd(c, lh, lh_lt_v3);
    const __m256d y = _mm256_blendv_pd(lh, c, lh_lt_v3);
    a = add(add(ll, x, double()), add(y, hh, double()), double());
#endif
}
template <typename T> Vc_INTRINSIC void fma(__m256i &a, __m256i b, __m256i c, T)
{
    a = add(mul(a, b, T()), c, T());
}

// shiftRight{{{1
template <int shift> Vc_INTRINSIC __m256i shiftRight(__m256i a,    int) { return AVX::srai_epi32<shift>(a); }
template <int shift> Vc_INTRINSIC __m256i shiftRight(__m256i a,   uint) { return AVX::srli_epi32<shift>(a); }
template <int shift> Vc_INTRINSIC __m256i shiftRight(__m256i a,  short) { return AVX::srai_epi16<shift>(a); }
template <int shift> Vc_INTRINSIC __m256i shiftRight(__m256i a, ushort) { return AVX::srli_epi16<shift>(a); }
//template <int shift> Vc_INTRINSIC __m256i shiftRight(__m256i a,  schar) { return AVX::srai_epi8 <shift>(a); }
//template <int shift> Vc_INTRINSIC __m256i shiftRight(__m256i a,  uchar) { return AVX::srli_epi8 <shift>(a); }

Vc_INTRINSIC __m256i shiftRight(__m256i a, int shift,    int) { return AVX::sra_epi32(a, _mm_cvtsi32_si128(shift)); }
Vc_INTRINSIC __m256i shiftRight(__m256i a, int shift,   uint) { return AVX::srl_epi32(a, _mm_cvtsi32_si128(shift)); }
Vc_INTRINSIC __m256i shiftRight(__m256i a, int shift,  short) { return AVX::sra_epi16(a, _mm_cvtsi32_si128(shift)); }
Vc_INTRINSIC __m256i shiftRight(__m256i a, int shift, ushort) { return AVX::srl_epi16(a, _mm_cvtsi32_si128(shift)); }
//Vc_INTRINSIC __m256i shiftRight(__m256i a, int shift,  schar) { return AVX::sra_epi8 (a, _mm_cvtsi32_si128(shift)); }
//Vc_INTRINSIC __m256i shiftRight(__m256i a, int shift,  uchar) { return AVX::srl_epi8 (a, _mm_cvtsi32_si128(shift)); }

// shiftLeft{{{1
template <int shift> Vc_INTRINSIC __m256i shiftLeft(__m256i a,    int) { return AVX::slli_epi32<shift>(a); }
template <int shift> Vc_INTRINSIC __m256i shiftLeft(__m256i a,   uint) { return AVX::slli_epi32<shift>(a); }
template <int shift> Vc_INTRINSIC __m256i shiftLeft(__m256i a,  short) { return AVX::slli_epi16<shift>(a); }
template <int shift> Vc_INTRINSIC __m256i shiftLeft(__m256i a, ushort) { return AVX::slli_epi16<shift>(a); }
//template <int shift> Vc_INTRINSIC __m256i shiftLeft(__m256i a,  schar) { return AVX::slli_epi8 <shift>(a); }
//template <int shift> Vc_INTRINSIC __m256i shiftLeft(__m256i a,  uchar) { return AVX::slli_epi8 <shift>(a); }

Vc_INTRINSIC __m256i shiftLeft(__m256i a, int shift,    int) { return AVX::sll_epi32(a, _mm_cvtsi32_si128(shift)); }
Vc_INTRINSIC __m256i shiftLeft(__m256i a, int shift,   uint) { return AVX::sll_epi32(a, _mm_cvtsi32_si128(shift)); }
Vc_INTRINSIC __m256i shiftLeft(__m256i a, int shift,  short) { return AVX::sll_epi16(a, _mm_cvtsi32_si128(shift)); }
Vc_INTRINSIC __m256i shiftLeft(__m256i a, int shift, ushort) { return AVX::sll_epi16(a, _mm_cvtsi32_si128(shift)); }
//Vc_INTRINSIC __m256i shiftLeft(__m256i a, int shift,  schar) { return AVX::sll_epi8 (a, _mm_cvtsi32_si128(shift)); }
//Vc_INTRINSIC __m256i shiftLeft(__m256i a, int shift,  uchar) { return AVX::sll_epi8 (a, _mm_cvtsi32_si128(shift)); }

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

// testc{{{1
Vc_INTRINSIC Vc_CONST int testc(__m128  a, __m128  b) { return _mm_testc_si128(_mm_castps_si128(a), _mm_castps_si128(b)); }
Vc_INTRINSIC Vc_CONST int testc(__m256  a, __m256  b) { return _mm256_testc_ps(a, b); }
Vc_INTRINSIC Vc_CONST int testc(__m256d a, __m256d b) { return _mm256_testc_pd(a, b); }
Vc_INTRINSIC Vc_CONST int testc(__m256i a, __m256i b) { return _mm256_testc_si256(a, b); }

// testz{{{1
Vc_INTRINSIC Vc_CONST int testz(__m128  a, __m128  b) { return _mm_testz_si128(_mm_castps_si128(a), _mm_castps_si128(b)); }
Vc_INTRINSIC Vc_CONST int testz(__m256  a, __m256  b) { return _mm256_testz_ps(a, b); }
Vc_INTRINSIC Vc_CONST int testz(__m256d a, __m256d b) { return _mm256_testz_pd(a, b); }
Vc_INTRINSIC Vc_CONST int testz(__m256i a, __m256i b) { return _mm256_testz_si256(a, b); }

// testnzc{{{1
Vc_INTRINSIC Vc_CONST int testnzc(__m128 a, __m128 b) { return _mm_testnzc_si128(_mm_castps_si128(a), _mm_castps_si128(b)); }
Vc_INTRINSIC Vc_CONST int testnzc(__m256  a, __m256  b) { return _mm256_testnzc_ps(a, b); }
Vc_INTRINSIC Vc_CONST int testnzc(__m256d a, __m256d b) { return _mm256_testnzc_pd(a, b); }
Vc_INTRINSIC Vc_CONST int testnzc(__m256i a, __m256i b) { return _mm256_testnzc_si256(a, b); }

// movemask{{{1
Vc_INTRINSIC Vc_CONST int movemask(__m256i a) { return AVX::movemask_epi8(a); }
Vc_INTRINSIC Vc_CONST int movemask(__m128i a) { return _mm_movemask_epi8(a); }
Vc_INTRINSIC Vc_CONST int movemask(__m256d a) { return _mm256_movemask_pd(a); }
Vc_INTRINSIC Vc_CONST int movemask(__m128d a) { return _mm_movemask_pd(a); }
Vc_INTRINSIC Vc_CONST int movemask(__m256  a) { return _mm256_movemask_ps(a); }
Vc_INTRINSIC Vc_CONST int movemask(__m128  a) { return _mm_movemask_ps(a); }

// mask_store{{{1
template <size_t N, typename Flags>
Vc_INTRINSIC void mask_store(__m256i k, bool *mem, Flags)
{
    static_assert(
        N == 4 || N == 8 || N == 16,
        "mask_store(__m256i, bool *) is only implemented for 4, 8, and 16 entries");
    switch (N) {
    case 4:
        *reinterpret_cast<MayAlias<int32_t> *>(mem) =
            (_mm_movemask_epi8(AVX::lo128(k)) |
             (_mm_movemask_epi8(AVX::hi128(k)) << 16)) &
            0x01010101;
        break;
    case 8: {
        const auto k2 = _mm_srli_epi16(_mm_packs_epi16(AVX::lo128(k), AVX::hi128(k)), 15);
        const auto k3 = _mm_packs_epi16(k2, _mm_setzero_si128());
#ifdef __x86_64__
        *reinterpret_cast<MayAlias<int64_t> *>(mem) = _mm_cvtsi128_si64(k3);
#else
        *reinterpret_cast<MayAlias<int32_t> *>(mem) = _mm_cvtsi128_si32(k3);
        *reinterpret_cast<MayAlias<int32_t> *>(mem + 4) = _mm_extract_epi32(k3, 1);
#endif
    } break;
    case 16: {
        const auto bools = Detail::and_(AVX::_mm_setone_epu8(),
                                        _mm_packs_epi16(AVX::lo128(k), AVX::hi128(k)));
        if (Flags::IsAligned) {
            _mm_store_si128(reinterpret_cast<__m128i *>(mem), bools);
        } else {
            _mm_storeu_si128(reinterpret_cast<__m128i *>(mem), bools);
        }
    } break;
    }
}

// mask_load{{{1
template <typename R, size_t N, typename Flags>
Vc_INTRINSIC R mask_load(const bool *mem, Flags,
                         enable_if<std::is_same<R, __m128>::value> = nullarg)
{
    static_assert(N == 4 || N == 8,
                  "mask_load<__m128>(const bool *) is only implemented for 4, 8 entries");
    switch (N) {
    case 4: {
        __m128i k = _mm_cvtsi32_si128(*reinterpret_cast<const int32_t *>(mem));
        k = _mm_unpacklo_epi8(k, k);
        k = _mm_unpacklo_epi16(k, k);
        k = _mm_cmpgt_epi32(k, _mm_setzero_si128());
        return AVX::avx_cast<__m128>(k);
    }
    case 8: {
        __m128i k = _mm_cvtsi64_si128(*reinterpret_cast<const int64_t *>(mem));
        return AVX::avx_cast<__m128>(
            _mm_cmpgt_epi16(_mm_unpacklo_epi8(k, k), _mm_setzero_si128()));
    }
    }
}

template <typename R, size_t N, typename Flags>
Vc_INTRINSIC R mask_load(const bool *mem, Flags,
                         enable_if<std::is_same<R, __m256>::value> = nullarg)
{
    static_assert(
        N == 4 || N == 8 || N == 16,
        "mask_load<__m256>(const bool *) is only implemented for 4, 8, and 16 entries");
    switch (N) {
    case 4: {
        __m128i k = AVX::avx_cast<__m128i>(_mm_and_ps(
            _mm_set1_ps(*reinterpret_cast<const float *>(mem)),
            AVX::avx_cast<__m128>(_mm_setr_epi32(0x1, 0x100, 0x10000, 0x1000000))));
        k = _mm_cmpgt_epi32(k, _mm_setzero_si128());
        return AVX::avx_cast<__m256>(
            AVX::concat(_mm_unpacklo_epi32(k, k), _mm_unpackhi_epi32(k, k)));
    }
    case 8: {
        __m128i k = _mm_cvtsi64_si128(*reinterpret_cast<const int64_t *>(mem));
        k = _mm_cmpgt_epi16(_mm_unpacklo_epi8(k, k), _mm_setzero_si128());
        return AVX::avx_cast<__m256>(
            AVX::concat(_mm_unpacklo_epi16(k, k), _mm_unpackhi_epi16(k, k)));
    }
    case 16: {
        const auto k128 = _mm_cmpgt_epi8(
            Flags::IsAligned ? _mm_load_si128(reinterpret_cast<const __m128i *>(mem))
                             : _mm_loadu_si128(reinterpret_cast<const __m128i *>(mem)),
            _mm_setzero_si128());
        return AVX::avx_cast<__m256>(
            AVX::concat(_mm_unpacklo_epi8(k128, k128), _mm_unpackhi_epi8(k128, k128)));
    }
    }
}

// mask_to_int{{{1
template <size_t Size>
Vc_INTRINSIC_L Vc_CONST_L int mask_to_int(__m256i x) Vc_INTRINSIC_R Vc_CONST_R;
template <> Vc_INTRINSIC Vc_CONST int mask_to_int<4>(__m256i k)
{
    return movemask(AVX::avx_cast<__m256d>(k));
}
template <> Vc_INTRINSIC Vc_CONST int mask_to_int<8>(__m256i k)
{
    return movemask(AVX::avx_cast<__m256>(k));
}
#ifdef VC_IMPL_BMI2
template <> Vc_INTRINSIC Vc_CONST int mask_to_int<16>(__m256i k)
{
    return _pext_u32(movemask(k), 0x55555555u);
}
#endif
template <> Vc_INTRINSIC Vc_CONST int mask_to_int<32>(__m256i k)
{
    return movemask(k);
}

//}}}1
}  // namespace Detail
}  // namespace Vc

#include "undomacros.h"

#endif  // VC_AVX_DETAIL_H_

// vim: foldmethod=marker
