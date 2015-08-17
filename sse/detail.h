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

#ifndef VC_SSE_DETAIL_H_
#define VC_SSE_DETAIL_H_

#include "casts.h"
#ifdef VC_IMPL_AVX
#include "../avx/intrinsics.h"
#endif
#include "vectorhelper.h"

#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Detail
{
// (converting) load functions {{{1
template <typename V, typename DstT> struct LoadTag
{
};

template <typename V, typename DstT, typename SrcT, typename Flags,
          typename =
              enable_if<(!std::is_integral<DstT>::value ||
                         !std::is_integral<SrcT>::value || sizeof(DstT) >= sizeof(SrcT))>>
Vc_INTRINSIC V load(const SrcT *mem, Flags flags)
{
    return load(mem, flags, LoadTag<V, DstT>());
}

// no conversion load from any T {{{2
template <typename V, typename T, typename Flags>
Vc_INTRINSIC V
    load(const T *mem, Flags, LoadTag<V, T>, enable_if<sizeof(V) == 16> = nullarg)
{
    return SSE::VectorHelper<V>::template load<Flags>(mem);
}

// short {{{2
template <typename Flags>
Vc_INTRINSIC __m128i load(const ushort *mem, Flags, LoadTag<__m128i, short>)
{
    return SSE::VectorHelper<__m128i>::load<Flags>(mem);
}
template <typename Flags>
Vc_INTRINSIC __m128i load(const uchar *mem, Flags, LoadTag<__m128i, short>)
{
    // the only available streaming load loads 16 bytes - twice as much as we need =>
    // can't use it, or we risk an out-of-bounds read and an unaligned load exception
    return SSE::cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem)));
}
template <typename Flags>
Vc_INTRINSIC __m128i load(const schar *mem, Flags, LoadTag<__m128i, short>)
{
    // the only available streaming load loads 16 bytes - twice as much as we need =>
    // can't use it, or we risk an out-of-bounds read and an unaligned load exception
    return SSE::cvtepi8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem)));
}

// ushort {{{2
template <typename Flags>
Vc_INTRINSIC __m128i load(const uchar *mem, Flags, LoadTag<__m128i, ushort>)
{
    // the only available streaming load loads 16 bytes - twice as much as we need =>
    // can't use it, or we risk an out-of-bounds read and an unaligned load exception
    return SSE::cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem)));
}

// int {{{2
template <typename Flags>
Vc_INTRINSIC __m128i load(const uint *mem, Flags, LoadTag<__m128i, int>)
{
    return SSE::VectorHelper<__m128i>::load<Flags>(mem);
}
// no difference between streaming and alignment, because the
// 32/64 bit loads are not available as streaming loads, and can always be unaligned
template <typename Flags>
Vc_INTRINSIC __m128i load(const ushort *mem, Flags, LoadTag<__m128i, int>)
{
    return SSE::cvtepu16_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem)));
}
template <typename Flags>
Vc_INTRINSIC __m128i load(const short *mem, Flags, LoadTag<__m128i, int>)
{
    return SSE::cvtepi16_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem)));
}
template <typename Flags>
Vc_INTRINSIC __m128i load(const uchar *mem, Flags, LoadTag<__m128i, int>)
{
    return SSE::cvtepu8_epi32(_mm_cvtsi32_si128(*reinterpret_cast<const int *>(mem)));
}
template <typename Flags>
Vc_INTRINSIC __m128i load(const schar *mem, Flags, LoadTag<__m128i, int>)
{
    return SSE::cvtepi8_epi32(_mm_cvtsi32_si128(*reinterpret_cast<const int *>(mem)));
}

// uint {{{2
template <typename Flags>
Vc_INTRINSIC __m128i load(const ushort *mem, Flags, LoadTag<__m128i, uint>)
{
    return SSE::cvtepu16_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem)));
}
template <typename Flags>
Vc_INTRINSIC __m128i load(const uchar *mem, Flags, LoadTag<__m128i, uint>)
{
    return SSE::cvtepu8_epi32(_mm_cvtsi32_si128(*reinterpret_cast<const int *>(mem)));
}

// double {{{2
template <typename Flags>
Vc_INTRINSIC __m128d load(const float *mem, Flags, LoadTag<__m128d, double>)
{
    return SSE::StaticCastHelper<float, double>::cast(
        _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64 *>(mem)));
}
template <typename Flags>
Vc_INTRINSIC __m128d load(const uint *mem, Flags f, LoadTag<__m128d, double>)
{
    return SSE::StaticCastHelper<uint, double>::cast(
        _mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem)));
}
template <typename Flags>
Vc_INTRINSIC __m128d load(const int *mem, Flags f, LoadTag<__m128d, double>)
{
    return SSE::StaticCastHelper<int, double>::cast(
        _mm_loadl_epi64(reinterpret_cast<const __m128i *>(mem)));
}
template <typename Flags>
Vc_INTRINSIC __m128d load(const ushort *mem, Flags f, LoadTag<__m128d, double>)
{
    return SSE::StaticCastHelper<ushort, double>::cast(
        _mm_cvtsi32_si128(*reinterpret_cast<const int *>(mem)));
}
template <typename Flags>
Vc_INTRINSIC __m128d load(const short *mem, Flags f, LoadTag<__m128d, double>)
{
    return SSE::StaticCastHelper<short, double>::cast(
        _mm_cvtsi32_si128(*reinterpret_cast<const int *>(mem)));
}
template <typename Flags>
Vc_INTRINSIC __m128d load(const uchar *mem, Flags f, LoadTag<__m128d, double>)
{
    return SSE::StaticCastHelper<uchar, double>::cast(
        _mm_set1_epi16(*reinterpret_cast<const short *>(mem)));
}
template <typename Flags>
Vc_INTRINSIC __m128d load(const schar *mem, Flags f, LoadTag<__m128d, double>)
{
    return SSE::StaticCastHelper<char, double>::cast(
        _mm_set1_epi16(*reinterpret_cast<const short *>(mem)));
}

// float {{{2
template <typename Flags>
Vc_INTRINSIC __m128 load(const double *mem, Flags, LoadTag<__m128, float>)
{
#ifdef VC_IMPL_AVX
    if (Flags::IsUnaligned) {
        return _mm256_cvtpd_ps(_mm256_loadu_pd(mem));
    } else if (Flags::IsStreaming) {
        return _mm256_cvtpd_ps(AvxIntrinsics::stream_load<__m256d>(mem));
    } else {  // Flags::IsAligned
        return _mm256_cvtpd_ps(_mm256_load_pd(mem));
    }
#else
    return _mm_movelh_ps(_mm_cvtpd_ps(SSE::VectorHelper<__m128d>::load<Flags>(&mem[0])),
                         _mm_cvtpd_ps(SSE::VectorHelper<__m128d>::load<Flags>(&mem[2])));
#endif
}
template <typename Flags>
Vc_INTRINSIC __m128 load(const uint *mem, Flags f, LoadTag<__m128, float>)
{
    return SSE::StaticCastHelper<uint, float>::cast(load<__m128i, uint>(mem, f));
}
template <typename T, typename Flags,
          typename = enable_if<!std::is_same<T, float>::value>>
Vc_INTRINSIC __m128 load(const T *mem, Flags f, LoadTag<__m128, float>)
{
    return _mm_cvtepi32_ps(load<__m128i, int>(mem, f));
}

// shifted{{{1
template <int amount, typename T>
Vc_INTRINSIC Vc_PURE enable_if<(sizeof(T) == 16 && amount > 0), T> shifted(T k)
{
    return _mm_srli_si128(k, amount);
}
template <int amount, typename T>
Vc_INTRINSIC Vc_PURE enable_if<(sizeof(T) == 16 && amount < 0), T> shifted(T k)
{
    return _mm_slli_si128(k, -amount);
}

// IndexesFromZero{{{1
template <typename T, int Size> Vc_INTRINSIC Vc_CONST const T *IndexesFromZero()
{
    if (Size == 4) {
        return reinterpret_cast<const T *>(SSE::_IndexesFromZero4);
    } else if (Size == 8) {
        return reinterpret_cast<const T *>(SSE::_IndexesFromZero8);
    } else if (Size == 16) {
        return reinterpret_cast<const T *>(SSE::_IndexesFromZero16);
    }
    return 0;
}

// mask_cast{{{1
template<size_t From, size_t To, typename R> Vc_INTRINSIC Vc_CONST R mask_cast(__m128i k)
{
    static_assert(From == To, "Incorrect mask cast.");
    static_assert(std::is_same<R, __m128>::value, "Incorrect mask cast.");
    return SSE::sse_cast<__m128>(k);
}

template<> Vc_INTRINSIC Vc_CONST __m128 mask_cast<2, 4, __m128>(__m128i k)
{
    return SSE::sse_cast<__m128>(_mm_packs_epi16(k, _mm_setzero_si128()));
}
template<> Vc_INTRINSIC Vc_CONST __m128 mask_cast<2, 8, __m128>(__m128i k)
{
    return SSE::sse_cast<__m128>(
        _mm_packs_epi16(_mm_packs_epi16(k, _mm_setzero_si128()), _mm_setzero_si128()));
}

template<> Vc_INTRINSIC Vc_CONST __m128 mask_cast<4, 2, __m128>(__m128i k)
{
    return SSE::sse_cast<__m128>(_mm_unpacklo_epi32(k, k));
}
template<> Vc_INTRINSIC Vc_CONST __m128 mask_cast<4, 8, __m128>(__m128i k)
{
    return SSE::sse_cast<__m128>(_mm_packs_epi16(k, _mm_setzero_si128()));
}

template<> Vc_INTRINSIC Vc_CONST __m128 mask_cast<8, 2, __m128>(__m128i k)
{
    const auto tmp = _mm_unpacklo_epi16(k, k);
    return SSE::sse_cast<__m128>(_mm_unpacklo_epi32(tmp, tmp));
}
template<> Vc_INTRINSIC Vc_CONST __m128 mask_cast<8, 4, __m128>(__m128i k)
{
    return SSE::sse_cast<__m128>(_mm_unpacklo_epi16(k, k));
}

template<> Vc_INTRINSIC Vc_CONST __m128 mask_cast<16, 8, __m128>(__m128i k)
{
    return SSE::sse_cast<__m128>(_mm_unpacklo_epi8(k, k));
}
template<> Vc_INTRINSIC Vc_CONST __m128 mask_cast<16, 4, __m128>(__m128i k)
{
    const auto tmp = SSE::sse_cast<__m128i>(mask_cast<16, 8, __m128>(k));
    return SSE::sse_cast<__m128>(_mm_unpacklo_epi16(tmp, tmp));
}
template<> Vc_INTRINSIC Vc_CONST __m128 mask_cast<16, 2, __m128>(__m128i k)
{
    const auto tmp = SSE::sse_cast<__m128i>(mask_cast<16, 4, __m128>(k));
    return SSE::sse_cast<__m128>(_mm_unpacklo_epi32(tmp, tmp));
}

// allone{{{1
template <typename V> Vc_INTRINSIC_L Vc_CONST_L V allone() Vc_INTRINSIC_R Vc_CONST_R;
template<> Vc_INTRINSIC Vc_CONST __m128  allone<__m128 >() { return SSE::_mm_setallone_ps(); }
template<> Vc_INTRINSIC Vc_CONST __m128i allone<__m128i>() { return SSE::_mm_setallone_si128(); }
template<> Vc_INTRINSIC Vc_CONST __m128d allone<__m128d>() { return SSE::_mm_setallone_pd(); }

// zero{{{1
template <typename V> Vc_INTRINSIC_L Vc_CONST_L V zero() Vc_INTRINSIC_R Vc_CONST_R;
template<> Vc_INTRINSIC Vc_CONST __m128  zero<__m128 >() { return _mm_setzero_ps(); }
template<> Vc_INTRINSIC Vc_CONST __m128i zero<__m128i>() { return _mm_setzero_si128(); }
template<> Vc_INTRINSIC Vc_CONST __m128d zero<__m128d>() { return _mm_setzero_pd(); }

// negate{{{1
Vc_ALWAYS_INLINE Vc_CONST __m128 negate(__m128 v, std::integral_constant<std::size_t, 4>)
{
    return _mm_xor_ps(v, SSE::_mm_setsignmask_ps());
}
Vc_ALWAYS_INLINE Vc_CONST __m128d negate(__m128d v, std::integral_constant<std::size_t, 8>)
{
    return _mm_xor_pd(v, SSE::_mm_setsignmask_pd());
}
Vc_ALWAYS_INLINE Vc_CONST __m128i negate(__m128i v, std::integral_constant<std::size_t, 4>)
{
#ifdef VC_IMPL_SSSE3
    return _mm_sign_epi32(v, allone<__m128i>());
#else
    return _mm_sub_epi32(_mm_setzero_si128(), v);
#endif
}
Vc_ALWAYS_INLINE Vc_CONST __m128i negate(__m128i v, std::integral_constant<std::size_t, 2>)
{
#ifdef VC_IMPL_SSSE3
    return _mm_sign_epi16(v, allone<__m128i>());
#else
    return _mm_sub_epi16(_mm_setzero_si128(), v);
#endif
}

// xor_{{{1
Vc_INTRINSIC __m128 xor_(__m128 a, __m128 b) { return _mm_xor_ps(a, b); }
Vc_INTRINSIC __m128d xor_(__m128d a, __m128d b) { return _mm_xor_pd(a, b); }
Vc_INTRINSIC __m128i xor_(__m128i a, __m128i b) { return _mm_xor_si128(a, b); }

// or_{{{1
Vc_INTRINSIC __m128 or_(__m128 a, __m128 b) { return _mm_or_ps(a, b); }
Vc_INTRINSIC __m128d or_(__m128d a, __m128d b) { return _mm_or_pd(a, b); }
Vc_INTRINSIC __m128i or_(__m128i a, __m128i b) { return _mm_or_si128(a, b); }

// and_{{{1
Vc_INTRINSIC __m128 and_(__m128 a, __m128 b) { return _mm_and_ps(a, b); }
Vc_INTRINSIC __m128d and_(__m128d a, __m128d b) { return _mm_and_pd(a, b); }
Vc_INTRINSIC __m128i and_(__m128i a, __m128i b) { return _mm_and_si128(a, b); }

// andnot_{{{1
Vc_INTRINSIC __m128 andnot_(__m128 a, __m128 b) { return _mm_andnot_ps(a, b); }
Vc_INTRINSIC __m128d andnot_(__m128d a, __m128d b) { return _mm_andnot_pd(a, b); }
Vc_INTRINSIC __m128i andnot_(__m128i a, __m128i b) { return _mm_andnot_si128(a, b); }

// add{{{1
Vc_INTRINSIC __m128  add(__m128  a, __m128  b,  float) { return _mm_add_ps(a, b); }
Vc_INTRINSIC __m128d add(__m128d a, __m128d b, double) { return _mm_add_pd(a, b); }
Vc_INTRINSIC __m128i add(__m128i a, __m128i b,    int) { return _mm_add_epi32(a, b); }
Vc_INTRINSIC __m128i add(__m128i a, __m128i b,   uint) { return _mm_add_epi32(a, b); }
Vc_INTRINSIC __m128i add(__m128i a, __m128i b,  short) { return _mm_add_epi16(a, b); }
Vc_INTRINSIC __m128i add(__m128i a, __m128i b, ushort) { return _mm_add_epi16(a, b); }
Vc_INTRINSIC __m128i add(__m128i a, __m128i b,  schar) { return _mm_add_epi8 (a, b); }
Vc_INTRINSIC __m128i add(__m128i a, __m128i b,  uchar) { return _mm_add_epi8 (a, b); }

// sub{{{1
Vc_INTRINSIC __m128  sub(__m128  a, __m128  b,  float) { return _mm_sub_ps(a, b); }
Vc_INTRINSIC __m128d sub(__m128d a, __m128d b, double) { return _mm_sub_pd(a, b); }
Vc_INTRINSIC __m128i sub(__m128i a, __m128i b,    int) { return _mm_sub_epi32(a, b); }
Vc_INTRINSIC __m128i sub(__m128i a, __m128i b,   uint) { return _mm_sub_epi32(a, b); }
Vc_INTRINSIC __m128i sub(__m128i a, __m128i b,  short) { return _mm_sub_epi16(a, b); }
Vc_INTRINSIC __m128i sub(__m128i a, __m128i b, ushort) { return _mm_sub_epi16(a, b); }
Vc_INTRINSIC __m128i sub(__m128i a, __m128i b,  schar) { return _mm_sub_epi8 (a, b); }
Vc_INTRINSIC __m128i sub(__m128i a, __m128i b,  uchar) { return _mm_sub_epi8 (a, b); }

// mul{{{1
Vc_INTRINSIC __m128  mul(__m128  a, __m128  b,  float) { return _mm_mul_ps(a, b); }
Vc_INTRINSIC __m128d mul(__m128d a, __m128d b, double) { return _mm_mul_pd(a, b); }
Vc_INTRINSIC __m128i mul(__m128i a, __m128i b,    int) {
#ifdef VC_IMPL_SSE4_1
    return _mm_mullo_epi32(a, b);
#else
    const __m128i aShift = _mm_srli_si128(a, 4);
    const __m128i ab02 = _mm_mul_epu32(a, b);  // [a0 * b0, a2 * b2]
    const __m128i bShift = _mm_srli_si128(b, 4);
    const __m128i ab13 = _mm_mul_epu32(aShift, bShift);  // [a1 * b1, a3 * b3]
    return _mm_unpacklo_epi32(_mm_shuffle_epi32(ab02, 8), _mm_shuffle_epi32(ab13, 8));
#endif
}
Vc_INTRINSIC __m128i mul(__m128i a, __m128i b,   uint) { return mul(a, b, int()); }
Vc_INTRINSIC __m128i mul(__m128i a, __m128i b,  short) { return _mm_mullo_epi16(a, b); }
Vc_INTRINSIC __m128i mul(__m128i a, __m128i b, ushort) { return _mm_mullo_epi16(a, b); }
Vc_INTRINSIC __m128i mul(__m128i a, __m128i b,  schar) {
#ifdef VC_USE_BUILTIN_VECTOR_TYPES
    using B = Common::BuiltinType<schar, 16>;
    const auto x = reinterpret_cast<const MayAlias<B> &>(a) *
                   reinterpret_cast<const MayAlias<B> &>(b);
    return reinterpret_cast<const __m128i &>(x);
#else
    return or_(
        and_(_mm_mullo_epi16(a, b), _mm_slli_epi16(allone<__m128i>(), 8)),
        _mm_slli_epi16(_mm_mullo_epi16(_mm_srli_si128(a, 1), _mm_srli_si128(b, 1)), 8));
#endif
}
Vc_INTRINSIC __m128i mul(__m128i a, __m128i b,  uchar) {
#ifdef VC_USE_BUILTIN_VECTOR_TYPES
    using B = Common::BuiltinType<uchar, 16>;
    const auto x = reinterpret_cast<const MayAlias<B> &>(a) *
                   reinterpret_cast<const MayAlias<B> &>(b);
    return reinterpret_cast<const __m128i &>(x);
#else
    return or_(
        and_(_mm_mullo_epi16(a, b), _mm_slli_epi16(allone<__m128i>(), 8)),
        _mm_slli_epi16(_mm_mullo_epi16(_mm_srli_si128(a, 1), _mm_srli_si128(b, 1)), 8));
#endif
}

// TODO: fma{{{1
//Vc_INTRINSIC __m128  fma(__m128  a, __m128  b, __m128  c,  float) { return _mm_mul_ps(a, b); }
//Vc_INTRINSIC __m128d fma(__m128d a, __m128d b, __m128d c, double) { return _mm_mul_pd(a, b); }
//Vc_INTRINSIC __m128i fma(__m128i a, __m128i b, __m128i c,    int) { }
//Vc_INTRINSIC __m128i fma(__m128i a, __m128i b, __m128i c,   uint) { return fma(a, b, int()); }
//Vc_INTRINSIC __m128i fma(__m128i a, __m128i b, __m128i c,  short) { return _mm_mullo_epi16(a, b); }
//Vc_INTRINSIC __m128i fma(__m128i a, __m128i b, __m128i c, ushort) { return _mm_mullo_epi16(a, b); }
//Vc_INTRINSIC __m128i fma(__m128i a, __m128i b, __m128i c,  schar) { }
//Vc_INTRINSIC __m128i fma(__m128i a, __m128i b, __m128i c,  uchar) { }

// min{{{1
Vc_INTRINSIC __m128  min(__m128  a, __m128  b,  float) { return _mm_min_ps(a, b); }
Vc_INTRINSIC __m128d min(__m128d a, __m128d b, double) { return _mm_min_pd(a, b); }
Vc_INTRINSIC __m128i min(__m128i a, __m128i b,    int) { return SSE::min_epi32(a, b); }
Vc_INTRINSIC __m128i min(__m128i a, __m128i b,   uint) { return SSE::min_epu32(a, b); }
Vc_INTRINSIC __m128i min(__m128i a, __m128i b,  short) { return _mm_min_epi16(a, b); }
Vc_INTRINSIC __m128i min(__m128i a, __m128i b, ushort) { return _mm_min_epu16(a, b); }
Vc_INTRINSIC __m128i min(__m128i a, __m128i b,  schar) { return _mm_min_epi8 (a, b); }
Vc_INTRINSIC __m128i min(__m128i a, __m128i b,  uchar) { return _mm_min_epu8 (a, b); }

// max{{{1
Vc_INTRINSIC __m128  max(__m128  a, __m128  b,  float) { return _mm_max_ps(a, b); }
Vc_INTRINSIC __m128d max(__m128d a, __m128d b, double) { return _mm_max_pd(a, b); }
Vc_INTRINSIC __m128i max(__m128i a, __m128i b,    int) { return SSE::max_epi32(a, b); }
Vc_INTRINSIC __m128i max(__m128i a, __m128i b,   uint) { return SSE::max_epu32(a, b); }
Vc_INTRINSIC __m128i max(__m128i a, __m128i b,  short) { return _mm_max_epi16(a, b); }
Vc_INTRINSIC __m128i max(__m128i a, __m128i b, ushort) { return _mm_max_epu16(a, b); }
Vc_INTRINSIC __m128i max(__m128i a, __m128i b,  schar) { return _mm_max_epi8 (a, b); }
Vc_INTRINSIC __m128i max(__m128i a, __m128i b,  uchar) { return _mm_max_epu8 (a, b); }

// horizontal add{{{1
Vc_INTRINSIC  float add(__m128  a,  float) {
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 1, 1, 1)));
    return _mm_cvtss_f32(a);
}
Vc_INTRINSIC double add(__m128d a, double) {
    a = _mm_add_sd(a, _mm_unpackhi_pd(a, a));
    return _mm_cvtsd_f64(a);
}
Vc_INTRINSIC    int add(__m128i a,    int) {
    a = add(a, _mm_srli_si128(a, 8), int());
    a = add(a, _mm_srli_si128(a, 4), int());
    return _mm_cvtsi128_si32(a);
}
Vc_INTRINSIC   uint add(__m128i a,   uint) { return add(a, int()); }
Vc_INTRINSIC  short add(__m128i a,  short) {
    a = add(a, _mm_srli_si128(a, 8), short());
    a = add(a, _mm_srli_si128(a, 4), short());
    a = add(a, _mm_srli_si128(a, 2), short());
    return _mm_cvtsi128_si32(a);  // & 0xffff is implicit
}
Vc_INTRINSIC ushort add(__m128i a, ushort) { return add(a, short()); }
Vc_INTRINSIC  schar add(__m128i a,  schar) {
    a = add(a, _mm_srli_si128(a, 8), schar());
    a = add(a, _mm_srli_si128(a, 4), schar());
    a = add(a, _mm_srli_si128(a, 2), schar());
    a = add(a, _mm_srli_si128(a, 1), schar());
    return _mm_cvtsi128_si32(a);  // & 0xff is implicit
}
Vc_INTRINSIC  uchar add(__m128i a,  uchar) { return add(a, schar()); }

// horizontal mul{{{1
Vc_INTRINSIC  float mul(__m128  a,  float) {
    a = _mm_mul_ps(a, _mm_movehl_ps(a, a));
    a = _mm_mul_ss(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 1, 1, 1)));
    return _mm_cvtss_f32(a);
}
Vc_INTRINSIC double mul(__m128d a, double) {
    a = _mm_mul_sd(a, _mm_unpackhi_pd(a, a));
    return _mm_cvtsd_f64(a);
}
Vc_INTRINSIC    int mul(__m128i a,    int) {
    a = mul(a, _mm_srli_si128(a, 8), int());
    a = mul(a, _mm_srli_si128(a, 4), int());
    return _mm_cvtsi128_si32(a);
}
Vc_INTRINSIC   uint mul(__m128i a,   uint) { return mul(a, int()); }
Vc_INTRINSIC  short mul(__m128i a,  short) {
    a = mul(a, _mm_srli_si128(a, 8), short());
    a = mul(a, _mm_srli_si128(a, 4), short());
    a = mul(a, _mm_srli_si128(a, 2), short());
    return _mm_cvtsi128_si32(a);  // & 0xffff is implicit
}
Vc_INTRINSIC ushort mul(__m128i a, ushort) { return mul(a, short()); }
Vc_INTRINSIC  schar mul(__m128i a,  schar) {
    // convert to two short vectors, multiply them and then do horizontal reduction
    const __m128i s0 = _mm_srai_epi16(a, 1);
    const __m128i s1 = Detail::and_(a, _mm_set1_epi32(0x0f0f0f0f));
    return mul(mul(s0, s1, short()), short());
}
Vc_INTRINSIC  uchar mul(__m128i a,  uchar) { return mul(a, schar()); }

// horizontal min{{{1
Vc_INTRINSIC  float min(__m128  a,  float) {
    a = _mm_min_ps(a, _mm_movehl_ps(a, a));
    a = _mm_min_ss(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 1, 1, 1)));
    return _mm_cvtss_f32(a);
}
Vc_INTRINSIC double min(__m128d a, double) {
    a = _mm_min_sd(a, _mm_unpackhi_pd(a, a));
    return _mm_cvtsd_f64(a);
}
Vc_INTRINSIC    int min(__m128i a,    int) {
    a = min(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)), int());
    a = min(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)), int());
    return _mm_cvtsi128_si32(a);
}
Vc_INTRINSIC   uint min(__m128i a,   uint) {
    a = min(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)), uint());
    a = min(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)), uint());
    return _mm_cvtsi128_si32(a);
}
Vc_INTRINSIC  short min(__m128i a,  short) {
    a = min(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)), short());
    a = min(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)), short());
    a = min(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)), short());
    return _mm_cvtsi128_si32(a);  // & 0xffff is implicit
}
Vc_INTRINSIC ushort min(__m128i a, ushort) {
    a = min(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)), ushort());
    a = min(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)), ushort());
    a = min(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)), ushort());
    return _mm_cvtsi128_si32(a);  // & 0xffff is implicit
}
Vc_INTRINSIC  schar min(__m128i a,  schar) {
    a = min(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)), schar());
    a = min(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)), schar());
    a = min(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)), schar());
    return std::min(schar(_mm_cvtsi128_si32(a) >> 8), schar(_mm_cvtsi128_si32(a)));
}
Vc_INTRINSIC  uchar min(__m128i a,  uchar) {
    a = min(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)), schar());
    a = min(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)), schar());
    a = min(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)), schar());
    return std::min((_mm_cvtsi128_si32(a) >> 8) & 0xff, _mm_cvtsi128_si32(a) & 0xff);
}

// horizontal max{{{1
Vc_INTRINSIC  float max(__m128  a,  float) {
    a = _mm_max_ps(a, _mm_movehl_ps(a, a));
    a = _mm_max_ss(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 1, 1, 1)));
    return _mm_cvtss_f32(a);
}
Vc_INTRINSIC double max(__m128d a, double) {
    a = _mm_max_sd(a, _mm_unpackhi_pd(a, a));
    return _mm_cvtsd_f64(a);
}
Vc_INTRINSIC    int max(__m128i a,    int) {
    a = max(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)), int());
    a = max(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)), int());
    return _mm_cvtsi128_si32(a);
}
Vc_INTRINSIC   uint max(__m128i a,   uint) {
    a = max(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)), uint());
    a = max(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)), uint());
    return _mm_cvtsi128_si32(a);
}
Vc_INTRINSIC  short max(__m128i a,  short) {
    a = max(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)), short());
    a = max(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)), short());
    a = max(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)), short());
    return _mm_cvtsi128_si32(a);  // & 0xffff is implicit
}
Vc_INTRINSIC ushort max(__m128i a, ushort) {
    a = max(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)), ushort());
    a = max(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)), ushort());
    a = max(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)), ushort());
    return _mm_cvtsi128_si32(a);  // & 0xffff is implicit
}
Vc_INTRINSIC  schar max(__m128i a,  schar) {
    a = max(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)), schar());
    a = max(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)), schar());
    a = max(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)), schar());
    return std::max(schar(_mm_cvtsi128_si32(a) >> 8), schar(_mm_cvtsi128_si32(a)));
}
Vc_INTRINSIC  uchar max(__m128i a,  uchar) {
    a = max(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)), schar());
    a = max(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)), schar());
    a = max(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)), schar());
    return std::max((_mm_cvtsi128_si32(a) >> 8) & 0xff, _mm_cvtsi128_si32(a) & 0xff);
}

// sorted{{{1
template <Vc::Implementation, typename T>
Vc_CONST_L SSE::Vector<T> sorted(VC_ALIGNED_PARAMETER(SSE::Vector<T>) x) Vc_CONST_R;
template <typename T>
Vc_INTRINSIC Vc_CONST SSE::Vector<T> sorted(VC_ALIGNED_PARAMETER(SSE::Vector<T>) x)
{
    static_assert(!CurrentImplementation::is(ScalarImpl),
                  "Detail::sorted can only be instantiated if a non-Scalar "
                  "implementation is selected.");
    return sorted < CurrentImplementation::is_between(SSE2Impl, SSSE3Impl)
               ? SSE2Impl
               : CurrentImplementation::is_between(SSE41Impl, SSE42Impl)
                     ? SSE41Impl
                     : CurrentImplementation::current() > (x);
}
//}}}1
}  // namespace Detail
}  // namespace Vc

#include "undomacros.h"

#endif  // VC_SSE_DETAIL_H_

// vim: foldmethod=marker
