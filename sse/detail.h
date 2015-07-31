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

#ifdef VC_IMPL_AVX
#include "../avx/vectorhelper.h"
#endif

#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Detail
{
using uint = unsigned int;
using ushort = unsigned short;
using uchar = unsigned char;
using schar = signed char;

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
    return _mm256_cvtpd_ps(AVX::VectorHelper<__m256d>::load<Flags>(mem));
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

//}}}1
}  // namespace Detail
}  // namespace Vc

#include "undomacros.h"

#endif  // VC_SSE_DETAIL_H_

// vim: foldmethod=marker
