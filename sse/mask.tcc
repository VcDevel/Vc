/*  This file is part of the Vc library. {{{
Copyright © 2013-2014 Matthias Kretz <kretz@kde.org>
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

#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace SSE
{
namespace internal {

// mask_cast/*{{{*/
template<size_t From, size_t To> Vc_ALWAYS_INLINE Vc_CONST __m128i mask_cast(__m128i k)
{
    static_assert(From == To, "Incorrect mask cast.");
    return k;
}
template<> Vc_ALWAYS_INLINE Vc_CONST __m128i mask_cast<2, 4>(__m128i k)
{
    return _mm_packs_epi16(k, _mm_setzero_si128());
}
template<> Vc_ALWAYS_INLINE Vc_CONST __m128i mask_cast<2, 8>(__m128i k)
{
    auto tmp = _mm_packs_epi16(k, _mm_setzero_si128());
    return _mm_packs_epi16(tmp, _mm_setzero_si128());
}

template<> Vc_ALWAYS_INLINE Vc_CONST __m128i mask_cast<4, 2>(__m128i k)
{
    return _mm_unpacklo_epi32(k, k);
}
template<> Vc_ALWAYS_INLINE Vc_CONST __m128i mask_cast<4, 8>(__m128i k)
{
    return _mm_packs_epi16(k, _mm_setzero_si128());
}

template<> Vc_ALWAYS_INLINE Vc_CONST __m128i mask_cast<8, 2>(__m128i k)
{
    auto tmp = _mm_unpacklo_epi16(k, k);
    return _mm_unpacklo_epi32(tmp, tmp);
}
template<> Vc_ALWAYS_INLINE Vc_CONST __m128i mask_cast<8, 4>(__m128i k)
{
    return _mm_unpacklo_epi16(k, k);
}

template<> Vc_ALWAYS_INLINE Vc_CONST __m128i mask_cast<16, 8>(__m128i k)
{
    return _mm_unpacklo_epi8(k, k);
}
template<> Vc_ALWAYS_INLINE Vc_CONST __m128i mask_cast<16, 4>(__m128i k)
{
    auto tmp = mask_cast<16, 8>(k);
    return _mm_unpacklo_epi16(tmp, tmp);
}
template<> Vc_ALWAYS_INLINE Vc_CONST __m128i mask_cast<16, 2>(__m128i k)
{
    auto tmp = mask_cast<16, 4>(k);
    return _mm_unpacklo_epi32(tmp, tmp);
}
/*}}}*/
// mask_to_int/*{{{*/
template<> Vc_ALWAYS_INLINE Vc_CONST int mask_to_int<2>(__m128i k)
{
    return _mm_movemask_pd(_mm_castsi128_pd(k));
}
template<> Vc_ALWAYS_INLINE Vc_CONST int mask_to_int<4>(__m128i k)
{
    return _mm_movemask_ps(_mm_castsi128_ps(k));
}
template<> Vc_ALWAYS_INLINE Vc_CONST int mask_to_int<8>(__m128i k)
{
    return _mm_movemask_epi8(_mm_packs_epi16(k, _mm_setzero_si128()));
}
template<> Vc_ALWAYS_INLINE Vc_CONST int mask_to_int<16>(__m128i k)
{
    return _mm_movemask_epi8(k);
}
/*}}}*/
/*mask_count{{{*/
template<> Vc_ALWAYS_INLINE Vc_CONST int mask_count<2>(__m128i k)
{
    int mask = _mm_movemask_pd(_mm_castsi128_pd(k));
    return (mask & 1) + (mask >> 1);
}

template<> Vc_ALWAYS_INLINE Vc_CONST int mask_count<4>(__m128i k)
{
#ifdef VC_IMPL_POPCNT
    return _mm_popcnt_u32(_mm_movemask_ps(_mm_castsi128_ps(k)));
//X     tmp = (tmp & 5) + ((tmp >> 1) & 5);
//X     return (tmp & 3) + ((tmp >> 2) & 3);
#else
    auto x = _mm_srli_epi32(k, 31);
    x = _mm_add_epi32(x, _mm_shuffle_epi32(x, _MM_SHUFFLE(0, 1, 2, 3)));
    x = _mm_add_epi32(x, _mm_shufflelo_epi16(x, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si32(x);
#endif
}

template<> Vc_ALWAYS_INLINE Vc_CONST int mask_count<8>(__m128i k)
{
#ifdef VC_IMPL_POPCNT
    return _mm_popcnt_u32(_mm_movemask_epi8(k)) / 2;
#else
//X     int tmp = _mm_movemask_epi8(dataI());
//X     tmp = (tmp & 0x1111) + ((tmp >> 2) & 0x1111);
//X     tmp = (tmp & 0x0303) + ((tmp >> 4) & 0x0303);
//X     return (tmp & 0x000f) + ((tmp >> 8) & 0x000f);
    auto x = _mm_srli_epi16(k, 15);
    x = _mm_add_epi16(x, _mm_shuffle_epi32(x, _MM_SHUFFLE(0, 1, 2, 3)));
    x = _mm_add_epi16(x, _mm_shufflelo_epi16(x, _MM_SHUFFLE(0, 1, 2, 3)));
    x = _mm_add_epi16(x, _mm_shufflelo_epi16(x, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_extract_epi16(x, 0);
#endif
}

template<> Vc_ALWAYS_INLINE Vc_CONST int mask_count<16>(__m128i k)
{
    int tmp = _mm_movemask_epi8(k);
#ifdef VC_IMPL_POPCNT
    return _mm_popcnt_u32(tmp);
#else
    tmp = (tmp & 0x5555) + ((tmp >> 1) & 0x5555);
    tmp = (tmp & 0x3333) + ((tmp >> 2) & 0x3333);
    tmp = (tmp & 0x0f0f) + ((tmp >> 4) & 0x0f0f);
    return (tmp & 0x00ff) + ((tmp >> 8) & 0x00ff);
#endif
}
/*}}}*/
// mask_store/*{{{*/
template<size_t> Vc_ALWAYS_INLINE void mask_store(__m128i k, bool *mem);
template<> Vc_ALWAYS_INLINE void mask_store<4>(__m128i k, bool *mem)
{
    const auto k2 = _mm_srli_epi16(_mm_packs_epi16(k, _mm_setzero_si128()), 15);
    typedef int boolAlias Vc_MAY_ALIAS;
    *reinterpret_cast<boolAlias *>(mem) = _mm_cvtsi128_si32(_mm_packs_epi16(k2, _mm_setzero_si128()));
}
template<> Vc_ALWAYS_INLINE void mask_store<8>(__m128i k, bool *mem)
{
    k = _mm_srli_epi16(k, 15);
    typedef int64_t boolAlias Vc_MAY_ALIAS;
    const auto k2 = _mm_packs_epi16(k, _mm_setzero_si128());
#ifdef __x86_64__
    *reinterpret_cast<boolAlias *>(mem) = _mm_cvtsi128_si64(k2);
#else
    *reinterpret_cast<boolAlias *>(mem) = _mm_cvtsi128_si32(k2);
    *reinterpret_cast<boolAlias *>(mem + 4) = extract_epi32<1>(k2);
#endif
}
/*}}}*/
// mask_load/*{{{*/
template<size_t> Vc_ALWAYS_INLINE __m128 mask_load(const bool *mem);
template<> Vc_ALWAYS_INLINE __m128 mask_load<8>(const bool *mem)
{
    __m128i k = _mm_cvtsi64_si128(*reinterpret_cast<const int64_t *>(mem));
    return sse_cast<__m128>(_mm_cmpgt_epi16(_mm_unpacklo_epi8(k, k), _mm_setzero_si128()));
}
template<> Vc_ALWAYS_INLINE __m128 mask_load<4>(const bool *mem)
{
    __m128i k = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(mem));
    k = _mm_cmpgt_epi16(_mm_unpacklo_epi8(k, k), _mm_setzero_si128());
    return sse_cast<__m128>(_mm_unpacklo_epi16(k, k));
}
/*}}}*/

} // namespace internal

template<> Vc_ALWAYS_INLINE void Mask<double>::store(bool *mem) const
{
    typedef uint16_t boolAlias Vc_MAY_ALIAS;
    boolAlias *ptr = reinterpret_cast<boolAlias *>(mem);
    *ptr = _mm_movemask_epi8(dataI()) & 0x0101;
}
template<typename T> Vc_ALWAYS_INLINE void Mask<T>::store(bool *mem) const
{
    internal::mask_store<Size>(dataI(), mem);
}
template<> Vc_ALWAYS_INLINE void Mask<double>::load(const bool *mem)
{
    d.set(0, MaskBool(mem[0]));
    d.set(1, MaskBool(mem[1]));
}
template<typename T> Vc_ALWAYS_INLINE void Mask<T>::load(const bool *mem)
{
    d.v() = internal::mask_load<Size>(mem);
}

template<> Vc_ALWAYS_INLINE Vc_PURE bool Mask< int16_t>::operator[](size_t index) const { return shiftMask() & (1 << 2 * index); }
template<> Vc_ALWAYS_INLINE Vc_PURE bool Mask<uint16_t>::operator[](size_t index) const { return shiftMask() & (1 << 2 * index); }
template<typename T> Vc_ALWAYS_INLINE Vc_PURE int Mask<T>::firstOne() const
{
    const int mask = toInt();
#ifdef _MSC_VER
    unsigned long bit;
    _BitScanForward(&bit, mask);
#else
    int bit;
    __asm__("bsf %1,%0" : "=&r"(bit) : "r"(mask));
#endif
    return bit;
}
/*operators{{{*/
/*}}}*/

template <typename M, typename G>
Vc_INTRINSIC M generate_impl(G &&gen, std::integral_constant<int, 2>)
{
    return _mm_set_epi64x(gen(1) ? 0xffffffffffffffffull : 0,
                          gen(0) ? 0xffffffffffffffffull : 0);
}
template <typename M, typename G>
Vc_INTRINSIC M generate_impl(G &&gen, std::integral_constant<int, 4>)
{
    return _mm_setr_epi32(gen(0) ? 0xfffffffful : 0, gen(1) ? 0xfffffffful : 0,
                          gen(2) ? 0xfffffffful : 0, gen(3) ? 0xfffffffful : 0);
}
template <typename M, typename G>
Vc_INTRINSIC M generate_impl(G &&gen, std::integral_constant<int, 8>)
{
    return _mm_setr_epi16(gen(0) ? 0xffffu : 0, gen(1) ? 0xffffu : 0,
                          gen(2) ? 0xffffu : 0, gen(3) ? 0xffffu : 0,
                          gen(4) ? 0xffffu : 0, gen(5) ? 0xffffu : 0,
                          gen(6) ? 0xffffu : 0, gen(7) ? 0xffffu : 0);
}
template <typename T>
template <typename G>
Vc_INTRINSIC Mask<T> Mask<T>::generate(G &&gen)
{
    return generate_impl<Mask<T>>(std::forward<G>(gen),
                                  std::integral_constant<int, Size>());
}
// shifted {{{1
template <int amount, typename T>
Vc_INTRINSIC Vc_PURE enable_if<(amount > 0), T> shifted_impl(T k)
{
    return _mm_srli_si128(k, amount);
}
template <int amount, typename T>
Vc_INTRINSIC Vc_PURE enable_if<(amount < 0), T> shifted_impl(T k)
{
    return _mm_slli_si128(k, -amount);
}
template <typename T> Vc_INTRINSIC Vc_PURE Mask<T> Mask<T>::shifted(int amount) const
{
    switch (amount * int(sizeof(VectorEntryType))) {
    case   0: return *this;
    case   1: return shifted_impl<  1>(dataI());
    case   2: return shifted_impl<  2>(dataI());
    case   3: return shifted_impl<  3>(dataI());
    case   4: return shifted_impl<  4>(dataI());
    case   5: return shifted_impl<  5>(dataI());
    case   6: return shifted_impl<  6>(dataI());
    case   7: return shifted_impl<  7>(dataI());
    case   8: return shifted_impl<  8>(dataI());
    case   9: return shifted_impl<  9>(dataI());
    case  10: return shifted_impl< 10>(dataI());
    case  11: return shifted_impl< 11>(dataI());
    case  12: return shifted_impl< 12>(dataI());
    case  13: return shifted_impl< 13>(dataI());
    case  14: return shifted_impl< 14>(dataI());
    case  15: return shifted_impl< 15>(dataI());
    case  16: return shifted_impl< 16>(dataI());
    case  -1: return shifted_impl< -1>(dataI());
    case  -2: return shifted_impl< -2>(dataI());
    case  -3: return shifted_impl< -3>(dataI());
    case  -4: return shifted_impl< -4>(dataI());
    case  -5: return shifted_impl< -5>(dataI());
    case  -6: return shifted_impl< -6>(dataI());
    case  -7: return shifted_impl< -7>(dataI());
    case  -8: return shifted_impl< -8>(dataI());
    case  -9: return shifted_impl< -9>(dataI());
    case -10: return shifted_impl<-10>(dataI());
    case -11: return shifted_impl<-11>(dataI());
    case -12: return shifted_impl<-12>(dataI());
    case -13: return shifted_impl<-13>(dataI());
    case -14: return shifted_impl<-14>(dataI());
    case -15: return shifted_impl<-15>(dataI());
    case -16: return shifted_impl<-16>(dataI());
    }
    return Zero();
}
// }}}1

}
}

#include "undomacros.h"

// vim: foldmethod=marker
