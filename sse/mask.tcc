/*  This file is part of the Vc library. {{{

    Copyright (C) 2013 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with Vc.  If not, see <http://www.gnu.org/licenses/>.

}}}*/

#include "macros.h"

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)
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
    *reinterpret_cast<boolAlias *>(mem + 4) = _mm_extract_epi32(k2, 1);
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
    d.m(0) = mem[0];
    d.m(1) = mem[1];
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

Vc_NAMESPACE_END

#include "undomacros.h"

// vim: foldmethod=marker
