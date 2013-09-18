/*  This file is part of the Vc library. {{{

    Copyright (C) 2012 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_SSE_INTERLEAVEDMEMORY_TCC
#define VC_SSE_INTERLEAVEDMEMORY_TCC

#include "macros.h"

Vc_NAMESPACE_BEGIN(Common)

namespace
{
template<typename V, int Size> struct InterleaveImpl;
template<typename V> struct InterleaveImpl<V, 8> {
    template<typename I> static inline void interleave(typename V::EntryType *const data, const I &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1)
    {
        const __m128i tmp0 = _mm_unpacklo_epi16(v0.data(), v1.data());
        const __m128i tmp1 = _mm_unpackhi_epi16(v0.data(), v1.data());
#ifdef __x86_64__
        const long long tmp00 = _mm_cvtsi128_si64(tmp0);
        const long long tmp01 = _mm_cvtsi128_si64(_mm_unpackhi_epi64(tmp0, tmp0));
        const long long tmp10 = _mm_cvtsi128_si64(tmp1);
        const long long tmp11 = _mm_cvtsi128_si64(_mm_unpackhi_epi64(tmp1, tmp1));
        *reinterpret_cast<int *>(&data[i[0]]) = tmp00;
        *reinterpret_cast<int *>(&data[i[1]]) = tmp00 >> 32;
        *reinterpret_cast<int *>(&data[i[2]]) = tmp01;
        *reinterpret_cast<int *>(&data[i[3]]) = tmp01 >> 32;
        *reinterpret_cast<int *>(&data[i[4]]) = tmp10;
        *reinterpret_cast<int *>(&data[i[5]]) = tmp10 >> 32;
        *reinterpret_cast<int *>(&data[i[6]]) = tmp11;
        *reinterpret_cast<int *>(&data[i[7]]) = tmp11 >> 32;
#elif defined(VC_IMPL_SSE4_1)
        *reinterpret_cast<int *>(&data[i[0]]) = _mm_cvtsi128_si32(tmp0);
        *reinterpret_cast<int *>(&data[i[1]]) = _mm_extract_epi32(tmp0, 1);
        *reinterpret_cast<int *>(&data[i[2]]) = _mm_extract_epi32(tmp0, 2);
        *reinterpret_cast<int *>(&data[i[3]]) = _mm_extract_epi32(tmp0, 3);
        *reinterpret_cast<int *>(&data[i[4]]) = _mm_cvtsi128_si32(tmp1);
        *reinterpret_cast<int *>(&data[i[5]]) = _mm_extract_epi32(tmp1, 1);
        *reinterpret_cast<int *>(&data[i[6]]) = _mm_extract_epi32(tmp1, 2);
        *reinterpret_cast<int *>(&data[i[7]]) = _mm_extract_epi32(tmp1, 3);
#else
        *reinterpret_cast<int *>(&data[i[0]]) = _mm_cvtsi128_si32(tmp0);
        *reinterpret_cast<int *>(&data[i[1]]) = _mm_cvtsi128_si32(_mm_srli_si128(tmp0, 4));
        *reinterpret_cast<int *>(&data[i[2]]) = _mm_cvtsi128_si32(_mm_srli_si128(tmp0, 8));
        *reinterpret_cast<int *>(&data[i[3]]) = _mm_cvtsi128_si32(_mm_srli_si128(tmp0, 12));
        *reinterpret_cast<int *>(&data[i[4]]) = _mm_cvtsi128_si32(tmp1);
        *reinterpret_cast<int *>(&data[i[5]]) = _mm_cvtsi128_si32(_mm_srli_si128(tmp1, 4));
        *reinterpret_cast<int *>(&data[i[6]]) = _mm_cvtsi128_si32(_mm_srli_si128(tmp1, 8));
        *reinterpret_cast<int *>(&data[i[7]]) = _mm_cvtsi128_si32(_mm_srli_si128(tmp1, 12));
#endif
    }/*}}}*/
    static inline void interleave(typename V::EntryType *const data, const SuccessiveEntries<2> &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1)
    {
        const __m128i tmp0 = _mm_unpacklo_epi16(v0.data(), v1.data());
        const __m128i tmp1 = _mm_unpackhi_epi16(v0.data(), v1.data());
        V(tmp0).store(&data[i[0]], Vc::Unaligned);
        V(tmp1).store(&data[i[4]], Vc::Unaligned);
    }/*}}}*/
    template<typename I> static inline void interleave(typename V::EntryType *const data, const I &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2)
    {
#ifdef VC_USE_MASKMOV_SCATTER
        const __m128i maskLo = _mm_set_epi16(0, 0, 0, 0, 0, -1, -1, -1);
        const __m128i maskHi = _mm_set_epi16(0, -1, -1, -1, 0, 0, 0, 0);
        typename V::EntryType *const dataHi = data - 4;
        const __m128i tmp0 = _mm_unpacklo_epi16(v0.data(), v2.data());
        const __m128i tmp1 = _mm_unpackhi_epi16(v0.data(), v2.data());
        const __m128i tmp2 = _mm_unpacklo_epi16(v1.data(), v1.data());
        const __m128i tmp3 = _mm_unpackhi_epi16(v1.data(), v1.data());

        const __m128i tmp4 = _mm_unpacklo_epi16(tmp0, tmp2);
        const __m128i tmp5 = _mm_unpackhi_epi16(tmp0, tmp2);
        const __m128i tmp6 = _mm_unpacklo_epi16(tmp1, tmp3);
        const __m128i tmp7 = _mm_unpackhi_epi16(tmp1, tmp3);
        _mm_maskmoveu_si128(tmp4, maskLo, reinterpret_cast<char *>(&data[i[0]]));
        _mm_maskmoveu_si128(tmp4, maskHi, reinterpret_cast<char *>(&dataHi[i[1]]));
        _mm_maskmoveu_si128(tmp5, maskLo, reinterpret_cast<char *>(&data[i[2]]));
        _mm_maskmoveu_si128(tmp5, maskHi, reinterpret_cast<char *>(&dataHi[i[3]]));
        _mm_maskmoveu_si128(tmp6, maskLo, reinterpret_cast<char *>(&data[i[4]]));
        _mm_maskmoveu_si128(tmp6, maskHi, reinterpret_cast<char *>(&dataHi[i[5]]));
        _mm_maskmoveu_si128(tmp7, maskLo, reinterpret_cast<char *>(&data[i[6]]));
        _mm_maskmoveu_si128(tmp7, maskHi, reinterpret_cast<char *>(&dataHi[i[7]]));
#else
        interleave(data, i, v0, v1);
        v2.scatter(data + 2, i);
#endif
    }/*}}}*/
    template<typename I> static inline void interleave(typename V::EntryType *const data, const I &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1,
            const typename V::AsArg v2, const typename V::AsArg v3)
    {
        const __m128i tmp0 = _mm_unpacklo_epi16(v0.data(), v2.data());
        const __m128i tmp1 = _mm_unpackhi_epi16(v0.data(), v2.data());
        const __m128i tmp2 = _mm_unpacklo_epi16(v1.data(), v3.data());
        const __m128i tmp3 = _mm_unpackhi_epi16(v1.data(), v3.data());

        const __m128i tmp4 = _mm_unpacklo_epi16(tmp0, tmp2);
        const __m128i tmp5 = _mm_unpackhi_epi16(tmp0, tmp2);
        const __m128i tmp6 = _mm_unpacklo_epi16(tmp1, tmp3);
        const __m128i tmp7 = _mm_unpackhi_epi16(tmp1, tmp3);

        _mm_storel_epi64(reinterpret_cast<__m128i *>(&data[i[0]]), tmp4);
        _mm_storel_epi64(reinterpret_cast<__m128i *>(&data[i[2]]), tmp5);
        _mm_storel_epi64(reinterpret_cast<__m128i *>(&data[i[4]]), tmp6);
        _mm_storel_epi64(reinterpret_cast<__m128i *>(&data[i[6]]), tmp7);
        _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[1]]), _mm_castsi128_ps(tmp4));
        _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[3]]), _mm_castsi128_ps(tmp5));
        _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[5]]), _mm_castsi128_ps(tmp6));
        _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[7]]), _mm_castsi128_ps(tmp7));
    }/*}}}*/
    static inline void interleave(typename V::EntryType *const data, const SuccessiveEntries<4> &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1,
            const typename V::AsArg v2, const typename V::AsArg v3)
    {
        const __m128i tmp0 = _mm_unpacklo_epi16(v0.data(), v2.data());
        const __m128i tmp1 = _mm_unpackhi_epi16(v0.data(), v2.data());
        const __m128i tmp2 = _mm_unpacklo_epi16(v1.data(), v3.data());
        const __m128i tmp3 = _mm_unpackhi_epi16(v1.data(), v3.data());

        const __m128i tmp4 = _mm_unpacklo_epi16(tmp0, tmp2);
        const __m128i tmp5 = _mm_unpackhi_epi16(tmp0, tmp2);
        const __m128i tmp6 = _mm_unpacklo_epi16(tmp1, tmp3);
        const __m128i tmp7 = _mm_unpackhi_epi16(tmp1, tmp3);

        V(tmp4).store(&data[i[0]], ::Vc::Unaligned);
        V(tmp5).store(&data[i[2]], ::Vc::Unaligned);
        V(tmp6).store(&data[i[4]], ::Vc::Unaligned);
        V(tmp7).store(&data[i[6]], ::Vc::Unaligned);
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1)
    {
        const __m128i a = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(&data[i[0]]));
        const __m128i b = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(&data[i[1]]));
        const __m128i c = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(&data[i[2]]));
        const __m128i d = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(&data[i[3]]));
        const __m128i e = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(&data[i[4]]));
        const __m128i f = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(&data[i[5]]));
        const __m128i g = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(&data[i[6]]));
        const __m128i h = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(&data[i[7]]));

        const __m128i tmp2  = _mm_unpacklo_epi16(a, e); // a0 a4 b0 b4 c0 c4 d0 d4
        const __m128i tmp3  = _mm_unpacklo_epi16(c, g); // a2 a6 b2 b6 c2 c6 d2 d6
        const __m128i tmp4  = _mm_unpacklo_epi16(b, f); // a1 a5 b1 b5 c1 c5 d1 d5
        const __m128i tmp5  = _mm_unpacklo_epi16(d, h); // a3 a7 b3 b7 c3 c7 d3 d7

        const __m128i tmp0  = _mm_unpacklo_epi16(tmp2, tmp3); // a0 a2 a4 a6 b0 b2 b4 b6
        const __m128i tmp1  = _mm_unpacklo_epi16(tmp4, tmp5); // a1 a3 a5 a7 b1 b3 b5 b7

        v0.data() = _mm_unpacklo_epi16(tmp0, tmp1);
        v1.data() = _mm_unpackhi_epi16(tmp0, tmp1);
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1, V &v2)
    {
        const __m128i a = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&data[i[0]]));
        const __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&data[i[1]]));
        const __m128i c = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&data[i[2]]));
        const __m128i d = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&data[i[3]]));
        const __m128i e = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&data[i[4]]));
        const __m128i f = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&data[i[5]]));
        const __m128i g = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&data[i[6]]));
        const __m128i h = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&data[i[7]]));

        const __m128i tmp2  = _mm_unpacklo_epi16(a, e); // a0 a4 b0 b4 c0 c4 d0 d4
        const __m128i tmp4  = _mm_unpacklo_epi16(b, f); // a1 a5 b1 b5 c1 c5 d1 d5
        const __m128i tmp3  = _mm_unpacklo_epi16(c, g); // a2 a6 b2 b6 c2 c6 d2 d6
        const __m128i tmp5  = _mm_unpacklo_epi16(d, h); // a3 a7 b3 b7 c3 c7 d3 d7

        const __m128i tmp0  = _mm_unpacklo_epi16(tmp2, tmp3); // a0 a2 a4 a6 b0 b2 b4 b6
        const __m128i tmp1  = _mm_unpacklo_epi16(tmp4, tmp5); // a1 a3 a5 a7 b1 b3 b5 b7
        const __m128i tmp6  = _mm_unpackhi_epi16(tmp2, tmp3); // c0 c2 c4 c6 d0 d2 d4 d6
        const __m128i tmp7  = _mm_unpackhi_epi16(tmp4, tmp5); // c1 c3 c5 c7 d1 d3 d5 d7

        v0.data() = _mm_unpacklo_epi16(tmp0, tmp1);
        v1.data() = _mm_unpackhi_epi16(tmp0, tmp1);
        v2.data() = _mm_unpacklo_epi16(tmp6, tmp7);
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1, V &v2, V &v3)
    {
        const __m128i a = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&data[i[0]]));
        const __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&data[i[1]]));
        const __m128i c = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&data[i[2]]));
        const __m128i d = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&data[i[3]]));
        const __m128i e = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&data[i[4]]));
        const __m128i f = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&data[i[5]]));
        const __m128i g = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&data[i[6]]));
        const __m128i h = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&data[i[7]]));

        const __m128i tmp2  = _mm_unpacklo_epi16(a, e); // a0 a4 b0 b4 c0 c4 d0 d4
        const __m128i tmp4  = _mm_unpacklo_epi16(b, f); // a1 a5 b1 b5 c1 c5 d1 d5
        const __m128i tmp3  = _mm_unpacklo_epi16(c, g); // a2 a6 b2 b6 c2 c6 d2 d6
        const __m128i tmp5  = _mm_unpacklo_epi16(d, h); // a3 a7 b3 b7 c3 c7 d3 d7

        const __m128i tmp0  = _mm_unpacklo_epi16(tmp2, tmp3); // a0 a2 a4 a6 b0 b2 b4 b6
        const __m128i tmp1  = _mm_unpacklo_epi16(tmp4, tmp5); // a1 a3 a5 a7 b1 b3 b5 b7
        const __m128i tmp6  = _mm_unpackhi_epi16(tmp2, tmp3); // c0 c2 c4 c6 d0 d2 d4 d6
        const __m128i tmp7  = _mm_unpackhi_epi16(tmp4, tmp5); // c1 c3 c5 c7 d1 d3 d5 d7

        v0.data() = _mm_unpacklo_epi16(tmp0, tmp1);
        v1.data() = _mm_unpackhi_epi16(tmp0, tmp1);
        v2.data() = _mm_unpacklo_epi16(tmp6, tmp7);
        v3.data() = _mm_unpackhi_epi16(tmp6, tmp7);
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1, V &v2, V &v3, V &v4)
    {
        const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[0]]));
        const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[1]]));
        const __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[2]]));
        const __m128i d = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[3]]));
        const __m128i e = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[4]]));
        const __m128i f = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[5]]));
        const __m128i g = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[6]]));
        const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[7]]));

        const __m128i tmp2  = _mm_unpacklo_epi16(a, e); // a0 a4 b0 b4 c0 c4 d0 d4
        const __m128i tmp4  = _mm_unpacklo_epi16(b, f); // a1 a5 b1 b5 c1 c5 d1 d5
        const __m128i tmp3  = _mm_unpacklo_epi16(c, g); // a2 a6 b2 b6 c2 c6 d2 d6
        const __m128i tmp5  = _mm_unpacklo_epi16(d, h); // a3 a7 b3 b7 c3 c7 d3 d7
        const __m128i tmp10 = _mm_unpackhi_epi16(a, e); // e0 e4 f0 f4 g0 g4 h0 h4
        const __m128i tmp11 = _mm_unpackhi_epi16(c, g); // e1 e5 f1 f5 g1 g5 h1 h5
        const __m128i tmp12 = _mm_unpackhi_epi16(b, f); // e2 e6 f2 f6 g2 g6 h2 h6
        const __m128i tmp13 = _mm_unpackhi_epi16(d, h); // e3 e7 f3 f7 g3 g7 h3 h7

        const __m128i tmp0  = _mm_unpacklo_epi16(tmp2, tmp3); // a0 a2 a4 a6 b0 b2 b4 b6
        const __m128i tmp1  = _mm_unpacklo_epi16(tmp4, tmp5); // a1 a3 a5 a7 b1 b3 b5 b7
        const __m128i tmp6  = _mm_unpackhi_epi16(tmp2, tmp3); // c0 c2 c4 c6 d0 d2 d4 d6
        const __m128i tmp7  = _mm_unpackhi_epi16(tmp4, tmp5); // c1 c3 c5 c7 d1 d3 d5 d7
        const __m128i tmp8  = _mm_unpacklo_epi16(tmp10, tmp11); // e0 e2 e4 e6 f0 f2 f4 f6
        const __m128i tmp9  = _mm_unpacklo_epi16(tmp12, tmp13); // e1 e3 e5 e7 f1 f3 f5 f7

        v0.data() = _mm_unpacklo_epi16(tmp0, tmp1);
        v1.data() = _mm_unpackhi_epi16(tmp0, tmp1);
        v2.data() = _mm_unpacklo_epi16(tmp6, tmp7);
        v3.data() = _mm_unpackhi_epi16(tmp6, tmp7);
        v4.data() = _mm_unpacklo_epi16(tmp8, tmp9);
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1, V &v2, V &v3, V &v4, V &v5)
    {
        const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[0]]));
        const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[1]]));
        const __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[2]]));
        const __m128i d = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[3]]));
        const __m128i e = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[4]]));
        const __m128i f = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[5]]));
        const __m128i g = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[6]]));
        const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[7]]));

        const __m128i tmp2  = _mm_unpacklo_epi16(a, e); // a0 a4 b0 b4 c0 c4 d0 d4
        const __m128i tmp4  = _mm_unpacklo_epi16(b, f); // a1 a5 b1 b5 c1 c5 d1 d5
        const __m128i tmp3  = _mm_unpacklo_epi16(c, g); // a2 a6 b2 b6 c2 c6 d2 d6
        const __m128i tmp5  = _mm_unpacklo_epi16(d, h); // a3 a7 b3 b7 c3 c7 d3 d7
        const __m128i tmp10 = _mm_unpackhi_epi16(a, e); // e0 e4 f0 f4 g0 g4 h0 h4
        const __m128i tmp11 = _mm_unpackhi_epi16(c, g); // e1 e5 f1 f5 g1 g5 h1 h5
        const __m128i tmp12 = _mm_unpackhi_epi16(b, f); // e2 e6 f2 f6 g2 g6 h2 h6
        const __m128i tmp13 = _mm_unpackhi_epi16(d, h); // e3 e7 f3 f7 g3 g7 h3 h7

        const __m128i tmp0  = _mm_unpacklo_epi16(tmp2, tmp3); // a0 a2 a4 a6 b0 b2 b4 b6
        const __m128i tmp1  = _mm_unpacklo_epi16(tmp4, tmp5); // a1 a3 a5 a7 b1 b3 b5 b7
        const __m128i tmp6  = _mm_unpackhi_epi16(tmp2, tmp3); // c0 c2 c4 c6 d0 d2 d4 d6
        const __m128i tmp7  = _mm_unpackhi_epi16(tmp4, tmp5); // c1 c3 c5 c7 d1 d3 d5 d7
        const __m128i tmp8  = _mm_unpacklo_epi16(tmp10, tmp11); // e0 e2 e4 e6 f0 f2 f4 f6
        const __m128i tmp9  = _mm_unpacklo_epi16(tmp12, tmp13); // e1 e3 e5 e7 f1 f3 f5 f7

        v0.data() = _mm_unpacklo_epi16(tmp0, tmp1);
        v1.data() = _mm_unpackhi_epi16(tmp0, tmp1);
        v2.data() = _mm_unpacklo_epi16(tmp6, tmp7);
        v3.data() = _mm_unpackhi_epi16(tmp6, tmp7);
        v4.data() = _mm_unpacklo_epi16(tmp8, tmp9);
        v5.data() = _mm_unpackhi_epi16(tmp8, tmp9);
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1, V &v2, V &v3, V &v4, V &v5, V &v6)
    {
        const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[0]]));
        const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[1]]));
        const __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[2]]));
        const __m128i d = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[3]]));
        const __m128i e = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[4]]));
        const __m128i f = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[5]]));
        const __m128i g = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[6]]));
        const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[7]]));

        const __m128i tmp2  = _mm_unpacklo_epi16(a, e); // a0 a4 b0 b4 c0 c4 d0 d4
        const __m128i tmp4  = _mm_unpacklo_epi16(b, f); // a1 a5 b1 b5 c1 c5 d1 d5
        const __m128i tmp3  = _mm_unpacklo_epi16(c, g); // a2 a6 b2 b6 c2 c6 d2 d6
        const __m128i tmp5  = _mm_unpacklo_epi16(d, h); // a3 a7 b3 b7 c3 c7 d3 d7
        const __m128i tmp10 = _mm_unpackhi_epi16(a, e); // e0 e4 f0 f4 g0 g4 h0 h4
        const __m128i tmp11 = _mm_unpackhi_epi16(c, g); // e1 e5 f1 f5 g1 g5 h1 h5
        const __m128i tmp12 = _mm_unpackhi_epi16(b, f); // e2 e6 f2 f6 g2 g6 h2 h6
        const __m128i tmp13 = _mm_unpackhi_epi16(d, h); // e3 e7 f3 f7 g3 g7 h3 h7

        const __m128i tmp0  = _mm_unpacklo_epi16(tmp2, tmp3); // a0 a2 a4 a6 b0 b2 b4 b6
        const __m128i tmp1  = _mm_unpacklo_epi16(tmp4, tmp5); // a1 a3 a5 a7 b1 b3 b5 b7
        const __m128i tmp6  = _mm_unpackhi_epi16(tmp2, tmp3); // c0 c2 c4 c6 d0 d2 d4 d6
        const __m128i tmp7  = _mm_unpackhi_epi16(tmp4, tmp5); // c1 c3 c5 c7 d1 d3 d5 d7
        const __m128i tmp8  = _mm_unpacklo_epi16(tmp10, tmp11); // e0 e2 e4 e6 f0 f2 f4 f6
        const __m128i tmp9  = _mm_unpacklo_epi16(tmp12, tmp13); // e1 e3 e5 e7 f1 f3 f5 f7
        const __m128i tmp14 = _mm_unpackhi_epi16(tmp10, tmp11); // g0 g2 g4 g6 h0 h2 h4 h6
        const __m128i tmp15 = _mm_unpackhi_epi16(tmp12, tmp13); // g1 g3 g5 g7 h1 h3 h5 h7

        v0.data() = _mm_unpacklo_epi16(tmp0, tmp1);
        v1.data() = _mm_unpackhi_epi16(tmp0, tmp1);
        v2.data() = _mm_unpacklo_epi16(tmp6, tmp7);
        v3.data() = _mm_unpackhi_epi16(tmp6, tmp7);
        v4.data() = _mm_unpacklo_epi16(tmp8, tmp9);
        v5.data() = _mm_unpackhi_epi16(tmp8, tmp9);
        v6.data() = _mm_unpacklo_epi16(tmp14, tmp15);
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1, V &v2, V &v3, V &v4, V &v5, V &v6, V &v7)
    {
        const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[0]]));
        const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[1]]));
        const __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[2]]));
        const __m128i d = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[3]]));
        const __m128i e = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[4]]));
        const __m128i f = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[5]]));
        const __m128i g = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[6]]));
        const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[7]]));

        const __m128i tmp2  = _mm_unpacklo_epi16(a, e); // a0 a4 b0 b4 c0 c4 d0 d4
        const __m128i tmp4  = _mm_unpacklo_epi16(b, f); // a1 a5 b1 b5 c1 c5 d1 d5
        const __m128i tmp3  = _mm_unpacklo_epi16(c, g); // a2 a6 b2 b6 c2 c6 d2 d6
        const __m128i tmp5  = _mm_unpacklo_epi16(d, h); // a3 a7 b3 b7 c3 c7 d3 d7
        const __m128i tmp10 = _mm_unpackhi_epi16(a, e); // e0 e4 f0 f4 g0 g4 h0 h4
        const __m128i tmp11 = _mm_unpackhi_epi16(c, g); // e1 e5 f1 f5 g1 g5 h1 h5
        const __m128i tmp12 = _mm_unpackhi_epi16(b, f); // e2 e6 f2 f6 g2 g6 h2 h6
        const __m128i tmp13 = _mm_unpackhi_epi16(d, h); // e3 e7 f3 f7 g3 g7 h3 h7

        const __m128i tmp0  = _mm_unpacklo_epi16(tmp2, tmp3); // a0 a2 a4 a6 b0 b2 b4 b6
        const __m128i tmp1  = _mm_unpacklo_epi16(tmp4, tmp5); // a1 a3 a5 a7 b1 b3 b5 b7
        const __m128i tmp6  = _mm_unpackhi_epi16(tmp2, tmp3); // c0 c2 c4 c6 d0 d2 d4 d6
        const __m128i tmp7  = _mm_unpackhi_epi16(tmp4, tmp5); // c1 c3 c5 c7 d1 d3 d5 d7
        const __m128i tmp8  = _mm_unpacklo_epi16(tmp10, tmp11); // e0 e2 e4 e6 f0 f2 f4 f6
        const __m128i tmp9  = _mm_unpacklo_epi16(tmp12, tmp13); // e1 e3 e5 e7 f1 f3 f5 f7
        const __m128i tmp14 = _mm_unpackhi_epi16(tmp10, tmp11); // g0 g2 g4 g6 h0 h2 h4 h6
        const __m128i tmp15 = _mm_unpackhi_epi16(tmp12, tmp13); // g1 g3 g5 g7 h1 h3 h5 h7

        v0.data() = _mm_unpacklo_epi16(tmp0, tmp1);
        v1.data() = _mm_unpackhi_epi16(tmp0, tmp1);
        v2.data() = _mm_unpacklo_epi16(tmp6, tmp7);
        v3.data() = _mm_unpackhi_epi16(tmp6, tmp7);
        v4.data() = _mm_unpacklo_epi16(tmp8, tmp9);
        v5.data() = _mm_unpackhi_epi16(tmp8, tmp9);
        v6.data() = _mm_unpacklo_epi16(tmp14, tmp15);
        v7.data() = _mm_unpackhi_epi16(tmp14, tmp15);
    }/*}}}*/
};
template<typename V> struct InterleaveImpl<V, 4> {
    template<typename I> static inline void interleave(typename V::EntryType *const data, const I &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1)
    {
        const __m128 tmp0 = _mm_unpacklo_ps(SSE::sse_cast<__m128>(v0.data()),SSE::sse_cast<__m128>(v1.data()));
        const __m128 tmp1 = _mm_unpackhi_ps(SSE::sse_cast<__m128>(v0.data()),SSE::sse_cast<__m128>(v1.data()));
        _mm_storel_pi(reinterpret_cast<__m64 *>(&data[i[0]]), tmp0);
        _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[1]]), tmp0);
        _mm_storel_pi(reinterpret_cast<__m64 *>(&data[i[2]]), tmp1);
        _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[3]]), tmp1);
    }/*}}}*/
    static inline void interleave(typename V::EntryType *const data, const SuccessiveEntries<2> &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1)
    {
        const __m128 tmp0 = _mm_unpacklo_ps(SSE::sse_cast<__m128>(v0.data()),SSE::sse_cast<__m128>(v1.data()));
        const __m128 tmp1 = _mm_unpackhi_ps(SSE::sse_cast<__m128>(v0.data()),SSE::sse_cast<__m128>(v1.data()));
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[0]]), tmp0);
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[2]]), tmp1);
    }/*}}}*/
    template<typename I> static inline void interleave(typename V::EntryType *const data, const I &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2)
    {
#ifdef VC_USE_MASKMOV_SCATTER
        const __m128 tmp0 = _mm_unpacklo_ps(SSE::sse_cast<__m128>(v0.data()), SSE::sse_cast<__m128>(v1.data()));
        const __m128 tmp1 = _mm_unpackhi_ps(SSE::sse_cast<__m128>(v0.data()), SSE::sse_cast<__m128>(v1.data()));
        const __m128 tmp2 = _mm_unpacklo_ps(SSE::sse_cast<__m128>(v2.data()), SSE::sse_cast<__m128>(v2.data()));
        const __m128 tmp3 = _mm_unpackhi_ps(SSE::sse_cast<__m128>(v2.data()), SSE::sse_cast<__m128>(v2.data()));
        const __m128i mask = _mm_set_epi32(0, -1, -1, -1);
        _mm_maskmoveu_si128(_mm_castps_si128(_mm_movelh_ps(tmp0, tmp2)), mask, reinterpret_cast<char *>(&data[i[0]]));
        _mm_maskmoveu_si128(_mm_castps_si128(_mm_movehl_ps(tmp2, tmp0)), mask, reinterpret_cast<char *>(&data[i[1]]));
        _mm_maskmoveu_si128(_mm_castps_si128(_mm_movelh_ps(tmp1, tmp3)), mask, reinterpret_cast<char *>(&data[i[2]]));
        _mm_maskmoveu_si128(_mm_castps_si128(_mm_movehl_ps(tmp3, tmp1)), mask, reinterpret_cast<char *>(&data[i[3]]));
#else
        const __m128 tmp0 = _mm_unpacklo_ps(SSE::sse_cast<__m128>(v0.data()),SSE::sse_cast<__m128>(v1.data()));
        const __m128 tmp1 = _mm_unpackhi_ps(SSE::sse_cast<__m128>(v0.data()),SSE::sse_cast<__m128>(v1.data()));
        _mm_storel_pi(reinterpret_cast<__m64 *>(&data[i[0]]), tmp0);
        _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[1]]), tmp0);
        _mm_storel_pi(reinterpret_cast<__m64 *>(&data[i[2]]), tmp1);
        _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[3]]), tmp1);
        v2.scatter(data + 2, i);
#endif
    }/*}}}*/
    template<typename I> static inline void interleave(typename V::EntryType *const data, const I &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1,
            const typename V::AsArg v2, const typename V::AsArg v3)
    {
        const __m128 tmp0 = _mm_unpacklo_ps(SSE::sse_cast<__m128>(v0.data()),SSE::sse_cast<__m128>(v1.data()));
        const __m128 tmp1 = _mm_unpackhi_ps(SSE::sse_cast<__m128>(v0.data()),SSE::sse_cast<__m128>(v1.data()));
        const __m128 tmp2 = _mm_unpacklo_ps(SSE::sse_cast<__m128>(v2.data()),SSE::sse_cast<__m128>(v3.data()));
        const __m128 tmp3 = _mm_unpackhi_ps(SSE::sse_cast<__m128>(v2.data()),SSE::sse_cast<__m128>(v3.data()));
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[0]]), _mm_movelh_ps(tmp0, tmp2));
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[1]]), _mm_movehl_ps(tmp2, tmp0));
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[2]]), _mm_movelh_ps(tmp1, tmp3));
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[3]]), _mm_movehl_ps(tmp3, tmp1));
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1)
    {
        const __m128 a = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(&data[i[0]])));
        const __m128 b = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(&data[i[1]])));
        const __m128 c = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(&data[i[2]])));
        const __m128 d = _mm_castpd_ps(_mm_load_sd(reinterpret_cast<const double *>(&data[i[3]])));

        const __m128 tmp0 = _mm_unpacklo_ps(a, b); // [a0 a1 b0 b1]
        const __m128 tmp1 = _mm_unpacklo_ps(c, d); // [a2 a3 b2 b3]

        v0.data() = SSE::sse_cast<typename V::VectorType>(_mm_movelh_ps(tmp0, tmp1));
        v1.data() = SSE::sse_cast<typename V::VectorType>(_mm_movehl_ps(tmp1, tmp0));
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1, V &v2)
    {
        const __m128 a = _mm_loadu_ps(reinterpret_cast<const float *>(&data[i[0]]));
        const __m128 b = _mm_loadu_ps(reinterpret_cast<const float *>(&data[i[1]]));
        const __m128 c = _mm_loadu_ps(reinterpret_cast<const float *>(&data[i[2]]));
        const __m128 d = _mm_loadu_ps(reinterpret_cast<const float *>(&data[i[3]]));

        const __m128 tmp0 = _mm_unpacklo_ps(a, b); // [a0 a1 b0 b1]
        const __m128 tmp1 = _mm_unpacklo_ps(c, d); // [a2 a3 b2 b3]
        const __m128 tmp2 = _mm_unpackhi_ps(a, b); // [c0 c1 XX XX]
        const __m128 tmp3 = _mm_unpackhi_ps(c, d); // [c2 c3 XX XX]

        v0.data() = SSE::sse_cast<typename V::VectorType>(_mm_movelh_ps(tmp0, tmp1));
        v1.data() = SSE::sse_cast<typename V::VectorType>(_mm_movehl_ps(tmp1, tmp0));
        v2.data() = SSE::sse_cast<typename V::VectorType>(_mm_movelh_ps(tmp2, tmp3));
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1, V &v2, V &v3)
    {
        const __m128 a = _mm_loadu_ps(reinterpret_cast<const float *>(&data[i[0]]));
        const __m128 b = _mm_loadu_ps(reinterpret_cast<const float *>(&data[i[1]]));
        const __m128 c = _mm_loadu_ps(reinterpret_cast<const float *>(&data[i[2]]));
        const __m128 d = _mm_loadu_ps(reinterpret_cast<const float *>(&data[i[3]]));

        const __m128 tmp0 = _mm_unpacklo_ps(a, b); // [a0 a1 b0 b1]
        const __m128 tmp1 = _mm_unpacklo_ps(c, d); // [a2 a3 b2 b3]
        const __m128 tmp2 = _mm_unpackhi_ps(a, b); // [c0 c1 d0 d1]
        const __m128 tmp3 = _mm_unpackhi_ps(c, d); // [c2 c3 d2 d3]

        v0.data() = SSE::sse_cast<typename V::VectorType>(_mm_movelh_ps(tmp0, tmp1));
        v1.data() = SSE::sse_cast<typename V::VectorType>(_mm_movehl_ps(tmp1, tmp0));
        v2.data() = SSE::sse_cast<typename V::VectorType>(_mm_movelh_ps(tmp2, tmp3));
        v3.data() = SSE::sse_cast<typename V::VectorType>(_mm_movehl_ps(tmp3, tmp2));
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1, V &v2, V &v3, V &v4)
    {
        deinterleave(data, i, v0, v1, v2, v3);
        v4.gather(data + 4, i);
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1, V &v2, V &v3, V &v4, V &v5)
    {
        deinterleave(data, i, v0, v1, v2, v3);
        deinterleave(data + 4, i, v4, v5);
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1, V &v2, V &v3, V &v4, V &v5, V &v6)
    {
        deinterleave(data, i, v0, v1, v2, v3);
        deinterleave(data + 4, i, v4, v5, v6);
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1, V &v2, V &v3, V &v4, V &v5, V &v6, V &v7)
    {
        deinterleave(data, i, v0, v1, v2, v3);
        deinterleave(data + 4, i, v4, v5, v6, v7);
    }/*}}}*/
};
template<typename V> struct InterleaveImpl<V, 2> {
    template<typename I> static inline void interleave(typename V::EntryType *const data, const I &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1)
    {
        const __m128d tmp0 = _mm_unpacklo_pd(v0.data(), v1.data());
        const __m128d tmp1 = _mm_unpackhi_pd(v0.data(), v1.data());
        _mm_storeu_pd(&data[i[0]], tmp0);
        _mm_storeu_pd(&data[i[1]], tmp1);
    }/*}}}*/
    template<typename I> static inline void interleave(typename V::EntryType *const data, const I &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2)
    {
        interleave(data, i, v0, v1);
        v2.scatter(data + 2, i);
    }/*}}}*/
    template<typename I> static inline void interleave(typename V::EntryType *const data, const I &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1,
            const typename V::AsArg v2, const typename V::AsArg v3)
    {
        interleave(data, i, v0, v1);
        interleave(data + 2, i, v2, v3);
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1)
    {
        const __m128d a = _mm_loadu_pd(&data[i[0]]);
        const __m128d b = _mm_loadu_pd(&data[i[1]]);

        v0.data() = _mm_unpacklo_pd(a, b);
        v1.data() = _mm_unpackhi_pd(a, b);
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1, V &v2)
    {
        v2.gather(data + 2, i);
        deinterleave(data, i, v0, v1);
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1, V &v2, V &v3)
    {
        deinterleave(data, i, v0, v1);
        deinterleave(data + 2, i, v2, v3);
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1, V &v2, V &v3, V &v4)
    {
        deinterleave(data, i, v0, v1);
        deinterleave(data + 2, i, v2, v3);
        v4.gather(data + 4, i);
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1, V &v2, V &v3, V &v4, V &v5)
    {
        deinterleave(data, i, v0, v1);
        deinterleave(data + 2, i, v2, v3);
        deinterleave(data + 4, i, v4, v5);
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1, V &v2, V &v3, V &v4, V &v5, V &v6)
    {
        deinterleave(data, i, v0, v1);
        deinterleave(data + 2, i, v2, v3);
        deinterleave(data + 4, i, v4, v5);
        v6.gather(data + 6, i);
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1, V &v2, V &v3, V &v4, V &v5, V &v6, V &v7)
    {
        deinterleave(data, i, v0, v1);
        deinterleave(data + 2, i, v2, v3);
        deinterleave(data + 4, i, v4, v5);
        deinterleave(data + 6, i, v6, v7);
    }/*}}}*/
};
} // anonymous namespace

template<typename V, typename I> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I>::interleave(const typename V::AsArg v0,/*{{{*/
        const typename V::AsArg v1)
{
    InterleaveImpl<V, V::Size>::interleave(m_data, m_indexes, v0, v1);
}/*}}}*/
template<typename V, typename I> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I>::interleave(const typename V::AsArg v0,/*{{{*/
        const typename V::AsArg v1, const typename V::AsArg v2)
{
    InterleaveImpl<V, V::Size>::interleave(m_data, m_indexes, v0, v1, v2);
}/*}}}*/
template<typename V, typename I> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I>::interleave(const typename V::AsArg v0,/*{{{*/
        const typename V::AsArg v1, const typename V::AsArg v2, const typename V::AsArg v3)
{
    InterleaveImpl<V, V::Size>::interleave(m_data, m_indexes, v0, v1, v2, v3);
}/*}}}*/
template<typename V, typename I> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I>::interleave(const typename V::AsArg v0,/*{{{*/
        const typename V::AsArg v1, const typename V::AsArg v2, const typename V::AsArg v3, const typename V::AsArg v4)
{
    InterleaveImpl<V, V::Size>::interleave(m_data, m_indexes, v0, v1, v2, v3);
    v4.scatter(m_data + 4, m_indexes);
}/*}}}*/
template<typename V, typename I> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I>::interleave(const typename V::AsArg v0,/*{{{*/
        const typename V::AsArg v1, const typename V::AsArg v2, const typename V::AsArg v3, const typename V::AsArg v4,
        const typename V::AsArg v5)
{
    InterleaveImpl<V, V::Size>::interleave(m_data    , m_indexes, v0, v1, v2, v3);
    InterleaveImpl<V, V::Size>::interleave(m_data + 4, m_indexes, v4, v5);
}/*}}}*/
template<typename V, typename I> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I>::interleave(const typename V::AsArg v0,/*{{{*/
        const typename V::AsArg v1, const typename V::AsArg v2, const typename V::AsArg v3, const typename V::AsArg v4,
        const typename V::AsArg v5, const typename V::AsArg v6)
{
    InterleaveImpl<V, V::Size>::interleave(m_data + 0, m_indexes, v0, v1, v2, v3);
    InterleaveImpl<V, V::Size>::interleave(m_data + 4, m_indexes, v4, v5, v6);
}/*}}}*/
template<typename V, typename I> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I>::interleave(const typename V::AsArg v0,/*{{{*/
        const typename V::AsArg v1, const typename V::AsArg v2, const typename V::AsArg v3, const typename V::AsArg v4,
        const typename V::AsArg v5, const typename V::AsArg v6, const typename V::AsArg v7)
{
    InterleaveImpl<V, V::Size>::interleave(m_data + 0, m_indexes, v0, v1, v2, v3);
    InterleaveImpl<V, V::Size>::interleave(m_data + 4, m_indexes, v4, v5, v6, v7);
}/*}}}*/

template<typename V, typename I> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I>::deinterleave(V &v0, V &v1) const/*{{{*/
{
    InterleaveImpl<V, V::Size>::deinterleave(m_data, m_indexes, v0, v1);
}/*}}}*/
template<typename V, typename I> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I>::deinterleave(V &v0, V &v1, V &v2) const/*{{{*/
{
    InterleaveImpl<V, V::Size>::deinterleave(m_data, m_indexes, v0, v1, v2);
}/*}}}*/
template<typename V, typename I> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I>::deinterleave(V &v0, V &v1, V &v2, V &v3) const/*{{{*/
{
    InterleaveImpl<V, V::Size>::deinterleave(m_data, m_indexes, v0, v1, v2, v3);
}/*}}}*/
template<typename V, typename I> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I>::deinterleave(V &v0, V &v1, V &v2, V &v3, V &v4) const/*{{{*/
{
    InterleaveImpl<V, V::Size>::deinterleave(m_data, m_indexes, v0, v1, v2, v3, v4);
}/*}}}*/
template<typename V, typename I> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I>::deinterleave(V &v0, V &v1, V &v2, V &v3, V &v4, V &v5) const/*{{{*/
{
    InterleaveImpl<V, V::Size>::deinterleave(m_data, m_indexes, v0, v1, v2, v3, v4, v5);
}/*}}}*/
template<typename V, typename I> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I>::deinterleave(V &v0, V &v1, V &v2, V &v3, V &v4, V &v5, V &v6) const/*{{{*/
{
    InterleaveImpl<V, V::Size>::deinterleave(m_data, m_indexes, v0, v1, v2, v3, v4, v5, v6);
}/*}}}*/
template<typename V, typename I> Vc_ALWAYS_INLINE void InterleavedMemoryAccessBase<V, I>::deinterleave(V &v0, V &v1, V &v2, V &v3, V &v4, V &v5, V &v6, V &v7) const/*{{{*/
{
    InterleaveImpl<V, V::Size>::deinterleave(m_data, m_indexes, v0, v1, v2, v3, v4, v5, v6, v7);
}/*}}}*/

Vc_NAMESPACE_END

#include "undomacros.h"

#endif // VC_SSE_INTERLEAVEDMEMORY_TCC

// vim: foldmethod=marker
