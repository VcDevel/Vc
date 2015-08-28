/*  This file is part of the Vc library. {{{
Copyright © 2012-2013 Matthias Kretz <kretz@kde.org>
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

#ifndef VC_AVX_INTERLEAVEDMEMORY_TCC
#define VC_AVX_INTERLEAVEDMEMORY_TCC

#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Common
{

namespace
{
    using namespace Vc::AVX;
template<typename V, int Size, size_t VSize> struct InterleaveImpl;
template<typename V> struct InterleaveImpl<V, 16, 32> {
    template<typename I> static inline void interleave(typename V::EntryType *const data, const I &i,/*{{{*/
            const typename V::AsArg v0, // a0 a1 a2 a3 a4 a5 a6 a7 | a8 a9 ...
            const typename V::AsArg v1) // b0 b1 b2 b3 b4 b5 b6 b7 | b8 b9 ...
    {
        const __m256i tmp0 = _mm256_unpacklo_epi16(v0.data(), v1.data()); // a0 b0 a1 b1 a2 b2 a3 b3 | a8 b8 a9 ...
        const __m256i tmp1 = _mm256_unpackhi_epi16(v0.data(), v1.data()); // a4 b4 a5 ...
        *reinterpret_cast<uint32_t *>(&data[i[ 0]]) = _mm_cvtsi128_si32(lo128(tmp0));
        *reinterpret_cast<uint32_t *>(&data[i[ 1]]) = _mm_extract_epi32(lo128(tmp0), 1);
        *reinterpret_cast<uint32_t *>(&data[i[ 2]]) = _mm_extract_epi32(lo128(tmp0), 2);
        *reinterpret_cast<uint32_t *>(&data[i[ 3]]) = _mm_extract_epi32(lo128(tmp0), 3);
        *reinterpret_cast<uint32_t *>(&data[i[ 4]]) = _mm_cvtsi128_si32(lo128(tmp1));
        *reinterpret_cast<uint32_t *>(&data[i[ 5]]) = _mm_extract_epi32(lo128(tmp1), 1);
        *reinterpret_cast<uint32_t *>(&data[i[ 6]]) = _mm_extract_epi32(lo128(tmp1), 2);
        *reinterpret_cast<uint32_t *>(&data[i[ 7]]) = _mm_extract_epi32(lo128(tmp1), 3);
        *reinterpret_cast<uint32_t *>(&data[i[ 8]]) = _mm_cvtsi128_si32(hi128(tmp0));
        *reinterpret_cast<uint32_t *>(&data[i[ 9]]) = _mm_extract_epi32(hi128(tmp0), 1);
        *reinterpret_cast<uint32_t *>(&data[i[10]]) = _mm_extract_epi32(hi128(tmp0), 2);
        *reinterpret_cast<uint32_t *>(&data[i[11]]) = _mm_extract_epi32(hi128(tmp0), 3);
        *reinterpret_cast<uint32_t *>(&data[i[12]]) = _mm_cvtsi128_si32(hi128(tmp1));
        *reinterpret_cast<uint32_t *>(&data[i[13]]) = _mm_extract_epi32(hi128(tmp1), 1);
        *reinterpret_cast<uint32_t *>(&data[i[14]]) = _mm_extract_epi32(hi128(tmp1), 2);
        *reinterpret_cast<uint32_t *>(&data[i[15]]) = _mm_extract_epi32(hi128(tmp1), 3);
    }/*}}}*/
    static inline void interleave(typename V::EntryType *const data, const SuccessiveEntries<2> &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1)
    {
        const __m256i tmp0 = _mm256_unpacklo_epi16(v0.data(), v1.data()); // a0 b0 a1 b1 a2 b2 a3 b3 | a8 b8 a9 ...
        const __m256i tmp1 = _mm256_unpackhi_epi16(v0.data(), v1.data()); // a4 b4 a5 ...
        V(Mem::shuffle128<X0, Y0>(tmp0, tmp1)).store(&data[i[0]], Vc::Unaligned);
        V(Mem::shuffle128<X1, Y1>(tmp0, tmp1)).store(&data[i[8]], Vc::Unaligned);
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
        const __m256i tmp0 = _mm256_unpacklo_epi16(v0.data(), v2.data()); // a0 c0 a1 c1 a2 c2 a3 c3 | a8 c8 a9 c9 ...
        const __m256i tmp1 = _mm256_unpackhi_epi16(v0.data(), v2.data()); // a4 c4 a5 c5 a6 c6 a7 c7 | a12 c12 ...
        const __m256i tmp2 = _mm256_unpacklo_epi16(v1.data(), v3.data()); // b0 d0 b1 d1 b2 d2 b3 d3 | b8 d8 b9 d9 ...
        const __m256i tmp3 = _mm256_unpackhi_epi16(v1.data(), v3.data()); // b4 d4 b5 ...

        const __m256i tmp4 = _mm256_unpacklo_epi16(tmp0, tmp2); // a0 b0 c0 d0 a1 b1 c1 d1 | a8 b8 c8 d8 a9 b9 ...
        const __m256i tmp5 = _mm256_unpackhi_epi16(tmp0, tmp2); // [abcd]2 [abcd]3 | [abcd]10 [abcd]11
        const __m256i tmp6 = _mm256_unpacklo_epi16(tmp1, tmp3); // [abcd]4 [abcd]5 | [abcd]12 [abcd]13
        const __m256i tmp7 = _mm256_unpackhi_epi16(tmp1, tmp3); // [abcd]6 [abcd]7 | [abcd]14 [abcd]15

        auto &&store = [&](__m256i x, int offset) {
            _mm_storel_epi64(reinterpret_cast<__m128i *>(&data[i[offset + 0]]), lo128(x));
            _mm_storel_epi64(reinterpret_cast<__m128i *>(&data[i[offset + 8]]), hi128(x));
            _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[offset + 1]]), avx_cast<__m128>(x));
            _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[offset + 9]]), avx_cast<__m128>(hi128(x)));
        };
        store(tmp4, 0);
        store(tmp5, 2);
        store(tmp6, 4);
        store(tmp7, 6);
    }/*}}}*/
    static inline void interleave(typename V::EntryType *const data, const SuccessiveEntries<4> &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1,
            const typename V::AsArg v2, const typename V::AsArg v3)
    {
        const __m256i tmp0 = _mm256_unpacklo_epi16(v0.data(), v2.data()); // a0 c0 a1 c1 a2 c2 a3 c3 | a8 c8 a9 c9 ...
        const __m256i tmp1 = _mm256_unpackhi_epi16(v0.data(), v2.data()); // a4 c4 a5 c5 a6 c6 a7 c7 | a12 c12 ...
        const __m256i tmp2 = _mm256_unpacklo_epi16(v1.data(), v3.data()); // b0 d0 b1 d1 b2 d2 b3 d3 | b8 d8 b9 d9 ...
        const __m256i tmp3 = _mm256_unpackhi_epi16(v1.data(), v3.data()); // b4 d4 b5 ...

        const __m256i tmp4 = _mm256_unpacklo_epi16(tmp0, tmp2); // a0 b0 c0 d0 a1 b1 c1 d1 | a8 b8 c8 d8 a9 b9 ...
        const __m256i tmp5 = _mm256_unpackhi_epi16(tmp0, tmp2); // [abcd]2 [abcd]3 | [abcd]10 [abcd]11
        const __m256i tmp6 = _mm256_unpacklo_epi16(tmp1, tmp3); // [abcd]4 [abcd]5 | [abcd]12 [abcd]13
        const __m256i tmp7 = _mm256_unpackhi_epi16(tmp1, tmp3); // [abcd]6 [abcd]7 | [abcd]14 [abcd]15

        V(Mem::shuffle128<X0, Y0>(tmp4, tmp5)).store(&data[i[0]], ::Vc::Unaligned);
        V(Mem::shuffle128<X0, Y0>(tmp6, tmp7)).store(&data[i[4]], ::Vc::Unaligned);
        V(Mem::shuffle128<X1, Y1>(tmp4, tmp5)).store(&data[i[8]], ::Vc::Unaligned);
        V(Mem::shuffle128<X1, Y1>(tmp6, tmp7)).store(&data[i[12]], ::Vc::Unaligned);
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1)
    {
        const __m256i tmp4 =  // a0 b0 a1 b1 a2 b2 a3 b3 | a8 b8 a9 b9 a10 b10 a11 b11
            _mm256_setr_epi32(*reinterpret_cast<const int *>(&data[i[0]]),
                              *reinterpret_cast<const int *>(&data[i[1]]),
                              *reinterpret_cast<const int *>(&data[i[2]]),
                              *reinterpret_cast<const int *>(&data[i[3]]),
                              *reinterpret_cast<const int *>(&data[i[8]]),
                              *reinterpret_cast<const int *>(&data[i[9]]),
                              *reinterpret_cast<const int *>(&data[i[10]]),
                              *reinterpret_cast<const int *>(&data[i[11]]));
        const __m256i tmp5 = // a4 b4 a5 b5 a6 b6 a7 b7 | a12 b12 a13 b13 a14 b14 a15 b15
            _mm256_setr_epi32(*reinterpret_cast<const int *>(&data[i[4]]),
                              *reinterpret_cast<const int *>(&data[i[5]]),
                              *reinterpret_cast<const int *>(&data[i[6]]),
                              *reinterpret_cast<const int *>(&data[i[7]]),
                              *reinterpret_cast<const int *>(&data[i[12]]),
                              *reinterpret_cast<const int *>(&data[i[13]]),
                              *reinterpret_cast<const int *>(&data[i[14]]),
                              *reinterpret_cast<const int *>(&data[i[15]]));

        const __m256i tmp2 = _mm256_unpacklo_epi16(tmp4, tmp5);  // a0 a4 b0 b4 a1 a5 b1 b5 | a8 a12 b8 b12 a9 a13 b9 b13
        const __m256i tmp3 = _mm256_unpackhi_epi16(tmp4, tmp5);  // a2 a6 b2 b6 a3 a7 b3 b7 | a10 a14 b10 b14 a11 a15 b11 b15

        const __m256i tmp0 = _mm256_unpacklo_epi16(tmp2, tmp3);  // a0 a2 a4 a6 b0 b2 b4 b6 | a8 a10 a12 a14 b8 ...
        const __m256i tmp1 = _mm256_unpackhi_epi16(tmp2, tmp3);  // a1 a3 a5 a7 b1 b3 b5 b7 | a9 a11 a13 a15 b9 ...

        v0.data() = _mm256_unpacklo_epi16(tmp0, tmp1); // a0 a1 a2 a3 a4 a5 a6 a7 | a8 a9 ...
        v1.data() = _mm256_unpackhi_epi16(tmp0, tmp1); // b0 b1 b2 b3 b4 b5 b6 b7 | b8 b9 ...
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1, V &v2)
    {
        const __m256i tmp0 = avx_cast<__m256i>(_mm256_setr_pd(*reinterpret_cast<const double *>(&data[i[0]]),
                                                              *reinterpret_cast<const double *>(&data[i[1]]),
                                                              *reinterpret_cast<const double *>(&data[i[8]]),
                                                              *reinterpret_cast<const double *>(&data[i[9]])));
        const __m256i tmp1 = avx_cast<__m256i>(_mm256_setr_pd(*reinterpret_cast<const double *>(&data[i[2]]),
                                                              *reinterpret_cast<const double *>(&data[i[3]]),
                                                              *reinterpret_cast<const double *>(&data[i[10]]),
                                                              *reinterpret_cast<const double *>(&data[i[11]])));
        const __m256i tmp2 = avx_cast<__m256i>(_mm256_setr_pd(*reinterpret_cast<const double *>(&data[i[4]]),
                                                              *reinterpret_cast<const double *>(&data[i[5]]),
                                                              *reinterpret_cast<const double *>(&data[i[12]]),
                                                              *reinterpret_cast<const double *>(&data[i[13]])));
        const __m256i tmp3 = avx_cast<__m256i>(_mm256_setr_pd(*reinterpret_cast<const double *>(&data[i[6]]),
                                                              *reinterpret_cast<const double *>(&data[i[7]]),
                                                              *reinterpret_cast<const double *>(&data[i[14]]),
                                                              *reinterpret_cast<const double *>(&data[i[15]])));
        const __m256i tmp4 = _mm256_unpacklo_epi16(tmp0, tmp2); // a0 a4 b0 b4 c0 c4 XX XX | a8 a12 b8 ...
        const __m256i tmp5 = _mm256_unpackhi_epi16(tmp0, tmp2); // a1 a5 ...
        const __m256i tmp6 = _mm256_unpacklo_epi16(tmp1, tmp3); // a2 a6 ...
        const __m256i tmp7 = _mm256_unpackhi_epi16(tmp1, tmp3); // a3 a7 ...

        const __m256i tmp8  = _mm256_unpacklo_epi16(tmp4, tmp6); // a0 a2 a4 a6 b0 ...
        const __m256i tmp9  = _mm256_unpackhi_epi16(tmp4, tmp6); // c0 c2 c4 c6 d0 ...
        const __m256i tmp10 = _mm256_unpacklo_epi16(tmp5, tmp7); // a1 a3 a5 a7 b1 ...
        const __m256i tmp11 = _mm256_unpackhi_epi16(tmp5, tmp7); // c1 c3 c5 c7 XX ...

        v0.data() = _mm256_unpacklo_epi16(tmp8, tmp10); // a0 a1 a2 a3 a4 a5 a6 a7 | a8 ...
        v1.data() = _mm256_unpackhi_epi16(tmp8, tmp10);
        v2.data() = _mm256_unpacklo_epi16(tmp9, tmp11);
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1, V &v2, V &v3)
    {
        const __m256i tmp0 = avx_cast<__m256i>(_mm256_setr_pd(*reinterpret_cast<const double *>(&data[i[0]]),
                                                              *reinterpret_cast<const double *>(&data[i[1]]),
                                                              *reinterpret_cast<const double *>(&data[i[8]]),
                                                              *reinterpret_cast<const double *>(&data[i[9]])));
        const __m256i tmp1 = avx_cast<__m256i>(_mm256_setr_pd(*reinterpret_cast<const double *>(&data[i[2]]),
                                                              *reinterpret_cast<const double *>(&data[i[3]]),
                                                              *reinterpret_cast<const double *>(&data[i[10]]),
                                                              *reinterpret_cast<const double *>(&data[i[11]])));
        const __m256i tmp2 = avx_cast<__m256i>(_mm256_setr_pd(*reinterpret_cast<const double *>(&data[i[4]]),
                                                              *reinterpret_cast<const double *>(&data[i[5]]),
                                                              *reinterpret_cast<const double *>(&data[i[12]]),
                                                              *reinterpret_cast<const double *>(&data[i[13]])));
        const __m256i tmp3 = avx_cast<__m256i>(_mm256_setr_pd(*reinterpret_cast<const double *>(&data[i[6]]),
                                                              *reinterpret_cast<const double *>(&data[i[7]]),
                                                              *reinterpret_cast<const double *>(&data[i[14]]),
                                                              *reinterpret_cast<const double *>(&data[i[15]])));
        const __m256i tmp4 = _mm256_unpacklo_epi16(tmp0, tmp2); // a0 a4 b0 b4 c0 c4 d0 d4 | a8 a12 b8 ...
        const __m256i tmp5 = _mm256_unpackhi_epi16(tmp0, tmp2); // a1 a5 ...
        const __m256i tmp6 = _mm256_unpacklo_epi16(tmp1, tmp3); // a2 a6 ...
        const __m256i tmp7 = _mm256_unpackhi_epi16(tmp1, tmp3); // a3 a7 ...

        const __m256i tmp8  = _mm256_unpacklo_epi16(tmp4, tmp6); // a0 a2 a4 a6 b0 ...
        const __m256i tmp9  = _mm256_unpackhi_epi16(tmp4, tmp6); // c0 c2 c4 c6 d0 ...
        const __m256i tmp10 = _mm256_unpacklo_epi16(tmp5, tmp7); // a1 a3 a5 a7 b1 ...
        const __m256i tmp11 = _mm256_unpackhi_epi16(tmp5, tmp7); // c1 c3 c5 c7 d1 ...

        v0.data() = _mm256_unpacklo_epi16(tmp8, tmp10); // a0 a1 a2 a3 a4 a5 a6 a7 | a8 ...
        v1.data() = _mm256_unpackhi_epi16(tmp8, tmp10);
        v2.data() = _mm256_unpacklo_epi16(tmp9, tmp11);
        v3.data() = _mm256_unpackhi_epi16(tmp9, tmp11);
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1, V &v2, V &v3, V &v4)
    {
        const __m256i a = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[0]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[8]])));
        const __m256i b = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[1]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[9]])));
        const __m256i c = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[2]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[10]])));
        const __m256i d = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[3]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[11]])));
        const __m256i e = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[4]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[12]])));
        const __m256i f = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[5]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[13]])));
        const __m256i g = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[6]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[14]])));
        const __m256i h = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[7]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[15]])));

        const __m256i tmp2  = _mm256_unpacklo_epi16(a, e); // a0 a4 b0 b4 c0 c4 d0 d4 | a8 ...
        const __m256i tmp4  = _mm256_unpacklo_epi16(b, f); // a1 a5 b1 b5 c1 c5 d1 d5
        const __m256i tmp3  = _mm256_unpacklo_epi16(c, g); // a2 a6 b2 b6 c2 c6 d2 d6
        const __m256i tmp5  = _mm256_unpacklo_epi16(d, h); // a3 a7 b3 b7 c3 c7 d3 d7
        const __m256i tmp10 = _mm256_unpackhi_epi16(a, e); // e0 e4 f0 f4 g0 g4 h0 h4
        const __m256i tmp11 = _mm256_unpackhi_epi16(c, g); // e1 e5 f1 f5 g1 g5 h1 h5
        const __m256i tmp12 = _mm256_unpackhi_epi16(b, f); // e2 e6 f2 f6 g2 g6 h2 h6
        const __m256i tmp13 = _mm256_unpackhi_epi16(d, h); // e3 e7 f3 f7 g3 g7 h3 h7

        const __m256i tmp0  = _mm256_unpacklo_epi16(tmp2, tmp3); // a0 a2 a4 a6 b0 b2 b4 b6 | a8 ...
        const __m256i tmp1  = _mm256_unpacklo_epi16(tmp4, tmp5); // a1 a3 a5 a7 b1 b3 b5 b7
        const __m256i tmp6  = _mm256_unpackhi_epi16(tmp2, tmp3); // c0 c2 c4 c6 d0 d2 d4 d6
        const __m256i tmp7  = _mm256_unpackhi_epi16(tmp4, tmp5); // c1 c3 c5 c7 d1 d3 d5 d7
        const __m256i tmp8  = _mm256_unpacklo_epi16(tmp10, tmp11); // e0 e2 e4 e6 f0 f2 f4 f6
        const __m256i tmp9  = _mm256_unpacklo_epi16(tmp12, tmp13); // e1 e3 e5 e7 f1 f3 f5 f7

        v0.data() = _mm256_unpacklo_epi16(tmp0, tmp1);
        v1.data() = _mm256_unpackhi_epi16(tmp0, tmp1);
        v2.data() = _mm256_unpacklo_epi16(tmp6, tmp7);
        v3.data() = _mm256_unpackhi_epi16(tmp6, tmp7);
        v4.data() = _mm256_unpacklo_epi16(tmp8, tmp9);
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1, V &v2, V &v3, V &v4, V &v5)
    {
        const __m256i a = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[0]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[8]])));
        const __m256i b = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[1]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[9]])));
        const __m256i c = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[2]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[10]])));
        const __m256i d = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[3]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[11]])));
        const __m256i e = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[4]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[12]])));
        const __m256i f = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[5]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[13]])));
        const __m256i g = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[6]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[14]])));
        const __m256i h = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[7]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[15]])));

        const __m256i tmp2  = _mm256_unpacklo_epi16(a, e); // a0 a4 b0 b4 c0 c4 d0 d4 | a8 ...
        const __m256i tmp4  = _mm256_unpacklo_epi16(b, f); // a1 a5 b1 b5 c1 c5 d1 d5
        const __m256i tmp3  = _mm256_unpacklo_epi16(c, g); // a2 a6 b2 b6 c2 c6 d2 d6
        const __m256i tmp5  = _mm256_unpacklo_epi16(d, h); // a3 a7 b3 b7 c3 c7 d3 d7
        const __m256i tmp10 = _mm256_unpackhi_epi16(a, e); // e0 e4 f0 f4 g0 g4 h0 h4
        const __m256i tmp11 = _mm256_unpackhi_epi16(c, g); // e1 e5 f1 f5 g1 g5 h1 h5
        const __m256i tmp12 = _mm256_unpackhi_epi16(b, f); // e2 e6 f2 f6 g2 g6 h2 h6
        const __m256i tmp13 = _mm256_unpackhi_epi16(d, h); // e3 e7 f3 f7 g3 g7 h3 h7

        const __m256i tmp0  = _mm256_unpacklo_epi16(tmp2, tmp3); // a0 a2 a4 a6 b0 b2 b4 b6 | a8 ...
        const __m256i tmp1  = _mm256_unpacklo_epi16(tmp4, tmp5); // a1 a3 a5 a7 b1 b3 b5 b7
        const __m256i tmp6  = _mm256_unpackhi_epi16(tmp2, tmp3); // c0 c2 c4 c6 d0 d2 d4 d6
        const __m256i tmp7  = _mm256_unpackhi_epi16(tmp4, tmp5); // c1 c3 c5 c7 d1 d3 d5 d7
        const __m256i tmp8  = _mm256_unpacklo_epi16(tmp10, tmp11); // e0 e2 e4 e6 f0 f2 f4 f6
        const __m256i tmp9  = _mm256_unpacklo_epi16(tmp12, tmp13); // e1 e3 e5 e7 f1 f3 f5 f7

        v0.data() = _mm256_unpacklo_epi16(tmp0, tmp1);
        v1.data() = _mm256_unpackhi_epi16(tmp0, tmp1);
        v2.data() = _mm256_unpacklo_epi16(tmp6, tmp7);
        v3.data() = _mm256_unpackhi_epi16(tmp6, tmp7);
        v4.data() = _mm256_unpacklo_epi16(tmp8, tmp9);
        v5.data() = _mm256_unpackhi_epi16(tmp8, tmp9);
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1, V &v2, V &v3, V &v4, V &v5, V &v6)
    {
        const __m256i a = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[0]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[8]])));
        const __m256i b = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[1]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[9]])));
        const __m256i c = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[2]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[10]])));
        const __m256i d = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[3]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[11]])));
        const __m256i e = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[4]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[12]])));
        const __m256i f = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[5]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[13]])));
        const __m256i g = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[6]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[14]])));
        const __m256i h = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[7]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[15]])));

        const __m256i tmp2  = _mm256_unpacklo_epi16(a, e); // a0 a4 b0 b4 c0 c4 d0 d4 | a8 ...
        const __m256i tmp4  = _mm256_unpacklo_epi16(b, f); // a1 a5 b1 b5 c1 c5 d1 d5
        const __m256i tmp3  = _mm256_unpacklo_epi16(c, g); // a2 a6 b2 b6 c2 c6 d2 d6
        const __m256i tmp5  = _mm256_unpacklo_epi16(d, h); // a3 a7 b3 b7 c3 c7 d3 d7
        const __m256i tmp10 = _mm256_unpackhi_epi16(a, e); // e0 e4 f0 f4 g0 g4 h0 h4
        const __m256i tmp11 = _mm256_unpackhi_epi16(c, g); // e1 e5 f1 f5 g1 g5 h1 h5
        const __m256i tmp12 = _mm256_unpackhi_epi16(b, f); // e2 e6 f2 f6 g2 g6 h2 h6
        const __m256i tmp13 = _mm256_unpackhi_epi16(d, h); // e3 e7 f3 f7 g3 g7 h3 h7

        const __m256i tmp0  = _mm256_unpacklo_epi16(tmp2, tmp3); // a0 a2 a4 a6 b0 b2 b4 b6 | a8 ...
        const __m256i tmp1  = _mm256_unpacklo_epi16(tmp4, tmp5); // a1 a3 a5 a7 b1 b3 b5 b7
        const __m256i tmp6  = _mm256_unpackhi_epi16(tmp2, tmp3); // c0 c2 c4 c6 d0 d2 d4 d6
        const __m256i tmp7  = _mm256_unpackhi_epi16(tmp4, tmp5); // c1 c3 c5 c7 d1 d3 d5 d7
        const __m256i tmp8  = _mm256_unpacklo_epi16(tmp10, tmp11); // e0 e2 e4 e6 f0 f2 f4 f6
        const __m256i tmp9  = _mm256_unpacklo_epi16(tmp12, tmp13); // e1 e3 e5 e7 f1 f3 f5 f7
        const __m256i tmp14 = _mm256_unpackhi_epi16(tmp10, tmp11); // g0 g2 g4 g6 h0 h2 h4 h6
        const __m256i tmp15 = _mm256_unpackhi_epi16(tmp12, tmp13); // g1 g3 g5 g7 h1 h3 h5 h7

        v0.data() = _mm256_unpacklo_epi16(tmp0, tmp1);
        v1.data() = _mm256_unpackhi_epi16(tmp0, tmp1);
        v2.data() = _mm256_unpacklo_epi16(tmp6, tmp7);
        v3.data() = _mm256_unpackhi_epi16(tmp6, tmp7);
        v4.data() = _mm256_unpacklo_epi16(tmp8, tmp9);
        v5.data() = _mm256_unpackhi_epi16(tmp8, tmp9);
        v6.data() = _mm256_unpacklo_epi16(tmp14, tmp15);
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1, V &v2, V &v3, V &v4, V &v5, V &v6, V &v7)
    {
        const __m256i a = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[0]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[8]])));
        const __m256i b = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[1]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[9]])));
        const __m256i c = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[2]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[10]])));
        const __m256i d = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[3]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[11]])));
        const __m256i e = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[4]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[12]])));
        const __m256i f = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[5]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[13]])));
        const __m256i g = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[6]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[14]])));
        const __m256i h = concat(_mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[7]])),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i *>(&data[i[15]])));

        const __m256i tmp2  = _mm256_unpacklo_epi16(a, e); // a0 a4 b0 b4 c0 c4 d0 d4 | a8 ...
        const __m256i tmp4  = _mm256_unpacklo_epi16(b, f); // a1 a5 b1 b5 c1 c5 d1 d5
        const __m256i tmp3  = _mm256_unpacklo_epi16(c, g); // a2 a6 b2 b6 c2 c6 d2 d6
        const __m256i tmp5  = _mm256_unpacklo_epi16(d, h); // a3 a7 b3 b7 c3 c7 d3 d7
        const __m256i tmp10 = _mm256_unpackhi_epi16(a, e); // e0 e4 f0 f4 g0 g4 h0 h4
        const __m256i tmp11 = _mm256_unpackhi_epi16(c, g); // e1 e5 f1 f5 g1 g5 h1 h5
        const __m256i tmp12 = _mm256_unpackhi_epi16(b, f); // e2 e6 f2 f6 g2 g6 h2 h6
        const __m256i tmp13 = _mm256_unpackhi_epi16(d, h); // e3 e7 f3 f7 g3 g7 h3 h7

        const __m256i tmp0  = _mm256_unpacklo_epi16(tmp2, tmp3); // a0 a2 a4 a6 b0 b2 b4 b6 | a8 ...
        const __m256i tmp1  = _mm256_unpacklo_epi16(tmp4, tmp5); // a1 a3 a5 a7 b1 b3 b5 b7
        const __m256i tmp6  = _mm256_unpackhi_epi16(tmp2, tmp3); // c0 c2 c4 c6 d0 d2 d4 d6
        const __m256i tmp7  = _mm256_unpackhi_epi16(tmp4, tmp5); // c1 c3 c5 c7 d1 d3 d5 d7
        const __m256i tmp8  = _mm256_unpacklo_epi16(tmp10, tmp11); // e0 e2 e4 e6 f0 f2 f4 f6
        const __m256i tmp9  = _mm256_unpacklo_epi16(tmp12, tmp13); // e1 e3 e5 e7 f1 f3 f5 f7
        const __m256i tmp14 = _mm256_unpackhi_epi16(tmp10, tmp11); // g0 g2 g4 g6 h0 h2 h4 h6
        const __m256i tmp15 = _mm256_unpackhi_epi16(tmp12, tmp13); // g1 g3 g5 g7 h1 h3 h5 h7

        v0.data() = _mm256_unpacklo_epi16(tmp0, tmp1);
        v1.data() = _mm256_unpackhi_epi16(tmp0, tmp1);
        v2.data() = _mm256_unpacklo_epi16(tmp6, tmp7);
        v3.data() = _mm256_unpackhi_epi16(tmp6, tmp7);
        v4.data() = _mm256_unpacklo_epi16(tmp8, tmp9);
        v5.data() = _mm256_unpackhi_epi16(tmp8, tmp9);
        v6.data() = _mm256_unpacklo_epi16(tmp14, tmp15);
        v7.data() = _mm256_unpackhi_epi16(tmp14, tmp15);
    }/*}}}*/
};
template<typename V> struct InterleaveImpl<V, 8, 32> {
    template<typename I> static inline void interleave(typename V::EntryType *const data, const I &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1)
    {
        using namespace Vc::Vc_IMPL_NAMESPACE;
        // [0a 1a 0b 1b 0e 1e 0f 1f]:
        const m256 tmp0 = _mm256_unpacklo_ps(avx_cast<m256>(v0.data()), avx_cast<m256>(v1.data()));
        // [0c 1c 0d 1d 0g 1g 0h 1h]:
        const m256 tmp1 = _mm256_unpackhi_ps(avx_cast<m256>(v0.data()), avx_cast<m256>(v1.data()));
        _mm_storel_pi(reinterpret_cast<__m64 *>(&data[i[0]]), lo128(tmp0));
        _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[1]]), lo128(tmp0));
        _mm_storel_pi(reinterpret_cast<__m64 *>(&data[i[2]]), lo128(tmp1));
        _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[3]]), lo128(tmp1));
        _mm_storel_pi(reinterpret_cast<__m64 *>(&data[i[4]]), hi128(tmp0));
        _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[5]]), hi128(tmp0));
        _mm_storel_pi(reinterpret_cast<__m64 *>(&data[i[6]]), hi128(tmp1));
        _mm_storeh_pi(reinterpret_cast<__m64 *>(&data[i[7]]), hi128(tmp1));
    }/*}}}*/
    static inline void interleave(typename V::EntryType *const data, const SuccessiveEntries<2> &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1)
    {
        using namespace Vc::Vc_IMPL_NAMESPACE;
        // [0a 1a 0b 1b 0e 1e 0f 1f]:
        const m256 tmp0 = _mm256_unpacklo_ps(avx_cast<m256>(v0.data()), avx_cast<m256>(v1.data()));
        // [0c 1c 0d 1d 0g 1g 0h 1h]:
        const m256 tmp1 = _mm256_unpackhi_ps(avx_cast<m256>(v0.data()), avx_cast<m256>(v1.data()));
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[0]]), lo128(tmp0));
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[2]]), lo128(tmp1));
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[4]]), hi128(tmp0));
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[6]]), hi128(tmp1));
    }/*}}}*/
    template<typename I> static inline void interleave(typename V::EntryType *const data, const I &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2)
    {
        using namespace Vc::Vc_IMPL_NAMESPACE;
#ifdef VC_USE_MASKMOV_SCATTER
        // [0a 2a 0b 2b 0e 2e 0f 2f]:
        const m256 tmp0 = _mm256_unpacklo_ps(avx_cast<m256>(v0.data()), avx_cast<m256>(v2.data()));
        // [0c 2c 0d 2d 0g 2g 0h 2h]:
        const m256 tmp1 = _mm256_unpackhi_ps(avx_cast<m256>(v0.data()), avx_cast<m256>(v2.data()));
        // [1a __ 1b __ 1e __ 1f __]:
        const m256 tmp2 = _mm256_unpacklo_ps(avx_cast<m256>(v1.data()), avx_cast<m256>(v1.data()));
        // [1c __ 1d __ 1g __ 1h __]:
        const m256 tmp3 = _mm256_unpackhi_ps(avx_cast<m256>(v1.data()), avx_cast<m256>(v1.data()));
        const m256 tmp4 = _mm256_unpacklo_ps(tmp0, tmp2);
        const m256 tmp5 = _mm256_unpackhi_ps(tmp0, tmp2);
        const m256 tmp6 = _mm256_unpacklo_ps(tmp1, tmp3);
        const m256 tmp7 = _mm256_unpackhi_ps(tmp1, tmp3);
        const m128i mask = _mm_set_epi32(0, -1, -1, -1);
        _mm_maskstore_ps(reinterpret_cast<float *>(&data[i[0]]), mask, lo128(tmp4));
        _mm_maskstore_ps(reinterpret_cast<float *>(&data[i[1]]), mask, lo128(tmp5));
        _mm_maskstore_ps(reinterpret_cast<float *>(&data[i[2]]), mask, lo128(tmp6));
        _mm_maskstore_ps(reinterpret_cast<float *>(&data[i[3]]), mask, lo128(tmp7));
        _mm_maskstore_ps(reinterpret_cast<float *>(&data[i[4]]), mask, hi128(tmp4));
        _mm_maskstore_ps(reinterpret_cast<float *>(&data[i[5]]), mask, hi128(tmp5));
        _mm_maskstore_ps(reinterpret_cast<float *>(&data[i[6]]), mask, hi128(tmp6));
        _mm_maskstore_ps(reinterpret_cast<float *>(&data[i[7]]), mask, hi128(tmp7));
#else
        interleave(data, i, v0, v1);
        v2.scatter(data + 2, i);
#endif
    }/*}}}*/
    template<typename I> static inline void interleave(typename V::EntryType *const data, const I &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1,
            const typename V::AsArg v2, const typename V::AsArg v3)
    {
        using namespace Vc::Vc_IMPL_NAMESPACE;
        const m256 tmp0 = _mm256_unpacklo_ps(avx_cast<m256>(v0.data()), avx_cast<m256>(v2.data()));
        const m256 tmp1 = _mm256_unpackhi_ps(avx_cast<m256>(v0.data()), avx_cast<m256>(v2.data()));
        const m256 tmp2 = _mm256_unpacklo_ps(avx_cast<m256>(v1.data()), avx_cast<m256>(v3.data()));
        const m256 tmp3 = _mm256_unpackhi_ps(avx_cast<m256>(v1.data()), avx_cast<m256>(v3.data()));
        const m256 tmp4 = _mm256_unpacklo_ps(tmp0, tmp2);
        const m256 tmp5 = _mm256_unpackhi_ps(tmp0, tmp2);
        const m256 tmp6 = _mm256_unpacklo_ps(tmp1, tmp3);
        const m256 tmp7 = _mm256_unpackhi_ps(tmp1, tmp3);
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[0]]), lo128(tmp4));
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[1]]), lo128(tmp5));
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[2]]), lo128(tmp6));
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[3]]), lo128(tmp7));
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[4]]), hi128(tmp4));
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[5]]), hi128(tmp5));
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[6]]), hi128(tmp6));
        _mm_storeu_ps(reinterpret_cast<float *>(&data[i[7]]), hi128(tmp7));
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1)
    {
        using namespace Vc::Vc_IMPL_NAMESPACE;
        const m128  il0 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&data[i[0]])); // a0 b0
        const m128  il2 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&data[i[2]])); // a2 b2
        const m128  il4 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&data[i[4]])); // a4 b4
        const m128  il6 = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<__m64 const *>(&data[i[6]])); // a6 b6
        const m128 il01 = _mm_loadh_pi(             il0, reinterpret_cast<__m64 const *>(&data[i[1]])); // a0 b0 a1 b1
        const m128 il23 = _mm_loadh_pi(             il2, reinterpret_cast<__m64 const *>(&data[i[3]])); // a2 b2 a3 b3
        const m128 il45 = _mm_loadh_pi(             il4, reinterpret_cast<__m64 const *>(&data[i[5]])); // a4 b4 a5 b5
        const m128 il67 = _mm_loadh_pi(             il6, reinterpret_cast<__m64 const *>(&data[i[7]])); // a6 b6 a7 b7

        const m256 tmp2 = concat(il01, il45);
        const m256 tmp3 = concat(il23, il67);

        const m256 tmp0 = _mm256_unpacklo_ps(tmp2, tmp3);
        const m256 tmp1 = _mm256_unpackhi_ps(tmp2, tmp3);

        v0.data() = avx_cast<typename V::VectorType>(_mm256_unpacklo_ps(tmp0, tmp1));
        v1.data() = avx_cast<typename V::VectorType>(_mm256_unpackhi_ps(tmp0, tmp1));
    }/*}}}*/
    static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const SuccessiveEntries<2> &i, V &v0, V &v1)
    {
        using namespace Vc::Vc_IMPL_NAMESPACE;
        const m256 il0123 = _mm256_loadu_ps(reinterpret_cast<const float *>(&data[i[0]])); // a0 b0 a1 b1 a2 b2 a3 b3
        const m256 il4567 = _mm256_loadu_ps(reinterpret_cast<const float *>(&data[i[4]])); // a4 b4 a5 b5 a6 b6 a7 b7

        const m256 tmp2 = Mem::shuffle128<X0, Y0>(il0123, il4567);
        const m256 tmp3 = Mem::shuffle128<X1, Y1>(il0123, il4567);

        const m256 tmp0 = _mm256_unpacklo_ps(tmp2, tmp3);
        const m256 tmp1 = _mm256_unpackhi_ps(tmp2, tmp3);

        v0.data() = avx_cast<typename V::VectorType>(_mm256_unpacklo_ps(tmp0, tmp1));
        v1.data() = avx_cast<typename V::VectorType>(_mm256_unpackhi_ps(tmp0, tmp1));
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1, V &v2)
    {
        using namespace Vc::Vc_IMPL_NAMESPACE;
        const m128  il0 = _mm_loadu_ps(reinterpret_cast<const float *>(&data[i[0]])); // a0 b0 c0 d0
        const m128  il1 = _mm_loadu_ps(reinterpret_cast<const float *>(&data[i[1]])); // a1 b1 c1 d1
        const m128  il2 = _mm_loadu_ps(reinterpret_cast<const float *>(&data[i[2]])); // a2 b2 c2 d2
        const m128  il3 = _mm_loadu_ps(reinterpret_cast<const float *>(&data[i[3]])); // a3 b3 c3 d3
        const m128  il4 = _mm_loadu_ps(reinterpret_cast<const float *>(&data[i[4]])); // a4 b4 c4 d4
        const m128  il5 = _mm_loadu_ps(reinterpret_cast<const float *>(&data[i[5]])); // a5 b5 c5 d5
        const m128  il6 = _mm_loadu_ps(reinterpret_cast<const float *>(&data[i[6]])); // a6 b6 c6 d6
        const m128  il7 = _mm_loadu_ps(reinterpret_cast<const float *>(&data[i[7]])); // a7 b7 c7 d7

        const m256 il04 = concat(il0, il4);
        const m256 il15 = concat(il1, il5);
        const m256 il26 = concat(il2, il6);
        const m256 il37 = concat(il3, il7);
        const m256 ab0246 = _mm256_unpacklo_ps(il04, il26);
        const m256 ab1357 = _mm256_unpacklo_ps(il15, il37);
        const m256 cd0246 = _mm256_unpackhi_ps(il04, il26);
        const m256 cd1357 = _mm256_unpackhi_ps(il15, il37);
        v0.data() = avx_cast<typename V::VectorType>(_mm256_unpacklo_ps(ab0246, ab1357));
        v1.data() = avx_cast<typename V::VectorType>(_mm256_unpackhi_ps(ab0246, ab1357));
        v2.data() = avx_cast<typename V::VectorType>(_mm256_unpacklo_ps(cd0246, cd1357));
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1, V &v2, V &v3)
    {
        using namespace Vc::Vc_IMPL_NAMESPACE;
        const m128  il0 = _mm_loadu_ps(reinterpret_cast<const float *>(&data[i[0]])); // a0 b0 c0 d0
        const m128  il1 = _mm_loadu_ps(reinterpret_cast<const float *>(&data[i[1]])); // a1 b1 c1 d1
        const m128  il2 = _mm_loadu_ps(reinterpret_cast<const float *>(&data[i[2]])); // a2 b2 c2 d2
        const m128  il3 = _mm_loadu_ps(reinterpret_cast<const float *>(&data[i[3]])); // a3 b3 c3 d3
        const m128  il4 = _mm_loadu_ps(reinterpret_cast<const float *>(&data[i[4]])); // a4 b4 c4 d4
        const m128  il5 = _mm_loadu_ps(reinterpret_cast<const float *>(&data[i[5]])); // a5 b5 c5 d5
        const m128  il6 = _mm_loadu_ps(reinterpret_cast<const float *>(&data[i[6]])); // a6 b6 c6 d6
        const m128  il7 = _mm_loadu_ps(reinterpret_cast<const float *>(&data[i[7]])); // a7 b7 c7 d7

        const m256 il04 = concat(il0, il4);
        const m256 il15 = concat(il1, il5);
        const m256 il26 = concat(il2, il6);
        const m256 il37 = concat(il3, il7);
        const m256 ab0246 = _mm256_unpacklo_ps(il04, il26);
        const m256 ab1357 = _mm256_unpacklo_ps(il15, il37);
        const m256 cd0246 = _mm256_unpackhi_ps(il04, il26);
        const m256 cd1357 = _mm256_unpackhi_ps(il15, il37);
        v0.data() = avx_cast<typename V::VectorType>(_mm256_unpacklo_ps(ab0246, ab1357));
        v1.data() = avx_cast<typename V::VectorType>(_mm256_unpackhi_ps(ab0246, ab1357));
        v2.data() = avx_cast<typename V::VectorType>(_mm256_unpacklo_ps(cd0246, cd1357));
        v3.data() = avx_cast<typename V::VectorType>(_mm256_unpackhi_ps(cd0246, cd1357));
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1, V &v2, V &v3, V &v4)
    {
        v4.gather(data + 4, i);
        deinterleave(data, i, v0, v1, v2, v3);
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1, V &v2, V &v3, V &v4, V &v5)
    {
        deinterleave(data, i, v0, v1, v2, v3);
        deinterleave(data + 4, i, v4, v5);
    }/*}}}*/
    static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const SuccessiveEntries<6> &i, V &v0, V &v1, V &v2, V &v3, V &v4, V &v5)
    {
        using namespace Vc::Vc_IMPL_NAMESPACE;
        const m256 a = _mm256_loadu_ps(reinterpret_cast<const float *>(&data[i[0]]));
        const m256 b = _mm256_loadu_ps(reinterpret_cast<const float *>(&data[i[0] + 1 * V::Size]));
        const m256 c = _mm256_loadu_ps(reinterpret_cast<const float *>(&data[i[0] + 2 * V::Size]));
        const m256 d = _mm256_loadu_ps(reinterpret_cast<const float *>(&data[i[0] + 3 * V::Size]));
        const m256 e = _mm256_loadu_ps(reinterpret_cast<const float *>(&data[i[0] + 4 * V::Size]));
        const m256 f = _mm256_loadu_ps(reinterpret_cast<const float *>(&data[i[0] + 5 * V::Size]));
        const __m256 tmp2 = Mem::shuffle128<X0, Y0>(a, d);
        const __m256 tmp3 = Mem::shuffle128<X1, Y1>(b, e);
        const __m256 tmp4 = Mem::shuffle128<X1, Y1>(a, d);
        const __m256 tmp5 = Mem::shuffle128<X0, Y0>(c, f);
        const __m256 tmp8 = Mem::shuffle128<X0, Y0>(b, e);
        const __m256 tmp9 = Mem::shuffle128<X1, Y1>(c, f);
        const __m256 tmp0 = _mm256_unpacklo_ps(tmp2, tmp3);
        const __m256 tmp1 = _mm256_unpackhi_ps(tmp4, tmp5);
        const __m256 tmp6 = _mm256_unpackhi_ps(tmp2, tmp3);
        const __m256 tmp7 = _mm256_unpacklo_ps(tmp8, tmp9);
        const __m256 tmp10 = _mm256_unpacklo_ps(tmp4, tmp5);
        const __m256 tmp11 = _mm256_unpackhi_ps(tmp8, tmp9);
        v0.data() = avx_cast<typename V::VectorType>(_mm256_unpacklo_ps(tmp0, tmp1));
        v1.data() = avx_cast<typename V::VectorType>(_mm256_unpackhi_ps(tmp0, tmp1));
        v2.data() = avx_cast<typename V::VectorType>(_mm256_unpacklo_ps(tmp6, tmp7));
        v3.data() = avx_cast<typename V::VectorType>(_mm256_unpackhi_ps(tmp6, tmp7));
        v4.data() = avx_cast<typename V::VectorType>(_mm256_unpacklo_ps(tmp10, tmp11));
        v5.data() = avx_cast<typename V::VectorType>(_mm256_unpackhi_ps(tmp10, tmp11));
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
template<typename V> struct InterleaveImpl<V, 4, 32> {
    template<typename I> static inline void interleave(typename V::EntryType *const data, const I &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1)
    {
        using namespace Vc::Vc_IMPL_NAMESPACE;
        const m256d tmp0 = _mm256_unpacklo_pd(v0.data(), v1.data());
        const m256d tmp1 = _mm256_unpackhi_pd(v0.data(), v1.data());
        _mm_storeu_pd(&data[i[0]], lo128(tmp0));
        _mm_storeu_pd(&data[i[1]], lo128(tmp1));
        _mm_storeu_pd(&data[i[2]], hi128(tmp0));
        _mm_storeu_pd(&data[i[3]], hi128(tmp1));
    }/*}}}*/
    template<typename I> static inline void interleave(typename V::EntryType *const data, const I &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2)
    {
        using namespace Vc::Vc_IMPL_NAMESPACE;
#ifdef VC_USE_MASKMOV_SCATTER
        const m256d tmp0 = _mm256_unpacklo_pd(v0.data(), v1.data());
        const m256d tmp1 = _mm256_unpackhi_pd(v0.data(), v1.data());
        const m256d tmp2 = _mm256_unpacklo_pd(v2.data(), v2.data());
        const m256d tmp3 = _mm256_unpackhi_pd(v2.data(), v2.data());

#if defined(VC_MSVC) && (VC_MSVC < 170000000 || !defined(_WIN64))
        // MSVC needs to be at Version 2012 before _mm256_set_epi64x works
        const m256i mask = concat(_mm_setallone_si128(), _mm_set_epi32(0, 0, -1, -1));
#else
        const m256i mask = _mm256_set_epi64x(0, -1, -1, -1);
#endif
        _mm256_maskstore_pd(&data[i[0]], mask, Mem::shuffle128<X0, Y0>(tmp0, tmp2));
        _mm256_maskstore_pd(&data[i[1]], mask, Mem::shuffle128<X0, Y0>(tmp1, tmp3));
        _mm256_maskstore_pd(&data[i[2]], mask, Mem::shuffle128<X1, Y1>(tmp0, tmp2));
        _mm256_maskstore_pd(&data[i[3]], mask, Mem::shuffle128<X1, Y1>(tmp1, tmp3));
#else
        interleave(data, i, v0, v1);
        v2.scatter(data + 2, i);
#endif
    }/*}}}*/
    template<typename I> static inline void interleave(typename V::EntryType *const data, const I &i,/*{{{*/
            const typename V::AsArg v0, const typename V::AsArg v1,
            const typename V::AsArg v2, const typename V::AsArg v3)
    {
        using namespace Vc::Vc_IMPL_NAMESPACE;
        // 0a 1a 0c 1c:
        const m256d tmp0 = _mm256_unpacklo_pd(v0.data(), v1.data());
        // 0b 1b 0b 1b:
        const m256d tmp1 = _mm256_unpackhi_pd(v0.data(), v1.data());
        // 2a 3a 2c 3c:
        const m256d tmp2 = _mm256_unpacklo_pd(v2.data(), v3.data());
        // 2b 3b 2b 3b:
        const m256d tmp3 = _mm256_unpackhi_pd(v2.data(), v3.data());
        /* The following might be more efficient once 256-bit stores are not split internally into 2
         * 128-bit stores.
        _mm256_storeu_pd(&data[i[0]], Mem::shuffle128<X0, Y0>(tmp0, tmp2));
        _mm256_storeu_pd(&data[i[1]], Mem::shuffle128<X0, Y0>(tmp1, tmp3));
        _mm256_storeu_pd(&data[i[2]], Mem::shuffle128<X1, Y1>(tmp0, tmp2));
        _mm256_storeu_pd(&data[i[3]], Mem::shuffle128<X1, Y1>(tmp1, tmp3));
        */
        _mm_storeu_pd(&data[i[0]  ], lo128(tmp0));
        _mm_storeu_pd(&data[i[0]+2], lo128(tmp2));
        _mm_storeu_pd(&data[i[1]  ], lo128(tmp1));
        _mm_storeu_pd(&data[i[1]+2], lo128(tmp3));
        _mm_storeu_pd(&data[i[2]  ], hi128(tmp0));
        _mm_storeu_pd(&data[i[2]+2], hi128(tmp2));
        _mm_storeu_pd(&data[i[3]  ], hi128(tmp1));
        _mm_storeu_pd(&data[i[3]+2], hi128(tmp3));
    }/*}}}*/
    template<typename I> static inline void deinterleave(typename V::EntryType const *const data,/*{{{*/
            const I &i, V &v0, V &v1)
    {
        using namespace Vc::Vc_IMPL_NAMESPACE;
        const m256d ab02 = concat(_mm_loadu_pd(&data[i[0]]), _mm_loadu_pd(&data[i[2]]));
        const m256d ab13 = concat(_mm_loadu_pd(&data[i[1]]), _mm_loadu_pd(&data[i[3]]));

        v0.data() = _mm256_unpacklo_pd(ab02, ab13);
        v1.data() = _mm256_unpackhi_pd(ab02, ab13);
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
        v4.gather(data + 4, i);
        deinterleave(data, i, v0, v1);
        deinterleave(data + 2, i, v2, v3);
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
        v6.gather(data + 6, i);
        deinterleave(data, i, v0, v1);
        deinterleave(data + 2, i, v2, v3);
        deinterleave(data + 4, i, v4, v5);
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

// InterleavedMemoryAccessBase::interleave {{{1
// 2 args {{{2
template <typename V, typename I, bool RO>
inline void InterleavedMemoryAccessBase<V, I, RO>::interleave(const typename V::AsArg v0,
                                                              const typename V::AsArg v1)
{
    InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data, m_indexes, v0, v1);
}
// 3 args {{{2
template <typename V, typename I, bool RO>
inline void InterleavedMemoryAccessBase<V, I, RO>::interleave(const typename V::AsArg v0,
                                                              const typename V::AsArg v1,
                                                              const typename V::AsArg v2)
{
    InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data, m_indexes, v0, v1, v2);
}
// 4 args {{{2
template <typename V, typename I, bool RO>
inline void InterleavedMemoryAccessBase<V, I, RO>::interleave(const typename V::AsArg v0,
                                                              const typename V::AsArg v1,
                                                              const typename V::AsArg v2,
                                                              const typename V::AsArg v3)
{
    InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data, m_indexes, v0, v1, v2, v3);
}
// 5 args {{{2
template <typename V, typename I, bool RO>
inline void InterleavedMemoryAccessBase<V, I, RO>::interleave(const typename V::AsArg v0,
                                                              const typename V::AsArg v1,
                                                              const typename V::AsArg v2,
                                                              const typename V::AsArg v3,
                                                              const typename V::AsArg v4)
{
    InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data, m_indexes, v0, v1, v2, v3);
    v4.scatter(m_data + 4, m_indexes);
}
// 6 args {{{2
template <typename V, typename I, bool RO>
inline void InterleavedMemoryAccessBase<V, I, RO>::interleave(
    const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2,
    const typename V::AsArg v3, const typename V::AsArg v4, const typename V::AsArg v5)
{
    InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data, m_indexes, v0, v1, v2, v3);
    InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data + 4, m_indexes, v4, v5);
}
// 7 args {{{2
template <typename V, typename I, bool RO>
inline void InterleavedMemoryAccessBase<V, I, RO>::interleave(
    const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2,
    const typename V::AsArg v3, const typename V::AsArg v4, const typename V::AsArg v5,
    const typename V::AsArg v6)
{
    InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data + 0, m_indexes, v0, v1, v2,
                                                      v3);
    InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data + 4, m_indexes, v4, v5, v6);
}
// 8 args {{{2
template <typename V, typename I, bool RO>
inline void InterleavedMemoryAccessBase<V, I, RO>::interleave(
    const typename V::AsArg v0, const typename V::AsArg v1, const typename V::AsArg v2,
    const typename V::AsArg v3, const typename V::AsArg v4, const typename V::AsArg v5,
    const typename V::AsArg v6, const typename V::AsArg v7)
{
    InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data + 0, m_indexes, v0, v1, v2,
                                                      v3);
    InterleaveImpl<V, V::Size, sizeof(V)>::interleave(m_data + 4, m_indexes, v4, v5, v6,
                                                      v7);
}

// InterleavedMemoryAccessBase::deinterleave {{{1
// 2 args {{{2
template <typename V, typename I, bool RO>
inline void InterleavedMemoryAccessBase<V, I, RO>::deinterleave(V &v0, V &v1) const
{
    InterleaveImpl<V, V::Size, sizeof(V)>::deinterleave(m_data, m_indexes, v0, v1);
}
// 3 args {{{2
template <typename V, typename I, bool RO>
inline void InterleavedMemoryAccessBase<V, I, RO>::deinterleave(V &v0, V &v1, V &v2) const
{
    InterleaveImpl<V, V::Size, sizeof(V)>::deinterleave(m_data, m_indexes, v0, v1, v2);
}
// 4 args {{{2
template <typename V, typename I, bool RO>
inline void InterleavedMemoryAccessBase<V, I, RO>::deinterleave(V &v0, V &v1, V &v2,
                                                                V &v3) const
{
    InterleaveImpl<V, V::Size, sizeof(V)>::deinterleave(m_data, m_indexes, v0, v1, v2,
                                                        v3);
}
// 5 args {{{2
template <typename V, typename I, bool RO>
inline void InterleavedMemoryAccessBase<V, I, RO>::deinterleave(V &v0, V &v1, V &v2,
                                                                V &v3, V &v4) const
{
    InterleaveImpl<V, V::Size, sizeof(V)>::deinterleave(m_data, m_indexes, v0, v1, v2, v3,
                                                        v4);
}
// 6 args {{{2
template <typename V, typename I, bool RO>
inline void InterleavedMemoryAccessBase<V, I, RO>::deinterleave(V &v0, V &v1, V &v2,
                                                                V &v3, V &v4, V &v5) const
{
    InterleaveImpl<V, V::Size, sizeof(V)>::deinterleave(m_data, m_indexes, v0, v1, v2, v3,
                                                        v4, v5);
}
// 7 args {{{2
template <typename V, typename I, bool RO>
inline void InterleavedMemoryAccessBase<V, I, RO>::deinterleave(V &v0, V &v1, V &v2,
                                                                V &v3, V &v4, V &v5,
                                                                V &v6) const
{
    InterleaveImpl<V, V::Size, sizeof(V)>::deinterleave(m_data, m_indexes, v0, v1, v2, v3,
                                                        v4, v5, v6);
}
// 8 args {{{2
template <typename V, typename I, bool RO>
inline void InterleavedMemoryAccessBase<V, I, RO>::deinterleave(V &v0, V &v1, V &v2,
                                                                V &v3, V &v4, V &v5,
                                                                V &v6, V &v7) const
{
    InterleaveImpl<V, V::Size, sizeof(V)>::deinterleave(m_data, m_indexes, v0, v1, v2, v3,
                                                        v4, v5, v6, v7);
}
// }}}1
}
}

#include "undomacros.h"

#endif // VC_AVX_INTERLEAVEDMEMORY_TCC

// vim: foldmethod=marker
