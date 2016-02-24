/*  This file is part of the Vc library. {{{
Copyright © 2009-2015 Matthias Kretz <kretz@kde.org>
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

#ifndef VC_MIC_VECTORHELPER_H_
#define VC_MIC_VECTORHELPER_H_

#include "types.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace MIC
{

template<typename T> struct VectorHelper
{
    typedef typename VectorTypeHelper<T>::Type VectorType;
    typedef typename MaskTypeHelper<T>::Type MaskType;
    static Vc_INTRINSIC VectorType zero();
    static Vc_INTRINSIC VectorType one();
};

#define Vc_OP1(op) \
        static Vc_INTRINSIC VectorType op(const VectorType &a) { return Vc_CAT2(_mm512_##op##_, Vc_SUFFIX)(a); } \
        static Vc_INTRINSIC VectorType op(const VectorType &a, const __mmask &k) { return Vc_CAT2(_mm512_mask_##op##_, Vc_SUFFIX)(a, k, a); } \
        static Vc_INTRINSIC VectorType op(const VectorType &a, const __mmask &k, const VectorType &o) { return Vc_CAT2(_mm512_mask_##op##_, Vc_SUFFIX)(o, k, a); } \
        static Vc_INTRINSIC VectorType op##_s(unsigned int , const VectorType &a) { return Vc_CAT2(_mm512_##op##_, Vc_SUFFIX)(a); } \
        static Vc_INTRINSIC VectorType op##_s(unsigned int , const VectorType &a, const __mmask &k) { return Vc_CAT2(_mm512_mask_##op##_, Vc_SUFFIX)(a, k, a); } \
        static Vc_INTRINSIC VectorType op##_s(unsigned int , const VectorType &a, const __mmask &k, const VectorType &o) { return Vc_CAT2(_mm512_mask_##op##_, Vc_SUFFIX)(o, k, a); }
#define Vc_OP(op) \
        static Vc_INTRINSIC VectorType op(const VectorType &a, const VectorType &b) { return Vc_CAT2(_mm512_##op##_, Vc_SUFFIX)(a, b); } \
        static Vc_INTRINSIC VectorType op(const VectorType &a, const VectorType &b, const __mmask &k) { return Vc_CAT2(_mm512_mask_##op##_, Vc_SUFFIX)(a, k, a, b); } \
        static Vc_INTRINSIC VectorType op(const VectorType &a, const VectorType &b, const __mmask &k, const VectorType &o) { return Vc_CAT2(_mm512_mask_##op##_, Vc_SUFFIX)(o, k, a, b); } \
        static Vc_INTRINSIC VectorType op##_s(unsigned int , const VectorType &a, const VectorType &b) { return Vc_CAT2(_mm512_##op##_, Vc_SUFFIX)(a, b); } \
        static Vc_INTRINSIC VectorType op##_s(unsigned int , const VectorType &a, const VectorType &b, const __mmask &k) { return Vc_CAT2(_mm512_mask_##op##_, Vc_SUFFIX)(a, k, a, b); } \
        static Vc_INTRINSIC VectorType op##_s(unsigned int , const VectorType &a, const VectorType &b, const __mmask &k, const VectorType &o) { return Vc_CAT2(_mm512_mask_##op##_, Vc_SUFFIX)(o, k, a, b); }
#define Vc_OPx(op, op2) \
        static Vc_INTRINSIC VectorType op(const VectorType &a, const VectorType &b) { return Vc_CAT2(_mm512_##op2##_, Vc_SUFFIX)(a, b); } \
        static Vc_INTRINSIC VectorType op(const VectorType &a, const VectorType &b, const __mmask &k) { return Vc_CAT2(_mm512_mask_##op2##_, Vc_SUFFIX)(a, k, a, b); } \
        static Vc_INTRINSIC VectorType op(const VectorType &a, const VectorType &b, const __mmask &k, const VectorType &o) { return Vc_CAT2(_mm512_mask_##op2##_, Vc_SUFFIX)(o, k, a, b); } \
        static Vc_INTRINSIC VectorType op##_s(unsigned int , const VectorType &a, const VectorType &b) { return Vc_CAT2(_mm512_##op2##_, Vc_SUFFIX)(a, b); } \
        static Vc_INTRINSIC VectorType op##_s(unsigned int , const VectorType &a, const VectorType &b, const __mmask &k) { return Vc_CAT2(_mm512_mask_##op2##_, Vc_SUFFIX)(a, k, a, b); } \
        static Vc_INTRINSIC VectorType op##_s(unsigned int , const VectorType &a, const VectorType &b, const __mmask &k, const VectorType &o) { return Vc_CAT2(_mm512_mask_##op2##_, Vc_SUFFIX)(o, k, a, b); }

template<> struct VectorHelper<double> {
    typedef double EntryType;
    typedef __m512d VectorType;
#define Vc_SUFFIX pd
    template<typename A> static Vc_INTRINSIC VectorType load(const EntryType *x, A);
    template<typename A> static Vc_INTRINSIC void store(EntryType *mem, VectorType x, A);
    template<typename A> static Vc_INTRINSIC void store(EntryType *mem, VectorType x, __mmask8 k, A);

    static Vc_INTRINSIC VectorType zero() { return Vc_CAT2(_mm512_setzero_, Vc_SUFFIX)(); }
    static Vc_INTRINSIC VectorType set(EntryType x) { return Vc_CAT2(_mm512_set_1to8_, Vc_SUFFIX)(x); }

    static Vc_INTRINSIC EntryType reduce_max(const VectorType &a) { return _mm512_reduce_max_pd(a); }
    static Vc_INTRINSIC EntryType reduce_min(const VectorType &a) { return _mm512_reduce_min_pd(a); }
    static Vc_INTRINSIC EntryType reduce_mul(const VectorType &a) { return _mm512_reduce_mul_pd(a); }
    static Vc_INTRINSIC EntryType reduce_add(const VectorType &a) { return _mm512_reduce_add_pd(a); }

    static Vc_INTRINSIC VectorType abs(VectorType a) {
        const __m512i absMask = _mm512_set_4to16_epi32(0xffffffff, 0x7fffffff, 0xffffffff, 0x7fffffff);
        return mic_cast<VectorType>(_mm512_and_epi32(mic_cast<__m512i>(a), absMask));
    }

    Vc_OP(max) Vc_OP(min)
    Vc_OP1(sqrt) Vc_OP1(rsqrt) Vc_OP1(recip)
    Vc_OP(pow)
    Vc_OP1(sin) Vc_OP1(sinh) Vc_OP1(asin)
    Vc_OP1(cos) Vc_OP1(cosh) Vc_OP1(acos)
    Vc_OP1(tan) Vc_OP1(tanh) Vc_OP1(atan) Vc_OP(atan2)
    Vc_OP1(log) Vc_OP1(log2) Vc_OP1(log10)
    Vc_OP1(exp) Vc_OP1(exp2)
    Vc_OP1(floor) Vc_OP1(ceil)
    static Vc_INTRINSIC VectorType round(VectorType x) { return _mm512_roundfxpnt_adjust_pd(x, _MM_FROUND_TO_NEAREST_INT, _MM_EXPADJ_NONE); }
#undef Vc_SUFFIX
};

template<> struct VectorHelper<float> {
    typedef float EntryType;
    typedef __m512 VectorType;
#define Vc_SUFFIX ps

    template<typename T2, typename A> static Vc_INTRINSIC VectorType load(const T2 *x, A);
    template<typename T2, typename A> static Vc_INTRINSIC void store(T2 *mem, VectorType x, A);
    template<typename T2, typename A> static Vc_INTRINSIC void store(T2 *mem, VectorType x, __mmask16 k, A);

    static Vc_INTRINSIC VectorType zero() { return Vc_CAT2(_mm512_setzero_, Vc_SUFFIX)(); }
    static Vc_INTRINSIC VectorType set(EntryType x) { return Vc_CAT2(_mm512_set_1to16_, Vc_SUFFIX)(x); }

    static Vc_INTRINSIC EntryType reduce_max(const VectorType &a) { return _mm512_reduce_max_ps(a); }
    static Vc_INTRINSIC EntryType reduce_min(const VectorType &a) { return _mm512_reduce_min_ps(a); }
    static Vc_INTRINSIC EntryType reduce_mul(const VectorType &a) { return _mm512_reduce_mul_ps(a); }
    static Vc_INTRINSIC EntryType reduce_add(const VectorType &a) { return _mm512_reduce_add_ps(a); }

    static Vc_INTRINSIC VectorType abs(VectorType a) {
        const __m512i absMask = _mm512_set_1to16_epi32(0x7fffffff);
        return mic_cast<VectorType>(_mm512_and_epi32(mic_cast<__m512i>(a), absMask));
    }

    Vc_OP(max) Vc_OP(min)
    Vc_OP1(sqrt) Vc_OP1(rsqrt) Vc_OP1(recip)
    Vc_OP(pow)
    Vc_OP1(sin) Vc_OP1(sinh) Vc_OP1(asin)
    Vc_OP1(cos) Vc_OP1(cosh) Vc_OP1(acos)
    Vc_OP1(tan) Vc_OP1(tanh) Vc_OP1(atan) Vc_OP(atan2)
    Vc_OP1(log) Vc_OP1(log2) Vc_OP1(log10)
    Vc_OP1(exp) Vc_OP1(exp2)
    Vc_OP1(floor) Vc_OP1(ceil)
    static Vc_INTRINSIC VectorType round(VectorType x) { return _mm512_round_ps(x, _MM_FROUND_TO_NEAREST_INT, _MM_EXPADJ_NONE); }
#undef Vc_SUFFIX
};

template<> struct VectorHelper<int> {
    typedef int EntryType;
    typedef __m512i VectorType;
#define Vc_SUFFIX epi32
    template<typename T2, typename A> static Vc_INTRINSIC VectorType load(const T2 *x, A);
    template<typename T2, typename A> static Vc_INTRINSIC void store(T2 *mem, VectorType x, A);
    template<typename T2, typename A> static Vc_INTRINSIC void store(T2 *mem, VectorType x, __mmask16 k, A);

    static Vc_INTRINSIC VectorType set(EntryType x) { return Vc_CAT2(_mm512_set_1to16_, Vc_SUFFIX)(x); }

    static Vc_INTRINSIC EntryType reduce_max(const VectorType &a) { return _mm512_reduce_max_epi32(a); }
    static Vc_INTRINSIC EntryType reduce_min(const VectorType &a) { return _mm512_reduce_min_epi32(a); }
    static Vc_INTRINSIC EntryType reduce_mul(const VectorType &a) { return _mm512_reduce_mul_epi32(a); }
    static Vc_INTRINSIC EntryType reduce_add(const VectorType &a) { return _mm512_reduce_add_epi32(a); }

    static Vc_INTRINSIC VectorType abs(VectorType a) {
        VectorType zero = mic_cast<VectorType>(_mm512_setzero());
        const VectorType minusOne = _mm512_set_1to16_epi32( -1 );
        return _mm512_mask_mullo_epi32(a, _mm512_cmplt_epi32_mask(a, zero), a, minusOne);
    }

    Vc_OP(max) Vc_OP(min)
    Vc_OP(sllv) Vc_OP(srlv)
#undef Vc_SUFFIX
    static Vc_INTRINSIC VectorType round(VectorType x) { return x; }
};

template<> struct VectorHelper<unsigned int> {
    typedef unsigned int EntryType;
    typedef __m512i VectorType;
#define Vc_SUFFIX epu32
    template<typename T2, typename A> static Vc_INTRINSIC VectorType load(const T2 *x, A);
    template<typename T2, typename A> static Vc_INTRINSIC void store(T2 *mem, VectorType x, A);
    template<typename T2, typename A> static Vc_INTRINSIC void store(T2 *mem, VectorType x, __mmask16 k, A);

    static Vc_INTRINSIC EntryType reduce_max(const VectorType &a) { return _mm512_reduce_max_epi32(a); }
    static Vc_INTRINSIC EntryType reduce_min(const VectorType &a) { return _mm512_reduce_min_epi32(a); }
    static Vc_INTRINSIC EntryType reduce_mul(const VectorType &a) { return _mm512_reduce_mul_epi32(a); }
    static Vc_INTRINSIC EntryType reduce_add(const VectorType &a) { return _mm512_reduce_add_epi32(a); }

    Vc_OP(max) Vc_OP(min)
#undef Vc_SUFFIX
#define Vc_SUFFIX epi32
    static Vc_INTRINSIC VectorType set(EntryType x) { return Vc_CAT2(_mm512_set_1to16_, Vc_SUFFIX)(static_cast<int>(x)); }

    Vc_OP(sllv) Vc_OP(srlv)
#undef Vc_SUFFIX
    static Vc_INTRINSIC VectorType round(VectorType x) { return x; }
};
#undef Vc_OP
#undef Vc_OPx

}  // namespace MIC
}  // namespace Vc

#include "vectorhelper.tcc"

#endif // VC_MIC_VECTORHELPER_H_
