/*  This file is part of the Vc library. {{{

    Copyright (C) 2009-2012 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_MIC_VECTORHELPER_H
#define VC_MIC_VECTORHELPER_H

#include "types.h"
#include "macros.h"

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)

template<typename T> struct VectorHelper
{
    typedef typename VectorTypeHelper<T>::Type VectorType;
    typedef typename MaskTypeHelper<T>::Type MaskType;
    static Vc_INTRINSIC VectorType zero();
    static Vc_INTRINSIC VectorType one();
};

#define OP1(op) \
        static Vc_INTRINSIC VectorType op(const VectorType &a) { return CAT(_mm512_##op##_, SUFFIX)(a); } \
        static Vc_INTRINSIC VectorType op(const VectorType &a, const __mmask &k) { return CAT(_mm512_mask_##op##_, SUFFIX)(a, k, a); } \
        static Vc_INTRINSIC VectorType op(const VectorType &a, const __mmask &k, const VectorType &o) { return CAT(_mm512_mask_##op##_, SUFFIX)(o, k, a); } \
        static Vc_INTRINSIC VectorType op##_s(unsigned int , const VectorType &a) { return CAT(_mm512_##op##_, SUFFIX)(a); } \
        static Vc_INTRINSIC VectorType op##_s(unsigned int , const VectorType &a, const __mmask &k) { return CAT(_mm512_mask_##op##_, SUFFIX)(a, k, a); } \
        static Vc_INTRINSIC VectorType op##_s(unsigned int , const VectorType &a, const __mmask &k, const VectorType &o) { return CAT(_mm512_mask_##op##_, SUFFIX)(o, k, a); }
#define OP(op) \
        static Vc_INTRINSIC VectorType op(const VectorType &a, const VectorType &b) { return CAT(_mm512_##op##_, SUFFIX)(a, b); } \
        static Vc_INTRINSIC VectorType op(const VectorType &a, const VectorType &b, const __mmask &k) { return CAT(_mm512_mask_##op##_, SUFFIX)(a, k, a, b); } \
        static Vc_INTRINSIC VectorType op(const VectorType &a, const VectorType &b, const __mmask &k, const VectorType &o) { return CAT(_mm512_mask_##op##_, SUFFIX)(o, k, a, b); } \
        static Vc_INTRINSIC VectorType op##_s(unsigned int , const VectorType &a, const VectorType &b) { return CAT(_mm512_##op##_, SUFFIX)(a, b); } \
        static Vc_INTRINSIC VectorType op##_s(unsigned int , const VectorType &a, const VectorType &b, const __mmask &k) { return CAT(_mm512_mask_##op##_, SUFFIX)(a, k, a, b); } \
        static Vc_INTRINSIC VectorType op##_s(unsigned int , const VectorType &a, const VectorType &b, const __mmask &k, const VectorType &o) { return CAT(_mm512_mask_##op##_, SUFFIX)(o, k, a, b); }
#define OP_(op) \
        static Vc_INTRINSIC VectorType op(const VectorType &a, const VectorType &b) { return CAT(_mm512_##op, SUFFIX)(a, b); } \
        static Vc_INTRINSIC VectorType op(const VectorType &a, const VectorType &b, const __mmask &k) { return CAT(_mm512_mask_##op, SUFFIX)(a, k, a, b); } \
        static Vc_INTRINSIC VectorType op(const VectorType &a, const VectorType &b, const __mmask &k, const VectorType &o) { return CAT(_mm512_mask_##op, SUFFIX)(o, k, a, b); } \
        static Vc_INTRINSIC VectorType op##_s(unsigned int , const VectorType &a, const VectorType &b) { return CAT(_mm512_##op, SUFFIX)(a, b); } \
        static Vc_INTRINSIC VectorType op##_s(unsigned int , const VectorType &a, const VectorType &b, const __mmask &k) { return CAT(_mm512_mask_##op, SUFFIX)(a, k, a, b); } \
        static Vc_INTRINSIC VectorType op##_s(unsigned int , const VectorType &a, const VectorType &b, const __mmask &k, const VectorType &o) { return CAT(_mm512_mask_##op, SUFFIX)(o, k, a, b); }
#define OPx(op, op2) \
        static Vc_INTRINSIC VectorType op(const VectorType &a, const VectorType &b) { return CAT(_mm512_##op2##_, SUFFIX)(a, b); } \
        static Vc_INTRINSIC VectorType op(const VectorType &a, const VectorType &b, const __mmask &k) { return CAT(_mm512_mask_##op2##_, SUFFIX)(a, k, a, b); } \
        static Vc_INTRINSIC VectorType op(const VectorType &a, const VectorType &b, const __mmask &k, const VectorType &o) { return CAT(_mm512_mask_##op2##_, SUFFIX)(o, k, a, b); } \
        static Vc_INTRINSIC VectorType op##_s(unsigned int , const VectorType &a, const VectorType &b) { return CAT(_mm512_##op2##_, SUFFIX)(a, b); } \
        static Vc_INTRINSIC VectorType op##_s(unsigned int , const VectorType &a, const VectorType &b, const __mmask &k) { return CAT(_mm512_mask_##op2##_, SUFFIX)(a, k, a, b); } \
        static Vc_INTRINSIC VectorType op##_s(unsigned int , const VectorType &a, const VectorType &b, const __mmask &k, const VectorType &o) { return CAT(_mm512_mask_##op2##_, SUFFIX)(o, k, a, b); }
#define OPcmp(op, _enum_) \
        static Vc_INTRINSIC __mmask cmp##op(VectorType a, VectorType b) { return _VC_CAT(_mm512_cmp_, SUFFIX, _mask,)(a, b, _enum_); } \
        static Vc_INTRINSIC __mmask cmp##op(VectorType a, VectorType b, __mmask k) { return _VC_CAT(_mm512_mask_cmp_, SUFFIX, _mask,)(k, a, b, _enum_); }

template<> struct VectorHelper<double> {
    typedef double EntryType;
    typedef __m512d VectorType;
#define SUFFIX pd
    // double doesn't support any upconversion
    static Vc_INTRINSIC VectorType load1(const EntryType  x) {
        return _mm512_extload_pd(&x, _MM_UPCONV_PD_NONE, _MM_BROADCAST_1X8, _MM_HINT_NONE);
    }
    static Vc_INTRINSIC VectorType load4(const EntryType *x) {
        return _mm512_extload_pd(&x, _MM_UPCONV_PD_NONE, _MM_BROADCAST_4X8, _MM_HINT_NONE);
    }

    template<typename A> static Vc_INTRINSIC VectorType load(const EntryType *x, A);
    template<typename A> static Vc_INTRINSIC void store(EntryType *mem, VectorType x, A);
    template<typename A> static Vc_INTRINSIC void store(EntryType *mem, VectorType x, __mmask8 k, A);

    static Vc_INTRINSIC VectorType zero() { return CAT(_mm512_setzero_, SUFFIX)(); }
    static Vc_INTRINSIC VectorType set(EntryType x) { return CAT(_mm512_set_1to8_, SUFFIX)(x); }

    static Vc_INTRINSIC VectorType multiplyAndAdd(const VectorType &v1, const VectorType &v2, const VectorType &v3) { return _mm512_fmadd_pd(v1, v2, v3); }
    //static Vc_INTRINSIC VectorType multiplyAndAdd(const VectorType &v1, const VectorType &v2, const VectorType &v3, const __mmask8 &k) { return _mm512_mask_fmadd_pd(v1, k, v2, v3); }
    static Vc_INTRINSIC VectorType multiplyAndSub(const VectorType &v1, const VectorType &v2, const VectorType &v3) { return _mm512_fmsub_pd(v1, v2, v3); }

    static Vc_INTRINSIC EntryType reduce_max(const VectorType &a) { return _mm512_reduce_max_pd(a); }
    static Vc_INTRINSIC EntryType reduce_min(const VectorType &a) { return _mm512_reduce_min_pd(a); }
    static Vc_INTRINSIC EntryType reduce_mul(const VectorType &a) { return _mm512_reduce_mul_pd(a); }
    static Vc_INTRINSIC EntryType reduce_add(const VectorType &a) { return _mm512_reduce_add_pd(a); }

    static Vc_INTRINSIC VectorType abs(VectorType a) {
        const __m512i absMask = _mm512_set_4to16_epi32(0xffffffff, 0x7fffffff, 0xffffffff, 0x7fffffff);
        return mic_cast<VectorType>(_mm512_and_epi32(mic_cast<__m512i>(a), absMask));
    }

    OP(max) OP(min)
    OP1(sqrt) OP1(rsqrt) OP1(recip)
    OP(pow)
    OP1(sin) OP1(sinh) OP1(asin)
    OP1(cos) OP1(cosh) OP1(acos)
    OP1(tan) OP1(tanh) OP1(atan) OP(atan2)
    OP1(log) OP1(log2) OP1(log10)
    OP1(exp) OP1(exp2)
    OP1(floor) OP1(ceil)
    OP(add) OP(sub) OP(mul) OP(div)
    OPcmp( eq, _CMP_EQ_OQ)
    OPcmp(neq, _CMP_NEQ_UQ)
    OPcmp( lt, _CMP_LT_OS)
    OPcmp(nlt, _CMP_NLT_US)
    OPcmp( le, _CMP_LE_OS)
    OPcmp(nle, _CMP_NLE_US)
    static Vc_INTRINSIC VectorType round(VectorType x) { return _mm512_roundfxpnt_adjust_pd(x, _MM_FROUND_TO_NEAREST_INT, _MM_EXPADJ_NONE); }
#undef SUFFIX
};

template<> struct VectorHelper<float> {
    typedef float EntryType;
    typedef __m512 VectorType;
#define SUFFIX ps

    template<typename T2, typename A> static Vc_INTRINSIC VectorType load(const T2 *x, A);
    template<typename T2, typename A> static Vc_INTRINSIC void store(T2 *mem, VectorType x, A);
    template<typename T2, typename A> static Vc_INTRINSIC void store(T2 *mem, VectorType x, __mmask16 k, A);

    static Vc_INTRINSIC VectorType zero() { return CAT(_mm512_setzero_, SUFFIX)(); }
    static Vc_INTRINSIC VectorType set(EntryType x) { return CAT(_mm512_set_1to16_, SUFFIX)(x); }

    static Vc_INTRINSIC VectorType multiplyAndAdd(const VectorType &v1, const VectorType &v2, const VectorType &v3) { return _mm512_fmadd_ps(v1, v2, v3); }
    //static Vc_INTRINSIC VectorType multiplyAndAdd(const VectorType &v1, const VectorType &v2, const VectorType &v3, const __mmask16 &k) { return _mm512_mask_fmadd_ps(v1, k, v2, v3); }
    static Vc_INTRINSIC VectorType multiplyAndSub(const VectorType &v1, const VectorType &v2, const VectorType &v3) { return _mm512_fmsub_ps(v1, v2, v3); }

    static Vc_INTRINSIC EntryType reduce_max(const VectorType &a) { return _mm512_reduce_max_ps(a); }
    static Vc_INTRINSIC EntryType reduce_min(const VectorType &a) { return _mm512_reduce_min_ps(a); }
    static Vc_INTRINSIC EntryType reduce_mul(const VectorType &a) { return _mm512_reduce_mul_ps(a); }
    static Vc_INTRINSIC EntryType reduce_add(const VectorType &a) { return _mm512_reduce_add_ps(a); }

    static Vc_INTRINSIC VectorType abs(VectorType a) {
        const __m512i absMask = _mm512_set_1to16_epi32(0x7fffffff);
        return mic_cast<VectorType>(_mm512_and_epi32(mic_cast<__m512i>(a), absMask));
    }

    OP(max) OP(min)
    OP1(sqrt) OP1(rsqrt) OP1(recip)
    OP(pow)
    OP1(sin) OP1(sinh) OP1(asin)
    OP1(cos) OP1(cosh) OP1(acos)
    OP1(tan) OP1(tanh) OP1(atan) OP(atan2)
    OP1(log) OP1(log2) OP1(log10)
    OP1(exp) OP1(exp2)
    OP1(floor) OP1(ceil)
    OP(add) OP(sub) OP(mul) OP(div)
    OPcmp( eq, _CMP_EQ_OQ)
    OPcmp(neq, _CMP_NEQ_UQ)
    OPcmp( lt, _CMP_LT_OS)
    OPcmp(nlt, _CMP_NLT_US)
    OPcmp( le, _CMP_LE_OS)
    OPcmp(nle, _CMP_NLE_US)
    static Vc_INTRINSIC VectorType round(VectorType x) { return _mm512_round_ps(x, _MM_FROUND_TO_NEAREST_INT, _MM_EXPADJ_NONE); }
#undef SUFFIX
};

template<> struct VectorHelper<int> {
    typedef int EntryType;
    typedef __m512i VectorType;
#define SUFFIX epi32
    template<typename T2, typename A> static Vc_INTRINSIC VectorType load(const T2 *x, A);
    template<typename T2, typename A> static Vc_INTRINSIC void store(T2 *mem, VectorType x, A);
    template<typename T2, typename A> static Vc_INTRINSIC void store(T2 *mem, VectorType x, __mmask16 k, A);

    static Vc_INTRINSIC VectorType set(EntryType x) { return CAT(_mm512_set_1to16_, SUFFIX)(x); }

    static Vc_INTRINSIC VectorType multiplyAndAdd(const VectorType &v1, const VectorType &v2, const VectorType &v3) { return _mm512_fmadd_epi32(v1, v2, v3); }
    //static Vc_INTRINSIC VectorType multiplyAndAdd(const VectorType &v1, const VectorType &v2, const VectorType &v3, const __mmask16 &k) { return _mm512_mask_fmadd_epi32(v1, k, v2, v3); }
    //static Vc_INTRINSIC VectorType multiplyAndSub(const VectorType &v1, const VectorType &v2, const VectorType &v3) { return _mm512_sub_epi32(_mm512_mullo_epi32(v1, v2), v3); }

    static Vc_INTRINSIC EntryType reduce_max(const VectorType &a) { return _mm512_reduce_max_epi32(a); }
    static Vc_INTRINSIC EntryType reduce_min(const VectorType &a) { return _mm512_reduce_min_epi32(a); }
    static Vc_INTRINSIC EntryType reduce_mul(const VectorType &a) { return _mm512_reduce_mul_epi32(a); }
    static Vc_INTRINSIC EntryType reduce_add(const VectorType &a) { return _mm512_reduce_add_epi32(a); }

    static Vc_INTRINSIC VectorType abs(VectorType a) {
        VectorType zero = mic_cast<VectorType>(_mm512_setzero());
        const VectorType minusOne = _mm512_set_1to16_epi32( -1 );
        return mul(a, minusOne, cmplt(a, zero), a);
    }

    OP(max) OP(min)
    OP(add) OP(sub) OPx(mul, mullo) OP(div) OP(rem)
    OP_(or_) OP_(and_) OP_(xor_)
    OPcmp( eq, _MM_CMPINT_EQ)
    OPcmp(neq, _MM_CMPINT_NE)
    OPcmp( lt, _MM_CMPINT_LT)
    OPcmp(nlt, _MM_CMPINT_NLT)
    OPcmp( le, _MM_CMPINT_LE)
    OPcmp(nle, _MM_CMPINT_NLE)
    OP(sllv) OP(srlv)
#undef SUFFIX
    static Vc_INTRINSIC VectorType round(VectorType x) { return x; }
};

template<> struct VectorHelper<unsigned int> {
    typedef unsigned int EntryType;
    typedef __m512i VectorType;
#define SUFFIX epu32
    template<typename T2, typename A> static Vc_INTRINSIC VectorType load(const T2 *x, A);
    template<typename T2, typename A> static Vc_INTRINSIC void store(T2 *mem, VectorType x, A);
    template<typename T2, typename A> static Vc_INTRINSIC void store(T2 *mem, VectorType x, __mmask16 k, A);

    static Vc_INTRINSIC VectorType multiplyAndAdd(const VectorType &v1, const VectorType &v2, const VectorType &v3) { return _mm512_fmadd_epi32(v1, v2, v3); }
    //static Vc_INTRINSIC VectorType multiplyAndAdd(const VectorType &v1, const VectorType &v2, const VectorType &v3, const __mmask16 &k) { return _mm512_mask_fmadd_epi32(v1, k, v2, v3); }
    //static Vc_INTRINSIC VectorType multiplyAndSub(const VectorType &v1, const VectorType &v2, const VectorType &v3) { return _mm512_sub_epi32(_mm512_mullo_epi32(v1, v2), v3); }

    static Vc_INTRINSIC EntryType reduce_max(const VectorType &a) { return _mm512_reduce_max_epi32(a); }
    static Vc_INTRINSIC EntryType reduce_min(const VectorType &a) { return _mm512_reduce_min_epi32(a); }
    static Vc_INTRINSIC EntryType reduce_mul(const VectorType &a) { return _mm512_reduce_mul_epi32(a); }
    static Vc_INTRINSIC EntryType reduce_add(const VectorType &a) { return _mm512_reduce_add_epi32(a); }

    OP(max) OP(min)
    OP(div) OP(rem)
    OPcmp( eq, _MM_CMPINT_EQ)
    OPcmp(neq, _MM_CMPINT_NE)
    OPcmp( lt, _MM_CMPINT_LT)
    OPcmp(nlt, _MM_CMPINT_NLT)
    OPcmp( le, _MM_CMPINT_LE)
    OPcmp(nle, _MM_CMPINT_NLE)
#undef SUFFIX
#define SUFFIX epi32
    static Vc_INTRINSIC VectorType set(EntryType x) { return CAT(_mm512_set_1to16_, SUFFIX)(static_cast<int>(x)); }

    OP(sllv) OP(srlv)
    OP(add) OP(sub) OPx(mul, mullo)
    OP_(or_) OP_(and_) OP_(xor_)
#undef SUFFIX
    static Vc_INTRINSIC VectorType round(VectorType x) { return x; }
};
#undef OP
#undef OP_
#undef OPx
#undef OPcmp

Vc_NAMESPACE_END

#include "vectorhelper.tcc"

#endif // VC_MIC_VECTORHELPER_H
