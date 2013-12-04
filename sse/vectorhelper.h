/*  This file is part of the Vc library.

    Copyright (C) 2009-2011 Matthias Kretz <kretz@kde.org>

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

*/

#ifndef SSE_VECTORHELPER_H
#define SSE_VECTORHELPER_H

#include "types.h"
#include "../common/loadstoreflags.h"
#include <limits>
#include "const_data.h"
#include "macros.h"

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)

namespace Internal
{
Vc_INTRINSIC Vc_CONST __m128 exponent(__m128 v)
{
    __m128i tmp = _mm_srli_epi32(_mm_castps_si128(v), 23);
    tmp = _mm_sub_epi32(tmp, _mm_set1_epi32(0x7f));
    return _mm_cvtepi32_ps(tmp);
}
Vc_INTRINSIC Vc_CONST __m128d exponent(__m128d v)
{
    __m128i tmp = _mm_srli_epi64(_mm_castpd_si128(v), 52);
    tmp = _mm_sub_epi32(tmp, _mm_set1_epi32(0x3ff));
    return _mm_cvtepi32_pd(_mm_shuffle_epi32(tmp, 0x08));
}
} // namespace Internal

    template<typename VectorType, unsigned int Size> struct SortHelper
    {
        static inline Vc_CONST_L VectorType sort(VectorType) Vc_CONST_R;
    };

#define OP0(name, code) static Vc_ALWAYS_INLINE Vc_CONST VectorType name() { return code; }
#define OP1(name, code) static Vc_ALWAYS_INLINE Vc_CONST VectorType name(const VectorType a) { return code; }
#define OP2(name, code) static Vc_ALWAYS_INLINE Vc_CONST VectorType name(const VectorType a, const VectorType b) { return code; }
#define OP3(name, code) static Vc_ALWAYS_INLINE Vc_CONST VectorType name(const VectorType a, const VectorType b, const VectorType c) { return code; }

        template<> struct VectorHelper<_M128>
        {
            typedef _M128 VectorType;

            template<typename Flags> static Vc_ALWAYS_INLINE Vc_PURE VectorType load(const float *x, typename EnableIfAligned  <Flags>::type = nullptr) { return _mm_load_ps(x); }
            template<typename Flags> static Vc_ALWAYS_INLINE Vc_PURE VectorType load(const float *x, typename EnableIfUnaligned<Flags>::type = nullptr) { return _mm_loadu_ps(x); }
            template<typename Flags> static Vc_ALWAYS_INLINE Vc_PURE VectorType load(const float *x, typename EnableIfStreaming<Flags>::type = nullptr) { return _mm_stream_load(x); }

            template<typename Flags> static Vc_ALWAYS_INLINE void store(float *mem, VectorType x, typename EnableIfAligned              <Flags>::type = nullptr) { _mm_store_ps(mem, x); }
            template<typename Flags> static Vc_ALWAYS_INLINE void store(float *mem, VectorType x, typename EnableIfUnalignedNotStreaming<Flags>::type = nullptr) { _mm_storeu_ps(mem, x); }
            template<typename Flags> static Vc_ALWAYS_INLINE void store(float *mem, VectorType x, typename EnableIfStreaming            <Flags>::type = nullptr) { _mm_stream_ps(mem, x); }
            template<typename Flags> static Vc_ALWAYS_INLINE void store(float *mem, VectorType x, typename EnableIfUnalignedAndStreaming<Flags>::type = nullptr) { _mm_maskmoveu_si128(_mm_castps_si128(x), _mm_setallone_si128(), reinterpret_cast<char *>(mem)); }

            // before AVX there was only one maskstore. load -> blend -> store would break the C++ memory model (read/write of memory that is actually not touched by this thread)
            template<typename Flags> static Vc_ALWAYS_INLINE void store(float *mem, VectorType x, VectorType m) { _mm_maskmoveu_si128(_mm_castps_si128(x), _mm_castps_si128(m), reinterpret_cast<char *>(mem)); }

            OP0(allone, _mm_setallone_ps())
            OP0(zero, _mm_setzero_ps())
            OP2(or_, _mm_or_ps(a, b))
            OP2(xor_, _mm_xor_ps(a, b))
            OP2(and_, _mm_and_ps(a, b))
            OP2(andnot_, _mm_andnot_ps(a, b))
            OP3(blend, _mm_blendv_ps(a, b, c))
        };


        template<> struct VectorHelper<_M128D>
        {
            typedef _M128D VectorType;

            template<typename Flags> static Vc_ALWAYS_INLINE Vc_PURE VectorType load(const double *x, typename EnableIfAligned  <Flags>::type = nullptr) { return _mm_load_pd(x); }
            template<typename Flags> static Vc_ALWAYS_INLINE Vc_PURE VectorType load(const double *x, typename EnableIfUnaligned<Flags>::type = nullptr) { return _mm_loadu_pd(x); }
            template<typename Flags> static Vc_ALWAYS_INLINE Vc_PURE VectorType load(const double *x, typename EnableIfStreaming<Flags>::type = nullptr) { return _mm_stream_load(x); }

            template<typename Flags> static Vc_ALWAYS_INLINE void store(double *mem, VectorType x, typename EnableIfAligned              <Flags>::type = nullptr) { _mm_store_pd(mem, x); }
            template<typename Flags> static Vc_ALWAYS_INLINE void store(double *mem, VectorType x, typename EnableIfUnalignedNotStreaming<Flags>::type = nullptr) { _mm_storeu_pd(mem, x); }
            template<typename Flags> static Vc_ALWAYS_INLINE void store(double *mem, VectorType x, typename EnableIfStreaming            <Flags>::type = nullptr) { _mm_stream_pd(mem, x); }
            template<typename Flags> static Vc_ALWAYS_INLINE void store(double *mem, VectorType x, typename EnableIfUnalignedAndStreaming<Flags>::type = nullptr) { _mm_maskmoveu_si128(_mm_castpd_si128(x), _mm_setallone_si128(), reinterpret_cast<char *>(mem)); }

            // before AVX there was only one maskstore. load -> blend -> store would break the C++ memory model (read/write of memory that is actually not touched by this thread)
            template<typename Flags> static Vc_ALWAYS_INLINE void store(double *mem, VectorType x, VectorType m) { _mm_maskmoveu_si128(_mm_castpd_si128(x), _mm_castpd_si128(m), reinterpret_cast<char *>(mem)); }

            OP0(allone, _mm_setallone_pd())
            OP0(zero, _mm_setzero_pd())
            OP2(or_, _mm_or_pd(a, b))
            OP2(xor_, _mm_xor_pd(a, b))
            OP2(and_, _mm_and_pd(a, b))
            OP2(andnot_, _mm_andnot_pd(a, b))
            OP3(blend, _mm_blendv_pd(a, b, c))
        };

        template<> struct VectorHelper<_M128I>
        {
            typedef _M128I VectorType;

            template<typename Flags, typename T> static Vc_ALWAYS_INLINE Vc_PURE VectorType load(const T *x, typename EnableIfAligned  <Flags>::type = nullptr) { return _mm_load_si128(reinterpret_cast<const VectorType *>(x)); }
            template<typename Flags, typename T> static Vc_ALWAYS_INLINE Vc_PURE VectorType load(const T *x, typename EnableIfUnaligned<Flags>::type = nullptr) { return _mm_loadu_si128(reinterpret_cast<const VectorType *>(x)); }
            template<typename Flags, typename T> static Vc_ALWAYS_INLINE Vc_PURE VectorType load(const T *x, typename EnableIfStreaming<Flags>::type = nullptr) { return _mm_stream_load(x); }

            template<typename Flags, typename T> static Vc_ALWAYS_INLINE void store(T *mem, VectorType x, typename EnableIfAligned              <Flags>::type = nullptr) { _mm_store_si128(reinterpret_cast<VectorType *>(mem), x); }
            template<typename Flags, typename T> static Vc_ALWAYS_INLINE void store(T *mem, VectorType x, typename EnableIfUnalignedNotStreaming<Flags>::type = nullptr) { _mm_storeu_si128(reinterpret_cast<VectorType *>(mem), x); }
            template<typename Flags, typename T> static Vc_ALWAYS_INLINE void store(T *mem, VectorType x, typename EnableIfStreaming            <Flags>::type = nullptr) { _mm_stream_si128(reinterpret_cast<VectorType *>(mem), x); }
            template<typename Flags, typename T> static Vc_ALWAYS_INLINE void store(T *mem, VectorType x, typename EnableIfUnalignedAndStreaming<Flags>::type = nullptr) { _mm_maskmoveu_si128(x, _mm_setallone_si128(), reinterpret_cast<char *>(mem)); }

            // before AVX there was only one maskstore. load -> blend -> store would break the C++ memory model (read/write of memory that is actually not touched by this thread)
            template<typename Flags, typename T> static Vc_ALWAYS_INLINE void store(T *mem, VectorType x, VectorType m) { _mm_maskmoveu_si128(x, m, reinterpret_cast<char *>(mem)); }

            OP0(allone, _mm_setallone_si128())
            OP0(zero, _mm_setzero_si128())
            OP2(or_, _mm_or_si128(a, b))
            OP2(xor_, _mm_xor_si128(a, b))
            OP2(and_, _mm_and_si128(a, b))
            OP2(andnot_, _mm_andnot_si128(a, b))
            OP3(blend, _mm_blendv_epi8(a, b, c))
        };

#undef OP1
#undef OP2
#undef OP3

#define OP1(op) \
        static Vc_ALWAYS_INLINE Vc_CONST VectorType op(const VectorType a) { return CAT(_mm_##op##_, SUFFIX)(a); }
#define OP(op) \
        static Vc_ALWAYS_INLINE Vc_CONST VectorType op(const VectorType a, const VectorType b) { return CAT(_mm_##op##_ , SUFFIX)(a, b); }
#define OP_(op) \
        static Vc_ALWAYS_INLINE Vc_CONST VectorType op(const VectorType a, const VectorType b) { return CAT(_mm_##op    , SUFFIX)(a, b); }
#define OPx(op, op2) \
        static Vc_ALWAYS_INLINE Vc_CONST VectorType op(const VectorType a, const VectorType b) { return CAT(_mm_##op2##_, SUFFIX)(a, b); }
#define OPcmp(op) \
        static Vc_ALWAYS_INLINE Vc_CONST VectorType cmp##op(const VectorType a, const VectorType b) { return CAT(_mm_cmp##op##_, SUFFIX)(a, b); }
#define OP_CAST_(op) \
        static Vc_ALWAYS_INLINE Vc_CONST VectorType op(const VectorType a, const VectorType b) { return CAT(_mm_castps_, SUFFIX)( \
            _mm_##op##ps(CAT(CAT(_mm_cast, SUFFIX), _ps)(a), \
              CAT(CAT(_mm_cast, SUFFIX), _ps)(b))); \
        }
#define MINMAX \
        static Vc_ALWAYS_INLINE Vc_CONST VectorType min(VectorType a, VectorType b) { return CAT(_mm_min_, SUFFIX)(a, b); } \
        static Vc_ALWAYS_INLINE Vc_CONST VectorType max(VectorType a, VectorType b) { return CAT(_mm_max_, SUFFIX)(a, b); }

        template<> struct VectorHelper<double> {
            typedef _M128D VectorType;
            typedef double EntryType;
#define SUFFIX pd

            OP_(or_) OP_(and_) OP_(xor_)
            static Vc_ALWAYS_INLINE Vc_CONST VectorType notMaskedToZero(VectorType a, _M128 mask) { return CAT(_mm_and_, SUFFIX)(_mm_castps_pd(mask), a); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType set(const double a) { return CAT(_mm_set1_, SUFFIX)(a); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType set(const double a, const double b) { return CAT(_mm_set_, SUFFIX)(a, b); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType zero() { return CAT(_mm_setzero_, SUFFIX)(); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType one()  { return CAT(_mm_setone_, SUFFIX)(); }// set(1.); }

#ifdef VC_IMPL_FMA4
            static Vc_ALWAYS_INLINE void fma(VectorType &v1, VectorType v2, VectorType v3) {
                v1 = _mm_macc_pd(v1, v2, v3);
            }
#else
            static inline void fma(VectorType &v1, VectorType v2, VectorType v3) {
                VectorType h1 = _mm_and_pd(v1, _mm_load_pd(reinterpret_cast<const double *>(&c_general::highMaskDouble)));
                VectorType h2 = _mm_and_pd(v2, _mm_load_pd(reinterpret_cast<const double *>(&c_general::highMaskDouble)));
#if defined(VC_GCC) && VC_GCC < 0x40703
                // GCC before 4.7.3 uses an incorrect optimization where it replaces the subtraction with an andnot
                // http://gcc.gnu.org/bugzilla/show_bug.cgi?id=54703
                asm("":"+x"(h1), "+x"(h2));
#endif
                const VectorType l1 = _mm_sub_pd(v1, h1);
                const VectorType l2 = _mm_sub_pd(v2, h2);
                const VectorType ll = mul(l1, l2);
                const VectorType lh = add(mul(l1, h2), mul(h1, l2));
                const VectorType hh = mul(h1, h2);
                // ll < lh < hh for all entries is certain
                const VectorType lh_lt_v3 = cmplt(abs(lh), abs(v3)); // |lh| < |v3|
                const VectorType b = _mm_blendv_pd(v3, lh, lh_lt_v3);
                const VectorType c = _mm_blendv_pd(lh, v3, lh_lt_v3);
                v1 = add(add(ll, b), add(c, hh));
            }
#endif

            OP(add) OP(sub) OP(mul)
            OPcmp(eq) OPcmp(neq)
            OPcmp(lt) OPcmp(nlt)
            OPcmp(le) OPcmp(nle)

            OP1(sqrt)
            static Vc_ALWAYS_INLINE Vc_CONST VectorType rsqrt(VectorType x) {
                return _mm_div_pd(one(), sqrt(x));
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType reciprocal(VectorType x) {
                return _mm_div_pd(one(), x);
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType isNaN(VectorType x) {
                return _mm_cmpunord_pd(x, x);
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType isFinite(VectorType x) {
                return _mm_cmpord_pd(x, _mm_mul_pd(zero(), x));
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType isInfinite(VectorType x) {
                return _mm_castsi128_pd(_mm_cmpeq_epi64(_mm_castpd_si128(abs(x)), _mm_castpd_si128(_mm_load_pd(c_log<double>::d(1)))));
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType abs(const VectorType a) {
                return CAT(_mm_and_, SUFFIX)(a, _mm_setabsmask_pd());
            }

            MINMAX
            static Vc_ALWAYS_INLINE Vc_CONST EntryType min(VectorType a) {
                a = _mm_min_sd(a, _mm_unpackhi_pd(a, a));
                return _mm_cvtsd_f64(a);
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType max(VectorType a) {
                a = _mm_max_sd(a, _mm_unpackhi_pd(a, a));
                return _mm_cvtsd_f64(a);
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType mul(VectorType a) {
                a = _mm_mul_sd(a, _mm_shuffle_pd(a, a, _MM_SHUFFLE2(0, 1)));
                return _mm_cvtsd_f64(a);
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType add(VectorType a) {
                a = _mm_add_sd(a, _mm_shuffle_pd(a, a, _MM_SHUFFLE2(0, 1)));
                return _mm_cvtsd_f64(a);
            }
#undef SUFFIX
            static Vc_ALWAYS_INLINE Vc_CONST VectorType round(VectorType a) {
#ifdef VC_IMPL_SSE4_1
                return _mm_round_pd(a, _MM_FROUND_NINT);
#else
                //XXX: slow: _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
                return _mm_cvtepi32_pd(_mm_cvtpd_epi32(a));
#endif
            }
        };

        template<> struct VectorHelper<float> {
            typedef float EntryType;
            typedef _M128 VectorType;
#define SUFFIX ps

            OP_(or_) OP_(and_) OP_(xor_)
            static Vc_ALWAYS_INLINE Vc_CONST VectorType notMaskedToZero(VectorType a, _M128 mask) { return CAT(_mm_and_, SUFFIX)(mask, a); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType set(const float a) { return CAT(_mm_set1_, SUFFIX)(a); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType set(const float a, const float b, const float c, const float d) { return CAT(_mm_set_, SUFFIX)(a, b, c, d); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType zero() { return CAT(_mm_setzero_, SUFFIX)(); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType one()  { return CAT(_mm_setone_, SUFFIX)(); }// set(1.f); }
            static Vc_ALWAYS_INLINE Vc_CONST _M128 concat(_M128D a, _M128D b) { return _mm_movelh_ps(_mm_cvtpd_ps(a), _mm_cvtpd_ps(b)); }

#ifdef VC_IMPL_FMA4
            static Vc_ALWAYS_INLINE void fma(VectorType &v1, VectorType v2, VectorType v3) {
                v1 = _mm_macc_ps(v1, v2, v3);
            }
#else
            static inline void fma(VectorType &v1, VectorType v2, VectorType v3) {
                __m128d v1_0 = _mm_cvtps_pd(v1);
                __m128d v1_1 = _mm_cvtps_pd(_mm_movehl_ps(v1, v1));
                __m128d v2_0 = _mm_cvtps_pd(v2);
                __m128d v2_1 = _mm_cvtps_pd(_mm_movehl_ps(v2, v2));
                __m128d v3_0 = _mm_cvtps_pd(v3);
                __m128d v3_1 = _mm_cvtps_pd(_mm_movehl_ps(v3, v3));
                v1 = _mm_movelh_ps(
                        _mm_cvtpd_ps(_mm_add_pd(_mm_mul_pd(v1_0, v2_0), v3_0)),
                        _mm_cvtpd_ps(_mm_add_pd(_mm_mul_pd(v1_1, v2_1), v3_1)));
            }
#endif

            OP(add) OP(sub) OP(mul)
            OPcmp(eq) OPcmp(neq)
            OPcmp(lt) OPcmp(nlt)
            OPcmp(le) OPcmp(nle)

            OP1(sqrt) OP1(rsqrt)
            static Vc_ALWAYS_INLINE Vc_CONST VectorType isNaN(VectorType x) {
                return _mm_cmpunord_ps(x, x);
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType isFinite(VectorType x) {
                return _mm_cmpord_ps(x, _mm_mul_ps(zero(), x));
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType isInfinite(VectorType x) {
                return _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_castps_si128(abs(x)), _mm_castps_si128(_mm_load_ps(c_log<float>::d(1)))));
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType reciprocal(VectorType x) {
                return _mm_rcp_ps(x);
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType abs(const VectorType a) {
                return CAT(_mm_and_, SUFFIX)(a, _mm_setabsmask_ps());
            }

            MINMAX
            static Vc_ALWAYS_INLINE Vc_CONST EntryType min(VectorType a) {
                a = _mm_min_ps(a, _mm_movehl_ps(a, a));   // a = min(a0, a2), min(a1, a3), min(a2, a2), min(a3, a3)
                a = _mm_min_ss(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 1, 1, 1))); // a = min(a0, a1), a1, a2, a3
                return _mm_cvtss_f32(a);
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType max(VectorType a) {
                a = _mm_max_ps(a, _mm_movehl_ps(a, a));   // a = max(a0, a2), max(a1, a3), max(a2, a2), max(a3, a3)
                a = _mm_max_ss(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 1, 1, 1))); // a = max(a0, a1), a1, a2, a3
                return _mm_cvtss_f32(a);
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType mul(VectorType a) {
                a = _mm_mul_ps(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3)));
                a = _mm_mul_ss(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 2, 0, 1)));
                return _mm_cvtss_f32(a);
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType add(VectorType a) {
                a = _mm_add_ps(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3)));
                a = _mm_add_ss(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 2, 0, 1)));
                return _mm_cvtss_f32(a);
            }
#undef SUFFIX
            static Vc_ALWAYS_INLINE Vc_CONST VectorType round(VectorType a) {
#ifdef VC_IMPL_SSE4_1
                return _mm_round_ps(a, _MM_FROUND_NINT);
#else
                //XXX slow: _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
                return _mm_cvtepi32_ps(_mm_cvtps_epi32(a));
#endif
            }
        };

        template<> struct VectorHelper<int> {
            typedef int EntryType;
            typedef _M128I VectorType;
#define SUFFIX si128

            OP_(or_) OP_(and_) OP_(xor_)
            static Vc_ALWAYS_INLINE Vc_CONST VectorType zero() { return CAT(_mm_setzero_, SUFFIX)(); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType notMaskedToZero(VectorType a, _M128 mask) { return CAT(_mm_and_, SUFFIX)(_mm_castps_si128(mask), a); }
#undef SUFFIX
#define SUFFIX epi32
            static Vc_ALWAYS_INLINE Vc_CONST VectorType one() { return CAT(_mm_setone_, SUFFIX)(); }

            static Vc_ALWAYS_INLINE Vc_CONST VectorType set(const int a) { return CAT(_mm_set1_, SUFFIX)(a); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType set(const int a, const int b, const int c, const int d) { return CAT(_mm_set_, SUFFIX)(a, b, c, d); }

            static Vc_ALWAYS_INLINE void fma(VectorType &v1, VectorType v2, VectorType v3) { v1 = add(mul(v1, v2), v3); }

            static Vc_ALWAYS_INLINE Vc_CONST VectorType shiftLeft(VectorType a, int shift) {
                return CAT(_mm_slli_, SUFFIX)(a, shift);
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType shiftRight(VectorType a, int shift) {
                return CAT(_mm_srai_, SUFFIX)(a, shift);
            }
            OP1(abs)

            MINMAX
            static Vc_ALWAYS_INLINE Vc_CONST EntryType min(VectorType a) {
                a = min(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                // using lo_epi16 for speed here
                a = min(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                return _mm_cvtsi128_si32(a);
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType max(VectorType a) {
                a = max(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                // using lo_epi16 for speed here
                a = max(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                return _mm_cvtsi128_si32(a);
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType add(VectorType a) {
                a = add(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = add(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                return _mm_cvtsi128_si32(a);
            }
#ifdef VC_IMPL_SSE4_1
            static Vc_ALWAYS_INLINE Vc_CONST VectorType mul(VectorType a, VectorType b) { return _mm_mullo_epi32(a, b); }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType mul(VectorType a) {
                a = mul(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = mul(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                return _mm_cvtsi128_si32(a);
            }
#else
            static inline Vc_CONST VectorType mul(const VectorType a, const VectorType b) {
                const VectorType aShift = _mm_srli_si128(a, 4);
                const VectorType ab02 = _mm_mul_epu32(a, b); // [a0 * b0, a2 * b2]
                const VectorType bShift = _mm_srli_si128(b, 4);
                const VectorType ab13 = _mm_mul_epu32(aShift, bShift); // [a1 * b1, a3 * b3]
                return _mm_unpacklo_epi32(_mm_shuffle_epi32(ab02, 8), _mm_shuffle_epi32(ab13, 8));
            }
#endif

            OP(add) OP(sub)
            OPcmp(eq)
            OPcmp(lt)
            OPcmp(gt)
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cmpneq(const VectorType a, const VectorType b) { _M128I x = cmpeq(a, b); return _mm_andnot_si128(x, _mm_setallone_si128()); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cmpnlt(const VectorType a, const VectorType b) { _M128I x = cmplt(a, b); return _mm_andnot_si128(x, _mm_setallone_si128()); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cmple (const VectorType a, const VectorType b) { _M128I x = cmpgt(a, b); return _mm_andnot_si128(x, _mm_setallone_si128()); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cmpnle(const VectorType a, const VectorType b) { return cmpgt(a, b); }
#undef SUFFIX
            static Vc_ALWAYS_INLINE Vc_CONST VectorType round(VectorType a) { return a; }
        };

        template<> struct VectorHelper<unsigned int> {
            typedef unsigned int EntryType;
            typedef _M128I VectorType;
#define SUFFIX si128
            OP_CAST_(or_) OP_CAST_(and_) OP_CAST_(xor_)
            static Vc_ALWAYS_INLINE Vc_CONST VectorType zero() { return CAT(_mm_setzero_, SUFFIX)(); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType notMaskedToZero(VectorType a, _M128 mask) { return CAT(_mm_and_, SUFFIX)(_mm_castps_si128(mask), a); }

#undef SUFFIX
#define SUFFIX epu32
            static Vc_ALWAYS_INLINE Vc_CONST VectorType one() { return CAT(_mm_setone_, SUFFIX)(); }

            MINMAX
            static Vc_ALWAYS_INLINE Vc_CONST EntryType min(VectorType a) {
                a = min(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                // using lo_epi16 for speed here
                a = min(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                return _mm_cvtsi128_si32(a);
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType max(VectorType a) {
                a = max(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                // using lo_epi16 for speed here
                a = max(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                return _mm_cvtsi128_si32(a);
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType mul(VectorType a) {
                a = mul(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                // using lo_epi16 for speed here
                a = mul(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                return _mm_cvtsi128_si32(a);
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType add(VectorType a) {
                a = add(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                // using lo_epi16 for speed here
                a = add(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                return _mm_cvtsi128_si32(a);
            }

            static Vc_ALWAYS_INLINE void fma(VectorType &v1, VectorType v2, VectorType v3) { v1 = add(mul(v1, v2), v3); }

            static Vc_ALWAYS_INLINE Vc_CONST VectorType mul(const VectorType a, const VectorType b) {
                return VectorHelper<int>::mul(a, b);
            }
//X             template<unsigned int b> static Vc_ALWAYS_INLINE Vc_CONST VectorType mul(const VectorType a) {
//X                 switch (b) {
//X                     case    0: return zero();
//X                     case    1: return a;
//X                     case    2: return _mm_slli_epi32(a,  1);
//X                     case    4: return _mm_slli_epi32(a,  2);
//X                     case    8: return _mm_slli_epi32(a,  3);
//X                     case   16: return _mm_slli_epi32(a,  4);
//X                     case   32: return _mm_slli_epi32(a,  5);
//X                     case   64: return _mm_slli_epi32(a,  6);
//X                     case  128: return _mm_slli_epi32(a,  7);
//X                     case  256: return _mm_slli_epi32(a,  8);
//X                     case  512: return _mm_slli_epi32(a,  9);
//X                     case 1024: return _mm_slli_epi32(a, 10);
//X                     case 2048: return _mm_slli_epi32(a, 11);
//X                 }
//X                 return mul(a, set(b));
//X             }

#undef SUFFIX
#define SUFFIX epi32
            static Vc_ALWAYS_INLINE Vc_CONST VectorType shiftLeft(VectorType a, int shift) {
                return CAT(_mm_slli_, SUFFIX)(a, shift);
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType shiftRight(VectorType a, int shift) {
                return CAT(_mm_srli_, SUFFIX)(a, shift);
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType set(const unsigned int a) { return CAT(_mm_set1_, SUFFIX)(a); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType set(const unsigned int a, const unsigned int b, const unsigned int c, const unsigned int d) { return CAT(_mm_set_, SUFFIX)(a, b, c, d); }

            OP(add) OP(sub)
            OPcmp(eq)
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cmpneq(const VectorType a, const VectorType b) { return _mm_andnot_si128(cmpeq(a, b), _mm_setallone_si128()); }

#ifndef USE_INCORRECT_UNSIGNED_COMPARE
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cmplt(const VectorType a, const VectorType b) {
                return _mm_cmplt_epu32(a, b);
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cmpgt(const VectorType a, const VectorType b) {
                return _mm_cmpgt_epu32(a, b);
            }
#else
            OPcmp(lt)
            OPcmp(gt)
#endif
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cmpnlt(const VectorType a, const VectorType b) { return _mm_andnot_si128(cmplt(a, b), _mm_setallone_si128()); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cmple (const VectorType a, const VectorType b) { return _mm_andnot_si128(cmpgt(a, b), _mm_setallone_si128()); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cmpnle(const VectorType a, const VectorType b) { return cmpgt(a, b); }

#undef SUFFIX
            static Vc_ALWAYS_INLINE Vc_CONST VectorType round(VectorType a) { return a; }
        };

        template<> struct VectorHelper<signed short> {
            typedef _M128I VectorType;
            typedef signed short EntryType;
#define SUFFIX si128

            OP_(or_) OP_(and_) OP_(xor_)
            static Vc_ALWAYS_INLINE Vc_CONST VectorType zero() { return CAT(_mm_setzero_, SUFFIX)(); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType notMaskedToZero(VectorType a, _M128 mask) { return CAT(_mm_and_, SUFFIX)(_mm_castps_si128(mask), a); }
            static Vc_ALWAYS_INLINE Vc_CONST _M128I concat(_M128I a, _M128I b) { return _mm_packs_epi32(a, b); }
            static Vc_ALWAYS_INLINE Vc_CONST _M128I expand0(_M128I x) { return _mm_srai_epi32(_mm_unpacklo_epi16(x, x), 16); }
            static Vc_ALWAYS_INLINE Vc_CONST _M128I expand1(_M128I x) { return _mm_srai_epi32(_mm_unpackhi_epi16(x, x), 16); }

#undef SUFFIX
#define SUFFIX epi16
            static Vc_ALWAYS_INLINE Vc_CONST VectorType one() { return CAT(_mm_setone_, SUFFIX)(); }

            static Vc_ALWAYS_INLINE Vc_CONST VectorType shiftLeft(VectorType a, int shift) {
                return CAT(_mm_slli_, SUFFIX)(a, shift);
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType shiftRight(VectorType a, int shift) {
                return CAT(_mm_srai_, SUFFIX)(a, shift);
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType set(const EntryType a) { return CAT(_mm_set1_, SUFFIX)(a); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType set(const EntryType a, const EntryType b, const EntryType c, const EntryType d,
                    const EntryType e, const EntryType f, const EntryType g, const EntryType h) {
                return CAT(_mm_set_, SUFFIX)(a, b, c, d, e, f, g, h);
            }

            static Vc_ALWAYS_INLINE void fma(VectorType &v1, VectorType v2, VectorType v3) {
                v1 = add(mul(v1, v2), v3); }

            OP1(abs)

            OPx(mul, mullo)
            OP(min) OP(max)
            static Vc_ALWAYS_INLINE Vc_CONST EntryType min(VectorType a) {
                // reminder: _MM_SHUFFLE(3, 2, 1, 0) means "no change"
                a = min(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = min(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = min(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)));
                return _mm_cvtsi128_si32(a); // & 0xffff is implicit
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType max(VectorType a) {
                // reminder: _MM_SHUFFLE(3, 2, 1, 0) means "no change"
                a = max(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = max(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = max(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)));
                return _mm_cvtsi128_si32(a); // & 0xffff is implicit
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType mul(VectorType a) {
                a = mul(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = mul(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = mul(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)));
                return _mm_cvtsi128_si32(a); // & 0xffff is implicit
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType add(VectorType a) {
                a = add(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = add(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = add(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)));
                return _mm_cvtsi128_si32(a); // & 0xffff is implicit
            }

            OP(add) OP(sub)
            OPcmp(eq)
            OPcmp(lt)
            OPcmp(gt)
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cmpneq(const VectorType a, const VectorType b) { _M128I x = cmpeq(a, b); return _mm_andnot_si128(x, _mm_setallone_si128()); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cmpnlt(const VectorType a, const VectorType b) { _M128I x = cmplt(a, b); return _mm_andnot_si128(x, _mm_setallone_si128()); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cmple (const VectorType a, const VectorType b) { _M128I x = cmpgt(a, b); return _mm_andnot_si128(x, _mm_setallone_si128()); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cmpnle(const VectorType a, const VectorType b) { return cmpgt(a, b); }
#undef SUFFIX
            static Vc_ALWAYS_INLINE Vc_CONST VectorType round(VectorType a) { return a; }
        };

        template<> struct VectorHelper<unsigned short> {
            typedef _M128I VectorType;
            typedef unsigned short EntryType;
#define SUFFIX si128
            OP_CAST_(or_) OP_CAST_(and_) OP_CAST_(xor_)
            static Vc_ALWAYS_INLINE Vc_CONST VectorType zero() { return CAT(_mm_setzero_, SUFFIX)(); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType notMaskedToZero(VectorType a, _M128 mask) { return CAT(_mm_and_, SUFFIX)(_mm_castps_si128(mask), a); }
#ifdef VC_IMPL_SSE4_1
            static Vc_ALWAYS_INLINE Vc_CONST _M128I concat(_M128I a, _M128I b) { return _mm_packus_epi32(a, b); }
#else
            // FIXME too bad, but this is broken without SSE 4.1
            static Vc_ALWAYS_INLINE Vc_CONST _M128I concat(_M128I a, _M128I b) {
                auto tmp0 = _mm_unpacklo_epi16(a, b); // 0 4 X X 1 5 X X
                auto tmp1 = _mm_unpackhi_epi16(a, b); // 2 6 X X 3 7 X X
                auto tmp2 = _mm_unpacklo_epi16(tmp0, tmp1); // 0 2 4 6 X X X X
                auto tmp3 = _mm_unpackhi_epi16(tmp0, tmp1); // 1 3 5 7 X X X X
                return _mm_unpacklo_epi16(tmp2, tmp3); // 0 1 2 3 4 5 6 7
            }
#endif
            static Vc_ALWAYS_INLINE Vc_CONST _M128I expand0(_M128I x) { return _mm_unpacklo_epi16(x, _mm_setzero_si128()); }
            static Vc_ALWAYS_INLINE Vc_CONST _M128I expand1(_M128I x) { return _mm_unpackhi_epi16(x, _mm_setzero_si128()); }

#undef SUFFIX
#define SUFFIX epu16
            static Vc_ALWAYS_INLINE Vc_CONST VectorType one() { return CAT(_mm_setone_, SUFFIX)(); }

//X             template<unsigned int b> static Vc_ALWAYS_INLINE Vc_CONST VectorType mul(const VectorType a) {
//X                 switch (b) {
//X                     case    0: return zero();
//X                     case    1: return a;
//X                     case    2: return _mm_slli_epi16(a,  1);
//X                     case    4: return _mm_slli_epi16(a,  2);
//X                     case    8: return _mm_slli_epi16(a,  3);
//X                     case   16: return _mm_slli_epi16(a,  4);
//X                     case   32: return _mm_slli_epi16(a,  5);
//X                     case   64: return _mm_slli_epi16(a,  6);
//X                     case  128: return _mm_slli_epi16(a,  7);
//X                     case  256: return _mm_slli_epi16(a,  8);
//X                     case  512: return _mm_slli_epi16(a,  9);
//X                     case 1024: return _mm_slli_epi16(a, 10);
//X                     case 2048: return _mm_slli_epi16(a, 11);
//X                 }
//X                 return mul(a, set(b));
//X             }
#if !defined(USE_INCORRECT_UNSIGNED_COMPARE) || VC_IMPL_SSE4_1
            OP(min) OP(max)
#endif
#undef SUFFIX
#define SUFFIX epi16
            static Vc_ALWAYS_INLINE Vc_CONST VectorType shiftLeft(VectorType a, int shift) {
                return CAT(_mm_slli_, SUFFIX)(a, shift);
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType shiftRight(VectorType a, int shift) {
                return CAT(_mm_srli_, SUFFIX)(a, shift);
            }

            static Vc_ALWAYS_INLINE void fma(VectorType &v1, VectorType v2, VectorType v3) { v1 = add(mul(v1, v2), v3); }

            OPx(mul, mullo) // should work correctly for all values
#if defined(USE_INCORRECT_UNSIGNED_COMPARE) && !defined(VC_IMPL_SSE4_1)
            OP(min) OP(max) // XXX breaks for values with MSB set
#endif
            static Vc_ALWAYS_INLINE Vc_CONST EntryType min(VectorType a) {
                // reminder: _MM_SHUFFLE(3, 2, 1, 0) means "no change"
                a = min(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = min(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = min(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)));
                return _mm_cvtsi128_si32(a); // & 0xffff is implicit
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType max(VectorType a) {
                // reminder: _MM_SHUFFLE(3, 2, 1, 0) means "no change"
                a = max(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = max(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = max(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)));
                return _mm_cvtsi128_si32(a); // & 0xffff is implicit
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType mul(VectorType a) {
                // reminder: _MM_SHUFFLE(3, 2, 1, 0) means "no change"
                a = mul(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = mul(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = mul(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)));
                return _mm_cvtsi128_si32(a); // & 0xffff is implicit
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType add(VectorType a) {
                // reminder: _MM_SHUFFLE(3, 2, 1, 0) means "no change"
                a = add(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = add(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = add(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)));
                return _mm_cvtsi128_si32(a); // & 0xffff is implicit
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType set(const EntryType a) { return CAT(_mm_set1_, SUFFIX)(a); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType set(const EntryType a, const EntryType b, const EntryType c,
                    const EntryType d, const EntryType e, const EntryType f,
                    const EntryType g, const EntryType h) {
                return CAT(_mm_set_, SUFFIX)(a, b, c, d, e, f, g, h);
            }

            OP(add) OP(sub)
            OPcmp(eq)
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cmpneq(const VectorType a, const VectorType b) { return _mm_andnot_si128(cmpeq(a, b), _mm_setallone_si128()); }

#ifndef USE_INCORRECT_UNSIGNED_COMPARE
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cmplt(const VectorType a, const VectorType b) {
                return _mm_cmplt_epu16(a, b);
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cmpgt(const VectorType a, const VectorType b) {
                return _mm_cmpgt_epu16(a, b);
            }
#else
            OPcmp(lt)
            OPcmp(gt)
#endif
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cmpnlt(const VectorType a, const VectorType b) { return _mm_andnot_si128(cmplt(a, b), _mm_setallone_si128()); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cmple (const VectorType a, const VectorType b) { return _mm_andnot_si128(cmpgt(a, b), _mm_setallone_si128()); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cmpnle(const VectorType a, const VectorType b) { return cmpgt(a, b); }
#undef SUFFIX
            static Vc_ALWAYS_INLINE Vc_CONST VectorType round(VectorType a) { return a; }
        };
#undef OP1
#undef OP
#undef OP_
#undef OPx
#undef OPcmp

Vc_IMPL_NAMESPACE_END

#include "vectorhelper.tcc"
#include "undomacros.h"

#endif // SSE_VECTORHELPER_H
