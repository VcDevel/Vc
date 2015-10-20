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

#ifndef AVX_VECTORHELPER_H
#define AVX_VECTORHELPER_H

#include <limits>
#include "types.h"
#include "intrinsics.h"
#include "casts.h"
#include "../common/loadstoreflags.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace AVX
{

#define OP0(name, code) static Vc_ALWAYS_INLINE Vc_CONST VectorType name() { return code; }
#define OP1(name, code) static Vc_ALWAYS_INLINE Vc_CONST VectorType name(VTArg a) { return code; }
#define OP2(name, code) static Vc_ALWAYS_INLINE Vc_CONST VectorType name(VTArg a, VTArg b) { return code; }
#define OP3(name, code) static Vc_ALWAYS_INLINE Vc_CONST VectorType name(VTArg a, VTArg b, VTArg c) { return code; }

        template<> struct VectorHelper<__m256>
        {
            typedef __m256 VectorType;
#ifdef Vc_PASSING_VECTOR_BY_VALUE_IS_BROKEN
            typedef const VectorType & VTArg;
#else
            typedef const VectorType VTArg;
#endif

            template<typename Flags> static Vc_ALWAYS_INLINE void store(float *mem, VTArg x, typename Flags::EnableIfAligned               = nullptr) { _mm256_store_ps(mem, x); }
            template<typename Flags> static Vc_ALWAYS_INLINE void store(float *mem, VTArg x, typename Flags::EnableIfUnalignedNotStreaming = nullptr) { _mm256_storeu_ps(mem, x); }
            template<typename Flags> static Vc_ALWAYS_INLINE void store(float *mem, VTArg x, typename Flags::EnableIfStreaming             = nullptr) { _mm256_stream_ps(mem, x); }
            template<typename Flags> static Vc_ALWAYS_INLINE void store(float *mem, VTArg x, typename Flags::EnableIfUnalignedAndStreaming = nullptr) { AvxIntrinsics::stream_store(mem, x, setallone_ps()); }

            template<typename Flags> static Vc_ALWAYS_INLINE void store(float *mem, VTArg x, VTArg m, typename std::enable_if<!Flags::IsStreaming, void *>::type = nullptr) { _mm256_maskstore(mem, m, x); }
            template<typename Flags> static Vc_ALWAYS_INLINE void store(float *mem, VTArg x, VTArg m, typename std::enable_if< Flags::IsStreaming, void *>::type = nullptr) { AvxIntrinsics::stream_store(mem, x, m); }

            static Vc_ALWAYS_INLINE Vc_CONST VectorType cdab(VTArg x) { return _mm256_permute_ps(x, _MM_SHUFFLE(2, 3, 0, 1)); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType badc(VTArg x) { return _mm256_permute_ps(x, _MM_SHUFFLE(1, 0, 3, 2)); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType aaaa(VTArg x) { return _mm256_permute_ps(x, _MM_SHUFFLE(0, 0, 0, 0)); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType bbbb(VTArg x) { return _mm256_permute_ps(x, _MM_SHUFFLE(1, 1, 1, 1)); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType cccc(VTArg x) { return _mm256_permute_ps(x, _MM_SHUFFLE(2, 2, 2, 2)); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType dddd(VTArg x) { return _mm256_permute_ps(x, _MM_SHUFFLE(3, 3, 3, 3)); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType dacb(VTArg x) { return _mm256_permute_ps(x, _MM_SHUFFLE(3, 0, 2, 1)); }

            OP0(allone, setallone_ps())
            OP0(zero, _mm256_setzero_ps())
            OP2(or_, _mm256_or_ps(a, b))
            OP2(xor_, _mm256_xor_ps(a, b))
            OP2(and_, _mm256_and_ps(a, b))
            OP2(andnot_, _mm256_andnot_ps(a, b))
            OP3(blend, _mm256_blendv_ps(a, b, c))
        };

        template<> struct VectorHelper<__m256d>
        {
            typedef __m256d VectorType;
#ifdef Vc_PASSING_VECTOR_BY_VALUE_IS_BROKEN
            typedef const VectorType & VTArg;
#else
            typedef const VectorType VTArg;
#endif

            template<typename Flags> static Vc_ALWAYS_INLINE void store(double *mem, VTArg x, typename Flags::EnableIfAligned               = nullptr) { _mm256_store_pd(mem, x); }
            template<typename Flags> static Vc_ALWAYS_INLINE void store(double *mem, VTArg x, typename Flags::EnableIfUnalignedNotStreaming = nullptr) { _mm256_storeu_pd(mem, x); }
            template<typename Flags> static Vc_ALWAYS_INLINE void store(double *mem, VTArg x, typename Flags::EnableIfStreaming             = nullptr) { _mm256_stream_pd(mem, x); }
            template<typename Flags> static Vc_ALWAYS_INLINE void store(double *mem, VTArg x, typename Flags::EnableIfUnalignedAndStreaming = nullptr) { AvxIntrinsics::stream_store(mem, x, setallone_pd()); }

            template<typename Flags> static Vc_ALWAYS_INLINE void store(double *mem, VTArg x, VTArg m, typename std::enable_if<!Flags::IsStreaming, void *>::type = nullptr) { _mm256_maskstore(mem, m, x); }
            template<typename Flags> static Vc_ALWAYS_INLINE void store(double *mem, VTArg x, VTArg m, typename std::enable_if< Flags::IsStreaming, void *>::type = nullptr) { AvxIntrinsics::stream_store(mem, x, m); }

            static VectorType cdab(VTArg x) { return _mm256_permute_pd(x, 5); }
            static VectorType badc(VTArg x) { return _mm256_permute2f128_pd(x, x, 1); }
            // aaaa bbbb cccc dddd specialized in vector.tcc
            static VectorType dacb(VTArg x) {
                const __m128d cb = avx_cast<__m128d>(_mm_alignr_epi8(avx_cast<__m128i>(lo128(x)),
                            avx_cast<__m128i>(hi128(x)), sizeof(double))); // XXX: lo and hi swapped?
                const __m128d da = _mm_blend_pd(lo128(x), hi128(x), 0 + 2); // XXX: lo and hi swapped?
                return concat(cb, da);
            }

            OP0(allone, setallone_pd())
            OP0(zero, _mm256_setzero_pd())
            OP2(or_, _mm256_or_pd(a, b))
            OP2(xor_, _mm256_xor_pd(a, b))
            OP2(and_, _mm256_and_pd(a, b))
            OP2(andnot_, _mm256_andnot_pd(a, b))
            OP3(blend, _mm256_blendv_pd(a, b, c))
        };

        template<> struct VectorHelper<__m256i>
        {
            typedef __m256i VectorType;
#ifdef Vc_PASSING_VECTOR_BY_VALUE_IS_BROKEN
            typedef const VectorType & VTArg;
#else
            typedef const VectorType VTArg;
#endif

            template<typename Flags, typename T> static Vc_ALWAYS_INLINE void store(T *mem, VTArg x, typename Flags::EnableIfAligned               = nullptr) { _mm256_store_si256(reinterpret_cast<__m256i *>(mem), x); }
            template<typename Flags, typename T> static Vc_ALWAYS_INLINE void store(T *mem, VTArg x, typename Flags::EnableIfUnalignedNotStreaming = nullptr) { _mm256_storeu_si256(reinterpret_cast<__m256i *>(mem), x); }
            template<typename Flags, typename T> static Vc_ALWAYS_INLINE void store(T *mem, VTArg x, typename Flags::EnableIfStreaming             = nullptr) { _mm256_stream_si256(reinterpret_cast<__m256i *>(mem), x); }
            template<typename Flags, typename T> static Vc_ALWAYS_INLINE void store(T *mem, VTArg x, typename Flags::EnableIfUnalignedAndStreaming = nullptr) { AvxIntrinsics::stream_store(mem, x, setallone_si256()); }

            template<typename Flags, typename T> static Vc_ALWAYS_INLINE void store(T *mem, VTArg x, VTArg m, typename std::enable_if<!Flags::IsStreaming, void *>::type = nullptr) { _mm256_maskstore(mem, m, x); }
            template<typename Flags, typename T> static Vc_ALWAYS_INLINE void store(T *mem, VTArg x, VTArg m, typename std::enable_if< Flags::IsStreaming, void *>::type = nullptr) { AvxIntrinsics::stream_store(mem, x, m); }

            static VectorType cdab(VTArg x) { return avx_cast<VectorType>(_mm256_permute_ps(avx_cast<__m256>(x), _MM_SHUFFLE(2, 3, 0, 1))); }
            static VectorType badc(VTArg x) { return avx_cast<VectorType>(_mm256_permute_ps(avx_cast<__m256>(x), _MM_SHUFFLE(1, 0, 3, 2))); }
            static VectorType aaaa(VTArg x) { return avx_cast<VectorType>(_mm256_permute_ps(avx_cast<__m256>(x), _MM_SHUFFLE(0, 0, 0, 0))); }
            static VectorType bbbb(VTArg x) { return avx_cast<VectorType>(_mm256_permute_ps(avx_cast<__m256>(x), _MM_SHUFFLE(1, 1, 1, 1))); }
            static VectorType cccc(VTArg x) { return avx_cast<VectorType>(_mm256_permute_ps(avx_cast<__m256>(x), _MM_SHUFFLE(2, 2, 2, 2))); }
            static VectorType dddd(VTArg x) { return avx_cast<VectorType>(_mm256_permute_ps(avx_cast<__m256>(x), _MM_SHUFFLE(3, 3, 3, 3))); }
            static VectorType dacb(VTArg x) { return avx_cast<VectorType>(_mm256_permute_ps(avx_cast<__m256>(x), _MM_SHUFFLE(3, 0, 2, 1))); }

            OP0(allone, setallone_si256())
            OP0(zero, _mm256_setzero_si256())
            OP2(or_, or_si256(a, b))
            OP2(xor_, xor_si256(a, b))
            OP2(and_, and_si256(a, b))
            OP2(andnot_, andnot_si256(a, b))
            OP3(blend, blendv_epi8(a, b, c))
        };
#undef OP0
#undef OP1
#undef OP2
#undef OP3

#define OP1(op) \
        static Vc_INTRINSIC VectorType Vc_CONST op(VTArg a) { return Vc_CAT2(_mm256_##op##_, SUFFIX)(a); }
#define OP(op) \
        static Vc_INTRINSIC VectorType Vc_CONST op(VTArg a, VTArg b) { return Vc_CAT2(op##_ , SUFFIX)(a, b); }
#define OP_(op) \
        static Vc_INTRINSIC VectorType Vc_CONST op(VTArg a, VTArg b) { return Vc_CAT2(_mm256_##op    , SUFFIX)(a, b); }
#define OPx(op, op2) \
        static Vc_INTRINSIC VectorType Vc_CONST op(VTArg a, VTArg b) { return Vc_CAT2(_mm256_##op2##_, SUFFIX)(a, b); }
#define OPcmp(op) \
        static Vc_INTRINSIC VectorType Vc_CONST cmp##op(VTArg a, VTArg b) { return Vc_CAT2(cmp##op##_, SUFFIX)(a, b); }
#define OP_CAST_(op) \
        static Vc_INTRINSIC VectorType Vc_CONST op(VTArg a, VTArg b) { return Vc_CAT2(_mm256_castps_, SUFFIX)( \
            _mm256_##op##ps(Vc_CAT2(Vc_CAT2(_mm256_cast, SUFFIX), _ps)(a), \
              Vc_CAT2(Vc_CAT2(_mm256_cast, SUFFIX), _ps)(b))); \
        }
#define MINMAX \
        static Vc_INTRINSIC VectorType Vc_CONST min(VTArg a, VTArg b) { return Vc_CAT2(min_, SUFFIX)(a, b); } \
        static Vc_INTRINSIC VectorType Vc_CONST max(VTArg a, VTArg b) { return Vc_CAT2(max_, SUFFIX)(a, b); }

        template<> struct VectorHelper<double> {
            typedef __m256d VectorType;
#ifdef Vc_PASSING_VECTOR_BY_VALUE_IS_BROKEN
            typedef const VectorType & VTArg;
#else
            typedef const VectorType VTArg;
#endif
            typedef double EntryType;
#define SUFFIX pd

            static Vc_ALWAYS_INLINE VectorType notMaskedToZero(VTArg a, __m256 mask) { return Vc_CAT2(_mm256_and_, SUFFIX)(_mm256_castps_pd(mask), a); }
            static Vc_ALWAYS_INLINE VectorType set(const double a) { return Vc_CAT2(_mm256_set1_, SUFFIX)(a); }
            static Vc_ALWAYS_INLINE VectorType set(const double a, const double b, const double c, const double d) {
                return Vc_CAT2(_mm256_set_, SUFFIX)(a, b, c, d);
            }
            static Vc_ALWAYS_INLINE VectorType zero() { return Vc_CAT2(_mm256_setzero_, SUFFIX)(); }
            static Vc_ALWAYS_INLINE VectorType one()  { return Vc_CAT2(setone_, SUFFIX)(); }// set(1.); }

            static inline void fma(VectorType &v1, VTArg v2, VTArg v3) {
#ifdef Vc_IMPL_FMA4
                v1 = _mm256_macc_pd(v1, v2, v3);
#else
                VectorType h1 = _mm256_and_pd(v1, _mm256_broadcast_sd(reinterpret_cast<const double *>(&c_general::highMaskDouble)));
                VectorType h2 = _mm256_and_pd(v2, _mm256_broadcast_sd(reinterpret_cast<const double *>(&c_general::highMaskDouble)));
#if defined(Vc_GCC) && Vc_GCC < 0x40703
                // GCC before 4.7.3 uses an incorrect optimization where it replaces the subtraction with an andnot
                // http://gcc.gnu.org/bugzilla/show_bug.cgi?id=54703
                asm("":"+x"(h1), "+x"(h2));
#endif
                const VectorType l1 = _mm256_sub_pd(v1, h1);
                const VectorType l2 = _mm256_sub_pd(v2, h2);
                const VectorType ll = mul(l1, l2);
                const VectorType lh = add(mul(l1, h2), mul(h1, l2));
                const VectorType hh = mul(h1, h2);
                // ll < lh < hh for all entries is certain
                const VectorType lh_lt_v3 = cmplt(abs(lh), abs(v3)); // |lh| < |v3|
                const VectorType b = _mm256_blendv_pd(v3, lh, lh_lt_v3);
                const VectorType c = _mm256_blendv_pd(lh, v3, lh_lt_v3);
                v1 = add(add(ll, b), add(c, hh));
#endif
            }

            static Vc_INTRINSIC VectorType Vc_CONST add(VTArg a, VTArg b) { return _mm256_add_pd(a,b); }
            static Vc_INTRINSIC VectorType Vc_CONST sub(VTArg a, VTArg b) { return _mm256_sub_pd(a,b); }
            static Vc_INTRINSIC VectorType Vc_CONST mul(VTArg a, VTArg b) { return _mm256_mul_pd(a,b); }
            OPcmp(eq) OPcmp(neq)
            OPcmp(lt) OPcmp(nlt)
            OPcmp(le) OPcmp(nle)

            OP1(sqrt)
            static Vc_ALWAYS_INLINE Vc_CONST VectorType rsqrt(VTArg x) {
                return _mm256_div_pd(one(), sqrt(x));
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType reciprocal(VTArg x) {
                return _mm256_div_pd(one(), x);
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType isNaN(VTArg x) {
                return cmpunord_pd(x, x);
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType isFinite(VTArg x) {
                return cmpord_pd(x, _mm256_mul_pd(zero(), x));
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType isInfinite(VectorType x) {
                return _mm256_castsi256_pd(cmpeq_epi64(_mm256_castpd_si256(abs(x)), _mm256_castpd_si256(set1_pd(c_log<double>::d(1)))));
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType abs(VTArg a) {
                return Vc_CAT2(_mm256_and_, SUFFIX)(a, setabsmask_pd());
            }

            static Vc_INTRINSIC VectorType Vc_CONST min(VTArg a, VTArg b) { return _mm256_min_pd(a, b); }
            static Vc_INTRINSIC VectorType Vc_CONST max(VTArg a, VTArg b) { return _mm256_max_pd(a, b); }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType min(VTArg a) {
                __m128d b = _mm_min_pd(avx_cast<__m128d>(a), _mm256_extractf128_pd(a, 1));
                b = _mm_min_sd(b, _mm_unpackhi_pd(b, b));
                return _mm_cvtsd_f64(b);
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType max(VTArg a) {
                __m128d b = _mm_max_pd(avx_cast<__m128d>(a), _mm256_extractf128_pd(a, 1));
                b = _mm_max_sd(b, _mm_unpackhi_pd(b, b));
                return _mm_cvtsd_f64(b);
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType mul(VTArg a) {
                __m128d b = _mm_mul_pd(avx_cast<__m128d>(a), _mm256_extractf128_pd(a, 1));
                b = _mm_mul_sd(b, _mm_shuffle_pd(b, b, _MM_SHUFFLE2(0, 1)));
                return _mm_cvtsd_f64(b);
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType add(VTArg a) {
                __m128d b = _mm_add_pd(avx_cast<__m128d>(a), _mm256_extractf128_pd(a, 1));
                b = _mm_hadd_pd(b, b); // or: b = _mm_add_sd(b, _mm256_shuffle_pd(b, b, _MM_SHUFFLE2(0, 1)));
                return _mm_cvtsd_f64(b);
            }
#undef SUFFIX
            static Vc_ALWAYS_INLINE Vc_CONST VectorType round(VTArg a) {
                return _mm256_round_pd(a, _MM_FROUND_NINT);
            }
        };

        template<> struct VectorHelper<float> {
            typedef float EntryType;
            typedef __m256 VectorType;
#ifdef Vc_PASSING_VECTOR_BY_VALUE_IS_BROKEN
            typedef const VectorType & VTArg;
#else
            typedef const VectorType VTArg;
#endif
#define SUFFIX ps

            static Vc_ALWAYS_INLINE Vc_CONST VectorType notMaskedToZero(VTArg a, __m256 mask) { return Vc_CAT2(_mm256_and_, SUFFIX)(mask, a); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType set(const float a) { return Vc_CAT2(_mm256_set1_, SUFFIX)(a); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType set(const float a, const float b, const float c, const float d,
                    const float e, const float f, const float g, const float h) {
                return Vc_CAT2(_mm256_set_, SUFFIX)(a, b, c, d, e, f, g, h); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType zero() { return Vc_CAT2(_mm256_setzero_, SUFFIX)(); }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType one()  { return Vc_CAT2(setone_, SUFFIX)(); }// set(1.f); }
            static Vc_ALWAYS_INLINE Vc_CONST __m256 concat(__m256d a, __m256d b) { return _mm256_insertf128_ps(avx_cast<__m256>(_mm256_cvtpd_ps(a)), _mm256_cvtpd_ps(b), 1); }

            static inline void fma(VectorType &v1, VTArg v2, VTArg v3) {
#ifdef Vc_IMPL_FMA4
                v1 = _mm256_macc_ps(v1, v2, v3);
#else
                __m256d v1_0 = _mm256_cvtps_pd(lo128(v1));
                __m256d v1_1 = _mm256_cvtps_pd(hi128(v1));
                __m256d v2_0 = _mm256_cvtps_pd(lo128(v2));
                __m256d v2_1 = _mm256_cvtps_pd(hi128(v2));
                __m256d v3_0 = _mm256_cvtps_pd(lo128(v3));
                __m256d v3_1 = _mm256_cvtps_pd(hi128(v3));
                v1 = AVX::concat(
                        _mm256_cvtpd_ps(_mm256_add_pd(_mm256_mul_pd(v1_0, v2_0), v3_0)),
                        _mm256_cvtpd_ps(_mm256_add_pd(_mm256_mul_pd(v1_1, v2_1), v3_1)));
#endif
            }

            static Vc_INTRINSIC VectorType Vc_CONST add(VTArg a, VTArg b) { return _mm256_add_ps(a, b); }
            static Vc_INTRINSIC VectorType Vc_CONST sub(VTArg a, VTArg b) { return _mm256_sub_ps(a, b); }
            static Vc_INTRINSIC VectorType Vc_CONST mul(VTArg a, VTArg b) { return _mm256_mul_ps(a, b); }
            OPcmp(eq) OPcmp(neq)
            OPcmp(lt) OPcmp(nlt)
            OPcmp(le) OPcmp(nle)

            OP1(sqrt) OP1(rsqrt)
            static Vc_ALWAYS_INLINE Vc_CONST VectorType isNaN(VTArg x) {
                return cmpunord_ps(x, x);
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType isFinite(VTArg x) {
                return cmpord_ps(x, _mm256_mul_ps(zero(), x));
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType isInfinite(VectorType x) {
                return _mm256_castsi256_ps(cmpeq_epi32(_mm256_castps_si256(abs(x)), _mm256_castps_si256(set1_ps(c_log<float>::d(1)))));
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType reciprocal(VTArg x) {
                return _mm256_rcp_ps(x);
            }
            static Vc_ALWAYS_INLINE Vc_CONST VectorType abs(VTArg a) {
                return Vc_CAT2(_mm256_and_, SUFFIX)(a, setabsmask_ps());
            }

            static Vc_INTRINSIC VectorType Vc_CONST min(VTArg a, VTArg b) { return _mm256_min_ps(a, b); }
            static Vc_INTRINSIC VectorType Vc_CONST max(VTArg a, VTArg b) { return _mm256_max_ps(a, b); }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType min(VTArg a) {
                __m128 b = _mm_min_ps(lo128(a), hi128(a));
                b = _mm_min_ps(b, _mm_movehl_ps(b, b));   // b = min(a0, a2), min(a1, a3), min(a2, a2), min(a3, a3)
                b = _mm_min_ss(b, _mm_shuffle_ps(b, b, _MM_SHUFFLE(1, 1, 1, 1))); // b = min(a0, a1), a1, a2, a3
                return _mm_cvtss_f32(b);
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType max(VTArg a) {
                __m128 b = _mm_max_ps(avx_cast<__m128>(a), _mm256_extractf128_ps(a, 1));
                b = _mm_max_ps(b, _mm_movehl_ps(b, b));   // b = max(a0, a2), max(a1, a3), max(a2, a2), max(a3, a3)
                b = _mm_max_ss(b, _mm_shuffle_ps(b, b, _MM_SHUFFLE(1, 1, 1, 1))); // b = max(a0, a1), a1, a2, a3
                return _mm_cvtss_f32(b);
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType mul(VTArg a) {
                __m128 b = _mm_mul_ps(avx_cast<__m128>(a), _mm256_extractf128_ps(a, 1));
                b = _mm_mul_ps(b, _mm_shuffle_ps(b, b, _MM_SHUFFLE(0, 1, 2, 3)));
                b = _mm_mul_ss(b, _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 2, 0, 1)));
                return _mm_cvtss_f32(b);
            }
            static Vc_ALWAYS_INLINE Vc_CONST EntryType add(VTArg a) {
                __m128 b = _mm_add_ps(avx_cast<__m128>(a), _mm256_extractf128_ps(a, 1));
                b = _mm_add_ps(b, _mm_shuffle_ps(b, b, _MM_SHUFFLE(0, 1, 2, 3)));
                b = _mm_add_ss(b, _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 2, 0, 1)));
                return _mm_cvtss_f32(b);
            }
#undef SUFFIX
            static Vc_ALWAYS_INLINE Vc_CONST VectorType round(VTArg a) {
                return _mm256_round_ps(a, _MM_FROUND_NINT);
            }
        };

#undef OP1
#undef OP
#undef OP_
#undef OPx
#undef OPcmp
#undef OP_CAST_
#undef MINMAX

}  // namespace AVX(2)
}  // namespace Vc

#endif // AVX_VECTORHELPER_H
