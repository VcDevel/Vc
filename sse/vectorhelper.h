/*  This file is part of the Vc library.

    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

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

/* The log implementations are based on code from Julien Pommier which carries the following
   copyright information:
 */
/*
   Inspired by Intel Approximate Math library, and based on the
   corresponding algorithms of the cephes math library
*/
/* Copyright (C) 2007  Julien Pommier

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

#ifndef SSE_VECTORHELPER_H
#define SSE_VECTORHELPER_H

#include "vectorbase.h"
#include <limits>

namespace Vc
{
namespace SSE
{
    template<typename VectorType, unsigned int Size> struct SortHelper
    {
        static VectorType sort(VectorType) PURE;
    };
    template<unsigned int Size> struct SortHelper<M256, Size>
    {
        static M256 sort(const M256 &) PURE;
    };

#undef OP_DECL
#undef PARENT_DATA
#undef PARENT_DATA_CONST

#define OP0(name, code) static inline VectorType name() PURE { return code; }
#define OP1(name, code) static inline VectorType name(const VectorType &a) PURE { return code; }
#define OP2(name, code) static inline VectorType name(const VectorType &a, const VectorType &b) PURE { return code; }
#define OP3(name, code) static inline VectorType name(const VectorType &a, const VectorType &b, const VectorType &c) PURE { return code; }

        template<> struct VectorHelper<_M128>
        {
            typedef _M128 VectorType;
            template<typename A> static VectorType load(const float *x, A) PURE;
            static void store(float *mem, const VectorType x, AlignedFlag);
            static void store(float *mem, const VectorType x, UnalignedFlag);
            static void store(float *mem, const VectorType x, StreamingAndAlignedFlag);
            static void store(float *mem, const VectorType x, StreamingAndUnalignedFlag);
            static void store(float *mem, const VectorType x, const VectorType m, AlignedFlag);
            static void store(float *mem, const VectorType x, const VectorType m, UnalignedFlag);
            static void store(float *mem, const VectorType x, const VectorType m, StreamingAndAlignedFlag);
            static void store(float *mem, const VectorType x, const VectorType m, StreamingAndUnalignedFlag);

            OP0(allone, _mm_setallone_ps())
            OP0(zero, _mm_setzero_ps())
            OP2(or_, _mm_or_ps(a, b))
            OP2(xor_, _mm_xor_ps(a, b))
            OP2(and_, _mm_and_ps(a, b))
            OP2(andnot_, _mm_andnot_ps(a, b))
            OP3(blend, _mm_blendv_ps(a, b, c))
        };


        template<> struct VectorHelper<M256>
        {
            typedef M256 VectorType;
            template<typename A> static VectorType load(const float *x, A) PURE;
            static void store(float *mem, const VectorType &x, AlignedFlag);
            static void store(float *mem, const VectorType &x, UnalignedFlag);
            static void store(float *mem, const VectorType &x, StreamingAndAlignedFlag);
            static void store(float *mem, const VectorType &x, StreamingAndUnalignedFlag);
            static void store(float *mem, const VectorType &x, const VectorType &m, AlignedFlag);
            static void store(float *mem, const VectorType &x, const VectorType &m, UnalignedFlag);
            static void store(float *mem, const VectorType &x, const VectorType &m, StreamingAndAlignedFlag);
            static void store(float *mem, const VectorType &x, const VectorType &m, StreamingAndUnalignedFlag);

            OP0(allone, VectorType::create(_mm_setallone_ps(), _mm_setallone_ps()))
            OP0(zero, VectorType::create(_mm_setzero_ps(), _mm_setzero_ps()))
            OP2(or_, VectorType::create(_mm_or_ps(a[0], b[0]), _mm_or_ps(a[1], b[1])))
            OP2(xor_, VectorType::create(_mm_xor_ps(a[0], b[0]), _mm_xor_ps(a[1], b[1])))
            OP2(and_, VectorType::create(_mm_and_ps(a[0], b[0]), _mm_and_ps(a[1], b[1])))
            OP2(andnot_, VectorType::create(_mm_andnot_ps(a[0], b[0]), _mm_andnot_ps(a[1], b[1])))
            OP3(blend, VectorType::create(_mm_blendv_ps(a[0], b[0], c[0]), _mm_blendv_ps(a[1], b[1], c[1])))
        };

        template<> struct VectorHelper<_M128D>
        {
            typedef _M128D VectorType;
            template<typename A> static VectorType load(const double *x, A) PURE;
            static void store(double *mem, const VectorType x, AlignedFlag);
            static void store(double *mem, const VectorType x, UnalignedFlag);
            static void store(double *mem, const VectorType x, StreamingAndAlignedFlag);
            static void store(double *mem, const VectorType x, StreamingAndUnalignedFlag);
            static void store(double *mem, const VectorType x, const VectorType m, AlignedFlag);
            static void store(double *mem, const VectorType x, const VectorType m, UnalignedFlag);
            static void store(double *mem, const VectorType x, const VectorType m, StreamingAndAlignedFlag);
            static void store(double *mem, const VectorType x, const VectorType m, StreamingAndUnalignedFlag);

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
            template<typename T> static VectorType load(const T *x, AlignedFlag) PURE;
            template<typename T> static VectorType load(const T *x, UnalignedFlag) PURE;
            template<typename T> static VectorType load(const T *x, StreamingAndAlignedFlag) PURE;
            template<typename T> static VectorType load(const T *x, StreamingAndUnalignedFlag) PURE;
            template<typename T> static void store(T *mem, const VectorType x, AlignedFlag);
            template<typename T> static void store(T *mem, const VectorType x, UnalignedFlag);
            template<typename T> static void store(T *mem, const VectorType x, StreamingAndAlignedFlag);
            template<typename T> static void store(T *mem, const VectorType x, StreamingAndUnalignedFlag);
            template<typename T> static void store(T *mem, const VectorType x, const VectorType m, AlignedFlag);
            template<typename T> static void store(T *mem, const VectorType x, const VectorType m, UnalignedFlag);
            template<typename T> static void store(T *mem, const VectorType x, const VectorType m, StreamingAndAlignedFlag);
            template<typename T> static void store(T *mem, const VectorType x, const VectorType m, StreamingAndUnalignedFlag);

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
        static inline VectorType op(const VectorType &a) PURE { return CAT(_mm_##op##_, SUFFIX)(a); }
#define OP(op) \
        static inline VectorType op(const VectorType &a, const VectorType &b) PURE { return CAT(_mm_##op##_ , SUFFIX)(a, b); }
#define OP_(op) \
        static inline VectorType op(const VectorType &a, const VectorType &b) PURE { return CAT(_mm_##op    , SUFFIX)(a, b); }
#define OPx(op, op2) \
        static inline VectorType op(const VectorType &a, const VectorType &b) PURE { return CAT(_mm_##op2##_, SUFFIX)(a, b); }
#define OPcmp(op) \
        static inline VectorType cmp##op(const VectorType &a, const VectorType &b) PURE { return CAT(_mm_cmp##op##_, SUFFIX)(a, b); }
#define OP_CAST_(op) \
        static inline VectorType op(const VectorType &a, const VectorType &b) PURE { return CAT(_mm_castps_, SUFFIX)( \
            _mm_##op##ps(CAT(CAT(_mm_cast, SUFFIX), _ps)(a), \
              CAT(CAT(_mm_cast, SUFFIX), _ps)(b))); \
        }
#define MINMAX \
        static inline VectorType min(VectorType a, VectorType b) PURE { return CAT(_mm_min_, SUFFIX)(a, b); } \
        static inline VectorType max(VectorType a, VectorType b) PURE { return CAT(_mm_max_, SUFFIX)(a, b); }

        template<> struct VectorHelper<double> {
            typedef _M128D VectorType;
            typedef double EntryType;
#define SUFFIX pd

            static inline VectorType notMaskedToZero(VectorType a, _M128 mask) PURE { return CAT(_mm_and_, SUFFIX)(_mm_castps_pd(mask), a); }
            static inline VectorType set(const double a) PURE { return CAT(_mm_set1_, SUFFIX)(a); }
            static inline VectorType set(const double a, const double b) PURE { return CAT(_mm_set_, SUFFIX)(a, b); }
            static inline VectorType zero() PURE { return CAT(_mm_setzero_, SUFFIX)(); }
            static inline VectorType one()  PURE { return CAT(_mm_setone_, SUFFIX)(); }// set(1.); }

            static inline void multiplyAndAdd(VectorType &v1, VectorType v2, VectorType v3) { v1 = add(mul(v1, v2), v3); }
            static inline VectorType mul(VectorType a, VectorType b, _M128 _mask) PURE {
                _M128D mask = _mm_castps_pd(_mask);
                return _mm_or_pd(
                    _mm_and_pd(mask, _mm_mul_pd(a, b)),
                    _mm_andnot_pd(mask, a)
                    );
            }
            static inline VectorType div(VectorType a, VectorType b, _M128 _mask) PURE {
                _M128D mask = _mm_castps_pd(_mask);
                return _mm_or_pd(
                    _mm_and_pd(mask, _mm_div_pd(a, b)),
                    _mm_andnot_pd(mask, a)
                    );
            }

            OP(add) OP(sub) OP(mul)
            OPcmp(eq) OPcmp(neq)
            OPcmp(lt) OPcmp(nlt)
            OPcmp(le) OPcmp(nle)

            OP1(sqrt)
            static inline VectorType rsqrt(VectorType x) PURE {
                return _mm_div_pd(one(), sqrt(x));
            }
            static inline VectorType reciprocal(VectorType x) PURE {
                return _mm_div_pd(one(), x);
            }
            static inline VectorType isNaN(VectorType x) PURE {
                return _mm_cmpunord_pd(x, x);
            }
            static inline VectorType isFinite(VectorType x) PURE {
                return _mm_cmpord_pd(x, _mm_mul_pd(zero(), x));
            }
            static VectorType log(VectorType x) PURE {
                const _M128D one = set(1.);
                const _M128D invalid_mask = cmplt(x, zero());
                const _M128D infinity_mask = cmpeq(x, zero());

                x = max(x, set(std::numeric_limits<double>::min()));  // lazy: cut off denormalized numbers

                _M128I emm0 = _mm_srli_epi64(_mm_castpd_si128(x), 52);
                emm0 = _mm_sub_epi32(emm0, _mm_set1_epi32(1023));
                _M128D e = _mm_cvtepi32_pd(_mm_shuffle_epi32(emm0, _MM_SHUFFLE(2, 0, 2, 0)));
                e = add(e, one);

                // keep only the fractional part
                const union {
                    unsigned long long int i;
                    double d;
                } mantissa_mask = { 0x800fffffffffffffull };
                x = _mm_and_pd(x, set(mantissa_mask.d));
                x = _mm_or_pd(x, set(0.5));

                const _M128D mask = cmplt(x, set(0.70710678118654757273731092936941422522068023681640625));

                const _M128D tmp = _mm_and_pd(x, mask);
                x = sub(x, one);
                x = add(x, tmp);

                e = sub(e, _mm_and_pd(one, mask));

                const _M128D z = mul(x, x);

                static const union {
                    unsigned short s[6 * 4];
                    double d[6];
                } P = { {
                    0x1bb0,0x93c3,0xb4c2,0x3f1a,
                    0x52f2,0x3f56,0xd6f5,0x3fdf,
                    0x6911,0xed92,0xd2ba,0x4012,
                    0xeb2e,0xc63e,0xff72,0x402c,
                    0xc84d,0x924b,0xefd6,0x4031,
                    0xdcf8,0x7d7e,0xd563,0x401e
                } };
                static const union {
                    unsigned short s[5 * 4];
                    double d[5];
                } Q = { {
                    0xef8e,0xae97,0x9320,0x4026,
                    0xc033,0x4e19,0x9d2c,0x4046,
                    0xbdbd,0xa326,0xbf33,0x4054,
                    0xae21,0xeb5e,0xc9e2,0x4051,
                    0x25b2,0x9e1f,0x200a,0x4037
                } };

                _M128D y = set(P.d[0]);
                for (int i = 1; i < 6; ++i) {
                    y = add(mul(y, x), set(P.d[i]));
                }
                _M128D y2 = add(set(Q.d[0]), x);
                for (int i = 1; i < 5; ++i) {
                    y2 = add(mul(y2, x), set(Q.d[i]));
                }
                y = mul(y, x);
                y = CAT(_mm_div_, SUFFIX)(y, y2);

                y = mul(y, z);
                y = sub(y, mul(e, set(2.121944400546905827679e-4)));
                y = sub(y, mul(z, set(0.5)));

                x = add(x, y);
                x = add(x, mul(e, set(0.693359375)));
                x = _mm_or_pd(x, invalid_mask); // negative arg will be NAN
                x = _mm_or_pd(_mm_andnot_pd(infinity_mask, x), _mm_and_pd(infinity_mask, set(-std::numeric_limits<double>::infinity())));
                return x;
            }
            static inline VectorType abs(const VectorType a) PURE {
                return CAT(_mm_and_, SUFFIX)(a, _mm_setabsmask_pd());
            }

            MINMAX
            static inline EntryType min(VectorType a) PURE {
                a = _mm_min_sd(a, _mm_unpackhi_pd(a, a));
                return _mm_cvtsd_f64(a);
            }
            static inline EntryType max(VectorType a) PURE {
                a = _mm_max_sd(a, _mm_unpackhi_pd(a, a));
                return _mm_cvtsd_f64(a);
            }
            static inline EntryType mul(VectorType a) PURE {
                a = _mm_mul_sd(a, _mm_shuffle_pd(a, a, _MM_SHUFFLE2(0, 1)));
                return _mm_cvtsd_f64(a);
            }
            static inline EntryType add(VectorType a) PURE {
                a = _mm_add_sd(a, _mm_shuffle_pd(a, a, _MM_SHUFFLE2(0, 1)));
                return _mm_cvtsd_f64(a);
            }
#undef SUFFIX
            static inline VectorType round(VectorType a) PURE {
#if VC_IMPL_SSE4_1
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

            static inline VectorType notMaskedToZero(VectorType a, _M128 mask) PURE { return CAT(_mm_and_, SUFFIX)(mask, a); }
            static inline VectorType set(const float a) PURE { return CAT(_mm_set1_, SUFFIX)(a); }
            static inline VectorType set(const float a, const float b, const float c, const float d) PURE { return CAT(_mm_set_, SUFFIX)(a, b, c, d); }
            static inline VectorType zero() PURE { return CAT(_mm_setzero_, SUFFIX)(); }
            static inline VectorType one()  PURE { return CAT(_mm_setone_, SUFFIX)(); }// set(1.f); }
            static inline _M128 concat(_M128D a, _M128D b) PURE { return _mm_movelh_ps(_mm_cvtpd_ps(a), _mm_cvtpd_ps(b)); }

            static inline void multiplyAndAdd(VectorType &v1, VectorType v2, VectorType v3) { v1 = add(mul(v1, v2), v3); }
            static inline VectorType mul(VectorType a, VectorType b, _M128 mask) PURE {
                return _mm_or_ps(
                    _mm_and_ps(mask, _mm_mul_ps(a, b)),
                    _mm_andnot_ps(mask, a)
                    );
            }
            static inline VectorType div(VectorType a, VectorType b, _M128 mask) PURE {
                return _mm_or_ps(
                    _mm_and_ps(mask, _mm_div_ps(a, b)),
                    _mm_andnot_ps(mask, a)
                    );
            }

            OP(add) OP(sub) OP(mul)
            OPcmp(eq) OPcmp(neq)
            OPcmp(lt) OPcmp(nlt)
            OPcmp(le) OPcmp(nle)

            OP1(sqrt) OP1(rsqrt)
            static inline VectorType isNaN(VectorType x) PURE {
                return _mm_cmpunord_ps(x, x);
            }
            static inline VectorType isFinite(VectorType x) PURE {
                return _mm_cmpord_ps(x, _mm_mul_ps(zero(), x));
            }
            static inline VectorType reciprocal(VectorType x) PURE {
                return _mm_rcp_ps(x);
            }
            static VectorType log(VectorType x) PURE {
                const _M128 one = set(1.);
                const _M128 invalid_mask = cmplt(x, zero());
                const _M128 infinity_mask = cmpeq(x, zero());

                x = max(x, set(std::numeric_limits<float>::min()));  // cut off denormalized stuff

                const _M128I emm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);
                _M128 e = _mm_cvtepi32_ps(_mm_sub_epi32(emm0, _mm_set1_epi32(127)));
                e = add(e, one);

                // keep only the fractional part
                const union {
                    unsigned int i;
                    float f;
                } mantissa_mask = { 0x807fffff };
                x = _mm_and_ps(x, set(mantissa_mask.f));
                x = _mm_or_ps(x, set(0.5));

                const _M128 mask = cmplt(x, set(0.707106781186547524f));

                const _M128 tmp = _mm_and_ps(x, mask);
                x = sub(x, one);
                x = add(x, tmp);

                e = sub(e, _mm_and_ps(one, mask));

                const _M128 z = mul(x, x);

                _M128 y = set( 7.0376836292e-2f);
                y = mul(y, x);
                y = add(y, set(-1.1514610310e-1f));
                y = mul(y, x);
                y = add(y, set( 1.1676998740e-1f));
                y = mul(y, x);
                y = add(y, set(-1.2420140846e-1f));
                y = mul(y, x);
                y = add(y, set( 1.4249322787e-1f));
                y = mul(y, x);
                y = add(y, set(-1.6668057665e-1f));
                y = mul(y, x);
                y = add(y, set( 2.0000714765e-1f));
                y = mul(y, x);
                y = add(y, set(-2.4999993993e-1f));
                y = mul(y, x);
                y = add(y, set( 3.3333331174e-1f));
                y = mul(y, x);

                y = mul(y, z);
                y = add(y, mul(e, set(-2.12194440e-4f)));
                y = sub(y, mul(z, set(0.5)));

                x = add(x, y);
                x = add(x, mul(e, set(0.693359375f)));
                x = _mm_or_ps(x, invalid_mask); // negative arg will be NAN
                x = _mm_or_ps(_mm_andnot_ps(infinity_mask, x), _mm_and_ps(infinity_mask, set(-std::numeric_limits<float>::infinity())));
                return x;
            }
            static inline VectorType abs(const VectorType a) PURE {
                return CAT(_mm_and_, SUFFIX)(a, _mm_setabsmask_ps());
            }

            MINMAX
            static inline EntryType min(VectorType a) PURE {
                a = _mm_min_ps(a, _mm_movehl_ps(a, a));   // a = min(a0, a2), min(a1, a3), min(a2, a2), min(a3, a3)
                a = _mm_min_ss(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 1, 1, 1))); // a = min(a0, a1), a1, a2, a3
                return _mm_cvtss_f32(a);
            }
            static inline EntryType max(VectorType a) PURE {
                a = _mm_max_ps(a, _mm_movehl_ps(a, a));   // a = max(a0, a2), max(a1, a3), max(a2, a2), max(a3, a3)
                a = _mm_max_ss(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 1, 1, 1))); // a = max(a0, a1), a1, a2, a3
                return _mm_cvtss_f32(a);
            }
            static inline EntryType mul(VectorType a) PURE {
                a = _mm_mul_ps(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3)));
                a = _mm_mul_ss(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 2, 0, 1)));
                return _mm_cvtss_f32(a);
            }
            static inline EntryType add(VectorType a) PURE {
                a = _mm_add_ps(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 1, 2, 3)));
                a = _mm_add_ss(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 2, 0, 1)));
                return _mm_cvtss_f32(a);
            }
#undef SUFFIX
            static inline VectorType round(VectorType a) PURE {
#if VC_IMPL_SSE4_1
                return _mm_round_ps(a, _MM_FROUND_NINT);
#else
                //XXX slow: _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
                return _mm_cvtepi32_ps(_mm_cvtps_epi32(a));
#endif
            }
        };

        template<> struct VectorHelper<float8> {
            typedef float EntryType;
            typedef M256 VectorType;

            static inline VectorType set(const float a) PURE {
                const _M128 x = _mm_set1_ps(a);
                return VectorType::create(x, x);
            }
            static inline VectorType set(const float a, const float b, const float c, const float d) PURE {
                const _M128 x = _mm_set_ps(a, b, c, d);
                return VectorType::create(x, x);
            }
            static inline VectorType set(const float a, const float b, const float c, const float d,
                    const float e, const float f, const float g, const float h) PURE {
                return VectorType::create(_mm_set_ps(a, b, c, d), _mm_set_ps(e, f, g, h));
            }
            static inline VectorType zero() PURE { return VectorType::create(_mm_setzero_ps(), _mm_setzero_ps()); }
            static inline VectorType one()  PURE { return set(1.f); }

#define REUSE_FLOAT_IMPL1(fun) \
            static inline VectorType fun(const VectorType &x) PURE { \
                return VectorType::create(VectorHelper<float>::fun(x[0]), VectorHelper<float>::fun(x[1])); \
            }
#define REUSE_FLOAT_IMPL2(fun) \
            static inline VectorType fun(const VectorType &x, const VectorType &y) PURE { \
                return VectorType::create(VectorHelper<float>::fun(x[0], y[0]), VectorHelper<float>::fun(x[1], y[1])); \
            }
#define REUSE_FLOAT_IMPL3(fun) \
            static inline VectorType fun(const VectorType &x, const VectorType &y, const VectorType &z) PURE { \
                return VectorType::create(VectorHelper<float>::fun(x[0], y[0], z[0]), VectorHelper<float>::fun(x[1], y[1], z[1])); \
            }
            REUSE_FLOAT_IMPL1(reciprocal)
            REUSE_FLOAT_IMPL1(sqrt)
            REUSE_FLOAT_IMPL1(rsqrt)
            REUSE_FLOAT_IMPL1(isNaN)
            REUSE_FLOAT_IMPL1(isFinite)
            REUSE_FLOAT_IMPL1(log)
            REUSE_FLOAT_IMPL1(abs)
            REUSE_FLOAT_IMPL1(round)

            REUSE_FLOAT_IMPL2(notMaskedToZero)
            REUSE_FLOAT_IMPL2(add)
            REUSE_FLOAT_IMPL2(sub)
            REUSE_FLOAT_IMPL2(mul)
            REUSE_FLOAT_IMPL2(cmple)
            REUSE_FLOAT_IMPL2(cmpnle)
            REUSE_FLOAT_IMPL2(cmplt)
            REUSE_FLOAT_IMPL2(cmpnlt)
            REUSE_FLOAT_IMPL2(cmpeq)
            REUSE_FLOAT_IMPL2(cmpneq)
            REUSE_FLOAT_IMPL2(min)
            REUSE_FLOAT_IMPL2(max)

            static inline EntryType min(const VectorType &a) PURE {
                return VectorHelper<float>::min(VectorHelper<float>::min(a[0], a[1]));
            }
            static inline EntryType max(const VectorType &a) PURE {
                return VectorHelper<float>::max(VectorHelper<float>::max(a[0], a[1]));
            }
            static inline EntryType mul(const VectorType &a) PURE {
                return VectorHelper<float>::mul(VectorHelper<float>::mul(a[0], a[1]));
            }
            static inline EntryType add(const VectorType &a) PURE {
                return VectorHelper<float>::add(VectorHelper<float>::add(a[0], a[1]));
            }

            static inline void multiplyAndAdd(VectorType &a, const VectorType &b, const VectorType &c) {
                VectorHelper<float>::multiplyAndAdd(a[0], b[0], c[0]);
                VectorHelper<float>::multiplyAndAdd(a[1], b[1], c[1]);
            }
            REUSE_FLOAT_IMPL3(mul)
            REUSE_FLOAT_IMPL3(div)
#undef REUSE_FLOAT_IMPL3
#undef REUSE_FLOAT_IMPL2
#undef REUSE_FLOAT_IMPL1
        };

        template<> struct VectorHelper<int> {
            typedef int EntryType;
            typedef _M128I VectorType;
#define SUFFIX si128

            OP_(or_) OP_(and_) OP_(xor_)
            static inline VectorType zero() PURE { return CAT(_mm_setzero_, SUFFIX)(); }
            static inline VectorType notMaskedToZero(VectorType a, _M128 mask) PURE { return CAT(_mm_and_, SUFFIX)(_mm_castps_si128(mask), a); }
#undef SUFFIX
#define SUFFIX epi32
            static inline VectorType one() PURE { return CAT(_mm_setone_, SUFFIX)(); }

            static inline VectorType set(const int a) PURE { return CAT(_mm_set1_, SUFFIX)(a); }
            static inline VectorType set(const int a, const int b, const int c, const int d) PURE { return CAT(_mm_set_, SUFFIX)(a, b, c, d); }

            static inline void multiplyAndAdd(VectorType &v1, VectorType v2, VectorType v3) { v1 = add(mul(v1, v2), v3); }

            static inline VectorType shiftLeft(VectorType a, int shift) {
                return CAT(_mm_slli_, SUFFIX)(a, shift);
            }
            static inline VectorType shiftRight(VectorType a, int shift) {
                return CAT(_mm_srai_, SUFFIX)(a, shift);
            }
            OP1(abs)

            MINMAX
            static inline EntryType min(VectorType a) PURE {
                a = min(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                // using lo_epi16 for speed here
                a = min(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                return _mm_cvtsi128_si32(a);
            }
            static inline EntryType max(VectorType a) PURE {
                a = max(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                // using lo_epi16 for speed here
                a = max(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                return _mm_cvtsi128_si32(a);
            }
            static inline EntryType add(VectorType a) PURE {
                a = add(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = add(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                return _mm_cvtsi128_si32(a);
            }
#if VC_IMPL_SSE4_1
            static inline VectorType mul(VectorType a, VectorType b) PURE { return _mm_mullo_epi32(a, b); }
            static inline EntryType mul(VectorType a) PURE {
                a = mul(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = mul(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                return _mm_cvtsi128_si32(a);
            }
#else
            static inline VectorType mul(const VectorType &a, const VectorType &b) PURE {
                const VectorType &aShift = _mm_srli_si128(a, 4);
                const VectorType &ab02 = _mm_mul_epu32(a, b); // [a0 * b0, a2 * b2]
                const VectorType &bShift = _mm_srli_si128(b, 4);
                const VectorType &ab13 = _mm_mul_epu32(aShift, bShift); // [a1 * b1, a3 * b3]
                return _mm_unpacklo_epi32(_mm_shuffle_epi32(ab02, 8), _mm_shuffle_epi32(ab13, 8));
            }
            static inline EntryType mul(VectorType a) PURE {
                STORE_VECTOR(int, _a, a);
                return _a[0] * _a[1] * _a[2] * _a[3];
            }
#endif
            static inline VectorType mul(const VectorType a, const VectorType b, _M128 _mask) PURE {
                return _mm_blendv_epi8(a, mul(a, b), _mm_castps_si128(_mask));
            }

            static inline VectorType div(const VectorType a, const VectorType b, _M128 _mask) PURE {
                const int mask = _mm_movemask_ps(_mask);
                STORE_VECTOR(int, _a, a);
                STORE_VECTOR(int, _b, b);
                union {
                    int i[4];
                    VectorType v;
                } x = { {
                    (mask & 1 ? _a[0] / _b[0] : _a[0]),
                    (mask & 2 ? _a[1] / _b[1] : _a[1]),
                    (mask & 4 ? _a[2] / _b[2] : _a[2]),
                    (mask & 8 ? _a[3] / _b[3] : _a[3])
                } };
                return x.v;
            }

            OP(add) OP(sub)
            OPcmp(eq)
            OPcmp(lt)
            OPcmp(gt)
            static inline VectorType cmpneq(const VectorType &a, const VectorType &b) PURE { _M128I x = cmpeq(a, b); return _mm_andnot_si128(x, _mm_setallone_si128()); }
            static inline VectorType cmpnlt(const VectorType &a, const VectorType &b) PURE { _M128I x = cmplt(a, b); return _mm_andnot_si128(x, _mm_setallone_si128()); }
            static inline VectorType cmple (const VectorType &a, const VectorType &b) PURE { _M128I x = cmpgt(a, b); return _mm_andnot_si128(x, _mm_setallone_si128()); }
            static inline VectorType cmpnle(const VectorType &a, const VectorType &b) PURE { return cmpgt(a, b); }
#undef SUFFIX
            static inline VectorType round(VectorType a) PURE { return a; }
        };

        template<> struct VectorHelper<unsigned int> {
            typedef unsigned int EntryType;
            typedef _M128I VectorType;
#define SUFFIX si128
            OP_CAST_(or_) OP_CAST_(and_) OP_CAST_(xor_)
            static inline VectorType zero() PURE { return CAT(_mm_setzero_, SUFFIX)(); }
            static inline VectorType notMaskedToZero(VectorType a, _M128 mask) PURE { return CAT(_mm_and_, SUFFIX)(_mm_castps_si128(mask), a); }

#undef SUFFIX
#define SUFFIX epu32
            static inline VectorType one() PURE { return CAT(_mm_setone_, SUFFIX)(); }

            MINMAX
            static inline EntryType min(VectorType a) PURE {
                a = min(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                // using lo_epi16 for speed here
                a = min(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                return _mm_cvtsi128_si32(a);
            }
            static inline EntryType max(VectorType a) PURE {
                a = max(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                // using lo_epi16 for speed here
                a = max(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                return _mm_cvtsi128_si32(a);
            }
            static inline EntryType mul(VectorType a) PURE {
                a = mul(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                // using lo_epi16 for speed here
                a = mul(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                return _mm_cvtsi128_si32(a);
            }
            static inline EntryType add(VectorType a) PURE {
                a = add(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                // using lo_epi16 for speed here
                a = add(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                return _mm_cvtsi128_si32(a);
            }

            static inline VectorType mul(const VectorType a, const VectorType b, _M128 _mask) PURE {
                return _mm_blendv_epi8(a, mul(a, b), _mm_castps_si128(_mask));
            }
            static inline VectorType mul(const VectorType &a, const VectorType &b) PURE {
                return VectorHelper<int>::mul(a, b);
            }
//X             template<unsigned int b> static inline VectorType mul(const VectorType a) PURE {
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
            static inline VectorType div(const VectorType &_a, const VectorType &_b, const _M128 _mask) PURE {
                const int mask = _mm_movemask_ps(_mask);
                VectorType _r = _a;
                typedef unsigned int uintA MAY_ALIAS;
                const uintA *b = reinterpret_cast<const uintA *>(&_b);
                uintA *r = reinterpret_cast<uintA *>(&_r);
                unrolled_loop16(i, 0, 4,
                    if (mask & (1 << i)) r[i] /= b[i];
                    );
                return _r;
            }

#undef SUFFIX
#define SUFFIX epi32
            static inline VectorType shiftLeft(VectorType a, int shift) {
                return CAT(_mm_slli_, SUFFIX)(a, shift);
            }
            static inline VectorType shiftRight(VectorType a, int shift) {
                return CAT(_mm_srli_, SUFFIX)(a, shift);
            }
            static inline VectorType set(const unsigned int a) PURE { return CAT(_mm_set1_, SUFFIX)(a); }
            static inline VectorType set(const unsigned int a, const unsigned int b, const unsigned int c, const unsigned int d) PURE { return CAT(_mm_set_, SUFFIX)(a, b, c, d); }

            OP(add) OP(sub)
            OPcmp(eq)
            static inline VectorType cmpneq(const VectorType &a, const VectorType &b) PURE { return _mm_andnot_si128(cmpeq(a, b), _mm_setallone_si128()); }

#ifndef USE_INCORRECT_UNSIGNED_COMPARE
            static inline VectorType cmplt(const VectorType &a, const VectorType &b) PURE {
                return _mm_cmplt_epu32(a, b);
            }
            static inline VectorType cmpgt(const VectorType &a, const VectorType &b) PURE {
                return _mm_cmpgt_epu32(a, b);
            }
#else
            OPcmp(lt)
            OPcmp(gt)
#endif
            static inline VectorType cmpnlt(const VectorType &a, const VectorType &b) PURE { return _mm_andnot_si128(cmplt(a, b), _mm_setallone_si128()); }
            static inline VectorType cmple (const VectorType &a, const VectorType &b) PURE { return _mm_andnot_si128(cmpgt(a, b), _mm_setallone_si128()); }
            static inline VectorType cmpnle(const VectorType &a, const VectorType &b) PURE { return cmpgt(a, b); }

#undef SUFFIX
            static inline VectorType round(VectorType a) PURE { return a; }
        };

        template<> struct VectorHelper<signed short> {
            typedef _M128I VectorType;
            typedef signed short EntryType;
#define SUFFIX si128

            OP_(or_) OP_(and_) OP_(xor_)
            static inline VectorType zero() PURE { return CAT(_mm_setzero_, SUFFIX)(); }
            static inline VectorType notMaskedToZero(VectorType a, _M128 mask) PURE { return CAT(_mm_and_, SUFFIX)(_mm_castps_si128(mask), a); }
            static inline _M128I concat(_M128I a, _M128I b) PURE { return _mm_packs_epi32(a, b); }
            static inline _M128I expand0(_M128I x) PURE { return _mm_srai_epi32(_mm_unpacklo_epi16(x, x), 16); }
            static inline _M128I expand1(_M128I x) PURE { return _mm_srai_epi32(_mm_unpackhi_epi16(x, x), 16); }

#undef SUFFIX
#define SUFFIX epi16
            static inline VectorType one() PURE { return CAT(_mm_setone_, SUFFIX)(); }

            static inline VectorType shiftLeft(VectorType a, int shift) {
                return CAT(_mm_slli_, SUFFIX)(a, shift);
            }
            static inline VectorType shiftRight(VectorType a, int shift) {
                return CAT(_mm_srai_, SUFFIX)(a, shift);
            }
            static inline VectorType set(const EntryType a) PURE { return CAT(_mm_set1_, SUFFIX)(a); }
            static inline VectorType set(const EntryType a, const EntryType b, const EntryType c, const EntryType d,
                    const EntryType e, const EntryType f, const EntryType g, const EntryType h) PURE {
                return CAT(_mm_set_, SUFFIX)(a, b, c, d, e, f, g, h);
            }

            static inline void multiplyAndAdd(VectorType &v1, VectorType v2, VectorType v3) {
                v1 = add(mul(v1, v2), v3); }

            OP1(abs)

            static inline VectorType mul(VectorType a, VectorType b, _M128 _mask) PURE {
                _M128I mask = _mm_castps_si128(_mask);
                return _mm_or_si128(
                    _mm_and_si128(mask, mul(a, b)),
                    _mm_andnot_si128(mask, a)
                    );
            }
            OPx(mul, mullo)
            OP(min) OP(max)
            static inline EntryType min(VectorType a) PURE {
                // reminder: _MM_SHUFFLE(3, 2, 1, 0) means "no change"
                a = min(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = min(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = min(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)));
                return _mm_cvtsi128_si32(a); // & 0xffff is implicit
            }
            static inline EntryType max(VectorType a) PURE {
                // reminder: _MM_SHUFFLE(3, 2, 1, 0) means "no change"
                a = max(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = max(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = max(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)));
                return _mm_cvtsi128_si32(a); // & 0xffff is implicit
            }
            static inline EntryType mul(VectorType a) PURE {
                a = mul(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = mul(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = mul(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)));
                return _mm_cvtsi128_si32(a); // & 0xffff is implicit
            }
            static inline EntryType add(VectorType a) PURE {
                a = add(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = add(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = add(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)));
                return _mm_cvtsi128_si32(a); // & 0xffff is implicit
            }

            static inline VectorType div(const VectorType &a, const VectorType &b, const _M128 _mask) PURE {
                const int mask = _mm_movemask_epi8(_mm_castps_si128(_mask));
                VectorType r = a;
                typedef EntryType Alias MAY_ALIAS;
                const Alias *bb = reinterpret_cast<const Alias *>(&b);
                Alias *rr = reinterpret_cast<Alias *>(&r);
                unrolled_loop16(i, 0, 8,
                    if (mask & (1 << i * 2)) rr[i] /= bb[i];
                    );
                return r;
            }

            OP(add) OP(sub)
            OPcmp(eq)
            OPcmp(lt)
            OPcmp(gt)
            static inline VectorType cmpneq(const VectorType &a, const VectorType &b) PURE { _M128I x = cmpeq(a, b); return _mm_andnot_si128(x, _mm_setallone_si128()); }
            static inline VectorType cmpnlt(const VectorType &a, const VectorType &b) PURE { _M128I x = cmplt(a, b); return _mm_andnot_si128(x, _mm_setallone_si128()); }
            static inline VectorType cmple (const VectorType &a, const VectorType &b) PURE { _M128I x = cmpgt(a, b); return _mm_andnot_si128(x, _mm_setallone_si128()); }
            static inline VectorType cmpnle(const VectorType &a, const VectorType &b) PURE { return cmpgt(a, b); }
#undef SUFFIX
            static inline VectorType round(VectorType a) PURE { return a; }
        };

        template<> struct VectorHelper<unsigned short> {
            typedef _M128I VectorType;
            typedef unsigned short EntryType;
#define SUFFIX si128
            OP_CAST_(or_) OP_CAST_(and_) OP_CAST_(xor_)
            static inline VectorType zero() PURE { return CAT(_mm_setzero_, SUFFIX)(); }
            static inline VectorType notMaskedToZero(VectorType a, _M128 mask) PURE { return CAT(_mm_and_, SUFFIX)(_mm_castps_si128(mask), a); }
#if VC_IMPL_SSE4_1
            static inline _M128I concat(_M128I a, _M128I b) PURE { return _mm_packus_epi32(a, b); }
#else
            // XXX too bad, but this is broken without SSE 4.1
            static inline _M128I concat(_M128I a, _M128I b) PURE { return _mm_packs_epi32(a, b); }
#endif
            static inline _M128I expand0(_M128I x) PURE { return _mm_srli_epi32(_mm_unpacklo_epi16(x, x), 16); }
            static inline _M128I expand1(_M128I x) PURE { return _mm_srli_epi32(_mm_unpackhi_epi16(x, x), 16); }

#undef SUFFIX
#define SUFFIX epu16
            static inline VectorType one() PURE { return CAT(_mm_setone_, SUFFIX)(); }
            static inline VectorType div(const VectorType &a, const VectorType &b, const _M128 _mask) PURE {
                const int mask = _mm_movemask_epi8(_mm_castps_si128(_mask));
                VectorType r = a;
                typedef EntryType Alias MAY_ALIAS;
                const Alias *bb = reinterpret_cast<const Alias *>(&b);
                Alias *rr = reinterpret_cast<Alias *>(&r);
                unrolled_loop16(i, 0, 8,
                    if (mask & (1 << i * 2)) rr[i] /= bb[i];
                    );
                return r;
            }

            static inline VectorType mul(VectorType a, VectorType b, _M128 _mask) PURE {
                _M128I mask = _mm_castps_si128(_mask);
                return _mm_or_si128(
                    _mm_and_si128(mask, mul(a, b)),
                    _mm_andnot_si128(mask, a)
                    );
            }
//X             template<unsigned int b> static inline VectorType mul(const VectorType a) PURE {
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
            static inline VectorType shiftLeft(VectorType a, int shift) {
                return CAT(_mm_slli_, SUFFIX)(a, shift);
            }
            static inline VectorType shiftRight(VectorType a, int shift) {
                return CAT(_mm_srli_, SUFFIX)(a, shift);
            }
            OPx(mul, mullo) // should work correctly for all values
#if defined(USE_INCORRECT_UNSIGNED_COMPARE) && !defined(VC_IMPL_SSE4_1)
            OP(min) OP(max) // XXX breaks for values with MSB set
#endif
            static inline EntryType min(VectorType a) PURE {
                // reminder: _MM_SHUFFLE(3, 2, 1, 0) means "no change"
                a = min(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = min(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = min(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)));
                return _mm_cvtsi128_si32(a); // & 0xffff is implicit
            }
            static inline EntryType max(VectorType a) PURE {
                // reminder: _MM_SHUFFLE(3, 2, 1, 0) means "no change"
                a = max(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = max(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = max(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)));
                return _mm_cvtsi128_si32(a); // & 0xffff is implicit
            }
            static inline EntryType mul(VectorType a) PURE {
                // reminder: _MM_SHUFFLE(3, 2, 1, 0) means "no change"
                a = mul(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = mul(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = mul(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)));
                return _mm_cvtsi128_si32(a); // & 0xffff is implicit
            }
            static inline EntryType add(VectorType a) PURE {
                // reminder: _MM_SHUFFLE(3, 2, 1, 0) means "no change"
                a = add(a, _mm_shuffle_epi32(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = add(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 0, 3, 2)));
                a = add(a, _mm_shufflelo_epi16(a, _MM_SHUFFLE(1, 1, 1, 1)));
                return _mm_cvtsi128_si32(a); // & 0xffff is implicit
            }
            static inline VectorType set(const EntryType a) PURE { return CAT(_mm_set1_, SUFFIX)(a); }
            static inline VectorType set(const EntryType a, const EntryType b, const EntryType c,
                    const EntryType d, const EntryType e, const EntryType f,
                    const EntryType g, const EntryType h) PURE {
                return CAT(_mm_set_, SUFFIX)(a, b, c, d, e, f, g, h);
            }

            OP(add) OP(sub)
            OPcmp(eq)
            static inline VectorType cmpneq(const VectorType &a, const VectorType &b) PURE { return _mm_andnot_si128(cmpeq(a, b), _mm_setallone_si128()); }

#ifndef USE_INCORRECT_UNSIGNED_COMPARE
            static inline VectorType cmplt(const VectorType &a, const VectorType &b) PURE {
                return _mm_cmplt_epu16(a, b);
            }
            static inline VectorType cmpgt(const VectorType &a, const VectorType &b) PURE {
                return _mm_cmpgt_epu16(a, b);
            }
#else
            OPcmp(lt)
            OPcmp(gt)
#endif
            static inline VectorType cmpnlt(const VectorType &a, const VectorType &b) PURE { return _mm_andnot_si128(cmplt(a, b), _mm_setallone_si128()); }
            static inline VectorType cmple (const VectorType &a, const VectorType &b) PURE { return _mm_andnot_si128(cmpgt(a, b), _mm_setallone_si128()); }
            static inline VectorType cmpnle(const VectorType &a, const VectorType &b) PURE { return cmpgt(a, b); }
#undef SUFFIX
            static inline VectorType round(VectorType a) PURE { return a; }
        };
#undef OP1
#undef OP
#undef OP_
#undef OPx
#undef OPcmp

} // namespace SSE
} // namespace Vc

#include "vectorhelper.tcc"

#endif // SSE_VECTORHELPER_H
