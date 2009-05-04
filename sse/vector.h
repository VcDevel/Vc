/*
    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) version 3.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Library General Public License for more details.

    You should have received a copy of the GNU Library General Public License
    along with this library; see the file COPYING.LIB.  If not, write to
    the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
    Boston, MA 02110-1301, USA.

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

#ifndef SSE_VECTOR_H
#define SSE_VECTOR_H

#include "intrinsics.h"
#include <algorithm>
#include <limits>

#ifndef _M128
# define _M128 __m128
#endif

#ifndef _M128I
# define _M128I __m128i
#endif

#ifndef _M128D
# define _M128D __m128d
#endif

#ifndef ALIGN
# ifdef __GNUC__
#  define ALIGN(n) __attribute__((aligned(n)))
# else
#  define ALIGN(n) __declspec(align(n))
# endif
#endif

#define CAT_HELPER(a, b) a##b
#define CAT(a, b) CAT_HELPER(a, b)

namespace SSE
{
    namespace Internal
    {
        ALIGN(16) extern const unsigned int   _IndexesFromZero4[4];
        ALIGN(16) extern const unsigned short _IndexesFromZero8[8];
    } // namespace Internal

    enum { VectorAlignment = 16 };
    template<typename T> class Vector;
    template<unsigned int VectorSize> class Mask;

    ALIGN(16) static const int _FullMaskData[4] = { 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff };
#define _0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF _mm_load_si128(reinterpret_cast<const __m128i *const>(_FullMaskData))

#define STORE_VECTOR(type, name, vec) \
    union { __m128i p; type v[16 / sizeof(type)]; } CAT(u, __LINE__); \
    _mm_store_si128(&CAT(u, __LINE__).p, vec); \
    const type *const name = &CAT(u, __LINE__).v[0]

#define PARENT_DATA (static_cast<Parent *>(this)->data)
#define PARENT_DATA_CONST (static_cast<const Parent *>(this)->data)
        template<typename T, typename Parent>
        struct VectorBase
        {
            typedef _M128 IntrinType;
            operator _M128() { return PARENT_DATA; }
            operator const _M128() const { return PARENT_DATA_CONST; }
        };
        template<typename Parent>
        struct VectorBase<float, Parent>
        {
            typedef _M128 IntrinType;
            operator _M128() { return PARENT_DATA; }
            operator const _M128() const { return PARENT_DATA_CONST; }
        };
        template<typename Parent>
        struct VectorBase<double, Parent>
        {
            typedef _M128D IntrinType;
            operator _M128D() { return PARENT_DATA; }
            operator const _M128D() const { return PARENT_DATA_CONST; }
        };
#define OP_DECL(symbol) \
            inline Vector<T> &operator symbol##=(const Vector<T> &x); \
            inline Vector<T> operator symbol(const Vector<T> &x) const;
        template<typename Parent>
        struct VectorBase<int, Parent>
        {
            typedef _M128I IntrinType;
            operator _M128I() { return PARENT_DATA; }
            operator const _M128I() const { return PARENT_DATA_CONST; }
#define T int
            OP_DECL(|)
            OP_DECL(&)
            OP_DECL(^)
            OP_DECL(>>)
            OP_DECL(<<)
#undef T

            Vector<short> operator,(Vector<int>) const;

            protected:
                const int *_IndexesFromZero() { return reinterpret_cast<const int *>(Internal::_IndexesFromZero4); }
        };
        template<typename Parent>
        struct VectorBase<unsigned int, Parent>
        {
            typedef _M128I IntrinType;
            operator _M128I() { return PARENT_DATA; }
            operator const _M128I() const { return PARENT_DATA_CONST; }
#define T unsigned int
            OP_DECL(|)
            OP_DECL(&)
            OP_DECL(^)
            OP_DECL(>>)
            OP_DECL(<<)
#undef T

            Vector<unsigned short> operator,(Vector<unsigned int>) const;

            protected:
                const unsigned int *_IndexesFromZero() { return reinterpret_cast<const unsigned int *>(Internal::_IndexesFromZero4); }
        };
        template<typename Parent>
        struct VectorBase<short, Parent>
        {
            typedef _M128I IntrinType;
            operator _M128I() { return PARENT_DATA; }
            operator const _M128I() const { return PARENT_DATA_CONST; }
#define T short
            OP_DECL(|)
            OP_DECL(&)
            OP_DECL(^)
            OP_DECL(>>)
            OP_DECL(<<)
#undef T
            protected:
                const short *_IndexesFromZero() { return reinterpret_cast<const short *>(Internal::_IndexesFromZero8); }
        };
        template<typename Parent>
        struct VectorBase<unsigned short, Parent>
        {
            typedef _M128I IntrinType;
            operator _M128I() { return PARENT_DATA; }
            operator const _M128I() const { return PARENT_DATA_CONST; }
#define T unsigned short
            OP_DECL(|)
            OP_DECL(&)
            OP_DECL(^)
            OP_DECL(>>)
            OP_DECL(<<)
#undef T
            protected:
                const unsigned short *_IndexesFromZero() { return reinterpret_cast<const unsigned short *>(Internal::_IndexesFromZero8); }
        };
#undef OP_DECL
#undef PARENT_DATA
#undef PARENT_DATA_CONST

        template<typename From, typename To> struct StaticCastHelper {};
        template<> struct StaticCastHelper<float       , int         > { static _M128I cast(const _M128  &v) { return _mm_cvtps_epi32(v); } };
        template<> struct StaticCastHelper<double      , int         > { static _M128I cast(const _M128D &v) { return _mm_cvtpd_epi32(v); } };
        template<> struct StaticCastHelper<int         , int         > { static _M128I cast(const _M128I &v) { return v; } };
        template<> struct StaticCastHelper<unsigned int, int         > { static _M128I cast(const _M128I &v) { return v; } };
        template<> struct StaticCastHelper<float       , unsigned int> { static _M128I cast(const _M128  &v) { return _mm_cvtps_epi32(v); } };
        template<> struct StaticCastHelper<double      , unsigned int> { static _M128I cast(const _M128D &v) { return _mm_cvtpd_epi32(v); } };
        template<> struct StaticCastHelper<int         , unsigned int> { static _M128I cast(const _M128I &v) { return v; } };
        template<> struct StaticCastHelper<unsigned int, unsigned int> { static _M128I cast(const _M128I &v) { return v; } };
        template<> struct StaticCastHelper<float       , float       > { static _M128  cast(const _M128  &v) { return v; } };
        template<> struct StaticCastHelper<double      , float       > { static _M128  cast(const _M128D &v) { return _mm_cvtpd_ps(v); } };
        template<> struct StaticCastHelper<int         , float       > { static _M128  cast(const _M128I &v) { return _mm_cvtepi32_ps(v); } };
        template<> struct StaticCastHelper<unsigned int, float       > { static _M128  cast(const _M128I &v) { return _mm_cvtepi32_ps(v); } };
        template<> struct StaticCastHelper<float       , double      > { static _M128D cast(const _M128  &v) { return _mm_cvtps_pd(v); } };
        template<> struct StaticCastHelper<double      , double      > { static _M128D cast(const _M128D &v) { return v; } };
        template<> struct StaticCastHelper<int         , double      > { static _M128D cast(const _M128I &v) { return _mm_cvtepi32_pd(v); } };
        template<> struct StaticCastHelper<unsigned int, double      > { static _M128D cast(const _M128I &v) { return _mm_cvtepi32_pd(v); } };

        template<> struct StaticCastHelper<unsigned short, short         > { static _M128I cast(const _M128I &v) { return v; } };
        template<> struct StaticCastHelper<unsigned short, unsigned short> { static _M128I cast(const _M128I &v) { return v; } };
        template<> struct StaticCastHelper<short         , unsigned short> { static _M128I cast(const _M128I &v) { return v; } };
        template<> struct StaticCastHelper<short         , short         > { static _M128I cast(const _M128I &v) { return v; } };

        template<typename From, typename To> struct ReinterpretCastHelper {};
        template<> struct ReinterpretCastHelper<float       , int         > { static _M128I cast(const _M128  &v) { return _mm_castps_si128(v); } };
        template<> struct ReinterpretCastHelper<double      , int         > { static _M128I cast(const _M128D &v) { return _mm_castpd_si128(v); } };
        template<> struct ReinterpretCastHelper<int         , int         > { static _M128I cast(const _M128I &v) { return v; } };
        template<> struct ReinterpretCastHelper<unsigned int, int         > { static _M128I cast(const _M128I &v) { return v; } };
        template<> struct ReinterpretCastHelper<float       , unsigned int> { static _M128I cast(const _M128  &v) { return _mm_castps_si128(v); } };
        template<> struct ReinterpretCastHelper<double      , unsigned int> { static _M128I cast(const _M128D &v) { return _mm_castpd_si128(v); } };
        template<> struct ReinterpretCastHelper<int         , unsigned int> { static _M128I cast(const _M128I &v) { return v; } };
        template<> struct ReinterpretCastHelper<unsigned int, unsigned int> { static _M128I cast(const _M128I &v) { return v; } };
        template<> struct ReinterpretCastHelper<float       , float       > { static _M128  cast(const _M128  &v) { return v; } };
        template<> struct ReinterpretCastHelper<double      , float       > { static _M128  cast(const _M128D &v) { return _mm_castpd_ps(v); } };
        template<> struct ReinterpretCastHelper<int         , float       > { static _M128  cast(const _M128I &v) { return _mm_castsi128_ps(v); } };
        template<> struct ReinterpretCastHelper<unsigned int, float       > { static _M128  cast(const _M128I &v) { return _mm_castsi128_ps(v); } };
        template<> struct ReinterpretCastHelper<float       , double      > { static _M128D cast(const _M128  &v) { return _mm_castps_pd(v); } };
        template<> struct ReinterpretCastHelper<double      , double      > { static _M128D cast(const _M128D &v) { return v; } };
        template<> struct ReinterpretCastHelper<int         , double      > { static _M128D cast(const _M128I &v) { return _mm_castsi128_pd(v); } };
        template<> struct ReinterpretCastHelper<unsigned int, double      > { static _M128D cast(const _M128I &v) { return _mm_castsi128_pd(v); } };

        template<> struct ReinterpretCastHelper<unsigned short, short         > { static _M128I cast(const _M128I &v) { return v; } };
        template<> struct ReinterpretCastHelper<unsigned short, unsigned short> { static _M128I cast(const _M128I &v) { return v; } };
        template<> struct ReinterpretCastHelper<short         , unsigned short> { static _M128I cast(const _M128I &v) { return v; } };
        template<> struct ReinterpretCastHelper<short         , short         > { static _M128I cast(const _M128I &v) { return v; } };

        template<typename To, typename From> static inline To mm128_reinterpret_cast(From v) { return v; }
        template<> inline _M128I mm128_reinterpret_cast<_M128I, _M128 >(_M128  v) { return _mm_castps_si128(v); }
        template<> inline _M128I mm128_reinterpret_cast<_M128I, _M128D>(_M128D v) { return _mm_castpd_si128(v); }
        template<> inline _M128  mm128_reinterpret_cast<_M128 , _M128D>(_M128D v) { return _mm_castpd_ps(v);    }
        template<> inline _M128  mm128_reinterpret_cast<_M128 , _M128I>(_M128I v) { return _mm_castsi128_ps(v); }
        template<> inline _M128D mm128_reinterpret_cast<_M128D, _M128I>(_M128I v) { return _mm_castsi128_pd(v); }
        template<> inline _M128D mm128_reinterpret_cast<_M128D, _M128 >(_M128  v) { return _mm_castps_pd(v);    }

        template<typename T> struct VectorHelper {};

#define OP1(op) \
        static inline TYPE op(const TYPE &a) { return CAT(_mm_##op##_, SUFFIX)(a); }
#define OP(op) \
        static inline TYPE op(const TYPE &a, const TYPE &b) { return CAT(_mm_##op##_ , SUFFIX)(a, b); }
#define OP_(op) \
        static inline TYPE op(const TYPE &a, const TYPE &b) { return CAT(_mm_##op    , SUFFIX)(a, b); }
#define OPx(op, op2) \
        static inline TYPE op(const TYPE &a, const TYPE &b) { return CAT(_mm_##op2##_, SUFFIX)(a, b); }
#define OPcmp(op) \
        static inline TYPE cmp##op(const TYPE &a, const TYPE &b) { return CAT(_mm_cmp##op##_, SUFFIX)(a, b); }
#define OP_CAST_(op) \
        static inline TYPE op(const TYPE &a, const TYPE &b) { return CAT(_mm_castps_, SUFFIX)( \
            _mm_##op##ps(CAT(CAT(_mm_cast, SUFFIX), _ps)(a), \
              CAT(CAT(_mm_cast, SUFFIX), _ps)(b))); \
        }
#define MINMAX \
        static inline TYPE min(TYPE a, TYPE b) { return CAT(_mm_min_, SUFFIX)(a, b); } \
        static inline TYPE max(TYPE a, TYPE b) { return CAT(_mm_max_, SUFFIX)(a, b); }
#define LOAD(T) \
        static inline TYPE load (const T *x) { return CAT(_mm_load_, SUFFIX)(x); }
#define LOAD_CAST(T) \
        static inline TYPE load (const T *x) { return CAT(_mm_load_, SUFFIX)(reinterpret_cast<const TYPE *>(x)); }
#define STORE(T) \
            static inline void store (T *mem, TYPE x) { return CAT(_mm_store_ , SUFFIX)(mem, x); } \
            static inline void storeStreaming(T *mem, TYPE x) { return CAT(_mm_stream_ , SUFFIX)(mem, x); }
#define STORE_CAST(T) \
            static inline void store (T *mem, TYPE x) { return CAT(_mm_store_, SUFFIX)(reinterpret_cast<TYPE *>(mem), x); } \
            static inline void storeStreaming(T *mem, TYPE x) { return CAT(_mm_stream_, SUFFIX)(reinterpret_cast<TYPE *>(mem), x); }

        // _mm_sll_* does not take a count parameter with different counts per vector element. So we
        // have to do it manually
#define SHIFT4 \
            static inline TYPE sll(TYPE v, __m128i count) { \
                STORE_VECTOR(unsigned int, shifts, count); \
                union { __m128i v; unsigned int i[4]; } data; \
                _mm_store_si128(&data.v, v); \
                data.i[0] <<= shifts[0]; \
                data.i[1] <<= shifts[1]; \
                data.i[2] <<= shifts[2]; \
                data.i[3] <<= shifts[3]; \
                return data.v; } \
            static inline TYPE slli(TYPE v, int count) { return CAT(_mm_slli_, SUFFIX)(v, count); } \
            static inline TYPE srl(TYPE v, __m128i count) { \
                STORE_VECTOR(unsigned int, shifts, count); \
                union { __m128i v; unsigned int i[4]; } data; \
                _mm_store_si128(&data.v, v); \
                data.i[0] >>= shifts[0]; \
                data.i[1] >>= shifts[1]; \
                data.i[2] >>= shifts[2]; \
                data.i[3] >>= shifts[3]; \
                return data.v; } \
            static inline TYPE srli(TYPE v, int count) { return CAT(_mm_srli_, SUFFIX)(v, count); }
#define SHIFT8 \
            static inline TYPE sll(TYPE v, __m128i count) { \
                STORE_VECTOR(unsigned short, shifts, count); \
                union { __m128i v; unsigned short i[8]; } data; \
                _mm_store_si128(&data.v, v); \
                data.i[0] <<= shifts[0]; \
                data.i[1] <<= shifts[1]; \
                data.i[2] <<= shifts[2]; \
                data.i[3] <<= shifts[3]; \
                data.i[4] <<= shifts[4]; \
                data.i[5] <<= shifts[5]; \
                data.i[6] <<= shifts[6]; \
                data.i[7] <<= shifts[7]; \
                return data.v; } \
            static inline TYPE slli(TYPE v, int count) { return CAT(_mm_slli_, SUFFIX)(v, count); } \
            static inline TYPE srl(TYPE v, __m128i count) { \
                STORE_VECTOR(unsigned short, shifts, count); \
                union { __m128i v; unsigned short i[8]; } data; \
                _mm_store_si128(&data.v, v); \
                data.i[0] >>= shifts[0]; \
                data.i[1] >>= shifts[1]; \
                data.i[2] >>= shifts[2]; \
                data.i[3] >>= shifts[3]; \
                data.i[4] >>= shifts[4]; \
                data.i[5] >>= shifts[5]; \
                data.i[6] >>= shifts[6]; \
                data.i[7] >>= shifts[7]; \
                return data.v; } \
            static inline TYPE srli(TYPE v, int count) { return CAT(_mm_srli_, SUFFIX)(v, count); }
#define GATHER_SCATTER(T) \
            static inline void gather(TYPE &v, const _M128I &indexes, const T *baseAddr) { \
                union { __m128i p; unsigned int i[4]; } u; _mm_store_si128(&u.p, indexes); \
                v = CAT(_mm_set_, SUFFIX)( \
                        baseAddr[u.i[3]], \
                        baseAddr[u.i[2]], \
                        baseAddr[u.i[1]], \
                        baseAddr[u.i[0]] \
                        ); \
            } \
            template<typename S> \
            static inline void gather(TYPE &v, const _M128I &indexes, const S *baseAddr, const T S::* member1) { \
                union { __m128i p; unsigned int i[4]; } u; _mm_store_si128(&u.p, indexes); \
                v = CAT(_mm_set_, SUFFIX)( \
                        baseAddr[u.i[3]].*(member1), \
                        baseAddr[u.i[2]].*(member1), \
                        baseAddr[u.i[1]].*(member1), \
                        baseAddr[u.i[0]].*(member1) \
                        ); \
            } \
            template<typename S1, typename S2> \
            static inline void gather(TYPE &v, const _M128I &indexes, const S1 *baseAddr, const S2 S1::* member1, const T S2::* member2) { \
                union { __m128i p; unsigned int i[4]; } u; _mm_store_si128(&u.p, indexes); \
                v = CAT(_mm_set_, SUFFIX)( \
                        baseAddr[u.i[3]].*(member1).*(member2), \
                        baseAddr[u.i[2]].*(member1).*(member2), \
                        baseAddr[u.i[1]].*(member1).*(member2), \
                        baseAddr[u.i[0]].*(member1).*(member2) \
                        ); \
            } \
            static inline void scatter(const TYPE &v, const _M128I &indexes, T *baseAddr) { \
                union { __m128i p; unsigned int i[4]; } u; _mm_store_si128(&u.p, indexes); \
                union { TYPE p; T v[4]; } w; store(w.v, v); \
                baseAddr[u.i[0]] = w.v[0]; \
                baseAddr[u.i[1]] = w.v[1]; \
                baseAddr[u.i[2]] = w.v[2]; \
                baseAddr[u.i[3]] = w.v[3]; \
            } \
            template<typename S> \
            static inline void scatter(const TYPE &v, const _M128I &indexes, S *baseAddr, T S::* member1) { \
                union { __m128i p; unsigned int i[4]; } u; _mm_store_si128(&u.p, indexes); \
                union { TYPE p; T v[4]; } w; store(w.v, v); \
                baseAddr[u.i[0]].*(member1) = w.v[0]; \
                baseAddr[u.i[1]].*(member1) = w.v[1]; \
                baseAddr[u.i[2]].*(member1) = w.v[2]; \
                baseAddr[u.i[3]].*(member1) = w.v[3]; \
            } \
            template<typename S1, typename S2> \
            static inline void scatter(const TYPE &v, const _M128I &indexes, S1 *baseAddr, S2 S1::* member1, T S2::* member2) { \
                union { __m128i p; unsigned int i[4]; } u; _mm_store_si128(&u.p, indexes); \
                union { TYPE p; T v[4]; } w; store(w.v, v); \
                baseAddr[u.i[0]].*(member1).*(member2) = w.v[0]; \
                baseAddr[u.i[1]].*(member1).*(member2) = w.v[1]; \
                baseAddr[u.i[2]].*(member1).*(member2) = w.v[2]; \
                baseAddr[u.i[3]].*(member1).*(member2) = w.v[3]; \
            }
#define GATHER_SCATTER_16(T) \
            static inline void gather(TYPE &v, const _M128I &indexes, const T *baseAddr) { \
                union { __m128i p; unsigned short i[8]; } u; _mm_store_si128(&u.p, indexes); \
                v = CAT(_mm_setr_, SUFFIX)(baseAddr[u.i[0]], baseAddr[u.i[1]], baseAddr[u.i[2]], baseAddr[u.i[3]], baseAddr[u.i[4]], baseAddr[u.i[5]], baseAddr[u.i[6]], baseAddr[u.i[7]]); \
            } \
            template<typename S> \
            static inline void gather(TYPE &v, const _M128I &indexes, const S *baseAddr, const T S::* member1) { \
                union { __m128i p; unsigned short i[8]; } u; _mm_store_si128(&u.p, indexes); \
                v = CAT(_mm_setr_, SUFFIX)(baseAddr[u.i[0]].*(member1), baseAddr[u.i[1]].*(member1), \
                        baseAddr[u.i[2]].*(member1), baseAddr[u.i[3]].*(member1), baseAddr[u.i[4]].*(member1), \
                        baseAddr[u.i[5]].*(member1), baseAddr[u.i[6]].*(member1), baseAddr[u.i[7]].*(member1)); \
            } \
            template<typename S1, typename S2> \
            static inline void gather(TYPE &v, const _M128I &indexes, const S1 *baseAddr, const S2 S1::* member1, const T S2::* member2) { \
                union { __m128i p; unsigned short i[8]; } u; _mm_store_si128(&u.p, indexes); \
                v = CAT(_mm_setr_, SUFFIX)(baseAddr[u.i[0]].*(member1).*(member2), \
                        baseAddr[u.i[1]].*(member1).*(member2), baseAddr[u.i[2]].*(member1).*(member2), \
                        baseAddr[u.i[3]].*(member1).*(member2), baseAddr[u.i[4]].*(member1).*(member2), \
                        baseAddr[u.i[5]].*(member1).*(member2), baseAddr[u.i[6]].*(member1).*(member2), \
                        baseAddr[u.i[7]].*(member1).*(member2)); \
            } \
            static inline void scatter(const TYPE &v, const _M128I &indexes, T *baseAddr) { \
                union { __m128i p; unsigned short i[8]; } u; _mm_store_si128(&u.p, indexes); \
                union { TYPE p; T v[8]; } w; store(w.v, v); \
                baseAddr[u.i[0]] = w.v[0]; \
                baseAddr[u.i[1]] = w.v[1]; \
                baseAddr[u.i[2]] = w.v[2]; \
                baseAddr[u.i[3]] = w.v[3]; \
                baseAddr[u.i[4]] = w.v[4]; \
                baseAddr[u.i[5]] = w.v[5]; \
                baseAddr[u.i[6]] = w.v[6]; \
                baseAddr[u.i[7]] = w.v[7]; \
            } \
            template<typename S> \
            static inline void scatter(const TYPE &v, const _M128I &indexes, S *baseAddr, T S::* member1) { \
                union { __m128i p; unsigned short i[8]; } u; _mm_store_si128(&u.p, indexes); \
                union { TYPE p; T v[8]; } w; store(w.v, v); \
                baseAddr[u.i[0]].*(member1) = w.v[0]; \
                baseAddr[u.i[1]].*(member1) = w.v[1]; \
                baseAddr[u.i[2]].*(member1) = w.v[2]; \
                baseAddr[u.i[3]].*(member1) = w.v[3]; \
                baseAddr[u.i[4]].*(member1) = w.v[4]; \
                baseAddr[u.i[5]].*(member1) = w.v[5]; \
                baseAddr[u.i[6]].*(member1) = w.v[6]; \
                baseAddr[u.i[7]].*(member1) = w.v[7]; \
            } \
            template<typename S1, typename S2> \
            static inline void scatter(const TYPE &v, const _M128I &indexes, S1 *baseAddr, S2 S1::* member1, T S2::* member2) { \
                union { __m128i p; unsigned short i[8]; } u; _mm_store_si128(&u.p, indexes); \
                union { TYPE p; T v[8]; } w; store(w.v, v); \
                baseAddr[u.i[0]].*(member1).*(member2) = w.v[0]; \
                baseAddr[u.i[1]].*(member1).*(member2) = w.v[1]; \
                baseAddr[u.i[2]].*(member1).*(member2) = w.v[2]; \
                baseAddr[u.i[3]].*(member1).*(member2) = w.v[3]; \
                baseAddr[u.i[4]].*(member1).*(member2) = w.v[4]; \
                baseAddr[u.i[5]].*(member1).*(member2) = w.v[5]; \
                baseAddr[u.i[6]].*(member1).*(member2) = w.v[6]; \
                baseAddr[u.i[7]].*(member1).*(member2) = w.v[7]; \
            }

        template<> struct VectorHelper<double> {
#define TYPE _M128D
#define SUFFIX pd
            LOAD(double)
            STORE(double)
            static inline void gather(TYPE &v, const _M128I &indexes, const double *baseAddr) {
                union { __m128i p; unsigned int i[4]; } u; _mm_store_si128(&u.p, indexes);
                v = _mm_setr_pd(baseAddr[u.i[0]], baseAddr[u.i[1]]);
            }
            template<typename S>
            static inline void gather(TYPE &v, const _M128I &indexes, const S *baseAddr, const double S::* member1) {
                union { __m128i p; unsigned int i[4]; } u; _mm_store_si128(&u.p, indexes);
                v = CAT(_mm_setr_, SUFFIX)(baseAddr[u.i[0]].*(member1), baseAddr[u.i[1]].*(member1));
            }
            template<typename S1, typename S2>
            static inline void gather(TYPE &v, const _M128I &indexes, const S1 *baseAddr, const S2 S1::* member1, const double S2::* member2) {
                union { __m128i p; unsigned int i[4]; } u; _mm_store_si128(&u.p, indexes);
                v = CAT(_mm_setr_, SUFFIX)(baseAddr[u.i[0]].*(member1).*(member2), baseAddr[u.i[1]].*(member1).*(member2));
            }
            static inline void scatter(const TYPE &v, const _M128I &indexes, double *baseAddr) {
                union { __m128i p; unsigned int i[4]; } u; _mm_store_si128(&u.p, indexes);
                _mm_storel_pd(&baseAddr[u.i[0]], v);
                _mm_storeh_pd(&baseAddr[u.i[1]], v);
            }
            template<typename S>
            static inline void scatter(const TYPE &v, const _M128I &indexes, S *baseAddr, double S::* member1) {
                union { __m128i p; unsigned int i[4]; } u; _mm_store_si128(&u.p, indexes);
                _mm_storel_pd(&(baseAddr[u.i[0]].*(member1)), v);
                _mm_storeh_pd(&(baseAddr[u.i[1]].*(member1)), v);
            }
            template<typename S1, typename S2>
            static inline void scatter(const TYPE &v, const _M128I &indexes, S1 *baseAddr, S2 S1::* member1, double S2::* member2) {
                union { __m128i p; unsigned int i[4]; } u; _mm_store_si128(&u.p, indexes);
                _mm_storel_pd(&(baseAddr[u.i[0]].*(member1).*(member2)), v);
                _mm_storeh_pd(&(baseAddr[u.i[1]].*(member1).*(member2)), v);
            }


            static inline TYPE notMaskedToZero(TYPE a, _M128 mask) { return CAT(_mm_and_, SUFFIX)(_mm_castps_pd(mask), a); }
            static inline TYPE set(const double a) { return CAT(_mm_set1_, SUFFIX)(a); }
            static inline TYPE set(const double a, const double b) { return CAT(_mm_set_, SUFFIX)(a, b); }
            static inline void setZero(TYPE &v) { v = CAT(_mm_setzero_, SUFFIX)(); }
            static inline TYPE zero() { return CAT(_mm_setzero_, SUFFIX)(); }

            static inline void multiplyAndAdd(TYPE &v1, TYPE v2, TYPE v3) { v1 = add(mul(v1, v2), v3); }
            static inline TYPE mul(TYPE a, TYPE b, _M128 _mask) {
                _M128D mask = _mm_castps_pd(_mask);
                return _mm_or_pd(
                    _mm_and_pd(mask, _mm_mul_pd(a, b)),
                    _mm_andnot_pd(mask, a)
                    );
            }
            static inline TYPE div(TYPE a, TYPE b, _M128 _mask) {
                _M128D mask = _mm_castps_pd(_mask);
                return _mm_or_pd(
                    _mm_and_pd(mask, _mm_div_pd(a, b)),
                    _mm_andnot_pd(mask, a)
                    );
            }

            OP(add) OP(sub) OP(mul) OP(div)
            OPcmp(eq) OPcmp(neq)
            OPcmp(lt) OPcmp(nlt)
            OPcmp(le) OPcmp(nle)

            OP1(sqrt)
            static TYPE log(TYPE x) {
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
                y = div(y, y2);

                y = mul(y, z);
                y = sub(y, mul(e, set(2.121944400546905827679e-4)));
                y = sub(y, mul(z, set(0.5)));

                x = add(x, y);
                x = add(x, mul(e, set(0.693359375)));
                x = _mm_or_pd(x, invalid_mask); // negative arg will be NAN
                x = _mm_or_pd(_mm_andnot_pd(infinity_mask, x), _mm_and_pd(infinity_mask, set(-std::numeric_limits<double>::infinity())));
                return x;
            }
            static inline TYPE abs(const TYPE a) {
                static const TYPE mask = _mm_castsi128_pd(_mm_set_epi32(0x7fffffff, 0xffffffff, 0x7fffffff, 0xffffffff));
                return CAT(_mm_and_, SUFFIX)(a, mask);
            }

            MINMAX
#undef TYPE
#undef SUFFIX
        };

        template<> struct VectorHelper<float> {
#define TYPE _M128
#define SUFFIX ps
            LOAD(float)
            STORE(float)
            GATHER_SCATTER(float)

            static inline TYPE notMaskedToZero(TYPE a, _M128 mask) { return CAT(_mm_and_, SUFFIX)(mask, a); }
            static inline TYPE set(const float a) { return CAT(_mm_set1_, SUFFIX)(a); }
            static inline TYPE set(const float a, const float b, const float c, const float d) { return CAT(_mm_set_, SUFFIX)(a, b, c, d); }
            static inline void setZero(TYPE &v) { v = CAT(_mm_setzero_, SUFFIX)(); }
            static inline TYPE zero() { return CAT(_mm_setzero_, SUFFIX)(); }

            static bool pack(TYPE &v1, _M128I &_m1, TYPE &v2, _M128I &_m2) {
                {
                    TYPE &m1 = reinterpret_cast<TYPE &>(_m1);
                    TYPE &m2 = reinterpret_cast<TYPE &>(_m2);
                    const int m1Mask = _mm_movemask_ps(m1);
                    const int m2Mask = _mm_movemask_ps(m2);
                    if (0 == (m1Mask & 8)) {
                        if (0 == (m1Mask & 4)) {
                            if (0 == (m1Mask & 2)) {
                                if (0 == m1Mask) {
                                    m1 = m2;
                                    v1 = v2;
                                    setZero(m2);
                                    return _mm_movemask_ps(m1) == 15;
                                }
                                if (0 == (m2Mask & 8)) {
#ifdef __SSSE3__
                                    v1 = _mm_alignr_epi8(v1, v2, sizeof(float));
#else
                                    v1 = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(3, 3, 1, 2));
                                    v1 = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(0, 2, 2, 3));
#endif
                                }
                            }
                        }
                    }
                }



                // there are 256 different m1.m2 combinations
                TYPE &m1 = reinterpret_cast<TYPE &>(_m1);
                TYPE &m2 = reinterpret_cast<TYPE &>(_m2);
                const int m1Mask = _mm_movemask_ps(m1);
                switch (m1Mask) {
                case 15: // v1 is full, nothing to do
                    return true;
                // 240 left
                case 0:  // v1 is empty, take v2
                    m1 = m2;
                    v1 = v2;
                    setZero(m2);
                    return _mm_movemask_ps(m1) == 15;
                // 224 left
                default:
                    {
                        TYPE tmp;
                        const int m2Mask = _mm_movemask_ps(m2);
                        switch (m2Mask) {
                        case 15: // v2 is full, just swap
                            tmp = v1;
                            v1 = v2;
                            v2 = tmp;
                            tmp = m1;
                            m1 = m2;
                            m2 = tmp;
                            return true;
                // 210 left
                        case 0: // v2 is empty, nothing to be gained from packing
                            return false;
                // 196 left
                        }
                        // m1 and m2 are neither full nor empty
                        tmp = _mm_or_ps(m1, m2);
                        const int m3Mask = _mm_movemask_ps(tmp);
                        // m3Mask tells use where both vectors have no entries
                        const int m4Mask = _mm_movemask_ps(_mm_and_ps(m1, m2));
                        // m3Mask tells use where both vectors have entries
                        if (m4Mask == 0 || m3Mask == 15) {
                            // m4Mask == 0: No overlap, simply move all we can from v2 into v1.
                            //              Empties v2.
                            // m3Mask == 15: Simply merge the parts from v2 into v1 where v1 is
                            //               empty.
                            const TYPE m2Move = _mm_andnot_ps(m1, m2); // the part to be moved into v1
                            v1 = _mm_add_ps(
                                    _mm_and_ps(v1, m1),
                                    _mm_and_ps(v2, m2Move)
                                    );
                            m1 = tmp;
                            m2 = _mm_andnot_ps(m2Move, m2);
                            return m3Mask == 15;
                // 
                        }
                        if ((m4Mask & 3) == 3) {
                            // the high values are available
                            tmp = _mm_unpackhi_ps(v1, v2);
                            v2  = _mm_unpacklo_ps(v1, v2);
                            v1  = tmp;
                            tmp = _mm_unpackhi_ps(m1, m2);
                            m2  = _mm_unpacklo_ps(m1, m2);
                            m1  = tmp;
                            return true;
                        }
                        if ((m4Mask & 12) == 12) {
                            // the low values are available
                            tmp = _mm_unpacklo_ps(v1, v2);
                            v2  = _mm_unpackhi_ps(v1, v2);
                            v1  = tmp;
                            tmp = _mm_unpacklo_ps(m1, m2);
                            m2  = _mm_unpackhi_ps(m1, m2);
                            m1  = tmp;
                            return true;
                        }
                        if ((m4Mask & 5) == 5) {
                            tmp = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(0, 2, 0, 2));
                            v2  = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(1, 3, 1, 3));
                            v1  = tmp;
                            tmp = _mm_shuffle_ps(m1, m2, _MM_SHUFFLE(0, 2, 0, 2));
                            m2  = _mm_shuffle_ps(m1, m2, _MM_SHUFFLE(1, 3, 1, 3));
                            m1  = tmp;
                            return true;
                        }
                        if ((m4Mask & 6) == 6) {
                            tmp = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(1, 2, 1, 2));
                            v2  = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(0, 3, 0, 3));
                            v1  = tmp;
                            tmp = _mm_shuffle_ps(m1, m2, _MM_SHUFFLE(1, 2, 1, 2));
                            m2  = _mm_shuffle_ps(m1, m2, _MM_SHUFFLE(0, 3, 0, 3));
                            m1  = tmp;
                            return true;
                        }
                        if ((m4Mask & 9) == 9) {
                            tmp = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(0, 3, 0, 3));
                            v2  = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(1, 2, 1, 2));
                            v1  = tmp;
                            tmp = _mm_shuffle_ps(m1, m2, _MM_SHUFFLE(0, 3, 0, 3));
                            m2  = _mm_shuffle_ps(m1, m2, _MM_SHUFFLE(1, 2, 1, 2));
                            m1  = tmp;
                            return true;
                        }
                        if ((m4Mask & 10) == 10) {
                            tmp = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(1, 3, 1, 3));
                            v2  = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(0, 2, 0, 2));
                            v1  = tmp;
                            tmp = _mm_shuffle_ps(m1, m2, _MM_SHUFFLE(1, 3, 1, 3));
                            m2  = _mm_shuffle_ps(m1, m2, _MM_SHUFFLE(0, 2, 0, 2));
                            m1  = tmp;
                            return true;
                        }
                        float *const vv1 = reinterpret_cast<float *>(&v1);
                        float *const vv2 = reinterpret_cast<float *>(&v2);
                        unsigned int *const mm1 = reinterpret_cast<unsigned int *>(&_m1);
                        unsigned int *const mm2 = reinterpret_cast<unsigned int *>(&_m2);
                        int j = 0;
                        for (int i = 0; i < 4; ++i) {
                            if (!(m1Mask & (1 << i))) { // v1 entry not set, take the first from v2
                                while (j < 4 && !(m2Mask & (1 << j))) {
                                    ++j;
                                }
                                if (j < 4) {
                                    vv1[i] = vv2[j];
                                    mm1[i] = 0xffffffff;
                                    mm2[j] = 0;
                                    ++j;
                                }
                            }
                        }
                        return _mm_movemask_ps(m1) == 15;
//X                         // m4Mask has exactly one bit set
//X                         switch (m4Mask) {
//X                         case 1:
//X                             // x___    xx__    xx__    xx__    xxx_    x_x_    x_x_    x_xx    x__x  + mirrored horizontally
//X                             // x___    x___    x_x_    x__x    x___    x___    x__x    x___    x___
//X                             break;
//X                         case 2:
//X                             break;
//X                         case 4:
//X                             break;
//X                         case 8:
//X                             break;
//X                         }
                    }
                }
            }

            static inline void multiplyAndAdd(TYPE &v1, TYPE v2, TYPE v3) { v1 = add(mul(v1, v2), v3); }
            static inline TYPE mul(TYPE a, TYPE b, _M128 mask) {
                return _mm_or_ps(
                    _mm_and_ps(mask, _mm_mul_ps(a, b)),
                    _mm_andnot_ps(mask, a)
                    );
            }
            static inline TYPE div(TYPE a, TYPE b, _M128 mask) {
                return _mm_or_ps(
                    _mm_and_ps(mask, _mm_div_ps(a, b)),
                    _mm_andnot_ps(mask, a)
                    );
            }

            OP(add) OP(sub) OP(mul) OP(div)
            OPcmp(eq) OPcmp(neq)
            OPcmp(lt) OPcmp(nlt)
            OPcmp(le) OPcmp(nle)

            OP1(sqrt)
            static TYPE log(TYPE x) {
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
            static inline TYPE abs(const TYPE a) {
                static const TYPE mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
                return CAT(_mm_and_, SUFFIX)(a, mask);
            }

            MINMAX
#undef TYPE
#undef SUFFIX
        };

        template<> struct VectorHelper<int> {
#define TYPE _M128I
#define SUFFIX si128
            LOAD_CAST(int)
            STORE_CAST(int)

            OP_(or_) OP_(and_) OP_(xor_)
            static inline void setZero(TYPE &v) { v = CAT(_mm_setzero_, SUFFIX)(); }
            static inline TYPE notMaskedToZero(TYPE a, _M128 mask) { return CAT(_mm_and_, SUFFIX)(_mm_castps_si128(mask), a); }
#undef SUFFIX
#define SUFFIX epi32
            GATHER_SCATTER(int)

            static inline TYPE set(const int a) { return CAT(_mm_set1_, SUFFIX)(a); }
            static inline TYPE set(const int a, const int b, const int c, const int d) { return CAT(_mm_set_, SUFFIX)(a, b, c, d); }

            static inline void multiplyAndAdd(TYPE &v1, TYPE v2, TYPE v3) { v1 = add(mul(v1, v2), v3); }

            SHIFT4


#ifdef __SSSE3__
            OP1(abs)
#else
            static inline TYPE abs(TYPE a) {
              TYPE zero; setZero( zero );
              const TYPE one = set( 1 );
              TYPE negative = cmplt(a, zero);
              a = xor_( a, negative );
              return add( a, and_( one, negative ) );
            }
#endif

#ifdef __SSE4_1__
            static inline TYPE mul(TYPE a, TYPE b) { return _mm_mullo_epi32(a, b); }
            static inline TYPE mul(TYPE a, TYPE b, _M128 _mask) {
                _M128I mask = _mm_castps_si128(_mask);
                return _mm_or_si128(
                    _mm_and_si128(mask, _mm_mullo_epi32(a, b)),
                    _mm_andnot_si128(mask, a)
                    );
            }
            MINMAX
#else
            static inline TYPE min(TYPE a, TYPE b) {
                STORE_VECTOR(int, _a, a);
                STORE_VECTOR(int, _b, b);
                union {
                    int i[4];
                    TYPE v;
                } x = { {
                    std::min(_a[0], _b[0]),
                    std::min(_a[1], _b[1]),
                    std::min(_a[2], _b[2]),
                    std::min(_a[3], _b[3])
                } };
                return x.v;
            }
            static inline TYPE max(TYPE a, TYPE b) {
                STORE_VECTOR(int, _a, a);
                STORE_VECTOR(int, _b, b);
                union {
                    int i[4];
                    TYPE v;
                } x = { {
                    std::max(_a[0], _b[0]),
                    std::max(_a[1], _b[1]),
                    std::max(_a[2], _b[2]),
                    std::max(_a[3], _b[3])
                } };
                return x.v;
            }
            static inline TYPE mul(const TYPE a, const TYPE b, _M128 _mask) {
                const int mask = _mm_movemask_ps(_mask);
                STORE_VECTOR(int, _a, a);
                STORE_VECTOR(int, _b, b);
                union {
                    int i[4];
                    TYPE v;
                } x = { {
                    (mask & 1 ? _a[0] * _b[0] : _a[0]),
                    (mask & 2 ? _a[1] * _b[1] : _a[1]),
                    (mask & 4 ? _a[2] * _b[2] : _a[2]),
                    (mask & 8 ? _a[3] * _b[3] : _a[3])
                } };
                return x.v;
            }
            static inline TYPE mul(const TYPE a, const TYPE b) {
                STORE_VECTOR(int, _a, a);
                STORE_VECTOR(int, _b, b);
                union {
                    int i[4];
                    TYPE v;
                } x = { {
                    _a[0] * _b[0],
                    _a[1] * _b[1],
                    _a[2] * _b[2],
                    _a[3] * _b[3]
                } };
                return x.v;
//X                 TYPE hi = _mm_mulhi_epi16(a, b);
//X                 hi = _mm_slli_epi32(hi, 16);
//X                 TYPE lo = _mm_mullo_epi16(a, b);
//X                 return or_(hi, lo);
            }
#endif

            static inline TYPE div(const TYPE a, const TYPE b, _M128 _mask) {
                const int mask = _mm_movemask_ps(_mask);
                STORE_VECTOR(int, _a, a);
                STORE_VECTOR(int, _b, b);
                union {
                    int i[4];
                    TYPE v;
                } x = { {
                    (mask & 1 ? _a[0] / _b[0] : _a[0]),
                    (mask & 2 ? _a[1] / _b[1] : _a[1]),
                    (mask & 4 ? _a[2] / _b[2] : _a[2]),
                    (mask & 8 ? _a[3] / _b[3] : _a[3])
                } };
                return x.v;
            }
            static inline TYPE div(const TYPE a, const TYPE b) {
                STORE_VECTOR(int, _a, a);
                STORE_VECTOR(int, _b, b);
                union {
                    int i[4];
                    TYPE v;
                } x = { {
                    _a[0] / _b[0],
                    _a[1] / _b[1],
                    _a[2] / _b[2],
                    _a[3] / _b[3]
                } };
                return x.v;
            }

            OP(add) OP(sub)
            OPcmp(eq)
            OPcmp(lt)
            OPcmp(gt)
            static inline TYPE cmpneq(const TYPE &a, const TYPE &b) { _M128I x = cmpeq(a, b); return _mm_xor_si128(x, _0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF); }
            static inline TYPE cmpnlt(const TYPE &a, const TYPE &b) { _M128I x = cmplt(a, b); return _mm_xor_si128(x, _0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF); }
            static inline TYPE cmple (const TYPE &a, const TYPE &b) { _M128I x = cmpgt(a, b); return _mm_xor_si128(x, _0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF); }
            static inline TYPE cmpnle(const TYPE &a, const TYPE &b) { return cmpgt(a, b); }
#undef TYPE
#undef SUFFIX
        };

        template<> struct VectorHelper<unsigned int> {
#define TYPE _M128I
#define SUFFIX si128
            LOAD_CAST(unsigned int)
            STORE_CAST(unsigned int)
            OP_CAST_(or_) OP_CAST_(and_) OP_CAST_(xor_)
            static inline void setZero(TYPE &v) { v = CAT(_mm_setzero_, SUFFIX)(); }
            static inline TYPE notMaskedToZero(TYPE a, _M128 mask) { return CAT(_mm_and_, SUFFIX)(_mm_castps_si128(mask), a); }

#undef SUFFIX
#define SUFFIX epu32

#ifdef __SSE4_1__
            MINMAX
#else
            static inline TYPE min(TYPE a, TYPE b) {
                STORE_VECTOR(unsigned int, _a, a);
                STORE_VECTOR(unsigned int, _b, b);
                union {
                    unsigned int i[4];
                    TYPE v;
                } x = { {
                    std::min(_a[0], _b[0]),
                    std::min(_a[1], _b[1]),
                    std::min(_a[2], _b[2]),
                    std::min(_a[3], _b[3])
                } };
                return x.v;
            }
            static inline TYPE max(TYPE a, TYPE b) {
                STORE_VECTOR(unsigned int, _a, a);
                STORE_VECTOR(unsigned int, _b, b);
                union {
                    unsigned int i[4];
                    TYPE v;
                } x = { {
                    std::max(_a[0], _b[0]),
                    std::max(_a[1], _b[1]),
                    std::max(_a[2], _b[2]),
                    std::max(_a[3], _b[3])
                } };
                return x.v;
            }
#endif

            static inline TYPE mul(TYPE a, TYPE b, _M128 _mask) {
                _M128I mask = _mm_castps_si128(_mask);
                return _mm_or_si128(
                    _mm_and_si128(mask, mul(a, b)),
                    _mm_andnot_si128(mask, a)
                    );
            }
            static inline TYPE mul(const TYPE a, const TYPE b) {
                TYPE hi = _mm_mulhi_epu16(a, b);
                hi = _mm_slli_epi32(hi, 16);
                TYPE lo = _mm_mullo_epi16(a, b);
                return or_(hi, lo);
            }
            static inline TYPE div(const TYPE a, const TYPE b, _M128 _mask) {
                const int mask = _mm_movemask_ps(_mask);
                STORE_VECTOR(unsigned int, _a, a);
                STORE_VECTOR(unsigned int, _b, b);
                union {
                    unsigned int i[4];
                    TYPE v;
                } x = { {
                    (mask & 1 ? _a[0] / _b[0] : _a[0]),
                    (mask & 2 ? _a[1] / _b[1] : _a[1]),
                    (mask & 4 ? _a[2] / _b[2] : _a[2]),
                    (mask & 8 ? _a[3] / _b[3] : _a[3])
                } };
                return x.v;
            }
            static inline TYPE div(const TYPE a, const TYPE b) {
                STORE_VECTOR(unsigned int, _a, a);
                STORE_VECTOR(unsigned int, _b, b);
                union {
                    unsigned int i[4];
                    TYPE v;
                } x = { {
                    _a[0] / _b[0],
                    _a[1] / _b[1],
                    _a[2] / _b[2],
                    _a[3] / _b[3]
                } };
                return x.v;
            }

#undef SUFFIX
#define SUFFIX epi32
            GATHER_SCATTER(unsigned int)
            static inline TYPE set(const unsigned int a) { return CAT(_mm_set1_, SUFFIX)(a); }
            static inline TYPE set(const unsigned int a, const unsigned int b, const unsigned int c, const unsigned int d) { return CAT(_mm_set_, SUFFIX)(a, b, c, d); }

            SHIFT4
            OP(add) OP(sub)
            OPcmp(eq)
            OPcmp(lt)
            OPcmp(gt)
            static inline TYPE cmpneq(const TYPE &a, const TYPE &b) { return _mm_xor_si128(cmpeq(a, b), _0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF); }
            static inline TYPE cmpnlt(const TYPE &a, const TYPE &b) { return _mm_xor_si128(cmplt(a, b), _0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF); }
            static inline TYPE cmple (const TYPE &a, const TYPE &b) { return _mm_xor_si128(cmpgt(a, b), _0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF); }
            static inline TYPE cmpnle(const TYPE &a, const TYPE &b) { return cmpgt(a, b); }
#undef TYPE
#undef SUFFIX
        };

        template<> struct VectorHelper<signed short> {
#define TYPE _M128I
#define SUFFIX si128
            LOAD_CAST(short)
            STORE_CAST(short)

            OP_(or_) OP_(and_) OP_(xor_)
            static inline void setZero(TYPE &v) { v = CAT(_mm_setzero_, SUFFIX)(); }
            static inline TYPE notMaskedToZero(TYPE a, _M128 mask) { return CAT(_mm_and_, SUFFIX)(_mm_castps_si128(mask), a); }
#undef SUFFIX
#define SUFFIX epi16
            GATHER_SCATTER_16(signed short)
            SHIFT8

            static inline TYPE set(const short a) { return CAT(_mm_set1_, SUFFIX)(a); }
            static inline TYPE set(const short a, const short b, const short c, const short d,
                    const short e, const short f, const short g, const short h) {
                return CAT(_mm_set_, SUFFIX)(a, b, c, d, e, f, g, h);
            }

            static inline void multiplyAndAdd(TYPE &v1, TYPE v2, TYPE v3) {
                v1 = add(mul(v1, v2), v3); }

#ifdef __SSSE3__
            OP1(abs)
#else
            static inline TYPE abs(TYPE a) {
              TYPE zero; setZero( zero );
              const TYPE one = set( 1 );
              TYPE negative = cmplt(a, zero);
              a = xor_( a, negative );
              return add( a, and_( one, negative ) );
            }
#endif

            static inline TYPE mul(TYPE a, TYPE b, _M128 _mask) {
                _M128I mask = _mm_castps_si128(_mask);
                return _mm_or_si128(
                    _mm_and_si128(mask, mul(a, b)),
                    _mm_andnot_si128(mask, a)
                    );
            }
            OPx(mul, mullo)
            OP(min) OP(max)

            static inline TYPE div(const TYPE a, const TYPE b, _M128 _mask) {
                const int mask = _mm_movemask_epi8(_mm_castps_si128(_mask));
                STORE_VECTOR(short, _a, a);
                STORE_VECTOR(short, _b, b);
                union {
                    short i[8];
                    TYPE v;
                } x = { {
                    (mask & 0x0001 ? _a[0] / _b[0] : _a[0]),
                    (mask & 0x0004 ? _a[1] / _b[1] : _a[1]),
                    (mask & 0x0010 ? _a[2] / _b[2] : _a[2]),
                    (mask & 0x0040 ? _a[3] / _b[3] : _a[3]),
                    (mask & 0x0100 ? _a[4] / _b[4] : _a[4]),
                    (mask & 0x0400 ? _a[5] / _b[5] : _a[5]),
                    (mask & 0x1000 ? _a[6] / _b[6] : _a[6]),
                    (mask & 0x4000 ? _a[7] / _b[7] : _a[7])
                } };
                return x.v;
            }
            static inline TYPE div(const TYPE a, const TYPE b) {
                STORE_VECTOR(short, _a, a);
                STORE_VECTOR(short, _b, b);
                union {
                    short i[8];
                    TYPE v;
                } x = { {
                    _a[0] / _b[0],
                    _a[1] / _b[1],
                    _a[2] / _b[2],
                    _a[3] / _b[3],
                    _a[4] / _b[4],
                    _a[5] / _b[5],
                    _a[6] / _b[6],
                    _a[7] / _b[7]
                } };
                return x.v;
            }

            OP(add) OP(sub)
            OPcmp(eq)
            OPcmp(lt)
            OPcmp(gt)
            static inline TYPE cmpneq(const TYPE &a, const TYPE &b) { _M128I x = cmpeq(a, b); return _mm_xor_si128(x, _0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF); }
            static inline TYPE cmpnlt(const TYPE &a, const TYPE &b) { _M128I x = cmplt(a, b); return _mm_xor_si128(x, _0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF); }
            static inline TYPE cmple (const TYPE &a, const TYPE &b) { _M128I x = cmpgt(a, b); return _mm_xor_si128(x, _0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF); }
            static inline TYPE cmpnle(const TYPE &a, const TYPE &b) { return cmpgt(a, b); }
#undef TYPE
#undef SUFFIX
        };

        template<> struct VectorHelper<unsigned short> {
#define TYPE _M128I
#define SUFFIX si128
            LOAD_CAST(unsigned short)
            STORE_CAST(unsigned short)
            OP_CAST_(or_) OP_CAST_(and_) OP_CAST_(xor_)
            static inline void setZero(TYPE &v) { v = CAT(_mm_setzero_, SUFFIX)(); }
            static inline TYPE notMaskedToZero(TYPE a, _M128 mask) { return CAT(_mm_and_, SUFFIX)(_mm_castps_si128(mask), a); }

#undef SUFFIX
#define SUFFIX epu16
            static inline TYPE div(const TYPE a, const TYPE b, _M128 _mask) {
                const int mask = _mm_movemask_epi8(_mm_castps_si128(_mask));
                STORE_VECTOR(unsigned short, _a, a);
                STORE_VECTOR(unsigned short, _b, b);
                union {
                    unsigned short i[8];
                    TYPE v;
                } x = { {
                    (mask & 0x0001 ? _a[0] / _b[0] : _a[0]),
                    (mask & 0x0004 ? _a[1] / _b[1] : _a[1]),
                    (mask & 0x0010 ? _a[2] / _b[2] : _a[2]),
                    (mask & 0x0040 ? _a[3] / _b[3] : _a[3]),
                    (mask & 0x0100 ? _a[4] / _b[4] : _a[4]),
                    (mask & 0x0400 ? _a[5] / _b[5] : _a[5]),
                    (mask & 0x1000 ? _a[6] / _b[6] : _a[6]),
                    (mask & 0x4000 ? _a[7] / _b[7] : _a[7])
                } };
                return x.v;
            }
            static inline TYPE div(const TYPE a, const TYPE b) {
                STORE_VECTOR(unsigned short, _a, a);
                STORE_VECTOR(unsigned short, _b, b);
                union {
                    unsigned short i[8];
                    TYPE v;
                } x = { {
                    _a[0] / _b[0],
                    _a[1] / _b[1],
                    _a[2] / _b[2],
                    _a[3] / _b[3],
                    _a[4] / _b[4],
                    _a[5] / _b[5],
                    _a[6] / _b[6],
                    _a[7] / _b[7]
                } };
                return x.v;
            }

            static inline TYPE mul(TYPE a, TYPE b, _M128 _mask) {
                _M128I mask = _mm_castps_si128(_mask);
                return _mm_or_si128(
                    _mm_and_si128(mask, mul(a, b)),
                    _mm_andnot_si128(mask, a)
                    );
            }
#undef SUFFIX
#define SUFFIX epi16
            SHIFT8
            OPx(mul, mullo) // should work correctly for all values
            OP(min) OP(max) // XXX breaks for values with MSB set
            GATHER_SCATTER_16(unsigned short)
            static inline TYPE set(const unsigned short a) { return CAT(_mm_set1_, SUFFIX)(a); }
            static inline TYPE set(const unsigned short a, const unsigned short b, const unsigned short c,
                    const unsigned short d, const unsigned short e, const unsigned short f,
                    const unsigned short g, const unsigned short h) {
                return CAT(_mm_set_, SUFFIX)(a, b, c, d, e, f, g, h);
            }

            OP(add) OP(sub)
            OPcmp(eq)
            OPcmp(lt)
            OPcmp(gt)
            static inline TYPE cmpneq(const TYPE &a, const TYPE &b) { return _mm_xor_si128(cmpeq(a, b), _0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF); }
            static inline TYPE cmpnlt(const TYPE &a, const TYPE &b) { return _mm_xor_si128(cmplt(a, b), _0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF); }
            static inline TYPE cmple (const TYPE &a, const TYPE &b) { return _mm_xor_si128(cmpgt(a, b), _0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF); }
            static inline TYPE cmpnle(const TYPE &a, const TYPE &b) { return cmpgt(a, b); }
#undef TYPE
#undef SUFFIX
        };
#undef GATHER_SCATTER_16
#undef GATHER_SCATTER
#undef SHIFT4
#undef SHIFT8
#undef STORE
#undef LOAD
#undef OP1
#undef OP
#undef OP_
#undef OPx
#undef OPcmp
#undef CAT
#undef CAT_HELPER

namespace VectorSpecialInitializerZero { enum Enum { Zero }; }
namespace VectorSpecialInitializerRandom { enum Enum { Random }; }
namespace VectorSpecialInitializerIndexesFromZero { enum Enum { IndexesFromZero }; }

template<unsigned int Size1, unsigned int Size2> struct MaskHelper;
template<> struct MaskHelper<2, 2> {
    static inline bool cmpeq (_M128 k1, _M128 k2) { return _mm_movemask_pd(_mm_castps_pd(k1)) == _mm_movemask_pd(_mm_castps_pd(k2)); }
    static inline bool cmpneq(_M128 k1, _M128 k2) { return _mm_movemask_pd(_mm_castps_pd(k1)) != _mm_movemask_pd(_mm_castps_pd(k2)); }
};
template<> struct MaskHelper<2, 4> {
    static inline bool cmpeq (_M128 k1, _M128 k2) { return _mm_movemask_pd(_mm_castps_pd(k1)) == (_mm_movemask_ps(k2) & 3); }
    static inline bool cmpneq(_M128 k1, _M128 k2) { return _mm_movemask_pd(_mm_castps_pd(k1)) != (_mm_movemask_ps(k2) & 3); }
};
template<> struct MaskHelper<2, 8> {
    static inline bool cmpeq (_M128 k1, _M128 k2) { return _mm_movemask_pd(_mm_castps_pd(k1)) == (_mm_movemask_epi8(_mm_castps_si128(k2)) & 0xf); }
    static inline bool cmpneq(_M128 k1, _M128 k2) { return _mm_movemask_pd(_mm_castps_pd(k1)) != (_mm_movemask_epi8(_mm_castps_si128(k2)) & 0xf); }
};
template<> struct MaskHelper<4, 2> {
    static inline bool cmpeq (_M128 k1, _M128 k2) { return MaskHelper<2, 4>::cmpeq (k2, k1); }
    static inline bool cmpneq(_M128 k1, _M128 k2) { return MaskHelper<2, 4>::cmpneq(k2, k1); }
};
template<> struct MaskHelper<4, 4> {
    static inline bool cmpeq (_M128 k1, _M128 k2) { return _mm_movemask_ps(k1) == _mm_movemask_ps(k2); }
    static inline bool cmpneq(_M128 k1, _M128 k2) { return _mm_movemask_ps(k1) != _mm_movemask_ps(k2); }
};
template<> struct MaskHelper<4, 8> {
    static inline bool cmpeq (_M128 k1, _M128 k2) { return _mm_movemask_ps(k1) == _mm_movemask_epi8(_mm_castps_si128(k2)); }
    static inline bool cmpneq(_M128 k1, _M128 k2) { return _mm_movemask_ps(k1) != _mm_movemask_epi8(_mm_castps_si128(k2)); }
};
template<> struct MaskHelper<8, 2> {
    static inline bool cmpeq (_M128 k1, _M128 k2) { return MaskHelper<2, 8>::cmpeq (k2, k1); }
    static inline bool cmpneq(_M128 k1, _M128 k2) { return MaskHelper<2, 8>::cmpneq(k2, k1); }
};
template<> struct MaskHelper<8, 4> {
    static inline bool cmpeq (_M128 k1, _M128 k2) { return MaskHelper<4, 8>::cmpeq (k2, k1); }
    static inline bool cmpneq(_M128 k1, _M128 k2) { return MaskHelper<4, 8>::cmpneq(k2, k1); }
};
template<> struct MaskHelper<8, 8> {
    static inline bool cmpeq (_M128 k1, _M128 k2) { return _mm_movemask_epi8(_mm_castps_si128(k1)) == _mm_movemask_epi8(_mm_castps_si128(k2)); }
    static inline bool cmpneq(_M128 k1, _M128 k2) { return _mm_movemask_epi8(_mm_castps_si128(k1)) != _mm_movemask_epi8(_mm_castps_si128(k2)); }
};

template<unsigned int VectorSize> class Mask
{
    friend class Mask<2u>;
    friend class Mask<4u>;
    friend class Mask<8u>;
    public:
        inline Mask() {}
        inline Mask(const __m128  &x) : k(x) {}
        inline Mask(const __m128d &x) : k(_mm_castpd_ps(x)) {}
        inline Mask(const __m128i &x) : k(_mm_castsi128_ps(x)) {}

        // _mm_movemask_epi8 creates a 16 bit mask containing the most significant bit of every byte in k

        template<unsigned int OtherSize>
        inline bool operator==(const Mask<OtherSize> &rhs) const { return MaskHelper<VectorSize, OtherSize>::cmpeq (k, rhs.k); }
        template<unsigned int OtherSize>
        inline bool operator!=(const Mask<OtherSize> &rhs) const { return MaskHelper<VectorSize, OtherSize>::cmpneq(k, rhs.k); }

        inline Mask operator&&(const Mask &rhs) const { return _mm_and_ps(k, rhs.k); }
        inline Mask operator||(const Mask &rhs) const { return _mm_or_ps (k, rhs.k); }
        inline Mask operator!() const { return _mm_andnot_ps(k, _mm_castsi128_ps(_mm_set1_epi32(-1))); }

        inline bool isFull () const { return _mm_movemask_epi8(dataI()) == 0xffff; }
        inline bool isEmpty() const { return _mm_movemask_epi8(dataI()) == 0x0000; }

        inline operator bool() const { return isFull(); }

        inline _M128  data () const { return k; }
        inline _M128I dataI() const { return _mm_castps_si128(k); }
        inline _M128D dataD() const { return _mm_castps_pd(k); }

        template<unsigned int OtherSize>
        inline Mask<OtherSize> cast() const { return Mask<OtherSize>(k); }

        inline bool operator[](int index) const {
            if (VectorSize == 2) {
                return _mm_movemask_pd(dataD()) & (1 << index);
            } else if (VectorSize == 4) {
                return _mm_movemask_ps(k) & (1 << index);
            } else if (VectorSize == 8) {
                return _mm_movemask_epi8(dataI()) & (1 << 2 * index);
            }
            return false;
        }

        inline Mask<VectorSize * 2> combine(Mask other) const { return _mm_packs_epi16(dataI(), other.dataI()); }

    private:
        _M128 k;
};

template<unsigned int A, unsigned int B> struct MaskCastHelper
{
    static inline Mask<A> cast(const Mask<B> &m) { return Mask<A>(m.data()); }
};

template<typename T>
class WriteMaskedVector
{
    friend class Vector<T>;
    typedef SSE::Mask<16 / sizeof(T)> Mask;
    public:
        //prefix
        inline Vector<T> &operator++() {
            vec->data = VectorHelper<T>::add(vec->data,
                    VectorHelper<T>::notMaskedToZero(VectorHelper<T>::set(1), mask.data())
                    );
            return *vec;
        }
        inline Vector<T> &operator--() {
            vec->data = VectorHelper<T>::sub(vec->data,
                    VectorHelper<T>::notMaskedToZero(VectorHelper<T>::set(1), mask.data())
                    );
            return *vec;
        }
        //postfix
        inline Vector<T> operator++(int) {
            Vector<T> ret(*vec);
            vec->data = VectorHelper<T>::add(vec->data,
                    VectorHelper<T>::notMaskedToZero(VectorHelper<T>::set(1), mask.data())
                    );
            return ret;
        }
        inline Vector<T> operator--(int) {
            Vector<T> ret(*vec);
            vec->data = VectorHelper<T>::sub(vec->data,
                    VectorHelper<T>::notMaskedToZero(VectorHelper<T>::set(1), mask.data())
                    );
            return ret;
        }

        inline Vector<T> &operator+=(Vector<T> x) {
            vec->data = VectorHelper<T>::add(vec->data, VectorHelper<T>::notMaskedToZero(x, mask.data()));
            return *vec;
        }
        inline Vector<T> &operator-=(Vector<T> x) {
            vec->data = VectorHelper<T>::sub(vec->data, VectorHelper<T>::notMaskedToZero(x, mask.data()));
            return *vec;
        }
        inline Vector<T> &operator*=(Vector<T> x) {
            vec->data = VectorHelper<T>::mul(vec->data, x.data, mask.data());
            return *vec;
        }
        inline Vector<T> &operator/=(Vector<T> x) {
            vec->data = VectorHelper<T>::div(vec->data, x.data, mask.data());
            return *vec;
        }

        inline Vector<T> &operator=(Vector<T> x) {
            vec->assign(x, mask);
            return *vec;
        }
    private:
        WriteMaskedVector(Vector<T> *v, Mask k) : vec(v), mask(k) {}
        Vector<T> *vec;
        Mask mask;
};

template<typename T>
class Vector : public VectorBase<T, Vector<T> >
{
    friend struct VectorBase<T, Vector<T> >;
    friend class WriteMaskedVector<T>;
    protected:
        typedef typename VectorBase<T, Vector<T> >::IntrinType IntrinType;
        IntrinType data;
    public:
        typedef T Type;
        enum { Size = 16 / sizeof(T) };
        typedef SSE::Mask<Size> Mask;

        /**
         * uninitialized
         */
        inline Vector() {}
        /**
         * initialized to 0 in all 128 bits
         */
        inline explicit Vector(VectorSpecialInitializerZero::Enum) { makeZero(); }
        /**
         * initialized to 0, 1 (, 2, 3 (, 4, 5, 6, 7))
         */
        inline explicit Vector(VectorSpecialInitializerIndexesFromZero::Enum) : data(VectorHelper<T>::load(VectorBase<T, Vector<T> >::_IndexesFromZero())) {}
        /**
         * initialize with given _M128 vector
         */
        inline Vector(const IntrinType &x) : data(x) {}
        inline explicit Vector(const Mask &m) : data(m.dataI()) {}
        /**
         * initialize all 16 or 8 values with the given value
         */
        inline Vector(T a)
        {
            data = VectorHelper<T>::set(a);
        }
        /**
         * initialize consecutive four vector entries with the given values
         */
        template<typename Other>
        inline Vector(Other a, Other b, Other c, Other d)
        {
            data = VectorHelper<T>::set(a, b, c, d);
        }
        /**
         * Initialize the vector with the given data. \param x must point to 64 byte aligned 512
         * byte data.
         */
        template<typename Other> inline explicit Vector(const Other *x) : data(VectorHelper<T>::load(x)) {}

        template<typename Other> static inline Vector broadcast4(const Other *x) { return Vector<T>(x); }

        template<typename Other> inline void load(const Other *mem) { data = VectorHelper<T>::load(mem); }

        inline void makeZero() { VectorHelper<T>::setZero(data); }

        /**
         * Store the vector data to the given memory. The memory must be 64 byte aligned and of 512
         * bytes size.
         */
        template<typename Other> inline void store(Other *mem) const { VectorHelper<T>::store(mem, data); }

        /**
         * Non-temporal store variant. Writes to the memory without polluting the cache.
         */
        template<typename Other> inline void storeStreaming(Other *mem) const { VectorHelper<T>::storeStreaming(mem, data); }

        inline const Vector<T> &dcba() const { return *this; }
        inline const Vector<T> cdab() const { return reinterpret_cast<IntrinType>(_mm_shuffle_epi32(data, _MM_SHUFFLE(2, 3, 0, 1))); }
        inline const Vector<T> badc() const { return reinterpret_cast<IntrinType>(_mm_shuffle_epi32(data, _MM_SHUFFLE(1, 0, 3, 2))); }
        inline const Vector<T> aaaa() const { return reinterpret_cast<IntrinType>(_mm_shuffle_epi32(data, _MM_SHUFFLE(0, 0, 0, 0))); }
        inline const Vector<T> bbbb() const { return reinterpret_cast<IntrinType>(_mm_shuffle_epi32(data, _MM_SHUFFLE(1, 1, 1, 1))); }
        inline const Vector<T> cccc() const { return reinterpret_cast<IntrinType>(_mm_shuffle_epi32(data, _MM_SHUFFLE(2, 2, 2, 2))); }
        inline const Vector<T> dddd() const { return reinterpret_cast<IntrinType>(_mm_shuffle_epi32(data, _MM_SHUFFLE(3, 3, 3, 3))); }
        inline const Vector<T> dacb() const { return reinterpret_cast<IntrinType>(_mm_shuffle_epi32(data, _MM_SHUFFLE(3, 0, 2, 1))); }

        inline Vector(const T *array, const Vector<unsigned int> &indexes) { VectorHelper<T>::gather(data, indexes, array); }
        inline Vector(const T *array, const Vector<unsigned int> &indexes, const Mask &mask) {
            VectorHelper<T>::gather(data, indexes & Vector<unsigned int>(MaskCastHelper<4, Size>::cast(mask)), array);
        }

        inline void gather(const T *array, const Vector<unsigned int> &indexes) { VectorHelper<T>::gather(data, indexes, array); }
        inline void gather(const T *array, const Vector<unsigned int> &indexes, const Mask &mask) {
            VectorHelper<T>::gather(data, indexes & Vector<unsigned int>(MaskCastHelper<4, Size>::cast(mask)), array);
        }

        inline void scatter(T *array, const Vector<unsigned int> &indexes) const { VectorHelper<T>::scatter(data, indexes, array); }
        inline void scatter(T *array, const Vector<unsigned int> &indexes, const Mask &mask) const {
            VectorHelper<T>::scatter(data, indexes & Vector<unsigned int>(MaskCastHelper<4, Size>::cast(mask)), array);
        }

        inline Vector(const T *array, const Vector<unsigned short> &indexes) { VectorHelper<T>::gather(data, indexes, array); }
        inline Vector(const T *array, const Vector<unsigned short> &indexes, const Mask &mask) {
            VectorHelper<T>::gather(data, indexes & Vector<unsigned short>(MaskCastHelper<8, Size>::cast(mask)), array);
        }
        inline void gather(const T *array, const Vector<unsigned short> &indexes) { VectorHelper<T>::gather(data, indexes, array); }
        inline void gather(const T *array, const Vector<unsigned short> &indexes, const Mask &mask) {
            VectorHelper<T>::gather(data, indexes & Vector<unsigned short>(MaskCastHelper<8, Size>::cast(mask)), array);
        }
        inline void scatter(T *array, const Vector<unsigned short> &indexes) const { VectorHelper<T>::scatter(data, indexes, array); }
        inline void scatter(T *array, const Vector<unsigned short> &indexes, const Mask &mask) const {
            VectorHelper<T>::scatter(data, indexes & Vector<unsigned short>(MaskCastHelper<8, Size>::cast(mask)), array);
        }

        /**
         * \param array An array of objects where one member should be gathered
         * \param member1 A member pointer to the member of the class/struct that should be gathered
         * \param indexes The indexes in the array. The correct offsets are calculated
         *                automatically.
         * \param mask Optional mask to select only parts of the vector that should be gathered
         */
        template<typename S1> inline Vector(const S1 *array, const T S1::* member1, const Vector<unsigned int> &indexes) {
            VectorHelper<T>::gather(data, indexes, array, member1);
        }
        template<typename S1> inline Vector(const S1 *array, const T S1::* member1,
                const Vector<unsigned int> &indexes, const Mask &mask) {
            VectorHelper<T>::gather(data, (indexes & Vector<unsigned int>(MaskCastHelper<4, Size>::cast(mask))), array, member1);
        }
        template<typename S1, typename S2> inline Vector(const S1 *array, const S2 S1::* member1,
                const T S2::* member2, const Vector<unsigned int> &indexes) {
            VectorHelper<T>::gather(data, indexes, array, member1, member2);
        }
        template<typename S1, typename S2> inline Vector(const S1 *array, const S2 S1::* member1,
                const T S2::* member2, const Vector<unsigned int> &indexes, const Mask &mask) {
            VectorHelper<T>::gather(data, (indexes & Vector<unsigned int>(MaskCastHelper<4, Size>::cast(mask))), array, member1, member2);
        }

        template<typename S1> inline void gather(const S1 *array, const T S1::* member1,
                const Vector<unsigned int> &indexes) {
            VectorHelper<T>::gather(data, indexes, array, member1);
        }
        template<typename S1> inline void gather(const S1 *array, const T S1::* member1,
                const Vector<unsigned int> &indexes, const Mask &mask) {
            VectorHelper<T>::gather(data, (indexes & Vector<unsigned int>(MaskCastHelper<4, Size>::cast(mask))), array, member1);
        }
        template<typename S1, typename S2> inline void gather(const S1 *array, const S2 S1::* member1,
                const T S2::* member2, const Vector<unsigned int> &indexes) {
            VectorHelper<T>::gather(data, indexes, array, member1, member2);
        }
        template<typename S1, typename S2> inline void gather(const S1 *array, const S2 S1::* member1,
                const T S2::* member2, const Vector<unsigned int> &indexes, const Mask &mask) {
            VectorHelper<T>::gather(data, (indexes & Vector<unsigned int>(MaskCastHelper<4, Size>::cast(mask))), array, member1, member2);
        }

        template<typename S1> inline void scatter(S1 *array, T S1::* member1,
                const Vector<unsigned int> &indexes) const {
            VectorHelper<T>::scatter(data, indexes, array, member1);
        }
        template<typename S1> inline void scatter(S1 *array, T S1::* member1,
                const Vector<unsigned int> &indexes, const Mask &mask) const {
            VectorHelper<T>::scatter(data, (indexes & Vector<unsigned int>(MaskCastHelper<4, Size>::cast(mask))), array, member1);
        }
        template<typename S1, typename S2> inline void scatter(S1 *array, S2 S1::* member1,
                T S2::* member2, const Vector<unsigned int> &indexes) const {
            VectorHelper<T>::scatter(data, indexes, array, member1, member2);
        }
        template<typename S1, typename S2> inline void scatter(S1 *array, S2 S1::* member1,
                T S2::* member2, const Vector<unsigned int> &indexes, const Mask &mask) const {
            VectorHelper<T>::scatter(data, (indexes & Vector<unsigned int>(MaskCastHelper<4, Size>::cast(mask))), array, member1, member2);
        }

        template<typename S1> inline Vector(const S1 *array, const T S1::* member1,
                const Vector<unsigned short> &indexes) {
            VectorHelper<T>::gather(data, indexes, array, member1);
        }
        template<typename S1> inline Vector(const S1 *array, const T S1::* member1,
                const Vector<unsigned short> &indexes, const Mask &mask) {
            VectorHelper<T>::gather(data, (indexes & Vector<unsigned short>(MaskCastHelper<8, Size>::cast(mask))), array, member1);
        }
        template<typename S1, typename S2> inline Vector(const S1 *array, const S2 S1::* member1,
                const T S2::* member2, const Vector<unsigned short> &indexes) {
            VectorHelper<T>::gather(data, indexes, array, member1, member2);
        }
        template<typename S1, typename S2> inline Vector(const S1 *array, const S2 S1::* member1,
                const T S2::* member2, const Vector<unsigned short> &indexes, const Mask &mask) {
            VectorHelper<T>::gather(data, (indexes & Vector<unsigned short>(MaskCastHelper<8, Size>::cast(mask))), array, member1, member2);
        }

        template<typename S1> inline void gather(const S1 *array, const T S1::* member1,
                const Vector<unsigned short> &indexes) {
            VectorHelper<T>::gather(data, indexes, array, member1);
        }
        template<typename S1> inline void gather(const S1 *array, const T S1::* member1,
                const Vector<unsigned short> &indexes, const Mask &mask) {
            VectorHelper<T>::gather(data, (indexes & Vector<unsigned short>(MaskCastHelper<8, Size>::cast(mask))), array, member1);
        }
        template<typename S1, typename S2> inline void gather(const S1 *array, const S2 S1::* member1,
                const T S2::* member2, const Vector<unsigned short> &indexes) {
            VectorHelper<T>::gather(data, indexes, array, member1, member2);
        }
        template<typename S1, typename S2> inline void gather(const S1 *array, const S2 S1::* member1,
                const T S2::* member2, const Vector<unsigned short> &indexes, const Mask &mask) {
            VectorHelper<T>::gather(data, (indexes & Vector<unsigned short>(MaskCastHelper<8, Size>::cast(mask))), array, member1, member2);
        }

        template<typename S1> inline void scatter(S1 *array, T S1::* member1,
                const Vector<unsigned short> &indexes) const {
            VectorHelper<T>::scatter(data, indexes, array, member1);
        }
        template<typename S1> inline void scatter(S1 *array, T S1::* member1,
                const Vector<unsigned short> &indexes, const Mask &mask) const {
            VectorHelper<T>::scatter(data, (indexes & Vector<unsigned short>(MaskCastHelper<8, Size>::cast(mask))), array, member1);
        }
        template<typename S1, typename S2> inline void scatter(S1 *array, S2 S1::* member1,
                T S2::* member2, const Vector<unsigned short> &indexes) const {
            VectorHelper<T>::scatter(data, indexes, array, member1, member2);
        }
        template<typename S1, typename S2> inline void scatter(S1 *array, S2 S1::* member1,
                T S2::* member2, const Vector<unsigned short> &indexes, const Mask &mask) const {
            VectorHelper<T>::scatter(data, (indexes & Vector<unsigned short>(MaskCastHelper<8, Size>::cast(mask))), array, member1, member2);
        }

        //prefix
        inline Vector &operator++() { data = VectorHelper<T>::add(data, Vector<T>(1)); return *this; }
        //postfix
        inline Vector operator++(int) { const Vector<T> r = *this; data = VectorHelper<T>::add(data, Vector<T>(1)); return r; }

        inline T operator[](int index) const {
            union { IntrinType p; T v[16 / sizeof(T)]; } u;
            VectorHelper<T>::store(u.v, data);
            return u.v[index];
        }

#define OP1(fun) \
        inline Vector fun() const { return Vector<T>(VectorHelper<T>::fun(data)); } \
        inline Vector &fun##_eq() { data = VectorHelper<T>::fun(data); return *this; }
        OP1(sqrt)
        OP1(abs)
#undef OP1

#define OP(symbol, fun) \
        inline Vector &operator symbol##=(const Vector<T> &x) { data = VectorHelper<T>::fun(data, x.data); return *this; } \
        inline Vector operator symbol(const Vector<T> &x) const { return Vector<T>(VectorHelper<T>::fun(data, x.data)); }

        OP(+, add)
        OP(-, sub)
        OP(*, mul)
        OP(/, div)
        OP(|, or_)
        OP(&, and_)
        OP(^, xor_)
#undef OP
#define OPcmp(symbol, fun) \
        inline Mask operator symbol(const Vector<T> &x) const { return VectorHelper<T>::fun(data, x.data); }

        OPcmp(==, cmpeq)
        OPcmp(!=, cmpneq)
        OPcmp(>=, cmpnlt)
        OPcmp(>, cmpnle)
        OPcmp(<, cmplt)
        OPcmp(<=, cmple)
#undef OPcmp

        inline void multiplyAndAdd(const Vector<T> &factor, const Vector<T> &summand) {
            VectorHelper<T>::multiplyAndAdd(data, factor, summand);
        }

        inline void assign( const Vector<T> &v, const Mask &mask ) {
            data = mm128_reinterpret_cast<IntrinType>(
                    _mm_or_ps(
                        _mm_andnot_ps(mask.data(), mm128_reinterpret_cast<_M128>(data)),
                        _mm_and_ps(mask.data(), mm128_reinterpret_cast<_M128>(v.data))
                        )
                    );
        }

        template<typename T2> inline Vector<T2> staticCast() const { return StaticCastHelper<T, T2>::cast(data); }
        template<typename T2> inline Vector<T2> reinterpretCast() const { return ReinterpretCastHelper<T, T2>::cast(data); }

        inline WriteMaskedVector<T> operator()(Mask k) { return WriteMaskedVector<T>(this, k); }

        /**
         * \return \p true  This vector was completely filled. m2 might be 0 or != 0. You still have
         *                  to test this.
         *         \p false This vector was not completely filled. m2 is all 0.
         */
        //inline bool pack(Mask &m1, Vector<T> &v2, Mask &m2) {
            //return VectorHelper<T>::pack(data, m1.data, v2.data, m2.data);
        //}
};

template<typename T> class SwizzledVector : public Vector<T> {};

template<typename T> inline Vector<T> operator+(const T &x, const Vector<T> &v) { return v.operator+(x); }
template<typename T> inline Vector<T> operator*(const T &x, const Vector<T> &v) { return v.operator*(x); }
template<typename T> inline Vector<T> operator-(const T &x, const Vector<T> &v) { return Vector<T>(x) - v; }
template<typename T> inline Vector<T> operator/(const T &x, const Vector<T> &v) { return Vector<T>(x) / v; }
template<typename T> inline typename Vector<T>::Mask  operator< (const T &x, const Vector<T> &v) { return Vector<T>(x) <  v; }
template<typename T> inline typename Vector<T>::Mask  operator<=(const T &x, const Vector<T> &v) { return Vector<T>(x) <= v; }
template<typename T> inline typename Vector<T>::Mask  operator> (const T &x, const Vector<T> &v) { return Vector<T>(x) >  v; }
template<typename T> inline typename Vector<T>::Mask  operator>=(const T &x, const Vector<T> &v) { return Vector<T>(x) >= v; }
template<typename T> inline typename Vector<T>::Mask  operator==(const T &x, const Vector<T> &v) { return Vector<T>(x) == v; }
template<typename T> inline typename Vector<T>::Mask  operator!=(const T &x, const Vector<T> &v) { return Vector<T>(x) != v; }

#define OP_IMPL(T, symbol, fun) \
  template<> inline Vector<T> &VectorBase<T, Vector<T> >::operator symbol##=(const Vector<T> &x) { (static_cast<Vector<T> *>(this)->data) = VectorHelper<T>::fun((static_cast<Vector<T> *>(this)->data), x.data); return *static_cast<Vector<T> *>(this); } \
  template<> inline Vector<T>  VectorBase<T, Vector<T> >::operator symbol(const Vector<T> &x) const { return Vector<T>(VectorHelper<T>::fun((static_cast<const Vector<T> *>(this)->data), x.data)); }
  OP_IMPL(int, &, and_)
  OP_IMPL(int, |, or_)
  OP_IMPL(int, ^, xor_)
  OP_IMPL(int, <<, sll)
  OP_IMPL(int, >>, srl)
  OP_IMPL(unsigned int, &, and_)
  OP_IMPL(unsigned int, |, or_)
  OP_IMPL(unsigned int, ^, xor_)
  OP_IMPL(unsigned int, <<, sll)
  OP_IMPL(unsigned int, >>, srl)
  OP_IMPL(short, &, and_)
  OP_IMPL(short, |, or_)
  OP_IMPL(short, ^, xor_)
  OP_IMPL(short, <<, sll)
  OP_IMPL(short, >>, srl)
  OP_IMPL(unsigned short, &, and_)
  OP_IMPL(unsigned short, |, or_)
  OP_IMPL(unsigned short, ^, xor_)
  OP_IMPL(unsigned short, <<, sll)
  OP_IMPL(unsigned short, >>, srl)
#undef OP_IMPL
#undef OP_IMPL2

  template<> inline Vector<unsigned short> VectorBase<unsigned int, Vector<unsigned int> >::operator,(Vector<unsigned int> x) const {
    return _mm_packs_epi32(static_cast<const Vector<unsigned int> *>(this)->data, x.data);
  }
  template<> inline Vector<short> VectorBase<int, Vector<int> >::operator,(Vector<int> x) const {
    return _mm_packs_epi32(static_cast<const Vector<int> *>(this)->data, x.data);
  }

  template<typename T> static inline Vector<T> min (const Vector<T> &x, const Vector<T> &y) { return VectorHelper<T>::min(x, y); }
  template<typename T> static inline Vector<T> max (const Vector<T> &x, const Vector<T> &y) { return VectorHelper<T>::max(x, y); }
  template<typename T> static inline Vector<T> min (const Vector<T> &x, const T &y) { return min(x, Vector<T>(y)); }
  template<typename T> static inline Vector<T> max (const Vector<T> &x, const T &y) { return max(x, Vector<T>(y)); }
  template<typename T> static inline Vector<T> min (const T &x, const Vector<T> &y) { return min(Vector<T>(x), y); }
  template<typename T> static inline Vector<T> max (const T &x, const Vector<T> &y) { return max(Vector<T>(x), y); }
  template<typename T> static inline Vector<T> sqrt(const Vector<T> &x) { return VectorHelper<T>::sqrt(x); }
  template<typename T> static inline Vector<T> abs (const Vector<T> &x) { return VectorHelper<T>::abs(x); }
  template<typename T> static inline Vector<T> sin (const Vector<T> &x) { return VectorHelper<T>::sin(x); }
  template<typename T> static inline Vector<T> cos (const Vector<T> &x) { return VectorHelper<T>::cos(x); }
  template<typename T> static inline Vector<T> log (const Vector<T> &x) { return VectorHelper<T>::log(x); }
  template<typename T> static inline Vector<T> log10(const Vector<T> &x) { return VectorHelper<T>::log10(x); }
#undef ALIGN
#undef STORE_VECTOR
} // namespace SSE

#endif // SSE_VECTOR_H
