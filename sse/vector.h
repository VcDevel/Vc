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

#ifndef SSE_VECTOR_H
#define SSE_VECTOR_H

#include "intrinsics.h"

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

namespace SSE
{
    enum { VectorAlignment = 16 };
    template<typename T> class Vector;

    ALIGN(16) static const int _OneMaskData[4]  = { 0x00000001, 0x00000001, 0x00000001, 0x00000001 };
    ALIGN(16) static const int _FullMaskData[4] = { 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff };
#define _FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF *reinterpret_cast<const _M128I *const>(_FullMaskData)

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
            enum Upconvert {
                UpconvertNone     = 0x00   /* no conversion      */
            };
        };
        template<typename Parent>
        struct VectorBase<double, Parent>
        {
            typedef _M128D IntrinType;
            operator _M128D() { return PARENT_DATA; }
            operator const _M128D() const { return PARENT_DATA_CONST; }
            enum Upconvert {
                UpconvertNone     = 0x00   /* no conversion      */
            };
        };
        template<typename Parent>
        struct VectorBase<int, Parent>
        {
            typedef _M128I IntrinType;
            operator _M128I() { return PARENT_DATA; }
            operator const _M128I() const { return PARENT_DATA_CONST; }

            enum Upconvert {
                UpconvertNone  = 0x00  /* no conversion      */
            };
#define OP_DECL(symbol, fun) \
            inline Vector<int> &operator symbol##=(const Vector<int> &x); \
            inline Vector<int> &operator symbol##=(const int &x); \
            inline Vector<int> operator symbol(const Vector<int> &x) const; \
            inline Vector<int> operator symbol(const int &x) const;

            OP_DECL(|, or_)
            OP_DECL(&, and_)
            OP_DECL(^, xor_)
#undef OP_DECL
        };
        template<typename Parent>
        struct VectorBase<unsigned int, Parent>
        {
            typedef _M128I IntrinType;
            operator _M128I() { return PARENT_DATA; }
            operator const _M128I() const { return PARENT_DATA_CONST; }
            enum Upconvert {
                UpconvertNone   = 0x00  /* no conversion      */
            };
        };
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

        template<typename T> struct VectorHelper {};

#define CAT_HELPER2(a, b) a##b
#define CAT_HELPER(a, b) CAT_HELPER2(a, b)
#define OP1(op) \
        static inline TYPE op(const TYPE &a) { return CAT_HELPER(_mm_##op##_, SUFFIX)(a); }
#define OP(op) \
        static inline TYPE op(const TYPE &a, const TYPE &b) { return CAT_HELPER(_mm_##op##_ , SUFFIX)(a, b); }
#define OP_(op) \
        static inline TYPE op(const TYPE &a, const TYPE &b) { return CAT_HELPER(_mm_##op    , SUFFIX)(a, b); }
#define OPx(op, op2) \
        static inline TYPE op(const TYPE &a, const TYPE &b) { return CAT_HELPER(_mm_##op2##_, SUFFIX)(a, b); }
#define OPcmp(op) \
        static inline TYPE cmp##op(const TYPE &a, const TYPE &b) { return CAT_HELPER(_mm_cmp##op##_, SUFFIX)(a, b); }
#define OP_CAST_(op) \
        static inline TYPE op(const TYPE &a, const TYPE &b) { return CAT_HELPER(_mm_castps_, SUFFIX)( \
            _mm_##op##ps(CAT_HELPER(CAT_HELPER(_mm_cast, SUFFIX), _ps)(a), \
              CAT_HELPER(CAT_HELPER(_mm_cast, SUFFIX), _ps)(b))); \
        }
#define MINMAX \
        static inline TYPE min(TYPE &a, TYPE &b) { return CAT_HELPER(_mm_min_, SUFFIX)(a, b); } \
        static inline TYPE max(TYPE &a, TYPE &b) { return CAT_HELPER(_mm_max_, SUFFIX)(a, b); }
#define LOAD(T) \
        static inline TYPE load1(const T *x) { return CAT_HELPER(_mm_load1_, SUFFIX)(x); } \
        static inline TYPE load (const T *x) { return CAT_HELPER(_mm_load_ , SUFFIX)(x); }
#define LOAD_CAST(T) \
        static inline TYPE load1(const T *x) { return CAT_HELPER(_mm_castps_, SUFFIX)(_mm_load1_ps(reinterpret_cast<const float *>(x))); } \
        static inline TYPE load (const T *x) { return CAT_HELPER(_mm_castps_, SUFFIX)(_mm_load_ps (reinterpret_cast<const float *>(x))); }
#define STORE(T) \
            static inline void store1(T *mem, TYPE x) { return CAT_HELPER(_mm_store1_, SUFFIX)(mem, x); } \
            static inline void store (T *mem, TYPE x) { return CAT_HELPER(_mm_store_ , SUFFIX)(mem, x); } \
            static inline void storeStreaming(T *mem, TYPE x) { return CAT_HELPER(_mm_stream_ , SUFFIX)(mem, x); }
#define STORE_CAST(T) \
            static inline void store1(T *mem, TYPE x) { return _mm_store1_ps(reinterpret_cast<float *>(mem), CAT_HELPER(CAT_HELPER(_mm_cast, SUFFIX), _ps)(x)); } \
            static inline void store (T *mem, TYPE x) { return _mm_store_ps (reinterpret_cast<float *>(mem), CAT_HELPER(CAT_HELPER(_mm_cast, SUFFIX), _ps)(x)); } \
            static inline void storeStreaming(T *mem, TYPE x) { return _mm_stream_ps(reinterpret_cast<float *>(mem), CAT_HELPER(CAT_HELPER(_mm_cast, SUFFIX), _ps)(x)); }
#define GATHER(T) \
            static inline void gather(TYPE &v, const _M128I &indexes, const T *baseAddr) { \
                const int *const i = reinterpret_cast<const int *>(&indexes); \
                v = CAT_HELPER(_mm_setr_, SUFFIX)(baseAddr[i[0]], baseAddr[i[1]], baseAddr[i[2]], baseAddr[i[3]]); \
            } \
            template<typename S> \
            static inline void gather(TYPE &v, const _M128I &indexes, const S *baseAddr, const T S::* member1) { \
                const int *const i = reinterpret_cast<const int *>(&indexes); \
                v = CAT_HELPER(_mm_setr_, SUFFIX)(baseAddr[i[0]].*(member1), baseAddr[i[1]].*(member1), baseAddr[i[2]].*(member1), baseAddr[i[3]].*(member1)); \
            } \
            template<typename S1, typename S2> \
            static inline void gather(TYPE &v, const _M128I &indexes, const S1 *baseAddr, const S2 S1::* member1, const T S2::* member2) { \
                const int *const i = reinterpret_cast<const int *>(&indexes); \
                v = CAT_HELPER(_mm_setr_, SUFFIX)(baseAddr[i[0]].*(member1).*(member2), baseAddr[i[1]].*(member1).*(member2), baseAddr[i[2]].*(member1).*(member2), baseAddr[i[3]].*(member1).*(member2)); \
            }

        template<> struct VectorHelper<double> {
#define TYPE _M128D
#define SUFFIX pd
            LOAD(double)
            STORE(double)
            static inline void gather(TYPE &v, const _M128I &indexes, const double *baseAddr) {
                const int *const i = reinterpret_cast<const int *>(&indexes);
                v = _mm_setr_pd(baseAddr[i[0]], baseAddr[i[1]]);
            }
            template<typename S>
            static inline void gather(TYPE &v, const _M128I &indexes, const S *baseAddr, const double S::* member1) {
                const int *const i = reinterpret_cast<const int *>(&indexes);
                v = CAT_HELPER(_mm_setr_, SUFFIX)(baseAddr[i[0]].*(member1), baseAddr[i[1]].*(member1));
            }
            template<typename S1, typename S2>
            static inline void gather(TYPE &v, const _M128I &indexes, const S1 *baseAddr, const S2 S1::* member1, const double S2::* member2) {
                const int *const i = reinterpret_cast<const int *>(&indexes);
                v = CAT_HELPER(_mm_setr_, SUFFIX)(baseAddr[i[0]].*(member1).*(member2), baseAddr[i[1]].*(member1).*(member2));
            }


            static inline TYPE set(const double a) { return CAT_HELPER(_mm_set1_, SUFFIX)(a); }
            static inline TYPE set(const double a, const double b) { return CAT_HELPER(_mm_set_, SUFFIX)(a, b); }
            static inline void setZero(TYPE &v) { v = CAT_HELPER(_mm_setzero_, SUFFIX)(); }

            static inline void multiplyAndAdd(TYPE &v1, TYPE v2, TYPE v3) { v1 = add(mul(v1, v2), v3); }

            OP(add) OP(sub) OP(mul) OP(div)
            OPcmp(eq) OPcmp(neq)
            OPcmp(lt) OPcmp(nlt)
            OPcmp(le) OPcmp(nle)

            OP1(sqrt)
            static inline TYPE abs(const TYPE a) {
              static const TYPE mask = { 0x7fffffffffffffff, 0x7fffffffffffffff };
              return CAT_HELPER(_mm_and_, SUFFIX)(a, mask);
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
            GATHER(float)

            static inline TYPE set(const float a) { return CAT_HELPER(_mm_set1_, SUFFIX)(a); }
            static inline TYPE set(const float a, const float b, const float c, const float d) { return CAT_HELPER(_mm_set_, SUFFIX)(a, b, c, d); }
            static inline void setZero(TYPE &v) { v = CAT_HELPER(_mm_setzero_, SUFFIX)(); }

            static inline void multiplyAndAdd(TYPE &v1, TYPE v2, TYPE v3) { v1 = add(mul(v1, v2), v3); }

            OP(add) OP(sub) OP(mul) OP(div)
            OPcmp(eq) OPcmp(neq)
            OPcmp(lt) OPcmp(nlt)
            OPcmp(le) OPcmp(nle)

            OP1(sqrt)
            static inline TYPE abs(const TYPE a) {
              static const TYPE mask = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
              return CAT_HELPER(_mm_and_, SUFFIX)(a, mask);
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
            static inline void setZero(TYPE &v) { v = CAT_HELPER(_mm_setzero_, SUFFIX)(); }
#undef SUFFIX
#define SUFFIX epi32
            GATHER(int)

            static inline TYPE set(const int a) { return CAT_HELPER(_mm_set1_, SUFFIX)(a); }
            static inline TYPE set(const int a, const int b, const int c, const int d) { return CAT_HELPER(_mm_set_, SUFFIX)(a, b, c, d); }

            static inline void multiplyAndAdd(TYPE &v1, TYPE v2, TYPE v3) { v1 = add(mul(v1, v2), v3); }

#ifdef __SSSE3__
            OP1(abs)
#endif

#ifdef __SSE4_1__
            static inline TYPE mul(const TYPE a, const TYPE b) { return _mm_mullo_epi32(a, b); }
            MINMAX
#else
            static inline TYPE mul(const TYPE a, const TYPE b) {
                const int *const _a = reinterpret_cast<const int *>(&a);
                const int *const _b = reinterpret_cast<const int *>(&b);
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

            static inline TYPE div(const TYPE a, const TYPE b) {
                const int *const _a = reinterpret_cast<const int *>(&a);
                const int *const _b = reinterpret_cast<const int *>(&b);
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
            static inline TYPE cmpneq(const TYPE &a, const TYPE &b) { _M128I x = cmpeq(a, b); return _mm_xor_si128(x, _FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF); }
            static inline TYPE cmpnlt(const TYPE &a, const TYPE &b) { _M128I x = cmplt(a, b); return _mm_xor_si128(x, _FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF); }
            static inline TYPE cmple (const TYPE &a, const TYPE &b) { _M128I x = cmpgt(a, b); return _mm_xor_si128(x, _FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF); }
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
            static inline void setZero(TYPE &v) { v = CAT_HELPER(_mm_setzero_, SUFFIX)(); }

#undef SUFFIX
#define SUFFIX epu32

#ifdef __SSE4_1__
            MINMAX
#endif

            static inline TYPE mul(const TYPE a, const TYPE b) {
                TYPE hi = _mm_mulhi_epu16(a, b);
                hi = _mm_slli_epi32(hi, 16);
                TYPE lo = _mm_mullo_epi16(a, b);
                return or_(hi, lo);
            }
            static inline TYPE div(const TYPE a, const TYPE b) {
                const unsigned int *const _a = reinterpret_cast<const unsigned int *>(&a);
                const unsigned int *const _b = reinterpret_cast<const unsigned int *>(&b);
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
            GATHER(unsigned int)
            static inline TYPE set(const unsigned int a) { return CAT_HELPER(_mm_set1_, SUFFIX)(a); }
            static inline TYPE set(const unsigned int a, const unsigned int b, const unsigned int c, const unsigned int d) { return CAT_HELPER(_mm_set_, SUFFIX)(a, b, c, d); }

            OP(add) OP(sub)
            OPcmp(eq)
            OPcmp(lt)
            OPcmp(gt)
            static inline TYPE cmpneq(const TYPE &a, const TYPE &b) { return _mm_xor_si128(cmpeq(a, b), _FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF); }
            static inline TYPE cmpnlt(const TYPE &a, const TYPE &b) { return _mm_xor_si128(cmplt(a, b), _FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF); }
            static inline TYPE cmple (const TYPE &a, const TYPE &b) { return _mm_xor_si128(cmpgt(a, b), _FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF); }
            static inline TYPE cmpnle(const TYPE &a, const TYPE &b) { return cmpgt(a, b); }
#undef GATHER
#undef STORE
#undef LOAD
#undef TYPE
#undef SUFFIX
        };
#undef OP1
#undef OP
#undef OPx
#undef OPcmp
#undef CAT_HELPER
#undef CAT_HELPER2

namespace VectorSpecialInitializerZero { enum Enum { Zero }; }
namespace VectorSpecialInitializerRandom { enum Enum { Random }; }

class Mask;
extern bool cmpeq32_64(const Mask &, const Mask &);
class Mask : public VectorBase<int, Mask>
{
    friend struct VectorBase<int, Mask>;
    friend inline bool cmpeq32_64(const Mask &m1, const Mask &m2) {
        // ps gives 4 bits (MSB from 4 32bit values)
        // pd gives 2 bits (MSB from 2 64bit values)
        return (_mm_movemask_ps(_mm_castsi128_ps(m1.data)) & 3) == _mm_movemask_pd(_mm_castsi128_pd(m2.data));
    }
    protected:
        _M128I data;
    public:
        enum { Size = 16 / sizeof(int) };
        inline Mask() : data(*reinterpret_cast<const _M128I *>(_FullMaskData)) {}
        inline Mask(const __m128 &x) : data(_mm_castps_si128(x)) {}
        inline Mask(const __m128d &x) : data(_mm_castpd_si128(x)) {}
        inline Mask(const __m128i &x) : data(x) {}

        inline operator const Vector<int> &() const { return *reinterpret_cast<const Vector<int> *>(this); }
        inline operator bool() const {
            // _mm_movemask_epi8 creates a 16 bit mask containing the most significant bit of every byte in data
            return _mm_movemask_epi8(data);
        }
        inline bool operator==(const Mask &m) const {
            return _mm_movemask_epi8(data) == _mm_movemask_epi8(m.data);
        }
};
static const Mask &OneMask = *reinterpret_cast<const Mask *const>(_OneMaskData);
static const Mask &FullMask = *reinterpret_cast<const Mask *const>(_FullMaskData);
static inline Mask maskNthElement( int n ) {
    ALIGN(16) union
    {
        int i[4];
        _M128I m;
    } x = { { 0, 0, 0, 0 } };
    x.i[n] = 0xffffffff;
    return x.m;
}

template<typename T>
class Vector : public VectorBase<T, Vector<T> >
{
    friend struct VectorBase<T, Vector<T> >;
    protected:
        typedef typename VectorBase<T, Vector<T> >::IntrinType IntrinType;
        IntrinType data;
    public:
        typedef T Type;
        enum { Size = 16 / sizeof(T) };
        /**
         * uninitialized
         */
        inline Vector() {}
        /**
         * initialzed to 0 in all 512 bits
         */
        inline Vector(VectorSpecialInitializerZero::Enum) { makeZero(); }
        /**
         * initialize with given _M128 vector
         */
        inline Vector(const IntrinType &x) : data(x) {}
        inline Vector(const Mask &m) : data(m) {}
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
        template<typename Other>
        inline Vector(const Other *x)
        {
            data = VectorHelper<T>::load(x);
        }

        template<typename Other>
        static inline Vector broadcast4(const Other *x)
        {
            return Vector<T>(x);
        }

        inline void makeZero()
        {
            VectorHelper<T>::setZero(data);
        }

        /**
         * Store the vector data to the given memory. The memory must be 64 byte aligned and of 512
         * bytes size.
         */
        inline void store(void *mem) const
        {
            VectorHelper<T>::store(mem, data);
        }

        /**
         * Non-temporal store variant. Writes to the memory without polluting the cache.
         */
        inline void storeStreaming(void *mem) const
        {
            VectorHelper<T>::storeStreaming(mem, data);
        }

        inline const Vector<T> &dcba() const { return *this; }
        inline const Vector<T> cdab() const { return _mm_shuffle_epi32(data, _MM_SHUFFLE(2, 3, 0, 1)); }
        inline const Vector<T> badc() const { return _mm_shuffle_epi32(data, _MM_SHUFFLE(1, 0, 3, 2)); }
        inline const Vector<T> aaaa() const { return _mm_shuffle_epi32(data, _MM_SHUFFLE(0, 0, 0, 0)); }
        inline const Vector<T> bbbb() const { return _mm_shuffle_epi32(data, _MM_SHUFFLE(1, 1, 1, 1)); }
        inline const Vector<T> cccc() const { return _mm_shuffle_epi32(data, _MM_SHUFFLE(2, 2, 2, 2)); }
        inline const Vector<T> dddd() const { return _mm_shuffle_epi32(data, _MM_SHUFFLE(3, 3, 3, 3)); }
        inline const Vector<T> dbac() const { return _mm_shuffle_epi32(data, _MM_SHUFFLE(3, 1, 0, 2)); }

        inline Vector(const T *array, const Vector<int> &indexes) { VectorHelper<T>::gather(data, indexes, array); }
        inline Vector(const T *array, const Vector<int> &indexes, const Mask &m) {
            VectorHelper<T>::gather(data, indexes & Vector<int>(m), array);
        }
        inline void gather(const T *array, const Vector<int> &indexes) { VectorHelper<T>::gather(data, indexes, array); }
        inline void gather(const T *array, const Vector<int> &indexes, const Mask &m) {
            VectorHelper<T>::gather(data, indexes & Vector<int>(m), array);
        }

        /**
         * \param array An array of objects where one member should be gathered
         * \param member1 A member pointer to the member of the class/struct that should be gathered
         * \param indexes The indexes in the array. The correct offsets are calculated
         *                automatically.
         * \param mask Optional mask to select only parts of the vector that should be gathered
         */
        template<typename S> inline Vector(const S *array, const T S::* member1,
                const Vector<int> &indexes, const Mask &mask = Mask()) {
            VectorHelper<T>::gather(data, (indexes & Vector<int>(mask)), array, member1);
        }
        template<typename S1, typename S2> inline Vector(const S1 *array, const S2 S1::* member1,
                const T S2::* member2, const Vector<int> &indexes, const Mask &mask = Mask()) {
            VectorHelper<T>::gather(data, (indexes & Vector<int>(mask)), array, member1, member2);
        }
        template<typename S> inline void gather(const S *array, const T S::* member1,
                const Vector<int> &indexes, const Mask &mask = Mask()) {
            VectorHelper<T>::gather(data, (indexes & Vector<int>(mask)), array, member1);
        }
        template<typename S1, typename S2> inline void gather(const S1 *array, const S2 S1::* member1,
                const T S2::* member2, const Vector<int> &indexes, const Mask &mask = Mask()) {
            VectorHelper<T>::gather(data, (indexes & Vector<int>(mask)), array, member1, member2);
        }

        //prefix
        inline Vector &operator++() { data = VectorHelper<T>::add(data, Vector<T>(1)); return *this; }
        //postfix
        inline Vector operator++(int) { const Vector<T> r = *this; data = VectorHelper<T>::add(data, Vector<T>(1)); return r; }

        inline T operator[](int index) const {
            const T *const x = reinterpret_cast<const T *>(&data);
            return x[index];
        }

#define OP1(fun) \
        inline Vector fun() const { return Vector<T>(VectorHelper<T>::fun(data)); } \
        inline Vector &fun##_eq() { data = VectorHelper<T>::fun(data); return *this; }
        OP1(sqrt)
        OP1(abs)
#undef OP1

#define OP(symbol, fun) \
        inline Vector &operator symbol##=(const Vector<T> &x) { data = VectorHelper<T>::fun(data, x.data); return *this; } \
        inline Vector &operator symbol##=(const T &x) { return operator symbol##=(Vector<T>(x)); } \
        inline Vector operator symbol(const Vector<T> &x) const { return Vector<T>(VectorHelper<T>::fun(data, x.data)); } \
        inline Vector operator symbol(const T &x) const { return operator symbol(Vector<T>(x)); }

        OP(+, add)
        OP(-, sub)
        OP(*, mul)
        OP(/, div)
        OP(|, or_)
        OP(&, and_)
        OP(^, xor_)
#undef OP
#define OPcmp(symbol, fun) \
        inline Mask operator symbol(const Vector<T> &x) const { return VectorHelper<T>::fun(data, x.data); } \
        inline Mask operator symbol(const T &x) const { return operator symbol(Vector<T>(x)); }

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

        inline Vector &assign( const Vector<T> &v, const Mask &mask ) {
          // TODO
        }

        template<typename T2> inline Vector<T2> staticCast() const { return StaticCastHelper<T, T2>::cast(data); }
        template<typename T2> inline Vector<T2> reinterpretCast() const { return ReinterpretCastHelper<T, T2>::cast(data); }
};

template<typename T> class SwizzledVector : public Vector<T> {};

template<typename T> inline Vector<T> operator+(const T &x, const Vector<T> &v) { return v.operator+(x); }
template<typename T> inline Vector<T> operator*(const T &x, const Vector<T> &v) { return v.operator+(x); }
template<typename T> inline Vector<T> operator-(const T &x, const Vector<T> &v) { return Vector<T>(x) - v; }
template<typename T> inline Vector<T> operator/(const T &x, const Vector<T> &v) { return Vector<T>(x) / v; }
template<typename T> inline Mask  operator< (const T &x, const Vector<T> &v) { return Vector<T>(x) <  v; }
template<typename T> inline Mask  operator<=(const T &x, const Vector<T> &v) { return Vector<T>(x) <= v; }
template<typename T> inline Mask  operator> (const T &x, const Vector<T> &v) { return Vector<T>(x) >  v; }
template<typename T> inline Mask  operator>=(const T &x, const Vector<T> &v) { return Vector<T>(x) >= v; }
template<typename T> inline Mask  operator==(const T &x, const Vector<T> &v) { return Vector<T>(x) == v; }
template<typename T> inline Mask  operator!=(const T &x, const Vector<T> &v) { return Vector<T>(x) != v; }

#define PARENT_DATA (static_cast<Vector<int> *>(this)->data)
#define PARENT_DATA_CONST (static_cast<const Vector<int> *>(this)->data)
#define OP_IMPL(symbol, fun) \
  template<> inline Vector<int> &VectorBase<int, Vector<int> >::operator symbol##=(const Vector<int> &x) { PARENT_DATA = VectorHelper<int>::fun(PARENT_DATA, x.data); return *static_cast<Vector<int> *>(this); } \
  template<> inline Vector<int> &VectorBase<int, Vector<int> >::operator symbol##=(const int &x) { return operator symbol##=(Vector<int>(x)); } \
  template<> inline Vector<int> VectorBase<int, Vector<int> >::operator symbol(const Vector<int> &x) const { return Vector<int>(VectorHelper<int>::fun(PARENT_DATA_CONST, x.data)); } \
  template<> inline Vector<int> VectorBase<int, Vector<int> >::operator symbol(const int &x) const { return operator symbol(Vector<int>(x)); }
  OP_IMPL(&, and_)
  OP_IMPL(|, or_)
  OP_IMPL(^, xor_)
#undef OP_IMPL
#undef ALIGN

#undef PARENT_DATA_CONST
#undef PARENT_DATA

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
} // namespace SSE

#endif // SSE_VECTOR_H
