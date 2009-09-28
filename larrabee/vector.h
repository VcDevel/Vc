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

#ifndef LARRABEE_VECTOR_H
#define LARRABEE_VECTOR_H

#include "intrinsics.h"

#define VC_HAVE_FMA

#define CAT_HELPER(a, b) a##b
#define CAT(a, b) CAT_HELPER(a, b)

namespace Vc
{
#ifndef HAVE_FLOAT16
#define HAVE_FLOAT16
#ifdef HALF_MAX
    typedef half float16;
#else
    class float16 {
        public:
            unsigned short data;
    };
#endif
#endif

#ifdef isfinite
#undef isfinite
#endif
#ifdef isnan
#undef isnan
#endif

#include <mm_malloc.h>

namespace LRBni
{
    enum { VectorAlignment = 64 };

    class VectorAlignedBase
    {
        public:
            void *operator new(size_t size) { return _mm_malloc(size, VectorAlignment); }
            void *operator new[](size_t size) { return _mm_malloc(size, VectorAlignment); }
            void operator delete(void *ptr, size_t) { _mm_free(ptr); }
            void operator delete[](void *ptr, size_t) { _mm_free(ptr); }
    } LRB_ALIGN(64);

    namespace VectorSpecialInitializerZero { enum ZEnum { Zero = 0 }; }
    namespace VectorSpecialInitializerOne { enum OEnum { One = 1 }; }
    namespace VectorSpecialInitializerRandom { enum REnum { Random }; }
    namespace VectorSpecialInitializerIndexesFromZero { enum IEnum { IndexesFromZero }; }

    LRB_ALIGN(16) extern const char _IndexesFromZero[16];

    template<typename T> struct ReturnTypeHelper { typedef char Type; };
    template<> struct ReturnTypeHelper<unsigned int> { typedef unsigned char Type; };
    template<> struct ReturnTypeHelper<int> { typedef signed char Type; };
    template<typename T> const typename ReturnTypeHelper<T>::Type *IndexesFromZero() {
        return reinterpret_cast<const typename ReturnTypeHelper<T>::Type *>(&_IndexesFromZero[0]);
    }

    template<bool> class STATIC_ASSERT_FAILURE;
    template<> class STATIC_ASSERT_FAILURE<true> {};

#define LRB_STATIC_ASSERT_NC(cond, msg) \
    typedef STATIC_ASSERT_FAILURE<cond> CAT(_STATIC_ASSERTION_FAILED_##msg, __LINE__); \
    CAT(_STATIC_ASSERTION_FAILED_##msg, __LINE__) Error_##msg
#define LRB_STATIC_ASSERT(cond, msg) LRB_STATIC_ASSERT_NC(cond, msg); (void) Error_##msg

    template<typename T> class Vector;
    template<typename T> struct SwizzledVector;

    template<unsigned int VectorSize> struct MaskHelper;
    template<> struct MaskHelper<8> {
        static inline bool isFull (__mmask k) { return (k & 0xff) == 0xff; }
        static inline bool isEmpty(__mmask k) { return (k & 0xff) == 0x00; }
        static inline bool isMix  (__mmask k) { const int tmp = k & 0xff; return tmp != 0 && (tmp ^ 0xff) != 0; }
        static inline bool cmpeq  (__mmask k1, __mmask k2) { return (k1 & 0xff) == (k2 & 0xff); }
        static inline bool cmpneq (__mmask k1, __mmask k2) { return (k1 & 0xff) != (k2 & 0xff); }
    };
    template<> struct MaskHelper<16> {
        static inline bool isFull (__mmask k) { return k == 0xffff; }
        static inline bool isEmpty(__mmask k) { return k == 0x0000; }
        static inline bool isMix  (__mmask k) { return k != 0 && (k ^ 0xffff) != 0; }
        static inline bool cmpeq  (__mmask k1, __mmask k2) { return k1 == k2; }
        static inline bool cmpneq (__mmask k1, __mmask k2) { return k1 != k2; }
    };

    template<unsigned int VectorSize> class Mask
    {
        friend class Mask<8u>;
        friend class Mask<16u>;
        public:
            inline Mask() {}
            inline Mask(__mmask _k) : k(_k) {}
            inline explicit Mask(VectorSpecialInitializerZero::ZEnum) : k(0) {}
            inline explicit Mask(VectorSpecialInitializerOne::OEnum) : k(0xffffu) {}
            inline explicit Mask(bool b) : k(b ? 0xffffu : 0) {}
            inline Mask(const Mask<VectorSize / 2> &a, const Mask<VectorSize / 2> &b) : k(a.k | (b.k << 8)) {}
            template<unsigned int OtherSize> explicit inline Mask(const Mask<OtherSize> &x) : k(x.k) {
                if (OtherSize != VectorSize) {
                    enum { Shift = VectorSize < OtherSize ? VectorSize : OtherSize };
                    const unsigned short mask = (0xffffu << Shift) & 0xffffu;
                    k &= ~mask;
                }
            }

            inline bool operator==(const Mask &rhs) const { return MaskHelper<VectorSize>::cmpeq (k, rhs.k); }
            inline bool operator!=(const Mask &rhs) const { return MaskHelper<VectorSize>::cmpneq(k, rhs.k); }

            inline Mask operator&&(const Mask &rhs) const { return _mm512_vkand(k, rhs.k); }
            inline Mask operator& (const Mask &rhs) const { return _mm512_vkand(k, rhs.k); }
            inline Mask operator||(const Mask &rhs) const { return _mm512_vkor (k, rhs.k); }
            inline Mask operator| (const Mask &rhs) const { return _mm512_vkor (k, rhs.k); }
            inline Mask operator^ (const Mask &rhs) const { return _mm512_vkxor(k, rhs.k); }
            inline Mask operator!() const { return ~k; }

            inline Mask &operator&=(const Mask &rhs) { k = _mm512_vkand(k, rhs.k); return *this; }
            inline Mask &operator|=(const Mask &rhs) { k = _mm512_vkor (k, rhs.k); return *this; }

            inline bool isFull () const { return MaskHelper<VectorSize>::isFull (k); }
            inline bool isEmpty() const { return MaskHelper<VectorSize>::isEmpty(k); }
            inline bool isMix  () const { return MaskHelper<VectorSize>::isMix  (k); }

            inline operator bool() const { return isFull(); }

            inline __mmask data () const { return k; }
            inline __mmask dataI() const { return k; }
            inline __mmask dataD() const { return k; }

            template<unsigned int OtherSize>
            inline Mask<OtherSize> cast() const { return Mask<OtherSize>(k); }

            inline bool operator[](int index) const { return static_cast<bool>(k & (1 << index)); }

            inline int count() const { return _mm_countbits_16(k); }

            /**
             * Returns the index of the first one in the mask.
             *
             * The return value is undefined if the mask is empty.
             */
            int firstOne() const { return _mm_bsff_16(k); }

        private:
            __mmask k;
    };

struct ForeachHelper
{
    unsigned short mask;
    short bit;
    bool brk;
    inline ForeachHelper(unsigned short _mask) :
        mask(_mask),
        bit(_mm_bsff_16(mask)),
        brk(false)
    {}
    inline bool outer() const { return bit != -1; }
    inline bool inner() { return (brk = !brk); }
    inline short next() const { return bit; }
    inline void step() { bit = _mm_bsfi_16(bit, mask); }
};

/**
 * Loop over all set bits in the mask. The iterator variable will be set to the position of the set
 * bits. A mask of e.g. 00011010 would result in the loop being called with the iterator being set to
 * 1, 3, and 4.
 *
 * This allows you to write:
 * \code
 * float_v a = ...;
 * foreach_bit(int i, a < 0.f) {
 *   std::cout << a[i] << "\n";
 * }
 * \endcode
 * The example prints all the values in \p a that are negative, and only those.
 *
 * \param it   The iterator variable. For example "int i".
 * \param mask The mask to iterate over. You can also just write a vector operation that returns a
 *             mask.
 */
#define Vc_foreach_bit(_it_, _mask_) \
    for (Vc::LRBni::ForeachHelper _Vc_foreach_bit_helper((_mask_).data()); \
            _Vc_foreach_bit_helper.outer(); \
            _Vc_foreach_bit_helper.step()) \
        for (_it_ = _Vc_foreach_bit_helper.next(); _Vc_foreach_bit_helper.inner(); )

#define foreach_bit(_it_, _mask_) Vc_foreach_bit(_it_, _mask_)

    class float11_11_10 { public:
        enum Component {
            X = _MM_FULLUPC_FLOAT11A,
            Y = _MM_FULLUPC_FLOAT11B,
            Z = _MM_FULLUPC_FLOAT10C
        };
        unsigned int data;
    };

    namespace
    {
        template<typename From, typename To> struct StaticCastHelper {};
        template<> struct StaticCastHelper<float       , int         > { static _M512I cast(const _M512  &v) { return _mm512_cvt_ps2pi(v, _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE); } };
        template<> struct StaticCastHelper<float       , unsigned int> { static _M512I cast(const _M512  &v) { return _mm512_cvt_ps2pu(v, _MM_ROUND_MODE_TOWARD_ZERO, _MM_EXPADJ_NONE); } };
        template<> struct StaticCastHelper<float       , float       > { static _M512  cast(const _M512  &v) { return v; } };
        template<> struct StaticCastHelper<float       , double      > { static _M512D cast(const _M512  &v) { return _mm512_cvtl_ps2pd(v); } };
        template<> struct StaticCastHelper<double      , int         > { static _M512I cast(const _M512D &v) { return _mm512_cvtl_pd2pi(_M512I(), v, _MM_ROUND_MODE_TOWARD_ZERO); } };
        template<> struct StaticCastHelper<double      , unsigned int> { static _M512I cast(const _M512D &v) { return _mm512_cvtl_pd2pu(_M512I(), v, _MM_ROUND_MODE_TOWARD_ZERO); } };
        template<> struct StaticCastHelper<double      , float       > { static _M512  cast(const _M512D &v) { return _mm512_cvtl_pd2ps(_M512(), v, _MM_ROUND_MODE_NEAREST); } };
        template<> struct StaticCastHelper<double      , double      > { static _M512D cast(const _M512D &v) { return v; } };
        template<> struct StaticCastHelper<int         , int         > { static _M512I cast(const _M512I &v) { return v; } };
        template<> struct StaticCastHelper<int         , unsigned int> { static _M512I cast(const _M512I &v) { return v; } };
        template<> struct StaticCastHelper<int         , float       > { static _M512  cast(const _M512I &v) { return _mm512_cvt_pi2ps(v, _MM_EXPADJ_NONE); } };
        template<> struct StaticCastHelper<int         , double      > { static _M512D cast(const _M512I &v) { return _mm512_cvtl_pi2pd(v); } };
        template<> struct StaticCastHelper<unsigned int, int         > { static _M512I cast(const _M512I &v) { return v; } };
        template<> struct StaticCastHelper<unsigned int, unsigned int> { static _M512I cast(const _M512I &v) { return v; } };
        template<> struct StaticCastHelper<unsigned int, float       > { static _M512  cast(const _M512I &v) { return _mm512_cvt_pu2ps(v, _MM_EXPADJ_NONE); } };
        template<> struct StaticCastHelper<unsigned int, double      > { static _M512D cast(const _M512I &v) { return _mm512_cvtl_pu2pd(v); } };

        template<typename From, typename To> struct ReinterpretCastHelper {};
        template<> struct ReinterpretCastHelper<float       , int         > { static _M512I cast(const _M512  &v) { return _mm512_castps_si512(v); } };
        template<> struct ReinterpretCastHelper<double      , int         > { static _M512I cast(const _M512D &v) { return _mm512_castpd_si512(v); } };
        template<> struct ReinterpretCastHelper<int         , int         > { static _M512I cast(const _M512I &v) { return v; } };
        template<> struct ReinterpretCastHelper<unsigned int, int         > { static _M512I cast(const _M512I &v) { return v; } };
        template<> struct ReinterpretCastHelper<float       , unsigned int> { static _M512I cast(const _M512  &v) { return _mm512_castps_si512(v); } };
        template<> struct ReinterpretCastHelper<double      , unsigned int> { static _M512I cast(const _M512D &v) { return _mm512_castpd_si512(v); } };
        template<> struct ReinterpretCastHelper<int         , unsigned int> { static _M512I cast(const _M512I &v) { return v; } };
        template<> struct ReinterpretCastHelper<unsigned int, unsigned int> { static _M512I cast(const _M512I &v) { return v; } };
        template<> struct ReinterpretCastHelper<float       , float       > { static _M512  cast(const _M512  &v) { return v; } };
        template<> struct ReinterpretCastHelper<double      , float       > { static _M512  cast(const _M512D &v) { return _mm512_castpd_ps(v);    } };
        template<> struct ReinterpretCastHelper<int         , float       > { static _M512  cast(const _M512I &v) { return _mm512_castsi512_ps(v); } };
        template<> struct ReinterpretCastHelper<unsigned int, float       > { static _M512  cast(const _M512I &v) { return _mm512_castsi512_ps(v); } };
        template<> struct ReinterpretCastHelper<float       , double      > { static _M512D cast(const _M512  &v) { return _mm512_castps_pd(v);    } };
        template<> struct ReinterpretCastHelper<double      , double      > { static _M512D cast(const _M512D &v) { return v; } };
        template<> struct ReinterpretCastHelper<int         , double      > { static _M512D cast(const _M512I &v) { return _mm512_castsi512_pd(v); } };
        template<> struct ReinterpretCastHelper<unsigned int, double      > { static _M512D cast(const _M512I &v) { return _mm512_castsi512_pd(v); } };

        template<typename To, typename From> To mm512_reinterpret_cast(From v) { return v; }
        template<> _M512I mm512_reinterpret_cast<_M512I, _M512 >(_M512  v) { return _mm512_castps_si512(v); }
        template<> _M512I mm512_reinterpret_cast<_M512I, _M512D>(_M512D v) { return _mm512_castpd_si512(v); }
        template<> _M512  mm512_reinterpret_cast<_M512 , _M512D>(_M512D v) { return _mm512_castpd_ps(v);    }
        template<> _M512  mm512_reinterpret_cast<_M512 , _M512I>(_M512I v) { return _mm512_castsi512_ps(v); }
        template<> _M512D mm512_reinterpret_cast<_M512D, _M512I>(_M512I v) { return _mm512_castsi512_pd(v); }
        template<> _M512D mm512_reinterpret_cast<_M512D, _M512 >(_M512  v) { return _mm512_castps_pd(v);    }
    } // anonymous namespace

#define PARENT_DATA (static_cast<Parent *>(this)->data)
#define PARENT_DATA_CONST (static_cast<const Parent *>(this)->data)
#define OP_DECL(symbol, fun) \
    inline Vector<T> &fun##_eq(const Vector<T> &x, const __mmask m); \
    inline Vector<T> &fun##_eq(const Vector<T> &x, const __mmask m, const Vector<T> &old); \
    inline Vector<T> &fun##_eq(const T &x, const __mmask m); \
    inline Vector<T> &fun##_eq(const T &x, const __mmask m, const Vector<T> &old); \
    inline Vector<T>  fun(const Vector<T> &x, const __mmask m) const; \
    inline Vector<T>  fun(const Vector<T> &x, const __mmask m, const Vector<T> &old) const; \
    inline Vector<T>  fun(const T &x, const __mmask m) const; \
    inline Vector<T>  fun(const T &x, const __mmask m, const Vector<T> &old) const; \
    inline Vector<T> &operator symbol##=(const Vector<T> &x); \
    inline Vector<T> &operator symbol##=(const T &x); \
    inline Vector<T>  operator symbol(const Vector<T> &x) const; \
    inline Vector<T>  operator symbol(const T &x) const;
    template<typename T, typename Parent> struct VectorBase
    {
        operator _M512() { return PARENT_DATA; }
        operator const _M512() const { return PARENT_DATA_CONST; }
    };
    template<typename Parent> struct VectorBase<float, Parent>
    {
        operator _M512() { return PARENT_DATA; }
        operator const _M512() const { return PARENT_DATA_CONST; }
        enum Upconvert {
            UpconvertNone     = _MM_FULLUPC_NONE,     /* no conversion      */
            UpconvertFloat16  = _MM_FULLUPC_FLOAT16,  /* float16 => float32 */
            UpconvertSrgb8    = _MM_FULLUPC_SRGB8,    /* srgb8   => float32 */
            UpconvertUint8    = _MM_FULLUPC_UINT8,    /* uint8   => float32 */
            UpconvertSint8    = _MM_FULLUPC_SINT8,    /* sint8   => float32 */
            UpconvertUnorm8   = _MM_FULLUPC_UNORM8,   /* unorm8  => float32 */
            UpconvertSnorm8   = _MM_FULLUPC_SNORM8,   /* snorm8  => float32 */
            UpconvertUint16   = _MM_FULLUPC_UINT16,   /* uint16  => float32 */
            UpconvertSint16   = _MM_FULLUPC_SINT16,   /* sint16  => float32 */
            UpconvertUnorm16  = _MM_FULLUPC_UNORM16,  /* unorm16 => float32 */
            UpconvertSnorm16  = _MM_FULLUPC_SNORM16,  /* snorm16 => float32 */
            UpconvertUnorm10A = _MM_FULLUPC_UNORM10A, /* unorm10A10B10C2D field A => float32 */
            UpconvertUnorm10B = _MM_FULLUPC_UNORM10B, /* unorm10A10B10C2D field B => float32 */
            UpconvertUnorm10C = _MM_FULLUPC_UNORM10C, /* unorm10A10B10C2D field C => float32 */
            UpconvertUnorm2D  = _MM_FULLUPC_UNORM2D,  /* unorm10A10B10C2D field D => float32 */
            UpconvertFloat11A = _MM_FULLUPC_FLOAT11A, /* float11A11B10C field A   => float32 */
            UpconvertFloat11B = _MM_FULLUPC_FLOAT11B, /* float11A11B10C field B   => float32 */
            UpconvertFloat10C = _MM_FULLUPC_FLOAT10C  /* float11A11B10C field C   => float32 */
        };
    };
    template<typename Parent> struct VectorBase<double, Parent>
    {
        operator _M512D() { return PARENT_DATA; }
        operator const _M512D() const { return PARENT_DATA_CONST; }
        enum Upconvert {
            UpconvertNone     = _MM_FULLUPC_NONE   /* no conversion      */
            // no other upconversions for double yet
        };
    };
    template<typename Parent> struct VectorBase<int, Parent>
    {
        typedef int T;
        operator _M512I() { return PARENT_DATA; }
        operator const _M512I() const { return PARENT_DATA_CONST; }
        enum Upconvert {
            UpconvertNone  = _MM_FULLUPC_NONE,   /* no conversion      */
            UpconvertInt8  = _MM_FULLUPC_SINT8I, /* sint8   => sint32  */
            UpconvertInt16 = _MM_FULLUPC_SINT16I /* sint16  => sint32  */
        };
        OP_DECL(|, or_)
        OP_DECL(&, and_)
        OP_DECL(^, xor_)
        OP_DECL(>>, srl)
        OP_DECL(<<, sll)
    };
    template<typename Parent> struct VectorBase<unsigned int, Parent>
    {
        typedef unsigned int T;
        operator _M512I() { return PARENT_DATA; }
        operator const _M512I() const { return PARENT_DATA_CONST; }
        enum Upconvert {
            UpconvertNone   = _MM_FULLUPC_NONE,   /* no conversion      */
            UpconvertUint8  = _MM_FULLUPC_UINT8I, /* uint8   => uint32  */
            UpconvertUint16 = _MM_FULLUPC_UINT16I /* uint16  => uint32  */
        };

        OP_DECL(|, or_)
        OP_DECL(&, and_)
        OP_DECL(^, xor_)
        OP_DECL(>>, srl)
        OP_DECL(<<, sll)
    };
#undef OP_DECL
#undef PARENT_DATA
#undef PARENT_DATA_CONST

    namespace
    {
#define OP1(op) \
        static inline VectorType op(const VectorType &a) { return CAT(_mm512_##op##_, SUFFIX)(a); } \
        static inline VectorType op(const VectorType &a, const __mmask &k) { return CAT(_mm512_mask_##op##_, SUFFIX)(a, k, a); } \
        static inline VectorType op(const VectorType &a, const __mmask &k, const VectorType &o) { return CAT(_mm512_mask_##op##_, SUFFIX)(o, k, a); } \
        static inline VectorType op##_s(unsigned int , const VectorType &a) { return CAT(_mm512_##op##_, SUFFIX)(a); } \
        static inline VectorType op##_s(unsigned int , const VectorType &a, const __mmask &k) { return CAT(_mm512_mask_##op##_, SUFFIX)(a, k, a); } \
        static inline VectorType op##_s(unsigned int , const VectorType &a, const __mmask &k, const VectorType &o) { return CAT(_mm512_mask_##op##_, SUFFIX)(o, k, a); }
#define OP(op) \
        static inline VectorType op(const VectorType &a, const VectorType &b) { return CAT(_mm512_##op##_, SUFFIX)(a, b); } \
        static inline VectorType op(const VectorType &a, const VectorType &b, const __mmask &k) { return CAT(_mm512_mask_##op##_, SUFFIX)(a, k, a, b); } \
        static inline VectorType op(const VectorType &a, const VectorType &b, const __mmask &k, const VectorType &o) { return CAT(_mm512_mask_##op##_, SUFFIX)(o, k, a, b); } \
        static inline VectorType op##_s(unsigned int , const VectorType &a, const VectorType &b) { return CAT(_mm512_##op##_, SUFFIX)(a, b); } \
        static inline VectorType op##_s(unsigned int , const VectorType &a, const VectorType &b, const __mmask &k) { return CAT(_mm512_mask_##op##_, SUFFIX)(a, k, a, b); } \
        static inline VectorType op##_s(unsigned int , const VectorType &a, const VectorType &b, const __mmask &k, const VectorType &o) { return CAT(_mm512_mask_##op##_, SUFFIX)(o, k, a, b); }
#define OP_(op) \
        static inline VectorType op(const VectorType &a, const VectorType &b) { return CAT(_mm512_##op, SUFFIX)(a, b); } \
        static inline VectorType op(const VectorType &a, const VectorType &b, const __mmask &k) { return CAT(_mm512_mask_##op, SUFFIX)(a, k, a, b); } \
        static inline VectorType op(const VectorType &a, const VectorType &b, const __mmask &k, const VectorType &o) { return CAT(_mm512_mask_##op, SUFFIX)(o, k, a, b); } \
        static inline VectorType op##_s(unsigned int , const VectorType &a, const VectorType &b) { return CAT(_mm512_##op, SUFFIX)(a, b); } \
        static inline VectorType op##_s(unsigned int , const VectorType &a, const VectorType &b, const __mmask &k) { return CAT(_mm512_mask_##op, SUFFIX)(a, k, a, b); } \
        static inline VectorType op##_s(unsigned int , const VectorType &a, const VectorType &b, const __mmask &k, const VectorType &o) { return CAT(_mm512_mask_##op, SUFFIX)(o, k, a, b); }
#define OPx(op, op2) \
        static inline VectorType op(const VectorType &a, const VectorType &b) { return CAT(_mm512_##op2##_, SUFFIX)(a, b); } \
        static inline VectorType op(const VectorType &a, const VectorType &b, const __mmask &k) { return CAT(_mm512_mask_##op2##_, SUFFIX)(a, k, a, b); } \
        static inline VectorType op(const VectorType &a, const VectorType &b, const __mmask &k, const VectorType &o) { return CAT(_mm512_mask_##op2##_, SUFFIX)(o, k, a, b); } \
        static inline VectorType op##_s(unsigned int , const VectorType &a, const VectorType &b) { return CAT(_mm512_##op2##_, SUFFIX)(a, b); } \
        static inline VectorType op##_s(unsigned int , const VectorType &a, const VectorType &b, const __mmask &k) { return CAT(_mm512_mask_##op2##_, SUFFIX)(a, k, a, b); } \
        static inline VectorType op##_s(unsigned int , const VectorType &a, const VectorType &b, const __mmask &k, const VectorType &o) { return CAT(_mm512_mask_##op2##_, SUFFIX)(o, k, a, b); }
#define OPcmp(op) \
        static inline __mmask cmp##op(const VectorType &a, const VectorType &b) { return CAT(_mm512_cmp##op##_, SUFFIX)(a, b); } \
        static inline __mmask cmp##op(const VectorType &a, const VectorType &b, const __mmask &k) { return CAT(_mm512_mask_cmp##op##_, SUFFIX)(k, a, b); } \
        static inline __mmask cmp##op##_s(unsigned int , const VectorType &a, const VectorType &b) { return CAT(_mm512_cmp##op##_, SUFFIX)(a, b); } \
        static inline __mmask cmp##op##_s(unsigned int , const VectorType &a, const VectorType &b, const __mmask &k) { return CAT(_mm512_mask_cmp##op##_, SUFFIX)(k, a, b); }
#define OPcmpQ(op) \
        static inline __mmask cmp##op(const VectorType &a, const VectorType &b) { return CAT(_mm512_mask_cmp##op##_, SUFFIX)(0x00ff, a, b); } \
        static inline __mmask cmp##op(const VectorType &a, const VectorType &b, const __mmask &k) { return CAT(_mm512_mask_cmp##op##_, SUFFIX)(k, a, b); } \
        static inline __mmask cmp##op##_s(unsigned int , const VectorType &a, const VectorType &b) { return CAT(_mm512_mask_cmp##op##_, SUFFIX)(0x00ff, a, b); } \
        static inline __mmask cmp##op##_s(unsigned int , const VectorType &a, const VectorType &b, const __mmask &k) { return CAT(_mm512_mask_cmp##op##_, SUFFIX)(k, a, b); }

        struct VectorDHelper
        {
            template<typename T> static inline void mov(T &v1, __mmask k, T v2 ) {
                v1 = mm512_reinterpret_cast<T>(_mm512_mask_movd(mm512_reinterpret_cast<_M512>(v1), k, mm512_reinterpret_cast<_M512>(v2)));
            }
        };
        struct VectorQHelper
        {
            static inline void mov(_M512D &v1, __mmask k, _M512D v2 ) {
                v1 = mm512_reinterpret_cast<_M512D>(_mm512_mask_movq(mm512_reinterpret_cast<_M512>(v1), k, mm512_reinterpret_cast<_M512>(v2)));
            }
        };

        template<typename T> struct VectorDQHelper : public VectorDHelper {};
        template<> struct VectorDQHelper<double> : public VectorQHelper {};

        template<typename T> struct VectorHelper;

        template<> struct VectorHelper<double> {
            typedef double EntryType;
            typedef _M512D VectorType;
#define SUFFIX pd
            // double doesn't support any upconversion
            static inline VectorType load1(const EntryType  x) { return mm512_reinterpret_cast<VectorType>(FixedIntrinsics::_mm512_loadq(&x, _MM_FULLUPC64_NONE, _MM_BROADCAST_1X8)); }
            static inline VectorType load4(const EntryType *x) { return mm512_reinterpret_cast<VectorType>(FixedIntrinsics::_mm512_loadq( x, _MM_FULLUPC64_NONE, _MM_BROADCAST_4X8)); }
            static inline VectorType load (const EntryType *x) { return mm512_reinterpret_cast<VectorType>(FixedIntrinsics::_mm512_loadq( x, _MM_FULLUPC64_NONE, _MM_BROADCAST_8X8)); }

            static inline void store         (void *mem, VectorType x) { _mm512_storeq(mem, mm512_reinterpret_cast<_M512>(x), _MM_DOWNC64_NONE, _MM_SUBSET64_8, _MM_HINT_NONE); }
            static inline void store         (void *mem, VectorType x, __mmask k) { _mm512_mask_storeq(mem, k, mm512_reinterpret_cast<_M512>(x), _MM_DOWNC64_NONE, _MM_SUBSET64_8, _MM_HINT_NONE); }
            static inline void storeStreaming(void *mem, VectorType x) { _mm512_storeq(mem, mm512_reinterpret_cast<_M512>(x), _MM_DOWNC64_NONE, _MM_SUBSET64_8, _MM_HINT_NT); }
            static inline void storeStreaming(void *mem, VectorType x, __mmask k) { _mm512_mask_storeq(mem, k, mm512_reinterpret_cast<_M512>(x), _MM_DOWNC64_NONE, _MM_SUBSET64_8, _MM_HINT_NT); }

            static inline VectorType zero() { return CAT(_mm512_setzero_, SUFFIX)(); }
            static inline VectorType set(EntryType x) { return CAT(_mm512_set_1to8_, SUFFIX)(x); }

            static inline void prepareGatherIndexes(_M512I &indexes) {
                indexes = mm512_reinterpret_cast<_M512I>(_mm512_mask_movq(
                            _mm512_shuf128x32(mm512_reinterpret_cast<_M512>(indexes), _MM_PERM_BBAA, _MM_PERM_DDCC),
                            0x33,
                            _mm512_shuf128x32(mm512_reinterpret_cast<_M512>(indexes), _MM_PERM_BBAA, _MM_PERM_BBAA)
                            ));
                indexes = _mm512_add_pi(indexes, _mm512_set_4to16_pi(0, 1, 0, 1));
            }
            static inline __mmask scaleMask(__mmask k) {
                __mmask r = 0;
                for (int i = 7; i >= 0; --i) {
                    if (k & (1 << i)) {
                        r |= (3 << 2 * i);
                    }
                }
                return r;
            }
            static inline VectorType gather(_M512I indexes, const EntryType *baseAddr) {
                prepareGatherIndexes(indexes);
                return mm512_reinterpret_cast<VectorType>(
                        _mm512_gatherd(indexes, const_cast<EntryType *>(baseAddr), _MM_FULLUPC_NONE, _MM_SCALE_4, _MM_HINT_NONE)
                        );
            }
            static inline void gather(VectorType &data, _M512I indexes, const EntryType *baseAddr, __mmask k) {
                prepareGatherIndexes(indexes);
                data = mm512_reinterpret_cast<VectorType>(
                        _mm512_mask_gatherd(mm512_reinterpret_cast<_M512>(data), scaleMask(k), indexes, const_cast<EntryType *>(baseAddr), _MM_FULLUPC_NONE, _MM_SCALE_4, _MM_HINT_NONE)
                        );
            }
            static inline void gatherScale1(VectorType &data, _M512I indexes, const EntryType *baseAddr, __mmask k) {
                indexes = mm512_reinterpret_cast<_M512I>(_mm512_mask_movq(
                            _mm512_shuf128x32(mm512_reinterpret_cast<_M512>(indexes), _MM_PERM_BBAA, _MM_PERM_DDCC),
                            0x33,
                            _mm512_shuf128x32(mm512_reinterpret_cast<_M512>(indexes), _MM_PERM_BBAA, _MM_PERM_BBAA)
                            ));
                indexes = _mm512_add_pi(indexes, _mm512_set_4to16_pi(0, 4, 0, 4));
                data = mm512_reinterpret_cast<VectorType>(
                        _mm512_mask_gatherd(mm512_reinterpret_cast<_M512>(data), scaleMask(k), indexes, const_cast<EntryType *>(baseAddr), _MM_FULLUPC_NONE, _MM_SCALE_1, _MM_HINT_NONE)
                        );
            }
            static inline VectorType gatherStreaming(_M512I indexes, const EntryType *baseAddr) {
                prepareGatherIndexes(indexes);
                return mm512_reinterpret_cast<VectorType>(
                        _mm512_gatherd(indexes, const_cast<EntryType *>(baseAddr), _MM_FULLUPC_NONE, _MM_SCALE_4, _MM_HINT_NT)
                        );
            }
            static inline void gatherStreaming(VectorType &data, _M512I indexes, const EntryType *baseAddr, __mmask k) {
                prepareGatherIndexes(indexes);
                data = mm512_reinterpret_cast<VectorType>(
                        _mm512_mask_gatherd(mm512_reinterpret_cast<_M512>(data), scaleMask(k), indexes, const_cast<EntryType *>(baseAddr), _MM_FULLUPC_NONE, _MM_SCALE_4, _MM_HINT_NT)
                        );
            }
            static inline void scatter(const VectorType data, _M512I indexes, EntryType *baseAddr) {
                prepareGatherIndexes(indexes);
                _mm512_scatterd(baseAddr, indexes, mm512_reinterpret_cast<_M512>(data), _MM_DOWNC_NONE,  _MM_SCALE_4, _MM_HINT_NONE);
            }
            static inline void scatter(const VectorType data, _M512I indexes, EntryType *baseAddr, __mmask k) {
                prepareGatherIndexes(indexes);
                _mm512_mask_scatterd(baseAddr, scaleMask(k), indexes, mm512_reinterpret_cast<_M512>(data), _MM_DOWNC_NONE, _MM_SCALE_4, _MM_HINT_NONE);
            }
            static inline void scatterStreaming(const VectorType data, _M512I indexes, EntryType *baseAddr) {
                prepareGatherIndexes(indexes);
                _mm512_scatterd(baseAddr, indexes, mm512_reinterpret_cast<_M512>(data), _MM_DOWNC_NONE,  _MM_SCALE_4, _MM_HINT_NT);
            }
            static inline void scatterStreaming(const VectorType data, _M512I indexes, EntryType *baseAddr, __mmask k) {
                prepareGatherIndexes(indexes);
                _mm512_mask_scatterd(baseAddr, scaleMask(k), indexes, mm512_reinterpret_cast<_M512>(data), _MM_DOWNC_NONE, _MM_SCALE_4, _MM_HINT_NT);
            }

            static inline VectorType multiplyAndAdd(const VectorType &v1, const VectorType &v2, const VectorType &v3) { return _mm512_madd132_pd(v1, v3, v2); }
            static inline VectorType multiplyAndAdd(const VectorType &v1, const VectorType &v2, const VectorType &v3, const __mmask &k) { return _mm512_mask_madd132_pd(v1, k, v3, v2); }
            static inline VectorType multiplyAndSub(const VectorType &v1, const VectorType &v2, const VectorType &v3) { return _mm512_msub132_pd(v1, v3, v2); }

            static inline EntryType reduce_max(const VectorType &a) { return _mm512_reduce_max_pd(a); }
            static inline EntryType reduce_min(const VectorType &a) { return _mm512_reduce_min_pd(a); }
            static inline EntryType reduce_mul(const VectorType &a) { return _mm512_reduce_mul_pd(a); }
            static inline EntryType reduce_add(const VectorType &a) { return _mm512_reduce_add_pd(a); }

            static inline VectorType abs(VectorType a) {
                const _M512I absMask = _mm512_set_4to16_pi(0xffffffff, 0x7fffffff, 0xffffffff, 0x7fffffff);
                return mm512_reinterpret_cast<VectorType>(_mm512_and_pq(mm512_reinterpret_cast<_M512I>(a), absMask));
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
            OPcmpQ(eq) OPcmpQ(neq)
            OPcmpQ(lt) OPcmpQ(nlt)
            OPcmpQ(le) OPcmpQ(nle)
            static inline __mmask isNaN(VectorType x) {
                return CAT(_mm512_cmpunord_, SUFFIX)(x, x);
            }
            static inline __mmask isFinite(VectorType x) {
                return CAT(_mm512_cmpord_, SUFFIX)(x, mul(zero(), x));
            }
#undef SUFFIX
            static inline VectorType round(VectorType x) {
                return _mm512_cvtl_pi2pd(_mm512_cvtl_pd2pi(_mm512_setzero_pi(), x, _MM_ROUND_MODE_NEAREST));
            }
        };

#define LOAD(T, conv) \
            static inline VectorType load1         (const T  x) { return mm512_reinterpret_cast<VectorType>(FixedIntrinsics::_mm512_loadd(&x, conv, _MM_BROADCAST_1X16 , _MM_HINT_NONE)); } \
            static inline VectorType load4         (const T *x) { return mm512_reinterpret_cast<VectorType>(FixedIntrinsics::_mm512_loadd( x, conv, _MM_BROADCAST_4X16 , _MM_HINT_NONE)); } \
            static inline VectorType load          (const T *x) { return mm512_reinterpret_cast<VectorType>(FixedIntrinsics::_mm512_loadd( x, conv, _MM_BROADCAST_16X16, _MM_HINT_NONE)); } \
            static inline VectorType load1Streaming(const T  x) { return mm512_reinterpret_cast<VectorType>(FixedIntrinsics::_mm512_loadd(&x, conv, _MM_BROADCAST_1X16 , _MM_HINT_NT  )); } \
            static inline VectorType load4Streaming(const T *x) { return mm512_reinterpret_cast<VectorType>(FixedIntrinsics::_mm512_loadd( x, conv, _MM_BROADCAST_4X16 , _MM_HINT_NT  )); } \
            static inline VectorType loadStreaming (const T *x) { return mm512_reinterpret_cast<VectorType>(FixedIntrinsics::_mm512_loadd( x, conv, _MM_BROADCAST_16X16, _MM_HINT_NT  )); }

#define GATHERSCATTER(T, upconv, downconv) \
            static inline VectorType gather(_M512I indexes, const T *baseAddr) { \
                return mm512_reinterpret_cast<VectorType>(_mm512_gatherd(indexes, const_cast<T *>(baseAddr), upconv, \
                            sizeof(T) == 4 ? _MM_SCALE_4 : (sizeof(T) == 2 ? _MM_SCALE_2 : _MM_SCALE_1), _MM_HINT_NONE \
                            )); \
            } \
            static inline void gather(VectorType &data, _M512I indexes, const T *baseAddr, __mmask k) { \
                data = mm512_reinterpret_cast<VectorType>(_mm512_mask_gatherd(mm512_reinterpret_cast<_M512>(data), k, indexes, const_cast<T *>(baseAddr), upconv, \
                        sizeof(T) == 4 ? _MM_SCALE_4 : (sizeof(T) == 2 ? _MM_SCALE_2 : _MM_SCALE_1), _MM_HINT_NONE \
                        )); \
            } \
            static inline void gatherScale1(VectorType &data, _M512I indexes, const T *baseAddr, __mmask k) { \
                data = mm512_reinterpret_cast<VectorType>(_mm512_mask_gatherd(mm512_reinterpret_cast<_M512>(data), k, indexes, const_cast<T *>(baseAddr), upconv, \
                            _MM_SCALE_1, _MM_HINT_NONE \
                        )); \
            } \
            static inline VectorType gatherStreaming(_M512I indexes, const T *baseAddr) { \
                return mm512_reinterpret_cast<VectorType>(_mm512_gatherd(indexes, const_cast<T *>(baseAddr), upconv, \
                            sizeof(T) == 4 ? _MM_SCALE_4 : (sizeof(T) == 2 ? _MM_SCALE_2 : _MM_SCALE_1), _MM_HINT_NT \
                            )); \
            } \
            static inline void gatherStreaming(VectorType &data, _M512I indexes, const T *baseAddr, __mmask k) { \
                data = mm512_reinterpret_cast<VectorType>(_mm512_mask_gatherd(mm512_reinterpret_cast<_M512>(data), k, indexes, const_cast<T *>(baseAddr), upconv, \
                        sizeof(T) == 4 ? _MM_SCALE_4 : (sizeof(T) == 2 ? _MM_SCALE_2 : _MM_SCALE_1), _MM_HINT_NT \
                        )); \
            } \
            static inline void scatter(const VectorType data, _M512I indexes, T *baseAddr) { \
                _mm512_scatterd(baseAddr, indexes, mm512_reinterpret_cast<_M512>(data), downconv, \
                        sizeof(T) == 4 ? _MM_SCALE_4 : (sizeof(T) == 2 ? _MM_SCALE_2 : _MM_SCALE_1), _MM_HINT_NONE \
                        ); \
            } \
            static inline void scatter(const VectorType data, _M512I indexes, T *baseAddr, __mmask k) { \
                _mm512_mask_scatterd(baseAddr, k, indexes, mm512_reinterpret_cast<_M512>(data), downconv, \
                        sizeof(T) == 4 ? _MM_SCALE_4 : (sizeof(T) == 2 ? _MM_SCALE_2 : _MM_SCALE_1), _MM_HINT_NONE \
                        ); \
            } \
            static inline void scatterStreaming(const VectorType data, _M512I indexes, T *baseAddr) { \
                _mm512_scatterd(baseAddr, indexes, mm512_reinterpret_cast<_M512>(data), downconv, \
                        sizeof(T) == 4 ? _MM_SCALE_4 : (sizeof(T) == 2 ? _MM_SCALE_2 : _MM_SCALE_1), _MM_HINT_NT \
                        ); \
            } \
            static inline void scatterStreaming(const VectorType data, _M512I indexes, T *baseAddr, __mmask k) { \
                _mm512_mask_scatterd(baseAddr, k, indexes, mm512_reinterpret_cast<_M512>(data), downconv, \
                        sizeof(T) == 4 ? _MM_SCALE_4 : (sizeof(T) == 2 ? _MM_SCALE_2 : _MM_SCALE_1), _MM_HINT_NT \
                        ); \
            }

#define STORE(T, conv) \
            static inline void store1         (T *mem, VectorType x) { _mm512_stored(mem, mm512_reinterpret_cast<_M512>(x), conv, _MM_SUBSET32_1 , _MM_HINT_NONE); } \
            static inline void store4         (T *mem, VectorType x) { _mm512_stored(mem, mm512_reinterpret_cast<_M512>(x), conv, _MM_SUBSET32_4 , _MM_HINT_NONE); } \
            static inline void store          (T *mem, VectorType x) { _mm512_stored(mem, mm512_reinterpret_cast<_M512>(x), conv, _MM_SUBSET32_16, _MM_HINT_NONE); } \
            static inline void store1Streaming(T *mem, VectorType x) { _mm512_stored(mem, mm512_reinterpret_cast<_M512>(x), conv, _MM_SUBSET32_1 , _MM_HINT_NT  ); } \
            static inline void store4Streaming(T *mem, VectorType x) { _mm512_stored(mem, mm512_reinterpret_cast<_M512>(x), conv, _MM_SUBSET32_4 , _MM_HINT_NT  ); } \
            static inline void storeStreaming (T *mem, VectorType x) { _mm512_stored(mem, mm512_reinterpret_cast<_M512>(x), conv, _MM_SUBSET32_16, _MM_HINT_NT  ); } \
            static inline void store1         (T *mem, VectorType x, __mmask k) { _mm512_mask_stored(mem, k, mm512_reinterpret_cast<_M512>(x), conv, _MM_SUBSET32_1 , _MM_HINT_NONE); } \
            static inline void store4         (T *mem, VectorType x, __mmask k) { _mm512_mask_stored(mem, k, mm512_reinterpret_cast<_M512>(x), conv, _MM_SUBSET32_4 , _MM_HINT_NONE); } \
            static inline void store          (T *mem, VectorType x, __mmask k) { _mm512_mask_stored(mem, k, mm512_reinterpret_cast<_M512>(x), conv, _MM_SUBSET32_16, _MM_HINT_NONE); } \
            static inline void store1Streaming(T *mem, VectorType x, __mmask k) { _mm512_mask_stored(mem, k, mm512_reinterpret_cast<_M512>(x), conv, _MM_SUBSET32_1 , _MM_HINT_NT  ); } \
            static inline void store4Streaming(T *mem, VectorType x, __mmask k) { _mm512_mask_stored(mem, k, mm512_reinterpret_cast<_M512>(x), conv, _MM_SUBSET32_4 , _MM_HINT_NT  ); } \
            static inline void storeStreaming (T *mem, VectorType x, __mmask k) { _mm512_mask_stored(mem, k, mm512_reinterpret_cast<_M512>(x), conv, _MM_SUBSET32_16, _MM_HINT_NT  ); }

        template<> struct VectorHelper<float> {
            typedef float EntryType;
            typedef _M512 VectorType;
#define SUFFIX ps

            LOAD(EntryType, _MM_FULLUPC_NONE)
            LOAD(float16, _MM_FULLUPC_FLOAT16)
            LOAD(unsigned char, _MM_FULLUPC_UINT8)
            LOAD(signed char, _MM_FULLUPC_SINT8)
            LOAD(char, _MM_FULLUPC_SINT8)
            LOAD(unsigned short, _MM_FULLUPC_UINT16)
            LOAD(signed short, _MM_FULLUPC_SINT16)

            static inline VectorType load1(const float11_11_10  x, float11_11_10::Component c) { return FixedIntrinsics::_mm512_loadd(&x, static_cast<_MM_FULLUP32_ENUM>(c), _MM_BROADCAST_1X16);  }
            static inline VectorType load4(const float11_11_10 *x, float11_11_10::Component c) { return FixedIntrinsics::_mm512_loadd( x, static_cast<_MM_FULLUP32_ENUM>(c), _MM_BROADCAST_4X16);  }
            static inline VectorType load (const float11_11_10 *x, float11_11_10::Component c) { return FixedIntrinsics::_mm512_loadd( x, static_cast<_MM_FULLUP32_ENUM>(c), _MM_BROADCAST_16X16); }

            STORE(EntryType, _MM_DOWNC_NONE)
            STORE(float16, _MM_DOWNC_FLOAT16)
            STORE(unsigned char, _MM_DOWNC_UINT8)
            STORE(signed char, _MM_DOWNC_SINT8)
            STORE(unsigned short, _MM_DOWNC_UINT16)
            STORE(signed short, _MM_DOWNC_SINT16)

            static inline VectorType zero() { return CAT(_mm512_setzero_, SUFFIX)(); }
            static inline VectorType set(EntryType x) { return CAT(_mm512_set_1to16_, SUFFIX)(x); }

            GATHERSCATTER(EntryType,      _MM_FULLUPC_NONE,    _MM_DOWNC_NONE   )
            GATHERSCATTER(float16,        _MM_FULLUPC_FLOAT16, _MM_DOWNC_FLOAT16)
            GATHERSCATTER(unsigned char,  _MM_FULLUPC_UINT8,   _MM_DOWNC_UINT8  )
            GATHERSCATTER(signed char,    _MM_FULLUPC_SINT8,   _MM_DOWNC_SINT8  )
            GATHERSCATTER(unsigned short, _MM_FULLUPC_UINT16,  _MM_DOWNC_UINT16 )
            GATHERSCATTER(signed short,   _MM_FULLUPC_SINT16,  _MM_DOWNC_SINT16 )

            static inline VectorType multiplyAndAdd(const VectorType &v1, const VectorType &v2, const VectorType &v3) { return _mm512_madd132_ps(v1, v3, v2); }
            static inline VectorType multiplyAndAdd(const VectorType &v1, const VectorType &v2, const VectorType &v3, const __mmask &k) { return _mm512_mask_madd132_ps(v1, k, v3, v2); }
            static inline VectorType multiplyAndSub(const VectorType &v1, const VectorType &v2, const VectorType &v3) { return _mm512_msub132_ps(v1, v3, v2); }

            static inline EntryType reduce_max(const VectorType &a) { return _mm512_reduce_max_ps(a); }
            static inline EntryType reduce_min(const VectorType &a) { return _mm512_reduce_min_ps(a); }
            static inline EntryType reduce_mul(const VectorType &a) { return _mm512_reduce_mul_ps(a); }
            static inline EntryType reduce_add(const VectorType &a) { return _mm512_reduce_add_ps(a); }

            static inline VectorType abs(VectorType a) {
                const _M512I absMask = _mm512_set_1to16_pi(0x7fffffff);
                return mm512_reinterpret_cast<VectorType>(_mm512_and_pi(mm512_reinterpret_cast<_M512I>(a), absMask));
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
            OPcmp(eq) OPcmp(neq)
            OPcmp(lt) OPcmp(nlt)
            OPcmp(le) OPcmp(nle)
            static inline __mmask isNaN(VectorType x) {
                return CAT(_mm512_cmpunord_, SUFFIX)(x, x);
            }
            static inline __mmask isFinite(VectorType x) {
                return CAT(_mm512_cmpord_, SUFFIX)(x, mul(zero(), x));
            }
#undef SUFFIX
            static inline VectorType round(VectorType x) {
                return _mm512_round_ps(x, _MM_ROUND_MODE_NEAREST, _MM_EXPADJ_NONE);
            }
        };

        template<> struct VectorHelper<int> {
            typedef int EntryType;
            typedef _M512I VectorType;
#define SUFFIX pi
            LOAD(EntryType, _MM_FULLUPC_NONE)
            LOAD(signed char, _MM_FULLUPC_SINT8I)
            LOAD(signed short, _MM_FULLUPC_SINT16I)

            STORE(EntryType, _MM_DOWNC_NONE)
            STORE(signed char, _MM_DOWNC_SINT8I)
            STORE(signed short, _MM_DOWNC_SINT16I)

            static inline VectorType set(EntryType x) { return CAT(_mm512_set_1to16_, SUFFIX)(x); }

            GATHERSCATTER(EntryType,    _MM_FULLUPC_NONE,    _MM_DOWNC_NONE)
            GATHERSCATTER(signed char,  _MM_FULLUPC_SINT8I,  _MM_DOWNC_SINT8I)
            GATHERSCATTER(signed short, _MM_FULLUPC_SINT16I, _MM_DOWNC_SINT16I)

            static inline VectorType multiplyAndAdd(const VectorType &v1, const VectorType &v2, const VectorType &v3) { return _mm512_madd231_pi(v3, v1, v2); }
            static inline VectorType multiplyAndAdd(const VectorType &v1, const VectorType &v2, const VectorType &v3, const __mmask &k) { return _mm512_mask_madd231_pi(v1, k, v3, v2); }
            static inline VectorType multiplyAndSub(const VectorType &v1, const VectorType &v2, const VectorType &v3) { return _mm512_sub_pi(_mm512_mull_pi(v1, v2), v3); }

            static inline EntryType reduce_max(const VectorType &a) { return _mm512_reduce_max_pi(a); }
            static inline EntryType reduce_min(const VectorType &a) { return _mm512_reduce_min_pi(a); }
            static inline EntryType reduce_mul(const VectorType &a) { return _mm512_reduce_mul_pi(a); }
            static inline EntryType reduce_add(const VectorType &a) { return _mm512_reduce_add_pi(a); }

            static inline VectorType abs(VectorType a) {
                VectorType zero = mm512_reinterpret_cast<VectorType>(_mm512_setzero());
                const VectorType minusOne = _mm512_set_1to16_pi( -1 );
                return mul(a, minusOne, cmplt(a, zero), a);
            }

            OP(max) OP(min)
            OP(add) OP(sub) OPx(mul, mull) OP(div) OP(rem)
            OP_(or_) OP_(and_) OP_(xor_)
            OPcmp(eq) OPcmp(neq)
            OPcmp(lt) OPcmp(nlt)
            OPcmp(le) OPcmp(nle)
            OP(sll) OP(srl)
#undef SUFFIX
            static inline VectorType round(VectorType x) { return x; }
        };

        template<> struct VectorHelper<unsigned int> {
            typedef unsigned int EntryType;
            typedef _M512I VectorType;
#define SUFFIX pu
            LOAD(EntryType, _MM_FULLUPC_NONE)
            LOAD(unsigned char, _MM_FULLUPC_UINT8I)
            LOAD(unsigned short, _MM_FULLUPC_UINT16I)

            STORE(EntryType, _MM_DOWNC_NONE)
            STORE(unsigned char, _MM_DOWNC_UINT8I)
            STORE(unsigned short, _MM_DOWNC_UINT16I)

            GATHERSCATTER(EntryType,      _MM_FULLUPC_NONE,    _MM_DOWNC_NONE)
            GATHERSCATTER(unsigned char,  _MM_FULLUPC_UINT8I,  _MM_DOWNC_UINT8I)
            GATHERSCATTER(unsigned short, _MM_FULLUPC_UINT16I, _MM_DOWNC_UINT16I)

            static inline VectorType multiplyAndAdd(const VectorType &v1, const VectorType &v2, const VectorType &v3) { return _mm512_madd231_pi(v3, v1, v2); }
            static inline VectorType multiplyAndAdd(const VectorType &v1, const VectorType &v2, const VectorType &v3, const __mmask &k) { return _mm512_mask_madd231_pi(v1, k, v3, v2); }
            static inline VectorType multiplyAndSub(const VectorType &v1, const VectorType &v2, const VectorType &v3) { return _mm512_sub_pi(_mm512_mull_pi(v1, v2), v3); }

            static inline EntryType reduce_max(const VectorType &a) { return _mm512_reduce_max_pi(a); }
            static inline EntryType reduce_min(const VectorType &a) { return _mm512_reduce_min_pi(a); }
            static inline EntryType reduce_mul(const VectorType &a) { return _mm512_reduce_mul_pi(a); }
            static inline EntryType reduce_add(const VectorType &a) { return _mm512_reduce_add_pi(a); }

            OP(max) OP(min)
            OP(div) OP(rem)
            OPcmp(eq) OPcmp(neq)
            OPcmp(lt) OPcmp(nlt)
            OPcmp(le) OPcmp(nle)
#undef SUFFIX
#define SUFFIX pi
            static inline VectorType set(EntryType x) { return CAT(_mm512_set_1to16_, SUFFIX)(static_cast<int>(x)); }

            OP(sll) OP(srl)
            OP(add) OP(sub) OPx(mul, mull)
            OP_(or_) OP_(and_) OP_(xor_)
#undef GATHERSCATTER
#undef STORE
#undef LOAD
#undef SUFFIX
            static inline VectorType round(VectorType x) { return x; }
        };
#undef OP
#undef OP_
#undef OPx
#undef OPcmp
    } // anonymous namespace

    template<typename T> class _Memory : public VectorAlignedBase
    {
        private:
            enum { Size = 64 / sizeof(T) };
            T d[Size];
        public:
            inline int size() const { return Size; }
            inline T &operator[](int i) { return d[i]; }
            inline T operator[](int i) const { return d[i]; }
            inline operator T*() { return &d[0]; }
            inline operator const T*() const { return &d[0]; }

            inline _Memory<T> &operator=(const _Memory<T> &rhs) {
                typedef typename VectorHelper<T>::VectorType VectorType;
                const VectorType tmp = VectorHelper<T>::load(&rhs.d[0]);
                VectorHelper<T>::store(&d[0], tmp);
                return *this;
            }
            inline _Memory<T> &operator=(const Vector<T> &rhs) {
                VectorHelper<T>::store(&d[0], rhs);
                return *this;
            }
    };

template<typename T>
class VectorMultiplication
{
    friend class Vector<T>;
    public:
        inline T operator[](int index) const { return Vector<T>(product())[index]; }

        inline VectorMultiplication<T> operator*(const Vector<T> &x) const { return VectorMultiplication<T>(product(), x.data); }
        inline Vector<T> operator+(const Vector<T> &x) const { return VectorHelper<T>::multiplyAndAdd(left, right, x.data); }
        inline Vector<T> operator-(const Vector<T> &x) const { return VectorHelper<T>::multiplyAndSub(left, right, x.data); }
        inline Vector<T> operator/(const Vector<T> &x) const { return Vector<T>(product()) / x; }
        inline Vector<T> operator%(const Vector<T> &x) const { return Vector<T>(product()) % x; }
        inline Vector<T> operator|(const Vector<T> &x) const { return Vector<T>(product()) | x; }
        inline Vector<T> operator&(const Vector<T> &x) const { return Vector<T>(product()) & x; }
        inline Vector<T> operator^(const Vector<T> &x) const { return Vector<T>(product()) ^ x; }
        inline typename Vector<T>::Mask operator==(const Vector<T> &x) const { return Vector<T>(product()) == x; }
        inline typename Vector<T>::Mask operator!=(const Vector<T> &x) const { return Vector<T>(product()) != x; }
        inline typename Vector<T>::Mask operator>=(const Vector<T> &x) const { return Vector<T>(product()) >= x; }
        inline typename Vector<T>::Mask operator<=(const Vector<T> &x) const { return Vector<T>(product()) <= x; }
        inline typename Vector<T>::Mask operator> (const Vector<T> &x) const { return Vector<T>(product()) >  x; }
        inline typename Vector<T>::Mask operator< (const Vector<T> &x) const { return Vector<T>(product()) <  x; }

        inline operator Vector<T>() const { return product(); }

    private:
        typedef typename VectorHelper<T>::VectorType VectorType;

        VectorMultiplication(const VectorType &a, const VectorType &b) : left(a), right(b) {}

        const VectorType &left;
        const VectorType &right;

        VectorType product() const { return VectorHelper<T>::mul(left, right); }
};

template<typename T> inline Vector<T> operator+(const Vector<T> &x, const VectorMultiplication<T> &y) {
    return VectorHelper<T>::multiplyAndAdd(y.left, y.right, x.data);
}
template<typename T> inline Vector<T> operator-(const Vector<T> &x, const VectorMultiplication<T> &y) {
    return VectorHelper<T>::multiplyAndSub(y.left, y.right, x.data);
}

template<typename T>
class WriteMaskedVector
{
    friend class Vector<T>;
    typedef LRBni::Mask<64 / sizeof(T)> Mask;
    public:
        //prefix
        inline Vector<T> &operator++() {
            vec->data = VectorHelper<T>::add(vec->data, VectorHelper<T>::set(static_cast<T>(1)), mask, vec->data);
            return *vec;
        }
        inline Vector<T> &operator--() {
            vec->data = VectorHelper<T>::sub(vec->data, VectorHelper<T>::set(static_cast<T>(1)), mask, vec->data);
            return *vec;
        }
        //postfix
        inline Vector<T> operator++(int) {
            Vector<T> ret(*vec);
            vec->data = VectorHelper<T>::add(vec->data, VectorHelper<T>::set(static_cast<T>(1)), mask, vec->data);
            return ret;
        }
        inline Vector<T> operator--(int) {
            Vector<T> ret(*vec);
            vec->data = VectorHelper<T>::sub(vec->data, VectorHelper<T>::set(static_cast<T>(1)), mask, vec->data);
            return ret;
        }

        inline Vector<T> &operator+=(Vector<T> x) {
            vec->data = VectorHelper<T>::add(vec->data, x.data, mask, vec->data);
            return *vec;
        }
        inline Vector<T> &operator-=(Vector<T> x) {
            vec->data = VectorHelper<T>::sub(vec->data, x.data, mask, vec->data);
            return *vec;
        }
        inline Vector<T> &operator*=(Vector<T> x) {
            vec->data = VectorHelper<T>::mul(vec->data, x.data, mask, vec->data);
            return *vec;
        }
        inline Vector<T> &operator/=(Vector<T> x) {
            vec->data = VectorHelper<T>::div(vec->data, x.data, mask, vec->data);
            return *vec;
        }

        inline Vector<T> &operator=(Vector<T> x) {
            vec->assign(x, mask);
            return *vec;
        }
    private:
        WriteMaskedVector(Vector<T> *v, Mask k) : vec(v), mask(k.data()) {}
        Vector<T> *vec;
        __mmask mask;
};

template<typename T>
class Vector : public VectorBase<T, Vector<T> >
{
    friend struct VectorBase<T, Vector<T> >;
    friend class VectorMultiplication<T>;
    friend class WriteMaskedVector<T>;
    friend class Vector<float>;
    friend class Vector<double>;
    friend class Vector<int>;
    friend class Vector<unsigned int>;
    protected:
        typedef typename VectorHelper<T>::VectorType VectorType;
        VectorType data;
    public:
        typedef T EntryType;
        typedef Vector<unsigned int> IndexType;
        typedef _Memory<T> Memory;

        enum { Size = 64 / sizeof(T) };
        typedef LRBni::Mask<Size> Mask;

        /**
         * Reinterpret some array of T as a vector of T. You may only do this if the pointer is
         * aligned correctly and the content of the memory isn't changed from somewhere else because
         * the load operation will happen implicitly at some later point(s).
         */
        static inline Vector fromMemory(T *mem) {
            assert(0 == (mem & (VectorAlignment - 1)));
            return reinterpret_cast<Vector<T> >(mem);
        }

        /**
         * uninitialized
         */
        inline Vector() {}

        /**
         * initialized to 0 in all 512 bits
         */
        inline explicit Vector(VectorSpecialInitializerZero::ZEnum) : data(mm512_reinterpret_cast<VectorType>(_mm512_setzero())) {}
        /**
         * initialized to 1 in all vector entries
         */
        inline explicit Vector(VectorSpecialInitializerOne::OEnum) : data(VectorHelper<T>::set(EntryType(1))) {}
        /**
         * initialized to 0, 1, 2, 3 (, 4, 5, 6, 7 (, 8, 9, 10, 11, 12, 13, 14, 15))
         */
        inline explicit Vector(VectorSpecialInitializerIndexesFromZero::IEnum) : data(VectorHelper<T>::load(IndexesFromZero<T>())) {}
//X         /**
//X          * initialzed to random numbers
//X          */
//X         inline explicit Vector(VectorSpecialInitializerRandom::Enum) { makeRandom(); }
        /**
         * initialize with given __m512 vector
         */
        inline Vector(VectorType x) : data(x) {}
        template<typename OtherT>
        explicit inline Vector(const Vector<OtherT> &x) : data(StaticCastHelper<OtherT, T>::cast(x.data)) {}
        /**
         * initialize all 16 or 8 values with the given value
         */
        inline Vector(T a) : data(VectorHelper<T>::load1(a)) {}
        /**
         * initialize consecutive four vector entries with the given values
         */
        template<typename Other>
        inline Vector(Other a, Other b, Other c, Other d)
        {
            LRB_ALIGN(64) const Other x[4] = {
                a, b, c, d
            };
            data = VectorHelper<T>::load4(x);
        }
        inline explicit Vector(const T *x) : data(VectorHelper<T>::load(x)) {}
        /**
         * Initialize the vector with the given data. \param x must point to 64 byte aligned 512
         * byte data.
         */
        template<typename Other> inline explicit Vector(const Other *x) : data(VectorHelper<T>::load(x)) {}

        // TODO: handle 8 <-> 16 conversions
        inline explicit Vector(const Vector *x) : data(x->data) {}
        inline void expand(Vector *x) const { x->data = data; }

        template<typename Other> static inline Vector broadcast4(const Other *x) { return Vector<T>(VectorHelper<T>::load4(x)); }

        template<typename Other> inline void load(const Other *x) { data = VectorHelper<T>::load(x); }

        inline void makeZero() { data = mm512_reinterpret_cast<VectorType>(_mm512_setzero()); }

        inline void makeZero(Mask k)
        {
            if (Size == 16) {
                _M512I tmp = mm512_reinterpret_cast<_M512I>(data);
                data = mm512_reinterpret_cast<VectorType>(VectorHelper<int>::xor_(tmp, tmp, k.data()));
            } else if (Size == 8) {
                VectorDQHelper<T>::mov(data, k.data(), mm512_reinterpret_cast<VectorType>(_mm512_setzero()));
            }
        }

//X         inline void makeRandom()
//X         {
//X             data = VectorHelper<int>::random();
//X         }
//X 
//X         inline void makeRandom(Mask k)
//X         {
//X             data = VectorHelper<int>::random(k.data());
//X         }

        /**
         * Store the vector data to the given memory. The memory must be 64 byte aligned and of 512
         * bytes size.
         */
        template<typename OtherT>
        inline void store(OtherT *mem) const
        {
            VectorHelper<T>::store(mem, data);
        }

        template<typename OtherT>
        inline void store(OtherT *mem, Mask mask) const
        {
            VectorHelper<T>::store(mem, data, mask.data());
        }

        inline void store(T *mem) const
        {
            VectorHelper<T>::store(mem, data);
        }

        inline void store(T *mem, Mask mask) const
        {
            VectorHelper<T>::store(mem, data, mask.data());
        }

        /**
         * Non-temporal store variant. Writes to the memory without polluting the cache.
         */
        template<typename OtherT>
        inline void storeStreaming(OtherT *mem) const
        {
            VectorHelper<T>::storeStreaming(mem, data);
        }

        inline void storeStreaming(T *mem) const
        {
            VectorHelper<T>::storeStreaming(mem, data);
        }

        inline const Vector<T> &dcba() const { return *this; }
        inline const SwizzledVector<T> cdab() const { const SwizzledVector<T> sv = { *this, _MM_SWIZ_REG_CDAB }; return sv; }
        inline const SwizzledVector<T> badc() const { const SwizzledVector<T> sv = { *this, _MM_SWIZ_REG_BADC }; return sv; }
        inline const SwizzledVector<T> aaaa() const { const SwizzledVector<T> sv = { *this, _MM_SWIZ_REG_AAAA }; return sv; }
        inline const SwizzledVector<T> bbbb() const { const SwizzledVector<T> sv = { *this, _MM_SWIZ_REG_BBBB }; return sv; }
        inline const SwizzledVector<T> cccc() const { const SwizzledVector<T> sv = { *this, _MM_SWIZ_REG_CCCC }; return sv; }
        inline const SwizzledVector<T> dddd() const { const SwizzledVector<T> sv = { *this, _MM_SWIZ_REG_DDDD }; return sv; }
        inline const SwizzledVector<T> dacb() const { const SwizzledVector<T> sv = { *this, _MM_SWIZ_REG_DACB }; return sv; }

        inline Vector(const T *array, const IndexType &indexes)
            : data(VectorHelper<T>::gather(sizeof(T) == 8 ? IndexType(indexes * 2) : indexes, array)) {}

        inline Vector(const T *array, const IndexType &indexes, Mask mask) {
            VectorHelper<T>::gather(data, sizeof(T) == 8 ? IndexType(indexes * 2) : indexes, array, mask.data());
        }
        inline Vector(const T *array, const IndexType &indexes, Mask mask, VectorSpecialInitializerZero::ZEnum)
            : data(mm512_reinterpret_cast<VectorType>(_mm512_setzero()))
        {
            VectorHelper<T>::gather(data, sizeof(T) == 8 ? IndexType(indexes * 2) : indexes, array, mask.data());
        }
        inline Vector(const T *array, const IndexType &indexes, Mask mask, EntryType def)
            : data(VectorHelper<T>::load1(def))
        {
            VectorHelper<T>::gather(data, sizeof(T) == 8 ? IndexType(indexes * 2) : indexes, array, mask.data());
        }

        inline void gather(const T *array, const IndexType &indexes) {
            data = VectorHelper<T>::gather(sizeof(T) == 8 ? IndexType(indexes * 2) : indexes, array);
        }

        inline void gather(const T *array, const IndexType &indexes, Mask mask) {
            VectorHelper<T>::gather(data, sizeof(T) == 8 ? IndexType(indexes * 2) : indexes, array, mask.data());
        }

        inline void scatter(T *array, const IndexType &indexes) const {
            VectorHelper<T>::scatter(data, sizeof(T) == 8 ? IndexType(indexes * 2) : indexes, array);
        }
        inline void scatter(T *array, const IndexType &indexes, Mask mask) const {
            VectorHelper<T>::scatter(data, sizeof(T) == 8 ? IndexType(indexes * 2) : indexes, array, mask.data());
        }

        /**
         * \param array An array of objects where one member should be gathered
         * \param member A member pointer to the member of the class/struct that should be gathered
         * \param indexes The indexes in the array. The correct offsets are calculated
         *                automatically.
         * \param mask Optional mask to select only parts of the vector that should be gathered
         */
        template<typename S, typename OtherT>
        inline Vector(const S *array, const OtherT S::* member1, const IndexType &indexes, Mask mask = Mask(VectorSpecialInitializerOne::One))
        {
            enum { Scale = sizeof(OtherT) == 8 ? 4 : sizeof(OtherT) };
            LRB_STATIC_ASSERT((sizeof(S) % Scale) == 0, Struct_size_needs_to_be_a_multiple_of_the_gathered_member_size);
            const IndexType &offsets = indexes * (sizeof(S) / Scale);
            VectorHelper<OtherT>::gather(data, offsets, &(array->*(member1)), mask.data());
        }

        template<typename S1, typename S2, typename OtherT>
        inline Vector(const S1 *array, const S2 S1::* member1, const OtherT S2::* member2, const IndexType &indexes, Mask mask = Mask(VectorSpecialInitializerOne::One))
        {
            enum { Scale = sizeof(OtherT) == 8 ? 4 : sizeof(OtherT) };
            LRB_STATIC_ASSERT((sizeof(S1) % Scale) == 0, Struct_size_needs_to_be_a_multiple_of_the_gathered_member_size);
            const IndexType &offsets = indexes * (sizeof(S1) / Scale);
            VectorHelper<OtherT>::gather(data, offsets, &(array->*(member1).*(member2)), mask.data());
        }

        template<typename S, typename OtherT>
        inline void gather(const S *array, const OtherT S::* member1, const IndexType &indexes, Mask mask = Mask(VectorSpecialInitializerOne::One))
        {
            enum { Scale = sizeof(OtherT) == 8 ? 4 : sizeof(OtherT) };
            LRB_STATIC_ASSERT((sizeof(S) % Scale) == 0, Struct_size_needs_to_be_a_multiple_of_the_gathered_member_size);
            const IndexType &offsets = indexes * (sizeof(S) / Scale);
            VectorHelper<OtherT>::gather(data, offsets, &(array->*(member1)), mask.data());
        }

        template<typename S1, typename S2, typename OtherT>
        inline void gather(const S1 *array, const S2 S1::* member1, const OtherT S2::* member2, const IndexType &indexes, Mask mask = Mask(VectorSpecialInitializerOne::One))
        {
            enum { Scale = sizeof(OtherT) == 8 ? 4 : sizeof(OtherT) };
            LRB_STATIC_ASSERT((sizeof(S1) % Scale) == 0, Struct_size_needs_to_be_a_multiple_of_the_gathered_member_size);
            const IndexType &offsets = indexes * (sizeof(S1) / Scale);
            VectorHelper<OtherT>::gather(data, offsets, &(array->*(member1).*(member2)), mask.data());
        }

        template<typename S1, typename OtherT>
        inline Vector(const S1 *array, const OtherT *const S1::* ptrMember1, const IndexType &outerIndex, const IndexType &innerIndex, Mask mask) {
            gather(array, ptrMember1, outerIndex, innerIndex, mask);
        }
        template<typename S1, typename OtherT>
        inline void gather(const S1 *array, const OtherT *const S1::* ptrMember1, const IndexType &outerIndex, const IndexType &innerIndex, Mask mask) {
            // FIXME there must be a nicer way to implement this
            enum {
                OuterStride = sizeof(S1) / 4
            };
            const int *const outerArray = reinterpret_cast<const int *>(&(array->*ptrMember1));// + (sizeof(void *) / sizeof(int) - 1);
            _M512I offsets = _mm512_setzero_pi();
            // bah, ugly hack:
            // gather the LSB of the pointers (this breaks if some point to the heap and others to
            // the stack, in that case the MSB differs (at least on Linux, dunno on FreeBSD))
            const _M512I index = _mm512_mull_pi(outerIndex, _mm512_set_1to16_pi(OuterStride));
            VectorHelper<int>::gather(offsets, index, outerArray, mask.data());
            // and calculate the offsets to (array[0].*ptrMember1)
            offsets = _mm512_sub_pi(offsets, _mm512_set_1to16_pi(outerArray[0]));
            offsets = _mm512_madd231_pi(offsets, innerIndex, _mm512_set_1to16_pi(sizeof(OtherT)));
            VectorHelper<T>::gatherScale1(data, offsets, array->*ptrMember1, mask.data());
        }

        template<typename S, typename OtherT>
        inline void scatter(S *array, OtherT S::* member1, const IndexType &indexes, Mask mask = Mask(VectorSpecialInitializerOne::One))
        {
            enum { Scale = sizeof(OtherT) == 8 ? 4 : sizeof(OtherT) };
            LRB_STATIC_ASSERT((sizeof(S) % Scale) == 0, Struct_size_needs_to_be_a_multiple_of_the_scattered_member_size);
            const IndexType &offsets = indexes * (sizeof(S) / Scale);
            VectorHelper<OtherT>::scatter(data, offsets, &(array->*(member1)), mask.data());
        }

        template<typename S1, typename S2, typename OtherT>
        inline void scatter(S1 *array, S2 S1::* member1, OtherT S2::* member2, const IndexType &indexes, Mask mask = Mask(VectorSpecialInitializerOne::One))
        {
            enum { Scale = sizeof(OtherT) == 8 ? 4 : sizeof(OtherT) };
            LRB_STATIC_ASSERT((sizeof(S1) % Scale) == 0, Struct_size_needs_to_be_a_multiple_of_the_scattered_member_size);
            const IndexType &offsets = indexes * (sizeof(S1) / Scale);
            VectorHelper<OtherT>::scatter(data, offsets, &(array->*(member1).*(member2)), mask.data());
        }

        //prefix
        inline Vector &operator++() { data = VectorHelper<T>::add(data, Vector<T>(1)); return *this; }
        //postfix
        inline Vector operator++(int) { const Vector<T> r = *this; data = VectorHelper<T>::add(data, Vector<T>(1)); return r; }
        inline void increment(Mask k) { data = VectorHelper<T>::add(data, Vector<T>(1), k.data()); }
        inline void decrement(Mask k) { data = VectorHelper<T>::sub(data, Vector<T>(1), k.data()); }

        inline T operator[](int index) const {
            union { VectorType v; T d[Size]; } u;
            VectorHelper<T>::store(u.d, data);
            return u.d[index];
        }

        inline VectorMultiplication<T> operator*(const Vector<T> &x) const { return VectorMultiplication<T>(data, x.data); }

        inline Vector &mul_eq(const SwizzledVector<T> &x, const Mask m) { data = VectorHelper<T>::mul_s(x.s, data, x.v.data, m); return *this; }
        inline Vector &mul_eq(const Vector<T> &x, const Mask m) { data = VectorHelper<T>::mul(data, x.data, m); return *this; }
        inline Vector &mul_eq(const Vector<T> &x, const Mask m, const Vector<T> &old) { data = VectorHelper<T>::mul(data, x.data, m, old.data); return *this; }
        inline Vector mul(const Vector<T> &x, const Mask m) const { return VectorHelper<T>::mul(data, x.data, m); }
        inline Vector mul(const Vector<T> &x, const Mask m, const Vector<T> &old) const { return VectorHelper<T>::mul(data, x.data, m, old.data); }
        inline Vector &operator*=(const Vector<T> &x) { data = VectorHelper<T>::mul(data, x.data); return *this; }

        inline Vector operator~() const { return mm512_reinterpret_cast<VectorType>(_mm512_andn_pi(mm512_reinterpret_cast<_M512I>(data), _mm512_setallone_pi())); }

#define OP(symbol, fun) \
        inline Vector &fun##_eq(const SwizzledVector<T> &x, const Mask m) { data = VectorHelper<T>::fun##_s(x.s, data, x.v.data, m); return *this; } \
        inline Vector &fun##_eq(const Vector<T> &x, const Mask m) { data = VectorHelper<T>::fun(data, x.data, m); return *this; } \
        inline Vector &fun##_eq(const Vector<T> &x, const Mask m, const Vector<T> &old) { data = VectorHelper<T>::fun(data, x.data, m, old.data); return *this; } \
        inline Vector fun(const Vector<T> &x, const Mask m) const { return VectorHelper<T>::fun(data, x.data, m); } \
        inline Vector fun(const Vector<T> &x, const Mask m, const Vector<T> &old) const { return VectorHelper<T>::fun(data, x.data, m, old.data); } \
        inline Vector &operator symbol##=(const Vector<T> &x) { data = VectorHelper<T>::fun(data, x.data); return *this; } \
        inline Vector operator symbol(const Vector<T> &x) const { return Vector<T>(VectorHelper<T>::fun(data, x.data)); }

        OP(+, add)
        OP(-, sub)
        OP(/, div)
        OP(%, rem)
        OP(|, or_)
        OP(&, and_)
        OP(^, xor_)
#undef OP
#define OPcmp(symbol, fun) \
        inline Mask fun(const Vector<T> &x, const Mask mask) const { return VectorHelper<T>::fun(data, x.data, mask.data()); } \
        inline Mask operator symbol(const Vector<T> &x) const { return VectorHelper<T>::fun(data, x.data); } \

        OPcmp(==, cmpeq)
        OPcmp(!=, cmpneq)
        OPcmp(>=, cmpnlt)
        OPcmp(>, cmpnle)
        OPcmp(<, cmplt)
        OPcmp(<=, cmple)
#undef OPcmp
#undef OPcmpQ

        inline void multiplyAndAdd(const Vector<T> &factor, const Vector<T> &summand) {
            VectorHelper<T>::multiplyAndAdd(data, factor, summand);
        }

        inline void multiplyAndAdd(const Vector<T> &factor, const Vector<T> &summand, Mask k) {
            VectorHelper<T>::multiplyAndAdd(data, factor, summand, k.data());
        }

        inline Vector multiplyAndAdd(const Vector<T> &factor, const Vector<T> &summand) const {
            Vector<T> r(*this);
            VectorHelper<T>::multiplyAndAdd(r.data, factor, summand);
            return r;
        }

        inline Vector multiplyAndAdd(const Vector<T> &factor, const Vector<T> &summand, Mask k) const {
            Vector<T> r(*this);
            VectorHelper<T>::multiplyAndAdd(r.data, factor, summand, k.data());
            return r;
        }

        inline void assign(const Vector<T> &v, const Mask &mask) {
            VectorDQHelper<T>::mov(data, mask.data(), v.data);
        }

        template<typename T2> inline Vector<T2> staticCast() const { return StaticCastHelper<T, T2>::cast(data); }
        template<typename T2> inline Vector<T2> reinterpretCast() const { return ReinterpretCastHelper<T, T2>::cast(data); }

        inline WriteMaskedVector<T> operator()(Mask k) { return WriteMaskedVector<T>(this, k); }

        inline T max() const { return VectorHelper<T>::reduce_max(data); }
        inline T min() const { return VectorHelper<T>::reduce_min(data); }
        inline T product() const { return VectorHelper<T>::reduce_mul(data); }
        inline T sum() const { return VectorHelper<T>::reduce_add(data); }

        template<typename F> void callWithValuesSorted(F &f) {
            union { VectorType v; EntryType d[Size]; } u;
            u.v = data; //VectorHelper<T>::store(u.d, data);

            EntryType value = u.d[0];
            f(value);
            for (int i = 1; i < Size; ++i) {
                if (u.d[i] != value) {
                    value = u.d[i];
                    f(value);
                }
            }
        }
};

template<typename T> inline Vector<T> operator+(const T &x, const Vector<T> &v) { return v.operator+(x); }
template<typename T> inline Vector<T> operator*(const T &x, const Vector<T> &v) { return v.operator*(x); }
template<typename T> inline Vector<T> operator-(const T &x, const Vector<T> &v) { return Vector<T>(x) - v; }
template<typename T> inline Vector<T> operator/(const T &x, const Vector<T> &v) { return Vector<T>(x) / v; }
template<typename T> inline Vector<T> operator%(const T &x, const Vector<T> &v) { return Vector<T>(x) % v; }
template<typename T> inline __mmask  operator< (const T &x, const Vector<T> &v) { return Vector<T>(x) <  v; }
template<typename T> inline __mmask  operator<=(const T &x, const Vector<T> &v) { return Vector<T>(x) <= v; }
template<typename T> inline __mmask  operator> (const T &x, const Vector<T> &v) { return Vector<T>(x) >  v; }
template<typename T> inline __mmask  operator>=(const T &x, const Vector<T> &v) { return Vector<T>(x) >= v; }
template<typename T> inline __mmask  operator==(const T &x, const Vector<T> &v) { return Vector<T>(x) == v; }
template<typename T> inline __mmask  operator!=(const T &x, const Vector<T> &v) { return Vector<T>(x) != v; }

template<typename T> struct SwizzledVector
{
    Vector<T> v;
    unsigned int s;
};

} // namespace LRBni

namespace LRBni
{
#define PARENT_DATA(T) (static_cast<Vector<T> *>(this)->data)
#define PARENT_DATA_CONST(T) (static_cast<const Vector<T> *>(this)->data)
#define OP_IMPL(EntryType, symbol, fun) \
        template<> inline Vector<EntryType> &VectorBase<EntryType, Vector<EntryType> >::fun##_eq(const Vector<EntryType> &x, const __mmask m) { PARENT_DATA(EntryType) = VectorHelper<EntryType>::fun(PARENT_DATA(EntryType), x.data, m); return *static_cast<Vector<EntryType> *>(this); } \
        template<> inline Vector<EntryType> &VectorBase<EntryType, Vector<EntryType> >::fun##_eq(const Vector<EntryType> &x, const __mmask m, const Vector<EntryType> &old) { PARENT_DATA(EntryType) = VectorHelper<EntryType>::fun(PARENT_DATA(EntryType), x.data, m, old.data); return *static_cast<Vector<EntryType> *>(this); } \
        template<> inline Vector<EntryType> &VectorBase<EntryType, Vector<EntryType> >::fun##_eq(const EntryType &x        , const __mmask m) { return fun##_eq(Vector<EntryType>(x), m); } \
        template<> inline Vector<EntryType> &VectorBase<EntryType, Vector<EntryType> >::fun##_eq(const EntryType &x        , const __mmask m, const Vector<EntryType> &old) { return fun##_eq(Vector<EntryType>(x), m, old); } \
        template<> inline Vector<EntryType>  VectorBase<EntryType, Vector<EntryType> >::fun     (const Vector<EntryType> &x, const __mmask m) const { return VectorHelper<EntryType>::fun(PARENT_DATA_CONST(EntryType), x.data, m); } \
        template<> inline Vector<EntryType>  VectorBase<EntryType, Vector<EntryType> >::fun     (const Vector<EntryType> &x, const __mmask m, const Vector<EntryType> &old) const { return VectorHelper<EntryType>::fun(PARENT_DATA_CONST(EntryType), x.data, m, old.data); } \
        template<> inline Vector<EntryType>  VectorBase<EntryType, Vector<EntryType> >::fun     (const EntryType &x        , const __mmask m) const { return fun(Vector<EntryType>(x), m); } \
        template<> inline Vector<EntryType>  VectorBase<EntryType, Vector<EntryType> >::fun     (const EntryType &x        , const __mmask m, const Vector<EntryType> &old) const { return fun(Vector<EntryType>(x), m, old); } \
        template<> inline Vector<EntryType> &VectorBase<EntryType, Vector<EntryType> >::operator symbol##=(const Vector<EntryType> &x) { PARENT_DATA(EntryType) = VectorHelper<EntryType>::fun(PARENT_DATA(EntryType), x.data); return *static_cast<Vector<EntryType> *>(this); } \
        template<> inline Vector<EntryType> &VectorBase<EntryType, Vector<EntryType> >::operator symbol##=(const EntryType &x) { return operator symbol##=(Vector<EntryType>(x)); } \
        template<> inline Vector<EntryType>  VectorBase<EntryType, Vector<EntryType> >::operator symbol(const Vector<EntryType> &x) const { return Vector<EntryType>(VectorHelper<EntryType>::fun(PARENT_DATA_CONST(EntryType), x.data)); } \
        template<> inline Vector<EntryType>  VectorBase<EntryType, Vector<EntryType> >::operator symbol(const EntryType &x) const { return operator symbol(Vector<EntryType>(x)); }
        OP_IMPL(int, &, and_)
        OP_IMPL(int, |, or_)
        OP_IMPL(int, ^, xor_)
        OP_IMPL(int, >>, srl)
        OP_IMPL(int, <<, sll)
        OP_IMPL(unsigned int, &, and_)
        OP_IMPL(unsigned int, |, or_)
        OP_IMPL(unsigned int, ^, xor_)
        OP_IMPL(unsigned int, >>, srl)
        OP_IMPL(unsigned int, <<, sll)
#undef PARENT_DATA_CONST
#undef PARENT_DATA
#undef OP_IMPL

#define MATH_OP1(name, call) \
    template<typename T> static inline LRBni::Vector<T> name(const LRBni::Vector<T> &x)               { return VectorHelper<T>::call(x); } \
    template<typename T> static inline LRBni::Vector<T> name(const LRBni::VectorMultiplication<T> &x) { return VectorHelper<T>::call(static_cast<Vector<T> >(x)); }
#define MATH_OP2(name, call) \
    template<typename T> static inline LRBni::Vector<T> name(const T &x, const LRBni::Vector<T> &y)   { return VectorHelper<T>::call(Vector<T>(x), y); } \
    template<typename T> static inline LRBni::Vector<T> name(const LRBni::Vector<T> &x, const T &y)   { return VectorHelper<T>::call(x, Vector<T>(y)); } \
    template<typename T> static inline LRBni::Vector<T> name(const LRBni::Vector<T> &x, const LRBni::Vector<T> &y)                             { return VectorHelper<T>::call(x, y); } \
    template<typename T> static inline LRBni::Vector<T> name(const LRBni::Vector<T> &x, const LRBni::VectorMultiplication<T> &y) { return VectorHelper<T>::call(x, static_cast<Vector<T> >(y)); } \
    template<typename T> static inline LRBni::Vector<T> name(const LRBni::VectorMultiplication<T> &x, const LRBni::Vector<T> &y) { return VectorHelper<T>::call(static_cast<Vector<T> >(x), y); } \
    template<typename T> static inline LRBni::Vector<T> name(const T &x, const LRBni::VectorMultiplication<T> &y) { return VectorHelper<T>::call(Vector<T>(x), static_cast<Vector<T> >(y)); } \
    template<typename T> static inline LRBni::Vector<T> name(const LRBni::VectorMultiplication<T> &x, const T &y) { return VectorHelper<T>::call(static_cast<Vector<T> >(x), Vector<T>(y)); } \
    template<typename T> static inline LRBni::Vector<T> name(const LRBni::VectorMultiplication<T> &x, const LRBni::VectorMultiplication<T> &y) { return VectorHelper<T>::call(static_cast<Vector<T> >(x), static_cast<Vector<T> >(y)); }

    MATH_OP2(min, min)
    MATH_OP2(max, max)
    MATH_OP1(sqrt, sqrt)
    MATH_OP1(rsqrt, rsqrt)
    MATH_OP1(abs, abs)
    MATH_OP1(sin, sin)
    MATH_OP1(cos, cos)
    MATH_OP1(log, log)
    MATH_OP1(log10, log10)
    MATH_OP1(atan, atan)
    MATH_OP2(atan2, atan2)
    MATH_OP1(reciprocal, recip)
    MATH_OP1(round, round)
    MATH_OP1(asin, asin)

    template<typename T> static inline LRBni::Mask<Vector<T>::Size> isfinite(const LRBni::Vector<T> &x) { return VectorHelper<T>::isFinite(x); }
    template<typename T> static inline LRBni::Mask<Vector<T>::Size> isfinite(const LRBni::VectorMultiplication<T> &x) { return VectorHelper<T>::isFinite(x); }
    template<typename T> static inline LRBni::Mask<Vector<T>::Size> isnan(const LRBni::Vector<T> &x) { return VectorHelper<T>::isNaN(x); }
    template<typename T> static inline LRBni::Mask<Vector<T>::Size> isnan(const LRBni::VectorMultiplication<T> &x) { return VectorHelper<T>::isNaN(x); }

  template<typename T> static inline void forceToRegisters(const Vector<T> &) {}
  template<typename T1, typename T2> static inline void forceToRegisters(
      const Vector<T1> &, const Vector<T2> &) {}
  template<typename T1, typename T2, typename T3> static inline void forceToRegisters(
        const Vector<T1>  &,  const Vector<T2>  &, const Vector<T3>  &) {}
  template<typename T1, typename T2, typename T3, typename T4> static inline void forceToRegisters(
        const Vector<T1>  &,  const Vector<T2>  &,
        const Vector<T3>  &,  const Vector<T4>  &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5>
    static inline void forceToRegisters(
        const Vector<T1>  &,  const Vector<T2>  &,
        const Vector<T3>  &,  const Vector<T4>  &,
        const Vector<T5>  &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
    static inline void forceToRegisters(
        const Vector<T1>  &,  const Vector<T2>  &,
        const Vector<T3>  &,  const Vector<T4>  &,
        const Vector<T5>  &,  const Vector<T6>  &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7>
    static inline void forceToRegisters(
        const Vector<T1>  &,  const Vector<T2>  &,
        const Vector<T3>  &,  const Vector<T4>  &,
        const Vector<T5>  &,  const Vector<T6>  &,
        const Vector<T7>  &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7, typename T8>
    static inline void forceToRegisters(
        const Vector<T1>  &,  const Vector<T2>  &,
        const Vector<T3>  &,  const Vector<T4>  &,
        const Vector<T5>  &,  const Vector<T6>  &,
        const Vector<T7>  &,  const Vector<T8>  &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7, typename T8, typename T9>
    static inline void forceToRegisters(
        const Vector<T1>  &,  const Vector<T2>  &,
        const Vector<T3>  &,  const Vector<T4>  &,
        const Vector<T5>  &,  const Vector<T6>  &,
        const Vector<T7>  &,  const Vector<T8>  &,
        const Vector<T9>  &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7, typename T8, typename T9, typename T10>
    static inline void forceToRegisters(
        const Vector<T1>  &, const Vector<T2>  &,
        const Vector<T3>  &, const Vector<T4>  &,
        const Vector<T5>  &, const Vector<T6>  &,
        const Vector<T7>  &, const Vector<T8>  &,
        const Vector<T9>  &, const Vector<T10> &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7, typename T8, typename T9, typename T10, typename T11>
    static inline void forceToRegisters(
        const Vector<T1>  &, const Vector<T2>  &,
        const Vector<T3>  &, const Vector<T4>  &,
        const Vector<T5>  &, const Vector<T6>  &,
        const Vector<T7>  &, const Vector<T8>  &,
        const Vector<T9>  &, const Vector<T10> &,
        const Vector<T11> &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7, typename T8, typename T9, typename T10, typename T11, typename T12>
    static inline void forceToRegisters(
        const Vector<T1>  &, const Vector<T2>  &,
        const Vector<T3>  &, const Vector<T4>  &,
        const Vector<T5>  &, const Vector<T6>  &,
        const Vector<T7>  &, const Vector<T8>  &,
        const Vector<T9>  &, const Vector<T10> &,
        const Vector<T11> &, const Vector<T12> &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13>
    static inline void forceToRegisters(
        const Vector<T1>  &, const Vector<T2>  &,
        const Vector<T3>  &, const Vector<T4>  &,
        const Vector<T5>  &, const Vector<T6>  &,
        const Vector<T7>  &, const Vector<T8>  &,
        const Vector<T9>  &, const Vector<T10> &,
        const Vector<T11> &, const Vector<T12> &,
        const Vector<T13> &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13,
    typename T14> static inline void forceToRegisters(
        const Vector<T1>  &, const Vector<T2>  &,
        const Vector<T3>  &, const Vector<T4>  &,
        const Vector<T5>  &, const Vector<T6>  &,
        const Vector<T7>  &, const Vector<T8>  &,
        const Vector<T9>  &, const Vector<T10> &,
        const Vector<T11> &, const Vector<T12> &,
        const Vector<T13> &, const Vector<T14> &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13,
    typename T14, typename T15> static inline void forceToRegisters(
        const Vector<T1>  &, const Vector<T2>  &,
        const Vector<T3>  &, const Vector<T4>  &,
        const Vector<T5>  &, const Vector<T6>  &,
        const Vector<T7>  &, const Vector<T8>  &,
        const Vector<T9>  &, const Vector<T10> &,
        const Vector<T11> &, const Vector<T12> &,
        const Vector<T13> &, const Vector<T14> &,
        const Vector<T15> &) {}
  template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6,
    typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13,
    typename T14, typename T15, typename T16> static inline void forceToRegisters(
        const Vector<T1>  &, const Vector<T2>  &,
        const Vector<T3>  &, const Vector<T4>  &,
        const Vector<T5>  &, const Vector<T6>  &,
        const Vector<T7>  &, const Vector<T8>  &,
        const Vector<T9>  &, const Vector<T10> &,
        const Vector<T11> &, const Vector<T12> &,
        const Vector<T13> &, const Vector<T14> &,
        const Vector<T15> &, const Vector<T16> &) {}
} // namespace LRBni
} // namespace Vc

#undef LRB_STATIC_ASSERT_NC
#undef LRB_STATIC_ASSERT
#undef CAT
#undef CAT_HELPER

#endif // LARRABEE_VECTOR_H
