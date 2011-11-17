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

#ifndef SSE_MASK_H
#define SSE_MASK_H

#include "intrinsics.h"

namespace Vc
{
namespace SSE
{

template<unsigned int Size1> struct MaskHelper;
template<> struct MaskHelper<2> {
    static inline bool cmpeq (_M128 k1, _M128 k2) { return _mm_movemask_pd(_mm_castps_pd(k1)) == _mm_movemask_pd(_mm_castps_pd(k2)); }
    static inline bool cmpneq(_M128 k1, _M128 k2) { return _mm_movemask_pd(_mm_castps_pd(k1)) != _mm_movemask_pd(_mm_castps_pd(k2)); }
};
template<> struct MaskHelper<4> {
    static inline bool cmpeq (_M128 k1, _M128 k2) { return _mm_movemask_ps(k1) == _mm_movemask_ps(k2); }
    static inline bool cmpneq(_M128 k1, _M128 k2) { return _mm_movemask_ps(k1) != _mm_movemask_ps(k2); }
};
template<> struct MaskHelper<8> {
    static inline bool cmpeq (_M128 k1, _M128 k2) { return _mm_movemask_epi8(_mm_castps_si128(k1)) == _mm_movemask_epi8(_mm_castps_si128(k2)); }
    static inline bool cmpneq(_M128 k1, _M128 k2) { return _mm_movemask_epi8(_mm_castps_si128(k1)) != _mm_movemask_epi8(_mm_castps_si128(k2)); }
};

class Float8Mask;
template<unsigned int VectorSize> class Mask
{
    friend class Mask<2u>;
    friend class Mask<4u>;
    friend class Mask<8u>;
    friend class Mask<16u>;
    friend class Float8Mask;
    public:
        FREE_STORE_OPERATORS_ALIGNED(16)
        
        // abstracts the way Masks are passed to functions, it can easily be changed to const ref here
        // Also Float8Mask requires const ref on MSVC 32bit.
#if defined VC_MSVC && defined _WIN32
        typedef const Mask<VectorSize> &Argument;
#else
        typedef Mask<VectorSize> Argument;
#endif


        inline Mask() {}
        inline Mask(const __m128  &x) : k(x) {}
        inline Mask(const __m128d &x) : k(_mm_castpd_ps(x)) {}
        inline Mask(const __m128i &x) : k(_mm_castsi128_ps(x)) {}
        inline explicit Mask(VectorSpecialInitializerZero::ZEnum) : k(_mm_setzero_ps()) {}
        inline explicit Mask(VectorSpecialInitializerOne::OEnum) : k(_mm_setallone_ps()) {}
        inline explicit Mask(bool b) : k(b ? _mm_setallone_ps() : _mm_setzero_ps()) {}
        inline Mask(const Mask &rhs) : k(rhs.k) {}
        inline Mask(const Mask<VectorSize / 2> *a)
          : k(_mm_castsi128_ps(_mm_packs_epi16(a[0].dataI(), a[1].dataI()))) {}
        inline explicit Mask(const Float8Mask &m);

        template<unsigned int OtherSize> explicit Mask(const Mask<OtherSize> &x);
//X         {
//X             _M128I tmp = x.dataI();
//X             if (OtherSize < VectorSize) {
//X                 tmp = _mm_packs_epi16(tmp, _mm_setzero_si128());
//X                 if (VectorSize / OtherSize >= 4u) { tmp = _mm_packs_epi16(tmp, _mm_setzero_si128()); }
//X                 if (VectorSize / OtherSize >= 8u) { tmp = _mm_packs_epi16(tmp, _mm_setzero_si128()); }
//X             } else if (OtherSize > VectorSize) {
//X                 tmp = _mm_unpacklo_epi8(tmp, tmp);
//X                 if (OtherSize / VectorSize >= 4u) { tmp = _mm_unpacklo_epi8(tmp, tmp); }
//X                 if (OtherSize / VectorSize >= 8u) { tmp = _mm_unpacklo_epi8(tmp, tmp); }
//X             }
//X             k = _mm_castsi128_ps(tmp);
//X         }

        void expand(Mask<VectorSize / 2> *x) const;

        inline bool operator==(const Mask &rhs) const { return MaskHelper<VectorSize>::cmpeq (k, rhs.k); }
        inline bool operator!=(const Mask &rhs) const { return MaskHelper<VectorSize>::cmpneq(k, rhs.k); }

        inline Mask operator&&(const Mask &rhs) const { return _mm_and_ps(k, rhs.k); }
        inline Mask operator& (const Mask &rhs) const { return _mm_and_ps(k, rhs.k); }
        inline Mask operator||(const Mask &rhs) const { return _mm_or_ps (k, rhs.k); }
        inline Mask operator| (const Mask &rhs) const { return _mm_or_ps (k, rhs.k); }
        inline Mask operator^ (const Mask &rhs) const { return _mm_xor_ps(k, rhs.k); }
        inline Mask operator!() const { return _mm_andnot_si128(dataI(), _mm_setallone_si128()); }

        inline Mask &operator&=(const Mask &rhs) { k = _mm_and_ps(k, rhs.k); return *this; }
        inline Mask &operator|=(const Mask &rhs) { k = _mm_or_ps (k, rhs.k); return *this; }

        inline bool isFull () const { return
#ifdef VC_USE_PTEST
            _mm_testc_si128(dataI(), _mm_setallone_si128()); // return 1 if (0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff) == (~0 & k)
#else
            _mm_movemask_epi8(dataI()) == 0xffff;
#endif
        }
        inline bool isEmpty() const { return
#ifdef VC_USE_PTEST
            _mm_testz_si128(dataI(), dataI()); // return 1 if (0, 0, 0, 0) == (k & k)
#else
            _mm_movemask_epi8(dataI()) == 0x0000;
#endif
        }
        inline bool isMix() const {
#ifdef VC_USE_PTEST
            return _mm_test_mix_ones_zeros(dataI(), _mm_setallone_si128());
#else
            const int tmp = _mm_movemask_epi8(dataI());
            return tmp != 0 && (tmp ^ 0xffff) != 0;
#endif
        }

        inline operator bool() const { return isFull(); }

        inline int CONST_L shiftMask() const CONST_R;

        int CONST_L toInt() const CONST_R;

        inline _M128  data () const { return k; }
        inline _M128I dataI() const { return _mm_castps_si128(k); }
        inline _M128D dataD() const { return _mm_castps_pd(k); }

        template<unsigned int OtherSize> inline Mask<OtherSize> cast() const { return Mask<OtherSize>(k); }

        bool operator[](int index) const;

        int count() const;

        /**
         * Returns the index of the first one in the mask.
         *
         * The return value is undefined if the mask is empty.
         */
        int firstOne() const;

    private:
        _M128 k;
};

struct ForeachHelper
{
    _long mask;
    bool brk;
    inline ForeachHelper(_long _mask) : mask(_mask), brk(false) {}
    inline bool outer() const { return mask != 0; }
    inline bool inner() { return (brk = !brk); }
    inline _long next() {
#ifdef __GNUC__ // the compiler understands inline asm
        const _long bit = __builtin_ctzl(mask);
        __asm__("btr %1,%0" : "+r"(mask) : "r"(bit));
#elif defined(_WIN64)
       unsigned long bit;
       _BitScanForward64(&bit, mask);
       _bittestandreset64(&mask, bit);
#elif defined(_WIN32)
       unsigned long bit;
       _BitScanForward(&bit, mask);
       _bittestandreset(&mask, bit);
#else
#error "Not implemented yet. Please contact vc-devel@compeng.uni-frankfurt.de"
#endif
        return bit;
    }
};

#define Vc_foreach_bit(_it_, _mask_) \
    for (Vc::SSE::ForeachHelper _Vc_foreach_bit_helper((_mask_).toInt()); _Vc_foreach_bit_helper.outer(); ) \
        for (_it_ = _Vc_foreach_bit_helper.next(); _Vc_foreach_bit_helper.inner(); )

template<unsigned int Size> inline int Mask<Size>::shiftMask() const
{
    return _mm_movemask_epi8(dataI());
}

template<> template<> inline Mask<2>::Mask(const Mask<4> &x) {
    k = _mm_unpacklo_ps(x.data(), x.data());
}
template<> template<> inline Mask<2>::Mask(const Mask<8> &x) {
    _M128I tmp = _mm_unpacklo_epi16(x.dataI(), x.dataI());
    k = _mm_castsi128_ps(_mm_unpacklo_epi32(tmp, tmp));
}
template<> template<> inline Mask<2>::Mask(const Mask<16> &x) {
    _M128I tmp = _mm_unpacklo_epi8(x.dataI(), x.dataI());
    tmp = _mm_unpacklo_epi16(tmp, tmp);
    k = _mm_castsi128_ps(_mm_unpacklo_epi32(tmp, tmp));
}
template<> template<> inline Mask<4>::Mask(const Mask<2> &x) {
    k = _mm_castsi128_ps(_mm_packs_epi16(x.dataI(), _mm_setzero_si128()));
}
template<> template<> inline Mask<4>::Mask(const Mask<8> &x) {
    k = _mm_castsi128_ps(_mm_unpacklo_epi16(x.dataI(), x.dataI()));
}
template<> template<> inline Mask<4>::Mask(const Mask<16> &x) {
    _M128I tmp = _mm_unpacklo_epi8(x.dataI(), x.dataI());
    k = _mm_castsi128_ps(_mm_unpacklo_epi16(tmp, tmp));
}
template<> template<> inline Mask<8>::Mask(const Mask<2> &x) {
    _M128I tmp = _mm_packs_epi16(x.dataI(), x.dataI());
    k = _mm_castsi128_ps(_mm_packs_epi16(tmp, tmp));
}
template<> template<> inline Mask<8>::Mask(const Mask<4> &x) {
    k = _mm_castsi128_ps(_mm_packs_epi16(x.dataI(), x.dataI()));
}
template<> template<> inline Mask<8>::Mask(const Mask<16> &x) {
    k = _mm_castsi128_ps(_mm_unpacklo_epi8(x.dataI(), x.dataI()));
}

template<> inline void Mask< 4>::expand(Mask<2> *x) const {
    x[0].k = _mm_unpacklo_ps(data(), data());
    x[1].k = _mm_unpackhi_ps(data(), data());
}
template<> inline void Mask< 8>::expand(Mask<4> *x) const {
    x[0].k = _mm_castsi128_ps(_mm_unpacklo_epi16(dataI(), dataI()));
    x[1].k = _mm_castsi128_ps(_mm_unpackhi_epi16(dataI(), dataI()));
}
template<> inline void Mask<16>::expand(Mask<8> *x) const {
    x[0].k = _mm_castsi128_ps(_mm_unpacklo_epi8 (dataI(), dataI()));
    x[1].k = _mm_castsi128_ps(_mm_unpackhi_epi8 (dataI(), dataI()));
}

template<> inline int Mask< 2>::toInt() const { return _mm_movemask_pd(dataD()); }
template<> inline int Mask< 4>::toInt() const { return _mm_movemask_ps(data ()); }
template<> inline int Mask< 8>::toInt() const { return _mm_movemask_epi8(_mm_packs_epi16(dataI(), _mm_setzero_si128())); }
template<> inline int Mask<16>::toInt() const { return _mm_movemask_epi8(dataI()); }

template<> inline bool Mask< 2>::operator[](int index) const { return toInt() & (1 << index); }
template<> inline bool Mask< 4>::operator[](int index) const { return toInt() & (1 << index); }
template<> inline bool Mask< 8>::operator[](int index) const { return shiftMask() & (1 << 2 * index); }
template<> inline bool Mask<16>::operator[](int index) const { return toInt() & (1 << index); }

template<> inline int Mask<2>::count() const
{
    int mask = _mm_movemask_pd(dataD());
    return (mask & 1) + (mask >> 1);
}

template<> inline int Mask<4>::count() const
{
#ifdef VC_IMPL_SSE4_2
    return _mm_popcnt_u32(_mm_movemask_ps(data()));
//X     tmp = (tmp & 5) + ((tmp >> 1) & 5);
//X     return (tmp & 3) + ((tmp >> 2) & 3);
#else
    _M128I x = _mm_srli_epi32(dataI(), 31);
    x = _mm_add_epi32(x, _mm_shuffle_epi32(x, _MM_SHUFFLE(0, 1, 2, 3)));
    x = _mm_add_epi32(x, _mm_shufflelo_epi16(x, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si32(x);
#endif
}

template<> inline int Mask<8>::count() const
{
#ifdef VC_IMPL_SSE4_2
    return _mm_popcnt_u32(_mm_movemask_epi8(dataI())) / 2;
#else
//X     int tmp = _mm_movemask_epi8(dataI());
//X     tmp = (tmp & 0x1111) + ((tmp >> 2) & 0x1111);
//X     tmp = (tmp & 0x0303) + ((tmp >> 4) & 0x0303);
//X     return (tmp & 0x000f) + ((tmp >> 8) & 0x000f);
    _M128I x = _mm_srli_epi16(dataI(), 15);
    x = _mm_add_epi16(x, _mm_shuffle_epi32(x, _MM_SHUFFLE(0, 1, 2, 3)));
    x = _mm_add_epi16(x, _mm_shufflelo_epi16(x, _MM_SHUFFLE(0, 1, 2, 3)));
    x = _mm_add_epi16(x, _mm_shufflelo_epi16(x, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_extract_epi16(x, 0);
#endif
}

template<> inline int Mask<16>::count() const
{
    int tmp = _mm_movemask_epi8(dataI());
#ifdef VC_IMPL_SSE4_2
    return _mm_popcnt_u32(tmp);
#else
    tmp = (tmp & 0x5555) + ((tmp >> 1) & 0x5555);
    tmp = (tmp & 0x3333) + ((tmp >> 2) & 0x3333);
    tmp = (tmp & 0x0f0f) + ((tmp >> 4) & 0x0f0f);
    return (tmp & 0x00ff) + ((tmp >> 8) & 0x00ff);
#endif
}


class Float8Mask
{
    enum {
        PartialSize = 4,
        VectorSize = 8
    };
    public:
        FREE_STORE_OPERATORS_ALIGNED(16)

        // abstracts the way Masks are passed to functions, it can easily be changed to const ref here
        // Also Float8Mask requires const ref on MSVC 32bit.
#if defined VC_MSVC && defined _WIN32
        typedef const Float8Mask & Argument;
#else
        typedef Float8Mask Argument;
#endif

        inline Float8Mask() {}
        inline Float8Mask(const M256 &x) : k(x) {}
        inline explicit Float8Mask(VectorSpecialInitializerZero::ZEnum) {
            k[0] = _mm_setzero_ps();
            k[1] = _mm_setzero_ps();
        }
        inline explicit Float8Mask(VectorSpecialInitializerOne::OEnum) {
            k[0] = _mm_setallone_ps();
            k[1] = _mm_setallone_ps();
        }
        inline explicit Float8Mask(bool b) {
            const __m128 tmp = b ? _mm_setallone_ps() : _mm_setzero_ps();
            k[0] = tmp;
            k[1] = tmp;
        }
        inline Float8Mask(const Mask<VectorSize> &a) {
            k[0] = _mm_castsi128_ps(_mm_unpacklo_epi16(a.dataI(), a.dataI()));
            k[1] = _mm_castsi128_ps(_mm_unpackhi_epi16(a.dataI(), a.dataI()));
        }

        inline bool operator==(const Float8Mask &rhs) const {
            return MaskHelper<PartialSize>::cmpeq (k[0], rhs.k[0])
                && MaskHelper<PartialSize>::cmpeq (k[1], rhs.k[1]);
        }
        inline bool operator!=(const Float8Mask &rhs) const {
            return MaskHelper<PartialSize>::cmpneq(k[0], rhs.k[0])
                && MaskHelper<PartialSize>::cmpneq(k[1], rhs.k[1]);
        }

        inline Float8Mask operator&&(const Float8Mask &rhs) const {
            Float8Mask r;
            r.k[0] = _mm_and_ps(k[0], rhs.k[0]);
            r.k[1] = _mm_and_ps(k[1], rhs.k[1]);
            return r;
        }
        inline Float8Mask operator& (const Float8Mask &rhs) const {
            Float8Mask r;
            r.k[0] = _mm_and_ps(k[0], rhs.k[0]);
            r.k[1] = _mm_and_ps(k[1], rhs.k[1]);
            return r;
        }
        inline Float8Mask operator||(const Float8Mask &rhs) const {
            Float8Mask r;
            r.k[0] = _mm_or_ps(k[0], rhs.k[0]);
            r.k[1] = _mm_or_ps(k[1], rhs.k[1]);
            return r;
        }
        inline Float8Mask operator| (const Float8Mask &rhs) const {
            Float8Mask r;
            r.k[0] = _mm_or_ps(k[0], rhs.k[0]);
            r.k[1] = _mm_or_ps(k[1], rhs.k[1]);
            return r;
        }
        inline Float8Mask operator^ (const Float8Mask &rhs) const {
            Float8Mask r;
            r.k[0] = _mm_xor_ps(k[0], rhs.k[0]);
            r.k[1] = _mm_xor_ps(k[1], rhs.k[1]);
            return r;
        }
        inline Float8Mask operator!() const {
            Float8Mask r;
            r.k[0] = _mm_andnot_ps(k[0], _mm_setallone_ps());
            r.k[1] = _mm_andnot_ps(k[1], _mm_setallone_ps());
            return r;
        }
        inline Float8Mask &operator&=(const Float8Mask &rhs) {
            k[0] = _mm_and_ps(k[0], rhs.k[0]);
            k[1] = _mm_and_ps(k[1], rhs.k[1]);
            return *this;
        }
        inline Float8Mask &operator|=(const Float8Mask &rhs) {
            k[0] = _mm_or_ps (k[0], rhs.k[0]);
            k[1] = _mm_or_ps (k[1], rhs.k[1]);
            return *this;
        }

        inline bool isFull () const {
            const _M128 tmp = _mm_and_ps(k[0], k[1]);
#ifdef VC_USE_PTEST
            return _mm_testc_si128(_mm_castps_si128(tmp), _mm_setallone_si128());
#else
            return _mm_movemask_ps(tmp) == 0xf;
            //_mm_movemask_ps(k[0]) == 0xf &&
            //_mm_movemask_ps(k[1]) == 0xf;
#endif
        }
        inline bool isEmpty() const {
            const _M128 tmp = _mm_or_ps(k[0], k[1]);
#ifdef VC_USE_PTEST
            return _mm_testz_si128(_mm_castps_si128(tmp), _mm_castps_si128(tmp));
#else
            return _mm_movemask_ps(tmp) == 0x0;
            //_mm_movemask_ps(k[0]) == 0x0 &&
            //_mm_movemask_ps(k[1]) == 0x0;
#endif
        }
        inline bool isMix() const {
#ifdef VC_USE_PTEST
            return _mm_test_mix_ones_zeros(_mm_castps_si128(k[0]), _mm_castps_si128(k[0])) &&
            _mm_test_mix_ones_zeros(_mm_castps_si128(k[1]), _mm_castps_si128(k[1]));
#else
            const int tmp = _mm_movemask_ps(k[0]) + _mm_movemask_ps(k[1]);
            return tmp > 0x0 && tmp < (0xf + 0xf);
#endif
        }

        inline operator bool() const { return isFull(); }

        inline int shiftMask() const {
            return (_mm_movemask_ps(k[1]) << 4) + _mm_movemask_ps(k[0]);
        }
        inline int toInt() const { return (_mm_movemask_ps(k[1]) << 4) + _mm_movemask_ps(k[0]); }

        inline const M256 &data () const { return k; }

        inline bool operator[](int index) const {
            return (toInt() & (1 << index)) != 0;
        }

        inline int count() const {
#ifdef VC_IMPL_SSE4_2
		return _mm_popcnt_u32(toInt());
#else
//X             int tmp1 = _mm_movemask_ps(k[0]);
//X             int tmp2 = _mm_movemask_ps(k[1]);
//X             tmp1 = (tmp1 & 5) + ((tmp1 >> 1) & 5);
//X             tmp2 = (tmp2 & 5) + ((tmp2 >> 1) & 5);
//X             return (tmp1 & 3) + (tmp2 & 3) + ((tmp1 >> 2) & 3) + ((tmp2 >> 2) & 3);
            _M128I x = _mm_add_epi32(_mm_srli_epi32(_mm_castps_si128(k[0]), 31),
                                     _mm_srli_epi32(_mm_castps_si128(k[1]), 31));
            x = _mm_add_epi32(x, _mm_shuffle_epi32(x, _MM_SHUFFLE(0, 1, 2, 3)));
            x = _mm_add_epi32(x, _mm_shufflelo_epi16(x, _MM_SHUFFLE(1, 0, 3, 2)));
            return _mm_cvtsi128_si32(x);
#endif
        }

        int firstOne() const;

    private:
        M256 k;
};

template<unsigned int Size> inline int Mask<Size>::firstOne() const
{
    const int mask = toInt();
#ifdef _MSC_VER
    unsigned long bit;
    _BitScanForward(&bit, mask);
#else
    int bit;
    __asm__("bsf %1,%0" : "=&r"(bit) : "r"(mask));
#endif
    return bit;
}
inline int Float8Mask::firstOne() const
{
    const int mask = toInt();
#ifdef _MSC_VER
    unsigned long bit;
    _BitScanForward(&bit, mask);
#else
    int bit;
    __asm__("bsf %1,%0" : "=&r"(bit) : "r"(mask));
#endif
    return bit;
}

template<unsigned int VectorSize>
inline Mask<VectorSize>::Mask(const Float8Mask &m)
    : k(_mm_castsi128_ps(_mm_packs_epi32(_mm_castps_si128(m.data()[0]), _mm_castps_si128(m.data()[1])))) {}

class Float8GatherMask
{
    public:
        Float8GatherMask(const Mask<8u> &k)   : mask(k.toInt()) {}
        Float8GatherMask(const Float8Mask &k) : mask(k.toInt()) {}
        int toInt() const { return mask; }
    private:
        const int mask;
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
//X #define foreach_bit(it, mask)
//X     for (int _sse_vector_foreach_inner = 1, ForeachScope _sse_vector_foreach_scope(mask.toInt()), int it = _sse_vector_foreach_scope.bit(); _sse_vector_foreach_inner; --_sse_vector_foreach_inner)
//X     for (int _sse_vector_foreach_mask = (mask).toInt(), int _sse_vector_foreach_it = _sse_bitscan(mask.toInt());
//X             _sse_vector_foreach_it > 0;
//X             _sse_vector_foreach_it = _sse_bitscan_initialized(_sse_vector_foreach_it, mask.data()))
//X         for (int _sse_vector_foreach_inner = 1, it = _sse_vector_foreach_it; _sse_vector_foreach_inner; --_sse_vector_foreach_inner)

} // namespace SSE
} // namespace Vc

#endif // SSE_MASK_H
