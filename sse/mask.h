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
    friend class Float8Mask;
    public:
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

        template<unsigned int OtherSize> explicit inline Mask(const Mask<OtherSize> &x)
        {
            _M128I tmp = x.dataI();
            if (OtherSize < VectorSize) {
                tmp = _mm_packs_epi16(tmp, _mm_setzero_si128());
                if (VectorSize / OtherSize >= 4u) { tmp = _mm_packs_epi16(tmp, _mm_setzero_si128()); }
                if (VectorSize / OtherSize >= 8u) { tmp = _mm_packs_epi16(tmp, _mm_setzero_si128()); }
            } else if (OtherSize > VectorSize) {
                tmp = _mm_unpacklo_epi8(tmp, tmp);
                if (OtherSize / VectorSize >= 4u) { tmp = _mm_unpacklo_epi8(tmp, tmp); }
                if (OtherSize / VectorSize >= 8u) { tmp = _mm_unpacklo_epi8(tmp, tmp); }
            }
            k = _mm_castsi128_ps(tmp);
        }

        inline void expand(Mask<VectorSize / 2> *x) const
        {
            enum { Shuf = _MM_SHUFFLE(1, 1, 0, 0) };
            if (VectorSize == 16u) {
                x[0].k = _mm_castsi128_ps(_mm_unpacklo_epi8 (dataI(), dataI()));
                x[1].k = _mm_castsi128_ps(_mm_unpackhi_epi8 (dataI(), dataI()));
            } else if (VectorSize == 8u) {
                x[0].k = _mm_castsi128_ps(_mm_unpacklo_epi16(dataI(), dataI()));
                x[1].k = _mm_castsi128_ps(_mm_unpackhi_epi16(dataI(), dataI()));
            } else if (VectorSize == 4u) {
                x[0].k = _mm_castsi128_ps(_mm_unpacklo_epi32(dataI(), dataI()));
                x[1].k = _mm_castsi128_ps(_mm_unpackhi_epi32(dataI(), dataI()));
            } else if (VectorSize == 2u) {
                x[0].k = _mm_castsi128_ps(_mm_unpacklo_epi64(dataI(), dataI()));
                x[1].k = _mm_castsi128_ps(_mm_unpackhi_epi64(dataI(), dataI()));
            }
        }

        inline bool operator==(const Mask &rhs) const { return MaskHelper<VectorSize>::cmpeq (k, rhs.k); }
        inline bool operator!=(const Mask &rhs) const { return MaskHelper<VectorSize>::cmpneq(k, rhs.k); }

        inline Mask operator&&(const Mask &rhs) const { return _mm_and_ps(k, rhs.k); }
        inline Mask operator& (const Mask &rhs) const { return _mm_and_ps(k, rhs.k); }
        inline Mask operator||(const Mask &rhs) const { return _mm_or_ps (k, rhs.k); }
        inline Mask operator| (const Mask &rhs) const { return _mm_or_ps (k, rhs.k); }
        inline Mask operator!() const { return _mm_andnot_si128(dataI(), _mm_setallone_si128()); }

        inline Mask &operator&=(const Mask &rhs) { k = _mm_and_ps(k, rhs.k); return *this; }
        inline Mask &operator|=(const Mask &rhs) { k = _mm_or_ps (k, rhs.k); return *this; }

        inline bool isFull () const { return
#ifdef __SSE4_1__
            _mm_testc_si128(dataI(), _mm_setallone_si128()); // return 1 if (0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff) == (~0 & k)
#else
            _mm_movemask_epi8(dataI()) == 0xffff;
#endif
        }
        inline bool isEmpty() const { return
#ifdef __SSE4_1__
            _mm_testz_si128(dataI(), dataI()); // return 1 if (0, 0, 0, 0) == (k & k)
#else
            _mm_movemask_epi8(dataI()) == 0x0000;
#endif
        }

        inline operator bool() const { return isFull(); }

        inline int shiftMask() const {
            return _mm_movemask_epi8(dataI());
        }

        inline int toInt() const {
            if (VectorSize == 2) {
                return _mm_movemask_pd(dataD());
            } else if (VectorSize == 4) {
                return _mm_movemask_ps(data());
            } else if (VectorSize == 8) {
                return _mm_movemask_epi8(_mm_packs_epi16(dataI(), _mm_setzero_si128()));
            } else if (VectorSize == 16) {
                return _mm_movemask_epi8(dataI());
            }
        }

        inline _M128  data () const { return k; }
        inline _M128I dataI() const { return _mm_castps_si128(k); }
        inline _M128D dataD() const { return _mm_castps_pd(k); }

        template<unsigned int OtherSize> inline Mask<OtherSize> cast() const { return Mask<OtherSize>(k); }

        inline bool operator[](int index) const {
            if (VectorSize == 2) {
                return _mm_movemask_pd(dataD()) & (1 << index);
            } else if (VectorSize == 4) {
                return _mm_movemask_ps(k) & (1 << index);
            } else if (VectorSize == 8) {
                return _mm_movemask_epi8(dataI()) & (1 << 2 * index);
            } else if (VectorSize == 16) {
                return _mm_movemask_epi8(dataI()) & (1 << index);
            }
            return false;
        }

    private:
        _M128 k;
};

class Float8Mask
{
    enum {
        PartialSize = 4,
        VectorSize = 8
    };
    public:
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

        inline bool isFull () const { return
#ifdef __SSE4_1__
            _mm_testc_si128(_mm_castps_si128(k[0]), _mm_setallone_si128()) &&
            _mm_testc_si128(_mm_castps_si128(k[1]), _mm_setallone_si128());
#else
            _mm_movemask_ps(k[0]) == 0xf &&
            _mm_movemask_ps(k[1]) == 0xf;
#endif
        }
        inline bool isEmpty() const { return
#ifdef __SSE4_1__
            _mm_testz_si128(_mm_castps_si128(k[0]), _mm_castps_si128(k[0])) &&
            _mm_testz_si128(_mm_castps_si128(k[1]), _mm_castps_si128(k[1]));
#else
            _mm_movemask_ps(k[0]) == 0x0 &&
            _mm_movemask_ps(k[1]) == 0x0;
#endif
        }

        inline operator bool() const { return isFull(); }

        inline int shiftMask() const {
            return (_mm_movemask_ps(k[1]) << 4) + _mm_movemask_ps(k[0]);
        }
        inline int toInt() const { return (_mm_movemask_ps(k[1]) << 4) + _mm_movemask_ps(k[0]); }

        inline const M256 &data () const { return k; }

        inline bool operator[](int index) const {
            return toInt() & (1 << index);
        }

    private:
        M256 k;
};

template<unsigned int VectorSize>
inline Mask<VectorSize>::Mask(const Float8Mask &m)
    : k(_mm_castsi128_ps(_mm_packs_epi32(_mm_castps_si128(m.data()[0]), _mm_castps_si128(m.data()[1])))) {}

template<typename M, typename F>
inline void foreach_bit(const M &mask, F func) {
    unsigned short m = mask.toInt();
    short i;
    asm("bsf %1,%0"                "\n\t"
        "jz _sse_bitscan_end"      "\n\t"
        "_sse_bitscan_loop:"       "\n\t"
        "btr %0,%1"                "\n\t"
        "bsf %1,%0"                "\n\t"
        "jnz _sse_bitscan_loop"    "\n\t"
        "_sse_bitscan_end:"        "\n\t"
        : "=a"(i)
        : "m"(m)
       );
    func(i);
}

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
#endif // SSE_MASK_H
