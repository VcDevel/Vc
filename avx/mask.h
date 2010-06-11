/*  This file is part of the Vc library.

    Copyright (C) 2009-2010 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Leavxr General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Leavxr General Public License for more details.

    You should have received a copy of the GNU Leavxr General Public
    License along with Vc.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifndef AVX_MASK_H
#define AVX_MASK_H

#include "intrinsics.h"

namespace Vc
{
namespace AVX
{

template<unsigned int Size1> struct MaskHelper;
template<> struct MaskHelper<2> {
    static inline bool cmpeq (_M256 k1, _M256 k2) { return _mm256_movemask_pd(_mm256_castps_pd(k1)) == _mm256_movemask_pd(_mm256_castps_pd(k2)); }
    static inline bool cmpneq(_M256 k1, _M256 k2) { return _mm256_movemask_pd(_mm256_castps_pd(k1)) != _mm256_movemask_pd(_mm256_castps_pd(k2)); }
};
template<> struct MaskHelper<4> {
    static inline bool cmpeq (_M256 k1, _M256 k2) { return _mm256_movemask_ps(k1) == _mm256_movemask_ps(k2); }
    static inline bool cmpneq(_M256 k1, _M256 k2) { return _mm256_movemask_ps(k1) != _mm256_movemask_ps(k2); }
};
template<> struct MaskHelper<8> {
    static inline bool cmpeq (_M256 k1, _M256 k2) { return _mm256_movemask_epi8(_mm256_castps_si128(k1)) == _mm256_movemask_epi8(_mm256_castps_si128(k2)); }
    static inline bool cmpneq(_M256 k1, _M256 k2) { return _mm256_movemask_epi8(_mm256_castps_si128(k1)) != _mm256_movemask_epi8(_mm256_castps_si128(k2)); }
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
        inline Mask() {}
        inline Mask(const __m256  &x) : k(x) {}
        inline Mask(const __m256d &x) : k(_mm256_castpd_ps(x)) {}
        inline Mask(const __m256i &x) : k(_mm256_castsi128_ps(x)) {}
        inline explicit Mask(VectorSpecialInitializerZero::ZEnum) : k(_mm256_setzero_ps()) {}
        inline explicit Mask(VectorSpecialInitializerOne::OEnum) : k(_mm256_setallone_ps()) {}
        inline explicit Mask(bool b) : k(b ? _mm256_setallone_ps() : _mm256_setzero_ps()) {}
        inline Mask(const Mask &rhs) : k(rhs.k) {}
        inline Mask(const Mask<VectorSize / 2> *a)
          : k(_mm256_castsi128_ps(_mm256_packs_epi16(a[0].dataI(), a[1].dataI()))) {}
        inline explicit Mask(const Float8Mask &m);

        template<unsigned int OtherSize> explicit Mask(const Mask<OtherSize> &x);
//X         {
//X             _M256I tmp = x.dataI();
//X             if (OtherSize < VectorSize) {
//X                 tmp = _mm256_packs_epi16(tmp, _mm256_setzero_si128());
//X                 if (VectorSize / OtherSize >= 4u) { tmp = _mm256_packs_epi16(tmp, _mm256_setzero_si128()); }
//X                 if (VectorSize / OtherSize >= 8u) { tmp = _mm256_packs_epi16(tmp, _mm256_setzero_si128()); }
//X             } else if (OtherSize > VectorSize) {
//X                 tmp = _mm256_unpacklo_epi8(tmp, tmp);
//X                 if (OtherSize / VectorSize >= 4u) { tmp = _mm256_unpacklo_epi8(tmp, tmp); }
//X                 if (OtherSize / VectorSize >= 8u) { tmp = _mm256_unpacklo_epi8(tmp, tmp); }
//X             }
//X             k = _mm256_castsi128_ps(tmp);
//X         }

        void expand(Mask<VectorSize / 2> *x) const;

        inline bool operator==(const Mask &rhs) const { return MaskHelper<VectorSize>::cmpeq (k, rhs.k); }
        inline bool operator!=(const Mask &rhs) const { return MaskHelper<VectorSize>::cmpneq(k, rhs.k); }

        inline Mask operator&&(const Mask &rhs) const { return _mm256_and_ps(k, rhs.k); }
        inline Mask operator& (const Mask &rhs) const { return _mm256_and_ps(k, rhs.k); }
        inline Mask operator||(const Mask &rhs) const { return _mm256_or_ps (k, rhs.k); }
        inline Mask operator| (const Mask &rhs) const { return _mm256_or_ps (k, rhs.k); }
        inline Mask operator^ (const Mask &rhs) const { return _mm256_xor_ps(k, rhs.k); }
        inline Mask operator!() const { return _mm256_andnot_si128(dataI(), _mm256_setallone_si128()); }

        inline Mask &operator&=(const Mask &rhs) { k = _mm256_and_ps(k, rhs.k); return *this; }
        inline Mask &operator|=(const Mask &rhs) { k = _mm256_or_ps (k, rhs.k); return *this; }

        inline bool isFull () const { return
#ifdef VC_USE_PTEST
            _mm256_testc_si128(dataI(), _mm256_setallone_si128()); // return 1 if (0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff) == (~0 & k)
#else
            _mm256_movemask_epi8(dataI()) == 0xffff;
#endif
        }
        inline bool isEmpty() const { return
#ifdef VC_USE_PTEST
            _mm256_testz_si128(dataI(), dataI()); // return 1 if (0, 0, 0, 0) == (k & k)
#else
            _mm256_movemask_epi8(dataI()) == 0x0000;
#endif
        }
        inline bool isMix() const {
#ifdef VC_USE_PTEST
            return _mm256_test_mix_ones_zeros(dataI(), _mm256_setallone_si128());
#else
            const int tmp = _mm256_movemask_epi8(dataI());
            return tmp != 0 && (tmp ^ 0xffff) != 0;
#endif
        }

        inline operator bool() const { return isFull(); }

        inline int shiftMask() const CONST;

        int toInt() const CONST;

        inline _M256  data () const { return k; }
#ifdef VC_GATHER_SET
        inline _M256  dataIndex() const { return k; }
#endif
        inline _M256I dataI() const { return _mm256_castps_si128(k); }
        inline _M256D dataD() const { return _mm256_castps_pd(k); }

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
        _M256 k;
};

struct ForeachHelper
{
    unsigned long mask;
    bool brk;
    inline ForeachHelper(unsigned long _mask) : mask(_mask), brk(false) {}
    inline bool outer() const { return mask != 0; }
    inline bool inner() { return (brk = !brk); }
    inline unsigned long next() {
        const unsigned long bit = __builtin_ctzl(mask);
        __asm__("btr %1,%0" : "+r"(mask) : "r"(bit));
        return bit;
    }
};

#define Vc_foreach_bit(_it_, _mask_) \
    for (Vc::AVX::ForeachHelper _Vc_foreach_bit_helper((_mask_).toInt()); _Vc_foreach_bit_helper.outer(); ) \
        for (_it_ = _Vc_foreach_bit_helper.next(); _Vc_foreach_bit_helper.inner(); )

#define foreach_bit(_it_, _mask_) Vc_foreach_bit(_it_, _mask_)

template<unsigned int Size> inline int Mask<Size>::shiftMask() const
{
    return _mm256_movemask_epi8(dataI());
}

template<> template<> inline Mask<2>::Mask(const Mask<4> &x) {
    k = _mm256_unpacklo_ps(x.data(), x.data());
}
template<> template<> inline Mask<2>::Mask(const Mask<8> &x) {
    _M256I tmp = _mm256_unpacklo_epi16(x.dataI(), x.dataI());
    k = _mm256_castsi128_ps(_mm256_unpacklo_epi32(tmp, tmp));
}
template<> template<> inline Mask<2>::Mask(const Mask<16> &x) {
    _M256I tmp = _mm256_unpacklo_epi8(x.dataI(), x.dataI());
    tmp = _mm256_unpacklo_epi16(tmp, tmp);
    k = _mm256_castsi128_ps(_mm256_unpacklo_epi32(tmp, tmp));
}
template<> template<> inline Mask<4>::Mask(const Mask<2> &x) {
    k = _mm256_castsi128_ps(_mm256_packs_epi16(x.dataI(), _mm256_setzero_si128()));
}
template<> template<> inline Mask<4>::Mask(const Mask<8> &x) {
    k = _mm256_castsi128_ps(_mm256_unpacklo_epi16(x.dataI(), x.dataI()));
}
template<> template<> inline Mask<4>::Mask(const Mask<16> &x) {
    _M256I tmp = _mm256_unpacklo_epi8(x.dataI(), x.dataI());
    k = _mm256_castsi128_ps(_mm256_unpacklo_epi16(tmp, tmp));
}
template<> template<> inline Mask<8>::Mask(const Mask<2> &x) {
    _M256I tmp = _mm256_packs_epi16(x.dataI(), x.dataI());
    k = _mm256_castsi128_ps(_mm256_packs_epi16(tmp, tmp));
}
template<> template<> inline Mask<8>::Mask(const Mask<4> &x) {
    k = _mm256_castsi128_ps(_mm256_packs_epi16(x.dataI(), x.dataI()));
}
template<> template<> inline Mask<8>::Mask(const Mask<16> &x) {
    k = _mm256_castsi128_ps(_mm256_unpacklo_epi8(x.dataI(), x.dataI()));
}

template<> inline void Mask< 4>::expand(Mask<2> *x) const {
    x[0].k = _mm256_unpacklo_ps(data(), data());
    x[1].k = _mm256_unpackhi_ps(data(), data());
}
template<> inline void Mask< 8>::expand(Mask<4> *x) const {
    x[0].k = _mm256_castsi128_ps(_mm256_unpacklo_epi16(dataI(), dataI()));
    x[1].k = _mm256_castsi128_ps(_mm256_unpackhi_epi16(dataI(), dataI()));
}
template<> inline void Mask<16>::expand(Mask<8> *x) const {
    x[0].k = _mm256_castsi128_ps(_mm256_unpacklo_epi8 (dataI(), dataI()));
    x[1].k = _mm256_castsi128_ps(_mm256_unpackhi_epi8 (dataI(), dataI()));
}

template<> inline int Mask< 2>::toInt() const { return _mm256_movemask_pd(dataD()); }
template<> inline int Mask< 4>::toInt() const { return _mm256_movemask_ps(data ()); }
template<> inline int Mask< 8>::toInt() const { return _mm256_movemask_epi8(_mm256_packs_epi16(dataI(), _mm256_setzero_si128())); }
template<> inline int Mask<16>::toInt() const { return _mm256_movemask_epi8(dataI()); }

template<> inline bool Mask< 2>::operator[](int index) const { return toInt() & (1 << index); }
template<> inline bool Mask< 4>::operator[](int index) const { return toInt() & (1 << index); }
template<> inline bool Mask< 8>::operator[](int index) const { return shiftMask() & (1 << 2 * index); }
template<> inline bool Mask<16>::operator[](int index) const { return toInt() & (1 << index); }

template<> inline int Mask<2>::count() const
{
    int mask = _mm256_movemask_pd(dataD());
    return (mask & 1) + (mask >> 1);
}

template<> inline int Mask<4>::count() const
{
//X     int tmp = _mm256_movemask_ps(data());
//X     tmp = (tmp & 5) + ((tmp >> 1) & 5);
//X     return (tmp & 3) + ((tmp >> 2) & 3);
    _M256I x = _mm256_srli_epi32(dataI(), 31);
    x = _mm256_add_epi32(x, _mm256_shuffle_epi32(x, _MM_SHUFFLE(0, 1, 2, 3)));
    x = _mm256_add_epi32(x, _mm256_shufflelo_epi16(x, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm256_cvtsi128_si32(x);
}

template<> inline int Mask<8>::count() const
{
//X     int tmp = _mm256_movemask_epi8(dataI());
//X     tmp = (tmp & 0x1111) + ((tmp >> 2) & 0x1111);
//X     tmp = (tmp & 0x0303) + ((tmp >> 4) & 0x0303);
//X     return (tmp & 0x000f) + ((tmp >> 8) & 0x000f);
    _M256I x = _mm256_srli_epi16(dataI(), 15);
    x = _mm256_add_epi16(x, _mm256_shuffle_epi32(x, _MM_SHUFFLE(0, 1, 2, 3)));
    x = _mm256_add_epi16(x, _mm256_shufflelo_epi16(x, _MM_SHUFFLE(0, 1, 2, 3)));
    x = _mm256_add_epi16(x, _mm256_shufflelo_epi16(x, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm256_extract_epi16(x, 0);
}

template<> inline int Mask<16>::count() const
{
    int tmp = _mm256_movemask_epi8(dataI());
    tmp = (tmp & 0x5555) + ((tmp >> 1) & 0x5555);
    tmp = (tmp & 0x3333) + ((tmp >> 2) & 0x3333);
    tmp = (tmp & 0x0f0f) + ((tmp >> 4) & 0x0f0f);
    return (tmp & 0x00ff) + ((tmp >> 8) & 0x00ff);
}


class Float8Mask
{
    enum {
        PartialSize = 4,
        VectorSize = 8
    };
    public:
        FREE_STORE_OPERATORS_ALIGNED(16)
        inline Float8Mask() {}
        inline Float8Mask(const M256 &x) : k(x) {}
        inline explicit Float8Mask(VectorSpecialInitializerZero::ZEnum) {
            k[0] = _mm256_setzero_ps();
            k[1] = _mm256_setzero_ps();
        }
        inline explicit Float8Mask(VectorSpecialInitializerOne::OEnum) {
            k[0] = _mm256_setallone_ps();
            k[1] = _mm256_setallone_ps();
        }
        inline explicit Float8Mask(bool b) {
            const __m256 tmp = b ? _mm256_setallone_ps() : _mm256_setzero_ps();
            k[0] = tmp;
            k[1] = tmp;
        }
        inline Float8Mask(const Mask<VectorSize> &a) {
            k[0] = _mm256_castsi128_ps(_mm256_unpacklo_epi16(a.dataI(), a.dataI()));
            k[1] = _mm256_castsi128_ps(_mm256_unpackhi_epi16(a.dataI(), a.dataI()));
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
            r.k[0] = _mm256_and_ps(k[0], rhs.k[0]);
            r.k[1] = _mm256_and_ps(k[1], rhs.k[1]);
            return r;
        }
        inline Float8Mask operator& (const Float8Mask &rhs) const {
            Float8Mask r;
            r.k[0] = _mm256_and_ps(k[0], rhs.k[0]);
            r.k[1] = _mm256_and_ps(k[1], rhs.k[1]);
            return r;
        }
        inline Float8Mask operator||(const Float8Mask &rhs) const {
            Float8Mask r;
            r.k[0] = _mm256_or_ps(k[0], rhs.k[0]);
            r.k[1] = _mm256_or_ps(k[1], rhs.k[1]);
            return r;
        }
        inline Float8Mask operator| (const Float8Mask &rhs) const {
            Float8Mask r;
            r.k[0] = _mm256_or_ps(k[0], rhs.k[0]);
            r.k[1] = _mm256_or_ps(k[1], rhs.k[1]);
            return r;
        }
        inline Float8Mask operator^ (const Float8Mask &rhs) const {
            Float8Mask r;
            r.k[0] = _mm256_xor_ps(k[0], rhs.k[0]);
            r.k[1] = _mm256_xor_ps(k[1], rhs.k[1]);
            return r;
        }
        inline Float8Mask operator!() const {
            Float8Mask r;
            r.k[0] = _mm256_andnot_ps(k[0], _mm256_setallone_ps());
            r.k[1] = _mm256_andnot_ps(k[1], _mm256_setallone_ps());
            return r;
        }
        inline Float8Mask &operator&=(const Float8Mask &rhs) {
            k[0] = _mm256_and_ps(k[0], rhs.k[0]);
            k[1] = _mm256_and_ps(k[1], rhs.k[1]);
            return *this;
        }
        inline Float8Mask &operator|=(const Float8Mask &rhs) {
            k[0] = _mm256_or_ps (k[0], rhs.k[0]);
            k[1] = _mm256_or_ps (k[1], rhs.k[1]);
            return *this;
        }

        inline bool isFull () const {
            const _M256 tmp = _mm256_and_ps(k[0], k[1]);
#ifdef VC_USE_PTEST
            return _mm256_testc_si128(_mm256_castps_si128(tmp), _mm256_setallone_si128());
#else
            return _mm256_movemask_ps(tmp) == 0xf;
            //_mm256_movemask_ps(k[0]) == 0xf &&
            //_mm256_movemask_ps(k[1]) == 0xf;
#endif
        }
        inline bool isEmpty() const {
            const _M256 tmp = _mm256_or_ps(k[0], k[1]);
#ifdef VC_USE_PTEST
            return _mm256_testz_si128(_mm256_castps_si128(tmp), _mm256_castps_si128(tmp));
#else
            return _mm256_movemask_ps(tmp) == 0x0;
            //_mm256_movemask_ps(k[0]) == 0x0 &&
            //_mm256_movemask_ps(k[1]) == 0x0;
#endif
        }
        inline bool isMix() const {
#ifdef VC_USE_PTEST
            return _mm256_test_mix_ones_zeros(_mm256_castps_si128(k[0]), _mm256_castps_si128(k[0])) &&
            _mm256_test_mix_ones_zeros(_mm256_castps_si128(k[1]), _mm256_castps_si128(k[1]));
#else
            const int tmp = _mm256_movemask_ps(k[0]) + _mm256_movemask_ps(k[1]);
            return tmp > 0x0 && tmp < (0xf + 0xf);
#endif
        }

        inline operator bool() const { return isFull(); }

        inline int shiftMask() const {
            return (_mm256_movemask_ps(k[1]) << 4) + _mm256_movemask_ps(k[0]);
        }
        inline int toInt() const { return (_mm256_movemask_ps(k[1]) << 4) + _mm256_movemask_ps(k[0]); }

        inline const M256 &data () const { return k; }

        inline bool operator[](int index) const {
            return (toInt() & (1 << index)) != 0;
        }

        inline int count() const {
//X             int tmp1 = _mm256_movemask_ps(k[0]);
//X             int tmp2 = _mm256_movemask_ps(k[1]);
//X             tmp1 = (tmp1 & 5) + ((tmp1 >> 1) & 5);
//X             tmp2 = (tmp2 & 5) + ((tmp2 >> 1) & 5);
//X             return (tmp1 & 3) + (tmp2 & 3) + ((tmp1 >> 2) & 3) + ((tmp2 >> 2) & 3);
            _M256I x = _mm256_add_epi32(_mm256_srli_epi32(_mm256_castps_si128(k[0]), 31),
                                     _mm256_srli_epi32(_mm256_castps_si128(k[1]), 31));
            x = _mm256_add_epi32(x, _mm256_shuffle_epi32(x, _MM_SHUFFLE(0, 1, 2, 3)));
            x = _mm256_add_epi32(x, _mm256_shufflelo_epi16(x, _MM_SHUFFLE(1, 0, 3, 2)));
            return _mm256_cvtsi128_si32(x);
        }

        int firstOne() const;

    private:
        M256 k;
};

template<unsigned int Size> inline int Mask<Size>::firstOne() const
{
    const int mask = toInt();
    int bit;
    __asm__("bsf %1,%0" : "=&r"(bit) : "r"(mask));
    return bit;
}
inline int Float8Mask::firstOne() const
{
    const int mask = toInt();
    int bit;
    __asm__("bsf %1,%0" : "=&r"(bit) : "r"(mask));
    return bit;
}

template<unsigned int VectorSize>
inline Mask<VectorSize>::Mask(const Float8Mask &m)
    : k(_mm256_castsi128_ps(_mm256_packs_epi32(_mm256_castps_si128(m.data()[0]), _mm256_castps_si128(m.data()[1])))) {}

class Float8GatherMask
{
    public:
#ifdef VC_GATHER_SET
        Float8GatherMask(const Mask<8u> &k)   : smallMask(k), bigMask(k), mask(k.toInt()) {}
        Float8GatherMask(const Float8Mask &k) : smallMask(k), bigMask(k), mask(k.toInt()) {}
        const __m256 dataIndex() const { return smallMask.data(); }
        const M256 data() const { return bigMask.data(); }
#else
        Float8GatherMask(const Mask<8u> &k)   : mask(k.toInt()) {}
        Float8GatherMask(const Float8Mask &k) : mask(k.toInt()) {}
#endif
        int toInt() const { return mask; }
    private:
#ifdef VC_GATHER_SET
        const Mask<8u> smallMask;
        const Float8Mask bigMask;
#endif
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
//X     for (int _avx_vector_foreach_inner = 1, ForeachScope _avx_vector_foreach_scope(mask.toInt()), int it = _avx_vector_foreach_scope.bit(); _avx_vector_foreach_inner; --_avx_vector_foreach_inner)
//X     for (int _avx_vector_foreach_mask = (mask).toInt(), int _avx_vector_foreach_it = _avx_bitscan(mask.toInt());
//X             _avx_vector_foreach_it > 0;
//X             _avx_vector_foreach_it = _avx_bitscan_initialized(_avx_vector_foreach_it, mask.data()))
//X         for (int _avx_vector_foreach_inner = 1, it = _avx_vector_foreach_it; _avx_vector_foreach_inner; --_avx_vector_foreach_inner)

} // namespace AVX
} // namespace Vc

#endif // AVX_MASK_H
