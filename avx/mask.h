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

#ifndef AVX_MASK_H
#define AVX_MASK_H

#include "intrinsics.h"

namespace Vc
{
namespace AVX
{

template<unsigned int VectorSize, size_t RegisterWidth> class Mask;

template<unsigned int VectorSize> class Mask<VectorSize, 32u>
{
    friend class Mask<4u, 32u>; // double_v
    friend class Mask<8u, 32u>; // float_v, (u)int_v
    friend class Mask<8u, 16u>; // (u)short_v
    friend class Mask<16u, 16u>; // (u)char_v
    public:
        FREE_STORE_OPERATORS_ALIGNED(32)
        inline Mask() {}
        inline Mask(const __m256  &x) : k(x) {}
        inline Mask(const __m256d &x) : k(_mm256_castpd_ps(x)) {}
        inline Mask(const __m256i &x) : k(_mm256_castsi256_ps(x)) {}
        inline explicit Mask(VectorSpecialInitializerZero::ZEnum) : k(_mm256_setzero_ps()) {}
        inline explicit Mask(VectorSpecialInitializerOne::OEnum) : k(_mm256_setallone_ps()) {}
        inline explicit Mask(bool b) : k(b ? _mm256_setallone_ps() : _mm256_setzero_ps()) {}
        inline Mask(const Mask &rhs) : k(rhs.k) {}
        inline Mask(const Mask<VectorSize, 16u> &rhs) : k(
                _mm256_insertf128_ps(
                    avx_cast<__m256>(_mm_unpacklo_epi16(rhs.data(), rhs.data())),
                    avx_cast<__m128>(_mm_unpackhi_epi16(rhs.data(), rhs.data())),
                    1)) {}
        inline Mask(const Mask<VectorSize / 2> *a)
          : k(_mm256_castsi256_ps(_mm256_packs_epi16(a[0].dataI(), a[1].dataI()))) {}

        template<unsigned int OtherSize> explicit Mask(const Mask<OtherSize> &x);
//X         {
//X             _M256I tmp = x.dataI();
//X             if (OtherSize < VectorSize) {
//X                 tmp = _mm256_packs_epi16(tmp, _mm256_setzero_si256());
//X                 if (VectorSize / OtherSize >= 4u) { tmp = _mm256_packs_epi16(tmp, _mm256_setzero_si256()); }
//X                 if (VectorSize / OtherSize >= 8u) { tmp = _mm256_packs_epi16(tmp, _mm256_setzero_si256()); }
//X             } else if (OtherSize > VectorSize) {
//X                 tmp = _mm256_unpacklo_epi8(tmp, tmp);
//X                 if (OtherSize / VectorSize >= 4u) { tmp = _mm256_unpacklo_epi8(tmp, tmp); }
//X                 if (OtherSize / VectorSize >= 8u) { tmp = _mm256_unpacklo_epi8(tmp, tmp); }
//X             }
//X             k = _mm256_castsi256_ps(tmp);
//X         }

        void expand(Mask<VectorSize / 2> *x) const;

        inline bool operator==(const Mask &rhs) const { return 0 != _mm256_testc_ps(k, rhs.k); }
        inline bool operator!=(const Mask &rhs) const { return 0 == _mm256_testc_ps(k, rhs.k); }

        inline Mask operator&&(const Mask &rhs) const { return _mm256_and_ps(k, rhs.k); }
        inline Mask operator& (const Mask &rhs) const { return _mm256_and_ps(k, rhs.k); }
        inline Mask operator||(const Mask &rhs) const { return _mm256_or_ps (k, rhs.k); }
        inline Mask operator| (const Mask &rhs) const { return _mm256_or_ps (k, rhs.k); }
        inline Mask operator^ (const Mask &rhs) const { return _mm256_xor_ps(k, rhs.k); }
        inline Mask operator!() const { return _mm256_andnot_ps(data(), _mm256_setallone_ps()); }

        inline Mask &operator&=(const Mask &rhs) { k = _mm256_and_ps(k, rhs.k); return *this; }
        inline Mask &operator|=(const Mask &rhs) { k = _mm256_or_ps (k, rhs.k); return *this; }

        // no need for expression template optimizations because cmp(n)eq for floats are not bitwise
        // compares
        inline bool isFull () const { return 0 != _mm256_testc_ps(k, _mm256_setallone_ps()); }
        inline bool isEmpty() const { return 0 != _mm256_testz_ps(k, k); }
        inline bool isMix  () const { return 0 != _mm256_testnzc_ps(k, _mm256_setallone_ps()); }

        inline operator bool() const { return isFull(); }

        inline int shiftMask() const CONST;
        int toInt() const CONST;

        inline _M256  data () const { return k; }
        inline _M256I dataI() const { return _mm256_castps_si256(k); }
        inline _M256D dataD() const { return _mm256_castps_pd(k); }

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

template<unsigned int VectorSize> class Mask<VectorSize, 16u>
{
    friend class Mask<4u, 32u>; // double_v
    friend class Mask<8u, 32u>; // float_v, (u)int_v
    friend class Mask<8u, 16u>; // (u)short_v
    friend class Mask<16u, 16u>; // (u)char_v
    public:
        FREE_STORE_OPERATORS_ALIGNED(16)
        inline Mask() {}
        inline Mask(const __m128  &x) : k(x) {}
        inline Mask(const __m128d &x) : k(_mm_castpd_ps(x)) {}
        inline Mask(const __m128i &x) : k(_mm_castsi128_ps(x)) {}
        inline explicit Mask(VectorSpecialInitializerZero::ZEnum) : k(_mm_setzero_ps()) {}
        inline explicit Mask(VectorSpecialInitializerOne::OEnum) : k(_mm_setallone_ps()) {}
        inline explicit Mask(bool b) : k(b ? _mm_setallone_ps() : _mm_setzero_ps()) {}
        inline Mask(const Mask &rhs) : k(rhs.k) {}
        inline Mask(const Mask<VectorSize, 32u> &rhs) : k(avx_cast<__m128>(
                _mm_packs_epi32(avx_cast<__m128i>(rhs.data()), _mm256_extractf128_si256(rhs.dataI(), 1)))) {}
        inline Mask(const Mask<VectorSize / 2, 16u> *a) : k(avx_cast<__m128>(
                _mm_packs_epi16(a[0].dataI(), a[1].dataI()))) {}

        inline bool operator==(const Mask &rhs) const { return 0 != _mm_testc_si128(dataI(), rhs.dataI()); }
        inline bool operator!=(const Mask &rhs) const { return 0 == _mm_testc_si128(dataI(), rhs.dataI()); }

        inline Mask operator&&(const Mask &rhs) const { return _mm_and_ps(k, rhs.k); }
        inline Mask operator& (const Mask &rhs) const { return _mm_and_ps(k, rhs.k); }
        inline Mask operator||(const Mask &rhs) const { return _mm_or_ps (k, rhs.k); }
        inline Mask operator| (const Mask &rhs) const { return _mm_or_ps (k, rhs.k); }
        inline Mask operator^ (const Mask &rhs) const { return _mm_xor_ps(k, rhs.k); }
        inline Mask operator!() const { return _mm_andnot_ps(data(), _mm_setallone_ps()); }

        inline Mask &operator&=(const Mask &rhs) { k = _mm_and_ps(k, rhs.k); return *this; }
        inline Mask &operator|=(const Mask &rhs) { k = _mm_or_ps (k, rhs.k); return *this; }

        // TODO: use expression templates to optimize (v1 == v2).isFull() and friends
        inline bool isFull () const { return 0 != _mm_testc_si128(dataI(), _mm_setallone_si128()); }
        inline bool isEmpty() const { return 0 != _mm_testz_si128(dataI(), dataI()); }
        inline bool isMix  () const { return 0 != _mm_testnzc_si128(dataI(), _mm_setallone_si128()); }

        inline operator bool() const { return isFull(); }

        inline int shiftMask() const CONST;
        int toInt() const CONST;

        inline __m128  data () const { return k; }
        inline __m128i dataI() const { return avx_cast<__m128i>(k); }
        inline __m128d dataD() const { return avx_cast<__m128d>(k); }

        bool operator[](int index) const;

        int count() const;
        int firstOne() const;

    private:
        __m128 k;
};

struct ForeachHelper
{
    size_t mask;
    bool brk;
    inline ForeachHelper(size_t _mask) : mask(_mask), brk(false) {}
    inline bool outer() const { return mask != 0; }
    inline bool inner() { return (brk = !brk); }
    inline size_t next() {
#if defined(__GNUC__) && !defined(VC_NO_INLINE_ASM)
        const size_t bit = __builtin_ctzl(mask);
        __asm__("btr %1,%0" : "+r"(mask) : "r"(bit));
#else
        const size_t bit = _bit_scan_forward(mask);
        mask &= ~(1 << bit);
#endif
        return bit;
    }
};

#define Vc_foreach_bit(_it_, _mask_) \
    for (Vc::AVX::ForeachHelper _Vc_foreach_bit_helper((_mask_).toInt()); _Vc_foreach_bit_helper.outer(); ) \
        for (_it_ = _Vc_foreach_bit_helper.next(); _Vc_foreach_bit_helper.inner(); )

#define foreach_bit(_it_, _mask_) Vc_foreach_bit(_it_, _mask_)

template<unsigned int Size> inline int Mask<Size, 32u>::shiftMask() const
{
    return _mm256_movemask_epi8(dataI());
}
template<unsigned int Size> inline int Mask<Size, 16u>::shiftMask() const
{
    return _mm_movemask_epi8(dataI());
}

template<> inline int Mask< 4, 32>::toInt() const { return _mm256_movemask_pd(dataD()); }
template<> inline int Mask< 8, 32>::toInt() const { return _mm256_movemask_ps(data ()); }
template<> inline int Mask< 8, 16>::toInt() const { return _mm_movemask_epi8(_mm_packs_epi16(dataI(), _mm_setzero_si256())); }
template<> inline int Mask<16, 16>::toInt() const { return _mm_movemask_epi8(dataI()); }

template<> inline bool Mask< 2>::operator[](int index) const { return toInt() & (1 << index); }
template<> inline bool Mask< 4>::operator[](int index) const { return toInt() & (1 << index); }
template<> inline bool Mask< 8>::operator[](int index) const { return shiftMask() & (1 << 2 * index); }
template<> inline bool Mask<16>::operator[](int index) const { return toInt() & (1 << index); }

template<unsigned int Size, size_t Width> inline int Mask<Size, Width>::count() const
{
    return _mm_popcnt_u32(toInt());
}

template<unsigned int Size, size_t Width> inline int Mask<Size, Width>::firstOne() const
{
    return _bit_scan_forward(toInt());
}

} // namespace AVX
} // namespace Vc

#endif // AVX_MASK_H
