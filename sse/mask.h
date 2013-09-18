/*  This file is part of the Vc library.

    Copyright (C) 2009-2012 Matthias Kretz <kretz@kde.org>

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
#include "../common/maskentry.h"
#include "macros.h"

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)

template<unsigned int Size1> struct MaskHelper
{
#ifdef VC_USE_PTEST
    static Vc_ALWAYS_INLINE Vc_CONST bool cmpeq (__m128 x, __m128 y) {
        return 0 != _mm_testc_si128(_mm_castps_si128(x), _mm_castps_si128(y));
    }
    static Vc_ALWAYS_INLINE Vc_CONST bool cmpneq(__m128 x, __m128 y) {
        return 0 == _mm_testc_si128(_mm_castps_si128(x), _mm_castps_si128(y));
    }
#endif
};
#ifndef VC_USE_PTEST
template<> struct MaskHelper<2> {
    static Vc_ALWAYS_INLINE Vc_CONST bool cmpeq (_M128 k1, _M128 k2) { return _mm_movemask_pd(_mm_castps_pd(k1)) == _mm_movemask_pd(_mm_castps_pd(k2)); }
    static Vc_ALWAYS_INLINE Vc_CONST bool cmpneq(_M128 k1, _M128 k2) { return _mm_movemask_pd(_mm_castps_pd(k1)) != _mm_movemask_pd(_mm_castps_pd(k2)); }
};
template<> struct MaskHelper<4> {
    static Vc_ALWAYS_INLINE Vc_CONST bool cmpeq (_M128 k1, _M128 k2) { return _mm_movemask_ps(k1) == _mm_movemask_ps(k2); }
    static Vc_ALWAYS_INLINE Vc_CONST bool cmpneq(_M128 k1, _M128 k2) { return _mm_movemask_ps(k1) != _mm_movemask_ps(k2); }
};
template<> struct MaskHelper<8> {
    static Vc_ALWAYS_INLINE Vc_CONST bool cmpeq (_M128 k1, _M128 k2) { return _mm_movemask_epi8(_mm_castps_si128(k1)) == _mm_movemask_epi8(_mm_castps_si128(k2)); }
    static Vc_ALWAYS_INLINE Vc_CONST bool cmpneq(_M128 k1, _M128 k2) { return _mm_movemask_epi8(_mm_castps_si128(k1)) != _mm_movemask_epi8(_mm_castps_si128(k2)); }
};
#endif

class Float8Mask;
template<unsigned int VectorSize> class Mask
{
    friend class Mask<2u>;
    friend class Mask<4u>;
    friend class Mask<8u>;
    friend class Mask<16u>;
    friend class Float8Mask;
    typedef Common::MaskBool<16 / VectorSize> MaskBool;
    typedef Common::VectorMemoryUnion<__m128, MaskBool> Storage;
    public:
        FREE_STORE_OPERATORS_ALIGNED(16)

        static constexpr size_t Size = VectorSize;

        // abstracts the way Masks are passed to functions, it can easily be changed to const ref here
        // Also Float8Mask requires const ref on MSVC 32bit.
#if defined VC_MSVC && defined _WIN32
        typedef const Mask<VectorSize> &Argument;
#else
        typedef Mask<VectorSize> Argument;
#endif

        Vc_ALWAYS_INLINE Mask() {}
        Vc_ALWAYS_INLINE Mask(const __m128  &x) : d(x) {}
        Vc_ALWAYS_INLINE Mask(const __m128d &x) : d(_mm_castpd_ps(x)) {}
        Vc_ALWAYS_INLINE Mask(const __m128i &x) : d(_mm_castsi128_ps(x)) {}
        Vc_ALWAYS_INLINE explicit Mask(VectorSpecialInitializerZero::ZEnum) : d(_mm_setzero_ps()) {}
        Vc_ALWAYS_INLINE explicit Mask(VectorSpecialInitializerOne::OEnum) : d(_mm_setallone_ps()) {}
        Vc_ALWAYS_INLINE explicit Mask(bool b) : d(b ? _mm_setallone_ps() : _mm_setzero_ps()) {}
        Vc_ALWAYS_INLINE Mask(const Mask &rhs) : d(rhs.d) {}
        Vc_ALWAYS_INLINE Mask(const Mask<VectorSize / 2> *a)
          : d(_mm_castsi128_ps(_mm_packs_epi16(a[0].dataI(), a[1].dataI()))) {}
        Vc_ALWAYS_INLINE explicit Mask(const Float8Mask &m);

        template<unsigned int OtherSize> Vc_ALWAYS_INLINE_L explicit Mask(const Mask<OtherSize> &x) Vc_ALWAYS_INLINE_R;
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
//X             d.v() = _mm_castsi128_ps(tmp);
//X         }

        inline void expand(Mask<VectorSize / 2> *x) const;

        Vc_ALWAYS_INLINE Vc_PURE bool operator==(const Mask &rhs) const { return MaskHelper<VectorSize>::cmpeq (d.v(), rhs.d.v()); }
        Vc_ALWAYS_INLINE Vc_PURE bool operator!=(const Mask &rhs) const { return MaskHelper<VectorSize>::cmpneq(d.v(), rhs.d.v()); }

        Vc_ALWAYS_INLINE Vc_PURE Mask operator!() const { return _mm_andnot_si128(dataI(), _mm_setallone_si128()); }

        Vc_ALWAYS_INLINE Mask &operator&=(const Mask &rhs) { d.v() = _mm_and_ps(d.v(), rhs.d.v()); return *this; }
        Vc_ALWAYS_INLINE Mask &operator|=(const Mask &rhs) { d.v() = _mm_or_ps (d.v(), rhs.d.v()); return *this; }
        Vc_ALWAYS_INLINE Mask &operator^=(const Mask &rhs) { d.v() = _mm_xor_ps(d.v(), rhs.d.v()); return *this; }

        Vc_ALWAYS_INLINE Vc_PURE bool isFull () const { return
#ifdef VC_USE_PTEST
            _mm_testc_si128(dataI(), _mm_setallone_si128()); // return 1 if (0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff) == (~0 & d.v())
#else
            _mm_movemask_epi8(dataI()) == 0xffff;
#endif
        }
        Vc_ALWAYS_INLINE Vc_PURE bool isNotEmpty() const { return
#ifdef VC_USE_PTEST
            0 == _mm_testz_si128(dataI(), dataI()); // return 1 if (0, 0, 0, 0) == (d.v() & d.v())
#else
            _mm_movemask_epi8(dataI()) != 0x0000;
#endif
        }
        Vc_ALWAYS_INLINE Vc_PURE bool isEmpty() const { return
#ifdef VC_USE_PTEST
            0 != _mm_testz_si128(dataI(), dataI()); // return 1 if (0, 0, 0, 0) == (d.v() & d.v())
#else
            _mm_movemask_epi8(dataI()) == 0x0000;
#endif
        }
        Vc_ALWAYS_INLINE Vc_PURE bool isMix() const {
#ifdef VC_USE_PTEST
            return _mm_test_mix_ones_zeros(dataI(), _mm_setallone_si128());
#else
            const int tmp = _mm_movemask_epi8(dataI());
            return tmp != 0 && (tmp ^ 0xffff) != 0;
#endif
        }

#ifndef VC_NO_AUTOMATIC_BOOL_FROM_MASK
        Vc_ALWAYS_INLINE Vc_PURE operator bool() const { return isFull(); }
#endif

        Vc_ALWAYS_INLINE_L Vc_PURE_L int shiftMask() const Vc_ALWAYS_INLINE_R Vc_PURE_R;

        Vc_ALWAYS_INLINE_L Vc_PURE_L int toInt() const Vc_ALWAYS_INLINE_R Vc_PURE_R;

        Vc_ALWAYS_INLINE Vc_PURE _M128  data () const { return d.v(); }
        Vc_ALWAYS_INLINE Vc_PURE _M128I dataI() const { return _mm_castps_si128(d.v()); }
        Vc_ALWAYS_INLINE Vc_PURE _M128D dataD() const { return _mm_castps_pd(d.v()); }

        template<unsigned int OtherSize> Vc_ALWAYS_INLINE Vc_PURE Mask<OtherSize> cast() const { return Mask<OtherSize>(d.v()); }

        Vc_ALWAYS_INLINE MaskBool &operator[](size_t index) { return d.m(index); }
        Vc_ALWAYS_INLINE_L Vc_PURE_L bool operator[](size_t index) const Vc_ALWAYS_INLINE_R Vc_PURE_R;

        Vc_ALWAYS_INLINE_L Vc_PURE_L unsigned int count() const Vc_ALWAYS_INLINE_R Vc_PURE_R;

        /**
         * Returns the index of the first one in the mask.
         *
         * The return value is undefined if the mask is empty.
         */
        Vc_ALWAYS_INLINE_L Vc_PURE_L unsigned int firstOne() const Vc_ALWAYS_INLINE_R Vc_PURE_R;

    private:
#ifdef VC_COMPILE_BENCHMARKS
    public:
#endif
        Storage d;
};

template<unsigned int Size> Vc_ALWAYS_INLINE Vc_PURE int Mask<Size>::shiftMask() const
{
    return _mm_movemask_epi8(dataI());
}

template<> template<> Vc_ALWAYS_INLINE Mask<2>::Mask(const Mask<4> &x) {
    d.v() = _mm_unpacklo_ps(x.data(), x.data());
}
template<> template<> Vc_ALWAYS_INLINE Mask<2>::Mask(const Mask<8> &x) {
    _M128I tmp = _mm_unpacklo_epi16(x.dataI(), x.dataI());
    d.v() = _mm_castsi128_ps(_mm_unpacklo_epi32(tmp, tmp));
}
template<> template<> Vc_ALWAYS_INLINE Mask<2>::Mask(const Mask<16> &x) {
    _M128I tmp = _mm_unpacklo_epi8(x.dataI(), x.dataI());
    tmp = _mm_unpacklo_epi16(tmp, tmp);
    d.v() = _mm_castsi128_ps(_mm_unpacklo_epi32(tmp, tmp));
}
template<> template<> Vc_ALWAYS_INLINE Mask<4>::Mask(const Mask<2> &x) {
    d.v() = _mm_castsi128_ps(_mm_packs_epi16(x.dataI(), _mm_setzero_si128()));
}
template<> template<> Vc_ALWAYS_INLINE Mask<4>::Mask(const Mask<8> &x) {
    d.v() = _mm_castsi128_ps(_mm_unpacklo_epi16(x.dataI(), x.dataI()));
}
template<> template<> Vc_ALWAYS_INLINE Mask<4>::Mask(const Mask<16> &x) {
    _M128I tmp = _mm_unpacklo_epi8(x.dataI(), x.dataI());
    d.v() = _mm_castsi128_ps(_mm_unpacklo_epi16(tmp, tmp));
}
template<> template<> Vc_ALWAYS_INLINE Mask<8>::Mask(const Mask<2> &x) {
    _M128I tmp = _mm_packs_epi16(x.dataI(), x.dataI());
    d.v() = _mm_castsi128_ps(_mm_packs_epi16(tmp, tmp));
}
template<> template<> Vc_ALWAYS_INLINE Mask<8>::Mask(const Mask<4> &x) {
    d.v() = _mm_castsi128_ps(_mm_packs_epi16(x.dataI(), x.dataI()));
}
template<> template<> Vc_ALWAYS_INLINE Mask<8>::Mask(const Mask<16> &x) {
    d.v() = _mm_castsi128_ps(_mm_unpacklo_epi8(x.dataI(), x.dataI()));
}

template<> inline void Mask< 4>::expand(Mask<2> *x) const {
    x[0].d.v() = _mm_unpacklo_ps(data(), data());
    x[1].d.v() = _mm_unpackhi_ps(data(), data());
}
template<> inline void Mask< 8>::expand(Mask<4> *x) const {
    x[0].d.v() = _mm_castsi128_ps(_mm_unpacklo_epi16(dataI(), dataI()));
    x[1].d.v() = _mm_castsi128_ps(_mm_unpackhi_epi16(dataI(), dataI()));
}
template<> inline void Mask<16>::expand(Mask<8> *x) const {
    x[0].d.v() = _mm_castsi128_ps(_mm_unpacklo_epi8 (dataI(), dataI()));
    x[1].d.v() = _mm_castsi128_ps(_mm_unpackhi_epi8 (dataI(), dataI()));
}

template<> Vc_ALWAYS_INLINE Vc_PURE int Mask< 2>::toInt() const { return _mm_movemask_pd(dataD()); }
template<> Vc_ALWAYS_INLINE Vc_PURE int Mask< 4>::toInt() const { return _mm_movemask_ps(data ()); }
template<> Vc_ALWAYS_INLINE Vc_PURE int Mask< 8>::toInt() const { return _mm_movemask_epi8(_mm_packs_epi16(dataI(), _mm_setzero_si128())); }
template<> Vc_ALWAYS_INLINE Vc_PURE int Mask<16>::toInt() const { return _mm_movemask_epi8(dataI()); }

template<> Vc_ALWAYS_INLINE Vc_PURE bool Mask< 2>::operator[](size_t index) const { return toInt() & (1 << index); }
template<> Vc_ALWAYS_INLINE Vc_PURE bool Mask< 4>::operator[](size_t index) const { return toInt() & (1 << index); }
template<> Vc_ALWAYS_INLINE Vc_PURE bool Mask< 8>::operator[](size_t index) const { return shiftMask() & (1 << 2 * index); }
template<> Vc_ALWAYS_INLINE Vc_PURE bool Mask<16>::operator[](size_t index) const { return toInt() & (1 << index); }

template<> Vc_ALWAYS_INLINE Vc_PURE unsigned int Mask<2>::count() const
{
    int mask = _mm_movemask_pd(dataD());
    return (mask & 1) + (mask >> 1);
}

template<> Vc_ALWAYS_INLINE Vc_PURE unsigned int Mask<4>::count() const
{
#ifdef VC_IMPL_POPCNT
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

template<> Vc_ALWAYS_INLINE Vc_PURE unsigned int Mask<8>::count() const
{
#ifdef VC_IMPL_POPCNT
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

template<> Vc_ALWAYS_INLINE Vc_PURE unsigned int Mask<16>::count() const
{
    int tmp = _mm_movemask_epi8(dataI());
#ifdef VC_IMPL_POPCNT
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
    enum PrivateConstants {
        PartialSize = 4,
        VectorSize = 8
    };
    typedef Common::MaskBool<32 / VectorSize> MaskBool;
    typedef Common::VectorMemoryUnion<M256, MaskBool> Storage;
    public:
        FREE_STORE_OPERATORS_ALIGNED(16)

        static constexpr size_t Size = VectorSize;

        // abstracts the way Masks are passed to functions, it can easily be changed to const ref here
        // Also Float8Mask requires const ref on MSVC 32bit.
#if defined VC_MSVC && defined _WIN32
        typedef const Float8Mask & Argument;
#else
        typedef Float8Mask Argument;
#endif

        Vc_ALWAYS_INLINE Float8Mask() {}
        Vc_ALWAYS_INLINE Float8Mask(const M256 &x) : d(x) {}
        Vc_ALWAYS_INLINE explicit Float8Mask(VectorSpecialInitializerZero::ZEnum) {
            d.v()[0] = _mm_setzero_ps();
            d.v()[1] = _mm_setzero_ps();
        }
        Vc_ALWAYS_INLINE explicit Float8Mask(VectorSpecialInitializerOne::OEnum) {
            d.v()[0] = _mm_setallone_ps();
            d.v()[1] = _mm_setallone_ps();
        }
        Vc_ALWAYS_INLINE explicit Float8Mask(bool b) {
            const __m128 tmp = b ? _mm_setallone_ps() : _mm_setzero_ps();
            d.v()[0] = tmp;
            d.v()[1] = tmp;
        }
        Vc_ALWAYS_INLINE Float8Mask(const Mask<VectorSize> &a) {
            d.v()[0] = _mm_castsi128_ps(_mm_unpacklo_epi16(a.dataI(), a.dataI()));
            d.v()[1] = _mm_castsi128_ps(_mm_unpackhi_epi16(a.dataI(), a.dataI()));
        }

        Vc_ALWAYS_INLINE Vc_PURE bool operator==(const Float8Mask &rhs) const {
            return MaskHelper<PartialSize>::cmpeq (d.v()[0], rhs.d.v()[0])
                && MaskHelper<PartialSize>::cmpeq (d.v()[1], rhs.d.v()[1]);
        }
        Vc_ALWAYS_INLINE Vc_PURE bool operator!=(const Float8Mask &rhs) const {
            return MaskHelper<PartialSize>::cmpneq(d.v()[0], rhs.d.v()[0])
                || MaskHelper<PartialSize>::cmpneq(d.v()[1], rhs.d.v()[1]);
        }

        Vc_ALWAYS_INLINE Vc_PURE Float8Mask operator&&(const Float8Mask &rhs) const {
            Float8Mask r;
            r.d.v()[0] = _mm_and_ps(d.v()[0], rhs.d.v()[0]);
            r.d.v()[1] = _mm_and_ps(d.v()[1], rhs.d.v()[1]);
            return r;
        }
        Vc_ALWAYS_INLINE Vc_PURE Float8Mask operator& (const Float8Mask &rhs) const {
            Float8Mask r;
            r.d.v()[0] = _mm_and_ps(d.v()[0], rhs.d.v()[0]);
            r.d.v()[1] = _mm_and_ps(d.v()[1], rhs.d.v()[1]);
            return r;
        }
        Vc_ALWAYS_INLINE Vc_PURE Float8Mask operator||(const Float8Mask &rhs) const {
            Float8Mask r;
            r.d.v()[0] = _mm_or_ps(d.v()[0], rhs.d.v()[0]);
            r.d.v()[1] = _mm_or_ps(d.v()[1], rhs.d.v()[1]);
            return r;
        }
        Vc_ALWAYS_INLINE Vc_PURE Float8Mask operator| (const Float8Mask &rhs) const {
            Float8Mask r;
            r.d.v()[0] = _mm_or_ps(d.v()[0], rhs.d.v()[0]);
            r.d.v()[1] = _mm_or_ps(d.v()[1], rhs.d.v()[1]);
            return r;
        }
        Vc_ALWAYS_INLINE Vc_PURE Float8Mask operator^ (const Float8Mask &rhs) const {
            Float8Mask r;
            r.d.v()[0] = _mm_xor_ps(d.v()[0], rhs.d.v()[0]);
            r.d.v()[1] = _mm_xor_ps(d.v()[1], rhs.d.v()[1]);
            return r;
        }
        Vc_ALWAYS_INLINE Vc_PURE Float8Mask operator!() const {
            Float8Mask r;
            r.d.v()[0] = _mm_andnot_ps(d.v()[0], _mm_setallone_ps());
            r.d.v()[1] = _mm_andnot_ps(d.v()[1], _mm_setallone_ps());
            return r;
        }
        Vc_ALWAYS_INLINE Float8Mask &operator&=(const Float8Mask &rhs) {
            d.v()[0] = _mm_and_ps(d.v()[0], rhs.d.v()[0]);
            d.v()[1] = _mm_and_ps(d.v()[1], rhs.d.v()[1]);
            return *this;
        }
        Vc_ALWAYS_INLINE Float8Mask &operator|=(const Float8Mask &rhs) {
            d.v()[0] = _mm_or_ps (d.v()[0], rhs.d.v()[0]);
            d.v()[1] = _mm_or_ps (d.v()[1], rhs.d.v()[1]);
            return *this;
        }
        Vc_ALWAYS_INLINE Float8Mask &operator^=(const Float8Mask &rhs) {
            d.v()[0] = _mm_xor_ps(d.v()[0], rhs.d.v()[0]);
            d.v()[1] = _mm_xor_ps(d.v()[1], rhs.d.v()[1]);
            return *this;
        }

        Vc_ALWAYS_INLINE Vc_PURE bool isFull () const {
            const _M128 tmp = _mm_and_ps(d.v()[0], d.v()[1]);
#ifdef VC_USE_PTEST
            return _mm_testc_si128(_mm_castps_si128(tmp), _mm_setallone_si128());
#else
            return _mm_movemask_ps(tmp) == 0xf;
            //_mm_movemask_ps(d.v()[0]) == 0xf &&
            //_mm_movemask_ps(d.v()[1]) == 0xf;
#endif
        }
        Vc_ALWAYS_INLINE Vc_PURE bool isNotEmpty() const {
            const _M128 tmp = _mm_or_ps(d.v()[0], d.v()[1]);
#ifdef VC_USE_PTEST
            return 0 == _mm_testz_si128(_mm_castps_si128(tmp), _mm_castps_si128(tmp));
#else
            return _mm_movemask_ps(tmp) != 0x0;
#endif
        }
        Vc_ALWAYS_INLINE Vc_PURE bool isEmpty() const {
            const _M128 tmp = _mm_or_ps(d.v()[0], d.v()[1]);
#ifdef VC_USE_PTEST
            return _mm_testz_si128(_mm_castps_si128(tmp), _mm_castps_si128(tmp));
#else
            return _mm_movemask_ps(tmp) == 0x0;
            //_mm_movemask_ps(d.v()[0]) == 0x0 &&
            //_mm_movemask_ps(d.v()[1]) == 0x0;
#endif
        }
        Vc_ALWAYS_INLINE Vc_PURE bool isMix() const {
            // consider [1111 0000]
            // solution:
            // if d.v()[0] != d.v()[1] => return true
            // if d.v()[0] == d.v()[1] => return d.v()[0].isMix
#ifdef VC_USE_PTEST
            __m128i tmp = _mm_castps_si128(_mm_xor_ps(d.v()[0], d.v()[1]));
            // tmp == 0 <=> d.v()[0] == d.v()[1]
            return !_mm_testz_si128(tmp, tmp) ||
                _mm_test_mix_ones_zeros(_mm_castps_si128(d.v()[0]), _mm_setallone_si128());
#else
            const int tmp = _mm_movemask_ps(d.v()[0]) + _mm_movemask_ps(d.v()[1]);
            return tmp > 0x0 && tmp < (0xf + 0xf);
#endif
        }

#ifndef VC_NO_AUTOMATIC_BOOL_FROM_MASK
        Vc_ALWAYS_INLINE Vc_PURE operator bool() const { return isFull(); }
#endif

        Vc_ALWAYS_INLINE Vc_PURE int shiftMask() const {
            return (_mm_movemask_ps(d.v()[1]) << 4) + _mm_movemask_ps(d.v()[0]);
        }
        Vc_ALWAYS_INLINE Vc_PURE int toInt() const { return (_mm_movemask_ps(d.v()[1]) << 4) + _mm_movemask_ps(d.v()[0]); }

        Vc_ALWAYS_INLINE Vc_PURE const M256 &data () const { return d.v(); }

        Vc_ALWAYS_INLINE MaskBool &operator[](size_t index) { return d.m(index); }
        Vc_ALWAYS_INLINE Vc_PURE bool operator[](size_t index) const {
            return (toInt() & (1 << index)) != 0;
        }

        Vc_ALWAYS_INLINE Vc_PURE unsigned int count() const {
#ifdef VC_IMPL_POPCNT
		return _mm_popcnt_u32(toInt());
#else
//X             int tmp1 = _mm_movemask_ps(d.v()[0]);
//X             int tmp2 = _mm_movemask_ps(d.v()[1]);
//X             tmp1 = (tmp1 & 5) + ((tmp1 >> 1) & 5);
//X             tmp2 = (tmp2 & 5) + ((tmp2 >> 1) & 5);
//X             return (tmp1 & 3) + (tmp2 & 3) + ((tmp1 >> 2) & 3) + ((tmp2 >> 2) & 3);
            _M128I x = _mm_add_epi32(_mm_srli_epi32(_mm_castps_si128(d.v()[0]), 31),
                                     _mm_srli_epi32(_mm_castps_si128(d.v()[1]), 31));
            x = _mm_add_epi32(x, _mm_shuffle_epi32(x, _MM_SHUFFLE(0, 1, 2, 3)));
            x = _mm_add_epi32(x, _mm_shufflelo_epi16(x, _MM_SHUFFLE(1, 0, 3, 2)));
            return _mm_cvtsi128_si32(x);
#endif
        }

        Vc_ALWAYS_INLINE_L Vc_PURE_L unsigned int firstOne() const Vc_ALWAYS_INLINE_R Vc_PURE_R;

    private:
#ifdef VC_COMPILE_BENCHMARKS
    public:
#endif
        Storage d;
};

template<unsigned int Size> Vc_ALWAYS_INLINE Vc_PURE unsigned int Mask<Size>::firstOne() const
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
Vc_ALWAYS_INLINE Vc_PURE unsigned int Float8Mask::firstOne() const
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
Vc_ALWAYS_INLINE Mask<VectorSize>::Mask(const Float8Mask &m)
    : d(_mm_castsi128_ps(_mm_packs_epi32(_mm_castps_si128(m.data()[0]), _mm_castps_si128(m.data()[1])))) {}

class Float8GatherMask
{
    public:
        Float8GatherMask(const Mask<8u> &k)   : mask(k.toInt()) {}
        Float8GatherMask(const Float8Mask &k) : mask(k.toInt()) {}
        int toInt() const { return mask; }
    private:
        const int mask;
};

// Operators
// let binary and/or/xor work for any combination of masks (as long as they have the same sizeof)
template<unsigned int LSize, unsigned int RSize> Mask<LSize> operator& (const Mask<LSize> &lhs, const Mask<RSize> &rhs) { return _mm_and_ps(lhs.data(), rhs.data()); }
template<unsigned int LSize, unsigned int RSize> Mask<LSize> operator| (const Mask<LSize> &lhs, const Mask<RSize> &rhs) { return _mm_or_ps (lhs.data(), rhs.data()); }
template<unsigned int LSize, unsigned int RSize> Mask<LSize> operator^ (const Mask<LSize> &lhs, const Mask<RSize> &rhs) { return _mm_xor_ps(lhs.data(), rhs.data()); }

// binary and/or/xor cannot work with one operand larger than the other
template<unsigned int Size> void operator& (const Mask<Size> &lhs, const Float8Mask &rhs);
template<unsigned int Size> void operator| (const Mask<Size> &lhs, const Float8Mask &rhs);
template<unsigned int Size> void operator^ (const Mask<Size> &lhs, const Float8Mask &rhs);
template<unsigned int Size> void operator& (const Float8Mask &rhs, const Mask<Size> &lhs);
template<unsigned int Size> void operator| (const Float8Mask &rhs, const Mask<Size> &lhs);
template<unsigned int Size> void operator^ (const Float8Mask &rhs, const Mask<Size> &lhs);

// disable logical and/or for incompatible masks
template<unsigned int LSize, unsigned int RSize> void operator&&(const Mask<LSize> &lhs, const Mask<RSize> &rhs);
template<unsigned int LSize, unsigned int RSize> void operator||(const Mask<LSize> &lhs, const Mask<RSize> &rhs);
template<unsigned int Size> void operator&&(const Mask<Size> &lhs, const Float8Mask &rhs);
template<unsigned int Size> void operator||(const Mask<Size> &lhs, const Float8Mask &rhs);
template<unsigned int Size> void operator&&(const Float8Mask &rhs, const Mask<Size> &lhs);
template<unsigned int Size> void operator||(const Float8Mask &rhs, const Mask<Size> &lhs);

// logical and/or for compatible masks
template<unsigned int Size> Vc_ALWAYS_INLINE Vc_PURE Mask<Size> operator&&(const Mask<Size> &lhs, const Mask<Size> &rhs) { return _mm_and_ps(lhs.data(), rhs.data()); }
template<unsigned int Size> Vc_ALWAYS_INLINE Vc_PURE Mask<Size> operator||(const Mask<Size> &lhs, const Mask<Size> &rhs) { return _mm_or_ps (lhs.data(), rhs.data()); }
Vc_ALWAYS_INLINE Vc_PURE Mask<8> operator&&(const Float8Mask &rhs, const Mask<8> &lhs) { return static_cast<Mask<8> >(rhs) && lhs; }
Vc_ALWAYS_INLINE Vc_PURE Mask<8> operator||(const Float8Mask &rhs, const Mask<8> &lhs) { return static_cast<Mask<8> >(rhs) || lhs; }
Vc_ALWAYS_INLINE Vc_PURE Mask<8> operator&&(const Mask<8> &rhs, const Float8Mask &lhs) { return rhs && static_cast<Mask<8> >(lhs); }
Vc_ALWAYS_INLINE Vc_PURE Mask<8> operator||(const Mask<8> &rhs, const Float8Mask &lhs) { return rhs || static_cast<Mask<8> >(lhs); }

Vc_IMPL_NAMESPACE_END

#include "undomacros.h"

#endif // SSE_MASK_H
