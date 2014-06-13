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

template<unsigned int Size1> struct MaskHelper;
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

namespace internal
{
    template<size_t From, size_t To> Vc_ALWAYS_INLINE_L Vc_CONST_L __m128i mask_cast(__m128i) Vc_ALWAYS_INLINE_R Vc_CONST_R;
    template<size_t Size> Vc_ALWAYS_INLINE_L Vc_CONST_L int mask_to_int(__m128i) Vc_ALWAYS_INLINE_R Vc_CONST_R;
    template<size_t Size> Vc_ALWAYS_INLINE_L Vc_CONST_L int mask_count(__m128i) Vc_ALWAYS_INLINE_R Vc_CONST_R;
} // namespace internal

template<typename T> class Mask
{
    friend class Mask<  double>;
    friend class Mask<   float>;
    friend class Mask< int32_t>;
    friend class Mask<uint32_t>;
    friend class Mask< int16_t>;
    friend class Mask<uint16_t>;

    public:
        FREE_STORE_OPERATORS_ALIGNED(16)
        static constexpr size_t Size = VectorTraits<T>::Size;

    private:
        typedef Common::MaskBool<sizeof(T)> MaskBool;
        typedef Common::VectorMemoryUnion<__m128, MaskBool> Storage;

    public:
        // abstracts the way Masks are passed to functions, it can easily be changed to const ref here
#if defined VC_MSVC && defined _WIN32
        typedef const Mask<T> &Argument;
#else
        typedef Mask<T> Argument;
#endif

        Vc_ALWAYS_INLINE Mask() {}
        Vc_ALWAYS_INLINE Mask(const __m128  &x) : d(x) {}
        Vc_ALWAYS_INLINE Mask(const __m128d &x) : d(_mm_castpd_ps(x)) {}
        Vc_ALWAYS_INLINE Mask(const __m128i &x) : d(_mm_castsi128_ps(x)) {}
        Vc_ALWAYS_INLINE explicit Mask(VectorSpecialInitializerZero::ZEnum) : d(_mm_setzero_ps()) {}
        Vc_ALWAYS_INLINE explicit Mask(VectorSpecialInitializerOne::OEnum) : d(_mm_setallone_ps()) {}
        Vc_ALWAYS_INLINE explicit Mask(bool b) : d(b ? _mm_setallone_ps() : _mm_setzero_ps()) {}

        template<typename U> Vc_ALWAYS_INLINE Mask(const Mask<U> &rhs,
          typename std::enable_if<is_implicit_cast_allowed_mask<U, T>::value, void *>::type = nullptr)
            : d(sse_cast<__m128>(internal::mask_cast<Mask<U>::Size, Size>(rhs.dataI()))) {}

        template<typename U> Vc_ALWAYS_INLINE explicit Mask(const Mask<U> &rhs,
          typename std::enable_if<!is_implicit_cast_allowed_mask<U, T>::value, void *>::type = nullptr)
            : d(sse_cast<__m128>(internal::mask_cast<Mask<U>::Size, Size>(rhs.dataI()))) {}

        Vc_ALWAYS_INLINE explicit Mask(const bool *mem) { load(mem); }
        template<typename Flags> Vc_ALWAYS_INLINE explicit Mask(const bool *mem, Flags f) { load(mem, f); }

        Vc_ALWAYS_INLINE_L void load(const bool *mem) Vc_ALWAYS_INLINE_R;
        template<typename Flags> Vc_ALWAYS_INLINE void load(const bool *mem, Flags) { load(mem); }

        Vc_ALWAYS_INLINE_L void store(bool *) const Vc_ALWAYS_INLINE_R;
        template<typename Flags> Vc_ALWAYS_INLINE void store(bool *mem, Flags) const { store(mem); }

        Vc_ALWAYS_INLINE Vc_PURE bool operator==(const Mask &rhs) const { return MaskHelper<Size>::cmpeq (d.v(), rhs.d.v()); }
        Vc_ALWAYS_INLINE Vc_PURE bool operator!=(const Mask &rhs) const { return MaskHelper<Size>::cmpneq(d.v(), rhs.d.v()); }

        Vc_ALWAYS_INLINE Vc_PURE Mask operator!() const { return _mm_andnot_si128(dataI(), _mm_setallone_si128()); }

        Vc_ALWAYS_INLINE Mask &operator&=(const Mask &rhs) { d.v() = _mm_and_ps(d.v(), rhs.d.v()); return *this; }
        Vc_ALWAYS_INLINE Mask &operator|=(const Mask &rhs) { d.v() = _mm_or_ps (d.v(), rhs.d.v()); return *this; }
        Vc_ALWAYS_INLINE Mask &operator^=(const Mask &rhs) { d.v() = _mm_xor_ps(d.v(), rhs.d.v()); return *this; }

        Vc_ALWAYS_INLINE Vc_PURE Mask operator&(const Mask &rhs) const { return _mm_and_ps(data(), rhs.data()); }
        Vc_ALWAYS_INLINE Vc_PURE Mask operator|(const Mask &rhs) const { return _mm_or_ps (data(), rhs.data()); }
        Vc_ALWAYS_INLINE Vc_PURE Mask operator^(const Mask &rhs) const { return _mm_xor_ps(data(), rhs.data()); }

        Vc_ALWAYS_INLINE Vc_PURE Mask operator&&(const Mask &rhs) const { return _mm_and_ps(data(), rhs.data()); }
        Vc_ALWAYS_INLINE Vc_PURE Mask operator||(const Mask &rhs) const { return _mm_or_ps (data(), rhs.data()); }

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

        Vc_ALWAYS_INLINE Vc_PURE int shiftMask() const { return _mm_movemask_epi8(dataI()); }

        Vc_ALWAYS_INLINE Vc_PURE int toInt() const { return internal::mask_to_int<Size>(dataI()); }

        Vc_ALWAYS_INLINE Vc_PURE _M128  data () const { return d.v(); }
        Vc_ALWAYS_INLINE Vc_PURE _M128I dataI() const { return _mm_castps_si128(d.v()); }
        Vc_ALWAYS_INLINE Vc_PURE _M128D dataD() const { return _mm_castps_pd(d.v()); }

        Vc_ALWAYS_INLINE MaskBool &operator[](size_t index) { return d.m(index); }
        Vc_ALWAYS_INLINE Vc_PURE bool operator[](size_t index) const { return toInt() & (1 << index); }

        Vc_ALWAYS_INLINE Vc_PURE int count() const { return internal::mask_count<Size>(dataI()); }

        /**
         * Returns the index of the first one in the mask.
         *
         * The return value is undefined if the mask is empty.
         */
        Vc_ALWAYS_INLINE_L Vc_PURE_L int firstOne() const Vc_ALWAYS_INLINE_R Vc_PURE_R;

    private:
#ifdef VC_COMPILE_BENCHMARKS
    public:
#endif
        Storage d;
};
template<typename T> constexpr size_t Mask<T>::Size;

struct ForeachHelper
{
    _long mask;
    bool brk;
    bool outerBreak;
    Vc_ALWAYS_INLINE ForeachHelper(_long _mask) : mask(_mask), brk(false), outerBreak(false) {}
    Vc_ALWAYS_INLINE bool outer() const { return (mask != 0) && !outerBreak; }
    Vc_ALWAYS_INLINE bool inner() { return (brk = !brk); }
    Vc_ALWAYS_INLINE void noBreak() { outerBreak = false; }
    Vc_ALWAYS_INLINE _long next() {
        outerBreak = true;
#ifdef VC_GNU_ASM
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
    for (Vc::SSE::ForeachHelper Vc__make_unique(foreach_bit_obj)((_mask_).toInt()); Vc__make_unique(foreach_bit_obj).outer(); ) \
        for (_it_ = Vc__make_unique(foreach_bit_obj).next(); Vc__make_unique(foreach_bit_obj).inner(); Vc__make_unique(foreach_bit_obj).noBreak())

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

Vc_IMPL_NAMESPACE_END

#include "undomacros.h"
#include "mask.tcc"

#endif // SSE_MASK_H
