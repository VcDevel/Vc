/*  This file is part of the Vc library. {{{
Copyright © 2009-2014 Matthias Kretz <kretz@kde.org>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the names of contributing organizations nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#ifndef SSE_MASK_H
#define SSE_MASK_H

#include "intrinsics.h"
#include "../common/maskentry.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace SSE
{

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
    friend class Common::MaskEntry<Mask>;

    /**
     * A helper type for aliasing the entries in the mask but behaving like a bool.
     */
    typedef Common::MaskBool<sizeof(T)> MaskBool;

    typedef Common::Storage<T, VectorTraits<T>::Size> Storage;

public:

    /**
     * The \c EntryType of masks is always bool, independent of \c T.
     */
    typedef bool EntryType;

    /**
     * The \c VectorEntryType, in contrast to \c EntryType, reveals information about the SIMD
     * implementation. This type is useful for the \c sizeof operator in generic functions.
     */
    typedef MaskBool VectorEntryType;

    /**
     * The \c VectorType reveals the implementation-specific internal type used for the SIMD type.
     */
    using VectorType = typename Storage::VectorType;

    /**
     * The associated Vector<T> type.
     */
    using Vector = SSE::Vector<T>;

public:
    FREE_STORE_OPERATORS_ALIGNED(16)
    static constexpr size_t Size = VectorTraits<T>::Size;
    static constexpr std::size_t size() { return Size; }

        // abstracts the way Masks are passed to functions, it can easily be changed to const ref here
#if defined VC_MSVC && defined _WIN32
        typedef const Mask<T> &Argument;
#else
        typedef Mask<T> Argument;
#endif

        Vc_INTRINSIC Mask() {}
        Vc_INTRINSIC Mask(const __m128  &x) : d(sse_cast<VectorType>(x)) {}
        Vc_INTRINSIC Mask(const __m128d &x) : d(sse_cast<VectorType>(x)) {}
        Vc_INTRINSIC Mask(const __m128i &x) : d(sse_cast<VectorType>(x)) {}
        Vc_INTRINSIC explicit Mask(VectorSpecialInitializerZero::ZEnum) : Mask(_mm_setzero_ps()) {}
        Vc_INTRINSIC explicit Mask(VectorSpecialInitializerOne::OEnum) : Mask(_mm_setallone_ps()) {}
        Vc_INTRINSIC explicit Mask(bool b) : Mask(b ? _mm_setallone_ps() : _mm_setzero_ps()) {}
        Vc_INTRINSIC static Mask Zero() { return Mask{VectorSpecialInitializerZero::Zero}; }
        Vc_INTRINSIC static Mask One() { return Mask{VectorSpecialInitializerOne::One}; }

        // implicit cast
        template <typename U>
        Vc_INTRINSIC Mask(U &&rhs,
                          Common::enable_if_mask_converts_implicitly<T, U> = nullarg)
            : d(sse_cast<VectorType>(
                  internal::mask_cast<Traits::simd_vector_size<U>::value, Size>(
                      rhs.dataI())))
        {
        }

        // explicit cast, implemented via simd_cast (implementation in sse/simd_cast.h)
        template <typename U>
        Vc_INTRINSIC explicit Mask(U &&rhs,
                                   Common::enable_if_mask_converts_explicitly<T, U> =
                                       nullarg);

        Vc_ALWAYS_INLINE explicit Mask(const bool *mem) { load(mem); }
        template<typename Flags> Vc_ALWAYS_INLINE explicit Mask(const bool *mem, Flags f) { load(mem, f); }

        Vc_ALWAYS_INLINE_L void load(const bool *mem) Vc_ALWAYS_INLINE_R;
        template<typename Flags> Vc_ALWAYS_INLINE void load(const bool *mem, Flags) { load(mem); }

        Vc_ALWAYS_INLINE_L void store(bool *) const Vc_ALWAYS_INLINE_R;
        template<typename Flags> Vc_ALWAYS_INLINE void store(bool *mem, Flags) const { store(mem); }

        Vc_ALWAYS_INLINE Vc_PURE bool operator==(const Mask &rhs) const { return MaskHelper<Size>::cmpeq (data(), rhs.data()); }
        Vc_ALWAYS_INLINE Vc_PURE bool operator!=(const Mask &rhs) const { return MaskHelper<Size>::cmpneq(data(), rhs.data()); }

        Vc_ALWAYS_INLINE Vc_PURE Mask operator!() const { return _mm_andnot_si128(dataI(), _mm_setallone_si128()); }

        Vc_ALWAYS_INLINE Mask &operator&=(const Mask &rhs) { d.v() = sse_cast<VectorType>(_mm_and_ps(data(), rhs.data())); return *this; }
        Vc_ALWAYS_INLINE Mask &operator|=(const Mask &rhs) { d.v() = sse_cast<VectorType>(_mm_or_ps (data(), rhs.data())); return *this; }
        Vc_ALWAYS_INLINE Mask &operator^=(const Mask &rhs) { d.v() = sse_cast<VectorType>(_mm_xor_ps(data(), rhs.data())); return *this; }

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

        Vc_ALWAYS_INLINE Vc_PURE int shiftMask() const { return _mm_movemask_epi8(dataI()); }

        Vc_ALWAYS_INLINE Vc_PURE int toInt() const { return internal::mask_to_int<Size>(dataI()); }

        Vc_ALWAYS_INLINE Vc_PURE __m128  data () const { return sse_cast<__m128 >(d.v()); }
        Vc_ALWAYS_INLINE Vc_PURE __m128i dataI() const { return sse_cast<__m128i>(d.v()); }
        Vc_ALWAYS_INLINE Vc_PURE __m128d dataD() const { return sse_cast<__m128d>(d.v()); }

        Vc_ALWAYS_INLINE Common::MaskEntry<Mask> operator[](size_t index)
        {
            return Common::MaskEntry<Mask>(*this, index);
        }
        Vc_ALWAYS_INLINE Vc_PURE bool operator[](size_t index) const
        {
            return toInt() & (1 << index);
        }

        Vc_ALWAYS_INLINE Vc_PURE int count() const { return internal::mask_count<Size>(dataI()); }

        /**
         * Returns the index of the first one in the mask.
         *
         * The return value is undefined if the mask is empty.
         */
        Vc_ALWAYS_INLINE_L Vc_PURE_L int firstOne() const Vc_ALWAYS_INLINE_R Vc_PURE_R;

        template <typename G> static Vc_INTRINSIC_L Mask generate(G &&gen) Vc_INTRINSIC_R;
        Vc_INTRINSIC_L Vc_PURE_L Mask shifted(int amount) const Vc_INTRINSIC_R Vc_PURE_R;

    private:
#ifdef VC_COMPILE_BENCHMARKS
    public:
#endif
        Storage d;

        void setEntry(size_t i, bool x) { d.set(i, MaskBool(x)); }
};
template<typename T> constexpr size_t Mask<T>::Size;

}  // namespace SSE
}  // namespace Vc

#include "undomacros.h"
#include "mask.tcc"

#endif // SSE_MASK_H
