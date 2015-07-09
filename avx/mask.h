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

#ifndef VC_AVX_MASK_H
#define VC_AVX_MASK_H

#include <array>

#include "intrinsics.h"
#include "../common/storage.h"
#include "../common/bitscanintrinsics.h"
#include "../common/maskentry.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Vc_IMPL_NAMESPACE
{

namespace internal {
template<typename V> Vc_ALWAYS_INLINE_L Vc_CONST_L V zero() Vc_ALWAYS_INLINE_R Vc_CONST_R;
template<typename V> Vc_ALWAYS_INLINE_L Vc_CONST_L V allone() Vc_ALWAYS_INLINE_R Vc_CONST_R;
template<size_t From, size_t To, typename R> Vc_ALWAYS_INLINE_L Vc_CONST_L R mask_cast(m128i) Vc_ALWAYS_INLINE_R Vc_CONST_R;
template<size_t From, size_t To, typename R> Vc_ALWAYS_INLINE_L Vc_CONST_L R mask_cast(m256i) Vc_ALWAYS_INLINE_R Vc_CONST_R;
template<size_t Size> Vc_ALWAYS_INLINE_L Vc_CONST_L int mask_to_int(m128i) Vc_ALWAYS_INLINE_R Vc_CONST_R;
template<size_t Size> Vc_ALWAYS_INLINE_L Vc_CONST_L int mask_to_int(m256i) Vc_ALWAYS_INLINE_R Vc_CONST_R;
Vc_INTRINSIC Vc_CONST int testc(m128 a, m128 b) { return _mm_testc_si128(_mm_castps_si128(a), _mm_castps_si128(b)); }
Vc_INTRINSIC Vc_CONST int testc(m256 a, m256 b) { return _mm256_testc_ps(a, b); }
Vc_INTRINSIC Vc_CONST int testz(m128 a, m128 b) { return _mm_testz_si128(_mm_castps_si128(a), _mm_castps_si128(b)); }
Vc_INTRINSIC Vc_CONST int testz(m256 a, m256 b) { return _mm256_testz_ps(a, b); }
Vc_INTRINSIC Vc_CONST int testnzc(m128 a, m128 b) { return _mm_testnzc_si128(_mm_castps_si128(a), _mm_castps_si128(b)); }
Vc_INTRINSIC Vc_CONST int testnzc(m256 a, m256 b) { return _mm256_testnzc_ps(a, b); }
Vc_INTRINSIC Vc_CONST m256 andnot_(param256 a, param256 b) { return _mm256_andnot_ps(a, b); }
Vc_INTRINSIC Vc_CONST m128 andnot_(param128 a, param128 b) { return _mm_andnot_ps(a, b); }
Vc_INTRINSIC Vc_CONST m256 and_(param256 a, param256 b) { return _mm256_and_ps(a, b); }
Vc_INTRINSIC Vc_CONST m128 and_(param128 a, param128 b) { return _mm_and_ps(a, b); }
Vc_INTRINSIC Vc_CONST m256 or_(param256 a, param256 b) { return _mm256_or_ps(a, b); }
Vc_INTRINSIC Vc_CONST m128 or_(param128 a, param128 b) { return _mm_or_ps(a, b); }
Vc_INTRINSIC Vc_CONST m256 xor_(param256 a, param256 b) { return _mm256_xor_ps(a, b); }
Vc_INTRINSIC Vc_CONST m128 xor_(param128 a, param128 b) { return _mm_xor_ps(a, b); }
Vc_INTRINSIC Vc_CONST int movemask(param256i a) { return movemask_epi8(a); }
Vc_INTRINSIC Vc_CONST int movemask(param128i a) { return _mm_movemask_epi8(a); }
Vc_INTRINSIC Vc_CONST int movemask(param256d a) { return _mm256_movemask_pd(a); }
Vc_INTRINSIC Vc_CONST int movemask(param128d a) { return _mm_movemask_pd(a); }
Vc_INTRINSIC Vc_CONST int movemask(param256  a) { return _mm256_movemask_ps(a); }
Vc_INTRINSIC Vc_CONST int movemask(param128  a) { return _mm_movemask_ps(a); }

} // namespace internal

template<typename T> class Mask
{
    friend class Mask<  double>;
    friend class Mask<   float>;
    friend class Mask< int32_t>;
    friend class Mask<uint32_t>;
    friend class Mask< int16_t>;
    friend class Mask<uint16_t>;
    friend Common::MaskEntry<Mask>;

public:
    /**
     * The \c EntryType of masks is always bool, independent of \c T.
     */
    typedef bool EntryType;

    /**
     * The \c VectorEntryType, in contrast to \c EntryType, reveals information about the SIMD
     * implementation. This type is useful for the \c sizeof operator in generic functions.
     */
    using VectorEntryType = Common::MaskBool<sizeof(T)>;

    /**
     * The associated Vector<T> type.
     */
    using Vector = Vc_AVX_NAMESPACE::Vector<T>;

    ///\internal
    using VectorTypeF = FloatVectorType<typename VectorTypeHelper<T>::Type>;
    ///\internal
    using VectorTypeD = DoubleVectorType<VectorTypeF>;
    ///\internal
    using VectorTypeI = IntegerVectorType<VectorTypeF>;

private:
#ifdef VC_PASSING_VECTOR_BY_VALUE_IS_BROKEN
    typedef const VectorTypeF &VArg;
    typedef const VectorTypeD &VdArg;
    typedef const VectorTypeI &ViArg;
#else
    typedef const VectorTypeF VArg;
    typedef const VectorTypeD VdArg;
    typedef const VectorTypeI ViArg;
#endif

public:
    static constexpr size_t Size = sizeof(VectorTypeF) / sizeof(T);
    static constexpr std::size_t size() { return Size; }
    FREE_STORE_OPERATORS_ALIGNED(alignof(VectorType))

private:
    typedef Common::Storage<T, size()> Storage;

public:
    /**
     * The \c VectorType reveals the implementation-specific internal type used for the
     * SIMD type.
     */
    using VectorType = typename Storage::VectorType;

    using EntryReference = Common::MaskEntry<Mask>;

        // abstracts the way Masks are passed to functions, it can easily be changed to const ref here
#if defined VC_MSVC && defined _WIN32
        typedef const Mask<T> &AsArg;
#else
        typedef const Mask<T> AsArg;
#endif

        Vc_ALWAYS_INLINE Mask() {}
        Vc_ALWAYS_INLINE Mask(VArg  x) : d(avx_cast<VectorType>(x)) {}
        Vc_ALWAYS_INLINE Mask(VdArg x) : d(avx_cast<VectorType>(x)) {}
        Vc_ALWAYS_INLINE Mask(ViArg x) : d(avx_cast<VectorType>(x)) {}
        Vc_ALWAYS_INLINE explicit Mask(VectorSpecialInitializerZero::ZEnum) : d(internal::zero<VectorType>()) {}
        Vc_ALWAYS_INLINE explicit Mask(VectorSpecialInitializerOne::OEnum) : d(internal::allone<VectorType>()) {}
        Vc_ALWAYS_INLINE explicit Mask(bool b)
            : d(b ? internal::allone<VectorType>() : internal::zero<VectorType>())
        {
        }
        Vc_INTRINSIC static Mask Zero() { return Mask{VectorSpecialInitializerZero::Zero}; }
        Vc_INTRINSIC static Mask One() { return Mask{VectorSpecialInitializerOne::One}; }

        // implicit cast
        template <typename U>
        Vc_INTRINSIC Mask(U &&rhs,
                          Common::enable_if_mask_converts_implicitly<T, U> = nullarg)
            : d(avx_cast<VectorType>(
                  internal::mask_cast<Traits::decay<U>::Size, Size, VectorTypeF>(
                      rhs.dataI())))
        {
        }

        // explicit cast, implemented via simd_cast (in avx/simd_cast_caller.h)
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

        Vc_ALWAYS_INLINE Mask &operator=(const Mask &) = default;
        Vc_ALWAYS_INLINE_L Mask &operator=(const std::array<bool, Size> &values) Vc_ALWAYS_INLINE_R;
        Vc_ALWAYS_INLINE_L operator std::array<bool, Size>() const Vc_ALWAYS_INLINE_R;

        // specializations in mask.tcc
        Vc_INTRINSIC Vc_PURE bool operator==(const Mask &rhs) const
        { return internal::movemask(d.v()) == internal::movemask(rhs.d.v()); }

        Vc_INTRINSIC Vc_PURE bool operator!=(const Mask &rhs) const
        { return !operator==(rhs); }

        Vc_ALWAYS_INLINE Mask operator!() const { return internal::andnot_(data(), internal::allone<VectorTypeF>()); }

        Vc_ALWAYS_INLINE Mask &operator&=(const Mask &rhs) { d.v() = avx_cast<VectorType>(internal::and_(data(), rhs.data())); return *this; }
        Vc_ALWAYS_INLINE Mask &operator|=(const Mask &rhs) { d.v() = avx_cast<VectorType>(internal::or_ (data(), rhs.data())); return *this; }
        Vc_ALWAYS_INLINE Mask &operator^=(const Mask &rhs) { d.v() = avx_cast<VectorType>(internal::xor_(data(), rhs.data())); return *this; }

        Vc_ALWAYS_INLINE Vc_PURE Mask operator&(const Mask &rhs) const { return internal::and_(data(), rhs.data()); }
        Vc_ALWAYS_INLINE Vc_PURE Mask operator|(const Mask &rhs) const { return internal::or_(data(), rhs.data()); }
        Vc_ALWAYS_INLINE Vc_PURE Mask operator^(const Mask &rhs) const { return internal::xor_(data(), rhs.data()); }

        Vc_ALWAYS_INLINE Vc_PURE Mask operator&&(const Mask &rhs) const { return internal::and_(data(), rhs.data()); }
        Vc_ALWAYS_INLINE Vc_PURE Mask operator||(const Mask &rhs) const { return internal::or_(data(), rhs.data()); }

        // no need for expression template optimizations because cmp(n)eq for floats are not bitwise
        // compares
        Vc_ALWAYS_INLINE bool isFull () const { return 0 != internal::testc(data(), internal::allone<VectorTypeF>()); }
        Vc_ALWAYS_INLINE bool isNotEmpty() const { return 0 == internal::testz(data(), data()); }
        Vc_ALWAYS_INLINE bool isEmpty() const { return 0 != internal::testz(data(), data()); }
        Vc_ALWAYS_INLINE bool isMix  () const { return 0 != internal::testnzc(data(), internal::allone<VectorTypeF>()); }

        Vc_ALWAYS_INLINE Vc_PURE int shiftMask() const { return internal::movemask(dataI()); }
        Vc_ALWAYS_INLINE Vc_PURE int toInt() const { return internal::mask_to_int<Size>(dataI()); }

        Vc_ALWAYS_INLINE VectorTypeF data () const { return avx_cast<VectorTypeF>(d.v()); }
        Vc_ALWAYS_INLINE VectorTypeI dataI() const { return avx_cast<VectorTypeI>(d.v()); }
        Vc_ALWAYS_INLINE VectorTypeD dataD() const { return avx_cast<VectorTypeD>(d.v()); }

        Vc_ALWAYS_INLINE EntryReference operator[](size_t index)
        {
            return {*this, index};
        }
        Vc_ALWAYS_INLINE_L Vc_PURE_L bool operator[](size_t index) const Vc_ALWAYS_INLINE_R Vc_PURE_R;

        Vc_ALWAYS_INLINE Vc_PURE int count() const { return _mm_popcnt_u32(toInt()); }
        Vc_ALWAYS_INLINE Vc_PURE int firstOne() const { return _bit_scan_forward(toInt()); }

        template <typename G> static Vc_INTRINSIC_L Mask generate(G &&gen) Vc_INTRINSIC_R;
        Vc_INTRINSIC_L Vc_PURE_L Mask shifted(int amount) const Vc_INTRINSIC_R Vc_PURE_R;

        ///\internal Called indirectly from operator[]
        void setEntry(size_t i, bool x) { d.set(i, Common::MaskBool<sizeof(T)>(x)); }

    private:
#ifndef VC_IMPL_POPCNT
        static Vc_ALWAYS_INLINE Vc_CONST unsigned int _mm_popcnt_u32(unsigned int n) {
            n = (n & 0x55555555U) + ((n >> 1) & 0x55555555U);
            n = (n & 0x33333333U) + ((n >> 2) & 0x33333333U);
            n = (n & 0x0f0f0f0fU) + ((n >> 4) & 0x0f0f0f0fU);
            //n = (n & 0x00ff00ffU) + ((n >> 8) & 0x00ff00ffU);
            //n = (n & 0x0000ffffU) + ((n >>16) & 0x0000ffffU);
            return n;
        }
#endif

#ifdef VC_COMPILE_BENCHMARKS
    public:
#endif
        Storage d;
};
template<typename T> constexpr size_t Mask<T>::Size;

}  // namespace AVX(2)
}  // namespace Vc

#include "mask.tcc"
#include "undomacros.h"

#endif // VC_AVX_MASK_H
