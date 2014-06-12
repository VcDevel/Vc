/*  This file is part of the Vc library.

    Copyright (C) 2009-2013 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_AVX_MASK_H
#define VC_AVX_MASK_H

#include <array>

#include "intrinsics.h"
#include "../common/bitscanintrinsics.h"
#include "../common/maskentry.h"
#include "macros.h"

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)

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
Vc_INTRINSIC Vc_CONST int movemask(param256i a) { return _mm256_movemask_epi8(a); }
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

    typedef typename FloatVectorType<typename VectorTypeHelper<T>::Type>::Type VectorType;
    typedef typename  DoubleVectorType<VectorType>::Type VectorTypeD;
    typedef typename IntegerVectorType<VectorType>::Type VectorTypeI;
#ifdef VC_PASSING_VECTOR_BY_VALUE_IS_BROKEN
    typedef const VectorType  &VArg;
    typedef const VectorTypeD &VdArg;
    typedef const VectorTypeI &ViArg;
#else
    typedef const VectorType  VArg;
    typedef const VectorTypeD VdArg;
    typedef const VectorTypeI ViArg;
#endif
    static constexpr size_t VSize = sizeof(VectorType);

    public:
        static constexpr size_t Size = VSize / sizeof(T);
        FREE_STORE_OPERATORS_ALIGNED(alignof(VectorType))

    private:
        typedef Common::MaskBool<sizeof(T)> MaskBool;
        typedef Common::VectorMemoryUnion<VectorType, MaskBool> Storage;

    public:
        // abstracts the way Masks are passed to functions, it can easily be changed to const ref here
#if defined VC_MSVC && defined _WIN32
        typedef const Mask<T> &AsArg;
#else
        typedef const Mask<T> AsArg;
#endif

        Vc_ALWAYS_INLINE Mask() {}
        Vc_ALWAYS_INLINE Mask(VArg  x) : d(x) {}
        Vc_ALWAYS_INLINE Mask(VdArg x) : d(avx_cast<VectorType>(x)) {}
        Vc_ALWAYS_INLINE Mask(ViArg x) : d(avx_cast<VectorType>(x)) {}
#ifdef VC_UNCONDITIONAL_AVX2_INTRINSICS
        //Vc_ALWAYS_INLINE Mask(__m256  x) : d(x) {}
        //Vc_ALWAYS_INLINE Mask(__m256d x) : d(_mm256_castpd_ps(x)) {}
        //Vc_ALWAYS_INLINE Mask(__m256i x) : d(_mm256_castsi256_ps(x)) {}
#endif
        Vc_ALWAYS_INLINE explicit Mask(VectorSpecialInitializerZero::ZEnum) : d(internal::zero<VectorType>()) {}
        Vc_ALWAYS_INLINE explicit Mask(VectorSpecialInitializerOne::OEnum) : d(internal::allone<VectorType>()) {}
        Vc_ALWAYS_INLINE explicit Mask(bool b) : d(b ? internal::allone<VectorType>() : VectorType(internal::zero<VectorType>())) {}

        template<typename U> Vc_ALWAYS_INLINE Mask(const Mask<U> &rhs,
          typename std::enable_if<is_implicit_cast_allowed_mask<U, T>::value, void *>::type = nullptr)
            : d(internal::mask_cast<Mask<U>::Size, Size, VectorType>(rhs.dataI())) {}

        template<typename U> Vc_ALWAYS_INLINE explicit Mask(const Mask<U> &rhs,
          typename std::enable_if<!is_implicit_cast_allowed_mask<U, T>::value, void *>::type = nullptr)
            : d(internal::mask_cast<Mask<U>::Size, Size, VectorType>(rhs.dataI())) {}

        Vc_ALWAYS_INLINE explicit Mask(const bool *mem) { load(mem); }
        template<typename Flags> Vc_ALWAYS_INLINE explicit Mask(const bool *mem, Flags f) { load(mem, f); }

        Vc_ALWAYS_INLINE_L void load(const bool *mem) Vc_ALWAYS_INLINE_R;
        template<typename Flags> Vc_ALWAYS_INLINE void load(const bool *mem, Flags) { load(mem); }

        Vc_ALWAYS_INLINE_L void store(bool *) const Vc_ALWAYS_INLINE_R;
        template<typename Flags> Vc_ALWAYS_INLINE void store(bool *mem, Flags) const { store(mem); }

        Vc_ALWAYS_INLINE Mask &operator=(const Mask &) = default;
        Vc_ALWAYS_INLINE_L Mask &operator=(const std::array<bool, Size> &values) Vc_ALWAYS_INLINE_R;
        Vc_ALWAYS_INLINE_L operator std::array<bool, Size>() const Vc_ALWAYS_INLINE_R;

        Vc_ALWAYS_INLINE bool operator==(const Mask &rhs) const { return internal::movemask(d.v()) == internal::movemask(rhs.d.v()); }
        Vc_ALWAYS_INLINE bool operator!=(const Mask &rhs) const { return internal::movemask(d.v()) != internal::movemask(rhs.d.v()); }

        Vc_ALWAYS_INLINE Mask operator!() const { return internal::andnot_(data(), internal::allone<VectorType>()); }

        Vc_ALWAYS_INLINE Mask &operator&=(const Mask &rhs) { d.v() = internal::and_(d.v(), rhs.d.v()); return *this; }
        Vc_ALWAYS_INLINE Mask &operator|=(const Mask &rhs) { d.v() = internal::or_(d.v(), rhs.d.v()); return *this; }
        Vc_ALWAYS_INLINE Mask &operator^=(const Mask &rhs) { d.v() = internal::xor_(d.v(), rhs.d.v()); return *this; }

        Vc_ALWAYS_INLINE Vc_PURE Mask operator&(const Mask &rhs) const { return internal::and_(data(), rhs.data()); }
        Vc_ALWAYS_INLINE Vc_PURE Mask operator|(const Mask &rhs) const { return internal::or_(data(), rhs.data()); }
        Vc_ALWAYS_INLINE Vc_PURE Mask operator^(const Mask &rhs) const { return internal::xor_(data(), rhs.data()); }

        Vc_ALWAYS_INLINE Vc_PURE Mask operator&&(const Mask &rhs) const { return internal::and_(data(), rhs.data()); }
        Vc_ALWAYS_INLINE Vc_PURE Mask operator||(const Mask &rhs) const { return internal::or_(data(), rhs.data()); }

        // no need for expression template optimizations because cmp(n)eq for floats are not bitwise
        // compares
        Vc_ALWAYS_INLINE bool isFull () const { return 0 != internal::testc(d.v(), internal::allone<VectorType>()); }
        Vc_ALWAYS_INLINE bool isNotEmpty() const { return 0 == internal::testz(d.v(), d.v()); }
        Vc_ALWAYS_INLINE bool isEmpty() const { return 0 != internal::testz(d.v(), d.v()); }
        Vc_ALWAYS_INLINE bool isMix  () const { return 0 != internal::testnzc(d.v(), internal::allone<VectorType>()); }

#ifndef VC_NO_AUTOMATIC_BOOL_FROM_MASK
        Vc_ALWAYS_INLINE operator bool() const { return isFull(); }
#endif

        Vc_ALWAYS_INLINE Vc_PURE int shiftMask() const { return internal::movemask(dataI()); }
        Vc_ALWAYS_INLINE Vc_PURE int toInt() const { return internal::mask_to_int<Size>(dataI()); }

        Vc_ALWAYS_INLINE VectorType  data () const { return d.v(); }
        Vc_ALWAYS_INLINE VectorTypeI dataI() const { return avx_cast<VectorTypeI>(d.v()); }
        Vc_ALWAYS_INLINE VectorTypeD dataD() const { return avx_cast<VectorTypeD>(d.v()); }

        Vc_ALWAYS_INLINE MaskBool &operator[](size_t index) { return d.m(index); }
        Vc_ALWAYS_INLINE_L Vc_PURE_L bool operator[](size_t index) const Vc_ALWAYS_INLINE_R Vc_PURE_R;

        Vc_ALWAYS_INLINE Vc_PURE unsigned int count() const { return _mm_popcnt_u32(toInt()); }
        Vc_ALWAYS_INLINE Vc_PURE unsigned int firstOne() const { return _bit_scan_forward(toInt()); }

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

struct ForeachHelper
{
    size_t mask;
    bool brk;
    bool outerBreak;
    Vc_ALWAYS_INLINE ForeachHelper(size_t _mask) : mask(_mask), brk(false), outerBreak(false) {}
    Vc_ALWAYS_INLINE bool outer() const { return mask != 0 && !outerBreak; }
    Vc_ALWAYS_INLINE bool inner() { return (brk = !brk); }
    Vc_ALWAYS_INLINE void noBreak() { outerBreak = false; }
    Vc_ALWAYS_INLINE size_t next() {
        outerBreak = true;
#ifdef VC_GNU_ASM
        const size_t bit = __builtin_ctzl(mask);
        __asm__("btr %1,%0" : "+r"(mask) : "r"(bit));
#else
#ifdef VC_MSVC
#pragma warning(suppress : 4267) // conversion from 'size_t' to 'unsigned long', possible loss of data
#endif
        const size_t bit = _bit_scan_forward(mask);
        mask &= ~(1 << bit);
#endif
        return bit;
    }
};

#define Vc_foreach_bit(_it_, _mask_) \
    for (Vc::Vc_IMPL_NAMESPACE::ForeachHelper Vc__make_unique(foreach_bit_obj)((_mask_).toInt()); Vc__make_unique(foreach_bit_obj).outer(); ) \
        for (_it_ = Vc__make_unique(foreach_bit_obj).next(); Vc__make_unique(foreach_bit_obj).inner(); Vc__make_unique(foreach_bit_obj).noBreak())

Vc_IMPL_NAMESPACE_END

#include "mask.tcc"
#include "undomacros.h"

#endif // VC_AVX_MASK_H
