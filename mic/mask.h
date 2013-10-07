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

#ifndef VC_MIC_MASK_H
#define VC_MIC_MASK_H

#include "../common/maskentry.h"
#include "macros.h"

#ifdef CAN_OFFLOAD
#pragma offload_attribute(push, target(mic))
#endif

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)

template<unsigned int VectorSize> struct MaskHelper;
template<> struct MaskHelper<8> {
    typedef __mmask8  Type;
    static inline bool isFull (Type k) {
        // I can't find a way around ICC creating an unnecessary kmov to GPR. Every method involves
        // a cast to __mmask16 and thus induces the “problem”.
        //__mmask16 kk; asm("kmerge2l1l %[in],%[out]" : [out]"=k"(kk) : [in]"k"((__mmask16)k));
        __mmask16 kk = _mm512_kmovlhb(k, k);
        return _mm512_kortestc(kk, kk);
    }
    static inline bool isEmpty(Type k) { return _mm512_kortestz(k, k); }
    static inline bool isNotEmpty(Type k) { return !isEmpty(k); }
    static inline bool isMix  (Type k) { return !isFull(k) && !isEmpty(k); }
    static inline bool cmpeq  (Type k1, Type k2) { return isEmpty(_mm512_kxor(k1, k2)); }
    static inline bool cmpneq (Type k1, Type k2) { return isNotEmpty(_mm512_kxor(k1, k2)); }
    static Vc_INTRINSIC Vc_CONST Type cast(Type k) { return k; }
    static Vc_INTRINSIC Vc_CONST Type cast(__mmask16 k) { return _mm512_kand(k, 0xff); }
};
template<> struct MaskHelper<16> {
    typedef __mmask16 Type;
    static inline bool isFull (Type k) { return _mm512_kortestc(k, k); }
    static inline bool isEmpty(Type k) { return _mm512_kortestz(k, k); }
    static inline bool isNotEmpty(Type k) { return !isEmpty(k); }
    static inline bool isMix  (Type k) { return !isFull(k) && !isEmpty(k); }
    static inline bool cmpeq  (Type k1, Type k2) { return isEmpty(_mm512_kxor(k1, k2)); }
    static inline bool cmpneq (Type k1, Type k2) { return isNotEmpty(_mm512_kxor(k1, k2)); }
    static Vc_INTRINSIC Vc_CONST Type cast(Type k) { return k; }
    static Vc_INTRINSIC Vc_CONST Type cast(__mmask8 k) { return _mm512_kand(k, 0xff); }
};

template<typename T> class Mask
{
    friend class Mask<  double>;
    friend class Mask<   float>;
    friend class Mask< int32_t>;
    friend class Mask<uint32_t>;
    friend class Mask< int16_t>;
    friend class Mask<uint16_t>;

    typedef typename MaskTypeHelper<T>::Type MaskType;
    typedef typename VectorTypeHelper<T>::Type VectorType;
    typedef typename DetermineVectorEntryType<T>::Type VectorEntryType;

public:
    static constexpr size_t Size = sizeof(VectorType) / sizeof(VectorEntryType);
    typedef Mask<T> AsArg; // for now only ICC can compile this code and it is not broken :)
    inline Mask() {}
    inline Mask(MaskType _k) : k(_k) {}
    inline explicit Mask(VectorSpecialInitializerZero::ZEnum) : k(0) {}
    inline explicit Mask(VectorSpecialInitializerOne::OEnum) : k(Size == 16 ? 0xffff : 0xff) {}
    inline explicit Mask(bool b) : k(b ? (Size == 16 ? 0xffff : 0xff) : 0) {}

    template<typename U> Vc_ALWAYS_INLINE Mask(const Mask<U> &rhs,
      typename std::enable_if<is_implicit_cast_allowed_mask<U, T>::value, void *>::type = nullptr)
        : k(MaskHelper<Size>::cast(rhs.data())) {}

    template<typename U> Vc_ALWAYS_INLINE explicit Mask(const Mask<U> &rhs,
      typename std::enable_if<!is_implicit_cast_allowed_mask<U, T>::value, void *>::type = nullptr)
        : k(MaskHelper<Size>::cast(rhs.data())) {}

    inline explicit Mask(const bool *mem) { load(mem, Aligned); }
    template<typename Flags>
    inline explicit Mask(const bool *mem, Flags f) { load(mem, f); }

    inline void load(const bool *mem) { load(mem, Aligned); }
    template<typename Flags>
    inline void load(const bool *mem, Flags) {
        const __m512i ones = _mm512_loadu_epi32(mem, UpDownConversion<unsigned int, unsigned char>());
            //_mm512_extload_epi32(mem, UpDownConversion<unsigned int, unsigned char>(), _MM_BROADCAST32_NONE, _MM_HINT_NONE);
        k = _mm512_cmpneq_epi32_mask(ones, _mm512_setzero_epi32());
    }

    inline void store(bool *mem) const { store(mem, Aligned); }
    template<typename Flags>
    inline void store(bool *mem, Flags) const {
        const __m512i zero = _mm512_setzero_epi32();
        const __m512i one = VectorHelper<__m512i>::one();
        const __m512i tmp = _and(zero, k, one, one);
        MicIntrinsics::store<decltype(Unaligned)>(mem, tmp, UpDownConversion<unsigned int, unsigned char>());
    }

    inline bool operator==(const Mask &rhs) const { return MaskHelper<Size>::cmpeq (k, rhs.k); }
    inline bool operator!=(const Mask &rhs) const { return MaskHelper<Size>::cmpneq(k, rhs.k); }

    inline Mask operator&&(const Mask &rhs) const { return _mm512_kand(k, rhs.k); }
    inline Mask operator& (const Mask &rhs) const { return _mm512_kand(k, rhs.k); }
    inline Mask operator||(const Mask &rhs) const { return _mm512_kor (k, rhs.k); }
    inline Mask operator| (const Mask &rhs) const { return _mm512_kor (k, rhs.k); }
    inline Mask operator^ (const Mask &rhs) const { return _mm512_kxor(k, rhs.k); }
    inline Mask operator!() const { return ~k; }

    inline Mask &operator&=(const Mask &rhs) { k = _mm512_kand(k, rhs.k); return *this; }
    inline Mask &operator|=(const Mask &rhs) { k = _mm512_kor (k, rhs.k); return *this; }

    inline bool isFull () const { return MaskHelper<Size>::isFull (k); }
    inline bool isEmpty() const { return MaskHelper<Size>::isEmpty(k); }
    inline bool isMix  () const { return MaskHelper<Size>::isMix  (k); }
    inline bool isNotEmpty() const { return MaskHelper<Size>::isNotEmpty(k); }

    inline operator bool() const { return isFull(); }

    inline MaskType data () const { return k; }
    inline MaskType dataI() const { return k; }
    inline MaskType dataD() const { return k; }

    //internal function for MaskEntry::operator=
    inline void setEntry(size_t index, bool value) {
        if (value) {
            k |= (1 << index);
        } else {
            k &= ~(1 << index);
        }
    }

    inline Common::MaskEntry<Mask<T>> operator[](size_t index) { return Common::MaskEntry<Mask<T>>(*this, index); }
    inline bool operator[](size_t index) const { return static_cast<bool>(k & (1 << index)); }

    Vc_ALWAYS_INLINE Vc_PURE int count() const {
        if (Size == 16) {
            return _mm_countbits_32(k);
        } else {
            return _mm_countbits_32(k & 0xffu);
        }
    }

    /**
     * Returns the index of the first one in the mask.
     *
     * The return value is undefined if the mask is empty.
     */
    int firstOne() const { return _mm_tzcnt_32(k); }

    int toInt() const { return k; }

private:
    MaskType k;
};
template<typename T> constexpr size_t Mask<T>::Size;

struct ForeachHelper
{
    unsigned int mask;
    int bit;
    bool brk;
    template<typename T>
    inline ForeachHelper(Mask<T> _mask) :
        mask(_mask.data()),
        bit(_mm_tzcnt_32(mask)),
        brk(false)
    {}
    inline ForeachHelper(Mask<double> _mask) :
        mask(_mask.data() & 0xff),
        bit(_mm_tzcnt_32(mask)),
        brk(false)
    {}
    inline bool outer() const { return bit != sizeof(mask) * 8; }
    inline bool inner() { return (brk = !brk); }
    inline short next() const { return bit; }
    inline void step() { bit = _mm_tzcnti_32(bit, mask); }
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
    for (Vc::MIC::ForeachHelper _Vc_foreach_bit_helper(_mask_); \
            _Vc_foreach_bit_helper.outer(); \
            _Vc_foreach_bit_helper.step()) \
        for (_it_ = _Vc_foreach_bit_helper.next(); _Vc_foreach_bit_helper.inner(); )

Vc_NAMESPACE_END

#include "undomacros.h"
#include "mask.tcc"

#endif // VC_MIC_MASK_H
