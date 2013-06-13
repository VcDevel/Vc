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

#ifdef CAN_OFFLOAD
#pragma offload_attribute(push, target(mic))
#endif

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)

template<unsigned int VectorSize> struct MaskHelper;
template<> struct MaskHelper<8> {
    typedef __mmask8  Type;
    static inline bool isFull (Type k) { k = _mm512_kmovlhb(k, k); return _mm512_kortestc(k, k); }
    static inline bool isEmpty(Type k) { return _mm512_kortestz(k, k); }
    static inline bool isNotEmpty(Type k) { return !isEmpty(k); }
    static inline bool isMix  (Type k) { return !isFull(k) && !isEmpty(k); }
    static inline bool cmpeq  (Type k1, Type k2) { return isEmpty(_mm512_kxor(k1, k2)); }
    static inline bool cmpneq (Type k1, Type k2) { return isNotEmpty(_mm512_kxor(k1, k2)); }
};
template<> struct MaskHelper<16> {
    typedef __mmask16 Type;
    static inline bool isFull (Type k) { return _mm512_kortestc(k, k); }
    static inline bool isEmpty(Type k) { return _mm512_kortestz(k, k); }
    static inline bool isNotEmpty(Type k) { return !isEmpty(k); }
    static inline bool isMix  (Type k) { return !isFull(k) && !isEmpty(k); }
    static inline bool cmpeq  (Type k1, Type k2) { return isEmpty(_mm512_kxor(k1, k2)); }
    static inline bool cmpneq (Type k1, Type k2) { return isNotEmpty(_mm512_kxor(k1, k2)); }
};

template<unsigned int VectorSize> class Mask
{
    friend class Mask<8u>;
    friend class Mask<16u>;
    typedef typename MaskHelper<VectorSize>::Type M;
public:
    enum Constants {
        Size = VectorSize
    };
    typedef Mask<VectorSize> AsArg; // for now only ICC can compile this code and it is not broken :)
    inline Mask() {}
    inline Mask(M _k) : k(_k) {}
    inline explicit Mask(VectorSpecialInitializerZero::ZEnum) : k(0) {}
    inline explicit Mask(VectorSpecialInitializerOne::OEnum) : k(VectorSize == 16 ? 0xffff : 0xff) {}
    inline explicit Mask(bool b) : k(b ? (VectorSize == 16 ? 0xffff : 0xff) : 0) {}
    inline Mask(const Mask<VectorSize / 2> &a, const Mask<VectorSize / 2> &b) : k(a.k | (b.k << 8)) {}
    template<unsigned int OtherSize> explicit inline Mask(const Mask<OtherSize> &x) : k(x.k) {
        if (OtherSize != VectorSize) {
            enum { Shift = VectorSize < OtherSize ? VectorSize : OtherSize };
            const unsigned short mask = (0xffffu << Shift) & 0xffffu;
            k &= ~mask;
        }
    }

    inline bool operator==(const Mask &rhs) const { return MaskHelper<VectorSize>::cmpeq (k, rhs.k); }
    inline bool operator!=(const Mask &rhs) const { return MaskHelper<VectorSize>::cmpneq(k, rhs.k); }

    inline Mask operator&&(const Mask &rhs) const { return _mm512_kand(k, rhs.k); }
    inline Mask operator& (const Mask &rhs) const { return _mm512_kand(k, rhs.k); }
    inline Mask operator||(const Mask &rhs) const { return _mm512_kor (k, rhs.k); }
    inline Mask operator| (const Mask &rhs) const { return _mm512_kor (k, rhs.k); }
    inline Mask operator^ (const Mask &rhs) const { return _mm512_kxor(k, rhs.k); }
    inline Mask operator!() const { return ~k; }

    inline Mask &operator&=(const Mask &rhs) { k = _mm512_kand(k, rhs.k); return *this; }
    inline Mask &operator|=(const Mask &rhs) { k = _mm512_kor (k, rhs.k); return *this; }

    inline bool isFull () const { return MaskHelper<VectorSize>::isFull (k); }
    inline bool isEmpty() const { return MaskHelper<VectorSize>::isEmpty(k); }
    inline bool isMix  () const { return MaskHelper<VectorSize>::isMix  (k); }
    inline bool isNotEmpty() const { return MaskHelper<VectorSize>::isNotEmpty(k); }

    inline operator bool() const { return isFull(); }

    inline M data () const { return k; }
    inline M dataI() const { return k; }
    inline M dataD() const { return k; }

    template<unsigned int OtherSize>
    inline Mask<OtherSize> cast() const { return Mask<OtherSize>(k); }

    inline bool operator[](int index) const { return static_cast<bool>(k & (1 << index)); }

    inline int count() const { return _mm_countbits_32(k); }

    /**
     * Returns the index of the first one in the mask.
     *
     * The return value is undefined if the mask is empty.
     */
    int firstOne() const { return _mm_tzcnt_32(k); }

private:
    M k;
};

template<> inline int Mask<8u>::count() const { return _mm_countbits_32(k & 0xffu); }

struct ForeachHelper
{
    unsigned int mask;
    int bit;
    bool brk;
    inline ForeachHelper(Mask<16u> _mask) :
        mask(_mask.data()),
        bit(_mm_tzcnt_32(mask)),
        brk(false)
    {}
    inline ForeachHelper(Mask<8u> _mask) :
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

#endif // VC_MIC_MASK_H
