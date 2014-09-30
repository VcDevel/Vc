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

#ifndef VC_SCALAR_MASK_H
#define VC_SCALAR_MASK_H

#include "types.h"
#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
namespace Scalar
{
template<typename T> class Mask
{
    friend class Mask<  double>;
    friend class Mask<   float>;
    friend class Mask< int32_t>;
    friend class Mask<uint32_t>;
    friend class Mask< int16_t>;
    friend class Mask<uint16_t>;

public:
    static constexpr size_t Size = 1;
    static constexpr std::size_t size() { return Size; }

    /**
     * The \c EntryType of masks is always bool, independent of \c T.
     */
    typedef bool EntryType;

    /**
     * The \c VectorEntryType, in contrast to \c EntryType, reveals information about the SIMD
     * implementation. This type is useful for the \c sizeof operator in generic functions.
     */
    typedef bool VectorEntryType;

    /**
     * The \c VectorType reveals the implementation-specific internal type used for the SIMD type.
     */
    using VectorType = bool;

    /**
     * The associated Vector<T> type.
     */
    using Vector = Scalar::Vector<T>;

        Vc_ALWAYS_INLINE Mask() {}
        Vc_ALWAYS_INLINE explicit Mask(bool b) : m(b) {}
        Vc_ALWAYS_INLINE explicit Mask(VectorSpecialInitializerZero::ZEnum) : m(false) {}
        Vc_ALWAYS_INLINE explicit Mask(VectorSpecialInitializerOne::OEnum) : m(true) {}
        Vc_INTRINSIC static Mask Zero() { return Mask{VectorSpecialInitializerZero::Zero}; }
        Vc_INTRINSIC static Mask One() { return Mask{VectorSpecialInitializerOne::One}; }

    // implicit cast
    template <typename U>
    Vc_INTRINSIC Mask(U &&rhs, Common::enable_if_mask_converts_implicitly<T, U> = nullarg)
        : m(rhs.m) {}

    // explicit cast, implemented via simd_cast (in scalar/simd_cast_caller.h)
    template <typename U>
    Vc_INTRINSIC_L explicit Mask(U &&rhs,
                                 Common::enable_if_mask_converts_explicitly<T, U> =
                                     nullarg) Vc_INTRINSIC_R;

        Vc_ALWAYS_INLINE explicit Mask(const bool *mem) : m(mem[0]) {}
        template<typename Flags> Vc_ALWAYS_INLINE explicit Mask(const bool *mem, Flags) : m(mem[0]) {}

        Vc_ALWAYS_INLINE void load(const bool *mem) { m = mem[0]; }
        template<typename Flags> Vc_ALWAYS_INLINE void load(const bool *mem, Flags) { m = mem[0]; }

        Vc_ALWAYS_INLINE void store(bool *mem) const { *mem = m; }
        template<typename Flags> Vc_ALWAYS_INLINE void store(bool *mem, Flags) const { *mem = m; }

        Vc_ALWAYS_INLINE Mask &operator=(const Mask &rhs) { m = rhs.m; return *this; }
        Vc_ALWAYS_INLINE Mask &operator=(bool rhs) { m = rhs; return *this; }

        Vc_ALWAYS_INLINE bool operator==(const Mask &rhs) const { return m == rhs.m; }
        Vc_ALWAYS_INLINE bool operator!=(const Mask &rhs) const { return m != rhs.m; }

        Vc_ALWAYS_INLINE Mask operator&&(const Mask &rhs) const { return Mask(m && rhs.m); }
        Vc_ALWAYS_INLINE Mask operator& (const Mask &rhs) const { return Mask(m && rhs.m); }
        Vc_ALWAYS_INLINE Mask operator||(const Mask &rhs) const { return Mask(m || rhs.m); }
        Vc_ALWAYS_INLINE Mask operator| (const Mask &rhs) const { return Mask(m || rhs.m); }
        Vc_ALWAYS_INLINE Mask operator^ (const Mask &rhs) const { return Mask(m ^  rhs.m); }
        Vc_ALWAYS_INLINE Mask operator!() const { return Mask(!m); }

        Vc_ALWAYS_INLINE Mask &operator&=(const Mask &rhs) { m &= rhs.m; return *this; }
        Vc_ALWAYS_INLINE Mask &operator|=(const Mask &rhs) { m |= rhs.m; return *this; }
        Vc_ALWAYS_INLINE Mask &operator^=(const Mask &rhs) { m ^= rhs.m; return *this; }

        Vc_ALWAYS_INLINE bool isFull () const { return  m; }
        Vc_ALWAYS_INLINE bool isNotEmpty() const { return m; }
        Vc_ALWAYS_INLINE bool isEmpty() const { return !m; }
        Vc_ALWAYS_INLINE bool isMix  () const { return false; }

        Vc_ALWAYS_INLINE bool data () const { return m; }
        Vc_ALWAYS_INLINE bool dataI() const { return m; }
        Vc_ALWAYS_INLINE bool dataD() const { return m; }

        template<unsigned int OtherSize>
            Vc_ALWAYS_INLINE Mask cast() const { return *this; }

        Vc_ALWAYS_INLINE bool &operator[](size_t) { return m; }
        Vc_ALWAYS_INLINE bool operator[](size_t) const { return m; }

        Vc_ALWAYS_INLINE int count() const { return m ? 1 : 0; }

        /**
         * Returns the index of the first one in the mask.
         *
         * The return value is undefined if the mask is empty.
         */
        Vc_ALWAYS_INLINE int firstOne() const { return 0; }
        Vc_ALWAYS_INLINE int toInt() const { return m ? 1 : 0; }

        template <typename G> static Vc_INTRINSIC Mask generate(G &&gen)
        {
            return Mask(gen(0));
        }

        Vc_INTRINSIC Vc_PURE Mask shifted(int amount) const
        {
            if (amount == 0) {
                return *this;
            } else {
                return Zero();
            }
        }

    private:
        bool m;
};
template<typename T> constexpr size_t Mask<T>::Size;

}  // namespace Scalar
}  // namespace Vc

#include "undomacros.h"

#endif // VC_SCALAR_MASK_H
