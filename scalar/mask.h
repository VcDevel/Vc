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

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)
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

        Vc_ALWAYS_INLINE Mask() {}
        Vc_ALWAYS_INLINE explicit Mask(bool b) : m(b) {}
        Vc_ALWAYS_INLINE explicit Mask(VectorSpecialInitializerZero::ZEnum) : m(false) {}
        Vc_ALWAYS_INLINE explicit Mask(VectorSpecialInitializerOne::OEnum) : m(true) {}

        template<typename U> Vc_ALWAYS_INLINE Mask(const Mask<U> &a,
          typename std::enable_if<is_implicit_cast_allowed_mask<U, T>::value, void *>::type = nullptr)
            : m(a.m) {}

        template<typename U> Vc_ALWAYS_INLINE explicit Mask(const Mask<U> &a,
          typename std::enable_if<!is_implicit_cast_allowed_mask<U, T>::value, void *>::type = nullptr)
            : m(a.m) {}

        Vc_ALWAYS_INLINE explicit Mask(const bool *mem) : m(mem[0]) {}
        template<typename Flags> Vc_ALWAYS_INLINE explicit Mask(const bool *mem, Flags) : m(mem[0]) {}

        Vc_ALWAYS_INLINE void load(const bool *mem) { m = mem[0]; }
        template<typename Flags> Vc_ALWAYS_INLINE void load(const bool *mem, Flags) { m = mem[0]; }

        Vc_ALWAYS_INLINE void store(bool *mem) const { *mem = m; }
        template<typename Flags> Vc_ALWAYS_INLINE void store(bool *mem, Flags) const { *mem = m; }

        Vc_ALWAYS_INLINE Mask &operator=(const Mask &rhs) { m = rhs.m; return *this; }
        Vc_ALWAYS_INLINE Mask &operator=(bool rhs) { m = rhs; return *this; }

        Vc_ALWAYS_INLINE void expand(Mask *x) { x[0].m = m; }

        Vc_ALWAYS_INLINE bool operator==(const Mask &rhs) const { return Mask(m == rhs.m); }
        Vc_ALWAYS_INLINE bool operator!=(const Mask &rhs) const { return Mask(m != rhs.m); }

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

#ifndef VC_NO_AUTOMATIC_BOOL_FROM_MASK
        Vc_ALWAYS_INLINE operator bool() const { return isFull(); }
#endif

        template<unsigned int OtherSize>
            Vc_ALWAYS_INLINE Mask cast() const { return *this; }

        Vc_ALWAYS_INLINE bool &operator[](size_t) { return m; }
        Vc_ALWAYS_INLINE bool operator[](size_t) const { return m; }

        Vc_ALWAYS_INLINE unsigned int count() const { return m ? 1 : 0; }

        /**
         * Returns the index of the first one in the mask.
         *
         * The return value is undefined if the mask is empty.
         */
        Vc_ALWAYS_INLINE unsigned int firstOne() const { return 0; }
        Vc_ALWAYS_INLINE int toInt() const { return m ? 1 : 0; }

    private:
        bool m;
};
template<typename T> constexpr size_t Mask<T>::Size;

struct ForeachHelper
{
    bool continu;
    Vc_ALWAYS_INLINE ForeachHelper(bool mask) : continu(mask) {}
    Vc_ALWAYS_INLINE void next() { continu = false; }
};

#define Vc_foreach_bit(_it_, _mask_) \
    for (Vc::Scalar::ForeachHelper Vc__make_unique(foreach_bit_obj)(_mask_); Vc__make_unique(foreach_bit_obj).continu; Vc__make_unique(foreach_bit_obj).next()) \
        for (_it_ = 0; Vc__make_unique(foreach_bit_obj).continu; Vc__make_unique(foreach_bit_obj).next())

Vc_IMPL_NAMESPACE_END

#include "undomacros.h"

#endif // VC_SCALAR_MASK_H
