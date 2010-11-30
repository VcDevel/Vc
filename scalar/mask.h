/*  This file is part of the Vc library.

    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_SIMPLE_MASK_H
#define VC_SIMPLE_MASK_H

#include "types.h"

namespace Vc
{
namespace Simple
{
template<unsigned int VectorSize = 1> class Mask
{
    public:
        inline Mask() {}
        inline Mask(bool b) : m(b) {}
        inline explicit Mask(VectorSpecialInitializerZero::ZEnum) : m(false) {}
        inline explicit Mask(VectorSpecialInitializerOne::OEnum) : m(true) {}
        inline Mask(const Mask<VectorSize> *a) : m(a[0].m) {}

        inline void expand(Mask *x) { x[0].m = m; }

        inline bool operator==(const Mask &rhs) const { return m == rhs.m; }
        inline bool operator!=(const Mask &rhs) const { return m != rhs.m; }

        inline Mask operator&&(const Mask &rhs) const { return m && rhs.m; }
        inline Mask operator& (const Mask &rhs) const { return m && rhs.m; }
        inline Mask operator||(const Mask &rhs) const { return m || rhs.m; }
        inline Mask operator| (const Mask &rhs) const { return m || rhs.m; }
        inline Mask operator^ (const Mask &rhs) const { return m ^  rhs.m; }
        inline Mask operator!() const { return !m; }

        inline Mask &operator&=(const Mask &rhs) { m &= rhs.m; return *this; }
        inline Mask &operator|=(const Mask &rhs) { m |= rhs.m; return *this; }

        inline bool isFull () const { return  m; }
        inline bool isEmpty() const { return !m; }
        inline bool isMix  () const { return false; }

        inline bool data () const { return m; }
        inline bool dataI() const { return m; }
        inline bool dataD() const { return m; }

        inline operator bool() const { return isFull(); }

        template<unsigned int OtherSize>
            inline Mask cast() const { return *this; }

        inline bool operator[](int) const { return m; }

        inline int count() const { return m ? 1 : 0; }

        /**
         * Returns the index of the first one in the mask.
         *
         * The return value is undefined if the mask is empty.
         */
        int firstOne() const { return 0; }

    private:
        bool m;
};

struct ForeachHelper
{
    bool first;
    inline ForeachHelper(bool mask) : first(mask) {}
    inline void next() { first = false; }
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
    for (Vc::Simple::ForeachHelper _Vc_foreach_bit_helper(_mask_); _Vc_foreach_bit_helper.first; ) \
        for (_it_ = 0; _Vc_foreach_bit_helper.first; _Vc_foreach_bit_helper.next())

#define foreach_bit(_it_, _mask_) Vc_foreach_bit(_it_, _mask_)

} // namespace Simple
} // namespace Vc

#endif // VC_SIMPLE_MASK_H
