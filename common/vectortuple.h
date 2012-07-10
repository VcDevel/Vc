/*  This file is part of the Vc library. {{{

    Copyright (C) 2012 Matthias Kretz <kretz@kde.org>

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

}}}*/

#ifndef VC_COMMON_VECTORTUPLE_H
#define VC_COMMON_VECTORTUPLE_H

#include "macros.h"

namespace Vc
{
namespace Common
{

template<size_t StructSize, typename V> struct InterleavedMemoryAccess;

template<int Length, typename V> class VectorTuple
{
    typedef typename V::EntryType T;
    friend class VectorTuple<Length + 1, V>;
    typedef V *VC_RESTRICT Ptr;
    Ptr pointers[Length];

public:
    VectorTuple(V *const a, V *const b)
    {
        pointers[0] = a;
        pointers[1] = b;
    }

    VectorTuple(const VectorTuple<Length - 1, V> &list, V *const a)
    {
        int i = 0;
        for (; i < Length - 1; ++i) {
            pointers[i] = list.pointers[i];
        }
        pointers[i] = a;
    }

    VectorTuple<Length + 1, V> operator,(V &a) const
    {
        return VectorTuple<Length + 1, V>(*this, &a);
    }

    template<size_t StructSize> ALWAYS_INLINE
    void operator=(const InterleavedMemoryAccess<StructSize, V> &access) const
    {
        VC_STATIC_ASSERT(Length <= StructSize, You_are_trying_to_extract_more_data_from_the_struct_than_it_has);
        switch (Length) {
        case 2:
            access.deinterleave(*pointers[0], *pointers[1]);
            break;
        case 3:
            access.deinterleave(*pointers[0], *pointers[1], *pointers[2]);
            break;
        case 4:
            access.deinterleave(*pointers[0], *pointers[1], *pointers[2], *pointers[3]);
            break;
        case 5:
            access.deinterleave(*pointers[0], *pointers[1], *pointers[2], *pointers[3], *pointers[4]);
            break;
        case 6:
            access.deinterleave(*pointers[0], *pointers[1], *pointers[2], *pointers[3], *pointers[4], *pointers[5]);
            break;
        case 7:
            access.deinterleave(*pointers[0], *pointers[1], *pointers[2], *pointers[3], *pointers[4], *pointers[5], *pointers[6]);
            break;
        case 8:
            access.deinterleave(*pointers[0], *pointers[1], *pointers[2], *pointers[3], *pointers[4], *pointers[5], *pointers[6], *pointers[7]);
            break;
        }
    }
};

} // namespace Common

Common::VectorTuple<2, float_v> operator,(float_v &a, float_v &b)
{
    return Common::VectorTuple<2, float_v>(&a, &b);
}

} // namespace Vc

#include "undomacros.h"

#endif // VC_COMMON_VECTORTUPLE_H
