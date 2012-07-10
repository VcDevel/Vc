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

#ifndef VC_COMMON_INTERLEAVEDMEMORY_H
#define VC_COMMON_INTERLEAVEDMEMORY_H

#include "macros.h"

namespace Vc
{
namespace Common
{

template<typename V> struct InterleavedMemoryAccessBase
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;
    typedef T Ta MAY_ALIAS;
    const I m_indexes;
    Ta *const m_data;

    InterleavedMemoryAccessBase(I indexes, Ta *data)
        : m_indexes(indexes), m_data(data)
    {
    }

    // implementations of the following are in {scalar,sse,avx}/interleavedmemory.tcc
    void deinterleave(V &v0, V &v1) const;
    void deinterleave(V &v0, V &v1, V &v2) const;
    void deinterleave(V &v0, V &v1, V &v2, V &v3) const;
    void deinterleave(V &v0, V &v1, V &v2, V &v3, V &v4) const;
    void deinterleave(V &v0, V &v1, V &v2, V &v3, V &v4, V &v5) const;
    void deinterleave(V &v0, V &v1, V &v2, V &v3, V &v4, V &v5, V &v6) const;
    void deinterleave(V &v0, V &v1, V &v2, V &v3, V &v4, V &v5, V &v6, V &v7) const;
};

// delay execution of the deinterleaving gather until operator=
template<size_t StructSize, typename V> struct InterleavedMemoryAccess : public InterleavedMemoryAccessBase<V>
{
    typedef InterleavedMemoryAccessBase<V> Base;
    typedef typename Base::Ta Ta;
    typedef typename Base::I I;

    InterleavedMemoryAccess(Ta *data, I indexes)
        : Base(indexes * I(StructSize), data)
    {
    }
};

/**
 * \param S The type of the struct.
 * \param V The type of the vector to be returned when read. This should reflect the type of the
 * members inside the struct.
 *
 * Example:
 * \code
 * struct Foo {
 *   int a, b, c;
 * };
 * Foo *data = Vc::malloc<Vc::AlignOnVector>(1024);
 * Vc::InterleavedMemoryWrapper<Foo, int_v> data_v(data);
 * \endcode
 */
template<typename S, typename V> class InterleavedMemoryWrapper
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;
    typedef InterleavedMemoryAccess<sizeof(S) / sizeof(T), V> Access;
    typedef T Ta MAY_ALIAS;
    Ta *const m_data;

    VC_STATIC_ASSERT((sizeof(S) / sizeof(T)) * sizeof(T) == sizeof(S), InterleavedMemoryAccess_does_not_support_packed_structs);

public:
    InterleavedMemoryWrapper(S *s)
        : m_data(reinterpret_cast<Ta *>(s))
    {
    }

    Access operator[](I indexes) const
    {
        return Access(m_data, indexes);
    }
};
} // namespace Common

using Common::InterleavedMemoryWrapper;

} // namespace Vc

#include "undomacros.h"

#endif // VC_COMMON_INTERLEAVEDMEMORY_H
