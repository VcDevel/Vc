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

    inline ALWAYS_INLINE InterleavedMemoryAccessBase(typename I::AsArg indexes, Ta *data)
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

    inline ALWAYS_INLINE InterleavedMemoryAccess(Ta *data, I indexes)
        : Base(indexes * I(StructSize), data)
    {
    }
};

/**
 * Wraps a pointer to memory with convenience functions to access it via vectors.
 *
 * \param S The type of the struct.
 * \param V The type of the vector to be returned when read. This should reflect the type of the
 * members inside the struct.
 *
 * \see operator[]
 * \ingroup Utilities
 * \headerfile interleavedmemory.h <Vc/Memory>
 */
template<typename S, typename V> class InterleavedMemoryWrapper
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;
    typedef typename I::AsArg IndexType;
    typedef InterleavedMemoryAccess<sizeof(S) / sizeof(T), V> Access;
    typedef T Ta MAY_ALIAS;
    Ta *const m_data;

    VC_STATIC_ASSERT((sizeof(S) / sizeof(T)) * sizeof(T) == sizeof(S), InterleavedMemoryAccess_does_not_support_packed_structs);

public:
    /**
     * Constructs the wrapper object.
     *
     * \param s A pointer to a C-array.
     */
    inline ALWAYS_INLINE InterleavedMemoryWrapper(S *s)
        : m_data(reinterpret_cast<Ta *>(s))
    {
    }

    /**
     * Interleaved gather.
     *
     * Assuming you have a struct of floats and a vector of \p indexes into the array, this function
     * can be used to return the struct entries as vectors using the minimal number of load
     * instructions.
     *
     * Example:
     * \code
     * struct Foo {
     *   float x, y, z;
     * };
     *
     * float_v normalizeStuff(Foo *_data, uint_v indexes)
     * {
     *   Vc::InterleavedMemoryWrapper<Foo, float_v> data(_data);
     *   float_v x, y, z;
     *   (x, y, z) = data[indexes];
     *   return Vc::sqrt(x * x + y * y + z * z);
     * }
     * \endcode
     *
     * You may think of the operation like this:
\verbatim
             Memory: {x0 y0 z0 x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4 x5 y5 z5 x6 y6 z6 x7 y7 z7 x8 y8 z8}
            indexes: [5, 0, 1, 7]
Result in (x, y, z): ({x5 x0 x1 x7}, {y5 y0 y1 y7}, {z5 z0 z1 z7})
\endverbatim
     */
    inline ALWAYS_INLINE Access operator[](IndexType indexes) const
    {
        return Access(m_data, indexes);
    }

    /// alias of the above function
    inline ALWAYS_INLINE Access gather(IndexType indexes) const { return operator[](indexes); }

    //inline ALWAYS_INLINE void scatter(I indexes, const V &v0, V v1
};
} // namespace Common

using Common::InterleavedMemoryWrapper;

} // namespace Vc

#include "undomacros.h"

#endif // VC_COMMON_INTERLEAVEDMEMORY_H
