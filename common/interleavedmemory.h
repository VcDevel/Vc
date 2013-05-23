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

Vc_NAMESPACE_BEGIN(Common)

/**
 * \internal
 *
 * Helper interface to make m_indexes in InterleavedMemoryAccessBase behave like an integer vector.
 * Only that the entries are successive entries from the given start index.
 */
template<size_t StructSize> class SuccessiveEntries
{
    size_t m_first;
public:
    typedef SuccessiveEntries AsArg;
    constexpr SuccessiveEntries(size_t first) : m_first(first) {}
    constexpr Vc_PURE size_t operator[](size_t offset) const { return m_first + offset * StructSize; }
    constexpr Vc_PURE size_t data() const { return m_first; }
    constexpr Vc_PURE SuccessiveEntries operator+(const SuccessiveEntries &rhs) const { return SuccessiveEntries(m_first + rhs.m_first); }
    constexpr Vc_PURE SuccessiveEntries operator*(const SuccessiveEntries &rhs) const { return SuccessiveEntries(m_first * rhs.m_first); }
};

/**
 * \internal
 */
template<typename V, typename I> struct InterleavedMemoryAccessBase
{
    // Partial specialization doesn't work for functions without partial specialization of the whole
    // class. Therefore we capture the contents of InterleavedMemoryAccessBase in a macro to easily
    // copy it into its specializations.
    typedef typename V::EntryType T;
    typedef typename V::AsArg VArg;
    typedef T Ta Vc_MAY_ALIAS;
    const I m_indexes;
    Ta *const m_data;

    Vc_ALWAYS_INLINE InterleavedMemoryAccessBase(typename I::AsArg indexes, Ta *data)
        : m_indexes(indexes), m_data(data)
    {
    }

    // implementations of the following are in {scalar,sse,avx}/interleavedmemory.tcc
    inline void deinterleave(V &v0, V &v1) const;
    inline void deinterleave(V &v0, V &v1, V &v2) const;
    inline void deinterleave(V &v0, V &v1, V &v2, V &v3) const;
    inline void deinterleave(V &v0, V &v1, V &v2, V &v3, V &v4) const;
    inline void deinterleave(V &v0, V &v1, V &v2, V &v3, V &v4, V &v5) const;
    inline void deinterleave(V &v0, V &v1, V &v2, V &v3, V &v4, V &v5, V &v6) const;
    inline void deinterleave(V &v0, V &v1, V &v2, V &v3, V &v4, V &v5, V &v6, V &v7) const;

    inline void interleave(VArg v0, VArg v1);
    inline void interleave(VArg v0, VArg v1, VArg v2);
    inline void interleave(VArg v0, VArg v1, VArg v2, VArg v3);
    inline void interleave(VArg v0, VArg v1, VArg v2, VArg v3, VArg v4);
    inline void interleave(VArg v0, VArg v1, VArg v2, VArg v3, VArg v4, VArg v5);
    inline void interleave(VArg v0, VArg v1, VArg v2, VArg v3, VArg v4, VArg v5, VArg v6);
    inline void interleave(VArg v0, VArg v1, VArg v2, VArg v3, VArg v4, VArg v5, VArg v6, VArg v7);
};

/**
 * \internal
 */
// delay execution of the deinterleaving gather until operator=
template<size_t StructSize, typename V, typename I = typename V::IndexType> struct InterleavedMemoryReadAccess : public InterleavedMemoryAccessBase<V, I>
{
    typedef InterleavedMemoryAccessBase<V, I> Base;
    typedef typename Base::Ta Ta;

    Vc_ALWAYS_INLINE InterleavedMemoryReadAccess(Ta *data, typename I::AsArg indexes)
        : Base(indexes * I(StructSize), data)
    {
    }
};

template<typename I> struct CheckIndexesUnique
{
#ifdef NDEBUG
    static Vc_INTRINSIC void test(const I &) {}
#else
    static void test(const I &indexes)
    {
        const I test = indexes.sorted();
        VC_ASSERT(I::Size == 1 || (test == test.rotated(1)).isEmpty())
    }
#endif
};
template<size_t S> struct CheckIndexesUnique<SuccessiveEntries<S> >
{
    static Vc_INTRINSIC void test(const SuccessiveEntries<S> &) {}
};

/**
 * \internal
 */
template<size_t StructSize, typename V, typename I = typename V::IndexType> struct InterleavedMemoryAccess : public InterleavedMemoryReadAccess<StructSize, V, I>
{
    typedef InterleavedMemoryAccessBase<V, I> Base;
    typedef typename Base::Ta Ta;

    Vc_ALWAYS_INLINE InterleavedMemoryAccess(Ta *data, typename I::AsArg indexes)
        : InterleavedMemoryReadAccess<StructSize, V, I>(data, indexes)
    {
        CheckIndexesUnique<I>::test(indexes);
    }

#define _VC_SCATTER_ASSIGNMENT(LENGTH, parameters) \
    Vc_ALWAYS_INLINE void operator=(const VectorTuple<LENGTH, V> &rhs) \
    { \
        static_assert(LENGTH <= StructSize, "You_are_trying_to_scatter_more_data_into_the_struct_than_it_has"); \
        this->interleave parameters ; \
    } \
    Vc_ALWAYS_INLINE void operator=(const VectorTuple<LENGTH, const V> &rhs) \
    { \
        static_assert(LENGTH <= StructSize, "You_are_trying_to_scatter_more_data_into_the_struct_than_it_has"); \
        this->interleave parameters ; \
    }
    _VC_SCATTER_ASSIGNMENT(2, (rhs.l, rhs.r))
    _VC_SCATTER_ASSIGNMENT(3, (rhs.l.l, rhs.l.r, rhs.r));
    _VC_SCATTER_ASSIGNMENT(4, (rhs.l.l.l, rhs.l.l.r, rhs.l.r, rhs.r));
    _VC_SCATTER_ASSIGNMENT(5, (rhs.l.l.l.l, rhs.l.l.l.r, rhs.l.l.r, rhs.l.r, rhs.r));
    _VC_SCATTER_ASSIGNMENT(6, (rhs.l.l.l.l.l, rhs.l.l.l.l.r, rhs.l.l.l.r, rhs.l.l.r, rhs.l.r, rhs.r));
    _VC_SCATTER_ASSIGNMENT(7, (rhs.l.l.l.l.l.l, rhs.l.l.l.l.l.r, rhs.l.l.l.l.r, rhs.l.l.l.r, rhs.l.l.r, rhs.l.r, rhs.r));
    _VC_SCATTER_ASSIGNMENT(8, (rhs.l.l.l.l.l.l.l, rhs.l.l.l.l.l.l.r, rhs.l.l.l.l.l.r, rhs.l.l.l.l.r, rhs.l.l.l.r, rhs.l.l.r, rhs.l.r, rhs.r));
#undef _VC_SCATTER_ASSIGNMENT

private:
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
    typedef typename V::AsArg VArg;
    typedef typename I::AsArg IndexType;
    enum Constants { StructSize = sizeof(S) / sizeof(T) };
    typedef InterleavedMemoryAccess<StructSize, V> Access;
    typedef InterleavedMemoryReadAccess<StructSize, V> ReadAccess;
    typedef InterleavedMemoryAccess<StructSize, V, SuccessiveEntries<StructSize> > AccessSuccessiveEntries;
    typedef InterleavedMemoryReadAccess<StructSize, V, SuccessiveEntries<StructSize> > ReadSuccessiveEntries;
    typedef T Ta Vc_MAY_ALIAS;
    Ta *const m_data;

    static_assert((sizeof(S) / sizeof(T)) * sizeof(T) == sizeof(S), "InterleavedMemoryAccess_does_not_support_packed_structs");

public:
    /**
     * Constructs the wrapper object.
     *
     * \param s A pointer to a C-array.
     */
    Vc_ALWAYS_INLINE InterleavedMemoryWrapper(S *s)
        : m_data(reinterpret_cast<Ta *>(s))
    {
    }

    /**
     * Interleaved scatter/gather access.
     *
     * Assuming you have a struct of floats and a vector of \p indexes into the array, this function
     * can be used to access the struct entries as vectors using the minimal number of store or load
     * instructions.
     *
     * \param indexes Vector of indexes that determine the gather locations.
     *
     * \return A special (magic) object that executes the loads and deinterleave on assignment to a
     * vector tuple.
     *
     * Example:
     * \code
     * struct Foo {
     *   float x, y, z;
     * };
     *
     * void fillWithBar(Foo *_data, uint_v indexes)
     * {
     *   Vc::InterleavedMemoryWrapper<Foo, float_v> data(_data);
     *   const float_v x = bar(1);
     *   const float_v y = bar(2);
     *   const float_v z = bar(3);
     *   data[indexes] = (x, y, z);
     *   // it's also possible to just store a subset at the front of the struct:
     *   data[indexes] = (x, y);
     *   // if you want to store a single entry, use scatter:
     *   z.scatter(_data, &Foo::x, indexes);
     * }
     *
     * float_v normalizeStuff(Foo *_data, uint_v indexes)
     * {
     *   Vc::InterleavedMemoryWrapper<Foo, float_v> data(_data);
     *   float_v x, y, z;
     *   (x, y, z) = data[indexes];
     *   // it is also possible to just load a subset from the front of the struct:
     *   // (x, y) = data[indexes];
     *   return Vc::sqrt(x * x + y * y + z * z);
     * }
     * \endcode
     *
     * You may think of the gather operation (or scatter as the inverse) like this:
\verbatim
             Memory: {x0 y0 z0 x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4 x5 y5 z5 x6 y6 z6 x7 y7 z7 x8 y8 z8}
            indexes: [5, 0, 1, 7]
Result in (x, y, z): ({x5 x0 x1 x7}, {y5 y0 y1 y7}, {z5 z0 z1 z7})
\endverbatim
     *
     * \warning If \p indexes contains non-unique entries on scatter, the result is undefined. If
     * \c NDEBUG is not defined the implementation will assert that the \p indexes entries are unique.
     */
    Vc_ALWAYS_INLINE Access operator[](IndexType indexes)
    {
        return Access(m_data, indexes);
    }

    /// const overload (gathers only) of the above function
    Vc_ALWAYS_INLINE ReadAccess operator[](IndexType indexes) const
    {
        return ReadAccess(m_data, indexes);
    }

    /// alias of the above function
    Vc_ALWAYS_INLINE ReadAccess gather(IndexType indexes) const { return operator[](indexes); }

    /**
     * Interleaved access.
     *
     * This function is an optimization of the function above, for cases where the index vector
     * contains consecutive values. It will load \p V::Size consecutive entries from memory and
     * deinterleave them into Vc vectors.
     *
     * \param first The first of \p V::Size indizes to be accessed.
     *
     * \return A special (magic) object that executes the loads and deinterleave on assignment to a
     * vector tuple.
     *
     * Example:
     * \code
     * struct Foo {
     *   float x, y, z;
     * };
     *
     * void foo(Foo *_data)
     * {
     *   Vc::InterleavedMemoryWrapper<Foo, float_v> data(_data);
     *   for (size_t i = 0; i < 32U; i += float_v::Size) {
     *     float_v x, y, z;
     *     (x, y, z) = data[i];
     *     // now:
     *     // x = { _data[i].x, _data[i + 1].x, _data[i + 2].x, ... }
     *     // y = { _data[i].y, _data[i + 1].y, _data[i + 2].y, ... }
     *     // z = { _data[i].z, _data[i + 1].z, _data[i + 2].z, ... }
     *     ...
     *   }
     * }
     * \endcode
     */
    Vc_ALWAYS_INLINE ReadSuccessiveEntries operator[](size_t first) const
    {
        return ReadSuccessiveEntries(m_data, first);
    }

    Vc_ALWAYS_INLINE AccessSuccessiveEntries operator[](size_t first)
    {
        return AccessSuccessiveEntries(m_data, first);
    }

    //Vc_ALWAYS_INLINE Access scatter(I indexes, VArg v0, VArg v1);
};
Vc_NAMESPACE_END

Vc_PUBLIC_NAMESPACE_BEGIN
using Common::InterleavedMemoryWrapper;
Vc_NAMESPACE_END

#include "undomacros.h"

#endif // VC_COMMON_INTERLEAVEDMEMORY_H
