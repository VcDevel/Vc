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

#ifndef VC_COMMON_MEMORYBASE_H
#define VC_COMMON_MEMORYBASE_H

#include <assert.h>
#include <type_traits>
#include "macros.h"

Vc_NAMESPACE_BEGIN(Common)

#define VC_MEM_OPERATOR_EQ(op) \
        template<typename T> \
        Vc_ALWAYS_INLINE typename std::enable_if<std::is_same<T, T>::value && !std::is_const<_V>::value, MemoryVector &>::type operator op##=(const T &x) { \
            const V result = value() op x; \
            result.store(&m_data[0], A()); \
            return *this; \
        }
/*dox{{{*/
/**
 * Helper class for the Memory::vector(size_t) class of functions.
 *
 * You will never need to directly make use of this class. It is an implementation detail of the
 * Memory API.
 *
 * \headerfile memorybase.h <Vc/Memory>
 *//*}}}*/
template<typename _V, typename A> class MemoryVector/*{{{*/
{
    typedef typename std::remove_cv<_V>::type V;

    typedef typename V::EntryType EntryType;
    typedef typename V::Mask Mask;
    public:
        EntryType m_data[V::Size];

        Vc_ALWAYS_INLINE MemoryVector() {}

        // disable copies because this type is supposed to alias the data in a Memory object,
        // nothing else
        MemoryVector(const MemoryVector &) = delete;
#ifndef VC_NO_MOVE_CTOR
        MemoryVector(MemoryVector &&) = delete;
#endif
        // Do not disable MemoryVector &operator=(const MemoryVector &) = delete; because it is
        // covered nicely by the operator= below.

        Vc_ALWAYS_INLINE Vc_PURE V value() const { return V(&m_data[0], A()); }

        /**
         * Cast to \p V operator.
         *
         * This function allows to assign this object to any object of type \p V.
         */
        Vc_ALWAYS_INLINE Vc_PURE operator const V() const { return value(); }

        // TODO: change to T&& once ICC can do it
        template<typename T>
        Vc_ALWAYS_INLINE typename std::enable_if<std::is_same<T, T>::value && !std::is_const<_V>::value, MemoryVector &>::type operator=(const T &x) {
            V v;
            v = x;
            v.store(&m_data[0], A());
            return *this;
        }

        VC_ALL_BINARY(VC_MEM_OPERATOR_EQ)
        VC_ALL_ARITHMETICS(VC_MEM_OPERATOR_EQ)
};/*}}}*/
#undef VC_MEM_OPERATOR_EQ

#define VC_VPH_OPERATOR(op) \
template<typename V1, typename A1, typename V2, typename A2> \
decltype(V1() op V2()) operator op(const MemoryVector<V1, A1> &x, const MemoryVector<V2, A2> &y) { \
    return x.value() op y.value(); \
}
VC_ALL_ARITHMETICS(VC_VPH_OPERATOR)
VC_ALL_BINARY     (VC_VPH_OPERATOR)
VC_ALL_COMPARES   (VC_VPH_OPERATOR)
#undef VC_VPH_OPERATOR

template<typename V, typename Parent> class MemoryRange
{
    Parent *m_parent;
    size_t m_first;
    size_t m_last;

public:
    MemoryRange(Parent *p, size_t firstIndex, size_t lastIndex)
        : m_parent(p), m_first(firstIndex), m_last(lastIndex)
    {}

    auto begin()       -> decltype(&m_parent->firstVector()) { return &m_parent->firstVector(); }
    auto begin() const -> decltype(&m_parent->firstVector()) { return &m_parent->firstVector(); }
    auto end()       -> decltype(&m_parent->lastVector()) { return &m_parent->lastVector() + 1; }
    auto end() const -> decltype(&m_parent->lastVector()) { return &m_parent->lastVector() + 1; }
};
template<typename V, typename Parent, int Dimension, typename RowMemory> class MemoryDimensionBase;
template<typename V, typename Parent, typename RowMemory> class MemoryDimensionBase<V, Parent, 1, RowMemory> // {{{1
{
    private:
        Parent *p() { return static_cast<Parent *>(this); }
        const Parent *p() const { return static_cast<const Parent *>(this); }
    public:
        /**
         * The type of the scalar entries in the array.
         */
        typedef typename V::EntryType EntryType;

        /**
         * Returns a pointer to the start of the allocated memory.
         */
        Vc_ALWAYS_INLINE Vc_PURE       EntryType *entries()       { return &p()->m_mem[0]; }
        /// Const overload of the above function.
        Vc_ALWAYS_INLINE Vc_PURE const EntryType *entries() const { return &p()->m_mem[0]; }

        /**
         * Returns the \p i-th scalar value in the memory.
         */
        Vc_ALWAYS_INLINE Vc_PURE EntryType &scalar(size_t i) { return entries()[i]; }
        /// Const overload of the above function.
        Vc_ALWAYS_INLINE Vc_PURE const EntryType scalar(size_t i) const { return entries()[i]; }

        /**
         * Cast operator to the scalar type. This allows to use the object very much like a standard
         * C array.
         */
        Vc_ALWAYS_INLINE Vc_PURE operator       EntryType*()       { return entries(); }
        /// Const overload of the above function.
        Vc_ALWAYS_INLINE Vc_PURE operator const EntryType*() const { return entries(); }

        /**
         *
         */
        inline MemoryRange<V, Parent> range(size_t firstIndex, size_t lastIndex) {
            return MemoryRange<V, Parent>(p(), firstIndex, lastIndex);
        }

        // omit operator[] because the EntryType* cast operator suffices, for dox it makes sense to
        // show it, though because it helps API discoverability.
#ifdef DOXYGEN
        /**
         * Returns the \p i-th scalar value in the memory.
         */
        inline EntryType &operator[](size_t i);
        /// Const overload of the above function.
        inline const EntryType &operator[](size_t i) const;
#endif

        /**
         * Uses a vector gather to combine the entries at the indexes in \p i into the returned
         * vector object.
         *
         * \param i  An integer vector. It determines the entries to be gathered.
         * \returns  A vector object. Modification of this object will not modify the values in
         *           memory.
         *
         * \warning  The API of this function might change in future versions of Vc to additionally
         *           support scatters.
         */
        template<typename IndexT> Vc_ALWAYS_INLINE Vc_PURE V operator[](Vector<IndexT> i) const
        {
            return V(entries(), i);
        }
};
template<typename V, typename Parent, typename RowMemory> class MemoryDimensionBase<V, Parent, 2, RowMemory> // {{{1
{
    private:
        Parent *p() { return static_cast<Parent *>(this); }
        const Parent *p() const { return static_cast<const Parent *>(this); }
    public:
        /**
         * The type of the scalar entries in the array.
         */
        typedef typename V::EntryType EntryType;

        static constexpr size_t rowCount() { return Parent::RowCount; }

        /**
         * Returns a pointer to the start of the allocated memory.
         */
        Vc_ALWAYS_INLINE Vc_PURE       EntryType *entries(size_t x = 0)       { return &p()->m_mem[x][0]; }
        /// Const overload of the above function.
        Vc_ALWAYS_INLINE Vc_PURE const EntryType *entries(size_t x = 0) const { return &p()->m_mem[x][0]; }

        /**
         * Returns the \p i,j-th scalar value in the memory.
         */
        Vc_ALWAYS_INLINE Vc_PURE EntryType &scalar(size_t i, size_t j) { return entries(i)[j]; }
        /// Const overload of the above function.
        Vc_ALWAYS_INLINE Vc_PURE const EntryType scalar(size_t i, size_t j) const { return entries(i)[j]; }

        /**
         * Returns the \p i-th row in the memory.
         */
        Vc_ALWAYS_INLINE Vc_PURE RowMemory &operator[](size_t i) {
            return RowMemory::fromRawData(entries(i));
        }
        /// Const overload of the above function.
        Vc_ALWAYS_INLINE Vc_PURE const RowMemory &operator[](size_t i) const {
            return RowMemory::fromRawData(const_cast<EntryType *>(entries(i)));
        }

        /**
         * \return the number of rows in the array.
         *
         * \note This function can be eliminated by an optimizing compiler.
         */
        Vc_ALWAYS_INLINE Vc_PURE size_t rowsCount() const { return p()->rowsCount(); }
};

//dox{{{1
/**
 * \headerfile memorybase.h <Vc/Memory>
 *
 * Common interface to all Memory classes, independent of allocation on the stack or heap.
 *
 * \param V The vector type you want to operate on. (e.g. float_v or uint_v)
 * \param Parent This type is the complete type of the class that derives from MemoryBase.
 * \param Dimension The number of dimensions the implementation provides.
 * \param RowMemory Class to be used to work on a single row.
 */
template<typename V, typename Parent, int Dimension, typename RowMemory> class MemoryBase : public MemoryDimensionBase<V, Parent, Dimension, RowMemory> //{{{1
{
    private:
        Parent *p() { return static_cast<Parent *>(this); }
        const Parent *p() const { return static_cast<const Parent *>(this); }
    public:
        /**
         * The type of the scalar entries in the array.
         */
        typedef typename V::EntryType EntryType;

        /**
         * \return the number of scalar entries in the array. This function is optimized away
         * if a constant size array is used.
         */
        Vc_ALWAYS_INLINE Vc_PURE size_t entriesCount() const { return p()->entriesCount(); }
        /**
         * \return the number of vector entries that span the array. This function is optimized away
         * if a constant size array is used.
         */
        Vc_ALWAYS_INLINE Vc_PURE size_t vectorsCount() const { return p()->vectorsCount(); }

        using MemoryDimensionBase<V, Parent, Dimension, RowMemory>::entries;
        using MemoryDimensionBase<V, Parent, Dimension, RowMemory>::scalar;

        /**
         * Return a (vectorized) iterator to the start of this memory object.
         */
        Vc_ALWAYS_INLINE       MemoryVector<      V, AlignedFlag> *begin()       { return &firstVector(); }
        //! const overload of the above
        Vc_ALWAYS_INLINE const MemoryVector<const V, AlignedFlag> *begin() const { return &firstVector(); }

        /**
         * Return a (vectorized) iterator to the end of this memory object.
         */
        Vc_ALWAYS_INLINE       MemoryVector<      V, AlignedFlag> *  end()       { return &lastVector() + 1; }
        //! const overload of the above
        Vc_ALWAYS_INLINE const MemoryVector<const V, AlignedFlag> *  end() const { return &lastVector() + 1; }

        /**
         * \param i Selects the offset, where the vector should be read.
         *
         * \return a smart object to wrap the \p i-th vector in the memory.
         *
         * The return value can be used as any other vector object. I.e. you can substitute
         * something like
         * \code
         * float_v a = ..., b = ...;
         * a += b;
         * \endcode
         * with
         * \code
         * mem.vector(i) += b;
         * \endcode
         *
         * This function ensures that only \em aligned loads and stores are used. Thus it only allows to
         * access memory at fixed strides. If access to known offsets from the aligned vectors is
         * needed the vector(size_t, int) function can be used.
         */
        Vc_ALWAYS_INLINE Vc_PURE MemoryVector<V, AlignedFlag> &vector(size_t i) {
            return *new(&entries()[i * V::Size]) MemoryVector<V, AlignedFlag>;
        }
        /** \brief Const overload of the above function
         *
         * \param i Selects the offset, where the vector should be read.
         *
         * \return a smart object to wrap the \p i-th vector in the memory.
         */
        Vc_ALWAYS_INLINE Vc_PURE const MemoryVector<const V, AlignedFlag> &vector(size_t i) const {
            return *new(const_cast<EntryType *>(&entries()[i * V::Size])) MemoryVector<const V, AlignedFlag>;
        }

        /**
         * \return a smart object to wrap the vector starting from the \p i-th scalar entry in the memory.
         *
         * Example:
         * \code
         * Memory<float_v, N> mem;
         * mem.setZero();
         * for (int i = 0; i < mem.entriesCount(); i += float_v::Size) {
         *     mem.vectorAt(i) += b;
         * }
         * \endcode
         *
         * \param i      Specifies the scalar entry from where the vector will be loaded/stored. I.e. the
         * values scalar(i), scalar(i + 1), ..., scalar(i + V::Size - 1) will be read/overwritten.
         *
         * \param align  You must take care to determine whether an unaligned load/store is
         * required. Per default an aligned load/store is used. If \p i is not a multiple of \c V::Size
         * you must pass Vc::Unaligned here.
         */
#ifdef DOXYGEN
        template<typename A> inline MemoryVector<V, A> &vectorAt(size_t i, A align = Vc::Aligned);
        /** \brief Const overload of the above function
         *
         * \return a smart object to wrap the vector starting from the \p i-th scalar entry in the memory.
         *
         * \param i      Specifies the scalar entry from where the vector will be loaded/stored. I.e. the
         * values scalar(i), scalar(i + 1), ..., scalar(i + V::Size - 1) will be read/overwritten.
         *
         * \param align  You must take care to determine whether an unaligned load/store is
         * required. Per default an aligned load/store is used. If \p i is not a multiple of \c V::Size
         * you must pass Vc::Unaligned here.
         */
        template<typename A> inline const MemoryVector<const V, A> &vectorAt(size_t i, A align = Vc::Aligned) const;
#else
        template<typename A>
        Vc_ALWAYS_INLINE Vc_PURE MemoryVector<V, A> &vectorAt(size_t i, A) {
            return *new(&entries()[i]) MemoryVector<V, A>;
        }
        template<typename A>
        Vc_ALWAYS_INLINE Vc_PURE const MemoryVector<const V, A> &vectorAt(size_t i, A) const {
            return *new(const_cast<EntryType *>(&entries()[i])) MemoryVector<const V, A>;
        }

        Vc_ALWAYS_INLINE Vc_PURE MemoryVector<V, AlignedFlag> &vectorAt(size_t i) {
            return *new(&entries()[i]) MemoryVector<V, AlignedFlag>;
        }
        Vc_ALWAYS_INLINE Vc_PURE const MemoryVector<V, AlignedFlag> &vectorAt(size_t i) const {
            return *new(const_cast<EntryType *>(&entries()[i])) MemoryVector<const V, AlignedFlag>;
        }
#endif

        /**
         * \return a smart object to wrap the \p i-th vector + \p shift in the memory.
         *
         * This function ensures that only \em unaligned loads and stores are used.
         * It allows to access memory at any location aligned to the entry type.
         *
         * \param i Selects the memory location of the i-th vector. Thus if \p V::Size == 4 and
         *          \p i is set to 3 the base address for the load/store will be the 12th entry
         *          (same as \p &mem[12]).
         * \param shift Shifts the base address determined by parameter \p i by \p shift many
         *              entries. Thus \p vector(3, 1) for \p V::Size == 4 will load/store the
         *              13th - 16th entries (same as \p &mem[13]).
         *
         * \note Any shift value is allowed as long as you make sure it stays within bounds of the
         * allocated memory. Shift values that are a multiple of \p V::Size will \em not result in
         * aligned loads. You have to use the above vector(size_t) function for aligned loads
         * instead.
         *
         * \note Thus a simple way to access vectors randomly is to set \p i to 0 and use \p shift as the
         * parameter to select the memory address:
         * \code
         * // don't use:
         * mem.vector(i / V::Size, i % V::Size) += 1;
         * // instead use:
         * mem.vector(0, i) += 1;
         * \endcode
         */
        Vc_ALWAYS_INLINE Vc_PURE MemoryVector<V, UnalignedFlag> &vector(size_t i, int shift) {
            return *new(&entries()[i * V::Size + shift]) MemoryVector<V, UnalignedFlag>;
        }
        /// Const overload of the above function.
        Vc_ALWAYS_INLINE Vc_PURE const MemoryVector<const V, UnalignedFlag> &vector(size_t i, int shift) const {
            return *new(const_cast<EntryType *>(&entries()[i * V::Size + shift])) MemoryVector<const V, UnalignedFlag>;
        }

        /**
         * \return the first vector in the allocated memory.
         *
         * This function is simply a shorthand for vector(0).
         */
        Vc_ALWAYS_INLINE Vc_PURE MemoryVector<V, AlignedFlag> &firstVector() {
            return *new(entries()) MemoryVector<V, AlignedFlag>;
        }
        /// Const overload of the above function.
        Vc_ALWAYS_INLINE Vc_PURE const MemoryVector<const V, AlignedFlag> &firstVector() const {
            return *new(const_cast<EntryType *>(entries())) MemoryVector<const V, AlignedFlag>;
        }

        /**
         * \return the last vector in the allocated memory.
         *
         * This function is simply a shorthand for vector(vectorsCount() - 1).
         */
        Vc_ALWAYS_INLINE Vc_PURE MemoryVector<V, AlignedFlag> &lastVector() {
            return *new(&entries()[vectorsCount() * V::Size - V::Size]) MemoryVector<V, AlignedFlag>;
        }
        /// Const overload of the above function.
        Vc_ALWAYS_INLINE Vc_PURE const MemoryVector<const V, AlignedFlag> &lastVector() const {
            return *new(const_cast<EntryType *>(&entries()[vectorsCount() * V::Size - V::Size])) MemoryVector<const V, AlignedFlag>;
        }

        Vc_ALWAYS_INLINE Vc_PURE V gather(const unsigned char  *indexes) const { return V(entries(), indexes); }
        Vc_ALWAYS_INLINE Vc_PURE V gather(const unsigned short *indexes) const { return V(entries(), indexes); }
        Vc_ALWAYS_INLINE Vc_PURE V gather(const unsigned int   *indexes) const { return V(entries(), indexes); }
        Vc_ALWAYS_INLINE Vc_PURE V gather(const unsigned long  *indexes) const { return V(entries(), indexes); }

        Vc_ALWAYS_INLINE void setZero() {
            V zero(Vc::Zero);
            for (size_t i = 0; i < vectorsCount(); ++i) {
                vector(i) = zero;
            }
        }

        template<typename P2, typename RM>
        inline Parent &operator+=(const MemoryBase<V, P2, Dimension, RM> &rhs) {
            assert(vectorsCount() == rhs.vectorsCount());
            for (size_t i = 0; i < vectorsCount(); ++i) {
                vector(i) += rhs.vector(i);
            }
            return static_cast<Parent &>(*this);
        }
        template<typename P2, typename RM>
        inline Parent &operator-=(const MemoryBase<V, P2, Dimension, RM> &rhs) {
            assert(vectorsCount() == rhs.vectorsCount());
            for (size_t i = 0; i < vectorsCount(); ++i) {
                vector(i) -= rhs.vector(i);
            }
            return static_cast<Parent &>(*this);
        }
        template<typename P2, typename RM>
        inline Parent &operator*=(const MemoryBase<V, P2, Dimension, RM> &rhs) {
            assert(vectorsCount() == rhs.vectorsCount());
            for (size_t i = 0; i < vectorsCount(); ++i) {
                vector(i) *= rhs.vector(i);
            }
            return static_cast<Parent &>(*this);
        }
        template<typename P2, typename RM>
        inline Parent &operator/=(const MemoryBase<V, P2, Dimension, RM> &rhs) {
            assert(vectorsCount() == rhs.vectorsCount());
            for (size_t i = 0; i < vectorsCount(); ++i) {
                vector(i) /= rhs.vector(i);
            }
            return static_cast<Parent &>(*this);
        }
        inline Parent &operator+=(EntryType rhs) {
            V v(rhs);
            for (size_t i = 0; i < vectorsCount(); ++i) {
                vector(i) += v;
            }
            return static_cast<Parent &>(*this);
        }
        inline Parent &operator-=(EntryType rhs) {
            V v(rhs);
            for (size_t i = 0; i < vectorsCount(); ++i) {
                vector(i) -= v;
            }
            return static_cast<Parent &>(*this);
        }
        inline Parent &operator*=(EntryType rhs) {
            V v(rhs);
            for (size_t i = 0; i < vectorsCount(); ++i) {
                vector(i) *= v;
            }
            return static_cast<Parent &>(*this);
        }
        inline Parent &operator/=(EntryType rhs) {
            V v(rhs);
            for (size_t i = 0; i < vectorsCount(); ++i) {
                vector(i) /= v;
            }
            return static_cast<Parent &>(*this);
        }
        template<typename P2, typename RM>
        inline bool operator==(const MemoryBase<V, P2, Dimension, RM> &rhs) const {
            assert(vectorsCount() == rhs.vectorsCount());
            for (size_t i = 0; i < vectorsCount(); ++i) {
                if (!(V(vector(i)) == V(rhs.vector(i))).isFull()) {
                    return false;
                }
            }
            return true;
        }
        template<typename P2, typename RM>
        inline bool operator!=(const MemoryBase<V, P2, Dimension, RM> &rhs) const {
            assert(vectorsCount() == rhs.vectorsCount());
            for (size_t i = 0; i < vectorsCount(); ++i) {
                if (!(V(vector(i)) == V(rhs.vector(i))).isEmpty()) {
                    return false;
                }
            }
            return true;
        }
        template<typename P2, typename RM>
        inline bool operator<(const MemoryBase<V, P2, Dimension, RM> &rhs) const {
            assert(vectorsCount() == rhs.vectorsCount());
            for (size_t i = 0; i < vectorsCount(); ++i) {
                if (!(V(vector(i)) < V(rhs.vector(i))).isFull()) {
                    return false;
                }
            }
            return true;
        }
        template<typename P2, typename RM>
        inline bool operator<=(const MemoryBase<V, P2, Dimension, RM> &rhs) const {
            assert(vectorsCount() == rhs.vectorsCount());
            for (size_t i = 0; i < vectorsCount(); ++i) {
                if (!(V(vector(i)) <= V(rhs.vector(i))).isFull()) {
                    return false;
                }
            }
            return true;
        }
        template<typename P2, typename RM>
        inline bool operator>(const MemoryBase<V, P2, Dimension, RM> &rhs) const {
            assert(vectorsCount() == rhs.vectorsCount());
            for (size_t i = 0; i < vectorsCount(); ++i) {
                if (!(V(vector(i)) > V(rhs.vector(i))).isFull()) {
                    return false;
                }
            }
            return true;
        }
        template<typename P2, typename RM>
        inline bool operator>=(const MemoryBase<V, P2, Dimension, RM> &rhs) const {
            assert(vectorsCount() == rhs.vectorsCount());
            for (size_t i = 0; i < vectorsCount(); ++i) {
                if (!(V(vector(i)) >= V(rhs.vector(i))).isFull()) {
                    return false;
                }
            }
            return true;
        }
};

Vc_NAMESPACE_END

#include "undomacros.h"

#endif // VC_COMMON_MEMORYBASE_H

// vim: foldmethod=marker
