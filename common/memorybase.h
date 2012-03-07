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

#ifndef VC_COMMON_MEMORYBASE_H
#define VC_COMMON_MEMORYBASE_H

#include <assert.h>

namespace Vc
{

/**
 * Helper class for the Memory::vector(size_t) class of functions.
 *
 * You will never need to directly make use of this class. It is an implementation detail of the
 * Memory API.
 *
 * \headerfile memorybase.h <Vc/Memory>
 */
template<typename V, typename A> class VectorPointerHelperConst
{
    typedef typename V::EntryType EntryType;
    typedef typename V::Mask Mask;
    const EntryType *const m_ptr;
    public:
        VectorPointerHelperConst(const EntryType *ptr) : m_ptr(ptr) {}

        /**
         * Cast to \p V operator.
         *
         * This function allows to assign this object to any object of type \p V.
         */
        inline operator const V() const { return V(m_ptr, Internal::FlagObject<A>::the()); }
        inline V operator+(const V &v) const { return V(m_ptr, Internal::FlagObject<A>::the()) + v; }
        inline V operator-(const V &v) const { return V(m_ptr, Internal::FlagObject<A>::the()) - v; }
        inline V operator/(const V &v) const { return V(m_ptr, Internal::FlagObject<A>::the()) / v; }
        inline V operator*(const V &v) const { return V(m_ptr, Internal::FlagObject<A>::the()) * v; }
        inline Mask operator==(const V &v) const { return V(m_ptr, Internal::FlagObject<A>::the()) == v; }
        inline Mask operator!=(const V &v) const { return V(m_ptr, Internal::FlagObject<A>::the()) != v; }
        inline Mask operator<=(const V &v) const { return V(m_ptr, Internal::FlagObject<A>::the()) <= v; }
        inline Mask operator>=(const V &v) const { return V(m_ptr, Internal::FlagObject<A>::the()) >= v; }
        inline Mask operator< (const V &v) const { return V(m_ptr, Internal::FlagObject<A>::the()) <  v; }
        inline Mask operator> (const V &v) const { return V(m_ptr, Internal::FlagObject<A>::the()) >  v; }
};

/**
 * Helper class for the Memory::vector(size_t) class of functions.
 *
 * You will never need to directly make use of this class. It is an implementation detail of the
 * Memory API.
 *
 * \headerfile memorybase.h <Vc/Memory>
 */
template<typename V, typename A> class VectorPointerHelper
{
    typedef typename V::EntryType EntryType;
    typedef typename V::Mask Mask;
    EntryType *const m_ptr;
    public:
        VectorPointerHelper(EntryType *ptr) : m_ptr(ptr) {}

        /**
         * Cast to \p V operator.
         *
         * This function allows to assign this object to any object of type \p V.
         */
        inline operator const V() const { return V(m_ptr, Internal::FlagObject<A>::the()); }

        inline VectorPointerHelper &operator=(const V &v) {
            v.store(m_ptr, Internal::FlagObject<A>::the());
            return *this;
        }
        inline VectorPointerHelper &operator+=(const V &v) {
            V result = V(m_ptr, Internal::FlagObject<A>::the()) + v;
            result.store(m_ptr, Internal::FlagObject<A>::the());
            return *this;
        }
        inline VectorPointerHelper &operator-=(const V &v) {
            V result = V(m_ptr, Internal::FlagObject<A>::the()) - v;
            result.store(m_ptr, Internal::FlagObject<A>::the());
            return *this;
        }
        inline VectorPointerHelper &operator*=(const V &v) {
            V result = V(m_ptr, Internal::FlagObject<A>::the()) * v;
            result.store(m_ptr, Internal::FlagObject<A>::the());
            return *this;
        }
        inline VectorPointerHelper &operator/=(const V &v) {
            V result = V(m_ptr, Internal::FlagObject<A>::the()) / v;
            result.store(m_ptr, Internal::FlagObject<A>::the());
            return *this;
        }
        inline V operator+(const V &v) const { return V(m_ptr, Internal::FlagObject<A>::the()) + v; }
        inline V operator-(const V &v) const { return V(m_ptr, Internal::FlagObject<A>::the()) - v; }
        inline V operator*(const V &v) const { return V(m_ptr, Internal::FlagObject<A>::the()) * v; }
        inline V operator/(const V &v) const { return V(m_ptr, Internal::FlagObject<A>::the()) / v; }
        inline Mask operator==(const V &v) const { return V(m_ptr, Internal::FlagObject<A>::the()) == v; }
        inline Mask operator!=(const V &v) const { return V(m_ptr, Internal::FlagObject<A>::the()) != v; }
        inline Mask operator<=(const V &v) const { return V(m_ptr, Internal::FlagObject<A>::the()) <= v; }
        inline Mask operator>=(const V &v) const { return V(m_ptr, Internal::FlagObject<A>::the()) >= v; }
        inline Mask operator< (const V &v) const { return V(m_ptr, Internal::FlagObject<A>::the()) <  v; }
        inline Mask operator> (const V &v) const { return V(m_ptr, Internal::FlagObject<A>::the()) >  v; }
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
        inline       EntryType *entries()       { return &p()->m_mem[0]; }
        /// Const overload of the above function.
        inline const EntryType *entries() const { return &p()->m_mem[0]; }

        /**
         * Returns the \p i-th scalar value in the memory.
         */
        inline EntryType &scalar(size_t i) { return entries()[i]; }
        /// Const overload of the above function.
        inline const EntryType scalar(size_t i) const { return entries()[i]; }

        /**
         * Cast operator to the scalar type. This allows to use the object very much like a standard
         * C array.
         */
        inline operator       EntryType*()       { return entries(); }
        /// Const overload of the above function.
        inline operator const EntryType*() const { return entries(); }

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
        template<typename IndexT> inline V operator[](Vector<IndexT> i) const
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

        inline size_t rowCount() const { return Parent::RowCount; }

        /**
         * Returns a pointer to the start of the allocated memory.
         */
        inline       EntryType *entries(size_t x = 0)       { return &p()->m_mem[x][0]; }
        /// Const overload of the above function.
        inline const EntryType *entries(size_t x = 0) const { return &p()->m_mem[x][0]; }

        /**
         * Returns the \p i,j-th scalar value in the memory.
         */
        inline EntryType &scalar(size_t i, size_t j) { return entries(i)[j]; }
        /// Const overload of the above function.
        inline const EntryType scalar(size_t i, size_t j) const { return entries(i)[j]; }

        /**
         * Returns the \p i-th row in the memory.
         */
        inline RowMemory &operator[](size_t i) {
            return *new(entries(i)) RowMemory;
        }
        /// Const overload of the above function.
        inline const RowMemory &operator[](size_t i) const {
            return *new(const_cast<EntryType *>(entries(i))) RowMemory;
        }

        /**
         * \return the number of rows in the array.
         *
         * \note This function can be eliminated by an optimizing compiler.
         */
        inline size_t rowsCount() const { return p()->rowsCount(); }
};

//{{{1
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
        inline size_t entriesCount() const { return p()->entriesCount(); }
        /**
         * \return the number of vector entries that span the array. This function is optimized away
         * if a constant size array is used.
         */
        inline size_t vectorsCount() const { return p()->vectorsCount(); }

        using MemoryDimensionBase<V, Parent, Dimension, RowMemory>::entries;
        using MemoryDimensionBase<V, Parent, Dimension, RowMemory>::scalar;

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
        inline VectorPointerHelper<V, AlignedFlag> vector(size_t i) { return &entries()[i * V::Size]; }
        /** \brief Const overload of the above function
         *
         * \param i Selects the offset, where the vector should be read.
         *
         * \return a smart object to wrap the \p i-th vector in the memory.
         */
        inline const VectorPointerHelperConst<V, AlignedFlag> vector(size_t i) const { return &entries()[i * V::Size]; }

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
        template<typename A> inline VectorPointerHelper<V, A> vectorAt(size_t i, A align = Vc::Aligned);
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
        template<typename A> inline const VectorPointerHelperConst<V, A> vectorAt(size_t i, A align = Vc::Aligned) const;
#else
        template<typename A>
        inline VectorPointerHelper<V, A> vectorAt(size_t i, A) { return &entries()[i]; }
        template<typename A>
        inline const VectorPointerHelperConst<V, A> vectorAt(size_t i, A) const { return &entries()[i]; }

        inline VectorPointerHelper<V, AlignedFlag> vectorAt(size_t i) { return &entries()[i]; }
        inline const VectorPointerHelperConst<V, AlignedFlag> vectorAt(size_t i) const { return &entries()[i]; }
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
        inline VectorPointerHelper<V, UnalignedFlag> vector(size_t i, int shift) { return &entries()[i * V::Size + shift]; }
        /// Const overload of the above function.
        inline const VectorPointerHelperConst<V, UnalignedFlag> vector(size_t i, int shift) const { return &entries()[i * V::Size + shift]; }

        /**
         * \return the first vector in the allocated memory.
         *
         * This function is simply a shorthand for vector(0).
         */
        inline VectorPointerHelper<V, AlignedFlag> firstVector() { return entries(); }
        /// Const overload of the above function.
        inline const VectorPointerHelperConst<V, AlignedFlag> firstVector() const { return entries(); }

        /**
         * \return the last vector in the allocated memory.
         *
         * This function is simply a shorthand for vector(vectorsCount() - 1).
         */
        inline VectorPointerHelper<V, AlignedFlag> lastVector() { return &entries()[vectorsCount() * V::Size - V::Size]; }
        /// Const overload of the above function.
        inline const VectorPointerHelperConst<V, AlignedFlag> lastVector() const { return &entries()[vectorsCount() * V::Size - V::Size]; }

        inline V gather(const unsigned char  *indexes) const { return V(entries(), indexes); }
        inline V gather(const unsigned short *indexes) const { return V(entries(), indexes); }
        inline V gather(const unsigned int   *indexes) const { return V(entries(), indexes); }
        inline V gather(const unsigned long  *indexes) const { return V(entries(), indexes); }

        inline void setZero() {
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

} // namespace Vc

#endif // VC_COMMON_MEMORYBASE_H
