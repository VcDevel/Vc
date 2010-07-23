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
 * Helper class for the Memory::vector(unsigned int) class of functions.
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
 * Helper class for the Memory::vector(unsigned int) class of functions.
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

/**
 * \headerfile memorybase.h <Vc/Memory>
 */
template<typename V, typename Parent> class MemoryBase
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
         * Returns the number of scalar entries in the array. This function is optimized away
         * if a constant size array is used.
         */
        inline unsigned int entriesCount() const { return p()->entriesCount(); }
        /**
         * Returns the number of vector entries that span the array. This function is optimized away
         * if a constant size array is used.
         */
        inline unsigned int vectorsCount() const { return p()->vectorsCount(); }

        /**
         * Returns the \p i-th scalar value in the memory.
         */
        inline EntryType &scalar(unsigned int i) { return entries()[i]; }
        /// Const overload of the above function.
        inline const EntryType scalar(unsigned int i) const { return entries()[i]; }

        /**
         * Returns a pointer to the start of the allocated memory.
         */
        inline       EntryType *entries()       { return &p()->m_mem[0]; }
        /// Const overload of the above function.
        inline const EntryType *entries() const { return &p()->m_mem[0]; }

        // omit operator[] because the EntryType* cast operator suffices

        /**
         * Cast operator to the scalar type. This allows to use the object very much like a standard
         * C array.
         */
        inline operator       EntryType*()       { return entries(); }
        /// Const overload of the above function.
        inline operator const EntryType*() const { return entries(); }

        /**
         * Returns a smart object to wrap the \p i-th vector in the memory.
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
         * needed the vector(unsigned int, int) function can be used.
         */
        inline VectorPointerHelper<V, AlignedFlag> vector(unsigned int i) { return &entries()[i * V::Size]; }
        /// Const overload of the above function.
        inline const VectorPointerHelperConst<V, AlignedFlag> vector(unsigned int i) const { return &entries()[i * V::Size]; }

        /**
         * Returns a smart object to wrap the \p i-th vector + \p shift in the memory.
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
         * aligned loads. You have to use the above vector(unsigned int) function for aligned loads
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
        inline VectorPointerHelper<V, UnalignedFlag> vector(unsigned int i, int shift) { return &entries()[i * V::Size + shift]; }
        /// Const overload of the above function.
        inline const VectorPointerHelperConst<V, UnalignedFlag> vector(unsigned int i, int shift) const { return &entries()[i * V::Size + shift]; }

        /**
         * Returns the first vector in the allocated memory.
         *
         * This function is simply a shorthand for vector(0).
         */
        inline VectorPointerHelper<V, AlignedFlag> firstVector() { return entries(); }
        /// Const overload of the above function.
        inline const VectorPointerHelperConst<V, AlignedFlag> firstVector() const { return entries(); }

        /**
         * Returns the last vector in the allocated memory.
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
            for (unsigned int i = 0; i < vectorsCount(); ++i) {
                vector(i) = zero;
            }
        }

        template<typename P2>
        inline Parent &operator+=(const MemoryBase<V, P2> &rhs) {
            assert(vectorsCount() == rhs.vectorsCount());
            for (unsigned int i = 0; i < vectorsCount(); ++i) {
                vector(i) += rhs.vector(i);
            }
            return static_cast<Parent &>(*this);
        }
        template<typename P2>
        inline Parent &operator-=(const MemoryBase<V, P2> &rhs) {
            assert(vectorsCount() == rhs.vectorsCount());
            for (unsigned int i = 0; i < vectorsCount(); ++i) {
                vector(i) -= rhs.vector(i);
            }
            return static_cast<Parent &>(*this);
        }
        template<typename P2>
        inline Parent &operator*=(const MemoryBase<V, P2> &rhs) {
            assert(vectorsCount() == rhs.vectorsCount());
            for (unsigned int i = 0; i < vectorsCount(); ++i) {
                vector(i) *= rhs.vector(i);
            }
            return static_cast<Parent &>(*this);
        }
        template<typename P2>
        inline Parent &operator/=(const MemoryBase<V, P2> &rhs) {
            assert(vectorsCount() == rhs.vectorsCount());
            for (unsigned int i = 0; i < vectorsCount(); ++i) {
                vector(i) /= rhs.vector(i);
            }
            return static_cast<Parent &>(*this);
        }
        inline Parent &operator+=(EntryType rhs) {
            V v(rhs);
            for (unsigned int i = 0; i < vectorsCount(); ++i) {
                vector(i) += v;
            }
            return static_cast<Parent &>(*this);
        }
        inline Parent &operator-=(EntryType rhs) {
            V v(rhs);
            for (unsigned int i = 0; i < vectorsCount(); ++i) {
                vector(i) -= v;
            }
            return static_cast<Parent &>(*this);
        }
        inline Parent &operator*=(EntryType rhs) {
            V v(rhs);
            for (unsigned int i = 0; i < vectorsCount(); ++i) {
                vector(i) *= v;
            }
            return static_cast<Parent &>(*this);
        }
        inline Parent &operator/=(EntryType rhs) {
            V v(rhs);
            for (unsigned int i = 0; i < vectorsCount(); ++i) {
                vector(i) /= v;
            }
            return static_cast<Parent &>(*this);
        }
        template<typename P2>
        inline bool operator==(const MemoryBase<V, P2> &rhs) const {
            assert(vectorsCount() == rhs.vectorsCount());
            for (unsigned int i = 0; i < vectorsCount(); ++i) {
                if (!(V(vector(i)) == V(rhs.vector(i))).isFull()) {
                    return false;
                }
            }
            return true;
        }
        template<typename P2>
        inline bool operator!=(const MemoryBase<V, P2> &rhs) const {
            assert(vectorsCount() == rhs.vectorsCount());
            for (unsigned int i = 0; i < vectorsCount(); ++i) {
                if (!(V(vector(i)) == V(rhs.vector(i))).isEmpty()) {
                    return false;
                }
            }
            return true;
        }
        template<typename P2>
        inline bool operator<(const MemoryBase<V, P2> &rhs) const {
            assert(vectorsCount() == rhs.vectorsCount());
            for (unsigned int i = 0; i < vectorsCount(); ++i) {
                if (!(V(vector(i)) < V(rhs.vector(i))).isFull()) {
                    return false;
                }
            }
            return true;
        }
        template<typename P2>
        inline bool operator<=(const MemoryBase<V, P2> &rhs) const {
            assert(vectorsCount() == rhs.vectorsCount());
            for (unsigned int i = 0; i < vectorsCount(); ++i) {
                if (!(V(vector(i)) <= V(rhs.vector(i))).isFull()) {
                    return false;
                }
            }
            return true;
        }
        template<typename P2>
        inline bool operator>(const MemoryBase<V, P2> &rhs) const {
            assert(vectorsCount() == rhs.vectorsCount());
            for (unsigned int i = 0; i < vectorsCount(); ++i) {
                if (!(V(vector(i)) > V(rhs.vector(i))).isFull()) {
                    return false;
                }
            }
            return true;
        }
        template<typename P2>
        inline bool operator>=(const MemoryBase<V, P2> &rhs) const {
            assert(vectorsCount() == rhs.vectorsCount());
            for (unsigned int i = 0; i < vectorsCount(); ++i) {
                if (!(V(vector(i)) >= V(rhs.vector(i))).isFull()) {
                    return false;
                }
            }
            return true;
        }
};

} // namespace Vc

#endif // VC_COMMON_MEMORYBASE_H
