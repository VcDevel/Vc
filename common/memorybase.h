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

template<typename V> class VectorPointerHelperConst
{
    typedef typename V::EntryType EntryType;
    typedef typename V::Mask Mask;
    const EntryType *const m_ptr;
    public:
        VectorPointerHelperConst(const EntryType *ptr) : m_ptr(ptr) {}

        inline operator const V() const { return V(m_ptr); }
        inline V operator+(const V &v) const { return V(m_ptr) + v; }
        inline V operator-(const V &v) const { return V(m_ptr) - v; }
        inline V operator/(const V &v) const { return V(m_ptr) / v; }
        inline V operator*(const V &v) const { return V(m_ptr) * v; }
        inline Mask operator==(const V &v) const { return V(m_ptr) == v; }
        inline Mask operator!=(const V &v) const { return V(m_ptr) != v; }
        inline Mask operator<=(const V &v) const { return V(m_ptr) <= v; }
        inline Mask operator>=(const V &v) const { return V(m_ptr) >= v; }
        inline Mask operator< (const V &v) const { return V(m_ptr) <  v; }
        inline Mask operator> (const V &v) const { return V(m_ptr) >  v; }
};
template<typename V> class VectorPointerHelper
{
    typedef typename V::EntryType EntryType;
    typedef typename V::Mask Mask;
    EntryType *const m_ptr;
    public:
        VectorPointerHelper(EntryType *ptr) : m_ptr(ptr) {}

        inline operator const V() const { return V(m_ptr); }

        inline const V &operator=(const V &v) {
            v.store(m_ptr);
            return v;
        }
#define OP(op) \
        inline V operator op(const V &v) const { \
            return V(m_ptr) op v; \
        } \
        inline const VectorPointerHelper<V> &operator op##=(const V &v) { \
            V result = V(m_ptr) op v; \
            result.store(m_ptr); \
            return *this; \
        }
        OP(+)
        OP(-)
        OP(/)
        OP(*)
#undef OP
        inline Mask operator==(const V &v) const { return V(m_ptr) == v; }
        inline Mask operator!=(const V &v) const { return V(m_ptr) != v; }
        inline Mask operator<=(const V &v) const { return V(m_ptr) <= v; }
        inline Mask operator>=(const V &v) const { return V(m_ptr) >= v; }
        inline Mask operator< (const V &v) const { return V(m_ptr) <  v; }
        inline Mask operator> (const V &v) const { return V(m_ptr) >  v; }
};
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
        /// This is an overloaded function.
        inline const EntryType scalar(unsigned int i) const { return entries()[i]; }

        /**
         * Returns a pointer to the start of the allocated memory.
         */
        inline       EntryType *entries()       { return &p()->m_mem[0]; }
        /// This is an overloaded function.
        inline const EntryType *entries() const { return &p()->m_mem[0]; }

        // omit operator[] because the EntryType* cast operator suffices

        /**
         * Cast operator to the scalar type. This allows to use the object very much like a standard
         * C array.
         */
        inline operator       EntryType*()       { return entries(); }
        /// This is an overloaded function.
        inline operator const EntryType*() const { return entries(); }

        /**
         * Returns a pointer to memory for the i-th vector.
         *
         * vector(i + 1) - vector(i) is always equal to V::Size.
         *
         * The pointer is guaranteed to be aligned correctly.
         */
        inline VectorPointerHelper<V> vector(unsigned int i) { return &entries()[i * V::Size]; }
        /// This is an overloaded function.
        inline const VectorPointerHelperConst<V> vector(unsigned int i) const { return &entries()[i * V::Size]; }

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
