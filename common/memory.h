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

#ifndef VC_COMMON_MEMORY_H
#define VC_COMMON_MEMORY_H

#include "memorybase.h"
#include <assert.h>
#include <algorithm>
#include <cstring>

namespace Vc
{

/**
 * Allocates memory on the Heap with alignment and padding.
 *
 * Memory that was allocated with this function must be released with Vc::free! Other methods might
 * work but are not portable.
 *
 * \param n Specifies the number of scalar values the allocated memory must be able to store.
 *
 * \warning The standard malloc function specifies the number of Bytes to allocate whereas this
 *          function specifies the number of values, thus differing in a factor of sizeof(T)
 *
 * \see Vc::free
 *
 * \ingroup Utilities
 * \headerfile memory.h <Vc/Memory>
 */
template<typename T, Vc::MallocAlignment A>
inline T *ALWAYS_INLINE malloc(size_t n)
{
    return static_cast<T *>(Internal::Helper::malloc<A>(n * sizeof(T)));
}

/**
 * Frees memory that was allocated with Vc::malloc.
 *
 * \ingroup Utilities
 * \headerfile memory.h <Vc/Memory>
 */
template<typename T>
inline void ALWAYS_INLINE free(T *p)
{
    Internal::Helper::free(p);
}

/**
 * A helper class to simplify usage of correctly aligned and padded memory, allowing both vector and
 * scalar access.
 *
 * Example:
 * \code
    Vc::Memory<int_v, 11> array;

    // scalar access:
    for (int i = 0; i < array.entriesCount(); ++i) {
        int x = array[i]; // read
        array[i] = x;     // write
    }
    // more explicit alternative:
    for (int i = 0; i < array.entriesCount(); ++i) {
        int x = array.scalar(i); // read
        array.scalar(i) = x;     // write
    }

    // vector access:
    for (int i = 0; i < array.vectorsCount(); ++i) {
        int_v x = array.vector(i); // read
        array.vector(i) = x;       // write
    }
 * \endcode
 * This code allocates a small array and implements three equivalent loops (that do nothing useful).
 * The loops show how scalar and vector read/write access is best implemented.
 *
 * Since the size of 11 is not a multiple of int_v::Size (unless you use the
 * scalar Vc implementation) the last write access of the vector loop would normally be out of
 * bounds. But the Memory class automatically pads the memory such that the whole array can be
 * accessed with correctly aligned memory addresses.
 *
 * \param V The vector type you want to operate on. (e.g. float_v or uint_v)
 * \param Size The number of entries of the scalar base type the memory should hold. This
 * is thus the same number as you would use for a normal C array (e.g. float mem[11] becomes
 * Memory<float_v, 11> mem).
 *
 * \see Memory<V, 0u>
 *
 * \ingroup Utilities
 * \headerfile memory.h <Vc/Memory>
 */
template<typename V, unsigned int Size = 0u> class Memory : public VectorAlignedBase, public MemoryBase<V, Memory<V, Size> >
{
    public:
        typedef typename V::EntryType EntryType;
    private:
        typedef MemoryBase<V, Memory<V, Size> > Base;
        friend class MemoryBase<V, Memory<V, Size> >;
        enum {
            Alignment = V::Size,
            AlignmentMask = Alignment - 1,
            MaskedSize = Size & AlignmentMask,
            Padding = Alignment - MaskedSize,
            PaddedSize = MaskedSize == 0 ? Size : Size + Padding
        };
#if defined(__INTEL_COMPILER) && defined(_WIN32)
		__declspec(align(__alignof(VectorAlignedBase)))
#endif
        EntryType m_mem[PaddedSize];
    public:
        using Base::vector;
        enum {
            EntriesCount = Size,
            VectorsCount = PaddedSize / V::Size
        };
        inline unsigned int entriesCount() const { return EntriesCount; }
        inline unsigned int vectorsCount() const { return VectorsCount; }

        template<typename Parent>
        inline Memory<V> &operator=(const MemoryBase<V, Parent> &rhs) {
            assert(vectorsCount() == rhs.vectorsCount());
            std::memcpy(m_mem, rhs.m_mem, entriesCount() * sizeof(EntryType));
            return *this;
        }
        inline Memory<V> &operator=(const EntryType *rhs) {
            std::memcpy(m_mem, rhs, entriesCount() * sizeof(EntryType));
            return *this;
        }
        inline Memory &operator=(const V &v) {
            for (unsigned int i = 0; i < vectorsCount(); ++i) {
                vector(i) = v;
            }
            return *this;
        }
}
#if defined(__INTEL_COMPILER) && !defined(_WIN32)
__attribute__((__aligned__(__alignof(VectorAlignedBase))))
#endif
;

/**
 * A helper class that is very similar to Memory<V, Size> but with dynamically allocated memory and
 * thus dynamic size.
 *
 * Example:
 * \code
    unsigned int size = 11;
    Vc::Memory<int_v> array(size);

    // scalar access:
    for (int i = 0; i < array.entriesCount(); ++i) {
        array[i] = i;
    }

    // vector access:
    for (int i = 0; i < array.vectorsCount(); ++i) {
        array.vector(i) = int_v::IndexesFromZero() + i * int_v::Size;
    }
 * \endcode
 * This code allocates a small array with 11 scalar entries
 * and implements two equivalent loops that initialize the memory.
 * The scalar loop writes each individual int. The vectorized loop writes int_v::Size values to
 * memory per iteration. Since the size of 11 is not a multiple of int_v::Size (unless you use the
 * scalar Vc implementation) the last write access of the vector loop would normally be out of
 * bounds. But the Memory class automatically pads the memory such that the whole array can be
 * accessed with correctly aligned memory addresses.
 * (Note: the scalar loop can be auto-vectorized, except for the last three assignments.)
 *
 * \param V The vector type you want to operate on. (e.g. float_v or uint_v)
 *
 * \see Memory<V, Size>
 *
 * \ingroup Utilities
 * \headerfile memory.h <Vc/Memory>
 */
template<typename V> class Memory<V, 0u> : public MemoryBase<V, Memory<V, 0u> >
{
    public:
        typedef typename V::EntryType EntryType;
    private:
        typedef MemoryBase<V, Memory<V> > Base;
        friend class MemoryBase<V, Memory<V> >;
        enum {
            Alignment = V::Size,
            AlignmentMask = Alignment - 1
        };
        unsigned int m_entriesCount;
        unsigned int m_vectorsCount;
        EntryType *m_mem;
        unsigned int calcPaddedEntriesCount(unsigned int x)
        {
            unsigned int masked = x & AlignmentMask;
            return (masked == 0 ? x : x + (Alignment - masked));
        }
    public:
        using Base::vector;

        /**
         * Allocate enough memory to access \p size values of type \p V::EntryType.
         *
         * The allocated memory is aligned and padded correctly for fully vectorized access.
         */
        inline Memory(unsigned int size)
            : m_entriesCount(size),
            m_vectorsCount(calcPaddedEntriesCount(m_entriesCount)),
            m_mem(Vc::malloc<EntryType, Vc::AlignOnVector>(m_vectorsCount))
        {
            m_vectorsCount /= V::Size;
        }

        /**
         * Copy the memory into a new memory area.
         *
         * The allocated memory is aligned and padded correctly for fully vectorized access.
         */
        template<typename Parent>
        inline Memory(const MemoryBase<V, Parent> &rhs)
            : m_entriesCount(rhs.entriesCount()),
            m_vectorsCount(rhs.vectorsCount()),
            m_mem(Vc::malloc<EntryType, Vc::AlignOnVector>(m_vectorsCount * V::Size))
        {
            std::memcpy(m_mem, rhs.m_mem, entriesCount() * sizeof(EntryType));
        }

        /**
         * Overload of the above function.
         *
         * (Because C++ would otherwise not use the templated cctor and use a default-constructed cctor instead.)
         */
        inline Memory(const Memory<V, 0u> &rhs)
            : m_entriesCount(rhs.entriesCount()),
            m_vectorsCount(rhs.vectorsCount()),
            m_mem(Vc::malloc<EntryType, Vc::AlignOnVector>(m_vectorsCount * V::Size))
        {
            std::memcpy(m_mem, rhs.m_mem, entriesCount() * sizeof(EntryType));
        }

        /**
         * Frees the memory which was allocated in the constructor.
         */
        inline ~Memory()
        {
            Vc::free(m_mem);
        }

        inline void swap(Memory &rhs) {
            std::swap(m_mem, rhs.m_mem);
            std::swap(m_entriesCount, rhs.m_entriesCount);
            std::swap(m_vectorsCount, rhs.m_vectorsCount);
        }

        inline unsigned int entriesCount() const { return m_entriesCount; }
        inline unsigned int vectorsCount() const { return m_vectorsCount; }

        /**
         * Overwrite all entries with the values stored in rhs. This function requires the
         * vectorsCount() of the left and right object to be equal.
         */
        template<typename Parent>
        inline Memory<V> &operator=(const MemoryBase<V, Parent> &rhs) {
            assert(vectorsCount() == rhs.vectorsCount());
            std::memcpy(m_mem, rhs.m_mem, entriesCount() * sizeof(EntryType));
            return *this;
        }

        /**
         * Overwrite all entries with the values stored in the memory at \p rhs. Note that this
         * function assumes that there are entriesCount() many values accessible from \p rhs on.
         */
        inline Memory<V> &operator=(const EntryType *rhs) {
            std::memcpy(m_mem, rhs, entriesCount() * sizeof(EntryType));
            return *this;
        }
};

/**
 * Prefetch the cacheline containing \p addr for a single read access.
 *
 * This prefetch completely bypasses the cache, not evicting any other data.
 *
 * \warning The prefetch API is not finalized and likely to change in following versions.
 *
 * \ingroup Utilities
 * \headerfile memory.h <Vc/Memory>
 */
inline void ALWAYS_INLINE prefetchForOneRead(const void *addr)
{
    Internal::Helper::prefetchForOneRead(addr);
}

/**
 * Prefetch the cacheline containing \p addr for modification.
 *
 * This prefetch evicts data from the cache. So use it only for data you really will use. When the
 * target system supports it the cacheline will be marked as modified while prefetching, saving work
 * later on.
 *
 * \warning The prefetch API is not finalized and likely to change in following versions.
 *
 * \ingroup Utilities
 * \headerfile memory.h <Vc/Memory>
 */
inline void ALWAYS_INLINE prefetchForModify(const void *addr)
{
    Internal::Helper::prefetchForModify(addr);
}

/**
 * Prefetch the cacheline containing \p addr to L1 cache.
 *
 * This prefetch evicts data from the cache. So use it only for data you really will use.
 *
 * \warning The prefetch API is not finalized and likely to change in following versions.
 *
 * \ingroup Utilities
 * \headerfile memory.h <Vc/Memory>
 */
inline void ALWAYS_INLINE prefetchClose(const void *addr)
{
    Internal::Helper::prefetchClose(addr);
}

/**
 * Prefetch the cacheline containing \p addr to L2 cache.
 *
 * This prefetch evicts data from the cache. So use it only for data you really will use.
 *
 * \warning The prefetch API is not finalized and likely to change in following versions.
 *
 * \ingroup Utilities
 * \headerfile memory.h <Vc/Memory>
 */
inline void ALWAYS_INLINE prefetchMid(const void *addr)
{
    Internal::Helper::prefetchMid(addr);
}

/**
 * Prefetch the cacheline containing \p addr to L3 cache.
 *
 * This prefetch evicts data from the cache. So use it only for data you really will use.
 *
 * \warning The prefetch API is not finalized and likely to change in following versions.
 *
 * \ingroup Utilities
 * \headerfile memory.h <Vc/Memory>
 */
inline void ALWAYS_INLINE prefetchFar(const void *addr)
{
    Internal::Helper::prefetchFar(addr);
}

} // namespace Vc

namespace std
{
    template<typename V> inline void swap(Vc::Memory<V> &a, Vc::Memory<V> &b) { a.swap(b); }
} // namespace std

#endif // VC_COMMON_MEMORY_H
