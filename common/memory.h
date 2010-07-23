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

namespace Vc
{

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
            std::copy(rhs.m_mem, rhs.m_mem + entriesCount(), m_mem);
            return *this;
        }
        inline Memory<V> &operator=(const EntryType *rhs) {
            std::copy(rhs, rhs + entriesCount(), m_mem);
            return *this;
        }
        inline Memory &operator=(const V &v) {
            for (unsigned int i = 0; i < vectorsCount(); ++i) {
                vector(i) = v;
            }
            return *this;
        }
};

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
        unsigned int calcVectorsCount(unsigned int x)
        {
            unsigned int masked = x & AlignmentMask;
            return (masked == 0 ? x : x + (Alignment - masked)) / V::Size;
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
            m_vectorsCount(calcVectorsCount(m_entriesCount)),
            m_mem(reinterpret_cast<EntryType *>(new V[m_vectorsCount]))
        {}

        /**
         * Frees the memory which was allocated in the constructor.
         */
        inline ~Memory()
        {
            delete[] reinterpret_cast<V *>(m_mem);
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
            std::copy(rhs.m_mem, rhs.m_mem + entriesCount(), m_mem);
            return *this;
        }

        /**
         * Overwrite all entries with the values stored in the memory at \p rhs. Note that this
         * function assumes that there are entriesCount() many values accessible from \p rhs on.
         */
        inline Memory<V> &operator=(const EntryType *rhs) {
            std::copy(rhs, rhs + entriesCount(), m_mem);
            return *this;
        }
};
} // namespace Vc

#endif // VC_COMMON_MEMORY_H
