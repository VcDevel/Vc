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
#include <mm_malloc.h>

namespace Vc
{
template<typename V, unsigned int Size = 0u> class Memory : public VectorAlignedBase, public MemoryBase<V, Memory<V, Size> >
{
    public:
        typedef typename V::EntryType EntryType;
    private:
        typedef MemoryBase<V, Memory<V, Size> > Base;
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

        inline       EntryType *entries()       { return &m_mem[0]; }
        inline const EntryType *entries() const { return &m_mem[0]; }

        template<typename Parent>
        inline Memory<V, Size> &operator=(const MemoryBase<V, Parent> &rhs) {
            assert(VectorsCount == rhs.vectorsCount());
            for (unsigned int i = 0; i < VectorsCount; ++i) {
                vector(i) = rhs.vector(i);
            }
            return *this;
        }
        inline Memory<V, Size> &operator=(const V *rhs) {
            for (unsigned int i = 0; i < VectorsCount; ++i) {
                vector(i) = rhs[i];
            }
            return *this;
        }
};

template<typename V> class Memory<V, 0u> : public MemoryBase<V, Memory<V, 0u> >
{
    public:
        typedef typename V::EntryType EntryType;
    private:
        typedef MemoryBase<V, Memory<V> > Base;
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
        inline Memory(unsigned int size)
            : m_entriesCount(size),
            m_vectorsCount(calcVectorsCount(m_entriesCount)),
            m_mem(reinterpret_cast<EntryType *>(_mm_malloc(m_vectorsCount * sizeof(V), VectorAlignment)))
        {}
        inline ~Memory()
        {
            _mm_free(m_mem);
        }
        inline unsigned int entriesCount() const { return m_entriesCount; }
        inline unsigned int vectorsCount() const { return m_vectorsCount; }

        inline       EntryType *entries()       { return &m_mem[0]; }
        inline const EntryType *entries() const { return &m_mem[0]; }

        template<typename Parent>
        inline Memory<V> &operator=(const MemoryBase<V, Parent> &rhs) {
            assert(vectorsCount() == rhs.vectorsCount());
            for (unsigned int i = 0; i < vectorsCount(); ++i) {
                vector(i) = rhs.vector(i);
            }
            return *this;
        }
        inline Memory<V> &operator=(const V *rhs) {
            for (unsigned int i = 0; i < vectorsCount(); ++i) {
                vector(i) = rhs[i];
            }
            return *this;
        }
};
} // namespace Vc

#endif // VC_COMMON_MEMORY_H
