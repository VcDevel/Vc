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

#ifndef VC_SSE_MEMORY_H
#define VC_SSE_MEMORY_H

#include <assert.h>

namespace Vc
{
namespace SSE
{
    template<typename V, typename Parent> class MemoryBase
    {
        private:
            Parent *p() { return static_cast<Parent *>(this); }
            const Parent *p() const { return static_cast<const Parent *>(this); }
        public:
            typedef typename V::EntryType EntryType;

            inline unsigned int entriesCount() const { return p()->entriesCount; }
            inline unsigned int vectorsCount() const { return p()->vectorsCount; }

            inline       EntryType *entries()       { return p()->entries(); }
            inline const EntryType *entries() const { return p()->entries(); }

            inline EntryType &operator[](int i)       { return entries()[i]; }
            inline EntryType  operator[](int i) const { return entries()[i]; }

            inline operator       EntryType*()       { return entries(); }
            inline operator const EntryType*() const { return entries(); }

            inline       EntryType *vector(int i)       { return &entries()[i * V::Size]; }
            inline const EntryType *vector(int i) const { return &entries()[i * V::Size]; }
    };

    template<typename V, unsigned int Size> class FixedSizeMemory : public VectorAlignedBase, public MemoryBase<V, FixedSizeMemory<V, Size> >
    {
        public:
            typedef typename V::EntryType EntryType;
        private:
            typedef MemoryBase<V, FixedSizeMemory<V, Size> > Base;
            enum {
                Alignment = 16 / sizeof(EntryType),
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

            inline FixedSizeMemory<V, Size> &operator=(const FixedSizeMemory<V, Size> &rhs) {
                for (int i = 0; i < VectorsCount; ++i) {
                    vector(i) = rhs.vector(i);
                }
                return *this;
            }
            template<typename Parent>
            inline FixedSizeMemory<V, Size> &operator=(const MemoryBase<V, Parent> &rhs) {
                assert(VectorsCount == rhs.vectorsCount());
                for (int i = 0; i < VectorsCount; ++i) {
                    vector(i) = rhs.vector(i);
                }
                return *this;
            }
            inline FixedSizeMemory<V, Size> &operator=(const V *rhs) {
                for (int i = 0; i < VectorsCount; ++i) {
                    vector(i) = rhs[i];
                }
                return *this;
            }
    };

    template<typename V> class VarSizeMemory : public MemoryBase<V, VarSizeMemory<V> >
    {
        public:
            typedef typename V::EntryType EntryType;
        private:
            typedef MemoryBase<V, VarSizeMemory<V> > Base;
            enum {
                Alignment = 16 / sizeof(EntryType),
                AlignmentMask = Alignment - 1
            };
            EntryType *m_mem;
            unsigned int m_entriesCount;
            unsigned int m_vectorsCount;
            unsigned int calcVectorsCount(unsigned int x)
            {
                unsigned int masked = x & AlignmentMask;
                return (masked == 0 ? x : x + (Alignment - masked)) / V::Size;
            }
        public:
            using Base::vector;
            inline VarSizeMemory(unsigned int size)
                : m_entriesCount(size),
                m_vectorsCount(calcVectorsCount(m_entriesCount)),
                m_mem(_mm_malloc(m_vectorsCount * sizeof(V), 16))
            {}
            inline ~VarSizeMemory()
            {
                _mm_free(m_mem);
            }
            inline unsigned int entriesCount() const { return m_entriesCount; }
            inline unsigned int vectorsCount() const { return m_vectorsCount; }

            inline       EntryType *entries()       { return &m_mem.e[0]; }
            inline const EntryType *entries() const { return &m_mem.e[0]; }

            template<unsigned int RhsSize>
            inline VarSizeMemory<V> &operator=(const FixedSizeMemory<V, RhsSize> &rhs) {
                assert(rhs::VectorsCount == vectorsCount());
                for (int i = 0; i < FixedSizeMemory<V, RhsSize>::VectorsCount; ++i) {
                    vector(i) = rhs.vector(i);
                }
                return *this;
            }
            template<typename Parent>
            inline VarSizeMemory<V> &operator=(const MemoryBase<V, Parent> &rhs) {
                assert(vectorsCount() == rhs.vectorsCount());
                for (int i = 0; i < vectorsCount(); ++i) {
                    vector(i) = rhs.vector(i);
                }
                return *this;
            }
            inline VarSizeMemory<V> &operator=(const V *rhs) {
                for (int i = 0; i < vectorsCount(); ++i) {
                    vector(i) = rhs[i];
                }
                return *this;
            }
    };

} // namespace SSE
} // namespace Vc

#endif // VC_SSE_MEMORY_H
