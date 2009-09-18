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

namespace Vc
{
namespace SSE
{
    template<typename T> class _Memory : public VectorAlignedBase
    {
        private:
            enum {
                Size = VectorBase<T>::Size
            };
            typedef typename VectorBase<T>::EntryType EntryType;
            typedef typename VectorBase<T>::VectorType VectorType;
            EntryType d[Size];
        public:
            inline int size() const { return Size; }
            inline EntryType &operator[](int i) { return d[i]; }
            inline EntryType operator[](int i) const { return d[i]; }
            inline operator EntryType*() { return &d[0]; }
            inline operator const EntryType*() const { return &d[0]; }

            inline _Memory<T> &operator=(const _Memory<T> &rhs) {
                const VectorType tmp = VectorHelper<T>::load(&rhs.d[0]);
                VectorHelper<VectorType>::store(&d[0], tmp);
                return *this;
            }
            inline _Memory<T> &operator=(const VectorBase<T> &rhs) {
                VectorHelper<VectorType>::store(&d[0], rhs.data());
                return *this;
            }

    };

} // namespace SSE
} // namespace Vc

#endif // VC_SSE_MEMORY_H
