/*  This file is part of the Vc library.

    Copyright (C) 2009-2010 Matthias Kretz <kretz@kde.org>

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

#ifndef AVX_VECTORBASE_H
#define AVX_VECTORBASE_H

#include "intrinsics.h"
#include "types.h"
#include "casts.h"
#include "const.h"

namespace Vc
{
namespace AVX
{
    template<typename T> class VectorBase {
        friend struct VectorHelperSize<float>;
        friend struct VectorHelperSize<double>;
        friend struct VectorHelperSize<int>;
        friend struct VectorHelperSize<unsigned int>;
        friend struct VectorHelperSize<short>;
        friend struct VectorHelperSize<unsigned short>;
        friend struct GatherHelper<float>;
        friend struct GatherHelper<double>;
        friend struct GatherHelper<int>;
        friend struct GatherHelper<unsigned int>;
        friend struct GatherHelper<short>;
        friend struct GatherHelper<unsigned short>;
        friend struct ScatterHelper<float>;
        friend struct ScatterHelper<double>;
        friend struct ScatterHelper<int>;
        friend struct ScatterHelper<unsigned int>;
        friend struct ScatterHelper<short>;
        friend struct ScatterHelper<unsigned short>;
        friend struct GeneralHelpers;
        public:
            typedef typename VectorTypeHelper<T>::Type VectorType;
            typedef T EntryType;
            enum { Size = sizeof(VectorType) / sizeof(EntryType) };
            typedef VectorBase<typename IndexTypeHelper<T>::Type> IndexType;
            typedef Mask<Size, sizeof(VectorType)> MaskType;

            inline Vector<EntryType> &operator|= (const Vector<EntryType> &x) ALWAYS_INLINE;
            inline Vector<EntryType> &operator&= (const Vector<EntryType> &x) ALWAYS_INLINE;
            inline Vector<EntryType> &operator^= (const Vector<EntryType> &x) ALWAYS_INLINE;
            inline Vector<EntryType> &operator>>=(const VectorBase<EntryType> &x) ALWAYS_INLINE;
            inline Vector<EntryType> &operator<<=(const VectorBase<EntryType> &x) ALWAYS_INLINE;
            inline Vector<EntryType> &operator>>=(int x) ALWAYS_INLINE;
            inline Vector<EntryType> &operator<<=(int x) ALWAYS_INLINE;

            inline Vector<EntryType> operator| (const Vector<EntryType> &x) const ALWAYS_INLINE;
            inline Vector<EntryType> operator& (const Vector<EntryType> &x) const ALWAYS_INLINE;
            inline Vector<EntryType> operator^ (const Vector<EntryType> &x) const ALWAYS_INLINE;
            inline Vector<EntryType> operator>>(const VectorBase<EntryType> &x) const ALWAYS_INLINE;
            inline Vector<EntryType> operator<<(const VectorBase<EntryType> &x) const ALWAYS_INLINE;
            inline Vector<EntryType> operator>>(int x) const ALWAYS_INLINE;
            inline Vector<EntryType> operator<<(int x) const ALWAYS_INLINE;

            VectorType &data() { return d.v(); }
            const VectorType &data() const { return d.v(); }

        protected:
            enum { HasVectorDivision = 1 };
            inline VectorBase() {}
            inline VectorBase(VectorType x) : d(x) {}

            typedef Common::VectorMemoryUnion<VectorType, EntryType> StorageType;
            StorageType d;

            static const T *_IndexesFromZero() {
                switch (sizeof(EntryType)) {
                case 1: // char
                    return reinterpret_cast<const T *>(_IndexesFromZero8);
                case 2: // short
                    return reinterpret_cast<const T *>(_IndexesFromZero16);
                case 4: // int
                    return reinterpret_cast<const T *>(_IndexesFromZero32);
                }
                return 0;
            }
    };

    template<> class VectorBase<float> {
        friend struct VectorHelperSize<float>;
        friend struct GatherHelper<float>;
        friend struct ScatterHelper<float>;
        friend struct GeneralHelpers;
        public:
            typedef typename VectorTypeHelper<float>::Type VectorType;
            typedef float EntryType;
            enum { Size = sizeof(VectorType) / sizeof(EntryType) };
            typedef VectorBase<IndexTypeHelper<float>::Type> IndexType;
            typedef Mask<Size, sizeof(VectorType)> MaskType;

            VectorType &data() { return d.v(); }
            const VectorType &data() const { return d.v(); }

        protected:
            enum { HasVectorDivision = 1 };
            inline VectorBase() {}
            inline VectorBase(VectorType x) : d(x) {}

            typedef Common::VectorMemoryUnion<VectorType, EntryType> StorageType;
            StorageType d;
    };

    template<> class VectorBase<double> {
        friend struct VectorHelperSize<double>;
        friend struct GatherHelper<double>;
        friend struct ScatterHelper<double>;
        friend struct GeneralHelpers;
        public:
            typedef typename VectorTypeHelper<double>::Type VectorType;
            typedef double EntryType;
            enum { Size = sizeof(VectorType) / sizeof(EntryType) };
            typedef VectorBase<IndexTypeHelper<double>::Type> IndexType;
            typedef Mask<Size, sizeof(VectorType)> MaskType;

            VectorType &data() { return d.v(); }
            const VectorType &data() const { return d.v(); }

        protected:
            enum { HasVectorDivision = 1 };
            inline VectorBase() {}
            inline VectorBase(VectorType x) : d(x) {}

            typedef Common::VectorMemoryUnion<VectorType, EntryType> StorageType;
            StorageType d;
    };

} // namespace AVX
} // namespace Vc

#endif // AVX_VECTORBASE_H
