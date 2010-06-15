/*  This file is part of the Vc library.

    Copyright (C) 2009-2010 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Leavxr General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Leavxr General Public License for more details.

    You should have received a copy of the GNU Leavxr General Public
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
        // TODO
            enum { Size = 16 / sizeof(T) };
            typedef _M256I VectorType ALIGN(16);
            typedef T EntryType;
            typedef VectorBase<typename IndexTypeHelper<Size>::Type> IndexType;
            typedef Mask<Size> MaskType;
            typedef MaskType GatherMaskType;

            inline Vector<EntryType> &operator|= (const Vector<EntryType> &x) ALWAYS_INLINE;
            inline Vector<EntryType> &operator&= (const Vector<EntryType> &x) ALWAYS_INLINE;
            inline Vector<EntryType> &operator^= (const Vector<EntryType> &x) ALWAYS_INLINE;
            inline Vector<EntryType> &operator>>=(const Vector<EntryType> &x) ALWAYS_INLINE;
            inline Vector<EntryType> &operator<<=(const Vector<EntryType> &x) ALWAYS_INLINE;
            inline Vector<EntryType> &operator>>=(int x) ALWAYS_INLINE;
            inline Vector<EntryType> &operator<<=(int x) ALWAYS_INLINE;

            inline Vector<EntryType> operator| (const Vector<EntryType> &x) const ALWAYS_INLINE;
            inline Vector<EntryType> operator& (const Vector<EntryType> &x) const ALWAYS_INLINE;
            inline Vector<EntryType> operator^ (const Vector<EntryType> &x) const ALWAYS_INLINE;
            inline Vector<EntryType> operator>>(const Vector<EntryType> &x) const ALWAYS_INLINE;
            inline Vector<EntryType> operator<<(const Vector<EntryType> &x) const ALWAYS_INLINE;
            inline Vector<EntryType> operator>>(int x) const ALWAYS_INLINE;
            inline Vector<EntryType> operator<<(int x) const ALWAYS_INLINE;

            VectorType &data() { return d.v(); }
            const VectorType &data() const { return d.v(); }

            inline VectorBase(VectorType x) : d(x) {}
        protected:
            inline VectorBase() {}

            VectorMemoryUnion<VectorType, EntryType> d;

            static const T *_IndexesFromZero() {
                if (Size == 4) {
                    return reinterpret_cast<const T *>(_IndexesFromZero4);
                } else if (Size == 8) {
                    return reinterpret_cast<const T *>(_IndexesFromZero8);
                } else if (Size == 16) {
                    return reinterpret_cast<const T *>(_IndexesFromZero16);
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
            enum { Size = 16 / sizeof(float) };
            typedef _M256 VectorType ALIGN(16);
            typedef float EntryType;
            typedef VectorBase<IndexTypeHelper<Size>::Type> IndexType;
            typedef Mask<Size> MaskType;
            typedef MaskType GatherMaskType;

            VectorType &data() { return d.v(); }
            const VectorType &data() const { return d.v(); }

        protected:
            inline VectorBase() {}
            inline VectorBase(VectorType x) : d(x) {}

            VectorMemoryUnion<VectorType, EntryType> d;
    };

    template<> class VectorBase<double> {
        friend struct VectorHelperSize<double>;
        friend struct GatherHelper<double>;
        friend struct ScatterHelper<double>;
        friend struct GeneralHelpers;
        public:
            enum { Size = 16 / sizeof(double) };
            typedef _M256D VectorType ALIGN(16);
            typedef double EntryType;
            typedef VectorBase<IndexTypeHelper<Size>::Type> IndexType;
            typedef Mask<Size> MaskType;
            typedef MaskType GatherMaskType;

            VectorType &data() { return d.v(); }
            const VectorType &data() const { return d.v(); }

        protected:
            inline VectorBase() {}
            inline VectorBase(VectorType x) : d(x) {}

            VectorMemoryUnion<VectorType, EntryType> d;
    };

} // namespace AVX
} // namespace Vc
#endif // AVX_VECTORBASE_H
