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

#ifndef SSE_VECTORBASE_H
#define SSE_VECTORBASE_H

#include "intrinsics.h"
#include "types.h"
#include "casts.h"
#include "const.h"

namespace Vc
{
namespace SSE
{
    template<typename T> class VectorBase {
        friend struct VectorHelperSize<float>;
        friend struct VectorHelperSize<double>;
        friend struct VectorHelperSize<int>;
        friend struct VectorHelperSize<unsigned int>;
        friend struct VectorHelperSize<short>;
        friend struct VectorHelperSize<unsigned short>;
        friend struct VectorHelperSize<float8>;
        friend struct GeneralHelpers;
        public:
            enum { Size = 16 / sizeof(T) };
            typedef _M128I VectorType;
            typedef T EntryType;
            typedef VectorBase<typename IndexTypeHelper<Size>::Type> IndexType;
            typedef Mask<Size> MaskType;
            typedef MaskType GatherMaskType;
#if defined VC_MSVC && defined _WIN32
            typedef const Vector<T> &AsArg;
#else
            typedef Vector<T> AsArg;
#endif

            inline INTRINSIC_L Vector<EntryType> &operator|= (const VectorBase<EntryType> &x) INTRINSIC_R;
            inline INTRINSIC_L Vector<EntryType> &operator&= (const VectorBase<EntryType> &x) INTRINSIC_R;
            inline INTRINSIC_L Vector<EntryType> &operator^= (const VectorBase<EntryType> &x) INTRINSIC_R;
            inline Vector<EntryType> &operator>>=(const VectorBase<EntryType> &x);
            inline Vector<EntryType> &operator<<=(const VectorBase<EntryType> &x);
            inline INTRINSIC_L Vector<EntryType> &operator>>=(int x) INTRINSIC_R;
            inline INTRINSIC_L Vector<EntryType> &operator<<=(int x) INTRINSIC_R;

            inline INTRINSIC_L Vector<EntryType> operator| (const VectorBase<EntryType> &x) const INTRINSIC_R PURE;
            inline INTRINSIC_L Vector<EntryType> operator& (const VectorBase<EntryType> &x) const INTRINSIC_R PURE;
            inline INTRINSIC_L Vector<EntryType> operator^ (const VectorBase<EntryType> &x) const INTRINSIC_R PURE;
            inline INTRINSIC_L Vector<EntryType> operator>>(const VectorBase<EntryType> &x) const INTRINSIC_R PURE;
            inline INTRINSIC_L Vector<EntryType> operator<<(const VectorBase<EntryType> &x) const INTRINSIC_R PURE;
            inline INTRINSIC_L Vector<EntryType> operator>>(int x) const INTRINSIC_R PURE;
            inline INTRINSIC_L Vector<EntryType> operator<<(int x) const INTRINSIC_R PURE;

            VectorType &data() { return d.v(); }
            const VectorType &data() const { return d.v(); }

            inline VectorBase(VectorType x) : d(x) {}
        protected:
            enum { HasVectorDivision = 0 };
            inline VectorBase() {}

            typedef Common::VectorMemoryUnion<VectorType, EntryType> StorageType;
            StorageType d;

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

    template<> class VectorBase<float8> {
        friend struct VectorHelperSize<float8>;
        friend struct VectorHelperSize<float>;
        friend struct GeneralHelpers;
        public:
            enum { Size = 8 };
            typedef M256 VectorType;
            typedef float EntryType;
            typedef VectorBase<IndexTypeHelper<Size>::Type> IndexType;
            typedef Float8Mask MaskType;
            typedef Float8GatherMask GatherMaskType;

            typedef ParameterHelper<VectorType>::ByValue ByValue;
#if defined VC_MSVC && defined _WIN32
            typedef const Vector<float8> &AsArg;
#else
            typedef Vector<float8> AsArg;
#endif

            VectorType &data() { return d.v(); }
            const VectorType &data() const { return d.v(); }

        protected:
            enum { HasVectorDivision = 1 };
            inline VectorBase() {}
            inline VectorBase(ByValue x) : d(x) {}

            typedef Common::VectorMemoryUnion<VectorType, EntryType> StorageType;
            StorageType d;
    };

    template<> class VectorBase<float> {
        friend struct VectorHelperSize<float>;
        friend struct GeneralHelpers;
        public:
            enum { Size = 16 / sizeof(float) };
            typedef _M128 VectorType;
            typedef float EntryType;
            typedef VectorBase<IndexTypeHelper<Size>::Type> IndexType;
            typedef Mask<Size> MaskType;
            typedef MaskType GatherMaskType;
#if defined VC_MSVC && defined _WIN32
            typedef const Vector<float> &AsArg;
#else
            typedef Vector<float> AsArg;
#endif

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
        friend struct GeneralHelpers;
        public:
            enum { Size = 16 / sizeof(double) };
            typedef _M128D VectorType;
            typedef double EntryType;
            typedef VectorBase<IndexTypeHelper<Size>::Type> IndexType;
            typedef Mask<Size> MaskType;
            typedef MaskType GatherMaskType;
#if defined VC_MSVC && defined _WIN32
            typedef const Vector<double> &AsArg;
#else
            typedef Vector<double> AsArg;
#endif

            VectorType &data() { return d.v(); }
            const VectorType &data() const { return d.v(); }

        protected:
            enum { HasVectorDivision = 1 };
            inline VectorBase() {}
            inline VectorBase(VectorType x) : d(x) {}

            typedef Common::VectorMemoryUnion<VectorType, EntryType> StorageType;
            StorageType d;
    };

} // namespace SSE
} // namespace Vc
#endif // SSE_VECTORBASE_H
