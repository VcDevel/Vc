/*  This file is part of the Vc library.

    Copyright (C) 2009-2011 Matthias Kretz <kretz@kde.org>

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

#ifndef SSE_TYPES_H
#define SSE_TYPES_H

#include "intrinsics.h"
#include "../common/storage.h"

#define VC_DOUBLE_V_SIZE 2
#define VC_FLOAT_V_SIZE 4
#define VC_SFLOAT_V_SIZE 8
#define VC_INT_V_SIZE 4
#define VC_UINT_V_SIZE 4
#define VC_SHORT_V_SIZE 8
#define VC_USHORT_V_SIZE 8

#include "../common/types.h"
#include "macros.h"

Vc_NAMESPACE_BEGIN(Vc_IMPL_NAMESPACE)
    template<typename T> class Vector;
    template<typename T> class WriteMaskedVector;

    // define our own long because on Windows64 long == int while on Linux long == max. register width
    // since we want to have a type that depends on 32 vs. 64 bit we need to do some special casing on Windows
#ifdef _WIN64
    typedef __int64 _long;
    typedef unsigned __int64 _ulong;
#else
    typedef long _long;
    typedef unsigned long _ulong;
#endif

    template<typename T> class Mask;

    template<typename T> struct ParameterHelper {
        typedef T ByValue;
        typedef T & Reference;
        typedef const T & ConstRef;
    };

    template<typename T> struct VectorHelper {};

    template<unsigned int Size> struct IndexTypeHelper;
    template<> struct IndexTypeHelper<2u> { typedef          int   Type; };
    template<> struct IndexTypeHelper<4u> { typedef          int   Type; };
    template<> struct IndexTypeHelper<8u> { typedef unsigned short Type; };
    template<> struct IndexTypeHelper<16u>{ typedef unsigned char  Type; };

    template<typename T> struct CtorTypeHelper { typedef T Type; };
    template<> struct CtorTypeHelper<short> { typedef int Type; };
    template<> struct CtorTypeHelper<unsigned short> { typedef unsigned int Type; };
    template<> struct CtorTypeHelper<float> { typedef double Type; };

    template<typename T> struct ExpandTypeHelper { typedef T Type; };
    template<> struct ExpandTypeHelper<short> { typedef int Type; };
    template<> struct ExpandTypeHelper<unsigned short> { typedef unsigned int Type; };
    template<> struct ExpandTypeHelper<float> { typedef double Type; };

    template<typename T> struct VectorTypeHelper { typedef __m128i Type; };
    template<> struct VectorTypeHelper<double>   { typedef __m128d Type; };
    template<> struct VectorTypeHelper< float>   { typedef __m128  Type; };

    template<typename T> struct DetermineGatherMask { typedef T Type; };

    template<typename T> struct VectorTraits
    {
        typedef typename VectorTypeHelper<T>::Type VectorType;
        typedef typename DetermineEntryType<T>::Type EntryType;
        static constexpr size_t Size = sizeof(VectorType) / sizeof(EntryType);
        enum Constants {
            HasVectorDivision = !std::is_integral<T>::value
        };
        typedef Mask<T> MaskType;
        typedef typename DetermineGatherMask<MaskType>::Type GatherMaskType;
        typedef Vector<typename IndexTypeHelper<Size>::Type> IndexType;
        typedef Common::VectorMemoryUnion<VectorType, EntryType> StorageType;
    };

    template<typename T> struct VectorHelperSize;

    template<typename V = Vector<float> >
    class STRUCT_ALIGN1(16) VectorAlignedBaseT
    {
        public:
            FREE_STORE_OPERATORS_ALIGNED(16)
    } STRUCT_ALIGN2(16);

Vc_IMPL_NAMESPACE_END

#include "undomacros.h"

#endif // SSE_TYPES_H
