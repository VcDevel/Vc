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

#ifndef AVX_TYPES_H
#define AVX_TYPES_H

#include "intrinsics.h"
#include "../common/storage.h"
#include "macros.h"

namespace Vc
{
namespace AVX
{
    template<typename T> class Vector;

    template<unsigned int VectorSize, size_t RegisterWidth> class Mask;

    template<typename T> struct VectorHelper {};
    template<typename T> struct GatherHelper;
    template<typename T> struct ScatterHelper;

    template<typename T> struct IndexTypeHelper;
    template<> struct IndexTypeHelper<         char > { typedef unsigned char  Type; };
    template<> struct IndexTypeHelper<unsigned char > { typedef unsigned char  Type; };
    template<> struct IndexTypeHelper<         short> { typedef unsigned short Type; };
    template<> struct IndexTypeHelper<unsigned short> { typedef unsigned short Type; };
    template<> struct IndexTypeHelper<         int  > { typedef unsigned int   Type; };
    template<> struct IndexTypeHelper<unsigned int  > { typedef unsigned int   Type; };
    template<> struct IndexTypeHelper<         float> { typedef unsigned int   Type; };
    template<> struct IndexTypeHelper<        double> { typedef unsigned int   Type; }; // _M128I based int32 would be nice

    template<typename T> struct NegateTypeHelper { typedef T Type; };
    template<> struct NegateTypeHelper<unsigned char > { typedef char  Type; };
    template<> struct NegateTypeHelper<unsigned short> { typedef short Type; };
    template<> struct NegateTypeHelper<unsigned int  > { typedef int   Type; };

    template<typename T> struct VectorTypeHelper;
    template<> struct VectorTypeHelper<         char > { typedef __m128i Type; };
    template<> struct VectorTypeHelper<unsigned char > { typedef __m128i Type; };
    template<> struct VectorTypeHelper<         short> { typedef __m128i Type; };
    template<> struct VectorTypeHelper<unsigned short> { typedef __m128i Type; };
    template<> struct VectorTypeHelper<         int  > { typedef _M256I Type; };
    template<> struct VectorTypeHelper<unsigned int  > { typedef _M256I Type; };
    template<> struct VectorTypeHelper<         float> { typedef _M256  Type; };
    template<> struct VectorTypeHelper<        double> { typedef _M256D Type; };

    template<typename T> struct HasVectorDivisionHelper { enum { Value = 1 }; };
    //template<> struct HasVectorDivisionHelper<unsigned int> { enum { Value = 0 }; };

    template<typename T> struct VectorHelperSize;

    namespace VectorSpecialInitializerZero { enum ZEnum { Zero = 0 }; }
    namespace VectorSpecialInitializerOne { enum OEnum { One = 1 }; }
    namespace VectorSpecialInitializerIndexesFromZero { enum IEnum { IndexesFromZero }; }

    template<typename V = Vector<float> >
    class STRUCT_ALIGN1(sizeof(V)) VectorAlignedBaseT
    {
        public:
            FREE_STORE_OPERATORS_ALIGNED(sizeof(V))
    } STRUCT_ALIGN2(sizeof(V));

} // namespace AVX
} // namespace Vc

#endif // AVX_TYPES_H
