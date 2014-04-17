/*  This file is part of the Vc library.

    Copyright (C) 2009-2012 Matthias Kretz <kretz@kde.org>

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

#include "../common/types.h"
#include "macros.h"

#ifdef VC_DEFAULT_IMPL_AVX2
#define VC_DOUBLE_V_SIZE 4
#define VC_FLOAT_V_SIZE 8
#define VC_INT_V_SIZE 8
#define VC_UINT_V_SIZE 8
#define VC_SHORT_V_SIZE 16
#define VC_USHORT_V_SIZE 16
#elif defined VC_DEFAULT_IMPL_AVX
#define VC_DOUBLE_V_SIZE 4
#define VC_FLOAT_V_SIZE 8
#define VC_INT_V_SIZE 8
#define VC_UINT_V_SIZE 8
#define VC_SHORT_V_SIZE 8
#define VC_USHORT_V_SIZE 8
#endif

namespace Vc_VERSIONED_NAMESPACE
{
namespace Vc_AVX_NAMESPACE
{
    template<typename T> class Vector;

    template<typename T> class Mask;

#ifdef VC_MSVC
    // MSVC's __declspec(align(#)) only works with numbers, no enums or sizeof allowed ;(
    template<size_t size> class _VectorAlignedBaseHack;
    template<> class STRUCT_ALIGN1( 8) _VectorAlignedBaseHack< 8> {} STRUCT_ALIGN2( 8);
    template<> class STRUCT_ALIGN1(16) _VectorAlignedBaseHack<16> {} STRUCT_ALIGN2(16);
    template<> class STRUCT_ALIGN1(32) _VectorAlignedBaseHack<32> {} STRUCT_ALIGN2(32);
    template<> class STRUCT_ALIGN1(64) _VectorAlignedBaseHack<64> {} STRUCT_ALIGN2(64);
    template<typename V = Vector<float> >
    class VectorAlignedBaseT : public _VectorAlignedBaseHack<alignof(V)>
    {
        public:
            FREE_STORE_OPERATORS_ALIGNED(alignof(V))
    };
#else
    template<typename V = Vector<float> >
    class STRUCT_ALIGN1(alignof(V)) VectorAlignedBaseT
    {
        public:
            FREE_STORE_OPERATORS_ALIGNED(alignof(V))
    } STRUCT_ALIGN2(alignof(V));
#endif
}

namespace Traits
{
template<typename T> struct is_simd_mask_internal<Vc_AVX_NAMESPACE::Mask<T>> : public std::true_type {};
template<typename T> struct is_simd_vector_internal<Vc_AVX_NAMESPACE::Vector<T>> : public std::true_type {};
}
}

#include "undomacros.h"

#endif // AVX_TYPES_H
