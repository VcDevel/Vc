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

#include "../traits/type_traits.h"
#include "macros.h"

#ifdef VC_DEFAULT_IMPL_AVX2
#define VC_DOUBLE_V_SIZE 4
#define VC_FLOAT_V_SIZE 8
#define VC_INT_V_SIZE 8
#define VC_UINT_V_SIZE 8
// TODO: 16
#define VC_SHORT_V_SIZE 8
#define VC_USHORT_V_SIZE 8
#elif defined VC_DEFAULT_IMPL_AVX
#define VC_DOUBLE_V_SIZE 4
#define VC_FLOAT_V_SIZE 8
// TODO: 4
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

    template <typename V = Vector<float>> class alignas(alignof(V)) VectorAlignedBaseT;
}

namespace Traits
{
template<typename T> struct is_simd_mask_internal<Vc_AVX_NAMESPACE::Mask<T>> : public std::true_type {};
template<typename T> struct is_simd_vector_internal<Vc_AVX_NAMESPACE::Vector<T>> : public std::true_type {};
}
}

#include "undomacros.h"

#endif // AVX_TYPES_H
