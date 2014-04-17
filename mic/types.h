/*  This file is part of the Vc library.

    Copyright (C) 2010-2011 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_MIC_TYPES_H
#define VC_MIC_TYPES_H

#include <cstdlib>
#include "macros.h"

#ifdef VC_DEFAULT_IMPL_MIC
#define VC_DOUBLE_V_SIZE 8
#define VC_FLOAT_V_SIZE 16
#define VC_INT_V_SIZE 16
#define VC_UINT_V_SIZE 16
#define VC_SHORT_V_SIZE 16
#define VC_USHORT_V_SIZE 16
#endif

namespace Vc_VERSIONED_NAMESPACE
{
namespace MIC
{
    template<typename T> class Vector;
    template<typename T> class Mask;
    constexpr std::size_t VectorAlignment = 64;

    template <typename V = Vector<float>> class alignas(alignof(V)) VectorAlignedBaseT;
}

namespace Traits
{
template <typename T> struct is_simd_mask_internal<MIC::Mask<T>> : public std::true_type {};
template <typename T> struct is_simd_vector_internal<MIC::Vector<T>> : public std::true_type {};
}
}

#include "undomacros.h"

#endif // VC_MIC_TYPES_H
