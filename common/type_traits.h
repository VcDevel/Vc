/*  This file is part of the Vc library. {{{

    Copyright (C) 2013 Matthias Kretz <kretz@kde.org>

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

}}}*/

#ifndef VC_COMMON_TYPE_TRAITS_H
#define VC_COMMON_TYPE_TRAITS_H

#include <type_traits>
#include "macros.h"

Vc_NAMESPACE_BEGIN(Common)

template<typename T> struct is_simd_mask_internal : public std::false_type {};
template<typename T> struct is_simd_vector_internal : public std::false_type {};

template<typename T>
struct is_simd_mask
    : public Common::is_simd_mask_internal<
      typename std::remove_cv<typename std::remove_reference<T>::type>::type>
{};

template<typename T>
struct is_simd_vector
    : public Common::is_simd_vector_internal<
      typename std::remove_cv<typename std::remove_reference<T>::type>::type>
{};

Vc_NAMESPACE_END

#include "undomacros.h"

#endif // VC_COMMON_TYPE_TRAITS_H
