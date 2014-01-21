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

namespace Vc_VERSIONED_NAMESPACE
{
// meta-programming helpers
template <bool Test, typename T = void *> using enable_if = typename std::enable_if<Test, T>::type;

namespace Common
{

template<typename T> struct is_simd_mask_internal : public std::false_type {};
template<typename T> struct is_simd_vector_internal : public std::false_type {};
template<typename T> struct is_subscript_operation_internal : public std::false_type {};
template<typename T> struct is_simd_array_internal : public std::false_type {};
template<typename T> struct is_loadstoreflag_internal : public std::false_type {};
template <typename T>
struct is_good_for_gatherscatter_internal
    : public std::integral_constant<
          bool,
          std::is_array<T>::value || std::is_pointer<T>::value && !std::is_function<T>::value>
{
};

template <typename T>
struct is_simd_mask : public Common::is_simd_mask_internal<typename std::decay<T>::type>
{};

template <typename T>
struct is_simd_vector : public Common::is_simd_vector_internal<typename std::decay<T>::type>
{};

template <typename T> struct IsSubscriptOperation : public is_subscript_operation_internal<typename std::decay<T>::type> {};
template <typename T> struct IsSimdArray : public is_simd_array_internal<typename std::decay<T>::type> {};
template <typename T> struct IsLoadStoreFlag : public is_loadstoreflag_internal<typename std::decay<T>::type> {};
template <typename T> struct IsGoodForGatherScatter : public is_good_for_gatherscatter_internal<typename std::decay<T>::type> {};

}
}

#include "undomacros.h"

#endif // VC_COMMON_TYPE_TRAITS_H
