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
struct enable_if_default_type {};
static constexpr enable_if_default_type nullarg;
template <bool Test, typename T = enable_if_default_type> using enable_if = typename std::enable_if<Test, T>::type;

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

template <std::size_t, typename... Args> struct is_gather_arguments_internal;
template <std::size_t N__, typename Arg0, typename Arg1, typename... MoreArguments>
struct is_gather_arguments_internal<
    N__,
    Arg0,
    Arg1,
    MoreArguments...> : public std::
                            integral_constant<bool,
                                              is_good_for_gatherscatter_internal<Arg0>::value &&
                                                  !is_loadstoreflag_internal<Arg1>::value &&(
                                                       std::is_pointer<Arg1>::value ||
                                                       is_simd_vector_internal<Arg1>::value)>
{
};
template<typename... Args> struct is_gather_arguments_internal<0, Args...> : public std::false_type {};
template<typename... Args> struct is_gather_arguments_internal<1, Args...> : public std::false_type {};

template <typename T, bool = is_simd_vector_internal<T>::value> struct is_integral_internal;
template <typename T, bool = is_simd_vector_internal<T>::value> struct is_floating_point_internal;
template <typename T, bool = is_simd_vector_internal<T>::value> struct is_signed_internal;
template <typename T, bool = is_simd_vector_internal<T>::value> struct is_unsigned_internal;

template <typename T> struct is_integral_internal      <T, false> : public std::is_integral      <T> {};
template <typename T> struct is_floating_point_internal<T, false> : public std::is_floating_point<T> {};
template <typename T> struct is_signed_internal        <T, false> : public std::is_signed        <T> {};
template <typename T> struct is_unsigned_internal      <T, false> : public std::is_unsigned      <T> {};

template <typename V> struct is_integral_internal      <V, true> : public std::is_integral      <typename V::EntryType> {};
template <typename V> struct is_floating_point_internal<V, true> : public std::is_floating_point<typename V::EntryType> {};
template <typename V> struct is_signed_internal        <V, true> : public std::is_signed        <typename V::EntryType> {};
template <typename V> struct is_unsigned_internal      <V, true> : public std::is_unsigned      <typename V::EntryType> {};

template <typename T>
struct is_arithmetic_internal
    : public std::integral_constant<
          bool,
          is_floating_point_internal<T>::value || is_integral_internal<T>::value>
    {};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
template <typename... Args> struct IsGatherArguments : public is_gather_arguments_internal<sizeof...(Args), typename std::decay<Args>::type...> {};

template <typename T> struct is_integral : public is_integral_internal<typename std::decay<T>::type> {};
template <typename T> struct is_floating_point : public is_floating_point_internal<typename std::decay<T>::type> {};
template <typename T> struct is_arithmetic : public is_arithmetic_internal<typename std::decay<T>::type> {};
template <typename T> struct is_signed : public is_signed_internal<typename std::decay<T>::type> {};
template <typename T> struct is_unsigned : public is_unsigned_internal<typename std::decay<T>::type> {};

}
}

#include "undomacros.h"

#endif // VC_COMMON_TYPE_TRAITS_H
