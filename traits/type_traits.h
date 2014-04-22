/*  This file is part of the Vc library. {{{
Copyright © 2013-2014 Matthias Kretz <kretz@kde.org>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the names of contributing organizations nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#ifndef VC_TRAITS_TYPE_TRAITS_H
#define VC_TRAITS_TYPE_TRAITS_H

#include <type_traits>
#include "decay.h"
#include "has_no_allocated_data.h"
#include "has_contiguous_storage.h"
#include "is_initializer_list.h"
#include "is_load_arguments.h"

namespace Vc_VERSIONED_NAMESPACE
{
// meta-programming helpers
struct enable_if_default_type {};
static constexpr enable_if_default_type nullarg;
template <bool Test, typename T = enable_if_default_type> using enable_if = typename std::enable_if<Test, T>::type;

namespace Traits
{
#include "has_subscript_operator.h"
#include "has_multiply_operator.h"
#include "has_addition_operator.h"

template<typename T> struct is_simd_mask_internal : public std::false_type {};
template<typename T> struct is_simd_vector_internal : public std::false_type {};
template<typename T> struct is_subscript_operation_internal : public std::false_type {};
template<typename T> struct is_simd_array_internal : public std::false_type {};
template<typename T> struct is_loadstoreflag_internal : public std::false_type {};

#include "is_gather_signature.h"

template <std::size_t, typename... Args> struct is_cast_arguments_internal : public std::false_type {};
template <typename Arg>
struct is_cast_arguments_internal<1, Arg> : public std::integral_constant<
                                                bool,
                                                is_simd_array_internal<Arg>::value ||
                                                    is_simd_vector_internal<Arg>::value>
{
};

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
          (is_floating_point_internal<T>::value || is_integral_internal<T>::value)>
{
};

template <typename T,
          bool = (is_simd_vector_internal<T>::value || is_simd_mask_internal<T>::value ||
                  is_simd_array_internal<T>::value)>
struct vector_size_internal;

template <typename T>
struct vector_size_internal<T, true> : public std::integral_constant<std::size_t, T::Size>
{
};
template <typename T>
struct vector_size_internal<T, false> : public std::integral_constant<std::size_t, 0>
{
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct is_simd_mask : public is_simd_mask_internal<decay<T>>
{
};

template <typename T> struct is_simd_vector : public is_simd_vector_internal<decay<T>>
{
};

template <typename T>
struct is_simd_array : public is_simd_array_internal<decay<T>>
{
};

template <typename T> struct is_subscript_operation : public is_subscript_operation_internal<decay<T>> {};
template <typename T> struct is_load_store_flag : public is_loadstoreflag_internal<decay<T>> {};
template <typename... Args> struct is_cast_arguments : public is_cast_arguments_internal<sizeof...(Args), decay<Args>...> {};

template <typename T> struct simd_vector_size : public vector_size_internal<decay<T>> {};

template <typename T> struct is_integral : public is_integral_internal<decay<T>> {};
template <typename T> struct is_floating_point : public is_floating_point_internal<decay<T>> {};
template <typename T> struct is_arithmetic : public is_arithmetic_internal<decay<T>> {};
template <typename T> struct is_signed : public is_signed_internal<decay<T>> {};
template <typename T> struct is_unsigned : public is_unsigned_internal<decay<T>> {};

}  // namespace Traits
}  // namespace Vc

#include "entry_type_of.h"

#endif // VC_TRAITS_TYPE_TRAITS_H
