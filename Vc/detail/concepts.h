/*  This file is part of the Vc library. {{{
Copyright © 2017 Matthias Kretz <kretz@kde.org>

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
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#ifndef VC_DETAIL_CONCEPTS_H_
#define VC_DETAIL_CONCEPTS_H_

#include "type_traits.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
/**\internal
 * Deduces to a type T which is signed.
 */
template <class T, class = std::enable_if_t<std::is_signed<T>::value>>
using SignedArithmetic = T;

/**\internal
 * Deduces to a type T which is convertible to DstT with `sizeof(T) == ExpectedSizeof`.
 */
template <class T, size_t ExpectedSizeof, class DstT,
          class = enable_if_t<all<has_expected_sizeof<T, ExpectedSizeof>,
                                  is_convertible<T, DstT>>::value>>
using convertible_memory = T;

template <class From, class To,
          class = enable_if_t<
              negation<detail::is_narrowing_conversion<std::decay_t<From>, To>>::value>>
using value_preserving = From;

template <class From, class To, class DecayedFrom = std::decay_t<From>,
          class = enable_if_t<all<
              is_convertible<From, To>,
              any<is_same<DecayedFrom, To>, is_same<DecayedFrom, int>,
                  all<is_same<DecayedFrom, uint>, is_unsigned<To>>,
                  negation<detail::is_narrowing_conversion<DecayedFrom, To>>>>::value>>
using value_preserving_or_int = From;

/**\internal
 * Tag type for overload disambiguation. Typically default initialized via `tag<1> = {}`.
 */
template <int...> struct tag {
};

/**\internal
 * Deduces to an arithmetic type, but not bool.
 */
template <class T, class = enable_if_t<is_vectorizable_v<T>>>
using vectorizable = T;

/**\internal
 * Deduces to a type allowed for load/store with the given value type.
 */
template <
    class Ptr, class ValueType,
    class = enable_if_t<detail::is_possible_loadstore_conversion<Ptr, ValueType>::value>>
using loadstore_ptr_type = Ptr;

}  // namespace detail
Vc_VERSIONED_NAMESPACE_END

#endif  // VC_DETAIL_CONCEPTS_H_

// vim: foldmethod=marker
