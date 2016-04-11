/*  This file is part of the Vc library. {{{
Copyright © 2014-2015 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_COMMON_INDEXSEQUENCE_H_
#define VC_COMMON_INDEXSEQUENCE_H_

#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
/** \internal
 * Helper class for a sequence of size_t values from 0 to N. This type will be included in
 * C++14.
 */
template <std::size_t... I> struct index_sequence
{
    static constexpr std::size_t size() noexcept { return sizeof...(I); }
};

/** \internal
 * This struct builds an index_sequence type from a given upper bound \p N.
 * It does so recursively via appending N - 1 to make_index_sequence_impl<N - 1>.
 */
template <std::size_t N, typename Prev = void> struct make_index_sequence_impl;
/// \internal constructs an empty index_sequence
template <> struct make_index_sequence_impl<0, void>
{
    using type = index_sequence<>;
};
/// \internal appends `N-1` to make_index_sequence<N-1>
template <std::size_t N> struct make_index_sequence_impl<N, void>
{
    using type = typename make_index_sequence_impl<
        N, typename make_index_sequence_impl<N - 1>::type>::type;
};
/// \internal constructs the index_sequence `Ns..., N-1`
template <std::size_t N, std::size_t... Ns>
struct make_index_sequence_impl<N, index_sequence<Ns...>>
{
    using type = index_sequence<Ns..., N - 1>;
};

/** \internal
 * Creates an index_sequence type for the upper bound \p N.
 */
template <std::size_t N>
using make_index_sequence = typename make_index_sequence_impl<N>::type;
}

#endif  // VC_COMMON_INDEXSEQUENCE_H_

// vim: foldmethod=marker
