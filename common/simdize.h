/*  This file is part of the Vc library. {{{
Copyright © 2014 Matthias Kretz <kretz@kde.org>
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

#ifndef VC_COMMON_SIMDIZE_H_
#define VC_COMMON_SIMDIZE_H_

#include <tuple>
#include <array>

#include "macros.h"

namespace Vc_VERSIONED_NAMESPACE
{
template <
    typename T, std::size_t N,
    bool = (std::is_same<T, short>::value || std::is_same<T, unsigned short>::value ||
            std::is_same<T, int>::value || std::is_same<T, unsigned int>::value ||
            std::is_same<T, float>::value || std::is_same<T, double>::value)>
struct ReplaceTypes;

template <typename T, std::size_t N = 0>
using simdize = typename ReplaceTypes<T, N>::type;

// specialization for non-simdizable types
template <typename T, std::size_t N> struct ReplaceTypes<T, N, false>
{
    typedef T type;
};

// specialization for simdizable arithmetic types
template <typename T, std::size_t N>
struct ReplaceTypes<T, N, true>
    : public std::conditional<(N == 0 || Vector<T>::Size == N), Vector<T>,
                              simdarray<T, N>>
{
};

// Adapter wrapper class
template <typename Base, std::size_t N> class Adapter : public Base
{
public:
    static constexpr std::size_t Size = N;
    static constexpr std::size_t size() { return N; }

    using base_type = Base;
};
template <typename Base, std::size_t N> constexpr std::size_t Adapter<Base, N>::Size;

// Typelist to support multiple parameter packs in one class template
template <typename... Ts> struct Typelist;

// Try substituting one type after another - the first one that succeeds sets N (if it was 0)
template <std::size_t N, typename Replaced, typename... Remaining>
struct SubstituteOneByOne;
template <std::size_t N, typename... Replaced, typename T, typename... Remaining>
struct SubstituteOneByOne<N, Typelist<Replaced...>, T, Remaining...>
{
private:
    // U::Size or 0
    template <typename U, std::size_t M = U::Size>
    static std::integral_constant<std::size_t, M> size_or_0(int);
    template <typename U> static std::integral_constant<std::size_t, 0> size_or_0(...);
    using V = simdize<T, N>;
    static constexpr auto NN = N != 0 ? N : decltype(size_or_0<V>(int()))::value;

public:
    using type =
        typename SubstituteOneByOne<NN, Typelist<Replaced..., V>, Remaining...>::type;
};

// specialization for ending the recursion and setting the return type
template <std::size_t N_, typename... Replaced>
struct SubstituteOneByOne<N_, Typelist<Replaced...>>
{
    // Return type for returning the vector width and list of substituted types
    struct type
    {
        static constexpr auto N = N_;
        template <template <typename...> class C> using Substituted = C<Replaced...>;
    };
};

// specialization for class templates where all template arguments need to be substituted
template <template <typename...> class C, typename... Ts, std::size_t N>
struct ReplaceTypes<C<Ts...>, N, false>
{
    typedef typename SubstituteOneByOne<N, Typelist<>, Ts...>::type tmp;
    typedef typename tmp::template Substituted<C> Substituted;
    static constexpr auto NN = tmp::N;
    typedef typename std::conditional<std::is_same<C<Ts...>, Substituted>::value,
                                      C<Ts...>, Adapter<Substituted, NN>>::type type;
};

}  // namespace Vc

#include "undomacros.h"

#endif  // VC_COMMON_SIMDIZE_H_

// vim: foldmethod=marker
