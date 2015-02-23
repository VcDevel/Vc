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
    typename T, std::size_t N, typename MT,
    bool = (std::is_same<T, short>::value || std::is_same<T, unsigned short>::value ||
            std::is_same<T, int>::value || std::is_same<T, unsigned int>::value ||
            std::is_same<T, float>::value || std::is_same<T, double>::value)>
struct ReplaceTypes
{
    typedef T type;
};

template <typename T, std::size_t N = 0, typename MT = void>
using simdize = typename ReplaceTypes<T, N, MT>::type;

// specialization for simdizable arithmetic types
template <typename T, std::size_t N, typename MT>
struct ReplaceTypes<T, N, MT, true>
    : public std::conditional<(N == 0 || Vector<T>::Size == N), Vector<T>,
                              simdarray<T, N>>
{
};

// specialization for bool -> Mask
template <std::size_t N, typename MT>
struct ReplaceTypes<bool, N, MT, false>
    : public std::conditional<(N == 0 || Mask<MT>::Size == N), Mask<MT>,
                              simd_mask_array<MT, N>>
{
};
template <std::size_t N>
struct ReplaceTypes<bool, N, void, false> : public ReplaceTypes<bool, N, float, false>
{
};

// Typelist to support multiple parameter packs in one class template
template <typename... Ts> struct Typelist;

// Try substituting one type after another - the first one that succeeds sets N (if it was 0)
template <std::size_t N, typename MT, typename Replaced, typename... Remaining>
struct SubstituteOneByOne;
template <std::size_t N, typename MT, typename... Replaced, typename T,
          typename... Remaining>
struct SubstituteOneByOne<N, MT, Typelist<Replaced...>, T, Remaining...>
{
private:
    // U::Size or 0
    template <typename U, std::size_t M = U::Size>
    static std::integral_constant<std::size_t, M> size_or_0(int);
    template <typename U> static std::integral_constant<std::size_t, 0> size_or_0(...);
    using V = simdize<T, N, MT>;
    static constexpr auto NewN = N != 0 ? N : decltype(size_or_0<V>(int()))::value;

    typedef typename std::conditional<
        (N != NewN && std::is_same<MT, void>::value),
        typename std::conditional<std::is_same<T, bool>::value, float, T>::type,
        MT>::type NewMT;

public:
    using type = typename SubstituteOneByOne<NewN, NewMT, Typelist<Replaced..., V>,
                                             Remaining...>::type;
};

// specialization for ending the recursion and setting the return type
template <std::size_t N_, typename MT, typename Replaced0, typename... Replaced>
struct SubstituteOneByOne<N_, MT, Typelist<Replaced0, Replaced...>>
{
    // Return type for returning the vector width and list of substituted types
    struct type
    {
        static constexpr auto N = N_;
        template <template <typename...> class C>
        using Substituted = C<Replaced0, Replaced...>;
        template <typename ValueT, template <typename, ValueT...> class C,
                  ValueT... Values>
        using Substituted1 = C<Replaced0, Values...>;
    };
};

template <typename Scalar, typename Base, std::size_t N> class SimdizeAdapter;

// specialization for class templates where all template arguments need to be substituted
template <template <typename...> class C, typename... Ts, std::size_t N, typename MT>
struct ReplaceTypes<C<Ts...>, N, MT, false>
{
    typedef typename SubstituteOneByOne<N, MT, Typelist<>, Ts...>::type tmp;
    typedef typename tmp::template Substituted<C> Substituted;
    static constexpr auto NN = tmp::N;
    typedef typename std::conditional<
        std::is_same<C<Ts...>, Substituted>::value, C<Ts...>,
        SimdizeAdapter<C<Ts...>, Substituted, NN>>::type type;
};

// specialization for class templates where all template arguments need to be substituted
#define Vc_DEFINE_NONTYPE_REPLACETYPES__(ValueType__)                                    \
    template <template <typename, ValueType__...> class C, typename Ts,                  \
              ValueType__ Value0, ValueType__... Values, std::size_t N, typename MT>     \
    struct ReplaceTypes<C<Ts, Value0, Values...>, N, MT, false>                          \
    {                                                                                    \
        typedef typename SubstituteOneByOne<N, MT, Typelist<>, Ts>::type tmp;            \
        typedef typename tmp::template Substituted1<ValueType__, C, Value0, Values...>   \
            Substituted;                                                                 \
        static constexpr auto NN = tmp::N;                                               \
        typedef typename std::conditional<                                               \
            std::is_same<C<Ts, Value0, Values...>, Substituted>::value,                  \
            C<Ts, Value0, Values...>,                                                    \
            SimdizeAdapter<C<Ts, Value0, Values...>, Substituted, NN>>::type type;       \
    }
Vc_DEFINE_NONTYPE_REPLACETYPES__(bool);
Vc_DEFINE_NONTYPE_REPLACETYPES__(wchar_t);
Vc_DEFINE_NONTYPE_REPLACETYPES__(char);
Vc_DEFINE_NONTYPE_REPLACETYPES__(  signed char);
Vc_DEFINE_NONTYPE_REPLACETYPES__(unsigned char);
Vc_DEFINE_NONTYPE_REPLACETYPES__(  signed short);
Vc_DEFINE_NONTYPE_REPLACETYPES__(unsigned short);
Vc_DEFINE_NONTYPE_REPLACETYPES__(  signed int);
Vc_DEFINE_NONTYPE_REPLACETYPES__(unsigned int);
Vc_DEFINE_NONTYPE_REPLACETYPES__(  signed long);
Vc_DEFINE_NONTYPE_REPLACETYPES__(unsigned long);
Vc_DEFINE_NONTYPE_REPLACETYPES__(  signed long long);
Vc_DEFINE_NONTYPE_REPLACETYPES__(unsigned long long);
#undef Vc_DEFINE_NONTYPE_REPLACETYPES__

// SimdizeAdapter wrapper class
template <typename Scalar, typename Base, std::size_t N>
class SimdizeAdapter : public Base
{
public:
    static constexpr std::size_t Size = N;
    static constexpr std::size_t size() { return N; }

    using base_type = Base;
    using scalar_type = Scalar;
};
template <typename Scalar, typename Base, std::size_t N>
constexpr std::size_t SimdizeAdapter<Scalar, Base, N>::Size;

}  // namespace Vc

namespace std
{
// tuple_size
template <typename Scalar, typename Base, std::size_t N>
class tuple_size<Vc::SimdizeAdapter<Scalar, Base, N>> : public tuple_size<Base>
{
};
// tuple_element
template <std::size_t I, typename Scalar, typename Base, std::size_t N>
class tuple_element<I, Vc::SimdizeAdapter<Scalar, Base, N>>
    : public tuple_element<I, Base>
{
};
// std::get does not need additional work because Vc::SimdizeAdapter derives from
// C<Ts...> and therefore if get<N>(C<Ts...>) works it works for SimdizeAdapter as well.

// std::allocator<Vc::SimdizeAdapter<T, N>>
template <typename S, typename T, std::size_t N>
class allocator<Vc::SimdizeAdapter<S, T, N>>
    : public Vc::Allocator<Vc::SimdizeAdapter<S, T, N>>
{
public:
    template <typename U> struct rebind
    {
        typedef std::allocator<U> other;
    };
};
}  // namespace std

#include "undomacros.h"

#endif  // VC_COMMON_SIMDIZE_H_

// vim: foldmethod=marker
