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

#ifndef VC_TESTS_TYPELIST_H_
#define VC_TESTS_TYPELIST_H_

#include <type_traits>

template <typename... Ts> struct Typelist;

template <template <typename...> class T, typename... Fixed> struct Template {
    template <typename... Us> using type = T<Us..., Fixed...>;
};

// concat {{{1
template <typename... More> struct concat_impl;
/**
 * Concatenate two type arguments into a single Typelist.
 */
template <typename... Ts> using concat = typename concat_impl<Ts...>::type;

// concat implementation:
template <typename A> struct concat_impl<A> {
    using type = A;
};
template <typename A, typename B> struct concat_impl<A, B> {
    using type = Typelist<A, B>;
};
template <typename... As, typename B> struct concat_impl<Typelist<As...>, B> {
    using type = Typelist<As..., B>;
};
template <typename A, typename... Bs> struct concat_impl<A, Typelist<Bs...>> {
    using type = Typelist<A, Bs...>;
};
template <typename... As, typename... Bs>
struct concat_impl<Typelist<As...>, Typelist<Bs...>> {
    using type = Typelist<As..., Bs...>;
};
template <typename A, typename B, typename C, typename... More>
struct concat_impl<A, B, C, More...> {
    using type = typename concat_impl<typename concat_impl<A, B>::type,
                                      typename concat_impl<C, More...>::type>::type;
};

// split {{{1
template <std::size_t N, typename T> struct split_impl;
template <typename... Ts> struct split_impl<0, Typelist<Ts...>> {
    using first = Typelist<>;
    using second = Typelist<Ts...>;
};
template <typename T, typename... Ts> struct split_impl<1, Typelist<T, Ts...>> {
    using first = Typelist<T>;
    using second = Typelist<Ts...>;
};
template <typename T0, typename T1, typename... Ts>
struct split_impl<2, Typelist<T0, T1, Ts...>> {
    using first = Typelist<T0, T1>;
    using second = Typelist<Ts...>;
};
template <typename T0, typename T1, typename T2, typename T3, typename... Ts>
struct split_impl<4, Typelist<T0, T1, T2, T3, Ts...>> {
    using first = Typelist<T0, T1, T2, T3>;
    using second = Typelist<Ts...>;
};
template <std::size_t N, typename... Ts> struct split_impl<N, Typelist<Ts...>> {
private:
    using A = split_impl<N / 2, Typelist<Ts...>>;
    using B = split_impl<N - N / 2, typename A::second>;

public:
    using first = concat<typename A::first, typename B::first>;
    using second = typename B::second;
};
template <typename T> using split = split_impl<T::size() / 2, T>;

// outer_product {{{1
template <typename A, typename B> struct outer_product_impl;
template <typename... Bs> struct outer_product_impl<Typelist<>, Typelist<Bs...>>
{
    using type = Typelist<>;
};
template <typename A0, typename... Bs>
struct outer_product_impl<Typelist<A0>, Typelist<Bs...>>
{
    using type = Typelist<concat<A0, Bs>...>;
};
template <typename A0, typename... As, typename... Bs>
struct outer_product_impl<Typelist<A0, As...>, Typelist<Bs...>>
{
    using type =
        concat<Typelist<concat<A0, Bs>...>,
               typename outer_product_impl<Typelist<As...>, Typelist<Bs...>>::type>;
};

template <typename A, typename B>
using outer_product = typename outer_product_impl<A, B>::type;

// expand_one {{{1
template <typename A, typename B> struct expand_one_impl;
template <typename A> struct expand_one_impl<A, Typelist<>> {
    using type = Typelist<>;
};
template <typename A, typename B0, typename... Bs>
struct expand_one_impl<A, Typelist<B0, Bs...>> {
    using type = concat<typename A::template type<B0>,
                        typename expand_one_impl<A, Typelist<Bs...>>::type>;
};

template <typename A, typename B>
using expand_one = typename expand_one_impl<A, B>::type;

// expand_list {{{1
template <typename A, typename B> struct expand_list_impl;
template <typename B> struct expand_list_impl<Typelist<>, B> {
    using type = Typelist<>;
};
template <typename A0, typename... As, typename B>
struct expand_list_impl<Typelist<A0, As...>, B> {
    using type = concat<expand_one<A0, B>, typename expand_list_impl<Typelist<As...>, B>::type>;
};
template <typename A, typename B>
using expand_list = typename expand_list_impl<A, B>::type;

// extract_type_impl {{{1
struct TypelistSentinel;
template <std::size_t N, bool N_less_4, bool N_larger_32, typename... Ts>
struct extract_type_impl
{
    using type = TypelistSentinel;
};
template <typename T0, typename... Ts> struct extract_type_impl<0, true, false, T0, Ts...>
{
    using type = T0;
};
template <typename T0, typename T1, typename... Ts>
struct extract_type_impl<1, true, false, T0, T1, Ts...>
{
    using type = T1;
};
template <typename T0, typename T1, typename T2, typename... Ts>
struct extract_type_impl<2, true, false, T0, T1, T2, Ts...>
{
    using type = T2;
};
template <typename T0, typename T1, typename T2, typename T3, typename... Ts>
struct extract_type_impl<3, true, false, T0, T1, T2, T3, Ts...>
{
    using type = T3;
};
template <std::size_t N,
          typename T00,
          typename T01,
          typename T02,
          typename T03,
          typename T04,
          typename T05,
          typename T06,
          typename T07,
          typename T08,
          typename T09,
          typename T10,
          typename T11,
          typename T12,
          typename T13,
          typename T14,
          typename T15,
          typename T16,
          typename T17,
          typename T18,
          typename T19,
          typename T20,
          typename T21,
          typename T22,
          typename T23,
          typename T24,
          typename T25,
          typename T26,
          typename T27,
          typename T28,
          typename T29,
          typename T30,
          typename T31,
          typename... Ts>
struct extract_type_impl<N,
                    false,
                    true,
                    T00,
                    T01,
                    T02,
                    T03,
                    T04,
                    T05,
                    T06,
                    T07,
                    T08,
                    T09,
                    T10,
                    T11,
                    T12,
                    T13,
                    T14,
                    T15,
                    T16,
                    T17,
                    T18,
                    T19,
                    T20,
                    T21,
                    T22,
                    T23,
                    T24,
                    T25,
                    T26,
                    T27,
                    T28,
                    T29,
                    T30,
                    T31,
                    Ts...>
{
    static constexpr std::size_t NN = N - 32;
    using type = typename extract_type_impl<NN, (NN < 4), (NN >= 32), Ts...>::type;
};

template <std::size_t N,
          typename T0,
          typename T1,
          typename T2,
          typename T3,
          typename... Ts>
struct extract_type_impl<N, false, false, T0, T1, T2, T3, Ts...>
{
    static constexpr std::size_t NN = N - 4;
    using type = typename extract_type_impl<NN, (NN < 4), (NN >= 32), Ts...>::type;
};

template <std::size_t N, typename... Ts>
using extract_type = typename extract_type_impl<N, (N < 4), (N >= 32), Ts...>::type;

template <typename... Ts> struct Typelist
{
    template <std::size_t N>
    using at = typename extract_type_impl<N, (N < 4), (N >= 32), Ts...>::type;

    static constexpr std::size_t size() { return sizeof...(Ts); }
};

// static_asserts {{{1
static_assert(std::is_same<outer_product<Typelist<int, float>, Typelist<short, double>>,
                           Typelist<Typelist<int, short>,
                                    Typelist<int, double>,
                                    Typelist<float, short>,
                                    Typelist<float, double>>>::value,
              "outer_product does not work as expected");
static_assert(
    std::is_same<
        outer_product<Typelist<long, char>,
                      outer_product<Typelist<int, float>, Typelist<short, double>>>,
        Typelist<Typelist<long, int, short>,
                 Typelist<long, int, double>,
                 Typelist<long, float, short>,
                 Typelist<long, float, double>,
                 Typelist<char, int, short>,
                 Typelist<char, int, double>,
                 Typelist<char, float, short>,
                 Typelist<char, float, double>>>::value,
    "outer_product does not work as expected");

namespace
{
template<typename T> struct TestType {};
template<typename T> struct TestType2 {};
static_assert(std::is_same<expand_one<Template<TestType>, Typelist<int, float>>,
                           Typelist<TestType<int>, TestType<float>>>::value,
              "expand_one is broken");
static_assert(std::is_same<expand_list<Typelist<Template<TestType>, Template<TestType2>>,
                                       Typelist<int, float>>,
                           Typelist<TestType<int>, TestType<float>, TestType2<int>,
                                    TestType2<float>>>::value,
              "expand_list is broken");
}  // namespace
// }}}1

#endif  // VC_TESTS_TYPELIST_H_

// vim: foldmethod=marker
