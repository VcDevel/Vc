/*  This file is part of the Vc library. {{{
Copyright © 2016-2017 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_DATAPAR_DETAIL_H_
#define VC_DATAPAR_DETAIL_H_

#include <limits>
#include <functional>
#include "macros.h"
#include "flags.h"
#include "type_traits.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
// size_constant {{{1
template <size_t X> using size_constant = std::integral_constant<size_t, X>;

// integer type aliases{{{1
using uchar = unsigned char;
using schar = signed char;
using ushort = unsigned short;
using uint = unsigned int;
using ulong = unsigned long;
using llong = long long;
using ullong = unsigned long long;

// equal_int_type{{{1
/**
 * \internal
 * Type trait to find the equivalent integer type given a(n) (un)signed long type.
 */
template <class T, size_t = sizeof(T)> struct equal_int_type;
template <> struct equal_int_type< long, 4> { using type =    int; };
template <> struct equal_int_type< long, 8> { using type =  llong; };
template <> struct equal_int_type<ulong, 4> { using type =   uint; };
template <> struct equal_int_type<ulong, 8> { using type = ullong; };
template <class T> using equal_int_type_t = typename equal_int_type<T>::type;

// promote_preserving_unsigned{{{1
// work around crazy semantics of unsigned integers of lower rank than int:
// Before applying an operator the operands are promoted to int. In which case over- or
// underflow is UB, even though the operand types were unsigned.
template <class T> static Vc_INTRINSIC const T &promote_preserving_unsigned(const T &x)
{
    return x;
}
static Vc_INTRINSIC unsigned int promote_preserving_unsigned(const unsigned char &x)
{
    return x;
}
static Vc_INTRINSIC unsigned int promote_preserving_unsigned(const unsigned short &x)
{
    return x;
}

// exact_bool{{{1
class exact_bool {
    const bool d;

public:
    constexpr exact_bool(bool b) : d(b) {}
    exact_bool(int) = delete;
    constexpr operator bool() const { return d; }
};

// unused{{{1
template <class T> static constexpr void unused(T && ) {}

// execute_on_index_sequence{{{1
template <typename F, size_t... I>
Vc_INTRINSIC void execute_on_index_sequence(F && f, std::index_sequence<I...>)
{
    auto &&x = {(f(size_constant<I>()), 0)...};
    unused(x);
}

template <typename F>
Vc_INTRINSIC void execute_on_index_sequence(F &&, std::index_sequence<>)
{
}

template <typename R, typename F, size_t... I>
Vc_INTRINSIC R execute_on_index_sequence_with_return(F && f, std::index_sequence<I...>)
{
    return R{f(size_constant<I>())...};
}

// execute_n_times{{{1
template <size_t N, typename F> Vc_INTRINSIC void execute_n_times(F && f)
{
    execute_on_index_sequence(std::forward<F>(f), std::make_index_sequence<N>{});
}

// generate_from_n_evaluations{{{1
template <size_t N, typename R, typename F>
Vc_INTRINSIC R generate_from_n_evaluations(F && f)
{
    return execute_on_index_sequence_with_return<R>(std::forward<F>(f),
                                                    std::make_index_sequence<N>{});
}

// datapar_tuple {{{1
// why not std::tuple?
// 1. std::tuple gives no guarantee about the storage order, but I require storage
//    equivalent to std::array<T, N>
// 2. much less code to instantiate: I require a very small subset of std::tuple
//    functionality
// 3. direct access to the element type (first template argument)
// 4. enforces equal element type, only different Abi types are allowed

template <class T, class... Abis> struct datapar_tuple;
// datapar_tuple specializations {{{2
template <class T> struct datapar_tuple<T> {
    static constexpr size_t tuple_size = 0;
    //static constexpr size_t element_count = 0;
};
template <class T, class Abi0> struct datapar_tuple<T, Abi0> {
    using first_type = Vc::datapar<T, Abi0>;
    static constexpr size_t tuple_size = 1;
    //static constexpr size_t element_count = first_type::size();
    first_type first;

    template <size_t Offset = 0, class F>
    static Vc_INTRINSIC datapar_tuple generate(F &&gen)
    {
        return {gen(first_type(), size_constant<Offset>())};
    }

    template <class F, class... More>
    friend Vc_INTRINSIC datapar_tuple apply(F &&fun, const datapar_tuple &x,
                                            const More &... more)
    {
        return {fun(x.first, more.first...)};
    }
};
template <class T, class Abi0, class... Abis> struct datapar_tuple<T, Abi0, Abis...> {
    using first_type = Vc::datapar<T, Abi0>;
    using second_type = datapar_tuple<T, Abis...>;
    static constexpr size_t tuple_size = sizeof...(Abis) + 1;
    //static constexpr size_t element_count = first_type::size + second_type::element_count;
    first_type first;
    second_type second;

    template <size_t Offset = 0, class F>
    static Vc_INTRINSIC datapar_tuple generate(F &&gen)
    {
        return {gen(first_type(), size_constant<Offset>()),
                second_type::template generate<Offset + first_type::size()>(
                    std::forward<F>(gen))};
    }

    template <class F, class... More>
    friend Vc_INTRINSIC datapar_tuple apply(F &&fun, const datapar_tuple &x,
                                            const More &... more)
    {
        return {fun(x.first, more.first...),
                apply(std::forward<F>(fun), x.second, more.second...)};
    }
};

// make_tuple {{{2
template <class T, class A0> datapar_tuple<T, A0> make_tuple(const Vc::datapar<T, A0> &x0)
{
    return {x0};
}
template <class T, class A0, class... As>
datapar_tuple<T, A0, As...> make_tuple(const Vc::datapar<T, A0> &x0,
                                       const Vc::datapar<T, As> &... xs)
{
    return {x0, make_tuple(xs...)};
}

// get<N> {{{2
namespace datapar_tuple_impl
{
template <class T, class... Abis>
auto get_impl(const datapar_tuple<T, Abis...> &t, size_constant<0>)
{
    return t.first;
}
template <size_t N, class T, class... Abis>
auto get_impl(const datapar_tuple<T, Abis...> &t, size_constant<N>)
{
    return get_impl(t.second, size_constant<N - 1>());
}
}  // namespace datapar_tuple_impl
template <size_t N, class T, class... Abis> auto get(const datapar_tuple<T, Abis...> &t)
{
    return datapar_tuple_impl::get_impl(t, size_constant<N>());
}

// tuple_element {{{2
template <size_t I, class T> struct tuple_element;
template <class T, class A0, class... As>
struct tuple_element<0, datapar_tuple<T, A0, As...>> {
    using type = Vc::datapar<T, A0>;
};
template <size_t I, class T, class A0, class... As>
struct tuple_element<I, datapar_tuple<T, A0, As...>> {
    using type = typename tuple_element<I - 1, datapar_tuple<T, As...>>::type;
};
template <size_t I, class T> using tuple_element_t = typename tuple_element<I, T>::type;

// number_of_preceding_elements {{{2
template <size_t I, class T> struct number_of_preceding_elements;
template <class T, class A0, class... As>
struct number_of_preceding_elements<0, datapar_tuple<T, A0, As...>>
    : public size_constant<0> {
};
template <size_t I, class T, class A0, class... As>
struct number_of_preceding_elements<I, datapar_tuple<T, A0, As...>>
    : public std::integral_constant<
          size_t,
          datapar<T, A0>::size() +
              number_of_preceding_elements<I - 1, datapar_tuple<T, As...>>::value> {
};

// for_each(const datapar_tuple &, Fun) {{{2
template <size_t Offset = 0, class T, class A0, class F>
Vc_INTRINSIC void for_each(const datapar_tuple<T, A0> &t_, F &&fun_)
{
    std::forward<F>(fun_)(t_.first, size_constant<Offset>());
}
template <size_t Offset = 0, class T, class A0, class A1, class... As, class F>
Vc_INTRINSIC void for_each(const datapar_tuple<T, A0, A1, As...> &t_, F &&fun_)
{
    fun_(t_.first, size_constant<Offset>());
    for_each<Offset + t_.first.size()>(t_.second, std::forward<F>(fun_));
}

// for_each(datapar_tuple &, Fun) {{{2
template <size_t Offset = 0, class T, class A0, class F>
Vc_INTRINSIC void for_each(datapar_tuple<T, A0> &t_, F &&fun_)
{
    std::forward<F>(fun_)(t_.first, size_constant<Offset>());
}
template <size_t Offset = 0, class T, class A0, class A1, class... As, class F>
Vc_INTRINSIC void for_each(datapar_tuple<T, A0, A1, As...> &t_, F &&fun_)
{
    fun_(t_.first, size_constant<Offset>());
    for_each<Offset + t_.first.size()>(t_.second, std::forward<F>(fun_));
}

// for_each(datapar_tuple &, const datapar_tuple &, Fun) {{{2
template <size_t Offset = 0, class T, class A0, class F>
Vc_INTRINSIC void for_each(datapar_tuple<T, A0> &a_, const datapar_tuple<T, A0> &b_,
                           F &&fun_)
{
    std::forward<F>(fun_)(a_.first, b_.first, size_constant<Offset>());
}
template <size_t Offset = 0, class T, class A0, class A1, class... As, class F>
Vc_INTRINSIC void for_each(datapar_tuple<T, A0, A1, As...> & a_,
                           const datapar_tuple<T, A0, A1, As...> &b_, F &&fun_)
{
    fun_(a_.first, b_.first, size_constant<Offset>());
    for_each<Offset + a_.first.size()>(a_.second, b_.second, std::forward<F>(fun_));
}

// for_each(const datapar_tuple &, const datapar_tuple &, Fun) {{{2
template <size_t Offset = 0, class T, class A0, class F>
Vc_INTRINSIC void for_each(const datapar_tuple<T, A0> &a_, const datapar_tuple<T, A0> &b_,
                           F &&fun_)
{
    std::forward<F>(fun_)(a_.first, b_.first, size_constant<Offset>());
}
template <size_t Offset = 0, class T, class A0, class A1, class... As, class F>
Vc_INTRINSIC void for_each(const datapar_tuple<T, A0, A1, As...> &a_,
                           const datapar_tuple<T, A0, A1, As...> &b_, F &&fun_)
{
    fun_(a_.first, b_.first, size_constant<Offset>());
    for_each<Offset + a_.first.size()>(a_.second, b_.second, std::forward<F>(fun_));
}

// may_alias{{{1
template <typename T> struct may_alias_impl {
    typedef T type Vc_MAY_ALIAS;
};
/**\internal
 * Helper may_alias<T> that turns T into the type to be used for an aliasing pointer. This
 * adds the may_alias attribute to T (with compilers that support it). But for MaskBool this
 * attribute is already part of the type and applying it a second times leads to warnings/errors,
 * therefore MaskBool is simply forwarded as is.
 */
#ifdef Vc_ICC
template <typename T> using may_alias [[gnu::may_alias]] = T;
#else
template <typename T> using may_alias = typename may_alias_impl<T>::type;
#endif

    // traits forward declaration{{{1
    /**
     * \internal
     * Defines the implementation of a given <T, Abi>.
     *
     * Implementations must ensure that only valid <T, Abi> instantiations are possible.
     * Static assertions in the type definition do not suffice. It is important that
     * SFINAE works.
     */
template <class T, class Abi> struct traits {
    static constexpr size_t size() noexcept { return 0; }
    static constexpr size_t datapar_member_alignment = 1;
    struct datapar_impl_type;
    struct datapar_member_type {};
    struct datapar_cast_type;
    struct datapar_base {
        datapar_base() = delete;
        datapar_base(const datapar_base &) = delete;
        datapar_base &operator=(const datapar_base &) = delete;
        ~datapar_base() = delete;
    };
    static constexpr size_t mask_member_alignment = 1;
    struct mask_impl_type;
    struct mask_member_type {};
    struct mask_cast_type;
    struct mask_base {
        mask_base() = delete;
        mask_base(const mask_base &) = delete;
        mask_base &operator=(const mask_base &) = delete;
        ~mask_base() = delete;
    };
};

// get_impl(_t){{{1
template <class T> struct get_impl;
template <class T> using get_impl_t = typename get_impl<T>::type;

// next_power_of_2{{{1
/**
 * \internal
 * Returns the next power of 2 larger than or equal to \p x.
 */
static constexpr std::size_t next_power_of_2(std::size_t x)
{
    return (x & (x - 1)) == 0 ? x : next_power_of_2((x | (x >> 1)) + 1);
}

// default_neutral_element{{{1
template <class T, class BinaryOperation> struct default_neutral_element;
template <class T, class X> struct default_neutral_element<T, std::plus<X>> {
    static constexpr T value = 0;
};
template <class T, class X> struct default_neutral_element<T, std::multiplies<X>> {
    static constexpr T value = 1;
};
template <class T, class X> struct default_neutral_element<T, std::bit_and<X>> {
    static constexpr T value = ~T(0);
};
template <class T, class X> struct default_neutral_element<T, std::bit_or<X>> {
    static constexpr T value = 0;
};
template <class T, class X> struct default_neutral_element<T, std::bit_xor<X>> {
    static constexpr T value = 0;
};

// private_init, bitset_init{{{1
/**
 * \internal
 * Tag used for private init constructor of datapar and mask
 */
static constexpr struct private_init_t {} private_init = {};
static constexpr struct bitset_init_t {} bitset_init = {};

// size_tag{{{1
template <size_t N> static constexpr size_constant<N> size_tag = {};

// identity/id{{{1
template <class T> struct identity {
    using type = T;
};
template <class T> using id = typename identity<T>::type;

// bool_constant{{{1
template <bool Value> using bool_constant = std::integral_constant<bool, Value>;

// is_narrowing_conversion<From, To>{{{1
template <class From, class To, bool = std::is_arithmetic<From>::value,
          bool = std::is_arithmetic<To>::value>
struct is_narrowing_conversion;

#ifdef Vc_MSVC
// ignore "warning C4018: '<': signed/unsigned mismatch" in the following trait. The implicit
// conversions will do the right thing here.
#pragma warning(push)
#pragma warning(disable : 4018)
#endif
template <class From, class To>
struct is_narrowing_conversion<From, To, true, true>
    : public bool_constant<(
          std::numeric_limits<From>::digits > std::numeric_limits<To>::digits ||
          std::numeric_limits<From>::max() > std::numeric_limits<To>::max() ||
          std::numeric_limits<From>::lowest() < std::numeric_limits<To>::lowest() ||
          (std::is_signed<From>::value && std::is_unsigned<To>::value))> {
};
#ifdef Vc_MSVC
#pragma warning(pop)
#endif

template <class T> struct is_narrowing_conversion<bool, T, true, true> : public std::true_type {};
template <> struct is_narrowing_conversion<bool, bool, true, true> : public std::false_type {};
template <class T> struct is_narrowing_conversion<T, T, true, true> : public std::false_type {
};

template <class From, class To>
struct is_narrowing_conversion<From, To, false, true>
    : public negation<std::is_convertible<From, To>> {
};

// converts_to_higher_integer_rank{{{1
template <class From, class To, bool = (sizeof(From) < sizeof(To))>
struct converts_to_higher_integer_rank : public std::true_type {
};
template <class From, class To>
struct converts_to_higher_integer_rank<From, To, false>
    : public std::is_same<decltype(declval<From>() + declval<To>()), To> {
};

// is_aligned(_v){{{1
template <class Flag, size_t Alignment> struct is_aligned;
template <size_t Alignment>
struct is_aligned<flags::vector_aligned_tag, Alignment> : public std::true_type {
};
template <size_t Alignment>
struct is_aligned<flags::element_aligned_tag, Alignment> : public std::false_type {
};
template <size_t GivenAlignment, size_t Alignment>
struct is_aligned<flags::overaligned_tag<GivenAlignment>, Alignment>
    : public std::integral_constant<bool, (GivenAlignment >= Alignment)> {
};
template <class Flag, size_t Alignment>
constexpr bool is_aligned_v = is_aligned<Flag, Alignment>::value;

// when_(un)aligned{{{1
/**
 * \internal
 * Implicitly converts from flags that specify alignment
 */
template <size_t Alignment>
class when_aligned
{
public:
    constexpr when_aligned(flags::vector_aligned_tag) {}
    template <size_t Given, class = std::enable_if_t<(Given >= Alignment)>>
    constexpr when_aligned(flags::overaligned_tag<Given>)
    {
    }
};
template <size_t Alignment>
class when_unaligned
{
public:
    constexpr when_unaligned(flags::element_aligned_tag) {}
    template <size_t Given, class = std::enable_if_t<(Given < Alignment)>>
    constexpr when_unaligned(flags::overaligned_tag<Given>)
    {
    }
};

//}}}1
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END
#endif  // VC_DATAPAR_DETAIL_H_

// vim: foldmethod=marker
