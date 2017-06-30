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

#include <functional>
#ifndef NDEBUG
#include <iostream>
#endif  // NDEBUG
#include <limits>
#include "macros.h"
#include "flags.h"
#include "type_traits.h"

Vc_VERSIONED_NAMESPACE_BEGIN
namespace detail
{
// unused{{{1
template <class T> static constexpr void unused(T && ) {}

// dummy_assert {{{1
#ifdef NDEBUG
struct dummy_assert {
    template <class T> Vc_INTRINSIC dummy_assert &operator<<(T &&) noexcept
    {
        return *this;
    }
};
#else   // NDEBUG
// real_assert {{{1
struct real_assert {
    Vc_INTRINSIC real_assert(bool ok, const char *code, const char *file, int line)
        : failed(!ok)
    {
        if (Vc_IS_UNLIKELY(failed)) {
            printFirst(code, file, line);
        }
    }
    Vc_INTRINSIC ~real_assert()
    {
        if (Vc_IS_UNLIKELY(failed)) {
            finalize();
        }
    }
    template <class T> Vc_INTRINSIC real_assert &operator<<(T &&x) const
    {
        if (Vc_IS_UNLIKELY(failed)) {
            print(std::forward<T>(x));
        }
        return *this;
    }

private:
    void printFirst(const char *code, const char *file, int line)
    {
        std::cerr << file << ':' << line << ": assert(" << code << ") failed.";
    }
    template <class T> void print(T &&x) const { std::cerr << std::forward<T>(x); }
    void finalize()
    {
        std::cerr << std::endl;
        std::abort();
    }
    bool failed;
};
#endif  // NDEBUG

// assertCorrectAlignment {{{1
#if defined Vc_CHECK_ALIGNMENT || defined COMPILE_FOR_UNIT_TESTS
template <class V = void, class T>
Vc_ALWAYS_INLINE void assertCorrectAlignment(const T *ptr)
{
    auto &&is_aligned = [](const T *p) -> bool {
        constexpr size_t s =
            alignof(std::conditional_t<std::is_same<void, V>::value, T, V>);
        return (reinterpret_cast<size_t>(p) & (s - 1)) == 0;
    };
#ifdef COMPILE_FOR_UNIT_TESTS
    Vc_ASSERT(is_aligned(ptr))
        << " ptr = " << ptr << ", expected alignment = "
        << alignof(std::conditional_t<std::is_same<void, V>::value, T, V>);
    detail::unused(is_aligned);
#else
    if (Vc_IS_UNLIKELY(!is_aligned(ptr))) {
        std::fprintf(stderr, "A load with incorrect alignment has just been called. Look at the stacktrace to find the guilty load.\n");
        std::abort();
    }
#endif
}
#else
template <class V = void, class T>
Vc_ALWAYS_INLINE void assertCorrectAlignment(const T *)
{
}
#endif

// size_constant {{{1
template <size_t X> using size_constant = std::integral_constant<size_t, X>;

// size_tag_type {{{1
template <class T, class A>
auto size_tag_type_f(int)->size_constant<datapar_size<T, A>::value>;
template <class T, class A> auto size_tag_type_f(float)->size_constant<0>;
template <class T, class A> using size_tag_type = decltype(size_tag_type_f<T, A>(0));

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

// data(datapar/mask) {{{1
template <class T, class A> Vc_INTRINSIC_L const auto &data(const Vc::datapar<T, A> &x) Vc_INTRINSIC_R;
template <class T, class A> Vc_INTRINSIC_L auto &data(Vc::datapar<T, A> & x) Vc_INTRINSIC_R;

template <class T, class A> Vc_INTRINSIC_L const auto &data(const Vc::mask<T, A> &x) Vc_INTRINSIC_R;
template <class T, class A> Vc_INTRINSIC_L auto &data(Vc::mask<T, A> &x) Vc_INTRINSIC_R;

// to_value_type_or_member_type {{{1
template <class V>
Vc_INTRINSIC auto to_value_type_or_member_type(const V &x)->decltype(detail::data(x))
{
    return detail::data(x);
}

template <class V>
Vc_INTRINSIC const typename V::value_type &to_value_type_or_member_type(
    const typename V::value_type &x)
{
    return x;
}

//}}}1
}  // namespace detail
Vc_VERSIONED_NAMESPACE_END
#endif  // VC_DATAPAR_DETAIL_H_

// vim: foldmethod=marker
