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

#ifndef VC_SIMD_DETAIL_H_
#define VC_SIMD_DETAIL_H_

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
// nullarg{{{
inline constexpr struct nullarg_t {
} nullarg;

// }}}
// unused{{{1
template <class T> static constexpr void unused(T && ) {}

// assert_unreachable{{{1
template <class T> struct assert_unreachable {
    static_assert(!std::is_same_v<T, T>, "this should be unreachable");
};

// custom diagnostics for UB {{{1
#if defined Vc_GCC
template <class T>
[[gnu::weak, gnu::noinline,
gnu::warning("Your code is invoking undefined behavior. Please fix your code.")]]
const T &warn_ub(const T &x);
template <class T>
[[gnu::weak, gnu::noinline]]
const T &warn_ub(const T &x) { return x; }
#else
template <class T> const T &warn_ub(const T &x) { return x; }
#endif

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
    template <class T> Vc_INTRINSIC real_assert &operator<<(T &&x)
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

// size_tag_type {{{1
template <class T, class A>
auto size_tag_type_f(int)->size_constant<simd_size<T, A>::value>;
template <class T, class A> auto size_tag_type_f(float)->size_constant<0>;
template <class T, class A> using size_tag_type = decltype(size_tag_type_f<T, A>(0));

// promote_preserving_unsigned{{{1
// work around crazy semantics of unsigned integers of lower rank than int:
// Before applying an operator the operands are promoted to int. In which case over- or
// underflow is UB, even though the operand types were unsigned.
template <class T> Vc_INTRINSIC const T &promote_preserving_unsigned(const T &x)
{
    return x;
}
Vc_INTRINSIC unsigned int promote_preserving_unsigned(const unsigned char &x)
{
    return x;
}
Vc_INTRINSIC unsigned int promote_preserving_unsigned(const unsigned short &x)
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
/**\internal
 * Helper may_alias<T> that turns T into the type to be used for an aliasing pointer. This
 * adds the may_alias attribute to T (with compilers that support it).
 */
template <typename T> using may_alias [[gnu::may_alias]] = T;

// simd and simd_mask base for unsupported <T, Abi>{{{1
struct unsupported_base {
    unsupported_base() = delete;
    unsupported_base(const unsupported_base &) = delete;
    unsupported_base &operator=(const unsupported_base &) = delete;
    ~unsupported_base() = delete;
};

// traits forward declaration{{{1
/**
 * \internal
 * Defines the implementation of a given <T, Abi>.
 *
 * Implementations must ensure that only valid <T, Abi> instantiations are possible.
 * Static assertions in the type definition do not suffice. It is important that
 * SFINAE works.
 */
struct invalid_traits {
    using is_valid = std::false_type;
    static constexpr size_t simd_member_alignment = 1;
    struct simd_impl_type;
    struct simd_member_type {};
    struct simd_cast_type;
    using simd_base = unsupported_base;
    static constexpr size_t mask_member_alignment = 1;
    struct mask_impl_type;
    struct mask_member_type {};
    struct mask_cast_type;
    using mask_base = unsupported_base;
};
template <class T, class Abi, class = void_t<>> struct traits : invalid_traits {
};

// get_impl(_t){{{1
template <class T> struct get_impl;
template <class T> using get_impl_t = typename get_impl<std::decay_t<T>>::type;

// get_traits(_t){{{1
template <class T> struct get_traits;
template <class T> using get_traits_t = typename get_traits<std::decay_t<T>>::type;

// next_power_of_2{{{1
/**
 * \internal
 * Returns the next power of 2 larger than or equal to \p x.
 */
constexpr std::size_t next_power_of_2(std::size_t x)
{
    return (x & (x - 1)) == 0 ? x : next_power_of_2((x | (x >> 1)) + 1);
}

// private_init, bitset_init{{{1
/**
 * \internal
 * Tag used for private init constructor of simd and simd_mask
 */
inline constexpr struct private_init_t {} private_init = {};
inline constexpr struct bitset_init_t {} bitset_init = {};

// size_tag{{{1
template <size_t N> inline constexpr size_constant<N> size_tag = {};

// identity/id{{{1
template <class T> struct identity {
    using type = T;
};
template <class T> using id = typename identity<T>::type;

// bool_constant{{{1
template <bool Value> using bool_constant = std::integral_constant<bool, Value>;
using std::true_type;
using std::false_type;

// is_narrowing_conversion<From, To>{{{1
template <class From, class To, bool = std::is_arithmetic<From>::value,
          bool = std::is_arithmetic<To>::value>
struct is_narrowing_conversion;

// ignore "warning C4018: '<': signed/unsigned mismatch" in the following trait. The implicit
// conversions will do the right thing here.
template <class From, class To>
struct is_narrowing_conversion<From, To, true, true>
    : public bool_constant<(
          std::numeric_limits<From>::digits > std::numeric_limits<To>::digits ||
          std::numeric_limits<From>::max() > std::numeric_limits<To>::max() ||
          std::numeric_limits<From>::lowest() < std::numeric_limits<To>::lowest() ||
          (std::is_signed<From>::value && std::is_unsigned<To>::value))> {
};

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
    : public std::is_same<decltype(std::declval<From>() + std::declval<To>()), To> {
};

// is_aligned(_v){{{1
template <class Flag, size_t Alignment> struct is_aligned;
template <size_t Alignment>
struct is_aligned<vector_aligned_tag, Alignment> : public std::true_type {
};
template <size_t Alignment>
struct is_aligned<element_aligned_tag, Alignment> : public std::false_type {
};
template <size_t GivenAlignment, size_t Alignment>
struct is_aligned<overaligned_tag<GivenAlignment>, Alignment>
    : public std::integral_constant<bool, (GivenAlignment >= Alignment)> {
};
template <class Flag, size_t Alignment>
inline constexpr bool is_aligned_v = is_aligned<Flag, Alignment>::value;

// when_(un)aligned{{{1
/**
 * \internal
 * Implicitly converts from flags that specify alignment
 */
template <size_t Alignment>
class when_aligned
{
public:
    constexpr when_aligned(vector_aligned_tag) {}
    template <size_t Given, class = std::enable_if_t<(Given >= Alignment)>>
    constexpr when_aligned(overaligned_tag<Given>)
    {
    }
};
template <size_t Alignment>
class when_unaligned
{
public:
    constexpr when_unaligned(element_aligned_tag) {}
    template <size_t Given, class = std::enable_if_t<(Given < Alignment)>>
    constexpr when_unaligned(overaligned_tag<Given>)
    {
    }
};

// data(simd/simd_mask) {{{1
template <class T, class A> constexpr Vc_INTRINSIC_L const auto &data(const Vc::simd<T, A> &x) Vc_INTRINSIC_R;
template <class T, class A> constexpr Vc_INTRINSIC_L auto &data(Vc::simd<T, A> & x) Vc_INTRINSIC_R;

template <class T, class A> constexpr Vc_INTRINSIC_L const auto &data(const Vc::simd_mask<T, A> &x) Vc_INTRINSIC_R;
template <class T, class A> constexpr Vc_INTRINSIC_L auto &data(Vc::simd_mask<T, A> &x) Vc_INTRINSIC_R;

// simd_converter {{{1
template <class FromT, class FromA, class ToT, class ToA> struct simd_converter;
template <class T, class A> struct simd_converter<T, A, T, A> {
    template <class U> Vc_INTRINSIC const U &operator()(const U &x) { return x; }
};

// to_value_type_or_member_type {{{1
template <class V>
constexpr Vc_INTRINSIC auto to_value_type_or_member_type(const V &x)->decltype(detail::data(x))
{
    return detail::data(x);
}

template <class V>
constexpr Vc_INTRINSIC const typename V::value_type &to_value_type_or_member_type(
    const typename V::value_type &x)
{
    return x;
}

// constexpr_if {{{1
template <class IfFun, class ElseFun>
Vc_INTRINSIC auto impl_or_fallback_dispatch(std::true_type, IfFun &&fun, ElseFun &&)
{
    return fun(0);
}

template <class IfFun, class ElseFun>
Vc_INTRINSIC auto impl_or_fallback_dispatch(std::false_type, IfFun &&, ElseFun &&fun)
{
    return fun(0);
}

template <bool Condition, class IfFun, class ElseFun>
Vc_INTRINSIC auto constexpr_if(IfFun &&if_fun, ElseFun &&else_fun)
{
    return impl_or_fallback_dispatch(Vc::detail::bool_constant<Condition>(), if_fun,
                                     else_fun);
}

template <bool Condition, class IfFun> Vc_INTRINSIC auto constexpr_if(IfFun &&if_fun)
{
    return impl_or_fallback_dispatch(Vc::detail::bool_constant<Condition>(), if_fun,
                                     [](int) {});
}

template <bool Condition, bool Condition2, class IfFun, class... Remainder>
Vc_INTRINSIC auto constexpr_if(IfFun &&if_fun, Vc::detail::bool_constant<Condition2>,
                               Remainder &&... rem)
{
    return impl_or_fallback_dispatch(
        Vc::detail::bool_constant<Condition>(), if_fun, [&](auto tmp_) {
            return constexpr_if<(std::is_same<decltype(tmp_), int>::value && Condition2)>(
                rem...);
        });
}

#ifdef __cpp_if_constexpr
#define Vc_CONSTEXPR_IF_RETURNING(...) if constexpr (__VA_ARGS__) {
#define Vc_CONSTEXPR_IF(...) if constexpr (__VA_ARGS__) {
#define Vc_CONSTEXPR_ELSE_IF(...) } else if constexpr (__VA_ARGS__) {
#define Vc_CONSTEXPR_ELSE } else {
#define Vc_CONSTEXPR_ENDIF }
#else
#define Vc_CONSTEXPR_IF_RETURNING(...) return Vc::detail::constexpr_if<(__VA_ARGS__)>([&](auto) {
#define Vc_CONSTEXPR_IF(...) Vc::detail::constexpr_if<(__VA_ARGS__)>([&](auto) {
#define Vc_CONSTEXPR_ELSE_IF(...) }, Vc::detail::bool_constant<(__VA_ARGS__)>(), [&](auto) {
#define Vc_CONSTEXPR_ELSE }, [&](auto) {
#define Vc_CONSTEXPR_ENDIF });
#endif

// bool_storage_member_type{{{1
template <size_t Size> struct bool_storage_member_type;
template <size_t Size>
using bool_storage_member_type_t = typename bool_storage_member_type<Size>::type;

// fixed_size_storage fwd decl {{{1
template <class T, int N> struct fixed_size_storage_builder_wrapper;
template <class T, int N>
using fixed_size_storage = typename fixed_size_storage_builder_wrapper<T, N>::type;

// Storage fwd decl{{{1
template <class ValueType, size_t Size, class = std::void_t<>> struct Storage;
template <class T> using storage16_t = Storage<T, 16 / sizeof(T)>;
template <class T> using storage32_t = Storage<T, 32 / sizeof(T)>;
template <class T> using storage64_t = Storage<T, 64 / sizeof(T)>;

// bit_iteration{{{1
constexpr uint popcount(uint x) { return __builtin_popcount(x); }
constexpr ulong popcount(ulong x) { return __builtin_popcountl(x); }
constexpr ullong popcount(ullong x) { return __builtin_popcountll(x); }

constexpr uint ctz(uint x) { return __builtin_ctz(x); }
constexpr ulong ctz(ulong x) { return __builtin_ctzl(x); }
constexpr ullong ctz(ullong x) { return __builtin_ctzll(x); }
constexpr uint clz(uint x) { return __builtin_clz(x); }
constexpr ulong clz(ulong x) { return __builtin_clzl(x); }
constexpr ullong clz(ullong x) { return __builtin_clzll(x); }

template <class T, class F> void bit_iteration(T k_, F &&f)
{
    static_assert(sizeof(ullong) >= sizeof(T));
    std::conditional_t<sizeof(T) <= sizeof(uint), uint, ullong> k;
    if constexpr (std::is_convertible_v<T, decltype(k)>) {
        k = k_;
    } else {
        k = k_.to_ullong();
    }
    switch (popcount(k)) {
    default:
        do {
            f(ctz(k));
            k &= (k - 1);
        } while (k);
        break;
    /*case 3:
        f(ctz(k));
        k &= (k - 1);
        [[fallthrough]];*/
    case 2:
        f(ctz(k));
        [[fallthrough]];
    case 1:
        f(popcount(~decltype(k)()) - 1 - clz(k));
        [[fallthrough]];
    case 0:
        break;
    }
}

//}}}1
// simd_tuple {{{
// why not std::tuple?
// 1. std::tuple gives no guarantee about the storage order, but I require storage
//    equivalent to std::array<T, N>
// 2. much less code to instantiate: I require a very small subset of std::tuple
//    functionality
// 3. direct access to the element type (first template argument)
// 4. enforces equal element type, only different Abi types are allowed

template <class T, class... Abis> struct simd_tuple;

template <size_t N, class T, class... Abis>
constexpr auto get_simd(const simd_tuple<T, Abis...> &);
template <size_t K, class T>
constexpr inline const auto &tuple_pop_front(size_constant<K>, const T &);
template <size_t K, class T>
constexpr inline auto &tuple_pop_front(size_constant<K>, T &);
template <size_t K, class T, class A0, class... As>
constexpr auto tuple_front(const simd_tuple<T, A0, As...> &);

//}}}
// is_homogeneous_tuple (all ABI tags are equal) {{{
template <class T> struct is_homogeneous_tuple;
template <class T>
inline constexpr bool is_homogeneous_tuple_v = is_homogeneous_tuple<T>::value;

// only 1 member => homogeneous
template <class T, class Abi>
struct is_homogeneous_tuple<detail::simd_tuple<T, Abi>> : std::true_type {
};

// more than 1 member
template <class T, class A0, class... Abis>
struct is_homogeneous_tuple<detail::simd_tuple<T, A0, Abis...>> {
    static constexpr bool value = (std::is_same_v<A0, Abis> && ...);
};
// }}}

}  // namespace detail
Vc_VERSIONED_NAMESPACE_END
#endif  // VC_SIMD_DETAIL_H_

// vim: foldmethod=marker
