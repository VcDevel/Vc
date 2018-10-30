#ifndef BITS_SIMD_H_
#define BITS_SIMD_H_

#include "simd_detail.h"
#include <bitset>
#include <climits>
#include <cstring>
#include <functional>
#include <iosfwd>
#include <limits>
#include <utility>
#ifndef NDEBUG
#include <iostream>
#endif  // NDEBUG
#if defined Vc_HAVE_SSE || defined Vc_HAVE_MMX
#include <x86intrin.h>
#endif  // Vc_HAVE_SSE

Vc_VERSIONED_NAMESPACE_BEGIN
// load/store flags {{{
using size_t = std::size_t;

struct element_aligned_tag {};
struct vector_aligned_tag {};
template <size_t N> struct overaligned_tag {
    static constexpr size_t alignment = N;
};
inline constexpr element_aligned_tag element_aligned = {};
inline constexpr vector_aligned_tag vector_aligned = {};
template <size_t N> inline constexpr overaligned_tag<N> overaligned = {};
// }}}
namespace detail {
// type traits {{{
// integer type aliases{{{
using uchar = unsigned char;
using schar = signed char;
using ushort = unsigned short;
using uint = unsigned int;
using ulong = unsigned long;
using llong = long long;
using ullong = unsigned long long;
using wchar = wchar_t;
using char16 = char16_t;
using char32 = char32_t;
//}}}

template <class... Ts> using all = std::conjunction<Ts...>;
template <class... Ts> using any = std::disjunction<Ts...>;

// is_equal {{{
template <class T, T a, T b> struct is_equal : public std::false_type {
};
template <class T, T a> struct is_equal<T, a, a> : public std::true_type {
};

// }}}
// sizeof {{{
template <class T, std::size_t Expected>
struct has_expected_sizeof : public std::integral_constant<bool, sizeof(T) == Expected> {
};

// }}}
// value_type_or_identity {{{
template <class T> typename T::value_type value_type_or_identity_impl(int);
template <class T> T value_type_or_identity_impl(float);
template <class T>
using value_type_or_identity = decltype(value_type_or_identity_impl<T>(int()));

// }}}
// is_vectorizable {{{
template <class T> struct is_vectorizable : public std::is_arithmetic<T> {};
template <> struct is_vectorizable<bool> : public std::false_type {};
template <class T> inline constexpr bool is_vectorizable_v = is_vectorizable<T>::value;
// Deduces to a vectorizable type
template <class T, class = enable_if_t<is_vectorizable_v<T>>> using Vectorizable = T;

// }}}
// is_possible_loadstore_conversion {{{
template <class Ptr, class ValueType>
struct is_possible_loadstore_conversion
    : all<is_vectorizable<Ptr>, is_vectorizable<ValueType>> {
};
template <> struct is_possible_loadstore_conversion<bool, bool> : std::true_type {
};
// Deduces to a type allowed for load/store with the given value type.
template <class Ptr, class ValueType,
          class = enable_if_t<is_possible_loadstore_conversion<Ptr, ValueType>::value>>
using loadstore_ptr_type = Ptr;

// }}}
// is_less_than {{{
template <int A, int B> struct is_less_than : public std::integral_constant<bool, (A < B)> {
};
// }}}
// is_equal_to {{{
template <int A, int B>
struct is_equal_to : public std::integral_constant<bool, (A == B)> {
};
// }}}
template <size_t X> using size_constant = std::integral_constant<size_t, X>;
// is_bitmask{{{
template <class T, class = std::void_t<>> struct is_bitmask : std::false_type {
    constexpr is_bitmask(const T &) noexcept {}
};
template <class T> inline constexpr bool is_bitmask_v = is_bitmask<T>::value;

/* the Storage<bool, N> case:
template <class T>
struct is_bitmask<T, std::enable_if_t<(sizeof(T) < T::width)>> : std::true_type {
    constexpr is_bitmask(const T &) noexcept {}
};*/

// the __mmaskXX case:
template <class T>
struct is_bitmask<
    T, std::void_t<decltype(std::declval<unsigned &>() = std::declval<T>() & 1u)>>
    : std::true_type {
    constexpr is_bitmask(const T &) noexcept {}
};

// }}}
// int_for_sizeof{{{
template <class T, size_t = sizeof(T)> struct int_for_sizeof;
template <class T> struct int_for_sizeof<T, 1> { using type = signed char; };
template <class T> struct int_for_sizeof<T, 2> { using type = signed short; };
template <class T> struct int_for_sizeof<T, 4> { using type = signed int; };
template <class T> struct int_for_sizeof<T, 8> { using type = signed long long; };
template <class T> using int_for_sizeof_t = typename int_for_sizeof<T>::type;

// }}}
// equal_int_type{{{
/**
 * \internal
 * TODO: rename to same_value_representation
 * Type trait to find the equivalent integer type given a(n) (un)signed long type.
 */
template <class T, size_t = sizeof(T)> struct equal_int_type;
template <> struct equal_int_type< long, sizeof(int)> { using type =    int; };
template <> struct equal_int_type< long, sizeof(llong)> { using type =  llong; };
template <> struct equal_int_type<ulong, sizeof(uint)> { using type =   uint; };
template <> struct equal_int_type<ulong, sizeof(ullong)> { using type = ullong; };
template <> struct equal_int_type< char, 1> { using type = std::conditional_t<std::is_signed_v<char>, schar, uchar>; };
template <size_t N> struct equal_int_type<char16_t, N> { using type = std::uint_least16_t; };
template <size_t N> struct equal_int_type<char32_t, N> { using type = std::uint_least32_t; };
template <> struct equal_int_type<wchar_t, 1> { using type = std::conditional_t<std::is_signed_v<wchar_t>, schar, uchar>; };
template <> struct equal_int_type<wchar_t, sizeof(short)> { using type = std::conditional_t<std::is_signed_v<wchar_t>, short, ushort>; };
template <> struct equal_int_type<wchar_t, sizeof(int)> { using type = std::conditional_t<std::is_signed_v<wchar_t>, int, uint>; };

template <class T> using equal_int_type_t = typename equal_int_type<T>::type;

// }}}
// has_same_value_representation{{{
template <class T, class = std::void_t<>>
struct has_same_value_representation : std::false_type {
};

template <class T>
struct has_same_value_representation<T, std::void_t<typename equal_int_type<T>::type>>
    : std::true_type {
};

template <class T>
inline constexpr bool has_same_value_representation_v =
    has_same_value_representation<T>::value;

// }}}
// is_fixed_size_abi{{{
template <class T> struct is_fixed_size_abi : std::false_type {
};

template <int N> struct is_fixed_size_abi<simd_abi::fixed_size<N>> : std::true_type {
};

template <class T> inline constexpr bool is_fixed_size_abi_v = is_fixed_size_abi<T>::value;

// }}}
// is_callable{{{
/*
template <class F, class... Args>
auto is_callable_dispatch(int, F &&fun, Args &&... args)
    -> std::conditional_t<true, std::true_type,
                          decltype(fun(std::forward<Args>(args)...))>;
template <class F, class... Args>
std::false_type is_callable_dispatch(float, F &&, Args &&...);

template <class... Args, class F> constexpr bool is_callable(F &&)
{
    return decltype(is_callable_dispatch(int(), std::declval<F>(), std::declval<Args>()...))::value;
}
*/

template <class F, class = std::void_t<>, class... Args>
struct is_callable : std::false_type {
};

template <class F, class... Args>
struct is_callable<F, std::void_t<decltype(std::declval<F>()(std::declval<Args>()...))>,
                   Args...> : std::true_type {
};

template <class F, class... Args>
inline constexpr bool is_callable_v = is_callable<F, void, Args...>::value;

#define Vc_TEST_LAMBDA(...)                                                              \
    decltype(                                                                            \
        [](auto &&... arg_s_) -> decltype(fun_(std::declval<decltype(arg_s_)>()...)) * { \
            return nullptr;                                                              \
        })

#define Vc_IS_CALLABLE(fun_, ...)                                                        \
    decltype(Vc::detail::is_callable_dispatch(int(), Vc_TEST_LAMBDA(fun_),               \
                                              __VA_ARGS__))::value

// }}}
// constexpr feature detection{{{
constexpr inline bool have_mmx = 0
#ifdef Vc_HAVE_MMX
                                 + 1
#endif
    ;
constexpr inline bool have_sse = 0
#ifdef Vc_HAVE_SSE
                                 + 1
#endif
    ;
constexpr inline bool have_sse2 = 0
#ifdef Vc_HAVE_SSE2
                                  + 1
#endif
    ;
constexpr inline bool have_sse3 = 0
#ifdef Vc_HAVE_SSE3
                                  + 1
#endif
    ;
constexpr inline bool have_ssse3 = 0
#ifdef Vc_HAVE_SSSE3
                                   + 1
#endif
    ;
constexpr inline bool have_sse4_1 = 0
#ifdef Vc_HAVE_SSE4_1
                                    + 1
#endif
    ;
constexpr inline bool have_sse4_2 = 0
#ifdef Vc_HAVE_SSE4_2
                                    + 1
#endif
    ;
constexpr inline bool have_xop = 0
#ifdef Vc_HAVE_XOP
                                 + 1
#endif
    ;
constexpr inline bool have_avx = 0
#ifdef Vc_HAVE_AVX
                                 + 1
#endif
    ;
constexpr inline bool have_avx2 = 0
#ifdef Vc_HAVE_AVX2
                                  + 1
#endif
    ;
constexpr inline bool have_bmi = 0
#ifdef Vc_HAVE_BMI
                                 + 1
#endif
    ;
constexpr inline bool have_bmi2 = 0
#ifdef Vc_HAVE_BMI2
                                  + 1
#endif
    ;
constexpr inline bool have_lzcnt = 0
#ifdef Vc_HAVE_LZCNT
                                   + 1
#endif
    ;
constexpr inline bool have_sse4a = 0
#ifdef Vc_HAVE_SSE4A
                                   + 1
#endif
    ;
constexpr inline bool have_fma = 0
#ifdef Vc_HAVE_FMA
                                 + 1
#endif
    ;
constexpr inline bool have_fma4 = 0
#ifdef Vc_HAVE_FMA4
                                  + 1
#endif
    ;
constexpr inline bool have_f16c = 0
#ifdef Vc_HAVE_F16C
                                  + 1
#endif
    ;
constexpr inline bool have_popcnt = 0
#ifdef Vc_HAVE_POPCNT
                                    + 1
#endif
    ;
constexpr inline bool have_avx512f = 0
#ifdef Vc_HAVE_AVX512F
                                     + 1
#endif
    ;
constexpr inline bool have_avx512dq = 0
#ifdef Vc_HAVE_AVX512DQ
                                      + 1
#endif
    ;
constexpr inline bool have_avx512vl = 0
#ifdef Vc_HAVE_AVX512VL
                                      + 1
#endif
    ;
constexpr inline bool have_avx512bw = 0
#ifdef Vc_HAVE_AVX512BW
                                      + 1
#endif
    ;
constexpr inline bool have_avx512dq_vl = have_avx512dq && have_avx512vl;
constexpr inline bool have_avx512bw_vl = have_avx512bw && have_avx512vl;

constexpr inline bool have_neon = 0
#ifdef Vc_HAVE_NEON
                                  + 1
#endif
    ;
// }}}
// }}}
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
[[gnu::__weak__, gnu::__noinline__,
gnu::warning("Your code is invoking undefined behavior. Please fix your code.")]]
const T &warn_ub(const T &x);
template <class T>
[[gnu::__weak__, gnu::__noinline__]]
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

// size_or_zero {{{1
template <class T, class A, size_t N = simd_size<T, A>::value>
constexpr size_t size_or_zero_dispatch(int)
{
    return N;
}
template <class T, class A> constexpr size_t size_or_zero_dispatch(float) {
  return 0;
}
template <class T, class A>
inline constexpr size_t size_or_zero = size_or_zero_dispatch<T, A>(0);

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
template <typename T> using may_alias [[gnu::__may_alias__]] = T;

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
template <class T, class Abi, class = std::void_t<>> struct traits : invalid_traits {
};

// }}}1
// get_impl(_t){{{
template <class T> struct get_impl;
template <class T> using get_impl_t = typename get_impl<std::decay_t<T>>::type;

// }}}
// get_traits(_t){{{
template <class T> struct get_traits;
template <class T> using get_traits_t = typename get_traits<std::decay_t<T>>::type;

// }}}
// make_immediate{{{
template <unsigned Stride> constexpr unsigned make_immediate(unsigned a, unsigned b)
{
    return a + b * Stride;
}
template <unsigned Stride>
constexpr unsigned make_immediate(unsigned a, unsigned b, unsigned c, unsigned d)
{
    return a + Stride * (b + Stride * (c + Stride * d));
}

// }}}
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

// }}}1
// when_(un)aligned{{{
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

// }}}
// data(simd/simd_mask) {{{
template <class T, class A> Vc_INTRINSIC constexpr const auto &data(const Vc::simd<T, A> &x);
template <class T, class A> Vc_INTRINSIC constexpr auto &data(Vc::simd<T, A> & x);

template <class T, class A> Vc_INTRINSIC constexpr const auto &data(const Vc::simd_mask<T, A> &x);
template <class T, class A> Vc_INTRINSIC constexpr auto &data(Vc::simd_mask<T, A> &x);

// }}}
// simd_converter {{{
template <class FromT, class FromA, class ToT, class ToA> struct simd_converter;
template <class T, class A> struct simd_converter<T, A, T, A> {
    template <class U> Vc_INTRINSIC const U &operator()(const U &x) { return x; }
};

// }}}
// to_value_type_or_member_type {{{
template <class V>
Vc_INTRINSIC constexpr auto to_value_type_or_member_type(const V &x)->decltype(detail::data(x))
{
    return detail::data(x);
}

template <class V>
Vc_INTRINSIC constexpr const typename V::value_type &to_value_type_or_member_type(
    const typename V::value_type &x)
{
    return x;
}

// }}}
// bool_storage_member_type{{{
template <size_t Size> struct bool_storage_member_type;
template <size_t Size>
using bool_storage_member_type_t = typename bool_storage_member_type<Size>::type;

// }}}
// fixed_size_storage fwd decl {{{
template <class T, int N> struct fixed_size_storage_builder_wrapper;
template <class T, int N>
using fixed_size_storage = typename fixed_size_storage_builder_wrapper<T, N>::type;

// }}}
// Storage fwd decl{{{
template <class ValueType, size_t Size, class = std::void_t<>> struct Storage;
template <class T> using storage16_t = Storage<T, 16 / sizeof(T)>;
template <class T> using storage32_t = Storage<T, 32 / sizeof(T)>;
template <class T> using storage64_t = Storage<T, 64 / sizeof(T)>;

// }}}
// bit_iteration{{{
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

//}}}
// firstbit{{{
template <class T> Vc_INTRINSIC Vc_CONST auto firstbit(T bits)
{
    static_assert(std::is_integral_v<T>, "firstbit requires an integral argument");
    if constexpr (sizeof(T) <= sizeof(int)) {
        return __builtin_ctz(bits);
    } else if constexpr(alignof(ullong) == 8) {
        return __builtin_ctzll(bits);
    } else {
        uint lo = bits;
        return lo == 0 ? 32 + __builtin_ctz(bits >> 32) : __builtin_ctz(lo);
    }
}

// }}}
// lastbit{{{
template <class T> Vc_INTRINSIC Vc_CONST auto lastbit(T bits)
{
    static_assert(std::is_integral_v<T>, "firstbit requires an integral argument");
    if constexpr (sizeof(T) <= sizeof(int)) {
        return 31 - __builtin_clz(bits);
    } else if constexpr(alignof(ullong) == 8) {
        return 63 - __builtin_clzll(bits);
    } else {
        uint lo = bits;
        uint hi = bits >> 32u;
        return hi == 0 ? 31 - __builtin_clz(lo) : 63 - __builtin_clz(hi);
    }
}

// }}}
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
// convert_mask declaration {{{
template <class To, class From> inline To convert_mask(From k);

// }}}
// shift_left, shift_right, increment, decrement {{{
template <class T = void> struct shift_left {
    constexpr T operator()(const T &a, const T &b) const { return a << b; }
};
template <> struct shift_left<void> {
    template <typename L, typename R> constexpr auto operator()(L &&a, R &&b) const
    {
        return std::forward<L>(a) << std::forward<R>(b);
    }
};
template <class T = void> struct shift_right {
    constexpr T operator()(const T &a, const T &b) const { return a >> b; }
};
template <> struct shift_right<void> {
    template <typename L, typename R> constexpr auto operator()(L &&a, R &&b) const
    {
        return std::forward<L>(a) >> std::forward<R>(b);
    }
};
template <class T = void> struct increment {
    constexpr T operator()(T a) const { return ++a; }
};
template <> struct increment<void> {
    template <typename T> constexpr T operator()(T a) const { return ++a; }
};
template <class T = void> struct decrement {
    constexpr T operator()(T a) const { return --a; }
};
template <> struct decrement<void> {
    template <typename T> constexpr T operator()(T a) const { return --a; }
};

// }}}
// get_impl {{{
template <class T> struct get_impl {
    static_assert(
        std::is_arithmetic<T>::value,
        "Vc chose the wrong implementation class. This should not be possible.");

    template <class U, class F>
    Vc_INTRINSIC T masked_load(T d, bool k, const U *mem, F)
    {
        if (k) {
            d = static_cast<T>(mem[0]);
        }
        return d;
    }
};
template <> struct get_impl<bool> {
    template <class F> Vc_INTRINSIC bool masked_load(bool d, bool k, const bool *mem, F)
    {
        if (k) {
            d = mem[0];
        }
        return d;
    }
};
// }}}
// value_preserving(_or_int) {{{
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

// }}}
// intrinsic_type {{{
template <class T, size_t Bytes, class = std::void_t<>> struct intrinsic_type;
template <class T, size_t Size>
using intrinsic_type_t = typename intrinsic_type<T, Size * sizeof(T)>::type;
template <class T> using intrinsic_type2_t   = typename intrinsic_type<T, 2>::type;
template <class T> using intrinsic_type4_t   = typename intrinsic_type<T, 4>::type;
template <class T> using intrinsic_type8_t   = typename intrinsic_type<T, 8>::type;
template <class T> using intrinsic_type16_t  = typename intrinsic_type<T, 16>::type;
template <class T> using intrinsic_type32_t  = typename intrinsic_type<T, 32>::type;
template <class T> using intrinsic_type64_t  = typename intrinsic_type<T, 64>::type;
template <class T> using intrinsic_type128_t = typename intrinsic_type<T, 128>::type;

// }}}
// is_intrinsic{{{1
template <class T> struct is_intrinsic : public std::false_type {};
template <class T> inline constexpr bool is_intrinsic_v = is_intrinsic<T>::value;

// }}}
// builtin_type {{{1
template <class T, size_t N, class = void> struct builtin_type_n {};

// special case 1-element to be T itself
template <class T>
struct builtin_type_n<T, 1, std::enable_if_t<detail::is_vectorizable_v<T>>> {
    using type = T;
};

// else, use GNU-style builtin vector types
template <class T, size_t N>
struct builtin_type_n<T, N, std::enable_if_t<detail::is_vectorizable_v<T>>> {
    static constexpr size_t Bytes = N * sizeof(T);
    using type [[gnu::__vector_size__(Bytes)]] = T;
};

template <class T, size_t Bytes>
struct builtin_type : builtin_type_n<T, Bytes / sizeof(T)> {
    static_assert(Bytes % sizeof(T) == 0);
};

template <class T, size_t Size>
using builtin_type_t = typename builtin_type_n<T, Size>::type;
template <class T> using builtin_type2_t  = typename builtin_type<T, 2>::type;
template <class T> using builtin_type4_t  = typename builtin_type<T, 4>::type;
template <class T> using builtin_type8_t  = typename builtin_type<T, 8>::type;
template <class T> using builtin_type16_t = typename builtin_type<T, 16>::type;
template <class T> using builtin_type32_t = typename builtin_type<T, 32>::type;
template <class T> using builtin_type64_t = typename builtin_type<T, 64>::type;
template <class T> using builtin_type128_t = typename builtin_type<T, 128>::type;

// is_builtin_vector {{{1
template <class T, class = std::void_t<>> struct is_builtin_vector : std::false_type {};
template <class T>
struct is_builtin_vector<
    T,
    std::void_t<typename builtin_type<decltype(std::declval<T>()[0]), sizeof(T)>::type>>
    : std::is_same<
          T, typename builtin_type<decltype(std::declval<T>()[0]), sizeof(T)>::type> {
};

template <class T> inline constexpr bool is_builtin_vector_v = is_builtin_vector<T>::value;

// builtin_traits{{{1
template <class T, class = std::void_t<>> struct builtin_traits;
template <class T>
struct builtin_traits<T, std::void_t<std::enable_if_t<is_builtin_vector_v<T>>>> {
    using type = T;
    using value_type = decltype(std::declval<T>()[0]);
    static constexpr int width = sizeof(T) / sizeof(value_type);
    template <class U, int W = width>
    static constexpr bool is = std::is_same_v<value_type, U> &&W == width;
};
template <class T, size_t N>
struct builtin_traits<Storage<T, N>, std::void_t<builtin_type_t<T, N>>> {
    using type = builtin_type_t<T, N>;
    using value_type = T;
    static constexpr int width = N;
    template <class U, int W = width>
    static constexpr bool is = std::is_same_v<value_type, U> &&W == width;
};

// }}}
// builtin_cast{{{1
template <class To, class From, class Traits = builtin_traits<From>>
Vc_INTRINSIC constexpr typename builtin_type<To, sizeof(From)>::type builtin_cast(From x)
{
    return reinterpret_cast<typename builtin_type<To, sizeof(From)>::type>(x);
}
template <class To, class T, size_t N>
Vc_INTRINSIC constexpr typename builtin_type<To, sizeof(Storage<T, N>)>::type
builtin_cast(const Storage<T, N> &x)
{
    return reinterpret_cast<typename builtin_type<To, sizeof(Storage<T, N>)>::type>(x.d);
}

// }}}
// to_intrin {{{
template <class T, class Traits = builtin_traits<T>,
          class R = intrinsic_type_t<typename Traits::value_type, Traits::width>>
Vc_INTRINSIC constexpr R to_intrin(T x)
{
    return reinterpret_cast<R>(x);
}
template <class T, size_t N, class R = intrinsic_type_t<T, N>>
Vc_INTRINSIC constexpr R to_intrin(Storage<T, N> x)
{
    return reinterpret_cast<R>(x.d);
}

// }}}
// make_builtin{{{
template <class T, class... Args>
Vc_INTRINSIC constexpr builtin_type_t<T, sizeof...(Args)> make_builtin(Args &&... args)
{
    return builtin_type_t<T, sizeof...(Args)>{static_cast<T>(args)...};
}

// }}}
// builtin_broadcast{{{
template <size_t N, class T>
Vc_INTRINSIC constexpr builtin_type_t<T, N> builtin_broadcast(T x)
{
    if constexpr (N == 2) {
        return builtin_type_t<T, 2>{x, x};
    } else if constexpr (N == 4) {
        return builtin_type_t<T, 4>{x, x, x, x};
    } else if constexpr (N == 8) {
        return builtin_type_t<T, 8>{x, x, x, x, x, x, x, x};
    } else if constexpr (N == 16) {
        return builtin_type_t<T, 16>{x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x};
    } else if constexpr (N == 32) {
        return builtin_type_t<T, 32>{x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
                                     x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x};
    } else if constexpr (N == 64) {
        return builtin_type_t<T, 64>{x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
                                     x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
                                     x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
                                     x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x};
    } else if constexpr (N == 128) {
        return builtin_type_t<T, 128>{
            x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
            x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
            x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
            x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
            x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x};
    }
}

// }}}
// auto_broadcast{{{
template <class T> struct auto_broadcast {
    const T x;
    Vc_INTRINSIC constexpr auto_broadcast(T xx) : x(xx) {}
    template <class V> Vc_INTRINSIC constexpr operator V() const
    {
        static_assert(is_builtin_vector_v<V>);
        return reinterpret_cast<V>(builtin_broadcast<sizeof(V) / sizeof(T)>(x));
    }
};

// }}}
// generate_builtin{{{
template <class T, size_t N, class G, size_t... I>
Vc_INTRINSIC constexpr builtin_type_t<T, N> generate_builtin_impl(
    G &&gen, std::index_sequence<I...>)
{
    return builtin_type_t<T, N>{static_cast<T>(gen(size_constant<I>()))...};
}

template <class V, class Traits = builtin_traits<V>, class G>
Vc_INTRINSIC constexpr V generate_builtin(G &&gen)
{
    return generate_builtin_impl<typename Traits::value_type, Traits::width>(
        std::forward<G>(gen), std::make_index_sequence<Traits::width>());
}

template <class T, size_t N, class G>
Vc_INTRINSIC constexpr builtin_type_t<T, N> generate_builtin(G &&gen)
{
    return generate_builtin_impl<T, N>(std::forward<G>(gen),
                                       std::make_index_sequence<N>());
}

// }}}
// builtin_load{{{
template <class T, size_t N, size_t M = N * sizeof(T), class F>
builtin_type_t<T, N> builtin_load(const void *p, F)
{
#ifdef Vc_WORKAROUND_XXX_2
    using U = std::conditional_t<
        (std::is_integral_v<T> || M < 4), long long,
        std::conditional_t<(std::is_same_v<T, double> || M < 8), float, T>>;
    using V = builtin_type_t<U, N * sizeof(T) / sizeof(U)>;
#else   // Vc_WORKAROUND_XXX_2
    using V = builtin_type_t<T, N>;
#endif  // Vc_WORKAROUND_XXX_2
    V r;
    static_assert(M <= sizeof(V));
    if constexpr(std::is_same_v<F, element_aligned_tag>) {
    } else if constexpr(std::is_same_v<F, vector_aligned_tag>) {
        p = __builtin_assume_aligned(p, alignof(builtin_type_t<T, N>));
    } else {
        p = __builtin_assume_aligned(p, F::alignment);
    }
    std::memcpy(&r, p, M);
    return reinterpret_cast<builtin_type_t<T, N>>(r);
}

// }}}
// builtin_load16 {{{
template <class T, size_t M = 16, class F>
builtin_type16_t<T> builtin_load16(const void *p, F f)
{
    return builtin_load<T, 16 / sizeof(T), M>(p, f);
}

// }}}
// builtin_store{{{
template <size_t M = 0, class B, class Traits = builtin_traits<B>, class F>
void builtin_store(const B v, void *p, F)
{
    using T = typename Traits::value_type;
    constexpr size_t N = Traits::width;
    constexpr size_t Bytes = M == 0 ? N * sizeof(T) : M;
    static_assert(Bytes <= sizeof(v));
#ifdef Vc_WORKAROUND_XXX_2
    using U = std::conditional_t<
        (std::is_integral_v<T> || Bytes < 4), long long,
        std::conditional_t<(std::is_same_v<T, double> || Bytes < 8), float, T>>;
    const auto vv = builtin_cast<U>(v);
#else   // Vc_WORKAROUND_XXX_2
    const builtin_type_t<T, N> vv = v;
#endif  // Vc_WORKAROUND_XXX_2
    if constexpr(std::is_same_v<F, vector_aligned_tag>) {
        p = __builtin_assume_aligned(p, alignof(builtin_type_t<T, N>));
    } else if constexpr(!std::is_same_v<F, element_aligned_tag>) {
        p = __builtin_assume_aligned(p, F::alignment);
    }
    if constexpr ((Bytes & (Bytes - 1)) != 0) {
        constexpr size_t MoreBytes = next_power_of_2(Bytes);
        alignas(MoreBytes) char tmp[MoreBytes];
        std::memcpy(tmp, &vv, MoreBytes);
        std::memcpy(p, tmp, Bytes);
    } else {
        std::memcpy(p, &vv, Bytes);
    }
}

// }}}
// allbits{{{
template <typename V>
inline constexpr V allbits =
    reinterpret_cast<V>(~intrinsic_type_t<llong, sizeof(V) / sizeof(llong)>());

// }}}
// xor_{{{
template <class T, class Traits = builtin_traits<T>>
Vc_INTRINSIC constexpr T xor_(T a, typename Traits::type b) noexcept
{
    return reinterpret_cast<T>(builtin_cast<unsigned>(a) ^ builtin_cast<unsigned>(b));
}

// }}}
// or_{{{
template <class T, class Traits = builtin_traits<T>>
Vc_INTRINSIC constexpr T or_(T a, typename Traits::type b) noexcept
{
    return reinterpret_cast<T>(builtin_cast<unsigned>(a) | builtin_cast<unsigned>(b));
}

// }}}
// and_{{{
template <class T, class Traits = builtin_traits<T>>
Vc_INTRINSIC constexpr T and_(T a, typename Traits::type b) noexcept
{
    return reinterpret_cast<T>(builtin_cast<unsigned>(a) & builtin_cast<unsigned>(b));
}

// }}}
// andnot_{{{
template <class T, class Traits = builtin_traits<T>>
Vc_INTRINSIC constexpr T andnot_(T a, typename Traits::type b) noexcept
{
    return reinterpret_cast<T>(~builtin_cast<unsigned>(a) & builtin_cast<unsigned>(b));
}

// }}}
// not_{{{
template <class T, class Traits = builtin_traits<T>>
Vc_INTRINSIC constexpr T not_(T a) noexcept
{
    return reinterpret_cast<T>(~builtin_cast<unsigned>(a));
}

// }}}
// concat{{{
template <class T, class Trait = builtin_traits<T>,
          class R = builtin_type_t<typename Trait::value_type, Trait::width * 2>>
constexpr R concat(T a_, T b_) {
#ifdef Vc_WORKAROUND_XXX_1
    using W = std::conditional_t<std::is_floating_point_v<typename Trait::value_type>,
                                 double, long long>;
    constexpr int input_width = sizeof(T) / sizeof(W);
    const auto a = builtin_cast<W>(a_);
    const auto b = builtin_cast<W>(b_);
    using U = builtin_type_t<W, sizeof(R) / sizeof(W)>;
#else
    constexpr int input_width = Trait::width;
    const T &a = a_;
    const T &b = b_;
    using U = R;
#endif
    if constexpr(input_width == 2) {
        return reinterpret_cast<R>(U{a[0], a[1], b[0], b[1]});
    } else if constexpr (input_width == 4) {
        return reinterpret_cast<R>(U{a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]});
    } else if constexpr (input_width == 8) {
        return reinterpret_cast<R>(U{a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], b[0],
                                     b[1], b[2], b[3], b[4], b[5], b[6], b[7]});
    } else if constexpr (input_width == 16) {
        return reinterpret_cast<R>(
            U{a[0],  a[1],  a[2],  a[3],  a[4],  a[5],  a[6],  a[7],  a[8],  a[9], a[10],
              a[11], a[12], a[13], a[14], a[15], b[0],  b[1],  b[2],  b[3],  b[4], b[5],
              b[6],  b[7],  b[8],  b[9],  b[10], b[11], b[12], b[13], b[14], b[15]});
    } else if constexpr (input_width == 32) {
        return reinterpret_cast<R>(
            U{a[0],  a[1],  a[2],  a[3],  a[4],  a[5],  a[6],  a[7],  a[8],  a[9],  a[10],
              a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21],
              a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], b[0],
              b[1],  b[2],  b[3],  b[4],  b[5],  b[6],  b[7],  b[8],  b[9],  b[10], b[11],
              b[12], b[13], b[14], b[15], b[16], b[17], b[18], b[19], b[20], b[21], b[22],
              b[23], b[24], b[25], b[26], b[27], b[28], b[29], b[30], b[31]});
    }
}

// }}}
// zero_extend {{{
template <class T, class Traits = builtin_traits<T>>
Vc_INTRINSIC auto zero_extend(T x)
{
    using value_type = typename Traits::value_type;
    constexpr size_t N = Traits::width;
    struct {
        T x;
        operator builtin_type_t<value_type, N * 2>()
        {
#ifdef Vc_WORKAROUND_XXX_3
            if constexpr (have_avx && Traits::template is<float, 4>) {
                return _mm256_insertf128_ps(__m256(), x, 0);
            } else if constexpr (have_avx && Traits::template is<double, 2>) {
                return _mm256_insertf128_pd(__m256d(), x, 0);
            } else if constexpr (have_avx2 && sizeof(x) == 16) {
                return _mm256_insertf128_si256(__m256i(), x, 0);
            } else if constexpr (have_avx512f && Traits::template is<float, 8>) {
                if constexpr (have_avx512dq) {
                    return _mm512_insertf32x8(__m512(), x, 0);
                } else {
                    return reinterpret_cast<__m512>(
                        _mm512_insertf64x4(__m512d(), reinterpret_cast<__m256d>(x), 0));
                }
            } else if constexpr (have_avx512f && Traits::template is<double, 4>) {
                return _mm512_insertf64x4(__m512d(), x, 0);
            } else if constexpr (have_avx512f && sizeof(x) == 32) {
                return _mm512_inserti64x4(__m512i(), x, 0);
            }
#endif
            return concat(x, T());
        }
        operator builtin_type_t<value_type, N * 4>()
        {
#ifdef Vc_WORKAROUND_XXX_3
            if constexpr (have_avx && Traits::template is<float, 4>) {
#ifdef Vc_WORKAROUND_PR85480
                asm("vmovaps %0, %0" : "+x"(x));
                return _mm512_castps128_ps512(x);
#else
                return _mm512_insertf32x4(__m512(), x, 0);
#endif
            } else if constexpr (have_avx && Traits::template is<double, 2>) {
#ifdef Vc_WORKAROUND_PR85480
                asm("vmovapd %0, %0" : "+x"(x));
                return _mm512_castpd128_pd512(x);
#else
                return _mm512_insertf64x2(__m512d(), x, 0);
#endif
            } else if constexpr (have_avx512f && sizeof(x) == 16) {
#ifdef Vc_WORKAROUND_PR85480
                asm("vmovadq %0, %0" : "+x"(x));
                return _mm512_castsi128_si512(x);
#else
                return _mm512_inserti32x4(__m512i(), x, 0);
#endif
            }
#endif
            return concat(concat(x, T()), builtin_type_t<value_type, N * 2>());
        }
        operator builtin_type_t<value_type, N * 8>()
        {
            return concat(operator builtin_type_t<value_type, N * 4>(),
                          builtin_type_t<value_type, N * 4>());
        }
        operator builtin_type_t<value_type, N * 16>()
        {
            return concat(operator builtin_type_t<value_type, N * 8>(),
                          builtin_type_t<value_type, N * 8>());
        }
    } r{x};
    return r;
}

// }}}
// extract<N, By>{{{
template <int Offset, int SplitBy, class T, class Trait = builtin_traits<T>,
          class R = builtin_type_t<typename Trait::value_type, Trait::width / SplitBy>>
Vc_INTRINSIC constexpr R extract(T x_)
{
#ifdef Vc_WORKAROUND_XXX_1
    using W = std::conditional_t<std::is_floating_point_v<typename Trait::value_type>,
                                 double, long long>;
    constexpr int return_width = sizeof(R) / sizeof(W);
    using U = builtin_type_t<W, return_width>;
    const auto x = builtin_cast<W>(x_);
#else
    constexpr int return_width = Trait::width / SplitBy;
    using U = R;
    const builtin_type_t<typename Trait::value_type, Trait::width> &x =
        x_;  // only needed for T = Storage<value_type, N>
#endif
    constexpr int O = Offset * return_width;
    if constexpr (return_width == 2) {
        return reinterpret_cast<R>(U{x[O + 0], x[O + 1]});
    } else if constexpr (return_width == 4) {
        return reinterpret_cast<R>(U{x[O + 0], x[O + 1], x[O + 2], x[O + 3]});
    } else if constexpr (return_width == 8) {
        return reinterpret_cast<R>(U{x[O + 0], x[O + 1], x[O + 2], x[O + 3], x[O + 4],
                                     x[O + 5], x[O + 6], x[O + 7]});
    } else if constexpr (return_width == 16) {
        return reinterpret_cast<R>(U{x[O + 0], x[O + 1], x[O + 2], x[O + 3], x[O + 4],
                                     x[O + 5], x[O + 6], x[O + 7], x[O + 8], x[O + 9],
                                     x[O + 10], x[O + 11], x[O + 12], x[O + 13],
                                     x[O + 14], x[O + 15]});
    } else if constexpr (return_width == 32) {
        return reinterpret_cast<R>(
            U{x[O + 0],  x[O + 1],  x[O + 2],  x[O + 3],  x[O + 4],  x[O + 5],  x[O + 6],
              x[O + 7],  x[O + 8],  x[O + 9],  x[O + 10], x[O + 11], x[O + 12], x[O + 13],
              x[O + 14], x[O + 15], x[O + 16], x[O + 17], x[O + 18], x[O + 19], x[O + 20],
              x[O + 21], x[O + 22], x[O + 23], x[O + 24], x[O + 25], x[O + 26], x[O + 27],
              x[O + 28], x[O + 29], x[O + 30], x[O + 31]});
    }
}

// }}}
// lo/hi128{{{
template <class T> Vc_INTRINSIC constexpr auto lo128(T x)
{
    return extract<0, sizeof(T) / 16>(x);
}
template <class T> Vc_INTRINSIC constexpr auto hi128(T x)
{
    static_assert(sizeof(x) == 32);
    return extract<1, 2>(x);
}

// }}}
// lo/hi256{{{
template <class T> Vc_INTRINSIC constexpr auto lo256(T x)
{
    static_assert(sizeof(x) == 64);
    return extract<0, 2>(x);
}
template <class T> Vc_INTRINSIC constexpr auto hi256(T x)
{
    static_assert(sizeof(x) == 64);
    return extract<1, 2>(x);
}

// }}}
// intrin_cast{{{
template <class To, class From> Vc_INTRINSIC constexpr To intrin_cast(From v)
{
    static_assert(is_builtin_vector_v<From> && is_builtin_vector_v<To>);
    if constexpr (sizeof(To) == sizeof(From)) {
        return reinterpret_cast<To>(v);
    } else if constexpr (sizeof(From) > sizeof(To)) {
        return reinterpret_cast<const To &>(v);
    } else if constexpr (have_avx && sizeof(From) == 16 && sizeof(To) == 32) {
        return reinterpret_cast<To>(_mm256_castps128_ps256(
            reinterpret_cast<intrinsic_type_t<float, sizeof(From) / sizeof(float)>>(v)));
    } else if constexpr (have_avx512f && sizeof(From) == 16 && sizeof(To) == 64) {
        return reinterpret_cast<To>(_mm512_castps128_ps512(
            reinterpret_cast<intrinsic_type_t<float, sizeof(From) / sizeof(float)>>(v)));
    } else if constexpr (have_avx512f && sizeof(From) == 32 && sizeof(To) == 64) {
        return reinterpret_cast<To>(_mm512_castps256_ps512(
            reinterpret_cast<intrinsic_type_t<float, sizeof(From) / sizeof(float)>>(v)));
    } else {
        assert_unreachable<To>();
    }
}

// }}}
// auto_cast{{{
template <class T> struct auto_cast_t {
    static_assert(is_builtin_vector_v<T>);
    const T x;
    template <class U> Vc_INTRINSIC constexpr operator U() const
    {
        return intrin_cast<U>(x);
    }
};
template <class T> Vc_INTRINSIC constexpr auto_cast_t<T> auto_cast(const T &x)
{
    return {x};
}
template <class T, size_t N>
Vc_INTRINSIC constexpr auto_cast_t<typename Storage<T, N>::register_type> auto_cast(
    const Storage<T, N> &x)
{
    return {x.d};
}

// }}}
// to_bitset{{{
Vc_INTRINSIC constexpr std::bitset<1> to_bitset(bool x) { return unsigned(x); }

template <class T, class = std::enable_if_t<detail::is_bitmask_v<T> && have_avx512f>>
Vc_INTRINSIC constexpr std::bitset<8 * sizeof(T)> to_bitset(T x)
{
    if constexpr (std::is_integral_v<T>) {
        return x;
    } else {
        return x.d;
    }
}

template <class T, class Trait = builtin_traits<T>>
Vc_INTRINSIC std::bitset<Trait::width> to_bitset(T x)
{
    constexpr bool is_sse = have_sse && sizeof(T) == 16;
    constexpr bool is_avx = have_avx && sizeof(T) == 32;
    constexpr bool is_neon128 = have_neon && sizeof(T) == 16;
    constexpr int w = sizeof(typename Trait::value_type);
    const auto intrin = detail::to_intrin(x);
    constexpr auto zero = decltype(intrin)();
    detail::unused(zero);

    if constexpr (is_neon128 && w == 1) {
        x &= T{0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80,
               0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80};
        return builtin_cast<ushort>(
            vpaddq_s8(vpaddq_s8(vpaddq_s8(x, zero), zero), zero))[0];
    } else if constexpr (is_neon128 && w == 2) {
        x &= T{0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80};
        return vpaddq_s16(vpaddq_s16(vpaddq_s16(x, zero), zero), zero)[0];
    } else if constexpr (is_neon128 && w == 4) {
        x &= T{0x1, 0x2, 0x4, 0x8};
        return vpaddq_s32(vpaddq_s32(x, zero), zero)[0];
    } else if constexpr (is_neon128 && w == 8) {
        x &= T{0x1, 0x2};
        return x[0] | x[1];
    } else if constexpr (is_sse && w == 1) {
        return _mm_movemask_epi8(intrin);
    } else if constexpr (is_sse && w == 2) {
        if constexpr (detail::have_avx512bw_vl) {
            return _mm_cmplt_epi16_mask(intrin, zero);
        } else {
            return _mm_movemask_epi8(_mm_packs_epi16(intrin, zero));
        }
    } else if constexpr (is_sse && w == 4) {
        if constexpr (detail::have_avx512vl && std::is_integral_v<T>) {
            return _mm_cmplt_epi32_mask(intrin, zero);
        } else {
            return _mm_movemask_ps(builtin_cast<float>(x));
        }
    } else if constexpr (is_sse && w == 8) {
        if constexpr (detail::have_avx512vl && std::is_integral_v<T>) {
            return _mm_cmplt_epi64_mask(intrin, zero);
        } else {
            return _mm_movemask_pd(builtin_cast<double>(x));
        }
    } else if constexpr (is_avx && w == 1) {
        return _mm256_movemask_epi8(intrin);
    } else if constexpr (is_avx && w == 2) {
        if constexpr (detail::have_avx512bw_vl) {
            return _mm256_cmplt_epi16_mask(intrin, zero);
        } else {
            return _mm_movemask_epi8(_mm_packs_epi16(detail::extract<0, 2>(intrin),
                                                     detail::extract<1, 2>(intrin)));
        }
    } else if constexpr (is_avx && w == 4) {
        if constexpr (detail::have_avx512vl && std::is_integral_v<T>) {
            return _mm256_cmplt_epi32_mask(intrin, zero);
        } else {
            return _mm256_movemask_ps(builtin_cast<float>(x));
        }
    } else if constexpr (is_avx && w == 8) {
        if constexpr (detail::have_avx512vl && std::is_integral_v<T>) {
            return _mm256_cmplt_epi64_mask(intrin, zero);
        } else {
            return _mm256_movemask_pd(builtin_cast<double>(x));
        }
    } else {
        assert_unreachable<T>();
    }
}

// }}}
// testz{{{
template <class T, class Traits = builtin_traits<T>>
Vc_INTRINSIC Vc_CONST int testz(T a, T b)
{
    if constexpr (detail::have_avx) {
        if constexpr (sizeof(T) == 32 && Traits::template is<float>) {
            return _mm256_testz_ps(a, b);
        } else if constexpr (sizeof(T) == 32 && Traits::template is<double>) {
            return _mm256_testz_pd(a, b);
        } else if constexpr (sizeof(T) == 32) {
            return _mm256_testz_si256(builtin_cast<llong>(a), builtin_cast<llong>(b));
        } else if constexpr(Traits::template is<float, 4>) {
            return _mm_testz_ps(a, b);
        } else if constexpr(Traits::template is<double, 2>) {
            return _mm_testz_pd(a, b);
        } else {
            static_assert(sizeof(T) == 16);
            return _mm_testz_si128(builtin_cast<llong>(a), builtin_cast<llong>(b));
        }
    } else if constexpr (detail::have_sse4_1) {
        return _mm_testz_si128(builtin_cast<llong>(a), builtin_cast<llong>(b));
    } else if constexpr (have_sse && Traits::template is<float, 4>) {
        return _mm_movemask_ps(and_(a, b)) == 0;
    } else if constexpr (have_sse2 && Traits::template is<double, 2>) {
        return _mm_movemask_pd(and_(a, b)) == 0;
    } else if constexpr (have_sse2) {
        return _mm_movemask_epi8(a & b) == 0;
    } else {
        assert_unreachable<T>();
    }
}

// }}}
// testnzc{{{
template <class T, class Traits = builtin_traits<T>>
Vc_INTRINSIC Vc_CONST int testnzc(T a, T b)
{
    if constexpr (detail::have_avx) {
        if constexpr (sizeof(T) == 32 && Traits::template is<float>) {
            return _mm256_testnzc_ps(a, b);
        } else if constexpr (sizeof(T) == 32 && Traits::template is<double>) {
            return _mm256_testnzc_pd(a, b);
        } else if constexpr (sizeof(T) == 32) {
            return _mm256_testnzc_si256(builtin_cast<llong>(a), builtin_cast<llong>(b));
        } else if constexpr(Traits::template is<float, 4>) {
            return _mm_testnzc_ps(a, b);
        } else if constexpr(Traits::template is<double, 2>) {
            return _mm_testnzc_pd(a, b);
        } else {
            static_assert(sizeof(T) == 16);
            return _mm_testnzc_si128(builtin_cast<llong>(a), builtin_cast<llong>(b));
        }
    } else if constexpr (detail::have_sse4_1) {
        return _mm_testnzc_si128(builtin_cast<llong>(a), builtin_cast<llong>(b));
    } else if constexpr (have_sse && Traits::template is<float, 4>) {
        return _mm_movemask_ps(and_(a, b)) == 0 && _mm_movemask_ps(andnot_(a, b)) == 0;
    } else if constexpr (have_sse2 && Traits::template is<double, 2>) {
        return _mm_movemask_pd(and_(a, b)) == 0 && _mm_movemask_pd(andnot_(a, b)) == 0;
    } else if constexpr (have_sse2) {
        return _mm_movemask_epi8(and_(a, b)) == 0 &&
               _mm_movemask_epi8(andnot_(a, b)) == 0;
    } else {
        assert_unreachable<T>();
    }
}

// }}}
// movemask{{{
template <class T, class Traits = builtin_traits<T>>
Vc_INTRINSIC Vc_CONST int movemask(T a)
{
    if constexpr (have_sse && Traits::template is<float, 4>) {
        return _mm_movemask_ps(a);
    } else if constexpr (have_avx && Traits::template is<float, 8>) {
        return _mm256_movemask_ps(a);
    } else if constexpr (have_sse2 && Traits::template is<double, 2>) {
        return _mm_movemask_pd(a);
    } else if constexpr (have_avx && Traits::template is<double, 4>) {
        return _mm256_movemask_pd(a);
    } else if constexpr (have_sse2 && sizeof(T) == 16) {
        return _mm_movemask_epi8(a);
    } else if constexpr (have_avx2 && sizeof(T) == 32) {
        return _mm256_movemask_epi8(a);
    } else {
        assert_unreachable<T>();
    }
}

template <class T, class Traits = builtin_traits<T>>
Vc_INTRINSIC Vc_CONST int movemask_epi16(T a)
{
    static_assert(std::is_integral_v<typename Traits::value_type>);
    if constexpr(have_avx512bw_vl && sizeof(T) == 16) {
        return _mm_cmp_epi16_mask(a, __m128i(), _MM_CMPINT_NE);
    } else if constexpr(have_avx512bw_vl && sizeof(T) == 32) {
        return _mm256_cmp_epi16_mask(a, __m256i(), _MM_CMPINT_NE);
    } else if constexpr(sizeof(T) == 32) {
        return _mm_movemask_epi8(_mm_packs_epi16(lo128(a), hi128(a)));
    } else {
        static_assert(sizeof(T) == 16);
        return _mm_movemask_epi8(_mm_packs_epi16(a, __m128i()));
    }
}

// }}}
// {double,float}_const {{{
template <int exponent> constexpr double double_2_pow()
{
    if constexpr (exponent < 0) {
        return 1. / double_2_pow<-exponent>();
    } else if constexpr (exponent < std::numeric_limits<unsigned long long>::digits) {
        return 1ull << exponent;
    } else {
        return ((~0ull >> 1) + 1) * 2. *
               double_2_pow<exponent - std::numeric_limits<unsigned long long>::digits>();
    }
}

template <int sign, unsigned long long mantissa, int exponent>
constexpr double double_const = (static_cast<double>((mantissa & 0x000fffffffffffffull) |
                                                     0x0010000000000000ull) /
                                 0x0010000000000000ull) *
                                double_2_pow<exponent>() * sign;
template <int sign, unsigned int mantissa, int exponent>
constexpr float float_const = (float((mantissa & 0x007fffffu) | 0x00800000u) /
                               0x00800000u) *
                              float(double_2_pow<exponent>()) * sign;
// }}}
// trig constants {{{
template <class T> struct trig;
template <> struct trig<float> {
    static inline constexpr float pi_4      = float_const< 1, 0x490FDB, -1>;
    static inline constexpr float pi_4_hi   = float_const< 1, 0x491000, -1>;
    static inline constexpr float pi_4_rem1 = float_const<-1, 0x157000, -19>;
    static inline constexpr float pi_4_rem2 = float_const<-1, 0x6F4B9F, -32>;
    static inline constexpr float _1_16 = 0.0625f;
    static inline constexpr float _16 = 16.f;
    static inline constexpr float cos_c0 = 4.166664568298827e-2f;  // ~ 1/4!
    static inline constexpr float cos_c1 = -1.388731625493765e-3f; // ~-1/6!
    static inline constexpr float cos_c2 = 2.443315711809948e-5f;  // ~ 1/8!
    static inline constexpr float sin_c0 = -1.6666654611e-1f; // ~-1/3!
    static inline constexpr float sin_c1 = 8.3321608736e-3f;  // ~ 1/5!
    static inline constexpr float sin_c2 = -1.9515295891e-4f; // ~-1/7!
    static inline constexpr float loss_threshold = 8192.f; // loss threshold
    static inline constexpr float _4_pi = float_const< 1, 0x22F983, 0>; // 1.27323949337005615234375 = 4/π
    static inline constexpr float pi_2 = float_const< 1, 0x490FDB, 0>; // π/2
    static inline constexpr float pi = float_const< 1, 0x490FDB, 1>; // π
    static inline constexpr float atan_p0 = 8.05374449538e-2f; // atan P coefficients
    static inline constexpr float atan_p1 = 1.38776856032e-1f; // atan P coefficients
    static inline constexpr float atan_p2 = 1.99777106478e-1f; // atan P coefficients
    static inline constexpr float atan_p3 = 3.33329491539e-1f; // atan P coefficients
    static inline constexpr float atan_threshold_hi = 2.414213562373095f; // tan( 3/8 π )
    static inline constexpr float atan_threshold_lo = 0.414213562373095f; // tan( 1/8 π ) lower threshold for special casing in atan
    static inline constexpr float pi_2_rem = float_const<-1, 0x3BBD2E, -25>; // remainder of pi/2
    static inline constexpr float small_asin_input = 1.e-4f; // small asin input threshold
    static inline constexpr float large_asin_input = 0.f; // padding (for alignment with double)
    static inline constexpr float asin_c0_0 = 4.2163199048e-2f; // asinCoeff0
    static inline constexpr float asin_c0_1 = 2.4181311049e-2f; // asinCoeff0
    static inline constexpr float asin_c0_2 = 4.5470025998e-2f; // asinCoeff0
    static inline constexpr float asin_c0_3 = 7.4953002686e-2f; // asinCoeff0
    static inline constexpr float asin_c0_4 = 1.6666752422e-1f; // asinCoeff0
};

template <> struct trig<double> {
    static inline constexpr double pi_4      = double_const< 1, 0x921fb54442d18, -1>; // π/4
    static inline constexpr double pi_4_hi   = double_const< 1, 0x921fb40000000, -1>; // π/4 - 30bits precision
    static inline constexpr double pi_4_rem1 = double_const< 1, 0x4442d00000000, -25>; // π/4 remainder1 - 32bits precision
    static inline constexpr double pi_4_rem2 = double_const< 1, 0x8469898cc5170, -49>; // π/4 remainder2
    static inline constexpr double _1_16 = 0.0625;
    static inline constexpr double _16 = 16.;
    static inline constexpr double cos_c0  = double_const< 1, 0x555555555554b, -5 >; // ~ 1/4!
    static inline constexpr double cos_c1  = double_const<-1, 0x6c16c16c14f91, -10>; // ~-1/6!
    static inline constexpr double cos_c2  = double_const< 1, 0xa01a019c844f5, -16>; // ~ 1/8!
    static inline constexpr double cos_c3  = double_const<-1, 0x27e4f7eac4bc6, -22>; // ~-1/10!
    static inline constexpr double cos_c4  = double_const< 1, 0x1ee9d7b4e3f05, -29>; // ~ 1/12!
    static inline constexpr double cos_c5  = double_const<-1, 0x8fa49a0861a9b, -37>; // ~-1/14!
    static inline constexpr double sin_c0  = double_const<-1, 0x5555555555548, -3 >; // ~-1/3!
    static inline constexpr double sin_c1  = double_const< 1, 0x111111110f7d0, -7 >; // ~ 1/5!
    static inline constexpr double sin_c2  = double_const<-1, 0xa01a019bfdf03, -13>; // ~-1/7!
    static inline constexpr double sin_c3  = double_const< 1, 0x71de3567d48a1, -19>; // ~ 1/9!
    static inline constexpr double sin_c4  = double_const<-1, 0xae5e5a9291f5d, -26>; // ~-1/11!
    static inline constexpr double sin_c5  = double_const< 1, 0x5d8fd1fd19ccd, -33>; // ~ 1/13!
    static inline constexpr double _4_pi    = double_const< 1, 0x8BE60DB939105, 0 >; // 4/π
    static inline constexpr double pi_2    = double_const< 1, 0x921fb54442d18, 0 >; // π/2
    static inline constexpr double pi      = double_const< 1, 0x921fb54442d18, 1 >; // π
    static inline constexpr double atan_p0 = double_const<-1, 0xc007fa1f72594, -1>; // atan P coefficients
    static inline constexpr double atan_p1 = double_const<-1, 0x028545b6b807a, 4 >; // atan P coefficients
    static inline constexpr double atan_p2 = double_const<-1, 0x2c08c36880273, 6 >; // atan P coefficients
    static inline constexpr double atan_p3 = double_const<-1, 0xeb8bf2d05ba25, 6 >; // atan P coefficients
    static inline constexpr double atan_p4 = double_const<-1, 0x03669fd28ec8e, 6 >; // atan P coefficients
    static inline constexpr double atan_q0 = double_const< 1, 0x8dbc45b14603c, 4 >; // atan Q coefficients
    static inline constexpr double atan_q1 = double_const< 1, 0x4a0dd43b8fa25, 7 >; // atan Q coefficients
    static inline constexpr double atan_q2 = double_const< 1, 0xb0e18d2e2be3b, 8 >; // atan Q coefficients
    static inline constexpr double atan_q3 = double_const< 1, 0xe563f13b049ea, 8 >; // atan Q coefficients
    static inline constexpr double atan_q4 = double_const< 1, 0x8519efbbd62ec, 7 >; // atan Q coefficients
    static inline constexpr double atan_threshold_hi = double_const< 1, 0x3504f333f9de6, 1>; // tan( 3/8 π )
    static inline constexpr double atan_threshold_lo = 0.66;                                 // lower threshold for special casing in atan
    static inline constexpr double pi_2_rem = double_const< 1, 0x1A62633145C07, -54>; // remainder of pi/2
    static inline constexpr double small_asin_input = 1.e-8; // small asin input threshold
    static inline constexpr double large_asin_input = 0.625; // large asin input threshold
    static inline constexpr double asin_c0_0 = double_const< 1, 0x84fc3988e9f08, -9>; // asinCoeff0
    static inline constexpr double asin_c0_1 = double_const<-1, 0x2079259f9290f, -1>; // asinCoeff0
    static inline constexpr double asin_c0_2 = double_const< 1, 0xbdff5baf33e6a, 2 >; // asinCoeff0
    static inline constexpr double asin_c0_3 = double_const<-1, 0x991aaac01ab68, 4 >; // asinCoeff0
    static inline constexpr double asin_c0_4 = double_const< 1, 0xc896240f3081d, 4 >; // asinCoeff0
    static inline constexpr double asin_c1_0 = double_const<-1, 0x5f2a2b6bf5d8c, 4 >; // asinCoeff1
    static inline constexpr double asin_c1_1 = double_const< 1, 0x26219af6a7f42, 7 >; // asinCoeff1
    static inline constexpr double asin_c1_2 = double_const<-1, 0x7fe08959063ee, 8 >; // asinCoeff1
    static inline constexpr double asin_c1_3 = double_const< 1, 0x56709b0b644be, 8 >; // asinCoeff1
    static inline constexpr double asin_c2_0 = double_const< 1, 0x16b9b0bd48ad3, -8>; // asinCoeff2
    static inline constexpr double asin_c2_1 = double_const<-1, 0x34341333e5c16, -1>; // asinCoeff2
    static inline constexpr double asin_c2_2 = double_const< 1, 0x5c74b178a2dd9, 2 >; // asinCoeff2
    static inline constexpr double asin_c2_3 = double_const<-1, 0x04331de27907b, 4 >; // asinCoeff2
    static inline constexpr double asin_c2_4 = double_const< 1, 0x39007da779259, 4 >; // asinCoeff2
    static inline constexpr double asin_c2_5 = double_const<-1, 0x0656c06ceafd5, 3 >; // asinCoeff2
    static inline constexpr double asin_c3_0 = double_const<-1, 0xd7b590b5e0eab, 3 >; // asinCoeff3
    static inline constexpr double asin_c3_1 = double_const< 1, 0x19fc025fe9054, 6 >; // asinCoeff3
    static inline constexpr double asin_c3_2 = double_const<-1, 0x265bb6d3576d7, 7 >; // asinCoeff3
    static inline constexpr double asin_c3_3 = double_const< 1, 0x1705684ffbf9d, 7 >; // asinCoeff3
    static inline constexpr double asin_c3_4 = double_const<-1, 0x898220a3607ac, 5 >; // asinCoeff3
};

// }}}
#if defined Vc_HAVE_SSE_ABI
// bool_storage_member_type{{{
#ifdef Vc_HAVE_AVX512F
template <> struct bool_storage_member_type< 2> { using type = __mmask8 ; };
template <> struct bool_storage_member_type< 4> { using type = __mmask8 ; };
template <> struct bool_storage_member_type< 8> { using type = __mmask8 ; };
template <> struct bool_storage_member_type<16> { using type = __mmask16; };
template <> struct bool_storage_member_type<32> { using type = __mmask32; };
template <> struct bool_storage_member_type<64> { using type = __mmask64; };
#endif  // Vc_HAVE_AVX512F

// }}}
// intrinsic_type{{{
// the following excludes bool via is_vectorizable
template <class T>
using void_if_integral_t = std::void_t<std::enable_if_t<
    detail::all<std::is_integral<T>, detail::is_vectorizable<T>>::value>>;
#if defined Vc_HAVE_AVX512F
template <> struct intrinsic_type<double, 64, void> { using type = __m512d; };
template <> struct intrinsic_type< float, 64, void> { using type = __m512; };
template <typename T> struct intrinsic_type<T, 64, void_if_integral_t<T>> { using type = __m512i; };
#endif  // Vc_HAVE_AVX512F

#if defined Vc_HAVE_AVX
template <> struct intrinsic_type<double, 32, void> { using type = __m256d; };
template <> struct intrinsic_type< float, 32, void> { using type = __m256; };
template <typename T> struct intrinsic_type<T, 32, void_if_integral_t<T>> { using type = __m256i; };
#endif  // Vc_HAVE_AVX

#if defined Vc_HAVE_SSE
template <> struct intrinsic_type< float, 16, void> { using type = __m128; };
template <> struct intrinsic_type< float,  8, void> { using type = __m128; };
template <> struct intrinsic_type< float,  4, void> { using type = __m128; };
#endif  // Vc_HAVE_SSE
#if defined Vc_HAVE_SSE2
template <> struct intrinsic_type<double, 16, void> { using type = __m128d; };
template <> struct intrinsic_type<double,  8, void> { using type = __m128d; };
template <typename T> struct intrinsic_type<T, 16, void_if_integral_t<T>> { using type = __m128i; };
template <typename T> struct intrinsic_type<T,  8, void_if_integral_t<T>> { using type = __m128i; };
template <typename T> struct intrinsic_type<T,  4, void_if_integral_t<T>> { using type = __m128i; };
template <typename T> struct intrinsic_type<T,  2, void_if_integral_t<T>> { using type = __m128i; };
template <typename T> struct intrinsic_type<T,  1, void_if_integral_t<T>> { using type = __m128i; };
#endif  // Vc_HAVE_SSE2

// }}}
// is_intrinsic{{{
template <> struct is_intrinsic<__m128> : public std::true_type {};
#ifdef Vc_HAVE_SSE2
template <> struct is_intrinsic<__m128d> : public std::true_type {};
template <> struct is_intrinsic<__m128i> : public std::true_type {};
#endif  // Vc_HAVE_SSE2
#ifdef Vc_HAVE_AVX
template <> struct is_intrinsic<__m256 > : public std::true_type {};
template <> struct is_intrinsic<__m256d> : public std::true_type {};
template <> struct is_intrinsic<__m256i> : public std::true_type {};
#endif  // Vc_HAVE_AVX
#ifdef Vc_HAVE_AVX512F
template <> struct is_intrinsic<__m512 > : public std::true_type {};
template <> struct is_intrinsic<__m512d> : public std::true_type {};
template <> struct is_intrinsic<__m512i> : public std::true_type {};
#endif  // Vc_HAVE_AVX512F

// }}}
// (sse|avx|avx512)_(simd|mask)_member_type{{{
template <class T> using sse_simd_member_type = storage16_t<T>;
template <class T> using sse_mask_member_type = storage16_t<T>;

template <class T> using avx_simd_member_type = storage32_t<T>;
template <class T> using avx_mask_member_type = storage32_t<T>;

template <class T> using avx512_simd_member_type = storage64_t<T>;
template <class T> using avx512_mask_member_type = Storage<bool, 64 / sizeof(T)>;
template <size_t N> using avx512_mask_member_type_n = Storage<bool, N>;

//}}}
// x_ aliases {{{
#ifdef Vc_HAVE_SSE
using x_f32 = Storage< float,  4>;
#ifdef Vc_HAVE_SSE2
using x_f64 = Storage<double,  2>;
using x_i08 = Storage< schar, 16>;
using x_u08 = Storage< uchar, 16>;
using x_i16 = Storage< short,  8>;
using x_u16 = Storage<ushort,  8>;
using x_i32 = Storage<   int,  4>;
using x_u32 = Storage<  uint,  4>;
using x_i64 = Storage< llong,  2>;
using x_u64 = Storage<ullong,  2>;
using x_long = Storage<long,   16 / sizeof(long)>;
using x_ulong = Storage<ulong, 16 / sizeof(ulong)>;
using x_long_equiv = Storage<equal_int_type_t<long>, 16 / sizeof(long)>;
using x_ulong_equiv = Storage<equal_int_type_t<ulong>, 16 / sizeof(ulong)>;
using x_chr = Storage<    char, 16>;
using x_c16 = Storage<char16_t,  8>;
using x_c32 = Storage<char32_t,  4>;
using x_wch = Storage< wchar_t, 16 / sizeof(wchar_t)>;
#endif  // Vc_HAVE_SSE2
#endif  // Vc_HAVE_SSE

//}}}
// y_ aliases {{{
using y_f32 = Storage< float,  8>;
using y_f64 = Storage<double,  4>;
using y_i08 = Storage< schar, 32>;
using y_u08 = Storage< uchar, 32>;
using y_i16 = Storage< short, 16>;
using y_u16 = Storage<ushort, 16>;
using y_i32 = Storage<   int,  8>;
using y_u32 = Storage<  uint,  8>;
using y_i64 = Storage< llong,  4>;
using y_u64 = Storage<ullong,  4>;
using y_long = Storage<long,   32 / sizeof(long)>;
using y_ulong = Storage<ulong, 32 / sizeof(ulong)>;
using y_long_equiv = Storage<equal_int_type_t<long>, 32 / sizeof(long)>;
using y_ulong_equiv = Storage<equal_int_type_t<ulong>, 32 / sizeof(ulong)>;
using y_chr = Storage<    char, 32>;
using y_c16 = Storage<char16_t, 16>;
using y_c32 = Storage<char32_t,  8>;
using y_wch = Storage< wchar_t, 32 / sizeof(wchar_t)>;

//}}}
// z_ aliases {{{
using z_f32 = Storage< float, 16>;
using z_f64 = Storage<double,  8>;
using z_i32 = Storage<   int, 16>;
using z_u32 = Storage<  uint, 16>;
using z_i64 = Storage< llong,  8>;
using z_u64 = Storage<ullong,  8>;
using z_long = Storage<long,   64 / sizeof(long)>;
using z_ulong = Storage<ulong, 64 / sizeof(ulong)>;
using z_i08 = Storage< schar, 64>;
using z_u08 = Storage< uchar, 64>;
using z_i16 = Storage< short, 32>;
using z_u16 = Storage<ushort, 32>;
using z_long_equiv = Storage<equal_int_type_t<long>, 64 / sizeof(long)>;
using z_ulong_equiv = Storage<equal_int_type_t<ulong>, 64 / sizeof(ulong)>;
using z_chr = Storage<    char, 64>;
using z_c16 = Storage<char16_t, 32>;
using z_c32 = Storage<char32_t, 16>;
using z_wch = Storage< wchar_t, 64 / sizeof(wchar_t)>;

//}}}
#endif  // Vc_HAVE_SSE_ABI
// Storage<bool>{{{1
template <size_t Width>
struct Storage<bool, Width, std::void_t<typename bool_storage_member_type<Width>::type>> {
    using register_type = typename bool_storage_member_type<Width>::type;
    using value_type = bool;
    static constexpr size_t width = Width;
    [[deprecated]] static constexpr size_t size() { return Width; }

    Vc_INTRINSIC constexpr Storage() = default;
    Vc_INTRINSIC constexpr Storage(register_type k) : d(k){};

    Vc_INTRINSIC Vc_PURE operator const register_type &() const { return d; }
    Vc_INTRINSIC Vc_PURE operator register_type &() { return d; }

    Vc_INTRINSIC register_type intrin() const { return d; }

    Vc_INTRINSIC Vc_PURE value_type operator[](size_t i) const
    {
        return d & (register_type(1) << i);
    }
    Vc_INTRINSIC void set(size_t i, value_type x)
    {
        if (x) {
            d |= (register_type(1) << i);
        } else {
            d &= ~(register_type(1) << i);
        }
    }

    register_type d;
};

// StorageBase{{{1
template <class T, size_t Width, class RegisterType = builtin_type_t<T, Width>,
          bool = std::disjunction_v<
              std::is_same<builtin_type_t<T, Width>, intrinsic_type_t<T, Width>>,
              std::is_same<RegisterType, intrinsic_type_t<T, Width>>>>
struct StorageBase;

template <class T, size_t Width, class RegisterType>
struct StorageBase<T, Width, RegisterType, true> {
    RegisterType d;
    Vc_INTRINSIC constexpr StorageBase() = default;
    Vc_INTRINSIC constexpr StorageBase(builtin_type_t<T, Width> x)
        : d(reinterpret_cast<RegisterType>(x))
    {
    }
};

template <class T, size_t Width, class RegisterType>
struct StorageBase<T, Width, RegisterType, false> {
    using intrin_type = intrinsic_type_t<T, Width>;
    RegisterType d;

    Vc_INTRINSIC constexpr StorageBase() = default;
    Vc_INTRINSIC constexpr StorageBase(builtin_type_t<T, Width> x)
        : d(reinterpret_cast<RegisterType>(x))
    {
    }
    Vc_INTRINSIC constexpr StorageBase(intrin_type x)
        : d(reinterpret_cast<RegisterType>(x))
    {
    }

    Vc_INTRINSIC constexpr operator intrin_type() const
    {
        return reinterpret_cast<intrin_type>(d);
    }
};

// StorageEquiv {{{1
template <typename T, size_t Width, bool = detail::has_same_value_representation_v<T>>
struct StorageEquiv : StorageBase<T, Width> {
    using StorageBase<T, Width>::d;
    Vc_INTRINSIC constexpr StorageEquiv() = default;
    template <class U, class = decltype(StorageBase<T, Width>(std::declval<U>()))>
    Vc_INTRINSIC constexpr StorageEquiv(U &&x) : StorageBase<T, Width>(std::forward<U>(x))
    {
    }
    // I want to use ctor inheritance, but it breaks always_inline. Having a function that
    // does a single movaps is stupid.
    //using StorageBase<T, Width>::StorageBase;
};

// This base class allows conversion to & from
// * builtin_type_t<equal_int_type_t<T>, Width>, and
// * Storage<equal_int_type_t<T>, Width>
// E.g. Storage<long, 4> is convertible to & from
// * builtin_type_t<long long, 4>, and
// * Storage<long long, 4>
// on LP64
// * builtin_type_t<int, 4>, and
// * Storage<int, 4>
// on ILP32, and LLP64
template <class T, size_t Width>
struct StorageEquiv<T, Width, true>
    : StorageBase<equal_int_type_t<T>, Width, builtin_type_t<T, Width>> {
    using Base = StorageBase<equal_int_type_t<T>, Width, builtin_type_t<T, Width>>;
    using Base::d;
    template <class U,
              class = decltype(StorageBase<equal_int_type_t<T>, Width,
                                           builtin_type_t<T, Width>>(std::declval<U>()))>
    Vc_INTRINSIC constexpr StorageEquiv(U &&x)
        : StorageBase<equal_int_type_t<T>, Width, builtin_type_t<T, Width>>(
              std::forward<U>(x))
    {
    }
    // I want to use ctor inheritance, but it breaks always_inline. Having a function that
    // does a single movaps is stupid.
    //using Base::StorageBase;

    Vc_INTRINSIC constexpr StorageEquiv() = default;

    // convertible from intrin_type, builtin_type_t<equal_int_type_t<T>, Width> and
    // builtin_type_t<T, Width>, and Storage<equal_int_type_t<T>, Width>
    Vc_INTRINSIC constexpr StorageEquiv(builtin_type_t<T, Width> x)
        : Base(reinterpret_cast<builtin_type_t<equal_int_type_t<T>, Width>>(x))
    {
    }
    Vc_INTRINSIC constexpr StorageEquiv(Storage<equal_int_type_t<T>, Width> x)
        : Base(x.d)
    {
    }

    // convertible to intrin_type, builtin_type_t<equal_int_type_t<T>, Width> and
    // builtin_type_t<T, Width> (in Storage), and Storage<equal_int_type_t<T>, Width>
    //
    // intrin_type<T> is handled by StorageBase
    // builtin_type_t<T> is handled by Storage
    // builtin_type_t<equal_int_type_t<T>> is handled in StorageEquiv, i.e. here:
    Vc_INTRINSIC constexpr operator builtin_type_t<equal_int_type_t<T>, Width>() const
    {
        return reinterpret_cast<builtin_type_t<equal_int_type_t<T>, Width>>(d);
    }
    Vc_INTRINSIC constexpr operator Storage<equal_int_type_t<T>, Width>() const
    {
        return reinterpret_cast<builtin_type_t<equal_int_type_t<T>, Width>>(d);
    }

    Vc_INTRINSIC constexpr Storage<equal_int_type_t<T>, Width> equiv() const
    {
        return reinterpret_cast<builtin_type_t<equal_int_type_t<T>, Width>>(d);
    }
};

// StorageBroadcast{{{1
template <class T, size_t Width> struct StorageBroadcast;
template <class T> struct StorageBroadcast<T, 2> {
    Vc_INTRINSIC static constexpr Storage<T, 2> broadcast(T x)
    {
        return builtin_type_t<T, 2>{x, x};
    }
};
template <class T> struct StorageBroadcast<T, 4> {
    Vc_INTRINSIC static constexpr Storage<T, 4> broadcast(T x)
    {
        return builtin_type_t<T, 4>{x, x, x, x};
    }
};
template <class T> struct StorageBroadcast<T, 8> {
    Vc_INTRINSIC static constexpr Storage<T, 8> broadcast(T x)
    {
        return builtin_type_t<T, 8>{x, x, x, x, x, x, x, x};
    }
};
template <class T> struct StorageBroadcast<T, 16> {
    Vc_INTRINSIC static constexpr Storage<T, 16> broadcast(T x)
    {
        return builtin_type_t<T, 16>{x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x};
    }
};
template <class T> struct StorageBroadcast<T, 32> {
    Vc_INTRINSIC static constexpr Storage<T, 32> broadcast(T x)
    {
        return builtin_type_t<T, 32>{x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
                                     x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x};
    }
};
template <class T> struct StorageBroadcast<T, 64> {
    Vc_INTRINSIC static constexpr Storage<T, 64> broadcast(T x)
    {
        return builtin_type_t<T, 64>{x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
                                     x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
                                     x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x,
                                     x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x};
    }
};

// Storage{{{1
template <typename T, size_t Width>
struct Storage<T, Width,
               std::void_t<builtin_type_t<T, Width>, intrinsic_type_t<T, Width>>>
    : StorageEquiv<T, Width>, StorageBroadcast<T, Width> {
    static_assert(is_vectorizable_v<T>);
    static_assert(Width >= 2);  // 1 doesn't make sense, use T directly then
    using register_type = builtin_type_t<T, Width>;
    using value_type = T;
    static constexpr size_t width = Width;
    [[deprecated("use width instead")]] static constexpr size_t size() { return Width; }

    Vc_INTRINSIC constexpr Storage() = default;
    template <class U, class = decltype(StorageEquiv<T, Width>(std::declval<U>()))>
    Vc_INTRINSIC constexpr Storage(U &&x) : StorageEquiv<T, Width>(std::forward<U>(x))
    {
    }
    // I want to use ctor inheritance, but it breaks always_inline. Having a function that
    // does a single movaps is stupid.
    //using StorageEquiv<T, Width>::StorageEquiv;
    using StorageEquiv<T, Width>::d;

    template <class... As,
              class = std::enable_if_t<((std::is_same_v<simd_abi::scalar, As> && ...) &&
                                        sizeof...(As) <= Width)>>
    Vc_INTRINSIC constexpr operator simd_tuple<T, As...>() const
    {
        const auto &dd = d;  // workaround for GCC7 ICE
        return detail::generate_from_n_evaluations<sizeof...(As), simd_tuple<T, As...>>(
            [&](auto i) { return dd[int(i)]; });
    }

    Vc_INTRINSIC constexpr operator const register_type &() const { return d; }
    Vc_INTRINSIC constexpr operator register_type &() { return d; }

    [[deprecated("use .d instead")]] Vc_INTRINSIC constexpr const register_type &builtin() const { return d; }
    [[deprecated("use .d instead")]] Vc_INTRINSIC constexpr register_type &builtin() { return d; }

    template <class U = intrinsic_type_t<T, Width>>
    Vc_INTRINSIC constexpr U intrin() const
    {
        return reinterpret_cast<U>(d);
    }
    [[deprecated(
        "use intrin() instead")]] Vc_INTRINSIC constexpr intrinsic_type_t<T, Width>
    v() const
    {
        return intrin();
    }

    Vc_INTRINSIC constexpr T operator[](size_t i) const { return d[i]; }
    [[deprecated("use operator[] instead")]] Vc_INTRINSIC constexpr T m(size_t i) const
    {
        return d[i];
    }

    Vc_INTRINSIC void set(size_t i, T x) { d[i] = x; }
};

// to_storage {{{1
template <class T> class to_storage
{
    T d;

public:
    constexpr to_storage(T x) : d(x) {}

    template <size_t N> constexpr operator Storage<bool, N>() const
    {
        static_assert(std::is_integral_v<T>);
        return static_cast<detail::bool_storage_member_type_t<N>>(d);
    }

    template <class U, size_t N> constexpr operator Storage<U, N>() const
    {
        static_assert(detail::is_builtin_vector_v<T>);
        static_assert(sizeof(detail::builtin_type_t<U, N>) == sizeof(T));
        return {reinterpret_cast<detail::builtin_type_t<U, N>>(d)};
    }
};

// to_storage_unsafe {{{1
template <class T> class to_storage_unsafe
{
    T d;
public:
    constexpr to_storage_unsafe(T x) : d(x) {}
    template <class U, size_t N> constexpr operator Storage<U, N>() const
    {
        static_assert(sizeof(builtin_type_t<U, N>) <= sizeof(T));
        return {reinterpret_cast<builtin_type_t<U, N>>(d)};
    }
};

// storage_bitcast{{{1
template <class T, class U, size_t M, size_t N = sizeof(U) * M / sizeof(T)>
Vc_INTRINSIC constexpr Storage<T, N> storage_bitcast(Storage<U, M> x)
{
    static_assert(sizeof(builtin_type_t<T, N>) == sizeof(builtin_type_t<U, M>));
    return reinterpret_cast<builtin_type_t<T, N>>(x.d);
}

// make_storage{{{1
template <class T, class... Args>
Vc_INTRINSIC constexpr Storage<T, sizeof...(Args)> make_storage(Args &&... args)
{
    return {typename Storage<T, sizeof...(Args)>::register_type{static_cast<T>(args)...}};
}

// generate_storage{{{1
template <class T, size_t N, class G>
Vc_INTRINSIC constexpr Storage<T, N> generate_storage(G &&gen)
{
    return generate_builtin<T, N>(std::forward<G>(gen));
}

// work around clang miscompilation on set{{{1
#if defined Vc_CLANG && !defined Vc_HAVE_SSE4_1
#if Vc_CLANG <= 0x60000
template <> void Storage<double, 2, AliasStrategy::VectorBuiltin>::set(size_t i, double x)
{
    if (x == 0. && i == 1)
        asm("" : "+g"(x));  // make clang forget that x is 0
    d[i] = x;
}
#else
#warning "clang 5 failed operators_sse2_vectorbuiltin_ldouble_float_double_schar_uchar in operator<simd<double, __sse>> and required a workaround. Is this still the case for newer clang versions?"
#endif
#endif

// Storage ostream operators{{{1
template <class CharT, class T, size_t N>
inline std::basic_ostream<CharT> &operator<<(std::basic_ostream<CharT> & s,
                                             const Storage<T, N> &v)
{
    s << '[' << v[0];
    for (size_t i = 1; i < N; ++i) {
        s << ((i % 4) ? " " : " | ") << v[i];
    }
    return s << ']';
}

//}}}1
// fallback_abi_for_long_double {{{
template <class T, class A0, class A1> struct fallback_abi_for_long_double {
    using type = A0;
};
template <class A0, class A1> struct fallback_abi_for_long_double<long double, A0, A1> {
    using type = A1;
};
template <class T, class A0, class A1>
using fallback_abi_for_long_double_t =
    typename fallback_abi_for_long_double<T, A0, A1>::type;
// }}}
}  // namespace detail

namespace simd_abi
{
// most of simd_abi is defined in simd_detail.h
template <class T> inline constexpr int max_fixed_size = 32;
// compatible {{{
#if defined __x86_64__
template <class T>
using compatible = detail::fallback_abi_for_long_double_t<T, __sse, scalar>;
#elif defined Vc_IS_AARCH64
template <typename T>
using compatible = detail::fallback_abi_for_long_double_t<T, __neon, scalar>;
#else
template <typename> using compatible = scalar;
#endif

// }}}
// native {{{
#if defined Vc_HAVE_FULL_AVX512_ABI
template <typename T>
using native = detail::fallback_abi_for_long_double_t<T, __avx512, scalar>;
#elif defined Vc_HAVE_AVX512_ABI
template <typename T>
using native =
    std::conditional_t<(sizeof(T) >= 4),
                       detail::fallback_abi_for_long_double_t<T, __avx512, scalar>, __avx>;
#elif defined Vc_HAVE_FULL_AVX_ABI
template <typename T> using native = detail::fallback_abi_for_long_double_t<T, __avx, scalar>;
#elif defined Vc_HAVE_AVX_ABI
template <typename T>
using native =
    std::conditional_t<std::is_floating_point<T>::value,
                       detail::fallback_abi_for_long_double_t<T, __avx, scalar>, __sse>;
#elif defined Vc_HAVE_FULL_SSE_ABI
template <typename T>
using native = detail::fallback_abi_for_long_double_t<T, __sse, scalar>;
#elif defined Vc_HAVE_SSE_ABI
template <typename T>
using native = std::conditional_t<std::is_same<float, T>::value, __sse, scalar>;
#elif defined Vc_HAVE_FULL_NEON_ABI
template <typename T>
using native = detail::fallback_abi_for_long_double_t<T, __neon, scalar>;
#else
template <typename> using native = scalar;
#endif

// }}}
// __default_abi {{{
#if defined Vc_DEFAULT_ABI
template <typename T> using __default_abi = Vc_DEFAULT_ABI<T>;
#else
template <typename T> using __default_abi = compatible<T>;
#endif

// }}}
}  // namespace simd_abi

// traits {{{1
// is_abi_tag {{{2
template <class T, class = std::void_t<>> struct is_abi_tag : std::false_type {
};
template <class T>
struct is_abi_tag<T, std::void_t<typename T::is_valid_abi_tag>>
    : public T::is_valid_abi_tag {
};
template <class T> inline constexpr bool is_abi_tag_v = is_abi_tag<T>::value;

// is_simd(_mask) {{{2
template <class T> struct is_simd : public std::false_type {};
template <class T> inline constexpr bool is_simd_v = is_simd<T>::value;

template <class T> struct is_simd_mask : public std::false_type {};
template <class T> inline constexpr bool is_simd_mask_v = is_simd_mask<T>::value;

// simd_size {{{2
namespace detail
{
template <class T, class Abi, class = void> struct simd_size_impl {
};
template <class T, class Abi>
struct simd_size_impl<
    T, Abi,
    std::enable_if_t<std::conjunction_v<detail::is_vectorizable<T>, Vc::is_abi_tag<Abi>>>>
    : detail::size_constant<Abi::template size<T>> {
};
}  // namespace detail

template <class T, class Abi = simd_abi::__default_abi<T>>
struct simd_size : detail::simd_size_impl<T, Abi> {
};
template <class T, class Abi = simd_abi::__default_abi<T>>
inline constexpr size_t simd_size_v = simd_size<T, Abi>::value;

// simd_abi::deduce {{{2
namespace detail
{
template <class T, std::size_t N, class = void> struct deduce_impl;
}  // namespace detail
namespace simd_abi
{
/**
 * \tparam T    The requested `value_type` for the elements.
 * \tparam N    The requested number of elements.
 * \tparam Abis This parameter is ignored, since this implementation cannot make any use
 *              of it. Either a good native ABI is matched and used as `type` alias, or
 *              the `fixed_size<N>` ABI is used, which internally is built from the best
 *              matching native ABIs.
 */
template <class T, std::size_t N, class...>
struct deduce : Vc::detail::deduce_impl<T, N> {};

template <class T, size_t N, class... Abis>
using deduce_t = typename deduce<T, N, Abis...>::type;
}  // namespace simd_abi

// }}}2
// rebind_simd {{{2
template <class T, class V> struct rebind_simd;
template <class T, class U, class Abi> struct rebind_simd<T, simd<U, Abi>> {
    using type = simd<T, simd_abi::deduce_t<T, simd_size_v<U, Abi>, Abi>>;
};
template <class T, class U, class Abi> struct rebind_simd<T, simd_mask<U, Abi>> {
    using type = simd_mask<T, simd_abi::deduce_t<T, simd_size_v<U, Abi>, Abi>>;
};
template <class T, class V> using rebind_simd_t = typename rebind_simd<T, V>::type;

// resize_simd {{{2
template <int N, class V> struct resize_simd;
template <int N, class T, class Abi> struct resize_simd<N, simd<T, Abi>> {
    using type = simd<T, simd_abi::deduce_t<T, N, Abi>>;
};
template <int N, class T, class Abi> struct resize_simd<N, simd_mask<T, Abi>> {
    using type = simd_mask<T, simd_abi::deduce_t<T, N, Abi>>;
};
template <int N, class V> using resize_simd_t = typename resize_simd<N, V>::type;

// }}}2
// memory_alignment {{{2
template <class T, class U = typename T::value_type>
struct memory_alignment
    : public detail::size_constant<detail::next_power_of_2(sizeof(U) * T::size())> {
};
template <class T, class U = typename T::value_type>
inline constexpr size_t memory_alignment_v = memory_alignment<T, U>::value;

// class template simd [simd] {{{1
template <class T, class Abi = simd_abi::__default_abi<T>> class simd;
template <class T, class Abi> struct is_simd<simd<T, Abi>> : public std::true_type {};
template <class T> using native_simd = simd<T, simd_abi::native<T>>;
template <class T, int N> using fixed_size_simd = simd<T, simd_abi::fixed_size<N>>;
template <class T, size_t N> using __deduced_simd = simd<T, simd_abi::deduce_t<T, N>>;

// class template simd_mask [simd_mask] {{{1
template <class T, class Abi = simd_abi::__default_abi<T>> class simd_mask;
template <class T, class Abi> struct is_simd_mask<simd_mask<T, Abi>> : public std::true_type {};
template <class T> using native_simd_mask = simd_mask<T, simd_abi::native<T>>;
template <class T, int N> using fixed_size_simd_mask = simd_mask<T, simd_abi::fixed_size<N>>;
template <class T, size_t N>
using __deduced_simd_mask = simd_mask<T, simd_abi::deduce_t<T, N>>;

namespace detail
{
template <class T, class Abi> struct get_impl<Vc::simd_mask<T, Abi>> {
    using type = typename traits<T, Abi>::mask_impl_type;
};
template <class T, class Abi> struct get_impl<Vc::simd<T, Abi>> {
    using type = typename traits<T, Abi>::simd_impl_type;
};

template <class T, class Abi> struct get_traits<Vc::simd_mask<T, Abi>> {
    using type = detail::traits<T, Abi>;
};
template <class T, class Abi> struct get_traits<Vc::simd<T, Abi>> {
    using type = detail::traits<T, Abi>;
};
}  // namespace detail

// casts [simd.casts] {{{1
// static_simd_cast {{{2
namespace detail
{
template <class T, class U, class A, bool = is_simd_v<T>, class = void>
struct static_simd_cast_return_type;

template <class T, class A0, class U, class A>
struct static_simd_cast_return_type<simd_mask<T, A0>, U, A, false, void>
    : static_simd_cast_return_type<simd<T, A0>, U, A> {
};

template <class T, class U, class A>
struct static_simd_cast_return_type<T, U, A, true,
                                    std::enable_if_t<T::size() == simd_size_v<U, A>>> {
    using type = T;
};

template <class T, class A>
struct static_simd_cast_return_type<T, T, A, false,
#ifdef Vc_FIX_P2TS_ISSUE66
                                    std::enable_if_t<detail::is_vectorizable_v<T>>
#else
                                    void
#endif
                                    > {
    using type = simd<T, A>;
};

template <class T, class = void> struct safe_make_signed {
    using type = T;
};
template <class T> struct safe_make_signed<T, std::enable_if_t<std::is_integral_v<T>>> {
    // the extra make_unsigned_t is because of PR85951
    using type = std::make_signed_t<std::make_unsigned_t<T>>;
};
template <class T> using safe_make_signed_t = typename safe_make_signed<T>::type;

template <class T, class U, class A>
struct static_simd_cast_return_type<T, U, A, false,
#ifdef Vc_FIX_P2TS_ISSUE66
                                    std::enable_if_t<detail::is_vectorizable_v<T>>
#else
                                    void
#endif
                                    > {
    using type =
        std::conditional_t<(std::is_integral_v<U> && std::is_integral_v<T> &&
#ifndef Vc_FIX_P2TS_ISSUE65
                            std::is_signed_v<U> != std::is_signed_v<T> &&
#endif
                            std::is_same_v<safe_make_signed_t<U>, safe_make_signed_t<T>>),
                           simd<T, A>, fixed_size_simd<T, simd_size_v<U, A>>>;
};

// specialized in scalar.h
template <class To, class, class, class Native, class From>
Vc_INTRINSIC To mask_cast_impl(const Native *, const From &x)
{
    static_assert(std::is_same_v<Native, typename detail::get_traits_t<To>::mask_member_type>);
    if constexpr (std::is_same_v<Native, bool>) {
        return {Vc::detail::private_init, x[0]};
    } else if constexpr (std::is_same_v<From, bool>) {
        To r{};
        r[0] = x;
        return r;
    } else {
        return {private_init,
                convert_mask<typename detail::get_traits_t<To>::mask_member_type>(x)};
    }
}
template <class To, class, class, class Native, size_t N>
Vc_INTRINSIC To mask_cast_impl(const Native *, const std::bitset<N> &x)
{
    return {Vc::detail::bitset_init, x};
}
template <class To, class, class>
Vc_INTRINSIC To mask_cast_impl(const bool *, bool x)
{
    return To(x);
}
template <class To, class, class>
Vc_INTRINSIC To mask_cast_impl(const std::bitset<1> *, bool x)
{
    return To(x);
}
template <class To, class T, class Abi, size_t N, class From>
Vc_INTRINSIC To mask_cast_impl(const std::bitset<N> *, const From &x)
{
    return {Vc::detail::private_init, detail::to_bitset(x)};
}
template <class To, class, class, size_t N>
Vc_INTRINSIC To mask_cast_impl(const std::bitset<N> *, const std::bitset<N> &x)
{
    return {Vc::detail::private_init, x};
}
}  // namespace detail

template <class T, class U, class A,
          class R = typename detail::static_simd_cast_return_type<T, U, A>::type>
Vc_INTRINSIC R static_simd_cast(const simd<U, A> &x)
{
    if constexpr(std::is_same<R, simd<U, A>>::value) {
        return x;
    } else {
        detail::simd_converter<U, A, typename R::value_type, typename R::abi_type> c;
        return R(detail::private_init, c(detail::data(x)));
    }
}

template <class T, class U, class A,
          class R = typename detail::static_simd_cast_return_type<T, U, A>::type>
Vc_INTRINSIC typename R::mask_type static_simd_cast(const simd_mask<U, A> &x)
{
    using RM = typename R::mask_type;
    if constexpr(std::is_same<RM, simd_mask<U, A>>::value) {
        return x;
    } else {
        using traits = detail::traits<typename R::value_type, typename R::abi_type>;
        const typename traits::mask_member_type *tag = nullptr;
        return detail::mask_cast_impl<RM, U, A>(tag, detail::data(x));
    }
}

// simd_cast {{{2
template <class T, class U, class A, class To = detail::value_type_or_identity<T>>
Vc_INTRINSIC auto simd_cast(const simd<detail::value_preserving<U, To>, A> &x)
    ->decltype(static_simd_cast<T>(x))
{
    return static_simd_cast<T>(x);
}

template <class T, class U, class A, class To = detail::value_type_or_identity<T>>
Vc_INTRINSIC auto simd_cast(const simd_mask<detail::value_preserving<U, To>, A> &x)
    ->decltype(static_simd_cast<T>(x))
{
    return static_simd_cast<T>(x);
}

namespace __proposed
{
template <class T, class U, class A>
Vc_INTRINSIC T resizing_simd_cast(const simd_mask<U, A> &x)
{
    static_assert(is_simd_mask_v<T>);
    if constexpr (std::is_same_v<T, simd_mask<U, A>>) {
        return x;
    } else {
        using traits = detail::traits<typename T::simd_type::value_type, typename T::abi_type>;
        const typename traits::mask_member_type *tag = nullptr;
        return detail::mask_cast_impl<T, U, A>(tag, detail::data(x));
    }
}
}  // namespace __proposed

// to_fixed_size {{{2
template <class T, int N>
Vc_INTRINSIC fixed_size_simd<T, N> to_fixed_size(const fixed_size_simd<T, N> &x)
{
    return x;
}

template <class T, int N>
Vc_INTRINSIC fixed_size_simd_mask<T, N> to_fixed_size(const fixed_size_simd_mask<T, N> &x)
{
    return x;
}

template <class T, class A> Vc_INTRINSIC auto to_fixed_size(const simd<T, A> &x)
{
    return simd<T, simd_abi::fixed_size<simd_size_v<T, A>>>(
        [&x](auto i) { return x[i]; });
}

template <class T, class A> Vc_INTRINSIC auto to_fixed_size(const simd_mask<T, A> &x)
{
    constexpr int N = simd_mask<T, A>::size();
    fixed_size_simd_mask<T, N> r;
    detail::execute_n_times<N>([&](auto i) { r[i] = x[i]; });
    return r;
}

// to_native {{{2
template <class T, int N>
Vc_INTRINSIC std::enable_if_t<(N == native_simd<T>::size()), native_simd<T>>
to_native(const fixed_size_simd<T, N> &x)
{
    alignas(memory_alignment_v<native_simd<T>>) T mem[N];
    x.copy_to(mem, vector_aligned);
    return {mem, vector_aligned};
}

template <class T, size_t N>
Vc_INTRINSIC std::enable_if_t<(N == native_simd_mask<T>::size()), native_simd_mask<T>> to_native(
    const fixed_size_simd_mask<T, N> &x)
{
    return native_simd_mask<T>([&](auto i) { return x[i]; });
}

// to_compatible {{{2
template <class T, size_t N>
Vc_INTRINSIC std::enable_if_t<(N == simd<T>::size()), simd<T>> to_compatible(
    const simd<T, simd_abi::fixed_size<N>> &x)
{
    alignas(memory_alignment_v<simd<T>>) T mem[N];
    x.copy_to(mem, vector_aligned);
    return {mem, vector_aligned};
}

template <class T, size_t N>
Vc_INTRINSIC std::enable_if_t<(N == simd_mask<T>::size()), simd_mask<T>> to_compatible(
    const simd_mask<T, simd_abi::fixed_size<N>> &x)
{
    return simd_mask<T>([&](auto i) { return x[i]; });
}

// simd_reinterpret_cast {{{2
namespace detail
{
template <class To, size_t N> Vc_INTRINSIC To simd_reinterpret_cast_impl(std::bitset<N> x)
{
    return {bitset_init, x};
}

template <class To, class T, size_t N>
Vc_INTRINSIC To simd_reinterpret_cast_impl(Storage<T, N> x)
{
    return {private_init, x};
}
}  // namespace detail

namespace __proposed
{
template <class To, class T, class A,
          class = std::enable_if_t<sizeof(To) == sizeof(simd<T, A>) &&
                                   (is_simd_v<To> || is_simd_mask_v<To>)>>
Vc_INTRINSIC To simd_reinterpret_cast(const simd<T, A> &x)
{
    //return {detail::private_init, detail::data(x)};
    return reinterpret_cast<const To &>(x);
}

template <class To, class T, class A,
          class = std::enable_if_t<(is_simd_v<To> || is_simd_mask_v<To>)>>
Vc_INTRINSIC To simd_reinterpret_cast(const simd_mask<T, A> &x)
{
    return Vc::detail::simd_reinterpret_cast_impl<To>(detail::data(x));
    //return reinterpret_cast<const To &>(x);
}
}  // namespace __proposed

// masked assignment [simd_mask.where] {{{1
#ifdef Vc_EXPERIMENTAL
namespace detail {
template <class T, class A> class masked_simd_impl;
template <class T, class A>
masked_simd_impl<T, A> masked_simd(const typename simd<T, A>::mask_type &k,
                                         simd<T, A> &v);
}  // namespace detail
#endif  // Vc_EXPERIMENTAL

// where_expression {{{1
template <typename M, typename T> class const_where_expression  //{{{2
{
    using V = T;
    static_assert(std::is_same_v<V, std::decay_t<T>>);
    struct Wrapper {
        using value_type = V;
    };

protected:
    using value_type =
        typename std::conditional_t<std::is_arithmetic<V>::value, Wrapper, V>::value_type;
    Vc_INTRINSIC friend const M &get_mask(const const_where_expression &x) { return x.k; }
    Vc_INTRINSIC friend const T &get_lvalue(const const_where_expression &x) { return x.d; }
    const M &k;
    T &d;

public:
    const_where_expression(const const_where_expression &) = delete;
    const_where_expression &operator=(const const_where_expression &) = delete;

    Vc_INTRINSIC const_where_expression(const M &kk, const T &dd) : k(kk), d(const_cast<T &>(dd)) {}

    Vc_INTRINSIC V operator-() const &&
    {
        return {detail::private_init,
                detail::get_impl_t<V>::template masked_unary<std::negate>(
                    detail::data(k), detail::data(d))};
    }

    template <class U, class Flags>
    Vc_NODISCARD Vc_INTRINSIC V
    copy_from(const detail::loadstore_ptr_type<U, value_type> *mem, Flags f) const &&
    {
        return {detail::private_init, detail::get_impl_t<V>::masked_load(
                                          detail::data(d), detail::data(k), mem, f)};
    }

    template <class U, class Flags>
    Vc_INTRINSIC void copy_to(detail::loadstore_ptr_type<U, value_type> *mem,
                              Flags f) const &&
    {
        detail::get_impl_t<V>::masked_store(detail::data(d), mem, f, detail::data(k));
    }
};

template <typename T> class const_where_expression<bool, T>  //{{{2
{
    using M = bool;
    using V = T;
    static_assert(std::is_same_v<V, std::decay_t<T>>);
    struct Wrapper {
        using value_type = V;
    };

protected:
    using value_type =
        typename std::conditional_t<std::is_arithmetic<V>::value, Wrapper, V>::value_type;
    Vc_INTRINSIC friend const M &get_mask(const const_where_expression &x) { return x.k; }
    Vc_INTRINSIC friend const T &get_lvalue(const const_where_expression &x) { return x.d; }
    const bool k;
    T &d;

public:
    const_where_expression(const const_where_expression &) = delete;
    const_where_expression &operator=(const const_where_expression &) = delete;

    Vc_INTRINSIC const_where_expression(const bool kk, const T &dd) : k(kk), d(const_cast<T &>(dd)) {}

    Vc_INTRINSIC V operator-() const && { return k ? -d : d; }

    template <class U, class Flags>
    Vc_NODISCARD Vc_INTRINSIC V
    copy_from(const detail::loadstore_ptr_type<U, value_type> *mem, Flags) const &&
    {
        return k ? static_cast<V>(mem[0]) : d;
    }

    template <class U, class Flags>
    Vc_INTRINSIC void copy_to(detail::loadstore_ptr_type<U, value_type> *mem,
                              Flags) const &&
    {
        if (k) {
            mem[0] = d;
        }
    }
};

// where_expression {{{2
template <typename M, typename T>
class where_expression : public const_where_expression<M, T>
{
    static_assert(!std::is_const<T>::value, "where_expression may only be instantiated with a non-const T parameter");
    using typename const_where_expression<M, T>::value_type;
    using const_where_expression<M, T>::k;
    using const_where_expression<M, T>::d;
    static_assert(std::is_same<typename M::abi_type, typename T::abi_type>::value, "");
    static_assert(M::size() == T::size(), "");

    Vc_INTRINSIC friend T &get_lvalue(where_expression &x) { return x.d; }
public:
    where_expression(const where_expression &) = delete;
    where_expression &operator=(const where_expression &) = delete;

    Vc_INTRINSIC where_expression(const M &kk, T &dd)
        : const_where_expression<M, T>(kk, dd)
    {
    }

    template <class U> Vc_INTRINSIC void operator=(U &&x) &&
    {
        Vc::detail::get_impl_t<T>::masked_assign(
            detail::data(k), detail::data(d),
            detail::to_value_type_or_member_type<T>(std::forward<U>(x)));
    }

#define Vc_OP_(op_, name_)                                                               \
    template <class U> Vc_INTRINSIC void operator op_##=(U &&x) &&                       \
    {                                                                                    \
        Vc::detail::get_impl_t<T>::template masked_cassign<name_>(                       \
            detail::data(k), detail::data(d),                                            \
            detail::to_value_type_or_member_type<T>(std::forward<U>(x)));                \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON
    Vc_OP_(+, std::plus);
    Vc_OP_(-, std::minus);
    Vc_OP_(*, std::multiplies);
    Vc_OP_(/, std::divides);
    Vc_OP_(%, std::modulus);
    Vc_OP_(&, std::bit_and);
    Vc_OP_(|, std::bit_or);
    Vc_OP_(^, std::bit_xor);
    Vc_OP_(<<, detail::shift_left);
    Vc_OP_(>>, detail::shift_right);
#undef Vc_OP_

    Vc_INTRINSIC void operator++() &&
    {
        detail::data(d) = detail::get_impl_t<T>::template masked_unary<detail::increment>(
            detail::data(k), detail::data(d));
    }
    Vc_INTRINSIC void operator++(int) &&
    {
        detail::data(d) = detail::get_impl_t<T>::template masked_unary<detail::increment>(
            detail::data(k), detail::data(d));
    }
    Vc_INTRINSIC void operator--() &&
    {
        detail::data(d) = detail::get_impl_t<T>::template masked_unary<detail::decrement>(
            detail::data(k), detail::data(d));
    }
    Vc_INTRINSIC void operator--(int) &&
    {
        detail::data(d) = detail::get_impl_t<T>::template masked_unary<detail::decrement>(
            detail::data(k), detail::data(d));
    }

    // intentionally hides const_where_expression::copy_from
    template <class U, class Flags>
    Vc_INTRINSIC void copy_from(const detail::loadstore_ptr_type<U, value_type> *mem,
                                Flags f) &&
    {
        detail::data(d) =
            detail::get_impl_t<T>::masked_load(detail::data(d), detail::data(k), mem, f);
    }

#ifdef Vc_EXPERIMENTAL
    template <class F>
    Vc_INTRINSIC std::enable_if_t<
        detail::all<std::is_same<decltype(std::declval<F>()(detail::masked_simd(
                                     std::declval<const M &>(), std::declval<T &>()))),
                                 void>>::value,
        where_expression &&>
    apply(F &&f) &&
    {
        std::forward<F>(f)(detail::masked_simd(k, d));
        return std::move(*this);
    }

    template <class F>
    Vc_INTRINSIC std::enable_if_t<
        detail::all<std::is_same<decltype(std::declval<F>()(detail::masked_simd(
                                     std::declval<const M &>(), std::declval<T &>()))),
                                 void>>::value,
        where_expression &&>
    apply_inv(F &&f) &&
    {
        std::forward<F>(f)(detail::masked_simd(!k, d));
        return std::move(*this);
    }
#endif  // Vc_EXPERIMENTAL
};

// where_expression<bool> {{{2
template <typename T>
class where_expression<bool, T> : public const_where_expression<bool, T>
{
    using M = bool;
    using typename const_where_expression<M, T>::value_type;
    using const_where_expression<M, T>::k;
    using const_where_expression<M, T>::d;

public:
    where_expression(const where_expression &) = delete;
    where_expression &operator=(const where_expression &) = delete;

    Vc_INTRINSIC where_expression(const M &kk, T &dd)
        : const_where_expression<M, T>(kk, dd)
    {
    }

#define Vc_OP_(op_)                                                                      \
    template <class U> Vc_INTRINSIC void operator op_(U &&x) &&                          \
    {                                                                                    \
        if (k) {                                                                         \
            d op_ std::forward<U>(x);                                                    \
        }                                                                                \
    }                                                                                    \
    Vc_NOTHING_EXPECTING_SEMICOLON
    Vc_OP_(=);
    Vc_OP_(+=);
    Vc_OP_(-=);
    Vc_OP_(*=);
    Vc_OP_(/=);
    Vc_OP_(%=);
    Vc_OP_(&=);
    Vc_OP_(|=);
    Vc_OP_(^=);
    Vc_OP_(<<=);
    Vc_OP_(>>=);
#undef Vc_OP_
    Vc_INTRINSIC void operator++()    && { if (k) { ++d; } }
    Vc_INTRINSIC void operator++(int) && { if (k) { ++d; } }
    Vc_INTRINSIC void operator--()    && { if (k) { --d; } }
    Vc_INTRINSIC void operator--(int) && { if (k) { --d; } }

    // intentionally hides const_where_expression::copy_from
    template <class U, class Flags>
    Vc_INTRINSIC void copy_from(const detail::loadstore_ptr_type<U, value_type> *mem,
                                Flags) &&
    {
        if (k) {
            d = mem[0];
        }
    }
};

// where_expression<M, tuple<...>> {{{2
#ifdef Vc_EXPERIMENTAL
template <typename M, typename... Ts> class where_expression<M, std::tuple<Ts &...>>
{
    const M &k;
    std::tuple<Ts &...> d;

public:
    where_expression(const where_expression &) = delete;
    where_expression &operator=(const where_expression &) = delete;

    Vc_INTRINSIC where_expression(const M &kk, std::tuple<Ts &...> &&dd) : k(kk), d(dd) {}

private:
    template <class F, std::size_t... Is>
    Vc_INTRINSIC void apply_helper(F &&f, const M &simd_mask, std::index_sequence<Is...>)
    {
        return std::forward<F>(f)(detail::masked_simd(simd_mask, std::get<Is>(d))...);
    }

public:
    template <class F>
    Vc_INTRINSIC std::enable_if_t<
        detail::all<
            std::is_same<decltype(std::declval<F>()(detail::masked_simd(
                             std::declval<const M &>(), std::declval<Ts &>())...)),
                         void>>::value,
        where_expression &&>
    apply(F &&f) &&
    {
        apply_helper(std::forward<F>(f), k, std::make_index_sequence<sizeof...(Ts)>());
        return std::move(*this);
    }

    template <class F>
    Vc_INTRINSIC std::enable_if_t<
        detail::all<
            std::is_same<decltype(std::declval<F>()(detail::masked_simd(
                             std::declval<const M &>(), std::declval<Ts &>())...)),
                         void>>::value,
        where_expression &&>
    apply_inv(F &&f) &&
    {
        apply_helper(std::forward<F>(f), !k, std::make_index_sequence<sizeof...(Ts)>());
        return std::move(*this);
    }
};

template <class T, class A, class... Vs>
Vc_INTRINSIC where_expression<simd_mask<T, A>, std::tuple<simd<T, A> &, Vs &...>> where(
    const typename simd<T, A>::mask_type &k, simd<T, A> &v0, Vs &... vs)
{
    return {k, std::tie(v0, vs...)};
}
#endif  // Vc_EXPERIMENTAL

// where {{{1
template <class T, class A>
Vc_INTRINSIC where_expression<simd_mask<T, A>, simd<T, A>> where(
    const typename simd<T, A>::mask_type &k, simd<T, A> &d)
{
    return {k, d};
}
template <class T, class A>
Vc_INTRINSIC const_where_expression<simd_mask<T, A>, simd<T, A>> where(
    const typename simd<T, A>::mask_type &k, const simd<T, A> &d)
{
    return {k, d};
}
template <class T, class A>
Vc_INTRINSIC where_expression<simd_mask<T, A>, simd_mask<T, A>> where(
    const std::remove_const_t<simd_mask<T, A>> &k, simd_mask<T, A> &d)
{
    return {k, d};
}
template <class T, class A>
Vc_INTRINSIC const_where_expression<simd_mask<T, A>, simd_mask<T, A>> where(
    const std::remove_const_t<simd_mask<T, A>> &k, const simd_mask<T, A> &d)
{
    return {k, d};
}
template <class T>
Vc_INTRINSIC where_expression<bool, T> where(detail::exact_bool k, T &d)
{
    return {k, d};
}
template <class T>
Vc_INTRINSIC const_where_expression<bool, T> where(detail::exact_bool k, const T &d)
{
    return {k, d};
}
template <class T, class A> void where(bool k, simd<T, A> &d) = delete;
template <class T, class A> void where(bool k, const simd<T, A> &d) = delete;

// proposed mask iterations {{{1
namespace __proposed
{
template <size_t N> class where_range
{
    const std::bitset<N> bits;

public:
    where_range(std::bitset<N> b) : bits(b) {}

    class iterator
    {
        size_t mask;
        size_t bit;

        void next_bit()
        {
            bit = __builtin_ctzl(mask);
        }
        void reset_lsb()
        {
            // 01100100 - 1 = 01100011
            mask &= (mask - 1);
            /*
            __asm__("btr %1,%0" : "+r"(mask) : "r"(bit));
            */
        }
    public:
        iterator(decltype(mask) m) : mask(m) { next_bit(); }
        iterator(const iterator &) = default;
        iterator(iterator &&) = default;

        Vc_ALWAYS_INLINE size_t operator->() const { return bit; }
        Vc_ALWAYS_INLINE size_t operator*() const { return bit; }

        Vc_ALWAYS_INLINE iterator &operator++()
        {
            reset_lsb();
            next_bit();
            return *this;
        }
        Vc_ALWAYS_INLINE iterator operator++(int)
        {
            iterator tmp = *this;
            reset_lsb();
            next_bit();
            return tmp;
        }

        Vc_ALWAYS_INLINE bool operator==(const iterator &rhs) const
        {
            return mask == rhs.mask;
        }
        Vc_ALWAYS_INLINE bool operator!=(const iterator &rhs) const
        {
            return mask != rhs.mask;
        }
    };

    iterator begin() const { return bits.to_ullong(); }
    iterator end() const { return 0; }
};

template <class T, class A> where_range<simd_size_v<T, A>> where(const simd_mask<T, A> &k)
{
    return k.to_bitset();
}

}  // namespace __proposed

// }}}1
// reductions [simd.reductions] {{{1
template <class T, class Abi, class BinaryOperation = std::plus<>>
Vc_INTRINSIC T reduce(const simd<T, Abi> &v,
                      BinaryOperation binary_op = BinaryOperation())
{
    using V = simd<T, Abi>;
    return detail::get_impl_t<V>::reduce(v, binary_op);
}

template <class M, class V, class BinaryOperation = std::plus<>>
Vc_INTRINSIC typename V::value_type reduce(const const_where_expression<M, V> &x,
                                           typename V::value_type identity_element,
                                           BinaryOperation binary_op)
{
    V tmp = identity_element;
    detail::get_impl_t<V>::masked_assign(detail::data(get_mask(x)), detail::data(tmp),
                                         detail::data(get_lvalue(x)));
    return reduce(tmp, binary_op);
}

template <class M, class V>
Vc_INTRINSIC typename V::value_type reduce(const const_where_expression<M, V> &x,
                                           std::plus<> binary_op = {})
{
    return reduce(x, 0, binary_op);
}

template <class M, class V>
Vc_INTRINSIC typename V::value_type reduce(const const_where_expression<M, V> &x,
                                           std::multiplies<> binary_op)
{
    return reduce(x, 1, binary_op);
}

template <class M, class V>
Vc_INTRINSIC typename V::value_type reduce(const const_where_expression<M, V> &x,
                                           std::bit_and<> binary_op)
{
    return reduce(x, ~typename V::value_type(), binary_op);
}

template <class M, class V>
Vc_INTRINSIC typename V::value_type reduce(const const_where_expression<M, V> &x,
                                           std::bit_or<> binary_op)
{
    return reduce(x, 0, binary_op);
}

template <class M, class V>
Vc_INTRINSIC typename V::value_type reduce(const const_where_expression<M, V> &x,
                                           std::bit_xor<> binary_op)
{
    return reduce(x, 0, binary_op);
}

// }}}1
// algorithms [simd.alg] {{{
template <class T, class A>
Vc_INTRINSIC simd<T, A> min(const simd<T, A> &a, const simd<T, A> &b)
{
    return {detail::private_init,
            A::simd_impl_type::min(detail::data(a), detail::data(b))};
}
template <class T, class A>
Vc_INTRINSIC simd<T, A> max(const simd<T, A> &a, const simd<T, A> &b)
{
    return {detail::private_init,
            A::simd_impl_type::max(detail::data(a), detail::data(b))};
}
template <class T, class A>
Vc_INTRINSIC std::pair<simd<T, A>, simd<T, A>> minmax(const simd<T, A> &a,
                                                            const simd<T, A> &b)
{
    const auto pair_of_members =
        A::simd_impl_type::minmax(detail::data(a), detail::data(b));
    return {simd<T, A>(detail::private_init, pair_of_members.first),
            simd<T, A>(detail::private_init, pair_of_members.second)};
}
template <class T, class A>
Vc_INTRINSIC simd<T, A> clamp(const simd<T, A> &v, const simd<T, A> &lo,
                                 const simd<T, A> &hi)
{
    using Impl = typename A::simd_impl_type;
    return {detail::private_init,
            Impl::min(detail::data(hi), Impl::max(detail::data(lo), detail::data(v)))};
}

// }}}

namespace __proposed
{
// shuffle {{{1
template <int Stride, int Offset = 0> struct strided {
    static constexpr int stride = Stride;
    static constexpr int offset = Offset;
    template <class T, class A>
    using shuffle_return_type = simd<
        T, simd_abi::deduce_t<T, (simd_size_v<T, A> - Offset + Stride - 1) / Stride, A>>;
    // alternative, always use fixed_size:
    // fixed_size_simd<T, (simd_size_v<T, A> - Offset + Stride - 1) / Stride>;
    template <class T> static constexpr auto src_index(T dst_index)
    {
        return Offset + dst_index * Stride;
    }
};

// SFINAE for the return type ensures P is a type that provides the alias template member
// shuffle_return_type and the static member function src_index
template <class P, class T, class A,
          class R = typename P::template shuffle_return_type<T, A>,
          class = decltype(P::src_index(Vc::detail::size_constant<0>()))>
Vc_NOT_OPTIMIZED Vc_INTRINSIC R shuffle(const simd<T, A> &x)
{
    return R([&x](auto i) { return x[P::src_index(i)]; });
}

// }}}1
}  // namespace __proposed

template <size_t... Sizes, class T, class A,
          class = std::enable_if_t<((Sizes + ...) == simd<T, A>::size())>>
inline std::tuple<simd<T, simd_abi::deduce_t<T, Sizes>>...> split(const simd<T, A> &);

namespace detail
{
// extract_part {{{
template <int Index, int Parts, class T, size_t N>
auto extract_part(Storage<T, N> x);
template <int Index, int Parts, class T, class A0, class... As>
auto extract_part(const simd_tuple<T, A0, As...> &x);

// }}}
// size_list {{{
template <size_t V0, size_t... Values> struct size_list {
    static constexpr size_t size = sizeof...(Values) + 1;

    template <size_t I> static constexpr size_t at(size_constant<I> = {})
    {
        if constexpr (I == 0) {
            return V0;
        } else {
            return size_list<Values...>::template at<I - 1>();
        }
    }

    template <size_t I> static constexpr auto before(size_constant<I> = {})
    {
        if constexpr (I == 0) {
            return size_constant<0>();
        } else {
            return size_constant<V0 + size_list<Values...>::template before<I - 1>()>();
        }
    }

    template <size_t N> static constexpr auto pop_front(size_constant<N> = {})
    {
        if constexpr (N == 0) {
            return size_list();
        } else {
            return size_list<Values...>::template pop_front<N-1>();
        }
    }
};
// }}}
// extract_center {{{
template <class T, size_t N> inline Storage<T, N / 2> extract_center(Storage<T, N> x) {
    if constexpr (have_avx512f && sizeof(x) == 64) {
        if constexpr(std::is_integral_v<T>) {
            return _mm512_castsi512_si256(
                _mm512_shuffle_i32x4(x, x, 1 + 2 * 0x4 + 2 * 0x10 + 3 * 0x40));
        } else if constexpr (sizeof(T) == 4) {
            return _mm512_castps512_ps256(
                _mm512_shuffle_f32x4(x, x, 1 + 2 * 0x4 + 2 * 0x10 + 3 * 0x40));
        } else if constexpr (sizeof(T) == 8) {
            return _mm512_castpd512_pd256(
                _mm512_shuffle_f64x2(x, x, 1 + 2 * 0x4 + 2 * 0x10 + 3 * 0x40));
        } else {
            assert_unreachable<T>();
        }
    } else {
        assert_unreachable<T>();
    }
}
template <class T, class A>
inline Storage<T, simd_size_v<T, A>> extract_center(const simd_tuple<T, A, A> &x)
{
    return detail::concat(detail::extract<1, 2>(x.first.d),
                          detail::extract<0, 2>(x.second.first.d));
}
template <class T, class A>
inline Storage<T, simd_size_v<T, A> / 2> extract_center(const simd_tuple<T, A> &x)
{
    return detail::extract_center(x.first);
}

// }}}
// split_wrapper {{{
template <size_t... Sizes, class T, class... As>
auto split_wrapper(size_list<Sizes...>, const simd_tuple<T, As...> &x)
{
    return Vc::split<Sizes...>(
        fixed_size_simd<T, simd_tuple<T, As...>::size()>(private_init, x));
}

// }}}
}  // namespace detail

// split<simd>(simd) {{{
template <class V, class A,
          size_t Parts = simd_size_v<typename V::value_type, A> / V::size()>
inline std::enable_if_t<(is_simd<V>::value &&
                         simd_size_v<typename V::value_type, A> == Parts * V::size()),
                        std::array<V, Parts>>
split(const simd<typename V::value_type, A> &x)
{
    using T = typename V::value_type;
    if constexpr (Parts == 1) {
        return {simd_cast<V>(x)};
    } else if constexpr (detail::is_fixed_size_abi_v<A> &&
                         (std::is_same_v<typename V::abi_type, simd_abi::scalar> ||
                          (detail::is_fixed_size_abi_v<typename V::abi_type> &&
                           sizeof(V) == sizeof(T) * V::size()  // V doesn't have padding
                           ))) {
        // fixed_size -> fixed_size (w/o padding) or scalar
#ifdef Vc_USE_ALIASING_LOADS
        const detail::may_alias<T> *const element_ptr =
            reinterpret_cast<const detail::may_alias<T> *>(&detail::data(x));
        return detail::generate_from_n_evaluations<Parts, std::array<V, Parts>>(
            [&](auto i) { return V(element_ptr + i * V::size(), vector_aligned); });
#else
        const auto &xx = detail::data(x);
        return detail::generate_from_n_evaluations<Parts, std::array<V, Parts>>(
            [&](auto i) {
                constexpr size_t offset = decltype(i)::value * V::size();
                detail::unused(offset);  // not really
                return V([&](auto j) {
                    constexpr detail::size_constant<j + offset> k;
                    return xx[k];
                });
            });
#endif
    } else if constexpr (std::is_same_v<typename V::abi_type, simd_abi::scalar>) {
        // normally memcpy should work here as well
        return detail::generate_from_n_evaluations<Parts, std::array<V, Parts>>(
            [&](auto i) { return x[i]; });
    } else {
        return detail::generate_from_n_evaluations<Parts, std::array<V, Parts>>(
            [&](auto i) {
                return V(detail::private_init,
                         detail::extract_part<i, Parts>(detail::data(x)));
            });
    }
}

// }}}
// split<simd_mask>(simd_mask) {{{
template <class V, class A,
          size_t Parts = simd_size_v<typename V::simd_type::value_type, A> / V::size()>
std::enable_if_t<(is_simd_mask_v<V> &&
                  simd_size_v<typename V::simd_type::value_type, A> == Parts * V::size()),
                 std::array<V, Parts>>
split(const simd_mask<typename V::simd_type::value_type, A> &x)
{
    if constexpr (std::is_same_v<A, typename V::abi_type>) {
        return {x};
    } else if constexpr (Parts == 1) {
        return {static_simd_cast<V>(x)};
    } else if constexpr (Parts == 2) {
        return {V(detail::private_init, [&](size_t i) { return x[i]; }),
                V(detail::private_init, [&](size_t i) { return x[i + V::size()]; })};
    } else if constexpr (Parts == 3) {
        return {V(detail::private_init, [&](size_t i) { return x[i]; }),
                V(detail::private_init, [&](size_t i) { return x[i + V::size()]; }),
                V(detail::private_init, [&](size_t i) { return x[i + 2 * V::size()]; })};
    } else if constexpr (Parts == 4) {
        return {V(detail::private_init, [&](size_t i) { return x[i]; }),
                V(detail::private_init, [&](size_t i) { return x[i + V::size()]; }),
                V(detail::private_init, [&](size_t i) { return x[i + 2 * V::size()]; }),
                V(detail::private_init, [&](size_t i) { return x[i + 3 * V::size()]; })};
    } else {
        detail::assert_unreachable<V>();
    }
}

// }}}
// split<Sizes...>(simd) {{{
template <size_t... Sizes, class T, class A,
          class = std::enable_if_t<((Sizes + ...) == simd<T, A>::size())>>
Vc_ALWAYS_INLINE std::tuple<simd<T, simd_abi::deduce_t<T, Sizes>>...> split(
    const simd<T, A> &x)
{
    using SL = detail::size_list<Sizes...>;
    using Tuple = std::tuple<__deduced_simd<T, Sizes>...>;
    constexpr size_t N = simd_size_v<T, A>;
    constexpr size_t N0 = SL::template at<0>();
    using V = __deduced_simd<T, N0>;

    if constexpr (N == N0) {
        static_assert(sizeof...(Sizes) == 1);
        return {simd_cast<V>(x)};
    } else if constexpr (detail::is_fixed_size_abi_v<A> &&
                         detail::fixed_size_storage<T, N>::first_size_v == N0) {
        // if the first part of the simd_tuple input matches the first output vector
        // in the std::tuple, extract it and recurse
        static_assert(!detail::is_fixed_size_abi_v<typename V::abi_type>,
                      "How can <T, N> be a single simd_tuple entry but a fixed_size_simd "
                      "when deduced?");
        const detail::fixed_size_storage<T, N> &xx = detail::data(x);
        return std::tuple_cat(
            std::make_tuple(V(detail::private_init, xx.first)),
            detail::split_wrapper(SL::template pop_front<1>(), xx.second));
    } else if constexpr ((!std::is_same_v<simd_abi::scalar,
                                          simd_abi::deduce_t<T, Sizes>> &&
                          ...) &&
                         (!detail::is_fixed_size_abi_v<simd_abi::deduce_t<T, Sizes>> &&
                          ...)) {
        if constexpr (((Sizes * 2 == N)&&...)) {
            return {{detail::private_init, detail::extract_part<0, 2>(detail::data(x))},
                    {detail::private_init, detail::extract_part<1, 2>(detail::data(x))}};
        } else if constexpr (std::is_same_v<detail::size_list<Sizes...>,
                                            detail::size_list<N / 3, N / 3, N / 3>>) {
            return {{detail::private_init, detail::extract_part<0, 3>(detail::data(x))},
                    {detail::private_init, detail::extract_part<1, 3>(detail::data(x))},
                    {detail::private_init, detail::extract_part<2, 3>(detail::data(x))}};
        } else if constexpr (std::is_same_v<detail::size_list<Sizes...>,
                                            detail::size_list<2 * N / 3, N / 3>>) {
            return {{detail::private_init,
                     detail::concat(detail::extract_part<0, 3>(detail::data(x)),
                                    detail::extract_part<1, 3>(detail::data(x)))},
                    {detail::private_init, detail::extract_part<2, 3>(detail::data(x))}};
        } else if constexpr (std::is_same_v<detail::size_list<Sizes...>,
                                            detail::size_list<N / 3, 2 * N / 3>>) {
            return {{detail::private_init, detail::extract_part<0, 3>(detail::data(x))},
                    {detail::private_init,
                     detail::concat(detail::extract_part<1, 3>(detail::data(x)),
                                    detail::extract_part<2, 3>(detail::data(x)))}};
        } else if constexpr (std::is_same_v<detail::size_list<Sizes...>,
                                            detail::size_list<N / 2, N / 4, N / 4>>) {
            return {{detail::private_init, detail::extract_part<0, 2>(detail::data(x))},
                    {detail::private_init, detail::extract_part<2, 4>(detail::data(x))},
                    {detail::private_init, detail::extract_part<3, 4>(detail::data(x))}};
        } else if constexpr (std::is_same_v<detail::size_list<Sizes...>,
                                            detail::size_list<N / 4, N / 4, N / 2>>) {
            return {{detail::private_init, detail::extract_part<0, 4>(detail::data(x))},
                    {detail::private_init, detail::extract_part<1, 4>(detail::data(x))},
                    {detail::private_init, detail::extract_part<1, 2>(detail::data(x))}};
        } else if constexpr (std::is_same_v<detail::size_list<Sizes...>,
                                            detail::size_list<N / 4, N / 2, N / 4>>) {
            return {
                {detail::private_init, detail::extract_part<0, 4>(detail::data(x))},
                {detail::private_init, detail::extract_center(detail::data(x))},
                {detail::private_init, detail::extract_part<3, 4>(detail::data(x))}};
        } else if constexpr (((Sizes * 4 == N) && ...)) {
            return {{detail::private_init, detail::extract_part<0, 4>(detail::data(x))},
                    {detail::private_init, detail::extract_part<1, 4>(detail::data(x))},
                    {detail::private_init, detail::extract_part<2, 4>(detail::data(x))},
                    {detail::private_init, detail::extract_part<3, 4>(detail::data(x))}};
        //} else if constexpr (detail::is_fixed_size_abi_v<A>) {
        } else {
            detail::assert_unreachable<T>();
        }
    } else {
#ifdef Vc_USE_ALIASING_LOADS
        const detail::may_alias<T> *const element_ptr =
            reinterpret_cast<const detail::may_alias<T> *>(&x);
        return detail::generate_from_n_evaluations<sizeof...(Sizes), Tuple>([&](auto i) {
            using Vi = __deduced_simd<T, SL::at(i)>;
            constexpr size_t offset = SL::before(i);
            constexpr size_t base_align = alignof(simd<T, A>);
            constexpr size_t a = base_align - ((offset * sizeof(T)) % base_align);
            constexpr size_t b = ((a - 1) & a) ^ a;
            constexpr size_t alignment = b == 0 ? a : b;
            return Vi(element_ptr + offset, overaligned<alignment>);
        });
#else
        return detail::generate_from_n_evaluations<sizeof...(Sizes), Tuple>([&](auto i) {
            using Vi = __deduced_simd<T, SL::at(i)>;
            const auto &xx = detail::data(x);
            using Offset = decltype(SL::before(i));
            return Vi([&](auto j) {
                constexpr detail::size_constant<Offset::value + j> k;
                return xx[k];
            });
        });
#endif
    }
}

// }}}

namespace detail
{
// subscript_in_pack {{{
template <size_t I, class T, class A, class... As>
Vc_INTRINSIC constexpr T subscript_in_pack(const simd<T, A> &x, const simd<T, As> &... xs)
{
    if constexpr (I < simd_size_v<T, A>) {
        return x[I];
    } else {
        return subscript_in_pack<I - simd_size_v<T, A>>(xs...);
    }
}
// }}}
}  // namespace detail

// concat(simd...) {{{
template <class T, class... As>
simd<T, simd_abi::deduce_t<T, (simd_size_v<T, As> + ...)>> concat(
    const simd<T, As> &... xs)
{
    return simd<T, simd_abi::deduce_t<T, (simd_size_v<T, As> + ...)>>(
        [&](auto i) { return detail::subscript_in_pack<i>(xs...); });
}

// }}}

namespace detail {
// smart_reference {{{
template <class U, class Accessor = U, class ValueType = typename U::value_type>
class smart_reference
{
    friend Accessor;
    int index;
    U &obj;

    Vc_INTRINSIC constexpr ValueType read() const noexcept
    {
        if constexpr (std::is_arithmetic_v<U>) {
            Vc_ASSERT(index == 0);
            return obj;
        } else {
            return obj[index];
        }
    }

    template <class T> Vc_INTRINSIC constexpr void write(T &&x) const
    {
        Accessor::set(obj, index, std::forward<T>(x));
    }

public:
    Vc_INTRINSIC smart_reference(U &o, int i) noexcept : index(i), obj(o) {}

    using value_type = ValueType;

    Vc_INTRINSIC smart_reference(const smart_reference &) = delete;

    Vc_INTRINSIC constexpr operator value_type() const noexcept { return read(); }

    template <class T,
              class = detail::value_preserving_or_int<std::decay_t<T>, value_type>>
    Vc_INTRINSIC constexpr smart_reference operator=(T &&x) &&
    {
        write(std::forward<T>(x));
        return {obj, index};
    }

// TODO: improve with operator.()

#define Vc_OP_(op_)                                                                      \
    template <class T,                                                                   \
              class TT = decltype(std::declval<value_type>() op_ std::declval<T>()),     \
              class = detail::value_preserving_or_int<std::decay_t<T>, TT>,              \
              class = detail::value_preserving_or_int<TT, value_type>>                   \
        Vc_INTRINSIC smart_reference operator op_##=(T &&x) &&                           \
    {                                                                                    \
        const value_type &lhs = read();                                                  \
        write(lhs op_ x);                                                                \
        return {obj, index};                                                             \
    }
    Vc_ALL_ARITHMETICS(Vc_OP_);
    Vc_ALL_SHIFTS(Vc_OP_);
    Vc_ALL_BINARY(Vc_OP_);
#undef Vc_OP_

    template <class T = void,
              class = decltype(
                  ++std::declval<std::conditional_t<true, value_type, T> &>())>
    Vc_INTRINSIC smart_reference operator++() &&
    {
        value_type x = read();
        write(++x);
        return {obj, index};
    }

    template <class T = void,
              class = decltype(
                  std::declval<std::conditional_t<true, value_type, T> &>()++)>
    Vc_INTRINSIC value_type operator++(int) &&
    {
        const value_type r = read();
        value_type x = r;
        write(++x);
        return r;
    }

    template <class T = void,
              class = decltype(
                  --std::declval<std::conditional_t<true, value_type, T> &>())>
    Vc_INTRINSIC smart_reference operator--() &&
    {
        value_type x = read();
        write(--x);
        return {obj, index};
    }

    template <class T = void,
              class = decltype(
                  std::declval<std::conditional_t<true, value_type, T> &>()--)>
    Vc_INTRINSIC value_type operator--(int) &&
    {
        const value_type r = read();
        value_type x = r;
        write(--x);
        return r;
    }

    Vc_INTRINSIC friend void swap(smart_reference &&a, smart_reference &&b) noexcept(
        all<std::is_nothrow_constructible<value_type, smart_reference &&>,
            std::is_nothrow_assignable<smart_reference &&, value_type &&>>::value)
    {
        value_type tmp = static_cast<smart_reference &&>(a);
        static_cast<smart_reference &&>(a) = static_cast<value_type>(b);
        static_cast<smart_reference &&>(b) = std::move(tmp);
    }

    Vc_INTRINSIC friend void swap(value_type &a, smart_reference &&b) noexcept(
        all<std::is_nothrow_constructible<value_type, value_type &&>,
            std::is_nothrow_assignable<value_type &, value_type &&>,
            std::is_nothrow_assignable<smart_reference &&, value_type &&>>::value)
    {
        value_type tmp(std::move(a));
        a = static_cast<value_type>(b);
        static_cast<smart_reference &&>(b) = std::move(tmp);
    }

    Vc_INTRINSIC friend void swap(smart_reference &&a, value_type &b) noexcept(
        all<std::is_nothrow_constructible<value_type, smart_reference &&>,
            std::is_nothrow_assignable<value_type &, value_type &&>,
            std::is_nothrow_assignable<smart_reference &&, value_type &&>>::value)
    {
        value_type tmp(a);
        static_cast<smart_reference &&>(a) = std::move(b);
        b = std::move(tmp);
    }
};

// }}}
// abi impl fwd decls {{{
struct neon_simd_impl;
struct neon_mask_impl;
struct sse_mask_impl;
struct sse_simd_impl;
struct avx_mask_impl;
struct avx_simd_impl;
struct avx512_mask_impl;
struct avx512_simd_impl;
struct scalar_simd_impl;
struct scalar_mask_impl;
template <int N> struct fixed_size_simd_impl;
template <int N> struct fixed_size_mask_impl;
template <int N, class Abi> struct combine_simd_impl;
template <int N, class Abi> struct combine_mask_impl;

// }}}
// gnu_traits {{{1
template <class T, class MT, class Abi, size_t N> struct gnu_traits {
    using is_valid = std::true_type;
    using simd_impl_type = typename Abi::simd_impl_type;
    using mask_impl_type = typename Abi::mask_impl_type;

    // simd and simd_mask member types {{{2
    using simd_member_type = Storage<T, N>;
    using mask_member_type = Storage<MT, N>;
    static constexpr size_t simd_member_alignment = alignof(simd_member_type);
    static constexpr size_t mask_member_alignment = alignof(mask_member_type);

    // simd_base / base class for simd, providing extra conversions {{{2
    struct simd_base2 {
        explicit operator detail::intrinsic_type_t<T, N>() const
        {
            return static_cast<const simd<T, Abi> *>(this)->d.v();
        }
        explicit operator detail::builtin_type_t<T, N>() const
        {
            return static_cast<const simd<T, Abi> *>(this)->d.builtin();
        }
    };
    struct simd_base1 {
        explicit operator detail::intrinsic_type_t<T, N>() const
        {
            return detail::data(*static_cast<const simd<T, Abi> *>(this));
        }
    };
    using simd_base = std::conditional_t<
        std::is_same<detail::intrinsic_type_t<T, N>, detail::builtin_type_t<T, N>>::value,
        simd_base1, simd_base2>;

    // mask_base {{{2
    struct mask_base2 {
        explicit operator detail::intrinsic_type_t<T, N>() const
        {
            return static_cast<const simd_mask<T, Abi> *>(this)->d.intrin();
        }
        explicit operator detail::builtin_type_t<T, N>() const
        {
            return static_cast<const simd_mask<T, Abi> *>(this)->d.d;
        }
    };
    struct mask_base1 {
        explicit operator detail::intrinsic_type_t<T, N>() const
        {
            return detail::data(*static_cast<const simd_mask<T, Abi> *>(this));
        }
    };
    using mask_base = std::conditional_t<
        std::is_same<detail::intrinsic_type_t<T, N>, detail::builtin_type_t<T, N>>::value,
        mask_base1, mask_base2>;

    // mask_cast_type {{{2
    // parameter type of one explicit simd_mask constructor
    class mask_cast_type
    {
        using U = detail::intrinsic_type_t<T, N>;
        U d;

    public:
        mask_cast_type(U x) : d(x) {}
        operator mask_member_type() const { return d; }
    };

    // simd_cast_type {{{2
    // parameter type of one explicit simd constructor
    class simd_cast_type1
    {
        using A = detail::intrinsic_type_t<T, N>;
        A d;

    public:
        simd_cast_type1(A a) : d(a) {}
        //simd_cast_type1(simd_member_type x) : d(x) {}
        operator simd_member_type() const { return d; }
    };

    class simd_cast_type2
    {
        using A = detail::intrinsic_type_t<T, N>;
        using B = detail::builtin_type_t<T, N>;
        A d;

    public:
        simd_cast_type2(A a) : d(a) {}
        simd_cast_type2(B b) : d(detail::intrin_cast<A>(b)) {}
        //simd_cast_type2(simd_member_type x) : d(x) {}
        operator simd_member_type() const { return d; }
    };

    using simd_cast_type = std::conditional_t<
        std::is_same<detail::intrinsic_type_t<T, N>, detail::builtin_type_t<T, N>>::value,
        simd_cast_type1, simd_cast_type2>;
    //}}}2
};

// neon_is_vectorizable {{{1
#ifdef Vc_HAVE_NEON_ABI
template <class T> struct neon_is_vectorizable : detail::is_vectorizable<T> {};
template <> struct neon_is_vectorizable<long double> : std::false_type {};
#ifndef Vc_HAVE_FULL_NEON_ABI
template <> struct neon_is_vectorizable<double> : std::false_type {};
#endif
#else
template <class T> struct neon_is_vectorizable : std::false_type {};
#endif

// sse_is_vectorizable {{{1
#ifdef Vc_HAVE_FULL_SSE_ABI
template <class T> struct sse_is_vectorizable : detail::is_vectorizable<T> {};
template <> struct sse_is_vectorizable<long double> : std::false_type {};
#elif defined Vc_HAVE_SSE_ABI
template <class T> struct sse_is_vectorizable : detail::is_same<T, float> {};
#else
template <class T> struct sse_is_vectorizable : std::false_type {};
#endif

// avx_is_vectorizable {{{1
#ifdef Vc_HAVE_FULL_AVX_ABI
template <class T> struct avx_is_vectorizable : detail::is_vectorizable<T> {};
#elif defined Vc_HAVE_AVX_ABI
template <class T> struct avx_is_vectorizable : std::is_floating_point<T> {};
#else
template <class T> struct avx_is_vectorizable : std::false_type {};
#endif
template <> struct avx_is_vectorizable<long double> : std::false_type {};

// avx512_is_vectorizable {{{1
#ifdef Vc_HAVE_AVX512_ABI
template <class T> struct avx512_is_vectorizable : detail::is_vectorizable<T> {};
template <> struct avx512_is_vectorizable<long double> : std::false_type {};
#ifndef Vc_HAVE_FULL_AVX512_ABI
template <> struct avx512_is_vectorizable<  char> : std::false_type {};
template <> struct avx512_is_vectorizable< uchar> : std::false_type {};
template <> struct avx512_is_vectorizable< schar> : std::false_type {};
template <> struct avx512_is_vectorizable< short> : std::false_type {};
template <> struct avx512_is_vectorizable<ushort> : std::false_type {};
template <> struct avx512_is_vectorizable<char16_t> : std::false_type {};
template <> struct avx512_is_vectorizable<wchar_t> : detail::bool_constant<sizeof(wchar_t) >= 4> {};
#endif
#else
template <class T> struct avx512_is_vectorizable : std::false_type {};
#endif

// }}}
// implicit_mask_abi_base {{{
template <int Bytes, class Abi> struct implicit_mask_abi_base {
    template <class T>
    using implicit_mask_type =
        detail::builtin_type_t<detail::int_for_sizeof_t<T>, simd_size_v<T, Abi>>;

    template <class T>
    static constexpr auto implicit_mask =
        reinterpret_cast<builtin_type_t<T, simd_size_v<T, Abi>>>(
            Abi::is_partial ? detail::generate_builtin<implicit_mask_type<T>>([](auto i) {
                return i < Bytes / sizeof(T) ? -1 : 0;
            })
                            : ~implicit_mask_type<T>());

    template <class T, class Trait = detail::builtin_traits<T>>
    static constexpr auto masked(T x)
    {
        using U = typename Trait::value_type;
        if constexpr (Abi::is_partial) {
            return and_(x , implicit_mask<U>);
        } else {
            return x;
        }
    }
};

// }}}
}  // namespace detail

namespace simd_abi
{
// __combine {{{1
template <int N, class Abi> struct __combine {
    template <class T> static constexpr size_t size = N *Abi::template size<T>;
    template <class T> static constexpr size_t full_size = size<T>;

    static constexpr int factor = N;
    using member_abi = Abi;

    // validity traits {{{2
    // allow 2x, 3x, and 4x "unroll"
    struct is_valid_abi_tag
        : detail::bool_constant<(N > 1 && N <= 4) && Abi::is_valid_abi_tag> {
    };
    template <class T> struct is_valid_size_for : Abi::template is_valid_size_for<T> {
    };
    template <class T>
    struct is_valid : detail::all<is_valid_abi_tag, typename Abi::template is_valid<T>> {
    };
    template <class T> static constexpr bool is_valid_v = is_valid<T>::value;

    // simd/mask_impl_type {{{2
    using simd_impl_type = detail::combine_simd_impl<N, Abi>;
    using mask_impl_type = detail::combine_mask_impl<N, Abi>;

    // traits {{{2
    template <class T, bool = is_valid_v<T>> struct traits : detail::invalid_traits {
    };

    template <class T> struct traits<T, true> {
        using is_valid = std::true_type;
        using simd_impl_type = detail::combine_simd_impl<N, Abi>;
        using mask_impl_type = detail::combine_mask_impl<N, Abi>;

        // simd and simd_mask member types {{{2
        using simd_member_type =
            std::array<typename Abi::template traits<T>::simd_member_type, N>;
        using mask_member_type =
            std::array<typename Abi::template traits<T>::mask_member_type, N>;
        static constexpr size_t simd_member_alignment =
            Abi::template traits<T>::simd_member_alignment;
        static constexpr size_t mask_member_alignment =
            Abi::template traits<T>::mask_member_alignment;

        // simd_base / base class for simd, providing extra conversions {{{2
        struct simd_base {
            explicit operator const simd_member_type &() const
            {
                return static_cast<const simd<T, __combine> *>(this)->d;
            }
        };

        // mask_base {{{2
        // empty. The std::bitset interface suffices
        struct mask_base {
            explicit operator const mask_member_type &() const
            {
                return static_cast<const simd_mask<T, __combine> *>(this)->d;
            }
        };

        // simd_cast_type {{{2
        struct simd_cast_type {
            simd_cast_type(const simd_member_type &dd) : d(dd) {}
            explicit operator const simd_member_type &() const { return d; }

        private:
            const simd_member_type &d;
        };

        // mask_cast_type {{{2
        struct mask_cast_type {
            mask_cast_type(const mask_member_type &dd) : d(dd) {}
            explicit operator const mask_member_type &() const { return d; }

        private:
            const mask_member_type &d;
        };
        //}}}2
    };
    //}}}2
};
// __neon_abi {{{1
template <int Bytes>
struct __neon_abi : detail::implicit_mask_abi_base<Bytes, __neon_abi<Bytes>> {
    template <class T> static constexpr size_t size = Bytes / sizeof(T);
    template <class T> static constexpr size_t full_size = 16 / sizeof(T);
    static constexpr bool is_partial = Bytes < 16;

    // validity traits {{{2
    struct is_valid_abi_tag : detail::bool_constant<(Bytes > 0 && Bytes <= 16)> {
    };
    template <class T>
    struct is_valid_size_for
        : detail::bool_constant<(Bytes / sizeof(T) > 1 && Bytes % sizeof(T) == 0)> {
    };
    template <class T>
    struct is_valid : detail::all<is_valid_abi_tag, detail::neon_is_vectorizable<T>,
                                  is_valid_size_for<T>> {
    };
    template <class T> static constexpr bool is_valid_v = is_valid<T>::value;

    // simd/mask_impl_type {{{2
    using simd_impl_type = detail::neon_simd_impl;
    using mask_impl_type = detail::neon_mask_impl;

    // traits {{{2
    template <class T>
    using traits = std::conditional_t<is_valid_v<T>,
                                      detail::gnu_traits<T, T, __neon_abi, full_size<T>>,
                                      detail::invalid_traits>;
    //}}}2
};

// __sse_abi {{{1
template <int Bytes>
struct __sse_abi : detail::implicit_mask_abi_base<Bytes, __sse_abi<Bytes>> {
    template <class T> static constexpr size_t size = Bytes / sizeof(T);
    template <class T> static constexpr size_t full_size = 16 / sizeof(T);
    static constexpr bool is_partial = Bytes < 16;

    // validity traits {{{2
    // allow 2x, 3x, and 4x "unroll"
    struct is_valid_abi_tag : detail::bool_constant<Bytes == 16> {};
    //struct is_valid_abi_tag : detail::bool_constant<(Bytes > 0 && Bytes <= 16)> {};
    template <class T>
    struct is_valid_size_for
        : detail::bool_constant<(Bytes / sizeof(T) > 1 && Bytes % sizeof(T) == 0)> {
    };

    template <class T>
    struct is_valid
        : detail::all<is_valid_abi_tag, detail::sse_is_vectorizable<T>, is_valid_size_for<T>> {
    };
    template <class T> static constexpr bool is_valid_v = is_valid<T>::value;

    // simd/mask_impl_type {{{2
    using simd_impl_type = detail::sse_simd_impl;
    using mask_impl_type = detail::sse_mask_impl;

    // traits {{{2
    template <class T>
    using traits = std::conditional_t<is_valid_v<T>,
                                      detail::gnu_traits<T, T, __sse_abi, full_size<T>>,
                                      detail::invalid_traits>;
    //}}}2
};

// __avx_abi {{{1
template <int Bytes>
struct __avx_abi : detail::implicit_mask_abi_base<Bytes, __avx_abi<Bytes>> {
    template <class T> static constexpr size_t size = Bytes / sizeof(T);
    template <class T> static constexpr size_t full_size = 32 / sizeof(T);
    static constexpr bool is_partial = Bytes < 32;

    // validity traits {{{2
    // - allow 2x, 3x, and 4x "unroll"
    // - disallow <= 16 Bytes as that's covered by __sse_abi
    struct is_valid_abi_tag : detail::bool_constant<Bytes == 32> {};
    /* TODO:
    struct is_valid_abi_tag
        : detail::bool_constant<((Bytes > 16 && Bytes <= 32) || Bytes == 64 ||
                                 Bytes == 96 || Bytes == 128)> {
    };
    */
    template <class T>
    struct is_valid_size_for : detail::bool_constant<(Bytes % sizeof(T) == 0)> {
    };
    template <class T>
    struct is_valid
        : detail::all<is_valid_abi_tag, detail::avx_is_vectorizable<T>, is_valid_size_for<T>> {
    };
    template <class T> static constexpr bool is_valid_v = is_valid<T>::value;

    // simd/mask_impl_type {{{2
    using simd_impl_type = detail::avx_simd_impl;
    using mask_impl_type = detail::avx_mask_impl;

    // traits {{{2
    template <class T>
    using traits = std::conditional_t<is_valid_v<T>,
                                      detail::gnu_traits<T, T, __avx_abi, full_size<T>>,
                                      detail::invalid_traits>;
    //}}}2
};

// __avx512_abi {{{1
template <int Bytes> struct __avx512_abi {
    template <class T> static constexpr size_t size = Bytes / sizeof(T);
    template <class T> static constexpr size_t full_size = 64 / sizeof(T);
    static constexpr bool is_partial = Bytes < 64;

    // validity traits {{{2
    // - disallow <= 32 Bytes as that's covered by __sse_abi and __avx_abi
    // TODO: consider AVX512VL
    struct is_valid_abi_tag : detail::bool_constant<Bytes == 64> {};
    /* TODO:
    struct is_valid_abi_tag
        : detail::bool_constant<(Bytes > 32 && Bytes <= 64)> {
    };
    */
    template <class T>
    struct is_valid_size_for : detail::bool_constant<(Bytes % sizeof(T) == 0)> {
    };
    template <class T>
    struct is_valid
        : detail::all<is_valid_abi_tag, detail::avx512_is_vectorizable<T>, is_valid_size_for<T>> {
    };
    template <class T> static constexpr bool is_valid_v = is_valid<T>::value;

    // implicit mask {{{2
    template <class T>
    using implicit_mask_type = detail::bool_storage_member_type_t<64 / sizeof(T)>;

    template <class T>
    static constexpr implicit_mask_type<T> implicit_mask =
        Bytes == 64 ? ~implicit_mask_type<T>()
                    : (implicit_mask_type<T>(1) << (Bytes / sizeof(T))) - 1;

    template <class T, class = std::enable_if_t<detail::is_bitmask_v<T>>>
    static constexpr T masked(T x)
    {
        if constexpr (is_partial) {
            constexpr size_t N = sizeof(T) * 8;
            return x &
                   ((detail::bool_storage_member_type_t<N>(1) << (Bytes * N / 64)) - 1);
        } else {
            return x;
        }
    }

    // simd/mask_impl_type {{{2
    using simd_impl_type = detail::avx512_simd_impl;
    using mask_impl_type = detail::avx512_mask_impl;

    // traits {{{2
    template <class T>
    using traits =
        std::conditional_t<is_valid_v<T>,
                           detail::gnu_traits<T, bool, __avx512_abi, full_size<T>>,
                           detail::invalid_traits>;
    //}}}2
};

// __scalar_abi {{{1
struct __scalar_abi {
    template <class T> static constexpr size_t size = 1;
    template <class T> static constexpr size_t full_size = 1;
    struct is_valid_abi_tag : std::true_type {};
    template <class T> struct is_valid_size_for : std::true_type {};
    template <class T> struct is_valid : detail::is_vectorizable<T> {};
    template <class T> static constexpr bool is_valid_v = is_valid<T>::value;

    using simd_impl_type = detail::scalar_simd_impl;
    using mask_impl_type = detail::scalar_mask_impl;

    template <class T, bool = is_valid_v<T>> struct traits : detail::invalid_traits {
    };

    template <class T> struct traits<T, true> {
        using is_valid = std::true_type;
        using simd_impl_type = detail::scalar_simd_impl;
        using mask_impl_type = detail::scalar_mask_impl;
        using simd_member_type = T;
        using mask_member_type = bool;
        static constexpr size_t simd_member_alignment = alignof(simd_member_type);
        static constexpr size_t mask_member_alignment = alignof(mask_member_type);

        // nothing the user can spell converts to/from simd/simd_mask
        struct simd_cast_type {
            simd_cast_type() = delete;
        };
        struct mask_cast_type {
            mask_cast_type() = delete;
        };
        struct simd_base {};
        struct mask_base {};
    };
};

// __fixed_abi {{{1
template <int N> struct __fixed_abi {
    template <class T> static constexpr size_t size = N;
    template <class T> static constexpr size_t full_size = N;
    // validity traits {{{2
    struct is_valid_abi_tag
        : public detail::bool_constant<(N > 0)> {
    };
    template <class T>
    struct is_valid_size_for
        : detail::bool_constant<((N <= simd_abi::max_fixed_size<T>) ||
                                 (simd_abi::__neon::is_valid_v<char> &&
                                  N == simd_size_v<char, simd_abi::__neon>) ||
                                 (simd_abi::__sse::is_valid_v<char> &&
                                  N == simd_size_v<char, simd_abi::__sse>) ||
                                 (simd_abi::__avx::is_valid_v<char> &&
                                  N == simd_size_v<char, simd_abi::__avx>) ||
                                 (simd_abi::__avx512::is_valid_v<char> &&
                                  N == simd_size_v<char, simd_abi::__avx512>))> {
    };
    template <class T>
    struct is_valid
        : detail::all<is_valid_abi_tag, detail::is_vectorizable<T>, is_valid_size_for<T>> {
    };
    template <class T> static constexpr bool is_valid_v = is_valid<T>::value;

    // simd/mask_impl_type {{{2
    using simd_impl_type = detail::fixed_size_simd_impl<N>;
    using mask_impl_type = detail::fixed_size_mask_impl<N>;

    // traits {{{2
    template <class T, bool = is_valid_v<T>> struct traits : detail::invalid_traits {
    };

    template <class T> struct traits<T, true> {
        using is_valid = std::true_type;
        using simd_impl_type = detail::fixed_size_simd_impl<N>;
        using mask_impl_type = detail::fixed_size_mask_impl<N>;

        // simd and simd_mask member types {{{2
        using simd_member_type = detail::fixed_size_storage<T, N>;
        using mask_member_type = std::bitset<N>;
        static constexpr size_t simd_member_alignment =
#ifdef Vc_GCC
            std::min(size_t(
#ifdef __AVX__
                         256
#else
                         128
#endif
                         ),
#else
            (
#endif
                     detail::next_power_of_2(N * sizeof(T)));
        static constexpr size_t mask_member_alignment = alignof(mask_member_type);

        // simd_base / base class for simd, providing extra conversions {{{2
        struct simd_base {
            // The following ensures, function arguments are passed via the stack. This is
            // important for ABI compatibility across TU boundaries
            simd_base(const simd_base &) {}
            simd_base() = default;

            explicit operator const simd_member_type &() const
            {
                return static_cast<const simd<T, __fixed_abi> *>(this)->d;
            }
            explicit operator std::array<T, N>() const
            {
                std::array<T, N> r;
                // simd_member_type can be larger because of higher alignment
                static_assert(sizeof(r) <= sizeof(simd_member_type), "");
                std::memcpy(r.data(), &static_cast<const simd_member_type &>(*this),
                            sizeof(r));
                return r;
            }
        };

        // mask_base {{{2
        // empty. The std::bitset interface suffices
        struct mask_base {};

        // simd_cast_type {{{2
        struct simd_cast_type {
            simd_cast_type(const std::array<T, N> &);
            simd_cast_type(const simd_member_type &dd) : d(dd) {}
            explicit operator const simd_member_type &() const { return d; }

        private:
            const simd_member_type &d;
        };

        // mask_cast_type {{{2
        class mask_cast_type
        {
            mask_cast_type() = delete;
        };
        //}}}2
    };
};

//}}}
}  // namespace simd_abi

namespace detail
{
// scalar_abi_wrapper {{{1
template <int Bytes> struct scalar_abi_wrapper : simd_abi::__scalar_abi {
    template <class T>
    static constexpr bool is_valid_v = simd_abi::__scalar_abi::is_valid<T>::value &&
                                       sizeof(T) == Bytes;
};

// decay_abi metafunction {{{1
template <class T> struct decay_abi {
    using type = T;
};
template <int Bytes> struct decay_abi<scalar_abi_wrapper<Bytes>> {
    using type = simd_abi::scalar;
};

// full_abi metafunction {{{1
template <template <int> class ATemp> struct full_abi;
template <> struct full_abi<simd_abi::__neon_abi> { using type = simd_abi::__neon128; };
template <> struct full_abi<simd_abi::__sse_abi> { using type = simd_abi::__sse; };
template <> struct full_abi<simd_abi::__avx_abi> { using type = simd_abi::__avx; };
template <> struct full_abi<simd_abi::__avx512_abi> { using type = simd_abi::__avx512; };
template <> struct full_abi<scalar_abi_wrapper> {
    using type = simd_abi::scalar;
};

// abi_list {{{1
template <template <int> class...> struct abi_list {
    template <class, int> static constexpr bool has_valid_abi = false;
    template <class, int> using first_valid_abi = void;
    template <class, int> using best_abi = void;
};

template <template <int> class A0, template <int> class... Rest>
struct abi_list<A0, Rest...> {
    template <class T, int N>
    static constexpr bool has_valid_abi = A0<sizeof(T) * N>::template is_valid_v<T> ||
                                          abi_list<Rest...>::template has_valid_abi<T, N>;
    template <class T, int N>
    using first_valid_abi =
        std::conditional_t<A0<sizeof(T) * N>::template is_valid_v<T>,
                           typename decay_abi<A0<sizeof(T) * N>>::type,
                           typename abi_list<Rest...>::template first_valid_abi<T, N>>;
    using B = typename full_abi<A0>::type;
    template <class T, int N>
    using best_abi = std::conditional_t<
        A0<sizeof(T) * N>::template is_valid_v<T>,
        typename decay_abi<A0<sizeof(T) * N>>::type,
        std::conditional_t<(B::template is_valid_v<T> && B::template size<T> <= N), B,
                           typename abi_list<Rest...>::template best_abi<T, N>>>;
};

// }}}1

// the following lists all native ABIs, which makes them accessible to simd_abi::deduce
// and select_best_vector_type_t (for fixed_size). Order matters: Whatever comes first has
// higher priority.
using all_native_abis =
    abi_list<simd_abi::__avx512_abi, simd_abi::__avx_abi, simd_abi::__sse_abi,
             simd_abi::__neon_abi, scalar_abi_wrapper>;

// valid traits specialization {{{1
template <class T, class Abi>
struct traits<T, Abi, std::void_t<typename Abi::template is_valid<T>>>
    : Abi::template traits<T> {
};

// deduce_impl specializations {{{1
// try all native ABIs (including scalar) first
template <class T, std::size_t N>
struct deduce_impl<T, N,
                   std::enable_if_t<all_native_abis::template has_valid_abi<T, N>>> {
    using type = all_native_abis::first_valid_abi<T, N>;
};

// fall back to fixed_size only if scalar and native ABIs don't match
template <class T, std::size_t N, class = void> struct deduce_fixed_size_fallback {};
template <class T, std::size_t N>
struct deduce_fixed_size_fallback<
    T, N, std::enable_if_t<simd_abi::fixed_size<N>::template is_valid_v<T>>> {
    using type = simd_abi::fixed_size<N>;
};
template <class T, std::size_t N, class>
struct deduce_impl : public deduce_fixed_size_fallback<T, N> {
};

//}}}1
// is_abi {{{
template <template <int> class Abi, int Bytes> constexpr int abi_bytes_impl(Abi<Bytes> *)
{
    return Bytes;
}
template <class T> constexpr int abi_bytes_impl(T *) { return -1; }
template <class Abi>
inline constexpr int abi_bytes = abi_bytes_impl(static_cast<Abi *>(nullptr));

template <class Abi0, class Abi1> constexpr bool is_abi()
{
    return std::is_same_v<Abi0, Abi1>;
}
template <template <int> class Abi0, class Abi1> constexpr bool is_abi()
{
    return std::is_same_v<Abi0<abi_bytes<Abi1>>, Abi1>;
}
template <class Abi0, template <int> class Abi1> constexpr bool is_abi()
{
    return std::is_same_v<Abi1<abi_bytes<Abi0>>, Abi0>;
}
template <template <int> class Abi0, template <int> class Abi1> constexpr bool is_abi()
{
    return std::is_same_v<Abi0<0>, Abi1<0>>;
}

// }}}
// is_combined_abi{{{
template <template <int, class> class Combine, int N, class Abi>
constexpr bool is_combined_abi(Combine<N, Abi> *)
{
    return std::is_same_v<Combine<N, Abi>, simd_abi::__combine<N, Abi>>;
}
template <class Abi> constexpr bool is_combined_abi(Abi *)
{
    return false;
}

template <class Abi> constexpr bool is_combined_abi()
{
    return is_combined_abi(static_cast<Abi *>(nullptr));
}

// }}}
}  // namespace detail

// simd_mask {{{
template <class T, class Abi> class simd_mask : public detail::traits<T, Abi>::mask_base
{
    // types, tags, and friends {{{
    using traits = detail::traits<T, Abi>;
    using impl = typename traits::mask_impl_type;
    using member_type = typename traits::mask_member_type;
    static constexpr T *type_tag = nullptr;
    friend typename traits::mask_base;
    friend class simd<T, Abi>;  // to construct masks on return
    friend impl;
    friend typename traits::simd_impl_type;  // to construct masks on return and
                                             // inspect data on masked operations
    // }}}
    // is_<abi> {{{
    static constexpr bool is_scalar() { return detail::is_abi<Abi, simd_abi::scalar>(); }
    static constexpr bool is_sse() { return detail::is_abi<Abi, simd_abi::__sse_abi>(); }
    static constexpr bool is_avx() { return detail::is_abi<Abi, simd_abi::__avx_abi>(); }
    static constexpr bool is_avx512()
    {
        return detail::is_abi<Abi, simd_abi::__avx512_abi>();
    }
    static constexpr bool is_neon()
    {
        return detail::is_abi<Abi, simd_abi::__neon_abi>();
    }
    static constexpr bool is_fixed() { return detail::is_fixed_size_abi_v<Abi>; }
    static constexpr bool is_combined() { return detail::is_combined_abi<Abi>(); }

    // }}}

public:
    // member types {{{
    using value_type = bool;
    using reference = detail::smart_reference<member_type, impl, value_type>;
    using simd_type = simd<T, Abi>;
    using abi_type = Abi;

    // }}}
    static constexpr size_t size() { return detail::size_or_zero<T, Abi>; }
    // constructors & assignment {{{
    simd_mask() = default;
    simd_mask(const simd_mask &) = default;
    simd_mask(simd_mask &&) = default;
    simd_mask &operator=(const simd_mask &) = default;
    simd_mask &operator=(simd_mask &&) = default;

    // }}}

    // access to internal representation (suggested extension)
    Vc_ALWAYS_INLINE explicit simd_mask(typename traits::mask_cast_type init) : d{init} {}
    // conversions to internal type is done in mask_base

    // bitset interface (extension) {{{
    Vc_ALWAYS_INLINE static simd_mask from_bitset(std::bitset<size()> bs)
    {
        return {detail::bitset_init, bs};
    }
    Vc_ALWAYS_INLINE std::bitset<size()> to_bitset() const {
        if constexpr (is_scalar()) {
            return unsigned(d);
        } else if constexpr (is_fixed()) {
            return d;
        } else {
            return detail::to_bitset(builtin());
        }
    }

    // }}}
    // explicit broadcast constructor {{{
    Vc_ALWAYS_INLINE explicit constexpr simd_mask(value_type x) : d(broadcast(x)) {}

    // }}}
    // implicit type conversion constructor {{{
    template <class U>
    Vc_ALWAYS_INLINE simd_mask(
        const simd_mask<U, simd_abi::fixed_size<size()>> &x,
        std::enable_if_t<detail::all<std::is_same<abi_type, simd_abi::fixed_size<size()>>,
                                     std::is_same<U, U>>::value,
                         detail::nullarg_t> = detail::nullarg)
        : simd_mask{detail::bitset_init, detail::data(x)}
    {
    }
    // }}}
    /* reference implementation for explicit simd_mask casts {{{
    template <class U>
    simd_mask(const simd_mask<U, Abi> &x,
         enable_if<
             (size() == simd_mask<U, Abi>::size()) &&
             detail::all<std::is_integral<T>, std::is_integral<U>,
             detail::negation<std::is_same<Abi, simd_abi::fixed_size<size()>>>,
             detail::negation<std::is_same<T, U>>>::value> = nullarg)
        : d{x.d}
    {
    }
    template <class U, class Abi2>
    simd_mask(const simd_mask<U, Abi2> &x,
         enable_if<detail::all<
         detail::negation<std::is_same<abi_type, Abi2>>,
             std::is_same<abi_type, simd_abi::fixed_size<size()>>>::value> = nullarg)
    {
        x.copy_to(&d[0], vector_aligned);
    }
    }}} */

    // load impl {{{
private:
    template <class F>
    Vc_INTRINSIC static member_type load_wrapper(const value_type *mem, F f)
    {
        if constexpr (is_scalar()) {
            return mem[0];
        } else if constexpr (is_fixed()) {
            const fixed_size_simd<unsigned char, size()> bools(
                reinterpret_cast<const detail::may_alias<unsigned char> *>(mem), f);
            return detail::data(bools != 0);
        } else if constexpr (is_sse()) {
            if constexpr (size() == 2 && detail::have_sse2) {
                return detail::to_storage(_mm_set_epi32(-int(mem[1]), -int(mem[1]),
                                                        -int(mem[0]), -int(mem[0])));
            } else if constexpr (size() == 4 && detail::have_sse2) {
                __m128i k = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(mem));
                k = _mm_cmpgt_epi16(_mm_unpacklo_epi8(k, k), _mm_setzero_si128());
                return detail::to_storage(_mm_unpacklo_epi16(k, k));
            } else if constexpr (size() == 4 && detail::have_mmx) {
                __m128 k =
                    _mm_cvtpi8_ps(_mm_cvtsi32_si64(*reinterpret_cast<const int *>(mem)));
                _mm_empty();
                return detail::to_storage(_mm_cmpgt_ps(k, __m128()));
            } else if constexpr (size() == 8 && detail::have_sse2) {
                const auto k = detail::make_builtin<long long>(
                    *reinterpret_cast<const detail::may_alias<long long> *>(mem), 0);
                if constexpr (detail::have_sse2) {
                    return detail::to_storage(
                        detail::builtin_cast<short>(_mm_unpacklo_epi8(k, k)) != 0);
                }
            } else if constexpr (size() == 16 && detail::have_sse2) {
                return _mm_cmpgt_epi8(detail::builtin_load<long long, 2>(mem, f),
                                      __m128i());
            } else {
                detail::assert_unreachable<F>();
            }
        } else if constexpr (is_avx()) {
            if constexpr (size() == 4 && detail::have_avx) {
                int bool4;
                if constexpr (detail::is_aligned_v<F, 4>) {
                    bool4 = *reinterpret_cast<const detail::may_alias<int> *>(mem);
                } else {
                    std::memcpy(&bool4, mem, 4);
                }
                const auto k = detail::to_intrin(
                    (detail::builtin_broadcast<4>(bool4) &
                     detail::make_builtin<int>(0x1, 0x100, 0x10000, 0x1000000)) != 0);
                return detail::to_storage(
                    detail::concat(_mm_unpacklo_epi32(k, k), _mm_unpackhi_epi32(k, k)));
            } else if constexpr (size() == 8 && detail::have_avx) {
                auto k = detail::builtin_load<long long, 2, 8>(mem, f);
                k = _mm_cmpgt_epi16(_mm_unpacklo_epi8(k, k), __m128i());
                return detail::to_storage(
                    detail::concat(_mm_unpacklo_epi16(k, k), _mm_unpackhi_epi16(k, k)));
            } else if constexpr (size() == 16 && detail::have_avx) {
                const auto k =
                    _mm_cmpgt_epi8(detail::builtin_load<long long, 2>(mem, f), __m128i());
                return detail::concat(_mm_unpacklo_epi8(k, k), _mm_unpackhi_epi8(k, k));
            } else if constexpr (size() == 32 && detail::have_avx2) {
                return _mm256_cmpgt_epi8(detail::builtin_load<long long, 4>(mem, f),
                                         __m256i());
            } else {
                detail::assert_unreachable<F>();
            }
        } else if constexpr (is_avx512()) {
            if constexpr (size() == 8) {
                const auto a = detail::builtin_load<long long, 2, 8>(mem, f);
                if constexpr (detail::have_avx512bw_vl) {
                    return _mm_test_epi8_mask(a, a);
                } else {
                    const auto b = _mm512_cvtepi8_epi64(a);
                    return _mm512_test_epi64_mask(b, b);
                }
            } else if constexpr (size() == 16) {
                const auto a = detail::builtin_load<long long, 2>(mem, f);
                if constexpr (detail::have_avx512bw_vl) {
                    return _mm_test_epi8_mask(a, a);
                } else {
                    const auto b = _mm512_cvtepi8_epi32(a);
                    return _mm512_test_epi32_mask(b, b);
                }
            } else if constexpr (size() == 32) {
                if constexpr (detail::have_avx512bw_vl) {
                    const auto a = detail::builtin_load<long long, 4>(mem, f);
                    return _mm256_test_epi8_mask(a, a);
                } else {
                    const auto a =
                        _mm512_cvtepi8_epi32(detail::builtin_load<long long, 2>(mem, f));
                    const auto b = _mm512_cvtepi8_epi32(
                        detail::builtin_load<long long, 2>(mem + 16, f));
                    return _mm512_test_epi32_mask(a, a) |
                           (_mm512_test_epi32_mask(b, b) << 16);
                }
            } else if constexpr (size() == 64) {
                if constexpr (detail::have_avx512bw) {
                    const auto a = detail::builtin_load<long long, 8>(mem, f);
                    return _mm512_test_epi8_mask(a, a);
                } else {
                    const auto a =
                        _mm512_cvtepi8_epi32(detail::builtin_load<long long, 2>(mem, f));
                    const auto b = _mm512_cvtepi8_epi32(
                        detail::builtin_load<long long, 2>(mem + 16, f));
                    const auto c = _mm512_cvtepi8_epi32(
                        detail::builtin_load<long long, 2>(mem + 32, f));
                    const auto d = _mm512_cvtepi8_epi32(
                        detail::builtin_load<long long, 2>(mem + 48, f));
                    return _mm512_test_epi32_mask(a, a) |
                           (_mm512_test_epi32_mask(b, b) << 16) |
                           (_mm512_test_epi32_mask(b, b) << 32) |
                           (_mm512_test_epi32_mask(b, b) << 48);
                }
            } else {
                detail::assert_unreachable<F>();
            }
        } else {
            detail::assert_unreachable<F>();
        }
        detail::unused(f);  // not true, see PR85827
    }

public :
    // }}}
    // load constructor {{{
    template <class Flags>
    Vc_ALWAYS_INLINE simd_mask(const value_type *mem, Flags f) : d(load_wrapper(mem, f))
    {
    }
    template <class Flags>
    Vc_ALWAYS_INLINE simd_mask(const value_type *mem, simd_mask k, Flags f) : d{}
    {
        d = impl::masked_load(d, k.d, mem, f);
    }

    // }}}
    // loads [simd_mask.load] {{{
    template <class Flags> Vc_ALWAYS_INLINE void copy_from(const value_type *mem, Flags f)
    {
        d = load_wrapper(mem, f);
    }

    // }}}
    // stores [simd_mask.store] {{{
    template <class Flags> Vc_ALWAYS_INLINE void copy_to(value_type *mem, Flags f) const
    {
        impl::store(d, mem, f);
    }

    // }}}
    // scalar access {{{
    Vc_ALWAYS_INLINE reference operator[](size_t i) { return {d, int(i)}; }
    Vc_ALWAYS_INLINE value_type operator[](size_t i) const {
        if constexpr (is_scalar()) {
            Vc_ASSERT(i == 0);
            detail::unused(i);
            return d;
        } else {
            return d[i];
        }
    }

    // }}}
    // negation {{{
    Vc_ALWAYS_INLINE simd_mask operator!() const
    {
        if constexpr (is_scalar()) {
            return {detail::private_init, !d};
        } else if constexpr (is_avx512() || is_fixed()) {
            return simd_mask(detail::private_init, ~builtin());
        } else {
            return {detail::private_init,
                    detail::to_storage(~detail::builtin_cast<uint>(builtin()))};
        }
    }

    // }}}
    // simd_mask binary operators [simd_mask.binary] {{{
    Vc_ALWAYS_INLINE friend simd_mask operator&&(const simd_mask &x, const simd_mask &y)
    {
        return {detail::private_init, impl::logical_and(x.d, y.d)};
    }
    Vc_ALWAYS_INLINE friend simd_mask operator||(const simd_mask &x, const simd_mask &y)
    {
        return {detail::private_init, impl::logical_or(x.d, y.d)};
    }

    Vc_ALWAYS_INLINE friend simd_mask operator&(const simd_mask &x, const simd_mask &y)
    {
        return {detail::private_init, impl::bit_and(x.d, y.d)};
    }
    Vc_ALWAYS_INLINE friend simd_mask operator|(const simd_mask &x, const simd_mask &y)
    {
        return {detail::private_init, impl::bit_or(x.d, y.d)};
    }
    Vc_ALWAYS_INLINE friend simd_mask operator^(const simd_mask &x, const simd_mask &y)
    {
        return {detail::private_init, impl::bit_xor(x.d, y.d)};
    }

    Vc_ALWAYS_INLINE friend simd_mask &operator&=(simd_mask &x, const simd_mask &y)
    {
        x.d = impl::bit_and(x.d, y.d);
        return x;
    }
    Vc_ALWAYS_INLINE friend simd_mask &operator|=(simd_mask &x, const simd_mask &y)
    {
        x.d = impl::bit_or(x.d, y.d);
        return x;
    }
    Vc_ALWAYS_INLINE friend simd_mask &operator^=(simd_mask &x, const simd_mask &y)
    {
        x.d = impl::bit_xor(x.d, y.d);
        return x;
    }

    // }}}
    // simd_mask compares [simd_mask.comparison] {{{
    Vc_ALWAYS_INLINE friend simd_mask operator==(const simd_mask &x, const simd_mask &y)
    {
        return !operator!=(x, y);
    }
    Vc_ALWAYS_INLINE friend simd_mask operator!=(const simd_mask &x, const simd_mask &y)
    {
        return {detail::private_init, impl::bit_xor(x.d, y.d)};
    }

    // }}}
    // "private" because of the first arguments's namespace
    Vc_INTRINSIC simd_mask(detail::private_init_t, typename traits::mask_member_type init)
        : d(init)
    {
    }

    // "private" because of the first arguments's namespace
    template <class F, class = decltype(bool(std::declval<F>()(size_t())))>
    Vc_INTRINSIC simd_mask(detail::private_init_t, F &&gen)
    {
        for (size_t i = 0; i < size(); ++i) {
            impl::set(d, i, gen(i));
        }
    }

    // "private" because of the first arguments's namespace
    Vc_INTRINSIC simd_mask(detail::bitset_init_t, std::bitset<size()> init)
        : d(impl::from_bitset(init, type_tag))
    {
    }

private:
    Vc_INTRINSIC static constexpr member_type broadcast(value_type x)  // {{{
    {
        if constexpr (is_scalar()) {
            return x;
        } else if constexpr (is_fixed()) {
            return x ? ~member_type() : member_type();
        } else if constexpr (is_avx512()) {
            using mmask_type = typename detail::bool_storage_member_type<size()>::type;
            return x ? Abi::masked(static_cast<mmask_type>(~mmask_type())) : mmask_type();
        } else {
            using U = detail::builtin_type_t<detail::int_for_sizeof_t<T>, size()>;
            return detail::to_storage(x ? Abi::masked(~U()) : U());
        }
    }

    // }}}
    auto intrin() const  // {{{
    {
        if constexpr (!is_scalar() && !is_fixed()) {
            return detail::to_intrin(d.d);
        }
    }

    // }}}
    auto &builtin() {  // {{{
        if constexpr (is_scalar() || is_fixed()) {
            return d;
        } else {
            return d.d;
        }
    }
    const auto &builtin() const
    {
        if constexpr (is_scalar() || is_fixed()) {
            return d;
        } else {
            return d.d;
        }
    }

    // }}}
    friend const auto &detail::data<T, abi_type>(const simd_mask &);
    friend auto &detail::data<T, abi_type>(simd_mask &);
    alignas(traits::mask_member_alignment) member_type d;
};

// }}}

namespace detail
{
// data(simd_mask) {{{
template <class T, class A>
Vc_INTRINSIC constexpr const auto &data(const simd_mask<T, A> &x)
{
    return x.d;
}
template <class T, class A> Vc_INTRINSIC constexpr auto &data(simd_mask<T, A> &x)
{
    return x.d;
}
// }}}
// all_of {{{
template <class T, class Abi, class Data> Vc_INTRINSIC bool all_of(const Data &k)
{
    if constexpr (is_abi<Abi, simd_abi::scalar>()) {
        return k;
    } else if constexpr (is_abi<Abi, simd_abi::fixed_size>()) {
        return k.all();
    } else if constexpr (is_combined_abi<Abi>()) {
        for (int i = 0; i < Abi::factor; ++i) {
            if (!all_of<T, typename Abi::member_abi>(k[i])) {
                return false;
            }
        }
        return true;
    } else if constexpr (is_abi<Abi, simd_abi::__sse_abi>() ||
                         is_abi<Abi, simd_abi::__avx_abi>()) {
        constexpr size_t N = simd_size_v<T, Abi>;
        if constexpr (have_sse4_1) {
            constexpr auto b =
                reinterpret_cast<intrinsic_type_t<T, N>>(Abi::template implicit_mask<T>);
            if constexpr (std::is_same_v<T, float> && N > 4) {
                return 0 != _mm256_testc_ps(k, b);
            } else if constexpr (std::is_same_v<T, float> && have_avx) {
                return 0 != _mm_testc_ps(k, b);
            } else if constexpr (std::is_same_v<T, float> ) {
                return 0 != _mm_testc_si128(_mm_castps_si128(k), _mm_castps_si128(b));
            } else if constexpr (std::is_same_v<T, double> && N > 2) {
                return 0 != _mm256_testc_pd(k, b);
            } else if constexpr (std::is_same_v<T, double> && have_avx) {
                return 0 != _mm_testc_pd(k, b);
            } else if constexpr (std::is_same_v<T, double>) {
                return 0 != _mm_testc_si128(_mm_castpd_si128(k), _mm_castpd_si128(b));
            } else if constexpr (sizeof(b) == 32) {
                return _mm256_testc_si256(k, b);
            } else {
                return _mm_testc_si128(k, b);
            }
        } else if constexpr (std::is_same_v<T, float>) {
            return (_mm_movemask_ps(k) & ((1 << N) - 1)) == (1 << N) - 1;
        } else if constexpr (std::is_same_v<T, double>) {
            return (_mm_movemask_pd(k) & ((1 << N) - 1)) == (1 << N) - 1;
        } else {
            return (_mm_movemask_epi8(k) & ((1 << (N * sizeof(T))) - 1)) ==
                   (1 << (N * sizeof(T))) - 1;
        }
    } else if constexpr (is_abi<Abi, simd_abi::__avx512_abi>()) {
        constexpr auto Mask = Abi::template implicit_mask<T>;
        if constexpr (std::is_same_v<Data, Storage<bool, 8>>) {
            if constexpr (have_avx512dq) {
                return _kortestc_mask8_u8(k.d, Mask == 0xff ? k.d : __mmask8(~Mask));
            } else {
                return k.d == Mask;
            }
        } else if constexpr (std::is_same_v<Data, Storage<bool, 16>>) {
            return _kortestc_mask16_u8(k.d, Mask == 0xffff ? k.d : __mmask16(~Mask));
        } else if constexpr (std::is_same_v<Data, Storage<bool, 32>>) {
            if constexpr (detail::have_avx512bw) {
#ifdef Vc_WORKAROUND_PR85538
                return k.d == Mask;
#else
                return _kortestc_mask32_u8(k.d, Mask == 0xffffffffU ? k.d : __mmask32(~Mask));
#endif
            }
        } else if constexpr (std::is_same_v<Data, Storage<bool, 64>>) {
            if constexpr (detail::have_avx512bw) {
#ifdef Vc_WORKAROUND_PR85538
                return k.d == Mask;
#else
                return _kortestc_mask64_u8(
                    k.d, Mask == 0xffffffffffffffffULL ? k.d : __mmask64(~Mask));
#endif
            }
        }
    } else {
        assert_unreachable<T>();
    }
}

// }}}
// any_of {{{
template <class T, class Abi, class Data> Vc_INTRINSIC bool any_of(const Data &k)
{
    if constexpr (is_abi<Abi, simd_abi::scalar>()) {
        return k;
    } else if constexpr (is_abi<Abi, simd_abi::fixed_size>()) {
        return k.any();
    } else if constexpr (is_combined_abi<Abi>()) {
        for (int i = 0; i < Abi::factor; ++i) {
            if (any_of<T, typename Abi::member_abi>(k[i])) {
                return true;
            }
        }
        return false;
    } else if constexpr (is_abi<Abi, simd_abi::__sse_abi>() ||
                         is_abi<Abi, simd_abi::__avx_abi>()) {
        constexpr size_t N = simd_size_v<T, Abi>;
        if constexpr (have_sse4_1) {
            return 0 == testz(k.d, Abi::template implicit_mask<T>);
        } else if constexpr (std::is_same_v<T, float>) {
            return (_mm_movemask_ps(k) & ((1 << N) - 1)) != 0;
        } else if constexpr (std::is_same_v<T, double>) {
            return (_mm_movemask_pd(k) & ((1 << N) - 1)) != 0;
        } else {
            return (_mm_movemask_epi8(k) & ((1 << (N * sizeof(T))) - 1)) != 0;
        }
    } else if constexpr (is_abi<Abi, simd_abi::__avx512_abi>()) {
        return (k & Abi::template implicit_mask<T>) != 0;
    } else {
        assert_unreachable<T>();
    }
}

// }}}
// none_of {{{
template <class T, class Abi, class Data> Vc_INTRINSIC bool none_of(const Data &k)
{
    if constexpr (is_abi<Abi, simd_abi::scalar>()) {
        return !k;
    } else if constexpr (is_abi<Abi, simd_abi::fixed_size>()) {
        return k.none();
    } else if constexpr (is_combined_abi<Abi>()) {
        for (int i = 0; i < Abi::factor; ++i) {
            if (any_of<T, typename Abi::member_abi>(k[i])) {
                return false;
            }
        }
        return true;
    } else if constexpr (is_abi<Abi, simd_abi::__sse_abi>() ||
                         is_abi<Abi, simd_abi::__avx_abi>()) {
        constexpr size_t N = simd_size_v<T, Abi>;
        if constexpr (have_sse4_1) {
            return 0 != testz(k.d, Abi::template implicit_mask<T>);
        } else if constexpr (std::is_same_v<T, float>) {
            return (_mm_movemask_ps(k) & ((1 << N) - 1)) == 0;
        } else if constexpr (std::is_same_v<T, double>) {
            return (_mm_movemask_pd(k) & ((1 << N) - 1)) == 0;
        } else {
            return (_mm_movemask_epi8(k) & ((1 << (N * sizeof(T))) - 1)) == 0;
        }
    } else if constexpr (is_abi<Abi, simd_abi::__avx512_abi>()) {
        return (k & Abi::template implicit_mask<T>) == 0;
    } else {
        assert_unreachable<T>();
    }
}

// }}}
// some_of {{{
template <class T, class Abi, class Data> Vc_INTRINSIC bool some_of(const Data &k)
{
    if constexpr (is_abi<Abi, simd_abi::scalar>()) {
        return false;
    } else if constexpr (is_abi<Abi, simd_abi::fixed_size>()) {
        return k.any() && !k.all();
    } else if constexpr (is_combined_abi<Abi>()) {
        return any_of<T, Abi>(k) && !all_of<T, Abi>(k);
    } else if constexpr (is_abi<Abi, simd_abi::__sse_abi>() ||
                         is_abi<Abi, simd_abi::__avx_abi>()) {
        constexpr size_t N = simd_size_v<T, Abi>;
        if constexpr (have_sse4_1) {
            return 0 != testnzc(k.d, Abi::template implicit_mask<T>);
        } else if constexpr (std::is_same_v<T, float>) {
            constexpr int allbits = (1 << N) - 1;
            const auto tmp = _mm_movemask_ps(k) & allbits;
            return tmp > 0 && tmp < allbits;
        } else if constexpr (std::is_same_v<T, double>) {
            constexpr int allbits = (1 << N) - 1;
            const auto tmp = _mm_movemask_pd(k) & allbits;
            return tmp > 0 && tmp < allbits;
        } else {
            constexpr int allbits = (1 << (N * sizeof(T))) - 1;
            const auto tmp = _mm_movemask_epi8(k) & allbits;
            return tmp > 0 && tmp < allbits;
        }
    } else if constexpr (is_abi<Abi, simd_abi::__avx512_abi>()) {
        return any_of<T, Abi>(k) && !all_of<T, Abi>(k);
    } else {
        assert_unreachable<T>();
    }
}

// }}}
// popcount {{{
template <class T, class Abi, class Data> Vc_INTRINSIC int popcount(const Data &k)
{
    if constexpr (is_abi<Abi, simd_abi::scalar>()) {
        return k;
    } else if constexpr (is_abi<Abi, simd_abi::fixed_size>()) {
        return k.count();
    } else if constexpr (is_combined_abi<Abi>()) {
        int count = popcount<T, typename Abi::member_abi>(k[0]);
        for (int i = 1; i < Abi::factor; ++i) {
            count += popcount<T, typename Abi::member_abi>(k[i]);
        }
        return count;
    } else if constexpr (is_abi<Abi, simd_abi::__sse_abi>() ||
                         is_abi<Abi, simd_abi::__avx_abi>()) {
        constexpr size_t N = simd_size_v<T, Abi>;
        const auto kk = Abi::masked(k.d);
        if constexpr (have_popcnt) {
            int bits = movemask(to_intrin(builtin_cast<T>(kk)));
            const int count = __builtin_popcount(bits);
            return std::is_integral_v<T> ? count / sizeof(T) : count;
        } else if constexpr (N == 2) {
            const int mask = _mm_movemask_pd(auto_cast(kk));
            return mask - (mask >> 1);
        } else if constexpr (N == 4 && sizeof(kk) == 16 && have_sse2) {
            auto x = builtin_cast<llong>(kk);
            x = _mm_add_epi32(x, _mm_shuffle_epi32(x, _MM_SHUFFLE(0, 1, 2, 3)));
            x = _mm_add_epi32(x, _mm_shufflelo_epi16(x, _MM_SHUFFLE(1, 0, 3, 2)));
            return -_mm_cvtsi128_si32(x);
        } else if constexpr (N == 4 && sizeof(kk) == 16) {
            return __builtin_popcount(_mm_movemask_ps(auto_cast(kk)));
        } else if constexpr (N == 8 && sizeof(kk) == 16) {
            auto x = builtin_cast<llong>(kk);
            x = _mm_add_epi16(x, _mm_shuffle_epi32(x, _MM_SHUFFLE(0, 1, 2, 3)));
            x = _mm_add_epi16(x, _mm_shufflelo_epi16(x, _MM_SHUFFLE(0, 1, 2, 3)));
            x = _mm_add_epi16(x, _mm_shufflelo_epi16(x, _MM_SHUFFLE(2, 3, 0, 1)));
            return -short(_mm_extract_epi16(x, 0));
        } else if constexpr (N == 16 && sizeof(kk) == 16) {
            auto x = builtin_cast<llong>(kk);
            x = _mm_add_epi8(x, _mm_shuffle_epi32(x, _MM_SHUFFLE(0, 1, 2, 3)));
            x = _mm_add_epi8(x, _mm_shufflelo_epi16(x, _MM_SHUFFLE(0, 1, 2, 3)));
            x = _mm_add_epi8(x, _mm_shufflelo_epi16(x, _MM_SHUFFLE(2, 3, 0, 1)));
            auto y = -builtin_cast<uchar>(x);
            if constexpr (have_sse4_1) {
                return y[0] + y[1];
            } else {
                unsigned z = _mm_extract_epi16(builtin_cast<llong>(y), 0);
                return (z & 0xff) + (z >> 8);
            }
        } else if constexpr (N == 4 && sizeof(kk) == 32) {
            auto x = -(lo128(kk) + hi128(kk));
            return x[0] + x[1];
        } else if constexpr (sizeof(kk) == 32) {
            return popcount<T, simd_abi::__sse>(-(lo128(kk) + hi128(kk)));
        } else {
            assert_unreachable<T>();
        }
    } else if constexpr (is_abi<Abi, simd_abi::__avx512_abi>()) {
        constexpr size_t N = simd_size_v<T, Abi>;
        const auto kk = Abi::masked(k.d);
        if constexpr (N <= 4) {
            return __builtin_popcount(kk);
        } else if constexpr (N <= 8) {
            return __builtin_popcount(kk);
        } else if constexpr (N <= 16) {
            return __builtin_popcount(kk);
        } else if constexpr (N <= 32) {
            return __builtin_popcount(kk);
        } else if constexpr (N <= 64) {
            return __builtin_popcountll(kk);
        } else {
            assert_unreachable<T>();
        }
    } else {
        assert_unreachable<T>();
    }
}

// }}}
// find_first_set {{{
template <class T, class Abi, class Data> Vc_INTRINSIC int find_first_set(const Data &k)
{
    if constexpr (is_abi<Abi, simd_abi::scalar>()) {
        return 0;
    } else if constexpr (is_abi<Abi, simd_abi::fixed_size>()) {
        return firstbit(k.to_ullong());
    } else if constexpr (is_combined_abi<Abi>()) {
        using A2 = typename Abi::member_abi;
        for (int i = 0; i < Abi::factor - 1; ++i) {
            if (any_of<T, A2>(k[i])) {
                return i * simd_size_v<T, A2> + find_first_set(k[i]);
            }
        }
        return (Abi::factor - 1) * simd_size_v<T, A2> +
               find_first_set(k[Abi::factor - 1]);
    } else if constexpr (is_abi<Abi, simd_abi::__sse_abi>() ||
                         is_abi<Abi, simd_abi::__avx_abi>()) {
        return firstbit(detail::to_bitset(k.d).to_ullong());
    } else if constexpr (is_abi<Abi, simd_abi::__avx512_abi>()) {
        if constexpr (simd_size_v<T, Abi> <= 32) {
            return _tzcnt_u32(k.d);
        } else {
            return firstbit(k.d);
        }
    } else {
        assert_unreachable<T>();
    }
}

// }}}
// find_last_set {{{
template <class T, class Abi, class Data> Vc_INTRINSIC int find_last_set(const Data &k)
{
    if constexpr (is_abi<Abi, simd_abi::scalar>()) {
        return 0;
    } else if constexpr (is_abi<Abi, simd_abi::fixed_size>()) {
        return lastbit(k.to_ullong());
    } else if constexpr (is_combined_abi<Abi>()) {
        using A2 = typename Abi::member_abi;
        for (int i = 0; i < Abi::factor - 1; ++i) {
            if (any_of<T, A2>(k[i])) {
                return i * simd_size_v<T, A2> + find_last_set(k[i]);
            }
        }
        return (Abi::factor - 1) * simd_size_v<T, A2> + find_last_set(k[Abi::factor - 1]);
    } else if constexpr (is_abi<Abi, simd_abi::__sse_abi>() ||
                         is_abi<Abi, simd_abi::__avx_abi>()) {
        return lastbit(detail::to_bitset(k.d).to_ullong());
    } else if constexpr (is_abi<Abi, simd_abi::__avx512_abi>()) {
        if constexpr (simd_size_v<T, Abi> <= 32) {
            return 31 - _lzcnt_u32(k.d);
        } else {
            return lastbit(k.d);
        }
    } else {
        assert_unreachable<T>();
    }
}

// }}}
}  // namespace detail

// reductions [simd_mask.reductions] {{{
template <class T, class Abi> Vc_ALWAYS_INLINE bool all_of(const simd_mask<T, Abi> &k)
{
    return detail::all_of<T, Abi>(detail::data(k));
}
template <class T, class Abi> Vc_ALWAYS_INLINE bool any_of(const simd_mask<T, Abi> &k)
{
    return detail::any_of<T, Abi>(detail::data(k));
}
template <class T, class Abi> Vc_ALWAYS_INLINE bool none_of(const simd_mask<T, Abi> &k)
{
    return detail::none_of<T, Abi>(detail::data(k));
}
template <class T, class Abi> Vc_ALWAYS_INLINE bool some_of(const simd_mask<T, Abi> &k)
{
    return detail::some_of<T, Abi>(detail::data(k));
}
template <class T, class Abi> Vc_ALWAYS_INLINE int popcount(const simd_mask<T, Abi> &k)
{
    return detail::popcount<T, Abi>(detail::data(k));
}
template <class T, class Abi>
Vc_ALWAYS_INLINE int find_first_set(const simd_mask<T, Abi> &k)
{
    return detail::find_first_set<T, Abi>(detail::data(k));
}
template <class T, class Abi>
Vc_ALWAYS_INLINE int find_last_set(const simd_mask<T, Abi> &k)
{
    return detail::find_last_set<T, Abi>(detail::data(k));
}

constexpr bool all_of(detail::exact_bool x) { return x; }
constexpr bool any_of(detail::exact_bool x) { return x; }
constexpr bool none_of(detail::exact_bool x) { return !x; }
constexpr bool some_of(detail::exact_bool) { return false; }
constexpr int popcount(detail::exact_bool x) { return x; }
constexpr int find_first_set(detail::exact_bool) { return 0; }
constexpr int find_last_set(detail::exact_bool) { return 0; }

// }}}

namespace detail
{
template <class Abi> struct generic_simd_impl;
// allow_conversion_ctor2{{{1
template <class T0, class T1, class A, bool BothIntegral> struct allow_conversion_ctor2_1;

template <class T0, class T1, class A>
struct allow_conversion_ctor2
    : public allow_conversion_ctor2_1<
          T0, T1, A, detail::all<std::is_integral<T0>, std::is_integral<T1>>::value> {
};

// disallow 2nd conversion ctor (equal Abi), if the value_types are equal (copy ctor)
template <class T, class A> struct allow_conversion_ctor2<T, T, A> : public std::false_type {};

// disallow 2nd conversion ctor (equal Abi), if the Abi is a fixed_size instance
template <class T0, class T1, int N>
struct allow_conversion_ctor2<T0, T1, simd_abi::fixed_size<N>> : public std::false_type {};

// disallow 2nd conversion ctor (equal Abi), if both of the above are true
template <class T, int N>
struct allow_conversion_ctor2<T, T, simd_abi::fixed_size<N>> : public std::false_type {};

// disallow 2nd conversion ctor (equal Abi), the integers only differ in sign
template <class T0, class T1, class A>
struct allow_conversion_ctor2_1<T0, T1, A, true>
    : public std::is_same<std::make_signed_t<T0>, std::make_signed_t<T1>> {
};

// disallow 2nd conversion ctor (equal Abi), any value_type is not integral
template <class T0, class T1, class A>
struct allow_conversion_ctor2_1<T0, T1, A, false> : public std::false_type {
};

// allow_conversion_ctor3{{{1
template <class T0, class A0, class T1, class A1, bool = std::is_same<A0, A1>::value>
struct allow_conversion_ctor3 : public std::false_type {
    // disallow 3rd conversion ctor if A0 is not fixed_size<simd_size_v<T1, A1>>
};

template <class T0, class T1, class A1>
struct allow_conversion_ctor3<T0, simd_abi::fixed_size<simd_size_v<T1, A1>>, T1, A1,
                              false  // disallow 3rd conversion ctor if the Abi types are
                                     // equal (disambiguate copy ctor and the two
                                     // preceding conversion ctors)
                              > : public std::is_convertible<T1, T0> {
};

// simd_int_operators{{{1
template <class V, bool> class simd_int_operators {};
template <class V> class simd_int_operators<V, true>
{
    using impl = detail::get_impl_t<V>;

    Vc_INTRINSIC const V &derived() const { return *static_cast<const V *>(this); }

    template <class T> Vc_INTRINSIC static V make_derived(T &&d)
    {
        return {detail::private_init, std::forward<T>(d)};
    }

public:
    friend V &operator %=(V &lhs, const V &x) { return lhs = lhs  % x; }
    friend V &operator &=(V &lhs, const V &x) { return lhs = lhs  & x; }
    friend V &operator |=(V &lhs, const V &x) { return lhs = lhs  | x; }
    friend V &operator ^=(V &lhs, const V &x) { return lhs = lhs  ^ x; }
    friend V &operator<<=(V &lhs, const V &x) { return lhs = lhs << x; }
    friend V &operator>>=(V &lhs, const V &x) { return lhs = lhs >> x; }
    friend V &operator<<=(V &lhs, int x) { return lhs = lhs << x; }
    friend V &operator>>=(V &lhs, int x) { return lhs = lhs >> x; }

    friend V operator% (const V &x, const V &y) { return simd_int_operators::make_derived(impl::modulus        (data(x), data(y))); }
    friend V operator& (const V &x, const V &y) { return simd_int_operators::make_derived(impl::bit_and        (data(x), data(y))); }
    friend V operator| (const V &x, const V &y) { return simd_int_operators::make_derived(impl::bit_or         (data(x), data(y))); }
    friend V operator^ (const V &x, const V &y) { return simd_int_operators::make_derived(impl::bit_xor        (data(x), data(y))); }
    friend V operator<<(const V &x, const V &y) { return simd_int_operators::make_derived(impl::bit_shift_left (data(x), data(y))); }
    friend V operator>>(const V &x, const V &y) { return simd_int_operators::make_derived(impl::bit_shift_right(data(x), data(y))); }
    friend V operator<<(const V &x, int y)      { return simd_int_operators::make_derived(impl::bit_shift_left (data(x), y)); }
    friend V operator>>(const V &x, int y)      { return simd_int_operators::make_derived(impl::bit_shift_right(data(x), y)); }

    // unary operators (for integral T)
    V operator~() const
    {
        return {private_init, impl::complement(derived().d)};
    }
};

//}}}1
}  // namespace detail

// simd {{{
template <class T, class Abi>
class simd
    : public detail::simd_int_operators<
          simd<T, Abi>, detail::all<std::is_integral<T>,
                                    typename detail::traits<T, Abi>::is_valid>::value>,
      public detail::traits<T, Abi>::simd_base
{
    using traits = detail::traits<T, Abi>;
    using impl = typename traits::simd_impl_type;
    using member_type = typename traits::simd_member_type;
    using cast_type = typename traits::simd_cast_type;
    static constexpr T *type_tag = nullptr;
    friend typename traits::simd_base;
    friend impl;
    friend detail::generic_simd_impl<Abi>;
    friend detail::simd_int_operators<simd, true>;

public:
    using value_type = T;
    using reference = detail::smart_reference<member_type, impl, value_type>;
    using mask_type = simd_mask<T, Abi>;
    using abi_type = Abi;

    static constexpr size_t size() { return detail::size_or_zero<T, Abi>; }
    simd() = default;
    simd(const simd &) = default;
    simd(simd &&) = default;
    simd &operator=(const simd &) = default;
    simd &operator=(simd &&) = default;

    // implicit broadcast constructor
    template <class U, class = detail::value_preserving_or_int<U, value_type>>
    Vc_ALWAYS_INLINE constexpr simd(U &&x)
        : d(impl::broadcast(static_cast<value_type>(std::forward<U>(x))))
    {
    }

    // implicit type conversion constructor (convert from fixed_size to fixed_size)
    template <class U>
    Vc_ALWAYS_INLINE simd(
        const simd<U, simd_abi::fixed_size<size()>> &x,
        std::enable_if_t<
            detail::all<std::is_same<simd_abi::fixed_size<size()>, abi_type>,
                        std::negation<detail::is_narrowing_conversion<U, value_type>>,
                        detail::converts_to_higher_integer_rank<U, value_type>>::value,
            void *> = nullptr)
        : simd{static_cast<std::array<U, size()>>(x).data(), vector_aligned}
    {
    }

#ifdef Vc_EXPERIMENTAL
    // explicit type conversion constructor
    // 1st conversion ctor: convert from fixed_size<size()>
    template <class U>
    Vc_ALWAYS_INLINE explicit simd(
        const simd<U, simd_abi::fixed_size<size()>> &x,
        std::enable_if_t<
            detail::any<detail::all<std::negation<std::is_same<
                                        simd_abi::fixed_size<size()>, abi_type>>,
                                    std::is_convertible<U, value_type>>,
                        detail::is_narrowing_conversion<U, value_type>>::value,
            void *> = nullptr)
        : simd{static_cast<std::array<U, size()>>(x).data(), vector_aligned}
    {
    }

    // 2nd conversion ctor: convert equal Abi, integers that only differ in signedness
    template <class U>
    Vc_ALWAYS_INLINE explicit simd(
        const simd<U, Abi> &x,
        std::enable_if_t<detail::allow_conversion_ctor2<value_type, U, Abi>::value,
                         void *> = nullptr)
        : d{static_cast<cast_type>(x)}
    {
    }

    // 3rd conversion ctor: convert from non-fixed_size to fixed_size if U is convertible to
    // value_type
    template <class U, class Abi2>
    Vc_ALWAYS_INLINE explicit simd(
        const simd<U, Abi2> &x,
        std::enable_if_t<detail::allow_conversion_ctor3<value_type, Abi, U, Abi2>::value,
                         void *> = nullptr)
    {
        x.copy_to(d.data(), overaligned<alignof(simd)>);
    }
#endif  // Vc_EXPERIMENTAL

    // generator constructor
    template <class F>
    Vc_ALWAYS_INLINE explicit constexpr simd(
        F &&gen,
        detail::value_preserving_or_int<
            decltype(std::declval<F>()(std::declval<detail::size_constant<0> &>())),
            value_type> * = nullptr)
        : d(impl::generator(std::forward<F>(gen), type_tag))
    {
    }

#ifdef Vc_EXPERIMENTAL
    template <class U, U... Indexes>
    Vc_ALWAYS_INLINE static simd seq(std::integer_sequence<U, Indexes...>)
    {
        constexpr auto N = size();
        alignas(memory_alignment<simd>::value) static constexpr value_type mem[N] = {
            value_type(Indexes)...};
        return simd(mem, vector_aligned);
    }
    Vc_ALWAYS_INLINE static simd seq() {
        return seq(std::make_index_sequence<size()>());
    }
#endif  // Vc_EXPERIMENTAL

    // load constructor
    template <class U, class Flags>
    Vc_ALWAYS_INLINE simd(const U *mem, Flags f)
        : d(impl::load(mem, f, type_tag))
    {
    }

    // loads [simd.load]
    template <class U, class Flags>
    Vc_ALWAYS_INLINE void copy_from(const detail::Vectorizable<U> *mem, Flags f)
    {
        d = static_cast<decltype(d)>(impl::load(mem, f, type_tag));
    }

    // stores [simd.store]
    template <class U, class Flags>
    Vc_ALWAYS_INLINE void copy_to(detail::Vectorizable<U> *mem, Flags f) const
    {
        impl::store(d, mem, f, type_tag);
    }

    // scalar access
    Vc_ALWAYS_INLINE constexpr reference operator[](size_t i) { return {d, int(i)}; }
    Vc_ALWAYS_INLINE constexpr value_type operator[](size_t i) const
    {
        if constexpr (is_scalar()) {
            Vc_ASSERT(i == 0);
            detail::unused(i);
            return d;
        } else {
            return d[i];
        }
    }

    // increment and decrement:
    Vc_ALWAYS_INLINE simd &operator++() { impl::increment(d); return *this; }
    Vc_ALWAYS_INLINE simd operator++(int) { simd r = *this; impl::increment(d); return r; }
    Vc_ALWAYS_INLINE simd &operator--() { impl::decrement(d); return *this; }
    Vc_ALWAYS_INLINE simd operator--(int) { simd r = *this; impl::decrement(d); return r; }

    // unary operators (for any T)
    Vc_ALWAYS_INLINE mask_type operator!() const
    {
        return {detail::private_init, impl::negate(d)};
    }
    Vc_ALWAYS_INLINE simd operator+() const { return *this; }
    Vc_ALWAYS_INLINE simd operator-() const
    {
        return {detail::private_init, impl::unary_minus(d)};
    }

    // access to internal representation (suggested extension)
    Vc_ALWAYS_INLINE explicit simd(cast_type init) : d(init) {}

    // compound assignment [simd.cassign]
    Vc_ALWAYS_INLINE friend simd &operator+=(simd &lhs, const simd &x) { return lhs = lhs + x; }
    Vc_ALWAYS_INLINE friend simd &operator-=(simd &lhs, const simd &x) { return lhs = lhs - x; }
    Vc_ALWAYS_INLINE friend simd &operator*=(simd &lhs, const simd &x) { return lhs = lhs * x; }
    Vc_ALWAYS_INLINE friend simd &operator/=(simd &lhs, const simd &x) { return lhs = lhs / x; }

    // binary operators [simd.binary]
    Vc_ALWAYS_INLINE friend simd operator+(const simd &x, const simd &y)
    {
        return {detail::private_init, impl::plus(x.d, y.d)};
    }
    Vc_ALWAYS_INLINE friend simd operator-(const simd &x, const simd &y)
    {
        return {detail::private_init, impl::minus(x.d, y.d)};
    }
    Vc_ALWAYS_INLINE friend simd operator*(const simd &x, const simd &y)
    {
        return {detail::private_init, impl::multiplies(x.d, y.d)};
    }
    Vc_ALWAYS_INLINE friend simd operator/(const simd &x, const simd &y)
    {
        return {detail::private_init, impl::divides(x.d, y.d)};
    }

    // compares [simd.comparison]
    Vc_ALWAYS_INLINE friend mask_type operator==(const simd &x, const simd &y)
    {
        return simd::make_mask(impl::equal_to(x.d, y.d));
    }
    Vc_ALWAYS_INLINE friend mask_type operator!=(const simd &x, const simd &y)
    {
        return simd::make_mask(impl::not_equal_to(x.d, y.d));
    }
    Vc_ALWAYS_INLINE friend mask_type operator<(const simd &x, const simd &y)
    {
        return simd::make_mask(impl::less(x.d, y.d));
    }
    Vc_ALWAYS_INLINE friend mask_type operator<=(const simd &x, const simd &y)
    {
        return simd::make_mask(impl::less_equal(x.d, y.d));
    }
    Vc_ALWAYS_INLINE friend mask_type operator>(const simd &x, const simd &y)
    {
        return simd::make_mask(impl::less(y.d, x.d));
    }
    Vc_ALWAYS_INLINE friend mask_type operator>=(const simd &x, const simd &y)
    {
        return simd::make_mask(impl::less_equal(y.d, x.d));
    }

    // "private" because of the first arguments's namespace
    Vc_INTRINSIC simd(detail::private_init_t, const member_type &init) : d(init) {}

    // "private" because of the first arguments's namespace
    Vc_INTRINSIC simd(detail::bitset_init_t, std::bitset<size()> init) : d() {
        where(mask_type(detail::bitset_init, init), *this) = ~*this;
    }

private:
    static constexpr bool is_scalar() { return std::is_same_v<abi_type, simd_abi::scalar>; }
    static constexpr bool is_fixed() { return detail::is_fixed_size_abi_v<abi_type>; }

    Vc_INTRINSIC static mask_type make_mask(typename mask_type::member_type k)
    {
        return {detail::private_init, k};
    }
    friend const auto &detail::data<value_type, abi_type>(const simd &);
    friend auto &detail::data<value_type, abi_type>(simd &);
    alignas(traits::simd_member_alignment) member_type d;
};

// }}}
namespace detail
{
// data {{{
template <class T, class A> Vc_INTRINSIC constexpr const auto &data(const simd<T, A> &x)
{
    return x.d;
}
template <class T, class A> Vc_INTRINSIC constexpr auto &data(simd<T, A> &x)
{
    return x.d;
}
// }}}
}  // namespace detail

namespace __proposed
{
namespace float_bitwise_operators
{
// float_bitwise_operators {{{
template <class T, class A>
Vc_INTRINSIC simd<T, A> operator|(const simd<T, A> &a, const simd<T, A> &b)
{
    return {Vc::detail::private_init, Vc::detail::get_impl_t<simd<T, A>>::bit_or(
                                          Vc::detail::data(a), Vc::detail::data(b))};
}

template <class T, class A>
Vc_INTRINSIC simd<T, A> operator&(const simd<T, A> &a, const simd<T, A> &b)
{
    return {Vc::detail::private_init, Vc::detail::get_impl_t<simd<T, A>>::bit_and(
                                          Vc::detail::data(a), Vc::detail::data(b))};
}
// }}}
}  // namespace float_bitwise_operators
}  // namespace __proposed

Vc_VERSIONED_NAMESPACE_END

#endif  // BITS_SIMD_H_
