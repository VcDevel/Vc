#ifndef _GLIBCXX_EXPERIMENTAL_SIMD_H
#define _GLIBCXX_EXPERIMENTAL_SIMD_H

//#pragma GCC system_header

#if __cplusplus >= 201703L

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
#if _GLIBCXX_SIMD_HAVE_SSE || _GLIBCXX_SIMD_HAVE_MMX
#include <x86intrin.h>
#endif  // _GLIBCXX_SIMD_HAVE_SSE

_GLIBCXX_SIMD_BEGIN_NAMESPACE
// load/store flags {{{
struct element_aligned_tag {};
struct vector_aligned_tag {};
template <size_t _N> struct overaligned_tag {
    static constexpr size_t alignment = _N;
};
inline constexpr element_aligned_tag element_aligned = {};
inline constexpr vector_aligned_tag vector_aligned = {};
template <size_t _N> inline constexpr overaligned_tag<_N> overaligned = {};
// }}}

// vvv ---- type traits ---- vvv
// integer type aliases{{{
using __uchar = unsigned char;
using __schar = signed char;
using __ushort = unsigned short;
using __short = short;
using __uint = unsigned int;
using __int = int;
using __ulong = unsigned long;
using __long = long;
using __ullong = unsigned long long;
using __llong = long long;
using __wchar = wchar_t;
using __char16 = char16_t;
using __char32 = char32_t;
//}}}
// __is_equal {{{
template <class _T, _T a, _T b> struct __is_equal : public false_type {
};
template <class _T, _T a> struct __is_equal<_T, a, a> : public true_type {
};

// }}}
// __identity/__id{{{
template <class _T> struct __identity {
    using type = _T;
};
template <class _T> using __id = typename __identity<_T>::type;

// }}}
// __first_of_pack{{{
template <class _T0, class...> struct __first_of_pack {
    using type = _T0;
};
template <class... _Ts> using __first_of_pack_t = typename __first_of_pack<_Ts...>::type;

//}}}
// __value_type_or_identity {{{
template <class _T> typename _T::value_type __value_type_or_identity_impl(int);
template <class _T> _T __value_type_or_identity_impl(float);
template <class _T>
using __value_type_or_identity = decltype(__value_type_or_identity_impl<_T>(int()));

// }}}
// __is_vectorizable {{{
template <class _T> struct __is_vectorizable : public std::is_arithmetic<_T> {};
template <> struct __is_vectorizable<bool> : public false_type {};
template <class _T> inline constexpr bool __is_vectorizable_v = __is_vectorizable<_T>::value;
// Deduces to a vectorizable type
template <class _T, class = enable_if_t<__is_vectorizable_v<_T>>> using _Vectorizable = _T;

// }}}
// __loadstore_ptr_type / __is_possible_loadstore_conversion {{{
template <class _Ptr, class _ValueType>
struct __is_possible_loadstore_conversion
    : conjunction<__is_vectorizable<_Ptr>, __is_vectorizable<_ValueType>> {
};
template <> struct __is_possible_loadstore_conversion<bool, bool> : true_type {
};
// Deduces to a type allowed for load/store with the given value type.
template <class _Ptr, class _ValueType,
          class = enable_if_t<__is_possible_loadstore_conversion<_Ptr, _ValueType>::value>>
using __loadstore_ptr_type = _Ptr;

// }}}
// __size_constant{{{
template <size_t _X> using __size_constant = integral_constant<size_t, _X>;
// }}}
// __is_bitmask{{{
template <class _T, class = std::void_t<>> struct __is_bitmask : false_type {
    constexpr __is_bitmask(const _T &) noexcept {}
};
template <class _T> inline constexpr bool __is_bitmask_v = __is_bitmask<_T>::value;

/* the __storage<bool, _N> case:
template <class _T>
struct __is_bitmask<_T, enable_if_t<(sizeof(_T) < _T::width)>> : true_type {
    constexpr __is_bitmask(const _T &) noexcept {}
};*/

// the __mmaskXX case:
template <class _T>
struct __is_bitmask<
    _T, std::void_t<decltype(std::declval<unsigned &>() = std::declval<_T>() & 1u)>>
    : true_type {
    constexpr __is_bitmask(const _T &) noexcept {}
};

// }}}
// __int_for_sizeof{{{
template <size_t> struct __int_for_sizeof;
template <> struct __int_for_sizeof<1> { using type = signed char; };
template <> struct __int_for_sizeof<2> { using type = signed short; };
template <> struct __int_for_sizeof<4> { using type = signed int; };
template <> struct __int_for_sizeof<8> { using type = signed long long; };
template <class _T>
using __int_for_sizeof_t = typename __int_for_sizeof<sizeof(_T)>::type;
template <size_t _N>
using __int_with_sizeof_t = typename __int_for_sizeof<_N>::type;

// }}}
// __equal_int_type{{{
/**
 * \internal
 * TODO: rename to same_value_representation
 * Type trait to find the equivalent integer type given a(n) (un)signed long type.
 */
template <class _T, size_t = sizeof(_T)> struct __equal_int_type;
template <> struct __equal_int_type< long, sizeof(int)> { using type =    int; };
template <> struct __equal_int_type< long, sizeof(__llong)> { using type =  __llong; };
template <> struct __equal_int_type<ulong, sizeof(uint)> { using type =   uint; };
template <> struct __equal_int_type<ulong, sizeof(__ullong)> { using type = __ullong; };
template <> struct __equal_int_type< char, 1> { using type = std::conditional_t<std::is_signed_v<char>, __schar, __uchar>; };
template <size_t _N> struct __equal_int_type<char16_t, _N> { using type = std::uint_least16_t; };
template <size_t _N> struct __equal_int_type<char32_t, _N> { using type = std::uint_least32_t; };
template <> struct __equal_int_type<wchar_t, 1> { using type = std::conditional_t<std::is_signed_v<wchar_t>, __schar, __uchar>; };
template <> struct __equal_int_type<wchar_t, sizeof(short)> { using type = std::conditional_t<std::is_signed_v<wchar_t>, short, ushort>; };
template <> struct __equal_int_type<wchar_t, sizeof(int)> { using type = std::conditional_t<std::is_signed_v<wchar_t>, int, uint>; };

template <class _T> using __equal_int_type_t = typename __equal_int_type<_T>::type;

// }}}
// __has_same_value_representation{{{
template <class _T, class = std::void_t<>>
struct __has_same_value_representation : false_type {
};

template <class _T>
struct __has_same_value_representation<_T, std::void_t<typename __equal_int_type<_T>::type>>
    : true_type {
};

template <class _T>
inline constexpr bool __has_same_value_representation_v =
    __has_same_value_representation<_T>::value;

// }}}
// __is_fixed_size_abi{{{
template <class _T> struct __is_fixed_size_abi : false_type {
};

template <int _N> struct __is_fixed_size_abi<simd_abi::fixed_size<_N>> : true_type {
};

template <class _T>
inline constexpr bool __is_fixed_size_abi_v = __is_fixed_size_abi<_T>::value;

// }}}
// __is_callable{{{
template <class _F, class = std::void_t<>, class... Args>
struct __is_callable : false_type {
};

template <class _F, class... Args>
struct __is_callable<_F, std::void_t<decltype(std::declval<_F>()(std::declval<Args>()...))>,
                   Args...> : true_type {
};

template <class _F, class... Args>
inline constexpr bool __is_callable_v = __is_callable<_F, void, Args...>::value;

#define _GLIBCXX_SIMD_TEST_LAMBDA(...)                                                              \
    decltype(                                                                            \
        [](auto &&... arg_s_) -> decltype(__fun(std::declval<decltype(arg_s_)>()...)) * { \
            return nullptr;                                                              \
        })

#define _GLIBCXX_SIMD_IS_CALLABLE(__fun, ...)                                                        \
    decltype(std::experimental::__is_callable_dispatch(int(), _GLIBCXX_SIMD_TEST_LAMBDA(__fun),               \
                                              __VA_ARGS__))::value

// }}}
// constexpr feature detection{{{
constexpr inline bool __have_mmx = _GLIBCXX_SIMD_HAVE_MMX;
constexpr inline bool __have_sse = _GLIBCXX_SIMD_HAVE_SSE;
constexpr inline bool __have_sse2 = _GLIBCXX_SIMD_HAVE_SSE2;
constexpr inline bool __have_sse3 = _GLIBCXX_SIMD_HAVE_SSE3;
constexpr inline bool __have_ssse3 = _GLIBCXX_SIMD_HAVE_SSSE3;
constexpr inline bool __have_sse4_1 = _GLIBCXX_SIMD_HAVE_SSE4_1;
constexpr inline bool __have_sse4_2 = _GLIBCXX_SIMD_HAVE_SSE4_2;
constexpr inline bool __have_xop = _GLIBCXX_SIMD_HAVE_XOP;
constexpr inline bool __have_avx = _GLIBCXX_SIMD_HAVE_AVX;
constexpr inline bool __have_avx2 = _GLIBCXX_SIMD_HAVE_AVX2;
constexpr inline bool __have_bmi = _GLIBCXX_SIMD_HAVE_BMI1;
constexpr inline bool __have_bmi2 = _GLIBCXX_SIMD_HAVE_BMI2;
constexpr inline bool __have_lzcnt = _GLIBCXX_SIMD_HAVE_LZCNT;
constexpr inline bool __have_sse4a = _GLIBCXX_SIMD_HAVE_SSE4A;
constexpr inline bool __have_fma = _GLIBCXX_SIMD_HAVE_FMA;
constexpr inline bool __have_fma4 = _GLIBCXX_SIMD_HAVE_FMA4;
constexpr inline bool __have_f16c = _GLIBCXX_SIMD_HAVE_F16C;
constexpr inline bool __have_popcnt = _GLIBCXX_SIMD_HAVE_POPCNT;
constexpr inline bool __have_avx512f = _GLIBCXX_SIMD_HAVE_AVX512F;
constexpr inline bool __have_avx512dq = _GLIBCXX_SIMD_HAVE_AVX512DQ;
constexpr inline bool __have_avx512vl = _GLIBCXX_SIMD_HAVE_AVX512VL;
constexpr inline bool __have_avx512bw = _GLIBCXX_SIMD_HAVE_AVX512BW;
constexpr inline bool __have_avx512dq_vl = __have_avx512dq && __have_avx512vl;
constexpr inline bool __have_avx512bw_vl = __have_avx512bw && __have_avx512vl;

constexpr inline bool __have_neon = _GLIBCXX_SIMD_HAVE_NEON;
// }}}
// ^^^ ---- type traits ---- ^^^

// __unused{{{
template <class _T> static constexpr void __unused(_T && ) {}

// }}}
// __assert_unreachable{{{
template <class _T> struct __assert_unreachable {
    static_assert(!std::is_same_v<_T, _T>, "this should be unreachable");
};

// }}}
#ifdef NDEBUG
// __dummy_assert {{{
struct __dummy_assert {
    template <class _T> _GLIBCXX_SIMD_INTRINSIC __dummy_assert &operator<<(_T &&) noexcept
    {
        return *this;
    }
};
// }}}
#else   // NDEBUG
// __real_assert {{{
struct __real_assert {
    _GLIBCXX_SIMD_INTRINSIC __real_assert(bool ok, const char *code, const char *file, int line)
        : failed(!ok)
    {
        if (_GLIBCXX_SIMD_IS_UNLIKELY(failed)) {
            printFirst(code, file, line);
        }
    }
    _GLIBCXX_SIMD_INTRINSIC ~__real_assert()
    {
        if (_GLIBCXX_SIMD_IS_UNLIKELY(failed)) {
            finalize();
        }
    }
    template <class _T> _GLIBCXX_SIMD_INTRINSIC __real_assert &operator<<(_T &&__x)
    {
        if (_GLIBCXX_SIMD_IS_UNLIKELY(failed)) {
            print(std::forward<_T>(__x));
        }
        return *this;
    }

private:
    void printFirst(const char *code, const char *file, int line)
    {
        std::cerr << file << ':' << line << ": assert(" << code << ") failed.";
    }
    template <class _T> void print(_T &&__x) const { std::cerr << std::forward<_T>(__x); }
    void finalize()
    {
        std::cerr << std::endl;
        std::abort();
    }
    bool failed;
};
// }}}
#endif  // NDEBUG
// __size_or_zero {{{
template <class _T, class _A, size_t _N = simd_size<_T, _A>::value>
constexpr size_t __size_or_zero_dispatch(int)
{
    return _N;
}
template <class _T, class _A> constexpr size_t __size_or_zero_dispatch(float) {
  return 0;
}
template <class _T, class _A>
inline constexpr size_t __size_or_zero = __size_or_zero_dispatch<_T, _A>(0);

// }}}
// __promote_preserving_unsigned{{{
// work around crazy semantics of unsigned integers of lower rank than int:
// Before applying an operator the operands are promoted to int. In which case over- or
// underflow is UB, even though the operand types were unsigned.
template <class _T> _GLIBCXX_SIMD_INTRINSIC const _T &__promote_preserving_unsigned(const _T &__x)
{
    return __x;
}
_GLIBCXX_SIMD_INTRINSIC unsigned int __promote_preserving_unsigned(const unsigned char &__x)
{
    return __x;
}
_GLIBCXX_SIMD_INTRINSIC unsigned int __promote_preserving_unsigned(const unsigned short &__x)
{
    return __x;
}

// }}}
// __exact_bool{{{1
class __exact_bool {
    const bool _M_data;

public:
    constexpr __exact_bool(bool b) : _M_data(b) {}
    __exact_bool(int) = delete;
    constexpr operator bool() const { return _M_data; }
};

// __execute_on_index_sequence{{{
template <class _F, size_t... _I>
_GLIBCXX_SIMD_INTRINSIC void __execute_on_index_sequence(_F && f, std::index_sequence<_I...>)
{
    auto &&__x = {(f(__size_constant<_I>()), 0)...};
    __unused(__x);
}

template <class _F>
_GLIBCXX_SIMD_INTRINSIC void __execute_on_index_sequence(_F &&, std::index_sequence<>)
{
}

template <class _R, class _F, size_t... _I>
_GLIBCXX_SIMD_INTRINSIC _R __execute_on_index_sequence_with_return(_F && f, std::index_sequence<_I...>)
{
    return _R{f(__size_constant<_I>())...};
}

// }}}
// __execute_n_times{{{1
template <size_t _N, typename _F> _GLIBCXX_SIMD_INTRINSIC void __execute_n_times(_F && f)
{
    __execute_on_index_sequence(std::forward<_F>(f), std::make_index_sequence<_N>{});
}

// __generate_from_n_evaluations{{{
template <size_t _N, typename _R, typename _F>
_GLIBCXX_SIMD_INTRINSIC _R __generate_from_n_evaluations(_F && f)
{
    return __execute_on_index_sequence_with_return<_R>(std::forward<_F>(f),
                                                    std::make_index_sequence<_N>{});
}

// }}}
// __may_alias{{{
/**\internal
 * Helper __may_alias<_T> that turns _T into the type to be used for an aliasing pointer. This
 * adds the __may_alias attribute to _T (with compilers that support it).
 */
template <class _T> using __may_alias [[gnu::__may_alias__]] = _T;

// }}}
// __unsupported_base {{{
// simd and simd_mask base for unsupported <_T, _Abi>
struct __unsupported_base {
    __unsupported_base() = delete;
    __unsupported_base(const __unsupported_base &) = delete;
    __unsupported_base &operator=(const __unsupported_base &) = delete;
    ~__unsupported_base() = delete;
};

// }}}
// __invalid_traits {{{
/**
 * \internal
 * Defines the implementation of a given <_T, _Abi>.
 *
 * Implementations must ensure that only valid <_T, _Abi> instantiations are possible.
 * Static assertions in the type definition do not suffice. It is important that
 * SFINAE works.
 */
struct __invalid_traits {
    using is_valid = false_type;
    static constexpr size_t __simd_member_alignment = 1;
    struct __simd_impl_type;
    struct __simd_member_type {};
    struct __simd_cast_type;
    using __simd_base = __unsupported_base;
    static constexpr size_t __mask_member_alignment = 1;
    struct __mask_impl_type;
    struct __mask_member_type {};
    struct __mask_cast_type;
    using __mask_base = __unsupported_base;
};
// }}}
// __simd_traits {{{
template <class _T, class _Abi, class = std::void_t<>>
struct __simd_traits : __invalid_traits {
};

// }}}
// __get_impl(_t){{{
template <class _T> struct __get_impl;
template <class _T> using __get_impl_t = typename __get_impl<std::decay_t<_T>>::type;

// }}}
// __get_traits(_t){{{
template <class _T> struct __get_traits;
template <class _T> using __get_traits_t = typename __get_traits<std::decay_t<_T>>::type;

// }}}
// __make_immediate{{{
template <unsigned _Stride> constexpr unsigned __make_immediate(unsigned a, unsigned b)
{
    return a + b * _Stride;
}
template <unsigned _Stride>
constexpr unsigned __make_immediate(unsigned a, unsigned b, unsigned c, unsigned d)
{
    return a + _Stride * (b + _Stride * (c + _Stride * d));
}

// }}}
// __next_power_of_2{{{1
/**
 * \internal
 * Returns the next power of 2 larger than or equal to \p __x.
 */
constexpr std::size_t __next_power_of_2(std::size_t __x)
{
    return (__x & (__x - 1)) == 0 ? __x : __next_power_of_2((__x | (__x >> 1)) + 1);
}

// __private_init, __bitset_init{{{1
/**
 * \internal
 * Tag used for private init constructor of simd and simd_mask
 */
inline constexpr struct __private_init_t {} __private_init = {};
inline constexpr struct __bitset_init_t {} __bitset_init = {};

// __bool_constant{{{1
template <bool _Value> using __bool_constant = std::integral_constant<bool, _Value>;

// }}}1
// __is_narrowing_conversion<_From, _To>{{{1
template <class _From, class _To, bool = std::is_arithmetic<_From>::value,
          bool = std::is_arithmetic<_To>::value>
struct __is_narrowing_conversion;

// ignore "warning C4018: '<': signed/unsigned mismatch" in the following trait. The implicit
// conversions will do the right thing here.
template <class _From, class _To>
struct __is_narrowing_conversion<_From, _To, true, true>
    : public __bool_constant<(
          std::numeric_limits<_From>::digits > std::numeric_limits<_To>::digits ||
          std::numeric_limits<_From>::max() > std::numeric_limits<_To>::max() ||
          std::numeric_limits<_From>::lowest() < std::numeric_limits<_To>::lowest() ||
          (std::is_signed<_From>::value && std::is_unsigned<_To>::value))> {
};

template <class _T> struct __is_narrowing_conversion<bool, _T, true, true> : public true_type {};
template <> struct __is_narrowing_conversion<bool, bool, true, true> : public false_type {};
template <class _T> struct __is_narrowing_conversion<_T, _T, true, true> : public false_type {
};

template <class _From, class _To>
struct __is_narrowing_conversion<_From, _To, false, true>
    : public negation<std::is_convertible<_From, _To>> {
};

// __converts_to_higher_integer_rank{{{1
template <class _From, class _To, bool = (sizeof(_From) < sizeof(_To))>
struct __converts_to_higher_integer_rank : public true_type {
};
template <class _From, class _To>
struct __converts_to_higher_integer_rank<_From, _To, false>
    : public std::is_same<decltype(std::declval<_From>() + std::declval<_To>()), _To> {
};

// __is_aligned(_v){{{1
template <class _Flag, size_t Alignment> struct __is_aligned;
template <size_t Alignment>
struct __is_aligned<vector_aligned_tag, Alignment> : public true_type {
};
template <size_t Alignment>
struct __is_aligned<element_aligned_tag, Alignment> : public false_type {
};
template <size_t GivenAlignment, size_t Alignment>
struct __is_aligned<overaligned_tag<GivenAlignment>, Alignment>
    : public std::integral_constant<bool, (GivenAlignment >= Alignment)> {
};
template <class _Flag, size_t Alignment>
inline constexpr bool __is_aligned_v = __is_aligned<_Flag, Alignment>::value;

// }}}1
// __when_(un)aligned{{{
/**
 * \internal
 * Implicitly converts from flags that specify alignment
 */
template <size_t Alignment>
class __when_aligned
{
public:
    constexpr __when_aligned(vector_aligned_tag) {}
    template <size_t _Given, class = enable_if_t<(_Given >= Alignment)>>
    constexpr __when_aligned(overaligned_tag<_Given>)
    {
    }
};
template <size_t Alignment>
class __when_unaligned
{
public:
    constexpr __when_unaligned(element_aligned_tag) {}
    template <size_t _Given, class = enable_if_t<(_Given < Alignment)>>
    constexpr __when_unaligned(overaligned_tag<_Given>)
    {
    }
};

// }}}
// __data(simd/simd_mask) {{{
template <class _T, class _A> _GLIBCXX_SIMD_INTRINSIC constexpr const auto &__data(const std::experimental::simd<_T, _A> &__x);
template <class _T, class _A> _GLIBCXX_SIMD_INTRINSIC constexpr auto &__data(std::experimental::simd<_T, _A> & __x);

template <class _T, class _A> _GLIBCXX_SIMD_INTRINSIC constexpr const auto &__data(const std::experimental::simd_mask<_T, _A> &__x);
template <class _T, class _A> _GLIBCXX_SIMD_INTRINSIC constexpr auto &__data(std::experimental::simd_mask<_T, _A> &__x);

// }}}
// __simd_converter {{{
template <class _FromT, class _FromA, class _ToT, class _ToA> struct __simd_converter;
template <class _T, class _A> struct __simd_converter<_T, _A, _T, _A> {
    template <class _U> _GLIBCXX_SIMD_INTRINSIC const _U &operator()(const _U &__x) { return __x; }
};

// }}}
// __to_value_type_or_member_type {{{
template <class _V>
_GLIBCXX_SIMD_INTRINSIC constexpr auto __to_value_type_or_member_type(const _V &__x)->decltype(__data(__x))
{
    return __data(__x);
}

template <class _V>
_GLIBCXX_SIMD_INTRINSIC constexpr const typename _V::value_type &__to_value_type_or_member_type(
    const typename _V::value_type &__x)
{
    return __x;
}

// }}}
// __bool_storage_member_type{{{
template <size_t _Size> struct __bool_storage_member_type;
template <size_t _Size>
using __bool_storage_member_type_t = typename __bool_storage_member_type<_Size>::type;

// }}}
// __fixed_size_storage fwd decl {{{
template <class _T, int _N> struct __fixed_size_storage_builder_wrapper;
template <class _T, int _N>
using __fixed_size_storage = typename __fixed_size_storage_builder_wrapper<_T, _N>::type;

// }}}
// __storage fwd decl{{{
template <class _ValueType, size_t _Size, class = std::void_t<>> struct __storage;
template <class _T> using __storage16_t = __storage<_T, 16 / sizeof(_T)>;
template <class _T> using __storage32_t = __storage<_T, 32 / sizeof(_T)>;
template <class _T> using __storage64_t = __storage<_T, 64 / sizeof(_T)>;

// }}}
// __bit_iteration{{{
constexpr uint __popcount(uint __x) { return __builtin_popcount(__x); }
constexpr ulong __popcount(ulong __x) { return __builtin_popcountl(__x); }
constexpr __ullong __popcount(__ullong __x) { return __builtin_popcountll(__x); }

constexpr uint ctz(uint __x) { return __builtin_ctz(__x); }
constexpr ulong ctz(ulong __x) { return __builtin_ctzl(__x); }
constexpr __ullong ctz(__ullong __x) { return __builtin_ctzll(__x); }
constexpr uint clz(uint __x) { return __builtin_clz(__x); }
constexpr ulong clz(ulong __x) { return __builtin_clzl(__x); }
constexpr __ullong clz(__ullong __x) { return __builtin_clzll(__x); }

template <class _T, class _F> void __bit_iteration(_T k_, _F &&f)
{
    static_assert(sizeof(__ullong) >= sizeof(_T));
    std::conditional_t<sizeof(_T) <= sizeof(uint), uint, __ullong> __k;
    if constexpr (std::is_convertible_v<_T, decltype(__k)>) {
        __k = k_;
    } else {
        __k = k_.to_ullong();
    }
    switch (__popcount(__k)) {
    default:
        do {
            f(ctz(__k));
            __k &= (__k - 1);
        } while (__k);
        break;
    /*case 3:
        f(ctz(__k));
        __k &= (__k - 1);
        [[fallthrough]];*/
    case 2:
        f(ctz(__k));
        [[fallthrough]];
    case 1:
        f(__popcount(~decltype(__k)()) - 1 - clz(__k));
        [[fallthrough]];
    case 0:
        break;
    }
}

//}}}
// __firstbit{{{
template <class _T> _GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST auto __firstbit(_T bits)
{
    static_assert(std::is_integral_v<_T>, "__firstbit requires an integral argument");
    if constexpr (sizeof(_T) <= sizeof(int)) {
        return __builtin_ctz(bits);
    } else if constexpr(alignof(__ullong) == 8) {
        return __builtin_ctzll(bits);
    } else {
        uint lo = bits;
        return lo == 0 ? 32 + __builtin_ctz(bits >> 32) : __builtin_ctz(lo);
    }
}

// }}}
// __lastbit{{{
template <class _T> _GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST auto __lastbit(_T bits)
{
    static_assert(std::is_integral_v<_T>, "__firstbit requires an integral argument");
    if constexpr (sizeof(_T) <= sizeof(int)) {
        return 31 - __builtin_clz(bits);
    } else if constexpr(alignof(__ullong) == 8) {
        return 63 - __builtin_clzll(bits);
    } else {
        uint lo = bits;
        uint hi = bits >> 32u;
        return hi == 0 ? 31 - __builtin_clz(lo) : 63 - __builtin_clz(hi);
    }
}

// }}}
// __simd_tuple {{{
// why not std::tuple?
// 1. std::tuple gives no guarantee about the storage order, but I require storage
//    equivalent to std::array<_T, _N>
// 2. direct access to the element type (first template argument)
// 3. enforces equal element type, only different _Abi types are allowed
template <class _T, class... _Abis> struct __simd_tuple;

//}}}
// __convert_mask declaration {{{
template <class _To, class _From> inline _To __convert_mask(_From __k);

// }}}
// __shift_left, __shift_right, __increment, __decrement {{{
template <class _T = void> struct __shift_left {
    constexpr _T operator()(const _T &a, const _T &b) const { return a << b; }
};
template <> struct __shift_left<void> {
    template <class _L, class _R> constexpr auto operator()(_L &&a, _R &&b) const
    {
        return std::forward<_L>(a) << std::forward<_R>(b);
    }
};
template <class _T = void> struct __shift_right {
    constexpr _T operator()(const _T &a, const _T &b) const { return a >> b; }
};
template <> struct __shift_right<void> {
    template <class _L, class _R> constexpr auto operator()(_L &&a, _R &&b) const
    {
        return std::forward<_L>(a) >> std::forward<_R>(b);
    }
};
template <class _T = void> struct __increment {
    constexpr _T operator()(_T a) const { return ++a; }
};
template <> struct __increment<void> {
    template <class _T> constexpr _T operator()(_T a) const { return ++a; }
};
template <class _T = void> struct __decrement {
    constexpr _T operator()(_T a) const { return --a; }
};
template <> struct __decrement<void> {
    template <class _T> constexpr _T operator()(_T a) const { return --a; }
};

// }}}
// __get_impl {{{
template <class _T> struct __get_impl {
    static_assert(
        std::is_arithmetic<_T>::value,
        "Vc chose the wrong implementation class. This should not be possible.");

    template <class _U, class _F>
    _GLIBCXX_SIMD_INTRINSIC _T masked_load(_T d, bool __k, const _U *mem, _F)
    {
        if (__k) {
            d = static_cast<_T>(mem[0]);
        }
        return d;
    }
};
template <> struct __get_impl<bool> {
    template <class _F> _GLIBCXX_SIMD_INTRINSIC bool masked_load(bool d, bool __k, const bool *mem, _F)
    {
        if (__k) {
            d = mem[0];
        }
        return d;
    }
};
// }}}
// __value_preserving(_or_int) {{{
template <class _From, class _To,
          class = enable_if_t<
              negation<__is_narrowing_conversion<std::decay_t<_From>, _To>>::value>>
using __value_preserving = _From;

template <class _From, class _To, class _DecayedFrom = std::decay_t<_From>,
          class = enable_if_t<conjunction<
              is_convertible<_From, _To>,
              disjunction<is_same<_DecayedFrom, _To>, is_same<_DecayedFrom, int>,
                  conjunction<is_same<_DecayedFrom, uint>, is_unsigned<_To>>,
                  negation<__is_narrowing_conversion<_DecayedFrom, _To>>>>::value>>
using __value_preserving_or_int = _From;

// }}}
// __intrinsic_type {{{
template <class _T, size_t _Bytes, class = std::void_t<>> struct __intrinsic_type;
template <class _T, size_t _Size>
using __intrinsic_type_t = typename __intrinsic_type<_T, _Size * sizeof(_T)>::type;
template <class _T> using __intrinsic_type2_t   = typename __intrinsic_type<_T, 2>::type;
template <class _T> using __intrinsic_type4_t   = typename __intrinsic_type<_T, 4>::type;
template <class _T> using __intrinsic_type8_t   = typename __intrinsic_type<_T, 8>::type;
template <class _T> using __intrinsic_type16_t  = typename __intrinsic_type<_T, 16>::type;
template <class _T> using __intrinsic_type32_t  = typename __intrinsic_type<_T, 32>::type;
template <class _T> using __intrinsic_type64_t  = typename __intrinsic_type<_T, 64>::type;
template <class _T> using __intrinsic_type128_t = typename __intrinsic_type<_T, 128>::type;

// }}}
// __is_intrinsic{{{1
template <class _T> struct __is_intrinsic : public false_type {};
template <class _T> inline constexpr bool is_intrinsic_v = __is_intrinsic<_T>::value;

// }}}

// vvv ---- builtin vector types [[gnu::vector_size(N)]] and operations ---- vvv
// __vector_type {{{
template <class _T, size_t _N, class = void> struct __vector_type_n {};

// special case 1-element to be _T itself
template <class _T>
struct __vector_type_n<_T, 1, enable_if_t<__is_vectorizable_v<_T>>> {
    using type = _T;
};

// else, use GNU-style builtin vector types
template <class _T, size_t _N>
struct __vector_type_n<_T, _N, enable_if_t<__is_vectorizable_v<_T>>> {
    static constexpr size_t _Bytes = _N * sizeof(_T);
    using type [[gnu::__vector_size__(_Bytes)]] = _T;
};

template <class _T, size_t _Bytes>
struct __vector_type : __vector_type_n<_T, _Bytes / sizeof(_T)> {
    static_assert(_Bytes % sizeof(_T) == 0);
};

template <class _T, size_t _Size>
using __vector_type_t = typename __vector_type_n<_T, _Size>::type;
template <class _T> using __vector_type2_t  = typename __vector_type<_T, 2>::type;
template <class _T> using __vector_type4_t  = typename __vector_type<_T, 4>::type;
template <class _T> using __vector_type8_t  = typename __vector_type<_T, 8>::type;
template <class _T> using __vector_type16_t = typename __vector_type<_T, 16>::type;
template <class _T> using __vector_type32_t = typename __vector_type<_T, 32>::type;
template <class _T> using __vector_type64_t = typename __vector_type<_T, 64>::type;
template <class _T> using __vector_type128_t = typename __vector_type<_T, 128>::type;

// }}}
// __is_vector_type {{{
template <class _T, class = std::void_t<>> struct __is_vector_type : false_type {};
template <class _T>
struct __is_vector_type<
    _T,
    std::void_t<typename __vector_type<decltype(std::declval<_T>()[0]), sizeof(_T)>::type>>
    : std::is_same<
          _T, typename __vector_type<decltype(std::declval<_T>()[0]), sizeof(_T)>::type> {
};

template <class _T>
inline constexpr bool __is_vector_type_v = __is_vector_type<_T>::value;

// }}}
// __vector_traits{{{
template <class _T, class = std::void_t<>> struct __vector_traits;
template <class _T>
struct __vector_traits<_T, std::void_t<enable_if_t<__is_vector_type_v<_T>>>> {
    using type = _T;
    using value_type = decltype(std::declval<_T>()[0]);
    static constexpr int width = sizeof(_T) / sizeof(value_type);
    template <class _U, int _W = width>
    static constexpr bool is = std::is_same_v<value_type, _U> &&_W == width;
};
template <class _T, size_t _N>
struct __vector_traits<__storage<_T, _N>, std::void_t<__vector_type_t<_T, _N>>> {
    using type = __vector_type_t<_T, _N>;
    using value_type = _T;
    static constexpr int width = _N;
    template <class _U, int _W = width>
    static constexpr bool is = std::is_same_v<value_type, _U> &&_W == width;
};

// }}}
// __vector_bitcast{{{
template <class _To, class _From, class _FromVT = __vector_traits<_From>>
_GLIBCXX_SIMD_INTRINSIC constexpr typename __vector_type<_To, sizeof(_From)>::type __vector_bitcast(_From __x)
{
    return reinterpret_cast<typename __vector_type<_To, sizeof(_From)>::type>(__x);
}
template <class _To, class _T, size_t _N>
_GLIBCXX_SIMD_INTRINSIC constexpr typename __vector_type<_To, sizeof(__storage<_T, _N>)>::type
__vector_bitcast(const __storage<_T, _N> &__x)
{
    return reinterpret_cast<typename __vector_type<_To, sizeof(__storage<_T, _N>)>::type>(__x._M_data);
}

// }}}
// __convert_x86 declarations {{{
template <class _To, class _T, class _TVT = __vector_traits<_T>>
_To __convert_x86(_T);

template <class _To, class _T, class _TVT = __vector_traits<_T>>
_To __convert_x86(_T, _T);

template <class _To, class _T, class _TVT = __vector_traits<_T>>
_To __convert_x86(_T, _T, _T, _T);

template <class _To, class _T, class _TVT = __vector_traits<_T>>
_To __convert_x86(_T, _T, _T, _T, _T, _T, _T, _T);

//}}}
// __vector_convert {{{
// implementation requires an index sequence
template <class _To, class _From, size_t... _I>
_GLIBCXX_SIMD_INTRINSIC constexpr _To __vector_convert(_From __a, index_sequence<_I...>)
{
    using _T = typename __vector_traits<_To>::value_type;
    return _To{static_cast<_T>(__a[_I])...};
}

template <class _To, class _From, size_t... _I>
_GLIBCXX_SIMD_INTRINSIC constexpr _To __vector_convert(_From __a, _From __b,
                                                       index_sequence<_I...>)
{
    using _T = typename __vector_traits<_To>::value_type;
    return _To{static_cast<_T>(__a[_I])..., static_cast<_T>(__b[_I])...};
}

template <class _To, class _From, size_t... _I>
_GLIBCXX_SIMD_INTRINSIC constexpr _To __vector_convert(_From __a, _From __b, _From __c,
                                                       index_sequence<_I...>)
{
    using _T = typename __vector_traits<_To>::value_type;
    return _To{static_cast<_T>(__a[_I])..., static_cast<_T>(__b[_I])...,
               static_cast<_T>(__c[_I])...};
}

template <class _To, class _From, size_t... _I>
_GLIBCXX_SIMD_INTRINSIC constexpr _To __vector_convert(_From __a, _From __b, _From __c,
                                                       _From __d, index_sequence<_I...>)
{
    using _T = typename __vector_traits<_To>::value_type;
    return _To{static_cast<_T>(__a[_I])..., static_cast<_T>(__b[_I])...,
               static_cast<_T>(__c[_I])..., static_cast<_T>(__d[_I])...};
}

template <class _To, class _From, size_t... _I>
_GLIBCXX_SIMD_INTRINSIC constexpr _To __vector_convert(_From __a, _From __b, _From __c,
                                                       _From __d, _From __e, _From __f,
                                                       _From __g, _From __h,
                                                       index_sequence<_I...>)
{
    using _T = typename __vector_traits<_To>::value_type;
    return _To{static_cast<_T>(__a[_I])..., static_cast<_T>(__b[_I])...,
               static_cast<_T>(__c[_I])..., static_cast<_T>(__d[_I])...,
               static_cast<_T>(__e[_I])..., static_cast<_T>(__f[_I])...,
               static_cast<_T>(__g[_I])..., static_cast<_T>(__h[_I])...};
}

// Defer actual conversion to the overload that takes an index sequence. Note that this
// function adds zeros or drops values off the end if you don't ensure matching width.
template <class _To, class... _From, class _ToT = __vector_traits<_To>,
          class _FromT = __vector_traits<__first_of_pack_t<_From...>>>
_GLIBCXX_SIMD_INTRINSIC constexpr _To __vector_convert(_From... __xs)
{
#ifdef _GLIBCXX_SIMD_WORKAROUND_PR85048
    return __convert_x86<_To>(__xs...);
#else
    return __vector_convert<_To>(__xs...,
                            make_index_sequence<std::min(_ToT::width, _FromT::width)>());
#endif
}

// This overload takes a vectorizable type _To and produces a return type that matches the
// width.
template <class _To, class... _From, class = enable_if_t<__is_vectorizable_v<_To>>,
          class _FromT = __vector_traits<__first_of_pack_t<_From...>>, class = int>
_GLIBCXX_SIMD_INTRINSIC constexpr _To __vector_convert(_From... __xs)
{
    return __vector_convert<__vector_type_t<_To, _FromT::width>>(__xs...);
}

// }}}
// __to_intrin {{{
template <class _T, class _TVT = __vector_traits<_T>,
          class _R = __intrinsic_type_t<typename _TVT::value_type, _TVT::width>>
_GLIBCXX_SIMD_INTRINSIC constexpr _R __to_intrin(_T __x)
{
    return reinterpret_cast<_R>(__x);
}
template <class _T, size_t _N, class _R = __intrinsic_type_t<_T, _N>>
_GLIBCXX_SIMD_INTRINSIC constexpr _R __to_intrin(__storage<_T, _N> __x)
{
    return reinterpret_cast<_R>(__x._M_data);
}

// }}}
// __make_builtin{{{
template <class _T, class... Args>
_GLIBCXX_SIMD_INTRINSIC constexpr __vector_type_t<_T, sizeof...(Args)> __make_builtin(Args &&... args)
{
    return __vector_type_t<_T, sizeof...(Args)>{static_cast<_T>(args)...};
}

// }}}
// __vector_broadcast{{{
template <size_t _N, class _T>
_GLIBCXX_SIMD_INTRINSIC constexpr __vector_type_t<_T, _N> __vector_broadcast(_T __x)
{
    if constexpr (_N == 2) {
        return __vector_type_t<_T, 2>{__x, __x};
    } else if constexpr (_N == 4) {
        return __vector_type_t<_T, 4>{__x, __x, __x, __x};
    } else if constexpr (_N == 8) {
        return __vector_type_t<_T, 8>{__x, __x, __x, __x, __x, __x, __x, __x};
    } else if constexpr (_N == 16) {
        return __vector_type_t<_T, 16>{__x, __x, __x, __x, __x, __x, __x, __x,
                                       __x, __x, __x, __x, __x, __x, __x, __x};
    } else if constexpr (_N == 32) {
        return __vector_type_t<_T, 32>{__x, __x, __x, __x, __x, __x, __x, __x,
                                       __x, __x, __x, __x, __x, __x, __x, __x,
                                       __x, __x, __x, __x, __x, __x, __x, __x,
                                       __x, __x, __x, __x, __x, __x, __x, __x};
    } else if constexpr (_N == 64) {
        return __vector_type_t<_T, 64>{
            __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x,
            __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x,
            __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x,
            __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x,
            __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x};
    } else if constexpr (_N == 128) {
        return __vector_type_t<_T, 128>{
            __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x,
            __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x,
            __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x,
            __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x,
            __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x,
            __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x,
            __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x,
            __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x,
            __x, __x, __x, __x, __x, __x, __x, __x};
    }
}

// }}}
// __auto_broadcast{{{
template <class _T> struct __auto_broadcast {
    const _T __x;
    _GLIBCXX_SIMD_INTRINSIC constexpr __auto_broadcast(_T xx) : __x(xx) {}
    template <class _V> _GLIBCXX_SIMD_INTRINSIC constexpr operator _V() const
    {
        static_assert(__is_vector_type_v<_V>);
        return reinterpret_cast<_V>(__vector_broadcast<sizeof(_V) / sizeof(_T)>(__x));
    }
};

// }}}
// __generate_builtin{{{
template <class _T, size_t _N, class _G, size_t... _I>
_GLIBCXX_SIMD_INTRINSIC constexpr __vector_type_t<_T, _N> generate_builtin_impl(
    _G &&gen, std::index_sequence<_I...>)
{
    return __vector_type_t<_T, _N>{static_cast<_T>(gen(__size_constant<_I>()))...};
}

template <class _V, class _VVT = __vector_traits<_V>, class _G>
_GLIBCXX_SIMD_INTRINSIC constexpr _V __generate_builtin(_G &&gen)
{
    return generate_builtin_impl<typename _VVT::value_type, _VVT::width>(
        std::forward<_G>(gen), std::make_index_sequence<_VVT::width>());
}

template <class _T, size_t _N, class _G>
_GLIBCXX_SIMD_INTRINSIC constexpr __vector_type_t<_T, _N> __generate_builtin(_G &&gen)
{
    return generate_builtin_impl<_T, _N>(std::forward<_G>(gen),
                                       std::make_index_sequence<_N>());
}

// }}}
// __vector_load{{{
template <class _T, size_t _N, size_t _M = _N * sizeof(_T), class _F>
__vector_type_t<_T, _N> __vector_load(const void *p, _F)
{
#ifdef _GLIBCXX_SIMD_WORKAROUND_XXX_2
    using _U = std::conditional_t<
        (std::is_integral_v<_T> || _M < 4), long long,
        std::conditional_t<(std::is_same_v<_T, double> || _M < 8), float, _T>>;
    using _V = __vector_type_t<_U, _N * sizeof(_T) / sizeof(_U)>;
#else   // _GLIBCXX_SIMD_WORKAROUND_XXX_2
    using _V = __vector_type_t<_T, _N>;
#endif  // _GLIBCXX_SIMD_WORKAROUND_XXX_2
    _V __r;
    static_assert(_M <= sizeof(_V));
    if constexpr(std::is_same_v<_F, element_aligned_tag>) {
    } else if constexpr(std::is_same_v<_F, vector_aligned_tag>) {
        p = __builtin_assume_aligned(p, alignof(__vector_type_t<_T, _N>));
    } else {
        p = __builtin_assume_aligned(p, _F::alignment);
    }
    std::memcpy(&__r, p, _M);
    return reinterpret_cast<__vector_type_t<_T, _N>>(__r);
}

// }}}
// __vector_load16 {{{
template <class _T, size_t _M = 16, class _F>
__vector_type16_t<_T> __vector_load16(const void *p, _F f)
{
    return __vector_load<_T, 16 / sizeof(_T), _M>(p, f);
}

// }}}
// __vector_store{{{
template <size_t _M = 0, class _B, class _BVT = __vector_traits<_B>, class _F>
void __vector_store(const _B v, void *p, _F)
{
    using _T = typename _BVT::value_type;
    constexpr size_t _N = _BVT::width;
    constexpr size_t _Bytes = _M == 0 ? _N * sizeof(_T) : _M;
    static_assert(_Bytes <= sizeof(v));
#ifdef _GLIBCXX_SIMD_WORKAROUND_XXX_2
    using _U = std::conditional_t<
        (std::is_integral_v<_T> || _Bytes < 4), long long,
        std::conditional_t<(std::is_same_v<_T, double> || _Bytes < 8), float, _T>>;
    const auto vv = __vector_bitcast<_U>(v);
#else   // _GLIBCXX_SIMD_WORKAROUND_XXX_2
    const __vector_type_t<_T, _N> vv = v;
#endif  // _GLIBCXX_SIMD_WORKAROUND_XXX_2
    if constexpr(std::is_same_v<_F, vector_aligned_tag>) {
        p = __builtin_assume_aligned(p, alignof(__vector_type_t<_T, _N>));
    } else if constexpr(!std::is_same_v<_F, element_aligned_tag>) {
        p = __builtin_assume_aligned(p, _F::alignment);
    }
    if constexpr ((_Bytes & (_Bytes - 1)) != 0) {
        constexpr size_t MoreBytes = __next_power_of_2(_Bytes);
        alignas(MoreBytes) char __tmp[MoreBytes];
        std::memcpy(__tmp, &vv, MoreBytes);
        std::memcpy(p, __tmp, _Bytes);
    } else {
        std::memcpy(p, &vv, _Bytes);
    }
}

// }}}
// __allbits{{{
template <class _V>
inline constexpr _V __allbits =
    reinterpret_cast<_V>(~__intrinsic_type_t<__llong, sizeof(_V) / sizeof(__llong)>());

// }}}
// __xor{{{
template <class _T, class _TVT = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC constexpr _T __xor(_T a, typename _TVT::type b) noexcept
{
    return reinterpret_cast<_T>(__vector_bitcast<unsigned>(a) ^ __vector_bitcast<unsigned>(b));
}

// }}}
// __or{{{
template <class _T, class _TVT = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC constexpr _T __or(_T a, typename _TVT::type b) noexcept
{
    return reinterpret_cast<_T>(__vector_bitcast<unsigned>(a) | __vector_bitcast<unsigned>(b));
}

// }}}
// __and{{{
template <class _T, class _TVT = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC constexpr _T __and(_T a, typename _TVT::type b) noexcept
{
    return reinterpret_cast<_T>(__vector_bitcast<unsigned>(a) & __vector_bitcast<unsigned>(b));
}

// }}}
// __andnot{{{
template <class _T, class _TVT = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC constexpr _T __andnot(_T a, typename _TVT::type b) noexcept
{
    return reinterpret_cast<_T>(~__vector_bitcast<unsigned>(a) & __vector_bitcast<unsigned>(b));
}

// }}}
// __not{{{
template <class _T, class _TVT = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC constexpr _T __not(_T a) noexcept
{
    return reinterpret_cast<_T>(~__vector_bitcast<unsigned>(a));
}

// }}}
// __concat{{{
template <class _T, class _TVT = __vector_traits<_T>,
          class _R = __vector_type_t<typename _TVT::value_type, _TVT::width * 2>>
constexpr _R __concat(_T a_, _T b_) {
#ifdef _GLIBCXX_SIMD_WORKAROUND_XXX_1
    using _W = std::conditional_t<std::is_floating_point_v<typename _TVT::value_type>,
                                 double, long long>;
    constexpr int input_width = sizeof(_T) / sizeof(_W);
    const auto a = __vector_bitcast<_W>(a_);
    const auto b = __vector_bitcast<_W>(b_);
    using _U = __vector_type_t<_W, sizeof(_R) / sizeof(_W)>;
#else
    constexpr int input_width = _TVT::width;
    const _T &a = a_;
    const _T &b = b_;
    using _U = _R;
#endif
    if constexpr(input_width == 2) {
        return reinterpret_cast<_R>(_U{a[0], a[1], b[0], b[1]});
    } else if constexpr (input_width == 4) {
        return reinterpret_cast<_R>(_U{a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]});
    } else if constexpr (input_width == 8) {
        return reinterpret_cast<_R>(_U{a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], b[0],
                                     b[1], b[2], b[3], b[4], b[5], b[6], b[7]});
    } else if constexpr (input_width == 16) {
        return reinterpret_cast<_R>(
            _U{a[0],  a[1],  a[2],  a[3],  a[4],  a[5],  a[6],  a[7],  a[8],  a[9], a[10],
              a[11], a[12], a[13], a[14], a[15], b[0],  b[1],  b[2],  b[3],  b[4], b[5],
              b[6],  b[7],  b[8],  b[9],  b[10], b[11], b[12], b[13], b[14], b[15]});
    } else if constexpr (input_width == 32) {
        return reinterpret_cast<_R>(
            _U{a[0],  a[1],  a[2],  a[3],  a[4],  a[5],  a[6],  a[7],  a[8],  a[9],  a[10],
              a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20], a[21],
              a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30], a[31], b[0],
              b[1],  b[2],  b[3],  b[4],  b[5],  b[6],  b[7],  b[8],  b[9],  b[10], b[11],
              b[12], b[13], b[14], b[15], b[16], b[17], b[18], b[19], b[20], b[21], b[22],
              b[23], b[24], b[25], b[26], b[27], b[28], b[29], b[30], b[31]});
    }
}

// }}}
// __zero_extend {{{
template <class _T, class _TVT = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC auto __zero_extend(_T __x)
{
    using value_type = typename _TVT::value_type;
    constexpr size_t _N = _TVT::width;
    struct {
        _T __x;
        operator __vector_type_t<value_type, _N * 2>()
        {
#ifdef _GLIBCXX_SIMD_WORKAROUND_XXX_3
            if constexpr (__have_avx && _TVT::template is<float, 4>) {
                return __vector_bitcast<value_type>(_mm256_insertf128_ps(__m256(), __x, 0));
            } else if constexpr (__have_avx && _TVT::template is<double, 2>) {
                return __vector_bitcast<value_type>(_mm256_insertf128_pd(__m256d(), __x, 0));
            } else if constexpr (__have_avx2 && sizeof(__x) == 16) {
                return __vector_bitcast<value_type>(_mm256_insertf128_si256(__m256i(), __x, 0));
            } else if constexpr (__have_avx512f && _TVT::template is<float, 8>) {
                if constexpr (__have_avx512dq) {
                    return __vector_bitcast<value_type>(_mm512_insertf32x8(__m512(), __x, 0));
                } else {
                    return reinterpret_cast<__m512>(
                        _mm512_insertf64x4(__m512d(), reinterpret_cast<__m256d>(__x), 0));
                }
            } else if constexpr (__have_avx512f && _TVT::template is<double, 4>) {
                return __vector_bitcast<value_type>(_mm512_insertf64x4(__m512d(), __x, 0));
            } else if constexpr (__have_avx512f && sizeof(__x) == 32) {
                return __vector_bitcast<value_type>(_mm512_inserti64x4(__m512i(), __x, 0));
            }
#endif
            return __concat(__x, _T());
        }
        operator __vector_type_t<value_type, _N * 4>()
        {
#ifdef _GLIBCXX_SIMD_WORKAROUND_XXX_3
            if constexpr (__have_avx && _TVT::template is<float, 4>) {
#ifdef _GLIBCXX_SIMD_WORKAROUND_PR85480
                asm("vmovaps %0, %0" : "+__x"(__x));
                return __vector_bitcast<value_type>(_mm512_castps128_ps512(__x));
#else
                return __vector_bitcast<value_type>(_mm512_insertf32x4(__m512(), __x, 0));
#endif
            } else if constexpr (__have_avx && _TVT::template is<double, 2>) {
#ifdef _GLIBCXX_SIMD_WORKAROUND_PR85480
                asm("vmovapd %0, %0" : "+__x"(__x));
                return __vector_bitcast<value_type>(_mm512_castpd128_pd512(__x));
#else
                return __vector_bitcast<value_type>(_mm512_insertf64x2(__m512d(), __x, 0));
#endif
            } else if constexpr (__have_avx512f && sizeof(__x) == 16) {
#ifdef _GLIBCXX_SIMD_WORKAROUND_PR85480
                asm("vmovadq %0, %0" : "+__x"(__x));
                return __vector_bitcast<value_type>(_mm512_castsi128_si512(__x));
#else
                return __vector_bitcast<value_type>(_mm512_inserti32x4(__m512i(), __x, 0));
#endif
            }
#endif
            return __concat(__concat(__x, _T()), __vector_type_t<value_type, _N * 2>());
        }
        operator __vector_type_t<value_type, _N * 8>()
        {
            return __concat(operator __vector_type_t<value_type, _N * 4>(),
                          __vector_type_t<value_type, _N * 4>());
        }
        operator __vector_type_t<value_type, _N * 16>()
        {
            return __concat(operator __vector_type_t<value_type, _N * 8>(),
                          __vector_type_t<value_type, _N * 8>());
        }
    } __r{__x};
    return __r;
}

// }}}
// __extract<_N, By>{{{
template <int _Offset, int _SplitBy, class _T, class _TVT = __vector_traits<_T>,
          class _R = __vector_type_t<typename _TVT::value_type, _TVT::width / _SplitBy>>
_GLIBCXX_SIMD_INTRINSIC constexpr _R __extract(_T __in)
{
#ifdef _GLIBCXX_SIMD_WORKAROUND_XXX_1
    using _W = std::conditional_t<std::is_floating_point_v<typename _TVT::value_type>,
                                 double, long long>;
    constexpr int return_width = sizeof(_R) / sizeof(_W);
    using _U = __vector_type_t<_W, return_width>;
    const auto __x = __vector_bitcast<_W>(__in);
#else
    constexpr int return_width = _TVT::width / _SplitBy;
    using _U = _R;
    const __vector_type_t<typename _TVT::value_type, _TVT::width> &__x =
        __in;  // only needed for _T = __storage<value_type, _N>
#endif
    constexpr int _O = _Offset * return_width;
    if constexpr (return_width == 2) {
        return reinterpret_cast<_R>(_U{__x[_O + 0], __x[_O + 1]});
    } else if constexpr (return_width == 4) {
        return reinterpret_cast<_R>(
            _U{__x[_O + 0], __x[_O + 1], __x[_O + 2], __x[_O + 3]});
    } else if constexpr (return_width == 8) {
        return reinterpret_cast<_R>(_U{__x[_O + 0], __x[_O + 1], __x[_O + 2], __x[_O + 3],
                                      __x[_O + 4], __x[_O + 5], __x[_O + 6],
                                      __x[_O + 7]});
    } else if constexpr (return_width == 16) {
        return reinterpret_cast<_R>(_U{
            __x[_O + 0], __x[_O + 1], __x[_O + 2], __x[_O + 3], __x[_O + 4], __x[_O + 5],
            __x[_O + 6], __x[_O + 7], __x[_O + 8], __x[_O + 9], __x[_O + 10],
            __x[_O + 11], __x[_O + 12], __x[_O + 13], __x[_O + 14], __x[_O + 15]});
    } else if constexpr (return_width == 32) {
        return reinterpret_cast<_R>(
            _U{__x[_O + 0],  __x[_O + 1],  __x[_O + 2],  __x[_O + 3],  __x[_O + 4],
               __x[_O + 5],  __x[_O + 6],  __x[_O + 7],  __x[_O + 8],  __x[_O + 9],
               __x[_O + 10], __x[_O + 11], __x[_O + 12], __x[_O + 13], __x[_O + 14],
               __x[_O + 15], __x[_O + 16], __x[_O + 17], __x[_O + 18], __x[_O + 19],
               __x[_O + 20], __x[_O + 21], __x[_O + 22], __x[_O + 23], __x[_O + 24],
               __x[_O + 25], __x[_O + 26], __x[_O + 27], __x[_O + 28], __x[_O + 29],
               __x[_O + 30], __x[_O + 31]});
    }
}

// }}}
// __lo/__hi128{{{
template <class _T> _GLIBCXX_SIMD_INTRINSIC constexpr auto __lo128(_T __x)
{
    return __extract<0, sizeof(_T) / 16>(__x);
}
template <class _T> _GLIBCXX_SIMD_INTRINSIC constexpr auto __hi128(_T __x)
{
    static_assert(sizeof(__x) == 32);
    return __extract<1, 2>(__x);
}

// }}}
// __lo/__hi256{{{
template <class _T> _GLIBCXX_SIMD_INTRINSIC constexpr auto __lo256(_T __x)
{
    static_assert(sizeof(__x) == 64);
    return __extract<0, 2>(__x);
}
template <class _T> _GLIBCXX_SIMD_INTRINSIC constexpr auto __hi256(_T __x)
{
    static_assert(sizeof(__x) == 64);
    return __extract<1, 2>(__x);
}

// }}}
// __intrin_bitcast{{{
template <class _To, class _From> _GLIBCXX_SIMD_INTRINSIC constexpr _To __intrin_bitcast(_From v)
{
    static_assert(__is_vector_type_v<_From> && __is_vector_type_v<_To>);
    if constexpr (sizeof(_To) == sizeof(_From)) {
        return reinterpret_cast<_To>(v);
    } else if constexpr (sizeof(_From) > sizeof(_To)) {
        return reinterpret_cast<const _To &>(v);
    } else if constexpr (__have_avx && sizeof(_From) == 16 && sizeof(_To) == 32) {
        return reinterpret_cast<_To>(_mm256_castps128_ps256(
            reinterpret_cast<__intrinsic_type_t<float, sizeof(_From) / sizeof(float)>>(v)));
    } else if constexpr (__have_avx512f && sizeof(_From) == 16 && sizeof(_To) == 64) {
        return reinterpret_cast<_To>(_mm512_castps128_ps512(
            reinterpret_cast<__intrinsic_type_t<float, sizeof(_From) / sizeof(float)>>(v)));
    } else if constexpr (__have_avx512f && sizeof(_From) == 32 && sizeof(_To) == 64) {
        return reinterpret_cast<_To>(_mm512_castps256_ps512(
            reinterpret_cast<__intrinsic_type_t<float, sizeof(_From) / sizeof(float)>>(v)));
    } else {
        __assert_unreachable<_To>();
    }
}

// }}}
// __auto_bitcast{{{
template <class _T> struct auto_cast_t {
    static_assert(__is_vector_type_v<_T>);
    const _T __x;
    template <class _U> _GLIBCXX_SIMD_INTRINSIC constexpr operator _U() const
    {
        return __intrin_bitcast<_U>(__x);
    }
};
template <class _T> _GLIBCXX_SIMD_INTRINSIC constexpr auto_cast_t<_T> __auto_bitcast(const _T &__x)
{
    return {__x};
}
template <class _T, size_t _N>
_GLIBCXX_SIMD_INTRINSIC constexpr auto_cast_t<typename __storage<_T, _N>::register_type> __auto_bitcast(
    const __storage<_T, _N> &__x)
{
    return {__x._M_data};
}

// }}}
// __vector_to_bitset{{{
_GLIBCXX_SIMD_INTRINSIC constexpr std::bitset<1> __vector_to_bitset(bool __x) { return unsigned(__x); }

template <class _T, class = enable_if_t<__is_bitmask_v<_T> && __have_avx512f>>
_GLIBCXX_SIMD_INTRINSIC constexpr std::bitset<8 * sizeof(_T)> __vector_to_bitset(_T __x)
{
    if constexpr (std::is_integral_v<_T>) {
        return __x;
    } else {
        return __x._M_data;
    }
}

template <class _T, class _TVT = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC std::bitset<_TVT::width> __vector_to_bitset(_T __x)
{
    constexpr bool __is_sse = __have_sse && sizeof(_T) == 16;
    constexpr bool __is_avx = __have_avx && sizeof(_T) == 32;
    constexpr bool is_neon128 = __have_neon && sizeof(_T) == 16;
    constexpr int w = sizeof(typename _TVT::value_type);
    const auto intrin = __to_intrin(__x);
    constexpr auto zero = decltype(intrin)();
    __unused(zero);

    if constexpr (is_neon128 && w == 1) {
        __x &= _T{0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80,
               0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80};
        return __vector_bitcast<ushort>(
            vpaddq_s8(vpaddq_s8(vpaddq_s8(__x, zero), zero), zero))[0];
    } else if constexpr (is_neon128 && w == 2) {
        __x &= _T{0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80};
        return vpaddq_s16(vpaddq_s16(vpaddq_s16(__x, zero), zero), zero)[0];
    } else if constexpr (is_neon128 && w == 4) {
        __x &= _T{0x1, 0x2, 0x4, 0x8};
        return vpaddq_s32(vpaddq_s32(__x, zero), zero)[0];
    } else if constexpr (is_neon128 && w == 8) {
        __x &= _T{0x1, 0x2};
        return __x[0] | __x[1];
    } else if constexpr (__is_sse && w == 1) {
        return _mm_movemask_epi8(intrin);
    } else if constexpr (__is_sse && w == 2) {
        if constexpr (__have_avx512bw_vl) {
            return _mm_cmplt_epi16_mask(intrin, zero);
        } else {
            return _mm_movemask_epi8(_mm_packs_epi16(intrin, zero));
        }
    } else if constexpr (__is_sse && w == 4) {
        if constexpr (__have_avx512vl && std::is_integral_v<_T>) {
            return _mm_cmplt_epi32_mask(intrin, zero);
        } else {
            return _mm_movemask_ps(__vector_bitcast<float>(__x));
        }
    } else if constexpr (__is_sse && w == 8) {
        if constexpr (__have_avx512vl && std::is_integral_v<_T>) {
            return _mm_cmplt_epi64_mask(intrin, zero);
        } else {
            return _mm_movemask_pd(__vector_bitcast<double>(__x));
        }
    } else if constexpr (__is_avx && w == 1) {
        return _mm256_movemask_epi8(intrin);
    } else if constexpr (__is_avx && w == 2) {
        if constexpr (__have_avx512bw_vl) {
            return _mm256_cmplt_epi16_mask(intrin, zero);
        } else {
            return _mm_movemask_epi8(_mm_packs_epi16(__extract<0, 2>(intrin),
                                                     __extract<1, 2>(intrin)));
        }
    } else if constexpr (__is_avx && w == 4) {
        if constexpr (__have_avx512vl && std::is_integral_v<_T>) {
            return _mm256_cmplt_epi32_mask(intrin, zero);
        } else {
            return _mm256_movemask_ps(__vector_bitcast<float>(__x));
        }
    } else if constexpr (__is_avx && w == 8) {
        if constexpr (__have_avx512vl && std::is_integral_v<_T>) {
            return _mm256_cmplt_epi64_mask(intrin, zero);
        } else {
            return _mm256_movemask_pd(__vector_bitcast<double>(__x));
        }
    } else {
        __assert_unreachable<_T>();
    }
}

// }}}
// __blend{{{
template <class _K, class _V0, class _V1>
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST auto __blend(_K mask, _V0 at0, _V1 at1)
{
    using _V = _V0;
    if constexpr (!std::is_same_v<_V0, _V1>) {
        static_assert(sizeof(_V0) == sizeof(_V1));
        if constexpr (__is_vector_type_v<_V0> && !__is_vector_type_v<_V1>) {
            return __blend(mask, at0, reinterpret_cast<_V0>(at1._M_data));
        } else if constexpr (!__is_vector_type_v<_V0> && __is_vector_type_v<_V1>) {
            return __blend(mask, reinterpret_cast<_V1>(at0._M_data), at1);
        } else {
            __assert_unreachable<_K>();
        }
    } else if constexpr (sizeof(_V) < 16) {
        static_assert(sizeof(_K) == sizeof(_V0) && sizeof(_V0) == sizeof(_V1));
        return (mask & at1) | (~mask & at0);
    } else if constexpr (!__is_vector_type_v<_V>) {
        return __blend(mask, at0._M_data, at1._M_data);
    } else if constexpr (sizeof(_K) < 16) {
        using _T = typename __vector_traits<_V>::value_type;
        if constexpr (sizeof(_V) == 16 && __have_avx512bw_vl && sizeof(_T) <= 2) {
            if constexpr (sizeof(_T) == 1) {
                return __intrin_bitcast<_V>(
                    _mm_mask_mov_epi8(__to_intrin(at0), mask, __to_intrin(at1)));
            } else if constexpr (sizeof(_T) == 2) {
                return __intrin_bitcast<_V>(
                    _mm_mask_mov_epi16(__to_intrin(at0), mask, __to_intrin(at1)));
            }
        } else if constexpr (sizeof(_V) == 16 && __have_avx512vl && sizeof(_T) > 2) {
            if constexpr (std::is_integral_v<_T> && sizeof(_T) == 4) {
                return __intrin_bitcast<_V>(
                    _mm_mask_mov_epi32(__to_intrin(at0), mask, __to_intrin(at1)));
            } else if constexpr (std::is_integral_v<_T> && sizeof(_T) == 8) {
                return __intrin_bitcast<_V>(
                    _mm_mask_mov_epi64(__to_intrin(at0), mask, __to_intrin(at1)));
            } else if constexpr (std::is_floating_point_v<_T> && sizeof(_T) == 4) {
                return __intrin_bitcast<_V>(_mm_mask_mov_ps(at0, mask, at1));
            } else if constexpr (std::is_floating_point_v<_T> && sizeof(_T) == 8) {
                return __intrin_bitcast<_V>(_mm_mask_mov_pd(at0, mask, at1));
            }
        } else if constexpr (sizeof(_V) == 16 && __have_avx512f && sizeof(_T) > 2) {
            if constexpr (std::is_integral_v<_T> && sizeof(_T) == 4) {
                return __intrin_bitcast<_V>(__lo128(_mm512_mask_mov_epi32(
                    __auto_bitcast(at0), mask, __auto_bitcast(at1))));
            } else if constexpr (std::is_integral_v<_T> && sizeof(_T) == 8) {
                return __intrin_bitcast<_V>(__lo128(_mm512_mask_mov_epi64(
                    __auto_bitcast(at0), mask, __auto_bitcast(at1))));
            } else if constexpr (std::is_floating_point_v<_T> && sizeof(_T) == 4) {
                return __intrin_bitcast<_V>(__lo128(
                    _mm512_mask_mov_ps(__auto_bitcast(at0), mask, __auto_bitcast(at1))));
            } else if constexpr (std::is_floating_point_v<_T> && sizeof(_T) == 8) {
                return __intrin_bitcast<_V>(__lo128(
                    _mm512_mask_mov_pd(__auto_bitcast(at0), mask, __auto_bitcast(at1))));
            }
        } else if constexpr (sizeof(_V) == 32 && __have_avx512bw_vl && sizeof(_T) <= 2) {
            if constexpr (sizeof(_T) == 1) {
                return __intrin_bitcast<_V>(
                    _mm256_mask_mov_epi8(__to_intrin(at0), mask, __to_intrin(at1)));
            } else if constexpr (sizeof(_T) == 2) {
                return __intrin_bitcast<_V>(
                    _mm256_mask_mov_epi16(__to_intrin(at0), mask, __to_intrin(at1)));
            }
        } else if constexpr (sizeof(_V) == 32 && __have_avx512vl && sizeof(_T) > 2) {
            if constexpr (std::is_integral_v<_T> && sizeof(_T) == 4) {
                return __intrin_bitcast<_V>(
                    _mm256_mask_mov_epi32(__to_intrin(at0), mask, __to_intrin(at1)));
            } else if constexpr (std::is_integral_v<_T> && sizeof(_T) == 8) {
                return __intrin_bitcast<_V>(
                    _mm256_mask_mov_epi64(__to_intrin(at0), mask, __to_intrin(at1)));
            } else if constexpr (std::is_floating_point_v<_T> && sizeof(_T) == 4) {
                return __intrin_bitcast<_V>(_mm256_mask_mov_ps(at0, mask, at1));
            } else if constexpr (std::is_floating_point_v<_T> && sizeof(_T) == 8) {
                return __intrin_bitcast<_V>(_mm256_mask_mov_pd(at0, mask, at1));
            }
        } else if constexpr (sizeof(_V) == 32 && __have_avx512f && sizeof(_T) > 2) {
            if constexpr (std::is_integral_v<_T> && sizeof(_T) == 4) {
                return __intrin_bitcast<_V>(__lo256(_mm512_mask_mov_epi32(
                    __auto_bitcast(at0), mask, __auto_bitcast(at1))));
            } else if constexpr (std::is_integral_v<_T> && sizeof(_T) == 8) {
                return __intrin_bitcast<_V>(__lo256(_mm512_mask_mov_epi64(
                    __auto_bitcast(at0), mask, __auto_bitcast(at1))));
            } else if constexpr (std::is_floating_point_v<_T> && sizeof(_T) == 4) {
                return __intrin_bitcast<_V>(__lo256(
                    _mm512_mask_mov_ps(__auto_bitcast(at0), mask, __auto_bitcast(at1))));
            } else if constexpr (std::is_floating_point_v<_T> && sizeof(_T) == 8) {
                return __intrin_bitcast<_V>(__lo256(
                    _mm512_mask_mov_pd(__auto_bitcast(at0), mask, __auto_bitcast(at1))));
            }
        } else if constexpr (sizeof(_V) == 64 && __have_avx512bw && sizeof(_T) <= 2) {
            if constexpr (sizeof(_T) == 1) {
                return __intrin_bitcast<_V>(
                    _mm512_mask_mov_epi8(__to_intrin(at0), mask, __to_intrin(at1)));
            } else if constexpr (sizeof(_T) == 2) {
                return __intrin_bitcast<_V>(
                    _mm512_mask_mov_epi16(__to_intrin(at0), mask, __to_intrin(at1)));
            }
        } else if constexpr (sizeof(_V) == 64 && __have_avx512f && sizeof(_T) > 2) {
            if constexpr (std::is_integral_v<_T> && sizeof(_T) == 4) {
                return __intrin_bitcast<_V>(
                    _mm512_mask_mov_epi32(__to_intrin(at0), mask, __to_intrin(at1)));
            } else if constexpr (std::is_integral_v<_T> && sizeof(_T) == 8) {
                return __intrin_bitcast<_V>(
                    _mm512_mask_mov_epi64(__to_intrin(at0), mask, __to_intrin(at1)));
            } else if constexpr (std::is_floating_point_v<_T> && sizeof(_T) == 4) {
                return __intrin_bitcast<_V>(_mm512_mask_mov_ps(at0, mask, at1));
            } else if constexpr (std::is_floating_point_v<_T> && sizeof(_T) == 8) {
                return __intrin_bitcast<_V>(_mm512_mask_mov_pd(at0, mask, at1));
            }
        } else {
            __assert_unreachable<_K>();
        }
    } else {
        const _V __k = __auto_bitcast(mask);
        using _T = typename __vector_traits<_V>::value_type;
        if constexpr (sizeof(_V) == 16 && __have_sse4_1) {
            if constexpr (std::is_integral_v<_T>) {
                return __intrin_bitcast<_V>(
                    _mm_blendv_epi8(__to_intrin(at0), __to_intrin(at1), __to_intrin(__k)));
            } else if constexpr (sizeof(_T) == 4) {
                return __intrin_bitcast<_V>(_mm_blendv_ps(at0, at1, __k));
            } else if constexpr (sizeof(_T) == 8) {
                return __intrin_bitcast<_V>(_mm_blendv_pd(at0, at1, __k));
            }
        } else if constexpr (sizeof(_V) == 32) {
            if constexpr (std::is_integral_v<_T>) {
                return __intrin_bitcast<_V>(_mm256_blendv_epi8(
                    __to_intrin(at0), __to_intrin(at1), __to_intrin(__k)));
            } else if constexpr (sizeof(_T) == 4) {
                return __intrin_bitcast<_V>(_mm256_blendv_ps(at0, at1, __k));
            } else if constexpr (sizeof(_T) == 8) {
                return __intrin_bitcast<_V>(_mm256_blendv_pd(at0, at1, __k));
            }
        } else {
            return __or(__andnot(__k, at0), __and(__k, at1));
        }
    }
}

// }}}
// __is_zero{{{
template <class _T, class _TVT = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC constexpr bool __is_zero(_T __a)
{
    const auto __b = __vector_bitcast<__llong>(__a);
    if constexpr (sizeof(__b) / sizeof(__llong) == 2) {
        return __b[0] == 0 && __b[1] == 0;
    } else if constexpr (sizeof(__b) / sizeof(__llong) == 4) {
        return __b[0] == 0 && __b[1] == 0 && __b[2] == 0 && __b[3] == 0;
    } else if constexpr (sizeof(__b) / sizeof(__llong) == 8) {
        return __b[0] == 0 && __b[1] == 0 && __b[2] == 0 && __b[3] == 0 && __b[4] == 0 &&
               __b[5] == 0 && __b[6] == 0 && __b[7] == 0;
    } else {
        __assert_unreachable<_T>();
    }
}
// }}}
// ^^^ ---- builtin vector types [[gnu::vector_size(N)]] and operations ---- ^^^

// __testz{{{
template <class _T, class _TVT = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST int __testz(_T a, _T b)
{
    if constexpr (__have_avx) {
        if constexpr (sizeof(_T) == 32 && _TVT::template is<float>) {
            return _mm256_testz_ps(a, b);
        } else if constexpr (sizeof(_T) == 32 && _TVT::template is<double>) {
            return _mm256_testz_pd(a, b);
        } else if constexpr (sizeof(_T) == 32) {
            return _mm256_testz_si256(__vector_bitcast<__llong>(a),
                                      __vector_bitcast<__llong>(b));
        } else if constexpr(_TVT::template is<float, 4>) {
            return _mm_testz_ps(a, b);
        } else if constexpr(_TVT::template is<double, 2>) {
            return _mm_testz_pd(a, b);
        } else {
            static_assert(sizeof(_T) == 16);
            return _mm_testz_si128(__vector_bitcast<__llong>(a), __vector_bitcast<__llong>(b));
        }
    } else if constexpr (__have_sse4_1) {
        return _mm_testz_si128(__vector_bitcast<__llong>(a), __vector_bitcast<__llong>(b));
    } else if constexpr (__have_sse && _TVT::template is<float, 4>) {
        return _mm_movemask_ps(__and(a, b)) == 0;
    } else if constexpr (__have_sse2 && _TVT::template is<double, 2>) {
        return _mm_movemask_pd(__and(a, b)) == 0;
    } else if constexpr (__have_sse2) {
        return _mm_movemask_epi8(a & b) == 0;
    } else {
        return __is_zero(__and(a, b));
    }
}

// }}}
// __testnzc{{{
template <class _T, class _TVT = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST int __testnzc(_T a, _T b)
{
    if constexpr (__have_avx) {
        if constexpr (sizeof(_T) == 32 && _TVT::template is<float>) {
            return _mm256_testnzc_ps(a, b);
        } else if constexpr (sizeof(_T) == 32 && _TVT::template is<double>) {
            return _mm256_testnzc_pd(a, b);
        } else if constexpr (sizeof(_T) == 32) {
            return _mm256_testnzc_si256(__vector_bitcast<__llong>(a), __vector_bitcast<__llong>(b));
        } else if constexpr(_TVT::template is<float, 4>) {
            return _mm_testnzc_ps(a, b);
        } else if constexpr(_TVT::template is<double, 2>) {
            return _mm_testnzc_pd(a, b);
        } else {
            static_assert(sizeof(_T) == 16);
            return _mm_testnzc_si128(__vector_bitcast<__llong>(a), __vector_bitcast<__llong>(b));
        }
    } else if constexpr (__have_sse4_1) {
        return _mm_testnzc_si128(__vector_bitcast<__llong>(a), __vector_bitcast<__llong>(b));
    } else if constexpr (__have_sse && _TVT::template is<float, 4>) {
        return _mm_movemask_ps(__and(a, b)) == 0 && _mm_movemask_ps(__andnot(a, b)) == 0;
    } else if constexpr (__have_sse2 && _TVT::template is<double, 2>) {
        return _mm_movemask_pd(__and(a, b)) == 0 && _mm_movemask_pd(__andnot(a, b)) == 0;
    } else if constexpr (__have_sse2) {
        return _mm_movemask_epi8(__and(a, b)) == 0 &&
               _mm_movemask_epi8(__andnot(a, b)) == 0;
    } else {
        return !(__is_zero(__vector_bitcast<__llong>(__and(a, b))) ||
                 __is_zero(__vector_bitcast<__llong>(__andnot(a, b))));
    }
}

// }}}
// __movemask{{{
template <class _T, class _TVT = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST int __movemask(_T a)
{
    if constexpr (__have_sse && _TVT::template is<float, 4>) {
        return _mm_movemask_ps(a);
    } else if constexpr (__have_avx && _TVT::template is<float, 8>) {
        return _mm256_movemask_ps(a);
    } else if constexpr (__have_sse2 && _TVT::template is<double, 2>) {
        return _mm_movemask_pd(a);
    } else if constexpr (__have_avx && _TVT::template is<double, 4>) {
        return _mm256_movemask_pd(a);
    } else if constexpr (__have_sse2 && sizeof(_T) == 16) {
        return _mm_movemask_epi8(a);
    } else if constexpr (__have_avx2 && sizeof(_T) == 32) {
        return _mm256_movemask_epi8(a);
    } else {
        __assert_unreachable<_T>();
    }
}

template <class _T, class _TVT = __vector_traits<_T>>
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST int movemask_epi16(_T a)
{
    static_assert(std::is_integral_v<typename _TVT::value_type>);
    if constexpr(__have_avx512bw_vl && sizeof(_T) == 16) {
        return _mm_cmp_epi16_mask(a, __m128i(), _MM_CMPINT_NE);
    } else if constexpr(__have_avx512bw_vl && sizeof(_T) == 32) {
        return _mm256_cmp_epi16_mask(a, __m256i(), _MM_CMPINT_NE);
    } else if constexpr(sizeof(_T) == 32) {
        return _mm_movemask_epi8(_mm_packs_epi16(__lo128(a), __hi128(a)));
    } else {
        static_assert(sizeof(_T) == 16);
        return _mm_movemask_epi8(_mm_packs_epi16(a, __m128i()));
    }
}

// }}}
// __{double,float}_const {{{
template <int __exponent> constexpr double __double_2_pow()
{
    if constexpr (__exponent < 0) {
        return 1. / __double_2_pow<-__exponent>();
    } else if constexpr (__exponent < std::numeric_limits<unsigned long long>::digits) {
        return 1ull << __exponent;
    } else {
        return ((~0ull >> 1) + 1) * 2. *
               __double_2_pow<__exponent - std::numeric_limits<unsigned long long>::digits>();
    }
}

template <int __sign, unsigned long long __mantissa, int __exponent>
constexpr double __double_const = (static_cast<double>((__mantissa & 0x000fffffffffffffull) |
                                                     0x0010000000000000ull) /
                                 0x0010000000000000ull) *
                                __double_2_pow<__exponent>() * __sign;
template <int __sign, unsigned int __mantissa, int __exponent>
constexpr float __float_const = (float((__mantissa & 0x007fffffu) | 0x00800000u) /
                               0x00800000u) *
                              float(__double_2_pow<__exponent>()) * __sign;
// }}}
// __trig constants {{{
template <class _T> struct __trig;
template <> struct __trig<float> {
    static inline constexpr float pi_4      = __float_const< 1, 0x490FDB, -1>;
    static inline constexpr float pi_4_hi   = __float_const< 1, 0x491000, -1>;
    static inline constexpr float pi_4_rem1 = __float_const<-1, 0x157000, -19>;
    static inline constexpr float pi_4_rem2 = __float_const<-1, 0x6F4B9F, -32>;
    static inline constexpr float _1_16 = 0.0625f;
    static inline constexpr float _16 = 16.f;
    static inline constexpr float cos_c0 = 4.166664568298827e-2f;  // ~ 1/4!
    static inline constexpr float cos_c1 = -1.388731625493765e-3f; // ~-1/6!
    static inline constexpr float cos_c2 = 2.443315711809948e-5f;  // ~ 1/8!
    static inline constexpr float sin_c0 = -1.6666654611e-1f; // ~-1/3!
    static inline constexpr float sin_c1 = 8.3321608736e-3f;  // ~ 1/5!
    static inline constexpr float sin_c2 = -1.9515295891e-4f; // ~-1/7!
    static inline constexpr float loss_threshold = 8192.f; // loss threshold
    static inline constexpr float _4_pi = __float_const< 1, 0x22F983, 0>; // 1.27323949337005615234375 = 4/π
    static inline constexpr float pi_2 = __float_const< 1, 0x490FDB, 0>; // π/2
    static inline constexpr float pi = __float_const< 1, 0x490FDB, 1>; // π
    static inline constexpr float atan_p0 = 8.05374449538e-2f; // atan P coefficients
    static inline constexpr float atan_p1 = 1.38776856032e-1f; // atan P coefficients
    static inline constexpr float atan_p2 = 1.99777106478e-1f; // atan P coefficients
    static inline constexpr float atan_p3 = 3.33329491539e-1f; // atan P coefficients
    static inline constexpr float atan_threshold_hi = 2.414213562373095f; // tan( 3/8 π )
    static inline constexpr float atan_threshold_lo = 0.414213562373095f; // tan( 1/8 π ) lower threshold for special casing in atan
    static inline constexpr float pi_2_rem = __float_const<-1, 0x3BBD2E, -25>; // remainder of pi/2
    static inline constexpr float small_asin_input = 1.e-4f; // small asin input threshold
    static inline constexpr float large_asin_input = 0.f; // padding (for alignment with double)
    static inline constexpr float asin_c0_0 = 4.2163199048e-2f; // asinCoeff0
    static inline constexpr float asin_c0_1 = 2.4181311049e-2f; // asinCoeff0
    static inline constexpr float asin_c0_2 = 4.5470025998e-2f; // asinCoeff0
    static inline constexpr float asin_c0_3 = 7.4953002686e-2f; // asinCoeff0
    static inline constexpr float asin_c0_4 = 1.6666752422e-1f; // asinCoeff0
};

template <> struct __trig<double> {
    static inline constexpr double pi_4      = __double_const< 1, 0x921fb54442d18, -1>; // π/4
    static inline constexpr double pi_4_hi   = __double_const< 1, 0x921fb40000000, -1>; // π/4 - 30bits precision
    static inline constexpr double pi_4_rem1 = __double_const< 1, 0x4442d00000000, -25>; // π/4 remainder1 - 32bits precision
    static inline constexpr double pi_4_rem2 = __double_const< 1, 0x8469898cc5170, -49>; // π/4 remainder2
    static inline constexpr double _1_16 = 0.0625;
    static inline constexpr double _16 = 16.;
    static inline constexpr double cos_c0  = __double_const< 1, 0x555555555554b, -5 >; // ~ 1/4!
    static inline constexpr double cos_c1  = __double_const<-1, 0x6c16c16c14f91, -10>; // ~-1/6!
    static inline constexpr double cos_c2  = __double_const< 1, 0xa01a019c844f5, -16>; // ~ 1/8!
    static inline constexpr double cos_c3  = __double_const<-1, 0x27e4f7eac4bc6, -22>; // ~-1/10!
    static inline constexpr double cos_c4  = __double_const< 1, 0x1ee9d7b4e3f05, -29>; // ~ 1/12!
    static inline constexpr double cos_c5  = __double_const<-1, 0x8fa49a0861a9b, -37>; // ~-1/14!
    static inline constexpr double sin_c0  = __double_const<-1, 0x5555555555548, -3 >; // ~-1/3!
    static inline constexpr double sin_c1  = __double_const< 1, 0x111111110f7d0, -7 >; // ~ 1/5!
    static inline constexpr double sin_c2  = __double_const<-1, 0xa01a019bfdf03, -13>; // ~-1/7!
    static inline constexpr double sin_c3  = __double_const< 1, 0x71de3567d48a1, -19>; // ~ 1/9!
    static inline constexpr double sin_c4  = __double_const<-1, 0xae5e5a9291f5d, -26>; // ~-1/11!
    static inline constexpr double sin_c5  = __double_const< 1, 0x5d8fd1fd19ccd, -33>; // ~ 1/13!
    static inline constexpr double _4_pi    = __double_const< 1, 0x8BE60DB939105, 0 >; // 4/π
    static inline constexpr double pi_2    = __double_const< 1, 0x921fb54442d18, 0 >; // π/2
    static inline constexpr double pi      = __double_const< 1, 0x921fb54442d18, 1 >; // π
    static inline constexpr double atan_p0 = __double_const<-1, 0xc007fa1f72594, -1>; // atan P coefficients
    static inline constexpr double atan_p1 = __double_const<-1, 0x028545b6b807a, 4 >; // atan P coefficients
    static inline constexpr double atan_p2 = __double_const<-1, 0x2c08c36880273, 6 >; // atan P coefficients
    static inline constexpr double atan_p3 = __double_const<-1, 0xeb8bf2d05ba25, 6 >; // atan P coefficients
    static inline constexpr double atan_p4 = __double_const<-1, 0x03669fd28ec8e, 6 >; // atan P coefficients
    static inline constexpr double atan_q0 = __double_const< 1, 0x8dbc45b14603c, 4 >; // atan Q coefficients
    static inline constexpr double atan_q1 = __double_const< 1, 0x4a0dd43b8fa25, 7 >; // atan Q coefficients
    static inline constexpr double atan_q2 = __double_const< 1, 0xb0e18d2e2be3b, 8 >; // atan Q coefficients
    static inline constexpr double atan_q3 = __double_const< 1, 0xe563f13b049ea, 8 >; // atan Q coefficients
    static inline constexpr double atan_q4 = __double_const< 1, 0x8519efbbd62ec, 7 >; // atan Q coefficients
    static inline constexpr double atan_threshold_hi = __double_const< 1, 0x3504f333f9de6, 1>; // tan( 3/8 π )
    static inline constexpr double atan_threshold_lo = 0.66;                                 // lower threshold for special casing in atan
    static inline constexpr double pi_2_rem = __double_const< 1, 0x1A62633145C07, -54>; // remainder of pi/2
    static inline constexpr double small_asin_input = 1.e-8; // small asin input threshold
    static inline constexpr double large_asin_input = 0.625; // large asin input threshold
    static inline constexpr double asin_c0_0 = __double_const< 1, 0x84fc3988e9f08, -9>; // asinCoeff0
    static inline constexpr double asin_c0_1 = __double_const<-1, 0x2079259f9290f, -1>; // asinCoeff0
    static inline constexpr double asin_c0_2 = __double_const< 1, 0xbdff5baf33e6a, 2 >; // asinCoeff0
    static inline constexpr double asin_c0_3 = __double_const<-1, 0x991aaac01ab68, 4 >; // asinCoeff0
    static inline constexpr double asin_c0_4 = __double_const< 1, 0xc896240f3081d, 4 >; // asinCoeff0
    static inline constexpr double asin_c1_0 = __double_const<-1, 0x5f2a2b6bf5d8c, 4 >; // asinCoeff1
    static inline constexpr double asin_c1_1 = __double_const< 1, 0x26219af6a7f42, 7 >; // asinCoeff1
    static inline constexpr double asin_c1_2 = __double_const<-1, 0x7fe08959063ee, 8 >; // asinCoeff1
    static inline constexpr double asin_c1_3 = __double_const< 1, 0x56709b0b644be, 8 >; // asinCoeff1
    static inline constexpr double asin_c2_0 = __double_const< 1, 0x16b9b0bd48ad3, -8>; // asinCoeff2
    static inline constexpr double asin_c2_1 = __double_const<-1, 0x34341333e5c16, -1>; // asinCoeff2
    static inline constexpr double asin_c2_2 = __double_const< 1, 0x5c74b178a2dd9, 2 >; // asinCoeff2
    static inline constexpr double asin_c2_3 = __double_const<-1, 0x04331de27907b, 4 >; // asinCoeff2
    static inline constexpr double asin_c2_4 = __double_const< 1, 0x39007da779259, 4 >; // asinCoeff2
    static inline constexpr double asin_c2_5 = __double_const<-1, 0x0656c06ceafd5, 3 >; // asinCoeff2
    static inline constexpr double asin_c3_0 = __double_const<-1, 0xd7b590b5e0eab, 3 >; // asinCoeff3
    static inline constexpr double asin_c3_1 = __double_const< 1, 0x19fc025fe9054, 6 >; // asinCoeff3
    static inline constexpr double asin_c3_2 = __double_const<-1, 0x265bb6d3576d7, 7 >; // asinCoeff3
    static inline constexpr double asin_c3_3 = __double_const< 1, 0x1705684ffbf9d, 7 >; // asinCoeff3
    static inline constexpr double asin_c3_4 = __double_const<-1, 0x898220a3607ac, 5 >; // asinCoeff3
};

// }}}
#if _GLIBCXX_SIMD_HAVE_SSE_ABI
// __bool_storage_member_type{{{
#if _GLIBCXX_SIMD_HAVE_AVX512F
template <> struct __bool_storage_member_type< 2> { using type = __mmask8 ; };
template <> struct __bool_storage_member_type< 4> { using type = __mmask8 ; };
template <> struct __bool_storage_member_type< 8> { using type = __mmask8 ; };
template <> struct __bool_storage_member_type<16> { using type = __mmask16; };
template <> struct __bool_storage_member_type<32> { using type = __mmask32; };
template <> struct __bool_storage_member_type<64> { using type = __mmask64; };
#endif  // _GLIBCXX_SIMD_HAVE_AVX512F

// }}}
// __intrinsic_type{{{
// the following excludes bool via __is_vectorizable
template <class _T>
using void_if_integral_t = std::void_t<enable_if_t<
    conjunction<std::is_integral<_T>, __is_vectorizable<_T>>::value>>;
#if _GLIBCXX_SIMD_HAVE_AVX512F
template <> struct __intrinsic_type<double, 64, void> { using type = __m512d; };
template <> struct __intrinsic_type< float, 64, void> { using type = __m512; };
template <class _T> struct __intrinsic_type<_T, 64, void_if_integral_t<_T>> { using type = __m512i; };
#endif  // _GLIBCXX_SIMD_HAVE_AVX512F

#if _GLIBCXX_SIMD_HAVE_AVX
template <> struct __intrinsic_type<double, 32, void> { using type = __m256d; };
template <> struct __intrinsic_type< float, 32, void> { using type = __m256; };
template <class _T> struct __intrinsic_type<_T, 32, void_if_integral_t<_T>> { using type = __m256i; };
#endif  // _GLIBCXX_SIMD_HAVE_AVX

#if _GLIBCXX_SIMD_HAVE_SSE
template <> struct __intrinsic_type< float, 16, void> { using type = __m128; };
template <> struct __intrinsic_type< float,  8, void> { using type = __m128; };
template <> struct __intrinsic_type< float,  4, void> { using type = __m128; };
#endif  // _GLIBCXX_SIMD_HAVE_SSE
#if _GLIBCXX_SIMD_HAVE_SSE2
template <> struct __intrinsic_type<double, 16, void> { using type = __m128d; };
template <> struct __intrinsic_type<double,  8, void> { using type = __m128d; };
template <class _T> struct __intrinsic_type<_T, 16, void_if_integral_t<_T>> { using type = __m128i; };
template <class _T> struct __intrinsic_type<_T,  8, void_if_integral_t<_T>> { using type = __m128i; };
template <class _T> struct __intrinsic_type<_T,  4, void_if_integral_t<_T>> { using type = __m128i; };
template <class _T> struct __intrinsic_type<_T,  2, void_if_integral_t<_T>> { using type = __m128i; };
template <class _T> struct __intrinsic_type<_T,  1, void_if_integral_t<_T>> { using type = __m128i; };
#endif  // _GLIBCXX_SIMD_HAVE_SSE2

// }}}
// __is_intrinsic{{{
#pragma GCC diagnostic push
// [[gnu::may_alias]] of __mXXX? leads to -Wignored-attributes
#pragma GCC diagnostic ignored "-Wignored-attributes"
template <> struct __is_intrinsic<__m128> : public true_type {};
#if _GLIBCXX_SIMD_HAVE_SSE2
template <> struct __is_intrinsic<__m128d> : public true_type {};
template <> struct __is_intrinsic<__m128i> : public true_type {};
#endif  // _GLIBCXX_SIMD_HAVE_SSE2
#if _GLIBCXX_SIMD_HAVE_AVX
template <> struct __is_intrinsic<__m256 > : public true_type {};
template <> struct __is_intrinsic<__m256d> : public true_type {};
template <> struct __is_intrinsic<__m256i> : public true_type {};
#endif  // _GLIBCXX_SIMD_HAVE_AVX
#if _GLIBCXX_SIMD_HAVE_AVX512F
template <> struct __is_intrinsic<__m512 > : public true_type {};
template <> struct __is_intrinsic<__m512d> : public true_type {};
template <> struct __is_intrinsic<__m512i> : public true_type {};
#endif  // _GLIBCXX_SIMD_HAVE_AVX512F
#pragma GCC diagnostic pop

// }}}
// __(sse|avx|avx512)_(simd|mask)_member_type{{{
template <class _T> using __sse_simd_member_type = __storage16_t<_T>;
template <class _T> using __sse_mask_member_type = __storage16_t<_T>;

template <class _T> using __avx_simd_member_type = __storage32_t<_T>;
template <class _T> using __avx_mask_member_type = __storage32_t<_T>;

template <class _T> using __avx512_simd_member_type = __storage64_t<_T>;
template <class _T> using __avx512_mask_member_type = __storage<bool, 64 / sizeof(_T)>;
template <size_t _N> using __avx512_mask_member_type_n = __storage<bool, _N>;

//}}}
#endif  // _GLIBCXX_SIMD_HAVE_SSE_ABI
// __storage<bool>{{{1
template <size_t _Width>
struct __storage<bool, _Width, std::void_t<typename __bool_storage_member_type<_Width>::type>> {
    using register_type = typename __bool_storage_member_type<_Width>::type;
    using value_type = bool;
    static constexpr size_t width = _Width;

    _GLIBCXX_SIMD_INTRINSIC constexpr __storage() = default;
    _GLIBCXX_SIMD_INTRINSIC constexpr __storage(register_type __k) : _M_data(__k){};

    _GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_PURE operator const register_type &() const { return _M_data; }
    _GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_PURE operator register_type &() { return _M_data; }

    _GLIBCXX_SIMD_INTRINSIC register_type intrin() const { return _M_data; }

    _GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_PURE value_type operator[](size_t __i) const
    {
        return _M_data & (register_type(1) << __i);
    }
    _GLIBCXX_SIMD_INTRINSIC void set(size_t __i, value_type __x)
    {
        if (__x) {
            _M_data |= (register_type(1) << __i);
        } else {
            _M_data &= ~(register_type(1) << __i);
        }
    }

    register_type _M_data;
};

// __storage_base{{{1
template <class _T, size_t _Width, class _RegisterType = __vector_type_t<_T, _Width>,
          bool = std::disjunction_v<
              std::is_same<__vector_type_t<_T, _Width>, __intrinsic_type_t<_T, _Width>>,
              std::is_same<_RegisterType, __intrinsic_type_t<_T, _Width>>>>
struct __storage_base;

template <class _T, size_t _Width, class _RegisterType>
struct __storage_base<_T, _Width, _RegisterType, true> {
    _RegisterType _M_data;
    _GLIBCXX_SIMD_INTRINSIC constexpr __storage_base() = default;
    _GLIBCXX_SIMD_INTRINSIC constexpr __storage_base(__vector_type_t<_T, _Width> __x)
        : _M_data(reinterpret_cast<_RegisterType>(__x))
    {
    }
};

template <class _T, size_t _Width, class _RegisterType>
struct __storage_base<_T, _Width, _RegisterType, false> {
    using intrin_type = __intrinsic_type_t<_T, _Width>;
    _RegisterType _M_data;

    _GLIBCXX_SIMD_INTRINSIC constexpr __storage_base() = default;
    _GLIBCXX_SIMD_INTRINSIC constexpr __storage_base(__vector_type_t<_T, _Width> __x)
        : _M_data(reinterpret_cast<_RegisterType>(__x))
    {
    }
    _GLIBCXX_SIMD_INTRINSIC constexpr __storage_base(intrin_type __x)
        : _M_data(reinterpret_cast<_RegisterType>(__x))
    {
    }
};

// __storage{{{1
template <class _T, size_t _Width>
struct __storage<_T, _Width,
               std::void_t<__vector_type_t<_T, _Width>, __intrinsic_type_t<_T, _Width>>>
    : __storage_base<_T, _Width> {
    static_assert(__is_vectorizable_v<_T>);
    static_assert(_Width >= 2);  // 1 doesn't make sense, use _T directly then
    using register_type = __vector_type_t<_T, _Width>;
    using value_type = _T;
    static constexpr size_t width = _Width;

    _GLIBCXX_SIMD_INTRINSIC constexpr __storage() = default;
    template <class _U, class = decltype(__storage_base<_T, _Width>(std::declval<_U>()))>
    _GLIBCXX_SIMD_INTRINSIC constexpr __storage(_U &&__x) : __storage_base<_T, _Width>(std::forward<_U>(__x))
    {
    }
    // I want to use ctor inheritance, but it breaks always_inline. Having a function that
    // does a single movaps is stupid.
    //using __storage_base<_T, _Width>::__storage_base;
    using __storage_base<_T, _Width>::_M_data;

    template <class... As,
              class = enable_if_t<((std::is_same_v<simd_abi::scalar, As> && ...) &&
                                        sizeof...(As) <= _Width)>>
    _GLIBCXX_SIMD_INTRINSIC constexpr operator __simd_tuple<_T, As...>() const
    {
        const auto &dd = _M_data;  // workaround for GCC7 ICE
        return __generate_from_n_evaluations<sizeof...(As), __simd_tuple<_T, As...>>(
            [&](auto __i) { return dd[int(__i)]; });
    }

    _GLIBCXX_SIMD_INTRINSIC constexpr operator const register_type &() const { return _M_data; }
    _GLIBCXX_SIMD_INTRINSIC constexpr operator register_type &() { return _M_data; }

    _GLIBCXX_SIMD_INTRINSIC constexpr _T operator[](size_t __i) const { return _M_data[__i]; }

    _GLIBCXX_SIMD_INTRINSIC void set(size_t __i, _T __x) { _M_data[__i] = __x; }
};

// __to_storage {{{1
template <class _T> class __to_storage
{
    _T _M_data;

public:
    constexpr __to_storage(_T __x) : _M_data(__x) {}

    template <size_t _N> constexpr operator __storage<bool, _N>() const
    {
        static_assert(std::is_integral_v<_T>);
        return static_cast<__bool_storage_member_type_t<_N>>(_M_data);
    }

    template <class _U, size_t _N> constexpr operator __storage<_U, _N>() const
    {
        static_assert(__is_vector_type_v<_T>);
        static_assert(sizeof(__vector_type_t<_U, _N>) == sizeof(_T));
        return {reinterpret_cast<__vector_type_t<_U, _N>>(_M_data)};
    }
};

// __storage_bitcast{{{1
template <class _T, class _U, size_t _M, size_t _N = sizeof(_U) * _M / sizeof(_T)>
_GLIBCXX_SIMD_INTRINSIC constexpr __storage<_T, _N> __storage_bitcast(__storage<_U, _M> __x)
{
    static_assert(sizeof(__vector_type_t<_T, _N>) == sizeof(__vector_type_t<_U, _M>));
    return reinterpret_cast<__vector_type_t<_T, _N>>(__x._M_data);
}

// __make_storage{{{1
template <class _T, class... Args>
_GLIBCXX_SIMD_INTRINSIC constexpr __storage<_T, sizeof...(Args)> __make_storage(Args &&... args)
{
    return {typename __storage<_T, sizeof...(Args)>::register_type{static_cast<_T>(args)...}};
}

// __generate_storage{{{1
template <class _T, size_t _N, class _G>
_GLIBCXX_SIMD_INTRINSIC constexpr __storage<_T, _N> __generate_storage(_G &&gen)
{
    return __generate_builtin<_T, _N>(std::forward<_G>(gen));
}

// __storage ostream operators{{{1
#ifndef NDEBUG
template <class _CharT, class _T, size_t _N>
inline std::basic_ostream<_CharT> &operator<<(std::basic_ostream<_CharT> &s,
                                              const __storage<_T, _N> &v)
{
    s << '[' << v[0];
    for (size_t __i = 1; __i < _N; ++__i) {
        s << ((__i % 4) ? " " : " | ") << v[__i];
    }
    return s << ']';
}
#endif  // NDEBUG

//}}}1
// __fallback_abi_for_long_double {{{
template <class _T, class _A0, class _A1> struct __fallback_abi_for_long_double {
    using type = _A0;
};
template <class _A0, class _A1> struct __fallback_abi_for_long_double<long double, _A0, _A1> {
    using type = _A1;
};
template <class _T, class _A0, class _A1>
using __fallback_abi_for_long_double_t =
    typename __fallback_abi_for_long_double<_T, _A0, _A1>::type;
// }}}

namespace simd_abi
{
// most of simd_abi is defined in simd_detail.h
template <class _T> inline constexpr int max_fixed_size = 32;
// compatible {{{
#if defined __x86_64__
template <class _T>
using compatible = __fallback_abi_for_long_double_t<_T, __sse, scalar>;
#elif defined _GLIBCXX_SIMD_IS_AARCH64
template <class _T>
using compatible = __fallback_abi_for_long_double_t<_T, __neon, scalar>;
#else
template <class> using compatible = scalar;
#endif

// }}}
// native {{{
#if _GLIBCXX_SIMD_HAVE_FULL_AVX512_ABI
template <class _T> using native = __fallback_abi_for_long_double_t<_T, __avx512, scalar>;
#elif _GLIBCXX_SIMD_HAVE_AVX512_ABI
template <class _T>
using native =
    std::conditional_t<(sizeof(_T) >= 4),
                       __fallback_abi_for_long_double_t<_T, __avx512, scalar>, __avx>;
#elif _GLIBCXX_SIMD_HAVE_FULL_AVX_ABI
template <class _T> using native = __fallback_abi_for_long_double_t<_T, __avx, scalar>;
#elif _GLIBCXX_SIMD_HAVE_AVX_ABI
template <class _T>
using native =
    std::conditional_t<std::is_floating_point<_T>::value,
                       __fallback_abi_for_long_double_t<_T, __avx, scalar>, __sse>;
#elif _GLIBCXX_SIMD_HAVE_FULL_SSE_ABI
template <class _T> using native = __fallback_abi_for_long_double_t<_T, __sse, scalar>;
#elif _GLIBCXX_SIMD_HAVE_SSE_ABI
template <class _T>
using native = std::conditional_t<std::is_same<float, _T>::value, __sse, scalar>;
#elif defined _GLIBCXX_SIMD_HAVE_FULL_NEON_ABI
template <class _T> using native = __fallback_abi_for_long_double_t<_T, __neon, scalar>;
#else
template <class> using native = scalar;
#endif

// }}}
// __default_abi {{{
#if defined _GLIBCXX_SIMD_DEFAULT_ABI
template <class _T> using __default_abi = _GLIBCXX_SIMD_DEFAULT_ABI<_T>;
#else
template <class _T> using __default_abi = compatible<_T>;
#endif

// }}}
}  // namespace simd_abi

// traits {{{1
// is_abi_tag {{{2
template <class _T, class = std::void_t<>> struct is_abi_tag : false_type {
};
template <class _T>
struct is_abi_tag<_T, std::void_t<typename _T::is_valid_abi_tag>>
    : public _T::is_valid_abi_tag {
};
template <class _T> inline constexpr bool is_abi_tag_v = is_abi_tag<_T>::value;

// is_simd(_mask) {{{2
template <class _T> struct is_simd : public false_type {};
template <class _T> inline constexpr bool is_simd_v = is_simd<_T>::value;

template <class _T> struct is_simd_mask : public false_type {};
template <class _T> inline constexpr bool is_simd_mask_v = is_simd_mask<_T>::value;

// simd_size {{{2
template <class _T, class _Abi, class = void> struct __simd_size_impl {
};
template <class _T, class _Abi>
struct __simd_size_impl<_T, _Abi,
                        enable_if_t<std::conjunction_v<
                            __is_vectorizable<_T>, std::experimental::is_abi_tag<_Abi>>>>
    : __size_constant<_Abi::template size<_T>> {
};

template <class _T, class _Abi = simd_abi::__default_abi<_T>>
struct simd_size : __simd_size_impl<_T, _Abi> {
};
template <class _T, class _Abi = simd_abi::__default_abi<_T>>
inline constexpr size_t simd_size_v = simd_size<_T, _Abi>::value;

// simd_abi::deduce {{{2
template <class _T, std::size_t _N, class = void> struct __deduce_impl;
namespace simd_abi
{
/**
 * \tparam _T    The requested `value_type` for the elements.
 * \tparam _N    The requested number of elements.
 * \tparam _Abis This parameter is ignored, since this implementation cannot make any use
 *              of it. Either a good native ABI is matched and used as `type` alias, or
 *              the `fixed_size<_N>` ABI is used, which internally is built from the best
 *              matching native ABIs.
 */
template <class _T, std::size_t _N, class...>
struct deduce : std::experimental::__deduce_impl<_T, _N> {};

template <class _T, size_t _N, class... _Abis>
using deduce_t = typename deduce<_T, _N, _Abis...>::type;
}  // namespace simd_abi

// }}}2
// rebind_simd {{{2
template <class _T, class _V> struct rebind_simd;
template <class _T, class _U, class _Abi> struct rebind_simd<_T, simd<_U, _Abi>> {
    using type = simd<_T, simd_abi::deduce_t<_T, simd_size_v<_U, _Abi>, _Abi>>;
};
template <class _T, class _U, class _Abi> struct rebind_simd<_T, simd_mask<_U, _Abi>> {
    using type = simd_mask<_T, simd_abi::deduce_t<_T, simd_size_v<_U, _Abi>, _Abi>>;
};
template <class _T, class _V> using rebind_simd_t = typename rebind_simd<_T, _V>::type;

// resize_simd {{{2
template <int _N, class _V> struct resize_simd;
template <int _N, class _T, class _Abi> struct resize_simd<_N, simd<_T, _Abi>> {
    using type = simd<_T, simd_abi::deduce_t<_T, _N, _Abi>>;
};
template <int _N, class _T, class _Abi> struct resize_simd<_N, simd_mask<_T, _Abi>> {
    using type = simd_mask<_T, simd_abi::deduce_t<_T, _N, _Abi>>;
};
template <int _N, class _V> using resize_simd_t = typename resize_simd<_N, _V>::type;

// }}}2
// memory_alignment {{{2
template <class _T, class _U = typename _T::value_type>
struct memory_alignment
    : public __size_constant<__next_power_of_2(sizeof(_U) * _T::size())> {
};
template <class _T, class _U = typename _T::value_type>
inline constexpr size_t memory_alignment_v = memory_alignment<_T, _U>::value;

// class template simd [simd] {{{1
template <class _T, class _Abi = simd_abi::__default_abi<_T>> class simd;
template <class _T, class _Abi> struct is_simd<simd<_T, _Abi>> : public true_type {};
template <class _T> using native_simd = simd<_T, simd_abi::native<_T>>;
template <class _T, int _N> using fixed_size_simd = simd<_T, simd_abi::fixed_size<_N>>;
template <class _T, size_t _N> using __deduced_simd = simd<_T, simd_abi::deduce_t<_T, _N>>;

// class template simd_mask [simd_mask] {{{1
template <class _T, class _Abi = simd_abi::__default_abi<_T>> class simd_mask;
template <class _T, class _Abi> struct is_simd_mask<simd_mask<_T, _Abi>> : public true_type {};
template <class _T> using native_simd_mask = simd_mask<_T, simd_abi::native<_T>>;
template <class _T, int _N> using fixed_size_simd_mask = simd_mask<_T, simd_abi::fixed_size<_N>>;
template <class _T, size_t _N>
using __deduced_simd_mask = simd_mask<_T, simd_abi::deduce_t<_T, _N>>;

template <class _T, class _Abi> struct __get_impl<std::experimental::simd_mask<_T, _Abi>> {
    using type = typename __simd_traits<_T, _Abi>::__mask_impl_type;
};
template <class _T, class _Abi> struct __get_impl<std::experimental::simd<_T, _Abi>> {
    using type = typename __simd_traits<_T, _Abi>::__simd_impl_type;
};

template <class _T, class _Abi> struct __get_traits<std::experimental::simd_mask<_T, _Abi>> {
    using type = __simd_traits<_T, _Abi>;
};
template <class _T, class _Abi> struct __get_traits<std::experimental::simd<_T, _Abi>> {
    using type = __simd_traits<_T, _Abi>;
};

// casts [simd.casts] {{{1
// static_simd_cast {{{2
template <class _T, class _U, class _A, bool = is_simd_v<_T>, class = void>
struct __static_simd_cast_return_type;

template <class _T, class _A0, class _U, class _A>
struct __static_simd_cast_return_type<simd_mask<_T, _A0>, _U, _A, false, void>
    : __static_simd_cast_return_type<simd<_T, _A0>, _U, _A> {
};

template <class _T, class _U, class _A>
struct __static_simd_cast_return_type<_T, _U, _A, true,
                                    enable_if_t<_T::size() == simd_size_v<_U, _A>>> {
    using type = _T;
};

template <class _T, class _A>
struct __static_simd_cast_return_type<_T, _T, _A, false,
#ifdef _GLIBCXX_SIMD_FIX_P2TS_ISSUE66
                                    enable_if_t<__is_vectorizable_v<_T>>
#else
                                    void
#endif
                                    > {
    using type = simd<_T, _A>;
};

template <class _T, class = void> struct __safe_make_signed {
    using type = _T;
};
template <class _T> struct __safe_make_signed<_T, enable_if_t<std::is_integral_v<_T>>> {
    // the extra make_unsigned_t is because of PR85951
    using type = std::make_signed_t<std::make_unsigned_t<_T>>;
};
template <class _T> using safe_make_signed_t = typename __safe_make_signed<_T>::type;

template <class _T, class _U, class _A>
struct __static_simd_cast_return_type<_T, _U, _A, false,
#ifdef _GLIBCXX_SIMD_FIX_P2TS_ISSUE66
                                    enable_if_t<__is_vectorizable_v<_T>>
#else
                                    void
#endif
                                    > {
    using type =
        std::conditional_t<(std::is_integral_v<_U> && std::is_integral_v<_T> &&
#ifndef _GLIBCXX_SIMD_FIX_P2TS_ISSUE65
                            std::is_signed_v<_U> != std::is_signed_v<_T> &&
#endif
                            std::is_same_v<safe_make_signed_t<_U>, safe_make_signed_t<_T>>),
                           simd<_T, _A>, fixed_size_simd<_T, simd_size_v<_U, _A>>>;
};

// specialized in scalar.h
template <class _To, class, class, class _Native, class _From>
_GLIBCXX_SIMD_INTRINSIC _To __mask_cast_impl(const _Native *, const _From &__x)
{
    static_assert(std::is_same_v<_Native, typename __get_traits_t<_To>::__mask_member_type>);
    if constexpr (std::is_same_v<_Native, bool>) {
        return {std::experimental::__private_init, __x[0]};
    } else if constexpr (std::is_same_v<_From, bool>) {
        _To __r{};
        __r[0] = __x;
        return __r;
    } else {
        return {__private_init,
                __convert_mask<typename __get_traits_t<_To>::__mask_member_type>(__x)};
    }
}
template <class _To, class, class, class _Native, size_t _N>
_GLIBCXX_SIMD_INTRINSIC _To __mask_cast_impl(const _Native *, const std::bitset<_N> &__x)
{
    return {std::experimental::__bitset_init, __x};
}
template <class _To, class, class>
_GLIBCXX_SIMD_INTRINSIC _To __mask_cast_impl(const bool *, bool __x)
{
    return _To(__x);
}
template <class _To, class, class>
_GLIBCXX_SIMD_INTRINSIC _To __mask_cast_impl(const std::bitset<1> *, bool __x)
{
    return _To(__x);
}
template <class _To, class _T, class _Abi, size_t _N, class _From>
_GLIBCXX_SIMD_INTRINSIC _To __mask_cast_impl(const std::bitset<_N> *, const _From &__x)
{
    return {std::experimental::__private_init, __vector_to_bitset(__x)};
}
template <class _To, class, class, size_t _N>
_GLIBCXX_SIMD_INTRINSIC _To __mask_cast_impl(const std::bitset<_N> *, const std::bitset<_N> &__x)
{
    return {std::experimental::__private_init, __x};
}

template <class _T, class _U, class _A,
          class _R = typename __static_simd_cast_return_type<_T, _U, _A>::type>
_GLIBCXX_SIMD_INTRINSIC _R static_simd_cast(const simd<_U, _A> &__x)
{
    if constexpr(std::is_same<_R, simd<_U, _A>>::value) {
        return __x;
    } else {
        __simd_converter<_U, _A, typename _R::value_type, typename _R::abi_type> c;
        return _R(__private_init, c(__data(__x)));
    }
}

template <class _T, class _U, class _A,
          class _R = typename __static_simd_cast_return_type<_T, _U, _A>::type>
_GLIBCXX_SIMD_INTRINSIC typename _R::mask_type static_simd_cast(const simd_mask<_U, _A> &__x)
{
    using _RM = typename _R::mask_type;
    if constexpr(std::is_same<_RM, simd_mask<_U, _A>>::value) {
        return __x;
    } else {
        using __traits = __simd_traits<typename _R::value_type, typename _R::abi_type>;
        const typename __traits::__mask_member_type *tag = nullptr;
        return __mask_cast_impl<_RM, _U, _A>(tag, __data(__x));
    }
}

// simd_cast {{{2
template <class _T, class _U, class _A, class _To = __value_type_or_identity<_T>>
_GLIBCXX_SIMD_INTRINSIC auto simd_cast(const simd<__value_preserving<_U, _To>, _A> &__x)
    ->decltype(static_simd_cast<_T>(__x))
{
    return static_simd_cast<_T>(__x);
}

template <class _T, class _U, class _A, class _To = __value_type_or_identity<_T>>
_GLIBCXX_SIMD_INTRINSIC auto simd_cast(const simd_mask<__value_preserving<_U, _To>, _A> &__x)
    ->decltype(static_simd_cast<_T>(__x))
{
    return static_simd_cast<_T>(__x);
}

namespace __proposed
{
template <class _T, class _U, class _A>
_GLIBCXX_SIMD_INTRINSIC _T resizing_simd_cast(const simd_mask<_U, _A> &__x)
{
    static_assert(is_simd_mask_v<_T>);
    if constexpr (std::is_same_v<_T, simd_mask<_U, _A>>) {
        return __x;
    } else {
        using __traits = __simd_traits<typename _T::simd_type::value_type, typename _T::abi_type>;
        const typename __traits::__mask_member_type *tag = nullptr;
        return __mask_cast_impl<_T, _U, _A>(tag, __data(__x));
    }
}
}  // namespace __proposed

// to_fixed_size {{{2
template <class _T, int _N>
_GLIBCXX_SIMD_INTRINSIC fixed_size_simd<_T, _N> to_fixed_size(const fixed_size_simd<_T, _N> &__x)
{
    return __x;
}

template <class _T, int _N>
_GLIBCXX_SIMD_INTRINSIC fixed_size_simd_mask<_T, _N> to_fixed_size(const fixed_size_simd_mask<_T, _N> &__x)
{
    return __x;
}

template <class _T, class _A> _GLIBCXX_SIMD_INTRINSIC auto to_fixed_size(const simd<_T, _A> &__x)
{
    return simd<_T, simd_abi::fixed_size<simd_size_v<_T, _A>>>(
        [&__x](auto __i) { return __x[__i]; });
}

template <class _T, class _A> _GLIBCXX_SIMD_INTRINSIC auto to_fixed_size(const simd_mask<_T, _A> &__x)
{
    constexpr int _N = simd_mask<_T, _A>::size();
    fixed_size_simd_mask<_T, _N> __r;
    __execute_n_times<_N>([&](auto __i) { __r[__i] = __x[__i]; });
    return __r;
}

// to_native {{{2
template <class _T, int _N>
_GLIBCXX_SIMD_INTRINSIC enable_if_t<(_N == native_simd<_T>::size()), native_simd<_T>>
to_native(const fixed_size_simd<_T, _N> &__x)
{
    alignas(memory_alignment_v<native_simd<_T>>) _T mem[_N];
    __x.copy_to(mem, vector_aligned);
    return {mem, vector_aligned};
}

template <class _T, size_t _N>
_GLIBCXX_SIMD_INTRINSIC enable_if_t<(_N == native_simd_mask<_T>::size()), native_simd_mask<_T>> to_native(
    const fixed_size_simd_mask<_T, _N> &__x)
{
    return native_simd_mask<_T>([&](auto __i) { return __x[__i]; });
}

// to_compatible {{{2
template <class _T, size_t _N>
_GLIBCXX_SIMD_INTRINSIC enable_if_t<(_N == simd<_T>::size()), simd<_T>> to_compatible(
    const simd<_T, simd_abi::fixed_size<_N>> &__x)
{
    alignas(memory_alignment_v<simd<_T>>) _T mem[_N];
    __x.copy_to(mem, vector_aligned);
    return {mem, vector_aligned};
}

template <class _T, size_t _N>
_GLIBCXX_SIMD_INTRINSIC enable_if_t<(_N == simd_mask<_T>::size()), simd_mask<_T>> to_compatible(
    const simd_mask<_T, simd_abi::fixed_size<_N>> &__x)
{
    return simd_mask<_T>([&](auto __i) { return __x[__i]; });
}

// simd_reinterpret_cast {{{2
template <class _To, size_t _N> _GLIBCXX_SIMD_INTRINSIC _To __simd_reinterpret_cast_impl(std::bitset<_N> __x)
{
    return {__bitset_init, __x};
}

template <class _To, class _T, size_t _N>
_GLIBCXX_SIMD_INTRINSIC _To __simd_reinterpret_cast_impl(__storage<_T, _N> __x)
{
    return {__private_init, __x};
}

namespace __proposed
{
template <class _To, class _T, class _A,
          class = enable_if_t<sizeof(_To) == sizeof(simd<_T, _A>) &&
                                   (is_simd_v<_To> || is_simd_mask_v<_To>)>>
_GLIBCXX_SIMD_INTRINSIC _To simd_reinterpret_cast(const simd<_T, _A> &__x)
{
    //return {__private_init, __data(__x)};
    return reinterpret_cast<const _To &>(__x);
}

template <class _To, class _T, class _A,
          class = enable_if_t<(is_simd_v<_To> || is_simd_mask_v<_To>)>>
_GLIBCXX_SIMD_INTRINSIC _To simd_reinterpret_cast(const simd_mask<_T, _A> &__x)
{
    return std::experimental::__simd_reinterpret_cast_impl<_To>(__data(__x));
    //return reinterpret_cast<const _To &>(__x);
}
}  // namespace __proposed

// masked assignment [simd_mask.where] {{{1
#ifdef _GLIBCXX_SIMD_EXPERIMENTAL
template <class _T, class _A> class __masked_simd_impl;
template <class _T, class _A>
__masked_simd_impl<_T, _A> __masked_simd(const typename simd<_T, _A>::mask_type &__k,
                                         simd<_T, _A> &v);
#endif  // _GLIBCXX_SIMD_EXPERIMENTAL

// where_expression {{{1
template <class _M, class _T> class const_where_expression  //{{{2
{
    using _V = _T;
    static_assert(std::is_same_v<_V, std::decay_t<_T>>);
    struct Wrapper {
        using value_type = _V;
    };

protected:
    using value_type =
        typename std::conditional_t<std::is_arithmetic<_V>::value, Wrapper, _V>::value_type;
    _GLIBCXX_SIMD_INTRINSIC friend const _M &__get_mask(const const_where_expression &__x) { return __x.__k; }
    _GLIBCXX_SIMD_INTRINSIC friend const _T &__get_lvalue(const const_where_expression &__x) { return __x._M_value; }
    const _M &__k;
    _T &_M_value;

public:
    const_where_expression(const const_where_expression &) = delete;
    const_where_expression &operator=(const const_where_expression &) = delete;

    _GLIBCXX_SIMD_INTRINSIC const_where_expression(const _M &kk, const _T &dd) : __k(kk), _M_value(const_cast<_T &>(dd)) {}

    _GLIBCXX_SIMD_INTRINSIC _V operator-() const &&
    {
        return {__private_init,
                __get_impl_t<_V>::template masked_unary<std::negate>(
                    __data(__k), __data(_M_value))};
    }

    template <class _U, class _Flags>
    [[nodiscard]] _GLIBCXX_SIMD_INTRINSIC _V
    copy_from(const __loadstore_ptr_type<_U, value_type> *mem, _Flags f) const &&
    {
        return {__private_init, __get_impl_t<_V>::masked_load(
                                          __data(_M_value), __data(__k), mem, f)};
    }

    template <class _U, class _Flags>
    _GLIBCXX_SIMD_INTRINSIC void copy_to(__loadstore_ptr_type<_U, value_type> *mem,
                              _Flags f) const &&
    {
        __get_impl_t<_V>::masked_store(__data(_M_value), mem, f, __data(__k));
    }
};

template <class _T> class const_where_expression<bool, _T>  //{{{2
{
    using _M = bool;
    using _V = _T;
    static_assert(std::is_same_v<_V, std::decay_t<_T>>);
    struct Wrapper {
        using value_type = _V;
    };

protected:
    using value_type =
        typename std::conditional_t<std::is_arithmetic<_V>::value, Wrapper, _V>::value_type;
    _GLIBCXX_SIMD_INTRINSIC friend const _M &__get_mask(const const_where_expression &__x) { return __x.__k; }
    _GLIBCXX_SIMD_INTRINSIC friend const _T &__get_lvalue(const const_where_expression &__x) { return __x._M_value; }
    const bool __k;
    _T &_M_value;

public:
    const_where_expression(const const_where_expression &) = delete;
    const_where_expression &operator=(const const_where_expression &) = delete;

    _GLIBCXX_SIMD_INTRINSIC const_where_expression(const bool kk, const _T &dd) : __k(kk), _M_value(const_cast<_T &>(dd)) {}

    _GLIBCXX_SIMD_INTRINSIC _V operator-() const && { return __k ? -_M_value : _M_value; }

    template <class _U, class _Flags>
    [[nodiscard]] _GLIBCXX_SIMD_INTRINSIC _V
    copy_from(const __loadstore_ptr_type<_U, value_type> *mem, _Flags) const &&
    {
        return __k ? static_cast<_V>(mem[0]) : _M_value;
    }

    template <class _U, class _Flags>
    _GLIBCXX_SIMD_INTRINSIC void copy_to(__loadstore_ptr_type<_U, value_type> *mem,
                              _Flags) const &&
    {
        if (__k) {
            mem[0] = _M_value;
        }
    }
};

// where_expression {{{2
template <class _M, class _T>
class where_expression : public const_where_expression<_M, _T>
{
    static_assert(!std::is_const<_T>::value, "where_expression may only be instantiated with a non-const _T parameter");
    using typename const_where_expression<_M, _T>::value_type;
    using const_where_expression<_M, _T>::__k;
    using const_where_expression<_M, _T>::_M_value;
    static_assert(std::is_same<typename _M::abi_type, typename _T::abi_type>::value, "");
    static_assert(_M::size() == _T::size(), "");

    _GLIBCXX_SIMD_INTRINSIC friend _T &__get_lvalue(where_expression &__x) { return __x._M_value; }
public:
    where_expression(const where_expression &) = delete;
    where_expression &operator=(const where_expression &) = delete;

    _GLIBCXX_SIMD_INTRINSIC where_expression(const _M &kk, _T &dd)
        : const_where_expression<_M, _T>(kk, dd)
    {
    }

    template <class _U> _GLIBCXX_SIMD_INTRINSIC void operator=(_U &&__x) &&
    {
        std::experimental::__get_impl_t<_T>::masked_assign(
            __data(__k), __data(_M_value),
            __to_value_type_or_member_type<_T>(std::forward<_U>(__x)));
    }

#define _GLIBCXX_SIMD_OP_(op_, name_)                                                    \
    template <class _U> _GLIBCXX_SIMD_INTRINSIC void operator op_##=(_U &&__x) &&          \
    {                                                                                    \
        std::experimental::__get_impl_t<_T>::template __masked_cassign<name_>(           \
            __data(__k), __data(_M_value),                                                        \
            __to_value_type_or_member_type<_T>(std::forward<_U>(__x)));                    \
    }                                                                                    \
    _GLIBCXX_SIMD_NOTHING_EXPECTING_SEMICOLON
    _GLIBCXX_SIMD_OP_(+, std::plus);
    _GLIBCXX_SIMD_OP_(-, std::minus);
    _GLIBCXX_SIMD_OP_(*, std::multiplies);
    _GLIBCXX_SIMD_OP_(/, std::divides);
    _GLIBCXX_SIMD_OP_(%, std::modulus);
    _GLIBCXX_SIMD_OP_(&, std::bit_and);
    _GLIBCXX_SIMD_OP_(|, std::bit_or);
    _GLIBCXX_SIMD_OP_(^, std::bit_xor);
    _GLIBCXX_SIMD_OP_(<<, __shift_left);
    _GLIBCXX_SIMD_OP_(>>, __shift_right);
#undef _GLIBCXX_SIMD_OP_

    _GLIBCXX_SIMD_INTRINSIC void operator++() &&
    {
        __data(_M_value) = __get_impl_t<_T>::template masked_unary<__increment>(
            __data(__k), __data(_M_value));
    }
    _GLIBCXX_SIMD_INTRINSIC void operator++(int) &&
    {
        __data(_M_value) = __get_impl_t<_T>::template masked_unary<__increment>(
            __data(__k), __data(_M_value));
    }
    _GLIBCXX_SIMD_INTRINSIC void operator--() &&
    {
        __data(_M_value) = __get_impl_t<_T>::template masked_unary<__decrement>(
            __data(__k), __data(_M_value));
    }
    _GLIBCXX_SIMD_INTRINSIC void operator--(int) &&
    {
        __data(_M_value) = __get_impl_t<_T>::template masked_unary<__decrement>(
            __data(__k), __data(_M_value));
    }

    // intentionally hides const_where_expression::copy_from
    template <class _U, class _Flags>
    _GLIBCXX_SIMD_INTRINSIC void copy_from(const __loadstore_ptr_type<_U, value_type> *mem,
                                _Flags f) &&
    {
        __data(_M_value) =
            __get_impl_t<_T>::masked_load(__data(_M_value), __data(__k), mem, f);
    }

#ifdef _GLIBCXX_SIMD_EXPERIMENTAL
    template <class _F>
    _GLIBCXX_SIMD_INTRINSIC enable_if_t<
        conjunction<std::is_same<decltype(std::declval<_F>()(__masked_simd(
                                     std::declval<const _M &>(), std::declval<_T &>()))),
                                 void>>::value,
        where_expression &&>
    apply(_F &&f) &&
    {
        std::forward<_F>(f)(__masked_simd(__k, _M_value));
        return std::move(*this);
    }

    template <class _F>
    _GLIBCXX_SIMD_INTRINSIC enable_if_t<
        conjunction<std::is_same<decltype(std::declval<_F>()(__masked_simd(
                                     std::declval<const _M &>(), std::declval<_T &>()))),
                                 void>>::value,
        where_expression &&>
    apply_inv(_F &&f) &&
    {
        std::forward<_F>(f)(__masked_simd(!__k, _M_value));
        return std::move(*this);
    }
#endif  // _GLIBCXX_SIMD_EXPERIMENTAL
};

// where_expression<bool> {{{2
template <class _T>
class where_expression<bool, _T> : public const_where_expression<bool, _T>
{
    using _M = bool;
    using typename const_where_expression<_M, _T>::value_type;
    using const_where_expression<_M, _T>::__k;
    using const_where_expression<_M, _T>::_M_value;

public:
    where_expression(const where_expression &) = delete;
    where_expression &operator=(const where_expression &) = delete;

    _GLIBCXX_SIMD_INTRINSIC where_expression(const _M &kk, _T &dd)
        : const_where_expression<_M, _T>(kk, dd)
    {
    }

#define _GLIBCXX_SIMD_OP_(op_)                                                           \
    template <class _U> _GLIBCXX_SIMD_INTRINSIC void operator op_(_U &&__x) &&             \
    {                                                                                    \
        if (__k) {                                                                         \
            _M_value op_ std::forward<_U>(__x);                                                   \
        }                                                                                \
    }                                                                                    \
    _GLIBCXX_SIMD_NOTHING_EXPECTING_SEMICOLON
    _GLIBCXX_SIMD_OP_(=);
    _GLIBCXX_SIMD_OP_(+=);
    _GLIBCXX_SIMD_OP_(-=);
    _GLIBCXX_SIMD_OP_(*=);
    _GLIBCXX_SIMD_OP_(/=);
    _GLIBCXX_SIMD_OP_(%=);
    _GLIBCXX_SIMD_OP_(&=);
    _GLIBCXX_SIMD_OP_(|=);
    _GLIBCXX_SIMD_OP_(^=);
    _GLIBCXX_SIMD_OP_(<<=);
    _GLIBCXX_SIMD_OP_(>>=);
#undef _GLIBCXX_SIMD_OP_
    _GLIBCXX_SIMD_INTRINSIC void operator++()    && { if (__k) { ++_M_value; } }
    _GLIBCXX_SIMD_INTRINSIC void operator++(int) && { if (__k) { ++_M_value; } }
    _GLIBCXX_SIMD_INTRINSIC void operator--()    && { if (__k) { --_M_value; } }
    _GLIBCXX_SIMD_INTRINSIC void operator--(int) && { if (__k) { --_M_value; } }

    // intentionally hides const_where_expression::copy_from
    template <class _U, class _Flags>
    _GLIBCXX_SIMD_INTRINSIC void copy_from(const __loadstore_ptr_type<_U, value_type> *mem,
                                _Flags) &&
    {
        if (__k) {
            _M_value = mem[0];
        }
    }
};

// where_expression<_M, tuple<...>> {{{2
#ifdef _GLIBCXX_SIMD_EXPERIMENTAL
template <class _M, class... _Ts> class where_expression<_M, std::tuple<_Ts &...>>
{
    const _M &__k;
    std::tuple<_Ts &...> _M_value;

public:
    where_expression(const where_expression &) = delete;
    where_expression &operator=(const where_expression &) = delete;

    _GLIBCXX_SIMD_INTRINSIC where_expression(const _M &kk, std::tuple<_Ts &...> &&dd) : __k(kk), _M_value(dd) {}

private:
    template <class _F, std::size_t... Is>
    _GLIBCXX_SIMD_INTRINSIC void apply_helper(_F &&f, const _M &simd_mask, std::index_sequence<Is...>)
    {
        return std::forward<_F>(f)(__masked_simd(simd_mask, std::get<Is>(_M_value))...);
    }

public:
    template <class _F>
    _GLIBCXX_SIMD_INTRINSIC enable_if_t<
        conjunction<
            std::is_same<decltype(std::declval<_F>()(__masked_simd(
                             std::declval<const _M &>(), std::declval<_Ts &>())...)),
                         void>>::value,
        where_expression &&>
    apply(_F &&f) &&
    {
        apply_helper(std::forward<_F>(f), __k, std::make_index_sequence<sizeof...(_Ts)>());
        return std::move(*this);
    }

    template <class _F>
    _GLIBCXX_SIMD_INTRINSIC enable_if_t<
        conjunction<
            std::is_same<decltype(std::declval<_F>()(__masked_simd(
                             std::declval<const _M &>(), std::declval<_Ts &>())...)),
                         void>>::value,
        where_expression &&>
    apply_inv(_F &&f) &&
    {
        apply_helper(std::forward<_F>(f), !__k, std::make_index_sequence<sizeof...(_Ts)>());
        return std::move(*this);
    }
};

template <class _T, class _A, class... Vs>
_GLIBCXX_SIMD_INTRINSIC where_expression<simd_mask<_T, _A>, std::tuple<simd<_T, _A> &, Vs &...>> where(
    const typename simd<_T, _A>::mask_type &__k, simd<_T, _A> &v0, Vs &... vs)
{
    return {__k, std::tie(v0, vs...)};
}
#endif  // _GLIBCXX_SIMD_EXPERIMENTAL

// where {{{1
template <class _T, class _A>
_GLIBCXX_SIMD_INTRINSIC where_expression<simd_mask<_T, _A>, simd<_T, _A>> where(
    const typename simd<_T, _A>::mask_type &__k, simd<_T, _A> &__value)
{
    return {__k, __value};
}
template <class _T, class _A>
_GLIBCXX_SIMD_INTRINSIC const_where_expression<simd_mask<_T, _A>, simd<_T, _A>> where(
    const typename simd<_T, _A>::mask_type &__k, const simd<_T, _A> &__value)
{
    return {__k, __value};
}
template <class _T, class _A>
_GLIBCXX_SIMD_INTRINSIC where_expression<simd_mask<_T, _A>, simd_mask<_T, _A>> where(
    const std::remove_const_t<simd_mask<_T, _A>> &__k, simd_mask<_T, _A> &__value)
{
    return {__k, __value};
}
template <class _T, class _A>
_GLIBCXX_SIMD_INTRINSIC const_where_expression<simd_mask<_T, _A>, simd_mask<_T, _A>> where(
    const std::remove_const_t<simd_mask<_T, _A>> &__k, const simd_mask<_T, _A> &__value)
{
    return {__k, __value};
}
template <class _T>
_GLIBCXX_SIMD_INTRINSIC where_expression<bool, _T> where(__exact_bool __k, _T &__value)
{
    return {__k, __value};
}
template <class _T>
_GLIBCXX_SIMD_INTRINSIC const_where_expression<bool, _T> where(__exact_bool __k, const _T &__value)
{
    return {__k, __value};
}
template <class _T, class _A> void where(bool __k, simd<_T, _A> &__value) = delete;
template <class _T, class _A> void where(bool __k, const simd<_T, _A> &__value) = delete;

// proposed mask iterations {{{1
namespace __proposed
{
template <size_t _N> class where_range
{
    const std::bitset<_N> __bits;

public:
    where_range(std::bitset<_N> __b) : __bits(__b) {}

    class iterator
    {
        size_t __mask;
        size_t __bit;

        _GLIBCXX_SIMD_INTRINSIC void next_bit() { __bit = __builtin_ctzl(__mask); }
        _GLIBCXX_SIMD_INTRINSIC void reset_lsb()
        {
            // 01100100 - 1 = 01100011
            __mask &= (__mask - 1);
            // __asm__("btr %1,%0" : "+r"(__mask) : "r"(__bit));
        }

    public:
        iterator(decltype(__mask) m) : __mask(m) { next_bit(); }
        iterator(const iterator &) = default;
        iterator(iterator &&) = default;

        _GLIBCXX_SIMD_ALWAYS_INLINE size_t operator->() const { return __bit; }
        _GLIBCXX_SIMD_ALWAYS_INLINE size_t operator*() const { return __bit; }

        _GLIBCXX_SIMD_ALWAYS_INLINE iterator &operator++()
        {
            reset_lsb();
            next_bit();
            return *this;
        }
        _GLIBCXX_SIMD_ALWAYS_INLINE iterator operator++(int)
        {
            iterator __tmp = *this;
            reset_lsb();
            next_bit();
            return __tmp;
        }

        _GLIBCXX_SIMD_ALWAYS_INLINE bool operator==(const iterator &rhs) const
        {
            return __mask == rhs.__mask;
        }
        _GLIBCXX_SIMD_ALWAYS_INLINE bool operator!=(const iterator &rhs) const
        {
            return __mask != rhs.__mask;
        }
    };

    iterator begin() const { return __bits.to_ullong(); }
    iterator end() const { return 0; }
};

template <class _T, class _A>
where_range<simd_size_v<_T, _A>> where(const simd_mask<_T, _A> &__k)
{
    return __k.__to_bitset();
}

}  // namespace __proposed

// }}}1
// reductions [simd.reductions] {{{1
template <class _T, class _Abi, class _BinaryOperation = std::plus<>>
_GLIBCXX_SIMD_INTRINSIC _T reduce(const simd<_T, _Abi>& v,
                                  _BinaryOperation __binary_op = _BinaryOperation())
{
    using _V = simd<_T, _Abi>;
    return __get_impl_t<_V>::reduce(v, __binary_op);
}

template <class _M, class _V, class _BinaryOperation = std::plus<>>
_GLIBCXX_SIMD_INTRINSIC typename _V::value_type reduce(
    const const_where_expression<_M, _V>& __x, typename _V::value_type __identity_element,
    _BinaryOperation __binary_op)
{
    _V __tmp = __identity_element;
    __get_impl_t<_V>::masked_assign(__data(__get_mask(__x)), __data(__tmp),
                                    __data(__get_lvalue(__x)));
    return reduce(__tmp, __binary_op);
}

template <class _M, class _V>
_GLIBCXX_SIMD_INTRINSIC typename _V::value_type reduce(
    const const_where_expression<_M, _V>& __x, std::plus<> __binary_op = {})
{
    return reduce(__x, 0, __binary_op);
}

template <class _M, class _V>
_GLIBCXX_SIMD_INTRINSIC typename _V::value_type reduce(
    const const_where_expression<_M, _V>& __x, std::multiplies<> __binary_op)
{
    return reduce(__x, 1, __binary_op);
}

template <class _M, class _V>
_GLIBCXX_SIMD_INTRINSIC typename _V::value_type reduce(
    const const_where_expression<_M, _V>& __x, std::bit_and<> __binary_op)
{
    return reduce(__x, ~typename _V::value_type(), __binary_op);
}

template <class _M, class _V>
_GLIBCXX_SIMD_INTRINSIC typename _V::value_type reduce(
    const const_where_expression<_M, _V>& __x, std::bit_or<> __binary_op)
{
    return reduce(__x, 0, __binary_op);
}

template <class _M, class _V>
_GLIBCXX_SIMD_INTRINSIC typename _V::value_type reduce(
    const const_where_expression<_M, _V>& __x, std::bit_xor<> __binary_op)
{
    return reduce(__x, 0, __binary_op);
}

// }}}1
// algorithms [simd.alg] {{{
template <class _T, class _A>
_GLIBCXX_SIMD_INTRINSIC simd<_T, _A> min(const simd<_T, _A> &a, const simd<_T, _A> &b)
{
    return {__private_init,
            _A::__simd_impl_type::min(__data(a), __data(b))};
}
template <class _T, class _A>
_GLIBCXX_SIMD_INTRINSIC simd<_T, _A> max(const simd<_T, _A> &a, const simd<_T, _A> &b)
{
    return {__private_init,
            _A::__simd_impl_type::max(__data(a), __data(b))};
}
template <class _T, class _A>
_GLIBCXX_SIMD_INTRINSIC std::pair<simd<_T, _A>, simd<_T, _A>> minmax(const simd<_T, _A> &a,
                                                            const simd<_T, _A> &b)
{
    const auto pair_of_members =
        _A::__simd_impl_type::minmax(__data(a), __data(b));
    return {simd<_T, _A>(__private_init, pair_of_members.first),
            simd<_T, _A>(__private_init, pair_of_members.second)};
}
template <class _T, class _A>
_GLIBCXX_SIMD_INTRINSIC simd<_T, _A> clamp(const simd<_T, _A> &v, const simd<_T, _A> &lo,
                                 const simd<_T, _A> &hi)
{
    using _Impl = typename _A::__simd_impl_type;
    return {__private_init,
            _Impl::min(__data(hi), _Impl::max(__data(lo), __data(v)))};
}

// }}}

namespace __proposed
{
// shuffle {{{1
template <int _Stride, int _Offset = 0> struct strided {
    static constexpr int stride = _Stride;
    static constexpr int offset = _Offset;
    template <class _T, class _A>
    using shuffle_return_type = simd<
        _T, simd_abi::deduce_t<_T, (simd_size_v<_T, _A> - _Offset + _Stride - 1) / _Stride, _A>>;
    // alternative, always use fixed_size:
    // fixed_size_simd<_T, (simd_size_v<_T, _A> - _Offset + _Stride - 1) / _Stride>;
    template <class _T> static constexpr auto src_index(_T dst_index)
    {
        return _Offset + dst_index * _Stride;
    }
};

// SFINAE for the return type ensures _P is a type that provides the alias template member
// shuffle_return_type and the static member function src_index
template <class _P, class _T, class _A,
          class _R = typename _P::template shuffle_return_type<_T, _A>,
          class = decltype(_P::src_index(std::experimental::__size_constant<0>()))>
_GLIBCXX_SIMD_INTRINSIC _R shuffle(const simd<_T, _A> &__x)
{
    return _R([&__x](auto __i) { return __x[_P::src_index(__i)]; });
}

// }}}1
}  // namespace __proposed

template <size_t... _Sizes, class _T, class _A,
          class = enable_if_t<((_Sizes + ...) == simd<_T, _A>::size())>>
inline std::tuple<simd<_T, simd_abi::deduce_t<_T, _Sizes>>...> split(const simd<_T, _A> &);

// __extract_part {{{
template <size_t _Index, size_t _Total, class _T, size_t _N>
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST
    __vector_type_t<_T, std::max(16 / sizeof(_T), _N / _Total)>
        __extract_part(__storage<_T, _N>);
template <int Index, int Parts, class _T, class _A0, class... As>
auto __extract_part(const __simd_tuple<_T, _A0, As...> &__x);

// }}}
// __size_list {{{
template <size_t V0, size_t... Values> struct __size_list {
    static constexpr size_t size = sizeof...(Values) + 1;

    template <size_t _I> static constexpr size_t at(__size_constant<_I> = {})
    {
        if constexpr (_I == 0) {
            return V0;
        } else {
            return __size_list<Values...>::template at<_I - 1>();
        }
    }

    template <size_t _I> static constexpr auto before(__size_constant<_I> = {})
    {
        if constexpr (_I == 0) {
            return __size_constant<0>();
        } else {
            return __size_constant<V0 + __size_list<Values...>::template before<_I - 1>()>();
        }
    }

    template <size_t _N> static constexpr auto pop_front(__size_constant<_N> = {})
    {
        if constexpr (_N == 0) {
            return __size_list();
        } else {
            return __size_list<Values...>::template pop_front<_N-1>();
        }
    }
};
// }}}
// __extract_center {{{
template <class _T, size_t _N>
_GLIBCXX_SIMD_INTRINSIC __storage<_T, _N / 2> __extract_center(__storage<_T, _N> __x)
{
    if constexpr (__have_avx512f && sizeof(__x) == 64) {
        const auto __intrin = __to_intrin(__x);
        if constexpr (std::is_integral_v<_T>) {
            return __vector_bitcast<_T>(_mm512_castsi512_si256(_mm512_shuffle_i32x4(
                __intrin, __intrin, 1 + 2 * 0x4 + 2 * 0x10 + 3 * 0x40)));
        } else if constexpr (sizeof(_T) == 4) {
            return __vector_bitcast<_T>(_mm512_castps512_ps256(_mm512_shuffle_f32x4(
                __intrin, __intrin, 1 + 2 * 0x4 + 2 * 0x10 + 3 * 0x40)));
        } else if constexpr (sizeof(_T) == 8) {
            return __vector_bitcast<_T>(_mm512_castpd512_pd256(_mm512_shuffle_f64x2(
                __intrin, __intrin, 1 + 2 * 0x4 + 2 * 0x10 + 3 * 0x40)));
        } else {
            __assert_unreachable<_T>();
        }
    } else {
        __assert_unreachable<_T>();
    }
}
template <class _T, class _A>
inline __storage<_T, simd_size_v<_T, _A>> __extract_center(
    const __simd_tuple<_T, _A, _A>& __x)
{
    return __concat(__extract<1, 2>(__x.first._M_data), __extract<0, 2>(__x.second.first._M_data));
}
template <class _T, class _A>
inline __storage<_T, simd_size_v<_T, _A> / 2> __extract_center(
    const __simd_tuple<_T, _A>& __x)
{
    return __extract_center(__x.first);
}

// }}}
// __split_wrapper {{{
template <size_t... _Sizes, class _T, class... As>
auto __split_wrapper(__size_list<_Sizes...>, const __simd_tuple<_T, As...> &__x)
{
    return std::experimental::split<_Sizes...>(
        fixed_size_simd<_T, __simd_tuple<_T, As...>::size()>(__private_init, __x));
}

// }}}

// split<simd>(simd) {{{
template <class _V, class _A,
          size_t Parts = simd_size_v<typename _V::value_type, _A> / _V::size()>
inline enable_if_t<(is_simd<_V>::value &&
                         simd_size_v<typename _V::value_type, _A> == Parts * _V::size()),
                        std::array<_V, Parts>>
split(const simd<typename _V::value_type, _A> &__x)
{
    using _T = typename _V::value_type;
    if constexpr (Parts == 1) {
        return {simd_cast<_V>(__x)};
    } else if constexpr (__is_fixed_size_abi_v<_A> &&
                         (std::is_same_v<typename _V::abi_type, simd_abi::scalar> ||
                          (__is_fixed_size_abi_v<typename _V::abi_type> &&
                           sizeof(_V) == sizeof(_T) * _V::size()  // _V doesn't have padding
                           ))) {
        // fixed_size -> fixed_size (w/o padding) or scalar
#ifdef _GLIBCXX_SIMD_USE_ALIASING_LOADS
        const __may_alias<_T> *const element_ptr =
            reinterpret_cast<const __may_alias<_T> *>(&__data(__x));
        return __generate_from_n_evaluations<Parts, std::array<_V, Parts>>(
            [&](auto __i) { return _V(element_ptr + __i * _V::size(), vector_aligned); });
#else
        const auto &xx = __data(__x);
        return __generate_from_n_evaluations<Parts, std::array<_V, Parts>>(
            [&](auto __i) {
                constexpr size_t offset = decltype(__i)::value * _V::size();
                __unused(offset);  // not really
                return _V([&](auto j) {
                    constexpr __size_constant<j + offset> __k;
                    return xx[__k];
                });
            });
#endif
    } else if constexpr (std::is_same_v<typename _V::abi_type, simd_abi::scalar>) {
        // normally memcpy should work here as well
        return __generate_from_n_evaluations<Parts, std::array<_V, Parts>>(
            [&](auto __i) { return __x[__i]; });
    } else {
        return __generate_from_n_evaluations<Parts, std::array<_V, Parts>>([&](auto __i) {
            if constexpr (__is_fixed_size_abi_v<typename _V::abi_type>) {
                return _V([&](auto __j) { return __x[__i * _V::size() + __j]; });
            } else {
                return _V(__private_init,
                         __extract_part<__i, Parts>(__data(__x)));
            }
        });
    }
}

// }}}
// split<simd_mask>(simd_mask) {{{
template <class _V, class _A,
          size_t Parts = simd_size_v<typename _V::simd_type::value_type, _A> / _V::size()>
enable_if_t<(is_simd_mask_v<_V> &&
                  simd_size_v<typename _V::simd_type::value_type, _A> == Parts * _V::size()),
                 std::array<_V, Parts>>
split(const simd_mask<typename _V::simd_type::value_type, _A> &__x)
{
    if constexpr (std::is_same_v<_A, typename _V::abi_type>) {
        return {__x};
    } else if constexpr (Parts == 1) {
        return {static_simd_cast<_V>(__x)};
    } else if constexpr (Parts == 2) {
        return {_V(__private_init, [&](size_t __i) { return __x[__i]; }),
                _V(__private_init, [&](size_t __i) { return __x[__i + _V::size()]; })};
    } else if constexpr (Parts == 3) {
        return {_V(__private_init, [&](size_t __i) { return __x[__i]; }),
                _V(__private_init, [&](size_t __i) { return __x[__i + _V::size()]; }),
                _V(__private_init, [&](size_t __i) { return __x[__i + 2 * _V::size()]; })};
    } else if constexpr (Parts == 4) {
        return {_V(__private_init, [&](size_t __i) { return __x[__i]; }),
                _V(__private_init, [&](size_t __i) { return __x[__i + _V::size()]; }),
                _V(__private_init, [&](size_t __i) { return __x[__i + 2 * _V::size()]; }),
                _V(__private_init, [&](size_t __i) { return __x[__i + 3 * _V::size()]; })};
    } else {
        __assert_unreachable<_V>();
    }
}

// }}}
// split<_Sizes...>(simd) {{{
template <size_t... _Sizes, class _T, class _A,
          class = enable_if_t<((_Sizes + ...) == simd<_T, _A>::size())>>
_GLIBCXX_SIMD_ALWAYS_INLINE std::tuple<simd<_T, simd_abi::deduce_t<_T, _Sizes>>...> split(
    const simd<_T, _A> &__x)
{
    using _SL = __size_list<_Sizes...>;
    using _Tuple = std::tuple<__deduced_simd<_T, _Sizes>...>;
    constexpr size_t _N = simd_size_v<_T, _A>;
    constexpr size_t N0 = _SL::template at<0>();
    using _V = __deduced_simd<_T, N0>;

    if constexpr (_N == N0) {
        static_assert(sizeof...(_Sizes) == 1);
        return {simd_cast<_V>(__x)};
    } else if constexpr (__is_fixed_size_abi_v<_A> &&
                         __fixed_size_storage<_T, _N>::__first_size_v == N0) {
        // if the first part of the __simd_tuple input matches the first output vector
        // in the std::tuple, extract it and recurse
        static_assert(!__is_fixed_size_abi_v<typename _V::abi_type>,
                      "How can <_T, _N> be a single __simd_tuple entry but a fixed_size_simd "
                      "when deduced?");
        const __fixed_size_storage<_T, _N> &xx = __data(__x);
        return std::tuple_cat(
            std::make_tuple(_V(__private_init, xx.first)),
            __split_wrapper(_SL::template pop_front<1>(), xx.second));
    } else if constexpr ((!std::is_same_v<simd_abi::scalar,
                                          simd_abi::deduce_t<_T, _Sizes>> &&
                          ...) &&
                         (!__is_fixed_size_abi_v<simd_abi::deduce_t<_T, _Sizes>> &&
                          ...)) {
        if constexpr (((_Sizes * 2 == _N)&&...)) {
            return {{__private_init, __extract_part<0, 2>(__data(__x))},
                    {__private_init, __extract_part<1, 2>(__data(__x))}};
        } else if constexpr (std::is_same_v<__size_list<_Sizes...>,
                                            __size_list<_N / 3, _N / 3, _N / 3>>) {
            return {{__private_init, __extract_part<0, 3>(__data(__x))},
                    {__private_init, __extract_part<1, 3>(__data(__x))},
                    {__private_init, __extract_part<2, 3>(__data(__x))}};
        } else if constexpr (std::is_same_v<__size_list<_Sizes...>,
                                            __size_list<2 * _N / 3, _N / 3>>) {
            return {{__private_init,
                     __concat(__extract_part<0, 3>(__data(__x)),
                                    __extract_part<1, 3>(__data(__x)))},
                    {__private_init, __extract_part<2, 3>(__data(__x))}};
        } else if constexpr (std::is_same_v<__size_list<_Sizes...>,
                                            __size_list<_N / 3, 2 * _N / 3>>) {
            return {{__private_init, __extract_part<0, 3>(__data(__x))},
                    {__private_init,
                     __concat(__extract_part<1, 3>(__data(__x)),
                                    __extract_part<2, 3>(__data(__x)))}};
        } else if constexpr (std::is_same_v<__size_list<_Sizes...>,
                                            __size_list<_N / 2, _N / 4, _N / 4>>) {
            return {{__private_init, __extract_part<0, 2>(__data(__x))},
                    {__private_init, __extract_part<2, 4>(__data(__x))},
                    {__private_init, __extract_part<3, 4>(__data(__x))}};
        } else if constexpr (std::is_same_v<__size_list<_Sizes...>,
                                            __size_list<_N / 4, _N / 4, _N / 2>>) {
            return {{__private_init, __extract_part<0, 4>(__data(__x))},
                    {__private_init, __extract_part<1, 4>(__data(__x))},
                    {__private_init, __extract_part<1, 2>(__data(__x))}};
        } else if constexpr (std::is_same_v<__size_list<_Sizes...>,
                                            __size_list<_N / 4, _N / 2, _N / 4>>) {
            return {
                {__private_init, __extract_part<0, 4>(__data(__x))},
                {__private_init, __extract_center(__data(__x))},
                {__private_init, __extract_part<3, 4>(__data(__x))}};
        } else if constexpr (((_Sizes * 4 == _N) && ...)) {
            return {{__private_init, __extract_part<0, 4>(__data(__x))},
                    {__private_init, __extract_part<1, 4>(__data(__x))},
                    {__private_init, __extract_part<2, 4>(__data(__x))},
                    {__private_init, __extract_part<3, 4>(__data(__x))}};
        //} else if constexpr (__is_fixed_size_abi_v<_A>) {
        } else {
            __assert_unreachable<_T>();
        }
    } else {
#ifdef _GLIBCXX_SIMD_USE_ALIASING_LOADS
        const __may_alias<_T> *const element_ptr =
            reinterpret_cast<const __may_alias<_T> *>(&__x);
        return __generate_from_n_evaluations<sizeof...(_Sizes), _Tuple>([&](auto __i) {
            using _Vi = __deduced_simd<_T, _SL::at(__i)>;
            constexpr size_t offset = _SL::before(__i);
            constexpr size_t base_align = alignof(simd<_T, _A>);
            constexpr size_t a = base_align - ((offset * sizeof(_T)) % base_align);
            constexpr size_t b = ((a - 1) & a) ^ a;
            constexpr size_t alignment = b == 0 ? a : b;
            return _Vi(element_ptr + offset, overaligned<alignment>);
        });
#else
        return __generate_from_n_evaluations<sizeof...(_Sizes), _Tuple>([&](auto __i) {
            using _Vi = __deduced_simd<_T, _SL::at(__i)>;
            const auto &xx = __data(__x);
            using _Offset = decltype(_SL::before(__i));
            return _Vi([&](auto j) {
                constexpr __size_constant<_Offset::value + j> __k;
                return xx[__k];
            });
        });
#endif
    }
}

// }}}

// __subscript_in_pack {{{
template <size_t _I, class _T, class _A, class... As>
_GLIBCXX_SIMD_INTRINSIC constexpr _T __subscript_in_pack(const simd<_T, _A> &__x, const simd<_T, As> &... __xs)
{
    if constexpr (_I < simd_size_v<_T, _A>) {
        return __x[_I];
    } else {
        return __subscript_in_pack<_I - simd_size_v<_T, _A>>(__xs...);
    }
}
// }}}

// concat(simd...) {{{
template <class _T, class... As>
simd<_T, simd_abi::deduce_t<_T, (simd_size_v<_T, As> + ...)>> concat(
    const simd<_T, As> &... __xs)
{
    return simd<_T, simd_abi::deduce_t<_T, (simd_size_v<_T, As> + ...)>>(
        [&](auto __i) { return __subscript_in_pack<__i>(__xs...); });
}

// }}}

// __smart_reference {{{
template <class _U, class _Accessor = _U, class _ValueType = typename _U::value_type>
class __smart_reference
{
    friend _Accessor;
    int index;
    _U &obj;

    _GLIBCXX_SIMD_INTRINSIC constexpr _ValueType __read() const noexcept
    {
        if constexpr (std::is_arithmetic_v<_U>) {
            _GLIBCXX_SIMD_ASSERT(index == 0);
            return obj;
        } else {
            return obj[index];
        }
    }

    template <class _T> _GLIBCXX_SIMD_INTRINSIC constexpr void __write(_T &&__x) const
    {
        _Accessor::set(obj, index, std::forward<_T>(__x));
    }

public:
    _GLIBCXX_SIMD_INTRINSIC __smart_reference(_U &o, int __i) noexcept : index(__i), obj(o) {}

    using value_type = _ValueType;

    _GLIBCXX_SIMD_INTRINSIC __smart_reference(const __smart_reference &) = delete;

    _GLIBCXX_SIMD_INTRINSIC constexpr operator value_type() const noexcept { return __read(); }

    template <class _T,
              class = __value_preserving_or_int<std::decay_t<_T>, value_type>>
    _GLIBCXX_SIMD_INTRINSIC constexpr __smart_reference operator=(_T &&__x) &&
    {
        __write(std::forward<_T>(__x));
        return {obj, index};
    }

// TODO: improve with operator.()

#define _GLIBCXX_SIMD_OP_(op_)                                                           \
    template <class _T,                                                                  \
              class _TT = decltype(std::declval<value_type>() op_ std::declval<_T>()),   \
              class     = __value_preserving_or_int<std::decay_t<_T>, _TT>,              \
              class     = __value_preserving_or_int<_TT, value_type>>                    \
    _GLIBCXX_SIMD_INTRINSIC __smart_reference operator op_##=(_T&& __x)&&                \
    {                                                                                    \
        const value_type& lhs = __read();                                                \
        __write(lhs op_ __x);                                                            \
        return {obj, index};                                                             \
    }
    _GLIBCXX_SIMD_ALL_ARITHMETICS(_GLIBCXX_SIMD_OP_);
    _GLIBCXX_SIMD_ALL_SHIFTS(_GLIBCXX_SIMD_OP_);
    _GLIBCXX_SIMD_ALL_BINARY(_GLIBCXX_SIMD_OP_);
#undef _GLIBCXX_SIMD_OP_

    template <class _T = void,
              class = decltype(
                  ++std::declval<std::conditional_t<true, value_type, _T> &>())>
    _GLIBCXX_SIMD_INTRINSIC __smart_reference operator++() &&
    {
        value_type __x = __read();
        __write(++__x);
        return {obj, index};
    }

    template <class _T = void,
              class = decltype(
                  std::declval<std::conditional_t<true, value_type, _T> &>()++)>
    _GLIBCXX_SIMD_INTRINSIC value_type operator++(int) &&
    {
        const value_type __r = __read();
        value_type __x = __r;
        __write(++__x);
        return __r;
    }

    template <class _T = void,
              class = decltype(
                  --std::declval<std::conditional_t<true, value_type, _T> &>())>
    _GLIBCXX_SIMD_INTRINSIC __smart_reference operator--() &&
    {
        value_type __x = __read();
        __write(--__x);
        return {obj, index};
    }

    template <class _T = void,
              class = decltype(
                  std::declval<std::conditional_t<true, value_type, _T> &>()--)>
    _GLIBCXX_SIMD_INTRINSIC value_type operator--(int) &&
    {
        const value_type __r = __read();
        value_type __x = __r;
        __write(--__x);
        return __r;
    }

    _GLIBCXX_SIMD_INTRINSIC friend void swap(__smart_reference &&a, __smart_reference &&b) noexcept(
        conjunction<std::is_nothrow_constructible<value_type, __smart_reference &&>,
            std::is_nothrow_assignable<__smart_reference &&, value_type &&>>::value)
    {
        value_type __tmp = static_cast<__smart_reference &&>(a);
        static_cast<__smart_reference &&>(a) = static_cast<value_type>(b);
        static_cast<__smart_reference &&>(b) = std::move(__tmp);
    }

    _GLIBCXX_SIMD_INTRINSIC friend void swap(value_type &a, __smart_reference &&b) noexcept(
        conjunction<std::is_nothrow_constructible<value_type, value_type &&>,
            std::is_nothrow_assignable<value_type &, value_type &&>,
            std::is_nothrow_assignable<__smart_reference &&, value_type &&>>::value)
    {
        value_type __tmp(std::move(a));
        a = static_cast<value_type>(b);
        static_cast<__smart_reference &&>(b) = std::move(__tmp);
    }

    _GLIBCXX_SIMD_INTRINSIC friend void swap(__smart_reference &&a, value_type &b) noexcept(
        conjunction<std::is_nothrow_constructible<value_type, __smart_reference &&>,
            std::is_nothrow_assignable<value_type &, value_type &&>,
            std::is_nothrow_assignable<__smart_reference &&, value_type &&>>::value)
    {
        value_type __tmp(a);
        static_cast<__smart_reference &&>(a) = std::move(b);
        b = std::move(__tmp);
    }
};

// }}}
// abi impl fwd decls {{{
struct __neon_simd_impl;
struct __neon_mask_impl;
struct __sse_mask_impl;
struct __sse_simd_impl;
struct __avx_mask_impl;
struct __avx_simd_impl;
struct __avx512_mask_impl;
struct __avx512_simd_impl;
struct __scalar_simd_impl;
struct __scalar_mask_impl;
template <int _N> struct __fixed_size_simd_impl;
template <int _N> struct __fixed_size_mask_impl;
template <int _N, class _Abi> struct __combine_simd_impl;
template <int _N, class _Abi> struct __combine_mask_impl;

// }}}
// __gnu_traits {{{1
template <class _T, class _MT, class _Abi, size_t _N> struct __gnu_traits {
    using is_valid = true_type;
    using __simd_impl_type = typename _Abi::__simd_impl_type;
    using __mask_impl_type = typename _Abi::__mask_impl_type;

    // simd and simd_mask member types {{{2
    using __simd_member_type = __storage<_T, _N>;
    using __mask_member_type = __storage<_MT, _N>;
    static constexpr size_t __simd_member_alignment = alignof(__simd_member_type);
    static constexpr size_t __mask_member_alignment = alignof(__mask_member_type);

    // __simd_base / base class for simd, providing extra conversions {{{2
    struct simd_base2 {
        explicit operator __intrinsic_type_t<_T, _N>() const
        {
            return static_cast<const simd<_T, _Abi> *>(this)->_M_data.v();
        }
        explicit operator __vector_type_t<_T, _N>() const
        {
            return static_cast<const simd<_T, _Abi> *>(this)->_M_data.builtin();
        }
    };
    struct simd_base1 {
        explicit operator __intrinsic_type_t<_T, _N>() const
        {
            return __data(*static_cast<const simd<_T, _Abi> *>(this));
        }
    };
    using __simd_base = std::conditional_t<
        std::is_same<__intrinsic_type_t<_T, _N>, __vector_type_t<_T, _N>>::value,
        simd_base1, simd_base2>;

    // __mask_base {{{2
    struct mask_base2 {
        explicit operator __intrinsic_type_t<_T, _N>() const
        {
            return static_cast<const simd_mask<_T, _Abi> *>(this)->_M_data.intrin();
        }
        explicit operator __vector_type_t<_T, _N>() const
        {
            return static_cast<const simd_mask<_T, _Abi> *>(this)->_M_data._M_data;
        }
    };
    struct mask_base1 {
        explicit operator __intrinsic_type_t<_T, _N>() const
        {
            return __data(*static_cast<const simd_mask<_T, _Abi> *>(this));
        }
    };
    using __mask_base = std::conditional_t<
        std::is_same<__intrinsic_type_t<_T, _N>, __vector_type_t<_T, _N>>::value,
        mask_base1, mask_base2>;

    // __mask_cast_type {{{2
    // parameter type of one explicit simd_mask constructor
    class __mask_cast_type
    {
        using _U = __intrinsic_type_t<_T, _N>;
        _U _M_data;

    public:
        __mask_cast_type(_U __x) : _M_data(__x) {}
        operator __mask_member_type() const { return _M_data; }
    };

    // __simd_cast_type {{{2
    // parameter type of one explicit simd constructor
    class simd_cast_type1
    {
        using _A = __intrinsic_type_t<_T, _N>;
        __simd_member_type _M_data;

    public:
        simd_cast_type1(_A __a) : _M_data(__vector_bitcast<_T>(__a)) {}
        operator __simd_member_type() const { return _M_data; }
    };

    class simd_cast_type2
    {
        using _A = __intrinsic_type_t<_T, _N>;
        using _B = __vector_type_t<_T, _N>;
        __simd_member_type _M_data;

    public:
        simd_cast_type2(_A __a) : _M_data(__vector_bitcast<_T>(__a)) {}
        simd_cast_type2(_B __b) : _M_data(__b) {}
        operator __simd_member_type() const { return _M_data; }
    };

    using __simd_cast_type = std::conditional_t<
        std::is_same<__intrinsic_type_t<_T, _N>, __vector_type_t<_T, _N>>::value,
        simd_cast_type1, simd_cast_type2>;
    //}}}2
};

// __neon_is_vectorizable {{{1
#if _GLIBCXX_SIMD_HAVE_NEON_ABI
template <class _T> struct __neon_is_vectorizable : __is_vectorizable<_T> {};
template <> struct __neon_is_vectorizable<long double> : false_type {};
#if !_GLIBCXX_SIMD_HAVE_FULL_NEON_ABI
template <> struct __neon_is_vectorizable<double> : false_type {};
#endif
#else
template <class _T> struct __neon_is_vectorizable : false_type {};
#endif

// __sse_is_vectorizable {{{1
#if _GLIBCXX_SIMD_HAVE_FULL_SSE_ABI
template <class _T> struct __sse_is_vectorizable : __is_vectorizable<_T> {};
template <> struct __sse_is_vectorizable<long double> : false_type {};
#elif _GLIBCXX_SIMD_HAVE_SSE_ABI
template <class _T> struct __sse_is_vectorizable : __is_same<_T, float> {};
#else
template <class _T> struct __sse_is_vectorizable : false_type {};
#endif

// __avx_is_vectorizable {{{1
#if _GLIBCXX_SIMD_HAVE_FULL_AVX_ABI
template <class _T> struct __avx_is_vectorizable : __is_vectorizable<_T> {};
#elif _GLIBCXX_SIMD_HAVE_AVX_ABI
template <class _T> struct __avx_is_vectorizable : std::is_floating_point<_T> {};
#else
template <class _T> struct __avx_is_vectorizable : false_type {};
#endif
template <> struct __avx_is_vectorizable<long double> : false_type {};

// __avx512_is_vectorizable {{{1
#if _GLIBCXX_SIMD_HAVE_AVX512_ABI
template <class _T> struct __avx512_is_vectorizable : __is_vectorizable<_T> {};
template <> struct __avx512_is_vectorizable<long double> : false_type {};
#if !_GLIBCXX_SIMD_HAVE_FULL_AVX512_ABI
template <> struct __avx512_is_vectorizable<  char> : false_type {};
template <> struct __avx512_is_vectorizable< __uchar> : false_type {};
template <> struct __avx512_is_vectorizable< __schar> : false_type {};
template <> struct __avx512_is_vectorizable< short> : false_type {};
template <> struct __avx512_is_vectorizable<ushort> : false_type {};
template <> struct __avx512_is_vectorizable<char16_t> : false_type {};
template <> struct __avx512_is_vectorizable<wchar_t> : __bool_constant<sizeof(wchar_t) >= 4> {};
#endif
#else
template <class _T> struct __avx512_is_vectorizable : false_type {};
#endif

// }}}
// __implicit_mask_abi_base {{{
template <int _Bytes, class _Abi> struct __implicit_mask_abi_base {
    template <class _T>
    using implicit_mask_type =
        __vector_type_t<__int_for_sizeof_t<_T>, simd_size_v<_T, _Abi>>;

    template <class _T>
    static constexpr auto implicit_mask =
        reinterpret_cast<__vector_type_t<_T, simd_size_v<_T, _Abi>>>(
            _Abi::is_partial ? __generate_builtin<implicit_mask_type<_T>>([](auto __i) {
                return __i < _Bytes / sizeof(_T) ? -1 : 0;
            })
                            : ~implicit_mask_type<_T>());

    template <class _T, class _TVT = __vector_traits<_T>>
    static constexpr auto masked(_T __x)
    {
        using _U = typename _TVT::value_type;
        if constexpr (_Abi::is_partial) {
            return __and(__x , implicit_mask<_U>);
        } else {
            return __x;
        }
    }
};

// }}}

namespace simd_abi
{
// __combine {{{1
template <int _N, class _Abi> struct __combine {
    template <class _T> static constexpr size_t size = _N *_Abi::template size<_T>;
    template <class _T> static constexpr size_t full_size = size<_T>;

    static constexpr int factor = _N;
    using member_abi = _Abi;

    // validity traits {{{2
    // allow 2x, 3x, and 4x "unroll"
    struct is_valid_abi_tag
        : __bool_constant<(_N > 1 && _N <= 4) && _Abi::is_valid_abi_tag> {
    };
    template <class _T> struct is_valid_size_for : _Abi::template is_valid_size_for<_T> {
    };
    template <class _T>
    struct is_valid : conjunction<is_valid_abi_tag, typename _Abi::template is_valid<_T>> {
    };
    template <class _T> static constexpr bool is_valid_v = is_valid<_T>::value;

    // simd/__mask_impl_type {{{2
    using __simd_impl_type = __combine_simd_impl<_N, _Abi>;
    using __mask_impl_type = __combine_mask_impl<_N, _Abi>;

    // __traits {{{2
    template <class _T, bool = is_valid_v<_T>> struct __traits : __invalid_traits {
    };

    template <class _T> struct __traits<_T, true> {
        using is_valid = true_type;
        using __simd_impl_type = __combine_simd_impl<_N, _Abi>;
        using __mask_impl_type = __combine_mask_impl<_N, _Abi>;

        // simd and simd_mask member types {{{2
        using __simd_member_type =
            std::array<typename _Abi::template __traits<_T>::__simd_member_type, _N>;
        using __mask_member_type =
            std::array<typename _Abi::template __traits<_T>::__mask_member_type, _N>;
        static constexpr size_t __simd_member_alignment =
            _Abi::template __traits<_T>::__simd_member_alignment;
        static constexpr size_t __mask_member_alignment =
            _Abi::template __traits<_T>::__mask_member_alignment;

        // __simd_base / base class for simd, providing extra conversions {{{2
        struct __simd_base {
            explicit operator const __simd_member_type &() const
            {
                return static_cast<const simd<_T, __combine> *>(this)->_M_data;
            }
        };

        // __mask_base {{{2
        // empty. The std::bitset interface suffices
        struct __mask_base {
            explicit operator const __mask_member_type &() const
            {
                return static_cast<const simd_mask<_T, __combine> *>(this)->_M_data;
            }
        };

        // __simd_cast_type {{{2
        struct __simd_cast_type {
            __simd_cast_type(const __simd_member_type &dd) : _M_data(dd) {}
            explicit operator const __simd_member_type &() const { return _M_data; }

        private:
            const __simd_member_type &_M_data;
        };

        // __mask_cast_type {{{2
        struct __mask_cast_type {
            __mask_cast_type(const __mask_member_type &dd) : _M_data(dd) {}
            explicit operator const __mask_member_type &() const { return _M_data; }

        private:
            const __mask_member_type &_M_data;
        };
        //}}}2
    };
    //}}}2
};
// __neon_abi {{{1
template <int _Bytes>
struct __neon_abi : __implicit_mask_abi_base<_Bytes, __neon_abi<_Bytes>> {
    template <class _T> static constexpr size_t size = _Bytes / sizeof(_T);
    template <class _T> static constexpr size_t full_size = 16 / sizeof(_T);
    static constexpr bool is_partial = _Bytes < 16;

    // validity traits {{{2
    struct is_valid_abi_tag : __bool_constant<(_Bytes > 0 && _Bytes <= 16)> {
    };
    template <class _T>
    struct is_valid_size_for
        : __bool_constant<(_Bytes / sizeof(_T) > 1 && _Bytes % sizeof(_T) == 0)> {
    };
    template <class _T>
    struct is_valid : conjunction<is_valid_abi_tag, __neon_is_vectorizable<_T>,
                                  is_valid_size_for<_T>> {
    };
    template <class _T> static constexpr bool is_valid_v = is_valid<_T>::value;

    // simd/__mask_impl_type {{{2
    using __simd_impl_type = __neon_simd_impl;
    using __mask_impl_type = __neon_mask_impl;

    // __traits {{{2
    template <class _T>
    using __traits = std::conditional_t<is_valid_v<_T>,
                                      __gnu_traits<_T, _T, __neon_abi, full_size<_T>>,
                                      __invalid_traits>;
    //}}}2
};

// __sse_abi {{{1
template <int _Bytes>
struct __sse_abi : __implicit_mask_abi_base<_Bytes, __sse_abi<_Bytes>> {
    template <class _T> static constexpr size_t size = _Bytes / sizeof(_T);
    template <class _T> static constexpr size_t full_size = 16 / sizeof(_T);
    static constexpr bool is_partial = _Bytes < 16;

    // validity traits {{{2
    // allow 2x, 3x, and 4x "unroll"
    struct is_valid_abi_tag : __bool_constant<_Bytes == 16> {};
    //struct is_valid_abi_tag : __bool_constant<(_Bytes > 0 && _Bytes <= 16)> {};
    template <class _T>
    struct is_valid_size_for
        : __bool_constant<(_Bytes / sizeof(_T) > 1 && _Bytes % sizeof(_T) == 0)> {
    };

    template <class _T>
    struct is_valid
        : conjunction<is_valid_abi_tag, __sse_is_vectorizable<_T>, is_valid_size_for<_T>> {
    };
    template <class _T> static constexpr bool is_valid_v = is_valid<_T>::value;

    // simd/__mask_impl_type {{{2
    using __simd_impl_type = __sse_simd_impl;
    using __mask_impl_type = __sse_mask_impl;

    // __traits {{{2
    template <class _T>
    using __traits = std::conditional_t<is_valid_v<_T>,
                                      __gnu_traits<_T, _T, __sse_abi, full_size<_T>>,
                                      __invalid_traits>;
    //}}}2
};

// __avx_abi {{{1
template <int _Bytes>
struct __avx_abi : __implicit_mask_abi_base<_Bytes, __avx_abi<_Bytes>> {
    template <class _T> static constexpr size_t size = _Bytes / sizeof(_T);
    template <class _T> static constexpr size_t full_size = 32 / sizeof(_T);
    static constexpr bool is_partial = _Bytes < 32;

    // validity traits {{{2
    // - allow 2x, 3x, and 4x "unroll"
    // - disallow <= 16 _Bytes as that's covered by __sse_abi
    struct is_valid_abi_tag : __bool_constant<_Bytes == 32> {};
    /* TODO:
    struct is_valid_abi_tag
        : __bool_constant<((_Bytes > 16 && _Bytes <= 32) || _Bytes == 64 ||
                                 _Bytes == 96 || _Bytes == 128)> {
    };
    */
    template <class _T>
    struct is_valid_size_for : __bool_constant<(_Bytes % sizeof(_T) == 0)> {
    };
    template <class _T>
    struct is_valid
        : conjunction<is_valid_abi_tag, __avx_is_vectorizable<_T>, is_valid_size_for<_T>> {
    };
    template <class _T> static constexpr bool is_valid_v = is_valid<_T>::value;

    // simd/__mask_impl_type {{{2
    using __simd_impl_type = __avx_simd_impl;
    using __mask_impl_type = __avx_mask_impl;

    // __traits {{{2
    template <class _T>
    using __traits = std::conditional_t<is_valid_v<_T>,
                                      __gnu_traits<_T, _T, __avx_abi, full_size<_T>>,
                                      __invalid_traits>;
    //}}}2
};

// __avx512_abi {{{1
template <int _Bytes> struct __avx512_abi {
    template <class _T> static constexpr size_t size = _Bytes / sizeof(_T);
    template <class _T> static constexpr size_t full_size = 64 / sizeof(_T);
    static constexpr bool is_partial = _Bytes < 64;

    // validity traits {{{2
    // - disallow <= 32 _Bytes as that's covered by __sse_abi and __avx_abi
    // TODO: consider AVX512VL
    struct is_valid_abi_tag : __bool_constant<_Bytes == 64> {};
    /* TODO:
    struct is_valid_abi_tag
        : __bool_constant<(_Bytes > 32 && _Bytes <= 64)> {
    };
    */
    template <class _T>
    struct is_valid_size_for : __bool_constant<(_Bytes % sizeof(_T) == 0)> {
    };
    template <class _T>
    struct is_valid
        : conjunction<is_valid_abi_tag, __avx512_is_vectorizable<_T>, is_valid_size_for<_T>> {
    };
    template <class _T> static constexpr bool is_valid_v = is_valid<_T>::value;

    // implicit mask {{{2
    template <class _T>
    using implicit_mask_type = __bool_storage_member_type_t<64 / sizeof(_T)>;

    template <class _T>
    static constexpr implicit_mask_type<_T> implicit_mask =
        _Bytes == 64 ? ~implicit_mask_type<_T>()
                    : (implicit_mask_type<_T>(1) << (_Bytes / sizeof(_T))) - 1;

    template <class _T, class = enable_if_t<__is_bitmask_v<_T>>>
    static constexpr _T masked(_T __x)
    {
        if constexpr (is_partial) {
            constexpr size_t _N = sizeof(_T) * 8;
            return __x &
                   ((__bool_storage_member_type_t<_N>(1) << (_Bytes * _N / 64)) - 1);
        } else {
            return __x;
        }
    }

    // simd/__mask_impl_type {{{2
    using __simd_impl_type = __avx512_simd_impl;
    using __mask_impl_type = __avx512_mask_impl;

    // __traits {{{2
    template <class _T>
    using __traits =
        std::conditional_t<is_valid_v<_T>,
                           __gnu_traits<_T, bool, __avx512_abi, full_size<_T>>,
                           __invalid_traits>;
    //}}}2
};

// __scalar_abi {{{1
struct __scalar_abi {
    template <class _T> static constexpr size_t size = 1;
    template <class _T> static constexpr size_t full_size = 1;
    struct is_valid_abi_tag : true_type {};
    template <class _T> struct is_valid_size_for : true_type {};
    template <class _T> struct is_valid : __is_vectorizable<_T> {};
    template <class _T> static constexpr bool is_valid_v = is_valid<_T>::value;

    using __simd_impl_type = __scalar_simd_impl;
    using __mask_impl_type = __scalar_mask_impl;

    template <class _T, bool = is_valid_v<_T>> struct __traits : __invalid_traits {
    };

    template <class _T> struct __traits<_T, true> {
        using is_valid = true_type;
        using __simd_impl_type = __scalar_simd_impl;
        using __mask_impl_type = __scalar_mask_impl;
        using __simd_member_type = _T;
        using __mask_member_type = bool;
        static constexpr size_t __simd_member_alignment = alignof(__simd_member_type);
        static constexpr size_t __mask_member_alignment = alignof(__mask_member_type);

        // nothing the user can spell converts to/from simd/simd_mask
        struct __simd_cast_type {
            __simd_cast_type() = delete;
        };
        struct __mask_cast_type {
            __mask_cast_type() = delete;
        };
        struct __simd_base {};
        struct __mask_base {};
    };
};

// __fixed_abi {{{1
template <int _N> struct __fixed_abi {
    template <class _T> static constexpr size_t size = _N;
    template <class _T> static constexpr size_t full_size = _N;
    // validity traits {{{2
    struct is_valid_abi_tag
        : public __bool_constant<(_N > 0)> {
    };
    template <class _T>
    struct is_valid_size_for
        : __bool_constant<((_N <= simd_abi::max_fixed_size<_T>) ||
                                 (simd_abi::__neon::is_valid_v<char> &&
                                  _N == simd_size_v<char, simd_abi::__neon>) ||
                                 (simd_abi::__sse::is_valid_v<char> &&
                                  _N == simd_size_v<char, simd_abi::__sse>) ||
                                 (simd_abi::__avx::is_valid_v<char> &&
                                  _N == simd_size_v<char, simd_abi::__avx>) ||
                                 (simd_abi::__avx512::is_valid_v<char> &&
                                  _N == simd_size_v<char, simd_abi::__avx512>))> {
    };
    template <class _T>
    struct is_valid
        : conjunction<is_valid_abi_tag, __is_vectorizable<_T>, is_valid_size_for<_T>> {
    };
    template <class _T> static constexpr bool is_valid_v = is_valid<_T>::value;

    // simd/__mask_impl_type {{{2
    using __simd_impl_type = __fixed_size_simd_impl<_N>;
    using __mask_impl_type = __fixed_size_mask_impl<_N>;

    // __traits {{{2
    template <class _T, bool = is_valid_v<_T>> struct __traits : __invalid_traits {
    };

    template <class _T> struct __traits<_T, true> {
        using is_valid = true_type;
        using __simd_impl_type = __fixed_size_simd_impl<_N>;
        using __mask_impl_type = __fixed_size_mask_impl<_N>;

        // simd and simd_mask member types {{{2
        using __simd_member_type = __fixed_size_storage<_T, _N>;
        using __mask_member_type = std::bitset<_N>;
        static constexpr size_t __simd_member_alignment =
            __next_power_of_2(_N * sizeof(_T));
        static constexpr size_t __mask_member_alignment = alignof(__mask_member_type);

        // __simd_base / base class for simd, providing extra conversions {{{2
        struct __simd_base {
            // The following ensures, function arguments are passed via the stack. This is
            // important for ABI compatibility across TU boundaries
            __simd_base(const __simd_base &) {}
            __simd_base() = default;

            explicit operator const __simd_member_type &() const
            {
                return static_cast<const simd<_T, __fixed_abi> *>(this)->_M_data;
            }
            explicit operator std::array<_T, _N>() const
            {
                std::array<_T, _N> __r;
                // __simd_member_type can be larger because of higher alignment
                static_assert(sizeof(__r) <= sizeof(__simd_member_type), "");
                std::memcpy(__r.data(), &static_cast<const __simd_member_type &>(*this),
                            sizeof(__r));
                return __r;
            }
        };

        // __mask_base {{{2
        // empty. The std::bitset interface suffices
        struct __mask_base {};

        // __simd_cast_type {{{2
        struct __simd_cast_type {
            __simd_cast_type(const std::array<_T, _N> &);
            __simd_cast_type(const __simd_member_type &dd) : _M_data(dd) {}
            explicit operator const __simd_member_type &() const { return _M_data; }

        private:
            const __simd_member_type &_M_data;
        };

        // __mask_cast_type {{{2
        class __mask_cast_type
        {
            __mask_cast_type() = delete;
        };
        //}}}2
    };
};

//}}}
}  // namespace simd_abi

// __scalar_abi_wrapper {{{1
template <int _Bytes> struct __scalar_abi_wrapper : simd_abi::__scalar_abi {
    template <class _T>
    static constexpr bool is_valid_v = simd_abi::__scalar_abi::is_valid<_T>::value &&
                                       sizeof(_T) == _Bytes;
};

// __decay_abi metafunction {{{1
template <class _T> struct __decay_abi {
    using type = _T;
};
template <int _Bytes> struct __decay_abi<__scalar_abi_wrapper<_Bytes>> {
    using type = simd_abi::scalar;
};

// __full_abi metafunction {{{1
template <template <int> class> struct __full_abi;
template <> struct __full_abi<simd_abi::__neon_abi> { using type = simd_abi::__neon128; };
template <> struct __full_abi<simd_abi::__sse_abi> { using type = simd_abi::__sse; };
template <> struct __full_abi<simd_abi::__avx_abi> { using type = simd_abi::__avx; };
template <> struct __full_abi<simd_abi::__avx512_abi> { using type = simd_abi::__avx512; };
template <> struct __full_abi<__scalar_abi_wrapper> {
    using type = simd_abi::scalar;
};

// __abi_list {{{1
template <template <int> class...> struct __abi_list {
    template <class, int> static constexpr bool has_valid_abi = false;
    template <class, int> using first_valid_abi = void;
    template <class, int> using best_abi = void;
};

template <template <int> class _A0, template <int> class... Rest>
struct __abi_list<_A0, Rest...> {
    template <class _T, int _N>
    static constexpr bool has_valid_abi = _A0<sizeof(_T) * _N>::template is_valid_v<_T> ||
                                          __abi_list<Rest...>::template has_valid_abi<_T, _N>;
    template <class _T, int _N>
    using first_valid_abi =
        std::conditional_t<_A0<sizeof(_T) * _N>::template is_valid_v<_T>,
                           typename __decay_abi<_A0<sizeof(_T) * _N>>::type,
                           typename __abi_list<Rest...>::template first_valid_abi<_T, _N>>;
    using _B = typename __full_abi<_A0>::type;
    template <class _T, int _N>
    using best_abi = std::conditional_t<
        _A0<sizeof(_T) * _N>::template is_valid_v<_T>,
        typename __decay_abi<_A0<sizeof(_T) * _N>>::type,
        std::conditional_t<(_B::template is_valid_v<_T> && _B::template size<_T> <= _N), _B,
                           typename __abi_list<Rest...>::template best_abi<_T, _N>>>;
};

// }}}1

// the following lists all native ABIs, which makes them accessible to simd_abi::deduce
// and select_best_vector_type_t (for fixed_size). Order matters: Whatever comes first has
// higher priority.
using __all_native_abis =
    __abi_list<simd_abi::__avx512_abi, simd_abi::__avx_abi, simd_abi::__sse_abi,
             simd_abi::__neon_abi, __scalar_abi_wrapper>;

// valid __traits specialization {{{1
template <class _T, class _Abi>
struct __simd_traits<_T, _Abi, std::void_t<typename _Abi::template is_valid<_T>>>
    : _Abi::template __traits<_T> {
};

// __deduce_impl specializations {{{1
// try all native ABIs (including scalar) first
template <class _T, std::size_t _N>
struct __deduce_impl<_T, _N,
                   enable_if_t<__all_native_abis::template has_valid_abi<_T, _N>>> {
    using type = __all_native_abis::first_valid_abi<_T, _N>;
};

// fall back to fixed_size only if scalar and native ABIs don't match
template <class _T, std::size_t _N, class = void> struct __deduce_fixed_size_fallback {};
template <class _T, std::size_t _N>
struct __deduce_fixed_size_fallback<
    _T, _N, enable_if_t<simd_abi::fixed_size<_N>::template is_valid_v<_T>>> {
    using type = simd_abi::fixed_size<_N>;
};
template <class _T, std::size_t _N, class>
struct __deduce_impl : public __deduce_fixed_size_fallback<_T, _N> {
};

//}}}1
// __is_abi {{{
template <template <int> class _Abi, int _Bytes> constexpr int __abi_bytes_impl(_Abi<_Bytes> *)
{
    return _Bytes;
}
template <class _T> constexpr int __abi_bytes_impl(_T *) { return -1; }
template <class _Abi>
inline constexpr int __abi_bytes = __abi_bytes_impl(static_cast<_Abi *>(nullptr));

template <class _Abi0, class _Abi1> constexpr bool __is_abi()
{
    return std::is_same_v<_Abi0, _Abi1>;
}
template <template <int> class _Abi0, class _Abi1> constexpr bool __is_abi()
{
    return std::is_same_v<_Abi0<__abi_bytes<_Abi1>>, _Abi1>;
}
template <class _Abi0, template <int> class _Abi1> constexpr bool __is_abi()
{
    return std::is_same_v<_Abi1<__abi_bytes<_Abi0>>, _Abi0>;
}
template <template <int> class _Abi0, template <int> class _Abi1> constexpr bool __is_abi()
{
    return std::is_same_v<_Abi0<0>, _Abi1<0>>;
}

// }}}
// __is_combined_abi{{{
template <template <int, class> class _Combine, int _N, class _Abi>
constexpr bool __is_combined_abi(_Combine<_N, _Abi> *)
{
    return std::is_same_v<_Combine<_N, _Abi>, simd_abi::__combine<_N, _Abi>>;
}
template <class _Abi> constexpr bool __is_combined_abi(_Abi *)
{
    return false;
}

template <class _Abi> constexpr bool __is_combined_abi()
{
    return __is_combined_abi(static_cast<_Abi *>(nullptr));
}

// }}}

// simd_mask {{{
template <class _T, class _Abi> class simd_mask : public __simd_traits<_T, _Abi>::__mask_base
{
    // types, tags, and friends {{{
    using __traits = __simd_traits<_T, _Abi>;
    using __impl = typename __traits::__mask_impl_type;
    using __member_type = typename __traits::__mask_member_type;
    static constexpr _T *__type_tag = nullptr;
    friend typename __traits::__mask_base;
    friend class simd<_T, _Abi>;  // to construct masks on return
    friend __impl;
    friend typename __traits::__simd_impl_type;  // to construct masks on return and
                                             // inspect data on masked operations
    // }}}
    // is_<abi> {{{
    static constexpr bool __is_scalar() { return __is_abi<_Abi, simd_abi::scalar>(); }
    static constexpr bool __is_sse() { return __is_abi<_Abi, simd_abi::__sse_abi>(); }
    static constexpr bool __is_avx() { return __is_abi<_Abi, simd_abi::__avx_abi>(); }
    static constexpr bool __is_avx512()
    {
        return __is_abi<_Abi, simd_abi::__avx512_abi>();
    }
    static constexpr bool __is_neon()
    {
        return __is_abi<_Abi, simd_abi::__neon_abi>();
    }
    static constexpr bool __is_fixed() { return __is_fixed_size_abi_v<_Abi>; }
    static constexpr bool __is_combined() { return __is_combined_abi<_Abi>(); }

    // }}}

public:
    // member types {{{
    using value_type = bool;
    using reference = __smart_reference<__member_type, __impl, value_type>;
    using simd_type = simd<_T, _Abi>;
    using abi_type = _Abi;

    // }}}
    static constexpr size_t size() { return __size_or_zero<_T, _Abi>; }
    // constructors & assignment {{{
    simd_mask() = default;
    simd_mask(const simd_mask &) = default;
    simd_mask(simd_mask &&) = default;
    simd_mask &operator=(const simd_mask &) = default;
    simd_mask &operator=(simd_mask &&) = default;

    // }}}

    // access to internal representation (suggested extension)
    _GLIBCXX_SIMD_ALWAYS_INLINE explicit simd_mask(typename __traits::__mask_cast_type __init) : _M_data{__init} {}
    // conversions to internal type is done in __mask_base

    // bitset interface (extension to be proposed) {{{
    _GLIBCXX_SIMD_ALWAYS_INLINE static simd_mask __from_bitset(std::bitset<size()> bs)
    {
        return {__bitset_init, bs};
    }
    _GLIBCXX_SIMD_ALWAYS_INLINE std::bitset<size()> __to_bitset() const {
        if constexpr (__is_scalar()) {
            return unsigned(_M_data);
        } else if constexpr (__is_fixed()) {
            return _M_data;
        } else {
            return __vector_to_bitset(builtin());
        }
    }

    // }}}
    // explicit broadcast constructor {{{
    _GLIBCXX_SIMD_ALWAYS_INLINE explicit constexpr simd_mask(value_type __x) : _M_data(__broadcast(__x)) {}

    // }}}
    // implicit type conversion constructor {{{
    template <class _U, class = enable_if_t<
                            conjunction<is_same<abi_type, simd_abi::fixed_size<size()>>,
                                        is_same<_U, _U>>::value>>
    _GLIBCXX_SIMD_ALWAYS_INLINE simd_mask(
        const simd_mask<_U, simd_abi::fixed_size<size()>> &__x)
        : simd_mask{__bitset_init, __data(__x)}
    {
    }
    // }}}
    /* reference implementation for explicit simd_mask casts {{{
    template <class _U, class = enable_if<
             (size() == simd_mask<_U, _Abi>::size()) &&
             conjunction<std::is_integral<_T>, std::is_integral<_U>,
             __negation<std::is_same<_Abi, simd_abi::fixed_size<size()>>>,
             __negation<std::is_same<_T, _U>>>::value>>
    simd_mask(const simd_mask<_U, _Abi> &__x)
        : _M_data{__x._M_data}
    {
    }
    template <class _U, class _Abi2, class = enable_if<conjunction<
         __negation<std::is_same<abi_type, _Abi2>>,
             std::is_same<abi_type, simd_abi::fixed_size<size()>>>::value>>
    simd_mask(const simd_mask<_U, _Abi2> &__x)
    {
        __x.copy_to(&_M_data[0], vector_aligned);
    }
    }}} */

    // load __impl {{{
private:
    template <class _F>
    _GLIBCXX_SIMD_INTRINSIC static __member_type load_wrapper(const value_type* mem,
                                                              [[maybe_unused]] _F __f)
    {
        if constexpr (__is_scalar()) {
            return mem[0];
        } else if constexpr (__is_fixed()) {
            const fixed_size_simd<unsigned char, size()> bools(
                reinterpret_cast<const __may_alias<unsigned char> *>(mem), __f);
            return __data(bools != 0);
        } else if constexpr (__is_sse()) {
            if constexpr (size() == 2 && __have_sse2) {
                return __to_storage(_mm_set_epi32(-int(mem[1]), -int(mem[1]),
                                                        -int(mem[0]), -int(mem[0])));
            } else if constexpr (size() == 4 && __have_sse2) {
                __m128i __k = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(mem));
                __k = _mm_cmpgt_epi16(_mm_unpacklo_epi8(__k, __k), _mm_setzero_si128());
                return __to_storage(_mm_unpacklo_epi16(__k, __k));
            } else if constexpr (size() == 4 && __have_mmx) {
                __m128 __k =
                    _mm_cvtpi8_ps(_mm_cvtsi32_si64(*reinterpret_cast<const int *>(mem)));
                _mm_empty();
                return __to_storage(_mm_cmpgt_ps(__k, __m128()));
            } else if constexpr (size() == 8 && __have_sse2) {
                const auto __k = __make_builtin<long long>(
                    *reinterpret_cast<const __may_alias<long long> *>(mem), 0);
                if constexpr (__have_sse2) {
                    return __to_storage(
                        __vector_bitcast<short>(_mm_unpacklo_epi8(__k, __k)) != 0);
                }
            } else if constexpr (size() == 16 && __have_sse2) {
                return __vector_bitcast<_T>(
                    _mm_cmpgt_epi8(__vector_load<long long, 2>(mem, __f), __m128i()));
            } else {
                __assert_unreachable<_F>();
            }
        } else if constexpr (__is_avx()) {
            if constexpr (size() == 4 && __have_avx) {
                int bool4;
                if constexpr (__is_aligned_v<_F, 4>) {
                    bool4 = *reinterpret_cast<const __may_alias<int> *>(mem);
                } else {
                    std::memcpy(&bool4, mem, 4);
                }
                const auto __k = __to_intrin(
                    (__vector_broadcast<4>(bool4) &
                     __make_builtin<int>(0x1, 0x100, 0x10000, 0x1000000)) != 0);
                return __to_storage(
                    __concat(_mm_unpacklo_epi32(__k, __k), _mm_unpackhi_epi32(__k, __k)));
            } else if constexpr (size() == 8 && __have_avx) {
                auto __k = __vector_load<long long, 2, 8>(mem, __f);
                __k = _mm_cmpgt_epi16(_mm_unpacklo_epi8(__k, __k), __m128i());
                return __to_storage(
                    __concat(_mm_unpacklo_epi16(__k, __k), _mm_unpackhi_epi16(__k, __k)));
            } else if constexpr (size() == 16 && __have_avx) {
                const auto __k =
                    _mm_cmpgt_epi8(__vector_load<long long, 2>(mem, __f), __m128i());
                return __concat(_mm_unpacklo_epi8(__k, __k), _mm_unpackhi_epi8(__k, __k));
            } else if constexpr (size() == 32 && __have_avx2) {
                return __vector_bitcast<_T>(
                    _mm256_cmpgt_epi8(__vector_load<long long, 4>(mem, __f), __m256i()));
            } else {
                __assert_unreachable<_F>();
            }
        } else if constexpr (__is_avx512()) {
            if constexpr (size() == 8) {
                const auto __a = __vector_load<long long, 2, 8>(mem, __f);
                if constexpr (__have_avx512bw_vl) {
                    return _mm_test_epi8_mask(__a, __a);
                } else {
                    const auto __b = _mm512_cvtepi8_epi64(__a);
                    return _mm512_test_epi64_mask(__b, __b);
                }
            } else if constexpr (size() == 16) {
                const auto __a = __vector_load<long long, 2>(mem, __f);
                if constexpr (__have_avx512bw_vl) {
                    return _mm_test_epi8_mask(__a, __a);
                } else {
                    const auto __b = _mm512_cvtepi8_epi32(__a);
                    return _mm512_test_epi32_mask(__b, __b);
                }
            } else if constexpr (size() == 32) {
                if constexpr (__have_avx512bw_vl) {
                    const auto __a = __vector_load<long long, 4>(mem, __f);
                    return _mm256_test_epi8_mask(__a, __a);
                } else {
                    const auto __a =
                        _mm512_cvtepi8_epi32(__vector_load<long long, 2>(mem, __f));
                    const auto __b =
                        _mm512_cvtepi8_epi32(__vector_load<long long, 2>(mem + 16, __f));
                    return _mm512_test_epi32_mask(__a, __a) |
                           (_mm512_test_epi32_mask(__b, __b) << 16);
                }
            } else if constexpr (size() == 64) {
                if constexpr (__have_avx512bw) {
                    const auto __a = __vector_load<long long, 8>(mem, __f);
                    return _mm512_test_epi8_mask(__a, __a);
                } else {
                    const auto __a =
                        _mm512_cvtepi8_epi32(__vector_load<long long, 2>(mem, __f));
                    const auto __b = _mm512_cvtepi8_epi32(
                        __vector_load<long long, 2>(mem + 16, __f));
                    const auto __c = _mm512_cvtepi8_epi32(
                        __vector_load<long long, 2>(mem + 32, __f));
                    const auto __d = _mm512_cvtepi8_epi32(
                        __vector_load<long long, 2>(mem + 48, __f));
                    return _mm512_test_epi32_mask(__a, __a) |
                           (_mm512_test_epi32_mask(__b, __b) << 16) |
                           (_mm512_test_epi32_mask(__c, __c) << 32) |
                           (_mm512_test_epi32_mask(__d, __d) << 48);
                }
            } else {
                __assert_unreachable<_F>();
            }
        } else {
            __assert_unreachable<_F>();
        }
    }

public :
    // }}}
    // load constructor {{{
    template <class _Flags>
    _GLIBCXX_SIMD_ALWAYS_INLINE simd_mask(const value_type* mem, _Flags __f)
        : _M_data(load_wrapper(mem, __f))
    {
    }
    template <class _Flags>
    _GLIBCXX_SIMD_ALWAYS_INLINE simd_mask(const value_type *mem, simd_mask __k, _Flags f) : _M_data{}
    {
        _M_data = __impl::masked_load(_M_data, __k._M_data, mem, f);
    }

    // }}}
    // loads [simd_mask.load] {{{
    template <class _Flags> _GLIBCXX_SIMD_ALWAYS_INLINE void copy_from(const value_type *mem, _Flags f)
    {
        _M_data = load_wrapper(mem, f);
    }

    // }}}
    // stores [simd_mask.store] {{{
    template <class _Flags> _GLIBCXX_SIMD_ALWAYS_INLINE void copy_to(value_type *mem, _Flags f) const
    {
        __impl::store(_M_data, mem, f);
    }

    // }}}
    // scalar access {{{
    _GLIBCXX_SIMD_ALWAYS_INLINE reference operator[](size_t __i) { return {_M_data, int(__i)}; }
    _GLIBCXX_SIMD_ALWAYS_INLINE value_type operator[](size_t __i) const {
        if constexpr (__is_scalar()) {
            _GLIBCXX_SIMD_ASSERT(__i == 0);
            __unused(__i);
            return _M_data;
        } else {
            return _M_data[__i];
        }
    }

    // }}}
    // negation {{{
    _GLIBCXX_SIMD_ALWAYS_INLINE simd_mask operator!() const
    {
        if constexpr (__is_scalar()) {
            return {__private_init, !_M_data};
        } else if constexpr (__is_avx512() || __is_fixed()) {
            return simd_mask(__private_init, ~builtin());
        } else {
            return {__private_init,
                    __to_storage(~__vector_bitcast<uint>(builtin()))};
        }
    }

    // }}}
    // simd_mask binary operators [simd_mask.binary] {{{
    _GLIBCXX_SIMD_ALWAYS_INLINE friend simd_mask operator&&(const simd_mask &__x, const simd_mask &__y)
    {
        return {__private_init, __impl::logical_and(__x._M_data, __y._M_data)};
    }
    _GLIBCXX_SIMD_ALWAYS_INLINE friend simd_mask operator||(const simd_mask &__x, const simd_mask &__y)
    {
        return {__private_init, __impl::logical_or(__x._M_data, __y._M_data)};
    }

    _GLIBCXX_SIMD_ALWAYS_INLINE friend simd_mask operator&(const simd_mask &__x, const simd_mask &__y)
    {
        return {__private_init, __impl::bit_and(__x._M_data, __y._M_data)};
    }
    _GLIBCXX_SIMD_ALWAYS_INLINE friend simd_mask operator|(const simd_mask &__x, const simd_mask &__y)
    {
        return {__private_init, __impl::bit_or(__x._M_data, __y._M_data)};
    }
    _GLIBCXX_SIMD_ALWAYS_INLINE friend simd_mask operator^(const simd_mask &__x, const simd_mask &__y)
    {
        return {__private_init, __impl::bit_xor(__x._M_data, __y._M_data)};
    }

    _GLIBCXX_SIMD_ALWAYS_INLINE friend simd_mask &operator&=(simd_mask &__x, const simd_mask &__y)
    {
        __x._M_data = __impl::bit_and(__x._M_data, __y._M_data);
        return __x;
    }
    _GLIBCXX_SIMD_ALWAYS_INLINE friend simd_mask &operator|=(simd_mask &__x, const simd_mask &__y)
    {
        __x._M_data = __impl::bit_or(__x._M_data, __y._M_data);
        return __x;
    }
    _GLIBCXX_SIMD_ALWAYS_INLINE friend simd_mask &operator^=(simd_mask &__x, const simd_mask &__y)
    {
        __x._M_data = __impl::bit_xor(__x._M_data, __y._M_data);
        return __x;
    }

    // }}}
    // simd_mask compares [simd_mask.comparison] {{{
    _GLIBCXX_SIMD_ALWAYS_INLINE friend simd_mask operator==(const simd_mask &__x, const simd_mask &__y)
    {
        return !operator!=(__x, __y);
    }
    _GLIBCXX_SIMD_ALWAYS_INLINE friend simd_mask operator!=(const simd_mask &__x, const simd_mask &__y)
    {
        return {__private_init, __impl::bit_xor(__x._M_data, __y._M_data)};
    }

    // }}}
    // "private" because of the first arguments's namespace
    _GLIBCXX_SIMD_INTRINSIC simd_mask(__private_init_t, typename __traits::__mask_member_type __init)
        : _M_data(__init)
    {
    }

    // "private" because of the first arguments's namespace
    template <class _F, class = decltype(bool(std::declval<_F>()(size_t())))>
    _GLIBCXX_SIMD_INTRINSIC simd_mask(__private_init_t, _F &&gen)
    {
        for (size_t __i = 0; __i < size(); ++__i) {
            __impl::set(_M_data, __i, gen(__i));
        }
    }

    // "private" because of the first arguments's namespace
    _GLIBCXX_SIMD_INTRINSIC simd_mask(__bitset_init_t, std::bitset<size()> __init)
        : _M_data(__impl::__from_bitset(__init, __type_tag))
    {
    }

private:
    _GLIBCXX_SIMD_INTRINSIC static constexpr __member_type __broadcast(value_type __x)  // {{{
    {
        if constexpr (__is_scalar()) {
            return __x;
        } else if constexpr (__is_fixed()) {
            return __x ? ~__member_type() : __member_type();
        } else if constexpr (__is_avx512()) {
            using mmask_type = typename __bool_storage_member_type<size()>::type;
            return __x ? _Abi::masked(static_cast<mmask_type>(~mmask_type())) : mmask_type();
        } else {
            using _U = __vector_type_t<__int_for_sizeof_t<_T>, size()>;
            return __to_storage(__x ? _Abi::masked(~_U()) : _U());
        }
    }

    // }}}
    auto intrin() const  // {{{
    {
        if constexpr (!__is_scalar() && !__is_fixed()) {
            return __to_intrin(_M_data._M_data);
        }
    }

    // }}}
    auto &builtin() {  // {{{
        if constexpr (__is_scalar() || __is_fixed()) {
            return _M_data;
        } else {
            return _M_data._M_data;
        }
    }
    const auto &builtin() const
    {
        if constexpr (__is_scalar() || __is_fixed()) {
            return _M_data;
        } else {
            return _M_data._M_data;
        }
    }

    // }}}
    friend const auto &__data<_T, abi_type>(const simd_mask &);
    friend auto &__data<_T, abi_type>(simd_mask &);
    alignas(__traits::__mask_member_alignment) __member_type _M_data;
};

// }}}

// __data(simd_mask) {{{
template <class _T, class _A>
_GLIBCXX_SIMD_INTRINSIC constexpr const auto &__data(const simd_mask<_T, _A> &__x)
{
    return __x._M_data;
}
template <class _T, class _A> _GLIBCXX_SIMD_INTRINSIC constexpr auto &__data(simd_mask<_T, _A> &__x)
{
    return __x._M_data;
}
// }}}
// __all_of {{{
template <class _T, class _Abi, class _Data> _GLIBCXX_SIMD_INTRINSIC bool __all_of(const _Data &__k)
{
    // _Data = decltype(__data(simd_mask))
    if constexpr (__is_abi<_Abi, simd_abi::scalar>()) {
        return __k;
    } else if constexpr (__is_abi<_Abi, simd_abi::fixed_size>()) {
        return __k.all();
    } else if constexpr (__is_combined_abi<_Abi>()) {
        for (int __i = 0; __i < _Abi::factor; ++__i) {
            if (!__all_of<_T, typename _Abi::member_abi>(__k[__i])) {
                return false;
            }
        }
        return true;
    } else if constexpr (__is_abi<_Abi, simd_abi::__sse_abi>() ||
                         __is_abi<_Abi, simd_abi::__avx_abi>()) {
        constexpr size_t _N = simd_size_v<_T, _Abi>;
        if constexpr (__have_sse4_1) {
            constexpr auto b =
                reinterpret_cast<__intrinsic_type_t<_T, _N>>(_Abi::template implicit_mask<_T>);
            if constexpr (std::is_same_v<_T, float> && _N > 4) {
                return 0 != _mm256_testc_ps(__to_intrin(__k), b);
            } else if constexpr (std::is_same_v<_T, float> && __have_avx) {
                return 0 != _mm_testc_ps(__to_intrin(__k), b);
            } else if constexpr (std::is_same_v<_T, float> ) {
                return 0 != _mm_testc_si128(_mm_castps_si128(__to_intrin(__k)), _mm_castps_si128(b));
            } else if constexpr (std::is_same_v<_T, double> && _N > 2) {
                return 0 != _mm256_testc_pd(__to_intrin(__k), b);
            } else if constexpr (std::is_same_v<_T, double> && __have_avx) {
                return 0 != _mm_testc_pd(__to_intrin(__k), b);
            } else if constexpr (std::is_same_v<_T, double>) {
                return 0 != _mm_testc_si128(_mm_castpd_si128(__to_intrin(__k)), _mm_castpd_si128(b));
            } else if constexpr (sizeof(b) == 32) {
                return _mm256_testc_si256(__to_intrin(__k), b);
            } else {
                return _mm_testc_si128(__to_intrin(__k), b);
            }
        } else if constexpr (std::is_same_v<_T, float>) {
            return (_mm_movemask_ps(__to_intrin(__k)) & ((1 << _N) - 1)) == (1 << _N) - 1;
        } else if constexpr (std::is_same_v<_T, double>) {
            return (_mm_movemask_pd(__to_intrin(__k)) & ((1 << _N) - 1)) == (1 << _N) - 1;
        } else {
            return (_mm_movemask_epi8(__to_intrin(__k)) & ((1 << (_N * sizeof(_T))) - 1)) ==
                   (1 << (_N * sizeof(_T))) - 1;
        }
    } else if constexpr (__is_abi<_Abi, simd_abi::__avx512_abi>()) {
        constexpr auto Mask = _Abi::template implicit_mask<_T>;
        if constexpr (std::is_same_v<_Data, __storage<bool, 8>>) {
            if constexpr (__have_avx512dq) {
                return _kortestc_mask8_u8(__k._M_data, Mask == 0xff ? __k._M_data : __mmask8(~Mask));
            } else {
                return __k._M_data == Mask;
            }
        } else if constexpr (std::is_same_v<_Data, __storage<bool, 16>>) {
            return _kortestc_mask16_u8(__k._M_data, Mask == 0xffff ? __k._M_data : __mmask16(~Mask));
        } else if constexpr (std::is_same_v<_Data, __storage<bool, 32>>) {
            if constexpr (__have_avx512bw) {
#ifdef _GLIBCXX_SIMD_WORKAROUND_PR85538
                return __k._M_data == Mask;
#else
                return _kortestc_mask32_u8(__k._M_data, Mask == 0xffffffffU ? __k._M_data : __mmask32(~Mask));
#endif
            }
        } else if constexpr (std::is_same_v<_Data, __storage<bool, 64>>) {
            if constexpr (__have_avx512bw) {
#ifdef _GLIBCXX_SIMD_WORKAROUND_PR85538
                return __k._M_data == Mask;
#else
                return _kortestc_mask64_u8(
                    __k._M_data, Mask == 0xffffffffffffffffULL ? __k._M_data : __mmask64(~Mask));
#endif
            }
        }
    } else {
        __assert_unreachable<_T>();
    }
}

// }}}
// __any_of {{{
template <class _T, class _Abi, class _Data> _GLIBCXX_SIMD_INTRINSIC bool __any_of(const _Data &__k)
{
    if constexpr (__is_abi<_Abi, simd_abi::scalar>()) {
        return __k;
    } else if constexpr (__is_abi<_Abi, simd_abi::fixed_size>()) {
        return __k.any();
    } else if constexpr (__is_combined_abi<_Abi>()) {
        for (int __i = 0; __i < _Abi::factor; ++__i) {
            if (__any_of<_T, typename _Abi::member_abi>(__k[__i])) {
                return true;
            }
        }
        return false;
    } else if constexpr (__is_abi<_Abi, simd_abi::__sse_abi>() ||
                         __is_abi<_Abi, simd_abi::__avx_abi>()) {
        constexpr size_t _N = simd_size_v<_T, _Abi>;
        if constexpr (__have_sse4_1) {
            return 0 == __testz(__k._M_data, _Abi::template implicit_mask<_T>);
        } else if constexpr (std::is_same_v<_T, float>) {
            return (_mm_movemask_ps(__to_intrin(__k)) & ((1 << _N) - 1)) != 0;
        } else if constexpr (std::is_same_v<_T, double>) {
            return (_mm_movemask_pd(__to_intrin(__k)) & ((1 << _N) - 1)) != 0;
        } else {
            return (_mm_movemask_epi8(__to_intrin(__k)) & ((1 << (_N * sizeof(_T))) - 1)) != 0;
        }
    } else if constexpr (__is_abi<_Abi, simd_abi::__avx512_abi>()) {
        return (__k & _Abi::template implicit_mask<_T>) != 0;
    } else {
        __assert_unreachable<_T>();
    }
}

// }}}
// __none_of {{{
template <class _T, class _Abi, class _Data> _GLIBCXX_SIMD_INTRINSIC bool __none_of(const _Data &__k)
{
    if constexpr (__is_abi<_Abi, simd_abi::scalar>()) {
        return !__k;
    } else if constexpr (__is_abi<_Abi, simd_abi::fixed_size>()) {
        return __k.none();
    } else if constexpr (__is_combined_abi<_Abi>()) {
        for (int __i = 0; __i < _Abi::factor; ++__i) {
            if (__any_of<_T, typename _Abi::member_abi>(__k[__i])) {
                return false;
            }
        }
        return true;
    } else if constexpr (__is_abi<_Abi, simd_abi::__sse_abi>() ||
                         __is_abi<_Abi, simd_abi::__avx_abi>()) {
        constexpr size_t _N = simd_size_v<_T, _Abi>;
        if constexpr (__have_sse4_1) {
            return 0 != __testz(__k._M_data, _Abi::template implicit_mask<_T>);
        } else if constexpr (std::is_same_v<_T, float>) {
            return (_mm_movemask_ps(__to_intrin(__k)) & ((1 << _N) - 1)) == 0;
        } else if constexpr (std::is_same_v<_T, double>) {
            return (_mm_movemask_pd(__to_intrin(__k)) & ((1 << _N) - 1)) == 0;
        } else {
            return (_mm_movemask_epi8(__to_intrin(__k)) & ((1 << (_N * sizeof(_T))) - 1)) == 0;
        }
    } else if constexpr (__is_abi<_Abi, simd_abi::__avx512_abi>()) {
        return (__k & _Abi::template implicit_mask<_T>) == 0;
    } else {
        __assert_unreachable<_T>();
    }
}

// }}}
// __some_of {{{
template <class _T, class _Abi, class _Data> _GLIBCXX_SIMD_INTRINSIC bool __some_of(const _Data &__k)
{
    if constexpr (__is_abi<_Abi, simd_abi::scalar>()) {
        return false;
    } else if constexpr (__is_abi<_Abi, simd_abi::fixed_size>()) {
        return __k.any() && !__k.all();
    } else if constexpr (__is_combined_abi<_Abi>()) {
        return __any_of<_T, _Abi>(__k) && !__all_of<_T, _Abi>(__k);
    } else if constexpr (__is_abi<_Abi, simd_abi::__sse_abi>() ||
                         __is_abi<_Abi, simd_abi::__avx_abi>()) {
        constexpr size_t _N = simd_size_v<_T, _Abi>;
        if constexpr (__have_sse4_1) {
            return 0 != __testnzc(__k._M_data, _Abi::template implicit_mask<_T>);
        } else if constexpr (std::is_same_v<_T, float>) {
            constexpr int __allbits = (1 << _N) - 1;
            const auto __tmp = _mm_movemask_ps(__to_intrin(__k)) & __allbits;
            return __tmp > 0 && __tmp < __allbits;
        } else if constexpr (std::is_same_v<_T, double>) {
            constexpr int __allbits = (1 << _N) - 1;
            const auto __tmp = _mm_movemask_pd(__to_intrin(__k)) & __allbits;
            return __tmp > 0 && __tmp < __allbits;
        } else {
            constexpr int __allbits = (1 << (_N * sizeof(_T))) - 1;
            const auto __tmp = _mm_movemask_epi8(__to_intrin(__k)) & __allbits;
            return __tmp > 0 && __tmp < __allbits;
        }
    } else if constexpr (__is_abi<_Abi, simd_abi::__avx512_abi>()) {
        return __any_of<_T, _Abi>(__k) && !__all_of<_T, _Abi>(__k);
    } else {
        __assert_unreachable<_T>();
    }
}

// }}}
// __popcount {{{
template <class _T, class _Abi, class _Data> _GLIBCXX_SIMD_INTRINSIC int __popcount(const _Data &__k)
{
    if constexpr (__is_abi<_Abi, simd_abi::scalar>()) {
        return __k;
    } else if constexpr (__is_abi<_Abi, simd_abi::fixed_size>()) {
        return __k.count();
    } else if constexpr (__is_combined_abi<_Abi>()) {
        int count = __popcount<_T, typename _Abi::member_abi>(__k[0]);
        for (int __i = 1; __i < _Abi::factor; ++__i) {
            count += __popcount<_T, typename _Abi::member_abi>(__k[__i]);
        }
        return count;
    } else if constexpr (__is_abi<_Abi, simd_abi::__sse_abi>() ||
                         __is_abi<_Abi, simd_abi::__avx_abi>()) {
        constexpr size_t _N = simd_size_v<_T, _Abi>;
        const auto __kk = _Abi::masked(__k._M_data);
        if constexpr (__have_popcnt) {
            int bits = __movemask(__to_intrin(__vector_bitcast<_T>(__kk)));
            const int count = __builtin_popcount(bits);
            return std::is_integral_v<_T> ? count / sizeof(_T) : count;
        } else if constexpr (_N == 2) {
            const int mask = _mm_movemask_pd(__auto_bitcast(__kk));
            return mask - (mask >> 1);
        } else if constexpr (_N == 4 && sizeof(__kk) == 16 && __have_sse2) {
            auto __x = __vector_bitcast<__llong>(__kk);
            __x = _mm_add_epi32(__x, _mm_shuffle_epi32(__x, _MM_SHUFFLE(0, 1, 2, 3)));
            __x = _mm_add_epi32(__x, _mm_shufflelo_epi16(__x, _MM_SHUFFLE(1, 0, 3, 2)));
            return -_mm_cvtsi128_si32(__x);
        } else if constexpr (_N == 4 && sizeof(__kk) == 16) {
            return __builtin_popcount(_mm_movemask_ps(__auto_bitcast(__kk)));
        } else if constexpr (_N == 8 && sizeof(__kk) == 16) {
            auto __x = __vector_bitcast<__llong>(__kk);
            __x = _mm_add_epi16(__x, _mm_shuffle_epi32(__x, _MM_SHUFFLE(0, 1, 2, 3)));
            __x = _mm_add_epi16(__x, _mm_shufflelo_epi16(__x, _MM_SHUFFLE(0, 1, 2, 3)));
            __x = _mm_add_epi16(__x, _mm_shufflelo_epi16(__x, _MM_SHUFFLE(2, 3, 0, 1)));
            return -short(_mm_extract_epi16(__x, 0));
        } else if constexpr (_N == 16 && sizeof(__kk) == 16) {
            auto __x = __vector_bitcast<__llong>(__kk);
            __x = _mm_add_epi8(__x, _mm_shuffle_epi32(__x, _MM_SHUFFLE(0, 1, 2, 3)));
            __x = _mm_add_epi8(__x, _mm_shufflelo_epi16(__x, _MM_SHUFFLE(0, 1, 2, 3)));
            __x = _mm_add_epi8(__x, _mm_shufflelo_epi16(__x, _MM_SHUFFLE(2, 3, 0, 1)));
            auto __y = -__vector_bitcast<__uchar>(__x);
            if constexpr (__have_sse4_1) {
                return __y[0] + __y[1];
            } else {
                unsigned __z = _mm_extract_epi16(__vector_bitcast<__llong>(__y), 0);
                return (__z & 0xff) + (__z >> 8);
            }
        } else if constexpr (_N == 4 && sizeof(__kk) == 32) {
            auto __x = -(__lo128(__kk) + __hi128(__kk));
            return __x[0] + __x[1];
        } else if constexpr (sizeof(__kk) == 32) {
            return __popcount<_T, simd_abi::__sse>(-(__lo128(__kk) + __hi128(__kk)));
        } else {
            __assert_unreachable<_T>();
        }
    } else if constexpr (__is_abi<_Abi, simd_abi::__avx512_abi>()) {
        constexpr size_t _N = simd_size_v<_T, _Abi>;
        const auto __kk = _Abi::masked(__k._M_data);
        if constexpr (_N <= 4) {
            return __builtin_popcount(__kk);
        } else if constexpr (_N <= 8) {
            return __builtin_popcount(__kk);
        } else if constexpr (_N <= 16) {
            return __builtin_popcount(__kk);
        } else if constexpr (_N <= 32) {
            return __builtin_popcount(__kk);
        } else if constexpr (_N <= 64) {
            return __builtin_popcountll(__kk);
        } else {
            __assert_unreachable<_T>();
        }
    } else {
        __assert_unreachable<_T>();
    }
}

// }}}
// __find_first_set {{{
template <class _T, class _Abi, class _Data> _GLIBCXX_SIMD_INTRINSIC int __find_first_set(const _Data &__k)
{
    if constexpr (__is_abi<_Abi, simd_abi::scalar>()) {
        return 0;
    } else if constexpr (__is_abi<_Abi, simd_abi::fixed_size>()) {
        return __firstbit(__k.to_ullong());
    } else if constexpr (__is_combined_abi<_Abi>()) {
        using _A2 = typename _Abi::member_abi;
        for (int __i = 0; __i < _Abi::factor - 1; ++__i) {
            if (__any_of<_T, _A2>(__k[__i])) {
                return __i * simd_size_v<_T, _A2> + __find_first_set(__k[__i]);
            }
        }
        return (_Abi::factor - 1) * simd_size_v<_T, _A2> +
               __find_first_set(__k[_Abi::factor - 1]);
    } else if constexpr (__is_abi<_Abi, simd_abi::__sse_abi>() ||
                         __is_abi<_Abi, simd_abi::__avx_abi>()) {
        return __firstbit(__vector_to_bitset(__k._M_data).to_ullong());
    } else if constexpr (__is_abi<_Abi, simd_abi::__avx512_abi>()) {
        if constexpr (simd_size_v<_T, _Abi> <= 32) {
            return _tzcnt_u32(__k._M_data);
        } else {
            return __firstbit(__k._M_data);
        }
    } else {
        __assert_unreachable<_T>();
    }
}

// }}}
// __find_last_set {{{
template <class _T, class _Abi, class _Data> _GLIBCXX_SIMD_INTRINSIC int __find_last_set(const _Data &__k)
{
    if constexpr (__is_abi<_Abi, simd_abi::scalar>()) {
        return 0;
    } else if constexpr (__is_abi<_Abi, simd_abi::fixed_size>()) {
        return __lastbit(__k.to_ullong());
    } else if constexpr (__is_combined_abi<_Abi>()) {
        using _A2 = typename _Abi::member_abi;
        for (int __i = 0; __i < _Abi::factor - 1; ++__i) {
            if (__any_of<_T, _A2>(__k[__i])) {
                return __i * simd_size_v<_T, _A2> + __find_last_set(__k[__i]);
            }
        }
        return (_Abi::factor - 1) * simd_size_v<_T, _A2> + __find_last_set(__k[_Abi::factor - 1]);
    } else if constexpr (__is_abi<_Abi, simd_abi::__sse_abi>() ||
                         __is_abi<_Abi, simd_abi::__avx_abi>()) {
        return __lastbit(__vector_to_bitset(__k._M_data).to_ullong());
    } else if constexpr (__is_abi<_Abi, simd_abi::__avx512_abi>()) {
        if constexpr (simd_size_v<_T, _Abi> <= 32) {
            return 31 - _lzcnt_u32(__k._M_data);
        } else {
            return __lastbit(__k._M_data);
        }
    } else {
        __assert_unreachable<_T>();
    }
}

// }}}

// reductions [simd_mask.reductions] {{{
template <class _T, class _Abi> _GLIBCXX_SIMD_ALWAYS_INLINE bool all_of(const simd_mask<_T, _Abi> &__k)
{
    return __all_of<_T, _Abi>(__data(__k));
}
template <class _T, class _Abi> _GLIBCXX_SIMD_ALWAYS_INLINE bool any_of(const simd_mask<_T, _Abi> &__k)
{
    return __any_of<_T, _Abi>(__data(__k));
}
template <class _T, class _Abi> _GLIBCXX_SIMD_ALWAYS_INLINE bool none_of(const simd_mask<_T, _Abi> &__k)
{
    return __none_of<_T, _Abi>(__data(__k));
}
template <class _T, class _Abi> _GLIBCXX_SIMD_ALWAYS_INLINE bool some_of(const simd_mask<_T, _Abi> &__k)
{
    return __some_of<_T, _Abi>(__data(__k));
}
template <class _T, class _Abi> _GLIBCXX_SIMD_ALWAYS_INLINE int popcount(const simd_mask<_T, _Abi> &__k)
{
    return __popcount<_T, _Abi>(__data(__k));
}
template <class _T, class _Abi>
_GLIBCXX_SIMD_ALWAYS_INLINE int find_first_set(const simd_mask<_T, _Abi> &__k)
{
    return __find_first_set<_T, _Abi>(__data(__k));
}
template <class _T, class _Abi>
_GLIBCXX_SIMD_ALWAYS_INLINE int find_last_set(const simd_mask<_T, _Abi> &__k)
{
    return __find_last_set<_T, _Abi>(__data(__k));
}

constexpr bool all_of(__exact_bool __x) { return __x; }
constexpr bool any_of(__exact_bool __x) { return __x; }
constexpr bool none_of(__exact_bool __x) { return !__x; }
constexpr bool some_of(__exact_bool) { return false; }
constexpr int popcount(__exact_bool __x) { return __x; }
constexpr int find_first_set(__exact_bool) { return 0; }
constexpr int find_last_set(__exact_bool) { return 0; }

// }}}

template <class _Abi> struct __generic_simd_impl;
// __allow_conversion_ctor2{{{1
template <class _T0, class _T1, class _A, bool BothIntegral> struct __allow_conversion_ctor2_1;

template <class _T0, class _T1, class _A>
struct __allow_conversion_ctor2
    : public __allow_conversion_ctor2_1<
          _T0, _T1, _A, conjunction<std::is_integral<_T0>, std::is_integral<_T1>>::value> {
};

// disallow 2nd conversion ctor (equal _Abi), if the value_types are equal (copy ctor)
template <class _T, class _A> struct __allow_conversion_ctor2<_T, _T, _A> : public false_type {};

// disallow 2nd conversion ctor (equal _Abi), if the _Abi is a fixed_size instance
template <class _T0, class _T1, int _N>
struct __allow_conversion_ctor2<_T0, _T1, simd_abi::fixed_size<_N>> : public false_type {};

// disallow 2nd conversion ctor (equal _Abi), if both of the above are true
template <class _T, int _N>
struct __allow_conversion_ctor2<_T, _T, simd_abi::fixed_size<_N>> : public false_type {};

// disallow 2nd conversion ctor (equal _Abi), the integers only differ in sign
template <class _T0, class _T1, class _A>
struct __allow_conversion_ctor2_1<_T0, _T1, _A, true>
    : public std::is_same<std::make_signed_t<_T0>, std::make_signed_t<_T1>> {
};

// disallow 2nd conversion ctor (equal _Abi), any value_type is not integral
template <class _T0, class _T1, class _A>
struct __allow_conversion_ctor2_1<_T0, _T1, _A, false> : public false_type {
};

// __allow_conversion_ctor3{{{1
template <class _T0, class _A0, class _T1, class _A1, bool = std::is_same<_A0, _A1>::value>
struct __allow_conversion_ctor3 : public false_type {
    // disallow 3rd conversion ctor if _A0 is not fixed_size<simd_size_v<_T1, _A1>>
};

template <class _T0, class _T1, class _A1>
struct __allow_conversion_ctor3<_T0, simd_abi::fixed_size<simd_size_v<_T1, _A1>>, _T1, _A1,
                              false  // disallow 3rd conversion ctor if the _Abi types are
                                     // equal (disambiguate copy ctor and the two
                                     // preceding conversion ctors)
                              > : public std::is_convertible<_T1, _T0> {
};

// __simd_int_operators{{{1
template <class _V, bool> class __simd_int_operators {};
template <class _V> class __simd_int_operators<_V, true>
{
    using __impl = __get_impl_t<_V>;

    _GLIBCXX_SIMD_INTRINSIC const _V &__derived() const { return *static_cast<const _V *>(this); }

    template <class _T> _GLIBCXX_SIMD_INTRINSIC static _V __make_derived(_T &&__d)
    {
        return {__private_init, std::forward<_T>(__d)};
    }

public:
    friend _V &operator %=(_V &lhs, const _V &__x) { return lhs = lhs  % __x; }
    friend _V &operator &=(_V &lhs, const _V &__x) { return lhs = lhs  & __x; }
    friend _V &operator |=(_V &lhs, const _V &__x) { return lhs = lhs  | __x; }
    friend _V &operator ^=(_V &lhs, const _V &__x) { return lhs = lhs  ^ __x; }
    friend _V &operator<<=(_V &lhs, const _V &__x) { return lhs = lhs << __x; }
    friend _V &operator>>=(_V &lhs, const _V &__x) { return lhs = lhs >> __x; }
    friend _V &operator<<=(_V &lhs, int __x) { return lhs = lhs << __x; }
    friend _V &operator>>=(_V &lhs, int __x) { return lhs = lhs >> __x; }

    friend _V operator% (const _V &__x, const _V &__y) { return __simd_int_operators::__make_derived(__impl::modulus        (__data(__x), __data(__y))); }
    friend _V operator& (const _V &__x, const _V &__y) { return __simd_int_operators::__make_derived(__impl::bit_and        (__data(__x), __data(__y))); }
    friend _V operator| (const _V &__x, const _V &__y) { return __simd_int_operators::__make_derived(__impl::bit_or         (__data(__x), __data(__y))); }
    friend _V operator^ (const _V &__x, const _V &__y) { return __simd_int_operators::__make_derived(__impl::bit_xor        (__data(__x), __data(__y))); }
    friend _V operator<<(const _V &__x, const _V &__y) { return __simd_int_operators::__make_derived(__impl::bit_shift_left (__data(__x), __data(__y))); }
    friend _V operator>>(const _V &__x, const _V &__y) { return __simd_int_operators::__make_derived(__impl::bit_shift_right(__data(__x), __data(__y))); }
    friend _V operator<<(const _V &__x, int __y)      { return __simd_int_operators::__make_derived(__impl::bit_shift_left (__data(__x), __y)); }
    friend _V operator>>(const _V &__x, int __y)      { return __simd_int_operators::__make_derived(__impl::bit_shift_right(__data(__x), __y)); }

    // unary operators (for integral _T)
    _V operator~() const
    {
        return {__private_init, __impl::complement(__derived()._M_data)};
    }
};

//}}}1

// simd {{{
template <class _T, class _Abi>
class simd
    : public __simd_int_operators<
          simd<_T, _Abi>, conjunction<std::is_integral<_T>,
                                    typename __simd_traits<_T, _Abi>::is_valid>::value>,
      public __simd_traits<_T, _Abi>::__simd_base
{
    using __traits = __simd_traits<_T, _Abi>;
    using __impl = typename __traits::__simd_impl_type;
    using __member_type = typename __traits::__simd_member_type;
    using __cast_type = typename __traits::__simd_cast_type;
    static constexpr _T *__type_tag = nullptr;
    friend typename __traits::__simd_base;
    friend __impl;
    friend __generic_simd_impl<_Abi>;
    friend __simd_int_operators<simd, true>;

public:
    using value_type = _T;
    using reference = __smart_reference<__member_type, __impl, value_type>;
    using mask_type = simd_mask<_T, _Abi>;
    using abi_type = _Abi;

    static constexpr size_t size() { return __size_or_zero<_T, _Abi>; }
    simd() = default;
    simd(const simd &) = default;
    simd(simd &&) = default;
    simd &operator=(const simd &) = default;
    simd &operator=(simd &&) = default;

    // implicit broadcast constructor
    template <class _U, class = __value_preserving_or_int<_U, value_type>>
    _GLIBCXX_SIMD_ALWAYS_INLINE constexpr simd(_U &&__x)
        : _M_data(__impl::__broadcast(static_cast<value_type>(std::forward<_U>(__x))))
    {
    }

    // implicit type conversion constructor (convert from fixed_size to fixed_size)
    template <class _U>
    _GLIBCXX_SIMD_ALWAYS_INLINE simd(
        const simd<_U, simd_abi::fixed_size<size()>> &__x,
        enable_if_t<
            conjunction<std::is_same<simd_abi::fixed_size<size()>, abi_type>,
                        std::negation<__is_narrowing_conversion<_U, value_type>>,
                        __converts_to_higher_integer_rank<_U, value_type>>::value,
            void *> = nullptr)
        : simd{static_cast<std::array<_U, size()>>(__x).data(), vector_aligned}
    {
    }

    // generator constructor
    template <class _F>
    _GLIBCXX_SIMD_ALWAYS_INLINE explicit constexpr simd(
        _F &&gen,
        __value_preserving_or_int<
            decltype(std::declval<_F>()(std::declval<__size_constant<0> &>())),
            value_type> * = nullptr)
        : _M_data(__impl::generator(std::forward<_F>(gen), __type_tag))
    {
    }

    // load constructor
    template <class _U, class _Flags>
    _GLIBCXX_SIMD_ALWAYS_INLINE simd(const _U *mem, _Flags f)
        : _M_data(__impl::load(mem, f, __type_tag))
    {
    }

    // loads [simd.load]
    template <class _U, class _Flags>
    _GLIBCXX_SIMD_ALWAYS_INLINE void copy_from(const _Vectorizable<_U> *mem, _Flags f)
    {
        _M_data = static_cast<decltype(_M_data)>(__impl::load(mem, f, __type_tag));
    }

    // stores [simd.store]
    template <class _U, class _Flags>
    _GLIBCXX_SIMD_ALWAYS_INLINE void copy_to(_Vectorizable<_U> *mem, _Flags f) const
    {
        __impl::store(_M_data, mem, f, __type_tag);
    }

    // scalar access
    _GLIBCXX_SIMD_ALWAYS_INLINE constexpr reference operator[](size_t __i) { return {_M_data, int(__i)}; }
    _GLIBCXX_SIMD_ALWAYS_INLINE constexpr value_type operator[](size_t __i) const
    {
        if constexpr (__is_scalar()) {
            _GLIBCXX_SIMD_ASSERT(__i == 0);
            __unused(__i);
            return _M_data;
        } else {
            return _M_data[__i];
        }
    }

    // increment and decrement:
    _GLIBCXX_SIMD_ALWAYS_INLINE simd &operator++() { __impl::__increment(_M_data); return *this; }
    _GLIBCXX_SIMD_ALWAYS_INLINE simd operator++(int) { simd __r = *this; __impl::__increment(_M_data); return __r; }
    _GLIBCXX_SIMD_ALWAYS_INLINE simd &operator--() { __impl::__decrement(_M_data); return *this; }
    _GLIBCXX_SIMD_ALWAYS_INLINE simd operator--(int) { simd __r = *this; __impl::__decrement(_M_data); return __r; }

    // unary operators (for any _T)
    _GLIBCXX_SIMD_ALWAYS_INLINE mask_type operator!() const
    {
        return {__private_init, __impl::negate(_M_data)};
    }
    _GLIBCXX_SIMD_ALWAYS_INLINE simd operator+() const { return *this; }
    _GLIBCXX_SIMD_ALWAYS_INLINE simd operator-() const
    {
        return {__private_init, __impl::unary_minus(_M_data)};
    }

    // access to internal representation (suggested extension)
    _GLIBCXX_SIMD_ALWAYS_INLINE explicit simd(__cast_type __init) : _M_data(__init) {}

    // compound assignment [simd.cassign]
    _GLIBCXX_SIMD_ALWAYS_INLINE friend simd &operator+=(simd &lhs, const simd &__x) { return lhs = lhs + __x; }
    _GLIBCXX_SIMD_ALWAYS_INLINE friend simd &operator-=(simd &lhs, const simd &__x) { return lhs = lhs - __x; }
    _GLIBCXX_SIMD_ALWAYS_INLINE friend simd &operator*=(simd &lhs, const simd &__x) { return lhs = lhs * __x; }
    _GLIBCXX_SIMD_ALWAYS_INLINE friend simd &operator/=(simd &lhs, const simd &__x) { return lhs = lhs / __x; }

    // binary operators [simd.binary]
    _GLIBCXX_SIMD_ALWAYS_INLINE friend simd operator+(const simd &__x, const simd &__y)
    {
        return {__private_init, __impl::plus(__x._M_data, __y._M_data)};
    }
    _GLIBCXX_SIMD_ALWAYS_INLINE friend simd operator-(const simd &__x, const simd &__y)
    {
        return {__private_init, __impl::minus(__x._M_data, __y._M_data)};
    }
    _GLIBCXX_SIMD_ALWAYS_INLINE friend simd operator*(const simd &__x, const simd &__y)
    {
        return {__private_init, __impl::multiplies(__x._M_data, __y._M_data)};
    }
    _GLIBCXX_SIMD_ALWAYS_INLINE friend simd operator/(const simd &__x, const simd &__y)
    {
        return {__private_init, __impl::divides(__x._M_data, __y._M_data)};
    }

    // compares [simd.comparison]
    _GLIBCXX_SIMD_ALWAYS_INLINE friend mask_type operator==(const simd &__x, const simd &__y)
    {
        return simd::make_mask(__impl::equal_to(__x._M_data, __y._M_data));
    }
    _GLIBCXX_SIMD_ALWAYS_INLINE friend mask_type operator!=(const simd &__x, const simd &__y)
    {
        return simd::make_mask(__impl::not_equal_to(__x._M_data, __y._M_data));
    }
    _GLIBCXX_SIMD_ALWAYS_INLINE friend mask_type operator<(const simd &__x, const simd &__y)
    {
        return simd::make_mask(__impl::less(__x._M_data, __y._M_data));
    }
    _GLIBCXX_SIMD_ALWAYS_INLINE friend mask_type operator<=(const simd &__x, const simd &__y)
    {
        return simd::make_mask(__impl::less_equal(__x._M_data, __y._M_data));
    }
    _GLIBCXX_SIMD_ALWAYS_INLINE friend mask_type operator>(const simd &__x, const simd &__y)
    {
        return simd::make_mask(__impl::less(__y._M_data, __x._M_data));
    }
    _GLIBCXX_SIMD_ALWAYS_INLINE friend mask_type operator>=(const simd &__x, const simd &__y)
    {
        return simd::make_mask(__impl::less_equal(__y._M_data, __x._M_data));
    }

    // "private" because of the first arguments's namespace
    _GLIBCXX_SIMD_INTRINSIC simd(__private_init_t, const __member_type &__init) : _M_data(__init) {}

    // "private" because of the first arguments's namespace
    _GLIBCXX_SIMD_INTRINSIC simd(__bitset_init_t, std::bitset<size()> __init) : _M_data() {
        where(mask_type(__bitset_init, __init), *this) = ~*this;
    }

private:
    static constexpr bool __is_scalar() { return std::is_same_v<abi_type, simd_abi::scalar>; }
    static constexpr bool __is_fixed() { return __is_fixed_size_abi_v<abi_type>; }

    _GLIBCXX_SIMD_INTRINSIC static mask_type make_mask(typename mask_type::__member_type __k)
    {
        return {__private_init, __k};
    }
    friend const auto &__data<value_type, abi_type>(const simd &);
    friend auto &__data<value_type, abi_type>(simd &);
    alignas(__traits::__simd_member_alignment) __member_type _M_data;
};

// }}}
// __data {{{
template <class _T, class _A> _GLIBCXX_SIMD_INTRINSIC constexpr const auto &__data(const simd<_T, _A> &__x)
{
    return __x._M_data;
}
template <class _T, class _A> _GLIBCXX_SIMD_INTRINSIC constexpr auto &__data(simd<_T, _A> &__x)
{
    return __x._M_data;
}
// }}}

namespace __proposed
{
namespace float_bitwise_operators
{
// float_bitwise_operators {{{
template <class _T, class _A>
_GLIBCXX_SIMD_INTRINSIC simd<_T, _A> operator|(const simd<_T, _A> &a, const simd<_T, _A> &b)
{
    return {__private_init, __get_impl_t<simd<_T, _A>>::bit_or(__data(a), __data(b))};
}

template <class _T, class _A>
_GLIBCXX_SIMD_INTRINSIC simd<_T, _A> operator&(const simd<_T, _A> &a, const simd<_T, _A> &b)
{
    return {__private_init, __get_impl_t<simd<_T, _A>>::bit_and(__data(a), __data(b))};
}
// }}}
}  // namespace float_bitwise_operators
}  // namespace __proposed

_GLIBCXX_SIMD_END_NAMESPACE

#endif  // __cplusplus >= 201703L
#endif  // _GLIBCXX_EXPERIMENTAL_SIMD_H
// vim: foldmethod=marker
