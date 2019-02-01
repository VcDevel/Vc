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
#if _GLIBCXX_SIMD_HAVE_SSE || _GLIBCXX_SIMD_HAVE_MMX
#include <x86intrin.h>
#endif  // _GLIBCXX_SIMD_HAVE_SSE

_GLIBCXX_SIMD_BEGIN_NAMESPACE
// load/store flags {{{
struct element_aligned_tag {};
struct vector_aligned_tag {};
template <size_t _N>
struct overaligned_tag
{
  static constexpr size_t _S_alignment = _N;
};
inline constexpr element_aligned_tag element_aligned = {};
inline constexpr vector_aligned_tag  vector_aligned  = {};
template <size_t _N>
inline constexpr overaligned_tag<_N> overaligned = {};
// }}}

// vvv ---- type traits ---- vvv
// integer type aliases{{{
using _UChar = unsigned char;
using _SChar = signed char;
using _UShort = unsigned short;
using _UInt = unsigned int;
using _ULong = unsigned long;
using _ULLong = unsigned long long;
using _LLong = long long;
//}}}
// __is_equal {{{
template <typename _Tp, _Tp __a, _Tp __b>
struct __is_equal : public false_type
{
};
template <typename _Tp, _Tp __a>
struct __is_equal<_Tp, __a, __a> : public true_type
{
};

// }}}
// __identity/__id{{{
template <typename _Tp>
struct __identity
{
  using type = _Tp;
};
template <typename _Tp>
using __id = typename __identity<_Tp>::type;

// }}}
// __first_of_pack{{{
template <typename _T0, typename...>
struct __first_of_pack
{
  using type = _T0;
};
template <typename... _Ts>
using __first_of_pack_t = typename __first_of_pack<_Ts...>::type;

//}}}
// __value_type_or_identity_t {{{
template <typename _Tp>
typename _Tp::value_type __value_type_or_identity_impl(int);
template <typename _Tp>
_Tp __value_type_or_identity_impl(float);
template <typename _Tp>
using __value_type_or_identity_t =
  decltype(__value_type_or_identity_impl<_Tp>(int()));

// }}}
// __is_vectorizable {{{
template <typename _Tp>
struct __is_vectorizable : public std::is_arithmetic<_Tp>
{
};
template <>
struct __is_vectorizable<bool> : public false_type
{
};
template <typename _Tp>
inline constexpr bool __is_vectorizable_v = __is_vectorizable<_Tp>::value;
// Deduces to a vectorizable type
template <typename _Tp, typename = enable_if_t<__is_vectorizable_v<_Tp>>>
using _Vectorizable = _Tp;

// }}}
// _LoadStorePtr / __is_possible_loadstore_conversion {{{
template <typename _Ptr, typename _ValueType>
struct __is_possible_loadstore_conversion
    : conjunction<__is_vectorizable<_Ptr>, __is_vectorizable<_ValueType>> {
};
template <> struct __is_possible_loadstore_conversion<bool, bool> : true_type {
};
// Deduces to a type allowed for load/store with the given value type.
template <typename _Ptr, typename _ValueType,
          typename = enable_if_t<__is_possible_loadstore_conversion<_Ptr, _ValueType>::value>>
using _LoadStorePtr = _Ptr;

// }}}
// _SizeConstant{{{
template <size_t _X> using _SizeConstant = integral_constant<size_t, _X>;
// }}}
// __is_bitmask{{{
template <typename _Tp, typename = std::void_t<>>
struct __is_bitmask : false_type
{
  constexpr __is_bitmask(const _Tp&) noexcept {}
};
template <typename _Tp>
inline constexpr bool __is_bitmask_v = __is_bitmask<_Tp>::value;

// the __mmaskXX case:
template <typename _Tp>
struct __is_bitmask<
  _Tp,
  std::void_t<decltype(std::declval<unsigned&>() = std::declval<_Tp>() & 1u)>>
: true_type
{
  constexpr __is_bitmask(const _Tp&) noexcept {}
};

// }}}
// __int_for_sizeof{{{
template <size_t> struct __int_for_sizeof;
template <> struct __int_for_sizeof<1> { using type = signed char; };
template <> struct __int_for_sizeof<2> { using type = signed short; };
template <> struct __int_for_sizeof<4> { using type = signed int; };
template <> struct __int_for_sizeof<8> { using type = signed long long; };
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
template <> struct __int_for_sizeof<16> { using type = __int128; };
#pragma GCC diagnostic pop
template <typename _Tp>
using __int_for_sizeof_t = typename __int_for_sizeof<sizeof(_Tp)>::type;
template <size_t _N>
using __int_with_sizeof_t = typename __int_for_sizeof<_N>::type;

// }}}
// __is_fixed_size_abi{{{
template <typename _Tp>
struct __is_fixed_size_abi : false_type
{
};
template <int _N>
struct __is_fixed_size_abi<simd_abi::fixed_size<_N>> : true_type
{
};

template <typename _Tp>
inline constexpr bool __is_fixed_size_abi_v = __is_fixed_size_abi<_Tp>::value;

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
// __is_abi {{{
template <template <int> class _Abi, int _Bytes>
constexpr int __abi_bytes_impl(_Abi<_Bytes>*)
{
    return _Bytes;
}
template <typename _Tp>
constexpr int __abi_bytes_impl(_Tp*)
{
  return -1;
}
template <typename _Abi>
inline constexpr int
  __abi_bytes_v = __abi_bytes_impl(static_cast<_Abi*>(nullptr));

template <typename _Abi0, typename _Abi1>
constexpr bool __is_abi()
{
  return std::is_same_v<_Abi0, _Abi1>;
}
template <template <int> class _Abi0, typename _Abi1>
constexpr bool __is_abi()
{
  return std::is_same_v<_Abi0<__abi_bytes_v<_Abi1>>, _Abi1>;
}
template <typename _Abi0, template <int> class _Abi1>
constexpr bool __is_abi()
{
  return std::is_same_v<_Abi1<__abi_bytes_v<_Abi0>>, _Abi0>;
}
template <template <int> class _Abi0, template <int> class _Abi1>
constexpr bool __is_abi()
{
  return std::is_same_v<_Abi0<0>, _Abi1<0>>;
}

// }}}
// __is_combined_abi{{{
template <template <int, typename> class _Combine, int _N, typename _Abi>
constexpr bool __is_combined_abi(_Combine<_N, _Abi>*)
{
  return std::is_same_v<_Combine<_N, _Abi>, simd_abi::__combine<_N, _Abi>>;
}
template <typename _Abi>
constexpr bool __is_combined_abi(_Abi*)
{
  return false;
}

template <typename _Abi>
constexpr bool __is_combined_abi()
{
  return __is_combined_abi(static_cast<_Abi*>(nullptr));
}

// }}}
// ^^^ ---- type traits ---- ^^^

// __unused{{{
template <typename _Tp>
static constexpr void __unused(_Tp&&)
{
}

// }}}
// __assert_unreachable{{{
template <typename _Tp>
struct __assert_unreachable
{
  static_assert(!std::is_same_v<_Tp, _Tp>, "this should be unreachable");
};

// }}}
// __size_or_zero_v {{{
template <typename _Tp, typename _A, size_t _N = simd_size<_Tp, _A>::value>
constexpr size_t __size_or_zero_dispatch(int)
{
  return _N;
}
template <typename _Tp, typename _A>
constexpr size_t __size_or_zero_dispatch(float)
{
  return 0;
}
template <typename _Tp, typename _A>
inline constexpr size_t __size_or_zero_v = __size_or_zero_dispatch<_Tp, _A>(0);

// }}}
// __promote_preserving_unsigned{{{
// work around crazy semantics of unsigned integers of lower rank than int:
// Before applying an operator the operands are promoted to int. In which case over- or
// underflow is UB, even though the operand types were unsigned.
template <typename _Tp>
_GLIBCXX_SIMD_INTRINSIC constexpr const _Tp&
  __promote_preserving_unsigned(const _Tp& __x)
{
  return __x;
}
_GLIBCXX_SIMD_INTRINSIC constexpr unsigned int
  __promote_preserving_unsigned(const unsigned char& __x)
{
  return __x;
}
_GLIBCXX_SIMD_INTRINSIC constexpr unsigned int
  __promote_preserving_unsigned(const unsigned short& __x)
{
  return __x;
}

// }}}
// _ExactBool{{{
class _ExactBool
{
  const bool _M_data;

public:
  _GLIBCXX_SIMD_INTRINSIC constexpr _ExactBool(bool __b)
  : _M_data(__b)
  {
  }
  _ExactBool(int) = delete;
  _GLIBCXX_SIMD_INTRINSIC constexpr operator bool() const { return _M_data; }
};

// }}}
// __execute_on_index_sequence(_with_return){{{
template <typename _F, size_t... _I>
_GLIBCXX_SIMD_INTRINSIC constexpr void
  __execute_on_index_sequence(_F&& __f, std::index_sequence<_I...>)
{
  auto&& __x = {(__f(_SizeConstant<_I>()), 0)...};
  __unused(__x);
}

template <typename _F>
_GLIBCXX_SIMD_INTRINSIC constexpr void
  __execute_on_index_sequence(_F&&, std::index_sequence<>)
{
}

template <typename _R, typename _F, size_t... _I>
_GLIBCXX_SIMD_INTRINSIC constexpr _R
  __execute_on_index_sequence_with_return(_F&& __f, std::index_sequence<_I...>)
{
  return _R{__f(_SizeConstant<_I>())...};
}

// }}}
// __execute_n_times{{{
template <size_t _N, typename _F>
_GLIBCXX_SIMD_INTRINSIC constexpr void __execute_n_times(_F&& __f)
{
  __execute_on_index_sequence(std::forward<_F>(__f),
			      std::make_index_sequence<_N>{});
}

// }}}
// __generate_from_n_evaluations{{{
template <size_t _N, typename _R, typename _F>
_GLIBCXX_SIMD_INTRINSIC constexpr _R __generate_from_n_evaluations(_F&& __f)
{
  return __execute_on_index_sequence_with_return<_R>(
    std::forward<_F>(__f), std::make_index_sequence<_N>{});
}

// }}}
// __may_alias{{{
/**\internal
 * Helper __may_alias<_Tp> that turns _Tp into the type to be used for an aliasing pointer. This
 * adds the __may_alias attribute to _Tp (with compilers that support it).
 */
template <typename _Tp> using __may_alias [[__gnu__::__may_alias__]] = _Tp;

// }}}
// _UnsupportedBase {{{
// simd and simd_mask base for unsupported <_Tp, _Abi>
struct _UnsupportedBase
{
  _UnsupportedBase()                        = delete;
  _UnsupportedBase(const _UnsupportedBase&) = delete;
  _UnsupportedBase& operator=(const _UnsupportedBase&) = delete;
  ~_UnsupportedBase()                                  = delete;
};

// }}}
// _InvalidTraits {{{
/**
 * \internal
 * Defines the implementation of __a given <_Tp, _Abi>.
 *
 * Implementations must ensure that only valid <_Tp, _Abi> instantiations are possible.
 * Static assertions in the type definition do not suffice. It is important that
 * SFINAE works.
 */
struct _InvalidTraits
{
  using _IsValid   = false_type;
  using _Simd_base = _UnsupportedBase;
  using _Mask_base = _UnsupportedBase;

  static constexpr size_t _S_simd_align = 1;
  struct _SimdImpl;
  struct _SimdMember {};
  struct _SimdCastType;

  static constexpr size_t _S_mask_align = 1;
  struct _MaskImpl;
  struct _MaskMember {};
  struct _MaskCastType;
};
// }}}
// _SimdTraits {{{
template <typename _Tp, typename _Abi, typename = std::void_t<>>
struct _SimdTraits : _InvalidTraits
{
};

// }}}
// __get_impl(_t){{{
template <typename _Tp> struct __get_impl;
template <typename _Tp>
using __get_impl_t = typename __get_impl<__remove_cvref_t<_Tp>>::type;

// }}}
// __get_traits(_t){{{
template <typename _Tp>
struct __get_traits;
template <typename _Tp>
using __get_traits_t = typename __get_traits<__remove_cvref_t<_Tp>>::type;

// }}}
// __next_power_of_2{{{
/**
 * \internal
 * Returns the next power of 2 larger than or equal to \p __x.
 */
constexpr std::size_t __next_power_of_2(std::size_t __x)
{
  return (__x & (__x - 1)) == 0 ? __x
				: __next_power_of_2((__x | (__x >> 1)) + 1);
}

// }}}
// __private_init, __bitset_init{{{
/**
 * \internal
 * Tag used for private init constructor of simd and simd_mask
 */
inline constexpr struct _PrivateInit {} __private_init = {};
inline constexpr struct _BitsetInit {} __bitset_init = {};

// }}}
// __is_narrowing_conversion<_From, _To>{{{
template <typename _From,
	  typename _To,
	  bool = std::is_arithmetic<_From>::value,
	  bool = std::is_arithmetic<_To>::value>
struct __is_narrowing_conversion;

// ignore "warning C4018: '<': signed/unsigned mismatch" in the following trait. The implicit
// conversions will do the right thing here.
template <typename _From, typename _To>
struct __is_narrowing_conversion<_From, _To, true, true>
: public __bool_constant<(
    std::numeric_limits<_From>::digits > std::numeric_limits<_To>::digits ||
    std::numeric_limits<_From>::max() > std::numeric_limits<_To>::max() ||
    std::numeric_limits<_From>::lowest() < std::numeric_limits<_To>::lowest() ||
    (std::is_signed<_From>::value && std::is_unsigned<_To>::value))>
{
};

template <typename _Tp>
struct __is_narrowing_conversion<bool, _Tp, true, true> : public true_type
{
};
template <>
struct __is_narrowing_conversion<bool, bool, true, true> : public false_type
{
};
template <typename _Tp>
struct __is_narrowing_conversion<_Tp, _Tp, true, true> : public false_type
{
};

template <typename _From, typename _To>
struct __is_narrowing_conversion<_From, _To, false, true>
: public negation<std::is_convertible<_From, _To>>
{
};

// }}}
// __converts_to_higher_integer_rank{{{
template <typename _From, typename _To, bool = (sizeof(_From) < sizeof(_To))>
struct __converts_to_higher_integer_rank : public true_type
{
};
template <typename _From, typename _To>
struct __converts_to_higher_integer_rank<_From, _To, false>
: public std::is_same<decltype(std::declval<_From>() + std::declval<_To>()),
		      _To>
{
};

// }}}
// __is_aligned(_v){{{
template <typename _Flag, size_t _Alignment>
struct __is_aligned;
template <size_t _Alignment>
struct __is_aligned<vector_aligned_tag, _Alignment> : public true_type
{
};
template <size_t _Alignment>
struct __is_aligned<element_aligned_tag, _Alignment> : public false_type
{
};
template <size_t _GivenAlignment, size_t _Alignment>
struct __is_aligned<overaligned_tag<_GivenAlignment>, _Alignment>
: public std::integral_constant<bool, (_GivenAlignment >= _Alignment)>
{
};
template <typename _Flag, size_t _Alignment>
inline constexpr bool __is_aligned_v = __is_aligned<_Flag, _Alignment>::value;

// }}}
// __data(simd/simd_mask) {{{
template <typename _Tp, typename _A>
_GLIBCXX_SIMD_INTRINSIC constexpr const auto& __data(const simd<_Tp, _A>& __x);
template <typename _Tp, typename _A>
_GLIBCXX_SIMD_INTRINSIC constexpr auto& __data(simd<_Tp, _A>& __x);

template <typename _Tp, typename _A>
_GLIBCXX_SIMD_INTRINSIC constexpr const auto&
  __data(const simd_mask<_Tp, _A>& __x);
template <typename _Tp, typename _A>
_GLIBCXX_SIMD_INTRINSIC constexpr auto& __data(simd_mask<_Tp, _A>& __x);

// }}}
// _SimdConverter {{{
template <typename _FromT, typename _FromA, typename _ToT, typename _ToA>
struct _SimdConverter;

template <typename _Tp, typename _A>
struct _SimdConverter<_Tp, _A, _Tp, _A>
{
  template <typename _U>
  _GLIBCXX_SIMD_INTRINSIC const _U& operator()(const _U& __x)
  {
    return __x;
  }
};

// }}}
// __to_value_type_or_member_type {{{
template <typename _V>
_GLIBCXX_SIMD_INTRINSIC constexpr auto
  __to_value_type_or_member_type(const _V& __x) -> decltype(__data(__x))
{
  return __data(__x);
}

template <typename _V>
_GLIBCXX_SIMD_INTRINSIC constexpr const typename _V::value_type&
  __to_value_type_or_member_type(const typename _V::value_type& __x)
{
  return __x;
}

// }}}
// __bool_storage_member_type{{{
template <size_t _Size>
struct __bool_storage_member_type;

template <size_t _Size>
using __bool_storage_member_type_t =
  typename __bool_storage_member_type<_Size>::type;

// }}}
// _SimdTuple {{{
// why not std::tuple?
// 1. std::tuple gives no guarantee about the storage order, but I require storage
//    equivalent to std::array<_Tp, _N>
// 2. direct access to the element type (first template argument)
// 3. enforces equal element type, only different _Abi types are allowed
template <typename _Tp, typename... _Abis> struct _SimdTuple;

//}}}
// __fixed_size_storage_t {{{
template <typename _Tp, int _N>
struct __fixed_size_storage;

template <typename _Tp, int _N>
using __fixed_size_storage_t = typename __fixed_size_storage<_Tp, _N>::type;

// }}}
// _SimdWrapper fwd decl{{{
template <typename _Tp, size_t _Size, typename = std::void_t<>>
struct _SimdWrapper;

template <typename _Tp>
using _SimdWrapper16 = _SimdWrapper<_Tp, 16 / sizeof(_Tp)>;
template <typename _Tp>
using _SimdWrapper32 = _SimdWrapper<_Tp, 32 / sizeof(_Tp)>;
template <typename _Tp>
using _SimdWrapper64 = _SimdWrapper<_Tp, 64 / sizeof(_Tp)>;

// }}}
// __bit_iteration{{{
constexpr _UInt   __popcount(_UInt __x) { return __builtin_popcount(__x); }
constexpr _ULong  __popcount(_ULong __x) { return __builtin_popcountl(__x); }
constexpr _ULLong __popcount(_ULLong __x) { return __builtin_popcountll(__x); }

constexpr _UInt   __ctz(_UInt __x) { return __builtin_ctz(__x); }
constexpr _ULong  __ctz(_ULong __x) { return __builtin_ctzl(__x); }
constexpr _ULLong __ctz(_ULLong __x) { return __builtin_ctzll(__x); }
constexpr _UInt   __clz(_UInt __x) { return __builtin_clz(__x); }
constexpr _ULong  __clz(_ULong __x) { return __builtin_clzl(__x); }
constexpr _ULLong __clz(_ULLong __x) { return __builtin_clzll(__x); }

template <typename _Tp, typename _F>
void __bit_iteration(_Tp __mask, _F&& __f)
{
    static_assert(sizeof(_ULLong) >= sizeof(_Tp));
    std::conditional_t<sizeof(_Tp) <= sizeof(_UInt), _UInt, _ULLong> __k;
    if constexpr (std::is_convertible_v<_Tp, decltype(__k)>) {
        __k = __mask;
    } else {
        __k = __mask.to_ullong();
    }
    switch (__popcount(__k)) {
    default:
        do {
            __f(__ctz(__k));
            __k &= (__k - 1);
        } while (__k);
        break;
    /*case 3:
        __f(__ctz(__k));
        __k &= (__k - 1);
        [[fallthrough]];*/
    case 2:
        __f(__ctz(__k));
        [[fallthrough]];
    case 1:
        __f(__popcount(~decltype(__k)()) - 1 - __clz(__k));
        [[fallthrough]];
    case 0:
        break;
    }
}

//}}}
// __firstbit{{{
template <typename _Tp>
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST auto __firstbit(_Tp __bits)
{
  static_assert(std::is_integral_v<_Tp>,
		"__firstbit requires an integral argument");
  if constexpr (sizeof(_Tp) <= sizeof(int))
    {
      return __builtin_ctz(__bits);
    }
  else if constexpr (alignof(_ULLong) == 8)
    {
      return __builtin_ctzll(__bits);
    }
  else
    {
      _UInt __lo = __bits;
      return __lo == 0 ? 32 + __builtin_ctz(__bits >> 32) : __builtin_ctz(__lo);
    }
}

// }}}
// __lastbit{{{
template <typename _Tp>
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST auto __lastbit(_Tp __bits)
{
  static_assert(std::is_integral_v<_Tp>,
		"__firstbit requires an integral argument");
  if constexpr (sizeof(_Tp) <= sizeof(int))
    {
      return 31 - __builtin_clz(__bits);
    }
  else if constexpr (alignof(_ULLong) == 8)
    {
      return 63 - __builtin_clzll(__bits);
    }
  else
    {
      _UInt __lo = __bits;
      _UInt __hi = __bits >> 32u;
      return __hi == 0 ? 31 - __builtin_clz(__lo) : 63 - __builtin_clz(__hi);
    }
}

// }}}
// __convert_mask declaration {{{
template <typename _To, typename _From>
inline _To __convert_mask(_From __k);

// }}}
// __shift_left, __shift_right, __increment, __decrement {{{
template <typename _Tp = void>
struct __shift_left
{
  constexpr _Tp operator()(const _Tp& __a, const _Tp& __b) const
  {
    return __a << __b;
  }
};
template <>
struct __shift_left<void>
{
  template <typename _L, typename _R>
  constexpr auto operator()(_L&& __a, _R&& __b) const
  {
    return std::forward<_L>(__a) << std::forward<_R>(__b);
  }
};
template <typename _Tp = void>
struct __shift_right
{
  constexpr _Tp operator()(const _Tp& __a, const _Tp& __b) const
  {
    return __a >> __b;
  }
};
template <>
struct __shift_right<void>
{
  template <typename _L, typename _R>
  constexpr auto operator()(_L&& __a, _R&& __b) const
  {
    return std::forward<_L>(__a) >> std::forward<_R>(__b);
  }
};
template <typename _Tp = void>
struct __increment
{
  constexpr _Tp operator()(_Tp __a) const { return ++__a; }
};
template <>
struct __increment<void>
{
  template <typename _Tp>
  constexpr _Tp operator()(_Tp __a) const
  {
    return ++__a;
  }
};
template <typename _Tp = void>
struct __decrement
{
  constexpr _Tp operator()(_Tp __a) const { return --__a; }
};
template <>
struct __decrement<void>
{
  template <typename _Tp>
  constexpr _Tp operator()(_Tp __a) const
  {
    return --__a;
  }
};

// }}}
// __value_preserving(_or_int) {{{
template <typename _From,
	  typename _To,
	  typename = enable_if_t<negation<
	    __is_narrowing_conversion<std::decay_t<_From>, _To>>::value>>
using _ValuePreserving = _From;

template <typename _From,
	  typename _To,
	  typename _DecayedFrom = std::decay_t<_From>,
	  typename              = enable_if_t<conjunction<
            is_convertible<_From, _To>,
            disjunction<
              is_same<_DecayedFrom, _To>,
              is_same<_DecayedFrom, int>,
              conjunction<is_same<_DecayedFrom, _UInt>, is_unsigned<_To>>,
              negation<__is_narrowing_conversion<_DecayedFrom, _To>>>>::value>>
using _ValuePreservingOrInt = _From;

// }}}
// __intrinsic_type {{{
template <typename _Tp, size_t _Bytes, typename = std::void_t<>> struct __intrinsic_type;
template <typename _Tp, size_t _Size>
using __intrinsic_type_t = typename __intrinsic_type<_Tp, _Size * sizeof(_Tp)>::type;
template <typename _Tp> using __intrinsic_type2_t   = typename __intrinsic_type<_Tp, 2>::type;
template <typename _Tp> using __intrinsic_type4_t   = typename __intrinsic_type<_Tp, 4>::type;
template <typename _Tp> using __intrinsic_type8_t   = typename __intrinsic_type<_Tp, 8>::type;
template <typename _Tp> using __intrinsic_type16_t  = typename __intrinsic_type<_Tp, 16>::type;
template <typename _Tp> using __intrinsic_type32_t  = typename __intrinsic_type<_Tp, 32>::type;
template <typename _Tp> using __intrinsic_type64_t  = typename __intrinsic_type<_Tp, 64>::type;
template <typename _Tp> using __intrinsic_type128_t = typename __intrinsic_type<_Tp, 128>::type;

// }}}

// vvv ---- builtin vector types [[gnu::vector_size(N)]] and operations ---- vvv
// __vector_type {{{
template <typename _Tp, size_t _N, typename = void> struct __vector_type_n {};

// special case 1-element to be _Tp itself
template <typename _Tp>
struct __vector_type_n<_Tp, 1, enable_if_t<__is_vectorizable_v<_Tp>>> {
    using type = _Tp;
};

// else, use GNU-style builtin vector types
template <typename _Tp, size_t _N>
struct __vector_type_n<_Tp, _N, enable_if_t<__is_vectorizable_v<_Tp>>> {
    static constexpr size_t _Bytes = _N * sizeof(_Tp);
    using type [[__gnu__::__vector_size__(_Bytes)]] = _Tp;
};

template <typename _Tp, size_t _Bytes>
struct __vector_type : __vector_type_n<_Tp, _Bytes / sizeof(_Tp)> {
    static_assert(_Bytes % sizeof(_Tp) == 0);
};

template <typename _Tp, size_t _Size>
using __vector_type_t = typename __vector_type_n<_Tp, _Size>::type;
template <typename _Tp> using __vector_type2_t  = typename __vector_type<_Tp, 2>::type;
template <typename _Tp> using __vector_type4_t  = typename __vector_type<_Tp, 4>::type;
template <typename _Tp> using __vector_type8_t  = typename __vector_type<_Tp, 8>::type;
template <typename _Tp> using __vector_type16_t = typename __vector_type<_Tp, 16>::type;
template <typename _Tp> using __vector_type32_t = typename __vector_type<_Tp, 32>::type;
template <typename _Tp> using __vector_type64_t = typename __vector_type<_Tp, 64>::type;
template <typename _Tp> using __vector_type128_t = typename __vector_type<_Tp, 128>::type;

// }}}
// __is_vector_type {{{
template <typename _Tp, typename = std::void_t<>> struct __is_vector_type : false_type {};
template <typename _Tp>
struct __is_vector_type<
    _Tp,
    std::void_t<typename __vector_type<decltype(std::declval<_Tp>()[0]), sizeof(_Tp)>::type>>
    : std::is_same<
          _Tp, typename __vector_type<decltype(std::declval<_Tp>()[0]), sizeof(_Tp)>::type> {
};

template <typename _Tp>
inline constexpr bool __is_vector_type_v = __is_vector_type<_Tp>::value;

// }}}
// __vector_traits{{{
template <typename _Tp, typename = std::void_t<>> struct __vector_traits;
template <typename _Tp>
struct __vector_traits<_Tp, std::void_t<enable_if_t<__is_vector_type_v<_Tp>>>> {
    using type = _Tp;
    using value_type = decltype(std::declval<_Tp>()[0]);
    static constexpr int _S_width = sizeof(_Tp) / sizeof(value_type);
    template <typename _U, int _W = _S_width>
    static constexpr bool __is = std::is_same_v<value_type, _U> &&_W == _S_width;
};
template <typename _Tp, size_t _N>
struct __vector_traits<_SimdWrapper<_Tp, _N>, std::void_t<__vector_type_t<_Tp, _N>>> {
    using type = __vector_type_t<_Tp, _N>;
    using value_type = _Tp;
    static constexpr int _S_width = _N;
    template <typename _U, int _W = _S_width>
    static constexpr bool __is = std::is_same_v<value_type, _U> &&_W == _S_width;
};

// }}}
// __vector_bitcast{{{
template <typename _To, typename _From, typename _FromVT = __vector_traits<_From>>
_GLIBCXX_SIMD_INTRINSIC constexpr typename __vector_type<_To, sizeof(_From)>::type __vector_bitcast(_From __x)
{
    return reinterpret_cast<typename __vector_type<_To, sizeof(_From)>::type>(__x);
}
template <typename _To, typename _Tp, size_t _N>
_GLIBCXX_SIMD_INTRINSIC constexpr typename __vector_type<_To, sizeof(_SimdWrapper<_Tp, _N>)>::type
__vector_bitcast(const _SimdWrapper<_Tp, _N> &__x)
{
    return reinterpret_cast<typename __vector_type<_To, sizeof(_SimdWrapper<_Tp, _N>)>::type>(__x._M_data);
}

// }}}
// __convert_x86 declarations {{{
template <typename _To, typename _Tp, typename _TVT = __vector_traits<_Tp>>
_To __convert_x86(_Tp);

template <typename _To, typename _Tp, typename _TVT = __vector_traits<_Tp>>
_To __convert_x86(_Tp, _Tp);

template <typename _To, typename _Tp, typename _TVT = __vector_traits<_Tp>>
_To __convert_x86(_Tp, _Tp, _Tp, _Tp);

template <typename _To, typename _Tp, typename _TVT = __vector_traits<_Tp>>
_To __convert_x86(_Tp, _Tp, _Tp, _Tp, _Tp, _Tp, _Tp, _Tp);

//}}}
// __vector_convert {{{
// implementation requires an index sequence
template <typename _To, typename _From, size_t... _I>
_GLIBCXX_SIMD_INTRINSIC constexpr _To __vector_convert(_From __a, index_sequence<_I...>)
{
    using _Tp = typename __vector_traits<_To>::value_type;
    return _To{static_cast<_Tp>(__a[_I])...};
}

template <typename _To, typename _From, size_t... _I>
_GLIBCXX_SIMD_INTRINSIC constexpr _To __vector_convert(_From __a, _From __b,
                                                       index_sequence<_I...>)
{
    using _Tp = typename __vector_traits<_To>::value_type;
    return _To{static_cast<_Tp>(__a[_I])..., static_cast<_Tp>(__b[_I])...};
}

template <typename _To, typename _From, size_t... _I>
_GLIBCXX_SIMD_INTRINSIC constexpr _To __vector_convert(_From __a, _From __b, _From __c,
                                                       index_sequence<_I...>)
{
    using _Tp = typename __vector_traits<_To>::value_type;
    return _To{static_cast<_Tp>(__a[_I])..., static_cast<_Tp>(__b[_I])...,
               static_cast<_Tp>(__c[_I])...};
}

template <typename _To, typename _From, size_t... _I>
_GLIBCXX_SIMD_INTRINSIC constexpr _To __vector_convert(_From __a, _From __b, _From __c,
                                                       _From __d, index_sequence<_I...>)
{
    using _Tp = typename __vector_traits<_To>::value_type;
    return _To{static_cast<_Tp>(__a[_I])..., static_cast<_Tp>(__b[_I])...,
               static_cast<_Tp>(__c[_I])..., static_cast<_Tp>(__d[_I])...};
}

template <typename _To, typename _From, size_t... _I>
_GLIBCXX_SIMD_INTRINSIC constexpr _To __vector_convert(_From __a, _From __b, _From __c,
                                                       _From __d, _From __e, _From __f,
                                                       _From __g, _From __h,
                                                       index_sequence<_I...>)
{
    using _Tp = typename __vector_traits<_To>::value_type;
    return _To{static_cast<_Tp>(__a[_I])..., static_cast<_Tp>(__b[_I])...,
               static_cast<_Tp>(__c[_I])..., static_cast<_Tp>(__d[_I])...,
               static_cast<_Tp>(__e[_I])..., static_cast<_Tp>(__f[_I])...,
               static_cast<_Tp>(__g[_I])..., static_cast<_Tp>(__h[_I])...};
}

// Defer actual conversion to the overload that takes an index sequence. Note that this
// function adds zeros or drops values off the end if you don't ensure matching width.
template <typename _To, typename... _From, typename _ToT = __vector_traits<_To>,
          typename _FromT = __vector_traits<__first_of_pack_t<_From...>>>
_GLIBCXX_SIMD_INTRINSIC constexpr _To __vector_convert(_From... __xs)
{
#ifdef _GLIBCXX_SIMD_WORKAROUND_PR85048
    return __convert_x86<_To>(__xs...);
#else
    return __vector_convert<_To>(__xs...,
                            make_index_sequence<std::min(_ToT::_S_width, _FromT::_S_width)>());
#endif
}

// This overload takes a vectorizable type _To and produces a return type that matches the
// width.
template <typename _To, typename... _From, typename = enable_if_t<__is_vectorizable_v<_To>>,
          typename _FromT = __vector_traits<__first_of_pack_t<_From...>>, typename = int>
_GLIBCXX_SIMD_INTRINSIC constexpr _To __vector_convert(_From... __xs)
{
    return __vector_convert<__vector_type_t<_To, _FromT::_S_width>>(__xs...);
}

// }}}
// __to_intrin {{{
template <typename _Tp, typename _TVT = __vector_traits<_Tp>,
          typename _R = __intrinsic_type_t<typename _TVT::value_type, _TVT::_S_width>>
_GLIBCXX_SIMD_INTRINSIC constexpr _R __to_intrin(_Tp __x)
{
    return reinterpret_cast<_R>(__x);
}
template <typename _Tp, size_t _N, typename _R = __intrinsic_type_t<_Tp, _N>>
_GLIBCXX_SIMD_INTRINSIC constexpr _R __to_intrin(_SimdWrapper<_Tp, _N> __x)
{
    return reinterpret_cast<_R>(__x._M_data);
}

// }}}
// __make_builtin{{{
template <typename _Tp, typename... _Args>
_GLIBCXX_SIMD_INTRINSIC constexpr __vector_type_t<_Tp, sizeof...(_Args)> __make_builtin(_Args &&... args)
{
    return __vector_type_t<_Tp, sizeof...(_Args)>{static_cast<_Tp>(args)...};
}

// }}}
// __vector_broadcast{{{
template <size_t _N, typename _Tp>
_GLIBCXX_SIMD_INTRINSIC constexpr __vector_type_t<_Tp, _N> __vector_broadcast(_Tp __x)
{
    if constexpr (_N == 2) {
        return __vector_type_t<_Tp, 2>{__x, __x};
    } else if constexpr (_N == 4) {
        return __vector_type_t<_Tp, 4>{__x, __x, __x, __x};
    } else if constexpr (_N == 8) {
        return __vector_type_t<_Tp, 8>{__x, __x, __x, __x, __x, __x, __x, __x};
    } else if constexpr (_N == 16) {
        return __vector_type_t<_Tp, 16>{__x, __x, __x, __x, __x, __x, __x, __x,
                                       __x, __x, __x, __x, __x, __x, __x, __x};
    } else if constexpr (_N == 32) {
        return __vector_type_t<_Tp, 32>{__x, __x, __x, __x, __x, __x, __x, __x,
                                       __x, __x, __x, __x, __x, __x, __x, __x,
                                       __x, __x, __x, __x, __x, __x, __x, __x,
                                       __x, __x, __x, __x, __x, __x, __x, __x};
    } else if constexpr (_N == 64) {
        return __vector_type_t<_Tp, 64>{
            __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x,
            __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x,
            __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x,
            __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x,
            __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x, __x};
    } else if constexpr (_N == 128) {
        return __vector_type_t<_Tp, 128>{
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
template <typename _Tp> struct __auto_broadcast {
    const _Tp __x;
    _GLIBCXX_SIMD_INTRINSIC constexpr __auto_broadcast(_Tp __xx) : __x(__xx) {}
    template <typename _V> _GLIBCXX_SIMD_INTRINSIC constexpr operator _V() const
    {
        static_assert(__is_vector_type_v<_V>);
        return reinterpret_cast<_V>(__vector_broadcast<sizeof(_V) / sizeof(_Tp)>(__x));
    }
};

// }}}
// __generate_builtin{{{
template <typename _Tp, size_t _N, typename _G, size_t... _I>
_GLIBCXX_SIMD_INTRINSIC constexpr __vector_type_t<_Tp, _N> generate_builtin_impl(
    _G &&__gen, std::index_sequence<_I...>)
{
    return __vector_type_t<_Tp, _N>{static_cast<_Tp>(__gen(_SizeConstant<_I>()))...};
}

template <typename _V, typename _VVT = __vector_traits<_V>, typename _G>
_GLIBCXX_SIMD_INTRINSIC constexpr _V __generate_builtin(_G &&__gen)
{
    return generate_builtin_impl<typename _VVT::value_type, _VVT::_S_width>(
        std::forward<_G>(__gen), std::make_index_sequence<_VVT::_S_width>());
}

template <typename _Tp, size_t _N, typename _G>
_GLIBCXX_SIMD_INTRINSIC constexpr __vector_type_t<_Tp, _N> __generate_builtin(_G &&__gen)
{
    return generate_builtin_impl<_Tp, _N>(std::forward<_G>(__gen),
                                       std::make_index_sequence<_N>());
}

// }}}
// __vector_load{{{
template <typename _Tp, size_t _N, size_t _M = _N * sizeof(_Tp), typename _F>
__vector_type_t<_Tp, _N> __vector_load(const void *__p, _F)
{
#ifdef _GLIBCXX_SIMD_WORKAROUND_XXX_2
    using _U = std::conditional_t<
        (std::is_integral_v<_Tp> || _M < 4), long long,
        std::conditional_t<(std::is_same_v<_Tp, double> || _M < 8), float, _Tp>>;
    using _V = __vector_type_t<_U, _N * sizeof(_Tp) / sizeof(_U)>;
#else   // _GLIBCXX_SIMD_WORKAROUND_XXX_2
    using _V = __vector_type_t<_Tp, _N>;
#endif  // _GLIBCXX_SIMD_WORKAROUND_XXX_2
    _V __r;
    static_assert(_M <= sizeof(_V));
    if constexpr(std::is_same_v<_F, element_aligned_tag>) {
    } else if constexpr(std::is_same_v<_F, vector_aligned_tag>) {
        __p = __builtin_assume_aligned(__p, alignof(__vector_type_t<_Tp, _N>));
    } else {
        __p = __builtin_assume_aligned(__p, _F::_S_alignment);
    }
    std::memcpy(&__r, __p, _M);
    return reinterpret_cast<__vector_type_t<_Tp, _N>>(__r);
}

// }}}
// __vector_load16 {{{
template <typename _Tp, size_t _M = 16, typename _F>
__vector_type16_t<_Tp> __vector_load16(const void *__p, _F __f)
{
    return __vector_load<_Tp, 16 / sizeof(_Tp), _M>(__p, __f);
}

// }}}
// __vector_store{{{
template <size_t _M = 0, typename _B, typename _BVT = __vector_traits<_B>, typename _F>
void __vector_store(const _B __v, void *__p, _F)
{
    using _Tp = typename _BVT::value_type;
    constexpr size_t _N = _BVT::_S_width;
    constexpr size_t _Bytes = _M == 0 ? _N * sizeof(_Tp) : _M;
    static_assert(_Bytes <= sizeof(__v));
#ifdef _GLIBCXX_SIMD_WORKAROUND_XXX_2
    using _U = std::conditional_t<
        (std::is_integral_v<_Tp> || _Bytes < 4), long long,
        std::conditional_t<(std::is_same_v<_Tp, double> || _Bytes < 8), float, _Tp>>;
    const auto __vv = __vector_bitcast<_U>(__v);
#else   // _GLIBCXX_SIMD_WORKAROUND_XXX_2
    const __vector_type_t<_Tp, _N> __vv = __v;
#endif  // _GLIBCXX_SIMD_WORKAROUND_XXX_2
    if constexpr(std::is_same_v<_F, vector_aligned_tag>) {
        __p = __builtin_assume_aligned(__p, alignof(__vector_type_t<_Tp, _N>));
    } else if constexpr(!std::is_same_v<_F, element_aligned_tag>) {
        __p = __builtin_assume_aligned(__p, _F::_S_alignment);
    }
    if constexpr ((_Bytes & (_Bytes - 1)) != 0) {
        constexpr size_t _MoreBytes = __next_power_of_2(_Bytes);
        alignas(_MoreBytes) char __tmp[_MoreBytes];
        std::memcpy(__tmp, &__vv, _MoreBytes);
        std::memcpy(__p, __tmp, _Bytes);
    } else {
        std::memcpy(__p, &__vv, _Bytes);
    }
}

// }}}
// __allbits{{{
template <typename _V>
inline constexpr _V __allbits =
    reinterpret_cast<_V>(~__intrinsic_type_t<_LLong, sizeof(_V) / sizeof(_LLong)>());

// }}}
// __xor{{{
template <typename _Tp, typename _TVT = __vector_traits<_Tp>>
_GLIBCXX_SIMD_INTRINSIC constexpr _Tp __xor(_Tp __a, typename _TVT::type __b) noexcept
{
  if constexpr (_TVT::template __is<float, 4> && __have_sse)
    return _mm_xor_ps(__a, __b);
  else if constexpr (_TVT::template __is<double, 2> && __have_sse2)
    return _mm_xor_pd(__a, __b);
  else if constexpr (_TVT::template __is<float, 8> && __have_avx)
    return _mm256_xor_ps(__a, __b);
  else if constexpr (_TVT::template __is<double, 4> && __have_avx)
    return _mm256_xor_pd(__a, __b);
  else if constexpr (_TVT::template __is<float, 16> && __have_avx512dq)
    return _mm512_xor_ps(__a, __b);
  else if constexpr (_TVT::template __is<double, 8> && __have_avx512dq)
    return _mm512_xor_pd(__a, __b);
  else
    return reinterpret_cast<typename _TVT::type>(
      __vector_bitcast<unsigned>(__a) ^ __vector_bitcast<unsigned>(__b));
}

// }}}
// __or{{{
template <typename _Tp, typename _TVT = __vector_traits<_Tp>>
_GLIBCXX_SIMD_INTRINSIC constexpr _Tp __or(_Tp __a, typename _TVT::type __b) noexcept
{
  if constexpr (_TVT::template __is<float, 4> && __have_sse)
    return _mm_or_ps(__a, __b);
  else if constexpr (_TVT::template __is<double, 2> && __have_sse2)
    return _mm_or_pd(__a, __b);
  else if constexpr (_TVT::template __is<float, 8> && __have_avx)
    return _mm256_or_ps(__a, __b);
  else if constexpr (_TVT::template __is<double, 4> && __have_avx)
    return _mm256_or_pd(__a, __b);
  else if constexpr (_TVT::template __is<float, 16> && __have_avx512dq)
    return _mm512_or_ps(__a, __b);
  else if constexpr (_TVT::template __is<double, 8> && __have_avx512dq)
    return _mm512_or_pd(__a, __b);
  else
    return reinterpret_cast<typename _TVT::type>(
      __vector_bitcast<unsigned>(__a) | __vector_bitcast<unsigned>(__b));
}

// }}}
// __and{{{
template <typename _Tp, typename _TVT = __vector_traits<_Tp>, typename... _Dummy>
_GLIBCXX_SIMD_INTRINSIC constexpr _Tp
  __and(_Tp __a, typename _TVT::type __b, _Dummy...) noexcept
{
  static_assert(sizeof...(_Dummy) == 0);
  if constexpr (_TVT::template __is<float, 4> && __have_sse)
    return _mm_and_ps(__a, __b);
  else if constexpr (_TVT::template __is<double, 2> && __have_sse2)
    return _mm_and_pd(__a, __b);
  else if constexpr (_TVT::template __is<float, 8> && __have_avx)
    return _mm256_and_ps(__a, __b);
  else if constexpr (_TVT::template __is<double, 4> && __have_avx)
    return _mm256_and_pd(__a, __b);
  else if constexpr (_TVT::template __is<float, 16> && __have_avx512dq)
    return _mm512_and_ps(__a, __b);
  else if constexpr (_TVT::template __is<double, 8> && __have_avx512dq)
    return _mm512_and_pd(__a, __b);
  else
    return reinterpret_cast<typename _TVT::type>(
      __vector_bitcast<unsigned>(__a) & __vector_bitcast<unsigned>(__b));
}

template <typename _Tp, typename = decltype(_Tp() & _Tp())>
_GLIBCXX_SIMD_INTRINSIC constexpr _Tp __and(_Tp __a, _Tp __b) noexcept
{
  return __a & __b;
}

// }}}
// __andnot{{{
template <typename _Tp, typename _TVT = __vector_traits<_Tp>>
_GLIBCXX_SIMD_INTRINSIC constexpr _Tp __andnot(_Tp __a, typename _TVT::type __b) noexcept
{
  if constexpr (_TVT::template __is<float, 4> && __have_sse)
    return _mm_andnot_ps(__a, __b);
  else if constexpr (_TVT::template __is<double, 2> && __have_sse2)
    return _mm_andnot_pd(__a, __b);
  else if constexpr (_TVT::template __is<float, 8> && __have_avx)
    return _mm256_andnot_ps(__a, __b);
  else if constexpr (_TVT::template __is<double, 4> && __have_avx)
    return _mm256_andnot_pd(__a, __b);
  else if constexpr (_TVT::template __is<float, 16> && __have_avx512dq)
    return _mm512_andnot_ps(__a, __b);
  else if constexpr (_TVT::template __is<double, 8> && __have_avx512dq)
    return _mm512_andnot_pd(__a, __b);
  else
    return reinterpret_cast<typename _TVT::type>(
      ~__vector_bitcast<unsigned>(__a) & __vector_bitcast<unsigned>(__b));
}

// }}}
// __not{{{
template <typename _Tp, typename _TVT = __vector_traits<_Tp>>
_GLIBCXX_SIMD_INTRINSIC constexpr _Tp __not(_Tp __a) noexcept
{
    return reinterpret_cast<_Tp>(~__vector_bitcast<unsigned>(__a));
}

// }}}
// __concat{{{
template <typename _Tp, typename _TVT = __vector_traits<_Tp>,
          typename _R = __vector_type_t<typename _TVT::value_type, _TVT::_S_width * 2>>
constexpr _R __concat(_Tp a_, _Tp b_) {
#ifdef _GLIBCXX_SIMD_WORKAROUND_XXX_1
    using _W = std::conditional_t<std::is_floating_point_v<typename _TVT::value_type>,
                                 double, long long>;
    constexpr int input_width = sizeof(_Tp) / sizeof(_W);
    const auto __a = __vector_bitcast<_W>(a_);
    const auto __b = __vector_bitcast<_W>(b_);
    using _U = __vector_type_t<_W, sizeof(_R) / sizeof(_W)>;
#else
    constexpr int input_width = _TVT::_S_width;
    const _Tp &__a = a_;
    const _Tp &__b = b_;
    using _U = _R;
#endif
    if constexpr(input_width == 2) {
        return reinterpret_cast<_R>(_U{__a[0], __a[1], __b[0], __b[1]});
    } else if constexpr (input_width == 4) {
        return reinterpret_cast<_R>(_U{__a[0], __a[1], __a[2], __a[3], __b[0], __b[1], __b[2], __b[3]});
    } else if constexpr (input_width == 8) {
        return reinterpret_cast<_R>(_U{__a[0], __a[1], __a[2], __a[3], __a[4], __a[5], __a[6], __a[7], __b[0],
                                     __b[1], __b[2], __b[3], __b[4], __b[5], __b[6], __b[7]});
    } else if constexpr (input_width == 16) {
        return reinterpret_cast<_R>(
            _U{__a[0],  __a[1],  __a[2],  __a[3],  __a[4],  __a[5],  __a[6],  __a[7],  __a[8],  __a[9], __a[10],
              __a[11], __a[12], __a[13], __a[14], __a[15], __b[0],  __b[1],  __b[2],  __b[3],  __b[4], __b[5],
              __b[6],  __b[7],  __b[8],  __b[9],  __b[10], __b[11], __b[12], __b[13], __b[14], __b[15]});
    } else if constexpr (input_width == 32) {
        return reinterpret_cast<_R>(
            _U{__a[0],  __a[1],  __a[2],  __a[3],  __a[4],  __a[5],  __a[6],  __a[7],  __a[8],  __a[9],  __a[10],
              __a[11], __a[12], __a[13], __a[14], __a[15], __a[16], __a[17], __a[18], __a[19], __a[20], __a[21],
              __a[22], __a[23], __a[24], __a[25], __a[26], __a[27], __a[28], __a[29], __a[30], __a[31], __b[0],
              __b[1],  __b[2],  __b[3],  __b[4],  __b[5],  __b[6],  __b[7],  __b[8],  __b[9],  __b[10], __b[11],
              __b[12], __b[13], __b[14], __b[15], __b[16], __b[17], __b[18], __b[19], __b[20], __b[21], __b[22],
              __b[23], __b[24], __b[25], __b[26], __b[27], __b[28], __b[29], __b[30], __b[31]});
    }
}

// }}}
// __zero_extend {{{
template <typename _Tp, typename _TVT = __vector_traits<_Tp>>
_GLIBCXX_SIMD_INTRINSIC auto __zero_extend(_Tp __x)
{
    using value_type = typename _TVT::value_type;
    constexpr size_t _N = _TVT::_S_width;
    struct {
        _Tp __x;
        operator __vector_type_t<value_type, _N * 2>()
        {
#ifdef _GLIBCXX_SIMD_WORKAROUND_XXX_3
            if constexpr (__have_avx && _TVT::template __is<float, 4>) {
                return __vector_bitcast<value_type>(_mm256_insertf128_ps(__m256(), __x, 0));
            } else if constexpr (__have_avx && _TVT::template __is<double, 2>) {
                return __vector_bitcast<value_type>(_mm256_insertf128_pd(__m256d(), __x, 0));
            } else if constexpr (__have_avx2 && sizeof(__x) == 16) {
                return __vector_bitcast<value_type>(_mm256_insertf128_si256(__m256i(), __x, 0));
            } else if constexpr (__have_avx512f && _TVT::template __is<float, 8>) {
                if constexpr (__have_avx512dq) {
                    return __vector_bitcast<value_type>(_mm512_insertf32x8(__m512(), __x, 0));
                } else {
                    return reinterpret_cast<__m512>(
                        _mm512_insertf64x4(__m512d(), reinterpret_cast<__m256d>(__x), 0));
                }
            } else if constexpr (__have_avx512f && _TVT::template __is<double, 4>) {
                return __vector_bitcast<value_type>(_mm512_insertf64x4(__m512d(), __x, 0));
            } else if constexpr (__have_avx512f && sizeof(__x) == 32) {
                return __vector_bitcast<value_type>(_mm512_inserti64x4(__m512i(), __x, 0));
            }
#endif
            return __concat(__x, _Tp());
        }
        operator __vector_type_t<value_type, _N * 4>()
        {
#ifdef _GLIBCXX_SIMD_WORKAROUND_XXX_3
            if constexpr (__have_avx && _TVT::template __is<float, 4>) {
#ifdef _GLIBCXX_SIMD_WORKAROUND_PR85480
                asm("vmovaps %0, %0" : "+__x"(__x));
                return __vector_bitcast<value_type>(_mm512_castps128_ps512(__x));
#else
                return __vector_bitcast<value_type>(_mm512_insertf32x4(__m512(), __x, 0));
#endif
            } else if constexpr (__have_avx && _TVT::template __is<double, 2>) {
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
            return __concat(__concat(__x, _Tp()), __vector_type_t<value_type, _N * 2>());
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
template <int _Offset, int _SplitBy, typename _Tp, typename _TVT = __vector_traits<_Tp>,
          typename _R = __vector_type_t<typename _TVT::value_type, _TVT::_S_width / _SplitBy>>
_GLIBCXX_SIMD_INTRINSIC constexpr _R __extract(_Tp __in)
{
#ifdef _GLIBCXX_SIMD_WORKAROUND_XXX_1
    using _W = std::conditional_t<std::is_floating_point_v<typename _TVT::value_type>,
                                 double, long long>;
    constexpr int return_width = sizeof(_R) / sizeof(_W);
    using _U = __vector_type_t<_W, return_width>;
    const auto __x = __vector_bitcast<_W>(__in);
#else
    constexpr int return_width = _TVT::_S_width / _SplitBy;
    using _U = _R;
    const __vector_type_t<typename _TVT::value_type, _TVT::_S_width> &__x =
        __in;  // only needed for _Tp = _SimdWrapper<value_type, _N>
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
template <typename _Tp> _GLIBCXX_SIMD_INTRINSIC constexpr auto __lo128(_Tp __x)
{
    return __extract<0, sizeof(_Tp) / 16>(__x);
}
template <typename _Tp> _GLIBCXX_SIMD_INTRINSIC constexpr auto __hi128(_Tp __x)
{
    static_assert(sizeof(__x) == 32);
    return __extract<1, 2>(__x);
}

// }}}
// __lo/__hi256{{{
template <typename _Tp> _GLIBCXX_SIMD_INTRINSIC constexpr auto __lo256(_Tp __x)
{
    static_assert(sizeof(__x) == 64);
    return __extract<0, 2>(__x);
}
template <typename _Tp> _GLIBCXX_SIMD_INTRINSIC constexpr auto __hi256(_Tp __x)
{
    static_assert(sizeof(__x) == 64);
    return __extract<1, 2>(__x);
}

// }}}
// __intrin_bitcast{{{
template <typename _To, typename _From> _GLIBCXX_SIMD_INTRINSIC constexpr _To __intrin_bitcast(_From __v)
{
    static_assert(__is_vector_type_v<_From> && __is_vector_type_v<_To>);
    if constexpr (sizeof(_To) == sizeof(_From)) {
        return reinterpret_cast<_To>(__v);
    } else if constexpr (sizeof(_From) > sizeof(_To)) {
        return reinterpret_cast<const _To &>(__v);
    } else if constexpr (__have_avx && sizeof(_From) == 16 && sizeof(_To) == 32) {
        return reinterpret_cast<_To>(_mm256_castps128_ps256(
            reinterpret_cast<__intrinsic_type_t<float, sizeof(_From) / sizeof(float)>>(__v)));
    } else if constexpr (__have_avx512f && sizeof(_From) == 16 && sizeof(_To) == 64) {
        return reinterpret_cast<_To>(_mm512_castps128_ps512(
            reinterpret_cast<__intrinsic_type_t<float, sizeof(_From) / sizeof(float)>>(__v)));
    } else if constexpr (__have_avx512f && sizeof(_From) == 32 && sizeof(_To) == 64) {
        return reinterpret_cast<_To>(_mm512_castps256_ps512(
            reinterpret_cast<__intrinsic_type_t<float, sizeof(_From) / sizeof(float)>>(__v)));
    } else {
        __assert_unreachable<_To>();
    }
}

// }}}
// __auto_bitcast{{{
template <typename _Tp> struct auto_cast_t {
    static_assert(__is_vector_type_v<_Tp>);
    const _Tp __x;
    template <typename _U> _GLIBCXX_SIMD_INTRINSIC constexpr operator _U() const
    {
        return __intrin_bitcast<_U>(__x);
    }
};
template <typename _Tp> _GLIBCXX_SIMD_INTRINSIC constexpr auto_cast_t<_Tp> __auto_bitcast(const _Tp &__x)
{
    return {__x};
}
template <typename _Tp, size_t _N>
_GLIBCXX_SIMD_INTRINSIC constexpr auto_cast_t<typename _SimdWrapper<_Tp, _N>::_BuiltinType> __auto_bitcast(
    const _SimdWrapper<_Tp, _N> &__x)
{
    return {__x._M_data};
}

// }}}
// __vector_to_bitset{{{
_GLIBCXX_SIMD_INTRINSIC constexpr std::bitset<1> __vector_to_bitset(bool __x) { return unsigned(__x); }

template <typename _Tp, typename = enable_if_t<__is_bitmask_v<_Tp> && __have_avx512f>>
_GLIBCXX_SIMD_INTRINSIC constexpr std::bitset<8 * sizeof(_Tp)> __vector_to_bitset(_Tp __x)
{
    if constexpr (std::is_integral_v<_Tp>) {
        return __x;
    } else {
        return __x._M_data;
    }
}

template <typename _Tp, typename _TVT = __vector_traits<_Tp>>
_GLIBCXX_SIMD_INTRINSIC std::bitset<_TVT::_S_width> __vector_to_bitset(_Tp __x)
{
    constexpr bool __is_sse = __have_sse && sizeof(_Tp) == 16;
    constexpr bool __is_avx = __have_avx && sizeof(_Tp) == 32;
    constexpr bool is_neon128 = __have_neon && sizeof(_Tp) == 16;
    constexpr int __w = sizeof(typename _TVT::value_type);
    const auto __intrin = __to_intrin(__x);
    constexpr auto zero = decltype(__intrin)();
    __unused(zero);

    if constexpr (is_neon128 && __w == 1) {
        __x &= _Tp{0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80,
               0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80};
        return __vector_bitcast<ushort>(
            vpaddq_s8(vpaddq_s8(vpaddq_s8(__x, zero), zero), zero))[0];
    } else if constexpr (is_neon128 && __w == 2) {
        __x &= _Tp{0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80};
        return vpaddq_s16(vpaddq_s16(vpaddq_s16(__x, zero), zero), zero)[0];
    } else if constexpr (is_neon128 && __w == 4) {
        __x &= _Tp{0x1, 0x2, 0x4, 0x8};
        return vpaddq_s32(vpaddq_s32(__x, zero), zero)[0];
    } else if constexpr (is_neon128 && __w == 8) {
        __x &= _Tp{0x1, 0x2};
        return __x[0] | __x[1];
    } else if constexpr (__is_sse && __w == 1) {
        return _mm_movemask_epi8(__intrin);
    } else if constexpr (__is_sse && __w == 2) {
        if constexpr (__have_avx512bw_vl) {
            return _mm_cmplt_epi16_mask(__intrin, zero);
        } else {
            return _mm_movemask_epi8(_mm_packs_epi16(__intrin, zero));
        }
    } else if constexpr (__is_sse && __w == 4) {
        if constexpr (__have_avx512vl && std::is_integral_v<_Tp>) {
            return _mm_cmplt_epi32_mask(__intrin, zero);
        } else {
            return _mm_movemask_ps(__vector_bitcast<float>(__x));
        }
    } else if constexpr (__is_sse && __w == 8) {
        if constexpr (__have_avx512vl && std::is_integral_v<_Tp>) {
            return _mm_cmplt_epi64_mask(__intrin, zero);
        } else {
            return _mm_movemask_pd(__vector_bitcast<double>(__x));
        }
    } else if constexpr (__is_avx && __w == 1) {
        return _mm256_movemask_epi8(__intrin);
    } else if constexpr (__is_avx && __w == 2) {
        if constexpr (__have_avx512bw_vl) {
            return _mm256_cmplt_epi16_mask(__intrin, zero);
        } else {
            return _mm_movemask_epi8(_mm_packs_epi16(__extract<0, 2>(__intrin),
                                                     __extract<1, 2>(__intrin)));
        }
    } else if constexpr (__is_avx && __w == 4) {
        if constexpr (__have_avx512vl && std::is_integral_v<_Tp>) {
            return _mm256_cmplt_epi32_mask(__intrin, zero);
        } else {
            return _mm256_movemask_ps(__vector_bitcast<float>(__x));
        }
    } else if constexpr (__is_avx && __w == 8) {
        if constexpr (__have_avx512vl && std::is_integral_v<_Tp>) {
            return _mm256_cmplt_epi64_mask(__intrin, zero);
        } else {
            return _mm256_movemask_pd(__vector_bitcast<double>(__x));
        }
    } else {
        __assert_unreachable<_Tp>();
    }
}

// }}}
// __blend{{{
template <typename _K, typename _V0, typename _V1>
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST auto
			__blend(_K mask, _V0 at0, _V1 at1)
{
  using _V = _V0;
  if constexpr (!std::is_same_v<_V0, _V1>)
    {
      static_assert(sizeof(_V0) == sizeof(_V1));
      if constexpr (__is_vector_type_v<_V0> && !__is_vector_type_v<_V1>)
	{
	  return __blend(mask, at0, reinterpret_cast<_V0>(at1._M_data));
	}
      else if constexpr (!__is_vector_type_v<_V0> && __is_vector_type_v<_V1>)
	{
	  return __blend(mask, reinterpret_cast<_V1>(at0._M_data), at1);
	}
      else
	{
	  __assert_unreachable<_K>();
	}
    }
  else if constexpr (sizeof(_V) < 16)
    {
      static_assert(sizeof(_K) == sizeof(_V0) && sizeof(_V0) == sizeof(_V1));
      return (mask & at1) | (~mask & at0);
    }
  else if constexpr (!__is_vector_type_v<_V>)
    {
      return __blend(mask, at0._M_data, at1._M_data);
    }
  else if constexpr (sizeof(_K) < 16) // blend via bitmask (AVX512)
    {
      using _Tp = typename __vector_traits<_V>::value_type;
      if constexpr (sizeof(_V) == 16 && __have_avx512bw_vl && sizeof(_Tp) <= 2)
	{
	  if constexpr (sizeof(_Tp) == 1)
	    {
	      return __intrin_bitcast<_V>(
		_mm_mask_mov_epi8(__to_intrin(at0), mask, __to_intrin(at1)));
	    }
	  else if constexpr (sizeof(_Tp) == 2)
	    {
	      return __intrin_bitcast<_V>(
		_mm_mask_mov_epi16(__to_intrin(at0), mask, __to_intrin(at1)));
	    }
	}
      else if constexpr (sizeof(_V) == 16 && __have_avx512vl && sizeof(_Tp) > 2)
	{
	  if constexpr (std::is_integral_v<_Tp> && sizeof(_Tp) == 4)
	    {
	      return __intrin_bitcast<_V>(
		_mm_mask_mov_epi32(__to_intrin(at0), mask, __to_intrin(at1)));
	    }
	  else if constexpr (std::is_integral_v<_Tp> && sizeof(_Tp) == 8)
	    {
	      return __intrin_bitcast<_V>(
		_mm_mask_mov_epi64(__to_intrin(at0), mask, __to_intrin(at1)));
	    }
	  else if constexpr (std::is_floating_point_v<_Tp> && sizeof(_Tp) == 4)
	    {
	      return __intrin_bitcast<_V>(_mm_mask_mov_ps(at0, mask, at1));
	    }
	  else if constexpr (std::is_floating_point_v<_Tp> && sizeof(_Tp) == 8)
	    {
	      return __intrin_bitcast<_V>(_mm_mask_mov_pd(at0, mask, at1));
	    }
	}
      else if constexpr (sizeof(_V) == 16 && __have_avx512f && sizeof(_Tp) > 2)
	{
	  if constexpr (std::is_integral_v<_Tp> && sizeof(_Tp) == 4)
	    {
	      return __intrin_bitcast<_V>(__lo128(_mm512_mask_mov_epi32(
		__auto_bitcast(at0), mask, __auto_bitcast(at1))));
	    }
	  else if constexpr (std::is_integral_v<_Tp> && sizeof(_Tp) == 8)
	    {
	      return __intrin_bitcast<_V>(__lo128(_mm512_mask_mov_epi64(
		__auto_bitcast(at0), mask, __auto_bitcast(at1))));
	    }
	  else if constexpr (std::is_floating_point_v<_Tp> && sizeof(_Tp) == 4)
	    {
	      return __intrin_bitcast<_V>(__lo128(_mm512_mask_mov_ps(
		__auto_bitcast(at0), mask, __auto_bitcast(at1))));
	    }
	  else if constexpr (std::is_floating_point_v<_Tp> && sizeof(_Tp) == 8)
	    {
	      return __intrin_bitcast<_V>(__lo128(_mm512_mask_mov_pd(
		__auto_bitcast(at0), mask, __auto_bitcast(at1))));
	    }
	}
      else if constexpr (sizeof(_V) == 32 && __have_avx512bw_vl &&
			 sizeof(_Tp) <= 2)
	{
	  if constexpr (sizeof(_Tp) == 1)
	    {
	      return __intrin_bitcast<_V>(
		_mm256_mask_mov_epi8(__to_intrin(at0), mask, __to_intrin(at1)));
	    }
	  else if constexpr (sizeof(_Tp) == 2)
	    {
	      return __intrin_bitcast<_V>(_mm256_mask_mov_epi16(
		__to_intrin(at0), mask, __to_intrin(at1)));
	    }
	}
      else if constexpr (sizeof(_V) == 32 && __have_avx512vl && sizeof(_Tp) > 2)
	{
	  if constexpr (std::is_integral_v<_Tp> && sizeof(_Tp) == 4)
	    {
	      return __intrin_bitcast<_V>(_mm256_mask_mov_epi32(
		__to_intrin(at0), mask, __to_intrin(at1)));
	    }
	  else if constexpr (std::is_integral_v<_Tp> && sizeof(_Tp) == 8)
	    {
	      return __intrin_bitcast<_V>(_mm256_mask_mov_epi64(
		__to_intrin(at0), mask, __to_intrin(at1)));
	    }
	  else if constexpr (std::is_floating_point_v<_Tp> && sizeof(_Tp) == 4)
	    {
	      return __intrin_bitcast<_V>(_mm256_mask_mov_ps(at0, mask, at1));
	    }
	  else if constexpr (std::is_floating_point_v<_Tp> && sizeof(_Tp) == 8)
	    {
	      return __intrin_bitcast<_V>(_mm256_mask_mov_pd(at0, mask, at1));
	    }
	}
      else if constexpr (sizeof(_V) == 32 && __have_avx512f && sizeof(_Tp) > 2)
	{
	  if constexpr (std::is_integral_v<_Tp> && sizeof(_Tp) == 4)
	    {
	      return __intrin_bitcast<_V>(__lo256(_mm512_mask_mov_epi32(
		__auto_bitcast(at0), mask, __auto_bitcast(at1))));
	    }
	  else if constexpr (std::is_integral_v<_Tp> && sizeof(_Tp) == 8)
	    {
	      return __intrin_bitcast<_V>(__lo256(_mm512_mask_mov_epi64(
		__auto_bitcast(at0), mask, __auto_bitcast(at1))));
	    }
	  else if constexpr (std::is_floating_point_v<_Tp> && sizeof(_Tp) == 4)
	    {
	      return __intrin_bitcast<_V>(__lo256(_mm512_mask_mov_ps(
		__auto_bitcast(at0), mask, __auto_bitcast(at1))));
	    }
	  else if constexpr (std::is_floating_point_v<_Tp> && sizeof(_Tp) == 8)
	    {
	      return __intrin_bitcast<_V>(__lo256(_mm512_mask_mov_pd(
		__auto_bitcast(at0), mask, __auto_bitcast(at1))));
	    }
	}
      else if constexpr (sizeof(_V) == 64 && __have_avx512bw &&
			 sizeof(_Tp) <= 2)
	{
	  if constexpr (sizeof(_Tp) == 1)
	    {
	      return __intrin_bitcast<_V>(
		_mm512_mask_mov_epi8(__to_intrin(at0), mask, __to_intrin(at1)));
	    }
	  else if constexpr (sizeof(_Tp) == 2)
	    {
	      return __intrin_bitcast<_V>(_mm512_mask_mov_epi16(
		__to_intrin(at0), mask, __to_intrin(at1)));
	    }
	}
      else if constexpr (sizeof(_V) == 64 && __have_avx512f && sizeof(_Tp) > 2)
	{
	  if constexpr (std::is_integral_v<_Tp> && sizeof(_Tp) == 4)
	    {
	      return __intrin_bitcast<_V>(_mm512_mask_mov_epi32(
		__to_intrin(at0), mask, __to_intrin(at1)));
	    }
	  else if constexpr (std::is_integral_v<_Tp> && sizeof(_Tp) == 8)
	    {
	      return __intrin_bitcast<_V>(_mm512_mask_mov_epi64(
		__to_intrin(at0), mask, __to_intrin(at1)));
	    }
	  else if constexpr (std::is_floating_point_v<_Tp> && sizeof(_Tp) == 4)
	    {
	      return __intrin_bitcast<_V>(_mm512_mask_mov_ps(at0, mask, at1));
	    }
	  else if constexpr (std::is_floating_point_v<_Tp> && sizeof(_Tp) == 8)
	    {
	      return __intrin_bitcast<_V>(_mm512_mask_mov_pd(at0, mask, at1));
	    }
	}
      else
	{
	  __assert_unreachable<_K>();
	}
    }
  else if constexpr (((__have_avx512f && sizeof(_V) == 64) ||
		      __have_avx512vl) &&
		     (sizeof(typename __vector_traits<_V>::value_type) >= 4 ||
		      __have_avx512bw))
    { // convert mask to bitmask
      return __blend(
	__convert_mask<__bool_storage_member_type_t<__vector_traits<_V>::_S_width>>(mask), at0,
	at1);
    }
  else
    {
      const _V __k = __auto_bitcast(mask);
      using _Tp = typename __vector_traits<_V>::value_type;
      if constexpr (sizeof(_V) == 16 && __have_sse4_1)
	{
	  if constexpr (std::is_integral_v<_Tp>)
	    {
	      return __intrin_bitcast<_V>(_mm_blendv_epi8(
		__to_intrin(at0), __to_intrin(at1), __to_intrin(__k)));
	    }
	  else if constexpr (sizeof(_Tp) == 4)
	    {
	      return __intrin_bitcast<_V>(_mm_blendv_ps(at0, at1, __k));
	    }
	  else if constexpr (sizeof(_Tp) == 8)
	    {
	      return __intrin_bitcast<_V>(_mm_blendv_pd(at0, at1, __k));
	    }
	}
      else if constexpr (sizeof(_V) == 32)
	{
	  if constexpr (std::is_integral_v<_Tp>)
	    {
	      return __intrin_bitcast<_V>(_mm256_blendv_epi8(
		__to_intrin(at0), __to_intrin(at1), __to_intrin(__k)));
	    }
	  else if constexpr (sizeof(_Tp) == 4)
	    {
	      return __intrin_bitcast<_V>(_mm256_blendv_ps(at0, at1, __k));
	    }
	  else if constexpr (sizeof(_Tp) == 8)
	    {
	      return __intrin_bitcast<_V>(_mm256_blendv_pd(at0, at1, __k));
	    }
	}
      else
	{
	  return __or(__andnot(__k, at0), __and(__k, at1));
	}
    }
}

// }}}
// __is_zero{{{
template <typename _Tp, typename _TVT = __vector_traits<_Tp>>
_GLIBCXX_SIMD_INTRINSIC constexpr bool __is_zero(_Tp __a)
{
    const auto __b = __vector_bitcast<_LLong>(__a);
    if constexpr (sizeof(__b) / sizeof(_LLong) == 2) {
        return __b[0] == 0 && __b[1] == 0;
    } else if constexpr (sizeof(__b) / sizeof(_LLong) == 4) {
        return __b[0] == 0 && __b[1] == 0 && __b[2] == 0 && __b[3] == 0;
    } else if constexpr (sizeof(__b) / sizeof(_LLong) == 8) {
        return __b[0] == 0 && __b[1] == 0 && __b[2] == 0 && __b[3] == 0 && __b[4] == 0 &&
               __b[5] == 0 && __b[6] == 0 && __b[7] == 0;
    } else {
        __assert_unreachable<_Tp>();
    }
}
// }}}
// ^^^ ---- builtin vector types [[gnu::vector_size(N)]] and operations ---- ^^^

// __testz{{{
template <typename _Tp, typename _TVT = __vector_traits<_Tp>>
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST int __testz(_Tp __a, _Tp __b)
{
    if constexpr (__have_avx) {
        if constexpr (sizeof(_Tp) == 32 && _TVT::template __is<float>) {
            return _mm256_testz_ps(__a, __b);
        } else if constexpr (sizeof(_Tp) == 32 && _TVT::template __is<double>) {
            return _mm256_testz_pd(__a, __b);
        } else if constexpr (sizeof(_Tp) == 32) {
            return _mm256_testz_si256(__vector_bitcast<_LLong>(__a),
                                      __vector_bitcast<_LLong>(__b));
        } else if constexpr(_TVT::template __is<float, 4>) {
            return _mm_testz_ps(__a, __b);
        } else if constexpr(_TVT::template __is<double, 2>) {
            return _mm_testz_pd(__a, __b);
        } else {
            static_assert(sizeof(_Tp) == 16);
            return _mm_testz_si128(__vector_bitcast<_LLong>(__a), __vector_bitcast<_LLong>(__b));
        }
    } else if constexpr (__have_sse4_1) {
        return _mm_testz_si128(__vector_bitcast<_LLong>(__a), __vector_bitcast<_LLong>(__b));
    } else if constexpr (__have_sse && _TVT::template __is<float, 4>) {
        return _mm_movemask_ps(__and(__a, __b)) == 0;
    } else if constexpr (__have_sse2 && _TVT::template __is<double, 2>) {
        return _mm_movemask_pd(__and(__a, __b)) == 0;
    } else if constexpr (__have_sse2) {
        return _mm_movemask_epi8(__a & __b) == 0;
    } else {
        return __is_zero(__and(__a, __b));
    }
}

// }}}
// __testnzc{{{
template <typename _Tp, typename _TVT = __vector_traits<_Tp>>
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST int __testnzc(_Tp __a, _Tp __b)
{
    if constexpr (__have_avx) {
        if constexpr (sizeof(_Tp) == 32 && _TVT::template __is<float>) {
            return _mm256_testnzc_ps(__a, __b);
        } else if constexpr (sizeof(_Tp) == 32 && _TVT::template __is<double>) {
            return _mm256_testnzc_pd(__a, __b);
        } else if constexpr (sizeof(_Tp) == 32) {
            return _mm256_testnzc_si256(__vector_bitcast<_LLong>(__a), __vector_bitcast<_LLong>(__b));
        } else if constexpr(_TVT::template __is<float, 4>) {
            return _mm_testnzc_ps(__a, __b);
        } else if constexpr(_TVT::template __is<double, 2>) {
            return _mm_testnzc_pd(__a, __b);
        } else {
            static_assert(sizeof(_Tp) == 16);
            return _mm_testnzc_si128(__vector_bitcast<_LLong>(__a), __vector_bitcast<_LLong>(__b));
        }
    } else if constexpr (__have_sse4_1) {
        return _mm_testnzc_si128(__vector_bitcast<_LLong>(__a), __vector_bitcast<_LLong>(__b));
    } else if constexpr (__have_sse && _TVT::template __is<float, 4>) {
        return _mm_movemask_ps(__and(__a, __b)) == 0 && _mm_movemask_ps(__andnot(__a, __b)) == 0;
    } else if constexpr (__have_sse2 && _TVT::template __is<double, 2>) {
        return _mm_movemask_pd(__and(__a, __b)) == 0 && _mm_movemask_pd(__andnot(__a, __b)) == 0;
    } else if constexpr (__have_sse2) {
        return _mm_movemask_epi8(__and(__a, __b)) == 0 &&
               _mm_movemask_epi8(__andnot(__a, __b)) == 0;
    } else {
        return !(__is_zero(__vector_bitcast<_LLong>(__and(__a, __b))) ||
                 __is_zero(__vector_bitcast<_LLong>(__andnot(__a, __b))));
    }
}

// }}}
// __movemask{{{
template <typename _Tp, typename _TVT = __vector_traits<_Tp>>
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST int __movemask(_Tp __a)
{
    if constexpr (__have_sse && _TVT::template __is<float, 4>) {
        return _mm_movemask_ps(__a);
    } else if constexpr (__have_avx && _TVT::template __is<float, 8>) {
        return _mm256_movemask_ps(__a);
    } else if constexpr (__have_sse2 && _TVT::template __is<double, 2>) {
        return _mm_movemask_pd(__a);
    } else if constexpr (__have_avx && _TVT::template __is<double, 4>) {
        return _mm256_movemask_pd(__a);
    } else if constexpr (__have_sse2 && sizeof(_Tp) == 16) {
        return _mm_movemask_epi8(__a);
    } else if constexpr (__have_avx2 && sizeof(_Tp) == 32) {
        return _mm256_movemask_epi8(__a);
    } else {
        __assert_unreachable<_Tp>();
    }
}

template <typename _Tp, typename _TVT = __vector_traits<_Tp>>
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST int movemask_epi16(_Tp __a)
{
    static_assert(std::is_integral_v<typename _TVT::value_type>);
    if constexpr(__have_avx512bw_vl && sizeof(_Tp) == 16) {
        return _mm_cmp_epi16_mask(__a, __m128i(), _MM_CMPINT_NE);
    } else if constexpr(__have_avx512bw_vl && sizeof(_Tp) == 32) {
        return _mm256_cmp_epi16_mask(__a, __m256i(), _MM_CMPINT_NE);
    } else if constexpr(sizeof(_Tp) == 32) {
        return _mm_movemask_epi8(_mm_packs_epi16(__lo128(__a), __hi128(__a)));
    } else {
        static_assert(sizeof(_Tp) == 16);
        return _mm_movemask_epi8(_mm_packs_epi16(__a, __m128i()));
    }
}

// }}}
#if _GLIBCXX_SIMD_HAVE_SSE_ABI
// __bool_storage_member_type{{{
#if _GLIBCXX_SIMD_HAVE_AVX512F
template <size_t _Size>
struct __bool_storage_member_type
{
  static_assert((_Size & (_Size - 1)) == 0,
		"This trait may only be used for non-power-of-2 sizes. "
		"Power-of-2 sizes must be specialized.");
  using type =
    typename __bool_storage_member_type<__next_power_of_2(_Size)>::type;
};
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
#if _GLIBCXX_SIMD_HAVE_AVX512F
template <>
struct __intrinsic_type<double, 64, void>
{
  using type [[__gnu__::__vector_size__(64)]] = double;
};
template <>
struct __intrinsic_type<float, 64, void>
{
  using type [[__gnu__::__vector_size__(64)]] = float;
};
template <typename _Tp>
struct __intrinsic_type<_Tp, 64, enable_if_t<is_integral_v<_Tp>>>
{
  using type [[__gnu__::__vector_size__(64)]] = long long int;
};
#endif  // _GLIBCXX_SIMD_HAVE_AVX512F

#if _GLIBCXX_SIMD_HAVE_AVX
template <>
struct __intrinsic_type<double, 32, void>
{
  using type [[__gnu__::__vector_size__(32)]] = double;
};
template <>
struct __intrinsic_type<float, 32, void>
{
  using type [[__gnu__::__vector_size__(32)]] = float;
};
template <typename _Tp>
struct __intrinsic_type<_Tp, 32, enable_if_t<is_integral_v<_Tp>>>
{
  using type [[__gnu__::__vector_size__(32)]] = long long int;
};
#endif  // _GLIBCXX_SIMD_HAVE_AVX

#if _GLIBCXX_SIMD_HAVE_SSE
template <>
struct __intrinsic_type<float, 16, void>
{
  using type [[__gnu__::__vector_size__(16)]] = float;
};
template <>
struct __intrinsic_type<float, 8, void>
{
  using type [[__gnu__::__vector_size__(16)]] = float;
};
template <>
struct __intrinsic_type<float, 4, void>
{
  using type [[__gnu__::__vector_size__(16)]] = float;
};
template <>
struct __intrinsic_type<double, 16, void>
{
  using type [[__gnu__::__vector_size__(16)]] = double;
};
template <>
struct __intrinsic_type<double, 8, void>
{
  using type [[__gnu__::__vector_size__(16)]] = double;
};
template <typename _Tp, size_t _Bytes>
struct __intrinsic_type<
  _Tp,
  _Bytes,
  enable_if_t<(_Bytes <= 16 && _Bytes >= sizeof(_Tp) &&
	       ((_Bytes - 1) & _Bytes) == 0 && is_integral_v<_Tp>)>>
{
  using type [[__gnu__::__vector_size__(16)]] = long long int;
};
#endif  // _GLIBCXX_SIMD_HAVE_SSE

// }}}
// _(Sse|Avx|Avx512)(Simd|Mask)Member{{{
template <typename _Tp> using _SseSimdMember = _SimdWrapper16<_Tp>;
template <typename _Tp> using _SseMaskMember = _SimdWrapper16<_Tp>;

template <typename _Tp> using _AvxSimdMember = _SimdWrapper32<_Tp>;
template <typename _Tp> using _AvxMaskMember = _SimdWrapper32<_Tp>;

template <typename _Tp> using _Avx512SimdMember = _SimdWrapper64<_Tp>;
template <typename _Tp> using _Avx512MaskMember = _SimdWrapper<bool, 64 / sizeof(_Tp)>;
template <size_t _N> using _Avx512MaskMemberN = _SimdWrapper<bool, _N>;

//}}}
#endif  // _GLIBCXX_SIMD_HAVE_SSE_ABI
// _SimdWrapper<bool>{{{1
template <size_t _Width>
struct _SimdWrapper<
  bool,
  _Width,
  std::void_t<typename __bool_storage_member_type<_Width>::type>>
{
  using _BuiltinType = typename __bool_storage_member_type<_Width>::type;
  using value_type   = bool;
  static constexpr size_t _S_width = _Width;

  _GLIBCXX_SIMD_INTRINSIC constexpr _SimdWrapper() = default;
  _GLIBCXX_SIMD_INTRINSIC constexpr _SimdWrapper(_BuiltinType __k)
  : _M_data(__k){};

  _GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_PURE
			  operator const _BuiltinType&() const
  {
    return _M_data;
  }
  _GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_PURE operator _BuiltinType&()
  {
    return _M_data;
  }

  _GLIBCXX_SIMD_INTRINSIC _BuiltinType __intrin() const { return _M_data; }

  _GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_PURE value_type
					     operator[](size_t __i) const
  {
    return _M_data & (_BuiltinType(1) << __i);
  }
  _GLIBCXX_SIMD_INTRINSIC void set(size_t __i, value_type __x)
  {
    if (__x)
      _M_data |= (_BuiltinType(1) << __i);
    else
      _M_data &= ~(_BuiltinType(1) << __i);
  }

  _BuiltinType _M_data;
};

// _SimdWrapperBase{{{1
template <
  typename _Tp,
  size_t _Width,
  typename _RegisterType = __vector_type_t<_Tp, _Width>,
  bool                   = std::disjunction_v<
    std::is_same<__vector_type_t<_Tp, _Width>, __intrinsic_type_t<_Tp, _Width>>,
    std::is_same<_RegisterType, __intrinsic_type_t<_Tp, _Width>>>>
struct _SimdWrapperBase;

template <typename _Tp, size_t _Width, typename _RegisterType>
struct _SimdWrapperBase<_Tp, _Width, _RegisterType, true>
{
  _RegisterType _M_data;
  _GLIBCXX_SIMD_INTRINSIC constexpr _SimdWrapperBase() = default;
  _GLIBCXX_SIMD_INTRINSIC constexpr _SimdWrapperBase(
    __vector_type_t<_Tp, _Width> __x)
  : _M_data(reinterpret_cast<_RegisterType>(__x))
  {
  }
};

template <typename _Tp, size_t _Width, typename _RegisterType>
struct _SimdWrapperBase<_Tp, _Width, _RegisterType, false>
{
  using _IntrinType = __intrinsic_type_t<_Tp, _Width>;
  _RegisterType _M_data;

  _GLIBCXX_SIMD_INTRINSIC constexpr _SimdWrapperBase() = default;
  _GLIBCXX_SIMD_INTRINSIC constexpr _SimdWrapperBase(
    __vector_type_t<_Tp, _Width> __x)
  : _M_data(reinterpret_cast<_RegisterType>(__x))
  {
  }
  _GLIBCXX_SIMD_INTRINSIC constexpr _SimdWrapperBase(_IntrinType __x)
  : _M_data(reinterpret_cast<_RegisterType>(__x))
  {
  }
};

// }}}
// _SimdWrapper{{{
template <typename _Tp, size_t _Width>
struct _SimdWrapper<_Tp, _Width,
               std::void_t<__vector_type_t<_Tp, _Width>, __intrinsic_type_t<_Tp, _Width>>>
    : _SimdWrapperBase<_Tp, _Width> {
    static_assert(__is_vectorizable_v<_Tp>);
    static_assert(_Width >= 2);  // 1 doesn't make sense, use _Tp directly then
    using _BuiltinType = __vector_type_t<_Tp, _Width>;
    using value_type = _Tp;
    static constexpr size_t _S_width = _Width;

    _GLIBCXX_SIMD_INTRINSIC constexpr _SimdWrapper() = default;
    template <typename _U, typename = decltype(_SimdWrapperBase<_Tp, _Width>(std::declval<_U>()))>
    _GLIBCXX_SIMD_INTRINSIC constexpr _SimdWrapper(_U &&__x) : _SimdWrapperBase<_Tp, _Width>(std::forward<_U>(__x))
    {
    }
    // I want to use ctor inheritance, but it breaks always_inline. Having a function that
    // does a single movaps is stupid.
    //using _SimdWrapperBase<_Tp, _Width>::_SimdWrapperBase;
    using _SimdWrapperBase<_Tp, _Width>::_M_data;

    template <typename... _As,
              typename = enable_if_t<((std::is_same_v<simd_abi::scalar, _As> && ...) &&
                                        sizeof...(_As) <= _Width)>>
    _GLIBCXX_SIMD_INTRINSIC constexpr operator _SimdTuple<_Tp, _As...>() const
    {
        const auto &dd = _M_data;  // workaround for GCC7 ICE
        return __generate_from_n_evaluations<sizeof...(_As), _SimdTuple<_Tp, _As...>>(
            [&](auto __i) { return dd[int(__i)]; });
    }

    _GLIBCXX_SIMD_INTRINSIC constexpr operator const _BuiltinType &() const { return _M_data; }
    _GLIBCXX_SIMD_INTRINSIC constexpr operator _BuiltinType &() { return _M_data; }

    _GLIBCXX_SIMD_INTRINSIC constexpr _Tp operator[](size_t __i) const { return _M_data[__i]; }

    _GLIBCXX_SIMD_INTRINSIC void set(size_t __i, _Tp __x) { _M_data[__i] = __x; }
};

// }}}
// _ToWrapper {{{
template <typename _Tp> class _ToWrapper
{
    _Tp _M_data;

public:
    constexpr _ToWrapper(_Tp __x) : _M_data(__x) {}

    template <size_t _N> constexpr operator _SimdWrapper<bool, _N>() const
    {
        static_assert(std::is_integral_v<_Tp>);
        return static_cast<__bool_storage_member_type_t<_N>>(_M_data);
    }

    template <typename _U, size_t _N> constexpr operator _SimdWrapper<_U, _N>() const
    {
        static_assert(__is_vector_type_v<_Tp>);
        static_assert(sizeof(__vector_type_t<_U, _N>) == sizeof(_Tp));
        return {reinterpret_cast<__vector_type_t<_U, _N>>(_M_data)};
    }
};

// }}}
// __wrapper_bitcast{{{
template <typename _Tp, typename _U, size_t _M, size_t _N = sizeof(_U) * _M / sizeof(_Tp)>
_GLIBCXX_SIMD_INTRINSIC constexpr _SimdWrapper<_Tp, _N> __wrapper_bitcast(_SimdWrapper<_U, _M> __x)
{
    static_assert(sizeof(__vector_type_t<_Tp, _N>) == sizeof(__vector_type_t<_U, _M>));
    return reinterpret_cast<__vector_type_t<_Tp, _N>>(__x._M_data);
}

// }}}
// __make_wrapper{{{
template <typename _Tp, typename... _Args>
_GLIBCXX_SIMD_INTRINSIC constexpr _SimdWrapper<_Tp, sizeof...(_Args)> __make_wrapper(_Args &&... args)
{
    return {typename _SimdWrapper<_Tp, sizeof...(_Args)>::_BuiltinType{static_cast<_Tp>(args)...}};
}

// }}}
// __generate_wrapper{{{
template <typename _Tp, size_t _N, typename _G>
_GLIBCXX_SIMD_INTRINSIC constexpr _SimdWrapper<_Tp, _N> __generate_wrapper(_G &&__gen)
{
    return __generate_builtin<_Tp, _N>(std::forward<_G>(__gen));
}

//}}}
// __fallback_abi_for_long_double {{{
template <typename _Tp, typename _A0, typename _A1> struct __fallback_abi_for_long_double {
    using type = _A0;
};
template <typename _A0, typename _A1> struct __fallback_abi_for_long_double<long double, _A0, _A1> {
    using type = _A1;
};
template <typename _Tp, typename _A0, typename _A1>
using __fallback_abi_for_long_double_t =
    typename __fallback_abi_for_long_double<_Tp, _A0, _A1>::type;
// }}}

namespace simd_abi
{
// most of simd_abi is defined in simd_detail.h
template <typename _Tp> inline constexpr int max_fixed_size = 32;
// compatible {{{
#if defined __x86_64__
template <typename _Tp>
using compatible = __fallback_abi_for_long_double_t<_Tp, __sse, scalar>;
#elif defined _GLIBCXX_SIMD_IS_AARCH64
template <typename _Tp>
using compatible = __fallback_abi_for_long_double_t<_Tp, __neon, scalar>;
#else
template <typename> using compatible = scalar;
#endif

// }}}
// native {{{
#if _GLIBCXX_SIMD_HAVE_FULL_AVX512_ABI
template <typename _Tp> using native = __fallback_abi_for_long_double_t<_Tp, __avx512, scalar>;
#elif _GLIBCXX_SIMD_HAVE_AVX512_ABI
template <typename _Tp>
using native =
    std::conditional_t<(sizeof(_Tp) >= 4),
                       __fallback_abi_for_long_double_t<_Tp, __avx512, scalar>, __avx>;
#elif _GLIBCXX_SIMD_HAVE_FULL_AVX_ABI
template <typename _Tp> using native = __fallback_abi_for_long_double_t<_Tp, __avx, scalar>;
#elif _GLIBCXX_SIMD_HAVE_AVX_ABI
template <typename _Tp>
using native =
    std::conditional_t<std::is_floating_point<_Tp>::value,
                       __fallback_abi_for_long_double_t<_Tp, __avx, scalar>, __sse>;
#elif _GLIBCXX_SIMD_HAVE_FULL_SSE_ABI
template <typename _Tp> using native = __fallback_abi_for_long_double_t<_Tp, __sse, scalar>;
#elif _GLIBCXX_SIMD_HAVE_SSE_ABI
template <typename _Tp>
using native = std::conditional_t<std::is_same<float, _Tp>::value, __sse, scalar>;
#elif defined _GLIBCXX_SIMD_HAVE_FULL_NEON_ABI
template <typename _Tp> using native = __fallback_abi_for_long_double_t<_Tp, __neon, scalar>;
#else
template <typename> using native = scalar;
#endif

// }}}
// __default_abi {{{
#if defined _GLIBCXX_SIMD_DEFAULT_ABI
template <typename _Tp> using __default_abi = _GLIBCXX_SIMD_DEFAULT_ABI<_Tp>;
#else
template <typename _Tp> using __default_abi = compatible<_Tp>;
#endif

// }}}
}  // namespace simd_abi

// traits {{{1
// is_abi_tag {{{2
template <typename _Tp, typename = std::void_t<>> struct is_abi_tag : false_type {
};
template <typename _Tp>
struct is_abi_tag<_Tp, std::void_t<typename _Tp::is_valid_abi_tag>>
    : public _Tp::is_valid_abi_tag {
};
template <typename _Tp> inline constexpr bool is_abi_tag_v = is_abi_tag<_Tp>::value;

// is_simd(_mask) {{{2
template <typename _Tp> struct is_simd : public false_type {};
template <typename _Tp> inline constexpr bool is_simd_v = is_simd<_Tp>::value;

template <typename _Tp> struct is_simd_mask : public false_type {};
template <typename _Tp> inline constexpr bool is_simd_mask_v = is_simd_mask<_Tp>::value;

// simd_size {{{2
template <typename _Tp, typename _Abi, typename = void> struct __simd_size_impl {
};
template <typename _Tp, typename _Abi>
struct __simd_size_impl<_Tp, _Abi,
                        enable_if_t<std::conjunction_v<
                            __is_vectorizable<_Tp>, std::experimental::is_abi_tag<_Abi>>>>
    : _SizeConstant<_Abi::template size<_Tp>> {
};

template <typename _Tp, typename _Abi = simd_abi::__default_abi<_Tp>>
struct simd_size : __simd_size_impl<_Tp, _Abi> {
};
template <typename _Tp, typename _Abi = simd_abi::__default_abi<_Tp>>
inline constexpr size_t simd_size_v = simd_size<_Tp, _Abi>::value;

// simd_abi::deduce {{{2
template <typename _Tp, std::size_t _N, typename = void> struct __deduce_impl;
namespace simd_abi
{
/**
 * \tparam _Tp    The requested `value_type` for the elements.
 * \tparam _N    The requested number of elements.
 * \tparam _Abis This parameter is ignored, since this implementation cannot make any use
 *              of it. Either __a good native ABI is matched and used as `type` alias, or
 *              the `fixed_size<_N>` ABI is used, which internally is built from the best
 *              matching native ABIs.
 */
template <typename _Tp, std::size_t _N, typename...>
struct deduce : std::experimental::__deduce_impl<_Tp, _N> {};

template <typename _Tp, size_t _N, typename... _Abis>
using deduce_t = typename deduce<_Tp, _N, _Abis...>::type;
}  // namespace simd_abi

// }}}2
// rebind_simd {{{2
template <typename _Tp, typename _V> struct rebind_simd;
template <typename _Tp, typename _U, typename _Abi> struct rebind_simd<_Tp, simd<_U, _Abi>> {
    using type = simd<_Tp, simd_abi::deduce_t<_Tp, simd_size_v<_U, _Abi>, _Abi>>;
};
template <typename _Tp, typename _U, typename _Abi> struct rebind_simd<_Tp, simd_mask<_U, _Abi>> {
    using type = simd_mask<_Tp, simd_abi::deduce_t<_Tp, simd_size_v<_U, _Abi>, _Abi>>;
};
template <typename _Tp, typename _V> using rebind_simd_t = typename rebind_simd<_Tp, _V>::type;

// resize_simd {{{2
template <int _N, typename _V> struct resize_simd;
template <int _N, typename _Tp, typename _Abi> struct resize_simd<_N, simd<_Tp, _Abi>> {
    using type = simd<_Tp, simd_abi::deduce_t<_Tp, _N, _Abi>>;
};
template <int _N, typename _Tp, typename _Abi> struct resize_simd<_N, simd_mask<_Tp, _Abi>> {
    using type = simd_mask<_Tp, simd_abi::deduce_t<_Tp, _N, _Abi>>;
};
template <int _N, typename _V> using resize_simd_t = typename resize_simd<_N, _V>::type;

// }}}2
// memory_alignment {{{2
template <typename _Tp, typename _U = typename _Tp::value_type>
struct memory_alignment
    : public _SizeConstant<__next_power_of_2(sizeof(_U) * _Tp::size())> {
};
template <typename _Tp, typename _U = typename _Tp::value_type>
inline constexpr size_t memory_alignment_v = memory_alignment<_Tp, _U>::value;

// class template simd [simd] {{{1
template <typename _Tp, typename _Abi = simd_abi::__default_abi<_Tp>> class simd;
template <typename _Tp, typename _Abi> struct is_simd<simd<_Tp, _Abi>> : public true_type {};
template <typename _Tp> using native_simd = simd<_Tp, simd_abi::native<_Tp>>;
template <typename _Tp, int _N> using fixed_size_simd = simd<_Tp, simd_abi::fixed_size<_N>>;
template <typename _Tp, size_t _N> using __deduced_simd = simd<_Tp, simd_abi::deduce_t<_Tp, _N>>;

// class template simd_mask [simd_mask] {{{1
template <typename _Tp, typename _Abi = simd_abi::__default_abi<_Tp>> class simd_mask;
template <typename _Tp, typename _Abi> struct is_simd_mask<simd_mask<_Tp, _Abi>> : public true_type {};
template <typename _Tp> using native_simd_mask = simd_mask<_Tp, simd_abi::native<_Tp>>;
template <typename _Tp, int _N> using fixed_size_simd_mask = simd_mask<_Tp, simd_abi::fixed_size<_N>>;
template <typename _Tp, size_t _N>
using __deduced_simd_mask = simd_mask<_Tp, simd_abi::deduce_t<_Tp, _N>>;

template <typename _Tp, typename _Abi>
struct __get_impl<std::experimental::simd_mask<_Tp, _Abi>>
{
  using type = typename _SimdTraits<_Tp, _Abi>::_MaskImpl;
};
template <typename _Tp, typename _Abi>
struct __get_impl<std::experimental::simd<_Tp, _Abi>>
{
  using type = typename _SimdTraits<_Tp, _Abi>::_SimdImpl;
};

template <typename _Tp, typename _Abi>
struct __get_traits<std::experimental::simd_mask<_Tp, _Abi>>
{
  using type = _SimdTraits<_Tp, _Abi>;
};
template <typename _Tp, typename _Abi>
struct __get_traits<std::experimental::simd<_Tp, _Abi>>
{
  using type = _SimdTraits<_Tp, _Abi>;
};

// casts [simd.casts] {{{1
// static_simd_cast {{{2
template <typename _Tp, typename _U, typename _A, bool = is_simd_v<_Tp>, typename = void>
struct __static_simd_cast_return_type;

template <typename _Tp, typename _A0, typename _U, typename _A>
struct __static_simd_cast_return_type<simd_mask<_Tp, _A0>, _U, _A, false, void>
    : __static_simd_cast_return_type<simd<_Tp, _A0>, _U, _A> {
};

template <typename _Tp, typename _U, typename _A>
struct __static_simd_cast_return_type<_Tp, _U, _A, true,
                                    enable_if_t<_Tp::size() == simd_size_v<_U, _A>>> {
    using type = _Tp;
};

template <typename _Tp, typename _A>
struct __static_simd_cast_return_type<_Tp, _Tp, _A, false,
#ifdef _GLIBCXX_SIMD_FIX_P2TS_ISSUE66
                                    enable_if_t<__is_vectorizable_v<_Tp>>
#else
                                    void
#endif
                                    > {
    using type = simd<_Tp, _A>;
};

template <typename _Tp, typename = void> struct __safe_make_signed {
    using type = _Tp;
};
template <typename _Tp> struct __safe_make_signed<_Tp, enable_if_t<std::is_integral_v<_Tp>>> {
    // the extra make_unsigned_t is because of PR85951
    using type = std::make_signed_t<std::make_unsigned_t<_Tp>>;
};
template <typename _Tp> using safe_make_signed_t = typename __safe_make_signed<_Tp>::type;

template <typename _Tp, typename _U, typename _A>
struct __static_simd_cast_return_type<_Tp, _U, _A, false,
#ifdef _GLIBCXX_SIMD_FIX_P2TS_ISSUE66
                                    enable_if_t<__is_vectorizable_v<_Tp>>
#else
                                    void
#endif
                                    > {
    using type =
        std::conditional_t<(std::is_integral_v<_U> && std::is_integral_v<_Tp> &&
#ifndef _GLIBCXX_SIMD_FIX_P2TS_ISSUE65
                            std::is_signed_v<_U> != std::is_signed_v<_Tp> &&
#endif
                            std::is_same_v<safe_make_signed_t<_U>, safe_make_signed_t<_Tp>>),
                           simd<_Tp, _A>, fixed_size_simd<_Tp, simd_size_v<_U, _A>>>;
};

template <typename _To, typename, typename, typename _Native, typename _From>
_GLIBCXX_SIMD_INTRINSIC _To __mask_cast_impl(const _Native *, const _From &__x)
{
    static_assert(std::is_same_v<_Native, typename __get_traits_t<_To>::_MaskMember>);
    if constexpr (std::is_same_v<_Native, bool>) {
        return {std::experimental::__private_init, bool(__x[0])};
    } else if constexpr (std::is_same_v<_From, bool>) {
        _To __r{};
        __r[0] = __x;
        return __r;
    } else {
        return {__private_init,
                __convert_mask<typename __get_traits_t<_To>::_MaskMember>(__x)};
    }
}
template <typename _To, typename, typename, typename _Native, size_t _N>
_GLIBCXX_SIMD_INTRINSIC _To __mask_cast_impl(const _Native *, const std::bitset<_N> &__x)
{
  static_assert(_N <= sizeof(_ULLong) * CHAR_BIT ||
		  _To::size() <= sizeof(_ULLong) * CHAR_BIT,
		"bug in std::experimental::(static|resizing)_simd_cast");
  return {std::experimental::__bitset_init, __x.to_ullong()};
}
template <typename _To, typename, typename>
_GLIBCXX_SIMD_INTRINSIC _To __mask_cast_impl(const bool *, bool __x)
{
    return _To(__x);
}
template <typename _To, typename, typename>
_GLIBCXX_SIMD_INTRINSIC _To __mask_cast_impl(const std::bitset<1> *, bool __x)
{
    return _To(__x);
}
template <typename _To, typename _Tp, typename _Abi, size_t _N, typename _From>
_GLIBCXX_SIMD_INTRINSIC _To __mask_cast_impl(const std::bitset<_N> *, const _From &__x)
{
    return {std::experimental::__private_init, __vector_to_bitset(__x)};
}
template <typename _To, typename, typename, size_t _N>
_GLIBCXX_SIMD_INTRINSIC _To __mask_cast_impl(const std::bitset<_N> *, const std::bitset<_N> &__x)
{
    return {std::experimental::__private_init, __x};
}

template <typename _Tp, typename _U, typename _A,
          typename _R = typename __static_simd_cast_return_type<_Tp, _U, _A>::type>
_GLIBCXX_SIMD_INTRINSIC _R static_simd_cast(const simd<_U, _A> &__x)
{
    if constexpr(std::is_same<_R, simd<_U, _A>>::value) {
        return __x;
    } else {
        _SimdConverter<_U, _A, typename _R::value_type, typename _R::abi_type> __c;
        return _R(__private_init, __c(__data(__x)));
    }
}

template <typename _Tp, typename _U, typename _A,
          typename _R = typename __static_simd_cast_return_type<_Tp, _U, _A>::type>
_GLIBCXX_SIMD_INTRINSIC typename _R::mask_type static_simd_cast(const simd_mask<_U, _A> &__x)
{
    using _RM = typename _R::mask_type;
    if constexpr(std::is_same<_RM, simd_mask<_U, _A>>::value) {
        return __x;
    } else {
        using __traits = _SimdTraits<typename _R::value_type, typename _R::abi_type>;
        const typename __traits::_MaskMember *tag = nullptr;
        return __mask_cast_impl<_RM, _U, _A>(tag, __data(__x));
    }
}

// simd_cast {{{2
template <typename _Tp, typename _U, typename _A, typename _To = __value_type_or_identity_t<_Tp>>
_GLIBCXX_SIMD_INTRINSIC auto simd_cast(const simd<_ValuePreserving<_U, _To>, _A> &__x)
    ->decltype(static_simd_cast<_Tp>(__x))
{
    return static_simd_cast<_Tp>(__x);
}

template <typename _Tp, typename _U, typename _A, typename _To = __value_type_or_identity_t<_Tp>>
_GLIBCXX_SIMD_INTRINSIC auto simd_cast(const simd_mask<_ValuePreserving<_U, _To>, _A> &__x)
    ->decltype(static_simd_cast<_Tp>(__x))
{
    return static_simd_cast<_Tp>(__x);
}

namespace __proposed
{
template <typename _Tp, typename _U, typename _A>
_GLIBCXX_SIMD_INTRINSIC _Tp resizing_simd_cast(const simd_mask<_U, _A> &__x)
{
    static_assert(is_simd_mask_v<_Tp>);
    if constexpr (std::is_same_v<_Tp, simd_mask<_U, _A>>) {
        return __x;
    } else {
        using __traits = _SimdTraits<typename _Tp::simd_type::value_type, typename _Tp::abi_type>;
        const typename __traits::_MaskMember *tag = nullptr;
        return __mask_cast_impl<_Tp, _U, _A>(tag, __data(__x));
    }
}
}  // namespace __proposed

// to_fixed_size {{{2
template <typename _Tp, int _N>
_GLIBCXX_SIMD_INTRINSIC fixed_size_simd<_Tp, _N> to_fixed_size(const fixed_size_simd<_Tp, _N> &__x)
{
    return __x;
}

template <typename _Tp, int _N>
_GLIBCXX_SIMD_INTRINSIC fixed_size_simd_mask<_Tp, _N> to_fixed_size(const fixed_size_simd_mask<_Tp, _N> &__x)
{
    return __x;
}

template <typename _Tp, typename _A> _GLIBCXX_SIMD_INTRINSIC auto to_fixed_size(const simd<_Tp, _A> &__x)
{
    return simd<_Tp, simd_abi::fixed_size<simd_size_v<_Tp, _A>>>(
        [&__x](auto __i) { return __x[__i]; });
}

template <typename _Tp, typename _A> _GLIBCXX_SIMD_INTRINSIC auto to_fixed_size(const simd_mask<_Tp, _A> &__x)
{
    constexpr int _N = simd_mask<_Tp, _A>::size();
    fixed_size_simd_mask<_Tp, _N> __r;
    __execute_n_times<_N>([&](auto __i) { __r[__i] = __x[__i]; });
    return __r;
}

// to_native {{{2
template <typename _Tp, int _N>
_GLIBCXX_SIMD_INTRINSIC enable_if_t<(_N == native_simd<_Tp>::size()), native_simd<_Tp>>
to_native(const fixed_size_simd<_Tp, _N> &__x)
{
    alignas(memory_alignment_v<native_simd<_Tp>>) _Tp __mem[_N];
    __x.copy_to(__mem, vector_aligned);
    return {__mem, vector_aligned};
}

template <typename _Tp, size_t _N>
_GLIBCXX_SIMD_INTRINSIC enable_if_t<(_N == native_simd_mask<_Tp>::size()), native_simd_mask<_Tp>> to_native(
    const fixed_size_simd_mask<_Tp, _N> &__x)
{
    return native_simd_mask<_Tp>([&](auto __i) { return __x[__i]; });
}

// to_compatible {{{2
template <typename _Tp, size_t _N>
_GLIBCXX_SIMD_INTRINSIC enable_if_t<(_N == simd<_Tp>::size()), simd<_Tp>> to_compatible(
    const simd<_Tp, simd_abi::fixed_size<_N>> &__x)
{
    alignas(memory_alignment_v<simd<_Tp>>) _Tp __mem[_N];
    __x.copy_to(__mem, vector_aligned);
    return {__mem, vector_aligned};
}

template <typename _Tp, size_t _N>
_GLIBCXX_SIMD_INTRINSIC enable_if_t<(_N == simd_mask<_Tp>::size()), simd_mask<_Tp>> to_compatible(
    const simd_mask<_Tp, simd_abi::fixed_size<_N>> &__x)
{
    return simd_mask<_Tp>([&](auto __i) { return __x[__i]; });
}

// simd_reinterpret_cast {{{2
template <typename _To, size_t _N> _GLIBCXX_SIMD_INTRINSIC _To __simd_reinterpret_cast_impl(std::bitset<_N> __x)
{
    return {__bitset_init, __x};
}

template <typename _To, typename _Tp, size_t _N>
_GLIBCXX_SIMD_INTRINSIC _To __simd_reinterpret_cast_impl(_SimdWrapper<_Tp, _N> __x)
{
    return {__private_init, __x};
}

namespace __proposed
{
template <typename _To,
	  typename _Tp,
	  typename _A,
	  typename = enable_if_t<sizeof(_To) == sizeof(simd<_Tp, _A>) &&
			      (is_simd_v<_To> || is_simd_mask_v<_To>)>>
_GLIBCXX_SIMD_INTRINSIC _To simd_reinterpret_cast(const simd<_Tp, _A>& __x)
{
  _To __r;
  std::memcpy(&__data(__r), &__data(__x), sizeof(_To));
  return __r;
}

template <typename _To, typename _Tp, typename _A,
          typename = enable_if_t<(is_simd_v<_To> || is_simd_mask_v<_To>)>>
_GLIBCXX_SIMD_INTRINSIC _To simd_reinterpret_cast(const simd_mask<_Tp, _A> &__x)
{
    return std::experimental::__simd_reinterpret_cast_impl<_To>(__data(__x));
    //return reinterpret_cast<const _To &>(__x);
}
}  // namespace __proposed

// masked assignment [simd_mask.where] {{{1

// where_expression {{{1
template <class _M, class _Tp> class const_where_expression  //{{{2
{
    using _V = _Tp;
    static_assert(std::is_same_v<_V, std::decay_t<_Tp>>);
    struct Wrapper {
        using value_type = _V;
    };

protected:
    using value_type =
        typename std::conditional_t<std::is_arithmetic<_V>::value, Wrapper, _V>::value_type;
    _GLIBCXX_SIMD_INTRINSIC friend const _M &__get_mask(const const_where_expression &__x) { return __x.__k; }
    _GLIBCXX_SIMD_INTRINSIC friend const _Tp &__get_lvalue(const const_where_expression &__x) { return __x._M_value; }
    const _M &__k;
    _Tp &_M_value;

public:
    const_where_expression(const const_where_expression &) = delete;
    const_where_expression &operator=(const const_where_expression &) = delete;

    _GLIBCXX_SIMD_INTRINSIC const_where_expression(const _M &kk, const _Tp &dd) : __k(kk), _M_value(const_cast<_Tp &>(dd)) {}

    _GLIBCXX_SIMD_INTRINSIC _V operator-() const &&
    {
        return {__private_init,
                __get_impl_t<_V>::template masked_unary<std::negate>(
                    __data(__k), __data(_M_value))};
    }

    template <class _U, class _Flags>
    [[nodiscard]] _GLIBCXX_SIMD_INTRINSIC _V
    copy_from(const _LoadStorePtr<_U, value_type> *__mem, _Flags __f) const &&
    {
        return {__private_init, __get_impl_t<_V>::masked_load(
                                          __data(_M_value), __data(__k), __mem, __f)};
    }

    template <class _U, class _Flags>
    _GLIBCXX_SIMD_INTRINSIC void copy_to(_LoadStorePtr<_U, value_type> *__mem,
                              _Flags __f) const &&
    {
        __get_impl_t<_V>::masked_store(__data(_M_value), __mem, __f, __data(__k));
    }
};

template <class _Tp> class const_where_expression<bool, _Tp>  //{{{2
{
    using _M = bool;
    using _V = _Tp;
    static_assert(std::is_same_v<_V, std::decay_t<_Tp>>);
    struct Wrapper {
        using value_type = _V;
    };

protected:
    using value_type =
        typename std::conditional_t<std::is_arithmetic<_V>::value, Wrapper, _V>::value_type;
    _GLIBCXX_SIMD_INTRINSIC friend const _M &__get_mask(const const_where_expression &__x) { return __x.__k; }
    _GLIBCXX_SIMD_INTRINSIC friend const _Tp &__get_lvalue(const const_where_expression &__x) { return __x._M_value; }
    const bool __k;
    _Tp &_M_value;

public:
    const_where_expression(const const_where_expression &) = delete;
    const_where_expression &operator=(const const_where_expression &) = delete;

    _GLIBCXX_SIMD_INTRINSIC const_where_expression(const bool kk, const _Tp &dd) : __k(kk), _M_value(const_cast<_Tp &>(dd)) {}

    _GLIBCXX_SIMD_INTRINSIC _V operator-() const && { return __k ? -_M_value : _M_value; }

    template <class _U, class _Flags>
    [[nodiscard]] _GLIBCXX_SIMD_INTRINSIC _V
    copy_from(const _LoadStorePtr<_U, value_type> *__mem, _Flags) const &&
    {
        return __k ? static_cast<_V>(__mem[0]) : _M_value;
    }

    template <class _U, class _Flags>
    _GLIBCXX_SIMD_INTRINSIC void copy_to(_LoadStorePtr<_U, value_type> *__mem,
                              _Flags) const &&
    {
        if (__k) {
            __mem[0] = _M_value;
        }
    }
};

// where_expression {{{2
template <class _M, class _Tp>
class where_expression : public const_where_expression<_M, _Tp>
{
    static_assert(!std::is_const<_Tp>::value, "where_expression may only be instantiated with __a non-const _Tp parameter");
    using typename const_where_expression<_M, _Tp>::value_type;
    using const_where_expression<_M, _Tp>::__k;
    using const_where_expression<_M, _Tp>::_M_value;
    static_assert(std::is_same<typename _M::abi_type, typename _Tp::abi_type>::value, "");
    static_assert(_M::size() == _Tp::size(), "");

    _GLIBCXX_SIMD_INTRINSIC friend _Tp &__get_lvalue(where_expression &__x) { return __x._M_value; }
public:
    where_expression(const where_expression &) = delete;
    where_expression &operator=(const where_expression &) = delete;

    _GLIBCXX_SIMD_INTRINSIC where_expression(const _M &kk, _Tp &dd)
        : const_where_expression<_M, _Tp>(kk, dd)
    {
    }

    template <class _U> _GLIBCXX_SIMD_INTRINSIC void operator=(_U &&__x) &&
    {
        std::experimental::__get_impl_t<_Tp>::masked_assign(
            __data(__k), __data(_M_value),
            __to_value_type_or_member_type<_Tp>(std::forward<_U>(__x)));
    }

#define _GLIBCXX_SIMD_OP_(op_, name_)                                                    \
    template <class _U> _GLIBCXX_SIMD_INTRINSIC void operator op_##=(_U&& __x)&&         \
    {                                                                                    \
        std::experimental::__get_impl_t<_Tp>::template __masked_cassign<name_>(           \
            __data(__k), __data(_M_value),                                               \
            __to_value_type_or_member_type<_Tp>(std::forward<_U>(__x)));                  \
    }                                                                                    \
    static_assert(true)
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
        __data(_M_value) = __get_impl_t<_Tp>::template masked_unary<__increment>(
            __data(__k), __data(_M_value));
    }
    _GLIBCXX_SIMD_INTRINSIC void operator++(int) &&
    {
        __data(_M_value) = __get_impl_t<_Tp>::template masked_unary<__increment>(
            __data(__k), __data(_M_value));
    }
    _GLIBCXX_SIMD_INTRINSIC void operator--() &&
    {
        __data(_M_value) = __get_impl_t<_Tp>::template masked_unary<__decrement>(
            __data(__k), __data(_M_value));
    }
    _GLIBCXX_SIMD_INTRINSIC void operator--(int) &&
    {
        __data(_M_value) = __get_impl_t<_Tp>::template masked_unary<__decrement>(
            __data(__k), __data(_M_value));
    }

    // intentionally hides const_where_expression::copy_from
    template <class _U, class _Flags>
    _GLIBCXX_SIMD_INTRINSIC void copy_from(const _LoadStorePtr<_U, value_type> *__mem,
                                _Flags __f) &&
    {
        __data(_M_value) =
            __get_impl_t<_Tp>::masked_load(__data(_M_value), __data(__k), __mem, __f);
    }
};

// where_expression<bool> {{{2
template <class _Tp>
class where_expression<bool, _Tp> : public const_where_expression<bool, _Tp>
{
    using _M = bool;
    using typename const_where_expression<_M, _Tp>::value_type;
    using const_where_expression<_M, _Tp>::__k;
    using const_where_expression<_M, _Tp>::_M_value;

public:
    where_expression(const where_expression &) = delete;
    where_expression &operator=(const where_expression &) = delete;

    _GLIBCXX_SIMD_INTRINSIC where_expression(const _M &kk, _Tp &dd)
        : const_where_expression<_M, _Tp>(kk, dd)
    {
    }

#define _GLIBCXX_SIMD_OP_(op_)                                                           \
    template <class _U> _GLIBCXX_SIMD_INTRINSIC void operator op_(_U&& __x)&&            \
    {                                                                                    \
        if (__k) {                                                                       \
            _M_value op_ std::forward<_U>(__x);                                          \
        }                                                                                \
    }                                                                                    \
    static_assert(true)
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
    _GLIBCXX_SIMD_INTRINSIC void copy_from(const _LoadStorePtr<_U, value_type> *__mem,
                                _Flags) &&
    {
        if (__k) {
            _M_value = __mem[0];
        }
    }
};

// where_expression<_M, tuple<...>> {{{2

// where {{{1
template <class _Tp, class _A>
_GLIBCXX_SIMD_INTRINSIC where_expression<simd_mask<_Tp, _A>, simd<_Tp, _A>> where(
    const typename simd<_Tp, _A>::mask_type &__k, simd<_Tp, _A> &__value)
{
    return {__k, __value};
}
template <class _Tp, class _A>
_GLIBCXX_SIMD_INTRINSIC const_where_expression<simd_mask<_Tp, _A>, simd<_Tp, _A>> where(
    const typename simd<_Tp, _A>::mask_type &__k, const simd<_Tp, _A> &__value)
{
    return {__k, __value};
}
template <class _Tp, class _A>
_GLIBCXX_SIMD_INTRINSIC where_expression<simd_mask<_Tp, _A>, simd_mask<_Tp, _A>> where(
    const std::remove_const_t<simd_mask<_Tp, _A>> &__k, simd_mask<_Tp, _A> &__value)
{
    return {__k, __value};
}
template <class _Tp, class _A>
_GLIBCXX_SIMD_INTRINSIC const_where_expression<simd_mask<_Tp, _A>, simd_mask<_Tp, _A>> where(
    const std::remove_const_t<simd_mask<_Tp, _A>> &__k, const simd_mask<_Tp, _A> &__value)
{
    return {__k, __value};
}
template <class _Tp>
_GLIBCXX_SIMD_INTRINSIC where_expression<bool, _Tp> where(_ExactBool __k, _Tp &__value)
{
    return {__k, __value};
}
template <class _Tp>
_GLIBCXX_SIMD_INTRINSIC const_where_expression<bool, _Tp> where(_ExactBool __k, const _Tp &__value)
{
    return {__k, __value};
}
template <class _Tp, class _A> void where(bool __k, simd<_Tp, _A> &__value) = delete;
template <class _Tp, class _A> void where(bool __k, const simd<_Tp, _A> &__value) = delete;

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

        _GLIBCXX_SIMD_INTRINSIC void __next_bit() { __bit = __builtin_ctzl(__mask); }
        _GLIBCXX_SIMD_INTRINSIC void __reset_lsb()
        {
            // 01100100 - 1 = 01100011
            __mask &= (__mask - 1);
            // __asm__("btr %1,%0" : "+r"(__mask) : "r"(__bit));
        }

    public:
        iterator(decltype(__mask) __m) : __mask(__m) { __next_bit(); }
        iterator(const iterator &) = default;
        iterator(iterator &&) = default;

        _GLIBCXX_SIMD_ALWAYS_INLINE size_t operator->() const { return __bit; }
        _GLIBCXX_SIMD_ALWAYS_INLINE size_t operator*() const { return __bit; }

        _GLIBCXX_SIMD_ALWAYS_INLINE iterator &operator++()
        {
            __reset_lsb();
            __next_bit();
            return *this;
        }
        _GLIBCXX_SIMD_ALWAYS_INLINE iterator operator++(int)
        {
            iterator __tmp = *this;
            __reset_lsb();
            __next_bit();
            return __tmp;
        }

        _GLIBCXX_SIMD_ALWAYS_INLINE bool operator==(const iterator &__rhs) const
        {
            return __mask == __rhs.__mask;
        }
        _GLIBCXX_SIMD_ALWAYS_INLINE bool operator!=(const iterator &__rhs) const
        {
            return __mask != __rhs.__mask;
        }
    };

    iterator begin() const { return __bits.to_ullong(); }
    iterator end() const { return 0; }
};

template <class _Tp, class _A>
where_range<simd_size_v<_Tp, _A>> where(const simd_mask<_Tp, _A> &__k)
{
    return __k.__to_bitset();
}

}  // namespace __proposed

// }}}1
// reductions [simd.reductions] {{{1
template <class _Tp, class _Abi, class _BinaryOperation = std::plus<>>
_GLIBCXX_SIMD_INTRINSIC _Tp reduce(const simd<_Tp, _Abi>& __v,
                                  _BinaryOperation __binary_op = _BinaryOperation())
{
    using _V = simd<_Tp, _Abi>;
    return __get_impl_t<_V>::reduce(__v, __binary_op);
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
template <class _Tp, class _A>
_GLIBCXX_SIMD_INTRINSIC constexpr simd<_Tp, _A> min(const simd<_Tp, _A> &__a, const simd<_Tp, _A> &__b)
{
    return {__private_init,
            _A::_SimdImpl::min(__data(__a), __data(__b))};
}
template <class _Tp, class _A>
_GLIBCXX_SIMD_INTRINSIC constexpr simd<_Tp, _A> max(const simd<_Tp, _A> &__a, const simd<_Tp, _A> &__b)
{
    return {__private_init,
            _A::_SimdImpl::max(__data(__a), __data(__b))};
}
template <class _Tp, class _A>
_GLIBCXX_SIMD_INTRINSIC constexpr std::pair<simd<_Tp, _A>, simd<_Tp, _A>> minmax(const simd<_Tp, _A> &__a,
                                                            const simd<_Tp, _A> &__b)
{
    const auto pair_of_members =
        _A::_SimdImpl::minmax(__data(__a), __data(__b));
    return {simd<_Tp, _A>(__private_init, pair_of_members.first),
            simd<_Tp, _A>(__private_init, pair_of_members.second)};
}
template <class _Tp, class _A>
_GLIBCXX_SIMD_INTRINSIC constexpr simd<_Tp, _A> clamp(const simd<_Tp, _A> &__v, const simd<_Tp, _A> &__lo,
                                 const simd<_Tp, _A> &__hi)
{
    using _Impl = typename _A::_SimdImpl;
    return {__private_init,
            _Impl::min(__data(__hi), _Impl::max(__data(__lo), __data(__v)))};
}

// }}}

namespace __proposed
{
// shuffle {{{1
template <int _Stride, int _Offset = 0> struct strided {
    static constexpr int _S_stride = _Stride;
    static constexpr int _S_offset = _Offset;
    template <class _Tp, class _A>
    using __shuffle_return_type = simd<
        _Tp, simd_abi::deduce_t<_Tp, (simd_size_v<_Tp, _A> - _Offset + _Stride - 1) / _Stride, _A>>;
    // alternative, always use fixed_size:
    // fixed_size_simd<_Tp, (simd_size_v<_Tp, _A> - _Offset + _Stride - 1) / _Stride>;
    template <class _Tp> static constexpr auto __src_index(_Tp __dst_index)
    {
        return _Offset + __dst_index * _Stride;
    }
};

// SFINAE for the return type ensures _P is a type that provides the alias template member
// __shuffle_return_type and the static member function __src_index
template <class _P, class _Tp, class _A,
          class _R = typename _P::template __shuffle_return_type<_Tp, _A>,
          class = decltype(_P::__src_index(std::experimental::_SizeConstant<0>()))>
_GLIBCXX_SIMD_INTRINSIC _R shuffle(const simd<_Tp, _A> &__x)
{
    return _R([&__x](auto __i) { return __x[_P::__src_index(__i)]; });
}

// }}}1
}  // namespace __proposed

template <size_t... _Sizes, class _Tp, class _A,
          class = enable_if_t<((_Sizes + ...) == simd<_Tp, _A>::size())>>
inline std::tuple<simd<_Tp, simd_abi::deduce_t<_Tp, _Sizes>>...> split(const simd<_Tp, _A> &);

// __extract_part {{{
template <size_t _Index, size_t _Total, class _Tp, size_t _N>
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST
    __vector_type_t<_Tp, std::max(16 / sizeof(_Tp), _N / _Total)>
        __extract_part(_SimdWrapper<_Tp, _N>);
template <int Index, int Parts, class _Tp, class _A0, class... _As>
auto __extract_part(const _SimdTuple<_Tp, _A0, _As...> &__x);

// }}}
// __size_list {{{
template <size_t V0, size_t... Values> struct __size_list {
    static constexpr size_t size = sizeof...(Values) + 1;

    template <size_t _I> static constexpr size_t at(_SizeConstant<_I> = {})
    {
        if constexpr (_I == 0) {
            return V0;
        } else {
            return __size_list<Values...>::template at<_I - 1>();
        }
    }

    template <size_t _I> static constexpr auto before(_SizeConstant<_I> = {})
    {
        if constexpr (_I == 0) {
            return _SizeConstant<0>();
        } else {
            return _SizeConstant<V0 + __size_list<Values...>::template before<_I - 1>()>();
        }
    }

    template <size_t _N> static constexpr auto pop_front(_SizeConstant<_N> = {})
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
template <class _Tp, size_t _N>
_GLIBCXX_SIMD_INTRINSIC _SimdWrapper<_Tp, _N / 2> __extract_center(_SimdWrapper<_Tp, _N> __x)
{
    if constexpr (__have_avx512f && sizeof(__x) == 64) {
        const auto __intrin = __to_intrin(__x);
        if constexpr (std::is_integral_v<_Tp>) {
            return __vector_bitcast<_Tp>(_mm512_castsi512_si256(_mm512_shuffle_i32x4(
                __intrin, __intrin, 1 + 2 * 0x4 + 2 * 0x10 + 3 * 0x40)));
        } else if constexpr (sizeof(_Tp) == 4) {
            return __vector_bitcast<_Tp>(_mm512_castps512_ps256(_mm512_shuffle_f32x4(
                __intrin, __intrin, 1 + 2 * 0x4 + 2 * 0x10 + 3 * 0x40)));
        } else if constexpr (sizeof(_Tp) == 8) {
            return __vector_bitcast<_Tp>(_mm512_castpd512_pd256(_mm512_shuffle_f64x2(
                __intrin, __intrin, 1 + 2 * 0x4 + 2 * 0x10 + 3 * 0x40)));
        } else {
            __assert_unreachable<_Tp>();
        }
    } else {
        __assert_unreachable<_Tp>();
    }
}
template <class _Tp, class _A>
inline _SimdWrapper<_Tp, simd_size_v<_Tp, _A>> __extract_center(
    const _SimdTuple<_Tp, _A, _A>& __x)
{
    return __concat(__extract<1, 2>(__x.first._M_data), __extract<0, 2>(__x.second.first._M_data));
}
template <class _Tp, class _A>
inline _SimdWrapper<_Tp, simd_size_v<_Tp, _A> / 2> __extract_center(
    const _SimdTuple<_Tp, _A>& __x)
{
    return __extract_center(__x.first);
}

// }}}
// __split_wrapper {{{
template <size_t... _Sizes, class _Tp, class... _As>
auto __split_wrapper(__size_list<_Sizes...>, const _SimdTuple<_Tp, _As...> &__x)
{
    return std::experimental::split<_Sizes...>(
        fixed_size_simd<_Tp, _SimdTuple<_Tp, _As...>::size()>(__private_init, __x));
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
    using _Tp = typename _V::value_type;
    if constexpr (Parts == 1) {
        return {simd_cast<_V>(__x)};
    } else if constexpr (__is_fixed_size_abi_v<_A> &&
                         (std::is_same_v<typename _V::abi_type, simd_abi::scalar> ||
                          (__is_fixed_size_abi_v<typename _V::abi_type> &&
                           sizeof(_V) == sizeof(_Tp) * _V::size()  // _V doesn't have padding
                           ))) {
        // fixed_size -> fixed_size (w/o padding) or scalar
#ifdef _GLIBCXX_SIMD_USE_ALIASING_LOADS
        const __may_alias<_Tp> *const __element_ptr =
            reinterpret_cast<const __may_alias<_Tp> *>(&__data(__x));
        return __generate_from_n_evaluations<Parts, std::array<_V, Parts>>(
            [&](auto __i) { return _V(__element_ptr + __i * _V::size(), vector_aligned); });
#else
        const auto &__xx = __data(__x);
        return __generate_from_n_evaluations<Parts, std::array<_V, Parts>>(
            [&](auto __i) {
                constexpr size_t __offset = decltype(__i)::value * _V::size();
                __unused(__offset);  // not really
                return _V([&](auto __j) {
                    constexpr _SizeConstant<__j + __offset> __k;
                    return __xx[__k];
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
template <typename _V,
	  typename _A,
	  size_t _Parts = simd_size_v<typename _V::simd_type::value_type, _A> /
			  _V::size()>
enable_if_t<(is_simd_mask_v<_V> &&
	     simd_size_v<typename _V::simd_type::value_type, _A> ==
	       _Parts * _V::size()),
	    std::array<_V, _Parts>>
  split(const simd_mask<typename _V::simd_type::value_type, _A>& __x)
{
  if constexpr (std::is_same_v<_A, typename _V::abi_type>)
    {
      return {__x};
    }
  else if constexpr (_Parts == 1)
    {
      return {static_simd_cast<_V>(__x)};
    }
  else if constexpr (_Parts == 2 &&
		     __is_abi<typename _V::abi_type, simd_abi::__sse>() &&
		     __is_abi<_A, simd_abi::__avx>())
    {
      return {_V(__private_init, __lo128(__data(__x))),
	      _V(__private_init, __hi128(__data(__x)))};
    }
  else if constexpr (_V::size() <= CHAR_BIT * sizeof(_ULLong))
    {
      const std::bitset __bits = __x.__to_bitset();
      return __generate_from_n_evaluations<_Parts, std::array<_V, _Parts>>(
	[&](auto __i) {
	  constexpr size_t __offset = __i * _V::size();
	  return _V(__bitset_init, (__bits >> __offset).to_ullong());
	});
    }
  else
    {
      return __generate_from_n_evaluations<_Parts, std::array<_V, _Parts>>(
	[&](auto __i) {
	  constexpr size_t __offset = __i * _V::size();
	  return _V(__private_init,
		    [&](auto __j) { return __x[__j + __offset]; });
	});
    }
}

// }}}
// split<_Sizes...>(simd) {{{
template <size_t... _Sizes, class _Tp, class _A,
          class = enable_if_t<((_Sizes + ...) == simd<_Tp, _A>::size())>>
_GLIBCXX_SIMD_ALWAYS_INLINE std::tuple<simd<_Tp, simd_abi::deduce_t<_Tp, _Sizes>>...> split(
    const simd<_Tp, _A> &__x)
{
    using _SL = __size_list<_Sizes...>;
    using _Tuple = std::tuple<__deduced_simd<_Tp, _Sizes>...>;
    constexpr size_t _N = simd_size_v<_Tp, _A>;
    constexpr size_t _N0 = _SL::template at<0>();
    using _V = __deduced_simd<_Tp, _N0>;

    if constexpr (_N == _N0) {
        static_assert(sizeof...(_Sizes) == 1);
        return {simd_cast<_V>(__x)};
    } else if constexpr (__is_fixed_size_abi_v<_A> &&
                         __fixed_size_storage_t<_Tp, _N>::_S_first_size == _N0) {
        // if the first part of the _SimdTuple input matches the first output vector
        // in the std::tuple, extract it and recurse
        static_assert(!__is_fixed_size_abi_v<typename _V::abi_type>,
                      "How can <_Tp, _N> be __a single _SimdTuple entry but __a fixed_size_simd "
                      "when deduced?");
        const __fixed_size_storage_t<_Tp, _N> &__xx = __data(__x);
        return std::tuple_cat(
            std::make_tuple(_V(__private_init, __xx.first)),
            __split_wrapper(_SL::template pop_front<1>(), __xx.second));
    } else if constexpr ((!std::is_same_v<simd_abi::scalar,
                                          simd_abi::deduce_t<_Tp, _Sizes>> &&
                          ...) &&
                         (!__is_fixed_size_abi_v<simd_abi::deduce_t<_Tp, _Sizes>> &&
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
            __assert_unreachable<_Tp>();
        }
    } else {
#ifdef _GLIBCXX_SIMD_USE_ALIASING_LOADS
        const __may_alias<_Tp> *const __element_ptr =
            reinterpret_cast<const __may_alias<_Tp> *>(&__x);
        return __generate_from_n_evaluations<sizeof...(_Sizes), _Tuple>([&](auto __i) {
            using _Vi = __deduced_simd<_Tp, _SL::at(__i)>;
            constexpr size_t __offset = _SL::before(__i);
            constexpr size_t __base_align = alignof(simd<_Tp, _A>);
            constexpr size_t __a = __base_align - ((__offset * sizeof(_Tp)) % __base_align);
            constexpr size_t __b = ((__a - 1) & __a) ^ __a;
            constexpr size_t __alignment = __b == 0 ? __a : __b;
            return _Vi(__element_ptr + __offset, overaligned<__alignment>);
        });
#else
        return __generate_from_n_evaluations<sizeof...(_Sizes), _Tuple>([&](auto __i) {
            using _Vi = __deduced_simd<_Tp, _SL::at(__i)>;
            const auto &__xx = __data(__x);
            using _Offset = decltype(_SL::before(__i));
            return _Vi([&](auto __j) {
                constexpr _SizeConstant<_Offset::value + __j> __k;
                return __xx[__k];
            });
        });
#endif
    }
}

// }}}

// __subscript_in_pack {{{
template <size_t _I, class _Tp, class _A, class... _As>
_GLIBCXX_SIMD_INTRINSIC constexpr _Tp __subscript_in_pack(const simd<_Tp, _A> &__x, const simd<_Tp, _As> &... __xs)
{
    if constexpr (_I < simd_size_v<_Tp, _A>) {
        return __x[_I];
    } else {
        return __subscript_in_pack<_I - simd_size_v<_Tp, _A>>(__xs...);
    }
}
// }}}

// concat(simd...) {{{
template <class _Tp, class... _As>
simd<_Tp, simd_abi::deduce_t<_Tp, (simd_size_v<_Tp, _As> + ...)>> concat(
    const simd<_Tp, _As> &... __xs)
{
    return simd<_Tp, simd_abi::deduce_t<_Tp, (simd_size_v<_Tp, _As> + ...)>>(
        [&](auto __i) { return __subscript_in_pack<__i>(__xs...); });
}

// }}}

// _Smart_reference {{{
template <class _U, class _Accessor = _U, class _ValueType = typename _U::value_type>
class _Smart_reference
{
    friend _Accessor;
    int index;
    _U &obj;

    _GLIBCXX_SIMD_INTRINSIC constexpr _ValueType __read() const noexcept
    {
        if constexpr (std::is_arithmetic_v<_U>) {
            _GLIBCXX_DEBUG_ASSERT(index == 0);
            return obj;
        } else {
            return obj[index];
        }
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC constexpr void __write(_Tp &&__x) const
    {
        _Accessor::set(obj, index, std::forward<_Tp>(__x));
    }

public:
    _GLIBCXX_SIMD_INTRINSIC constexpr _Smart_reference(_U& __o, int __i) noexcept
        : index(__i), obj(__o)
    {
    }

    using value_type = _ValueType;

    _GLIBCXX_SIMD_INTRINSIC _Smart_reference(const _Smart_reference &) = delete;

    _GLIBCXX_SIMD_INTRINSIC constexpr operator value_type() const noexcept { return __read(); }

    template <class _Tp,
              class = _ValuePreservingOrInt<std::decay_t<_Tp>, value_type>>
    _GLIBCXX_SIMD_INTRINSIC constexpr _Smart_reference operator=(_Tp &&__x) &&
    {
        __write(std::forward<_Tp>(__x));
        return {obj, index};
    }

// TODO: improve with operator.()

#define _GLIBCXX_SIMD_OP_(op_)                                                 \
  template <class _Tp,                                                         \
	    class _TT =                                                        \
	      decltype(std::declval<value_type>() op_ std::declval<_Tp>()),    \
	    class = _ValuePreservingOrInt<std::decay_t<_Tp>, _TT>,         \
	    class = _ValuePreservingOrInt<_TT, value_type>>                \
  _GLIBCXX_SIMD_INTRINSIC constexpr _Smart_reference operator op_##=(          \
    _Tp&& __x)&&                                                               \
  {                                                                            \
    const value_type& __lhs = __read();                                        \
    __write(__lhs op_ __x);                                                    \
    return {obj, index};                                                       \
  }
    _GLIBCXX_SIMD_ALL_ARITHMETICS(_GLIBCXX_SIMD_OP_);
    _GLIBCXX_SIMD_ALL_SHIFTS(_GLIBCXX_SIMD_OP_);
    _GLIBCXX_SIMD_ALL_BINARY(_GLIBCXX_SIMD_OP_);
#undef _GLIBCXX_SIMD_OP_

    template <class _Tp = void,
              class = decltype(
                  ++std::declval<std::conditional_t<true, value_type, _Tp> &>())>
    _GLIBCXX_SIMD_INTRINSIC constexpr _Smart_reference operator++() &&
    {
        value_type __x = __read();
        __write(++__x);
        return {obj, index};
    }

    template <class _Tp = void,
              class = decltype(
                  std::declval<std::conditional_t<true, value_type, _Tp> &>()++)>
    _GLIBCXX_SIMD_INTRINSIC constexpr value_type operator++(int) &&
    {
        const value_type __r = __read();
        value_type __x = __r;
        __write(++__x);
        return __r;
    }

    template <class _Tp = void,
              class = decltype(
                  --std::declval<std::conditional_t<true, value_type, _Tp> &>())>
    _GLIBCXX_SIMD_INTRINSIC constexpr _Smart_reference operator--() &&
    {
        value_type __x = __read();
        __write(--__x);
        return {obj, index};
    }

    template <class _Tp = void,
              class = decltype(
                  std::declval<std::conditional_t<true, value_type, _Tp> &>()--)>
    _GLIBCXX_SIMD_INTRINSIC constexpr value_type operator--(int) &&
    {
        const value_type __r = __read();
        value_type __x = __r;
        __write(--__x);
        return __r;
    }

    _GLIBCXX_SIMD_INTRINSIC friend void swap(_Smart_reference &&__a, _Smart_reference &&__b) noexcept(
        conjunction<std::is_nothrow_constructible<value_type, _Smart_reference &&>,
            std::is_nothrow_assignable<_Smart_reference &&, value_type &&>>::value)
    {
        value_type __tmp = static_cast<_Smart_reference &&>(__a);
        static_cast<_Smart_reference &&>(__a) = static_cast<value_type>(__b);
        static_cast<_Smart_reference &&>(__b) = std::move(__tmp);
    }

    _GLIBCXX_SIMD_INTRINSIC friend void swap(value_type &__a, _Smart_reference &&__b) noexcept(
        conjunction<std::is_nothrow_constructible<value_type, value_type &&>,
            std::is_nothrow_assignable<value_type &, value_type &&>,
            std::is_nothrow_assignable<_Smart_reference &&, value_type &&>>::value)
    {
        value_type __tmp(std::move(__a));
        __a = static_cast<value_type>(__b);
        static_cast<_Smart_reference &&>(__b) = std::move(__tmp);
    }

    _GLIBCXX_SIMD_INTRINSIC friend void swap(_Smart_reference &&__a, value_type &__b) noexcept(
        conjunction<std::is_nothrow_constructible<value_type, _Smart_reference &&>,
            std::is_nothrow_assignable<value_type &, value_type &&>,
            std::is_nothrow_assignable<_Smart_reference &&, value_type &&>>::value)
    {
        value_type __tmp(__a);
        static_cast<_Smart_reference &&>(__a) = std::move(__b);
        __b = std::move(__tmp);
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
template <class _Tp, class _MT, class _Abi, size_t _N> struct __gnu_traits {
    using _IsValid = true_type;
    using _SimdImpl = typename _Abi::_SimdImpl;
    using _MaskImpl = typename _Abi::_MaskImpl;

    // simd and simd_mask member types {{{2
    using _SimdMember = _SimdWrapper<_Tp, _N>;
    using _MaskMember = _SimdWrapper<_MT, _N>;
    static constexpr size_t _S_simd_align = alignof(_SimdMember);
    static constexpr size_t _S_mask_align = alignof(_MaskMember);

    // _Simd_base / base class for simd, providing extra conversions {{{2
    struct simd_base2 {
        explicit operator __intrinsic_type_t<_Tp, _N>() const
        {
            return static_cast<const simd<_Tp, _Abi> *>(this)->_M_data.__v();
        }
        explicit operator __vector_type_t<_Tp, _N>() const
        {
            return static_cast<const simd<_Tp, _Abi> *>(this)->_M_data.builtin();
        }
    };
    struct simd_base1 {
        explicit operator __intrinsic_type_t<_Tp, _N>() const
        {
            return __data(*static_cast<const simd<_Tp, _Abi> *>(this));
        }
    };
    using _Simd_base = std::conditional_t<
        std::is_same<__intrinsic_type_t<_Tp, _N>, __vector_type_t<_Tp, _N>>::value,
        simd_base1, simd_base2>;

    // _Mask_base {{{2
    struct mask_base2 {
        explicit operator __intrinsic_type_t<_Tp, _N>() const
        {
            return static_cast<const simd_mask<_Tp, _Abi> *>(this)->_M_data.__intrin();
        }
        explicit operator __vector_type_t<_Tp, _N>() const
        {
            return static_cast<const simd_mask<_Tp, _Abi> *>(this)->_M_data._M_data;
        }
    };
    struct mask_base1 {
        explicit operator __intrinsic_type_t<_Tp, _N>() const
        {
            return __data(*static_cast<const simd_mask<_Tp, _Abi> *>(this));
        }
    };
    using _Mask_base = std::conditional_t<
        std::is_same<__intrinsic_type_t<_Tp, _N>, __vector_type_t<_Tp, _N>>::value,
        mask_base1, mask_base2>;

    // _MaskCastType {{{2
    // parameter type of one explicit simd_mask constructor
    class _MaskCastType
    {
        using _U = __intrinsic_type_t<_Tp, _N>;
        _U _M_data;

    public:
        _MaskCastType(_U __x) : _M_data(__x) {}
        operator _MaskMember() const { return _M_data; }
    };

    // _SimdCastType {{{2
    // parameter type of one explicit simd constructor
    class simd_cast_type1
    {
        using _A = __intrinsic_type_t<_Tp, _N>;
        _SimdMember _M_data;

    public:
        simd_cast_type1(_A __a) : _M_data(__vector_bitcast<_Tp>(__a)) {}
        operator _SimdMember() const { return _M_data; }
    };

    class simd_cast_type2
    {
        using _A = __intrinsic_type_t<_Tp, _N>;
        using _B = __vector_type_t<_Tp, _N>;
        _SimdMember _M_data;

    public:
        simd_cast_type2(_A __a) : _M_data(__vector_bitcast<_Tp>(__a)) {}
        simd_cast_type2(_B __b) : _M_data(__b) {}
        operator _SimdMember() const { return _M_data; }
    };

    using _SimdCastType = std::conditional_t<
        std::is_same<__intrinsic_type_t<_Tp, _N>, __vector_type_t<_Tp, _N>>::value,
        simd_cast_type1, simd_cast_type2>;
    //}}}2
};

// __neon_is_vectorizable {{{1
#if _GLIBCXX_SIMD_HAVE_NEON_ABI
template <class _Tp> struct __neon_is_vectorizable : __is_vectorizable<_Tp> {};
template <> struct __neon_is_vectorizable<long double> : false_type {};
#if !_GLIBCXX_SIMD_HAVE_FULL_NEON_ABI
template <> struct __neon_is_vectorizable<double> : false_type {};
#endif
#else
template <class _Tp> struct __neon_is_vectorizable : false_type {};
#endif

// __sse_is_vectorizable {{{1
#if _GLIBCXX_SIMD_HAVE_FULL_SSE_ABI
template <class _Tp> struct __sse_is_vectorizable : __is_vectorizable<_Tp> {};
template <> struct __sse_is_vectorizable<long double> : false_type {};
#elif _GLIBCXX_SIMD_HAVE_SSE_ABI
template <class _Tp> struct __sse_is_vectorizable : is_same<_Tp, float> {};
#else
template <class _Tp> struct __sse_is_vectorizable : false_type {};
#endif

// __avx_is_vectorizable {{{1
#if _GLIBCXX_SIMD_HAVE_FULL_AVX_ABI
template <class _Tp> struct __avx_is_vectorizable : __is_vectorizable<_Tp> {};
#elif _GLIBCXX_SIMD_HAVE_AVX_ABI
template <class _Tp> struct __avx_is_vectorizable : std::is_floating_point<_Tp> {};
#else
template <class _Tp> struct __avx_is_vectorizable : false_type {};
#endif
template <> struct __avx_is_vectorizable<long double> : false_type {};

// __avx512_is_vectorizable {{{1
#if _GLIBCXX_SIMD_HAVE_AVX512_ABI
template <class _Tp> struct __avx512_is_vectorizable : __is_vectorizable<_Tp> {};
template <> struct __avx512_is_vectorizable<long double> : false_type {};
#if !_GLIBCXX_SIMD_HAVE_FULL_AVX512_ABI
template <> struct __avx512_is_vectorizable<  char> : false_type {};
template <> struct __avx512_is_vectorizable< _UChar> : false_type {};
template <> struct __avx512_is_vectorizable< _SChar> : false_type {};
template <> struct __avx512_is_vectorizable< short> : false_type {};
template <> struct __avx512_is_vectorizable<ushort> : false_type {};
template <> struct __avx512_is_vectorizable<char16_t> : false_type {};
template <> struct __avx512_is_vectorizable<wchar_t> : __bool_constant<sizeof(wchar_t) >= 4> {};
#endif
#else
template <class _Tp> struct __avx512_is_vectorizable : false_type {};
#endif

// }}}
// __implicit_mask_abi_base {{{
template <int _Bytes, class _Abi> struct __implicit_mask_abi_base {
    template <class _Tp>
    using implicit_mask_type =
        __vector_type_t<__int_for_sizeof_t<_Tp>, simd_size_v<_Tp, _Abi>>;

    template <class _Tp>
    static constexpr auto implicit_mask =
        reinterpret_cast<__vector_type_t<_Tp, simd_size_v<_Tp, _Abi>>>(
            _Abi::is_partial ? __generate_builtin<implicit_mask_type<_Tp>>([](auto __i) {
                return __i < _Bytes / sizeof(_Tp) ? -1 : 0;
            })
                            : ~implicit_mask_type<_Tp>());

    template <class _Tp, class _TVT = __vector_traits<_Tp>>
    static constexpr auto masked(_Tp __x)
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
    template <class _Tp> static constexpr size_t size = _N *_Abi::template size<_Tp>;
    template <class _Tp> static constexpr size_t full_size = size<_Tp>;

    static constexpr int factor = _N;
    using member_abi = _Abi;

    // validity traits {{{2
    // allow 2x, 3x, and 4x "unroll"
    struct is_valid_abi_tag
        : __bool_constant<(_N > 1 && _N <= 4) && _Abi::is_valid_abi_tag> {
    };
    template <class _Tp> struct is_valid_size_for : _Abi::template is_valid_size_for<_Tp> {
    };
    template <class _Tp>
    struct _IsValid : conjunction<is_valid_abi_tag, typename _Abi::template _IsValid<_Tp>> {
    };
    template <class _Tp> static constexpr bool is_valid_v = _IsValid<_Tp>::value;

    // simd/_MaskImpl {{{2
    using _SimdImpl = __combine_simd_impl<_N, _Abi>;
    using _MaskImpl = __combine_mask_impl<_N, _Abi>;

    // __traits {{{2
    template <class _Tp, bool = is_valid_v<_Tp>> struct __traits : _InvalidTraits {
    };

    template <class _Tp> struct __traits<_Tp, true> {
        using _IsValid = true_type;
        using _SimdImpl = __combine_simd_impl<_N, _Abi>;
        using _MaskImpl = __combine_mask_impl<_N, _Abi>;

        // simd and simd_mask member types {{{2
        using _SimdMember =
            std::array<typename _Abi::template __traits<_Tp>::_SimdMember, _N>;
        using _MaskMember =
            std::array<typename _Abi::template __traits<_Tp>::_MaskMember, _N>;
        static constexpr size_t _S_simd_align =
            _Abi::template __traits<_Tp>::_S_simd_align;
        static constexpr size_t _S_mask_align =
            _Abi::template __traits<_Tp>::_S_mask_align;

        // _Simd_base / base class for simd, providing extra conversions {{{2
        struct _Simd_base {
            explicit operator const _SimdMember &() const
            {
                return static_cast<const simd<_Tp, __combine> *>(this)->_M_data;
            }
        };

        // _Mask_base {{{2
        // empty. The std::bitset interface suffices
        struct _Mask_base {
            explicit operator const _MaskMember &() const
            {
                return static_cast<const simd_mask<_Tp, __combine> *>(this)->_M_data;
            }
        };

        // _SimdCastType {{{2
        struct _SimdCastType {
            _SimdCastType(const _SimdMember &dd) : _M_data(dd) {}
            explicit operator const _SimdMember &() const { return _M_data; }

        private:
            const _SimdMember &_M_data;
        };

        // _MaskCastType {{{2
        struct _MaskCastType {
            _MaskCastType(const _MaskMember &dd) : _M_data(dd) {}
            explicit operator const _MaskMember &() const { return _M_data; }

        private:
            const _MaskMember &_M_data;
        };
        //}}}2
    };
    //}}}2
};
// __neon_abi {{{1
template <int _Bytes>
struct __neon_abi : __implicit_mask_abi_base<_Bytes, __neon_abi<_Bytes>> {
    template <class _Tp> static constexpr size_t size = _Bytes / sizeof(_Tp);
    template <class _Tp> static constexpr size_t full_size = 16 / sizeof(_Tp);
    static constexpr bool is_partial = _Bytes < 16;

    // validity traits {{{2
    struct is_valid_abi_tag : __bool_constant<(_Bytes > 0 && _Bytes <= 16)> {
    };
    template <class _Tp>
    struct is_valid_size_for
        : __bool_constant<(_Bytes / sizeof(_Tp) > 1 && _Bytes % sizeof(_Tp) == 0)> {
    };
    template <class _Tp>
    struct _IsValid : conjunction<is_valid_abi_tag, __neon_is_vectorizable<_Tp>,
                                  is_valid_size_for<_Tp>> {
    };
    template <class _Tp> static constexpr bool is_valid_v = _IsValid<_Tp>::value;

    // simd/_MaskImpl {{{2
    using _SimdImpl = __neon_simd_impl;
    using _MaskImpl = __neon_mask_impl;

    // __traits {{{2
    template <class _Tp>
    using __traits = std::conditional_t<is_valid_v<_Tp>,
                                      __gnu_traits<_Tp, _Tp, __neon_abi, full_size<_Tp>>,
                                      _InvalidTraits>;
    //}}}2
};

// __sse_abi {{{1
template <int _Bytes>
struct __sse_abi : __implicit_mask_abi_base<_Bytes, __sse_abi<_Bytes>> {
    template <class _Tp> static constexpr size_t size = _Bytes / sizeof(_Tp);
    template <class _Tp> static constexpr size_t full_size = 16 / sizeof(_Tp);
    static constexpr bool is_partial = _Bytes < 16;

    // validity traits {{{2
    // allow 2x, 3x, and 4x "unroll"
    struct is_valid_abi_tag : __bool_constant<_Bytes == 16> {};
    //struct is_valid_abi_tag : __bool_constant<(_Bytes > 0 && _Bytes <= 16)> {};
    template <class _Tp>
    struct is_valid_size_for
        : __bool_constant<(_Bytes / sizeof(_Tp) > 1 && _Bytes % sizeof(_Tp) == 0)> {
    };

    template <class _Tp>
    struct _IsValid
        : conjunction<is_valid_abi_tag, __sse_is_vectorizable<_Tp>, is_valid_size_for<_Tp>> {
    };
    template <class _Tp> static constexpr bool is_valid_v = _IsValid<_Tp>::value;

    // simd/_MaskImpl {{{2
    using _SimdImpl = __sse_simd_impl;
    using _MaskImpl = __sse_mask_impl;

    // __traits {{{2
    template <class _Tp>
    using __traits = std::conditional_t<is_valid_v<_Tp>,
                                      __gnu_traits<_Tp, _Tp, __sse_abi, full_size<_Tp>>,
                                      _InvalidTraits>;
    //}}}2
};

// __avx_abi {{{1
template <int _Bytes>
struct __avx_abi : __implicit_mask_abi_base<_Bytes, __avx_abi<_Bytes>> {
    template <class _Tp> static constexpr size_t size = _Bytes / sizeof(_Tp);
    template <class _Tp> static constexpr size_t full_size = 32 / sizeof(_Tp);
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
    template <class _Tp>
    struct is_valid_size_for : __bool_constant<(_Bytes % sizeof(_Tp) == 0)> {
    };
    template <class _Tp>
    struct _IsValid
        : conjunction<is_valid_abi_tag, __avx_is_vectorizable<_Tp>, is_valid_size_for<_Tp>> {
    };
    template <class _Tp> static constexpr bool is_valid_v = _IsValid<_Tp>::value;

    // simd/_MaskImpl {{{2
    using _SimdImpl = __avx_simd_impl;
    using _MaskImpl = __avx_mask_impl;

    // __traits {{{2
    template <class _Tp>
    using __traits = std::conditional_t<is_valid_v<_Tp>,
                                      __gnu_traits<_Tp, _Tp, __avx_abi, full_size<_Tp>>,
                                      _InvalidTraits>;
    //}}}2
};

// __avx512_abi {{{1
template <int _Bytes> struct __avx512_abi {
    template <class _Tp> static constexpr size_t size = _Bytes / sizeof(_Tp);
    template <class _Tp> static constexpr size_t full_size = 64 / sizeof(_Tp);
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
    template <class _Tp>
    struct is_valid_size_for : __bool_constant<(_Bytes % sizeof(_Tp) == 0)> {
    };
    template <class _Tp>
    struct _IsValid
        : conjunction<is_valid_abi_tag, __avx512_is_vectorizable<_Tp>, is_valid_size_for<_Tp>> {
    };
    template <class _Tp> static constexpr bool is_valid_v = _IsValid<_Tp>::value;

    // implicit mask {{{2
    template <class _Tp>
    using implicit_mask_type = __bool_storage_member_type_t<64 / sizeof(_Tp)>;

    template <class _Tp>
    static constexpr implicit_mask_type<_Tp> implicit_mask =
        _Bytes == 64 ? ~implicit_mask_type<_Tp>()
                    : (implicit_mask_type<_Tp>(1) << (_Bytes / sizeof(_Tp))) - 1;

    template <class _Tp, class = enable_if_t<__is_bitmask_v<_Tp>>>
    static constexpr _Tp masked(_Tp __x)
    {
        if constexpr (is_partial) {
            constexpr size_t _N = sizeof(_Tp) * 8;
            return __x &
                   ((__bool_storage_member_type_t<_N>(1) << (_Bytes * _N / 64)) - 1);
        } else {
            return __x;
        }
    }

    // simd/_MaskImpl {{{2
    using _SimdImpl = __avx512_simd_impl;
    using _MaskImpl = __avx512_mask_impl;

    // __traits {{{2
    template <class _Tp>
    using __traits =
        std::conditional_t<is_valid_v<_Tp>,
                           __gnu_traits<_Tp, bool, __avx512_abi, full_size<_Tp>>,
                           _InvalidTraits>;
    //}}}2
};

// __scalar_abi {{{1
struct __scalar_abi {
    template <class _Tp> static constexpr size_t size = 1;
    template <class _Tp> static constexpr size_t full_size = 1;
    struct is_valid_abi_tag : true_type {};
    template <class _Tp> struct is_valid_size_for : true_type {};
    template <class _Tp> struct _IsValid : __is_vectorizable<_Tp> {};
    template <class _Tp> static constexpr bool is_valid_v = _IsValid<_Tp>::value;

    using _SimdImpl = __scalar_simd_impl;
    using _MaskImpl = __scalar_mask_impl;

    template <class _Tp, bool = is_valid_v<_Tp>> struct __traits : _InvalidTraits {
    };

    template <class _Tp> struct __traits<_Tp, true> {
        using _IsValid = true_type;
        using _SimdImpl = __scalar_simd_impl;
        using _MaskImpl = __scalar_mask_impl;
        using _SimdMember = _Tp;
        using _MaskMember = bool;
        static constexpr size_t _S_simd_align = alignof(_SimdMember);
        static constexpr size_t _S_mask_align = alignof(_MaskMember);

        // nothing the user can spell converts to/from simd/simd_mask
        struct _SimdCastType {
            _SimdCastType() = delete;
        };
        struct _MaskCastType {
            _MaskCastType() = delete;
        };
        struct _Simd_base {};
        struct _Mask_base {};
    };
};

// __fixed_abi {{{1
template <int _N> struct __fixed_abi {
    template <class _Tp> static constexpr size_t size = _N;
    template <class _Tp> static constexpr size_t full_size = _N;
    // validity traits {{{2
    struct is_valid_abi_tag
        : public __bool_constant<(_N > 0)> {
    };
    template <class _Tp>
    struct is_valid_size_for
        : __bool_constant<((_N <= simd_abi::max_fixed_size<_Tp>) ||
                                 (simd_abi::__neon::is_valid_v<char> &&
                                  _N == simd_size_v<char, simd_abi::__neon>) ||
                                 (simd_abi::__sse::is_valid_v<char> &&
                                  _N == simd_size_v<char, simd_abi::__sse>) ||
                                 (simd_abi::__avx::is_valid_v<char> &&
                                  _N == simd_size_v<char, simd_abi::__avx>) ||
                                 (simd_abi::__avx512::is_valid_v<char> &&
                                  _N == simd_size_v<char, simd_abi::__avx512>))> {
    };
    template <class _Tp>
    struct _IsValid
        : conjunction<is_valid_abi_tag, __is_vectorizable<_Tp>, is_valid_size_for<_Tp>> {
    };
    template <class _Tp> static constexpr bool is_valid_v = _IsValid<_Tp>::value;

    // simd/_MaskImpl {{{2
    using _SimdImpl = __fixed_size_simd_impl<_N>;
    using _MaskImpl = __fixed_size_mask_impl<_N>;

    // __traits {{{2
    template <class _Tp, bool = is_valid_v<_Tp>> struct __traits : _InvalidTraits {
    };

    template <class _Tp> struct __traits<_Tp, true> {
        using _IsValid = true_type;
        using _SimdImpl = __fixed_size_simd_impl<_N>;
        using _MaskImpl = __fixed_size_mask_impl<_N>;

        // simd and simd_mask member types {{{2
        using _SimdMember = __fixed_size_storage_t<_Tp, _N>;
        using _MaskMember = std::bitset<_N>;
        static constexpr size_t _S_simd_align =
            __next_power_of_2(_N * sizeof(_Tp));
        static constexpr size_t _S_mask_align = alignof(_MaskMember);

        // _Simd_base / base class for simd, providing extra conversions {{{2
        struct _Simd_base {
            // The following ensures, function arguments are passed via the stack. This is
            // important for ABI compatibility across TU boundaries
            _Simd_base(const _Simd_base &) {}
            _Simd_base() = default;

            explicit operator const _SimdMember &() const
            {
                return static_cast<const simd<_Tp, __fixed_abi> *>(this)->_M_data;
            }
            explicit operator std::array<_Tp, _N>() const
            {
                std::array<_Tp, _N> __r;
                // _SimdMember can be larger because of higher alignment
                static_assert(sizeof(__r) <= sizeof(_SimdMember), "");
                std::memcpy(__r.data(), &static_cast<const _SimdMember &>(*this),
                            sizeof(__r));
                return __r;
            }
        };

        // _Mask_base {{{2
        // empty. The std::bitset interface suffices
        struct _Mask_base {};

        // _SimdCastType {{{2
        struct _SimdCastType {
            _SimdCastType(const std::array<_Tp, _N> &);
            _SimdCastType(const _SimdMember &dd) : _M_data(dd) {}
            explicit operator const _SimdMember &() const { return _M_data; }

        private:
            const _SimdMember &_M_data;
        };

        // _MaskCastType {{{2
        class _MaskCastType
        {
            _MaskCastType() = delete;
        };
        //}}}2
    };
};

//}}}
}  // namespace simd_abi

// __scalar_abi_wrapper {{{1
template <int _Bytes> struct __scalar_abi_wrapper : simd_abi::__scalar_abi {
    template <class _Tp>
    static constexpr bool is_valid_v = simd_abi::__scalar_abi::_IsValid<_Tp>::value &&
                                       sizeof(_Tp) == _Bytes;
};

// __decay_abi metafunction {{{1
template <class _Tp> struct __decay_abi {
    using type = _Tp;
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

template <template <int> class _A0, template <int> class... _Rest>
struct __abi_list<_A0, _Rest...> {
    template <class _Tp, int _N>
    static constexpr bool has_valid_abi = _A0<sizeof(_Tp) * _N>::template is_valid_v<_Tp> ||
                                          __abi_list<_Rest...>::template has_valid_abi<_Tp, _N>;
    template <class _Tp, int _N>
    using first_valid_abi =
        std::conditional_t<_A0<sizeof(_Tp) * _N>::template is_valid_v<_Tp>,
                           typename __decay_abi<_A0<sizeof(_Tp) * _N>>::type,
                           typename __abi_list<_Rest...>::template first_valid_abi<_Tp, _N>>;
    using _B = typename __full_abi<_A0>::type;
    template <class _Tp, int _N>
    using best_abi = std::conditional_t<
        _A0<sizeof(_Tp) * _N>::template is_valid_v<_Tp>,
        typename __decay_abi<_A0<sizeof(_Tp) * _N>>::type,
        std::conditional_t<(_B::template is_valid_v<_Tp> && _B::template size<_Tp> <= _N), _B,
                           typename __abi_list<_Rest...>::template best_abi<_Tp, _N>>>;
};

// }}}1

// the following lists all native ABIs, which makes them accessible to simd_abi::deduce
// and select_best_vector_type_t (for fixed_size). Order matters: Whatever comes first has
// higher priority.
using __all_native_abis =
    __abi_list<simd_abi::__avx512_abi, simd_abi::__avx_abi, simd_abi::__sse_abi,
             simd_abi::__neon_abi, __scalar_abi_wrapper>;

// valid __traits specialization {{{1
template <class _Tp, class _Abi>
struct _SimdTraits<_Tp, _Abi, std::void_t<typename _Abi::template _IsValid<_Tp>>>
    : _Abi::template __traits<_Tp> {
};

// __deduce_impl specializations {{{1
// try all native ABIs (including scalar) first
template <class _Tp, std::size_t _N>
struct __deduce_impl<_Tp, _N,
                   enable_if_t<__all_native_abis::template has_valid_abi<_Tp, _N>>> {
    using type = __all_native_abis::first_valid_abi<_Tp, _N>;
};

// fall back to fixed_size only if scalar and native ABIs don't match
template <class _Tp, std::size_t _N, class = void> struct __deduce_fixed_size_fallback {};
template <class _Tp, std::size_t _N>
struct __deduce_fixed_size_fallback<
    _Tp, _N, enable_if_t<simd_abi::fixed_size<_N>::template is_valid_v<_Tp>>> {
    using type = simd_abi::fixed_size<_N>;
};
template <class _Tp, std::size_t _N, class>
struct __deduce_impl : public __deduce_fixed_size_fallback<_Tp, _N> {
};

//}}}1

// simd_mask {{{
template <class _Tp, class _Abi> class simd_mask : public _SimdTraits<_Tp, _Abi>::_Mask_base
{
    // types, tags, and friends {{{
    using __traits = _SimdTraits<_Tp, _Abi>;
    using __impl = typename __traits::_MaskImpl;
    using __member_type = typename __traits::_MaskMember;
    static constexpr _Tp *_S_type_tag = nullptr;
    friend typename __traits::_Mask_base;
    friend class simd<_Tp, _Abi>;  // to construct masks on return
    friend __impl;
    friend typename __traits::_SimdImpl;  // to construct masks on return and
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
    using reference = _Smart_reference<__member_type, __impl, value_type>;
    using simd_type = simd<_Tp, _Abi>;
    using abi_type = _Abi;

    // }}}
    static constexpr size_t size() { return __size_or_zero_v<_Tp, _Abi>; }
    // constructors & assignment {{{
    simd_mask() = default;
    simd_mask(const simd_mask &) = default;
    simd_mask(simd_mask &&) = default;
    simd_mask &operator=(const simd_mask &) = default;
    simd_mask &operator=(simd_mask &&) = default;

    // }}}

    // access to internal representation (suggested extension) {{{
    _GLIBCXX_SIMD_ALWAYS_INLINE explicit simd_mask(
      typename __traits::_MaskCastType __init)
    : _M_data{__init}
    {
    }
    // conversions to internal type is done in _Mask_base

    // }}}
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
             conjunction<std::is_integral<_Tp>, std::is_integral<_U>,
             __negation<std::is_same<_Abi, simd_abi::fixed_size<size()>>>,
             __negation<std::is_same<_Tp, _U>>>::value>>
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
    _GLIBCXX_SIMD_INTRINSIC static __member_type load_wrapper(const value_type* __mem,
                                                              [[maybe_unused]] _F __f)
    {
        if constexpr (__is_scalar()) {
            return __mem[0];
        } else if constexpr (__is_fixed()) {
            const fixed_size_simd<unsigned char, size()> bools(
                reinterpret_cast<const __may_alias<unsigned char> *>(__mem), __f);
            return __data(bools != 0);
        } else if constexpr (__is_sse()) {
            if constexpr (size() == 2 && __have_sse2) {
                return _ToWrapper(_mm_set_epi32(-int(__mem[1]), -int(__mem[1]),
                                                        -int(__mem[0]), -int(__mem[0])));
            } else if constexpr (size() == 4 && __have_sse2) {
                __m128i __k = _mm_cvtsi32_si128(*reinterpret_cast<const int *>(__mem));
                __k = _mm_cmpgt_epi16(_mm_unpacklo_epi8(__k, __k), _mm_setzero_si128());
                return _ToWrapper(_mm_unpacklo_epi16(__k, __k));
            } else if constexpr (size() == 4 && __have_mmx) {
                __m128 __k =
                    _mm_cvtpi8_ps(_mm_cvtsi32_si64(*reinterpret_cast<const int *>(__mem)));
                _mm_empty();
                return _ToWrapper(_mm_cmpgt_ps(__k, __m128()));
            } else if constexpr (size() == 8 && __have_sse2) {
                const auto __k = __make_builtin<long long>(
                    *reinterpret_cast<const __may_alias<long long> *>(__mem), 0);
                if constexpr (__have_sse2) {
                    return _ToWrapper(
                        __vector_bitcast<short>(_mm_unpacklo_epi8(__k, __k)) != 0);
                }
            } else if constexpr (size() == 16 && __have_sse2) {
                return __vector_bitcast<_Tp>(
                    _mm_cmpgt_epi8(__vector_load<long long, 2>(__mem, __f), __m128i()));
            } else {
                __assert_unreachable<_F>();
            }
        } else if constexpr (__is_avx()) {
            if constexpr (size() == 4 && __have_avx) {
                int bool4;
                if constexpr (__is_aligned_v<_F, 4>) {
                    bool4 = *reinterpret_cast<const __may_alias<int> *>(__mem);
                } else {
                    std::memcpy(&bool4, __mem, 4);
                }
                const auto __k = __to_intrin(
                    (__vector_broadcast<4>(bool4) &
                     __make_builtin<int>(0x1, 0x100, 0x10000, 0x1000000)) != 0);
                return _ToWrapper(
                    __concat(_mm_unpacklo_epi32(__k, __k), _mm_unpackhi_epi32(__k, __k)));
            } else if constexpr (size() == 8 && __have_avx) {
                auto __k = __vector_load<long long, 2, 8>(__mem, __f);
                __k = _mm_cmpgt_epi16(_mm_unpacklo_epi8(__k, __k), __m128i());
                return _ToWrapper(
                    __concat(_mm_unpacklo_epi16(__k, __k), _mm_unpackhi_epi16(__k, __k)));
            } else if constexpr (size() == 16 && __have_avx) {
                const auto __k =
                    _mm_cmpgt_epi8(__vector_load<long long, 2>(__mem, __f), __m128i());
                return __concat(_mm_unpacklo_epi8(__k, __k), _mm_unpackhi_epi8(__k, __k));
            } else if constexpr (size() == 32 && __have_avx2) {
                return __vector_bitcast<_Tp>(
                    _mm256_cmpgt_epi8(__vector_load<long long, 4>(__mem, __f), __m256i()));
            } else {
                __assert_unreachable<_F>();
            }
        } else if constexpr (__is_avx512()) {
            if constexpr (size() == 8) {
                const auto __a = __vector_load<long long, 2, 8>(__mem, __f);
                if constexpr (__have_avx512bw_vl) {
                    return _mm_test_epi8_mask(__a, __a);
                } else {
                    const auto __b = _mm512_cvtepi8_epi64(__a);
                    return _mm512_test_epi64_mask(__b, __b);
                }
            } else if constexpr (size() == 16) {
                const auto __a = __vector_load<long long, 2>(__mem, __f);
                if constexpr (__have_avx512bw_vl) {
                    return _mm_test_epi8_mask(__a, __a);
                } else {
                    const auto __b = _mm512_cvtepi8_epi32(__a);
                    return _mm512_test_epi32_mask(__b, __b);
                }
            } else if constexpr (size() == 32) {
                if constexpr (__have_avx512bw_vl) {
                    const auto __a = __vector_load<long long, 4>(__mem, __f);
                    return _mm256_test_epi8_mask(__a, __a);
                } else {
                    const auto __a =
                        _mm512_cvtepi8_epi32(__vector_load<long long, 2>(__mem, __f));
                    const auto __b =
                        _mm512_cvtepi8_epi32(__vector_load<long long, 2>(__mem + 16, __f));
                    return _mm512_test_epi32_mask(__a, __a) |
                           (_mm512_test_epi32_mask(__b, __b) << 16);
                }
            } else if constexpr (size() == 64) {
                if constexpr (__have_avx512bw) {
                    const auto __a = __vector_load<long long, 8>(__mem, __f);
                    return _mm512_test_epi8_mask(__a, __a);
                } else {
                    const auto __a =
                        _mm512_cvtepi8_epi32(__vector_load<long long, 2>(__mem, __f));
                    const auto __b = _mm512_cvtepi8_epi32(
                        __vector_load<long long, 2>(__mem + 16, __f));
                    const auto __c = _mm512_cvtepi8_epi32(
                        __vector_load<long long, 2>(__mem + 32, __f));
                    const auto __d = _mm512_cvtepi8_epi32(
                        __vector_load<long long, 2>(__mem + 48, __f));
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
    _GLIBCXX_SIMD_ALWAYS_INLINE simd_mask(const value_type* __mem, _Flags __f)
        : _M_data(load_wrapper(__mem, __f))
    {
    }
    template <class _Flags>
    _GLIBCXX_SIMD_ALWAYS_INLINE simd_mask(const value_type *__mem, simd_mask __k, _Flags __f) : _M_data{}
    {
        _M_data = __impl::masked_load(_M_data, __k._M_data, __mem, __f);
    }

    // }}}
    // loads [simd_mask.load] {{{
    template <class _Flags> _GLIBCXX_SIMD_ALWAYS_INLINE void copy_from(const value_type *__mem, _Flags __f)
    {
        _M_data = load_wrapper(__mem, __f);
    }

    // }}}
    // stores [simd_mask.store] {{{
    template <class _Flags> _GLIBCXX_SIMD_ALWAYS_INLINE void copy_to(value_type *__mem, _Flags __f) const
    {
        __impl::store(_M_data, __mem, __f);
    }

    // }}}
    // scalar access {{{
    _GLIBCXX_SIMD_ALWAYS_INLINE reference operator[](size_t __i) { return {_M_data, int(__i)}; }
    _GLIBCXX_SIMD_ALWAYS_INLINE value_type operator[](size_t __i) const {
        if constexpr (__is_scalar()) {
            _GLIBCXX_DEBUG_ASSERT(__i == 0);
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
        } else if constexpr (__is_fixed()) {
            return {__private_init, ~builtin()};
        } else if constexpr (__have_avx512dq && __is_avx512() && size() <= 8) {
            return {__private_init, _knot_mask8(builtin())};
        } else if constexpr (__is_avx512() && size() <= 16) {
            // the following is a narrowing conversion on KNL for doubles (__mmask8)
            return simd_mask(__private_init, _knot_mask16(builtin()));
        } else if constexpr (__have_avx512bw && __is_avx512() && size() <= 32) {
            return {__private_init, _knot_mask32(builtin())};
        } else if constexpr (__have_avx512bw && __is_avx512() && size() <= 64) {
            return {__private_init, _knot_mask64(builtin())};
        } else {
            return {__private_init,
                    _ToWrapper(~__vector_bitcast<_UInt>(builtin()))};
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
    // private_init ctor {{{
    _GLIBCXX_SIMD_INTRINSIC simd_mask(_PrivateInit, typename __traits::_MaskMember __init)
        : _M_data(__init)
    {
    }

    // }}}
    // private_init generator ctor {{{
    template <class _F, class = decltype(bool(std::declval<_F>()(size_t())))>
    _GLIBCXX_SIMD_INTRINSIC simd_mask(_PrivateInit, _F &&__gen)
    {
        for (size_t __i = 0; __i < size(); ++__i) {
            __impl::set(_M_data, __i, __gen(__i));
        }
    }

    // }}}
    // bitset_init ctor {{{
    _GLIBCXX_SIMD_INTRINSIC simd_mask(_BitsetInit, std::bitset<size()> __init)
        : _M_data(__impl::__from_bitset(__init, _S_type_tag))
    {
    }

    // }}}
    // __cvt {{{
    struct _CvtProxy
    {
      template <
	typename _U,
	typename _A2,
	typename = enable_if_t<simd_size_v<_U, _A2> == simd_size_v<_Tp, _Abi>>>
      operator simd_mask<_U, _A2>() &&
      {
	return static_simd_cast<simd_mask<_U, _A2>>(_M_data);
      }

      const simd_mask<_Tp, _Abi>& _M_data;
    };
    _GLIBCXX_SIMD_INTRINSIC _CvtProxy __cvt() const { return {*this}; }
    // }}}

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
            using _U = __vector_type_t<__int_for_sizeof_t<_Tp>, size()>;
            return _ToWrapper(__x ? _Abi::masked(~_U()) : _U());
        }
    }

    // }}}
    // TODO remove __intrin:
    /*auto __intrin() const  // {{{
    {
        if constexpr (!__is_scalar() && !__is_fixed()) {
            return __to_intrin(_M_data._M_data);
        }
    }*/

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
    friend const auto &__data<_Tp, abi_type>(const simd_mask &);
    friend auto &__data<_Tp, abi_type>(simd_mask &);
    alignas(__traits::_S_mask_align) __member_type _M_data;
};

// }}}

// __data(simd_mask) {{{
template <class _Tp, class _A>
_GLIBCXX_SIMD_INTRINSIC constexpr const auto &__data(const simd_mask<_Tp, _A> &__x)
{
    return __x._M_data;
}
template <class _Tp, class _A> _GLIBCXX_SIMD_INTRINSIC constexpr auto &__data(simd_mask<_Tp, _A> &__x)
{
    return __x._M_data;
}
// }}}
// __all_of {{{
template <class _Tp, class _Abi, class _Data> _GLIBCXX_SIMD_INTRINSIC bool __all_of(const _Data &__k)
{
    // _Data = decltype(__data(simd_mask))
    if constexpr (__is_abi<_Abi, simd_abi::scalar>()) {
        return __k;
    } else if constexpr (__is_abi<_Abi, simd_abi::fixed_size>()) {
        return __k.all();
    } else if constexpr (__is_combined_abi<_Abi>()) {
        for (int __i = 0; __i < _Abi::factor; ++__i) {
            if (!__all_of<_Tp, typename _Abi::member_abi>(__k[__i])) {
                return false;
            }
        }
        return true;
    } else if constexpr (__is_abi<_Abi, simd_abi::__sse_abi>() ||
                         __is_abi<_Abi, simd_abi::__avx_abi>()) {
        constexpr size_t _N = simd_size_v<_Tp, _Abi>;
        if constexpr (__have_sse4_1) {
            constexpr auto __b =
                reinterpret_cast<__intrinsic_type_t<_Tp, _N>>(_Abi::template implicit_mask<_Tp>);
            if constexpr (std::is_same_v<_Tp, float> && _N > 4) {
                return 0 != _mm256_testc_ps(__to_intrin(__k), __b);
            } else if constexpr (std::is_same_v<_Tp, float> && __have_avx) {
                return 0 != _mm_testc_ps(__to_intrin(__k), __b);
            } else if constexpr (std::is_same_v<_Tp, float> ) {
                return 0 != _mm_testc_si128(_mm_castps_si128(__to_intrin(__k)), _mm_castps_si128(__b));
            } else if constexpr (std::is_same_v<_Tp, double> && _N > 2) {
                return 0 != _mm256_testc_pd(__to_intrin(__k), __b);
            } else if constexpr (std::is_same_v<_Tp, double> && __have_avx) {
                return 0 != _mm_testc_pd(__to_intrin(__k), __b);
            } else if constexpr (std::is_same_v<_Tp, double>) {
                return 0 != _mm_testc_si128(_mm_castpd_si128(__to_intrin(__k)), _mm_castpd_si128(__b));
            } else if constexpr (sizeof(__b) == 32) {
                return _mm256_testc_si256(__to_intrin(__k), __b);
            } else {
                return _mm_testc_si128(__to_intrin(__k), __b);
            }
        } else if constexpr (std::is_same_v<_Tp, float>) {
            return (_mm_movemask_ps(__to_intrin(__k)) & ((1 << _N) - 1)) == (1 << _N) - 1;
        } else if constexpr (std::is_same_v<_Tp, double>) {
            return (_mm_movemask_pd(__to_intrin(__k)) & ((1 << _N) - 1)) == (1 << _N) - 1;
        } else {
            return (_mm_movemask_epi8(__to_intrin(__k)) & ((1 << (_N * sizeof(_Tp))) - 1)) ==
                   (1 << (_N * sizeof(_Tp))) - 1;
        }
    } else if constexpr (__is_abi<_Abi, simd_abi::__avx512_abi>()) {
        constexpr auto _Mask = _Abi::template implicit_mask<_Tp>;
        if constexpr (std::is_same_v<_Data, _SimdWrapper<bool, 8>>) {
            if constexpr (__have_avx512dq) {
                return _kortestc_mask8_u8(__k._M_data, _Mask == 0xff ? __k._M_data : __mmask8(~_Mask));
            } else {
                return __k._M_data == _Mask;
            }
        } else if constexpr (std::is_same_v<_Data, _SimdWrapper<bool, 16>>) {
            return _kortestc_mask16_u8(__k._M_data, _Mask == 0xffff ? __k._M_data : __mmask16(~_Mask));
        } else if constexpr (std::is_same_v<_Data, _SimdWrapper<bool, 32>>) {
            if constexpr (__have_avx512bw) {
#ifdef _GLIBCXX_SIMD_WORKAROUND_PR85538
                return __k._M_data == _Mask;
#else
                return _kortestc_mask32_u8(__k._M_data, _Mask == 0xffffffffU ? __k._M_data : __mmask32(~_Mask));
#endif
            }
        } else if constexpr (std::is_same_v<_Data, _SimdWrapper<bool, 64>>) {
            if constexpr (__have_avx512bw) {
#ifdef _GLIBCXX_SIMD_WORKAROUND_PR85538
                return __k._M_data == _Mask;
#else
                return _kortestc_mask64_u8(
                    __k._M_data, _Mask == 0xffffffffffffffffULL ? __k._M_data : __mmask64(~_Mask));
#endif
            }
        }
    } else {
        __assert_unreachable<_Tp>();
    }
}

// }}}
// __any_of {{{
template <class _Tp, class _Abi, class _Data> _GLIBCXX_SIMD_INTRINSIC bool __any_of(const _Data &__k)
{
    if constexpr (__is_abi<_Abi, simd_abi::scalar>()) {
        return __k;
    } else if constexpr (__is_abi<_Abi, simd_abi::fixed_size>()) {
        return __k.any();
    } else if constexpr (__is_combined_abi<_Abi>()) {
        for (int __i = 0; __i < _Abi::factor; ++__i) {
            if (__any_of<_Tp, typename _Abi::member_abi>(__k[__i])) {
                return true;
            }
        }
        return false;
    } else if constexpr (__is_abi<_Abi, simd_abi::__sse_abi>() ||
                         __is_abi<_Abi, simd_abi::__avx_abi>()) {
        constexpr size_t _N = simd_size_v<_Tp, _Abi>;
        if constexpr (__have_sse4_1) {
            return 0 == __testz(__k._M_data, _Abi::template implicit_mask<_Tp>);
        } else if constexpr (std::is_same_v<_Tp, float>) {
            return (_mm_movemask_ps(__to_intrin(__k)) & ((1 << _N) - 1)) != 0;
        } else if constexpr (std::is_same_v<_Tp, double>) {
            return (_mm_movemask_pd(__to_intrin(__k)) & ((1 << _N) - 1)) != 0;
        } else {
            return (_mm_movemask_epi8(__to_intrin(__k)) & ((1 << (_N * sizeof(_Tp))) - 1)) != 0;
        }
    } else if constexpr (__is_abi<_Abi, simd_abi::__avx512_abi>()) {
        return (__k & _Abi::template implicit_mask<_Tp>) != 0;
    } else {
        __assert_unreachable<_Tp>();
    }
}

// }}}
// __none_of {{{
template <class _Tp, class _Abi, class _Data> _GLIBCXX_SIMD_INTRINSIC bool __none_of(const _Data &__k)
{
    if constexpr (__is_abi<_Abi, simd_abi::scalar>()) {
        return !__k;
    } else if constexpr (__is_abi<_Abi, simd_abi::fixed_size>()) {
        return __k.none();
    } else if constexpr (__is_combined_abi<_Abi>()) {
        for (int __i = 0; __i < _Abi::factor; ++__i) {
            if (__any_of<_Tp, typename _Abi::member_abi>(__k[__i])) {
                return false;
            }
        }
        return true;
    } else if constexpr (__is_abi<_Abi, simd_abi::__sse_abi>() ||
                         __is_abi<_Abi, simd_abi::__avx_abi>()) {
        constexpr size_t _N = simd_size_v<_Tp, _Abi>;
        if constexpr (__have_sse4_1) {
            return 0 != __testz(__k._M_data, _Abi::template implicit_mask<_Tp>);
        } else if constexpr (std::is_same_v<_Tp, float>) {
            return (_mm_movemask_ps(__to_intrin(__k)) & ((1 << _N) - 1)) == 0;
        } else if constexpr (std::is_same_v<_Tp, double>) {
            return (_mm_movemask_pd(__to_intrin(__k)) & ((1 << _N) - 1)) == 0;
        } else {
            return (_mm_movemask_epi8(__to_intrin(__k)) & ((1 << (_N * sizeof(_Tp))) - 1)) == 0;
        }
    } else if constexpr (__is_abi<_Abi, simd_abi::__avx512_abi>()) {
        return (__k & _Abi::template implicit_mask<_Tp>) == 0;
    } else {
        __assert_unreachable<_Tp>();
    }
}

// }}}
// __some_of {{{
template <class _Tp, class _Abi, class _Data> _GLIBCXX_SIMD_INTRINSIC bool __some_of(const _Data &__k)
{
    if constexpr (__is_abi<_Abi, simd_abi::scalar>()) {
        return false;
    } else if constexpr (__is_abi<_Abi, simd_abi::fixed_size>()) {
        return __k.any() && !__k.all();
    } else if constexpr (__is_combined_abi<_Abi>()) {
        return __any_of<_Tp, _Abi>(__k) && !__all_of<_Tp, _Abi>(__k);
    } else if constexpr (__is_abi<_Abi, simd_abi::__sse_abi>() ||
                         __is_abi<_Abi, simd_abi::__avx_abi>()) {
        constexpr size_t _N = simd_size_v<_Tp, _Abi>;
        if constexpr (__have_sse4_1) {
            return 0 != __testnzc(__k._M_data, _Abi::template implicit_mask<_Tp>);
        } else if constexpr (std::is_same_v<_Tp, float>) {
            constexpr int __allbits = (1 << _N) - 1;
            const auto __tmp = _mm_movemask_ps(__to_intrin(__k)) & __allbits;
            return __tmp > 0 && __tmp < __allbits;
        } else if constexpr (std::is_same_v<_Tp, double>) {
            constexpr int __allbits = (1 << _N) - 1;
            const auto __tmp = _mm_movemask_pd(__to_intrin(__k)) & __allbits;
            return __tmp > 0 && __tmp < __allbits;
        } else {
            constexpr int __allbits = (1 << (_N * sizeof(_Tp))) - 1;
            const auto __tmp = _mm_movemask_epi8(__to_intrin(__k)) & __allbits;
            return __tmp > 0 && __tmp < __allbits;
        }
    } else if constexpr (__is_abi<_Abi, simd_abi::__avx512_abi>()) {
        return __any_of<_Tp, _Abi>(__k) && !__all_of<_Tp, _Abi>(__k);
    } else {
        __assert_unreachable<_Tp>();
    }
}

// }}}
// __popcount {{{
template <class _Tp, class _Abi, class _Data> _GLIBCXX_SIMD_INTRINSIC int __popcount(const _Data &__k)
{
    if constexpr (__is_abi<_Abi, simd_abi::scalar>()) {
        return __k;
    } else if constexpr (__is_abi<_Abi, simd_abi::fixed_size>()) {
        return __k.count();
    } else if constexpr (__is_combined_abi<_Abi>()) {
        int __count = __popcount<_Tp, typename _Abi::member_abi>(__k[0]);
        for (int __i = 1; __i < _Abi::factor; ++__i) {
            __count += __popcount<_Tp, typename _Abi::member_abi>(__k[__i]);
        }
        return __count;
    } else if constexpr (__is_abi<_Abi, simd_abi::__sse_abi>() ||
                         __is_abi<_Abi, simd_abi::__avx_abi>()) {
        constexpr size_t _N = simd_size_v<_Tp, _Abi>;
        const auto __kk = _Abi::masked(__k._M_data);
        if constexpr (__have_popcnt) {
            int __bits = __movemask(__to_intrin(__vector_bitcast<_Tp>(__kk)));
            const int __count = __builtin_popcount(__bits);
            return std::is_integral_v<_Tp> ? __count / sizeof(_Tp) : __count;
        } else if constexpr (_N == 2) {
            const int mask = _mm_movemask_pd(__auto_bitcast(__kk));
            return mask - (mask >> 1);
        } else if constexpr (_N == 4 && sizeof(__kk) == 16 && __have_sse2) {
            auto __x = __vector_bitcast<_LLong>(__kk);
            __x = _mm_add_epi32(__x, _mm_shuffle_epi32(__x, _MM_SHUFFLE(0, 1, 2, 3)));
            __x = _mm_add_epi32(__x, _mm_shufflelo_epi16(__x, _MM_SHUFFLE(1, 0, 3, 2)));
            return -_mm_cvtsi128_si32(__x);
        } else if constexpr (_N == 4 && sizeof(__kk) == 16) {
            return __builtin_popcount(_mm_movemask_ps(__auto_bitcast(__kk)));
        } else if constexpr (_N == 8 && sizeof(__kk) == 16) {
            auto __x = __vector_bitcast<_LLong>(__kk);
            __x = _mm_add_epi16(__x, _mm_shuffle_epi32(__x, _MM_SHUFFLE(0, 1, 2, 3)));
            __x = _mm_add_epi16(__x, _mm_shufflelo_epi16(__x, _MM_SHUFFLE(0, 1, 2, 3)));
            __x = _mm_add_epi16(__x, _mm_shufflelo_epi16(__x, _MM_SHUFFLE(2, 3, 0, 1)));
            return -short(_mm_extract_epi16(__x, 0));
        } else if constexpr (_N == 16 && sizeof(__kk) == 16) {
            auto __x = __vector_bitcast<_LLong>(__kk);
            __x = _mm_add_epi8(__x, _mm_shuffle_epi32(__x, _MM_SHUFFLE(0, 1, 2, 3)));
            __x = _mm_add_epi8(__x, _mm_shufflelo_epi16(__x, _MM_SHUFFLE(0, 1, 2, 3)));
            __x = _mm_add_epi8(__x, _mm_shufflelo_epi16(__x, _MM_SHUFFLE(2, 3, 0, 1)));
            auto __y = -__vector_bitcast<_UChar>(__x);
            if constexpr (__have_sse4_1) {
                return __y[0] + __y[1];
            } else {
                unsigned __z = _mm_extract_epi16(__vector_bitcast<_LLong>(__y), 0);
                return (__z & 0xff) + (__z >> 8);
            }
        } else if constexpr (_N == 4 && sizeof(__kk) == 32) {
            auto __x = -(__lo128(__kk) + __hi128(__kk));
            return __x[0] + __x[1];
        } else if constexpr (sizeof(__kk) == 32) {
            return __popcount<_Tp, simd_abi::__sse>(-(__lo128(__kk) + __hi128(__kk)));
        } else {
            __assert_unreachable<_Tp>();
        }
    } else if constexpr (__is_abi<_Abi, simd_abi::__avx512_abi>()) {
        constexpr size_t _N = simd_size_v<_Tp, _Abi>;
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
            __assert_unreachable<_Tp>();
        }
    } else {
        __assert_unreachable<_Tp>();
    }
}

// }}}
// __find_first_set {{{
template <class _Tp, class _Abi, class _Data> _GLIBCXX_SIMD_INTRINSIC int __find_first_set(const _Data &__k)
{
    if constexpr (__is_abi<_Abi, simd_abi::scalar>()) {
        return 0;
    } else if constexpr (__is_abi<_Abi, simd_abi::fixed_size>()) {
        return __firstbit(__k.to_ullong());
    } else if constexpr (__is_combined_abi<_Abi>()) {
        using _A2 = typename _Abi::member_abi;
        for (int __i = 0; __i < _Abi::factor - 1; ++__i) {
            if (__any_of<_Tp, _A2>(__k[__i])) {
                return __i * simd_size_v<_Tp, _A2> + __find_first_set(__k[__i]);
            }
        }
        return (_Abi::factor - 1) * simd_size_v<_Tp, _A2> +
               __find_first_set(__k[_Abi::factor - 1]);
    } else if constexpr (__is_abi<_Abi, simd_abi::__sse_abi>() ||
                         __is_abi<_Abi, simd_abi::__avx_abi>()) {
        return __firstbit(__vector_to_bitset(__k._M_data).to_ullong());
    } else if constexpr (__is_abi<_Abi, simd_abi::__avx512_abi>()) {
        if constexpr (simd_size_v<_Tp, _Abi> <= 32) {
            return _tzcnt_u32(__k._M_data);
        } else {
            return __firstbit(__k._M_data);
        }
    } else {
        __assert_unreachable<_Tp>();
    }
}

// }}}
// __find_last_set {{{
template <class _Tp, class _Abi, class _Data> _GLIBCXX_SIMD_INTRINSIC int __find_last_set(const _Data &__k)
{
    if constexpr (__is_abi<_Abi, simd_abi::scalar>()) {
        return 0;
    } else if constexpr (__is_abi<_Abi, simd_abi::fixed_size>()) {
        return __lastbit(__k.to_ullong());
    } else if constexpr (__is_combined_abi<_Abi>()) {
        using _A2 = typename _Abi::member_abi;
        for (int __i = 0; __i < _Abi::factor - 1; ++__i) {
            if (__any_of<_Tp, _A2>(__k[__i])) {
                return __i * simd_size_v<_Tp, _A2> + __find_last_set(__k[__i]);
            }
        }
        return (_Abi::factor - 1) * simd_size_v<_Tp, _A2> + __find_last_set(__k[_Abi::factor - 1]);
    } else if constexpr (__is_abi<_Abi, simd_abi::__sse_abi>() ||
                         __is_abi<_Abi, simd_abi::__avx_abi>()) {
        return __lastbit(__vector_to_bitset(__k._M_data).to_ullong());
    } else if constexpr (__is_abi<_Abi, simd_abi::__avx512_abi>()) {
        if constexpr (simd_size_v<_Tp, _Abi> <= 32) {
            return 31 - _lzcnt_u32(__k._M_data);
        } else {
            return __lastbit(__k._M_data);
        }
    } else {
        __assert_unreachable<_Tp>();
    }
}

// }}}

// reductions [simd_mask.reductions] {{{
template <class _Tp, class _Abi> _GLIBCXX_SIMD_ALWAYS_INLINE bool all_of(const simd_mask<_Tp, _Abi> &__k)
{
    return __all_of<_Tp, _Abi>(__data(__k));
}
template <class _Tp, class _Abi> _GLIBCXX_SIMD_ALWAYS_INLINE bool any_of(const simd_mask<_Tp, _Abi> &__k)
{
    return __any_of<_Tp, _Abi>(__data(__k));
}
template <class _Tp, class _Abi> _GLIBCXX_SIMD_ALWAYS_INLINE bool none_of(const simd_mask<_Tp, _Abi> &__k)
{
    return __none_of<_Tp, _Abi>(__data(__k));
}
template <class _Tp, class _Abi> _GLIBCXX_SIMD_ALWAYS_INLINE bool some_of(const simd_mask<_Tp, _Abi> &__k)
{
    return __some_of<_Tp, _Abi>(__data(__k));
}
template <class _Tp, class _Abi> _GLIBCXX_SIMD_ALWAYS_INLINE int popcount(const simd_mask<_Tp, _Abi> &__k)
{
    return __popcount<_Tp, _Abi>(__data(__k));
}
template <class _Tp, class _Abi>
_GLIBCXX_SIMD_ALWAYS_INLINE int find_first_set(const simd_mask<_Tp, _Abi> &__k)
{
    return __find_first_set<_Tp, _Abi>(__data(__k));
}
template <class _Tp, class _Abi>
_GLIBCXX_SIMD_ALWAYS_INLINE int find_last_set(const simd_mask<_Tp, _Abi> &__k)
{
    return __find_last_set<_Tp, _Abi>(__data(__k));
}

constexpr bool all_of(_ExactBool __x) { return __x; }
constexpr bool any_of(_ExactBool __x) { return __x; }
constexpr bool none_of(_ExactBool __x) { return !__x; }
constexpr bool some_of(_ExactBool) { return false; }
constexpr int popcount(_ExactBool __x) { return __x; }
constexpr int find_first_set(_ExactBool) { return 0; }
constexpr int find_last_set(_ExactBool) { return 0; }

// }}}

template <class _Abi> struct __generic_simd_impl;
// __allow_conversion_ctor2{{{1
template <class _T0, class _T1, class _A, bool _BothIntegral> struct __allow_conversion_ctor2_1;

template <class _T0, class _T1, class _A>
struct __allow_conversion_ctor2
    : public __allow_conversion_ctor2_1<
          _T0, _T1, _A, conjunction<std::is_integral<_T0>, std::is_integral<_T1>>::value> {
};

// disallow 2nd conversion ctor (equal _Abi), if the value_types are equal (copy ctor)
template <class _Tp, class _A> struct __allow_conversion_ctor2<_Tp, _Tp, _A> : public false_type {};

// disallow 2nd conversion ctor (equal _Abi), if the _Abi is a fixed_size instance
template <class _T0, class _T1, int _N>
struct __allow_conversion_ctor2<_T0, _T1, simd_abi::fixed_size<_N>> : public false_type {};

// disallow 2nd conversion ctor (equal _Abi), if both of the above are true
template <class _Tp, int _N>
struct __allow_conversion_ctor2<_Tp, _Tp, simd_abi::fixed_size<_N>> : public false_type {};

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

// _Simd_int_operators{{{1
template <class _V, bool> class _Simd_int_operators {};
template <class _V> class _Simd_int_operators<_V, true>
{
    using __impl = __get_impl_t<_V>;

    _GLIBCXX_SIMD_INTRINSIC const _V &__derived() const { return *static_cast<const _V *>(this); }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC static _V __make_derived(_Tp &&__d)
    {
        return {__private_init, std::forward<_Tp>(__d)};
    }

public:
    constexpr friend _V &operator %=(_V &__lhs, const _V &__x) { return __lhs = __lhs  % __x; }
    constexpr friend _V &operator &=(_V &__lhs, const _V &__x) { return __lhs = __lhs  & __x; }
    constexpr friend _V &operator |=(_V &__lhs, const _V &__x) { return __lhs = __lhs  | __x; }
    constexpr friend _V &operator ^=(_V &__lhs, const _V &__x) { return __lhs = __lhs  ^ __x; }
    constexpr friend _V &operator<<=(_V &__lhs, const _V &__x) { return __lhs = __lhs << __x; }
    constexpr friend _V &operator>>=(_V &__lhs, const _V &__x) { return __lhs = __lhs >> __x; }
    constexpr friend _V &operator<<=(_V &__lhs, int __x) { return __lhs = __lhs << __x; }
    constexpr friend _V &operator>>=(_V &__lhs, int __x) { return __lhs = __lhs >> __x; }

    constexpr friend _V operator% (const _V &__x, const _V &__y) { return _Simd_int_operators::__make_derived(__impl::modulus        (__data(__x), __data(__y))); }
    constexpr friend _V operator& (const _V &__x, const _V &__y) { return _Simd_int_operators::__make_derived(__impl::bit_and        (__data(__x), __data(__y))); }
    constexpr friend _V operator| (const _V &__x, const _V &__y) { return _Simd_int_operators::__make_derived(__impl::bit_or         (__data(__x), __data(__y))); }
    constexpr friend _V operator^ (const _V &__x, const _V &__y) { return _Simd_int_operators::__make_derived(__impl::bit_xor        (__data(__x), __data(__y))); }
    constexpr friend _V operator<<(const _V &__x, const _V &__y) { return _Simd_int_operators::__make_derived(__impl::bit_shift_left (__data(__x), __data(__y))); }
    constexpr friend _V operator>>(const _V &__x, const _V &__y) { return _Simd_int_operators::__make_derived(__impl::bit_shift_right(__data(__x), __data(__y))); }
    constexpr friend _V operator<<(const _V &__x, int __y)      { return _Simd_int_operators::__make_derived(__impl::bit_shift_left (__data(__x), __y)); }
    constexpr friend _V operator>>(const _V &__x, int __y)      { return _Simd_int_operators::__make_derived(__impl::bit_shift_right(__data(__x), __y)); }

    // unary operators (for integral _Tp)
    constexpr _V operator~() const
    {
        return {__private_init, __impl::complement(__derived()._M_data)};
    }
};

//}}}1

// simd {{{
template <class _Tp, class _Abi>
class simd
    : public _Simd_int_operators<
          simd<_Tp, _Abi>, conjunction<std::is_integral<_Tp>,
                                    typename _SimdTraits<_Tp, _Abi>::_IsValid>::value>,
      public _SimdTraits<_Tp, _Abi>::_Simd_base
{
    using __traits = _SimdTraits<_Tp, _Abi>;
    using __impl = typename __traits::_SimdImpl;
    using __member_type = typename __traits::_SimdMember;
    using __cast_type = typename __traits::_SimdCastType;
    static constexpr _Tp *_S_type_tag = nullptr;
    friend typename __traits::_Simd_base;
    friend __impl;
    friend __generic_simd_impl<_Abi>;
    friend _Simd_int_operators<simd, true>;

public:
    using value_type = _Tp;
    using reference = _Smart_reference<__member_type, __impl, value_type>;
    using mask_type = simd_mask<_Tp, _Abi>;
    using abi_type = _Abi;

    static constexpr size_t size() { return __size_or_zero_v<_Tp, _Abi>; }
    constexpr simd() = default;
    constexpr simd(const simd &) = default;
    constexpr simd(simd &&) = default;
    constexpr simd &operator=(const simd &) = default;
    constexpr simd &operator=(simd &&) = default;

    // implicit broadcast constructor
    template <class _U, class = _ValuePreservingOrInt<_U, value_type>>
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
        _F &&__gen,
        _ValuePreservingOrInt<
            decltype(std::declval<_F>()(std::declval<_SizeConstant<0> &>())),
            value_type> * = nullptr)
        : _M_data(__impl::generator(std::forward<_F>(__gen), _S_type_tag))
    {
    }

    // load constructor
    template <class _U, class _Flags>
    _GLIBCXX_SIMD_ALWAYS_INLINE simd(const _U *__mem, _Flags __f)
        : _M_data(__impl::load(__mem, __f, _S_type_tag))
    {
    }

    // loads [simd.load]
    template <class _U, class _Flags>
    _GLIBCXX_SIMD_ALWAYS_INLINE void copy_from(const _Vectorizable<_U> *__mem, _Flags __f)
    {
        _M_data = static_cast<decltype(_M_data)>(__impl::load(__mem, __f, _S_type_tag));
    }

    // stores [simd.store]
    template <class _U, class _Flags>
    _GLIBCXX_SIMD_ALWAYS_INLINE void copy_to(_Vectorizable<_U> *__mem, _Flags __f) const
    {
        __impl::store(_M_data, __mem, __f, _S_type_tag);
    }

    // scalar access
    _GLIBCXX_SIMD_ALWAYS_INLINE constexpr reference operator[](size_t __i) { return {_M_data, int(__i)}; }
    _GLIBCXX_SIMD_ALWAYS_INLINE constexpr value_type operator[](size_t __i) const
    {
        if constexpr (__is_scalar()) {
            _GLIBCXX_DEBUG_ASSERT(__i == 0);
            __unused(__i);
            return _M_data;
        } else {
            return _M_data[__i];
        }
    }

    // increment and decrement:
    _GLIBCXX_SIMD_ALWAYS_INLINE constexpr simd &operator++() { __impl::__increment(_M_data); return *this; }
    _GLIBCXX_SIMD_ALWAYS_INLINE constexpr simd operator++(int) { simd __r = *this; __impl::__increment(_M_data); return __r; }
    _GLIBCXX_SIMD_ALWAYS_INLINE constexpr simd &operator--() { __impl::__decrement(_M_data); return *this; }
    _GLIBCXX_SIMD_ALWAYS_INLINE constexpr simd operator--(int) { simd __r = *this; __impl::__decrement(_M_data); return __r; }

    // unary operators (for any _Tp)
    _GLIBCXX_SIMD_ALWAYS_INLINE constexpr mask_type operator!() const
    {
        return {__private_init, __impl::negate(_M_data)};
    }
    _GLIBCXX_SIMD_ALWAYS_INLINE constexpr simd operator+() const { return *this; }
    _GLIBCXX_SIMD_ALWAYS_INLINE constexpr simd operator-() const
    {
        return {__private_init, __impl::unary_minus(_M_data)};
    }

    // access to internal representation (suggested extension)
    _GLIBCXX_SIMD_ALWAYS_INLINE explicit simd(__cast_type __init) : _M_data(__init) {}

    // compound assignment [simd.cassign]
    _GLIBCXX_SIMD_ALWAYS_INLINE constexpr friend simd &operator+=(simd &__lhs, const simd &__x) { return __lhs = __lhs + __x; }
    _GLIBCXX_SIMD_ALWAYS_INLINE constexpr friend simd &operator-=(simd &__lhs, const simd &__x) { return __lhs = __lhs - __x; }
    _GLIBCXX_SIMD_ALWAYS_INLINE constexpr friend simd &operator*=(simd &__lhs, const simd &__x) { return __lhs = __lhs * __x; }
    _GLIBCXX_SIMD_ALWAYS_INLINE constexpr friend simd &operator/=(simd &__lhs, const simd &__x) { return __lhs = __lhs / __x; }

    // binary operators [simd.binary]
    _GLIBCXX_SIMD_ALWAYS_INLINE constexpr friend simd operator+(const simd &__x, const simd &__y)
    {
        return {__private_init, __impl::plus(__x._M_data, __y._M_data)};
    }
    _GLIBCXX_SIMD_ALWAYS_INLINE constexpr friend simd operator-(const simd &__x, const simd &__y)
    {
        return {__private_init, __impl::minus(__x._M_data, __y._M_data)};
    }
    _GLIBCXX_SIMD_ALWAYS_INLINE constexpr friend simd operator*(const simd& __x, const simd& __y)
    {
        return {__private_init, __impl::multiplies(__x._M_data, __y._M_data)};
    }
    _GLIBCXX_SIMD_ALWAYS_INLINE constexpr friend simd operator/(const simd &__x, const simd &__y)
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
    _GLIBCXX_SIMD_INTRINSIC constexpr simd(_PrivateInit,
					   const __member_type& __init)
    : _M_data(__init)
    {
    }

    // "private" because of the first arguments's namespace
    _GLIBCXX_SIMD_INTRINSIC simd(_BitsetInit, std::bitset<size()> __init) : _M_data() {
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
    alignas(__traits::_S_simd_align) __member_type _M_data;
};

// }}}
// __data {{{
template <class _Tp, class _A> _GLIBCXX_SIMD_INTRINSIC constexpr const auto &__data(const simd<_Tp, _A> &__x)
{
    return __x._M_data;
}
template <class _Tp, class _A> _GLIBCXX_SIMD_INTRINSIC constexpr auto &__data(simd<_Tp, _A> &__x)
{
    return __x._M_data;
}
// }}}

namespace __proposed
{
namespace float_bitwise_operators
{
// float_bitwise_operators {{{
template <class _Tp, class _A>
_GLIBCXX_SIMD_INTRINSIC constexpr simd<_Tp, _A>
			operator^(const simd<_Tp, _A>& __a, const simd<_Tp, _A>& __b)
{
  return {__private_init,
	  __get_impl_t<simd<_Tp, _A>>::bit_xor(__data(__a), __data(__b))};
}

template <class _Tp, class _A>
_GLIBCXX_SIMD_INTRINSIC constexpr simd<_Tp, _A> operator|(const simd<_Tp, _A> &__a, const simd<_Tp, _A> &__b)
{
    return {__private_init, __get_impl_t<simd<_Tp, _A>>::bit_or(__data(__a), __data(__b))};
}

template <class _Tp, class _A>
_GLIBCXX_SIMD_INTRINSIC constexpr simd<_Tp, _A> operator&(const simd<_Tp, _A> &__a, const simd<_Tp, _A> &__b)
{
    return {__private_init, __get_impl_t<simd<_Tp, _A>>::bit_and(__data(__a), __data(__b))};
}
// }}}
}  // namespace float_bitwise_operators
}  // namespace __proposed

_GLIBCXX_SIMD_END_NAMESPACE

#endif  // __cplusplus >= 201703L
#endif  // _GLIBCXX_EXPERIMENTAL_SIMD_H
// vim: foldmethod=marker
