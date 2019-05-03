// Simd Abi specific implementations -*- C++ -*-

// Copyright © 2015-2019 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH
//                       Matthias Kretz <m.kretz@gsi.de>
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the names of contributing organizations nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef _GLIBCXX_EXPERIMENTAL_SIMD_ABIS_H_
#define _GLIBCXX_EXPERIMENTAL_SIMD_ABIS_H_

#if __cplusplus >= 201703L

#include "simd.h"
#include <array>
#include <cmath>
#include <cstdlib>

#include "simd_debug.h"
_GLIBCXX_SIMD_BEGIN_NAMESPACE
template <typename _V, typename = _VectorTraits<_V>>
static inline constexpr _V _S_signmask = __xor(_V() + 1, _V() - 1);
template <typename _V, typename = _VectorTraits<_V>>
static inline constexpr _V _S_absmask = __andnot(_S_signmask<_V>, __allbits<_V>);

// __subscript_read/_write {{{1
template <typename _Tp> _Tp __subscript_read(_Vectorizable<_Tp> __x, size_t) noexcept
{
    return __x;
}
template <typename _Tp>
void __subscript_write(_Vectorizable<_Tp> &__x, size_t, __id<_Tp> __y) noexcept
{
    __x = __y;
}

template <typename _Tp>
typename _Tp::value_type __subscript_read(const _Tp &__x, size_t __i) noexcept
{
    return __x[__i];
}
template <typename _Tp>
void __subscript_write(_Tp &__x, size_t __i, typename _Tp::value_type __y) noexcept
{
    return __x.__set(__i, __y);
}

// __simd_tuple_element {{{1
template <size_t _I, typename _Tp> struct __simd_tuple_element;
template <typename _Tp, typename _A0, typename... _As>
struct __simd_tuple_element<0, _SimdTuple<_Tp, _A0, _As...>> {
    using type = std::experimental::simd<_Tp, _A0>;
};
template <size_t _I, typename _Tp, typename _A0, typename... _As>
struct __simd_tuple_element<_I, _SimdTuple<_Tp, _A0, _As...>> {
    using type = typename __simd_tuple_element<_I - 1, _SimdTuple<_Tp, _As...>>::type;
};
template <size_t _I, typename _Tp>
using __simd_tuple_element_t = typename __simd_tuple_element<_I, _Tp>::type;

// __simd_tuple_concat {{{1
template <typename _Tp, typename... _A0s, typename... _A1s>
_GLIBCXX_SIMD_INTRINSIC constexpr _SimdTuple<_Tp, _A0s..., _A1s...>
  __simd_tuple_concat(const _SimdTuple<_Tp, _A0s...>& __left,
		      const _SimdTuple<_Tp, _A1s...>& __right)
{
  if constexpr (sizeof...(_A0s) == 0)
    return __right;
  else if constexpr (sizeof...(_A1s) == 0)
    return __left;
  else
    return {__left.first, __simd_tuple_concat(__left.second, __right)};
}

template <typename _Tp, typename _A10, typename... _A1s>
_GLIBCXX_SIMD_INTRINSIC constexpr _SimdTuple<_Tp,
					       simd_abi::scalar,
					       _A10,
					       _A1s...>
  __simd_tuple_concat(const _Tp&                              __left,
		      const _SimdTuple<_Tp, _A10, _A1s...>& __right)
{
  return {__left, __right};
}

// __simd_tuple_pop_front {{{1
template <size_t _N, typename _Tp>
_GLIBCXX_SIMD_INTRINSIC constexpr decltype(auto)
  __simd_tuple_pop_front(_Tp&& __x)
{
  if constexpr (_N == 0)
    return std::forward<_Tp>(__x);
  else
    return __simd_tuple_pop_front<_N - 1>(__x.second);
}

// __get_simd_at<_N> {{{1
struct __as_simd {};
struct __as_simd_tuple {};
template <typename _Tp, typename _A0, typename... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr simd<_Tp, _A0> __simd_tuple_get_impl(
    __as_simd, const _SimdTuple<_Tp, _A0, _Abis...> &__t, _SizeConstant<0>)
{
    return {__private_init, __t.first};
}
template <typename _Tp, typename _A0, typename... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr const auto &__simd_tuple_get_impl(
    __as_simd_tuple, const _SimdTuple<_Tp, _A0, _Abis...> &__t, _SizeConstant<0>)
{
    return __t.first;
}
template <typename _Tp, typename _A0, typename... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto &__simd_tuple_get_impl(
    __as_simd_tuple, _SimdTuple<_Tp, _A0, _Abis...> &__t, _SizeConstant<0>)
{
    return __t.first;
}

template <typename _R, size_t _N, typename _Tp, typename... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto __simd_tuple_get_impl(
    _R, const _SimdTuple<_Tp, _Abis...> &__t, _SizeConstant<_N>)
{
    return __simd_tuple_get_impl(_R(), __t.second, _SizeConstant<_N - 1>());
}
template <size_t _N, typename _Tp, typename... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto &__simd_tuple_get_impl(
    __as_simd_tuple, _SimdTuple<_Tp, _Abis...> &__t, _SizeConstant<_N>)
{
    return __simd_tuple_get_impl(__as_simd_tuple(), __t.second, _SizeConstant<_N - 1>());
}

template <size_t _N, typename _Tp, typename... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto __get_simd_at(const _SimdTuple<_Tp, _Abis...> &__t)
{
    return __simd_tuple_get_impl(__as_simd(), __t, _SizeConstant<_N>());
}

// }}}
// __get_tuple_at<_N> {{{
template <size_t _N, typename _Tp, typename... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto __get_tuple_at(const _SimdTuple<_Tp, _Abis...> &__t)
{
    return __simd_tuple_get_impl(__as_simd_tuple(), __t, _SizeConstant<_N>());
}

template <size_t _N, typename _Tp, typename... _Abis>
_GLIBCXX_SIMD_INTRINSIC constexpr auto &__get_tuple_at(_SimdTuple<_Tp, _Abis...> &__t)
{
    return __simd_tuple_get_impl(__as_simd_tuple(), __t, _SizeConstant<_N>());
}

// __tuple_element_meta {{{1
template <typename _Tp, typename _Abi, size_t _Offset>
struct __tuple_element_meta : public _Abi::_SimdImpl {
  static_assert(is_same_v<typename _Abi::_SimdImpl::abi_type,
			  _Abi>); // this fails e.g. when _SimdImpl is an alias
				  // for _SimdImplBuiltin<_DifferentAbi>
  using value_type                    = _Tp;
  using abi_type                      = _Abi;
  using _Traits                       = _SimdTraits<_Tp, _Abi>;
  using _MaskImpl                     = typename _Traits::_MaskImpl;
  using _MaskMember                   = typename _Traits::_MaskMember;
  using simd_type                     = std::experimental::simd<_Tp, _Abi>;
  static constexpr size_t    _S_offset = _Offset;
  static constexpr size_t    size() { return simd_size<_Tp, _Abi>::value; }
  static constexpr _MaskImpl _S_mask_impl = {};

  template <size_t _N>
  _GLIBCXX_SIMD_INTRINSIC static _MaskMember __make_mask(std::bitset<_N> __bits)
  {
    constexpr _Tp* __type_tag = nullptr;
    return _MaskImpl::__from_bitset(
      std::bitset<size()>((__bits >> _Offset).to_ullong()), __type_tag);
  }

  _GLIBCXX_SIMD_INTRINSIC static _ULLong __mask_to_shifted_ullong(_MaskMember __k)
  {
    return __vector_to_bitset(__k).to_ullong() << _Offset;
  }
};

template <size_t _Offset, typename _Tp, typename _Abi, typename... _As>
__tuple_element_meta<_Tp, _Abi, _Offset> __make_meta(const _SimdTuple<_Tp, _Abi, _As...> &)
{
    return {};
}

// }}}1
// _WithOffset wrapper class {{{
template <size_t _Offset, typename _Base>
struct _WithOffset : public _Base
{
  static inline constexpr size_t _S_offset = _Offset;

  _GLIBCXX_SIMD_INTRINSIC char* __as_charptr()
  {
    return reinterpret_cast<char*>(this) +
	   _S_offset * sizeof(typename _Base::value_type);
  }
  _GLIBCXX_SIMD_INTRINSIC const char* __as_charptr() const
  {
    return reinterpret_cast<const char*>(this) +
	   _S_offset * sizeof(typename _Base::value_type);
  }
};

// make _WithOffset<_WithOffset> ill-formed to use:
template <size_t _O0, size_t _O1, typename _Base>
struct _WithOffset<_O0, _WithOffset<_O1, _Base>> {};

template <size_t _Offset, typename _Tp>
decltype(auto) __add_offset(_Tp& __base)
{
  return static_cast<_WithOffset<_Offset, __remove_cvref_t<_Tp>>&>(__base);
}
template <size_t _Offset, typename _Tp>
decltype(auto) __add_offset(const _Tp& __base)
{
  return static_cast<const _WithOffset<_Offset, __remove_cvref_t<_Tp>>&>(
    __base);
}
template <size_t _Offset, size_t _ExistingOffset, typename _Tp>
decltype(auto) __add_offset(_WithOffset<_ExistingOffset, _Tp>& __base)
{
  return static_cast<_WithOffset<_Offset + _ExistingOffset, _Tp>&>(
    static_cast<_Tp&>(__base));
}
template <size_t _Offset, size_t _ExistingOffset, typename _Tp>
decltype(auto) __add_offset(const _WithOffset<_ExistingOffset, _Tp>& __base)
{
  return static_cast<const _WithOffset<_Offset + _ExistingOffset, _Tp>&>(
    static_cast<const _Tp&>(__base));
}

template <typename _Tp>
constexpr inline size_t __offset = 0;
template <size_t _Offset, typename _Tp>
constexpr inline size_t __offset<_WithOffset<_Offset, _Tp>> =
  _WithOffset<_Offset, _Tp>::_S_offset;
template <typename _Tp>
constexpr inline size_t __offset<const _Tp> = __offset<_Tp>;
template <typename _Tp>
constexpr inline size_t __offset<_Tp&> = __offset<_Tp>;
template <typename _Tp>
constexpr inline size_t __offset<_Tp&&> = __offset<_Tp>;

// }}}
// _SimdTuple specializations {{{1
// empty {{{2
template <typename _Tp> struct _SimdTuple<_Tp> {
    using value_type = _Tp;
    static constexpr size_t _S_tuple_size = 0;
    static constexpr size_t size() { return 0; }
};

// _SimdTupleData {{{2
template <typename _FirstType, typename _SecondType>
struct _SimdTupleData
{
  _FirstType  first;
  _SecondType second;
};

template <typename _FirstType, typename _Tp>
struct _SimdTupleData<_FirstType, _SimdTuple<_Tp>>
{
  _FirstType  first;
  static constexpr _SimdTuple<_Tp> second = {};
};

// 1 or more {{{2
template <class _Tp, class _Abi0, class... _Abis>
struct _SimdTuple<_Tp, _Abi0, _Abis...>
: _SimdTupleData<typename _SimdTraits<_Tp, _Abi0>::_SimdMember,
		 _SimdTuple<_Tp, _Abis...>>
{
  using value_type   = _Tp;
  using _First_type  = typename _SimdTraits<_Tp, _Abi0>::_SimdMember;
  using _First_abi   = _Abi0;
  using _Second_type = _SimdTuple<_Tp, _Abis...>;
  static constexpr size_t _S_tuple_size = sizeof...(_Abis) + 1;
  static constexpr size_t size()
  {
    return simd_size_v<_Tp, _Abi0> + _Second_type::size();
  }
  static constexpr size_t _S_first_size = simd_size_v<_Tp, _Abi0>;

  using _Base = _SimdTupleData<typename _SimdTraits<_Tp, _Abi0>::_SimdMember,
			       _SimdTuple<_Tp, _Abis...>>;
  using _Base::first;
  using _Base::second;

  _SimdTuple() = default;

  _GLIBCXX_SIMD_INTRINSIC char* __as_charptr()
  {
    return reinterpret_cast<char*>(this);
  }
  _GLIBCXX_SIMD_INTRINSIC const char* __as_charptr() const
  {
    return reinterpret_cast<const char*>(this);
  }

  template <size_t _Offset = 0, class _F>
  _GLIBCXX_SIMD_INTRINSIC static constexpr _SimdTuple
    __generate(_F&& __gen, _SizeConstant<_Offset> = {})
  {
    auto &&__first = __gen(__tuple_element_meta<_Tp, _Abi0, _Offset>());
    if constexpr (_S_tuple_size == 1)
      return {__first};
    else
      return {__first, _Second_type::__generate(
			 std::forward<_F>(__gen),
			 _SizeConstant<_Offset + simd_size_v<_Tp, _Abi0>>())};
  }

  template <size_t _Offset = 0, class _F, class... _More>
  _GLIBCXX_SIMD_INTRINSIC _SimdTuple
			  __apply_wrapped(_F&& __fun, const _More&... __more) const
  {
    _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
    auto&& __first= __fun(__make_meta<_Offset>(*this), first, __more.first...);
    if constexpr (_S_tuple_size == 1)
      return { __first };
    else
      return {
	__first,
	second.template __apply_wrapped<_Offset + simd_size_v<_Tp, _Abi0>>(
	  std::forward<_F>(__fun), __more.second...)};
  }

  template <size_t _Size,
	    size_t _Offset = 0,
	    typename _R    = __fixed_size_storage_t<_Tp, _Size>>
  _GLIBCXX_SIMD_INTRINSIC constexpr _R __extract_tuple_with_size() const
  {
    if constexpr (_Size == _S_first_size && _Offset == 0)
      return {first};
    else if constexpr (_Size > _S_first_size && _Offset == 0 && _S_tuple_size > 1)
      return {
	first,
	second.template __extract_tuple_with_size<_Size - _S_first_size>()};
    else if constexpr (_Size == 1)
      return {operator[](_SizeConstant<_Offset>())};
    else if constexpr (_R::_S_tuple_size == 1)
      {
	static_assert(_Offset % _Size == 0);
	static_assert(_S_first_size % _Size == 0);
	return {typename _R::_First_type(
	  __private_init,
	  __extract_part<_Offset / _Size, _S_first_size / _Size>(first))};
      }
    else
      __assert_unreachable<_SizeConstant<_Size>>();
  }

  template <typename _Tup>
  _GLIBCXX_SIMD_INTRINSIC constexpr decltype(auto)
    __extract_argument(_Tup&& __tup) const
  {
    using _TupT = typename __remove_cvref_t<_Tup>::value_type;
    if constexpr (is_same_v<_SimdTuple, __remove_cvref_t<_Tup>>)
      return __tup.first;
    else if (__builtin_is_constant_evaluated())
      return __fixed_size_storage_t<_TupT, _S_first_size>::__generate([&](
	auto __meta) constexpr {
	return __meta.__generator(
	  [&](auto __i) constexpr { return __tup[__i]; },
	  static_cast<_TupT*>(nullptr));
      });
    else
      return [&]() {
	__fixed_size_storage_t<_TupT, _S_first_size> __r;
	__builtin_memcpy(__r.__as_charptr(), __tup.__as_charptr(), sizeof(__r));
	return __r;
      }();
  }

  template <typename _Tup>
  _GLIBCXX_SIMD_INTRINSIC constexpr auto& __skip_argument(_Tup&& __tup) const
  {
    static_assert(_S_tuple_size > 1);
    using _U               = __remove_cvref_t<_Tup>;
    constexpr size_t __off = __offset<_U>;
    if constexpr (_S_first_size == _U::_S_first_size && __off == 0)
      return __tup.second;
    else if constexpr (_S_first_size > _U::_S_first_size &&
		       _S_first_size % _U::_S_first_size == 0 && __off == 0)
      return __simd_tuple_pop_front<_S_first_size / _U::_S_first_size>(__tup);
    else if constexpr (_S_first_size + __off < _U::_S_first_size)
      return __add_offset<_S_first_size>(__tup);
    else if constexpr (_S_first_size + __off == _U::_S_first_size)
      return __tup.second;
    else
      __assert_unreachable<_Tup>();
  }

  template <size_t _Offset, typename... _More>
  _GLIBCXX_SIMD_INTRINSIC constexpr void
    __assign_front(const _SimdTuple<_Tp, _Abi0, _More...>& __x) &
  {
    static_assert(_Offset == 0);
    first = __x.first;
    if constexpr (sizeof...(_More) > 0)
      {
	static_assert(sizeof...(_Abis) >= sizeof...(_More));
	second.template __assign_front<0>(__x.second);
      }
  }

  template <size_t _Offset>
  _GLIBCXX_SIMD_INTRINSIC constexpr void
    __assign_front(const _First_type& __x) &
  {
    static_assert(_Offset == 0);
    first = __x;
  }

  template <size_t _Offset, typename... _As>
  _GLIBCXX_SIMD_INTRINSIC constexpr void
    __assign_front(const _SimdTuple<_Tp, _As...>& __x) &
  {
    __builtin_memcpy(__as_charptr() + _Offset * sizeof(value_type),
		     __x.__as_charptr(),
		     sizeof(_Tp) * _SimdTuple<_Tp, _As...>::size());
  }

  /*
   * Iterate over the first objects in this _SimdTuple and call __fun for each
   * of them. If additional arguments are passed via __more, chunk them into
   * _SimdTuple or __vector_type_t objects of the same number of values.
   */
  template <class _F, class... _More>
  _GLIBCXX_SIMD_INTRINSIC constexpr _SimdTuple
    __apply_per_chunk(_F&& __fun, _More&&... __more) const
  {
    _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
    //_GLIBCXX_SIMD_DEBUG_DEFERRED("__more = ", __more...);
    static_assert(
      (... && ((_S_first_size % __remove_cvref_t<_More>::_S_first_size == 0) ||
	       (__remove_cvref_t<_More>::_S_first_size % _S_first_size == 0))));
    if constexpr ((... || conjunction_v<
			    is_lvalue_reference<_More>,
			    negation<is_const<remove_reference_t<_More>>>>))
      {
	// need to write back at least one of __more after calling __fun
	auto&& __first = [&](auto... __args) constexpr {
	  auto __r =
	    __fun(__tuple_element_meta<_Tp, _Abi0, 0>(), first, __args...);
	  [[maybe_unused]] auto&& __unused = {(
	    [](auto&& __dst, const auto& __src) {
	      if constexpr (is_assignable_v<decltype(__dst), decltype(__dst)>)
		{
		  __dst.template __assign_front<__offset<decltype(__dst)>>(
		    __src);
		}
	    }(std::forward<_More>(__more), __args),
	    0)...};
	  return __r;
	}(__extract_argument(__more)...);
	if constexpr (_S_tuple_size == 1)
	  return { __first };
	else
	  return {__first,
		  second.__apply_per_chunk(std::forward<_F>(__fun),
					   __skip_argument(__more)...)};
      }
    else
      {
	auto&& __first = __fun(__tuple_element_meta<_Tp, _Abi0, 0>(), first,
			       __extract_argument(__more)...);
	if constexpr (_S_tuple_size == 1)
	  return { __first };
	else
	  return {__first,
		  second.__apply_per_chunk(std::forward<_F>(__fun),
					   __skip_argument(__more)...)};
      }
  }

  template <typename _R = _Tp, typename _F, typename... _More>
  _GLIBCXX_SIMD_INTRINSIC auto
    __apply_r(_F&& __fun, const _More&... __more) const
  {
    _GLIBCXX_SIMD_DEBUG(_SIMD_TUPLE);
    auto&& __first =
      __fun(__tuple_element_meta<_Tp, _Abi0, 0>(), first, __more.first...);
    if constexpr (_S_tuple_size == 1)
      return __first;
    else
      return __simd_tuple_concat<_R>(
	__first, second.template __apply_r<_R>(std::forward<_F>(__fun),
					       __more.second...));
  }

  template <typename _F, typename... _More>
  _GLIBCXX_SIMD_INTRINSIC friend std::bitset<size()>
    __test(_F&& __fun, const _SimdTuple& __x, const _More&... __more)
  {
    auto&& __first = __vector_to_bitset(
      __fun(__tuple_element_meta<_Tp, _Abi0, 0>(), __x.first, __more.first...));
    if constexpr (_S_tuple_size == 1)
      return __first;
    else
      return __first.to_ullong() |
	     (__test(__fun, __x.second, __more.second...).to_ullong()
	      << simd_size_v<_Tp, _Abi0>);
  }

  template <typename _U, _U _I>
  _GLIBCXX_SIMD_INTRINSIC constexpr _Tp
    operator[](std::integral_constant<_U, _I>) const noexcept
  {
    if constexpr (_I < simd_size_v<_Tp, _Abi0>)
	return __subscript_read(first, _I);
    else
	return second[std::integral_constant<_U,
					     _I - simd_size_v<_Tp, _Abi0>>()];
  }

  _Tp operator[](size_t __i) const noexcept
  {
    if constexpr (_S_tuple_size == 1)
      return __subscript_read(first, __i);
    else
      {
#ifdef _GLIBCXX_SIMD_USE_ALIASING_LOADS
	return reinterpret_cast<const __may_alias<_Tp>*>(this)[__i];
#else
	if constexpr (__is_abi<_Abi0, simd_abi::scalar>())
	  {
	    const _Tp* ptr = &first;
	    return ptr[__i];
	  }
	else
	  return __i < simd_size_v<_Tp, _Abi0>
		   ? __subscript_read(first, __i)
		   : second[__i - simd_size_v<_Tp, _Abi0>];
#endif
      }
  }

  void __set(size_t __i, _Tp __val) noexcept
  {
    if constexpr (_S_tuple_size == 1)
      return __subscript_write(first, __i, __val);
    else
      {
#ifdef _GLIBCXX_SIMD_USE_ALIASING_LOADS
	reinterpret_cast<__may_alias<_Tp>*>(this)[__i] = __val;
#else
	if (__i < simd_size_v<_Tp, _Abi0>)
	  __subscript_write(first, __i, __val);
	else
	  second.__set(__i - simd_size_v<_Tp, _Abi0>, __val);
#endif
      }
  }
};

// __make_simd_tuple {{{1
template <typename _Tp, typename _A0>
_GLIBCXX_SIMD_INTRINSIC _SimdTuple<_Tp, _A0> __make_simd_tuple(
    std::experimental::simd<_Tp, _A0> __x0)
{
    return {__data(__x0)};
}
template <typename _Tp, typename _A0, typename... _As>
_GLIBCXX_SIMD_INTRINSIC _SimdTuple<_Tp, _A0, _As...> __make_simd_tuple(
    const std::experimental::simd<_Tp, _A0> &__x0,
    const std::experimental::simd<_Tp, _As> &... __xs)
{
    return {__data(__x0), __make_simd_tuple(__xs...)};
}

template <typename _Tp, typename _A0>
_GLIBCXX_SIMD_INTRINSIC _SimdTuple<_Tp, _A0> __make_simd_tuple(
    const typename _SimdTraits<_Tp, _A0>::_SimdMember &__arg0)
{
    return {__arg0};
}

template <typename _Tp, typename _A0, typename _A1, typename... _Abis>
_GLIBCXX_SIMD_INTRINSIC _SimdTuple<_Tp, _A0, _A1, _Abis...> __make_simd_tuple(
    const typename _SimdTraits<_Tp, _A0>::_SimdMember &__arg0,
    const typename _SimdTraits<_Tp, _A1>::_SimdMember &__arg1,
    const typename _SimdTraits<_Tp, _Abis>::_SimdMember &... __args)
{
    return {__arg0, __make_simd_tuple<_Tp, _A1, _Abis...>(__arg1, __args...)};
}

// __to_simd_tuple {{{1
template <size_t, class _Tp> using __to_tuple_helper = _Tp;
template <class _Tp, class _A0, size_t... _Indexes>
_GLIBCXX_SIMD_INTRINSIC _SimdTuple<_Tp, __to_tuple_helper<_Indexes, _A0>...>
__to_simd_tuple_impl(std::index_sequence<_Indexes...>,
                     const std::array<__vector_type_t<_Tp, simd_size_v<_Tp, _A0>>,
                                      sizeof...(_Indexes)> &__args)
{
    return __make_simd_tuple<_Tp, __to_tuple_helper<_Indexes, _A0>...>(__args[_Indexes]...);
}

template <class _Tp, class _A0, size_t _N>
_GLIBCXX_SIMD_INTRINSIC auto __to_simd_tuple(
    const std::array<__vector_type_t<_Tp, simd_size_v<_Tp, _A0>>, _N> &__args)
{
    return __to_simd_tuple_impl<_Tp, _A0>(std::make_index_sequence<_N>(), __args);
}

// __optimize_simd_tuple {{{1
template <class _Tp> _GLIBCXX_SIMD_INTRINSIC _SimdTuple<_Tp> __optimize_simd_tuple(const _SimdTuple<_Tp>)
{
    return {};
}

template <class _Tp, class _A>
_GLIBCXX_SIMD_INTRINSIC const _SimdTuple<_Tp, _A> &__optimize_simd_tuple(const _SimdTuple<_Tp, _A> &__x)
{
    return __x;
}

template <class _Tp, class _A0, class _A1, class... _Abis,
          class _R = __fixed_size_storage_t<_Tp, _SimdTuple<_Tp, _A0, _A1, _Abis...>::size()>>
_GLIBCXX_SIMD_INTRINSIC _R __optimize_simd_tuple(const _SimdTuple<_Tp, _A0, _A1, _Abis...> &__x)
{
    using _Tup = _SimdTuple<_Tp, _A0, _A1, _Abis...>;
    if constexpr (std::is_same_v<_R, _Tup>)
      {
	return __x;
      }
    else if constexpr (_R::_S_first_size == simd_size_v<_Tp, _A0>)
      {
	return __simd_tuple_concat(_SimdTuple<_Tp, typename _R::_First_abi>{__x.first},
                            __optimize_simd_tuple(__x.second));
      }
    else if constexpr (_R::_S_first_size ==
		       simd_size_v<_Tp, _A0> + simd_size_v<_Tp, _A1>)
      {
	return __simd_tuple_concat(_SimdTuple<_Tp, typename _R::_First_abi>{__data(
                                std::experimental::concat(__get_simd_at<0>(__x), __get_simd_at<1>(__x)))},
                            __optimize_simd_tuple(__x.second.second));
      }
    else if constexpr (_R::_S_first_size ==
		       4 * __simd_tuple_element_t<0, _Tup>::size())
      {
	return __simd_tuple_concat(
	  _SimdTuple<_Tp, typename _R::_First_abi>{
	    __data(concat(__get_simd_at<0>(__x), __get_simd_at<1>(__x),
			  __get_simd_at<2>(__x), __get_simd_at<3>(__x)))},
	  __optimize_simd_tuple(__x.second.second.second.second));
      }
    else if constexpr (_R::_S_first_size ==
		       8 * __simd_tuple_element_t<0, _Tup>::size())
      {
	return __simd_tuple_concat(
	  _SimdTuple<_Tp, typename _R::_First_abi>{__data(concat(
	    __get_simd_at<0>(__x), __get_simd_at<1>(__x), __get_simd_at<2>(__x),
	    __get_simd_at<3>(__x), __get_simd_at<4>(__x), __get_simd_at<5>(__x),
	    __get_simd_at<6>(__x), __get_simd_at<7>(__x)))},
	  __optimize_simd_tuple(
	    __x.second.second.second.second.second.second.second.second));
      }
    else if constexpr (_R::_S_first_size ==
		       16 * __simd_tuple_element_t<0, _Tup>::size())
      {
	return __simd_tuple_concat(
	  _SimdTuple<_Tp, typename _R::_First_abi>{__data(concat(
	    __get_simd_at<0>(__x), __get_simd_at<1>(__x), __get_simd_at<2>(__x),
	    __get_simd_at<3>(__x), __get_simd_at<4>(__x), __get_simd_at<5>(__x),
	    __get_simd_at<6>(__x), __get_simd_at<7>(__x), __get_simd_at<8>(__x),
	    __get_simd_at<9>(__x), __get_simd_at<10>(__x),
	    __get_simd_at<11>(__x), __get_simd_at<12>(__x),
	    __get_simd_at<13>(__x), __get_simd_at<14>(__x),
	    __get_simd_at<15>(__x)))},
	  __optimize_simd_tuple(
	    __x.second.second.second.second.second.second.second.second.second
	      .second.second.second.second.second.second.second));
      }
    else
      {
	return __x;
      }
}

// __for_each(const _SimdTuple &, Fun) {{{1
template <size_t _Offset = 0, class _Tp, class _A0, class _F>
_GLIBCXX_SIMD_INTRINSIC constexpr void
  __for_each(const _SimdTuple<_Tp, _A0>& __t, _F&& __fun)
{
  std::forward<_F>(__fun)(__make_meta<_Offset>(__t), __t.first);
}
template <size_t _Offset = 0,
	  class _Tp,
	  class _A0,
	  class _A1,
	  class... _As,
	  class _F>
_GLIBCXX_SIMD_INTRINSIC constexpr void
  __for_each(const _SimdTuple<_Tp, _A0, _A1, _As...>& __t, _F&& __fun)
{
  __fun(__make_meta<_Offset>(__t), __t.first);
  __for_each<_Offset + simd_size<_Tp, _A0>::value>(__t.second,
						   std::forward<_F>(__fun));
}

// __for_each(_SimdTuple &, Fun) {{{1
template <size_t _Offset = 0, class _Tp, class _A0, class _F>
_GLIBCXX_SIMD_INTRINSIC constexpr void
  __for_each(_SimdTuple<_Tp, _A0>& __t, _F&& __fun)
{
  std::forward<_F>(__fun)(__make_meta<_Offset>(__t), __t.first);
}
template <size_t _Offset = 0,
	  class _Tp,
	  class _A0,
	  class _A1,
	  class... _As,
	  class _F>
_GLIBCXX_SIMD_INTRINSIC constexpr void
  __for_each(_SimdTuple<_Tp, _A0, _A1, _As...>& __t, _F&& __fun)
{
  __fun(__make_meta<_Offset>(__t), __t.first);
  __for_each<_Offset + simd_size<_Tp, _A0>::value>(__t.second,
						   std::forward<_F>(__fun));
}

// __for_each(_SimdTuple &, const _SimdTuple &, Fun) {{{1
template <size_t _Offset = 0, class _Tp, class _A0, class _F>
_GLIBCXX_SIMD_INTRINSIC constexpr void
  __for_each(_SimdTuple<_Tp, _A0>&       __a,
	     const _SimdTuple<_Tp, _A0>& __b,
	     _F&&                          __fun)
{
  std::forward<_F>(__fun)(__make_meta<_Offset>(__a), __a.first, __b.first);
}
template <size_t _Offset = 0,
	  class _Tp,
	  class _A0,
	  class _A1,
	  class... _As,
	  class _F>
_GLIBCXX_SIMD_INTRINSIC constexpr void
  __for_each(_SimdTuple<_Tp, _A0, _A1, _As...>&       __a,
	     const _SimdTuple<_Tp, _A0, _A1, _As...>& __b,
	     _F&&                                      __fun)
{
  __fun(__make_meta<_Offset>(__a), __a.first, __b.first);
  __for_each<_Offset + simd_size<_Tp, _A0>::value>(__a.second, __b.second,
						   std::forward<_F>(__fun));
}

// __for_each(const _SimdTuple &, const _SimdTuple &, Fun) {{{1
template <size_t _Offset = 0, class _Tp, class _A0, class _F>
_GLIBCXX_SIMD_INTRINSIC constexpr void
  __for_each(const _SimdTuple<_Tp, _A0>& __a,
	     const _SimdTuple<_Tp, _A0>& __b,
	     _F&&                          __fun)
{
  std::forward<_F>(__fun)(__make_meta<_Offset>(__a), __a.first, __b.first);
}
template <size_t _Offset = 0,
	  class _Tp,
	  class _A0,
	  class _A1,
	  class... _As,
	  class _F>
_GLIBCXX_SIMD_INTRINSIC constexpr void
  __for_each(const _SimdTuple<_Tp, _A0, _A1, _As...>& __a,
	     const _SimdTuple<_Tp, _A0, _A1, _As...>& __b,
	     _F&&                                      __fun)
{
  __fun(__make_meta<_Offset>(__a), __a.first, __b.first);
  __for_each<_Offset + simd_size<_Tp, _A0>::value>(__a.second, __b.second,
						   std::forward<_F>(__fun));
}

// }}}1
// __cmpord{{{
template <class _Tp, class _TVT = _VectorTraits<_Tp>>
_GLIBCXX_SIMD_INTRINSIC auto __cmpord(_Tp __x, _Tp __y)
{
  static_assert(is_floating_point_v<typename _TVT::value_type>);
#if _GLIBCXX_SIMD_X86INTRIN // {{{
  if constexpr (__have_sse && _TVT::template __is<float, 4>)
    return __intrin_bitcast<_Tp>(_mm_cmpord_ps(__x, __y));
  else if constexpr (__have_sse2 && _TVT::template __is<double, 2>)
    return __intrin_bitcast<_Tp>(_mm_cmpord_pd(__x, __y));
  else if constexpr (__have_avx && _TVT::template __is<float, 8>)
    return __intrin_bitcast<_Tp>(_mm256_cmp_ps(__x, __y, _CMP_ORD_Q));
  else if constexpr (__have_avx && _TVT::template __is<double, 4>)
    return __intrin_bitcast<_Tp>(_mm256_cmp_pd(__x, __y, _CMP_ORD_Q));
  else if constexpr (__have_avx512f && _TVT::template __is<float, 16>)
    return _mm512_cmp_ps_mask(__x, __y, _CMP_ORD_Q);
  else if constexpr (__have_avx512f && _TVT::template __is<double, 8>)
    return _mm512_cmp_pd_mask(__x, __y, _CMP_ORD_Q);
  else
#endif // _GLIBCXX_SIMD_X86INTRIN }}}
    {
      return reinterpret_cast<_Tp>((__x < __y) != (__x >= __y));
    }
}

// }}}
// __cmpunord{{{
template <class _Tp, class _TVT = _VectorTraits<_Tp>>
_GLIBCXX_SIMD_INTRINSIC auto __cmpunord(_Tp __x, _Tp __y)
{
  static_assert(is_floating_point_v<typename _TVT::value_type>);
#if _GLIBCXX_SIMD_X86INTRIN // {{{
  if constexpr (__have_sse && _TVT::template __is<float, 4>)
    return __intrin_bitcast<_Tp>(_mm_cmpunord_ps(__x, __y));
  else if constexpr (__have_sse2 && _TVT::template __is<double, 2>)
    return __intrin_bitcast<_Tp>(_mm_cmpunord_pd(__x, __y));
  else if constexpr (__have_avx && _TVT::template __is<float, 8>)
    return __intrin_bitcast<_Tp>(_mm256_cmp_ps(__x, __y, _CMP_UNORD_Q));
  else if constexpr (__have_avx && _TVT::template __is<double, 4>)
    return __intrin_bitcast<_Tp>(_mm256_cmp_pd(__x, __y, _CMP_UNORD_Q));
  else if constexpr (__have_avx512f && _TVT::template __is<float, 16>)
    return _mm512_cmp_ps_mask(__x, __y, _CMP_UNORD_Q);
  else if constexpr (__have_avx512f && _TVT::template __is<double, 8>)
    return _mm512_cmp_pd_mask(__x, __y, _CMP_UNORD_Q);
  else
#endif // _GLIBCXX_SIMD_X86INTRIN }}}
    {
      return reinterpret_cast<_Tp>((__x < __y) == (__x >= __y));
    }
}

// }}}
// __maskstore (non-converting; with optimizations for SSE2-AVX512BWVL) {{{
#if _GLIBCXX_SIMD_X86INTRIN // {{{
template <class _Tp, class _F>
_GLIBCXX_SIMD_INTRINSIC void __maskstore(_SimdWrapper64<_Tp> __v, _Tp* __mem, _F,
                                         _SimdWrapper<bool, _SimdWrapper64<_Tp>::_S_width> __k)
{
    [[maybe_unused]] const auto __vi = __to_intrin(__v);
    static_assert(sizeof(__v) == 64 && __have_avx512f);
    if constexpr (__have_avx512bw && sizeof(_Tp) == 1) {
        _mm512_mask_storeu_epi8(__mem, __k, __vi);
    } else if constexpr (__have_avx512bw && sizeof(_Tp) == 2) {
        _mm512_mask_storeu_epi16(__mem, __k, __vi);
    } else if constexpr (__have_avx512f && sizeof(_Tp) == 4) {
        if constexpr (__is_aligned_v<_F, 64> && std::is_integral_v<_Tp>) {
            _mm512_mask_store_epi32(__mem, __k, __vi);
        } else if constexpr (__is_aligned_v<_F, 64> && std::is_floating_point_v<_Tp>) {
            _mm512_mask_store_ps(__mem, __k, __vi);
        } else if constexpr (std::is_integral_v<_Tp>) {
            _mm512_mask_storeu_epi32(__mem, __k, __vi);
        } else {
            _mm512_mask_storeu_ps(__mem, __k, __vi);
        }
    } else if constexpr (__have_avx512f && sizeof(_Tp) == 8) {
        if constexpr (__is_aligned_v<_F, 64> && std::is_integral_v<_Tp>) {
            _mm512_mask_store_epi64(__mem, __k, __vi);
        } else if constexpr (__is_aligned_v<_F, 64> && std::is_floating_point_v<_Tp>) {
            _mm512_mask_store_pd(__mem, __k, __vi);
        } else if constexpr (std::is_integral_v<_Tp>) {
            _mm512_mask_storeu_epi64(__mem, __k, __vi);
        } else {
            _mm512_mask_storeu_pd(__mem, __k, __vi);
        }
    } else if constexpr (__have_sse2) {
        constexpr int _N = 16 / sizeof(_Tp);
        using _M          = __vector_type_t<_Tp, _N>;
        _mm_maskmoveu_si128(__auto_bitcast(__extract<0, 4>(__v._M_data)),
                            __auto_bitcast(__convert_mask<_M>(__k._M_data)),
                            reinterpret_cast<char*>(__mem));
        _mm_maskmoveu_si128(__auto_bitcast(__extract<1, 4>(__v._M_data)),
                            __auto_bitcast(__convert_mask<_M>(__k._M_data >> 1 * _N)),
                            reinterpret_cast<char*>(__mem) + 1 * 16);
        _mm_maskmoveu_si128(__auto_bitcast(__extract<2, 4>(__v._M_data)),
                            __auto_bitcast(__convert_mask<_M>(__k._M_data >> 2 * _N)),
                            reinterpret_cast<char*>(__mem) + 2 * 16);
        _mm_maskmoveu_si128(__auto_bitcast(__extract<3, 4>(__v._M_data)),
                            __auto_bitcast(__convert_mask<_M>(__k._M_data >> 3 * _N)),
                            reinterpret_cast<char*>(__mem) + 3 * 16);
    } else {
        __assert_unreachable<_Tp>();
    }
}

template <class _Tp, class _F>
_GLIBCXX_SIMD_INTRINSIC void __maskstore(_SimdWrapper32<_Tp> __v, _Tp* __mem, _F,
                                         _SimdWrapper32<_Tp> __k)
{
    [[maybe_unused]] const auto __vi = __vector_bitcast<_LLong>(__v);
    [[maybe_unused]] const auto __ki = __vector_bitcast<_LLong>(__k);
    if constexpr (__have_avx512bw_vl && sizeof(_Tp) == 1) {
        _mm256_mask_storeu_epi8(__mem, _mm256_movepi8_mask(__ki), __vi);
    } else if constexpr (__have_avx512bw_vl && sizeof(_Tp) == 2) {
        _mm256_mask_storeu_epi16(__mem, _mm256_movepi16_mask(__ki), __vi);
    } else if constexpr (__have_avx2 && sizeof(_Tp) == 4 && std::is_integral_v<_Tp>) {
        _mm256_maskstore_epi32(reinterpret_cast<int*>(__mem), __ki, __vi);
    } else if constexpr (sizeof(_Tp) == 4) {
        _mm256_maskstore_ps(reinterpret_cast<float*>(__mem), __ki,
                            __vector_bitcast<float>(__v));
    } else if constexpr (__have_avx2 && sizeof(_Tp) == 8 && std::is_integral_v<_Tp>) {
        _mm256_maskstore_epi64(reinterpret_cast<_LLong*>(__mem), __ki, __vi);
    } else if constexpr (__have_avx && sizeof(_Tp) == 8) {
        _mm256_maskstore_pd(reinterpret_cast<double*>(__mem), __ki,
                            __vector_bitcast<double>(__v));
    } else if constexpr (__have_sse2) {
        _mm_maskmoveu_si128(__lo128(__vi), __lo128(__ki), reinterpret_cast<char*>(__mem));
        _mm_maskmoveu_si128(__hi128(__vi), __hi128(__ki),
                            reinterpret_cast<char*>(__mem) + 16);
    } else {
        __assert_unreachable<_Tp>();
    }
}

template <class _Tp, class _F>
_GLIBCXX_SIMD_INTRINSIC void __maskstore(_SimdWrapper32<_Tp> __v, _Tp* __mem, _F,
                                         _SimdWrapper<bool, _SimdWrapper32<_Tp>::_S_width> __k)
{
    [[maybe_unused]] const auto __vi = __to_intrin(__v);
    if constexpr (__have_avx512bw_vl && sizeof(_Tp) == 1) {
        _mm256_mask_storeu_epi8(__mem, __k, __vi);
    } else if constexpr (__have_avx512bw_vl && sizeof(_Tp) == 2) {
        _mm256_mask_storeu_epi16(__mem, __k, __vi);
    } else if constexpr (__have_avx512vl && sizeof(_Tp) == 4) {
        if constexpr (__is_aligned_v<_F, 32> && std::is_integral_v<_Tp>) {
            _mm256_mask_store_epi32(__mem, __k, __vi);
        } else if constexpr (__is_aligned_v<_F, 32> && std::is_floating_point_v<_Tp>) {
            _mm256_mask_store_ps(__mem, __k, __vi);
        } else if constexpr (std::is_integral_v<_Tp>) {
            _mm256_mask_storeu_epi32(__mem, __k, __vi);
        } else {
            _mm256_mask_storeu_ps(__mem, __k, __vi);
        }
    } else if constexpr (__have_avx512vl && sizeof(_Tp) == 8) {
        if constexpr (__is_aligned_v<_F, 32> && std::is_integral_v<_Tp>) {
            _mm256_mask_store_epi64(__mem, __k, __vi);
        } else if constexpr (__is_aligned_v<_F, 32> && std::is_floating_point_v<_Tp>) {
            _mm256_mask_store_pd(__mem, __k, __vi);
        } else if constexpr (std::is_integral_v<_Tp>) {
            _mm256_mask_storeu_epi64(__mem, __k, __vi);
        } else {
            _mm256_mask_storeu_pd(__mem, __k, __vi);
        }
    } else if constexpr (__have_avx512f && (sizeof(_Tp) >= 4 || __have_avx512bw)) {
        // use a 512-bit maskstore, using zero-extension of the bitmask
        __maskstore(
            _SimdWrapper64<_Tp>(__intrin_bitcast<__vector_type64_t<_Tp>>(__v._M_data)),
            __mem,
            // careful, vector_aligned has a stricter meaning in the 512-bit maskstore:
            std::conditional_t<std::is_same_v<_F, vector_aligned_tag>,
                               overaligned_tag<32>, _F>(),
            _SimdWrapper<bool, 64 / sizeof(_Tp)>(__k._M_data));
    } else {
        __maskstore(
            __v, __mem, _F(),
            _SimdWrapper32<_Tp>(__convert_mask<__vector_type_t<_Tp, 32 / sizeof(_Tp)>>(__k)));
    }
}
#endif // _GLIBCXX_SIMD_X86INTRIN }}}

template <class _Tp, class _F>
_GLIBCXX_SIMD_INTRINSIC void __maskstore(_SimdWrapper16<_Tp> __v, _Tp* __mem, _F,
                                         _SimdWrapper16<_Tp> __k)
{
    [[maybe_unused]] const auto __vi = __vector_bitcast<_LLong>(__v);
    [[maybe_unused]] const auto __ki = __vector_bitcast<_LLong>(__k);
    if constexpr (__have_avx512bw_vl && sizeof(_Tp) == 1) {
        _mm_mask_storeu_epi8(__mem, _mm_movepi8_mask(__ki), __vi);
    } else if constexpr (__have_avx512bw_vl && sizeof(_Tp) == 2) {
        _mm_mask_storeu_epi16(__mem, _mm_movepi16_mask(__ki), __vi);
    } else if constexpr (__have_avx2 && sizeof(_Tp) == 4 && std::is_integral_v<_Tp>) {
        _mm_maskstore_epi32(reinterpret_cast<int*>(__mem), __ki, __vi);
    } else if constexpr (__have_avx && sizeof(_Tp) == 4) {
        _mm_maskstore_ps(reinterpret_cast<float*>(__mem), __ki,
                         __vector_bitcast<float>(__v));
    } else if constexpr (__have_avx2 && sizeof(_Tp) == 8 && std::is_integral_v<_Tp>) {
        _mm_maskstore_epi64(reinterpret_cast<_LLong*>(__mem), __ki, __vi);
    } else if constexpr (__have_avx && sizeof(_Tp) == 8) {
        _mm_maskstore_pd(reinterpret_cast<double*>(__mem), __ki,
                         __vector_bitcast<double>(__v));
    } else if constexpr (__have_sse2) {
        _mm_maskmoveu_si128(__vi, __ki, reinterpret_cast<char*>(__mem));
    } else {
        __bit_iteration(__vector_to_bitset(__k._M_data).to_ullong(),
                        [&](auto __i) constexpr { __mem[__i] = __v[__i]; });
    }
}

template <class _Tp, class _F>
_GLIBCXX_SIMD_INTRINSIC void __maskstore(_SimdWrapper16<_Tp> __v, _Tp* __mem, _F,
                                         _SimdWrapper<bool, _SimdWrapper16<_Tp>::_S_width> __k)
{
    [[maybe_unused]] const auto __vi = __to_intrin(__v);
    if constexpr (__have_avx512bw_vl && sizeof(_Tp) == 1) {
        _mm_mask_storeu_epi8(__mem, __k, __vi);
    } else if constexpr (__have_avx512bw_vl && sizeof(_Tp) == 2) {
        _mm_mask_storeu_epi16(__mem, __k, __vi);
    } else if constexpr (__have_avx512vl && sizeof(_Tp) == 4) {
        if constexpr (__is_aligned_v<_F, 16> && std::is_integral_v<_Tp>) {
            _mm_mask_store_epi32(__mem, __k, __vi);
        } else if constexpr (__is_aligned_v<_F, 16> && std::is_floating_point_v<_Tp>) {
            _mm_mask_store_ps(__mem, __k, __vi);
        } else if constexpr (std::is_integral_v<_Tp>) {
            _mm_mask_storeu_epi32(__mem, __k, __vi);
        } else {
            _mm_mask_storeu_ps(__mem, __k, __vi);
        }
    } else if constexpr (__have_avx512vl && sizeof(_Tp) == 8) {
        if constexpr (__is_aligned_v<_F, 16> && std::is_integral_v<_Tp>) {
            _mm_mask_store_epi64(__mem, __k, __vi);
        } else if constexpr (__is_aligned_v<_F, 16> && std::is_floating_point_v<_Tp>) {
            _mm_mask_store_pd(__mem, __k, __vi);
        } else if constexpr (std::is_integral_v<_Tp>) {
            _mm_mask_storeu_epi64(__mem, __k, __vi);
        } else {
            _mm_mask_storeu_pd(__mem, __k, __vi);
        }
    } else if constexpr (__have_avx512f && (sizeof(_Tp) >= 4 || __have_avx512bw)) {
        // use a 512-bit maskstore, using zero-extension of the bitmask
        __maskstore(
            _SimdWrapper64<_Tp>(__intrin_bitcast<__intrinsic_type64_t<_Tp>>(__v._M_data)), __mem,
            // careful, vector_aligned has a stricter meaning in the 512-bit maskstore:
            std::conditional_t<std::is_same_v<_F, vector_aligned_tag>,
                               overaligned_tag<16>, _F>(),
            _SimdWrapper<bool, 64 / sizeof(_Tp)>(__k._M_data));
    } else {
        __maskstore(
            __v, __mem, _F(),
            _SimdWrapper16<_Tp>(__convert_mask<__vector_type_t<_Tp, 16 / sizeof(_Tp)>>(__k)));
    }
}

template <typename _Tp, typename _F, typename _TVT = _VectorTraits<_Tp>>
_GLIBCXX_SIMD_INTRINSIC enable_if_t<sizeof(_Tp) == 8, void>
			__maskstore(_Tp __v, typename _TVT::value_type* __mem, _F, _Tp __k)
{
  __bit_iteration(__vector_to_bitset(__k).to_ulong(),
		  [&](auto __i) constexpr { __mem[__i] = __v[__i]; });
}

// }}}
// __xzyw{{{
// shuffles the complete vector, swapping the inner two quarters. Often useful for AVX for
// fixing up a shuffle result.
template <class _Tp, class _TVT = _VectorTraits<_Tp>>
_GLIBCXX_SIMD_INTRINSIC _Tp __xzyw(_Tp __a)
{
    if constexpr (sizeof(_Tp) == 16) {
        static_assert(sizeof(float) == 4 && sizeof(int) == 4);
        const auto __x = __vector_bitcast<
            conditional_t<is_floating_point_v<typename _TVT::value_type>, float, int>>(
            __a);
        return reinterpret_cast<_Tp>(decltype(__x){__x[0], __x[2], __x[1], __x[3]});
    } else if constexpr (sizeof(_Tp) == 32) {
        static_assert(sizeof(double) == 8 && sizeof(_LLong) == 8);
        const auto __x =
            __vector_bitcast<conditional_t<is_floating_point_v<typename _TVT::value_type>,
                                           double, _LLong>>(__a);
        return reinterpret_cast<_Tp>(decltype(__x){__x[0], __x[2], __x[1], __x[3]});
    } else if constexpr (sizeof(_Tp) == 64) {
        static_assert(sizeof(double) == 8 && sizeof(_LLong) == 8);
        const auto __x =
            __vector_bitcast<conditional_t<is_floating_point_v<typename _TVT::value_type>,
                                           double, _LLong>>(__a);
        return reinterpret_cast<_Tp>(decltype(__x){__x[0], __x[1], __x[4], __x[5], __x[2],
                                                  __x[3], __x[6], __x[7]});
    } else {
        __assert_unreachable<_Tp>();
    }
}

// }}}
// __shift_elements_right{{{
// if (__shift % 2ⁿ == 0) => the low n Bytes are correct
template <unsigned __shift, class _Tp, class _TVT = _VectorTraits<_Tp>>
_GLIBCXX_SIMD_INTRINSIC _Tp __shift_elements_right(_Tp __v)
{
    static_assert(__shift <= sizeof(_Tp));
    if constexpr (__shift == 0) {
        return __v;
    } else if constexpr(__shift == sizeof(_Tp)) {
        return _Tp();
#if _GLIBCXX_SIMD_X86INTRIN // {{{
    } else if constexpr (__have_sse && __shift == 8 && _TVT::template __is<float, 4>) {
        return _mm_movehl_ps(__v, __v);
    } else if constexpr (__have_sse2 && __shift == 8 && _TVT::template __is<double, 2>) {
        return _mm_unpackhi_pd(__v, __v);
    } else if constexpr (__have_sse2 && sizeof(_Tp) == 16) {
        return __intrin_bitcast<_Tp>(
            _mm_srli_si128(__intrin_bitcast<__m128i>(__v), __shift));
/*
    } else if constexpr (__shift == 16 && sizeof(_Tp) == 32) {
        if constexpr (__have_avx && _TVT::template __is<double, 4>) {
            return _mm256_permute2f128_pd(__v, __v, 0x81);
        } else if constexpr (__have_avx && _TVT::template __is<float, 8>) {
            return _mm256_permute2f128_ps(__v, __v, 0x81);
        } else if constexpr (__have_avx) {
            return _mm256_permute2f128_si256(__v, __v, 0x81);
        } else {
            return __auto_bitcast(__hi128(__v));
        }
    } else if constexpr (__have_avx2 && sizeof(_Tp) == 32) {
        const auto __vi = __intrin_bitcast<__m256i>(__v);
        return __intrin_bitcast<_Tp>(_mm256_srli_si256(
            __shift < 16 ? __vi : _mm256_permute2x128_si256(__vi, __vi, 0x81),
            __shift % 16));
    } else if constexpr (sizeof(_Tp) == 32) {
        __shift % 16
        return __intrin_bitcast<_Tp>(
        __extract<_shift/16, 2>(__v)
        );
    } else if constexpr (__have512f && sizeof(_Tp) == 64) {
        if constexpr (__shift % 8 == 0) {
            return __mm512_alignr_epi64(__m512i(), __intrin_bitcast<__m512i>(__v),
                                        __shift / 8);
        } else if constexpr (__shift % 4 == 0) {
            return __mm512_alignr_epi32(__m512i(), __intrin_bitcast<__m512i>(__v),
                                        __shift / 4);
        } else {
            const auto __shifted = __mm512_alignr_epi8(
                __m512i(), __intrin_bitcast<__m512i>(__v), __shift % 16);
            return __intrin_bitcast<_Tp>(
                __shift < 16
                    ? __shifted
                    : _mm512_shuffle_i32x4(__shifted, __shifted, 0xe4 + (__shift / 16)));
        }
    } else if constexpr (__shift == 32 && sizeof(_Tp) == 64) {
        return __auto_bitcast(__hi256(__v));
    } else if constexpr (__shift % 16 == 0 && sizeof(_Tp) == 64) {
        return __auto_bitcast(__extract<__shift / 16, 4>(__v));
*/
#endif // _GLIBCXX_SIMD_X86INTRIN }}}
    }
  else
    {
      constexpr int __chunksize =
	__shift % 8 == 0 ? 8 : __shift % 4 == 0 ? 4 : __shift % 2 == 0 ? 2 : 1;
      auto __w = __vector_bitcast<__int_with_sizeof_t<__chunksize>>(__v);
      return __intrin_bitcast<_Tp>(decltype(__w){__w[__shift / __chunksize]});
    }
}

// }}}
// __extract_part(_Tp) {{{
template <size_t _Index, size_t _Total, class _Tp, typename _TVT>
_GLIBCXX_SIMD_INTRINSIC _GLIBCXX_SIMD_CONST
  typename __vector_type<typename _TVT::value_type,
			 std::max(__min_vector_size,
				  int(sizeof(_Tp) / _Total))>::type
  __extract_part(_Tp __x)
{
  constexpr size_t _N    = _TVT::_S_width;
  constexpr size_t _NewN = _N / _Total;
  static_assert(_Total > _Index, "_Total must be greater than _Index");
  static_assert(_NewN * _Total == _N, "_N must be divisible by _Total");
  if constexpr (_Index == 0 && _Total == 1)
    return __x;
  else if constexpr (sizeof(typename _TVT::value_type) * _NewN >=
		     __min_vector_size)
    return __extract<_Index, _Total>(__x);
  else
    {
      constexpr int __split = sizeof(__x) / __min_vector_size;
      constexpr int __shift =
	(sizeof(__x) / _Total * _Index) % __min_vector_size;
      return __shift_elements_right<__shift>(
	__extract_part<_Index * __split / _Total, __split>(__x));
    }
}

// }}}
// __extract_part(_SimdWrapper<bool, _N>) {{{
template <size_t _Index, size_t _Total, size_t _N>
_GLIBCXX_SIMD_INTRINSIC constexpr __bool_storage_member_type_t<_N / _Total>
__extract_part(_SimdWrapper<bool, _N> __x)
{
    static_assert(__have_avx512f && _N == _N);
    static_assert(_Total >= 2 && _Index < _Total && _Index >= 0);
    return __x._M_data >> (_Index * _N / _Total);
}

// }}}
// __extract_part(_SimdTuple) {{{
template <int _Index, int _Parts, class _Tp, class _A0, class... _As>
_GLIBCXX_SIMD_INTRINSIC auto  // __vector_type_t or _SimdTuple
__extract_part(const _SimdTuple<_Tp, _A0, _As...> &__x)
{
    // worst cases:
    // (a) 4, 4, 4 => 3, 3, 3, 3 (_Parts = 4)
    // (b) 2, 2, 2 => 3, 3       (_Parts = 2)
    // (c) 4, 2 => 2, 2, 2       (_Parts = 3)
    using _Tuple = _SimdTuple<_Tp, _A0, _As...>;
    static_assert(_Index < _Parts && _Index >= 0 && _Parts >= 1);
    constexpr size_t _N = _Tuple::size();
    static_assert(_N >= _Parts && _N % _Parts == 0);
    constexpr size_t __values_per_part = _N / _Parts;
    if constexpr (_Parts == 1) {
        if constexpr (_Tuple::_S_tuple_size == 1) {
            return __x.first._M_data;
        } else {
            return __x;
        }
    } else if constexpr (simd_size_v<_Tp, _A0> % __values_per_part != 0) {
        // nasty case: The requested partition does not match the partition of the
        // _SimdTuple. Fall back to construction via scalar copies.
#ifdef _GLIBCXX_SIMD_USE_ALIASING_LOADS
        const __may_alias<_Tp> *const element_ptr =
            reinterpret_cast<const __may_alias<_Tp> *>(&__x) + _Index * __values_per_part;
        return __data(simd<_Tp, simd_abi::deduce_t<_Tp, __values_per_part>>(
                          [&](auto __i) constexpr { return element_ptr[__i]; }))
            ._M_data;
#else
        [[maybe_unused]] constexpr size_t __offset = _Index * __values_per_part;
        return __data(simd<_Tp, simd_abi::deduce_t<_Tp, __values_per_part>>([&](auto __i) constexpr {
                   constexpr _SizeConstant<__i + __offset> __k;
                   return __x[__k];
               }))
            ._M_data;
#endif
    } else if constexpr (__values_per_part * _Index >= simd_size_v<_Tp, _A0>) {  // recurse
        constexpr int __parts_in_first = simd_size_v<_Tp, _A0> / __values_per_part;
        return __extract_part<_Index - __parts_in_first, _Parts - __parts_in_first>(__x.second);
    } else {  // at this point we know that all of the return values are in __x.first
        static_assert(__values_per_part * (1 + _Index) <= simd_size_v<_Tp, _A0>);
        if constexpr (simd_size_v<_Tp, _A0> == __values_per_part) {
            return __x.first._M_data;
        } else {
            return __extract_part<_Index, simd_size_v<_Tp, _A0> / __values_per_part>(
                __x.first);
        }
    }
}
// }}}
// _ToWrapper specializations for bitset and __mmask<_N> {{{
#if _GLIBCXX_SIMD_HAVE_AVX512_ABI
template <size_t _N> class _ToWrapper<std::bitset<_N>>
{
    std::bitset<_N> _M_data;

public:
    // can convert to larger storage for _Abi::_S_is_partial == true
    template <class _U, size_t _M> constexpr operator _SimdWrapper<_U, _M>() const
    {
        static_assert(_M >= _N);
        return __convert_mask<_SimdWrapper<_U, _M>>(_M_data);
    }
};

#define _GLIBCXX_SIMD_TO_STORAGE(_Type)                                                  \
    template <> class _ToWrapper<_Type>                                                \
    {                                                                                    \
        _Type _M_data;                                                                         \
                                                                                         \
    public:                                                                              \
        template <class _U, size_t _N> constexpr operator _SimdWrapper<_U, _N>() const      \
        {                                                                                \
            static_assert(_N >= sizeof(_Type) * CHAR_BIT);                               \
            return reinterpret_cast<__vector_type_t<_U, _N>>(                            \
                __convert_mask<_SimdWrapper<_U, _N>>(_M_data));                                   \
        }                                                                                \
                                                                                         \
        template <size_t _N> constexpr operator _SimdWrapper<bool, _N>() const              \
        {                                                                                \
            static_assert(                                                               \
                std::is_same_v<_Type, typename __bool_storage_member_type<_N>::type>);   \
            return _M_data;                                                                    \
        }                                                                                \
    }
_GLIBCXX_SIMD_TO_STORAGE(__mmask8);
_GLIBCXX_SIMD_TO_STORAGE(__mmask16);
_GLIBCXX_SIMD_TO_STORAGE(__mmask32);
_GLIBCXX_SIMD_TO_STORAGE(__mmask64);
#undef _GLIBCXX_SIMD_TO_STORAGE
#endif  // _GLIBCXX_SIMD_HAVE_AVX512_ABI

// }}}

#if _GLIBCXX_SIMD_HAVE_SSE && defined _GLIBCXX_SIMD_WORKAROUND_PR85048
#include "simd_x86_conversions.h"
#endif  // SSE && _GLIBCXX_SIMD_WORKAROUND_PR85048

// __convert function{{{
template <class _To, class _From, class... _More>
_GLIBCXX_SIMD_INTRINSIC auto __convert(_From __v0, _More... __vs)
{
    static_assert((true && ... && is_same_v<_From, _More>));
    if constexpr (__is_vectorizable_v<_From>) {
        if constexpr (__is_vector_type_v<_To>) {
            return __make_vector(__v0, __vs...);
        } else {
            using _Tp = typename _To::value_type;
            return __make_wrapper<_Tp>(__v0, __vs...);
        }
    } else if constexpr (!__is_vector_type_v<_From>) {
        return __convert<_To>(__v0._M_data, __vs._M_data...);
    } else if constexpr (!__is_vector_type_v<_To>) {
        return _To(__convert<typename _To::_BuiltinType>(__v0, __vs...));
    } else if constexpr (__is_vectorizable_v<_To>) {
        return __convert<__vector_type_t<_To, (_VectorTraits<_From>::_S_width *
                                            (1 + sizeof...(_More)))>>(__v0, __vs...)
            ._M_data;
    } else {
        static_assert(sizeof...(_More) == 0 ||
                          _VectorTraits<_To>::_S_width >=
                              (1 + sizeof...(_More)) * _VectorTraits<_From>::_S_width,
                      "__convert(...) requires the input to fit into the output");
        return __vector_convert<_To>(__v0, __vs...);
    }
}

// }}}
// __convert_all{{{
template <typename _To, typename _From, typename _FromVT = _VectorTraits<_From>>
_GLIBCXX_SIMD_INTRINSIC auto __convert_all(_From __v)
{
    static_assert(__is_vector_type_v<_To>);
    if constexpr (!__is_vector_type_v<_From>) {
        return __convert_all<_To>(__v._M_data);
    } else if constexpr (_FromVT::_S_width > _VectorTraits<_To>::_S_width) {
        constexpr size_t _N = _FromVT::_S_width / _VectorTraits<_To>::_S_width;
        return __generate_from_n_evaluations<_N, std::array<_To, _N>>([&](auto __i) constexpr {
            auto __part = __extract_part<decltype(__i)::value, _N>(__v);
            return __vector_convert<_To>(__part);
        });
    } else {
        return __vector_convert<_To>(__v);
    }
}

// }}}
// __converts_via_decomposition{{{
// This lists all cases where a __vector_convert needs to fall back to conversion of
// individual scalars (i.e. decompose the input vector into scalars, convert, compose
// output vector). In those cases, __masked_load & __masked_store prefer to use the
// __bit_iteration implementation.
template <class _From, class _To, size_t _ToSize> struct __converts_via_decomposition {
private:
    static constexpr bool _S_i_to_i = is_integral_v<_From> && is_integral_v<_To>;
    static constexpr bool _S_f_to_i = is_floating_point_v<_From> && is_integral_v<_To>;
    static constexpr bool _S_f_to_f = is_floating_point_v<_From> && is_floating_point_v<_To>;
    static constexpr bool _S_i_to_f = is_integral_v<_From> && is_floating_point_v<_To>;

    template <size_t _A, size_t _B>
    static constexpr bool _S_sizes = sizeof(_From) == _A && sizeof(_To) == _B;

public:
    static constexpr bool value =
        (_S_i_to_i && _S_sizes<8, 2> && !__have_ssse3 && _ToSize == 16) ||
        (_S_i_to_i && _S_sizes<8, 1> && !__have_avx512f && _ToSize == 16) ||
        (_S_f_to_i && _S_sizes<4, 8> && !__have_avx512dq) ||
        (_S_f_to_i && _S_sizes<8, 8> && !__have_avx512dq) ||
        (_S_f_to_i && _S_sizes<8, 4> && !__have_sse4_1 && _ToSize == 16) ||
        (_S_i_to_f && _S_sizes<8, 4> && !__have_avx512dq && _ToSize == 16) ||
        (_S_i_to_f && _S_sizes<8, 8> && !__have_avx512dq && _ToSize < 64);
};

template <class _From, class _To, size_t _ToSize>
inline constexpr bool __converts_via_decomposition_v =
    __converts_via_decomposition<_From, _To, _ToSize>::value;

// }}}
// __is_bitset {{{
template <class _Tp> struct __is_bitset : false_type {};
template <size_t _N> struct __is_bitset<std::bitset<_N>> : true_type {};
template <class _Tp> inline constexpr bool __is_bitset_v = __is_bitset<_Tp>::value;

// }}}
// __is_storage {{{
template <class _Tp> struct __is_storage : false_type {};
template <class _Tp, size_t _N> struct __is_storage<_SimdWrapper<_Tp, _N>> : true_type {};
template <class _Tp> inline constexpr bool __is_storage_v = __is_storage<_Tp>::value;

// }}}
// __convert_mask{{{
template <class _To, class _From>
inline _To __convert_mask(_From __k)
{
  if constexpr (std::is_same_v<_To, _From>)
    { // also covers bool -> bool
      return __k;
    } else if constexpr (std::is_unsigned_v<_From> && std::is_unsigned_v<_To>) {
        // bits -> bits
        return __k;  // zero-extends or truncates
    } else if constexpr (__is_bitset_v<_From>) {
        // from std::bitset {{{
        static_assert(__k.size() <= sizeof(_ULLong) * CHAR_BIT);
        using _Tp = std::conditional_t<
            (__k.size() <= sizeof(ushort) * CHAR_BIT),
            std::conditional_t<(__k.size() <= CHAR_BIT), _UChar, ushort>,
            std::conditional_t<(__k.size() <= sizeof(_UInt) * CHAR_BIT), _UInt, _ULLong>>;
        return __convert_mask<_To>(static_cast<_Tp>(__k.to_ullong()));
        // }}}
    } else if constexpr (__is_bitset_v<_To>) {
        // to std::bitset {{{
        static_assert(_To().size() <= sizeof(_ULLong) * CHAR_BIT);
        using _Tp = std::conditional_t<
            (_To().size() <= sizeof(ushort) * CHAR_BIT),
            std::conditional_t<(_To().size() <= CHAR_BIT), _UChar, ushort>,
            std::conditional_t<(_To().size() <= sizeof(_UInt) * CHAR_BIT), _UInt, _ULLong>>;
        return __convert_mask<_Tp>(__k);
        // }}}
    } else if constexpr (__is_storage_v<_From>) {
        return __convert_mask<_To>(__k._M_data);
    } else if constexpr (__is_storage_v<_To>) {
        return __convert_mask<typename _To::_BuiltinType>(__k);
    } else if constexpr (std::is_unsigned_v<_From> && __is_vector_type_v<_To>) {
        // bits -> vector {{{
        using _Trait = _VectorTraits<_To>;
        constexpr size_t _N_in = sizeof(_From) * CHAR_BIT;
        using _ToT = typename _Trait::value_type;
        constexpr size_t _N_out = _Trait::_S_width;
        constexpr size_t _N = std::min(_N_in, _N_out);
        constexpr size_t __bytes_per_output_element = sizeof(_ToT);
#if _GLIBCXX_SIMD_X86INTRIN // {{{
        if constexpr (__have_avx512f) {
            if constexpr (__bytes_per_output_element == 1 && sizeof(_To) == 16) {
                if constexpr (__have_avx512bw_vl) {
                    return __vector_bitcast<_ToT>(_mm_movm_epi8(__k));
                } else if constexpr (__have_avx512bw) {
                    return __vector_bitcast<_ToT>(__lo128(_mm512_movm_epi8(__k)));
                } else {
                    auto __as32bits = _mm512_maskz_mov_epi32(__k, ~__m512i());
                    auto __as16bits = __xzyw(
                        _mm256_packs_epi32(__lo256(__as32bits), __hi256(__as32bits)));
                    return __vector_bitcast<_ToT>(
                        _mm_packs_epi16(__lo128(__as16bits), __hi128(__as16bits)));
                }
            } else if constexpr (__bytes_per_output_element == 1 && sizeof(_To) == 32) {
                if constexpr (__have_avx512bw_vl) {
                    return __vector_bitcast<_ToT>(_mm256_movm_epi8(__k));
                } else if constexpr (__have_avx512bw) {
                    return __vector_bitcast<_ToT>(__lo256(_mm512_movm_epi8(__k)));
                } else {
                    auto __as16bits =  // 0 16 1 17 ... 15 31
                        _mm512_srli_epi32(_mm512_maskz_mov_epi32(__k, ~__m512i()), 16) |
                        _mm512_slli_epi32(_mm512_maskz_mov_epi32(__k >> 16, ~__m512i()),
                                          16);
                    auto __0_16_1_17 = __xzyw(_mm256_packs_epi16(
                        __lo256(__as16bits),
                        __hi256(__as16bits))  // 0 16 1 17 2 18 3 19 8 24 9 25 ...
                    );
                    // deinterleave:
                    return __vector_bitcast<_ToT>(__xzyw(_mm256_shuffle_epi8(
                        __0_16_1_17,  // 0 16 1 17 2 ...
                        _mm256_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13,
                                         15, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11,
                                         13, 15))));  // 0-7 16-23 8-15 24-31 -> xzyw
                                                      // 0-3  8-11 16-19 24-27
                                                      // 4-7 12-15 20-23 28-31
                }
            } else if constexpr (__bytes_per_output_element == 1 && sizeof(_To) == 64) {
                return reinterpret_cast<__vector_type_t<_SChar, 64>>(_mm512_movm_epi8(__k));
            } else if constexpr (__bytes_per_output_element == 2 && sizeof(_To) == 16) {
                if constexpr (__have_avx512bw_vl) {
                    return __vector_bitcast<_ToT>(_mm_movm_epi16(__k));
                } else if constexpr (__have_avx512bw) {
                    return __vector_bitcast<_ToT>(__lo128(_mm512_movm_epi16(__k)));
                } else {
                    __m256i __as32bits;
                    if constexpr (__have_avx512vl) {
                        __as32bits = _mm256_maskz_mov_epi32(__k, ~__m256i());
                    } else {
                        __as32bits = __lo256(_mm512_maskz_mov_epi32(__k, ~__m512i()));
                    }
                    return __vector_bitcast<_ToT>(
                        _mm_packs_epi32(__lo128(__as32bits), __hi128(__as32bits)));
                }
            } else if constexpr (__bytes_per_output_element == 2 && sizeof(_To) == 32) {
                if constexpr (__have_avx512bw_vl) {
                    return __vector_bitcast<_ToT>(_mm256_movm_epi16(__k));
                } else if constexpr (__have_avx512bw) {
                    return __vector_bitcast<_ToT>(__lo256(_mm512_movm_epi16(__k)));
                } else {
                    auto __as32bits = _mm512_maskz_mov_epi32(__k, ~__m512i());
                    return __vector_bitcast<_ToT>(__xzyw(
                        _mm256_packs_epi32(__lo256(__as32bits), __hi256(__as32bits))));
                }
            } else if constexpr (__bytes_per_output_element == 2 && sizeof(_To) == 64) {
                return __vector_bitcast<_ToT>(_mm512_movm_epi16(__k));
            } else if constexpr (__bytes_per_output_element == 4 && sizeof(_To) == 16) {
                return __vector_bitcast<_ToT>(
                    __have_avx512dq_vl
                        ? _mm_movm_epi32(__k)
                        : __have_avx512dq
                              ? __lo128(_mm512_movm_epi32(__k))
                              : __have_avx512vl
                                    ? _mm_maskz_mov_epi32(__k, ~__m128i())
                                    : __lo128(_mm512_maskz_mov_epi32(__k, ~__m512i())));
            } else if constexpr (__bytes_per_output_element == 4 && sizeof(_To) == 32) {
                return __vector_bitcast<_ToT>(
                    __have_avx512dq_vl
                        ? _mm256_movm_epi32(__k)
                        : __have_avx512dq
                              ? __lo256(_mm512_movm_epi32(__k))
                              : __have_avx512vl
                                    ? _mm256_maskz_mov_epi32(__k, ~__m256i())
                                    : __lo256(_mm512_maskz_mov_epi32(__k, ~__m512i())));
            } else if constexpr (__bytes_per_output_element == 4 && sizeof(_To) == 64) {
                return __vector_bitcast<_ToT>(__have_avx512dq
                                             ? _mm512_movm_epi32(__k)
                                             : _mm512_maskz_mov_epi32(__k, ~__m512i()));
            } else if constexpr (__bytes_per_output_element == 8 && sizeof(_To) == 16) {
                return __vector_bitcast<_ToT>(
                    __have_avx512dq_vl
                        ? _mm_movm_epi64(__k)
                        : __have_avx512dq
                              ? __lo128(_mm512_movm_epi64(__k))
                              : __have_avx512vl
                                    ? _mm_maskz_mov_epi64(__k, ~__m128i())
                                    : __lo128(_mm512_maskz_mov_epi64(__k, ~__m512i())));
            } else if constexpr (__bytes_per_output_element == 8 && sizeof(_To) == 32) {
                return __vector_bitcast<_ToT>(
                    __have_avx512dq_vl
                        ? _mm256_movm_epi64(__k)
                        : __have_avx512dq
                              ? __lo256(_mm512_movm_epi64(__k))
                              : __have_avx512vl
                                    ? _mm256_maskz_mov_epi64(__k, ~__m256i())
                                    : __lo256(_mm512_maskz_mov_epi64(__k, ~__m512i())));
            } else if constexpr (__bytes_per_output_element == 8 && sizeof(_To) == 64) {
                return __vector_bitcast<_ToT>(__have_avx512dq
                                             ? _mm512_movm_epi64(__k)
                                             : _mm512_maskz_mov_epi64(__k, ~__m512i()));
            } else {
                __assert_unreachable<_To>();
            }
        } else if constexpr (__have_sse) {
            using _U = std::make_unsigned_t<__int_for_sizeof_t<_ToT>>;
            using _V = __vector_type_t<_U, _N>;  // simd<_U, _Abi>;
            static_assert(sizeof(_V) <= 32);  // can't be AVX512
            constexpr size_t __bits_per_element = sizeof(_U) * CHAR_BIT;
            if constexpr (!__have_avx2 && __have_avx && sizeof(_V) == 32) {
                if constexpr (_N == 8) {
                    return _mm256_cmp_ps(
                        _mm256_and_ps(
                            _mm256_castsi256_ps(_mm256_set1_epi32(__k)),
                            _mm256_castsi256_ps(_mm256_setr_epi32(
                                0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80))),
                        _mm256_setzero_ps(), _CMP_NEQ_UQ);
                } else if constexpr (_N == 4) {
                    return _mm256_cmp_pd(
                        _mm256_and_pd(
                            _mm256_castsi256_pd(_mm256_set1_epi64x(__k)),
                            _mm256_castsi256_pd(
                                _mm256_setr_epi64x(0x01, 0x02, 0x04, 0x08))),
                        _mm256_setzero_pd(), _CMP_NEQ_UQ);
                } else {
                    __assert_unreachable<_To>();
                }
            } else if constexpr (__bits_per_element >= _N) {
                constexpr auto __bitmask = __generate_vector<__vector_type_t<_U, _N>>(
                    [](auto __i) constexpr -> _U { return 1ull << __i; });
                return __vector_bitcast<_ToT>(
                    (__vector_broadcast<_N, _U>(__k) & __bitmask) != 0);
            } else if constexpr (sizeof(_V) == 16 && sizeof(_ToT) == 1 && __have_ssse3) {
                const auto __bitmask = __to_intrin(__make_vector<_UChar>(
                    1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128));
                return __vector_bitcast<_ToT>(
                    __vector_bitcast<_ToT>(
                        _mm_shuffle_epi8(
                            __to_intrin(__vector_type_t<_ULLong, 2>{__k}),
                            _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                                          1)) &
                        __bitmask) != 0);
            } else if constexpr (sizeof(_V) == 32 && sizeof(_ToT) == 1 && __have_avx2) {
                const auto __bitmask =
                    _mm256_broadcastsi128_si256(__to_intrin(__make_vector<_UChar>(
                        1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128)));
                return __vector_bitcast<_ToT>(
                    __vector_bitcast<_ToT>(_mm256_shuffle_epi8(
                                        _mm256_broadcastsi128_si256(__to_intrin(
                                            __vector_type_t<_ULLong, 2>{__k})),
                                        _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                                                         1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                                                         2, 2, 3, 3, 3, 3, 3, 3, 3, 3)) &
                                    __bitmask) != 0);
                /* TODO:
                } else if constexpr (sizeof(_V) == 32 && sizeof(_ToT) == 2 && __have_avx2) {
                    constexpr auto __bitmask = _mm256_broadcastsi128_si256(
                        _mm_setr_epi8(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000)); return
                __vector_bitcast<_ToT>( _mm256_shuffle_epi8(
                                   _mm256_broadcastsi128_si256(__m128i{__k}),
                                   _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3)) & __bitmask) != 0;
                */
            } else {
                const _V __tmp = __generate_vector<_V>([&](auto __i) constexpr {
                                  return static_cast<_U>(
                                      __k >> (__bits_per_element * (__i / __bits_per_element)));
                              }) &
                              __generate_vector<_V>([](auto __i) constexpr {
                                  return static_cast<_U>(1ull << (__i % __bits_per_element));
                              });  // mask bit index
                return __vector_bitcast<_ToT>(__tmp != _V());
            }
        } else
#endif // _GLIBCXX_SIMD_X86INTRIN }}}
	  {
	    using _I = __int_for_sizeof_t<_ToT>;
	    return reinterpret_cast<_To>(
	      __generate_vector<__vector_type_t<_I, _N_out>>([&](auto __i) constexpr {
		return ((__k >> __i) & 1) == 0 ? _I() : ~_I();
	      }));
	  }
	// }}}
    } else if constexpr (__is_vector_type_v<_From> && std::is_unsigned_v<_To>) {
        // vector -> bits {{{
        using _Trait = _VectorTraits<_From>;
        using _Tp = typename _Trait::value_type;
        constexpr size_t _FromN = _Trait::_S_width;
        constexpr size_t cvt_id = _FromN * 10 + sizeof(_Tp);
        constexpr bool __have_avx512_int = __have_avx512f && std::is_integral_v<_Tp>;
        [[maybe_unused]]  // PR85827
        const auto __intrin = __to_intrin(__k);

#if _GLIBCXX_SIMD_X86INTRIN // {{{
             if constexpr (cvt_id == 16'1 && __have_avx512bw_vl) { return    _mm_movepi8_mask(__intrin); }
        else if constexpr (cvt_id == 16'1 && __have_avx512bw   ) { return _mm512_movepi8_mask(__zero_extend(__intrin)); }
        else if constexpr (cvt_id == 16'1                      ) { return    _mm_movemask_epi8(__intrin); }
        else if constexpr (cvt_id == 32'1 && __have_avx512bw_vl) { return _mm256_movepi8_mask(__intrin); }
        else if constexpr (cvt_id == 32'1 && __have_avx512bw   ) { return _mm512_movepi8_mask(__zero_extend(__intrin)); }
        else if constexpr (cvt_id == 32'1                      ) { return _mm256_movemask_epi8(__intrin); }
        else if constexpr (cvt_id == 64'1 && __have_avx512bw   ) { return _mm512_movepi8_mask(__intrin); }
        else if constexpr (cvt_id ==  8'2 && __have_avx512bw_vl) { return    _mm_movepi16_mask(__intrin); }
        else if constexpr (cvt_id ==  8'2 && __have_avx512bw   ) { return _mm512_movepi16_mask(__zero_extend(__intrin)); }
        else if constexpr (cvt_id ==  8'2                      ) { return movemask_epi16(__intrin); }
        else if constexpr (cvt_id == 16'2 && __have_avx512bw_vl) { return _mm256_movepi16_mask(__intrin); }
        else if constexpr (cvt_id == 16'2 && __have_avx512bw   ) { return _mm512_movepi16_mask(__zero_extend(__intrin)); }
        else if constexpr (cvt_id == 16'2                      ) { return movemask_epi16(__intrin); }
        else if constexpr (cvt_id == 32'2 && __have_avx512bw   ) { return _mm512_movepi16_mask(__intrin); }
        else if constexpr (cvt_id ==  4'4 && __have_avx512dq_vl) { return    _mm_movepi32_mask(__vector_bitcast<_LLong>(__k)); }
        else if constexpr (cvt_id ==  4'4 && __have_avx512dq   ) { return _mm512_movepi32_mask(__zero_extend(__vector_bitcast<_LLong>(__k))); }
        else if constexpr (cvt_id ==  4'4 && __have_avx512vl   ) { return    _mm_cmp_epi32_mask(__vector_bitcast<_LLong>(__k), __m128i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  4'4 && __have_avx512_int ) { return _mm512_cmp_epi32_mask(__zero_extend(__intrin), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  4'4                      ) { return    _mm_movemask_ps(__k); }
        else if constexpr (cvt_id ==  8'4 && __have_avx512dq_vl) { return _mm256_movepi32_mask(__vector_bitcast<_LLong>(__k)); }
        else if constexpr (cvt_id ==  8'4 && __have_avx512dq   ) { return _mm512_movepi32_mask(__zero_extend(__vector_bitcast<_LLong>(__k))); }
        else if constexpr (cvt_id ==  8'4 && __have_avx512vl   ) { return _mm256_cmp_epi32_mask(__vector_bitcast<_LLong>(__k), __m256i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  8'4 && __have_avx512_int ) { return _mm512_cmp_epi32_mask(__zero_extend(__intrin), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  8'4                      ) { return _mm256_movemask_ps(__k); }
        else if constexpr (cvt_id == 16'4 && __have_avx512dq   ) { return _mm512_movepi32_mask(__vector_bitcast<_LLong>(__k)); }
        else if constexpr (cvt_id == 16'4                      ) { return _mm512_cmp_epi32_mask(__vector_bitcast<_LLong>(__k), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  2'8 && __have_avx512dq_vl) { return    _mm_movepi64_mask(__vector_bitcast<_LLong>(__k)); }
        else if constexpr (cvt_id ==  2'8 && __have_avx512dq   ) { return _mm512_movepi64_mask(__zero_extend(__vector_bitcast<_LLong>(__k))); }
        else if constexpr (cvt_id ==  2'8 && __have_avx512vl   ) { return    _mm_cmp_epi64_mask(__vector_bitcast<_LLong>(__k), __m128i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  2'8 && __have_avx512_int ) { return _mm512_cmp_epi64_mask(__zero_extend(__intrin), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  2'8                      ) { return    _mm_movemask_pd(__k); }
        else if constexpr (cvt_id ==  4'8 && __have_avx512dq_vl) { return _mm256_movepi64_mask(__vector_bitcast<_LLong>(__k)); }
        else if constexpr (cvt_id ==  4'8 && __have_avx512dq   ) { return _mm512_movepi64_mask(__zero_extend(__vector_bitcast<_LLong>(__k))); }
        else if constexpr (cvt_id ==  4'8 && __have_avx512vl   ) { return _mm256_cmp_epi64_mask(__vector_bitcast<_LLong>(__k), __m256i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  4'8 && __have_avx512_int ) { return _mm512_cmp_epi64_mask(__zero_extend(__intrin), __m512i(), _MM_CMPINT_LT); }
        else if constexpr (cvt_id ==  4'8                      ) { return _mm256_movemask_pd(__k); }
        else if constexpr (cvt_id ==  8'8 && __have_avx512dq   ) { return _mm512_movepi64_mask(__vector_bitcast<_LLong>(__k)); }
        else if constexpr (cvt_id ==  8'8                      ) { return _mm512_cmp_epi64_mask(__vector_bitcast<_LLong>(__k), __m512i(), _MM_CMPINT_LT); }
        else
#endif // _GLIBCXX_SIMD_X86INTRIN }}}
        __assert_unreachable<_To>();
        // }}}
    } else if constexpr (__is_vector_type_v<_From> && __is_vector_type_v<_To>) {
        // vector -> vector {{{
        using _ToTrait = _VectorTraits<_To>;
        using _FromTrait = _VectorTraits<_From>;
        using _ToT = typename _ToTrait::value_type;
        using _Tp = typename _FromTrait::value_type;
        constexpr size_t _FromN = _FromTrait::_S_width;
        constexpr size_t _ToN = _ToTrait::_S_width;
        constexpr int _FromBytes = sizeof(_Tp);
        constexpr int _ToBytes = sizeof(_ToT);

        if constexpr (_FromN == _ToN && sizeof(_From) == sizeof(_To))
	  { // reinterpret the bits
	    return reinterpret_cast<_To>(__k);
	  }
#if _GLIBCXX_SIMD_X86INTRIN // {{{
        else if constexpr (sizeof(_To) == 16 && sizeof(__k) == 16)
	{ // SSE -> SSE {{{
            if constexpr (_FromBytes == 4 && _ToBytes == 8) {
                if constexpr(std::is_integral_v<_Tp>) {
                    return __vector_bitcast<_ToT>(__interleave128_lo(__k, __k));
                } else {
                    return __vector_bitcast<_ToT>(__interleave128_lo(__k, __k));
                }
            } else if constexpr (_FromBytes == 2 && _ToBytes == 8) {
                const auto __y = __vector_bitcast<int>(__interleave128_lo(__k, __k));
                return __vector_bitcast<_ToT>(__interleave128_lo(__y, __y));
            } else if constexpr (_FromBytes == 1 && _ToBytes == 8) {
                auto __y = __vector_bitcast<short>(__interleave128_lo(__k, __k));
                auto __z = __vector_bitcast<int>(__interleave128_lo(__y, __y));
                return __vector_bitcast<_ToT>(__interleave128_lo(__z, __z));
            } else if constexpr (_FromBytes == 8 && _ToBytes == 4) {
		if constexpr (__have_sse2)
		  return __vector_bitcast<_ToT>(
		    _mm_packs_epi32(__vector_bitcast<_LLong>(__k), __m128i()));
		else
		  return __vector_shuffle<1, 3, 6, 7>(
		    __vector_bitcast<_ToT>(__k), _To());
	    } else if constexpr (_FromBytes == 2 && _ToBytes == 4) {
                return __vector_bitcast<_ToT>(__interleave128_lo(__k, __k));
            } else if constexpr (_FromBytes == 1 && _ToBytes == 4) {
                const auto __y = __vector_bitcast<short>(__interleave128_lo(__k, __k));
                return __vector_bitcast<_ToT>(__interleave128_lo(__y, __y));
            } else if constexpr (_FromBytes == 8 && _ToBytes == 2) {
		if constexpr (__have_sse2 && !__have_ssse3)
		  return __vector_bitcast<_ToT>(_mm_packs_epi32(
		    _mm_packs_epi32(__vector_bitcast<_LLong>(__k), __m128i()),
		    __m128i()));
		else
		  return __vector_permute<3, 7, -1, -1, -1, -1, -1, -1>(
		    __vector_bitcast<_ToT>(__k));
	    } else if constexpr (_FromBytes == 4 && _ToBytes == 2) {
                return __vector_bitcast<_ToT>(
                    _mm_packs_epi32(__vector_bitcast<_LLong>(__k), __m128i()));
            } else if constexpr (_FromBytes == 1 && _ToBytes == 2) {
                return __vector_bitcast<_ToT>(__interleave128_lo(__k, __k));
            } else if constexpr (_FromBytes == 8 && _ToBytes == 1) {
                if constexpr(__have_ssse3) {
                    return __vector_bitcast<_ToT>(
                        _mm_shuffle_epi8(__vector_bitcast<_LLong>(__k),
                                         _mm_setr_epi8(7, 15, -1, -1, -1, -1, -1, -1, -1,
                                                       -1, -1, -1, -1, -1, -1, -1)));
                } else {
                    auto __y = _mm_packs_epi32(__vector_bitcast<_LLong>(__k), __m128i());
                    __y = _mm_packs_epi32(__y, __m128i());
                    return __vector_bitcast<_ToT>(_mm_packs_epi16(__y, __m128i()));
                }
		return __vector_permute<7, 15, -1, -1, -1, -1, -1, -1, -1, -1,
					-1, -1, -1, -1, -1, -1>(
		  __vector_bitcast<_ToT>(__k));
	    } else if constexpr (_FromBytes == 4 && _ToBytes == 1) {
                if constexpr(__have_ssse3) {
                    return __vector_bitcast<_ToT>(
                        _mm_shuffle_epi8(__vector_bitcast<_LLong>(__k),
                                         _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1,
                                                       -1, -1, -1, -1, -1, -1, -1)));
                } else {
                    const auto __y = _mm_packs_epi32(__vector_bitcast<_LLong>(__k), __m128i());
                    return __vector_bitcast<_ToT>(_mm_packs_epi16(__y, __m128i()));
                }
		return __vector_permute<3, 7, 11, 15, -1, -1, -1, -1, -1, -1,
					-1, -1, -1, -1, -1, -1>(
		  __vector_bitcast<_ToT>(__k));
	    } else if constexpr (_FromBytes == 2 && _ToBytes == 1) {
                return __vector_bitcast<_ToT>(_mm_packs_epi16(__vector_bitcast<_LLong>(__k), __m128i()));
	    } else {
                static_assert(!std::is_same_v<_Tp, _Tp>, "should be unreachable");
            }
	  } // }}}
	else if constexpr (sizeof(_To) == 32 && sizeof(__k) == 32)
	  { // AVX -> AVX {{{
            if constexpr (_FromBytes == _ToBytes) {  // keep low 1/2
                __assert_unreachable<_Tp>();
            } else if constexpr (_FromBytes == _ToBytes * 2) {
                const auto __y = __vector_bitcast<_LLong>(__k);
                return __vector_bitcast<_ToT>(
                    _mm256_castsi128_si256(_mm_packs_epi16(__lo128(__y), __hi128(__y))));
            } else if constexpr (_FromBytes == _ToBytes * 4) {
                const auto __y = __vector_bitcast<_LLong>(__k);
                return __vector_bitcast<_ToT>(_mm256_castsi128_si256(
                    _mm_packs_epi16(_mm_packs_epi16(__lo128(__y), __hi128(__y)), __m128i())));
            } else if constexpr (_FromBytes == _ToBytes * 8) {
                const auto __y = __vector_bitcast<_LLong>(__k);
                return __vector_bitcast<_ToT>(_mm256_castsi128_si256(
                    _mm_shuffle_epi8(_mm_packs_epi16(__lo128(__y), __hi128(__y)),
                                     _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1))));
            } else if constexpr (_FromBytes * 2 == _ToBytes) {
                auto __y = __xzyw(__to_intrin(__k));
                if constexpr(std::is_floating_point_v<_Tp>) {
                    return __vector_bitcast<_ToT>(_mm256_unpacklo_ps(__y, __y));
                } else {
                    return __vector_bitcast<_ToT>(_mm256_unpacklo_epi8(__y, __y));
                }
            } else if constexpr (_FromBytes * 4 == _ToBytes) {
                auto __y = _mm_unpacklo_epi8(__lo128(__vector_bitcast<_LLong>(__k)),
                                           __lo128(__vector_bitcast<_LLong>(__k)));  // drops 3/4 of input
                return __vector_bitcast<_ToT>(
                    __concat(_mm_unpacklo_epi16(__y, __y), _mm_unpackhi_epi16(__y, __y)));
            } else if constexpr (_FromBytes == 1 && _ToBytes == 8) {
                auto __y = _mm_unpacklo_epi8(__lo128(__vector_bitcast<_LLong>(__k)),
                                           __lo128(__vector_bitcast<_LLong>(__k)));  // drops 3/4 of input
                __y = _mm_unpacklo_epi16(__y, __y);  // drops another 1/2 => 7/8 total
                return __vector_bitcast<_ToT>(
                    __concat(_mm_unpacklo_epi32(__y, __y), _mm_unpackhi_epi32(__y, __y)));
            } else {
                __assert_unreachable<_Tp>();
            }
	  } // }}}
	else if constexpr (sizeof(_To) == 32 && sizeof(__k) == 16)
	  { // SSE -> AVX {{{
            if constexpr (_FromBytes == _ToBytes) {
                return __vector_bitcast<_ToT>(
                    __intrinsic_type_t<_Tp, 32 / sizeof(_Tp)>(__zero_extend(__to_intrin(__k))));
            } else if constexpr (_FromBytes * 2 == _ToBytes) {  // keep all
                return __vector_bitcast<_ToT>(__concat(_mm_unpacklo_epi8(__vector_bitcast<_LLong>(__k), __vector_bitcast<_LLong>(__k)),
                                         _mm_unpackhi_epi8(__vector_bitcast<_LLong>(__k), __vector_bitcast<_LLong>(__k))));
            } else if constexpr (_FromBytes * 4 == _ToBytes) {
                if constexpr (__have_avx2) {
                    return __vector_bitcast<_ToT>(_mm256_shuffle_epi8(
                        __concat(__vector_bitcast<_LLong>(__k), __vector_bitcast<_LLong>(__k)),
                        _mm256_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                         4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7,
                                         7)));
                } else {
                    return __vector_bitcast<_ToT>(
                        __concat(_mm_shuffle_epi8(__vector_bitcast<_LLong>(__k),
                                                _mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2,
                                                              2, 2, 2, 3, 3, 3, 3)),
                               _mm_shuffle_epi8(__vector_bitcast<_LLong>(__k),
                                                _mm_setr_epi8(4, 4, 4, 4, 5, 5, 5, 5, 6,
                                                              6, 6, 6, 7, 7, 7, 7))));
                }
            } else if constexpr (_FromBytes * 8 == _ToBytes) {
                if constexpr (__have_avx2) {
                    return __vector_bitcast<_ToT>(_mm256_shuffle_epi8(
                        __concat(__vector_bitcast<_LLong>(__k), __vector_bitcast<_LLong>(__k)),
                        _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
                                         3)));
                } else {
                    return __vector_bitcast<_ToT>(
                        __concat(_mm_shuffle_epi8(__vector_bitcast<_LLong>(__k),
                                                _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1,
                                                              1, 1, 1, 1, 1, 1, 1)),
                               _mm_shuffle_epi8(__vector_bitcast<_LLong>(__k),
                                                _mm_setr_epi8(2, 2, 2, 2, 2, 2, 2, 2, 3,
                                                              3, 3, 3, 3, 3, 3, 3))));
                }
            } else if constexpr (_FromBytes == _ToBytes * 2) {
                return __vector_bitcast<_ToT>(
                    __m256i(__zero_extend(_mm_packs_epi16(__vector_bitcast<_LLong>(__k), __m128i()))));
            } else if constexpr (_FromBytes == 8 && _ToBytes == 2) {
                return __vector_bitcast<_ToT>(__m256i(__zero_extend(
                    _mm_shuffle_epi8(__vector_bitcast<_LLong>(__k),
                                     _mm_setr_epi8(6, 7, 14, 15, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1)))));
            } else if constexpr (_FromBytes == 4 && _ToBytes == 1) {
                return __vector_bitcast<_ToT>(__m256i(__zero_extend(
                    _mm_shuffle_epi8(__vector_bitcast<_LLong>(__k),
                                     _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1)))));
            } else if constexpr (_FromBytes == 8 && _ToBytes == 1) {
                return __vector_bitcast<_ToT>(__m256i(__zero_extend(
                    _mm_shuffle_epi8(__vector_bitcast<_LLong>(__k),
                                     _mm_setr_epi8(7, 15, -1, -1, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1)))));
            } else {
                static_assert(!std::is_same_v<_Tp, _Tp>, "should be unreachable");
            }
	  } // }}}
	else if constexpr (sizeof(_To) == 16 && sizeof(__k) == 32)
	  { // AVX -> SSE {{{
            if constexpr (_FromBytes == _ToBytes) {  // keep low 1/2
                return __vector_bitcast<_ToT>(__lo128(__k));
            } else if constexpr (_FromBytes == _ToBytes * 2) {  // keep all
                auto __y = __vector_bitcast<_LLong>(__k);
                return __vector_bitcast<_ToT>(_mm_packs_epi16(__lo128(__y), __hi128(__y)));
            } else if constexpr (_FromBytes == _ToBytes * 4) {  // add 1/2 undef
                auto __y = __vector_bitcast<_LLong>(__k);
                return __vector_bitcast<_ToT>(
                    _mm_packs_epi16(_mm_packs_epi16(__lo128(__y), __hi128(__y)), __m128i()));
            } else if constexpr (_FromBytes == 8 && _ToBytes == 1) {  // add 3/4 undef
                auto __y = __vector_bitcast<_LLong>(__k);
                return __vector_bitcast<_ToT>(
                    _mm_shuffle_epi8(_mm_packs_epi16(__lo128(__y), __hi128(__y)),
                                     _mm_setr_epi8(3, 7, 11, 15, -1, -1, -1, -1, -1, -1,
                                                   -1, -1, -1, -1, -1, -1)));
            } else if constexpr (_FromBytes * 2 == _ToBytes) {  // keep low 1/4
                auto __y = __lo128(__vector_bitcast<_LLong>(__k));
                return __vector_bitcast<_ToT>(_mm_unpacklo_epi8(__y, __y));
            } else if constexpr (_FromBytes * 4 == _ToBytes) {  // keep low 1/8
                auto __y = __lo128(__vector_bitcast<_LLong>(__k));
                __y = _mm_unpacklo_epi8(__y, __y);
                return __vector_bitcast<_ToT>(_mm_unpacklo_epi8(__y, __y));
            } else if constexpr (_FromBytes * 8 == _ToBytes) {  // keep low 1/16
                auto __y = __lo128(__vector_bitcast<_LLong>(__k));
                __y = _mm_unpacklo_epi8(__y, __y);
                __y = _mm_unpacklo_epi8(__y, __y);
                return __vector_bitcast<_ToT>(_mm_unpacklo_epi8(__y, __y));
            } else {
                static_assert(!std::is_same_v<_Tp, _Tp>, "should be unreachable");
            }
	  } // }}}
#endif // _GLIBCXX_SIMD_X86INTRIN }}}
	else
	  {
	    using _I = __int_for_sizeof_t<_ToT>;
	    return reinterpret_cast<_To>(
	      __generate_vector<__vector_type_t<_I, _ToN>>([&](auto __i) constexpr {
		return __i >= _FromN || __k[int(__i)] == 0 ? _I() : ~_I();
	      }));
	  }
	/*
        } else if constexpr (_FromBytes > _ToBytes) {
	    const _To     __y      = __vector_bitcast<_ToT>(__k);
	    return [&] <std::size_t... _Is> (std::index_sequence<_Is...>) {
	      constexpr int _Stride = _FromBytes / _ToBytes;
	      return _To{__y[(_Is + 1) * _Stride - 1]...};
	    }(std::make_index_sequence<std::min(_ToN, _FromN)>());
	} else {
	    // {0, 0, 1, 1} (_Dups = 2, _Is<4>)
	    // {0, 0, 0, 0, 1, 1, 1, 1} (_Dups = 4, _Is<8>)
	    // {0, 0, 1, 1, 2, 2, 3, 3} (_Dups = 2, _Is<8>)
	    // ...
	    return [&] <std::size_t... _Is> (std::index_sequence<_Is...>) {
	      constexpr int __dup = _ToBytes / _FromBytes;
	      return __vector_bitcast<_ToT>(_From{__k[_Is / __dup]...});
	    }(std::make_index_sequence<_FromN>());
	}
	*/
        // }}}
    } else {
        __assert_unreachable<_To>();
    }
}

// }}}

template <class _Abi> struct _SimdMathFallback {  //{{{
  using _SuperImpl = typename _Abi::_SimdImpl;

#define _GLIBCXX_SIMD_MATH_FALLBACK(__name)                                    \
  template <typename _Tp, typename... _More>                                   \
  static _Tp __##__name(const _Tp& __x, const _More&... __more)                \
  {                                                                            \
    if constexpr ((__is_vectorizable_v<_Tp> && ... &&                          \
		   __is_vectorizable_v<_More>))                                \
      return std::__name(__x, __more...);                                      \
    else if constexpr (__is_vectorizable_v<_Tp>)                               \
      return std::__name(__x, __more[0]...);                                   \
    else                                                                       \
      return __generate_vector<_Tp>(                                           \
	[&](auto __i) { return std::__name(__x[__i], __more[__i]...); });      \
  }

#define _GLIBCXX_SIMD_MATH_FALLBACK_MASKRET(__name)                            \
  template <typename _Tp, typename... _More>                                   \
  static                                                                       \
    typename _Tp::mask_type __##__name(const _Tp& __x, const _More&... __more) \
  {                                                                            \
    if constexpr ((__is_vectorizable_v<_Tp> && ... &&                          \
		   __is_vectorizable_v<_More>))                                \
      return std::__name(__x, __more...);                                      \
    else if constexpr (__is_vectorizable_v<_Tp>)                               \
      return std::__name(__x, __more[0]...);                                   \
    else                                                                       \
      return __generate_vector<_Tp>(                                           \
	[&](auto __i) { return std::__name(__x[__i], __more[__i]...); });      \
  }

#define _GLIBCXX_SIMD_MATH_FALLBACK_FIXEDRET(_RetTp, __name)                   \
  template <typename _Tp, typename... _More>                                   \
  static auto __##__name(const _Tp& __x, const _More&... __more)               \
  {                                                                            \
    if constexpr (__is_vectorizable_v<_Tp>)                                    \
      return _SimdTuple<_RetTp, simd_abi::scalar>{                             \
	std::__name(__x, __more...)};                                          \
    else                                                                       \
      return __fixed_size_storage_t<_RetTp, _VectorTraits<_Tp>::_S_width>::    \
	__generate([&](auto __meta) constexpr {                                \
	  return __meta.__generator(                                           \
	    [&](auto __i) {                                                    \
	      return std::__name(__x[__meta._S_offset + __i],                  \
				 __more[__meta._S_offset + __i]...);           \
	    },                                                                 \
	    static_cast<_RetTp*>(nullptr));                                    \
	});                                                                    \
  }

  _GLIBCXX_SIMD_MATH_FALLBACK(acos)
  _GLIBCXX_SIMD_MATH_FALLBACK(asin)
  _GLIBCXX_SIMD_MATH_FALLBACK(atan)
  _GLIBCXX_SIMD_MATH_FALLBACK(atan2)
  _GLIBCXX_SIMD_MATH_FALLBACK(cos)
  _GLIBCXX_SIMD_MATH_FALLBACK(sin)
  _GLIBCXX_SIMD_MATH_FALLBACK(tan)
  _GLIBCXX_SIMD_MATH_FALLBACK(acosh)
  _GLIBCXX_SIMD_MATH_FALLBACK(asinh)
  _GLIBCXX_SIMD_MATH_FALLBACK(atanh)
  _GLIBCXX_SIMD_MATH_FALLBACK(cosh)
  _GLIBCXX_SIMD_MATH_FALLBACK(sinh)
  _GLIBCXX_SIMD_MATH_FALLBACK(tanh)
  _GLIBCXX_SIMD_MATH_FALLBACK(exp)
  _GLIBCXX_SIMD_MATH_FALLBACK(exp2)
  _GLIBCXX_SIMD_MATH_FALLBACK(expm1)
  _GLIBCXX_SIMD_MATH_FALLBACK(ldexp)
  _GLIBCXX_SIMD_MATH_FALLBACK_FIXEDRET(int, ilogb)
  _GLIBCXX_SIMD_MATH_FALLBACK(log)
  _GLIBCXX_SIMD_MATH_FALLBACK(log10)
  _GLIBCXX_SIMD_MATH_FALLBACK(log1p)
  _GLIBCXX_SIMD_MATH_FALLBACK(log2)
  _GLIBCXX_SIMD_MATH_FALLBACK(logb)

  //modf implemented in simd_math.h
  _GLIBCXX_SIMD_MATH_FALLBACK(scalbn)
  _GLIBCXX_SIMD_MATH_FALLBACK(scalbln)
  _GLIBCXX_SIMD_MATH_FALLBACK(cbrt)
  _GLIBCXX_SIMD_MATH_FALLBACK(abs)
  _GLIBCXX_SIMD_MATH_FALLBACK(fabs)
  _GLIBCXX_SIMD_MATH_FALLBACK(pow)
  _GLIBCXX_SIMD_MATH_FALLBACK(sqrt)
  _GLIBCXX_SIMD_MATH_FALLBACK(erf)
  _GLIBCXX_SIMD_MATH_FALLBACK(erfc)
  _GLIBCXX_SIMD_MATH_FALLBACK(lgamma)
  _GLIBCXX_SIMD_MATH_FALLBACK(tgamma)
  _GLIBCXX_SIMD_MATH_FALLBACK(ceil)
  _GLIBCXX_SIMD_MATH_FALLBACK(floor)
  _GLIBCXX_SIMD_MATH_FALLBACK(nearbyint)

  _GLIBCXX_SIMD_MATH_FALLBACK(rint)
  _GLIBCXX_SIMD_MATH_FALLBACK_FIXEDRET(long, lrint)
  _GLIBCXX_SIMD_MATH_FALLBACK_FIXEDRET(long long, llrint)

  _GLIBCXX_SIMD_MATH_FALLBACK(round)
  _GLIBCXX_SIMD_MATH_FALLBACK_FIXEDRET(long, lround)
  _GLIBCXX_SIMD_MATH_FALLBACK_FIXEDRET(long long, llround)

  _GLIBCXX_SIMD_MATH_FALLBACK(trunc)
  _GLIBCXX_SIMD_MATH_FALLBACK(fmod)
  _GLIBCXX_SIMD_MATH_FALLBACK(remainder)

  template <typename _Tp, typename _TVT = _VectorTraits<_Tp>>
  static _Tp __remquo(const _Tp                                   __x,
		      const _Tp                                   __y,
		      __fixed_size_storage_t<int, _TVT::_S_width>* __z)
  {
    return __generate_vector<_Tp>([&](auto __i) {
      int  __tmp;
      auto __r    = std::remquo(__x[__i], __y[__i], &__tmp);
      __z->__set(__i, __tmp);
      return __r;
    });
  }

  // copysign in simd_math.h
  _GLIBCXX_SIMD_MATH_FALLBACK(nextafter)
  _GLIBCXX_SIMD_MATH_FALLBACK(fdim)
  _GLIBCXX_SIMD_MATH_FALLBACK(fmax)
  _GLIBCXX_SIMD_MATH_FALLBACK(fmin)
  _GLIBCXX_SIMD_MATH_FALLBACK(fma)
  _GLIBCXX_SIMD_MATH_FALLBACK_FIXEDRET(int, fpclassify)

  template <class _Tp>
  _GLIBCXX_SIMD_INTRINSIC simd_mask<_Tp, _Abi>
			  __isfinite(const simd<_Tp, _Abi>& __x)
  {
    return simd<_Tp, _Abi>([&](auto __i) { return std::isfinite(__x[__i]); });
  }

  template <class _Tp>
  _GLIBCXX_SIMD_INTRINSIC simd_mask<_Tp, _Abi>
			  __isinf(const simd<_Tp, _Abi>& __x)
  {
    return simd<_Tp, _Abi>([&](auto __i) { return std::isinf(__x[__i]); });
  }

  template <class _Tp>
  _GLIBCXX_SIMD_INTRINSIC simd_mask<_Tp, _Abi>
			  __isnan(const simd<_Tp, _Abi>& __x)
  {
    return simd<_Tp, _Abi>([&](auto __i) { return std::isnan(__x[__i]); });
  }

  template <class _Tp>
  _GLIBCXX_SIMD_INTRINSIC simd_mask<_Tp, _Abi>
			  __isnormal(const simd<_Tp, _Abi>& __x)
  {
    return simd<_Tp, _Abi>([&](auto __i) { return std::isnormal(__x[__i]); });
  }

  template <class _Tp>
  _GLIBCXX_SIMD_INTRINSIC simd_mask<_Tp, _Abi>
			  __signbit(const simd<_Tp, _Abi>& __x)
  {
    return simd<_Tp, _Abi>([&](auto __i) { return std::signbit(__x[__i]); });
  }

  template <typename _Tp, typename _TVT = _VectorTraits<_Tp>>
  _GLIBCXX_SIMD_INTRINSIC static constexpr auto
    __isgreater(const _Tp& __x, const _Tp& __y)
  {
    return _SuperImpl::__less(__y, __x);
  }
  template <typename _Tp, typename _TVT = _VectorTraits<_Tp>>
  _GLIBCXX_SIMD_INTRINSIC static constexpr auto
    __isgreaterequal(const _Tp& __x, const _Tp& __y)
  {
    return _SuperImpl::__less_equal(__y, __x);
  }
  template <typename _Tp, typename _TVT = _VectorTraits<_Tp>>
  _GLIBCXX_SIMD_INTRINSIC static constexpr auto
    __isless(const _Tp& __x, const _Tp& __y)
  {
    return _SuperImpl::__less(__x, __y);
  }
  template <typename _Tp, typename _TVT = _VectorTraits<_Tp>>
  _GLIBCXX_SIMD_INTRINSIC static constexpr auto
    __islessequal(const _Tp& __x, const _Tp& __y)
  {
    return _SuperImpl::__less_equal(__x, __y);
  }
  template <typename _Tp, typename _TVT = _VectorTraits<_Tp>>
  _GLIBCXX_SIMD_INTRINSIC static constexpr auto
    __islessgreater(const _Tp& __x, const _Tp& __y)
  {
    return __or(_SuperImpl::__less(__y, __x), _SuperImpl::__less(__x, __y));
  }

  template <typename _Tp, typename _TVT = _VectorTraits<_Tp>>
  _GLIBCXX_SIMD_INTRINSIC static constexpr auto
    __isunordered(const _Tp& __x, const _Tp& __y)
  {
    return __cmpunord(__x, __y);
  }

#undef _GLIBCXX_SIMD_MATH_FALLBACK
#undef _GLIBCXX_SIMD_MATH_FALLBACK_FIXEDRET
};  // }}}
// _SimdImplScalar {{{
struct _SimdImplScalar : _SimdMathFallback<simd_abi::scalar> {
    // member types {{{2
    using abi_type = std::experimental::simd_abi::scalar;
    using _MaskMember = bool;
    template <class _Tp> using _SimdMember = _Tp;
    template <class _Tp> using _Simd = std::experimental::simd<_Tp, abi_type>;
    template <class _Tp> using _TypeTag = _Tp *;

    // broadcast {{{2
    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC static constexpr _Tp __broadcast(_Tp __x) noexcept
    {
        return __x;
    }

    // __generator {{{2
    template <class _F, class _Tp>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _Tp
      __generator(_F&& __gen, _TypeTag<_Tp>)
    {
        return __gen(_SizeConstant<0>());
    }

    // __load {{{2
    template <class _Tp, class _U, class _F>
    static inline _Tp __load(const _U *__mem, _F, _TypeTag<_Tp>) noexcept
    {
        return static_cast<_Tp>(__mem[0]);
    }

    // __masked_load {{{2
    template <class _Tp, class _U, class _F>
    static inline _Tp __masked_load(_Tp __merge, bool __k, const _U *__mem, _F) noexcept
    {
        if (__k) {
            __merge = static_cast<_Tp>(__mem[0]);
        }
        return __merge;
    }

    // __store {{{2
    template <class _Tp, class _U, class _F>
    static inline void __store(_Tp __v, _U *__mem, _F, _TypeTag<_Tp>) noexcept
    {
        __mem[0] = static_cast<_Tp>(__v);
    }

    // __masked_store {{{2
    template <class _Tp, class _U, class _F>
    static inline void __masked_store(const _Tp __v, _U *__mem, _F, const bool __k) noexcept
    {
        if (__k) {
            __mem[0] = __v;
        }
    }

    // __negate {{{2
    template <class _Tp> static inline bool __negate(_Tp __x) noexcept { return !__x; }

    // __reduce {{{2
    template <class _Tp, class _BinaryOperation>
    static inline _Tp __reduce(const _Simd<_Tp> &__x, _BinaryOperation &)
    {
        return __x._M_data;
    }

    // __min, __max {{{2
    template <class _Tp> static inline _Tp __min(const _Tp __a, const _Tp __b)
    {
        return std::min(__a, __b);
    }

    template <class _Tp> static inline _Tp __max(const _Tp __a, const _Tp __b)
    {
        return std::max(__a, __b);
    }

    // __complement {{{2
    template <class _Tp> static inline _Tp __complement(_Tp __x) noexcept
    {
        return static_cast<_Tp>(~__x);
    }

    // __unary_minus {{{2
    template <class _Tp> static inline _Tp __unary_minus(_Tp __x) noexcept
    {
        return static_cast<_Tp>(-__x);
    }

    // arithmetic operators {{{2
    template <class _Tp> static inline _Tp __plus(_Tp __x, _Tp __y)
    {
        return static_cast<_Tp>(__promote_preserving_unsigned(__x) +
                              __promote_preserving_unsigned(__y));
    }

    template <class _Tp> static inline _Tp __minus(_Tp __x, _Tp __y)
    {
        return static_cast<_Tp>(__promote_preserving_unsigned(__x) -
                              __promote_preserving_unsigned(__y));
    }

    template <class _Tp> static inline constexpr _Tp __multiplies(_Tp __x, _Tp __y)
    {
        return static_cast<_Tp>(__promote_preserving_unsigned(__x) *
                              __promote_preserving_unsigned(__y));
    }

    template <class _Tp> static inline _Tp __divides(_Tp __x, _Tp __y)
    {
        return static_cast<_Tp>(__promote_preserving_unsigned(__x) /
                              __promote_preserving_unsigned(__y));
    }

    template <class _Tp> static inline _Tp __modulus(_Tp __x, _Tp __y)
    {
        return static_cast<_Tp>(__promote_preserving_unsigned(__x) %
                              __promote_preserving_unsigned(__y));
    }

    template <class _Tp>
    static inline _Tp __bit_and(_Tp __x, _Tp __y)
    {
      if constexpr (is_floating_point_v<_Tp>)
	{
	  using _I     = __int_for_sizeof_t<_Tp>;
	  const _I __r = reinterpret_cast<const __may_alias<_I>&>(__x) &
			 reinterpret_cast<const __may_alias<_I>&>(__y);
	  return reinterpret_cast<const __may_alias<_Tp>&>(__r);
	}
      else
	{
	  return static_cast<_Tp>(__promote_preserving_unsigned(__x) &
				 __promote_preserving_unsigned(__y));
	}
    }

    template <class _Tp>
    static inline _Tp __bit_or(_Tp __x, _Tp __y)
    {
      if constexpr (is_floating_point_v<_Tp>)
	{
	  using _I     = __int_for_sizeof_t<_Tp>;
	  const _I __r = reinterpret_cast<const __may_alias<_I>&>(__x) |
			 reinterpret_cast<const __may_alias<_I>&>(__y);
	  return reinterpret_cast<const __may_alias<_Tp>&>(__r);
	}
      else
	{
	  return static_cast<_Tp>(__promote_preserving_unsigned(__x) |
				 __promote_preserving_unsigned(__y));
	}
    }

    template <class _Tp>
    static inline _Tp __bit_xor(_Tp __x, _Tp __y)
    {
      if constexpr (is_floating_point_v<_Tp>)
	{
	  using _I     = __int_for_sizeof_t<_Tp>;
	  const _I __r = reinterpret_cast<const __may_alias<_I>&>(__x) ^
			 reinterpret_cast<const __may_alias<_I>&>(__y);
	  return reinterpret_cast<const __may_alias<_Tp>&>(__r);
	}
      else
	{
	  return static_cast<_Tp>(__promote_preserving_unsigned(__x) ^
				 __promote_preserving_unsigned(__y));
	}
    }

    template <class _Tp> static inline _Tp __bit_shift_left(_Tp __x, int __y)
    {
        return static_cast<_Tp>(__promote_preserving_unsigned(__x) << __y);
    }

    template <class _Tp> static inline _Tp __bit_shift_right(_Tp __x, int __y)
    {
        return static_cast<_Tp>(__promote_preserving_unsigned(__x) >> __y);
    }

    // math {{{2
    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC static _Tp __abs(_Tp __x) { return _Tp(std::abs(__x)); }
    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC static _Tp __sqrt(_Tp __x) { return std::sqrt(__x); }
    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC static _Tp __trunc(_Tp __x) { return std::trunc(__x); }
    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC static _Tp __floor(_Tp __x) { return std::floor(__x); }
    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC static _Tp __ceil(_Tp __x) { return std::ceil(__x); }

    template <typename _Tp>
    _GLIBCXX_SIMD_INTRINSIC static _Tp
      __remquo(_Tp __x, _Tp __y, _SimdTuple<int, simd_abi::_ScalarAbi>* __z)
    {
      return std::remquo(__x, __y, &__z->first);
    }
    template <typename _Tp>
    _GLIBCXX_SIMD_INTRINSIC static _Tp __remquo(_Tp __x, _Tp __y, int* __z)
    {
      return std::remquo(__x, __y, __z);
    }

    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC static _SimdTuple<int, abi_type> __fpclassify(_Tp __x)
    {
        return {std::fpclassify(__x)};
    }
    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC static bool __isfinite(_Tp __x) { return std::isfinite(__x); }
    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC static bool __isinf(_Tp __x) { return std::isinf(__x); }
    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC static bool __isnan(_Tp __x) { return std::isnan(__x); }
    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC static bool __isnormal(_Tp __x) { return std::isnormal(__x); }
    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC static bool __signbit(_Tp __x) { return std::signbit(__x); }
    template <class _Tp> _GLIBCXX_SIMD_INTRINSIC static bool __isunordered(_Tp __x, _Tp __y) { return std::isunordered(__x, __y); }

    // __increment & __decrement{{{2
    template <class _Tp> static inline void __increment(_Tp &__x) { ++__x; }
    template <class _Tp> static inline void __decrement(_Tp &__x) { --__x; }

    // compares {{{2
    template <class _Tp> static bool __equal_to(_Tp __x, _Tp __y) { return __x == __y; }
    template <class _Tp> static bool __not_equal_to(_Tp __x, _Tp __y) { return __x != __y; }
    template <class _Tp> static bool __less(_Tp __x, _Tp __y) { return __x < __y; }
    template <class _Tp> static bool __less_equal(_Tp __x, _Tp __y) { return __x <= __y; }

    // smart_reference access {{{2
    template <class _Tp, class _U> static void __set(_Tp &__v, int __i, _U &&__x) noexcept
    {
        _GLIBCXX_DEBUG_ASSERT(__i == 0);
        __unused(__i);
        __v = std::forward<_U>(__x);
    }

    // __masked_assign {{{2
    template <typename _Tp> _GLIBCXX_SIMD_INTRINSIC static void __masked_assign(bool __k, _Tp &__lhs, _Tp __rhs)
    {
        if (__k) {
            __lhs = __rhs;
        }
    }

    // __masked_cassign {{{2
    template <template <typename> class _Op, typename _Tp>
    _GLIBCXX_SIMD_INTRINSIC static void __masked_cassign(const bool __k, _Tp &__lhs, const _Tp __rhs)
    {
        if (__k) {
            __lhs = _Op<_Tp>{}(__lhs, __rhs);
        }
    }

    // __masked_unary {{{2
    template <template <typename> class _Op, typename _Tp>
    _GLIBCXX_SIMD_INTRINSIC static _Tp __masked_unary(const bool __k, const _Tp __v)
    {
        return static_cast<_Tp>(__k ? _Op<_Tp>{}(__v) : __v);
    }

    // }}}2
};

// }}}
// _MaskImplScalar {{{
struct _MaskImplScalar {
    // member types {{{2
    template <class _Tp> using _TypeTag = _Tp *;

    // __from_bitset {{{2
    template <class _Tp>
    _GLIBCXX_SIMD_INTRINSIC static bool __from_bitset(std::bitset<1> __bs, _TypeTag<_Tp>) noexcept
    {
        return __bs[0];
    }

    // __masked_load {{{2
    template <class _F>
    _GLIBCXX_SIMD_INTRINSIC static bool __masked_load(bool __merge, bool __mask, const bool *__mem,
                                         _F) noexcept
    {
        if (__mask) {
            __merge = __mem[0];
        }
        return __merge;
    }

    // __store {{{2
    template <class _F> _GLIBCXX_SIMD_INTRINSIC static void __store(bool __v, bool *__mem, _F) noexcept
    {
        __mem[0] = __v;
    }

    // __masked_store {{{2
    template <class _F>
    _GLIBCXX_SIMD_INTRINSIC static void __masked_store(const bool __v, bool *__mem, _F,
                                          const bool __k) noexcept
    {
        if (__k) {
            __mem[0] = __v;
        }
    }

    // logical and bitwise operators {{{2
    static constexpr bool __logical_and(bool __x, bool __y) { return __x && __y; }
    static constexpr bool __logical_or(bool __x, bool __y) { return __x || __y; }
    static constexpr bool __bit_and(bool __x, bool __y) { return __x && __y; }
    static constexpr bool __bit_or(bool __x, bool __y) { return __x || __y; }
    static constexpr bool __bit_xor(bool __x, bool __y) { return __x != __y; }

    // smart_reference access {{{2
    static void __set(bool &__k, int __i, bool __x) noexcept
    {
        _GLIBCXX_DEBUG_ASSERT(__i == 0);
        __unused(__i);
        __k = __x;
    }

    // __masked_assign {{{2
    _GLIBCXX_SIMD_INTRINSIC static void __masked_assign(bool __k, bool &__lhs, bool __rhs)
    {
        if (__k) {
            __lhs = __rhs;
        }
    }

    // }}}2
};

// }}}

// ISA & type detection {{{1
template <class _Tp, size_t _N> constexpr bool __is_sse_ps()
{
    return __have_sse && std::is_same_v<_Tp, float> && _N == 4;
}
template <class _Tp, size_t _N> constexpr bool __is_sse_pd()
{
    return __have_sse2 && std::is_same_v<_Tp, double> && _N == 2;
}
template <class _Tp, size_t _N> constexpr bool __is_avx_ps()
{
    return __have_avx && std::is_same_v<_Tp, float> && _N == 8;
}
template <class _Tp, size_t _N> constexpr bool __is_avx_pd()
{
    return __have_avx && std::is_same_v<_Tp, double> && _N == 4;
}
template <class _Tp, size_t _N> constexpr bool __is_avx512_ps()
{
    return __have_avx512f && std::is_same_v<_Tp, float> && _N == 16;
}
template <class _Tp, size_t _N> constexpr bool __is_avx512_pd()
{
    return __have_avx512f && std::is_same_v<_Tp, double> && _N == 8;
}

template <class _Tp, size_t _N> constexpr bool __is_neon_ps()
{
    return __have_neon && std::is_same_v<_Tp, float> && _N == 4;
}
template <class _Tp, size_t _N> constexpr bool __is_neon_pd()
{
    return __have_neon && std::is_same_v<_Tp, double> && _N == 2;
}

// _SimdImplBuiltin {{{1
template <class _Abi> struct _SimdImplBuiltin : _SimdMathFallback<_Abi> {
    // member types {{{2
    using abi_type = _Abi;
    template <class _Tp> using _TypeTag = _Tp *;
    template <class _Tp>
    using _SimdMember = typename _Abi::template __traits<_Tp>::_SimdMember;
    template <class _Tp>
    using _MaskMember = typename _Abi::template __traits<_Tp>::_MaskMember;
    template <class _Tp> static constexpr size_t _S_full_size = _SimdMember<_Tp>::_S_width;
    using _SuperImpl = typename _Abi::_SimdImpl;

    // __make_simd(_SimdWrapper/__intrinsic_type_t) {{{2
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static simd<_Tp, _Abi> __make_simd(_SimdWrapper<_Tp, _N> __x)
    {
        return {__private_init, __x};
    }
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static simd<_Tp, _Abi> __make_simd(__intrinsic_type_t<_Tp, _N> __x)
    {
        return {__private_init, __vector_bitcast<_Tp>(__x)};
    }

    // __broadcast {{{2
    template <class _Tp>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _SimdMember<_Tp> __broadcast(_Tp __x) noexcept
    {
        return __vector_broadcast<_S_full_size<_Tp>>(__x);
    }

    // __generator {{{2
    template <class _F, class _Tp>
    inline static constexpr _SimdMember<_Tp>
      __generator(_F&& __gen, _TypeTag<_Tp>)
    {
        return __generate_wrapper<_Tp, _S_full_size<_Tp>>(std::forward<_F>(__gen));
    }

    // __load {{{2
    template <class _Tp, class _U, class _F>
    _GLIBCXX_SIMD_INTRINSIC static _SimdMember<_Tp> __load(const _U *__mem, _F,
                                                 _TypeTag<_Tp>) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        constexpr size_t _N = _SimdMember<_Tp>::_S_width;
        constexpr size_t __max_load_size =
            (sizeof(_U) >= 4 && __have_avx512f) || __have_avx512bw
                ? 64
                : (std::is_floating_point_v<_U> && __have_avx) || __have_avx2 ? 32 : 16;
        if constexpr (sizeof(_U) > 8) {
            return __generate_wrapper<_Tp, _N>(
                [&](auto __i) constexpr { return static_cast<_Tp>(__mem[__i]); });
        } else if constexpr (std::is_same_v<_U, _Tp>) {
            return __vector_load<_U, _N>(__mem, _F());
        } else if constexpr (sizeof(_U) * _N < 16) {
            return __convert<_SimdMember<_Tp>>(
                __vector_load16<_U, sizeof(_U) * _N>(__mem, _F()));
        } else if constexpr (sizeof(_U) * _N <= __max_load_size) {
            return __convert<_SimdMember<_Tp>>(__vector_load<_U, _N>(__mem, _F()));
        } else if constexpr (sizeof(_U) * _N == 2 * __max_load_size) {
            return __convert<_SimdMember<_Tp>>(
                __vector_load<_U, _N / 2>(__mem, _F()),
                __vector_load<_U, _N / 2>(__mem + _N / 2, _F()));
        } else if constexpr (sizeof(_U) * _N == 4 * __max_load_size) {
            return __convert<_SimdMember<_Tp>>(
                __vector_load<_U, _N / 4>(__mem, _F()),
                __vector_load<_U, _N / 4>(__mem + 1 * _N / 4, _F()),
                __vector_load<_U, _N / 4>(__mem + 2 * _N / 4, _F()),
                __vector_load<_U, _N / 4>(__mem + 3 * _N / 4, _F()));
        } else if constexpr (sizeof(_U) * _N == 8 * __max_load_size) {
            return __convert<_SimdMember<_Tp>>(
                __vector_load<_U, _N / 8>(__mem, _F()),
                __vector_load<_U, _N / 8>(__mem + 1 * _N / 8, _F()),
                __vector_load<_U, _N / 8>(__mem + 2 * _N / 8, _F()),
                __vector_load<_U, _N / 8>(__mem + 3 * _N / 8, _F()),
                __vector_load<_U, _N / 8>(__mem + 4 * _N / 8, _F()),
                __vector_load<_U, _N / 8>(__mem + 5 * _N / 8, _F()),
                __vector_load<_U, _N / 8>(__mem + 6 * _N / 8, _F()),
                __vector_load<_U, _N / 8>(__mem + 7 * _N / 8, _F()));
        } else {
            __assert_unreachable<_Tp>();
        }
    }

    // __masked_load {{{2
    template <class _Tp, size_t _N, class _U, class _F>
    static inline _SimdWrapper<_Tp, _N> __masked_load(_SimdWrapper<_Tp, _N> __merge,
                                                _MaskMember<_Tp> __k,
                                                const _U *__mem,
                                                _F) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
      __bit_iteration(__vector_to_bitset(__k._M_data).to_ullong(), [&](auto __i) {
		      __merge.__set(__i, static_cast<_Tp>(__mem[__i]));
		      });
      return __merge;
    }

    // __store {{{2
    template <class _Tp, class _U, class _F>
    _GLIBCXX_SIMD_INTRINSIC static void __store(_SimdMember<_Tp> __v, _U *__mem, _F,
                                   _TypeTag<_Tp>) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        // TODO: converting int -> "smaller int" can be optimized with AVX512
        constexpr size_t _N = _SimdMember<_Tp>::_S_width;
        constexpr size_t __max_store_size =
            (sizeof(_U) >= 4 && __have_avx512f) || __have_avx512bw
                ? 64
                : (std::is_floating_point_v<_U> && __have_avx) || __have_avx2 ? 32 : 16;
        if constexpr (sizeof(_U) > 8) {
            __execute_n_times<_N>([&](auto __i) constexpr { __mem[__i] = __v[__i]; });
        } else if constexpr (std::is_same_v<_U, _Tp>) {
            __vector_store(__v._M_data, __mem, _F());
        } else if constexpr (sizeof(_U) * _N < 16) {
            __vector_store<sizeof(_U) * _N>(__convert<__vector_type16_t<_U>>(__v),
                                                 __mem, _F());
        } else if constexpr (sizeof(_U) * _N <= __max_store_size) {
            __vector_store(__convert<__vector_type_t<_U, _N>>(__v), __mem, _F());
        } else {
            constexpr size_t __vsize = __max_store_size / sizeof(_U);
            constexpr size_t __stores = _N / __vsize;
            using _V = __vector_type_t<_U, __vsize>;
            const std::array<_V, __stores> __converted = __convert_all<_V>(__v);
            __execute_n_times<__stores>([&](auto __i) constexpr {
                __vector_store(__converted[__i], __mem + __i * __vsize, _F());
            });
        }
    }

    // __masked_store {{{2
    template <class _Tp, size_t _N, class _U, class _F>
    static inline void __masked_store(const _SimdWrapper<_Tp, _N> __v, _U *__mem, _F,
                                    const _MaskMember<_Tp> __k) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        [[maybe_unused]] const auto __vi = __to_intrin(__v);
        constexpr size_t __max_store_size =
            (sizeof(_U) >= 4 && __have_avx512f) || __have_avx512bw
                ? 64
                : (std::is_floating_point_v<_U> && __have_avx) || __have_avx2 ? 32 : 16;
        if constexpr (std::is_same_v<_Tp, _U> ||
                      (std::is_integral_v<_Tp> && std::is_integral_v<_U> &&
                       sizeof(_Tp) == sizeof(_U))) {
            // bitwise or no conversion, reinterpret:
            const auto __kk = [&]() {
                if constexpr (__is_bitmask_v<decltype(__k)>) {
                    return _MaskMember<_U>(__k._M_data);
                } else {
                    return __wrapper_bitcast<_U>(__k);
                }
            }();
            __maskstore(__wrapper_bitcast<_U>(__v), __mem, _F(), __kk);
        } else if constexpr (sizeof(_U) <= 8 &&  // no long double
                             !__converts_via_decomposition_v<
                                 _Tp, _U, __max_store_size>  // conversion via decomposition
                                                          // is better handled via the
                                                          // bit_iteration fallback below
        ) {
            using _VV = _SimdWrapper<_U, std::clamp(_N, 16 / sizeof(_U), __max_store_size / sizeof(_U))>;
            using _V = typename _VV::_BuiltinType;
            constexpr bool __prefer_bitmask =
                (__have_avx512f && sizeof(_U) >= 4) || __have_avx512bw;
            using _M = _SimdWrapper<std::conditional_t<__prefer_bitmask, bool, _U>, _VV::_S_width>;
            constexpr size_t _VN = _VectorTraits<_V>::_S_width;

            if constexpr (_VN >= _N) {
                __maskstore(_VV(__convert<_V>(__v)), __mem,
                               // careful, if _V has more elements than the input __v (_N),
                               // vector_aligned is incorrect:
                               std::conditional_t<(_VectorTraits<_V>::_S_width > _N),
                                                  overaligned_tag<sizeof(_U) * _N>, _F>(),
                               __convert_mask<_M>(__k));
            } else if constexpr (_VN * 2 == _N) {
                const std::array<_V, 2> converted = __convert_all<_V>(__v);
                __maskstore(_VV(converted[0]), __mem, _F(), __convert_mask<_M>(__extract_part<0, 2>(__k)));
                __maskstore(_VV(converted[1]), __mem + _VV::_S_width, _F(), __convert_mask<_M>(__extract_part<1, 2>(__k)));
            } else if constexpr (_VN * 4 == _N) {
                const std::array<_V, 4> converted = __convert_all<_V>(__v);
                __maskstore(_VV(converted[0]), __mem, _F(), __convert_mask<_M>(__extract_part<0, 4>(__k)));
                __maskstore(_VV(converted[1]), __mem + 1 * _VV::_S_width, _F(), __convert_mask<_M>(__extract_part<1, 4>(__k)));
                __maskstore(_VV(converted[2]), __mem + 2 * _VV::_S_width, _F(), __convert_mask<_M>(__extract_part<2, 4>(__k)));
                __maskstore(_VV(converted[3]), __mem + 3 * _VV::_S_width, _F(), __convert_mask<_M>(__extract_part<3, 4>(__k)));
            } else if constexpr (_VN * 8 == _N) {
                const std::array<_V, 8> converted = __convert_all<_V>(__v);
                __maskstore(_VV(converted[0]), __mem, _F(), __convert_mask<_M>(__extract_part<0, 8>(__k)));
                __maskstore(_VV(converted[1]), __mem + 1 * _VV::_S_width, _F(), __convert_mask<_M>(__extract_part<1, 8>(__k)));
                __maskstore(_VV(converted[2]), __mem + 2 * _VV::_S_width, _F(), __convert_mask<_M>(__extract_part<2, 8>(__k)));
                __maskstore(_VV(converted[3]), __mem + 3 * _VV::_S_width, _F(), __convert_mask<_M>(__extract_part<3, 8>(__k)));
                __maskstore(_VV(converted[4]), __mem + 4 * _VV::_S_width, _F(), __convert_mask<_M>(__extract_part<4, 8>(__k)));
                __maskstore(_VV(converted[5]), __mem + 5 * _VV::_S_width, _F(), __convert_mask<_M>(__extract_part<5, 8>(__k)));
                __maskstore(_VV(converted[6]), __mem + 6 * _VV::_S_width, _F(), __convert_mask<_M>(__extract_part<6, 8>(__k)));
                __maskstore(_VV(converted[7]), __mem + 7 * _VV::_S_width, _F(), __convert_mask<_M>(__extract_part<7, 8>(__k)));
            } else {
                __assert_unreachable<_Tp>();
            }
        } else {
            __bit_iteration(__vector_to_bitset(__k._M_data).to_ullong(),
                            [&](auto __i) constexpr { __mem[__i] = static_cast<_U>(__v[__i]); });
        }
    }

    // __complement {{{2
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _SimdWrapper<_Tp, _N> __complement(_SimdWrapper<_Tp, _N> __x) noexcept
    {
        return ~__x._M_data;
    }

    // __unary_minus {{{2
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _SimdWrapper<_Tp, _N> __unary_minus(_SimdWrapper<_Tp, _N> __x) noexcept
    {
        // GCC doesn't use the psign instructions, but pxor & psub seem to be just as good
        // a choice as pcmpeqd & psign. So meh.
        return -__x._M_data;
    }

    // arithmetic operators {{{2
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _SimdWrapper<_Tp, _N> __plus(_SimdWrapper<_Tp, _N> __x,
                                                                    _SimdWrapper<_Tp, _N> __y)
    {
      return __x._M_data + __y._M_data;
    }
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _SimdWrapper<_Tp, _N> __minus(_SimdWrapper<_Tp, _N> __x,
                                                                     _SimdWrapper<_Tp, _N> __y)
    {
      return __x._M_data - __y._M_data;
    }
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _SimdWrapper<_Tp, _N> __multiplies(
        _SimdWrapper<_Tp, _N> __x, _SimdWrapper<_Tp, _N> __y)
    {
      return __x._M_data * __y._M_data;
    }
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _SimdWrapper<_Tp, _N> __divides(
        _SimdWrapper<_Tp, _N> __x, _SimdWrapper<_Tp, _N> __y)
    {
        return __x._M_data / __y._M_data;
    }
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _SimdWrapper<_Tp, _N> __modulus(_SimdWrapper<_Tp, _N> __x, _SimdWrapper<_Tp, _N> __y)
    {
        static_assert(std::is_integral<_Tp>::value, "modulus is only supported for integral types");
        return __x._M_data % __y._M_data;
    }
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _SimdWrapper<_Tp, _N> __bit_and(_SimdWrapper<_Tp, _N> __x, _SimdWrapper<_Tp, _N> __y)
    {
        return __and(__x._M_data, __y._M_data);
    }
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _SimdWrapper<_Tp, _N> __bit_or(_SimdWrapper<_Tp, _N> __x, _SimdWrapper<_Tp, _N> __y)
    {
        return __or(__x._M_data, __y._M_data);
    }
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _SimdWrapper<_Tp, _N> __bit_xor(_SimdWrapper<_Tp, _N> __x, _SimdWrapper<_Tp, _N> __y)
    {
        return __xor(__x._M_data, __y._M_data);
    }
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _SimdWrapper<_Tp, _N> __bit_shift_left(_SimdWrapper<_Tp, _N> __x, _SimdWrapper<_Tp, _N> __y)
    {
        return __x._M_data << __y._M_data;
    }
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _SimdWrapper<_Tp, _N> __bit_shift_right(_SimdWrapper<_Tp, _N> __x, _SimdWrapper<_Tp, _N> __y)
    {
        return __x._M_data >> __y._M_data;
    }

    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _SimdWrapper<_Tp, _N> __bit_shift_left(_SimdWrapper<_Tp, _N> __x, int __y)
    {
        return __x._M_data << __y;
    }
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _SimdWrapper<_Tp, _N> __bit_shift_right(_SimdWrapper<_Tp, _N> __x,
                                                                         int __y)
    {
        return __x._M_data >> __y;
    }

    // compares {{{2
    // __equal_to {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _MaskMember<_Tp> __equal_to(
        _SimdWrapper<_Tp, _N> __x, _SimdWrapper<_Tp, _N> __y)
    {
      return _ToWrapper(__x._M_data == __y._M_data);
    }

    // __not_equal_to {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _MaskMember<_Tp> __not_equal_to(
        _SimdWrapper<_Tp, _N> __x, _SimdWrapper<_Tp, _N> __y)
    {
      return _ToWrapper(__x._M_data != __y._M_data);
    }

    // __less {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _MaskMember<_Tp> __less(_SimdWrapper<_Tp, _N> __x,
                                                           _SimdWrapper<_Tp, _N> __y)
    {
      return _ToWrapper(__x._M_data < __y._M_data);
    }

    // __less_equal {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _MaskMember<_Tp> __less_equal(_SimdWrapper<_Tp, _N> __x,
                                                                 _SimdWrapper<_Tp, _N> __y)
    {
      return _ToWrapper(__x._M_data <= __y._M_data);
    }

    // negation {{{2
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _MaskMember<_Tp> __negate(_SimdWrapper<_Tp, _N> __x) noexcept
    {
      return _ToWrapper(!__x._M_data);
    }

    // __min, __max, __minmax {{{2
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_NORMAL_MATH _GLIBCXX_SIMD_INTRINSIC static constexpr _SimdWrapper<_Tp, _N> __min(_SimdWrapper<_Tp, _N> __a,
                                                                   _SimdWrapper<_Tp, _N> __b)
    {
        return __a._M_data < __b._M_data ? __a._M_data : __b._M_data;
    }
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_NORMAL_MATH _GLIBCXX_SIMD_INTRINSIC static constexpr _SimdWrapper<_Tp, _N> __max(_SimdWrapper<_Tp, _N> __a,
                                                                   _SimdWrapper<_Tp, _N> __b)
    {
        return __a._M_data > __b._M_data ? __a._M_data : __b._M_data;
    }

    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_NORMAL_MATH _GLIBCXX_SIMD_INTRINSIC static constexpr std::pair<_SimdWrapper<_Tp, _N>, _SimdWrapper<_Tp, _N>>
    __minmax(_SimdWrapper<_Tp, _N> __a, _SimdWrapper<_Tp, _N> __b)
    {
        return {__a._M_data < __b._M_data ? __a._M_data : __b._M_data, __a._M_data < __b._M_data ? __b._M_data : __a._M_data};
    }

    // reductions {{{2
    template <class _Tp, class _BinaryOperation>
    _GLIBCXX_SIMD_INTRINSIC static _Tp
      __reduce(simd<_Tp, _Abi> __x, _BinaryOperation&& __binary_op)
    {
      constexpr size_t _N = simd_size_v<_Tp, _Abi>;
      if constexpr (sizeof(__x) > __min_vector_size && _N > 2)
	{
	  using _A = simd_abi::deduce_t<_Tp, _N / 2>;
	  using _V = std::experimental::simd<_Tp, _A>;
	  return _A::_SimdImpl::__reduce(
	    __binary_op(
	      _V(__private_init, __extract<0, 2>(__data(__x)._M_data)),
	      _V(__private_init, __extract<1, 2>(__data(__x)._M_data))),
	    std::forward<_BinaryOperation>(__binary_op));
	}
#if _GLIBCXX_SIMD_HAVE_NEON // {{{
      else if constexpr (sizeof(__x) == 8 || sizeof(__x) == 16)
	{
	  static_assert(_N <= 8); // either 64-bit vectors or 128-bit double
	  if constexpr (_N == 8)
	    {
	      __x = __binary_op(
		__x, __make_simd<_Tp, _N>(
		       __vector_permute<1, 0, 3, 2, 5, 4, 7, 6>(__x._M_data)));
	      __x = __binary_op(
		__x, __make_simd<_Tp, _N>(
		       __vector_permute<3, 2, 1, 0, 7, 6, 5, 4>(__x._M_data)));
	      __x = __binary_op(
		__x, __make_simd<_Tp, _N>(
		       __vector_permute<7, 6, 5, 4, 3, 2, 1, 0>(__x._M_data)));
              return __x[0];
	    }
	  else if constexpr (_N == 4)
	    {
	      __x = __binary_op(
		__x, __make_simd<_Tp, _N>(
		       __vector_permute<1, 0, 3, 2>(__x._M_data)));
	      __x = __binary_op(
		__x, __make_simd<_Tp, _N>(
		       __vector_permute<3, 2, 1, 0>(__x._M_data)));
              return __x[0];
	    }
	  else
	    {
	      static_assert(_N == 2);
	      __x = __binary_op(
		__x, __make_simd<_Tp, _N>(__vector_permute<1, 0>(__x._M_data)));
	      return __x[0];
	    }
	}
#endif // _GLIBCXX_SIMD_HAVE_NEON }}}
      else if constexpr (sizeof(__x) == 16)
	{
	  if constexpr (_N == 16)
	    {
	      const auto __y = __x._M_data;
	      __x            = __binary_op(
                __make_simd<_Tp, _N>(__vector_permute<0, 0, 1, 1, 2, 2, 3, 3, 4,
                                                    4, 5, 5, 6, 6, 7, 7>(__y)),
                __make_simd<_Tp, _N>(
                  __vector_permute<8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
                                   14, 14, 15, 15>(__y)));
	    }
	  if constexpr (_N >= 8)
	    {
	      const auto __y = __vector_bitcast<short>(__x._M_data);
	      __x =
		__binary_op(__make_simd<_Tp, _N>(__vector_bitcast<_Tp>(
			      __vector_permute<0, 0, 1, 1, 2, 2, 3, 3>(__y))),
			    __make_simd<_Tp, _N>(__vector_bitcast<_Tp>(
			      __vector_permute<4, 4, 5, 5, 6, 6, 7, 7>(__y))));
	    }
	  if constexpr (_N >= 4)
	    {
	      using _U =
		std::conditional_t<std::is_floating_point_v<_Tp>, float, int>;
	      const auto __y = __vector_bitcast<_U>(__x._M_data);
	      __x = __binary_op(__x, __make_simd<_Tp, _N>(__vector_bitcast<_Tp>(
				       __vector_permute<3, 2, 1, 0>(__y))));
	    }
	  using _U =
	    std::conditional_t<std::is_floating_point_v<_Tp>, double, _LLong>;
	  const auto __y = __vector_bitcast<_U>(__x._M_data);
	  __x = __binary_op(__x, __make_simd<_Tp, _N>(__vector_bitcast<_Tp>(
				   __vector_permute<1, 1>(__y))));
	  return __x[0];
	}
      else
	__assert_unreachable<_Tp>();
    }

    // math {{{2
    // __abs {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _SimdWrapper<_Tp, _N>
      __abs(_SimdWrapper<_Tp, _N> __x) noexcept
    {
      // if (__builtin_is_constant_evaluated())
      //  {
      //    return __x._M_data < 0 ? -__x._M_data : __x._M_data;
      //  }
      if constexpr (std::is_floating_point_v<_Tp>)
	// `v < 0 ? -v : v` cannot compile to the efficient implementation of
	// masking the signbit off because it must consider v == -0

	// ~(-0.) & v would be easy, but breaks with fno-signed-zeros
	return __and(_S_absmask<__vector_type_t<_Tp, _N>>, __x._M_data);
      else
	return __x._M_data < 0 ? -__x._M_data : __x._M_data;
    }

    // __nearbyint {{{3
    template <typename _Tp, typename _TVT = _VectorTraits<_Tp>>
    _GLIBCXX_SIMD_INTRINSIC static _Tp __nearbyint(_Tp __x_) noexcept
    {
      using value_type = typename _TVT::value_type;
      using _V        = typename _TVT::type;
      const _V __x    = __x_;
      const _V __absx = __and(__x, _S_absmask<_V>);
      static_assert(CHAR_BIT * sizeof(1ull) >=
		    std::numeric_limits<value_type>::digits);
      constexpr _V __shifter_abs =
	_V() + (1ull << (std::numeric_limits<value_type>::digits - 1));
      const _V __shifter = __or(__and(_S_signmask<_V>, __x), __shifter_abs);
      _V __shifted = __x + __shifter;
      // how can we stop -fassociative-math to break this pattern?
      //asm("" : "+X"(__shifted));
      __shifted -= __shifter;
      return __absx < __shifter_abs ? __shifted : __x;
    }

    // __rint {{{3
    template <typename _Tp, typename _TVT = _VectorTraits<_Tp>>
    _GLIBCXX_SIMD_INTRINSIC static _Tp __rint(_Tp __x) noexcept
    {
      return _SuperImpl::__nearbyint(__x);
    }

    // __trunc {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _SimdWrapper<_Tp, _N>
      __trunc(_SimdWrapper<_Tp, _N> __x)
    {
      using _V        = __vector_type_t<_Tp, _N>;
      const _V __absx = __and(__x._M_data, _S_absmask<_V>);
      static_assert(CHAR_BIT * sizeof(1ull) >=
		    std::numeric_limits<_Tp>::digits);
      constexpr _Tp __shifter = 1ull << (std::numeric_limits<_Tp>::digits - 1);
      _V            __truncated = (__absx + __shifter) - __shifter;
      __truncated -= __truncated > __absx ? _V() + 1 : _V();
      return __absx < __shifter ? __or(__xor(__absx, __x._M_data), __truncated)
				: __x._M_data;
    }

    // __round {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _SimdWrapper<_Tp, _N>
      __round(_SimdWrapper<_Tp, _N> __x)
    {
      using _V        = __vector_type_t<_Tp, _N>;
      const _V __absx = __and(__x._M_data, _S_absmask<_V>);
      static_assert(CHAR_BIT * sizeof(1ull) >=
		    std::numeric_limits<_Tp>::digits);
      constexpr _Tp __shifter = 1ull << (std::numeric_limits<_Tp>::digits - 1);
      _V            __truncated = (__absx + __shifter) - __shifter;
      __truncated -= __truncated > __absx ? _V() + 1 : _V();
      const _V __rounded =
	__or(__xor(__absx, __x._M_data),
	     __truncated + (__absx - __truncated >= _Tp(.5) ? _V() + 1 : _V()));
      return __absx < __shifter ? __rounded : __x._M_data;
    }

    // __floor {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _SimdWrapper<_Tp, _N> __floor(_SimdWrapper<_Tp, _N> __x)
    {
      const auto __y = _SuperImpl::__trunc(__x)._M_data;
      const auto __negative_input = __vector_bitcast<_Tp>(__x._M_data < __vector_broadcast<_N, _Tp>(0));
      const auto __mask = __andnot(__vector_bitcast<_Tp>(__y == __x._M_data), __negative_input);
      return __or(__andnot(__mask, __y), __and(__mask, __y - __vector_broadcast<_N, _Tp>(1)));
    }

    // __ceil {{{3
    template <class _Tp, size_t _N> _GLIBCXX_SIMD_INTRINSIC static _SimdWrapper<_Tp, _N> __ceil(_SimdWrapper<_Tp, _N> __x)
    {
      const auto __y = _SuperImpl::__trunc(__x)._M_data;
      const auto __negative_input = __vector_bitcast<_Tp>(__x._M_data < __vector_broadcast<_N, _Tp>(0));
      const auto __inv_mask = __or(__vector_bitcast<_Tp>(__y == __x._M_data), __negative_input);
      return __or(__and(__inv_mask, __y),
		  __andnot(__inv_mask, __y + __vector_broadcast<_N, _Tp>(1)));
    }

    // __isnan {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _MaskMember<_Tp> __isnan(_SimdWrapper<_Tp, _N> __x)
    {
#if __FINITE_MATH_ONLY__
      __unused(__x);
      return {}; // false
#else
      return __cmpunord(__x._M_data, __x._M_data);
#endif
    }

    // __isfinite {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _MaskMember<_Tp> __isfinite(_SimdWrapper<_Tp, _N> __x)
    {
#if __FINITE_MATH_ONLY__
      __unused(__x);
      return __vector_bitcast<_N>(_Tp()) == __vector_bitcast<_N>(_Tp());
#else
      // if all exponent bits are set, __x is either inf or NaN
      using _I = __int_for_sizeof_t<_Tp>;
      const auto __inf = __vector_bitcast<_I>(
	__vector_broadcast<_N>(std::numeric_limits<_Tp>::infinity()));
      return __vector_bitcast<_Tp>(__inf >
				   (__vector_bitcast<_I>(__x) & __inf));
#endif
    }

    // __isunordered {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _MaskMember<_Tp> __isunordered(_SimdWrapper<_Tp, _N> __x,
                                                          _SimdWrapper<_Tp, _N> __y)
    {
        return __cmpunord(__x._M_data, __y._M_data);
    }

    // __signbit {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _MaskMember<_Tp> __signbit(_SimdWrapper<_Tp, _N> __x)
    {
      using _I = __int_for_sizeof_t<_Tp>;
      const auto __xx = __vector_bitcast<_I>(__x._M_data);
      return __vector_bitcast<_Tp>(__xx >> std::numeric_limits<_I>::digits);
    }

    // __isinf {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _MaskMember<_Tp> __isinf(_SimdWrapper<_Tp, _N> __x)
    {
#if __FINITE_MATH_ONLY__
      __unused(__x);
      return {}; // false
#else
      return _SuperImpl::template __equal_to<_Tp, _N>(
	_SuperImpl::__abs(__x),
	__vector_broadcast<_N>(std::numeric_limits<_Tp>::infinity()));
      // alternative:
      // compare to inf using the corresponding integer type
      /*
	 return
	 __vector_bitcast<_Tp>(__vector_bitcast<__int_for_sizeof_t<_Tp>>(__abs(__x)._M_data)
	 ==
	 __vector_bitcast<__int_for_sizeof_t<_Tp>>(__vector_broadcast<_N>(
	 std::numeric_limits<_Tp>::infinity())));
	 */
#endif
    }

    // __isnormal {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _MaskMember<_Tp>
      __isnormal(_SimdWrapper<_Tp, _N> __x)
    {
#if __FINITE_MATH_ONLY__
      return _SuperImpl::template __less_equal<_Tp, _N>(
	__vector_broadcast<_N>(std::numeric_limits<_Tp>::min()), _SuperImpl::__abs(__x));
#else
      return __and(
	_SuperImpl::template __less_equal<_Tp, _N>(
	  __vector_broadcast<_N>(std::numeric_limits<_Tp>::min()), _SuperImpl::__abs(__x)),
	_SuperImpl::template __less<_Tp, _N>(
	  _SuperImpl::__abs(__x),
	  __vector_broadcast<_N>(std::numeric_limits<_Tp>::infinity())));
#endif
    }

    // __fpclassify {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __fixed_size_storage_t<int, _N> __fpclassify(_SimdWrapper<_Tp, _N> __x)
    {
      constexpr auto __fp_normal = __vector_bitcast<_Tp>(
	__vector_broadcast<_N, __int_for_sizeof_t<_Tp>>(FP_NORMAL));
      constexpr auto __fp_nan = __vector_bitcast<_Tp>(
	__vector_broadcast<_N, __int_for_sizeof_t<_Tp>>(FP_NAN));
      constexpr auto __fp_infinite = __vector_bitcast<_Tp>(
	__vector_broadcast<_N, __int_for_sizeof_t<_Tp>>(FP_INFINITE));
      constexpr auto __fp_subnormal = __vector_bitcast<_Tp>(
	__vector_broadcast<_N, __int_for_sizeof_t<_Tp>>(FP_SUBNORMAL));
      constexpr auto __fp_zero = __vector_bitcast<_Tp>(
	__vector_broadcast<_N, __int_for_sizeof_t<_Tp>>(FP_ZERO));

      const auto __tmp = __vector_bitcast<_LLong>(
	_SuperImpl::__abs(__x)._M_data < std::numeric_limits<_Tp>::min()
	  ? (__x._M_data == 0 ? __fp_zero : __fp_subnormal)
	  : __blend(__isinf(__x)._M_data,
		    __blend(__isnan(__x)._M_data, __fp_normal, __fp_nan),
		    __fp_infinite));
      if constexpr (sizeof(_Tp) == sizeof(int))
	{
	  if constexpr (__fixed_size_storage_t<int, _N>::_S_tuple_size == 1)
	    {
	      return {__vector_bitcast<int>(__tmp)};
	    }
	  else if constexpr (__fixed_size_storage_t<int, _N>::_S_tuple_size == 2)
	    {
	      return {__extract<0, 2>(__vector_bitcast<int>(__tmp)),
		      __extract<1, 2>(__vector_bitcast<int>(__tmp))};
	    }
	  else
	    {
	      __assert_unreachable<_Tp>();
	    }
	}
      else if constexpr (_N == 2 && sizeof(_Tp) == 8 &&
			 __fixed_size_storage_t<int, _N>::_S_tuple_size == 2)
	{
	  return {int(__tmp[0]), {int(__tmp[1])}};
	}
      else if constexpr (_N == 4 && sizeof(_Tp) == 8 &&
			 __fixed_size_storage_t<int, _N>::_S_tuple_size == 1)
	{
#if _GLIBCXX_SIMD_X86INTRIN
	  return {_mm_packs_epi32(__lo128(__tmp), __hi128(__tmp))};
#else  // _GLIBCXX_SIMD_X86INTRIN
	  return {__make_wrapper<int>(__tmp[0], __tmp[1], __tmp[2], __tmp[3])};
#endif // _GLIBCXX_SIMD_X86INTRIN
	}
      else if constexpr (_N == 2 && sizeof(_Tp) == 8 &&
			 __fixed_size_storage_t<int, _N>::_S_tuple_size == 1)
	return {__make_wrapper<int>(__tmp[0], __tmp[1])};
      else
	{
	  __assert_unreachable<_Tp>();
	}
    }

    // __increment & __decrement{{{2
    template <class _Tp, size_t _N> _GLIBCXX_SIMD_INTRINSIC static void __increment(_SimdWrapper<_Tp, _N> &__x)
    {
        __x = __x._M_data + 1;
    }
    template <class _Tp, size_t _N> _GLIBCXX_SIMD_INTRINSIC static void __decrement(_SimdWrapper<_Tp, _N> &__x)
    {
        __x = __x._M_data - 1;
    }

    // smart_reference access {{{2
    template <class _Tp, size_t _N, class _U>
    _GLIBCXX_SIMD_INTRINSIC static void __set(_SimdWrapper<_Tp, _N> &__v, int __i, _U &&__x) noexcept
    {
        __v.__set(__i, std::forward<_U>(__x));
    }

    // __masked_assign{{{2
    template <class _Tp, class _K, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static void __masked_assign(_SimdWrapper<_K, _N> __k,
                                                      _SimdWrapper<_Tp, _N> &__lhs,
                                                      __id<_SimdWrapper<_Tp, _N>> __rhs)
    {
        __lhs = __blend(__k._M_data, __lhs._M_data, __rhs._M_data);
    }

    template <class _Tp, class _K, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static void __masked_assign(_SimdWrapper<_K, _N> __k, _SimdWrapper<_Tp, _N> &__lhs,
                                           __id<_Tp> __rhs)
    {
        if (__builtin_constant_p(__rhs) && __rhs == 0 && std::is_same<_K, _Tp>::value) {
            if constexpr (!__is_bitmask(__k)) {
                // the __andnot optimization only makes sense if __k._M_data is a vector register
                __lhs._M_data = __andnot(__k._M_data, __lhs._M_data);
                return;
            } else {
                // for AVX512/__mmask, a _mm512_maskz_mov is best
                __lhs._M_data = __auto_bitcast(__blend(__k, __lhs, __intrinsic_type_t<_Tp, _N>()));
                return;
            }
        }
        __lhs._M_data = __blend(__k._M_data, __lhs._M_data, __vector_broadcast<_N>(__rhs));
    }

    // __masked_cassign {{{2
    template <template <typename> class _Op, class _Tp, class _K, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static void __masked_cassign(const _SimdWrapper<_K, _N> __k, _SimdWrapper<_Tp, _N> &__lhs,
                                            const __id<_SimdWrapper<_Tp, _N>> __rhs)
    {
        __lhs._M_data = __blend(__k._M_data, __lhs._M_data, _Op<void>{}(__lhs._M_data, __rhs._M_data));
    }

    template <template <typename> class _Op, class _Tp, class _K, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static void __masked_cassign(const _SimdWrapper<_K, _N> __k, _SimdWrapper<_Tp, _N> &__lhs,
                                            const __id<_Tp> __rhs)
    {
        __lhs._M_data = __blend(__k._M_data, __lhs._M_data, _Op<void>{}(__lhs._M_data, __vector_broadcast<_N>(__rhs)));
    }

    // __masked_unary {{{2
    template <template <typename> class _Op, class _Tp, class _K, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _SimdWrapper<_Tp, _N> __masked_unary(const _SimdWrapper<_K, _N> __k,
                                                            const _SimdWrapper<_Tp, _N> __v)
    {
        auto __vv = __make_simd(__v);
        _Op<decltype(__vv)> __op;
        return __blend(__k, __v, __data(__op(__vv)));
    }

    //}}}2
};

// __generic_mask_impl {{{1
template <class _Abi> struct __generic_mask_impl {
    // member types {{{2
    template <class _Tp> using _TypeTag = _Tp *;
    template <class _Tp>
    using _SimdMember = typename _Abi::template __traits<_Tp>::_SimdMember;
    template <class _Tp>
    using _MaskMember = typename _Abi::template __traits<_Tp>::_MaskMember;

    // __masked_load {{{2
    template <class _Tp, size_t _N, class _F>
    static inline _SimdWrapper<_Tp, _N> __masked_load(_SimdWrapper<_Tp, _N> __merge,
						    _SimdWrapper<_Tp, _N> __mask,
						    const bool*           __mem,
						    _F) noexcept
    {
      // AVX(2) has 32/64 bit maskload, but nothing at 8 bit granularity
      auto __tmp = __wrapper_bitcast<__int_for_sizeof_t<_Tp>>(__merge);
      __bit_iteration(__vector_to_bitset(__mask._M_data).to_ullong(),
		      [&](auto __i) { __tmp.__set(__i, -__mem[__i]); });
      __merge = __wrapper_bitcast<_Tp>(__tmp);
      return __merge;
    }

    // __store {{{2
    template <class _Tp, size_t _N, class _F>
    _GLIBCXX_SIMD_INTRINSIC static void __store(_SimdWrapper<_Tp, _N> __v, bool *__mem, _F) noexcept
    {
      __execute_n_times<_N>([&](auto __i) constexpr { __mem[__i] = __v[__i]; });
    }

    // __masked_store {{{2
    template <class _Tp, size_t _N, class _F>
    static inline void __masked_store(const _SimdWrapper<_Tp, _N> __v, bool *__mem, _F,
                                    const _SimdWrapper<_Tp, _N> __k) noexcept
    {
      __bit_iteration(__vector_to_bitset(__k._M_data).to_ullong(),
		      [&](auto __i) constexpr { __mem[__i] = __v[__i]; });
    }

    // __from_bitset{{{2
    template <size_t _N, class _Tp>
    _GLIBCXX_SIMD_INTRINSIC static _MaskMember<_Tp> __from_bitset(std::bitset<_N> __bits, _TypeTag<_Tp>)
    {
        return __convert_mask<typename _MaskMember<_Tp>::_BuiltinType>(__bits);
    }

    // logical and bitwise operators {{{2
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _SimdWrapper<_Tp, _N>
      __logical_and(const _SimdWrapper<_Tp, _N>& __x,
		  const _SimdWrapper<_Tp, _N>& __y)
    {
      return __and(__x._M_data, __y._M_data);
    }

    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _SimdWrapper<_Tp, _N>
      __logical_or(const _SimdWrapper<_Tp, _N>& __x,
		 const _SimdWrapper<_Tp, _N>& __y)
    {
      return __or(__x._M_data, __y._M_data);
    }

    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _SimdWrapper<_Tp, _N>
      __bit_and(const _SimdWrapper<_Tp, _N>& __x,
	      const _SimdWrapper<_Tp, _N>& __y)
    {
      return __and(__x._M_data, __y._M_data);
    }

    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _SimdWrapper<_Tp, _N>
      __bit_or(const _SimdWrapper<_Tp, _N>& __x, const _SimdWrapper<_Tp, _N>& __y)
    {
      return __or(__x._M_data, __y._M_data);
    }

    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _SimdWrapper<_Tp, _N>
      __bit_xor(const _SimdWrapper<_Tp, _N>& __x,
	      const _SimdWrapper<_Tp, _N>& __y)
    {
      return __xor(__x._M_data, __y._M_data);
    }

    // smart_reference access {{{2
    template <class _Tp, size_t _N>
    static void __set(_SimdWrapper<_Tp, _N>& __k, int __i, bool __x) noexcept
    {
      if constexpr (std::is_same_v<_Tp, bool>)
	__k.__set(__i, __x);
      else
	__k._M_data[__i] = __bit_cast<_Tp>(__int_for_sizeof_t<_Tp>(-__x));
    }

    // __masked_assign{{{2
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static void __masked_assign(_SimdWrapper<_Tp, _N> __k, _SimdWrapper<_Tp, _N> &__lhs,
                                           __id<_SimdWrapper<_Tp, _N>> __rhs)
    {
        __lhs = __blend(__k._M_data, __lhs._M_data, __rhs._M_data);
    }

    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static void __masked_assign(_SimdWrapper<_Tp, _N> __k, _SimdWrapper<_Tp, _N> &__lhs, bool __rhs)
    {
        if (__builtin_constant_p(__rhs)) {
            if (__rhs == false) {
                __lhs = __andnot(__k._M_data, __lhs._M_data);
            } else {
                __lhs = __or(__k._M_data, __lhs._M_data);
            }
            return;
        }
        __lhs = __blend(__k, __lhs, __data(simd_mask<_Tp, _Abi>(__rhs)));
    }

    //}}}2
};

//}}}1

#if _GLIBCXX_SIMD_X86INTRIN // {{{
// __x86_simd_impl {{{1
template <class _Abi> struct __x86_simd_impl : _SimdImplBuiltin<_Abi> {
  using _Base = _SimdImplBuiltin<_Abi>;
  template <typename _Tp>
  using _MaskMember = typename _Base::template _MaskMember<_Tp>;

  // __masked_load {{{2
  template <class _Tp, size_t _N, class _U, class _F>
  static inline _SimdWrapper<_Tp, _N>
    __masked_load(_SimdWrapper<_Tp, _N> __merge,
		_MaskMember<_Tp>      __k,
		const _U*             __mem,
		_F) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
  {
    if constexpr (std::is_same_v<_Tp, _U> || // no conversion
		  (sizeof(_Tp) == sizeof(_U) &&
		   std::is_integral_v<_Tp> ==
		     std::is_integral_v<_U>) // conversion via bit
					     // reinterpretation
    )
      {
	[[maybe_unused]] const auto __intrin = __to_intrin(__merge);
	constexpr bool              __have_avx512bw_vl_or_zmm =
	  __have_avx512bw_vl || (__have_avx512bw && sizeof(__merge) == 64);
	if constexpr (__have_avx512bw_vl_or_zmm && sizeof(_Tp) == 1)
	  {
	    if constexpr (sizeof(__merge) == 16)
	      {
		__merge = __vector_bitcast<_Tp>(_mm_mask_loadu_epi8(
		  __intrin, _mm_movemask_epi8(__to_intrin(__k)), __mem));
	      }
	    else if constexpr (sizeof(__merge) == 32)
	      {
		__merge = __vector_bitcast<_Tp>(_mm256_mask_loadu_epi8(
		  __intrin, _mm256_movemask_epi8(__to_intrin(__k)), __mem));
	      }
	    else if constexpr (sizeof(__merge) == 64)
	      {
		__merge = __vector_bitcast<_Tp>(
		  _mm512_mask_loadu_epi8(__intrin, __k, __mem));
	      }
	    else
	      {
		__assert_unreachable<_Tp>();
	      }
	  }
	else if constexpr (__have_avx512bw_vl_or_zmm && sizeof(_Tp) == 2)
	  {
	    if constexpr (sizeof(__merge) == 16)
	      {
		__merge = __vector_bitcast<_Tp>(_mm_mask_loadu_epi16(
		  __intrin, movemask_epi16(__to_intrin(__k)), __mem));
	      }
	    else if constexpr (sizeof(__merge) == 32)
	      {
		__merge = __vector_bitcast<_Tp>(_mm256_mask_loadu_epi16(
		  __intrin, movemask_epi16(__to_intrin(__k)), __mem));
	      }
	    else if constexpr (sizeof(__merge) == 64)
	      {
		__merge = __vector_bitcast<_Tp>(
		  _mm512_mask_loadu_epi16(__intrin, __k, __mem));
	      }
	    else
	      {
		__assert_unreachable<_Tp>();
	      }
	  }
	else if constexpr (__have_avx2 && sizeof(_Tp) == 4 &&
			   std::is_integral_v<_U>)
	  {
	    if constexpr (sizeof(__merge) == 16)
	      {
		__merge =
		  (~__k._M_data & __merge._M_data) |
		  __vector_bitcast<_Tp>(_mm_maskload_epi32(
		    reinterpret_cast<const int*>(__mem), __to_intrin(__k)));
	      }
	    else if constexpr (sizeof(__merge) == 32)
	      {
		__merge =
		  (~__k._M_data & __merge._M_data) |
		  __vector_bitcast<_Tp>(_mm256_maskload_epi32(
		    reinterpret_cast<const int*>(__mem), __to_intrin(__k)));
	      }
	    else if constexpr (__have_avx512f && sizeof(__merge) == 64)
	      {
		__merge = __vector_bitcast<_Tp>(
		  _mm512_mask_loadu_epi32(__intrin, __k, __mem));
	      }
	    else
	      {
		__assert_unreachable<_Tp>();
	      }
	  }
	else if constexpr (__have_avx && sizeof(_Tp) == 4)
	  {
	    if constexpr (sizeof(__merge) == 16)
	      {
		__merge = __or(__andnot(__k._M_data, __merge._M_data),
			       __vector_bitcast<_Tp>(_mm_maskload_ps(
				 reinterpret_cast<const float*>(__mem),
				 __vector_bitcast<_LLong>(__k))));
	      }
	    else if constexpr (sizeof(__merge) == 32)
	      {
		__merge =
		  __or(__andnot(__k._M_data, __merge._M_data),
		       _mm256_maskload_ps(reinterpret_cast<const float*>(__mem),
					  __vector_bitcast<_LLong>(__k)));
	      }
	    else if constexpr (__have_avx512f && sizeof(__merge) == 64)
	      {
		__merge = __vector_bitcast<_Tp>(
		  _mm512_mask_loadu_ps(__intrin, __k, __mem));
	      }
	    else
	      {
		__assert_unreachable<_Tp>();
	      }
	  }
	else if constexpr (__have_avx2 && sizeof(_Tp) == 8 &&
			   std::is_integral_v<_U>)
	  {
	    if constexpr (sizeof(__merge) == 16)
	      {
		__merge =
		  (~__k._M_data & __merge._M_data) |
		  __vector_bitcast<_Tp>(_mm_maskload_epi64(
		    reinterpret_cast<const _LLong*>(__mem), __to_intrin(__k)));
	      }
	    else if constexpr (sizeof(__merge) == 32)
	      {
		__merge =
		  (~__k._M_data & __merge._M_data) |
		  __vector_bitcast<_Tp>(_mm256_maskload_epi64(
		    reinterpret_cast<const _LLong*>(__mem), __to_intrin(__k)));
	      }
	    else if constexpr (__have_avx512f && sizeof(__merge) == 64)
	      {
		__merge = __vector_bitcast<_Tp>(
		  _mm512_mask_loadu_epi64(__intrin, __k, __mem));
	      }
	    else
	      {
		__assert_unreachable<_Tp>();
	      }
	  }
	else if constexpr (__have_avx && sizeof(_Tp) == 8)
	  {
	    if constexpr (sizeof(__merge) == 16)
	      {
		__merge = __or(__andnot(__k._M_data, __merge._M_data),
			       __vector_bitcast<_Tp>(_mm_maskload_pd(
				 reinterpret_cast<const double*>(__mem),
				 __vector_bitcast<_LLong>(__k))));
	      }
	    else if constexpr (sizeof(__merge) == 32)
	      {
		__merge = __or(
		  __andnot(__k._M_data, __merge._M_data),
		  _mm256_maskload_pd(reinterpret_cast<const double*>(__mem),
				     __vector_bitcast<_LLong>(__k)));
	      }
	    else if constexpr (__have_avx512f && sizeof(__merge) == 64)
	      {
		__merge = __vector_bitcast<_Tp>(
		  _mm512_mask_loadu_pd(__intrin, __k, __mem));
	      }
	    else
	      {
		__assert_unreachable<_Tp>();
	      }
	  }
	else
	  {
	    __bit_iteration(__vector_to_bitset(__k._M_data).to_ullong(),
			    [&](auto __i) {
			      __merge.__set(__i, static_cast<_Tp>(__mem[__i]));
			    });
	  }
      }
    else if constexpr (sizeof(_U) <= 8 && // no long double
		       !__converts_via_decomposition_v<
			 _U, _Tp,
			 sizeof(__merge)> // conversion via decomposition
					  // is better handled via the
					  // bit_iteration fallback below
    )
      {
	// TODO: copy pattern from __masked_store, which doesn't resort to
	// fixed_size
	using _A = simd_abi::deduce_t<
	  _U, std::max(_N, 16 / sizeof(_U)) // _N or more, so that at least a 16
					    // Byte vector is used instead of a
					    // fixed_size filled with scalars
	  >;
	using _ATraits = _SimdTraits<_U, _A>;
	using _AImpl   = typename _ATraits::_SimdImpl;
	typename _ATraits::_SimdMember __uncvted{};
	typename _ATraits::_MaskMember __kk;
	if constexpr (__is_fixed_size_abi_v<_A>)
	  {
	    __kk = __vector_to_bitset(__k._M_data);
	  }
	else
	  {
	    __kk = __convert_mask<typename _ATraits::_MaskMember>(__k);
	  }
	__uncvted = _AImpl::__masked_load(__uncvted, __kk, __mem, _F());
	_SimdConverter<_U, _A, _Tp, _Abi> __converter;
        _Base::__masked_assign(__k, __merge, __converter(__uncvted));
      }
    else
      {
	__bit_iteration(
	  __vector_to_bitset(__k._M_data).to_ullong(),
	  [&](auto __i) { __merge.__set(__i, static_cast<_Tp>(__mem[__i])); });
      }
    return __merge;
    }

    // __masked_store {{{2
    template <class _Tp, size_t _N, class _U, class _F>
    static inline void __masked_store(const _SimdWrapper<_Tp, _N> __v, _U *__mem, _F,
                                    const _MaskMember<_Tp> __k) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
      if constexpr (std::is_integral_v<_Tp> && std::is_integral_v<_U> &&
		    sizeof(_Tp) > sizeof(_U) && __have_avx512f &&
		    (sizeof(_Tp) >= 4 || __have_avx512bw) &&
		    (sizeof(__v) == 64 || __have_avx512vl)) {  // truncating store
	[[maybe_unused]] const auto __vi = __to_intrin(__v);
	const auto __kk = [&]() {
	  if constexpr (__is_bitmask_v<decltype(__k)>) {
	    return __k;
	  } else {
	    return __convert_mask<_SimdWrapper<bool, _N>>(__k);
	  }
	}();
	if constexpr (sizeof(_Tp) == 8 && sizeof(_U) == 4) {
	  if constexpr (sizeof(__vi) == 64) {
	    _mm512_mask_cvtepi64_storeu_epi32(__mem, __kk, __vi);
	  } else if constexpr (sizeof(__vi) == 32) {
	    _mm256_mask_cvtepi64_storeu_epi32(__mem, __kk, __vi);
	  } else if constexpr (sizeof(__vi) == 16) {
	    _mm_mask_cvtepi64_storeu_epi32(__mem, __kk, __vi);
	  }
	} else if constexpr (sizeof(_Tp) == 8 && sizeof(_U) == 2) {
	  if constexpr (sizeof(__vi) == 64) {
	    _mm512_mask_cvtepi64_storeu_epi16(__mem, __kk, __vi);
	  } else if constexpr (sizeof(__vi) == 32) {
	    _mm256_mask_cvtepi64_storeu_epi16(__mem, __kk, __vi);
	  } else if constexpr (sizeof(__vi) == 16) {
	    _mm_mask_cvtepi64_storeu_epi16(__mem, __kk, __vi);
	  }
	} else if constexpr (sizeof(_Tp) == 8 && sizeof(_U) == 1) {
	  if constexpr (sizeof(__vi) == 64) {
	    _mm512_mask_cvtepi64_storeu_epi8(__mem, __kk, __vi);
	  } else if constexpr (sizeof(__vi) == 32) {
	    _mm256_mask_cvtepi64_storeu_epi8(__mem, __kk, __vi);
	  } else if constexpr (sizeof(__vi) == 16) {
	    _mm_mask_cvtepi64_storeu_epi8(__mem, __kk, __vi);
	  }
	} else if constexpr (sizeof(_Tp) == 4 && sizeof(_U) == 2) {
	  if constexpr (sizeof(__vi) == 64) {
	    _mm512_mask_cvtepi32_storeu_epi16(__mem, __kk, __vi);
	  } else if constexpr (sizeof(__vi) == 32) {
	    _mm256_mask_cvtepi32_storeu_epi16(__mem, __kk, __vi);
	  } else if constexpr (sizeof(__vi) == 16) {
	    _mm_mask_cvtepi32_storeu_epi16(__mem, __kk, __vi);
	  }
	} else if constexpr (sizeof(_Tp) == 4 && sizeof(_U) == 1) {
	  if constexpr (sizeof(__vi) == 64) {
	    _mm512_mask_cvtepi32_storeu_epi8(__mem, __kk, __vi);
	  } else if constexpr (sizeof(__vi) == 32) {
	    _mm256_mask_cvtepi32_storeu_epi8(__mem, __kk, __vi);
	  } else if constexpr (sizeof(__vi) == 16) {
	    _mm_mask_cvtepi32_storeu_epi8(__mem, __kk, __vi);
	  }
	} else if constexpr (sizeof(_Tp) == 2 && sizeof(_U) == 1) {
	  if constexpr (sizeof(__vi) == 64) {
	    _mm512_mask_cvtepi16_storeu_epi8(__mem, __kk, __vi);
	  } else if constexpr (sizeof(__vi) == 32) {
	    _mm256_mask_cvtepi16_storeu_epi8(__mem, __kk, __vi);
	  } else if constexpr (sizeof(__vi) == 16) {
	    _mm_mask_cvtepi16_storeu_epi8(__mem, __kk, __vi);
	  }
	} else {
	  __assert_unreachable<_Tp>();
	}
      } else {
	_Base::__masked_store(__v,__mem,_F(),__k);
      }
    }

    // __multiplies {{{2
    template <typename _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _SimdWrapper<_Tp, _N>
      __multiplies(_SimdWrapper<_Tp, _N> __x, _SimdWrapper<_Tp, _N> __y)
    {
      if (__builtin_is_constant_evaluated())
	return __x._M_data * __y._M_data;
      else if constexpr (sizeof(_Tp) == 1)
	{
	  // codegen of `x*y` is suboptimal (as of GCC 9.0.1)
	  const auto __high_byte = __vector_broadcast<_N / 2, short>(-256);
	  const auto __even =
	    __vector_bitcast<short>(__x) * __vector_bitcast<short>(__y);
	  const auto __odd = (__vector_bitcast<short>(__x) >> 8) *
			     (__vector_bitcast<short>(__y) & __high_byte);
	  if constexpr (__have_avx512bw)
	    return __blend(0xaaaa'aaaa'aaaa'aaaaLL,
			   __vector_bitcast<_Tp>(__even),
			   __vector_bitcast<_Tp>(__odd));
	  else if constexpr (__have_sse4_1)
	    return __vector_bitcast<_Tp>(__blend(__high_byte, __even, __odd));
	  else
	    return __vector_bitcast<_Tp>(__andnot(__high_byte, __even) | __odd);
	}
      else
	return _Base::__multiplies(__x, __y);
    }

    // __divides {{{2
    template <typename _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _SimdWrapper<_Tp, _N>
      __divides(_SimdWrapper<_Tp, _N> __x, _SimdWrapper<_Tp, _N> __y)
    {
      if (__builtin_is_constant_evaluated())
	return __x._M_data / __y._M_data;
      else if constexpr (is_integral_v<_Tp> && sizeof(_Tp) <= 4)
	{ // use divps - codegen of `x/y` is suboptimal (as of GCC 9.0.1)
	  using _FloatV = typename __vector_type<conditional_t<sizeof(_Tp) ==4, double, float>,
	    __have_avx512f ? 64 : __have_avx ? 32 : 16>::type;
	  const auto __xf = __convert_all<_FloatV>(__x);
	  const auto __yf = __convert_all<_FloatV>(__y);
	  using _R = __vector_type_t<_Tp, _N>;
	  if constexpr(__is_vector_type_v<remove_const_t<decltype(__xf)>>)
	    return __vector_convert<_R>(__xf / __yf);
	  else if constexpr(__xf.size() == 2)
	    return __vector_convert<_R>(__xf[0] / __yf[0], __xf[1] / __yf[1]);
	  else if constexpr(__xf.size() == 4)
	    return __vector_convert<_R>(__xf[0] / __yf[0], __xf[1] / __yf[1],
					__xf[2] / __yf[2], __xf[3] / __yf[3]);
	  else
	    __assert_unreachable<_Tp>();
	}
      /* 64-bit int division is potentially optimizable via double division if
       * the value in __x is small enough and the conversion between
       * int<->double is efficient enough:
      else if constexpr (is_integral_v<_Tp> && is_unsigned_v<_Tp> &&
			 sizeof(_Tp) == 8)
	{
	  if constexpr (__have_sse4_1 && sizeof(__x) == 16)
	    {
	      if (_mm_test_all_zeros(__x, __m128i{0xffe0'0000'0000'0000ull,
						  0xffe0'0000'0000'0000ull}))
		{
		  __x._M_data | 0x __vector_convert<__m128d>(__x._M_data)
		}
	    }
	}
        */
      else
	return _Base::__divides(__x, __y);
    }

    // __bit_shift_left {{{2
    // Notes on UB. C++2a [expr.shift] says:
    // -1- [...] The operands shall be of integral or unscoped enumeration type
    //     and integral promotions are performed. The type of the result is that
    //     of the promoted left operand. The behavior is undefined if the right
    //     operand is negative, or greater than or equal to the width of the
    //     promoted left operand.
    // -2- The value of E1 << E2 is the unique value congruent to E1×2^E2 modulo
    //     2^N, where N is the width of the type of the result.
    //
    // C++17 [expr.shift] says:
    // -2- The value of E1 << E2 is E1 left-shifted E2 bit positions; vacated
    //     bits are zero-filled. If E1 has an unsigned type, the value of the
    //     result is E1 × 2^E2 , reduced modulo one more than the maximum value
    //     representable in the result type. Otherwise, if E1 has a signed type
    //     and non-negative value, and E1 × 2^E2 is representable in the
    //     corresponding unsigned type of the result type, then that value,
    //     converted to the result type, is the resulting value; otherwise, the
    //     behavior is undefined.
    //
    // Consequences:
    // With C++2a signed and unsigned types have the same UB
    // characteristics:
    // - left shift is not UB for 0 <= RHS < max(32, #bits(T))
    //
    // With C++17 there's little room for optimizations because the standard
    // requires all shifts to happen on promoted integrals (i.e. int). Thus,
    // short and char shifts must assume shifts affect bits of neighboring
    // values.
#ifndef _GLIBCXX_SIMD_NO_SHIFT_OPT
    template <typename _Tp, typename _TVT = _VectorTraits<_Tp>>
    inline _GLIBCXX_SIMD_CONST static typename _TVT::type
      __bit_shift_left(_Tp __xx, int __y)
    {
      using _V = typename _TVT::type;
      using _U = typename _TVT::value_type;
      _V __x   = __xx;
      [[maybe_unused]] const auto __ix = __to_intrin(__x);
      if (__builtin_is_constant_evaluated())
	return __x << __y;
#if __cplusplus > 201703
      // after C++17, signed shifts have no UB, and behave just like unsigned
      // shifts
      else if constexpr (sizeof(_U) == 1 && is_signed_v<_U>)
	return __vector_bitcast<_U>(
	  __bit_shift_left(__vector_bitcast<make_unsigned_t<_U>>(__x), __y));
#endif
      else if constexpr (sizeof(_U) == 1)
	{
	  // (cf. https://gcc.gnu.org/bugzilla/show_bug.cgi?id=83894)
	  if (__builtin_constant_p(__y))
	    {
	      if (__y == 0)
		return __x;
	      else if (__y == 1)
		return __x + __x;
	      else if (__y == 2)
		{
		  __x = __x + __x;
		  return __x + __x;
		}
	      else if (__y > 2 && __y < 8)
		{
		  const _UChar __mask = 0xff << __y; // precomputed vector
		  return __vector_bitcast<_U>(
		    __vector_bitcast<_UChar>(__vector_bitcast<unsigned>(__x)
					     << __y) &
		    __mask);
		}
	      else if (__y >= 8 && __y < 32)
		return _V();
	      else
		__builtin_unreachable();
	    }
	  // general strategy in the following: use an sllv instead of sll
	  // instruction, because it's 2 to 4 times faster:
	  else if constexpr (__have_avx512bw_vl && sizeof(__x) == 16)
	    return __vector_bitcast<_U>(_mm256_cvtepi16_epi8(_mm256_sllv_epi16(
	      _mm256_cvtepi8_epi16(__ix), _mm256_set1_epi16(__y))));
	  else if constexpr (__have_avx512bw && sizeof(__x) == 32)
	    return __vector_bitcast<_U>(_mm512_cvtepi16_epi8(_mm512_sllv_epi16(
	      _mm512_cvtepi8_epi16(__ix), _mm512_set1_epi16(__y))));
	  else if constexpr (__have_avx512bw && sizeof(__x) == 64)
	    {
	      const auto __shift = _mm512_set1_epi16(__y);
	      return __vector_bitcast<_U>(
		__concat(_mm512_cvtepi16_epi8(_mm512_sllv_epi16(
			   _mm512_cvtepi8_epi16(__lo256(__ix)), __shift)),
			 _mm512_cvtepi16_epi8(_mm512_sllv_epi16(
			   _mm512_cvtepi8_epi16(__hi256(__ix)), __shift))));
	    }
	  else if constexpr (__have_avx2 && sizeof(__x) == 32)
	    {
#if 1
	      const auto __shift = _mm_cvtsi32_si128(__y);
	      auto __k = _mm256_sll_epi16(_mm256_slli_epi16(~__m256i(), 8), __shift);
	      __k |= _mm256_srli_epi16(__k, 8);
	      return __vector_bitcast<_U>(_mm256_sll_epi32(__ix, __shift) &
					  __k);
#else
	      const _U __k = 0xff << __y;
	      return __vector_bitcast<_U>(__vector_bitcast<int>(__x) << __y) &
		     __k;
#endif
	    }
	  else
	    {
	      const auto __shift = _mm_cvtsi32_si128(__y);
	      auto __k = _mm_sll_epi16(_mm_slli_epi16(~__m128i(), 8), __shift);
	      __k |= _mm_srli_epi16(__k, 8);
	      return __vector_bitcast<_U>(_mm_sll_epi16(__ix, __shift) & __k);
	    }
	}
      return __x << __y;
    }

    template <typename _Tp, typename _TVT = _VectorTraits<_Tp>>
    inline _GLIBCXX_SIMD_CONST static typename _TVT::type
      __bit_shift_left(_Tp __xx, typename _TVT::type __y)
    {
      using _V                         = typename _TVT::type;
      using _U                         = typename _TVT::value_type;
      _V                          __x  = __xx;
      [[maybe_unused]] const auto __ix = __to_intrin(__x);
      [[maybe_unused]] const auto __iy = __to_intrin(__y);
      if (__builtin_is_constant_evaluated())
	return __x << __y;
#if __cplusplus > 201703
      // after C++17, signed shifts have no UB, and behave just like unsigned
      // shifts
      else if constexpr (is_signed_v<_U>)
	return __vector_bitcast<_U>(
	  __bit_shift_left(__vector_bitcast<make_unsigned_t<_U>>(__x),
			   __vector_bitcast<make_unsigned_t<_U>>(__y)));
#endif
      else if constexpr (sizeof(_U) == 1)
	{
	  if constexpr (sizeof __ix == 64 && __have_avx512bw)
	    return __vector_bitcast<_U>(__concat(
	      _mm512_cvtepi16_epi8(
		_mm512_sllv_epi16(_mm512_cvtepu8_epi16(__lo256(__ix)),
				  _mm512_cvtepu8_epi16(__lo256(__iy)))),
	      _mm512_cvtepi16_epi8(
		_mm512_sllv_epi16(_mm512_cvtepu8_epi16(__hi256(__ix)),
				  _mm512_cvtepu8_epi16(__hi256(__iy))))));
	  else if constexpr (sizeof __ix == 32 && __have_avx512bw)
	    return __vector_bitcast<_U>(_mm512_cvtepi16_epi8(_mm512_sllv_epi16(
	      _mm512_cvtepu8_epi16(__ix), _mm512_cvtepu8_epi16(__iy))));
	  else if constexpr (sizeof __ix == 16 && __have_avx512bw_vl)
	    return __vector_bitcast<_U>(_mm256_cvtepi16_epi8(_mm256_sllv_epi16(
	      _mm256_cvtepu8_epi16(__ix), _mm256_cvtepu8_epi16(__iy))));
	  else if constexpr (sizeof __ix == 16 && __have_avx512bw)
	    return __vector_bitcast<_U>(
	      __lo128(_mm512_cvtepi16_epi8(_mm512_sllv_epi16(
		_mm512_cvtepu8_epi16(_mm256_castsi128_si256(__ix)),
		_mm512_cvtepu8_epi16(_mm256_castsi128_si256(__iy))))));
	  else if constexpr (__have_sse4_1)
	    {
              auto __mask = __vector_bitcast<_U>(__vector_bitcast<short>(__y) << 5);
	      auto __x4   = __vector_bitcast<_U>(__vector_bitcast<short>(__x) << 4);
	      __x4 &= char(0xf0);
              __x = __blend(__mask, __x, __x4);
              __mask += __mask;
	      auto __x2   = __vector_bitcast<_U>(__vector_bitcast<short>(__x) << 2);
	      __x2 &= char(0xfc);
              __x = __blend(__mask, __x, __x2);
              __mask += __mask;
	      auto __x1   = __x + __x;
              __x = __blend(__mask, __x, __x1);
	      return __x & ((__y & char(0xf8)) == 0); // y > 7 nulls the result
	    }
          else
	    {
              auto __mask = __vector_bitcast<_UChar>(__vector_bitcast<short>(__y) << 5);
	      auto __x4   = __vector_bitcast<_U>(__vector_bitcast<short>(__x) << 4);
	      __x4 &= char(0xf0);
              __x = __blend(__vector_bitcast<_SChar>(__mask) < 0, __x, __x4);
              __mask += __mask;
	      auto __x2   = __vector_bitcast<_U>(__vector_bitcast<short>(__x) << 2);
	      __x2 &= char(0xfc);
              __x = __blend(__vector_bitcast<_SChar>(__mask) < 0, __x, __x2);
              __mask += __mask;
	      auto __x1   = __x + __x;
              __x = __blend(__vector_bitcast<_SChar>(__mask) < 0, __x, __x1);
	      return __x & ((__y & char(0xf8)) == 0); // y > 7 nulls the result
	    }
	}
      else if constexpr (sizeof(_U) == 2)
	{
	  if constexpr (sizeof __ix == 64 && __have_avx512bw)
	    return __vector_bitcast<_U>(_mm512_sllv_epi16(__ix, __iy));
	  else if constexpr (sizeof __ix == 32 && __have_avx512bw_vl)
	    return __vector_bitcast<_U>(_mm256_sllv_epi16(__ix, __iy));
	  else if constexpr (sizeof __ix == 32 && __have_avx512bw)
	    return __vector_bitcast<_U>(__lo256(_mm512_sllv_epi16(
	      _mm512_castsi256_si512(__ix), _mm512_castsi256_si512(__iy))));
	  else if constexpr (sizeof __ix == 32 && __have_avx2)
	    {
	      const auto __ux = __vector_bitcast<unsigned>(__x);
	      const auto __uy = __vector_bitcast<unsigned>(__y);
	      return __vector_bitcast<_U>(_mm256_blend_epi16(
		__auto_bitcast(__ux << (__uy & 0x0000ffffu)),
		__auto_bitcast((__ux & 0xffff0000u) << (__uy >> 16)), 0xaa));
	    }
	  else if constexpr (sizeof __ix == 16 && __have_avx512bw_vl)
	    return __vector_bitcast<_U>(_mm_sllv_epi16(__ix, __iy));
	  else if constexpr (sizeof __ix == 16 && __have_avx512bw)
	    return __vector_bitcast<_U>(__lo128(_mm512_sllv_epi16(
	      _mm512_castsi128_si512(__ix), _mm512_castsi128_si512(__iy))));
	  else if constexpr (sizeof __ix == 16 && __have_avx2)
	    {
	      const auto __ux = __vector_bitcast<unsigned>(__x);
	      const auto __uy = __vector_bitcast<unsigned>(__y);
	      return __vector_bitcast<_U>(_mm_blend_epi16(
		__auto_bitcast(__ux << (__uy & 0x0000ffffu)),
		__auto_bitcast((__ux & 0xffff0000u) << (__uy >> 16)), 0xaa));
	    }
	  else if constexpr (sizeof __ix == 16)
	    {
	      __y += 0x3f8 >> 3;
	      return __x *
		     __vector_bitcast<_U>(
		       __vector_convert<__vector_type16_t<int>>(
			 __vector_bitcast<float>(__vector_bitcast<unsigned>(__y)
						 << 23)) |
		       (__vector_convert<__vector_type16_t<int>>(
			  __vector_bitcast<float>(
			    (__vector_bitcast<unsigned>(__y) >> 16) << 23))
			<< 16));
	    }
	  else
	    __assert_unreachable<_Tp>();
	}
      else if constexpr (sizeof(_U) == 4 && sizeof __ix == 16 && !__have_avx2)
        // latency is suboptimal, but throughput is at full speedup
	return __vector_bitcast<_U>(
	  __vector_bitcast<unsigned>(__x) *
	  __vector_convert<__vector_type16_t<int>>(
	    __vector_bitcast<float>((__y << 23) + 0x3f80'0000)));
      else if constexpr (sizeof(_U) == 8 && sizeof __ix == 16 && !__have_avx2)
	{
	  const auto __lo = _mm_sll_epi64(__ix, __iy);
	  const auto __hi = _mm_sll_epi64(__ix, _mm_unpackhi_epi64(__iy, __iy));
	  if constexpr (__have_sse4_1)
	    return __vector_bitcast<_U>(_mm_blend_epi16(__lo, __hi, 0xf0));
	  else
	    return __vector_bitcast<_U>(_mm_move_sd(
	      __vector_bitcast<double>(__hi), __vector_bitcast<double>(__lo)));
	}
      else
	return __x << __y;
    }
#endif // _GLIBCXX_SIMD_NO_SHIFT_OPT

    // __bit_shift_right {{{2
#ifndef _GLIBCXX_SIMD_NO_SHIFT_OPT
    template <typename _Tp, typename _TVT = _VectorTraits<_Tp>>
    inline _GLIBCXX_SIMD_CONST static typename _TVT::type
      __bit_shift_right(_Tp __xx, int __y)
    {
      using _V = typename _TVT::type;
      using _U = typename _TVT::value_type;
      _V __x   = __xx;
      [[maybe_unused]] const auto __ix = __to_intrin(__x);
      if (__builtin_is_constant_evaluated())
	return __x >> __y;
      else if constexpr (sizeof(_U) == 1 && is_unsigned_v<_U>) //{{{
	return __vector_bitcast<_U>(__vector_bitcast<_UShort>(__x) >> __y) &
	       _U(0xff >> __y);
      //}}}
      else if constexpr (sizeof(_U) == 1 && is_signed_v<_U>) //{{{
	return __vector_bitcast<_U>(
	  (__vector_bitcast<_UShort>(__vector_bitcast<short>(__x) >> (__y + 8))
	   << 8) |
	  (__vector_bitcast<_UShort>(
	     __vector_bitcast<short>(__vector_bitcast<_UShort>(__x) << 8) >>
	     __y) >>
	   8));
      //}}}
      // GCC optimizes sizeof == 2, 4, and unsigned 8 as expected
      else if constexpr (sizeof(_U) == 8 && is_signed_v<_U>) //{{{
      {
	if (__y > 32)
	  return (__vector_bitcast<_U>(__vector_bitcast<int>(__x) >> 32) &
		  _U(0xffff'ffff'0000'0000ull)) |
		 __vector_bitcast<_U>(__vector_bitcast<int>(
					__vector_bitcast<_ULLong>(__x) >> 32) >>
				      (__y - 32));
	else
	  return __vector_bitcast<_U>(__vector_bitcast<_ULLong>(__x) >> __y) |
		 __vector_bitcast<_U>(
		   __vector_bitcast<int>(__x & -0x8000'0000'0000'0000ll) >>
		   __y);
      }
      //}}}
      else
	return __x >> __y;
    }

    template <typename _Tp, typename _TVT = _VectorTraits<_Tp>>
    inline _GLIBCXX_SIMD_CONST static typename _TVT::type
      __bit_shift_right(_Tp __xx, typename _TVT::type __y)
    {
      using _V                         = typename _TVT::type;
      using _U                         = typename _TVT::value_type;
      _V                          __x  = __xx;
      [[maybe_unused]] const auto __ix = __to_intrin(__x);
      [[maybe_unused]] const auto __iy = __to_intrin(__y);
      if (__builtin_is_constant_evaluated())
	return __x >> __y;
      else if constexpr (sizeof(_U) == 1) //{{{
	{
	  if constexpr (sizeof(__x) == 16 && __have_avx512bw_vl)
	    return __vector_bitcast<_U>(_mm256_cvtepi16_epi8(
	      is_signed_v<_U> ? _mm256_srav_epi16(_mm256_cvtepi8_epi16(__ix),
						  _mm256_cvtepi8_epi16(__iy))
			      : _mm256_srlv_epi16(_mm256_cvtepu8_epi16(__ix),
						  _mm256_cvtepu8_epi16(__iy))));
	  else if constexpr (sizeof(__x) == 32 && __have_avx512bw)
	    return __vector_bitcast<_U>(_mm512_cvtepi16_epi8(
	      is_signed_v<_U> ? _mm512_srav_epi16(_mm512_cvtepi8_epi16(__ix),
						  _mm512_cvtepi8_epi16(__iy))
			      : _mm512_srlv_epi16(_mm512_cvtepu8_epi16(__ix),
						  _mm512_cvtepu8_epi16(__iy))));
	  else if constexpr (sizeof(__x) == 64 && is_signed_v<_U>)
	    return __vector_bitcast<_U>(_mm512_mask_mov_epi8(
	      _mm512_srav_epi16(__ix, _mm512_srli_epi16(__iy, 8)),
	      0x5555'5555'5555'5555ull,
	      _mm512_srav_epi16(_mm512_slli_epi16(__ix, 8),
				_mm512_maskz_add_epi8(0x5555'5555'5555'5555ull,
						      __iy,
						      _mm512_set1_epi16(8)))));
	  else if constexpr (sizeof(__x) == 64 && is_unsigned_v<_U>)
	    return __vector_bitcast<_U>(_mm512_mask_mov_epi8(
	      _mm512_srlv_epi16(__ix, _mm512_srli_epi16(__iy, 8)),
	      0x5555'5555'5555'5555ull,
	      _mm512_srlv_epi16(
		_mm512_maskz_mov_epi8(0x5555'5555'5555'5555ull, __ix),
		_mm512_maskz_mov_epi8(0x5555'5555'5555'5555ull, __iy))));
	  /* This has better throughput but higher latency than the impl below
	  else if constexpr (__have_avx2 && sizeof(__x) == 16 &&
			     is_unsigned_v<_U>)
	    {
	      const auto __shorts = __to_intrin(__bit_shift_right(
		__vector_bitcast<_UShort>(_mm256_cvtepu8_epi16(__ix)),
		__vector_bitcast<_UShort>(_mm256_cvtepu8_epi16(__iy))));
	      return __vector_bitcast<_U>(
		_mm_packus_epi16(__lo128(__shorts), __hi128(__shorts)));
	    }
	    */
	  else if constexpr (__have_avx2)
	    // the following uses vpsr[al]vd, which requires AVX2
	    if constexpr (is_signed_v<_U>)
	      {
		const auto r3 = __vector_bitcast<_UInt>(
				  (__vector_bitcast<int>(__x) >>
				   (__vector_bitcast<_UInt>(__y) >> 24))) &
				0xff000000u;
		const auto r2 =
		  __vector_bitcast<_UInt>(
		    ((__vector_bitcast<int>(__x) << 8) >>
		     ((__vector_bitcast<_UInt>(__y) << 8) >> 24))) &
		  0xff000000u;
		const auto r1 =
		  __vector_bitcast<_UInt>(
		    ((__vector_bitcast<int>(__x) << 16) >>
		     ((__vector_bitcast<_UInt>(__y) << 16) >> 24))) &
		  0xff000000u;
		const auto r0 = __vector_bitcast<_UInt>(
		  (__vector_bitcast<int>(__x) << 24) >>
		  ((__vector_bitcast<_UInt>(__y) << 24) >> 24));
		return __vector_bitcast<_U>(r3 | (r2 >> 8) | (r1 >> 16) |
					    (r0 >> 24));
	      }
	    else
	      {
		const auto r3 = (__vector_bitcast<_UInt>(__x) >>
				 (__vector_bitcast<_UInt>(__y) >> 24)) &
				0xff000000u;
		const auto r2 = ((__vector_bitcast<_UInt>(__x) << 8) >>
				 ((__vector_bitcast<_UInt>(__y) << 8) >> 24)) &
				0xff000000u;
		const auto r1 = ((__vector_bitcast<_UInt>(__x) << 16) >>
				 ((__vector_bitcast<_UInt>(__y) << 16) >> 24)) &
				0xff000000u;
		const auto r0 = (__vector_bitcast<_UInt>(__x) << 24) >>
				((__vector_bitcast<_UInt>(__y) << 24) >> 24);
		return __vector_bitcast<_U>(r3 | (r2 >> 8) | (r1 >> 16) |
					    (r0 >> 24));
	      }
	  else if constexpr (__have_sse4_1 && is_unsigned_v<_U>)
	    {
	      auto __mask =
		__vector_bitcast<_U>(__vector_bitcast<_UShort>(__y) << 5);
	      auto __x4 = __vector_bitcast<_U>(
		(__vector_bitcast<_UShort>(__x) >> 4) & _UShort(0xff0f));
	      __x = __blend(__mask, __x, __x4);
	      __mask += __mask;
	      auto __x2 = __vector_bitcast<_U>(
		(__vector_bitcast<_UShort>(__x) >> 2) & _UShort(0xff3f));
	      __x = __blend(__mask, __x, __x2);
	      __mask += __mask;
	      auto __x1 = __vector_bitcast<_U>(
		(__vector_bitcast<_UShort>(__x) >> 1) & _UShort(0xff7f));
	      __x = __blend(__mask, __x, __x1);
	      return __x & ((__y & char(0xf8)) == 0); // y > 7 nulls the result
	    }
	  else if constexpr (__have_sse4_1 && is_signed_v<_U>)
	    {
	      auto __mask =
		__vector_bitcast<_UChar>(__vector_bitcast<_UShort>(__y) << 5);
	      auto __maskl = [&]() {
		return __vector_bitcast<_UShort>(__mask) << 8;
	      };
	      auto __xh  = __vector_bitcast<short>(__x);
	      auto __xl  = __vector_bitcast<short>(__x) << 8;
	      auto __xh4 = __xh >> 4;
	      auto __xl4 = __xl >> 4;
	      __xh       = __blend(__mask, __xh, __xh4);
	      __xl       = __blend(__maskl(), __xl, __xl4);
	      __mask += __mask;
	      auto __xh2 = __xh >> 2;
	      auto __xl2 = __xl >> 2;
	      __xh       = __blend(__mask, __xh, __xh2);
	      __xl       = __blend(__maskl(), __xl, __xl2);
	      __mask += __mask;
	      auto __xh1 = __xh >> 1;
	      auto __xl1 = __xl >> 1;
	      __xh       = __blend(__mask, __xh, __xh1);
	      __xl       = __blend(__maskl(), __xl, __xl1);
	      __x        = __vector_bitcast<_U>((__xh & short(0xff00))) |
		    __vector_bitcast<_U>(__vector_bitcast<_UShort>(__xl) >> 8);
	      return __x & ((__y & char(0xf8)) == 0); // y > 7 nulls the result
	    }
	  else if constexpr (is_unsigned_v<_U>) // SSE2
	    {
	      auto __mask =
		__vector_bitcast<_U>(__vector_bitcast<_UShort>(__y) << 5);
	      auto __x4 = __vector_bitcast<_U>(
		(__vector_bitcast<_UShort>(__x) >> 4) & _UShort(0xff0f));
	      __x = __blend(__mask > 0x7f, __x, __x4);
	      __mask += __mask;
	      auto __x2 = __vector_bitcast<_U>(
		(__vector_bitcast<_UShort>(__x) >> 2) & _UShort(0xff3f));
	      __x = __blend(__mask > 0x7f, __x, __x2);
	      __mask += __mask;
	      auto __x1 = __vector_bitcast<_U>(
		(__vector_bitcast<_UShort>(__x) >> 1) & _UShort(0xff7f));
	      __x = __blend(__mask > 0x7f, __x, __x1);
	      return __x & ((__y & char(0xf8)) == 0); // y > 7 nulls the result
	    }
	  else // signed SSE2
	    {
	      static_assert(is_signed_v<_U>);
	      auto __maskh = __vector_bitcast<_UShort>(__y) << 5;
	      auto __maskl = __vector_bitcast<_UShort>(__y) << (5 + 8);
	      auto __xh  = __vector_bitcast<short>(__x);
	      auto __xl  = __vector_bitcast<short>(__x) << 8;
	      auto __xh4 = __xh >> 4;
	      auto __xl4 = __xl >> 4;
	      __xh       = __blend(__maskh > 0x7fff, __xh, __xh4);
	      __xl       = __blend(__maskl > 0x7fff, __xl, __xl4);
	      __maskh += __maskh;
	      __maskl += __maskl;
	      auto __xh2 = __xh >> 2;
	      auto __xl2 = __xl >> 2;
	      __xh       = __blend(__maskh > 0x7fff, __xh, __xh2);
	      __xl       = __blend(__maskl > 0x7fff, __xl, __xl2);
	      __maskh += __maskh;
	      __maskl += __maskl;
	      auto __xh1 = __xh >> 1;
	      auto __xl1 = __xl >> 1;
	      __xh       = __blend(__maskh > 0x7fff, __xh, __xh1);
	      __xl       = __blend(__maskl > 0x7fff, __xl, __xl1);
	      __x        = __vector_bitcast<_U>((__xh & short(0xff00))) |
		    __vector_bitcast<_U>(__vector_bitcast<_UShort>(__xl) >> 8);
	      return __x & ((__y & char(0xf8)) == 0); // y > 7 nulls the result
	    }
	} //}}}
      else if constexpr (sizeof(_U) == 2) //{{{
	{
	  [[maybe_unused]] auto __blend_0xaa =
	    [](auto __a, auto __b) {
	      if constexpr (sizeof(__a) == 16)
		return _mm_blend_epi16(__to_intrin(__a), __to_intrin(__b),
				       0xaa);
	      else if constexpr (sizeof(__a) == 32)
		return _mm256_blend_epi16(__to_intrin(__a), __to_intrin(__b),
					  0xaa);
	      else if constexpr (sizeof(__a) == 64)
		return _mm512_mask_blend_epi16(0xaaaa'aaaaU, __to_intrin(__a),
					       __to_intrin(__b));
	      else
		__assert_unreachable<decltype(__a)>();
	    };
	  if constexpr (__have_avx512bw_vl && sizeof(_Tp) == 16)
	    return __vector_bitcast<_U>(is_signed_v<_U>
					  ? _mm_srav_epi16(__ix, __iy)
					  : _mm_srlv_epi16(__ix, __iy));
	  else if constexpr (__have_avx512bw_vl && sizeof(_Tp) == 32)
	    return __vector_bitcast<_U>(is_signed_v<_U>
					  ? _mm256_srav_epi16(__ix, __iy)
					  : _mm256_srlv_epi16(__ix, __iy));
	  else if constexpr (__have_avx512bw && sizeof(_Tp) == 64)
	    return __vector_bitcast<_U>(is_signed_v<_U>
					  ? _mm512_srav_epi16(__ix, __iy)
					  : _mm512_srlv_epi16(__ix, __iy));
	  else if constexpr (__have_avx2 && is_signed_v<_U>)
	    return __vector_bitcast<_U>(
	      __blend_0xaa(((__vector_bitcast<int>(__x) << 16) >>
			     (__vector_bitcast<int>(__y) & 0xffffu)) >> 16,
			   __vector_bitcast<int>(__x) >>
			     (__vector_bitcast<int>(__y) >> 16)));
	  else if constexpr (__have_avx2 && is_unsigned_v<_U>)
	    return __vector_bitcast<_U>(
	      __blend_0xaa((__vector_bitcast<_UInt>(__x) & 0xffffu) >>
			     (__vector_bitcast<_UInt>(__y) & 0xffffu),
			   __vector_bitcast<_UInt>(__x) >>
			     (__vector_bitcast<_UInt>(__y) >> 16)));
	  else if constexpr (__have_sse4_1)
	    {
	      auto __mask = __vector_bitcast<_UShort>(__y);
	      //__mask *= 0x0808;
	      __mask = (__mask << 3) | (__mask << 11);
	      // do __x = 0 where __y[4] is set
	      __x = __blend(__mask, __x, _V());
	      // do __x =>> 8 where __y[3] is set
	      __x = __blend(__mask += __mask, __x, __x >> 8);
	      // do __x =>> 4 where __y[2] is set
	      __x = __blend(__mask += __mask, __x, __x >> 4);
	      // do __x =>> 2 where __y[1] is set
	      __x = __blend(__mask += __mask, __x, __x >> 2);
	      // do __x =>> 1 where __y[0] is set
	      return __vector_bitcast<_U>(__blend(__mask + __mask, __x, __x >> 1));
	    }
	  else
	    {
	      auto __k = __vector_bitcast<_UShort>(__y) << 11;
	      auto __mask = [](__vector_type16_t<_UShort> __kk) {
		return __vector_bitcast<short>(__kk) < 0;
	      };
	      // do __x = 0 where __y[4] is set
	      __x = __blend(__mask(__k), __x, _V());
	      // do __x =>> 8 where __y[3] is set
	      __x = __blend(__mask(__k += __k), __x, __x >> 8);
	      // do __x =>> 4 where __y[2] is set
	      __x = __blend(__mask(__k += __k), __x, __x >> 4);
	      // do __x =>> 2 where __y[1] is set
	      __x = __blend(__mask(__k += __k), __x, __x >> 2);
	      // do __x =>> 1 where __y[0] is set
	      return __vector_bitcast<_U>(__blend(__mask(__k + __k), __x, __x >> 1));
	    }
	} //}}}
      else if constexpr (sizeof(_U) == 4 && !__have_avx2) //{{{
	{
	  if constexpr (is_unsigned_v<_U>)
	    {
	      // x >> y == x * 2^-y == (x * 2^(31-y)) >> 31
	      const __m128 __factor_f =
		reinterpret_cast<__m128>(0x4f00'0000u - (__y << 23));
	      const __m128i __factor = __builtin_constant_p(__factor_f)
					 ? __to_intrin(__make_vector<int>(
					     __factor_f[0], __factor_f[1],
					     __factor_f[2], __factor_f[3]))
					 : _mm_cvttps_epi32(__factor_f);
	      const auto __r02 = _mm_srli_epi64(_mm_mul_epu32(__ix, __factor), 31);
	      const auto __r13 = _mm_mul_epu32(_mm_srli_si128(__ix, 4),
					       _mm_srli_si128(__factor, 4));
	      if constexpr (__have_sse4_1)
		return __vector_bitcast<_U>(
		  _mm_blend_epi16(_mm_slli_epi64(__r13, 1), __r02, 0x33));
	      else
		return __vector_bitcast<_U>(
		  __r02 | _mm_slli_si128(_mm_srli_epi64(__r13, 31), 4));
	    }
	  else
	    {
	      auto __shift = [](auto __a, auto __b) {
		if constexpr (is_signed_v<_U>)
		  return _mm_sra_epi32(__a, __b);
		else
		  return _mm_srl_epi32(__a, __b);
	      };
	      const auto __r0 =
		__shift(__ix, _mm_unpacklo_epi32(__iy, __m128i()));
	      const auto __r1 = __shift(__ix, _mm_srli_epi64(__iy, 32));
	      const auto __r2 =
		__shift(__ix, _mm_unpackhi_epi32(__iy, __m128i()));
	      const auto __r3 = __shift(__ix, _mm_srli_si128(__iy, 12));
	      if constexpr (__have_sse4_1)
		return __vector_bitcast<_U>(
		  _mm_blend_epi16(_mm_blend_epi16(__r1, __r0, 0x3),
				  _mm_blend_epi16(__r3, __r2, 0x30), 0xf0));
	      else
		return __vector_bitcast<_U>(_mm_unpacklo_epi64(
		  _mm_unpacklo_epi32(__r0, _mm_srli_si128(__r1, 4)),
		  _mm_unpackhi_epi32(__r2, _mm_srli_si128(__r3, 4))));
	    }
	} //}}}
      else
	return __x >> __y;
    }
#endif // _GLIBCXX_SIMD_NO_SHIFT_OPT

    // compares {{{2
    // __equal_to {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _MaskMember<_Tp> __equal_to(
        _SimdWrapper<_Tp, _N> __x, _SimdWrapper<_Tp, _N> __y)
    {
      if constexpr (sizeof(__x) == 64) {  // AVX512
	[[maybe_unused]] const auto __xi = __to_intrin(__x);
	[[maybe_unused]] const auto __yi = __to_intrin(__y);
	if constexpr (std::is_floating_point_v<_Tp>) {
	  if constexpr (sizeof(_Tp) == 8) { return _mm512_cmp_pd_mask(__xi, __yi, _CMP_EQ_OQ);
	  } else if constexpr (sizeof(_Tp) == 4) { return _mm512_cmp_ps_mask(__xi, __yi, _CMP_EQ_OQ);
	  } else { __assert_unreachable<_Tp>(); }
	} else {
	  if constexpr (sizeof(_Tp) == 8) { return _mm512_cmpeq_epi64_mask(__xi, __yi);
	  } else if constexpr (sizeof(_Tp) == 4) { return _mm512_cmpeq_epi32_mask(__xi, __yi);
	  } else if constexpr (sizeof(_Tp) == 2) { return _mm512_cmpeq_epi16_mask(__xi, __yi);
	  } else if constexpr (sizeof(_Tp) == 1) { return _mm512_cmpeq_epi8_mask(__xi, __yi);
	  } else { __assert_unreachable<_Tp>(); }
	}
      } else
	return _Base::__equal_to(__x,__y);
    }

    // __not_equal_to {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _MaskMember<_Tp> __not_equal_to(
        _SimdWrapper<_Tp, _N> __x, _SimdWrapper<_Tp, _N> __y)
    {
      if constexpr (sizeof(__x) == 64) {  // AVX512
	[[maybe_unused]] const auto __xi = __to_intrin(__x);
	[[maybe_unused]] const auto __yi = __to_intrin(__y);
	if constexpr (std::is_floating_point_v<_Tp>) {
	  if constexpr (sizeof(_Tp) == 8) { return _mm512_cmp_pd_mask(__xi, __yi, _CMP_NEQ_UQ);
	  } else if constexpr (sizeof(_Tp) == 4) { return _mm512_cmp_ps_mask(__xi, __yi, _CMP_NEQ_UQ);
	  } else { __assert_unreachable<_Tp>(); }
	} else {
	  if constexpr (sizeof(_Tp) == 8) { return ~_mm512_cmpeq_epi64_mask(__xi, __yi);
	  } else if constexpr (sizeof(_Tp) == 4) { return ~_mm512_cmpeq_epi32_mask(__xi, __yi);
	  } else if constexpr (sizeof(_Tp) == 2) { return ~_mm512_cmpeq_epi16_mask(__xi, __yi);
	  } else if constexpr (sizeof(_Tp) == 1) { return ~_mm512_cmpeq_epi8_mask(__xi, __yi);
	  } else { __assert_unreachable<_Tp>(); }
	}
      } else
	return _Base::__not_equal_to(__x, __y);
    }

    // __less {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _MaskMember<_Tp> __less(_SimdWrapper<_Tp, _N> __x,
                                                           _SimdWrapper<_Tp, _N> __y)
    {
      if constexpr (sizeof(__x) == 64) {  // AVX512
	[[maybe_unused]] const auto __xi = __to_intrin(__x);
	[[maybe_unused]] const auto __yi = __to_intrin(__y);
	if constexpr (std::is_floating_point_v<_Tp>) {
	  if constexpr (sizeof(_Tp) == 8) { return _mm512_cmp_pd_mask(__xi, __yi, _CMP_LT_OS);
	  } else if constexpr (sizeof(_Tp) == 4) { return _mm512_cmp_ps_mask(__xi, __yi, _CMP_LT_OS);
	  } else { __assert_unreachable<_Tp>(); }
	} else if constexpr (std::is_signed_v<_Tp>) {
	  if constexpr (sizeof(_Tp) == 8) { return _mm512_cmplt_epi64_mask(__xi, __yi);
	  } else if constexpr (sizeof(_Tp) == 4) { return _mm512_cmplt_epi32_mask(__xi, __yi);
	  } else if constexpr (sizeof(_Tp) == 2) { return _mm512_cmplt_epi16_mask(__xi, __yi);
	  } else if constexpr (sizeof(_Tp) == 1) { return _mm512_cmplt_epi8_mask(__xi, __yi);
	  } else { __assert_unreachable<_Tp>(); }
	} else {
	  static_assert(std::is_unsigned_v<_Tp>);
	  if constexpr (sizeof(_Tp) == 8) { return _mm512_cmplt_epu64_mask(__xi, __yi);
	  } else if constexpr (sizeof(_Tp) == 4) { return _mm512_cmplt_epu32_mask(__xi, __yi);
	  } else if constexpr (sizeof(_Tp) == 2) { return _mm512_cmplt_epu16_mask(__xi, __yi);
	  } else if constexpr (sizeof(_Tp) == 1) { return _mm512_cmplt_epu8_mask(__xi, __yi);
	  } else { __assert_unreachable<_Tp>(); }
	}
      } else
	return _Base::__less(__x, __y);
    }

    // __less_equal {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _MaskMember<_Tp> __less_equal(_SimdWrapper<_Tp, _N> __x,
                                                                 _SimdWrapper<_Tp, _N> __y)
    {
      if constexpr (sizeof(__x) == 64) {  // AVX512
	[[maybe_unused]] const auto __xi = __to_intrin(__x);
	[[maybe_unused]] const auto __yi = __to_intrin(__y);
	if constexpr (std::is_floating_point_v<_Tp>) {
	  if constexpr (sizeof(_Tp) == 8) { return _mm512_cmp_pd_mask(__xi, __yi, _CMP_LE_OS);
	  } else if constexpr (sizeof(_Tp) == 4) { return _mm512_cmp_ps_mask(__xi, __yi, _CMP_LE_OS);
	  } else { __assert_unreachable<_Tp>(); }
	} else if constexpr (std::is_signed_v<_Tp>) {
	  if constexpr (sizeof(_Tp) == 8) { return _mm512_cmple_epi64_mask(__xi, __yi);
	  } else if constexpr (sizeof(_Tp) == 4) { return _mm512_cmple_epi32_mask(__xi, __yi);
	  } else if constexpr (sizeof(_Tp) == 2) { return _mm512_cmple_epi16_mask(__xi, __yi);
	  } else if constexpr (sizeof(_Tp) == 1) { return _mm512_cmple_epi8_mask(__xi, __yi);
	  } else { __assert_unreachable<_Tp>(); }
	} else {
	  static_assert(std::is_unsigned_v<_Tp>);
	  if constexpr (sizeof(_Tp) == 8) { return _mm512_cmple_epu64_mask(__xi, __yi);
	  } else if constexpr (sizeof(_Tp) == 4) { return _mm512_cmple_epu32_mask(__xi, __yi);
	  } else if constexpr (sizeof(_Tp) == 2) { return _mm512_cmple_epu16_mask(__xi, __yi);
	  } else if constexpr (sizeof(_Tp) == 1) { return _mm512_cmple_epu8_mask(__xi, __yi);
	  } else { __assert_unreachable<_Tp>(); }
	}
      } else
	return _Base::__less_equal(__x, __y);
    }

    // negation {{{2
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static constexpr _MaskMember<_Tp> __negate(_SimdWrapper<_Tp, _N> __x) noexcept
    {
      if constexpr (__is_abi<_Abi, simd_abi::_Avx512Abi>()) {
	  return __equal_to(__x, _SimdWrapper<_Tp, _N>());
      } else {
	return _Base::__negate(__x);
      }
    }

    // math {{{2
    using _Base::__abs;
    // __sqrt {{{3
    template <class _Tp, size_t _N> _GLIBCXX_SIMD_INTRINSIC static _SimdWrapper<_Tp, _N> __sqrt(_SimdWrapper<_Tp, _N> __x)
    {
               if constexpr (__is_sse_ps   <_Tp, _N>()) { return _mm_sqrt_ps(__x);
        } else if constexpr (__is_sse_pd   <_Tp, _N>()) { return _mm_sqrt_pd(__x);
        } else if constexpr (__is_avx_ps   <_Tp, _N>()) { return _mm256_sqrt_ps(__x);
        } else if constexpr (__is_avx_pd   <_Tp, _N>()) { return _mm256_sqrt_pd(__x);
        } else if constexpr (__is_avx512_ps<_Tp, _N>()) { return _mm512_sqrt_ps(__x);
        } else if constexpr (__is_avx512_pd<_Tp, _N>()) { return _mm512_sqrt_pd(__x);
        } else { __assert_unreachable<_Tp>(); }
    }

    // __trunc {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _SimdWrapper<_Tp, _N> __trunc(_SimdWrapper<_Tp, _N> __x)
    {
        if constexpr (__is_avx512_ps<_Tp, _N>()) {
            return _mm512_roundscale_ps(__x, 0x0b);
        } else if constexpr (__is_avx512_pd<_Tp, _N>()) {
            return _mm512_roundscale_pd(__x, 0x0b);
        } else if constexpr (__is_avx_ps<_Tp, _N>()) {
            return _mm256_round_ps(__x, 0x3);
        } else if constexpr (__is_avx_pd<_Tp, _N>()) {
            return _mm256_round_pd(__x, 0x3);
        } else if constexpr (__have_sse4_1 && __is_sse_ps<_Tp, _N>()) {
            return _mm_round_ps(__x, 0x3);
        } else if constexpr (__have_sse4_1 && __is_sse_pd<_Tp, _N>()) {
            return _mm_round_pd(__x, 0x3);
        } else if constexpr (__is_sse_ps<_Tp, _N>()) {
            auto __truncated = _mm_cvtepi32_ps(_mm_cvttps_epi32(__x));
            const auto __no_fractional_values = __vector_bitcast<float>(
                __vector_bitcast<int>(__vector_bitcast<_UInt>(__x._M_data) & 0x7f800000u) <
                0x4b000000);  // the exponent is so large that no mantissa bits signify
                              // fractional values (0x3f8 + 23*8 = 0x4b0)
            return __blend(__no_fractional_values, __x, __truncated);
        } else {
            return _Base::__trunc(__x);
        }
    }

    // __round {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _SimdWrapper<_Tp, _N>
      __round(_SimdWrapper<_Tp, _N> __x)
    {
      using _V = __vector_type_t<_Tp, _N>;
      _V __truncated;
      if constexpr (__is_avx512_ps<_Tp, _N>())
	__truncated = _mm512_roundscale_ps(__x._M_data, 0x0b);
      else if constexpr (__is_avx512_pd<_Tp, _N>())
	__truncated = _mm512_roundscale_pd(__x._M_data, 0x0b);
      else if constexpr (__is_avx_ps<_Tp, _N>())
	__truncated = _mm256_round_ps(__x._M_data,
			       _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
      else if constexpr (__is_avx_pd<_Tp, _N>())
	__truncated = _mm256_round_pd(__x._M_data,
			       _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
      else if constexpr (__have_sse4_1 && __is_sse_ps<_Tp, _N>())
	__truncated = _mm_round_ps(__x._M_data,
			    _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
      else if constexpr (__have_sse4_1 && __is_sse_pd<_Tp, _N>())
	__truncated = _mm_round_pd(__x._M_data,
			    _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
      else if constexpr (__is_sse_ps<_Tp, _N>())
	__truncated = _mm_cvtepi32_ps(_mm_cvttps_epi32(__to_intrin(__x)));
      else
	return _Base::__round(__x);

      // x < 0 => truncated <= 0 && truncated >= x => x - truncated <= 0
      // x > 0 => truncated >= 0 && truncated <= x => x - truncated >= 0

      const _V __rounded =
	__truncated +
	(__and(_S_absmask<_V>, __x._M_data - __truncated) >= _Tp(.5)
	   ? __or(__and(_S_signmask<_V>, __x._M_data), _V() + 1)
	   : _V());
      if constexpr(__have_sse4_1)
	return __rounded;
      else
	return __and(_S_absmask<_V>, __x._M_data) < 0x1p23f ? __rounded : __x._M_data;
    }

    // __nearbyint {{{3
    template <typename _Tp, typename _TVT = _VectorTraits<_Tp>>
    _GLIBCXX_SIMD_INTRINSIC static _Tp __nearbyint(_Tp __x) noexcept
    {
      if constexpr (_TVT::template __is<float, 16>)
	return _mm512_roundscale_ps(__x, 0x0c);
      else if constexpr (_TVT::template __is<double, 8>)
	return _mm512_roundscale_pd(__x, 0x0c);
      else if constexpr (_TVT::template __is<float, 8>)
	return _mm256_round_ps(__x,
			       _MM_FROUND_CUR_DIRECTION | _MM_FROUND_NO_EXC);
      else if constexpr (_TVT::template __is<double, 4>)
	return _mm256_round_pd(__x,
			       _MM_FROUND_CUR_DIRECTION | _MM_FROUND_NO_EXC);
      else if constexpr (__have_sse4_1 && _TVT::template __is<float, 4>)
	return _mm_round_ps(__x, _MM_FROUND_CUR_DIRECTION | _MM_FROUND_NO_EXC);
      else if constexpr (__have_sse4_1 && _TVT::template __is<double, 2>)
	return _mm_round_pd(__x, _MM_FROUND_CUR_DIRECTION | _MM_FROUND_NO_EXC);
      else
	return _Base::__nearbyint(__x);
    }

    // __rint {{{3
    template <typename _Tp, typename _TVT = _VectorTraits<_Tp>>
    _GLIBCXX_SIMD_INTRINSIC static _Tp __rint(_Tp __x) noexcept
    {
      if constexpr (_TVT::template __is<float, 16>)
	return _mm512_roundscale_ps(__x, 0x04);
      else if constexpr (_TVT::template __is<double, 8>)
	return _mm512_roundscale_pd(__x, 0x04);
      else if constexpr (_TVT::template __is<float, 8>)
	return _mm256_round_ps(__x, _MM_FROUND_CUR_DIRECTION);
      else if constexpr (_TVT::template __is<double, 4>)
	return _mm256_round_pd(__x, _MM_FROUND_CUR_DIRECTION);
      else if constexpr (__have_sse4_1 && _TVT::template __is<float, 4>)
	return _mm_round_ps(__x, _MM_FROUND_CUR_DIRECTION);
      else if constexpr (__have_sse4_1 && _TVT::template __is<double, 2>)
	return _mm_round_pd(__x, _MM_FROUND_CUR_DIRECTION);
      else
	return _Base::__rint(__x);
    }

    // __floor {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _SimdWrapper<_Tp, _N> __floor(_SimdWrapper<_Tp, _N> __x)
    {
        if constexpr (__is_avx512_ps<_Tp, _N>()) {
            return _mm512_roundscale_ps(__x, 0x09);
        } else if constexpr (__is_avx512_pd<_Tp, _N>()) {
            return _mm512_roundscale_pd(__x, 0x09);
        } else if constexpr (__is_avx_ps<_Tp, _N>()) {
            return _mm256_round_ps(__x, 0x1);
        } else if constexpr (__is_avx_pd<_Tp, _N>()) {
            return _mm256_round_pd(__x, 0x1);
        } else if constexpr (__have_sse4_1 && __is_sse_ps<_Tp, _N>()) {
            return _mm_floor_ps(__x);
        } else if constexpr (__have_sse4_1 && __is_sse_pd<_Tp, _N>()) {
            return _mm_floor_pd(__x);
        } else {
	  return _Base::__floor(__x);
        }
    }

    // __ceil {{{3
    template <class _Tp, size_t _N> _GLIBCXX_SIMD_INTRINSIC static _SimdWrapper<_Tp, _N> __ceil(_SimdWrapper<_Tp, _N> __x)
    {
        if constexpr (__is_avx512_ps<_Tp, _N>()) {
            return _mm512_roundscale_ps(__x, 0x0a);
        } else if constexpr (__is_avx512_pd<_Tp, _N>()) {
            return _mm512_roundscale_pd(__x, 0x0a);
        } else if constexpr (__is_avx_ps<_Tp, _N>()) {
            return _mm256_round_ps(__x, 0x2);
        } else if constexpr (__is_avx_pd<_Tp, _N>()) {
            return _mm256_round_pd(__x, 0x2);
        } else if constexpr (__have_sse4_1 && __is_sse_ps<_Tp, _N>()) {
            return _mm_ceil_ps(__x);
        } else if constexpr (__have_sse4_1 && __is_sse_pd<_Tp, _N>()) {
            return _mm_ceil_pd(__x);
        } else {
	  return _Base::__ceil(__x);
        }
    }

    // __signbit {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _MaskMember<_Tp> __signbit(_SimdWrapper<_Tp, _N> __x)
    {
        using _I = __int_for_sizeof_t<_Tp>;
        if constexpr (__have_avx512dq && __is_avx512_ps<_Tp, _N>()) {
            return _mm512_movepi32_mask(__vector_bitcast<_LLong>(__x));
        } else if constexpr (__have_avx512dq && __is_avx512_pd<_Tp, _N>()) {
            return _mm512_movepi64_mask(__vector_bitcast<_LLong>(__x));
        } else if constexpr (sizeof(__x) == 64) {
            const auto __signmask = __vector_broadcast<_N>(std::numeric_limits<_I>::min());
            return __equal_to(_SimdWrapper<_I, _N>(__vector_bitcast<_I>(__x._M_data) & __signmask),
                            _SimdWrapper<_I, _N>(__signmask));
        } else {
            const auto __xx = __vector_bitcast<_I>(__x._M_data);
            [[maybe_unused]] constexpr _I __signmask = std::numeric_limits<_I>::min();
            if constexpr ((sizeof(_Tp) == 4 && (__have_avx2 || sizeof(__x) == 16)) ||
                          __have_avx512vl) {
                return __vector_bitcast<_Tp>(__xx >> std::numeric_limits<_I>::digits);
            } else if constexpr ((__have_avx2 || (__have_ssse3 && sizeof(__x) == 16))) {
                return __vector_bitcast<_Tp>((__xx & __signmask) == __signmask);
            } else {  // SSE2/3 or AVX (w/o AVX2)
                constexpr auto __one = __vector_broadcast<_N, _Tp>(1);
                return __vector_bitcast<_Tp>(
                    __vector_bitcast<_Tp>((__xx & __signmask) | __vector_bitcast<_I>(__one))  // -1 or 1
                    != __one);
            }
        }
    }

    // __isnonzerovalue_mask (isnormal | is subnormal == !isinf & !isnan & !is zero) {{{3
    template <class _Tp>
    _GLIBCXX_SIMD_INTRINSIC static auto __isnonzerovalue_mask(_Tp __x)
    {
      using _Traits = _VectorTraits<_Tp>;
      if constexpr (__have_avx512dq_vl)
	{
	  if constexpr (_Traits::template __is<float, 4>)
	    return _knot_mask8(_mm_fpclass_ps_mask(__x, 0x9f));
	  else if constexpr (_Traits::template __is<float, 8>)
	    return _knot_mask8(_mm256_fpclass_ps_mask(__x, 0x9f));
	  else if constexpr (_Traits::template __is<float, 16>)
	    return _knot_mask16(_mm512_fpclass_ps_mask(__x, 0x9f));
	  else if constexpr (_Traits::template __is<double, 2>)
	    return _knot_mask8(_mm_fpclass_pd_mask(__x, 0x9f));
	  else if constexpr (_Traits::template __is<double, 4>)
	    return _knot_mask8(_mm256_fpclass_pd_mask(__x, 0x9f));
	  else if constexpr (_Traits::template __is<double, 8>)
	    return _knot_mask8(_mm512_fpclass_pd_mask(__x, 0x9f));
	  else
	    __assert_unreachable<_Tp>();
	}
      else
	{
	  using _U            = typename _Traits::value_type;
	  constexpr size_t _N = _Traits::_S_width;
	  const auto       __a =
	    __x * std::numeric_limits<_U>::infinity(); // NaN if __x == 0
	  const auto __b = __x * _U();                 // NaN if __x == inf
	  if constexpr (__have_avx512vl && __is_sse_ps<_U, _N>())
	    {
	      return _mm_cmp_ps_mask(__a, __b, _CMP_ORD_Q);
	    }
	  else if constexpr (__have_avx512f && __is_sse_ps<_U, _N>())
	    {
	      return __mmask8(0xf & _mm512_cmp_ps_mask(__auto_bitcast(__a),
						       __auto_bitcast(__b),
						       _CMP_ORD_Q));
	    }
	  else if constexpr (__have_avx512vl && __is_sse_pd<_U, _N>())
	    {
	      return _mm_cmp_pd_mask(__a, __b, _CMP_ORD_Q);
	    }
	  else if constexpr (__have_avx512f && __is_sse_pd<_U, _N>())
	    {
	      return __mmask8(0x3 & _mm512_cmp_pd_mask(__auto_bitcast(__a),
						       __auto_bitcast(__b),
						       _CMP_ORD_Q));
	    }
	  else if constexpr (__have_avx512vl && __is_avx_ps<_U, _N>())
	    {
	      return _mm256_cmp_ps_mask(__a, __b, _CMP_ORD_Q);
	    }
	  else if constexpr (__have_avx512f && __is_avx_ps<_U, _N>())
	    {
	      return __mmask8(_mm512_cmp_ps_mask(
		__auto_bitcast(__a), __auto_bitcast(__b), _CMP_ORD_Q));
	    }
	  else if constexpr (__have_avx512vl && __is_avx_pd<_U, _N>())
	    {
	      return _mm256_cmp_pd_mask(__a, __b, _CMP_ORD_Q);
	    }
	  else if constexpr (__have_avx512f && __is_avx_pd<_U, _N>())
	    {
	      return __mmask8(0xf & _mm512_cmp_pd_mask(__auto_bitcast(__a),
						       __auto_bitcast(__b),
						       _CMP_ORD_Q));
	    }
	  else if constexpr (__is_avx512_ps<_U, _N>())
	    {
	      return _mm512_cmp_ps_mask(__a, __b, _CMP_ORD_Q);
	    }
	  else if constexpr (__is_avx512_pd<_U, _N>())
	    {
	      return _mm512_cmp_pd_mask(__a, __b, _CMP_ORD_Q);
	    }
	  else
	    {
	      __assert_unreachable<_Tp>();
	    }
	}
    }

    // __isfinite {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _MaskMember<_Tp>
      __isfinite(_SimdWrapper<_Tp, _N> __x)
    {
#if __FINITE_MATH_ONLY__
      __unused(__x);
      return __equal_to(_SimdWrapper<_Tp, _N>(), _SimdWrapper<_Tp, _N>());
#else
      return __cmpord(__x._M_data, __x._M_data * _Tp());
#endif
    }

    // __isinf {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _MaskMember<_Tp> __isinf(_SimdWrapper<_Tp, _N> __x)
    {
#if __FINITE_MATH_ONLY__
      __unused(__x);
      return {}; // false
#else
      if constexpr (__is_avx512_pd<_Tp, _N>() && __have_avx512dq)
	return _mm512_fpclass_pd_mask(__x, 0x18);
      else if constexpr (__is_avx512_ps<_Tp, _N>() && __have_avx512dq)
	return _mm512_fpclass_ps_mask(__x, 0x18);
      else if constexpr (__have_avx512dq_vl)
	{
	  if constexpr (__is_sse_pd<_Tp, _N>())
	    return __vector_bitcast<double>(
	      _mm_movm_epi64(_mm_fpclass_pd_mask(__x, 0x18)));
	  else if constexpr (__is_avx_pd<_Tp, _N>())
	    return __vector_bitcast<double>(
	      _mm256_movm_epi64(_mm256_fpclass_pd_mask(__x, 0x18)));
	  else if constexpr (__is_sse_ps<_Tp, _N>())
	    return __vector_bitcast<float>(
	      _mm_movm_epi32(_mm_fpclass_ps_mask(__x, 0x18)));
	  else if constexpr (__is_avx_ps<_Tp, _N>())
	    return __vector_bitcast<float>(
	      _mm256_movm_epi32(_mm256_fpclass_ps_mask(__x, 0x18)));
	  else
	    __assert_unreachable<_Tp>();
	}
      else
	return _Base::__isinf(__x);
#endif
    }

    // __isnormal {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static _MaskMember<_Tp>
      __isnormal(_SimdWrapper<_Tp, _N> __x)
    {
      if constexpr (__have_avx512dq)
	{
	  if constexpr (__have_avx512vl && __is_sse_ps<_Tp, _N>())
	    return __vector_bitcast<float>(
	      _mm_movm_epi32(_knot_mask8(_mm_fpclass_ps_mask(__x, 0xbf))));
	  else if constexpr (__have_avx512vl && __is_avx_ps<_Tp, _N>())
	    return __vector_bitcast<float>(_mm256_movm_epi32(
	      _knot_mask8(_mm256_fpclass_ps_mask(__x, 0xbf))));
	  else if constexpr (__is_avx512_ps<_Tp, _N>())
	    return _knot_mask16(_mm512_fpclass_ps_mask(__x, 0xbf));
	  else if constexpr (__have_avx512vl && __is_sse_pd<_Tp, _N>())
	    return __vector_bitcast<double>(
	      _mm_movm_epi64(_knot_mask8(_mm_fpclass_pd_mask(__x, 0xbf))));
	  else if constexpr (__have_avx512vl && __is_avx_pd<_Tp, _N>())
	    return __vector_bitcast<double>(_mm256_movm_epi64(
	      _knot_mask8(_mm256_fpclass_pd_mask(__x, 0xbf))));
	  else if constexpr (__is_avx512_pd<_Tp, _N>())
	    return _knot_mask8(_mm512_fpclass_pd_mask(__x, 0xbf));
	  else
	    __assert_unreachable<_Tp>();
	}
      else
	return _Base::__isnormal(__x);
    }

    // __isnan {{{3
    using _Base::__isnan;

    // __fpclassify {{{3
    template <class _Tp, size_t _N>
    _GLIBCXX_SIMD_INTRINSIC static __fixed_size_storage_t<int, _N> __fpclassify(_SimdWrapper<_Tp, _N> __x)
    {
        if constexpr (__is_avx512_pd<_Tp, _N>()) {
            // AVX512 is special because we want to use an __mmask to blend int vectors
            // (coming from double vectors). GCC doesn't allow this combination on the
            // ternary operator. Thus, resort to intrinsics:
            if constexpr (__have_avx512vl) {
                auto &&__b = [](int __y) { return __to_intrin(__vector_broadcast<_N>(__y)); };
                return {_mm256_mask_mov_epi32(
                    _mm256_mask_mov_epi32(
                        _mm256_mask_mov_epi32(__b(FP_NORMAL), __isnan(__x), __b(FP_NAN)),
                        __isinf(__x), __b(FP_INFINITE)),
                    _mm512_cmp_pd_mask(
                        __abs(__x),
                        __vector_broadcast<_N>(std::numeric_limits<double>::min()),
                        _CMP_LT_OS),
                    _mm256_mask_mov_epi32(
                        __b(FP_SUBNORMAL),
                        _mm512_cmp_pd_mask(__x, _mm512_setzero_pd(), _CMP_EQ_OQ),
                        __b(FP_ZERO)))};
            } else {
                auto &&__b = [](int __y) {
                    return _mm512_castsi256_si512(__to_intrin(__vector_broadcast<_N>(__y)));
                };
                return {__lo256(_mm512_mask_mov_epi32(
                    _mm512_mask_mov_epi32(
                        _mm512_mask_mov_epi32(__b(FP_NORMAL), __isnan(__x), __b(FP_NAN)),
                        __isinf(__x), __b(FP_INFINITE)),
                    _mm512_cmp_pd_mask(
                        __abs(__x),
                        __vector_broadcast<_N>(std::numeric_limits<double>::min()),
                        _CMP_LT_OS),
                    _mm512_mask_mov_epi32(
                        __b(FP_SUBNORMAL),
                        _mm512_cmp_pd_mask(__x, _mm512_setzero_pd(), _CMP_EQ_OQ),
                        __b(FP_ZERO))))};
            }
        } else {
	  return _Base::__fpclassify(__x);
        }
    }

    //}}}2
};

// __x86_mask_impl {{{1
template <class _Abi>
struct __x86_mask_impl : __generic_mask_impl<_Abi>
{
  using _Base = __generic_mask_impl<_Abi>;

  // __masked_load {{{2
  template <class _Tp, size_t _N, class _F>
  static inline _SimdWrapper<_Tp, _N> __masked_load(_SimdWrapper<_Tp, _N> __merge,
						  _SimdWrapper<_Tp, _N> __mask,
						  const bool*           __mem,
						  _F) noexcept
  {
    if constexpr (__is_abi<_Abi, simd_abi::_Avx512Abi>())
      {
	if constexpr (__have_avx512bw_vl)
	  {
	    if constexpr (_N == 8)
	      {
		const auto __a = _mm_mask_loadu_epi8(__m128i(), __mask, __mem);
		return (__merge & ~__mask) | _mm_test_epi8_mask(__a, __a);
	      }
	    else if constexpr (_N == 16)
	      {
		const auto __a = _mm_mask_loadu_epi8(__m128i(), __mask, __mem);
		return (__merge & ~__mask) | _mm_test_epi8_mask(__a, __a);
	      }
	    else if constexpr (_N == 32)
	      {
		const auto __a = _mm256_mask_loadu_epi8(__m256i(), __mask, __mem);
		return (__merge & ~__mask) | _mm256_test_epi8_mask(__a, __a);
	      }
	    else if constexpr (_N == 64)
	      {
		const auto __a = _mm512_mask_loadu_epi8(__m512i(), __mask, __mem);
		return (__merge & ~__mask) | _mm512_test_epi8_mask(__a, __a);
	      }
	    else
	      {
		__assert_unreachable<_Tp>();
	      }
	  }
	else
	  {
	    __bit_iteration(__mask, [&](auto __i) { __merge.__set(__i, __mem[__i]); });
	    return __merge;
	  }
      }
    else if constexpr (__have_avx512bw_vl && _N == 32 && sizeof(_Tp) == 1)
      {
	const auto __k = __convert_mask<_SimdWrapper<bool, _N>>(__mask);
	__merge          = _ToWrapper(
          _mm256_mask_sub_epi8(__vector_bitcast<_LLong>(__merge), __k, __m256i(),
                               _mm256_mask_loadu_epi8(__m256i(), __k, __mem)));
      }
    else if constexpr (__have_avx512bw_vl && _N == 16 && sizeof(_Tp) == 1)
      {
	const auto __k = __convert_mask<_SimdWrapper<bool, _N>>(__mask);
	__merge          = _ToWrapper(
          _mm_mask_sub_epi8(__vector_bitcast<_LLong>(__merge), __k, __m128i(),
                            _mm_mask_loadu_epi8(__m128i(), __k, __mem)));
      }
    else if constexpr (__have_avx512bw_vl && _N == 16 && sizeof(_Tp) == 2)
      {
	const auto __k = __convert_mask<_SimdWrapper<bool, _N>>(__mask);
	__merge          = _ToWrapper(_mm256_mask_sub_epi16(
          __vector_bitcast<_LLong>(__merge), __k, __m256i(),
          _mm256_cvtepi8_epi16(_mm_mask_loadu_epi8(__m128i(), __k, __mem))));
      }
    else if constexpr (__have_avx512bw_vl && _N == 8 && sizeof(_Tp) == 2)
      {
	const auto __k = __convert_mask<_SimdWrapper<bool, _N>>(__mask);
	__merge          = _ToWrapper(_mm_mask_sub_epi16(
          __vector_bitcast<_LLong>(__merge), __k, __m128i(),
          _mm_cvtepi8_epi16(_mm_mask_loadu_epi8(__m128i(), __k, __mem))));
      }
    else if constexpr (__have_avx512bw_vl && _N == 8 && sizeof(_Tp) == 4)
      {
	const auto __k = __convert_mask<_SimdWrapper<bool, _N>>(__mask);
	__merge          = _ToWrapper(_mm256_mask_sub_epi32(
          __vector_bitcast<_LLong>(__merge), __k, __m256i(),
          _mm256_cvtepi8_epi32(_mm_mask_loadu_epi8(__m128i(), __k, __mem))));
      }
    else if constexpr (__have_avx512bw_vl && _N == 4 && sizeof(_Tp) == 4)
      {
	const auto __k = __convert_mask<_SimdWrapper<bool, _N>>(__mask);
	__merge          = _ToWrapper(_mm_mask_sub_epi32(
          __vector_bitcast<_LLong>(__merge), __k, __m128i(),
          _mm_cvtepi8_epi32(_mm_mask_loadu_epi8(__m128i(), __k, __mem))));
      }
    else if constexpr (__have_avx512bw_vl && _N == 4 && sizeof(_Tp) == 8)
      {
	const auto __k = __convert_mask<_SimdWrapper<bool, _N>>(__mask);
	__merge          = _ToWrapper(_mm256_mask_sub_epi64(
          __vector_bitcast<_LLong>(__merge), __k, __m256i(),
          _mm256_cvtepi8_epi64(_mm_mask_loadu_epi8(__m128i(), __k, __mem))));
      }
    else if constexpr (__have_avx512bw_vl && _N == 2 && sizeof(_Tp) == 8)
      {
	const auto __k = __convert_mask<_SimdWrapper<bool, _N>>(__mask);
	__merge          = _ToWrapper(_mm_mask_sub_epi64(
          __vector_bitcast<_LLong>(__merge), __k, __m128i(),
          _mm_cvtepi8_epi64(_mm_mask_loadu_epi8(__m128i(), __k, __mem))));
      }
    else
      {
	return _Base::__masked_load(__merge, __mask, __mem, _F{});
      }
    return __merge;
  }

  // __store {{{2
  template <class _Tp, size_t _N, class _F>
  _GLIBCXX_SIMD_INTRINSIC static void
    __store(_SimdWrapper<_Tp, _N> __v, bool* __mem, _F) noexcept
  {
    if constexpr (__is_abi<_Abi, simd_abi::_SseAbi>())
      {
	if constexpr (_N == 2 && __have_sse2)
	  {
	    const auto __k = __vector_bitcast<int>(__v);
	    __mem[0]       = -__k[1];
	    __mem[1]       = -__k[3];
	  }
	else if constexpr (_N == 4 && __have_sse2)
	  {
	    const unsigned __bool4 =
	      __vector_bitcast<_UInt>(_mm_packs_epi16(
		_mm_packs_epi32(__vector_bitcast<_LLong>(__v), __m128i()),
		__m128i()))[0] &
	      0x01010101u;
	    std::memcpy(__mem, &__bool4, 4);
	  }
	else if constexpr (std::is_same_v<_Tp, float> && __have_mmx)
	  {
	    const __m128 __k = __to_intrin(__v);
	    const __m64  __kk  = _mm_cvtps_pi8(__and(__k, _mm_set1_ps(1.f)));
	    __vector_store<4>(__kk, __mem, _F());
	    _mm_empty();
	  }
	else if constexpr (_N == 8 && __have_sse2)
	  {
	    __vector_store<8>(
	      _mm_packs_epi16(__to_intrin(__vector_bitcast<ushort>(__v) >> 15),
			      __m128i()),
	      __mem, _F());
	  }
	else if constexpr (_N == 16 && __have_sse2)
	  {
	    __vector_store(__v._M_data & 1, __mem, _F());
	  }
	else
	  {
	    __assert_unreachable<_Tp>();
	  }
      }
    else if constexpr (__is_abi<_Abi, simd_abi::_AvxAbi>())
      {
	if constexpr (_N == 4 && __have_avx)
	  {
	    auto __k = __vector_bitcast<_LLong>(__v);
	    int  __bool4;
	    if constexpr (__have_avx2)
	      {
		__bool4 = _mm256_movemask_epi8(__k);
	      }
	    else
	      {
		__bool4 = (_mm_movemask_epi8(__lo128(__k)) |
			 (_mm_movemask_epi8(__hi128(__k)) << 16));
	      }
	    __bool4 &= 0x01010101;
	    std::memcpy(__mem, &__bool4, 4);
	  }
	else if constexpr (_N == 8 && __have_avx)
	  {
	    const auto __k = __vector_bitcast<_LLong>(__v);
	    const auto __k2 =
	      _mm_srli_epi16(_mm_packs_epi16(__lo128(__k), __hi128(__k)), 15);
	    const auto __k3 = _mm_packs_epi16(__k2, __m128i());
	    __vector_store<8>(__k3, __mem, _F());
	  }
	else if constexpr (_N == 16 && __have_avx2)
	  {
	    const auto __x   = _mm256_srli_epi16(__to_intrin(__v), 15);
	    const auto __bools = _mm_packs_epi16(__lo128(__x), __hi128(__x));
	    __vector_store<16>(__bools, __mem, _F());
	  }
	else if constexpr (_N == 16 && __have_avx)
	  {
	    const auto __bools =
	      1 & __vector_bitcast<_UChar>(_mm_packs_epi16(
		    __lo128(__to_intrin(__v)), __hi128(__to_intrin(__v))));
	    __vector_store<16>(__bools, __mem, _F());
	  }
	else if constexpr (_N == 32 && __have_avx)
	  {
	    __vector_store<32>(1 & __v._M_data, __mem, _F());
	  }
	else
	  {
	    __assert_unreachable<_Tp>();
	  }
      }
    else if constexpr (__is_abi<_Abi, simd_abi::_Avx512Abi>())
      {
	if constexpr (_N == 8)
	  {
	    __vector_store<8>(
#if _GLIBCXX_SIMD_HAVE_AVX512VL && _GLIBCXX_SIMD_HAVE_AVX512BW
	      _mm_maskz_set1_epi8(__v._M_data, 1),
#elif defined __x86_64__
	      __make_wrapper<_ULLong>(
		_pdep_u64(__v._M_data, 0x0101010101010101ULL), 0ull),
#else
	      __make_wrapper<_UInt>(_pdep_u32(__v._M_data, 0x01010101U),
				    _pdep_u32(__v._M_data >> 4, 0x01010101U)),
#endif
	      __mem, _F());
	  }
	else if constexpr (_N == 16 && __have_avx512bw_vl)
	  {
	    __vector_store(_mm_maskz_set1_epi8(__v._M_data, 1), __mem, _F());
	  }
	else if constexpr (_N == 16 && __have_avx512f)
	  {
	    _mm512_mask_cvtepi32_storeu_epi8(
	      __mem, ~__mmask16(), _mm512_maskz_set1_epi32(__v._M_data, 1));
	  }
	else if constexpr (_N == 32 && __have_avx512bw_vl)
	  {
	    __vector_store(_mm256_maskz_set1_epi8(__v._M_data, 1), __mem, _F());
	  }
	else if constexpr (_N == 32 && __have_avx512bw)
	  {
	    __vector_store(__lo256(_mm512_maskz_set1_epi8(__v._M_data, 1)),
			   __mem, _F());
	  }
	else if constexpr (_N == 64 && __have_avx512bw)
	  {
	    __vector_store(_mm512_maskz_set1_epi8(__v._M_data, 1), __mem, _F());
	  }
	else
	  {
	    __assert_unreachable<_Tp>();
	  }
      }
    else
      {
	__assert_unreachable<_Tp>();
      }
  }

  // __masked_store {{{2
  template <class _Tp, size_t _N, class _F>
  static inline void __masked_store(const _SimdWrapper<_Tp, _N> __v,
				  bool*                       __mem,
				  _F,
				  const _SimdWrapper<_Tp, _N> __k) noexcept
  {
    if constexpr (__is_abi<_Abi, simd_abi::_Avx512Abi>())
      {
	if constexpr (_N == 8 && __have_avx512bw_vl)
	  {
	    _mm_mask_cvtepi16_storeu_epi8(__mem, __k,
					  _mm_maskz_set1_epi16(__v, 1));
	  }
	else if constexpr (_N == 8 && __have_avx512vl)
	  {
	    _mm256_mask_cvtepi32_storeu_epi8(__mem, __k,
					     _mm256_maskz_set1_epi32(__v, 1));
	  }
	else if constexpr (_N == 8)
	  {
	    // we rely on __k < 0x100:
	    _mm512_mask_cvtepi32_storeu_epi8(__mem, __k,
					     _mm512_maskz_set1_epi32(__v, 1));
	  }
	else if constexpr (_N == 16 && __have_avx512bw_vl)
	  {
	    _mm_mask_storeu_epi8(__mem, __k, _mm_maskz_set1_epi8(__v, 1));
	  }
	else if constexpr (_N == 16)
	  {
	    _mm512_mask_cvtepi32_storeu_epi8(__mem, __k,
					     _mm512_maskz_set1_epi32(__v, 1));
	  }
	else if constexpr (_N == 32 && __have_avx512bw_vl)
	  {
	    _mm256_mask_storeu_epi8(__mem, __k, _mm256_maskz_set1_epi8(__v, 1));
	  }
	else if constexpr (_N == 32 && __have_avx512bw)
	  {
	    _mm256_mask_storeu_epi8(__mem, __k,
				    __lo256(_mm512_maskz_set1_epi8(__v, 1)));
	  }
	else if constexpr (_N == 64 && __have_avx512bw)
	  {
	    _mm512_mask_storeu_epi8(__mem, __k, _mm512_maskz_set1_epi8(__v, 1));
	  }
	else
	  {
	    __assert_unreachable<_Tp>();
	  }
      }
    else
      {
	_Base::__masked_store(__v, __mem, _F(), __k);
      }
  }

  // logical and bitwise operators {{{2
  template <class _Tp, size_t _N>
  _GLIBCXX_SIMD_INTRINSIC static constexpr _SimdWrapper<_Tp, _N>
    __logical_and(const _SimdWrapper<_Tp, _N>& __x,
		const _SimdWrapper<_Tp, _N>& __y)
  {
    if constexpr (std::is_same_v<_Tp, bool>)
      {
	if constexpr (__have_avx512dq && _N <= 8)
	  return _kand_mask8(__x._M_data, __y._M_data);
	else if constexpr (_N <= 16)
	  return _kand_mask16(__x._M_data, __y._M_data);
	else if constexpr (__have_avx512bw && _N <= 32)
	  return _kand_mask32(__x._M_data, __y._M_data);
	else if constexpr (__have_avx512bw && _N <= 64)
	  return _kand_mask64(__x._M_data, __y._M_data);
	else
	  __assert_unreachable<_Tp>();
      }
    else
      return _Base::__logical_and(__x, __y);
  }

  template <class _Tp, size_t _N>
  _GLIBCXX_SIMD_INTRINSIC static constexpr _SimdWrapper<_Tp, _N>
    __logical_or(const _SimdWrapper<_Tp, _N>& __x,
	       const _SimdWrapper<_Tp, _N>& __y)
  {
    if constexpr (std::is_same_v<_Tp, bool>)
      {
	if constexpr (__have_avx512dq && _N <= 8)
	  return _kor_mask8(__x._M_data, __y._M_data);
	else if constexpr (_N <= 16)
	  return _kor_mask16(__x._M_data, __y._M_data);
	else if constexpr (__have_avx512bw && _N <= 32)
	  return _kor_mask32(__x._M_data, __y._M_data);
	else if constexpr (__have_avx512bw && _N <= 64)
	  return _kor_mask64(__x._M_data, __y._M_data);
	else
	  __assert_unreachable<_Tp>();
      }
    else
      return _Base::__logical_or(__x, __y);
  }

  template <class _Tp, size_t _N>
  _GLIBCXX_SIMD_INTRINSIC static constexpr _SimdWrapper<_Tp, _N>
    __bit_and(const _SimdWrapper<_Tp, _N>& __x, const _SimdWrapper<_Tp, _N>& __y)
  {
    if constexpr (std::is_same_v<_Tp, bool>)
      {
	if constexpr (__have_avx512dq && _N <= 8)
	  return _kand_mask8(__x._M_data, __y._M_data);
	else if constexpr (_N <= 16)
	  return _kand_mask16(__x._M_data, __y._M_data);
	else if constexpr (__have_avx512bw && _N <= 32)
	  return _kand_mask32(__x._M_data, __y._M_data);
	else if constexpr (__have_avx512bw && _N <= 64)
	  return _kand_mask64(__x._M_data, __y._M_data);
	else
	  __assert_unreachable<_Tp>();
      }
    else
      return _Base::__bit_and(__x, __y);
  }

  template <class _Tp, size_t _N>
  _GLIBCXX_SIMD_INTRINSIC static constexpr _SimdWrapper<_Tp, _N>
    __bit_or(const _SimdWrapper<_Tp, _N>& __x, const _SimdWrapper<_Tp, _N>& __y)
  {
    if constexpr (std::is_same_v<_Tp, bool>)
      {
	if constexpr (__have_avx512dq && _N <= 8)
	  return _kor_mask8(__x._M_data, __y._M_data);
	else if constexpr (_N <= 16)
	  return _kor_mask16(__x._M_data, __y._M_data);
	else if constexpr (__have_avx512bw && _N <= 32)
	  return _kor_mask32(__x._M_data, __y._M_data);
	else if constexpr (__have_avx512bw && _N <= 64)
	  return _kor_mask64(__x._M_data, __y._M_data);
	else
	  __assert_unreachable<_Tp>();
      }
    else
      return _Base::__bit_or(__x, __y);
  }

  template <class _Tp, size_t _N>
  _GLIBCXX_SIMD_INTRINSIC static constexpr _SimdWrapper<_Tp, _N>
    __bit_xor(const _SimdWrapper<_Tp, _N>& __x, const _SimdWrapper<_Tp, _N>& __y)
  {
    if constexpr (std::is_same_v<_Tp, bool>)
      {
	if constexpr (__have_avx512dq && _N <= 8)
	  return _kxor_mask8(__x._M_data, __y._M_data);
	else if constexpr (_N <= 16)
	  return _kxor_mask16(__x._M_data, __y._M_data);
	else if constexpr (__have_avx512bw && _N <= 32)
	  return _kxor_mask32(__x._M_data, __y._M_data);
	else if constexpr (__have_avx512bw && _N <= 64)
	  return _kxor_mask64(__x._M_data, __y._M_data);
	else
	  __assert_unreachable<_Tp>();
      }
    else
      return _Base::__bit_xor(__x, __y);
  }

  //}}}2
};

//}}}1

struct _MaskImplSse : __x86_mask_impl<simd_abi::__sse> {};
struct _SimdImplSse : __x86_simd_impl<simd_abi::__sse> {};

struct _MaskImplAvx : __x86_mask_impl<simd_abi::__avx> {};
struct _SimdImplAvx : __x86_simd_impl<simd_abi::__avx> {};

struct _SimdImplAvx512 : __x86_simd_impl<simd_abi::__avx512> {};
struct _MaskImplAvx512 : __x86_mask_impl<simd_abi::__avx512> {};
#endif // _GLIBCXX_SIMD_X86INTRIN }}}

#if _GLIBCXX_SIMD_HAVE_NEON // {{{
// _SimdImplNeon {{{
template <int _Bytes>
struct _SimdImplNeon : _SimdImplBuiltin<simd_abi::_NeonAbi<_Bytes>>
{
  using _Base = _SimdImplBuiltin<simd_abi::_NeonAbi<_Bytes>>;
  // math {{{
  // __sqrt {{{
  template <typename _Tp, typename _TVT = _VectorTraits<_Tp>>
  _GLIBCXX_SIMD_INTRINSIC static _Tp __sqrt(_Tp __x)
  {
    const auto __intrin = __to_intrin(__x);
    if constexpr (_TVT::template __is<float, 2>)
      return vsqrt_f32(__intrin);
    else if constexpr (_TVT::template __is<float, 4>)
      return vsqrtq_f32(__intrin);
    else if constexpr (_TVT::template __is<double, 1>)
      return vsqrt_f64(__intrin);
    else if constexpr (_TVT::template __is<double, 2>)
      return vsqrtq_f64(__intrin);
    else
      return _Base::__sqrt(__x);
  } // }}}
  // __trunc {{{
  template <typename _Tp, typename _TVT = _VectorTraits<_Tp>>
  _GLIBCXX_SIMD_INTRINSIC static _Tp __trunc(_Tp __x)
  {
    const auto __intrin = __to_intrin(__x);
    if constexpr (_TVT::template __is<float, 2>)
      return vrnd_f32(__intrin);
    else if constexpr (_TVT::template __is<float, 4>)
      return vrndq_f32(__intrin);
    else if constexpr (_TVT::template __is<double, 1>)
      return vrnd_f64(__intrin);
    else if constexpr (_TVT::template __is<double, 2>)
      return vrndq_f64(__intrin);
    else
      return _Base::__trunc(__x);
  } // }}}
  // __floor {{{
  template <typename _Tp, typename _TVT = _VectorTraits<_Tp>>
  _GLIBCXX_SIMD_INTRINSIC static _Tp __floor(_Tp __x)
  {
    const auto __intrin = __to_intrin(__x);
    if constexpr (_TVT::template __is<float, 2>)
      return vrndm_f32(__intrin);
    else if constexpr (_TVT::template __is<float, 4>)
      return vrndmq_f32(__intrin);
    else if constexpr (_TVT::template __is<double, 1>)
      return vrndm_f64(__intrin);
    else if constexpr (_TVT::template __is<double, 2>)
      return vrndmq_f64(__intrin);
    else
      return _Base::__floor(__x);
  } // }}}
  // __ceil {{{
  template <typename _Tp, typename _TVT = _VectorTraits<_Tp>>
  _GLIBCXX_SIMD_INTRINSIC static _Tp __ceil(_Tp __x)
  {
    const auto __intrin = __to_intrin(__x);
    if constexpr (_TVT::template __is<float, 2>)
      return vrndp_f32(__intrin);
    else if constexpr (_TVT::template __is<float, 4>)
      return vrndpq_f32(__intrin);
    else if constexpr (_TVT::template __is<double, 1>)
      return vrndp_f64(__intrin);
    else if constexpr (_TVT::template __is<double, 2>)
      return vrndpq_f64(__intrin);
    else
      return _Base::__ceil(__x);
  } //}}}
  //}}}
}; // }}}
// _MaskImplNeon {{{
template <int _Bytes>
struct _MaskImplNeon : __generic_mask_impl<simd_abi::_NeonAbi<_Bytes>>
{
}; // }}}
#endif // _GLIBCXX_SIMD_HAVE_NEON }}}

/**
 * The fixed_size ABI gives the following guarantees:
 *  - simd objects are passed via the stack
 *  - memory layout of `simd<_Tp, _N>` is equivalent to `std::array<_Tp, _N>`
 *  - alignment of `simd<_Tp, _N>` is `_N * sizeof(_Tp)` if _N is __a power-of-2 value,
 *    otherwise `__next_power_of_2(_N * sizeof(_Tp))` (Note: if the alignment were to
 *    exceed the system/compiler maximum, it is bounded to that maximum)
 *  - simd_mask objects are passed like std::bitset<_N>
 *  - memory layout of `simd_mask<_Tp, _N>` is equivalent to `std::bitset<_N>`
 *  - alignment of `simd_mask<_Tp, _N>` is equal to the alignment of `std::bitset<_N>`
 */
// __autocvt_to_simd {{{
template <class _Tp, bool = std::is_arithmetic_v<__remove_cvref_t<_Tp>>>
struct __autocvt_to_simd {
    _Tp _M_data;
    using _TT = __remove_cvref_t<_Tp>;
    operator _TT() { return _M_data; }
    operator _TT &()
    {
        static_assert(std::is_lvalue_reference<_Tp>::value, "");
        static_assert(!std::is_const<_Tp>::value, "");
        return _M_data;
    }
    operator _TT *()
    {
        static_assert(std::is_lvalue_reference<_Tp>::value, "");
        static_assert(!std::is_const<_Tp>::value, "");
        return &_M_data;
    }

    constexpr inline __autocvt_to_simd(_Tp dd) : _M_data(dd) {}

    template <class _Abi> operator simd<typename _TT::value_type, _Abi>()
    {
        return {__private_init, _M_data};
    }

    template <class _Abi> operator simd<typename _TT::value_type, _Abi> &()
    {
        return *reinterpret_cast<simd<typename _TT::value_type, _Abi> *>(&_M_data);
    }

    template <class _Abi> operator simd<typename _TT::value_type, _Abi> *()
    {
        return reinterpret_cast<simd<typename _TT::value_type, _Abi> *>(&_M_data);
    }
};
template <class _Tp> __autocvt_to_simd(_Tp &&)->__autocvt_to_simd<_Tp>;

template <class _Tp> struct __autocvt_to_simd<_Tp, true> {
    using _TT = __remove_cvref_t<_Tp>;
    _Tp _M_data;
    fixed_size_simd<_TT, 1> _M_fd;

    constexpr inline __autocvt_to_simd(_Tp dd) : _M_data(dd), _M_fd(_M_data) {}
    ~__autocvt_to_simd()
    {
        _M_data = __data(_M_fd).first;
    }

    operator fixed_size_simd<_TT, 1>()
    {
        return _M_fd;
    }
    operator fixed_size_simd<_TT, 1> &()
    {
        static_assert(std::is_lvalue_reference<_Tp>::value, "");
        static_assert(!std::is_const<_Tp>::value, "");
        return _M_fd;
    }
    operator fixed_size_simd<_TT, 1> *()
    {
        static_assert(std::is_lvalue_reference<_Tp>::value, "");
        static_assert(!std::is_const<_Tp>::value, "");
        return &_M_fd;
    }
};

// }}}
// __fixed_size_storage_t<_Tp, _N>{{{1
template <class _Tp, int _N, class _Tuple,
          class _Next = simd<_Tp, _AllNativeAbis::_BestAbi<_Tp, _N>>,
          int _Remain = _N - int(_Next::size())>
struct __fixed_size_storage_builder;

template <class _Tp, int _N>
struct __fixed_size_storage
    : public __fixed_size_storage_builder<_Tp, _N, _SimdTuple<_Tp>> {
};

template <class _Tp, int _N, class... _As, class _Next>
struct __fixed_size_storage_builder<_Tp, _N, _SimdTuple<_Tp, _As...>, _Next, 0> {
    using type = _SimdTuple<_Tp, _As..., typename _Next::abi_type>;
};

template <class _Tp, int _N, class... _As, class _Next, int _Remain>
struct __fixed_size_storage_builder<_Tp, _N, _SimdTuple<_Tp, _As...>, _Next, _Remain> {
    using type = typename __fixed_size_storage_builder<
        _Tp, _Remain, _SimdTuple<_Tp, _As..., typename _Next::abi_type>>::type;
};

// _AbisInSimdTuple {{{1
template <class _Tp> struct _SeqOp;
template <size_t _I0, size_t... _Is> struct _SeqOp<std::index_sequence<_I0, _Is...>> {
    using _FirstPlusOne = std::index_sequence<_I0 + 1, _Is...>;
    using _NotFirstPlusOne = std::index_sequence<_I0, (_Is + 1)...>;
    template <size_t _First, size_t _Add>
    using _Prepend = std::index_sequence<_First, _I0 + _Add, (_Is + _Add)...>;
};

template <class _Tp> struct _AbisInSimdTuple;
template <class _Tp> struct _AbisInSimdTuple<_SimdTuple<_Tp>> {
    using _Counts = std::index_sequence<0>;
    using _Begins = std::index_sequence<0>;
};
template <class _Tp, class _A> struct _AbisInSimdTuple<_SimdTuple<_Tp, _A>> {
    using _Counts = std::index_sequence<1>;
    using _Begins = std::index_sequence<0>;
};
template <class _Tp, class _A0, class... _As>
struct _AbisInSimdTuple<_SimdTuple<_Tp, _A0, _A0, _As...>> {
    using _Counts = typename _SeqOp<typename _AbisInSimdTuple<
        _SimdTuple<_Tp, _A0, _As...>>::_Counts>::_FirstPlusOne;
    using _Begins = typename _SeqOp<typename _AbisInSimdTuple<
        _SimdTuple<_Tp, _A0, _As...>>::_Begins>::_NotFirstPlusOne;
};
template <class _Tp, class _A0, class _A1, class... _As>
struct _AbisInSimdTuple<_SimdTuple<_Tp, _A0, _A1, _As...>> {
    using _Counts = typename _SeqOp<typename _AbisInSimdTuple<
        _SimdTuple<_Tp, _A1, _As...>>::_Counts>::template _Prepend<1, 0>;
    using _Begins = typename _SeqOp<typename _AbisInSimdTuple<
        _SimdTuple<_Tp, _A1, _As...>>::_Begins>::template _Prepend<0, 1>;
};

// _BinaryTreeReduce {{{1
template <size_t _Count, size_t _Begin> struct _BinaryTreeReduce {
    static_assert(_Count > 0,
                  "_BinaryTreeReduce requires at least one simd object to work with");
    template <class _Tp, class... _As, class _BinaryOperation>
    auto operator()(const _SimdTuple<_Tp, _As...> &__tup,
                    const _BinaryOperation &__binary_op) const noexcept
    {
        constexpr size_t __left = __next_power_of_2(_Count) / 2;
        constexpr size_t __right = _Count - __left;
        return __binary_op(_BinaryTreeReduce<__left, _Begin>()(__tup, __binary_op),
                         _BinaryTreeReduce<__right, _Begin + __left>()(__tup, __binary_op));
    }
};
template <size_t _Begin> struct _BinaryTreeReduce<1, _Begin> {
    template <class _Tp, class... _As, class _BinaryOperation>
    auto operator()(const _SimdTuple<_Tp, _As...> &__tup, const _BinaryOperation &) const
        noexcept
    {
        return __get_simd_at<_Begin>(__tup);
    }
};
template <size_t _Begin> struct _BinaryTreeReduce<2, _Begin> {
    template <class _Tp, class... _As, class _BinaryOperation>
    auto operator()(const _SimdTuple<_Tp, _As...> &__tup,
                    const _BinaryOperation &__binary_op) const noexcept
    {
        return __binary_op(__get_simd_at<_Begin>(__tup),
                         __get_simd_at<_Begin + 1>(__tup));
    }
};

// __vec_to_scalar_reduction {{{1
// This helper function implements the second step in a generic fixed_size reduction.
// -  Input: a tuple of native simd (or scalar) objects of decreasing size.
// - Output: a scalar (the reduction).
// - Approach:
//   1. reduce the first two tuple elements
//      a) If the number of elements differs by a factor of 2, split the first object into
//         two objects of the second type and reduce all three to one object of second
//         type.
//      b) If the number of elements differs by a factor of 4, split the first object into
//         two equally sized objects, reduce, and split to two objects of the second type.
//         Finally, reduce all three remaining objects to one object of second type.
//      c) Otherwise use std::experimental::reduce to reduce both inputs to a scalar, and binary_op to
//         reduce to a single scalar.
//
//      (This optimizes all native cases on x86, e.g. <AVX512, SSE, Scalar>.)
//
//   2. Concate the result of (1) with the remaining tuple elements to recurse into
//      __vec_to_scalar_reduction.
//
//   3. If __vec_to_scalar_reduction is called with a one-element tuple, call std::experimental::reduce to
//      reduce to a scalar and return.
template <class _Tp, class _A0, class _A1, class _BinaryOperation>
_GLIBCXX_SIMD_INTRINSIC simd<_Tp, _A1> __vec_to_scalar_reduction_first_pair(
    const simd<_Tp, _A0> __left, const simd<_Tp, _A1> __right, const _BinaryOperation &__binary_op,
    _SizeConstant<2>) noexcept
{
    const std::array<simd<_Tp, _A1>, 2> __splitted = split<simd<_Tp, _A1>>(__left);
    return __binary_op(__binary_op(__splitted[0], __right), __splitted[1]);
}

template <class _Tp, class _A0, class _A1, class _BinaryOperation>
_GLIBCXX_SIMD_INTRINSIC simd<_Tp, _A1> __vec_to_scalar_reduction_first_pair(
    const simd<_Tp, _A0> __left, const simd<_Tp, _A1> __right, const _BinaryOperation &__binary_op,
    _SizeConstant<4>) noexcept
{
    constexpr auto _N0 = simd_size_v<_Tp, _A0> / 2;
    const auto __left2 = split<simd<_Tp, simd_abi::deduce_t<_Tp, _N0>>>(__left);
    const std::array<simd<_Tp, _A1>, 2> __splitted =
        split<simd<_Tp, _A1>>(__binary_op(__left2[0], __left2[1]));
    return __binary_op(__binary_op(__splitted[0], __right), __splitted[1]);
}

template <class _Tp, class _A0, class _A1, class _BinaryOperation, size_t _Factor>
_GLIBCXX_SIMD_INTRINSIC simd<_Tp, simd_abi::scalar> __vec_to_scalar_reduction_first_pair(
    const simd<_Tp, _A0> __left, const simd<_Tp, _A1> __right, const _BinaryOperation &__binary_op,
    _SizeConstant<_Factor>) noexcept
{
    return __binary_op(std::experimental::reduce(__left, __binary_op), std::experimental::reduce(__right, __binary_op));
}

template <class _Tp, class _A0, class _BinaryOperation>
_GLIBCXX_SIMD_INTRINSIC _Tp __vec_to_scalar_reduction(const _SimdTuple<_Tp, _A0> &__tup,
                                       const _BinaryOperation &__binary_op) noexcept
{
    return std::experimental::reduce(simd<_Tp, _A0>(__private_init, __tup.first), __binary_op);
}

template <class _Tp, class _A0, class _A1, class... _As, class _BinaryOperation>
_GLIBCXX_SIMD_INTRINSIC _Tp __vec_to_scalar_reduction(const _SimdTuple<_Tp, _A0, _A1, _As...> &__tup,
                                       const _BinaryOperation &__binary_op) noexcept
{
    return __vec_to_scalar_reduction(
        __simd_tuple_concat(
            __make_simd_tuple(
                __vec_to_scalar_reduction_first_pair<_Tp, _A0, _A1, _BinaryOperation>(
                    {__private_init, __tup.first}, {__private_init, __tup.second.first},
                    __binary_op,
                    _SizeConstant<simd_size_v<_Tp, _A0> / simd_size_v<_Tp, _A1>>())),
            __tup.second.second),
        __binary_op);
}

// _SimdImplFixedSize {{{1
// fixed_size should not inherit from _SimdMathFallback in order for
// specializations in the used _SimdTuple Abis to get used
template <int _N> struct _SimdImplFixedSize {
    // member types {{{2
    using _MaskMember = std::bitset<_N>;
    template <class _Tp> using _SimdMember = __fixed_size_storage_t<_Tp, _N>;
    template <class _Tp>
    static constexpr std::size_t _S_tuple_size = _SimdMember<_Tp>::_S_tuple_size;
    template <class _Tp> using _Simd = std::experimental::simd<_Tp, simd_abi::fixed_size<_N>>;
    template <class _Tp> using _TypeTag = _Tp *;

    // broadcast {{{2
    template <class _Tp> static constexpr inline _SimdMember<_Tp> __broadcast(_Tp __x) noexcept
    {
        return _SimdMember<_Tp>::__generate(
            [&](auto __meta) constexpr { return __meta.__broadcast(__x); });
    }

    // __generator {{{2
    template <class _F, class _Tp>
    static constexpr inline _SimdMember<_Tp>
      __generator(_F&& __gen, _TypeTag<_Tp>)
    {
      return _SimdMember<_Tp>::__generate([&__gen](auto __meta) constexpr {
	return __meta.__generator(
	  [&](auto __i) constexpr {
	    static_assert(__i < _N);
	    return __gen(_SizeConstant<__meta._S_offset + __i>());
	  },
	  _TypeTag<_Tp>());
      });
    }

    // __load {{{2
    template <class _Tp, class _U, class _F>
    static inline _SimdMember<_Tp> __load(const _U *__mem, _F __f,
                                              _TypeTag<_Tp>) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
        return _SimdMember<_Tp>::__generate(
            [&](auto __meta) { return __meta.__load(&__mem[__meta._S_offset], __f, _TypeTag<_Tp>()); });
    }

    // __masked_load {{{2
    template <class _Tp, class... _As, class _U, class _F>
    static inline _SimdTuple<_Tp, _As...>
      __masked_load(const _SimdTuple<_Tp, _As...>& __old,
		  const _MaskMember          __bits,
		  const _U*                        __mem,
		  _F __f) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
      auto __merge = __old;
      __for_each(__merge, [&](auto __meta, auto& __native) {
	__native = __meta.__masked_load(__native, __meta.__make_mask(__bits),
				  &__mem[__meta._S_offset], __f);
      });
      return __merge;
    }

    // __store {{{2
    template <class _Tp, class _U, class _F>
    static inline void __store(const _SimdMember<_Tp>& __v,
			     _U*                           __mem,
			     _F                            __f,
			     _TypeTag<_Tp>) _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
      __for_each(__v, [&](auto __meta, auto __native) {
	__meta.__store(__native, &__mem[__meta._S_offset], __f, _TypeTag<_Tp>());
      });
    }

    // __masked_store {{{2
    template <class _Tp, class... _As, class _U, class _F>
    static inline void __masked_store(const _SimdTuple<_Tp, _As...>& __v,
				    _U*                              __mem,
				    _F                               __f,
				    const _MaskMember          __bits)
      _GLIBCXX_SIMD_NOEXCEPT_OR_IN_TEST
    {
      __for_each(__v, [&](auto __meta, auto __native) {
	__meta.__masked_store(__native, &__mem[__meta._S_offset], __f,
			  __meta.__make_mask(__bits));
      });
    }

    // negation {{{2
    template <class _Tp, class... _As>
    static inline _MaskMember
      __negate(const _SimdTuple<_Tp, _As...>& __x) noexcept
    {
        _MaskMember __bits = 0;
        __for_each(__x, [&__bits](auto __meta, auto __native) constexpr {
            __bits |= __meta.__mask_to_shifted_ullong(__meta.__negate(__native));
        });
        return __bits;
    }

    // reductions {{{2
private:
    template <class _Tp, class... _As, class _BinaryOperation, size_t... _Counts,
              size_t... _Begins>
    static inline _Tp __reduce(const _SimdTuple<_Tp, _As...> &__tup,
                           const _BinaryOperation &__binary_op,
                           std::index_sequence<_Counts...>, std::index_sequence<_Begins...>)
    {
        // 1. reduce all tuple elements with equal ABI to a single element in the output
        // tuple
        const auto __reduced_vec =
            __make_simd_tuple(_BinaryTreeReduce<_Counts, _Begins>()(__tup, __binary_op)...);
        // 2. split and reduce until a scalar results
        return __vec_to_scalar_reduction(__reduced_vec, __binary_op);
    }

public:
    template <class _Tp, class _BinaryOperation>
    static inline _Tp __reduce(const _Simd<_Tp> &__x, const _BinaryOperation &__binary_op)
    {
        using _Ranges = _AbisInSimdTuple<_SimdMember<_Tp>>;
        return _SimdImplFixedSize::__reduce(__x._M_data, __binary_op,
                                              typename _Ranges::_Counts(),
                                              typename _Ranges::_Begins());
    }

    // __min, __max {{{2
    template <typename _Tp, typename... _As>
    static inline constexpr _SimdTuple<_Tp, _As...>
      __min(const _SimdTuple<_Tp, _As...>& __a,
	  const _SimdTuple<_Tp, _As...>& __b)
    {
      return __a.__apply_per_chunk(
	[](auto __impl, auto __aa, auto __bb) constexpr {
	  return __impl.__min(__aa, __bb);
	},
	__b);
    }

    template <typename _Tp, typename... _As>
    static inline constexpr _SimdTuple<_Tp, _As...>
      __max(const _SimdTuple<_Tp, _As...>& __a,
	  const _SimdTuple<_Tp, _As...>& __b)
    {
      return __a.__apply_per_chunk(
	[](auto __impl, auto __aa, auto __bb) constexpr {
	  return __impl.__max(__aa, __bb);
	},
	__b);
    }

    // __complement {{{2
    template <typename _Tp, typename... _As>
    static inline constexpr _SimdTuple<_Tp, _As...>
      __complement(const _SimdTuple<_Tp, _As...>& __x) noexcept
    {
      return __x.__apply_per_chunk([](auto __impl, auto __xx) constexpr {
	return __impl.__complement(__xx);
      });
    }

    // __unary_minus {{{2
    template <typename _Tp, typename... _As>
    static inline constexpr _SimdTuple<_Tp, _As...>
      __unary_minus(const _SimdTuple<_Tp, _As...>& __x) noexcept
    {
      return __x.__apply_per_chunk([](auto __impl, auto __xx) constexpr {
	return __impl.__unary_minus(__xx);
      });
    }

    // arithmetic operators {{{2

#define _GLIBCXX_SIMD_FIXED_OP(name_, op_)                                     \
  template <typename _Tp, typename... _As>                                     \
  static inline constexpr _SimdTuple<_Tp, _As...> name_(                       \
    const _SimdTuple<_Tp, _As...>& __x, const _SimdTuple<_Tp, _As...>& __y)    \
  {                                                                            \
    return __x.__apply_per_chunk(                                              \
      [](auto __impl, auto __xx, auto __yy) constexpr {                        \
	return __impl.name_(__xx, __yy);                                       \
      },                                                                       \
      __y);                                                                    \
  }

    _GLIBCXX_SIMD_FIXED_OP(__plus, +)
    _GLIBCXX_SIMD_FIXED_OP(__minus, -)
    _GLIBCXX_SIMD_FIXED_OP(__multiplies, *)
    _GLIBCXX_SIMD_FIXED_OP(__divides, /)
    _GLIBCXX_SIMD_FIXED_OP(__modulus, %)
    _GLIBCXX_SIMD_FIXED_OP(__bit_and, &)
    _GLIBCXX_SIMD_FIXED_OP(__bit_or, |)
    _GLIBCXX_SIMD_FIXED_OP(__bit_xor, ^)
    _GLIBCXX_SIMD_FIXED_OP(__bit_shift_left, <<)
    _GLIBCXX_SIMD_FIXED_OP(__bit_shift_right, >>)
#undef _GLIBCXX_SIMD_FIXED_OP

    template <typename _Tp, typename... _As>
    static inline constexpr _SimdTuple<_Tp, _As...>
      __bit_shift_left(const _SimdTuple<_Tp, _As...>& __x, int __y)
    {
      return __x.__apply_per_chunk([__y](auto __impl, auto __xx) constexpr {
	return __impl.__bit_shift_left(__xx, __y);
      });
    }

    template <typename _Tp, typename... _As>
    static inline constexpr _SimdTuple<_Tp, _As...>
      __bit_shift_right(const _SimdTuple<_Tp, _As...>& __x, int __y)
    {
      return __x.__apply_per_chunk([__y](auto __impl, auto __xx) constexpr {
	return __impl.__bit_shift_right(__xx, __y);
      });
    }

    // math {{{2
#define _GLIBCXX_SIMD_APPLY_ON_TUPLE(_RetTp, __name)                           \
  template <typename _Tp, typename... _As, typename... _More>                  \
  static inline __fixed_size_storage_t<_RetTp,                                 \
				       _SimdTuple<_Tp, _As...>::size()>        \
    __##__name(const _SimdTuple<_Tp, _As...>& __x, const _More&... __more)     \
  {                                                                            \
    if constexpr (sizeof...(_More) == 0)                                       \
      {                                                                        \
	if constexpr (is_same_v<_Tp, _RetTp>)                                  \
	  return __x.__apply_per_chunk([](auto __impl, auto __xx) constexpr {  \
	    using _V = typename decltype(__impl)::simd_type;                   \
	    return __data(__name(_V(__private_init, __xx)));                   \
	  });                                                                  \
	else                                                                   \
	  return __optimize_simd_tuple(__x.template __apply_r<_RetTp>(         \
	    [](auto __impl, auto __xx) { return __impl.__##__name(__xx); }));  \
      }                                                                        \
    else if constexpr (is_same_v<_Tp, _RetTp> &&                               \
		       (... &&                                                 \
			std::is_same_v<_SimdTuple<_Tp, _As...>, _More>))       \
      return __x.__apply_per_chunk(                                            \
	[](auto __impl, auto __xx, auto... __pack) constexpr {                 \
	  using _V = typename decltype(__impl)::simd_type;                     \
	  return __data(                                                       \
	    __name(_V(__private_init, __xx), _V(__private_init, __pack)...));  \
	},                                                                     \
	__more...);                                                            \
    else if constexpr (is_same_v<_Tp, _RetTp>)                                 \
      return __x.__apply_per_chunk(                                            \
	[](auto __impl, auto __xx, auto... __pack) constexpr {                 \
	  using _V = typename decltype(__impl)::simd_type;                     \
	  return __data(                                                       \
	    __name(_V(__private_init, __xx), __autocvt_to_simd(__pack)...));   \
	},                                                                     \
	__more...);                                                            \
    else                                                                       \
      __assert_unreachable<_Tp>();                                             \
  }
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, acos)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, asin)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, atan)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, atan2)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, cos)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, sin)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, tan)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, acosh)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, asinh)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, atanh)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, cosh)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, sinh)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, tanh)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, exp)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, exp2)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, expm1)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(int, ilogb)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, log)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, log10)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, log1p)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, log2)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, logb)
    //modf implemented in simd_math.h
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, scalbn) //double scalbn(double x, int exp);
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, scalbln)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, cbrt)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, abs)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, fabs)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, pow)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, sqrt)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, erf)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, erfc)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, lgamma)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, tgamma)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, trunc)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, ceil)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, floor)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, nearbyint)

    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, rint)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(long, lrint)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(long long, llrint)

    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, round)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(long, lround)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(long long, llround)

    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, ldexp)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, fmod)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, remainder)
    // copysign in simd_math.h
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, nextafter)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, fdim)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, fmax)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, fmin)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(_Tp, fma)
    _GLIBCXX_SIMD_APPLY_ON_TUPLE(int, fpclassify)
#undef _GLIBCXX_SIMD_APPLY_ON_TUPLE

    template <typename _Tp, typename... _Abis>
    static _SimdTuple<_Tp, _Abis...> __remquo(
      const _SimdTuple<_Tp, _Abis...>&                                __x,
      const _SimdTuple<_Tp, _Abis...>&                                __y,
      __fixed_size_storage_t<int, _SimdTuple<_Tp, _Abis...>::size()>* __z)
    {
      return __x.__apply_per_chunk(
	[](auto __impl, const auto __xx, const auto __yy, auto& __zz) {
	  return __impl.__remquo(__xx, __yy, &__zz);
	},
	__y, *__z);
    }

    template <typename _Tp, typename... _As>
    static inline _SimdTuple<_Tp, _As...>
      __frexp(const _SimdTuple<_Tp, _As...>&   __x,
	      __fixed_size_storage_t<int, _N>& __exp) noexcept
    {
      return __x.__apply_per_chunk(
	[](auto __impl, const auto& __a, auto& __b) {
	  return __data(
	    frexp(typename decltype(__impl)::simd_type(__private_init, __a),
		  __autocvt_to_simd(__b)));
	},
	__exp);
    }

    template <typename _Tp, typename... _As>
    static inline __fixed_size_storage_t<int, _N>
      __fpclassify(const _SimdTuple<_Tp, _As...>& __x) noexcept
    {
      return __optimize_simd_tuple(__x.template __apply_r<int>(
	[](auto __impl, auto __xx) { return __impl.__fpclassify(__xx); }));
    }

#define _GLIBCXX_SIMD_TEST_ON_TUPLE_(name_)                                    \
  template <typename _Tp, typename... _As>                                     \
  static inline _MaskMember __##name_(                                   \
    const _SimdTuple<_Tp, _As...>& __x) noexcept                             \
  {                                                                            \
    return __test([](auto __impl, auto __xx) { return __impl.__##name_(__xx); }, \
		__x);                                                          \
  }
    _GLIBCXX_SIMD_TEST_ON_TUPLE_(isinf)
    _GLIBCXX_SIMD_TEST_ON_TUPLE_(isfinite)
    _GLIBCXX_SIMD_TEST_ON_TUPLE_(isnan)
    _GLIBCXX_SIMD_TEST_ON_TUPLE_(isnormal)
    _GLIBCXX_SIMD_TEST_ON_TUPLE_(signbit)
#undef _GLIBCXX_SIMD_TEST_ON_TUPLE_

    // __increment & __decrement{{{2
    template <typename... _Ts>
    _GLIBCXX_SIMD_INTRINSIC static constexpr void
      __increment(_SimdTuple<_Ts...>& __x)
    {
      __for_each(
	__x, [](auto __meta, auto& native) constexpr {
	  __meta.__increment(native);
	});
    }

    template <typename... _Ts>
    _GLIBCXX_SIMD_INTRINSIC static constexpr void
      __decrement(_SimdTuple<_Ts...>& __x)
    {
      __for_each(
	__x, [](auto __meta, auto& native) constexpr {
	  __meta.__decrement(native);
	});
    }

    // compares {{{2
#define _GLIBCXX_SIMD_CMP_OPERATIONS(__cmp)                                    \
  template <typename _Tp, typename... _As>                                     \
  _GLIBCXX_SIMD_INTRINSIC static _MaskMember __cmp(                            \
    const _SimdTuple<_Tp, _As...>& __x, const _SimdTuple<_Tp, _As...>& __y)    \
  {                                                                            \
    return __test([](auto __impl, auto __xx,                                   \
		     auto __yy) { return __impl.__cmp(__xx, __yy); },          \
		  __x, __y);                                                   \
  }
    _GLIBCXX_SIMD_CMP_OPERATIONS(__equal_to)
    _GLIBCXX_SIMD_CMP_OPERATIONS(__not_equal_to)
    _GLIBCXX_SIMD_CMP_OPERATIONS(__less)
    _GLIBCXX_SIMD_CMP_OPERATIONS(__less_equal)
    _GLIBCXX_SIMD_CMP_OPERATIONS(__isless)
    _GLIBCXX_SIMD_CMP_OPERATIONS(__islessequal)
    _GLIBCXX_SIMD_CMP_OPERATIONS(__isgreater)
    _GLIBCXX_SIMD_CMP_OPERATIONS(__isgreaterequal)
    _GLIBCXX_SIMD_CMP_OPERATIONS(__islessgreater)
    _GLIBCXX_SIMD_CMP_OPERATIONS(__isunordered)
#undef _GLIBCXX_SIMD_CMP_OPERATIONS

    // smart_reference access {{{2
    template <typename _Tp, typename... _As, typename _U>
    _GLIBCXX_SIMD_INTRINSIC static void __set(_SimdTuple<_Tp, _As...> &__v, int __i, _U &&__x) noexcept
    {
        __v.__set(__i, std::forward<_U>(__x));
    }

    // __masked_assign {{{2
    template <typename _Tp, typename... _As>
    _GLIBCXX_SIMD_INTRINSIC static void
      __masked_assign(const _MaskMember                __bits,
		    _SimdTuple<_Tp, _As...>&             __lhs,
		    const __id<_SimdTuple<_Tp, _As...>>& __rhs)
    {
      __for_each(__lhs, __rhs,
		 [&](auto __meta, auto& __native_lhs, auto __native_rhs) constexpr {
		   __meta.__masked_assign(__meta.__make_mask(__bits), __native_lhs,
					__native_rhs);
		 });
    }

    // Optimization for the case where the RHS is a scalar. No need to broadcast the
    // scalar to a simd first.
    template <typename _Tp, typename... _As>
    _GLIBCXX_SIMD_INTRINSIC static void
      __masked_assign(const _MaskMember    __bits,
		    _SimdTuple<_Tp, _As...>& __lhs,
		    const __id<_Tp>            __rhs)
    {
      __for_each(__lhs, [&](auto __meta, auto& __native_lhs) constexpr {
	__meta.__masked_assign(__meta.__make_mask(__bits), __native_lhs, __rhs);
      });
    }

    // __masked_cassign {{{2
    template <template <typename> class _Op, typename _Tp, typename... _As>
    static inline void __masked_cassign(const _MaskMember          __bits,
					_SimdTuple<_Tp, _As...>&       __lhs,
					const _SimdTuple<_Tp, _As...>& __rhs)
    {
      __for_each(__lhs, __rhs,
		 [&](auto __meta, auto& __native_lhs, auto __native_rhs) constexpr {
		   __meta.template __masked_cassign<_Op>(
		     __meta.__make_mask(__bits), __native_lhs, __native_rhs);
		 });
    }

    // Optimization for the case where the RHS is a scalar. No need to broadcast
    // the scalar to a simd first.
    template <template <typename> class _Op, typename _Tp, typename... _As>
    static inline void __masked_cassign(const _MaskMember    __bits,
					_SimdTuple<_Tp, _As...>& __lhs,
					const _Tp&                 __rhs)
    {
      __for_each(__lhs, [&](auto __meta, auto& __native_lhs) constexpr {
	__meta.template __masked_cassign<_Op>(__meta.__make_mask(__bits),
					      __native_lhs, __rhs);
      });
    }

    // __masked_unary {{{2
    template <template <typename> class _Op, typename _Tp, typename... _As>
    static inline _SimdTuple<_Tp, _As...>
      __masked_unary(const _MaskMember         __bits,
		   const _SimdTuple<_Tp, _As...> __v) // TODO: const-ref __v?
    {
      return __v.__apply_wrapped([&__bits](auto __meta, auto __native) constexpr {
	return __meta.template __masked_unary<_Op>(__meta.__make_mask(__bits),
						 __native);
      });
    }

    // }}}2
};

// _MaskImplFixedSize {{{1
template <int _N> struct _MaskImplFixedSize {
    static_assert(sizeof(_ULLong) * CHAR_BIT >= _N,
                  "The fixed_size implementation relies on one "
                  "_ULLong being able to store all boolean "
                  "elements.");  // required in load & store

    // member types {{{2
    using _MaskMember = std::bitset<_N>;
    template <typename _Tp> using _TypeTag = _Tp *;

    // __from_bitset {{{2
    template <typename _Tp>
    _GLIBCXX_SIMD_INTRINSIC static _MaskMember __from_bitset(const _MaskMember &__bs,
                                                     _TypeTag<_Tp>) noexcept
    {
        return __bs;
    }

    // __load {{{2
    template <typename _F> static inline _MaskMember __load(const bool *__mem, _F __f) noexcept
    {
        // TODO: _UChar is not necessarily the best type to use here. For smaller _N ushort,
        // _UInt, _ULLong, float, and double can be more efficient.
        _ULLong __r = 0;
        using _Vs = __fixed_size_storage_t<_UChar, _N>;
        __for_each(_Vs{}, [&](auto __meta, auto) {
            __r |= __meta.__mask_to_shifted_ullong(
                __meta._S_mask_impl.__load(&__mem[__meta._S_offset], __f, _SizeConstant<__meta.size()>()));
        });
        return __r;
    }

    // __masked_load {{{2
    template <typename _F>
    static inline _MaskMember __masked_load(_MaskMember __merge,
                                               _MaskMember __mask, const bool *__mem,
                                               _F) noexcept
    {
        __bit_iteration(__mask.to_ullong(), [&](auto __i) { __merge[__i] = __mem[__i]; });
        return __merge;
    }

    // __store {{{2
    template <typename _F>
    static inline void __store(_MaskMember __bs, bool *__mem, _F __f) noexcept
    {
#if _GLIBCXX_SIMD_HAVE_AVX512BW
        const __m512i bool64 = _mm512_movm_epi8(__bs.to_ullong()) & 0x0101010101010101ULL;
        __vector_store<_N>(bool64, __mem, __f);
#elif _GLIBCXX_SIMD_HAVE_BMI2
#ifdef __x86_64__
        __unused(__f);
        __execute_n_times<_N / 8>([&](auto __i) {
            constexpr size_t __offset = __i * 8;
            const _ULLong bool8 =
                _pdep_u64(__bs.to_ullong() >> __offset, 0x0101010101010101ULL);
            std::memcpy(&__mem[__offset], &bool8, 8);
        });
        if (_N % 8 > 0) {
            constexpr size_t __offset = (_N / 8) * 8;
            const _ULLong bool8 =
                _pdep_u64(__bs.to_ullong() >> __offset, 0x0101010101010101ULL);
            std::memcpy(&__mem[__offset], &bool8, _N % 8);
        }
#else   // __x86_64__
        __unused(__f);
        __execute_n_times<_N / 4>([&](auto __i) {
            constexpr size_t __offset = __i * 4;
            const _ULLong __bool4 =
                _pdep_u32(__bs.to_ullong() >> __offset, 0x01010101U);
            std::memcpy(&__mem[__offset], &__bool4, 4);
        });
        if (_N % 4 > 0) {
            constexpr size_t __offset = (_N / 4) * 4;
            const _ULLong __bool4 =
                _pdep_u32(__bs.to_ullong() >> __offset, 0x01010101U);
            std::memcpy(&__mem[__offset], &__bool4, _N % 4);
        }
#endif  // __x86_64__
#elif  _GLIBCXX_SIMD_HAVE_SSE2   // !AVX512BW && !BMI2
        using _V = simd<_UChar, simd_abi::__sse>;
        _ULLong __bits = __bs.to_ullong();
        __execute_n_times<(_N + 15) / 16>([&](auto __i) {
            constexpr size_t __offset = __i * 16;
            constexpr size_t __remaining = _N - __offset;
            if constexpr (__remaining == 1) {
                __mem[__offset] = static_cast<bool>(__bits >> __offset);
            } else if constexpr (__remaining <= 4) {
                const _UInt __bool4 = ((__bits >> __offset) * 0x00204081U) & 0x01010101U;
                std::memcpy(&__mem[__offset], &__bool4, __remaining);
            } else if constexpr (__remaining <= 7) {
                const _ULLong bool8 =
                    ((__bits >> __offset) * 0x40810204081ULL) & 0x0101010101010101ULL;
                std::memcpy(&__mem[__offset], &bool8, __remaining);
            } else if constexpr (__have_sse2) {
                auto __tmp = _mm_cvtsi32_si128(__bits >> __offset);
                __tmp = _mm_unpacklo_epi8(__tmp, __tmp);
                __tmp = _mm_unpacklo_epi16(__tmp, __tmp);
                __tmp = _mm_unpacklo_epi32(__tmp, __tmp);
                _V __tmp2(__tmp);
                __tmp2 &= _V([](auto __j) {
                    return static_cast<_UChar>(1 << (__j % CHAR_BIT));
                });  // mask bit index
                const __m128i __bool16 = __intrin_bitcast<__m128i>(
                    __vector_bitcast<_UChar>(__data(__tmp2 == 0)) +
                    1);  // 0xff -> 0x00 | 0x00 -> 0x01
                if constexpr (__remaining >= 16) {
                    __vector_store<16>(__bool16, &__mem[__offset], __f);
                } else if constexpr (__remaining & 3) {
                    constexpr int to_shift = 16 - int(__remaining);
                    _mm_maskmoveu_si128(__bool16,
                                        _mm_srli_si128(~__m128i(), to_shift),
                                        reinterpret_cast<char *>(&__mem[__offset]));
                } else  // at this point: 8 < __remaining < 16
                    if constexpr (__remaining >= 8) {
                    __vector_store<8>(__bool16, &__mem[__offset], __f);
                    if constexpr (__remaining == 12) {
                        __vector_store<4>(_mm_unpackhi_epi64(__bool16, __bool16),
                                         &__mem[__offset + 8], __f);
                    }
                }
            } else {
                __assert_unreachable<_F>();
            }
        });
#else
        // TODO: _UChar is not necessarily the best type to use here. For smaller _N ushort,
        // _UInt, _ULLong, float, and double can be more efficient.
        using _Vs = __fixed_size_storage_t<_UChar, _N>;
        __for_each(_Vs{}, [&](auto __meta, auto) {
            __meta._S_mask_impl.__store(__meta.__make_mask(__bs), &__mem[__meta._S_offset], __f);
        });
//#else
        //__execute_n_times<_N>([&](auto __i) { __mem[__i] = __bs[__i]; });
#endif  // _GLIBCXX_SIMD_HAVE_BMI2
    }

    // __masked_store {{{2
    template <typename _F>
    static inline void __masked_store(const _MaskMember __v, bool *__mem, _F,
                                    const _MaskMember __k) noexcept
    {
        __bit_iteration(__k, [&](auto __i) { __mem[__i] = __v[__i]; });
    }

    // logical and bitwise operators {{{2
    _GLIBCXX_SIMD_INTRINSIC static _MaskMember __logical_and(const _MaskMember &__x,
                                                     const _MaskMember &__y) noexcept
    {
        return __x & __y;
    }

    _GLIBCXX_SIMD_INTRINSIC static _MaskMember __logical_or(const _MaskMember &__x,
                                                    const _MaskMember &__y) noexcept
    {
        return __x | __y;
    }

    _GLIBCXX_SIMD_INTRINSIC static _MaskMember __bit_and(const _MaskMember &__x,
                                                 const _MaskMember &__y) noexcept
    {
        return __x & __y;
    }

    _GLIBCXX_SIMD_INTRINSIC static _MaskMember __bit_or(const _MaskMember &__x,
                                                const _MaskMember &__y) noexcept
    {
        return __x | __y;
    }

    _GLIBCXX_SIMD_INTRINSIC static _MaskMember __bit_xor(const _MaskMember &__x,
                                                 const _MaskMember &__y) noexcept
    {
        return __x ^ __y;
    }

    // smart_reference access {{{2
    _GLIBCXX_SIMD_INTRINSIC static void __set(_MaskMember &__k, int __i, bool __x) noexcept
    {
        __k.set(__i, __x);
    }

    // __masked_assign {{{2
    _GLIBCXX_SIMD_INTRINSIC static void __masked_assign(const _MaskMember __k,
                                           _MaskMember &__lhs,
                                           const _MaskMember __rhs)
    {
        __lhs = (__lhs & ~__k) | (__rhs & __k);
    }

    // Optimization for the case where the RHS is a scalar.
    _GLIBCXX_SIMD_INTRINSIC static void __masked_assign(const _MaskMember __k,
                                           _MaskMember &__lhs, const bool __rhs)
    {
        if (__rhs) {
            __lhs |= __k;
        } else {
            __lhs &= ~__k;
        }
    }

    // }}}2
};
// }}}1

// _SimdConverter <From, A> -> <To, A> (same ABI) {{{
template <typename _From, typename _To, typename _Abi>
struct _SimdConverter<_From, _Abi, _To, _Abi>
{
  template <typename _Tp>
  using _SimdMember = typename _Abi::template __traits<_Tp>::_SimdMember;
  using _Arg = _SimdMember<_From>;
  using _Ret = _SimdMember<_To>;
  using _V = __vector_type_t<_To, simd_size_v<_To, _Abi>>;

  _GLIBCXX_SIMD_INTRINSIC decltype(auto) operator()(const _Arg& __a)
  {
    if constexpr (is_same_v<_To, _From>)
      return __a;
    else
      return __convert<_V>(__a);
  }
  template <typename... _More>
  _GLIBCXX_SIMD_INTRINSIC _Ret operator()(_Arg __a, _More... __more)
  {
    static_assert(sizeof(_From) >= (sizeof...(_More)+1) * sizeof(_To));
    return __convert<_V>(__a, __more...);
  }

  _GLIBCXX_SIMD_INTRINSIC auto __all(const _Arg& __a)
  {
    return __convert_all<_V>(__a);
  }
};
// }}}
// _SimdConverter scalar -> scalar {{{
template <typename _From, typename _To>
struct _SimdConverter<_From, simd_abi::scalar, _To, simd_abi::scalar>
{
  _GLIBCXX_SIMD_INTRINSIC _To operator()(_From __a)
  {
    return static_cast<_To>(__a);
  }
};

// }}}
// _SimdConverter "native" -> scalar {{{
template <typename _From, typename _To, typename _Abi>
struct _SimdConverter<_From, _Abi, _To, simd_abi::scalar>
{
  using _Arg = typename _Abi::template __traits<_From>::_SimdMember;
  static constexpr size_t _S_n = _Arg::_S_width;

  _GLIBCXX_SIMD_INTRINSIC std::array<_To, _S_n> __all(_Arg __a)
  {
    return __call_with_subscripts(
      __a, make_index_sequence<_S_n>(),
      [&](auto... __values) constexpr -> std::array<_To, _S_n> {
	return {static_cast<_To>(__values)...};
      });
  }
};

// }}}
// _SimdConverter scalar -> "native" {{{
template <typename _From, typename _To, typename _Abi>
struct _SimdConverter<_From, simd_abi::scalar, _To, _Abi>
{
  using _Ret = typename _Abi::template __traits<_To>::_SimdMember;

  template <typename... _More>
  _GLIBCXX_SIMD_INTRINSIC constexpr _Ret operator()(_From __a, _More... __more)
  {
    static_assert(sizeof...(_More) + 1 == _Ret::_S_width);
    static_assert(std::conjunction_v<std::is_same<_From, _More>...>);
    return __make_vector<_To>(__a, __more...);
  }
};

// }}}
// _SimdConverter "native 1" -> "native 2" {{{
template <typename _From, typename _To, typename _AFrom, typename _ATo>
struct _SimdConverter<_From, _AFrom, _To, _ATo>
{
  using _Arg = typename _AFrom::template __traits<_From>::_SimdMember;
  using _Ret = typename _ATo::template __traits<_To>::_SimdMember;
  using _V   = __vector_type_t<_To, simd_size_v<_To, _ATo>>;

  _GLIBCXX_SIMD_INTRINSIC auto __all(_Arg __a)
  {
    return __convert_all<_V>(__a);
  }

  template <typename... _More>
  _GLIBCXX_SIMD_INTRINSIC _Ret operator()(_Arg __a, _More... __more)
  {
    static_assert(std::conjunction_v<std::is_same<_Arg, _More>...>);
    return __convert<_V>(__a, __more...);
  }
};

// }}}
// _SimdConverter scalar -> fixed_size<1> {{{1
template <typename _From, typename _To>
struct _SimdConverter<_From, simd_abi::scalar, _To, simd_abi::fixed_size<1>> {
    _SimdTuple<_To, simd_abi::scalar> operator()(_From __x) { return {static_cast<_To>(__x)}; }
};

// _SimdConverter fixed_size<1> -> scalar {{{1
template <typename _From, typename _To>
struct _SimdConverter<_From, simd_abi::fixed_size<1>, _To, simd_abi::scalar> {
    _GLIBCXX_SIMD_INTRINSIC _To operator()(_SimdTuple<_From, simd_abi::scalar> __x)
    {
        return {static_cast<_To>(__x.first)};
    }
};

// _SimdConverter fixed_size<_N> -> fixed_size<_N> {{{1
template <typename _Tp, int _N>
struct _SimdConverter<_Tp, simd_abi::fixed_size<_N>, _Tp, simd_abi::fixed_size<_N>> {
    using _Arg = __fixed_size_storage_t<_Tp, _N>;
    _GLIBCXX_SIMD_INTRINSIC const _Arg &operator()(const _Arg &__x) { return __x; }
};

template <size_t _ChunkSize, typename _Tp> struct determine_required_input_chunks;

template <typename _Tp, typename... _Abis>
struct determine_required_input_chunks<0, _SimdTuple<_Tp, _Abis...>>
    : public std::integral_constant<size_t, 0> {
};

template <size_t _ChunkSize, typename _Tp, typename _Abi0, typename... _Abis>
struct determine_required_input_chunks<_ChunkSize, _SimdTuple<_Tp, _Abi0, _Abis...>>
    : public std::integral_constant<
          size_t, determine_required_input_chunks<_ChunkSize - simd_size_v<_Tp, _Abi0>,
                                                  _SimdTuple<_Tp, _Abis...>>::value> {
};

template <typename _From, typename _To> struct _FixedSizeConverter {
    struct _OneToMultipleChunks {
    };
    template <int _N> struct _MultipleToOneChunk {
    };
    struct _EqualChunks {
    };
    template <typename _FromAbi,
	      typename _ToAbi,
	      size_t _ToSize   = simd_size_v<_To, _ToAbi>,
	      size_t _FromSize = simd_size_v<_From, _FromAbi>>
    using _ChunkRelation = std::conditional_t<
      (_ToSize < _FromSize),
      _OneToMultipleChunks,
      std::conditional_t<(_ToSize == _FromSize),
			 _EqualChunks,
			 _MultipleToOneChunk<int(_ToSize / _FromSize)>>>;

    template <typename... _Abis>
    using _ReturnType = __fixed_size_storage_t<_To, _SimdTuple<_From, _Abis...>::size()>;


protected:
    // _OneToMultipleChunks {{{2
    template <typename _A0>
    _GLIBCXX_SIMD_INTRINSIC _ReturnType<_A0> __impl(_OneToMultipleChunks, const _SimdTuple<_From, _A0> &__x)
    {
        using _R = _ReturnType<_A0>;
        _SimdConverter<_From, _A0, _To, typename _R::_First_abi> __native_cvt;
        auto &&__multiple_return_chunks = __native_cvt.__all(__x.first);
        return __to_simd_tuple<_To, typename _R::_First_abi>(__multiple_return_chunks);
    }

    template <typename... _Abis>
    _GLIBCXX_SIMD_INTRINSIC _ReturnType<_Abis...> __impl(_OneToMultipleChunks,
                                           const _SimdTuple<_From, _Abis...> &__x)
    {
        using _R = _ReturnType<_Abis...>;
        using _Arg = _SimdTuple<_From, _Abis...>;
        constexpr size_t __first_chunk = simd_size_v<_From, typename _Arg::_First_abi>;
        _SimdConverter<_From, typename _Arg::_First_abi, _To, typename _R::_First_abi>
            __native_cvt;
        auto &&__multiple_return_chunks = __native_cvt.__all(__x.first);
        constexpr size_t __n_output_chunks =
            __first_chunk / simd_size_v<_To, typename _R::_First_abi>;
        return __simd_tuple_concat(
            __to_simd_tuple<_To, typename _R::_First_abi>(__multiple_return_chunks),
            __impl(_ChunkRelation<typename _Arg::_Second_type::_First_abi,
                               typename __simd_tuple_element<__n_output_chunks, _R>::type::abi_type>(),
                 __x.second));
    }

    // _MultipleToOneChunk {{{2
    template <int _N, typename _A0, typename... _Abis>
    _GLIBCXX_SIMD_INTRINSIC _ReturnType<_A0, _Abis...> __impl(_MultipleToOneChunk<_N>,
                                               const _SimdTuple<_From, _A0, _Abis...> &__x)
    {
        return __impl_mto(std::integral_constant<bool, sizeof...(_Abis) + 1 == _N>(),
                        std::make_index_sequence<_N>(), __x);
    }

    template <size_t... _Indexes, typename _A0, typename... _Abis>
    _GLIBCXX_SIMD_INTRINSIC _ReturnType<_A0, _Abis...> __impl_mto(true_type,
                                                   std::index_sequence<_Indexes...>,
                                                   const _SimdTuple<_From, _A0, _Abis...> &__x)
    {
        using _R = _ReturnType<_A0, _Abis...>;
        _SimdConverter<_From, _A0, _To, typename _R::_First_abi> __native_cvt;
        return {__native_cvt(__get_tuple_at<_Indexes>(__x)...)};
    }

    template <size_t... _Indexes, typename _A0, typename... _Abis>
    _GLIBCXX_SIMD_INTRINSIC _ReturnType<_A0, _Abis...> __impl_mto(false_type,
                                                   std::index_sequence<_Indexes...>,
                                                   const _SimdTuple<_From, _A0, _Abis...> &__x)
    {
        using _R = _ReturnType<_A0, _Abis...>;
        _SimdConverter<_From, _A0, _To, typename _R::_First_abi> __native_cvt;
        return {
            __native_cvt(__get_tuple_at<_Indexes>(__x)...),
            __impl(
                _ChunkRelation<
                    typename __simd_tuple_element<sizeof...(_Indexes),
                                           _SimdTuple<_From, _A0, _Abis...>>::type::abi_type,
                    typename _R::_Second_type::_First_abi>(),
                __simd_tuple_pop_front<sizeof...(_Indexes)>(__x))};
    }

    // _EqualChunks {{{2
    template <typename _A0>
    _GLIBCXX_SIMD_INTRINSIC _ReturnType<_A0> __impl(_EqualChunks, const _SimdTuple<_From, _A0> &__x)
    {
        _SimdConverter<_From, _A0, _To, typename _ReturnType<_A0>::_First_abi> __native_cvt;
        return {__native_cvt(__x.first)};
    }

    template <typename _A0, typename _A1, typename... _Abis>
    _GLIBCXX_SIMD_INTRINSIC _ReturnType<_A0, _A1, _Abis...> __impl(
        _EqualChunks, const _SimdTuple<_From, _A0, _A1, _Abis...> &__x)
    {
        using _R = _ReturnType<_A0, _A1, _Abis...>;
        using _Rem = typename _R::_Second_type;
        _SimdConverter<_From, _A0, _To, typename _R::_First_abi> __native_cvt;
	return {__native_cvt(__x.first),
                __impl(_ChunkRelation<_A1, typename _Rem::_First_abi>(), __x.second)};
    }

    //}}}2
};

template <typename _From, typename _To, int _N>
struct _SimdConverter<_From,
		      simd_abi::fixed_size<_N>,
		      _To,
		      simd_abi::fixed_size<_N>>
: public _FixedSizeConverter<_From, _To>
{
  using _Base = _FixedSizeConverter<_From, _To>;
  using _Ret  = __fixed_size_storage_t<_To, _N>;
  using _Arg  = __fixed_size_storage_t<_From, _N>;

  _GLIBCXX_SIMD_INTRINSIC _Ret operator()(const _Arg& __x)
  {
    if constexpr (__is_abi<typename _Ret::_First_abi, simd_abi::scalar>())
      { // then all entries of _Ret are scalar
	return __call_with_subscripts(
	  __x, make_index_sequence<_N>(), [](auto... __values) constexpr -> _Ret {
	    return __to_simd_tuple<_To, simd_abi::scalar>(
	      array<_To, _N>{static_cast<_To>(__values)...});
	  });
      }
    else
      {
	using _CR =
	  typename _Base::template _ChunkRelation<typename _Arg::_First_abi,
						  typename _Ret::_First_abi>;
	return _Base::__impl(_CR(), __x);
      }
  }
};

// _SimdConverter "native" -> fixed_size<_N> {{{1
// i.e. 1 register to ? registers
template <typename _From, typename _A, typename _To, int _N>
struct _SimdConverter<_From, _A, _To, simd_abi::fixed_size<_N>> {
    using __traits = _SimdTraits<_From, _A>;
    using _Arg = typename __traits::_SimdMember;
    using _ReturnType = __fixed_size_storage_t<_To, _N>;
    static_assert(_N == simd_size_v<_From, _A>,
                  "_SimdConverter to fixed_size only works for equal element counts");

    _GLIBCXX_SIMD_INTRINSIC _ReturnType operator()(_Arg __x)
    {
        return __impl(std::make_index_sequence<_ReturnType::_S_tuple_size>(), __x);
    }

private:
    _ReturnType __impl(std::index_sequence<0>, _Arg __x)
    {
        _SimdConverter<_From, _A, _To, typename _ReturnType::_First_abi> __native_cvt;
        return {__native_cvt(__x)};
    }
    template <size_t... _Indexes> _ReturnType __impl(std::index_sequence<_Indexes...>, _Arg __x)
    {
        _SimdConverter<_From, _A, _To, typename _ReturnType::_First_abi> __native_cvt;
        const auto &__tmp = __native_cvt.__all(__x);
        return {__tmp[_Indexes]...};
    }
};

// _SimdConverter fixed_size<_N> -> "native" {{{1
// i.e. ? register to 1 registers
template <typename _From, int _N, typename _To, typename _A>
struct _SimdConverter<_From, simd_abi::fixed_size<_N>, _To, _A> {
    using __traits = _SimdTraits<_To, _A>;
    using _ReturnType = typename __traits::_SimdMember;
    using _Arg = __fixed_size_storage_t<_From, _N>;
    static_assert(_N == simd_size_v<_To, _A>,
                  "_SimdConverter to fixed_size only works for equal element counts");

    _GLIBCXX_SIMD_INTRINSIC _ReturnType operator()(_Arg __x)
    {
        return __impl(std::make_index_sequence<_Arg::_S_tuple_size>(), __x);
    }

private:
    template <size_t... _Indexes> _ReturnType __impl(std::index_sequence<_Indexes...>, _Arg __x)
    {
        _SimdConverter<_From, typename _Arg::_First_abi, _To, _A> __native_cvt;
        return __native_cvt(__get_tuple_at<_Indexes>(__x)...);
    }
};

// }}}1
_GLIBCXX_SIMD_END_NAMESPACE
#endif  // __cplusplus >= 201703L
#endif  // _GLIBCXX_EXPERIMENTAL_SIMD_ABIS_H_
// vim: foldmethod=marker sw=2 noet ts=8 sts=2 tw=80
